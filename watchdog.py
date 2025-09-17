# watchdog.py
"""
Watchdog per Nova - monitoraggio codice, esecuzione test, apply_patch, reload.

Prerequisiti consigliati:
- pytest installato nel virtualenv (opzionale ma raccomandato)
- git disponibile (per rollback se necessario)
- apply_patch.apply_patch(file_path) presente (come da tuo progetto)
"""

import os
import time
import hashlib
import subprocess
import importlib
import sys
from typing import Dict, List
from loguru import logger

# Moduli del progetto (best-effort wiring)
try:
    from apply_patch import apply_patch
except Exception:
    def apply_patch(path):
        logger.warning("apply_patch non trovato: chiamata di stub (ritorna False).")
        return False

try:
    from core import NovaCore
except Exception:
    class NovaCore:
        def think(self):
            logger.debug("Stub NovaCore.think() chiamato (core non disponibile).")

try:
    from memory_timeline import MemoryTimeline
except Exception:
    class MemoryTimeline:
        def add_experience(self, *args, **kwargs):
            logger.debug("Stub MemoryTimeline.add_experience() chiamata.")


WATCHED_DIR = os.environ.get("NOVA_WATCHED_DIR", "nova")
CHECK_INTERVAL = float(os.environ.get("NOVA_WATCHDOG_INTERVAL", "3"))
PYTEST_CMD = os.environ.get("NOVA_PYTEST_CMD", "pytest -q")  # command to run tests
MAX_CHANGE_BATCH = int(os.environ.get("NOVA_WATCHDOG_BATCH", "8"))
TEST_TIMEOUT_SEC = int(os.environ.get("NOVA_WATCHDOG_TEST_TIMEOUT", "120"))

# small utility
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def _rel_module_name(base_dir: str, path: str) -> str:
    # derive a python module name from file path relative to base_dir
    rel = os.path.relpath(path, base_dir)
    if rel.endswith(".py"):
        rel = rel[:-3]
    parts = []
    for p in rel.split(os.sep):
        # ignore __init__ and empty parts
        if p == "__init__" or p == "":
            continue
        parts.append(p)
    return ".".join(parts)

def _run_subprocess(cmd: str, timeout: int = TEST_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    """
    Esegue una shell command e ritorna CompletedProcess.
    Usa shell=True quando cmd è stringa (compatibilità con env).
    """
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        logger.error("Comando timeout: %s", cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout="", stderr=str(e))

def _git_revert(files: List[str], repo_dir: str = ".") -> bool:
    """
    Tenta rollback dei file modificati usando git. Ritorna True se ok.
    """
    try:
        # ensure files exist in repo and run checkout
        cmd = ["git", "checkout", "--"] + files
        p = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        if p.returncode == 0:
            logger.info("Rollback git eseguito per file: %s", files)
            return True
        else:
            logger.warning("Git rollback fallito: %s", p.stderr.strip())
            return False
    except Exception as e:
        logger.exception("Errore durante git revert: %s", e)
        return False

def _run_pytest(cmd: str = PYTEST_CMD, timeout: int = TEST_TIMEOUT_SEC) -> bool:
    """
    Esegue pytest (o comando definito). Ritorna True se exitcode == 0.
    """
    try:
        logger.info("Esecuzione test: %s", cmd)
        p = _run_subprocess(cmd, timeout=timeout)
        logger.debug("TEST stdout:\n%s", p.stdout)
        logger.debug("TEST stderr:\n%s", p.stderr)
        if p.returncode == 0:
            logger.info("Test passati (exit=0).")
            return True
        else:
            logger.warning("Test falliti (exit=%s).", p.returncode)
            return False
    except Exception:
        logger.exception("Errore eseguendo i test.")
        return False

class Watchdog:
    def __init__(self, nova_core: NovaCore, memory_timeline: MemoryTimeline, watched_dir: str = WATCHED_DIR):
        self.nova_core = nova_core
        self.memory_timeline = memory_timeline
        self.watched_dir = watched_dir
        self.files_hash: Dict[str,str] = {}
        self._init_scan()

    def _init_scan(self):
        """Inizializza mappa hash con i file correnti (senza trigger)."""
        logger.info("Watchdog: inizializzo scansione su %s", self.watched_dir)
        for root, _, files in os.walk(self.watched_dir):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        self.files_hash[path] = _file_sha256(path)
                    except Exception:
                        logger.debug("Non posso hashare %s al primo avvio.", path)

    def scan_files(self) -> List[str]:
        """Ritorna lista di file modificati (sha256 cambiato)"""
        changed = []
        for root, _, files in os.walk(self.watched_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                try:
                    h = _file_sha256(path)
                except Exception:
                    continue
                prev = self.files_hash.get(path)
                if prev is None:
                    # nuova comparsa: considerala cambiata
                    self.files_hash[path] = h
                    logger.info("Nuovo file rilevato: %s", path)
                    changed.append(path)
                elif prev != h:
                    self.files_hash[path] = h
                    logger.info("Modifica rilevata su: %s", path)
                    changed.append(path)
        return changed

    def handle_changes(self, changed_files: List[str]):
        """
        Per batch di file modifcati:
        - esegue test (pytest)
        - se passano: chiama apply_patch(file) e reload moduli
        - se falliscono: tenta rollback via git (se disponibile)
        - registra eventi in MemoryTimeline e chiama nova_core.think()
        """
        if not changed_files:
            return

        # limit batch size
        batch = changed_files[:MAX_CHANGE_BATCH]
        logger.info("Handle_changes: batch %d file", len(batch))

        # 1) esegui test (il test riflette lo stato corrente dei file modificati)
        tests_ok = _run_pytest()

        if not tests_ok:
            # prova a revertare se repository git
            try:
                repo_root = os.getcwd()
                if os.path.isdir(os.path.join(repo_root, ".git")):
                    logger.warning("Test falliti: tentativo rollback git sui file modificati.")
                    _git_revert(batch, repo_dir=repo_root)
                    self.memory_timeline.add_experience(f"Modifiche non valide annullate (tests falliti): {', '.join([os.path.relpath(p) for p in batch])}",
                                                       category="system_update", importance=4)
                else:
                    self.memory_timeline.add_experience(f"Modifiche rilevate ma test falliti: {', '.join([os.path.relpath(p) for p in batch])}",
                                                       category="system_update", importance=3)
            except Exception:
                logger.exception("Errore durante gestione test falliti.")
            return

        # 2) prova ad applicare patch per ogni file (best-effort)
        applied = []
        failed_apply = []
        for fpath in batch:
            try:
                ok = apply_patch(fpath)
            except Exception:
                logger.exception("apply_patch ha sollevato eccezione per %s", fpath)
                ok = False

            if ok:
                applied.append(fpath)
            else:
                failed_apply.append(fpath)

        # 3) reload dei moduli applicati (best-effort)
        reloaded = []
        reload_failed = []
        for path in applied:
            try:
                # ricava nome modulo relativo
                module_name = _rel_module_name(self.watched_dir, path)
                if module_name:
                    logger.info("Tentativo reload modulo: %s (da file %s)", module_name, path)
                    # import if not present
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])
                    else:
                        # try to import module by name (may fail if package not in path)
                        try:
                            importlib.import_module(module_name)
                        except Exception:
                            # last resort: exec file into new module (dangerous)
                            logger.debug("Import standard fallito per %s; skipping importlib.import_module.", module_name)
                    reloaded.append(module_name)
                else:
                    reloaded.append(os.path.basename(path))
            except Exception:
                logger.exception("Reload modulo fallito per %s", path)
                reload_failed.append(path)

        # 4) registra eventi nella timeline e notifica core
        if applied:
            self.memory_timeline.add_experience(f"Patch applicate a: {', '.join([os.path.relpath(p) for p in applied])}",
                                               category="system_update", importance=5)
        if failed_apply:
            self.memory_timeline.add_experience(f"Patch fallite per: {', '.join([os.path.relpath(p) for p in failed_apply])}",
                                               category="system_update", importance=4)
        if reload_failed:
            self.memory_timeline.add_experience(f"Reload fallito per: {', '.join([os.path.relpath(p) for p in reload_failed])}",
                                               category="system_update", importance=3)

        # after successful apply/reload, let Nova reflect
        try:
            logger.info("Notifico NovaCore per riflessione post-aggiornamento.")
            self.nova_core.think()
        except Exception:
            logger.exception("Chiamata nova_core.think() fallita.")

    def run(self):
        logger.info("Watchdog avviato. Monitoro modifiche ogni %s secondi.", CHECK_INTERVAL)
        try:
            while True:
                try:
                    changed = self.scan_files()
                    if changed:
                        # filter duplicates by relative path ordering and pass to handler
                        self.handle_changes(changed)
                    time.sleep(CHECK_INTERVAL)
                except KeyboardInterrupt:
                    logger.info("Watchdog interrotto manualmente.")
                    break
                except Exception as e:
                    logger.exception("Errore nel loop principale del Watchdog: %s", e)
                    # backoff su errore per non floodare
                    time.sleep(min(30, CHECK_INTERVAL * 2))
        finally:
            logger.info("Watchdog terminato.")
