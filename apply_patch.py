# apply_patch.py
"""
PatchManager avanzato per Nova.

Caratteristiche principali:
- Backup atomico dei file prima di scrivere
- Validazione sintattica per file .py (ast.parse)
- Controllo anti-pattern semplice (evita comandi shell hard-coded, import di moduli sospetti)
- Commit su git (branch temporaneo) e rollback automatico se i test falliscono
- Esecuzione pytest prima del commit (configurabile)
- Audit log persistente in backups/patch_log.yaml
- Funzione top-level apply_patch(path) compatibile con watchdog
"""

import os
import shutil
import subprocess
import tempfile
import time
import hashlib
import yaml
import ast
from datetime import datetime
from loguru import logger
from typing import Optional

# Configurazioni (puoi sovrascrivere con env var)
NOVA_DIR = os.environ.get("NOVA_DIR", "nova")
PATCHES_DIR = os.environ.get("PATCHES_DIR", "patches")
BACKUP_DIR = os.environ.get("BACKUP_DIR", "backups")
LOG_FILE = os.environ.get("PATCH_LOG", os.path.join(BACKUP_DIR, "patch_log.yaml"))
PYTEST_CMD = os.environ.get("PYTEST_CMD", "pytest -q")
RUN_TESTS = os.environ.get("APPLYPATCH_RUN_TESTS", "1") == "1"
GIT_ENABLED = os.environ.get("APPLYPATCH_USE_GIT", "1") == "1"
FORBIDDEN_PATTERNS = [
    "os.system(", "subprocess.Popen(", "subprocess.run(", "eval(", "exec(", "__import__(",
    "open('/etc/passwd", "rm -rf", "ssh ", "scp ", "wget ", "curl "
]

os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(PATCHES_DIR, exist_ok=True)
os.makedirs(NOVA_DIR, exist_ok=True)

def _now_str():
    return datetime.utcnow().isoformat()

def _sha256(text: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(text)
    return h.hexdigest()

def _atomic_write(path: str, content: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(content)
    os.replace(tmp, path)

def _log_patch_entry(entry: dict):
    # Append to YAML log (thread-unsafe simple append; fine for single-process dev)
    try:
        log = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf8") as f:
                log = yaml.safe_load(f) or []
        log.append(entry)
        with open(LOG_FILE, "w", encoding="utf8") as f:
            yaml.safe_dump(log, f, allow_unicode=True)
    except Exception:
        logger.exception("Impossibile scrivere log patch.")

def _run_cmd(cmd: str, cwd: Optional[str] = None, timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        logger.error("Comando timeout: %s", cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout="", stderr=str(e))

def _is_python_file(path: str) -> bool:
    return path.lower().endswith(".py")

def _validate_python_syntax(source: str) -> Optional[str]:
    """
    Restituisce None se OK, altrimenti messaggio di errore.
    """
    try:
        ast.parse(source)
        return None
    except SyntaxError as se:
        return f"{se.__class__.__name__}: {se}"

def _contains_forbidden_patterns(source: str) -> Optional[str]:
    low = source.lower()
    for p in FORBIDDEN_PATTERNS:
        if p in low:
            return f"Pattern proibito rilevato: {p}"
    return None

def _create_backup(original_path: str) -> Optional[str]:
    if not os.path.exists(original_path):
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(original_path)
    backup_name = f"{base}.{ts}.bak"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    try:
        shutil.copy2(original_path, backup_path)
        logger.info("Backup creato: %s", backup_path)
        return backup_path
    except Exception:
        logger.exception("Errore creando backup per %s", original_path)
        return None

def _git_commit_single_file(target_relpath: str, commit_message: str, repo_root: str = ".") -> bool:
    """Aggiunge e committa un singolo file; crea branch temporaneo se necessario."""
    try:
        # ensure git repo
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            logger.debug("Git non disponibile nella repo (%s). Skip git commit.", repo_root)
            return False

        branch = f"autopatch/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        # create new branch
        p = _run_cmd(f"git checkout -b {branch}", cwd=repo_root)
        if p.returncode != 0:
            logger.warning("Impossibile creare branch git '%s': %s", branch, p.stderr.strip())
            return False

        # add and commit
        p = _run_cmd(f"git add -- {target_relpath}", cwd=repo_root)
        if p.returncode != 0:
            logger.warning("git add fallito: %s", p.stderr.strip())
            return False
        p = _run_cmd(f"git commit -m \"{commit_message}\" --no-verify", cwd=repo_root)
        if p.returncode != 0:
            logger.warning("git commit fallito: %s", p.stderr.strip())
            return False

        logger.info("Patch committata su branch %s (file: %s)", branch, target_relpath)
        return True
    except Exception:
        logger.exception("Errore durante git commit")
        return False

def _git_revert_files(files, repo_root="."):
    try:
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            return False
        cmd = ["git", "checkout", "--"] + files
        p = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if p.returncode == 0:
            logger.info("Git revert eseguito su: %s", files)
            return True
        else:
            logger.warning("Git revert fallito: %s", p.stderr.strip())
            return False
    except Exception:
        logger.exception("Errore durante git revert")
        return False

class PatchManager:
    def __init__(self, nova_dir: str = NOVA_DIR, patches_dir: str = PATCHES_DIR, backup_dir: str = BACKUP_DIR):
        self.nova_dir = nova_dir
        self.patches_dir = patches_dir
        self.backup_dir = backup_dir
        logger.info("PatchManager inizializzato (nova_dir=%s)", self.nova_dir)

    def apply_patch_file(self, patch_file_path: str) -> bool:
        """
        Applica una patch file (patch_file_path è il path al file contenente il nuovo sorgente).
        La logica:
         - legge il contenuto della patch
         - determina target in nova/ (basename)
         - valida sintassi e pattern per python
         - backup del file originale
         - scrittura atomica sul file target
         - esegue test (pytest) se abilitati
         - committa su git su branch temporaneo (se abilitato)
         - registra entry di audit su LOG_FILE
        """
        try:
            if not os.path.exists(patch_file_path):
                logger.warning("Patch non trovata: %s", patch_file_path)
                return False

            with open(patch_file_path, "r", encoding="utf8") as f:
                content = f.read()

            target_basename = os.path.basename(patch_file_path)
            target_path = os.path.join(self.nova_dir, target_basename)

            # validate python syntax if file .py
            if _is_python_file(target_path):
                syntax_err = _validate_python_syntax(content)
                if syntax_err:
                    logger.warning("Validazione sintattica fallita per %s: %s", target_basename, syntax_err)
                    return False
                forbidden = _contains_forbidden_patterns(content)
                if forbidden:
                    logger.warning("Patch contiene pattern proibiti: %s", forbidden)
                    # qui non applichiamo automaticamente: ritorniamo False
                    return False

            # create backup
            backup_path = _create_backup(target_path)

            # write atomically
            try:
                _atomic_write(target_path, content)
            except Exception:
                logger.exception("Scrittura atomica fallita per %s", target_path)
                # attempt rollback from backup
                if backup_path:
                    shutil.copy2(backup_path, target_path)
                return False

            # optional: run tests
            tests_ok = True
            if RUN_TESTS:
                logger.info("Esecuzione test (pytest) dopo applicazione patch...")
                p = _run_cmd(PYTEST_CMD)
                if p.returncode != 0:
                    logger.warning("Test falliti dopo la patch. stdout:\n%s\nstderr:\n%s", p.stdout, p.stderr)
                    tests_ok = False

            if not tests_ok:
                # rollback: restore backup if exists, try git revert if possible
                if backup_path and os.path.exists(backup_path):
                    shutil.copy2(backup_path, target_path)
                else:
                    logger.warning("Nessun backup disponibile per rollback su %s", target_path)
                # git revert attempt
                _git_revert_files([os.path.relpath(target_path, ".")])
                entry = {
                    "timestamp": _now_str(),
                    "patch_file": patch_file_path,
                    "target": target_path,
                    "status": "tests_failed",
                    "backup": backup_path,
                    "tests_stdout": p.stdout if 'p' in locals() else "",
                    "tests_stderr": p.stderr if 'p' in locals() else ""
                }
                _log_patch_entry(entry)
                return False

            # git commit (best-effort)
            if GIT_ENABLED:
                rel = os.path.relpath(target_path, ".")
                commit_msg = f"Autopatch applied {os.path.basename(patch_file_path)} @ {_now_str()}"
                _git_commit_single_file(rel, commit_msg)

            # audit log
            entry = {
                "timestamp": _now_str(),
                "patch_file": patch_file_path,
                "target": target_path,
                "status": "applied",
                "backup": backup_path,
                "sha256": _sha256(content.encode("utf8"))
            }
            _log_patch_entry(entry)
            logger.info("Patch applicata con successo: %s -> %s", patch_file_path, target_path)
            return True

        except Exception:
            logger.exception("apply_patch_file eccezione")
            return False

    def apply_all_in_dir(self, patches_dir: Optional[str] = None) -> int:
        """Applica tutte le patch nella cartella (ordine alfabetico). Ritorna numero applicate."""
        patches_dir = patches_dir or self.patches_dir
        applied = 0
        for name in sorted(os.listdir(patches_dir)):
            full = os.path.join(patches_dir, name)
            if os.path.isfile(full):
                ok = self.apply_patch_file(full)
                if ok:
                    applied += 1
        return applied

# Top-level compat wrapper per watchdog.py e altre parti che chiamano apply_patch(path)
_global_patch_manager = PatchManager()

def apply_patch(path: str) -> bool:
    """
    Interfaccia semplice: path può essere un path assoluto/relativo a un file di patch (nella cartella patches/)
    o il path assoluto del file target già pronto (in caso tu voglia fornire direttamente il file sorgente).
    """
    # Se il file è nella cartella patches, usiamo apply_patch_file
    p = os.path.abspath(path)
    # Se è un file in patches directory -> treat as patch content to apply into nova/ by basename
    if os.path.commonpath([os.path.abspath(PATCHES_DIR)]) == os.path.commonpath([os.path.abspath(PATCHES_DIR), p]):
        return _global_patch_manager.apply_patch_file(p)
    # Else: maybe user passed a source file already in nova/ — in questo caso copy it as "patch" (safe path)
    if os.path.commonpath([os.path.abspath(NOVA_DIR)]) == os.path.commonpath([os.path.abspath(NOVA_DIR), p]):
        # create a temporary patch file from this source and apply via manager path
        # (we treat it as already applied and return True)
        logger.info("apply_patch: file già dentro nova/; trattato come update in-place.")
        # Basic validation: if python, validate syntax
        try:
            with open(p, "r", encoding="utf8") as f:
                content = f.read()
            if _is_python_file(p):
                err = _validate_python_syntax(content)
                if err:
                    logger.warning("apply_patch: errore sintassi nel file target: %s", err)
                    return False
            # backup already exists? create one
            _create_backup(p)
            # We've already written the file (it's the passed path), run tests if configured
            if RUN_TESTS:
                res = _run_cmd(PYTEST_CMD)
                if res.returncode != 0:
                    logger.warning("apply_patch: test falliti dopo update in-place.")
                    # attempt revert from backups if possible
                    _git_revert_files([os.path.relpath(p, ".")])
                    return False
            # log
            _log_patch_entry({"timestamp": _now_str(), "patch_file": p, "target": p, "status": "applied_inplace"})
            return True
        except Exception:
            logger.exception("apply_patch: eccezione processing in-place file")
            return False

    # If none matched, treat as generic patch file
    return _global_patch_manager.apply_patch_file(p)

# Self-test CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Applica patch di prova alla cartella nova/")
    parser.add_argument("patch", help="File patch (path) presente nella cartella patches/ o file sorgente")
    args = parser.parse_args()
    ok = apply_patch(args.patch)
    print("DONE:", ok)
