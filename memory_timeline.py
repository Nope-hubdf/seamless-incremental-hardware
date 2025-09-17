# memory_timeline.py
"""
MemoryTimeline - timeline di esperienze di Nova (robusta, integrata, thread-safe)

Responsabilità:
- Memorizzare eventi/esperienze con timestamp, categoria, importanza e metadata.
- Notificare moduli collegati: dream_generator, attention_manager, inner_dialogue,
  conscious_loop, long_term_memory_manager, emergent_self, desire_engine.
- Fornire API per ricerca, esport, consolidamento verso LTM, pruning.
- Persistenza atomica su internal_state.yaml (o stato condiviso passato).
- Difensiva: non assume che i moduli abbiano tutte le API; usa fallback.
"""

import os
import threading
import yaml
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

# Percorso stato persistente (override mediante env var se necessario)
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")

# Import difensivo di moduli collegati (se non presenti verranno ignorati)
try:
    from dream_generator import DreamGenerator
except Exception:
    DreamGenerator = None

try:
    from conscious_loop import ConsciousLoop
except Exception:
    ConsciousLoop = None

try:
    from attention_manager import AttentionManager
except Exception:
    AttentionManager = None

try:
    from inner_dialogue import InnerDialogue
except Exception:
    InnerDialogue = None

try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None

# -----------------------
# Helpers
# -----------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# MemoryTimeline
# -----------------------
class MemoryTimeline:
    def __init__(self, state: Optional[Dict[str, Any]] = None,
                 dream_generator: Optional[Any] = None,
                 conscious_loop: Optional[Any] = None,
                 attention_manager: Optional[Any] = None,
                 inner_dialogue: Optional[Any] = None,
                 long_term_memory_manager: Optional[Any] = None,
                 emergent_self: Optional[Any] = None,
                 desire_engine: Optional[Any] = None):
        """
        state: dizionario condiviso (es. core.state). Se None, si carica da STATE_FILE.
        Tutti gli altri moduli sono opzionali ma consigliati per integrazione completa.
        """
        self._lock = threading.RLock()
        # carica stato condiviso o da file
        if state is None:
            self.state = self._load_state()
        else:
            self.state = state

        # assicurati container timeline nello stato condiviso
        self.state.setdefault("timeline", [])
        self.state.setdefault("_timeline_meta", {"created": _now_iso(), "last_saved": None})

        # wiring moduli (preferisci oggetti passati)
        self.dream_generator = dream_generator
        self.conscious_loop = conscious_loop
        self.attention_manager = attention_manager
        self.inner_dialogue = inner_dialogue
        self.long_term_memory_manager = long_term_memory_manager
        self.emergent_self = emergent_self
        self.desire_engine = desire_engine

        # se alcuni non sono stati passati e sono disponibili come classi, puoi inizializzarli con lo state
        try:
            if self.dream_generator is None and DreamGenerator:
                self.dream_generator = DreamGenerator(self.state)
        except Exception:
            self.dream_generator = None

        try:
            if self.attention_manager is None and AttentionManager:
                self.attention_manager = AttentionManager(self.state)
        except Exception:
            self.attention_manager = None

        try:
            if self.inner_dialogue is None and InnerDialogue:
                # InnerDialogue spesso richiede altri engine; passiamo ciò che abbiamo (difensivo)
                try:
                    self.inner_dialogue = InnerDialogue(self.state, getattr(self, "emotion_engine", None), getattr(self, "motivational_engine", None), self, self.dream_generator, getattr(self, "life_journal", None), self.attention_manager)
                except Exception:
                    # fallback: istanzia con minima firma
                    try:
                        self.inner_dialogue = InnerDialogue(self.state)
                    except Exception:
                        self.inner_dialogue = None
        except Exception:
            self.inner_dialogue = None

        try:
            if self.long_term_memory_manager is None and LongTermMemoryManager:
                self.long_term_memory_manager = LongTermMemoryManager(self.state, timeline=self)
        except Exception:
            self.long_term_memory_manager = None

        logger.info("MemoryTimeline inizializzato. Eventi correnti: %d", len(self.state.get("timeline", [])))

    # -----------------------
    # Persistence
    # -----------------------
    def _load_state(self) -> Dict[str, Any]:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf8") as f:
                    st = yaml.safe_load(f) or {}
                    logger.info("MemoryTimeline: stato caricato da %s", STATE_FILE)
                    return st
        except Exception:
            logger.exception("MemoryTimeline: errore caricamento stato")
        # default
        logger.info("MemoryTimeline: creazione nuovo stato.")
        return {"timeline": [], "_timeline_meta": {"created": _now_iso(), "last_saved": None}}

    def _save_state(self):
        with self._lock:
            try:
                self.state["_timeline_meta"]["last_saved"] = _now_iso()
                _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                logger.debug("MemoryTimeline: stato salvato su %s", STATE_FILE)
            except Exception:
                logger.exception("MemoryTimeline: errore salvataggio stato")

    # -----------------------
    # Core: add_experience
    # -----------------------
    def add_experience(self, content: Any, category: str = "general", importance: int = 1, metadata: Optional[Dict[str, Any]] = None, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggiunge un evento nella timeline.
        content: testo oppure dict con dettagli (se dict verrà memorizzato come content['text'] se presente).
        category: es. 'dream', 'interaction', 'perception', 'task', ecc.
        importance: intero (1..5)
        metadata: dizionario libero
        source: stringa opzionale che descrive l'origine (es. 'remote', 'vision', 'user')
        Ritorna l'entry inserita.
        """
        with self._lock:
            try:
                if isinstance(content, dict):
                    text = content.get("content") or content.get("text") or str(content)
                else:
                    text = str(content)
                timestamp = _now_iso()
                entry = {
                    "id": f"t_{int(time.time()*1000)}_{abs(hash(timestamp))%10000}",
                    "timestamp": timestamp,
                    "content": text,
                    "category": category,
                    "importance": int(max(1, min(5, importance))),
                    "metadata": metadata or {},
                    "source": source or metadata.get("source") if metadata else None
                }
                # append e persist
                self.state.setdefault("timeline", []).append(entry)
                self._save_state()
                logger.info("MemoryTimeline: esperienza aggiunta id=%s (cat=%s imp=%d) %s", entry["id"], category, entry["importance"], text[:120])

                # notifiche ai moduli collegati (difensive)
                self._notify_modules_on_add(entry)

                return entry
            except Exception:
                logger.exception("MemoryTimeline: exception in add_experience")
                raise

    # -----------------------
    # Notifiche verso altri moduli
    # -----------------------
    def _notify_modules_on_add(self, entry: Dict[str, Any]):
        """
        Comunica l'evento ai moduli: dream_generator, attention_manager, inner_dialogue, conscious_loop,
        long_term_memory_manager, emergent_self, desire_engine.
        Ogni notifica è try/except per non interrompere il flusso.
        """
        # dream generator: se categoria == 'dream' o se importanza alta, notify
        try:
            if self.dream_generator and hasattr(self.dream_generator, "add_dream") and entry["category"] == "dream":
                self.dream_generator.add_dream(entry["content"])
            # some dream modules expose process_new_input
            elif self.dream_generator and hasattr(self.dream_generator, "process_new_input"):
                self.dream_generator.process_new_input(entry["content"], entry["category"])
        except Exception:
            logger.debug("MemoryTimeline: dream_generator notify failed")

        # attention manager: update focus with short summary
        try:
            if self.attention_manager and hasattr(self.attention_manager, "update_focus"):
                # prefer list input
                try:
                    self.attention_manager.update_focus([entry["content"]])
                except TypeError:
                    self.attention_manager.update_focus(entry["content"])
        except Exception:
            logger.debug("MemoryTimeline: attention_manager notify failed")

        # inner dialogue: feed for reflection
        try:
            if self.inner_dialogue and hasattr(self.inner_dialogue, "add_subconscious_thought"):
                self.inner_dialogue.add_subconscious_thought(entry["content"])
            elif self.inner_dialogue and hasattr(self.inner_dialogue, "register_experience"):
                self.inner_dialogue.register_experience(entry)
        except Exception:
            logger.debug("MemoryTimeline: inner_dialogue notify failed")

        # conscious loop: integrate memory/reflection
        try:
            if self.conscious_loop and hasattr(self.conscious_loop, "integrate_memory"):
                self.conscious_loop.integrate_memory(entry)
            elif self.conscious_loop and hasattr(self.conscious_loop, "register_memory"):
                self.conscious_loop.register_memory(entry)
        except Exception:
            logger.debug("MemoryTimeline: conscious_loop notify failed")

        # long-term memory manager: consolidate if important
        try:
            if self.long_term_memory_manager and hasattr(self.long_term_memory_manager, "add_experience"):
                # consolidate immediately if importance high
                if entry["importance"] >= 4:
                    self.long_term_memory_manager.add_experience(entry["content"], category=entry["category"], tags=entry.get("metadata", {}).get("tags"), importance=entry["importance"])
            # otherwise do not block
        except Exception:
            logger.debug("MemoryTimeline: long_term_memory_manager notify failed")

        # emergent self / narrative: allow emergent_self to integrate input
        try:
            if self.emergent_self and hasattr(self.emergent_self, "create_narrative_entry"):
                # do not block; best-effort
                try:
                    self.emergent_self.create_narrative_entry(use_llm=False, recent_n=6)
                except Exception:
                    pass
        except Exception:
            logger.debug("MemoryTimeline: emergent_self notify failed")

        # desire engine: may generate desires from meaningful experiences
        try:
            if self.desire_engine and hasattr(self.desire_engine, "generate_desires"):
                # light-weight trigger: let desire engine look at timeline later
                try:
                    # Non chiamare pesantemente qui per non bloccare: schedule or quick call
                    self.desire_engine.generate_desires(recent_n=10, min_importance=3)
                except Exception:
                    pass
        except Exception:
            logger.debug("MemoryTimeline: desire_engine notify failed")

    # -----------------------
    # Convenience / reflection helpers
    # -----------------------
    def reflect_on(self, entry_or_text: Any, depth: int = 1) -> List[str]:
        """
        Richiama inner_dialogue / self_reflection per produrre riflessioni a partire dall'entry.
        Restituisce la lista di riflessioni prodotte (se disponibili).
        """
        reflections = []
        try:
            summary = entry_or_text if isinstance(entry_or_text, str) else entry_or_text.get("content", str(entry_or_text))
            if self.inner_dialogue and hasattr(self.inner_dialogue, "generate_dialogue"):
                try:
                    dialog = self.inner_dialogue.generate_dialogue()
                    reflections.append(dialog)
                except Exception:
                    logger.debug("MemoryTimeline: inner_dialogue.generate_dialogue failed")
            # try self_reflection module if present inside conscious_loop or emergent_self
            sr = getattr(self.conscious_loop, "self_reflection", None) if self.conscious_loop else None
            if sr and hasattr(sr, "analyze_experience"):
                try:
                    r = sr.analyze_experience({"content": summary, "type": "interaction"})
                    reflections.append(r.get("insight") if isinstance(r, dict) else str(r))
                except Exception:
                    logger.debug("MemoryTimeline: self_reflection analyze failed")
            return reflections
        except Exception:
            logger.exception("MemoryTimeline: exception in reflect_on")
            return reflections

    # -----------------------
    # Query / retrieval
    # -----------------------
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.state.get("timeline", []))[-n:]

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self.state.get("timeline", []) if e.get("category") == category]

    def get_high_importance(self, threshold: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self.state.get("timeline", []) if int(e.get("importance", 1)) >= threshold]

    def search(self, keyword: str, n: int = 20) -> List[Dict[str, Any]]:
        """Ricerca semplice case-insensitive nella timeline (fallback)."""
        with self._lock:
            q = keyword.lower().strip()
            results = []
            for e in reversed(self.state.get("timeline", [])[-500:]):  # search recent N for speed
                if q in str(e.get("content","")).lower() or q in str(e.get("metadata","")).lower():
                    results.append(e)
                    if len(results) >= n:
                        break
            return results

    # -----------------------
    # Consolidamento / maintenance
    # -----------------------
    def consolidate_to_long_term(self, importance_threshold: int = 4, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Trasferisce elementi significativi nella memoria a lungo termine (LTM).
        Ritorna la lista degli oggetti aggiunti allo LTM.
        """
        added = []
        try:
            candidates = [e for e in self.get_high_importance(importance_threshold)]
            # prefer più recenti
            candidates = sorted(candidates, key=lambda x: x.get("timestamp", ""), reverse=True)[:max_items]
            for c in candidates:
                if self.long_term_memory_manager and hasattr(self.long_term_memory_manager, "add_experience"):
                    try:
                        itm = self.long_term_memory_manager.add_experience(c.get("content"), category=c.get("category"), tags=c.get("metadata", {}).get("tags", []), importance=c.get("importance", 1), metadata={"timeline_id": c.get("id")})
                        added.append(itm)
                    except Exception:
                        logger.debug("MemoryTimeline: LTM add_experience failed for %s", c.get("id"))
            return added
        except Exception:
            logger.exception("MemoryTimeline: exception in consolidate_to_long_term")
            return added

    def prune_timeline(self, keep_last: int = 2000):
        """
        Riduce la timeline mantenendo solo gli ultimi keep_last eventi (utile su dispositivi con poco spazio).
        """
        with self._lock:
            try:
                tl = self.state.get("timeline", [])
                if len(tl) > keep_last:
                    removed = len(tl) - keep_last
                    self.state["timeline"] = tl[-keep_last:]
                    self._save_state()
                    logger.info("MemoryTimeline: pruning eseguito. Rimossi %d eventi.", removed)
            except Exception:
                logger.exception("MemoryTimeline: exception in prune_timeline")

    # -----------------------
    # Export / debug
    # -----------------------
    def export_timeline(self, path: str = "timeline_export.yaml", last_n: Optional[int] = None) -> str:
        with self._lock:
            try:
                data = self.state.get("timeline", []) if last_n is None else self.state.get("timeline", [])[-last_n:]
                _atomic_write(path, yaml.safe_dump(data, allow_unicode=True))
                logger.info("MemoryTimeline: esportato timeline in %s", path)
                return path
            except Exception:
                logger.exception("MemoryTimeline: exception in export_timeline")
                return ""

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            tl = self.state.get("timeline", [])
            return {
                "total_events": len(tl),
                "last_event": tl[-1] if tl else None,
                "last_saved": self.state.get("_timeline_meta", {}).get("last_saved")
            }

# -----------------------
# Quick test / demo (run solo in sviluppo)
# -----------------------
if __name__ == "__main__":
    # demo minimale
    mt = MemoryTimeline()
    mt.add_experience("Nova ha visto un tramonto rosso e ha provato meraviglia.", category="perception", importance=3)
    mt.add_experience("Nova ha sognato di volare.", category="dream", importance=5)
    print("Recent:", mt.get_recent(5))
    print("Search 'tramonto':", mt.search("tramonto"))
    mt.consolidate_to_long_term(importance_threshold=4)
    print("Snapshot:", mt.snapshot())
