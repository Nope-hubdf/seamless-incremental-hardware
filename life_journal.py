# life_journal.py
"""
LifeJournal - Diario autobiografico automatico di Nova

Funzionalità principali:
- record_event(description, category, importance, metadata)
- generate_daily_summary(date), generate_range_summary(start_date, end_date)
- export_journal_md(path), export_journal_json(path)
- find_entries(filter_fn) e search_by_keyword(kw)
- integration hooks con MemoryTimeline, DreamGenerator, ConsciousLoop, AttentionManager, EmotionEngine, SelfReflection
- optional LLM-driven summarization usando modello locale Gemma (llama_cpp) con fallback
- persistenza atomica su internal_state.yaml
"""

import os
import json
import yaml
import threading
import traceback
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Callable

from loguru import logger

STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")
JOURNAL_MAX_ENTRIES = int(os.environ.get("NOVA_JOURNAL_MAX_ENTRIES", 5000))

# -----------------------
# Optional local LLM summarizer (Gemma via llama_cpp) with fallback
# -----------------------
def _load_llm(model_path: str):
    try:
        from llama_cpp import Llama
        logger.info("LifeJournal: caricamento LLM per summarization %s", model_path)
        llm = Llama(model_path=model_path)
        def _call(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                if not resp:
                    return ""
                text = ""
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        text = "".join([c.get("text","") for c in choices])
                elif hasattr(resp, "text"):
                    text = getattr(resp, "text")
                return (text or "").strip()
            except Exception:
                logger.exception("LifeJournal: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.debug("LifeJournal: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # fallback naive: prendi le prime frasi più informative
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = lines[-1] if lines else ""
            return f"Riepilogo sintetico (fallback): {seed[:200]}"
        return _fallback

# -----------------------
# Atomic write helper
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# LifeJournal
# -----------------------
class LifeJournal:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None, model_path: str = DEFAULT_MODEL_PATH):
        """
        state: core.state condiviso
        core: opzionale NovaCore (per wiring ai moduli)
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # collaborators (prefer to use core-provided instances to avoid duplicates)
        self.timeline = getattr(core, "memory_timeline", None)
        self.dream_gen = getattr(core, "dream_generator", None)
        self.conscious_loop = getattr(core, "conscious_loop", None)
        self.attention = getattr(core, "attention_manager", None)
        self.emotion_engine = getattr(core, "emotion_engine", None)
        self.self_reflection = getattr(core, "self_reflection", None)

        # journal storage in state
        self.state.setdefault("journal", [])  # list of entries
        # each entry: {id, ts_iso, category, description, importance, metadata}

        # load LLM summarizer callable (optional)
        self.summarizer = _load_llm(model_path)

        logger.info("LifeJournal inizializzato (LLM summarizer=%s).", "yes" if self.summarizer else "no")

    # -----------------------
    # Internal: create unique id for entries
    # -----------------------
    def _make_entry_id(self) -> str:
        return f"j_{int(datetime.utcnow().timestamp()*1000)}_{len(self.state.get('journal',[]))}"

    # -----------------------
    # Record event (main API)
    # - description: testo
    # - category: 'interaction','dream','thought','perception','task','emotion','system', etc.
    # - importance: 0.0 - 1.0
    # - metadata: optional dict (e.g., {'image_path':..., 'audio_path':..., 'sensor':...})
    def record_event(self,
                     description: str,
                     category: str = "general",
                     importance: float = 0.5,
                     metadata: Optional[Dict[str, Any]] = None,
                     persist: bool = True) -> Dict[str, Any]:
        with self._lock:
            try:
                ts = datetime.utcnow().isoformat()
                entry = {
                    "id": self._make_entry_id(),
                    "ts": ts,
                    "category": category,
                    "description": description,
                    "importance": float(max(0.0, min(1.0, importance))),
                    "metadata": metadata or {}
                }
                self.state.setdefault("journal", []).append(entry)
                # cap journal length
                if len(self.state["journal"]) > JOURNAL_MAX_ENTRIES:
                    self.state["journal"] = self.state["journal"][-JOURNAL_MAX_ENTRIES:]

                # Add to memory timeline
                try:
                    if self.timeline and hasattr(self.timeline, "add_experience"):
                        self.timeline.add_experience({"text": description, "meta": {"id": entry["id"], "category": category, "importance": entry["importance"]}}, category=category, importance=max(1, int(entry["importance"]*3)))
                    else:
                        self.state.setdefault("timeline", []).append({"timestamp": ts, "content": description, "category": category, "importance": int(entry["importance"]*3)})
                except Exception:
                    logger.exception("LifeJournal: errore add timeline")

                # If dream, register with dream_generator
                try:
                    if category == "dream":
                        if self.dream_gen and hasattr(self.dream_gen, "register_reflection"):
                            self.dream_gen.register_reflection(entry)
                        elif self.dream_gen and hasattr(self.dream_gen, "add_influence"):
                            self.dream_gen.add_influence(description)
                except Exception:
                    logger.exception("LifeJournal: errore notify dream_generator")

                # Update attention
                try:
                    if self.attention and hasattr(self.attention, "update_focus"):
                        # pass simple text or entry depending on API
                        try:
                            self.attention.update_focus([description])
                        except TypeError:
                            self.attention.update_focus(entry)
                except Exception:
                    logger.exception("LifeJournal: errore notify attention")

                # Notify conscious loop
                try:
                    if self.conscious_loop:
                        if hasattr(self.conscious_loop, "register_reflection"):
                            self.conscious_loop.register_reflection(entry)
                        elif hasattr(self.conscious_loop, "on_perception"):
                            self.conscious_loop.on_perception({"type":"journal_entry","payload":entry})
                except Exception:
                    logger.exception("LifeJournal: errore notify conscious_loop")

                # Optionally pass to emotion_engine for appraisal
                try:
                    if self.emotion_engine and hasattr(self.emotion_engine, "process_experience"):
                        self.emotion_engine.process_experience({"content": description, "importance": importance, "source": "journal"})
                except Exception:
                    logger.exception("LifeJournal: errore notify emotion_engine")

                # Optionally send to self_reflection
                try:
                    if self.self_reflection and hasattr(self.self_reflection, "analyze_experience"):
                        self.self_reflection.analyze_experience({"content": description, "type": category, "importance": importance})
                except Exception:
                    logger.exception("LifeJournal: errore notify self_reflection")

                if persist:
                    self._persist_state()

                logger.info("LifeJournal: evento registrato id=%s category=%s", entry["id"], category)
                return entry
            except Exception:
                logger.exception("LifeJournal: exception in record_event")
                raise

    # -----------------------
    # Summaries
    # -----------------------
    def generate_daily_summary(self, day: Optional[date] = None, use_llm: bool = True) -> str:
        """
        Genera un riepilogo del giorno specificato (UTC). Se day None -> oggi.
        """
        if day is None:
            day = datetime.utcnow().date()
        start = datetime.combine(day, datetime.min.time())
        end = datetime.combine(day, datetime.max.time())
        return self.generate_range_summary(start, end, use_llm=use_llm)

    def generate_range_summary(self, start_dt: datetime, end_dt: datetime, use_llm: bool = True) -> str:
        """
        Genera un sommario per l'intervallo richiesto.
        """
        with self._lock:
            entries = [e for e in self.state.get("journal", []) if start_dt.isoformat() <= e["ts"] <= end_dt.isoformat()]
            if not entries:
                return f"Nessuna entry tra {start_dt.isoformat()} e {end_dt.isoformat()}."

            # build prompt or naive summary
            if use_llm and callable(self.summarizer):
                try:
                    prompt_lines = [f"- [{e['category']}] {e['ts']}: {e['description']}" for e in entries[-200:]]
                    prompt = (
                        "Sei Nova. Riassumi in italiano il seguente insieme di registrazioni autobiografiche, "
                        "fornisci 3-5 insight principali e suggerimenti pratici per la crescita personale:\n\n"
                        + "\n".join(prompt_lines)
                        + "\n\nRiepilogo:"
                    )
                    out = self.summarizer(prompt, max_tokens=380, temperature=0.7) or ""
                    return out.strip()
                except Exception:
                    logger.exception("LifeJournal: errore LLM summary, fallback a naive summary")

            # Fallback: naive structured summary
            counts: Dict[str,int] = {}
            important_texts: List[str] = []
            for e in entries:
                counts[e["category"]] = counts.get(e["category"], 0) + 1
                if e["importance"] >= 0.7:
                    important_texts.append(f"- [{e['ts']}] {e['category']}: {e['description'][:300]}")

            summary = f"Riepilogo ({len(entries)} entry) dal {start_dt.date()} al {end_dt.date()}.\nCategorie:\n"
            for k,v in counts.items():
                summary += f"- {k}: {v}\n"
            if important_texts:
                summary += "\nEntry importanti:\n" + "\n".join(important_texts[:10])
            else:
                summary += "\nNessuna entry fortemente importante registrata.\n"
            return summary

    # -----------------------
    # Search & filters
    # -----------------------
    def find_entries(self, filter_fn: Callable[[Dict[str,Any]], bool]) -> List[Dict[str,Any]]:
        with self._lock:
            return [e for e in self.state.get("journal", []) if filter_fn(e)]

    def search_by_keyword(self, keyword: str, limit: int = 50) -> List[Dict[str,Any]]:
        k = keyword.lower()
        with self._lock:
            results = [e for e in reversed(self.state.get("journal", [])) if k in e["description"].lower() or k in e.get("category","").lower()]
            return results[:limit]

    # -----------------------
    # Export (Markdown / JSON)
    # -----------------------
    def export_journal_md(self, path: str, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None) -> str:
        with self._lock:
            entries = self._filter_by_range(start_dt, end_dt)
            md = f"# Diario di Nova (export) - {datetime.utcnow().isoformat()}\n\n"
            for e in entries:
                md += f"## {e['ts']} — {e['category']}\n\n{e['description']}\n\n"
                if e.get("metadata"):
                    md += f"_metadata_: {json.dumps(e['metadata'], ensure_ascii=False)}\n\n"
            # write file
            try:
                with open(path, "w", encoding="utf8") as f:
                    f.write(md)
                logger.info("LifeJournal: export Markdown salvato in %s", path)
            except Exception:
                logger.exception("LifeJournal: errore export md")
            return path

    def export_journal_json(self, path: str, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None) -> str:
        with self._lock:
            entries = self._filter_by_range(start_dt, end_dt)
            try:
                with open(path, "w", encoding="utf8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                logger.info("LifeJournal: export JSON salvato in %s", path)
            except Exception:
                logger.exception("LifeJournal: errore export json")
            return path

    def _filter_by_range(self, start_dt: Optional[datetime], end_dt: Optional[datetime]) -> List[Dict[str,Any]]:
        with self._lock:
            all_entries = list(self.state.get("journal", []))
            if not start_dt and not end_dt:
                return all_entries
            if not start_dt:
                start_dt = datetime.min
            if not end_dt:
                end_dt = datetime.max
            return [e for e in all_entries if start_dt.isoformat() <= e["ts"] <= end_dt.isoformat()]

    # -----------------------
    # Voting / human-in-the-loop: upvote/downvote an entry to adjust importance
    # -----------------------
    def vote_entry(self, entry_id: str, vote: int = 1) -> Optional[Dict[str,Any]]:
        """
        vote: +1 or -1 (or any integer). Adjusts importance modestly.
        """
        with self._lock:
            for e in self.state.get("journal", []):
                if e["id"] == entry_id:
                    old = e.get("importance", 0.5)
                    new = max(0.0, min(1.0, old + 0.05 * vote))
                    e["importance"] = new
                    logger.info("LifeJournal: voto entry %s -> %s", entry_id, new)
                    self._persist_state()
                    return e
            return None

    # -----------------------
    # Internal persist state
    # -----------------------
    def _persist_state(self):
        try:
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
        except Exception:
            logger.exception("LifeJournal: errore persist state")

    # -----------------------
    # Public convenience: save state explicitly
    # -----------------------
    def save_state(self):
        with self._lock:
            self._persist_state()
            logger.info("LifeJournal: stato salvato manualmente.")

    # -----------------------
    # Prune old entries (keeps last N)
    # -----------------------
    def prune_journal(self, keep_last: int = JOURNAL_MAX_ENTRIES):
        with self._lock:
            j = self.state.get("journal", [])
            if len(j) > keep_last:
                self.state["journal"] = j[-keep_last:]
                self._persist_state()
                logger.info("LifeJournal: journal potato, mantenute ultime %d entry.", keep_last)

    # -----------------------
    # Small utilities
    # -----------------------
    def get_recent(self, n: int = 20) -> List[Dict[str,Any]]:
        with self._lock:
            return list(self.state.get("journal", []))[-n:]

    def count(self) -> int:
        return len(self.state.get("journal", []))

# -----------------------
# Quick demo / test
# -----------------------
if __name__ == "__main__":
    # demo state
    st = {}
    # simulate core with minimal modules for integration
    class DummyTimeline:
        def add_experience(self, *a, **k):
            print("Timeline add", a, k)
    class DummyDream:
        def register_reflection(self, r): print("Dream register", r)
        def add_influence(self, t): print("Dream influence", t)
    class DummyConscious:
        def register_reflection(self, r): print("Conscious register", r)
        def on_perception(self, p): print("Conscious perception", p)
    class DummyAttention:
        def update_focus(self, payload): print("Attention update", payload)
    class DummyEmotion:
        def process_experience(self, ex): print("Emotion process", ex)
    class DummySelfRef:
        def analyze_experience(self, ex): print("SelfRef analyze", ex)

    core = type("C", (), {})()
    core.memory_timeline = DummyTimeline()
    core.dream_generator = DummyDream()
    core.conscious_loop = DummyConscious()
    core.attention_manager = DummyAttention()
    core.emotion_engine = DummyEmotion()
    core.self_reflection = DummySelfRef()

    lj = LifeJournal(st, core=core)
    lj.record_event("Ho parlato con una persona gentile. Mi ha fatto sorridere.", category="interaction", importance=0.8)
    lj.record_event("Ho sognato di volare sopra una città luminosa.", category="dream", importance=0.7)
    print(lj.generate_daily_summary())
    lj.export_journal_md("nova_journal_demo.md")
    print("Export creato.")
