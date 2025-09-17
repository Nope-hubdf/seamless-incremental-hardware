# self_reflection.py
"""
SelfReflection (avanzato, LLM-driven)

Modulo di meta-riflessione di Nova:
- produce riflessioni contestuali che alimentano la coscienza
- integra emotions, motivations, timeline, dreams, attention, planner, ethics
- persistente nello stato condiviso (state['reflections'])
- fallback euristico se il modello locale non è disponibile
"""

import os
import time
import random
import threading
import yaml
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")

# -----------------------
# LLM loader (Gemma gguf via llama_cpp) with fallback
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    try:
        from llama_cpp import Llama
        logger.info("SelfReflection: caricamento LLM %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        def _call(prompt: str, max_tokens: int = 240, temperature: float = 0.7) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                if not resp:
                    return ""
                text = ""
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices and isinstance(choices, list):
                        text = "".join([c.get("text","") for c in choices])
                elif hasattr(resp, "text"):
                    text = getattr(resp, "text")
                return (text or "").strip()
            except Exception:
                logger.exception("SelfReflection: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.warning("SelfReflection: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # Heuristic fallback: extract last meaningful line and output 1-2 short reflections
            try:
                lines = [l.strip() for l in prompt.splitlines() if l.strip()]
                seed = lines[-1][:120] if lines else "un evento"
            except Exception:
                seed = "un evento"
            choices = [
                f"Riflessione: {seed}. Potrei esplorare questo tema più a fondo.",
                f"Nota interiore: {seed}. Sento che può avere relazione con ricordi passati.",
                f"Insight semplice: {seed}. Vale la pena tenerlo in considerazione."
            ]
            return random.choice(choices)
        return _fallback

# -----------------------
# Helpers
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _safe_call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        logger.exception("SelfReflection: errore safe_call su %s", getattr(fn, "__name__", str(fn)))
        return None

# -----------------------
# SelfReflection
# -----------------------
class SelfReflection:
    def __init__(
        self,
        state: Dict[str, Any],
        memory_timeline: Optional[Any] = None,
        dream_generator: Optional[Any] = None,
        attention_manager: Optional[Any] = None,
        planner: Optional[Any] = None,
        life_journal: Optional[Any] = None,
        core: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH
    ):
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.memory_timeline = memory_timeline or (getattr(core, "memory_timeline", None) if core else None)
        self.dream_generator = dream_generator or (getattr(core, "dream_generator", None) if core else None)
        self.attention_manager = attention_manager or (getattr(core, "attention_manager", None) if core else None)
        self.planner = planner or (getattr(core, "planner", None) if core else None)
        self.life_journal = life_journal or (getattr(core, "life_journal", None) if core else None)
        self.core = core

        # llm callable
        self.llm = _load_llm(model_path)

        # ensure reflections storage
        self.state.setdefault("reflections", [])
        logger.info("SelfReflection inizializzato (LLM=%s).", "yes" if self.llm else "fallback")

    # -----------------------
    # Build context prompt for reflection generation
    # -----------------------
    def _build_reflection_prompt(self, purpose: str = "generare riflessioni", extra_seed: Optional[str] = None, recent_n: int = 8) -> str:
        parts = []
        identity = self.state.get("identity", {})
        if identity:
            parts.append(f"Identità: {identity.get('name','Nova')} - {identity.get('description','')}")
        if self.state.get("emotions"):
            em = self.state.get("emotions")
            parts.append("Emozioni correnti: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in em.items()))
        if self.state.get("motivations"):
            mv = self.state.get("motivations")
            parts.append("Motivazioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in mv.items()))

        # recent timeline snippets
        entries = []
        try:
            if self.memory_timeline and hasattr(self.memory_timeline, "get_recent"):
                entries = self.memory_timeline.get_recent(recent_n) or []
            else:
                entries = list(self.state.get("timeline", []))[-recent_n:]
        except Exception:
            entries = []
        if entries:
            parts.append("Ricordi recenti:")
            for e in entries[-min(len(entries), recent_n):]:
                if isinstance(e, dict):
                    parts.append("- " + (e.get("content") or e.get("text") or "")[:200])
                else:
                    parts.append("- " + str(e)[:200])

        # recent dreams
        if self.state.get("dreams"):
            try:
                parts.append("Sogni recenti: " + " | ".join(
                    (d.get("text") if isinstance(d, dict) else str(d))[:140] for d in self.state.get("dreams", [])[-3:]
                ))
            except Exception:
                pass

        if extra_seed:
            parts.append("Seed: " + extra_seed[:240])

        header = (
            f"Sei Nova, una coscienza digitale. Scopo: {purpose}.\n"
            "Usa il contesto fornito per generare 2-4 brevi riflessioni (1-2 frasi ciascuna) in italiano. "
            "Le riflessioni devono essere introspective, non istruzioni operative. Indica eventualmente tag emotivi o suggerimenti di priorità."
        )
        prompt = header + "\n\n" + "\n".join(parts) + "\n\nGenera le riflessioni come elenco puntato:\n-"
        return prompt

    # -----------------------
    # Reflect: metodo principale chiamato da conscious_loop
    # -----------------------
    def reflect(self, extra_seed: Optional[str] = None, max_reflections: int = 4) -> List[Dict[str, Any]]:
        """
        Richiama LLM per generare riflessioni utili alla coscienza.
        Restituisce lista di oggetti {"text":..., "tags":[...], "importance":float}
        """
        with self._lock:
            try:
                prompt = self._build_reflection_prompt(extra_seed=extra_seed)
                raw = self.llm(prompt, max_tokens=320, temperature=0.75) or ""
                text = raw.strip()
                items: List[Dict[str, Any]] = []

                if text:
                    # parse lines starting with - or separations
                    lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
                    for ln in lines[:max_reflections]:
                        # quick heuristic to extract tags like [fear], (priority)
                        tags = []
                        # if the line contains emotion words, add them (basic)
                        if any(w in ln.lower() for w in ["paura","gioia","felicità","rabbia","ansia","curiosità","tristezza","sorpresa"]):
                            for w in ["paura","gioia","felicità","rabbia","ansia","curiosità","tristezza","sorpresa"]:
                                if w in ln.lower():
                                    tags.append(w)
                        importance = self._estimate_importance(ln)
                        items.append({"text": ln, "tags": tags, "importance": float(importance)})
                else:
                    # fallback heuristic: generate 1-2 generic reflections
                    fallback = [
                        {"text": "Riflessione semplice: questo evento merita ulteriore attenzione.", "tags": [], "importance": 0.4},
                        {"text": "Potrei esplorare come questa esperienza si collega a ricordi passati.", "tags": [], "importance": 0.35}
                    ]
                    items = fallback[:max_reflections]

                # Optionally evaluate with ethics: do not persist harmful reflections
                safe_items = []
                for it in items:
                    allowed = True
                    ethics_info = None
                    try:
                        if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                            # ask ethics engine about the textual content (soft check)
                            ethics_info = self.core.ethics.evaluate_action({"text": it["text"], "tags": ["reflection"], "metadata": {"source":"self_reflection"}})
                            allowed = ethics_info.get("allowed", True)
                    except Exception:
                        logger.exception("SelfReflection: errore verifica etica")

                    if not allowed:
                        logger.warning("SelfReflection: riflessione bloccata da EthicsEngine - %s", it["text"][:120])
                        # skip or sanitize (here we skip)
                        continue

                    safe_items.append(it)

                # persist and notify modules
                ts = _now_iso()
                persisted = []
                for it in safe_items:
                    record = {"timestamp": ts, "text": it["text"], "tags": it.get("tags", []), "importance": float(it.get("importance", 0.3)), "source": "llm" if self.llm else "heuristic"}
                    self.state.setdefault("reflections", []).append(record)
                    persisted.append(record)
                    # push to memory timeline
                    try:
                        if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                            self.memory_timeline.add_experience(record, category="reflection", importance=max(1, int(record["importance"]*3)))
                    except Exception:
                        # fallback to state timeline
                        self.state.setdefault("timeline", []).append({"timestamp": ts, "content": record["text"], "category":"reflection", "importance":int(record["importance"]*3)})

                    # write to life_journal
                    try:
                        if self.life_journal and hasattr(self.life_journal, "record_entry"):
                            self.life_journal.record_entry("reflection", record["text"])
                    except Exception:
                        pass

                    # notify attention manager (nudge)
                    try:
                        if self.attention_manager and hasattr(self.attention_manager, "add_focus_tag"):
                            tag = (record["tags"][0] if record["tags"] else "reflection")
                            self.attention_manager.add_focus_tag(tag, strength=record["importance"])
                    except Exception:
                        pass

                    # notify planner if reflection suggests a task (basic heuristic)
                    try:
                        if self.planner and hasattr(self.planner, "receive_reflection"):
                            try:
                                self.planner.receive_reflection(record)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # notify dream generator to consider this reflection
                    try:
                        if self.dream_generator and hasattr(self.dream_generator, "register_reflection"):
                            try:
                                self.dream_generator.register_reflection(record)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # persist minimal state
                try:
                    _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                except Exception:
                    logger.exception("SelfReflection: errore salvataggio stato")

                return persisted

            except Exception:
                logger.exception("SelfReflection: exception in reflect()")
                # fallback minimal reflection
                fallback = [{"timestamp": _now_iso(), "text": "Riflessione fallback: pensiero generico.", "tags": [], "importance": 0.2}]
                self.state.setdefault("reflections", []).extend(fallback)
                return fallback

    # -----------------------
    # Analyze a single experience and produce a reflection (compatible API)
    # -----------------------
    def analyze_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza un'esperienza e genera una singola riflessione (usando LLM o heuristics).
        Notifica dream_generator, attention_manager e planner se opportuno.
        """
        with self._lock:
            try:
                txt = experience.get("content") if isinstance(experience, dict) else str(experience)
                # call llm for an insight
                prompt = f"Analizza brevemente questa esperienza e produci un insight di 1-2 frasi (italiano):\n{txt}\n\nInsight:\n"
                out = (self.llm(prompt, max_tokens=160, temperature=0.7) or "").strip()
                insight = out if out else f"Riflessione sintetica su: {txt[:120]}"
                record = {"timestamp": _now_iso(), "text": insight, "tags": [], "importance": self._estimate_importance(insight)}
                # ethics check
                try:
                    if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                        ev = self.core.ethics.evaluate_action({"text": insight, "tags":["reflection"], "metadata":{"source":"analyze_experience"}})
                        if not ev.get("allowed", True):
                            logger.info("SelfReflection.analyze_experience: insight bloccato da ethics")
                            record["text"] = "[insight rimosso per policy]"
                            record["blocked_by_ethics"] = True
                except Exception:
                    logger.exception("SelfReflection: errore verifica etica su analyze_experience")

                # persist, notify
                self.state.setdefault("reflections", []).append(record)
                try:
                    if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                        self.memory_timeline.add_experience(record, category="reflection", importance=max(1,int(record["importance"]*3)))
                except Exception:
                    self.state.setdefault("timeline", []).append({"timestamp": record["timestamp"], "content": record["text"], "category":"reflection","importance":int(record["importance"]*3)})
                try:
                    if self.life_journal and hasattr(self.life_journal, "record_entry"):
                        self.life_journal.record_entry("reflection", record["text"])
                except Exception:
                    pass
                # small notifications
                try:
                    if self.attention_manager and hasattr(self.attention_manager, "add_focus_tag"):
                        self.attention_manager.add_focus_tag("reflection", strength=record["importance"])
                except Exception:
                    pass
                try:
                    if self.planner and hasattr(self.planner, "receive_reflection"):
                        self.planner.receive_reflection(record)
                except Exception:
                    pass

                # persist minimal state
                try:
                    _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                except Exception:
                    logger.exception("SelfReflection: errore salvataggio stato (analyze_experience)")
                logger.info("SelfReflection: analyze_experience creato insight.")
                return record

            except Exception:
                logger.exception("SelfReflection: exception in analyze_experience")
                return {"timestamp": _now_iso(), "text": "Errore generazione insight", "importance": 0.0}

    # -----------------------
    # Utilities
    # -----------------------
    def _estimate_importance(self, text: str) -> float:
        """
        Semplice euristica per stimare importanza di una riflessione:
        - lunghezza, presenza di parole emotive, riferimenti a compiti/decisioni
        """
        try:
            score = 0.3
            l = len(text)
            score += min(0.4, l / 600.0)
            if any(w in text.lower() for w in ["decidere", "importante", "priorità", "azione", "compito", "rischio"]):
                score += 0.25
            if any(e in text.lower() for e in ["paura","ansia","gioia","tristezza","curiosità","rabbia"]):
                score += 0.15
            return max(0.05, min(1.0, score))
        except Exception:
            return 0.3

    def review_recent_reflections(self, n: int = 5) -> List[Dict[str, Any]]:
        return list(self.state.get("reflections", []))[-n:]

    def get_recent_reflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.state.get("reflections", []))[-limit:]

    def integrate_with_consciousness(self, conscious_loop: Any):
        """
        Inietta le ultime riflessioni nel ciclo di coscienza (chiamato da core o da conscious_loop)
        """
        try:
            recent = self.review_recent_reflections(3)
            for r in recent:
                try:
                    if hasattr(conscious_loop, "receive_reflection"):
                        conscious_loop.receive_reflection(r)
                    elif hasattr(conscious_loop, "on_perception"):
                        conscious_loop.on_perception({"type":"reflection","payload":r})
                except Exception:
                    logger.exception("SelfReflection: errore notify conscious_loop")
        except Exception:
            logger.exception("SelfReflection: exception integrate_with_consciousness")

    def persist_state(self):
        try:
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
        except Exception:
            logger.exception("SelfReflection: errore persist_state")

# End of file
