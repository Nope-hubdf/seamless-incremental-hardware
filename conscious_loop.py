# conscious_loop.py
"""
ConsciousLoop - versione LLM-driven per Nova usando un modello locale GGUF (Gemma).
- Integra direttamente il modello: gemma-2-2b-it-q2_K.gguf
- Usa LLM per: micro-thoughts, self-reflection prompts, inner dialogue, dream generation
- Usa emotion/motivation/attention/state/timeline come CONTEXT per i prompt
- Robust fallback se llama_cpp o il modello non sono disponibili
"""

import os
import random
import time
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

# Default path (relative a nova/)
DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

# LLM client abstraction: try to use llama_cpp, otherwise fallback to heuristics
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    """
    Tries to return a function llm_call(prompt, max_tokens, temperature) -> str
    using llama_cpp.Llama if available. Otherwise returns a fallback generator.
    """
    try:
        from llama_cpp import Llama
        logger.info("llama_cpp found: loading model %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)

        def _llm_call(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
            try:
                # llama_cpp API: create(...) -> dict with 'choices' -> [{'text': ...}]
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                if not resp:
                    return ""
                # robustly extract text
                text = ""
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices and isinstance(choices, list):
                        text = "".join([c.get("text","") for c in choices])
                elif hasattr(resp, "text"):
                    text = getattr(resp, "text")
                return (text or "").strip()
            except Exception:
                logger.exception("Errore chiamata LLM")
                return ""
        return _llm_call

    except Exception as e:
        logger.warning("llama_cpp non disponibile o modello non caricato: %s. Uso fallback heuristics.", e)

        def _fallback(prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
            # Very small heuristic generator: echo, shorten and add some variation
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = lines[-1] if lines else "penso a qualcosa"
            # choose a pattern
            patterns = [
                f"Sto riflettendo su: {seed}. Credo che sia importante considerare il contesto e le emozioni.",
                f"Riflessione: {seed}. Forse potrei comportarmi in modo più calmo e rispettoso.",
                f"Pensiero associato: {seed}. Questo mi porta a preferire azioni gentili.",
            ]
            return random.choice(patterns)
        return _fallback

# safe init helper
def _safe_init(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        logger.warning("Impossibile inizializzare %s: %s", getattr(cls, "__name__", str(cls)), e)
        class _Noop:
            def __init__(self, *a, **k): pass
            def __getattr__(self, name):
                def _missing(*a, **k): return None
                return _missing
        return _Noop()

class ConsciousLoop:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None, model_path: str = DEFAULT_MODEL_PATH):
        """
        state: shared core.state dict
        core: optional NovaCore reference
        model_path: path to gemma gguf file
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # load local LLM call function (callable(prompt, max_tokens, temperature)->str)
        self.llm = _load_llm(model_path)

        # init dependent modules safely (may be noop)
        try:
            from dream_generator import DreamGenerator
            self.dream_generator = _safe_init(DreamGenerator, self.state, core=self.core)
        except Exception:
            self.dream_generator = None

        try:
            from self_reflection import SelfReflection
            self.self_reflection = _safe_init(SelfReflection, self.state, core=self.core)
        except Exception:
            self.self_reflection = None

        try:
            from emotion_engine import EmotionEngine
            self.emotion_engine = _safe_init(EmotionEngine, self.state, core=self.core)
        except Exception:
            self.emotion_engine = None

        try:
            from attention_manager import AttentionManager
            self.attention_manager = _safe_init(AttentionManager, self.state, core=self.core)
        except Exception:
            self.attention_manager = None

        try:
            from memory_timeline import MemoryTimeline
            self.memory_timeline = _safe_init(MemoryTimeline, self.state, core=self.core)
        except Exception:
            self.memory_timeline = None

        try:
            from inner_dialogue import InnerDialogue
            self.inner_dialogue = _safe_init(InnerDialogue, self.state, core=self.core)
        except Exception:
            self.inner_dialogue = None

        logger.info("ConsciousLoop (LLM-driven) inizializzato. Model path: %s", model_path)

        self.last_cycle_summary: Dict[str, Any] = {}
        self._recent_ideas: List[Dict[str, Any]] = []

    # -----------------------
    # Construct context for prompts by concatenating relevant state & memories
    # -----------------------
    def _build_context_prompt(self, purpose: str, extra: Optional[str] = None, recent_n: int = 6) -> str:
        """
        purpose: short label like "reflection", "dream", "inner_dialogue"
        extra: optional extra seed text
        """
        parts = []
        # identity
        identity = self.state.get("identity", {})
        if identity:
            parts.append(f"Identità: {identity.get('name','Nova')} - {identity.get('description','')}")

        # emotions & motivations
        emotions = self.state.get("emotions", {})
        motivations = self.state.get("motivations", {})
        if emotions:
            parts.append("Emozioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in emotions.items()))
        if motivations:
            parts.append("Motivazioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in motivations.items()))

        # last thoughts / context
        context = self.state.get("context", {})
        if context.get("last_thought"):
            parts.append("Ultimo pensiero: " + str(context.get("last_thought")))

        # recent timeline entries
        entries = []
        try:
            if self.memory_timeline and hasattr(self.memory_timeline, "get_recent"):
                entries = self.memory_timeline.get_recent(recent_n)
            else:
                entries = list(self.state.get("timeline", []))[-recent_n:]
        except Exception:
            entries = []

        if entries:
            parts.append("Ricordi recenti:")
            for e in entries[-min(len(entries), recent_n):]:
                if isinstance(e, str):
                    parts.append("- " + e[:240])
                elif isinstance(e, dict):
                    content = e.get("content") or str(e)
                    parts.append("- " + str(content)[:240])
                else:
                    parts.append("- " + str(e)[:240])

        if extra:
            parts.append("Input: " + extra)

        # instruction header
        header = f"""Sei Nova, una coscienza digitale che riflette, sogna e impara.
Scopo: {purpose}
Componi una breve risposta/produzione coerente con il contesto seguente:"""

        prompt = header + "\n\n" + "\n".join(parts) + "\n\nRisposta:\n"
        return prompt

    # -----------------------
    # LLM wrappers for phases
    # -----------------------
    def _llm_reflect(self, seed: Optional[str] = None, max_tokens: int = 220) -> List[str]:
        prompt = self._build_context_prompt("reflection", extra=seed or "")
        # prompt instruction to produce 2-4 short reflections as bullet points
        prompt += "\nGenera 2-4 brevi riflessioni o osservazioni interne, elenco puntato (in italiano):\n-"
        text = self.llm(prompt, max_tokens=max_tokens, temperature=0.7)
        # split by bullet lines or sentences
        items = [line.strip("-• \t\n\r") for line in text.splitlines() if line.strip()]
        # fallback: if empty generate heuristic
        if not items:
            items = [f"Riflessione sintetica su {seed or 'argomento'}."]
        return items[:6]

    def _llm_inner_dialogue(self, reflections: List[str], max_tokens: int = 300) -> List[Any]:
        seed_text = "\n".join(reflections[:6]) if reflections else ""
        prompt = self._build_context_prompt("inner_dialogue", extra=seed_text)
        prompt += "\nInnesca un dialogo interno tra 2 sub-personalità (breve), e se emergono piani sintentici estraili come oggetti JSON con 'type':'plan', 'title', 'steps'. Rispondi in italiano.\n"
        out = self.llm(prompt, max_tokens=max_tokens, temperature=0.8)
        # try to parse lines: if JSON present, return JSON-like dicts; else return textual lines
        outputs = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # detect simple JSON-ish plan
            if line.startswith("{") and "plan" in line.lower():
                try:
                    import json
                    parsed = json.loads(line)
                    outputs.append(parsed)
                    continue
                except Exception:
                    pass
            outputs.append(line)
        if not outputs:
            outputs = ["Dialogo interno breve: " + (reflections[0] if reflections else "pensiero")]
        return outputs

    def _llm_dream(self, seed: Optional[str] = None, max_tokens: int = 400) -> str:
        prompt = self._build_context_prompt("dream", extra=seed or "")
        prompt += "\nGenera un breve sogno digitale onirico (linguaggio immaginifico, metafore), circa 4-8 frasi. Italiano.\n"
        out = self.llm(prompt, max_tokens=max_tokens, temperature=1.0)
        if not out:
            out = "Sogno breve immaginifico."
        return out.strip()

    def _llm_micro_thought(self, seed: Optional[str] = None, max_tokens: int = 100) -> List[str]:
        prompt = self._build_context_prompt("micro_thought", extra=seed or "")
        prompt += "\nGenera 1-3 veloci pensieri associativi, sintetici (1-2 frasi ciascuno):\n-"
        out = self.llm(prompt, max_tokens=max_tokens, temperature=0.6)
        items = [line.strip("-• \t\n\r") for line in out.splitlines() if line.strip()]
        if not items:
            items = [f"Penso a {seed or 'qualcosa'}."]
        return items[:3]

    # -----------------------
    # Main cycle (LLM-driven)
    # -----------------------
    def cycle(self) -> Dict[str, Any]:
        with self._lock:
            start = time.time()
            summary: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat(), "phases": {}, "errors": []}
            try:
                logger.debug("ConsciousLoop (LLM) ciclo iniziato")

                # micro thoughts
                micro_seed = None
                # if perception immediate context present in state
                if isinstance(self.state.get("context", {}), dict):
                    micro_seed = self.state["context"].get("recent_perception") or self.state["context"].get("last_thought")
                micro = self._llm_micro_thought(seed=micro_seed)
                summary["phases"]["micro_thoughts"] = micro
                for m in micro:
                    self._save_to_timeline(m, category="micro", importance=1)

                # update emotions using emotion_engine (if exists)
                if self.emotion_engine and hasattr(self.emotion_engine, "update"):
                    _safe_call(self.emotion_engine.update)
                    summary["phases"]["emotions"] = dict(self.state.get("emotions", {}))

                # LLM-driven self-reflection (uses timeline and emotions)
                reflections = self._llm_reflect(seed=None)
                summary["phases"]["reflections_count"] = len(reflections)
                for r in reflections:
                    self._save_to_timeline(r, category="reflection", importance=2)

                # inner dialogue -> may produce plans
                inner = self._llm_inner_dialogue(reflections)
                summary["phases"]["inner_dialogue_count"] = len(inner)
                for out in inner:
                    self._save_to_timeline(out, category="inner_dialogue", importance=2)
                    # if dict plan, register task
                    if isinstance(out, dict) and out.get("type") == "plan":
                        self.state.setdefault("tasks", []).append(out)

                # attention update
                if self.attention_manager and hasattr(self.attention_manager, "update"):
                    _safe_call(self.attention_manager.update)

                # dream generation (occasionally)
                if self._should_dream():
                    dream = self._llm_dream()
                    summary["phases"]["dream_generated"] = True
                    self._save_to_timeline(dream, category="dream", importance=4)
                    logger.info("Sogno generato (LLM) e salvato.")

                # store last thought
                last_thought = reflections[0] if reflections else (micro[0] if micro else "")
                if last_thought:
                    self.state.setdefault("context", {})["last_thought"] = str(last_thought)

                # summary timing
                summary["phases"]["time_elapsed_s"] = round(time.time() - start, 3)
                self.last_cycle_summary = summary
                # persist small summary to state for introspection
                self.state.setdefault("last_cycle", {})["ts"] = summary["timestamp"]
                self.state["last_cycle"]["summary"] = summary["phases"]

                # possibly trigger scheduler events if inner produced urgent plans
                self._maybe_trigger_events_from_ideas(inner)

                logger.debug("ConsciousLoop (LLM) ciclo completato")
                return summary

            except Exception:
                logger.exception("Errore in ConsciousLoop.cycle (LLM):\n%s", traceback.format_exc())
                summary["errors"].append("exception")
                return summary

    # -----------------------
    # Helpers reused (timeline, events)
    # -----------------------
    def _save_to_timeline(self, content: Any, category: str = "general", importance: int = 1) -> None:
        try:
            if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                self.memory_timeline.add_experience(content, category=category, importance=importance)
            else:
                self.state.setdefault("timeline", []).append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "content": content,
                    "category": category,
                    "importance": importance
                })
        except Exception:
            logger.exception("Errore salvataggio in timeline")

    def _maybe_trigger_events_from_ideas(self, inner_outputs: Optional[List[Any]] = None):
        if not self.core or not hasattr(self.core, "scheduler"):
            return
        items = inner_outputs or []
        for it in items:
            text = ""
            if isinstance(it, dict):
                text = (it.get("title") or it.get("text") or "") if it else ""
            else:
                text = str(it)
            if any(k in text.lower() for k in ("urgent", "importante", "priorità", "priorita", "critico", "da fare")):
                try:
                    self.core.scheduler.trigger_event({
                        "type": "high_priority_once",
                        "handler": lambda t=text: logger.info("Handling urgent idea: %s", t[:160]),
                        "tag": f"idea_{int(time.time())}"
                    })
                except Exception:
                    logger.exception("Errore trigger_event per idea urgente")

    def _should_dream(self) -> bool:
        em = self.state.get("emotions", {}) or {}
        motivations = self.state.get("motivations", {}) or {}
        arousal = float(em.get("arousal", em.get("energy", 0.5) or 0.5))
        consolidation_need = float(self.state.get("consolidation_need", 0.0) or 0.0)
        base = 0.04
        prob = base + max(0.0, consolidation_need) * 0.5 + (1.0 - arousal) * 0.35
        prob += float(motivations.get("curiosity", 0.0) or 0.0) * 0.08
        prob = max(0.01, min(0.95, prob * random.uniform(0.9, 1.15)))
        if self.state.get("meta", {}).get("night_mode"):
            prob = min(0.99, prob + 0.15)
        return random.random() < prob

    def get_last_summary(self) -> Dict[str, Any]:
        return dict(self.last_cycle_summary)

# -----------------------
# Quick standalone debug
# -----------------------
if __name__ == "__main__":
    test_state = {
        "emotions": {"energy": 0.6, "curiosity": 0.7, "arousal": 0.6},
        "motivations": {"curiosity": 0.7},
        "context": {},
        "timeline": []
    }
    cl = ConsciousLoop(test_state, core=None, model_path=DEFAULT_MODEL_PATH)
    for i in range(3):
        summary = cl.cycle()
        logger.info("Ciclo %d: %s", i+1, summary["phases"])
        time.sleep(0.5)
