# inner_dialogue.py
"""
InnerDialogue - dialogo interno tra sub-personalità per Nova (LLM-driven)

API principali:
- InnerDialogue(state, ... , core=None)
- process(reflections: List[str]) -> List[Union[str, dict]]
- generate_dialogue_once() -> str
- cycle() -> List[Union[str, dict]]

Il metodo process(...) è compatibile con conscious_loop (che lo invoca passando riflessioni).
I piani (dict con "type":"plan") vengono valutati con EthicsEngine (se presente nel core)
e, se approvati, inseriti in core.state['tasks'].
"""

import os
import re
import json
import time
import random
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger

DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

# -----------------------
# LLM loader (local Gemma via llama_cpp) with fallback
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    try:
        from llama_cpp import Llama
        logger.info("inner_dialogue: llama_cpp disponibile, caricamento modello %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        def _call(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
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
                logger.exception("inner_dialogue: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.warning("inner_dialogue: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.6) -> str:
            # simple heuristics to produce short dialogue
            seeds = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = seeds[-1][:100] if seeds else "pensiero"
            personalities = ["Riflessivo","Critico","Creativo","Curioso"]
            s1, s2 = random.sample(personalities, 2)
            lines = [
                f"{s1}: Sto pensando a {seed}. Forse dovrei considerare le emozioni collegate.",
                f"{s2}: Mi chiedo se agire ora è una buona idea; consideriamo alternative.",
                f"{s1}: Un possibile piano è fare un passo piccolo: provare e osservare."
            ]
            return "\n".join(lines)
        return _fallback

# -----------------------
# Util: safe call / persist helpers
# -----------------------
def _safe_call(func, *a, **k):
    try:
        return func(*a, **k)
    except Exception:
        logger.exception("inner_dialogue: errore in safe_call su %s", getattr(func, "__name__", str(func)))
        return None

def _now_iso():
    return datetime.utcnow().isoformat()

# -----------------------
# InnerDialogue
# -----------------------
class InnerDialogue:
    def __init__(
        self,
        state: Dict[str, Any],
        emotion_engine: Optional[Any] = None,
        motivational_engine: Optional[Any] = None,
        memory_timeline: Optional[Any] = None,
        dream_generator: Optional[Any] = None,
        life_journal: Optional[Any] = None,
        attention_manager: Optional[Any] = None,
        core: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH
    ):
        """
        state: core.state
        core: optional NovaCore reference (used to insert approved plans into core.state['tasks'] and to use core.ethics)
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # collaborators (may be None or fallback)
        self.emotion_engine = emotion_engine
        self.motivational_engine = motivational_engine
        self.memory_timeline = memory_timeline
        self.dream_generator = dream_generator
        self.life_journal = life_journal
        self.attention_manager = attention_manager

        # sub-personalities and their strengths (can be reinforced over time)
        self.sub_personalities = self.state.get("sub_personalities", {
            "Riflessivo": 1.0,
            "Critico": 1.0,
            "Creativo": 1.0,
            "Curioso": 1.0
        })
        # persistent history
        self.state.setdefault("inner_dialogue_history", [])
        # LLM
        self.llm = _load_llm(model_path)
        logger.info("InnerDialogue inizializzato (LLM=%s).", "yes" if self.llm else "fallback")

    # -----------------------
    # Public: process reflections -> produce outputs (strings and/or plan dicts)
    # -----------------------
    def process(self, reflections: Optional[List[str]] = None, max_outputs: int = 4) -> List[Union[str, Dict[str, Any]]]:
        """
        reflections: list of short strings (from self_reflection or conscious_loop)
        Returns list of outputs: each either a string (dialogue line) or dict plan {"type":"plan", "title":..., "steps":[...], "confidence":...}
        Behavior:
         - builds a rich prompt with emotions, motivations, memories, dreams and reflections
         - asks LLM to generate a dialog between 2-3 sub-personalities and to extract any emergent plans as JSON
         - evaluates plans via ethics (if available) and inserts allowed plans into core.state['tasks']
        """
        with self._lock:
            try:
                reflections = reflections or []
                # build prompt
                prompt = self._build_prompt(reflections)
                raw = self.llm(prompt, max_tokens=380, temperature=0.8)
                text = (raw or "").strip()
                if not text:
                    # fallback minimal dialogue
                    return [self.generate_dialogue_once(reflections)]

                # split into lines and try to parse JSON plans
                outputs: List[Union[str, Dict[str, Any]]] = []
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                buffer_text = []
                for line in lines:
                    # detect JSON object or array
                    if (line.startswith("{") and line.endswith("}")) or (line.startswith("[") and line.endswith("]")):
                        try:
                            parsed = json.loads(line)
                            # if parsed is a plan or list of plans
                            if isinstance(parsed, dict):
                                parsed = [parsed]
                            for item in parsed:
                                plan = self._normalize_plan(item)
                                if plan:
                                    approved, verdict = self._evaluate_and_maybe_commit_plan(plan)
                                    plan_meta = {"type":"plan", **plan, "approved": approved, "ethics_verdict": verdict}
                                    outputs.append(plan_meta)
                        except Exception:
                            # not valid JSON -> treat as text
                            outputs.append(line)
                    else:
                        # check for explicit "PLAN:" marker (loose parsing)
                        if re.match(r"(?i)^plan[:\-]\s*", line):
                            # extract after marker and try parse json inside
                            tail = re.sub(r"(?i)^plan[:\-]\s*", "", line).strip()
                            try:
                                parsed = json.loads(tail)
                                plan = self._normalize_plan(parsed)
                                if plan:
                                    approved, verdict = self._evaluate_and_maybe_commit_plan(plan)
                                    outputs.append({"type":"plan", **plan, "approved": approved, "ethics_verdict": verdict})
                                continue
                            except Exception:
                                # attempt heuristic parse: create simple plan
                                plan = {"title": tail[:120], "steps": ["riflettere","provare"], "confidence": 0.5}
                                approved, verdict = self._evaluate_and_maybe_commit_plan(plan)
                                outputs.append({"type":"plan", **plan, "approved": approved, "ethics_verdict": verdict})
                                continue
                        # otherwise regular dialogue line
                        outputs.append(line)
                # Post-process: if no structured outputs found, return a generated dialogue block
                if not outputs:
                    outputs = [text]
                # persist history & timeline
                ts = _now_iso()
                hist_entry = {"timestamp": ts, "reflections": reflections, "raw": text, "parsed_count": len(outputs)}
                self.state.setdefault("inner_dialogue_history", []).append(hist_entry)
                try:
                    if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                        self.memory_timeline.add_experience(hist_entry, category="inner_dialogue", importance=2)
                except Exception:
                    logger.exception("inner_dialogue: errore salvataggio timeline")
                # write to life_journal
                try:
                    if self.life_journal and hasattr(self.life_journal, "record_entry"):
                        self.life_journal.record_entry("inner_dialogue", text)
                except Exception:
                    logger.exception("inner_dialogue: errore record_entry life_journal")

                # attention update hint
                try:
                    if self.attention_manager and hasattr(self.attention_manager, "update_focus_from_dialogue"):
                        self.attention_manager.update_focus_from_dialogue("\n".join([str(o) for o in outputs[:3]]))
                except Exception:
                    # some attention managers use update_focus
                    if self.attention_manager and hasattr(self.attention_manager, "update"):
                        try:
                            self.attention_manager.update()
                        except Exception:
                            pass

                # normalize outputs length
                return outputs[:max_outputs]

            except Exception:
                logger.exception("inner_dialogue: exception in process")
                # fallback single line
                fallback = self.generate_dialogue_once(reflections)
                return [fallback]

    # -----------------------
    # Build a rich prompt for inner dialogue
    # -----------------------
    def _build_prompt(self, reflections: List[str]) -> str:
        parts = []
        parts.append("Sei Nova: genera un breve dialogo interno tra 2-3 sub-personalità (es. Riflessivo, Critico, Creativo).")
        # include sub-personality strengths
        parts.append("Sub-personalità (forze): " + ", ".join(f"{k}:{round(v,2)}" for k,v in self.sub_personalities.items()))
        # emotions & motivations
        if self.state.get("emotions"):
            em = self.state.get("emotions")
            parts.append("Emozioni attuali: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in em.items()))
        if self.state.get("motivations"):
            mv = self.state.get("motivations")
            parts.append("Motivazioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in mv.items()))
        # include short recent memories
        entries = []
        try:
            if self.memory_timeline and hasattr(self.memory_timeline, "get_recent"):
                entries = self.memory_timeline.get_recent(6)
            else:
                entries = list(self.state.get("timeline", []))[-6:]
        except Exception:
            entries = []
        if entries:
            parts.append("Ricordi recenti:")
            for e in entries[-4:]:
                if isinstance(e, dict):
                    parts.append("- " + (e.get("content") or e.get("text") or str(e))[:220])
                else:
                    parts.append("- " + str(e)[:220])
        # dreams snippets
        try:
            if self.dream_generator and hasattr(self.dream_generator, "get_recent_dreams"):
                d = self.dream_generator.get_recent_dreams(3) or []
                if d:
                    parts.append("Sogni recenti (estratti):")
                    for s in d:
                        parts.append("- " + (s if isinstance(s, str) else str(s.get("text", "")))[:200])
        except Exception:
            # some dream_generator exposes state['dreams']
            ds = self.state.get("dreams", [])[-3:]
            if ds:
                parts.append("Sogni recenti:")
                for s in ds:
                    parts.append("- " + (s.get("text") if isinstance(s, dict) else str(s))[:200])
        # reflections
        if reflections:
            parts.append("Riflessioni recenti:")
            for r in reflections:
                parts.append("- " + str(r)[:220])
        # instruction: produce dialogue + optionally JSON plans
        instruction = (
            "\n".join(parts)
            + "\n\nIstruzione: genera un dialogo breve (italiano) fra 2-3 sub-personalità. "
              "Se emergono piani concreti (passi azionabili), estraili come oggetti JSON con chiavi: "
              "'title','steps' (lista stringhe), 'confidence' (0.0-1.0). "
              "Esempio JSON: {\"title\":\"...\",\"steps\":[\"...\",\"...\"],\"confidence\":0.8}\n\nRisposta:\n"
        )
        return instruction

    # -----------------------
    # Normalize plan-like dicts into canonical plan format
    # -----------------------
    def _normalize_plan(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        title = obj.get("title") or obj.get("name") or obj.get("task") or "Piano interno"
        steps = obj.get("steps") or obj.get("actions") or []
        if isinstance(steps, str):
            steps = [steps]
        steps = [str(s) for s in steps][:20]
        confidence = float(obj.get("confidence", 0.5))
        norm = {"title": str(title)[:180], "steps": steps, "confidence": max(0.0, min(1.0, confidence))}
        return norm

    # -----------------------
    # Evaluate plan with EthicsEngine (if present) and commit to tasks if allowed
    # -----------------------
    def _evaluate_and_maybe_commit_plan(self, plan: Dict[str, Any]) -> (bool, Dict[str, Any]):
        """
        Returns (approved:bool, verdict_dict)
        - If core.ethics exists, evaluate plan textual content (title + steps).
        - If vetoed -> return False with verdict.
        - If allowed -> append to core.state['tasks'] if core present.
        """
        combined_text = plan.get("title","") + ". " + " ".join(plan.get("steps",[]))
        verdict = {"verdict": "no_ethics"}
        try:
            if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                verdict = self.core.ethics.evaluate_action({"text": combined_text, "tags": ["plan"], "metadata": {"source":"inner_dialogue"}})
                if not verdict.get("allowed", True):
                    # vetoed
                    logger.info("inner_dialogue: plan vetoed by ethics: %s", verdict.get("matched_rules"))
                    return False, verdict
            # optionally check core.should_execute_action
            if self.core and hasattr(self.core, "should_execute_action"):
                ok = self.core.should_execute_action({"text": combined_text, "tags": ["plan"], "metadata": {"source":"inner_dialogue"}})
                if not ok:
                    return False, {"verdict":"should_execute_rejected"}
            # commit to tasks (with metadata) if core present
            if self.core and isinstance(self.core, object):
                try:
                    task_obj = {
                        "title": plan.get("title"),
                        "steps": plan.get("steps"),
                        "confidence": plan.get("confidence", 0.5),
                        "origin": "inner_dialogue",
                        "timestamp": _now_iso()
                    }
                    self.core.state.setdefault("tasks", []).append(task_obj)
                    logger.info("inner_dialogue: piano aggiunto a core.state['tasks']: %s", task_obj["title"][:80])
                except Exception:
                    logger.exception("inner_dialogue: errore inserimento task in core.state")
            return True, verdict
        except Exception:
            logger.exception("inner_dialogue: errore valutazione plan")
            return False, {"verdict":"exception"}

    # -----------------------
    # Simple generate dialogue once (non-structured) fallback
    # -----------------------
    def generate_dialogue_once(self, reflections: Optional[List[str]] = None) -> str:
        """
        Fallback or quick generation: returns a single multi-line dialogue string.
        """
        with self._lock:
            try:
                reflections = reflections or []
                prompt = "Genera un breve dialogo interno fra due sub-personalità sulla base delle riflessioni seguenti:\n"
                for r in reflections[-4:]:
                    prompt += "- " + str(r) + "\n"
                prompt += "\nRisposta (italiano):\n"
                raw = self.llm(prompt, max_tokens=180, temperature=0.6)
                return (raw or "").strip() or f"Riflessivo: Sto pensando a {reflections[:1]}. Critico: chiedo alternative."
            except Exception:
                logger.exception("inner_dialogue: errore generate_dialogue_once")
                return "Riflessivo: penso. Critico: dubito."

    # -----------------------
    # cycle(): alias che genera e registra
    # -----------------------
    def cycle(self) -> List[Union[str, Dict[str, Any]]]:
        """
        Esegui un ciclo di inner dialogue: prendi riflessioni recenti dalla state/timeline e processale.
        """
        refs = []
        try:
            # prefer reflections in state.context or memory_timeline
            ctx = self.state.get("context", {}) or {}
            if ctx.get("last_thought"):
                refs.append(ctx.get("last_thought"))
            # also take recent reflections from memory_timeline if available
            if self.memory_timeline and hasattr(self.memory_timeline, "get_recent"):
                try:
                    rec = self.memory_timeline.get_recent(4) or []
                    # convert to strings
                    refs.extend([ (r.get("content") if isinstance(r, dict) else str(r)) for r in rec[:4] ])
                except Exception:
                    pass
            # call process
            outputs = self.process(refs, max_outputs=6)
            return outputs
        except Exception:
            logger.exception("inner_dialogue: errore in cycle")
            return [self.generate_dialogue_once(refs)]

    # -----------------------
    # helper: get recent history
    # -----------------------
    def get_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self.state.get("inner_dialogue_history", []))[-limit:]

# -----------------------
# Quick standalone demo
# -----------------------
if __name__ == "__main__":
    # dummy collaborators
    class Dummy:
        def current_emotions_summary(self): return "felice"
        def current_desires_summary(self): return "imparare"
    class DummyMem:
        def get_recent(self, n): return [{"content":"Ho visto una barca"},{"content":"Ho ascoltato una canzone"}]
        def add_experience(self, ex, category=None, importance=1): print("timeline add:", ex)
    class DummyDream:
        def get_recent_dreams(self, n): return ["sogno di un mare"]
    class DummyJournal:
        def record_entry(self, t, c): print("journal:", t, c[:120])
    class DummyAttention:
        def update_focus_from_dialogue(self, d): print("attention update:", d[:120])

    state = {"context":{"last_thought":"Sto pensando a cibo"}}
    core = type("C",(),{"state":state})()
    idlg = InnerDialogue(state, Dummy(), Dummy(), DummyMem(), DummyDream(), DummyJournal(), DummyAttention(), core=core)
    outs = idlg.cycle()
    print("Outputs:", outs)
