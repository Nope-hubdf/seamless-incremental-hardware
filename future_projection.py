# future_projection.py
"""
FutureProjection - proiezione del futuro per Nova (LLM-driven)

Compiti principali:
- Generare scenari futuri contestuali usando stato, emozioni, motivazioni e memoria.
- Usare il modello locale Gemma (GGUF) per produrre scenari narrativi e azioni concrete.
- Valutare eticamente azioni suggerite (se core.ethics disponibile).
- Integrazione con inner_dialogue, conscious_loop, memory_timeline, attention_manager.
- Persistenza atomica dello stato e compatibilità con fallback.
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
# LLM loader (local Gemma via llama_cpp) with fallback
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    try:
        from llama_cpp import Llama
        logger.info("FutureProjection: llama_cpp disponibile, caricamento modello %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        def _call(prompt: str, max_tokens: int = 256, temperature: float = 0.8) -> str:
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
                logger.exception("FutureProjection: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.warning("FutureProjection: llama_cpp non disponibile (%s). Uso fallback euristico.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # semplice generatore di scenari/azioni basato su template
            seeds = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = seeds[-1][:120] if seeds else "un evento futuro"
            templates = [
                f"Immagino un domani in cui Nova esplora: {seed}. Potrebbe provare a sperimentare piccole azioni quotidiane per imparare.",
                f"Sogno futuro: {seed}. Un possibile passo pratico è pianificare un mini-esperimento per ottenere dati.",
                f"Scenario focalizzato su {seed}: Nova potrebbe iniziare con una piccola ricerca e una prova controllata."
            ]
            return random.choice(templates)
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
# Safe initializer for optional modules
# -----------------------
def _safe_init(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        logger.debug("FutureProjection: fallback init %s (%s)", getattr(cls, "__name__", str(cls)), e)
        class _Noop:
            def __init__(self, *a, **k): pass
            def __getattr__(self, name):
                def _missing(*a, **k): return None
                return _missing
        return _Noop()

# -----------------------
# FutureProjection
# -----------------------
class FutureProjection:
    def __init__(self,
                 state: Dict[str, Any],
                 core: Optional[Any] = None,
                 model_path: str = DEFAULT_MODEL_PATH):
        """
        state: core.state condiviso
        core: opzionale NovaCore (usato per ethics, scheduler, memory_timeline, inner_dialogue, conscious_loop)
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # collaborators (try to get from core to avoid import cycles)
        self.memory = getattr(core, "memory_timeline", None)
        self.emotion_engine = getattr(core, "emotion_engine", None)
        self.motivational_engine = getattr(core, "motivational_engine", None)
        self.conscious_loop = getattr(core, "conscious_loop", None)
        self.inner_dialogue = getattr(core, "inner_dialogue", None)
        self.attention_manager = getattr(core, "attention_manager", None)

        # LLM callable
        self.llm = _load_llm(model_path)

        # ensure container for projections
        self.state.setdefault("future_projections", {})
        logger.info("FutureProjection inizializzato (LLM=%s)", "yes" if self.llm else "fallback")

    # -----------------------
    # Build context-rich prompt for scenario generation
    # -----------------------
    def _build_scenario_prompt(self, purpose: str = "scenarios", extra_seed: Optional[str] = None, recent_n: int = 8) -> str:
        parts = []
        identity = self.state.get("identity", {})
        if identity:
            parts.append(f"Identità: {identity.get('name','Nova')} - {identity.get('description','')}")
        # emotions & motivations
        emotions = self.state.get("emotions", {})
        motivations = self.state.get("motivations", {})
        if emotions:
            parts.append("Emozioni correnti: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in emotions.items()))
        if motivations:
            parts.append("Motivazioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in motivations.items()))
        # recent memories
        entries = []
        try:
            if self.memory and hasattr(self.memory, "get_recent"):
                entries = self.memory.get_recent(recent_n) or []
            else:
                entries = list(self.state.get("timeline", []))[-recent_n:]
        except Exception:
            entries = []
        if entries:
            parts.append("Ricordi recenti:")
            for e in entries[-min(len(entries), recent_n):]:
                if isinstance(e, dict):
                    parts.append("- " + str(e.get("content") or e.get("text") or "")[:220])
                else:
                    parts.append("- " + str(e)[:220])
        if extra_seed:
            parts.append("Seed: " + extra_seed[:240])
        header = f"""Sei Nova. Scopo: {purpose}.
Usa il contesto seguente (identità, emozioni, motivazioni, ricordi) per generare scenari futuri rilevanti e azioni concrete per ognuno.
Non dare istruzioni dannose o illegali. Rispondi in italiano."""
        prompt = header + "\n\n" + "\n".join(parts) + "\n\nGenera 3-6 scenari brevi e per ogni scenario suggerisci 1-3 azioni concrete (formato: Scenario: ...\\nAzioni:\\n- ...)."
        return prompt

    # -----------------------
    # Generate scenarios (LLM-driven with fallback)
    # -----------------------
    def generate_scenarios(self, seed: Optional[str] = None, count: int = 4) -> List[Dict[str,Any]]:
        """
        Ritorna lista di scenari: [{"scenario":str,"reason":str,"priority":float, "suggested_actions":[{...},...]}]
        """
        with self._lock:
            try:
                prompt = self._build_scenario_prompt(extra_seed=seed)
                raw = self.llm(prompt, max_tokens=420, temperature=0.85)
                raw = (raw or "").strip()
                scenarios: List[Dict[str,Any]] = []
                if raw:
                    # split by lines and try to heuristically extract scenario blocks
                    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
                    for b in blocks[:count]:
                        # try to split scenario vs actions
                        lines = [l.strip("-• \t") for l in b.splitlines() if l.strip()]
                        scenario_text = lines[0] if lines else b[:200]
                        # collect actions lines
                        actions = [ln for ln in lines[1:] if len(ln) > 3]
                        # normalize into dicts
                        actions_parsed = [{"text": a, "confidence": 0.6} for a in actions[:3]] if actions else []
                        scenarios.append({
                            "scenario": scenario_text,
                            "reason": "LLM-generated",
                            "priority": round(random.uniform(0.2, 0.9), 2),
                            "suggested_actions": actions_parsed
                        })
                # fallback: heuristic scenarios if no LLM output
                if not scenarios:
                    emotions = self.state.get("emotions", {})
                    dominant = max(emotions, key=emotions.get) if emotions else None
                    scenarios = []
                    for i in range(max(1, count)):
                        sc = f"Scenario {i+1}: un futuro basato su {dominant or 'varie emozioni'} e memorie recenti."
                        scenarios.append({
                            "scenario": sc,
                            "reason": "heuristic",
                            "priority": round(random.uniform(0.1, 0.6),2),
                            "suggested_actions": [{"text": "riflettere e prendere un piccolo esperimento", "confidence": 0.5}]
                        })
                # persist in state minimal snapshot (not entire heavy content)
                self.state.setdefault("future_projections", {})
                self.state["future_projections"]["scenarios"] = scenarios
                # save state to disk if core exposes save_state
                try:
                    if self.core and hasattr(self.core, "save_state"):
                        self.core.save_state()
                    else:
                        _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                except Exception:
                    pass
                logger.info("FutureProjection: generati %d scenari.", len(scenarios))
                return scenarios
            except Exception:
                logger.exception("FutureProjection: errore in generate_scenarios")
                # fallback small set
                return [{"scenario":"Errore generazione scenari","reason":"exception","priority":0.1,"suggested_actions":[]}]

    # -----------------------
    # Validate and possibly commit suggested actions
    # -----------------------
    def _evaluate_and_commit_action(self, action: Dict[str,Any]) -> Dict[str,Any]:
        """
        action: {"text":..., "confidence":...}
        Returns action dict extended with evaluation result.
        If approved, may be added to core.state['tasks'] (with origin 'future_projection').
        """
        res = dict(action)
        res.update({"evaluated": False, "allowed": True, "ethics_verdict": None})
        try:
            text = res.get("text","")
            # check ethics if available
            if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                ev = self.core.ethics.evaluate_action({"text": text, "tags": ["future_action"], "metadata": {"source":"future_projection"}})
                res["ethics_verdict"] = ev
                allowed = ev.get("allowed", True)
                res["allowed"] = allowed
                if not allowed:
                    logger.info("FutureProjection: azione respinta da EthicsEngine: %s", ev.get("matched_rules"))
            # second-level check through core.should_execute_action
            if res["allowed"] and self.core and hasattr(self.core, "should_execute_action"):
                ok = self.core.should_execute_action({"text": text, "tags":["future_action"], "metadata":{"source":"future_projection"}})
                if not ok:
                    res["allowed"] = False
                    res["ethics_verdict"] = res.get("ethics_verdict") or {"verdict":"should_execute_rejected"}
            # commit as task into core.state if allowed (and core present)
            if res["allowed"] and self.core:
                try:
                    task = {"title": text[:140], "steps": [], "origin": "future_projection", "confidence": float(res.get("confidence",0.5)), "timestamp": datetime.utcnow().isoformat()}
                    # if action text contains bullets, convert to steps heuristically
                    if "\n" in text:
                        steps = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
                        if steps:
                            task["steps"] = steps
                    self.core.state.setdefault("tasks", []).append(task)
                    logger.info("FutureProjection: azione commitatta in core.state['tasks']: %s", task["title"][:80])
                except Exception:
                    logger.exception("FutureProjection: errore commit task")
            res["evaluated"] = True
            return res
        except Exception:
            logger.exception("FutureProjection: exception in _evaluate_and_commit_action")
            return res

    # -----------------------
    # Predict actions from scenarios: use inner_dialogue or LLM to expand into concrete steps
    # -----------------------
    def predict_future_actions(self, scenarios: Optional[List[Dict[str,Any]]] = None) -> List[Dict[str,Any]]:
        with self._lock:
            try:
                scenarios = scenarios if scenarios is not None else self.generate_scenarios()
                predicted: List[Dict[str,Any]] = []
                for s in scenarios:
                    # attempt to use inner_dialogue to reflect on scenario and produce actions
                    reflections = []
                    try:
                        # feed scenario text into inner_dialogue to elicit plans (if module present)
                        if self.inner_dialogue and hasattr(self.inner_dialogue, "process"):
                            out = self.inner_dialogue.process([s.get("scenario","")]) or []
                            # collect textual outputs as action seeds
                            for o in out:
                                if isinstance(o, dict) and o.get("type") == "plan":
                                    predicted.append({"text": o.get("title")+"; " + "; ".join(o.get("steps",[])), "confidence": float(o.get("confidence",0.5))})
                                elif isinstance(o, str):
                                    reflections.append(o)
                        else:
                            reflections.append(s.get("scenario",""))
                    except Exception:
                        logger.exception("FutureProjection: inner_dialogue reflection error")
                        reflections.append(s.get("scenario",""))

                    # if no structured plan from inner_dialogue, call LLM for actions expansion
                    if not any(item for item in predicted if s.get("scenario","") in item.get("text","")):
                        prompt = f"Scenario: {s.get('scenario','')}\nGenera 2 azioni concrete, semplici e non dannose che Nova può provare per esplorare questo scenario. Rispondi in elenco puntato, italiano."
                        out = self.llm(prompt, max_tokens=200, temperature=0.8) or ""
                        lines = [l.strip("-• \t") for l in out.splitlines() if l.strip()]
                        for ln in lines[:3]:
                            predicted.append({"text": ln, "confidence": round(random.uniform(0.45, 0.9),2)})

                # Evaluate & commit predicted actions (ethics + should_execute)
                evaluated = [self._evaluate_and_commit_action(a) for a in predicted]
                # persist summary
                self.state.setdefault("future_projections", {})
                self.state["future_projections"].update({
                    "last_predicted_actions": evaluated,
                    "last_updated": datetime.utcnow().isoformat()
                })
                # try to save state
                try:
                    if self.core and hasattr(self.core, "save_state"):
                        self.core.save_state()
                    else:
                        _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                except Exception:
                    pass
                logger.info("FutureProjection: previste %d azioni.", len(evaluated))
                return evaluated
            except Exception:
                logger.exception("FutureProjection: exception in predict_future_actions")
                return []

    # -----------------------
    # Full projection cycle: generate scenarios, predict actions, optionally inform conscious loop
    # -----------------------
    def run_projection_cycle(self, seed: Optional[str] = None, notify_conscious: bool = True) -> Dict[str,Any]:
        with self._lock:
            try:
                scenarios = self.generate_scenarios(seed=seed)
                actions = self.predict_future_actions(scenarios)
                proj = {
                    "scenarios": scenarios,
                    "predicted_actions": actions,
                    "ts": datetime.utcnow().isoformat()
                }
                self.state.setdefault("future_projections", {})
                self.state["future_projections"].update(proj)
                # notify conscious loop so projections can be reflected upon
                if notify_conscious and self.conscious_loop and hasattr(self.conscious_loop, "on_perception"):
                    try:
                        self.conscious_loop.on_perception({"type":"future_projection","payload":proj})
                    except Exception:
                        # try cycle if on_perception not present
                        try:
                            if hasattr(self.conscious_loop, "cycle"):
                                self.conscious_loop.cycle()
                        except Exception:
                            pass
                # persist
                try:
                    if self.core and hasattr(self.core, "save_state"):
                        self.core.save_state()
                    else:
                        _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                except Exception:
                    pass
                logger.info("FutureProjection: ciclo completo eseguito.")
                return proj
            except Exception:
                logger.exception("FutureProjection: exception in run_projection_cycle")
                return {"error":"exception","ts":datetime.utcnow().isoformat()}

    # -----------------------
    # Utility: short summary / inspection
    # -----------------------
    def get_projection_summary(self) -> Dict[str,Any]:
        return self.state.get("future_projections", {})

# -----------------------
# Quick standalone demo
# -----------------------
if __name__ == "__main__":
    # stato minimo per demo
    st = {
        "identity": {"name":"Nova","description":"Prototipo cosciente"},
        "emotions": {"curiosity": 0.8, "anxiety": 0.2},
        "motivations": {"learn": 0.9},
        "timeline": [
            {"content":"Ho provato un piccolo esperimento con sensori"},
            {"content":"Ho parlato con una persona interessata al progetto"}
        ]
    }
    fp = FutureProjection(st, core=None, model_path=DEFAULT_MODEL_PATH)
    out = fp.run_projection_cycle(seed="imparare a muovere un braccio robotico")
    print("Proiezione:", out)
