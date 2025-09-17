# motivational_engine.py
"""
MotivationalEngine (avanzato)

Scopi:
- Gestire desideri, obiettivi, curiosità e spinte esplorative di Nova.
- Promuovere desideri a obiettivi, decomporli in passi concreti e proporre azioni.
- Usare modello locale Gemma (GGUF via llama_cpp) se disponibile per generare piani e suggerimenti.
- Valutare proposte con EthicsEngine (se presente) prima di committarle in core.state['tasks'].
- Interagire con memory_timeline, dream_generator, attention_manager, inner_dialogue, conscious_loop e scheduler.
- Persistenza atomica su internal_state.yaml (configurabile tramite env var).
"""

import os
import time
import random
import threading
import yaml
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")

# -----------------------
# LLM loader (local Gemma via llama_cpp) with fallback
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 2):
    try:
        from llama_cpp import Llama
        logger.info("MotivationalEngine: caricamento LLM %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        def _call(prompt: str, max_tokens: int = 200, temperature: float = 0.8) -> str:
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
                logger.exception("MotivationalEngine: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.warning("MotivationalEngine: llama_cpp non disponibile (%s). Uso fallback heuristics.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # semplice fallback per decomposizione di obiettivi
            try:
                lines = [l.strip() for l in prompt.splitlines() if l.strip()]
                seed = lines[-1][:120] if lines else "obiettivo"
            except Exception:
                seed = "obiettivo"
            choices = [
                f"Passi suggeriti per {seed}: 1) riflettere, 2) provare un piccolo esperimento, 3) valutare risultati.",
                f"Scomposizione semplice per {seed}: definire metrica, fare una prova, iterare.",
                f"Piano rapido per {seed}: osservare, testare, registrare."
            ]
            return random.choice(choices)
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
# MotivationalEngine
# -----------------------
class MotivationalEngine:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None, model_path: str = DEFAULT_MODEL_PATH):
        """
        state: shared core.state
        core: optional NovaCore reference (used for scheduler, ethics, memory, etc.)
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # LLM callable
        self.llm = _load_llm(model_path)

        # Initialize motivational structures
        self.state.setdefault("motivation", {
            "desires": [],         # list of {"text":..., "ts":..., "source":..., "strength": float}
            "goals": [],           # list of {"id","title","steps","priority","progress","ts","origin"}
            "curiosity_level": 0.5,
            "last_update": None
        })

        # tuning params
        self.desire_to_goal_prob_base = float(os.environ.get("MOTIV_DESIRE_PROMOTE_P", 0.18))
        self.goal_decay_rate = float(os.environ.get("MOTIV_GOAL_DECAY", 0.996))
        self.curiosity_drift = float(os.environ.get("MOTIV_CURIO_DRIFT", 0.01))

        # attempt to wire collaborators from core (avoid import cycles)
        self.memory = getattr(core, "memory_timeline", None)
        self.dream_generator = getattr(core, "dream_generator", None)
        self.attention = getattr(core, "attention_manager", None)
        self.inner_dialogue = getattr(core, "inner_dialogue", None)
        self.conscious_loop = getattr(core, "conscious_loop", None)
        self.scheduler = getattr(core, "scheduler", None)

        logger.info("MotivationalEngine inizializzato (LLM=%s).", "yes" if callable(self.llm) else "fallback")

    # -----------------------
    # Persistence helper
    # -----------------------
    def _persist_state(self):
        try:
            self.state["motivation"]["last_update"] = datetime.utcnow().isoformat()
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
        except Exception:
            logger.exception("MotivationalEngine: errore salvataggio stato")

    # -----------------------
    # Utilities
    # -----------------------
    def _make_desire_obj(self, text: str, source: str = "user", strength: float = 0.4) -> Dict[str, Any]:
        return {"text": text, "ts": datetime.utcnow().isoformat(), "source": source, "strength": float(max(0.01, min(1.0, strength)))}

    def _make_goal_obj(self, title: str, steps: List[str], priority: float = 0.5, origin: str = "generated") -> Dict[str, Any]:
        return {
            "id": f"goal_{int(time.time()*1000)}_{random.randint(0,999)}",
            "title": title,
            "steps": steps,
            "priority": float(max(0.0, min(1.0, priority))),
            "progress": 0.0,
            "ts": datetime.utcnow().isoformat(),
            "origin": origin
        }

    # -----------------------
    # Public API: desires & goals management
    # -----------------------
    def add_desire(self, desire_text: str, source: str = "user", strength: float = 0.4):
        with self._lock:
            try:
                desire = self._make_desire_obj(desire_text, source=source, strength=strength)
                self.state["motivation"].setdefault("desires", []).append(desire)
                logger.info("MotivationalEngine: desiderio aggiunto: %s", desire_text[:120])
                # memory + attention + dream influence
                try:
                    if self.memory and hasattr(self.memory, "add_experience"):
                        self.memory.add_experience({"text": desire_text, "meta": {"type":"desire","strength":desire["strength"]}}, category="motivation", importance=2)
                except Exception:
                    pass
                try:
                    if self.attention and hasattr(self.attention, "add_focus_tag"):
                        self.attention.add_focus_tag("desire", strength=desire["strength"])
                except Exception:
                    pass
                try:
                    if self.dream_generator and hasattr(self.dream_generator, "register_reflection"):
                        # let dreams be influenced by new desires later
                        self.dream_generator.register_reflection({"text": desire_text, "importance": desire["strength"]})
                except Exception:
                    pass
                self._persist_state()
            except Exception:
                logger.exception("MotivationalEngine: exception in add_desire")

    def remove_desire(self, desire_text: str):
        with self._lock:
            try:
                ds = self.state["motivation"].get("desires", [])
                self.state["motivation"]["desires"] = [d for d in ds if d.get("text") != desire_text]
                logger.info("MotivationalEngine: desiderio rimosso: %s", desire_text[:120])
                self._persist_state()
            except Exception:
                logger.exception("MotivationalEngine: exception in remove_desire")

    def list_desires(self) -> List[Dict[str, Any]]:
        return list(self.state["motivation"].get("desires", []))

    def list_goals(self) -> List[Dict[str, Any]]:
        return list(self.state["motivation"].get("goals", []))

    # -----------------------
    # Promote desire -> goal (with decomposition & ethics check)
    # -----------------------
    def promote_desire_to_goal(self, desire: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Tenta di trasformare un desire dict in un obiettivo concreto con steps.
        Se approved (ethics + should_execute) viene aggiunto a state['motivation']['goals'] e opzionalmente a core.state['tasks'].
        """
        with self._lock:
            try:
                text = desire.get("text") if isinstance(desire, dict) else str(desire)
                # probabilità base di promozione legata a curiosità e forza del desire
                curiosity = float(self.state["motivation"].get("curiosity_level", 0.5))
                p_base = self.desire_to_goal_prob_base * (0.6 + 0.8 * curiosity) * float(desire.get("strength", 0.5))
                if force or random.random() < p_base:
                    # decompose via LLM if available
                    steps = self.decompose_goal(text)
                    # build goal object
                    priority = float(min(1.0, 0.2 + 0.8 * desire.get("strength", 0.5) * curiosity))
                    goal = self._make_goal_obj(title=text, steps=steps, priority=priority, origin=desire.get("source","desire"))
                    # ethics check on goal text + steps (if core.ethics present)
                    allowed = True
                    verdict = None
                    if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                        try:
                            txt = goal["title"] + " " + " ".join(goal["steps"])
                            verdict = self.core.ethics.evaluate_action({"text": txt, "tags":["goal"], "metadata":{"source":"motivational_engine"}})
                            allowed = verdict.get("allowed", True)
                        except Exception:
                            logger.exception("MotivationalEngine: errore verifica etica")
                    if not allowed:
                        logger.warning("MotivationalEngine: goal bloccato da EthicsEngine: %s", verdict)
                        return None
                    # commit to goals
                    self.state["motivation"].setdefault("goals", []).append(goal)
                    # optionally add to core.state['tasks'] as suggested task (human-in-the-loop)
                    if self.core and hasattr(self.core, "state"):
                        try:
                            task = {"title": goal["title"], "steps": goal["steps"], "origin":"motivational_engine", "priority": goal["priority"], "timestamp": datetime.utcnow().isoformat()}
                            self.core.state.setdefault("tasks", []).append(task)
                            logger.info("MotivationalEngine: goal promosso e aggiunto ai tasks: %s", goal["title"][:120])
                        except Exception:
                            logger.exception("MotivationalEngine: errore aggiunta task al core.state")
                    # remove the desire (if present)
                    try:
                        self.state["motivation"]["desires"] = [d for d in self.state["motivation"].get("desires", []) if d.get("text") != text]
                    except Exception:
                        pass
                    self._persist_state()
                    return goal
                else:
                    logger.debug("MotivationalEngine: promozione desiderio->goal non avvenuta (probabilità).")
                    return None
            except Exception:
                logger.exception("MotivationalEngine: exception in promote_desire_to_goal")
                return None

    # -----------------------
    # Decompose goal into actionable steps using LLM or fallback heuristic
    # -----------------------
    def decompose_goal(self, goal_text: str, max_steps: int = 6) -> List[str]:
        try:
            prompt = (
                f"Obiettivo: {goal_text}\n"
                "Dividi questo obiettivo in 3-6 passi concreti, semplici e non dannosi, che Nova può eseguire per esplorare l'obiettivo. "
                "Rispondi in italiano come elenco puntato, uno step per linea.\n"
            )
            out = self.llm(prompt, max_tokens=220, temperature=0.8) or ""
            lines = [l.strip("-• \t") for l in out.splitlines() if l.strip()]
            if not lines:
                raise ValueError("no lines from llm")
            return [ln for ln in lines[:max_steps]]
        except Exception:
            # fallback heuristic: simple 3-step decomposition
            logger.debug("MotivationalEngine: fallback decompose_goal")
            return [
                f"Raccogli informazioni su: {goal_text}",
                f"Prova un esperimento o una piccola azione relativa a: {goal_text}",
                f"Valuta i risultati e annota osservazioni"
            ]

    # -----------------------
    # Prioritize goals (simple weighted scoring)
    # -----------------------
    def prioritize_goals(self):
        with self._lock:
            try:
                goals = self.state["motivation"].get("goals", [])
                emotions = self.state.get("emotions", {}) or {}
                # compute score for each goal
                scored = []
                for g in goals:
                    base = float(g.get("priority", 0.5))
                    # boost if matches attention tags or recent timeline mentions
                    att_boost = 0.0
                    try:
                        if self.attention and hasattr(self.attention, "get_current_focus"):
                            focus = self.attention.get_current_focus()
                            if focus and isinstance(focus, dict) and focus.get("text"):
                                if focus.get("text", "").lower() in g.get("title","").lower():
                                    att_boost = 0.25
                    except Exception:
                        pass
                    # emotion alignment: positive correlation with curiosity or joy
                    emotion_score = float(self.state.get("emotions", {}).get("curiosity", 0.0) or 0.0) * 0.4
                    score = max(0.0, min(1.0, base + att_boost + emotion_score))
                    scored.append((score, g))
                scored.sort(key=lambda x: x[0], reverse=True)
                # reassign priorities based on sorted order
                for idx, (s, g) in enumerate(scored):
                    g["priority"] = round(max(0.01, min(1.0, s)), 3)
                self.state["motivation"]["goals"] = [g for _, g in scored]
                self._persist_state()
            except Exception:
                logger.exception("MotivationalEngine: exception in prioritize_goals")

    # -----------------------
    # Periodic update: curiosity drift, promote desires, decay goals, trigger planning
    # -----------------------
    def update(self):
        with self._lock:
            try:
                # curiosity drift + small random fluctuations
                drift = random.uniform(-self.curiosity_drift, self.curiosity_drift*1.5)
                self.state["motivation"]["curiosity_level"] = float(max(0.0, min(1.0, self.state["motivation"].get("curiosity_level", 0.5) + drift)))
                # decay existing goals slightly
                for g in self.state["motivation"].get("goals", []):
                    g["priority"] = float(max(0.0, min(1.0, g.get("priority", 0.5) * self.goal_decay_rate)))
                    # small progress random drift
                    g["progress"] = float(min(1.0, max(0.0, g.get("progress", 0.0) + random.uniform(-0.02, 0.03))))
                # try to promote desires based on curiosity and strength
                desires = list(self.state["motivation"].get("desires", []))
                for d in desires:
                    self.promote_desire_to_goal(d)
                # prioritize goals
                self.prioritize_goals()
                # trigger inner reflection or conscious loop hint
                try:
                    if self.inner_dialogue and hasattr(self.inner_dialogue, "process"):
                        # give inner dialogue some motivational context
                        ctx = [f"Motivazione: {g.get('title')}" for g in self.state["motivation"].get("goals", [])[:3]]
                        if ctx:
                            self.inner_dialogue.process(ctx)
                except Exception:
                    pass
                # occasionally request dream generation when curiosity high
                if random.random() < 0.06 and self.state["motivation"].get("curiosity_level", 0.5) > 0.6:
                    try:
                        if self.dream_generator and hasattr(self.dream_generator, "generate_dream"):
                            self.dream_generator.generate_dream(seed="; ".join([d.get("text") for d in desires[:2]]))
                    except Exception:
                        pass
                # schedule follow-ups for top goals via scheduler
                try:
                    if self.scheduler and hasattr(self.scheduler, "add_recurring_task"):
                        top_goals = self.state["motivation"].get("goals", [])[:2]
                        for g in top_goals:
                            tag = f"follow_goal_{g.get('id')}"
                            # create a follow-up handler closure
                            def make_handler(goal_id):
                                def handler():
                                    logger.info("Follow-up handler eseguito per goal %s", goal_id)
                                    # small progress bump
                                    try:
                                        for gg in self.state["motivation"].get("goals", []):
                                            if gg.get("id") == goal_id:
                                                gg["progress"] = min(1.0, gg.get("progress",0.0) + 0.02)
                                                # commit as a task suggestion to core.state
                                                if self.core and hasattr(self.core, "state"):
                                                    task = {"title": f"Seguire passo per: {gg.get('title')}", "origin":"motivational_followup", "timestamp": datetime.utcnow().isoformat()}
                                                    self.core.state.setdefault("tasks", []).append(task)
                                    except Exception:
                                        logger.exception("Follow-up handler errore")
                                return handler
                            # attempt to register short recurring follow-up every few minutes (if scheduler supports seconds)
                            try:
                                self.scheduler.add_recurring_task(make_handler(g.get("id")), interval=60*10)  # every 10 minutes
                            except Exception:
                                # ignore if scheduler missing or interface different
                                pass
                except Exception:
                    logger.exception("MotivationalEngine: errore scheduling follow-ups")
                # persist
                self._persist_state()
            except Exception:
                logger.exception("MotivationalEngine: exception in update")

    # -----------------------
    # Reward / progress / reinforcement signals
    # -----------------------
    def reward_progress(self, goal_id: str, amount: float = 0.1):
        with self._lock:
            try:
                for g in self.state["motivation"].get("goals", []):
                    if g.get("id") == goal_id:
                        g["progress"] = min(1.0, g.get("progress", 0.0) + float(amount))
                        # small priority boost on progress
                        g["priority"] = min(1.0, g.get("priority", 0.0) + 0.05 * amount)
                        # influence curiosity slightly
                        self.state["motivation"]["curiosity_level"] = min(1.0, self.state["motivation"].get("curiosity_level",0.5) + 0.02*amount)
                        logger.info("MotivationalEngine: progresso goal %s -> progress=%.2f", goal_id, g["progress"])
                        self._persist_state()
                        return True
                return False
            except Exception:
                logger.exception("MotivationalEngine: exception in reward_progress")
                return False

    # -----------------------
    # Utility: suggest an exploratory action for current top goal
    # -----------------------
    def suggest_exploration_action(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            try:
                goals = self.state["motivation"].get("goals", [])
                if not goals:
                    return None
                top = sorted(goals, key=lambda x: x.get("priority",0.0), reverse=True)[0]
                # try to use inner_dialogue or LLM to produce a small action
                action_text = None
                try:
                    if self.inner_dialogue and hasattr(self.inner_dialogue, "process"):
                        out = self.inner_dialogue.process([f"Proponi un'azione esplorativa per: {top.get('title')}"])
                        # take first string output if exists
                        for o in out:
                            if isinstance(o, str) and len(o) > 10:
                                action_text = o
                                break
                    if not action_text:
                        prompt = f"Obiettivo: {top.get('title')}\nSuggerisci 1 azione esplorativa, semplice e non dannosa che Nova può provare adesso. Rispondi in italiano in una frase."
                        action_text = self.llm(prompt, max_tokens=120, temperature=0.7) or None
                except Exception:
                    logger.exception("MotivationalEngine: errore suggerimento azione")
                if not action_text:
                    action_text = f"Prova un piccolo esperimento relativo a: {top.get('title')}"
                suggestion = {"goal_id": top.get("id"), "text": action_text.strip(), "confidence": 0.5}
                # ethics check before suggesting external actions
                allowed = True
                verdict = None
                if self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                    try:
                        verdict = self.core.ethics.evaluate_action({"text": suggestion["text"], "tags":["suggestion"], "metadata":{"source":"motivational_engine"}})
                        allowed = verdict.get("allowed", True)
                    except Exception:
                        logger.exception("MotivationalEngine: errore verifica etica suggestion")
                suggestion["allowed"] = allowed
                suggestion["ethics_verdict"] = verdict
                return suggestion
            except Exception:
                logger.exception("MotivationalEngine: exception in suggest_exploration_action")
                return None

    # -----------------------
    # Quick diagnostics & state snapshot
    # -----------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "curiosity": self.state["motivation"].get("curiosity_level"),
                "num_desires": len(self.state["motivation"].get("desires", [])),
                "num_goals": len(self.state["motivation"].get("goals", [])),
                "top_goals": [g.get("title") for g in self.state["motivation"].get("goals", [])[:4]]
            }

# end file
