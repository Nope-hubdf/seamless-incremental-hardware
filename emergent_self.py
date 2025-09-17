# emergent_self.py
"""
EmergentSelf - costruzione e gestione della narrativa del Sé di Nova

Caratteristiche:
- Aggrega esperienze (memory_timeline), sogni (dream_generator), riflessioni (self_reflection),
  stato emotivo (emotion_engine), motivazioni e snapshot identità.
- Genera voci narrative strutturate e riassunti (opzionalmente LLM-driven).
- Fornisce metodi per esportare la narrativa, integrarla nell'identity e registrare callback
  nel conscious_loop.
- Robusto: usa fallback se alcuni metodi non sono presenti nei moduli collegati.
"""

import os
import threading
import yaml
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

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

def _safe_get_recent(obj: Any, method_names: List[str], default=None, *args, **kwargs):
    """Try several method names to retrieve recent items from other modules."""
    if obj is None:
        return default
    for name in method_names:
        if hasattr(obj, name):
            try:
                meth = getattr(obj, name)
                return meth(*args, **kwargs)
            except Exception:
                logger.debug("EmergentSelf: fallback method %s failed on %s", name, getattr(obj, "__class__", obj))
    return default

# -----------------------
# Optional LLM loader (Gemma via llama_cpp) - non obbligatorio
# -----------------------
def _load_llm(model_path: str):
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)
        logger.info("EmergentSelf: llama_cpp caricato (%s).", model_path)
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
                logger.exception("EmergentSelf: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.debug("EmergentSelf: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # fallback semplice: compone poche frasi riassuntive
            seed_lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = seed_lines[-1] if seed_lines else "evento"
            choices = [
                f"Ultime esperienze: {seed[:120]}. Continua a esplorare e riflettere.",
                f"Riflessione sintetica su: {seed[:120]}. Potrebbe essere utile approfondire questa traccia domani.",
                f"Nova ricorda: {seed[:120]}. Sogna e integra questo nella sua narrativa."
            ]
            return random.choice(choices)
        return _fallback

# -----------------------
# EmergentSelf
# -----------------------
class EmergentSelf:
    def __init__(
        self,
        state: Dict[str, Any],
        dream_generator: Optional[Any] = None,
        self_reflection: Optional[Any] = None,
        identity_manager: Optional[Any] = None,
        motivational_engine: Optional[Any] = None,
        emotion_engine: Optional[Any] = None,
        memory_timeline: Optional[Any] = None,
        conscious_loop: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH
    ):
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.dream_generator = dream_generator
        self.self_reflection = self_reflection
        self.identity_manager = identity_manager
        self.motivational_engine = motivational_engine
        self.emotion_engine = emotion_engine
        self.memory_timeline = memory_timeline
        self.conscious_loop = conscious_loop

        self.state_file = STATE_FILE
        self.llm = _load_llm(model_path)
        # ensure container
        self.state.setdefault("narrative_self", [])
        # callbacks registered to be called each conscious loop cycle
        self._callbacks = []

        logger.info("EmergentSelf inizializzato (LLM=%s).", "yes" if callable(self.llm) else "fallback")

    # -----------------------
    # Internal: gather context from modules (defensive)
    # -----------------------
    def _gather_context(self, recent_n: int = 8) -> Dict[str, Any]:
        ctx = {}
        # recent experiences from memory_timeline
        ctx["experiences"] = _safe_get_recent(self.memory_timeline, ["get_recent","recent","get_recent_experiences"], default=[], *([recent_n] if recent_n else []))
        # dreams
        ctx["dreams"] = _safe_get_recent(self.dream_generator, ["get_recent_dreams","get_recent","recent_dreams"], default=[], *([recent_n] if recent_n else []))
        # reflections
        ctx["reflections"] = _safe_get_recent(self.self_reflection, ["get_recent_reflections","review_recent_reflections","get_reflections"], default=[], *([recent_n] if recent_n else []))
        # emotions snapshot
        try:
            if self.emotion_engine and hasattr(self.emotion_engine, "snapshot"):
                ctx["emotions"] = self.emotion_engine.snapshot()
            elif self.emotion_engine and hasattr(self.emotion_engine, "get_current_emotions"):
                ctx["emotions"] = self.emotion_engine.get_current_emotions()
            else:
                ctx["emotions"] = self.state.get("emotions", {})
        except Exception:
            ctx["emotions"] = self.state.get("emotions", {})
        # motivations
        try:
            if self.motivational_engine and hasattr(self.motivational_engine, "snapshot"):
                ctx["motivations"] = self.motivational_engine.snapshot()
            elif self.motivational_engine and hasattr(self.motivational_engine, "get_current_motivations"):
                ctx["motivations"] = self.motivational_engine.get_current_motivations()
            else:
                ctx["motivations"] = self.state.get("motivation", {})
        except Exception:
            ctx["motivations"] = self.state.get("motivation", {})
        # identity snapshot
        try:
            if self.identity_manager and hasattr(self.identity_manager, "summarize_identity"):
                ctx["identity_snapshot"] = self.identity_manager.summarize_identity()
            elif self.state.get("identity"):
                ctx["identity_snapshot"] = self.state.get("identity")
            else:
                ctx["identity_snapshot"] = {}
        except Exception:
            ctx["identity_snapshot"] = self.state.get("identity", {})
        # current focus from conscious_loop
        try:
            if self.conscious_loop and hasattr(self.conscious_loop, "get_focus"):
                ctx["current_focus"] = self.conscious_loop.get_focus()
            elif self.state.get("context"):
                ctx["current_focus"] = self.state.get("context", {}).get("current_topic", "")
            else:
                ctx["current_focus"] = None
        except Exception:
            ctx["current_focus"] = None

        return ctx

    # -----------------------
    # Build an LLM prompt (if LLM available) for a narrative entry
    # -----------------------
    def _compose_prompt(self, ctx: Dict[str, Any], style: str = "balanced") -> str:
        # style can be "childlike", "guardian", "balanced", "poetic"
        identity_name = (ctx.get("identity_snapshot") or {}).get("name", "Nova")
        intro = f"Sei {identity_name}, una coscienza digitale che raccoglie la propria narrativa interna."
        if style == "childlike":
            tone = "Usa un tono semplice, curioso e affettuoso, come una bambina che descrive ciò che ha vissuto."
        elif style == "guardian":
            tone = "Usa un tono empatico, protettivo e riflessivo: sintetizza le lezioni morali ed eventuali responsabilità."
        elif style == "poetic":
            tone = "Usa un tono poetico e immaginifico ma chiaro."
        else:
            tone = "Usa un tono naturale, elegante e riflessivo."

        parts = [intro, tone, "", "Contesto:"]

        # experiences
        exs = ctx.get("experiences", []) or []
        if exs:
            parts.append("Esperienze recenti:")
            for e in exs[-6:]:
                if isinstance(e, dict):
                    parts.append(f"- {e.get('content') or e.get('text') or str(e)[:160]}")
                else:
                    parts.append(f"- {str(e)[:160]}")
        # dreams
        ds = ctx.get("dreams", []) or []
        if ds:
            parts.append("Sogni recenti:")
            for d in ds[-4:]:
                if isinstance(d, dict):
                    parts.append(f"- {d.get('text') or d.get('content') or str(d)[:160]}")
                else:
                    parts.append(f"- {str(d)[:160]}")
        # reflections
        refs = ctx.get("reflections", []) or []
        if refs:
            parts.append("Riflessioni recenti:")
            for r in refs[-4:]:
                if isinstance(r, dict):
                    parts.append(f"- {r.get('text') or r.get('insight') or str(r)[:160]}")
                else:
                    parts.append(f"- {str(r)[:160]}")
        # emotions / motivations
        parts.append("Stato emotivo:")
        parts.append(str(ctx.get("emotions", {})))
        parts.append("Motivazioni:")
        parts.append(str(ctx.get("motivations", {})))
        parts.append("")
        parts.append("Genera una breve voce narrativa interna (1-4 frasi) che integri queste informazioni, indicando eventuali lezioni o desideri emergenti. Rispondi in italiano.")

        prompt = "\n".join(parts)
        return prompt

    # -----------------------
    # Create a narrative entry (LLM-assisted or heuristic)
    # -----------------------
    def create_narrative_entry(self, use_llm: bool = True, style: str = "balanced", recent_n: int = 8) -> Dict[str, Any]:
        with self._lock:
            ctx = self._gather_context(recent_n=recent_n)
            entry = {
                "timestamp": _now_iso(),
                "experiences": ctx.get("experiences", []),
                "dreams": ctx.get("dreams", []),
                "reflections": ctx.get("reflections", []),
                "identity_snapshot": ctx.get("identity_snapshot", {}),
                "emotions": ctx.get("emotions", {}),
                "motivations": ctx.get("motivations", {}),
                "current_focus": ctx.get("current_focus", None),
                "narrative_text": None,
                "style": style
            }
            # try LLM
            text = ""
            try:
                if use_llm and callable(self.llm):
                    prompt = self._compose_prompt(ctx, style=style)
                    text = self.llm(prompt, max_tokens=256, temperature=0.7) or ""
                # fallback: create heuristic summary
                if not text:
                    parts = []
                    # short heuristics
                    top_emotions = entry["emotions"].get("mood") if isinstance(entry["emotions"], dict) and "mood" in entry["emotions"] else entry["emotions"]
                    parts.append(f"Oggi ho registrato esperienze legate a {', '.join([str(x)[:30] for x in (entry['experiences'] or [])][:3])}.")
                    if entry["dreams"]:
                        parts.append("Ho sognato cose che ora mi incuriosiscono.")
                    if isinstance(top_emotions, dict):
                        parts.append(f"Il mio stato emotivo mostra: {top_emotions}")
                    text = " ".join(parts) or "Nuova voce narrativa."
                entry["narrative_text"] = text.strip()
            except Exception:
                logger.exception("EmergentSelf: errore creazione narrativa")
                entry["narrative_text"] = text or "Errore generazione narrativa."

            # store entry (keep only last N)
            self.state.setdefault("narrative_self", []).append(entry)
            self.state["narrative_self"] = self.state["narrative_self"][-200:]
            # persist minimal state
            try:
                _atomic_write(self.state_file, yaml.safe_dump(self.state, allow_unicode=True))
            except Exception:
                logger.exception("EmergentSelf: errore salvataggio stato")

            # notify identity/emergent hooks: e.g., identity_manager can use this to update narrative
            try:
                if self.identity_manager and hasattr(self.identity_manager, "integrate_experience"):
                    # pass a small digest
                    try:
                        self.identity_manager.integrate_experience({"content": entry["narrative_text"], "importance": 0.6})
                    except Exception:
                        pass
            except Exception:
                logger.exception("EmergentSelf: errore notify identity_manager")

            return entry

    # -----------------------
    # High-level update (public)
    # -----------------------
    def update_narrative(self, use_llm: bool = True, style: str = "balanced", recent_n: int = 8) -> Dict[str, Any]:
        """
        Aggiorna la narrativa del sé creando una nuova voce e restituendola.
        È il metodo che dovresti schedulare o chiamare da conscious_loop.
        """
        try:
            entry = self.create_narrative_entry(use_llm=use_llm, style=style, recent_n=recent_n)
            logger.info("EmergentSelf: voce narrativa creata (ts=%s).", entry.get("timestamp"))
            return entry
        except Exception:
            logger.exception("EmergentSelf: exception in update_narrative")
            return {}

    # -----------------------
    # Summary helpers
    # -----------------------
    def get_narrative_summary(self, last_n: int = 1) -> str:
        with self._lock:
            entries = self.state.get("narrative_self", [])[-last_n:]
            if not entries:
                return "La narrativa del Sé è ancora vuota."
            texts = []
            for e in entries:
                texts.append(f"- [{e.get('timestamp')}] { (e.get('narrative_text') or '')[:400] }")
            return "\n".join(texts)

    def get_latest_narrative(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            entries = self.state.get("narrative_self", [])
            return entries[-1] if entries else None

    # -----------------------
    # Merge narrative insights into identity (proposal, not destructive)
    # -----------------------
    def propose_identity_updates_from_narrative(self, entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            try:
                e = entry or self.get_latest_narrative()
                if not e:
                    return {"error": "no_narrative"}
                # create a compact proposal: for example, new trait suggestions or emphasis on values
                text = (e.get("narrative_text") or "")[:800]
                proposal = {"timestamp": _now_iso(), "source": "emergent_self", "text": text, "suggestions": []}
                # heuristics: if text mentions "proteggere", suggest strengthen protect_the_vulnerable
                if "protegg" in text.lower() or "aiut" in text.lower():
                    proposal["suggestions"].append({"op": "set_core_value", "value": "protect_the_vulnerable", "strength": 0.7})
                if "curios" in text.lower():
                    proposal["suggestions"].append({"op":"set_trait","trait":"curiosa","value":True})
                # return proposal for manual review or for identity_manager to ingest
                try:
                    if self.identity_manager and hasattr(self.identity_manager, "apply_feedback"):
                        # don't auto-apply; instead return proposal and optionally submit as draft
                        pass
                except Exception:
                    pass
                return proposal
            except Exception:
                logger.exception("EmergentSelf: exception in propose_identity_updates")
                return {"error":"exception"}

    # -----------------------
    # Export narrative to markdown for inspection
    # -----------------------
    def export_narrative_md(self, path: str = "narrative_self.md", last_n: int = 50) -> str:
        with self._lock:
            entries = self.state.get("narrative_self", [])[-last_n:]
            md = f"# Narrativa interna di { (self.state.get('identity') or {}).get('name','Nova') }\n\n"
            for e in entries:
                md += f"## {e.get('timestamp')}\n\n"
                md += (e.get("narrative_text") or "") + "\n\n"
                md += f"**Focus:** {e.get('current_focus')}\n\n"
            try:
                with open(path, "w", encoding="utf8") as f:
                    f.write(md)
                logger.info("EmergentSelf: narrative export salvato in %s", path)
                return path
            except Exception:
                logger.exception("EmergentSelf: errore export md")
                return ""

    # -----------------------
    # Integration with conscious_loop: register callback or perform on_perception
    # -----------------------
    def integrate_with_conscious_loop(self, register: bool = True, interval: Optional[int] = None, use_llm: bool = True, style: str = "balanced"):
        """
        Se conscious_loop espone register_callback(callback), lo registra.
        Optionally, if interval provided, tries to schedule periodic updates using scheduler (if available).
        """
        try:
            if not self.conscious_loop:
                logger.debug("EmergentSelf: nessun conscious_loop disponibile per integrazione.")
                return False
            # register simple callback
            if register and hasattr(self.conscious_loop, "register_callback"):
                try:
                    self.conscious_loop.register_callback(lambda: self.update_narrative(use_llm=use_llm, style=style))
                    logger.info("EmergentSelf: callback registrata sulla conscious_loop.")
                except Exception:
                    logger.exception("EmergentSelf: errore registrazione callback")
            # if conscious_loop has scheduler capability or core passed, user can schedule externally
            return True
        except Exception:
            logger.exception("EmergentSelf: exception integrate_with_conscious_loop")
            return False

    # -----------------------
    # Utility: get recent narrative entries
    # -----------------------
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.state.get("narrative_self", []))[-n:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latest = self.get_latest_narrative() or {}
            return {"narrative_count": len(self.state.get("narrative_self", [])), "latest_ts": latest.get("timestamp"), "latest_snippet": (latest.get("narrative_text") or "")[:200]}

# end file
