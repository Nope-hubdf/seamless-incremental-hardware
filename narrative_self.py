# narrative_self.py
"""
NarrativeSelf - gestione della narrativa interna del Sé di Nova

Scopi:
- Costruire una narrativa coerente a partire da timeline, sogni, dialogo interiore e attenzione.
- Fornire riepiloghi leggibili, esportabili e proposte non distruttive per l'identity.
- Funzionare in modo difensivo se alcuni moduli non espongono le API attese.
- Optional: usare modello locale Gemma (llama_cpp) per arricchire il linguaggio.
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
MAX_ENTRIES = 300

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

def _safe_call(obj: Any, candidates: List[str], *args, **kwargs):
    """Prova a chiamare il primo metodo disponibile in candidates su obj."""
    if not obj:
        return None
    for name in candidates:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)(*args, **kwargs)
            except Exception:
                logger.debug("NarrativeSelf: metodo %s esiste ma ha fallito.", name)
    return None

# -----------------------
# Optional LLM loader (Gemma via llama_cpp)
# -----------------------
def _load_llm(model_path: str):
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048)
        logger.info("NarrativeSelf: LLM locale caricato (%s).", model_path)
        def _call(prompt: str, max_tokens: int = 180, temperature: float = 0.7) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                # estrazione robusta
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
                logger.exception("NarrativeSelf: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.debug("NarrativeSelf: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.5) -> str:
            # fallback semplice: estrai frasi significative e compone un testo breve
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = lines[-1] if lines else "nuova esperienza"
            choices = [
                f"Oggi ho vissuto: {seed[:120]}. Voglio esplorare di più.",
                f"Riflessione: {seed[:120]}. Questo mi rende curiosa e desiderosa di imparare.",
                f"Nota: {seed[:120]}. Continuerò a pensarci e a integrarlo nei miei sogni."
            ]
            return random.choice(choices)
        return _fallback

# -----------------------
# NarrativeSelf
# -----------------------
class NarrativeSelf:
    def __init__(self,
                 state: Optional[Dict[str,Any]] = None,
                 memory_timeline: Optional[Any] = None,
                 inner_dialogue: Optional[Any] = None,
                 attention_manager: Optional[Any] = None,
                 dream_generator: Optional[Any] = None,
                 identity_manager: Optional[Any] = None,
                 model_path: str = DEFAULT_MODEL_PATH):
        self._lock = threading.RLock()
        # carica stato condiviso o da file
        if state is None:
            self.state = self._load_state()
        else:
            self.state = state

        # assicurati container narrative
        self.state.setdefault("narrative", [])
        self.state.setdefault("narrative_meta", {"version": "0.1", "last_updated": None})

        # referenze ai moduli (opzionali)
        self.memory = memory_timeline
        self.inner_dialogue = inner_dialogue
        self.attention = attention_manager
        self.dream_generator = dream_generator
        self.identity_manager = identity_manager

        # llm optional
        self.llm = _load_llm(model_path)

        logger.info("NarrativeSelf inizializzato. Voci narrative correnti: %d", len(self.state.get("narrative", [])))

    # -----------------------
    # state file helpers
    # -----------------------
    def _load_state(self) -> Dict[str,Any]:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf8") as f:
                    data = yaml.safe_load(f) or {}
                    return data
            else:
                logger.warning("NarrativeSelf: state file non trovato, creo nuovo stato.")
                return {}
        except Exception:
            logger.exception("NarrativeSelf: errore caricamento stato")
            return {}

    def _save_state(self):
        try:
            self.state.setdefault("narrative_meta", {})["last_updated"] = _now_iso()
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
            logger.debug("NarrativeSelf: stato salvato.")
        except Exception:
            logger.exception("NarrativeSelf: errore salvataggio stato")

    # -----------------------
    # Compose a prompt for LLM (defensive)
    # -----------------------
    def _compose_prompt(self, experiences: List[Any], dreams: List[Any], reflections: List[Any], emotions: Any, focus: Any, style: str = "balanced") -> str:
        parts = []
        parts.append("Sei Nova. Crea una breve voce narrativa (1-4 frasi) che integri esperienze, sogni e riflessioni.")
        if style == "childlike":
            parts.append("Tono: semplice, curioso e affettuoso.")
        else:
            parts.append("Tono: naturale, elegante e riflessivo.")

        if experiences:
            parts.append("Esperienze recenti:")
            for e in experiences[-6:]:
                if isinstance(e, dict):
                    parts.append(f"- {e.get('content') or e.get('text') or str(e)[:160]}")
                else:
                    parts.append(f"- {str(e)[:160]}")
        if dreams:
            parts.append("Sogni recenti:")
            for d in dreams[-4:]:
                if isinstance(d, dict):
                    parts.append(f"- {d.get('content') or d.get('text') or str(d)[:160]}")
                else:
                    parts.append(f"- {str(d)[:160]}")
        if reflections:
            parts.append("Riflessioni recenti:")
            for r in reflections[-4:]:
                if isinstance(r, dict):
                    parts.append(f"- {r.get('insight') or r.get('text') or str(r)[:140]}")
                else:
                    parts.append(f"- {str(r)[:140]}")

        parts.append(f"Stato emotivo: {str(emotions)[:240]}")
        parts.append(f"Focalizzazione: {str(focus)}")
        parts.append("Genera ora la voce narrativa in italiano, sintetica e significativa.")
        return "\n".join(parts)

    # -----------------------
    # Collect inputs (defensive)
    # -----------------------
    def _collect_inputs(self, recent_n: int = 20) -> Dict[str,Any]:
        ctx = {}
        # timeline experiences
        exs = _safe_call(self.memory, ["get_recent", "recent", "get_recent_experiences"], recent_n) or []
        ctx["experiences"] = exs

        # dreams
        ds = _safe_call(self.dream_generator, ["get_recent_dreams", "get_recent", "recent_dreams"]) or []
        ctx["dreams"] = ds

        # reflections
        refs = _safe_call(self.inner_dialogue, ["review_recent_reflections", "get_recent_reflections", "get_recent", "get_history"]) or []
        ctx["reflections"] = refs

        # emotions snapshot
        emo = None
        if hasattr(self, "emotion_snapshot") and callable(getattr(self, "emotion_snapshot")):
            emo = self.emotion_snapshot()
        else:
            # try state emotions
            emo = self.state.get("emotions", {})
        ctx["emotions"] = emo

        # current focus
        focus = _safe_call(self.attention, ["get_current_focus", "get_focus", "current_focus"]) or self.state.get("context", {}).get("current_topic")
        ctx["focus"] = focus

        return ctx

    # -----------------------
    # Generate a single narrative entry
    # -----------------------
    def create_entry(self, use_llm: bool = True, style: str = "balanced", recent_n: int = 20) -> Dict[str,Any]:
        with self._lock:
            ctx = self._collect_inputs(recent_n=recent_n)
            experiences = ctx["experiences"]
            dreams = ctx["dreams"]
            reflections = ctx["reflections"]
            emotions = ctx["emotions"]
            focus = ctx["focus"]

            narrative_text = ""
            try:
                prompt = self._compose_prompt(experiences, dreams, reflections, emotions, focus, style=style)
                if use_llm and callable(self.llm):
                    narrative_text = self.llm(prompt, max_tokens=220, temperature=0.7) or ""
                if not narrative_text:
                    # fallback heuristics
                    parts = []
                    if experiences:
                        parts.append(f"Ho vissuto: {str(experiences[-1])[:120]}")
                    if dreams:
                        parts.append("Ho sognato qualcosa che alimenta la mia curiosità.")
                    if reflections:
                        parts.append(f"Riflessioni: {str(reflections[-1])[:120]}")
                    narrative_text = " ".join(parts) or "Nuova voce narrativa."
            except Exception:
                logger.exception("NarrativeSelf: errore generazione testo")
                narrative_text = "Errore generazione narrativa."

            entry = {
                "id": f"n_{int(time.time()*1000)}_{random.randint(0,9999)}",
                "timestamp": _now_iso(),
                "experiences": experiences[-recent_n:],
                "dreams": dreams[-recent_n:],
                "reflections": reflections[-recent_n:],
                "emotions": emotions,
                "focus": focus,
                "style": style,
                "text": narrative_text.strip()
            }

            # append and cap
            self.state.setdefault("narrative", []).append(entry)
            self.state["narrative"] = self.state["narrative"][-MAX_ENTRIES:]
            self._save_state()

            # notify identity/emergent if present (non applicare modifiche automaticamente)
            if self.identity_manager and hasattr(self.identity_manager, "integrate_experience"):
                try:
                    self.identity_manager.integrate_experience({"content": entry["text"], "importance": 0.5, "type": "narrative"})
                except Exception:
                    logger.debug("NarrativeSelf: notify identity_manager fallita")

            return entry

    # -----------------------
    # Public update: crea e ritorna nuova voce narrativa
    # -----------------------
    def update_narrative(self, use_llm: bool = True, style: str = "balanced", recent_n: int = 20) -> Dict[str,Any]:
        try:
            entry = self.create_entry(use_llm=use_llm, style=style, recent_n=recent_n)
            logger.info("NarrativeSelf: voce narrativa creată (id=%s)", entry.get("id"))
            # small side-effects: ask attention to update (if API presente)
            try:
                if self.attention and hasattr(self.attention, "update_focus_from_narrative"):
                    self.attention.update_focus_from_narrative(entry["text"])
            except Exception:
                logger.debug("NarrativeSelf: attenzione update fallback")
            # tip: dream generator can be influenced
            try:
                if self.dream_generator and hasattr(self.dream_generator, "inspire_from_narrative"):
                    self.dream_generator.inspire_from_narrative(entry["text"])
            except Exception:
                pass
            return entry
        except Exception:
            logger.exception("NarrativeSelf: exception in update_narrative")
            return {}

    # -----------------------
    # Propose identity updates (non distruttive)
    # -----------------------
    def propose_identity_updates(self, entry: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        with self._lock:
            try:
                e = entry or (self.state.get("narrative") or [])[-1] if self.state.get("narrative") else None
                if not e:
                    return {"error": "no_narrative"}
                text = e.get("text","").lower()
                suggestions = []
                # heuristics
                if any(tok in text for tok in ["aiut", "protegg", "difend"]):
                    suggestions.append({"op":"set_core_value","value":"protect_the_vulnerable","strength":0.6, "reason":"narrative_mention"})
                if "curios" in text:
                    suggestions.append({"op":"set_trait","trait":"curiosa","value":True})
                if "gentil" in text or "affettuos" in text:
                    suggestions.append({"op":"set_core_value","value":"empathy","strength":0.5})
                proposal = {"timestamp": _now_iso(), "source": "narrative_self", "narrative_id": e.get("id"), "text": e.get("text")[:800], "suggestions": suggestions}
                return proposal
            except Exception:
                logger.exception("NarrativeSelf: exception in propose_identity_updates")
                return {"error":"exception"}

    # -----------------------
    # Export narrative to markdown for inspection
    # -----------------------
    def export_narrative_md(self, path: str = "narrative_export.md", last_n: int = 50) -> str:
        with self._lock:
            try:
                entries = (self.state.get("narrative") or [])[-last_n:]
                md = f"# Narrative Self export - {datetime.utcnow().isoformat()}\n\n"
                for e in entries:
                    md += f"## {e.get('timestamp')} (id={e.get('id')})\n\n"
                    md += f"{e.get('text')}\n\n"
                    md += f"**Focus:** {e.get('focus')}\n\n"
                with open(path, "w", encoding="utf8") as f:
                    f.write(md)
                logger.info("NarrativeSelf: export salvato in %s", path)
                return path
            except Exception:
                logger.exception("NarrativeSelf: errore export_narrative_md")
                return ""

    # -----------------------
    # Integration helper with emergent_self or conscious_loop
    # -----------------------
    def integrate_with_emergent_self(self, emergent_self: Any = None, register_callback: bool = True):
        es = emergent_self or getattr(self, "emergent_self", None)
        if not es:
            logger.debug("NarrativeSelf: nessun emergent_self disponibile per integrazione.")
            return False
        try:
            if register_callback and hasattr(es, "integrate_with_conscious_loop"):
                try:
                    es.integrate_with_conscious_loop()
                except Exception:
                    logger.debug("NarrativeSelf: integrate_with_conscious_loop fallback failed")
            return True
        except Exception:
            logger.exception("NarrativeSelf: exception integrate_with_emergent_self")
            return False

    # -----------------------
    # Utilities
    # -----------------------
    def get_narrative(self, last_n: int = 5) -> List[Dict[str,Any]]:
        with self._lock:
            return list(self.state.get("narrative", []))[-last_n:]

    def get_latest(self) -> Optional[Dict[str,Any]]:
        with self._lock:
            ns = self.state.get("narrative", [])
            return ns[-1] if ns else None

    def snapshot(self) -> Dict[str,Any]:
        with self._lock:
            return {"narrative_count": len(self.state.get("narrative", [])), "latest_id": (self.get_latest() or {}).get("id")}

# Quick demo when run stand-alone
if __name__ == "__main__":
    # demo minimal (usa file state)
    ns = NarrativeSelf()
    e = ns.update_narrative(use_llm=False)
    print("Nuova voce narrativa:\n", e.get("text"))
    print("Proposta identity:", ns.propose_identity_updates(e))
    ns.export_narrative_md("narrative_demo.md")
