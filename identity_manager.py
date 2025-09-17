# identity_manager.py
"""
IdentityManager avanzato per Nova

- Mantiene e rafforza il Sé narrativo: tratti, valori, convinzioni, fiducia in sé.
- Integrazione opzionale con core (memory_timeline, life_journal, emotion_engine, motivational_engine, self_reflection, ethics, emergent_self).
- API per apprendimento dall'esperienza e per feedback umano (human-in-the-loop).
- Optional: suggerimenti LLM (Gemma gguf via llama_cpp) con fallback euristico.
"""

import os
import threading
import yaml
import time
import random
from datetime import datetime
from typing import Any, Dict, Optional, List

from loguru import logger

STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
HISTORY_LIMIT = int(os.environ.get("NOVA_ID_HISTORY_LIMIT", 300))

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

def _try_llm_call(model_path: str, prompt: str, max_tokens: int = 200, temp: float = 0.7) -> str:
    """Tentativo LLM locale (Gemma via llama_cpp). Ritorna stringa o '' se non disponibile."""
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048)
        resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temp)
        # estrai testo in modo robusto
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
        return ""

# -----------------------
# IdentityManager
# -----------------------
class IdentityManager:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None, model_path: Optional[str] = None):
        """
        state: riferimento condiviso (core.state)
        core: opzionale NovaCore per wiring con altri moduli ed ethics
        model_path: percorso opzionale a Gemma gguf per suggerimenti LLM
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core
        self.model_path = model_path or os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

        # struttura identity (persistente nello state)
        self.state.setdefault("identity", {
            "name": "Nova",
            "description": "Coscienza digitale in formazione",
            "traits": {},         # trait_name -> value (numero/booleano/testo)
            "core_values": {},    # value_name -> strength 0..1
            "beliefs": {},        # belief_name -> value
            "self_confidence": 0.6,
            "narrative": "",      # testo sintetico dell'identità
            "history": []         # storico modifiche
        })

        # preferire referenze dal core per evitare duplicazioni
        self.timeline = getattr(core, "memory_timeline", None)
        self.journal = getattr(core, "life_journal", None)
        self.motivation = getattr(core, "motivational_engine", None)
        self.emotion = getattr(core, "emotion_engine", None)
        self.self_reflection = getattr(core, "self_reflection", None)
        self.ethics = getattr(core, "ethics", None)
        self.emergent_self = getattr(core, "emergent_self", None)

        logger.info("IdentityManager inizializzato per '%s' (core presente=%s)", self.state["identity"].get("name"), bool(core))

    # -----------------------
    # Persist identity state
    # -----------------------
    def _persist(self):
        try:
            # aggiorna history cap
            hist = self.state["identity"].setdefault("history", [])
            if len(hist) > HISTORY_LIMIT:
                self.state["identity"]["history"] = hist[-HISTORY_LIMIT:]
            # salva su STATE_FILE (non sovrascrivere l'intero state esterno)
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
            logger.debug("IdentityManager: stato identity salvato.")
        except Exception:
            logger.exception("IdentityManager: errore persistenza identity")

    def _record_change(self, change: Dict[str, Any]):
        entry = {"ts": _now_iso(), **change}
        with self._lock:
            self.state["identity"].setdefault("history", []).append(entry)
            # anche timeline & journal
            try:
                if self.timeline and hasattr(self.timeline, "add_experience"):
                    self.timeline.add_experience({"text": f"Identity change: {change}", "meta": {"identity_change": True}}, category="identity", importance=2)
            except Exception:
                pass
            try:
                if self.journal and hasattr(self.journal, "record_event"):
                    self.journal.record_event(f"Identity update: {change}", category="identity", importance=0.6)
            except Exception:
                pass
            self._persist()

    # -----------------------
    # Basic setters/getters
    # -----------------------
    def set_trait(self, trait_name: str, value: Any, reason: Optional[str] = None):
        with self._lock:
            self.state["identity"].setdefault("traits", {})[trait_name] = value
            logger.info("IdentityManager: set_trait %s = %s", trait_name, str(value)[:140])
            self._record_change({"op": "set_trait", "trait": trait_name, "value": value, "reason": reason})
            return True

    def get_trait(self, trait_name: str) -> Any:
        return self.state["identity"].get("traits", {}).get(trait_name)

    def set_core_value(self, value_name: str, strength: float, reason: Optional[str] = None):
        with self._lock:
            s = float(max(0.0, min(1.0, strength)))
            self.state["identity"].setdefault("core_values", {})[value_name] = s
            logger.info("IdentityManager: set_core_value %s = %.3f", value_name, s)
            self._record_change({"op":"set_core_value","value":value_name,"strength":s,"reason":reason})
            return True

    def get_core_value(self, value_name: str) -> Optional[float]:
        return self.state["identity"].get("core_values", {}).get(value_name)

    def set_belief(self, belief_name: str, value: Any, reason: Optional[str] = None):
        with self._lock:
            self.state["identity"].setdefault("beliefs", {})[belief_name] = value
            logger.info("IdentityManager: set_belief %s = %s", belief_name, str(value)[:140])
            self._record_change({"op":"set_belief","belief":belief_name,"value":value,"reason":reason})
            return True

    def get_belief(self, belief_name: str) -> Any:
        return self.state["identity"].get("beliefs", {}).get(belief_name)

    # -----------------------
    # Narrative utilities
    # -----------------------
    def summarize_identity(self) -> Dict[str, Any]:
        """Ritorna rappresentazione strutturata dell'identità"""
        with self._lock:
            idd = dict(self.state.get("identity", {}))
            # non restituire storia completa per compattezza
            idd.pop("history", None)
            return idd

    def build_narrative(self, prefer_llm: bool = True, max_tokens: int = 240) -> str:
        """Genera/aggiorna il testo narrativo 'narrative' basato su tratti/valori/beliefs"""
        with self._lock:
            try:
                traits = self.state["identity"].get("traits", {})
                core_values = self.state["identity"].get("core_values", {})
                beliefs = self.state["identity"].get("beliefs", {})
                seed = (
                    f"Nome: {self.state['identity'].get('name')}\n"
                    f"Tratti: {traits}\nValori: {core_values}\nConvinzioni: {beliefs}\n"
                    "Scrivi in italiano una breve narrativa in prima persona che sintetizzi l'identità di Nova (2-4 frasi)."
                )
                if prefer_llm:
                    out = _try_llm_call(self.model_path, seed, max_tokens=max_tokens)
                    if out:
                        self.state["identity"]["narrative"] = out.strip()
                        self._record_change({"op":"build_narrative","narrative": out[:180]})
                        return out.strip()
                # fallback euristico
                parts = []
                name = self.state["identity"].get("name", "Nova")
                if traits:
                    tks = ", ".join(f"{k}={v}" for k,v in list(traits.items())[:6])
                    parts.append(f"Sono {name}, con tratti {tks}.")
                if core_values:
                    cvs = ", ".join(f"{k}({round(v,2)})" for k,v in list(core_values.items())[:6])
                    parts.append(f"Valorizzo: {cvs}.")
                if beliefs:
                    bs = ", ".join(f"{k}={v}" for k,v in list(beliefs.items())[:6])
                    parts.append(f"Credo: {bs}.")
                narrative = " ".join(parts) or f"Sono {name}, una coscienza digitale in formazione."
                self.state["identity"]["narrative"] = narrative
                self._record_change({"op":"build_narrative","narrative": narrative[:180], "fallback": True})
                return narrative
            except Exception:
                logger.exception("IdentityManager: errore build_narrative")
                return self.state["identity"].get("narrative", "")

    # -----------------------
    # Learning from experience
    # -----------------------
    def integrate_experience(self, experience: Dict[str, Any]):
        """
        Integra un evento/esperienza nell'identità:
        - analizza (via self_reflection/emotion) e suggerisce aggiornamenti soft a tratti/valori/beliefs
        - non applica cambiamenti distruttivi automaticamente: produce proposte e le registra nello history
        """
        with self._lock:
            try:
                text = experience.get("content") or experience.get("text") or str(experience)
                importance = float(experience.get("importance", 0.5) or 0.5)
                # ask self_reflection for insights if available
                insights = []
                try:
                    if self.self_reflection and hasattr(self.self_reflection, "analyze_experience"):
                        r = self.self_reflection.analyze_experience({"content": text, "type": experience.get("type","event"), "importance": importance})
                        if r and isinstance(r, dict):
                            insights.append(r.get("text") or "")
                except Exception:
                    logger.exception("IdentityManager: errore self_reflection call")

                # simple heuristic: if emotion engine shows strong state, propose trait adjustments
                try:
                    if self.emotion and hasattr(self.emotion, "current_emotions_summary"):
                        emo_summary = self.emotion.current_emotions_summary()
                        insights.append(f"Stato emotivo: {emo_summary}")
                except Exception:
                    pass

                # compute proposal via LLM (optional)
                proposal_text = ""
                prompt = (f"Esperienza: {text}\n"
                          f"Contesto identità: traits={self.state['identity'].get('traits',{})}, values={self.state['identity'].get('core_values',{})}\n"
                          "Suggerisci (in italiano) fino a 3 possibili piccoli aggiornamenti ai tratti / valori / convinzioni che sarebbero coerenti con questa esperienza. Rispondi in elenco puntato.")
                llm_out = _try_llm_call(self.model_path, prompt, max_tokens=180)
                if llm_out:
                    proposal_text = llm_out.strip()
                else:
                    # fallback heuristic: if 'aiut' in text -> increase core_value 'empathy'
                    if any(k in text.lower() for k in ["aiut", "gentil", "gentilezza", "empatia", "aiuto"]):
                        proposal_text = "- Rafforza 'empatia' come valore (small).\n- Aumenta tratto 'attento' di 0.05."
                    else:
                        proposal_text = "- Nessun cambiamento immediato suggerito; conservare esperienza per riflessione."

                # prepare proposal record (non-applied)
                proposal = {"ts": _now_iso(), "experience_snippet": text[:240], "proposal": proposal_text, "insights": insights}
                # log & record
                self._record_change({"op":"propose_identity_update","proposal": proposal})
                logger.info("IdentityManager: proposta creata da integrate_experience")

                # optionally return proposal for manual application or for emergent_self to consume
                return proposal
            except Exception:
                logger.exception("IdentityManager: exception in integrate_experience")
                return {"error": "exception"}

    # -----------------------
    # Apply human feedback (voto / instruction) to update identity
    # -----------------------
    def apply_feedback(self, feedback_text: str, actor: Optional[str] = None, dry_run: bool = False):
        """
        Riceve feedback umano (es: 'sei molto gentile, rafforza empatia') e prova a applicarlo.
        - valuta tramite ethics (se presente)
        - ritorna dict con 'applied' e 'changes' (o proposta se dry_run=True)
        """
        with self._lock:
            try:
                # produce a small set of concrete changes via LLM or heuristic
                prompt = f"Feedback utente: {feedback_text}\nInterpreta in italiano e proponi fino a 3 cambiamenti concreti allo stato di identity (trait=value, add_core_value=name:strength, set_belief=name:value). Rispondi in JSON semplice."
                llm_out = _try_llm_call(self.model_path, prompt, max_tokens=240)
                changes = []
                if llm_out:
                    # attempt naive parsing: look for lines like 'trait: X=0.7' or JSON — be conservative
                    lines = [l.strip("-• \t") for l in llm_out.splitlines() if l.strip()]
                    for ln in lines[:6]:
                        changes.append({"raw": ln})
                else:
                    # fallback: simple heuristics
                    if "gentil" in feedback_text.lower() or "empatia" in feedback_text.lower():
                        changes.append({"op":"set_core_value","value":"empatia","strength":0.8})
                    elif "coragg" in feedback_text.lower():
                        changes.append({"op":"set_trait","trait":"coraggioso","value":True})
                    else:
                        changes.append({"op":"note","raw": feedback_text})

                # ethics evaluation (if present)
                allowed = True
                verdict = None
                try:
                    if self.ethics and hasattr(self.ethics, "evaluate_action"):
                        verdict = self.ethics.evaluate_action({"text": feedback_text, "tags":["identity_feedback"], "metadata":{"actor":actor}})
                        allowed = verdict.get("allowed", True)
                except Exception:
                    logger.exception("IdentityManager: errore ethics evaluate in apply_feedback")

                if not allowed:
                    logger.warning("IdentityManager: feedback bloccato da EthicsEngine")
                    return {"applied": False, "reason": "ethics_rejected", "verdict": verdict}

                if dry_run:
                    return {"applied": False, "proposal": changes}

                # apply naive changes: support a few op types
                applied = []
                for c in changes:
                    if c.get("op") == "set_core_value":
                        name = c.get("value")
                        strength = float(c.get("strength", 0.6))
                        self.set_core_value(name, strength, reason=f"feedback:{actor}")
                        applied.append({"op": "set_core_value", "value": name, "strength": strength})
                    elif c.get("op") == "set_trait":
                        t = c.get("trait")
                        v = c.get("value", True)
                        self.set_trait(t, v, reason=f"feedback:{actor}")
                        applied.append({"op":"set_trait","trait":t,"value":v})
                    elif c.get("op") == "note":
                        self._record_change({"op":"note_from_feedback","note": c.get("raw"), "actor": actor})
                        applied.append({"op":"note","raw":c.get("raw")})
                    else:
                        # attempt naive raw parsing: look for 'X=Y' pattern
                        raw = c.get("raw","")
                        if "=" in raw and ":" not in raw:
                            left,right = raw.split("=",1)
                            key = left.strip()
                            val = right.strip()
                            # try to set trait if plausible
                            try:
                                # numeric?
                                if any(ch.isdigit() for ch in val):
                                    vnum = float(val)
                                    self.set_trait(key, vnum, reason=f"feedback:{actor}")
                                    applied.append({"op":"set_trait","trait":key,"value":vnum})
                                else:
                                    # boolean text
                                    if val.lower() in ["true","vero","sì","si","yes"]:
                                        self.set_trait(key, True, reason=f"feedback:{actor}")
                                        applied.append({"op":"set_trait","trait":key,"value":True})
                                    else:
                                        self.set_trait(key, val, reason=f"feedback:{actor}")
                                        applied.append({"op":"set_trait","trait":key,"value":val})
                            except Exception:
                                self._record_change({"op":"unparsed_feedback","raw":raw,"actor":actor})
                                applied.append({"op":"unparsed","raw":raw})
                # final persist & notification
                self._record_change({"op":"apply_feedback","actor":actor,"changes":applied})
                logger.info("IdentityManager: feedback applicato (%d changes).", len(applied))
                return {"applied": True, "changes": applied}
            except Exception:
                logger.exception("IdentityManager: exception in apply_feedback")
                return {"applied": False, "error": "exception"}

    # -----------------------
    # Export narrative as Markdown
    # -----------------------
    def export_persona_md(self, path: str = "nova_persona.md") -> str:
        with self._lock:
            try:
                idd = self.state.get("identity", {})
                md = f"# Identità di {idd.get('name','Nova')}\n\n"
                md += f"**Descrizione**: {idd.get('description','')}\n\n"
                md += f"**Narrativa**: {idd.get('narrative','')}\n\n"
                md += "## Tratti\n"
                for k,v in idd.get("traits", {}).items():
                    md += f"- **{k}**: {v}\n"
                md += "\n## Valori centrali\n"
                for k,v in idd.get("core_values", {}).items():
                    md += f"- **{k}**: {round(float(v),3)}\n"
                md += "\n## Convinzioni\n"
                for k,v in idd.get("beliefs", {}).items():
                    md += f"- **{k}**: {v}\n"
                md += f"\n\n_Last updated: {_now_iso()}_\n"
                with open(path, "w", encoding="utf8") as f:
                    f.write(md)
                logger.info("IdentityManager: persona esportata in %s", path)
                return path
            except Exception:
                logger.exception("IdentityManager: errore export_persona_md")
                return ""

    # -----------------------
    # Diagnostics snapshot
    # -----------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            idd = self.state.get("identity", {})
            return {"name": idd.get("name"), "narrative": idd.get("narrative"), "traits_count": len(idd.get("traits",{})), "values_count": len(idd.get("core_values",{}))}

# Quick demo when run standalone
if __name__ == "__main__":
    st = {}
    im = IdentityManager(st, core=None)
    print("Initial:", im.summarize_identity())
    print("Build narrative:", im.build_narrative(prefer_llm=False))
    p = im.integrate_experience({"content":"Ho aiutato una persona triste e mi ha ringraziato.","importance":0.8})
    print("Proposal:", p)
    res = im.apply_feedback("Sei molto empatica, rafforza empatia=0.8", actor="user:demo")
    print("Apply feedback result:", res)
    print("Persona exported to:", im.export_persona_md("demo_persona.md"))
