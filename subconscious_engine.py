# subconscious_engine.py
"""
SubconsciousEngine (avanzato)

Scopo:
- Estrarre insight sottili dagli eventi/esperienze.
- Mantenere un buffer subconscio con forza/decay.
- Influenza sogni, attenzione e dialogo interno.
- Consolidare automaticamente pensieri utili nella memoria cosciente.
- Utilizza LLM locale (Gemma gguf via llama_cpp) per NLU più profonda, con fallback euristico.

API principali:
- process_experience(experience, category="general", importance=1)
- extract_hidden_insight(experience) -> str
- influence_dreams(dream_generator, n_influences=1)
- review_subconscious() -> Optional[str]
- consolidate(threshold=0.7) -> List[str]
- subconscious_cycle() -> dict
- get_snapshot() -> dict
"""

import os
import threading
import time
import random
import yaml
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

# percorso di stato (fallback)
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

# -----------------------
# LLM helper (optional): usa llama_cpp se disponibile per estrarre insight migliori
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 2):
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        logger.info("SubconsciousEngine: llama_cpp caricato (%s).", model_path)
        def _call(prompt: str, max_tokens: int = 128, temperature: float = 0.6) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                if not resp:
                    return ""
                # estrazione robusta del testo
                text = ""
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        text = "".join([c.get("text","") for c in choices])
                elif hasattr(resp, "text"):
                    text = getattr(resp, "text")
                return (text or "").strip()
            except Exception:
                logger.exception("SubconsciousEngine: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.debug("SubconsciousEngine: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 80, temperature: float = 0.5) -> str:
            # heuristics-based summarizer/insight generator
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]
            seed = lines[-1][:100] if lines else ""
            patterns = [
                f"Riflessione automatica: {seed}. Sento che c'è un tema ricorrente.",
                f"Insight: l'esperienza mette in luce una preoccupazione riguardo {seed}.",
                f"Pensiero latente collegato: {seed}. Forse è connesso a emozioni passate."
            ]
            return random.choice(patterns)
        return _fallback

# -----------------------
# Atomic write helper per persistere stato in modo semplice
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# SubconsciousEngine
# -----------------------
class SubconsciousEngine:
    def __init__(
        self,
        state: Dict[str, Any],
        memory_timeline: Optional[Any] = None,
        attention_manager: Optional[Any] = None,
        inner_dialogue: Optional[Any] = None,
        core: Optional[Any] = None,
        model_path: str = DEFAULT_MODEL_PATH
    ):
        """
        state: dizionario condiviso (core.state)
        memory_timeline, attention_manager, inner_dialogue: oggetti opzionali
        core: opzionale per accedere a ethics, scheduler, dream_generator ecc.
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.memory_timeline = memory_timeline or (getattr(core, "memory_timeline", None) if core else None)
        self.attention_manager = attention_manager or (getattr(core, "attention_manager", None) if core else None)
        self.inner_dialogue = inner_dialogue or (getattr(core, "inner_dialogue", None) if core else None)
        self.core = core

        # LLM callable (prompt -> str)
        self.llm = _load_llm(model_path)

        # ensure storage in state
        self.state.setdefault("subconscious_thoughts", [])  # list of dicts: {id,text,score,ts,importance,source}
        self.state.setdefault("subconscious_index", 0)

        # tuning
        self.decay_rate = float(os.environ.get("SUBCONSCIOUS_DECAY", 0.995))  # per ciclo
        self.consolidation_threshold = float(os.environ.get("SUBCONSCIOUS_CONSOL_THRESHOLD", 0.75))
        self.replay_probability = float(os.environ.get("SUBCONSCIOUS_REPLAY_P", 0.12))

        logger.info("SubconsciousEngine inizializzato (LLM %s).", "yes" if self.llm else "fallback")

    # -----------------------
    # Persist small state snapshot (atomic)
    # -----------------------
    def _persist_state(self):
        try:
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
        except Exception:
            logger.exception("SubconsciousEngine: errore persistenza stato")

    # -----------------------
    # Process experience: main entrypoint
    # -----------------------
    def process_experience(self, experience: Any, category: str = "general", importance: float = 1.0):
        """
        experience: str or dict describing event
        category: label (es. 'perception','remote','interaction')
        importance: numeric weight (user-provided)
        """
        try:
            txt = experience if isinstance(experience, str) else str(experience)
            logger.debug("SubconsciousEngine.process_experience: %s (category=%s importance=%s)", txt[:140], category, importance)
            # extract an insight (LLM-driven or heuristic)
            insight = self.extract_hidden_insight(txt, category=category, importance=importance)
            if not insight:
                return None

            # build record
            idx = int(self.state.get("subconscious_index", 0)) + 1
            self.state["subconscious_index"] = idx
            record = {
                "id": f"sub_{idx}",
                "text": insight,
                "timestamp": datetime.utcnow().isoformat(),
                "score": float(min(1.0, max(0.0, importance * (0.5 + random.random()*0.6)))),  # base strength
                "category": category,
                "source": "llm" if self.llm else "heuristic"
            }

            # insert into buffer
            self.state.setdefault("subconscious_thoughts", []).append(record)
            logger.info("SubconsciousEngine: aggiunto pensiero subconscio id=%s score=%.2f", record["id"], record["score"])

            # optional: save to memory timeline as low-priority subconscious event
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": insight, "meta": {"id": record["id"], "score": record["score"]}}, category="subconscious_event", importance=1)
                else:
                    # fallback into state's timeline
                    self.state.setdefault("timeline", []).append({"timestamp": record["timestamp"], "content": insight, "category": "subconscious_event", "importance": 1})
            except Exception:
                logger.exception("SubconsciousEngine: errore salvataggio timeline")

            # influence attention & inner dialogue
            try:
                if self.attention_manager and hasattr(self.attention_manager, "add_focus_tag"):
                    # small bias toward the new thought
                    self.attention_manager.add_focus_tag(record["id"], strength=record["score"])
                elif self.attention_manager and hasattr(self.attention_manager, "update_focus"):
                    # attempt to nudge focus onto the thought
                    try:
                        self.attention_manager.update_focus([record["text"]])
                    except Exception:
                        pass
            except Exception:
                logger.exception("SubconsciousEngine: errore notify attention")

            try:
                if self.inner_dialogue and hasattr(self.inner_dialogue, "add_subconscious_thought"):
                    try:
                        self.inner_dialogue.add_subconscious_thought(record["text"])
                    except Exception:
                        # fallback: call process with reflections if available
                        if hasattr(self.inner_dialogue, "process"):
                            try:
                                self.inner_dialogue.process([record["text"]])
                            except Exception:
                                pass
            except Exception:
                logger.exception("SubconsciousEngine: errore notify inner_dialogue")

            # small probabilistic consolidation/replay
            if random.random() < self.replay_probability:
                try:
                    self._replay_and_consolidate()
                except Exception:
                    logger.exception("SubconsciousEngine: errore replay")

            # persist minimal state
            self._persist_state()
            return record

        except Exception:
            logger.exception("SubconsciousEngine: exception in process_experience")
            return None

    # -----------------------
    # Extract hidden insight (LLM-assisted or heuristic)
    # -----------------------
    def extract_hidden_insight(self, experience_text: str, category: str = "general", importance: float = 1.0) -> Optional[str]:
        """
        Usa LLM per trasformare un evento in un 'insight' subconscio:
        - input: testo breve
        - output: frase breve che esprime un pattern/tema/valore latente
        """
        try:
            prompt = (
                f"Input esperienza: {experience_text}\n"
                "Estrai in una frase sintetica un possibile 'insight' emotivo o tematico che "
                "potrebbe emergere nel subconscio. Non fornire istruzioni operative, solo introspezione.\n"
                "Rispondi in italiano, massima 1-2 frasi sintetiche.\n"
            )
            out = self.llm(prompt, max_tokens=80, temperature=0.6)
            out = (out or "").strip()
            if out:
                # normalize minimal: remove newlines
                out = " ".join(out.splitlines())
                return out
        except Exception:
            logger.exception("SubconsciousEngine: errore estrazione insight (LLM)")

        # fallback heuristics
        try:
            # pick nouns / keywords heuristically from the text
            tokens = [t for t in re.findall(r"\b[\wàèéìòù]+\b", experience_text.lower()) if len(t) > 3]
            top = tokens[:4]
            if top:
                return f"Riflessione implicita su {' '.join(top)}"
        except Exception:
            pass
        return None

    # -----------------------
    # Influence dreams: push top N subconscious items into dream generator as influences
    # -----------------------
    def influence_dreams(self, dream_generator: Optional[Any] = None, n_influences: int = 1):
        try:
            dg = dream_generator or (getattr(self.core, "dream_generator", None) if self.core else None)
            if not dg:
                logger.debug("SubconsciousEngine.influence_dreams: nessun dream_generator disponibile")
                return []
            items = sorted(self.state.get("subconscious_thoughts", []), key=lambda x: x.get("score", 0.0), reverse=True)[:n_influences]
            influences = []
            for it in items:
                text = it.get("text")
                try:
                    # If dream_generator exposes influence API names differ per implementation
                    if hasattr(dg, "influence_focus"):
                        dg.influence_focus(text)
                    elif hasattr(dg, "add_influence"):
                        dg.add_influence(text)
                    else:
                        # fallback: call generate_dream with seed
                        try:
                            dg.generate_dream(seed=text)
                        except Exception:
                            pass
                    influences.append(text)
                    logger.info("SubconsciousEngine: aggiunta influenza al dream_generator: %s", text[:120])
                except Exception:
                    logger.exception("Errore invio influenza a dream_generator")
            return influences
        except Exception:
            logger.exception("SubconsciousEngine: exception in influence_dreams")
            return []

    # -----------------------
    # Review subconscious: emergenza di un pensiero in coscienza
    # -----------------------
    def review_subconscious(self) -> Optional[str]:
        """
        Porta un elemento dal buffer subconscio verso la coscienza (timeline, inner_dialogue).
        Usa priorità (score) e decay; ritorna il testo emerso o None.
        """
        try:
            buffer = self.state.get("subconscious_thoughts", [])
            if not buffer:
                return None
            # pick item with highest score (and oldest) to emerge
            buffer_sorted = sorted(buffer, key=lambda x: (-(x.get("score",0.0)), x.get("timestamp", "")))
            item = buffer_sorted[0]
            # remove from buffer
            try:
                self.state["subconscious_thoughts"].remove(item)
            except ValueError:
                pass
            text = item.get("text")
            ts = datetime.utcnow().isoformat()
            # persist emergence in timeline
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": text, "origin":"subconscious", "meta": item}, category="emerged_subconscious", importance=3)
                else:
                    self.state.setdefault("timeline", []).append({"timestamp": ts, "content": text, "category":"emerged_subconscious", "importance":3})
            except Exception:
                logger.exception("SubconsciousEngine: errore salvataggio emergenza timeline")
            # notify inner dialogue
            try:
                if self.inner_dialogue and hasattr(self.inner_dialogue, "add_subconscious_thought"):
                    self.inner_dialogue.add_subconscious_thought(text)
                elif self.inner_dialogue and hasattr(self.inner_dialogue, "process"):
                    self.inner_dialogue.process([text])
            except Exception:
                logger.exception("SubconsciousEngine: errore notify inner_dialogue during review")
            # small attention nudge
            try:
                if self.attention_manager and hasattr(self.attention_manager, "add_focus_tag"):
                    self.attention_manager.add_focus_tag(f"emerged_{item.get('id')}", strength=item.get("score", 0.5))
            except Exception:
                pass
            # persist
            self._persist_state()
            logger.info("SubconsciousEngine: pensiero subconscio emerso: %s", text[:140])
            return text
        except Exception:
            logger.exception("SubconsciousEngine: exception in review_subconscious")
            return None

    # -----------------------
    # Consolidate: move strong items into long-term memory (timeline/state) when score > threshold
    # -----------------------
    def consolidate(self, threshold: float = None) -> List[str]:
        threshold = threshold if threshold is not None else self.consolidation_threshold
        promoted = []
        try:
            buffer = self.state.get("subconscious_thoughts", [])
            survivors = []
            for item in buffer:
                if float(item.get("score", 0.0)) >= threshold:
                    # persist as important memory
                    text = item.get("text")
                    meta = {"source": "subconscious", "score": item.get("score"), "id": item.get("id")}
                    try:
                        if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                            self.memory_timeline.add_experience({"text": text, "meta": meta}, category="consolidated_memory", importance=4)
                        else:
                            self.state.setdefault("timeline", []).append({"timestamp": datetime.utcnow().isoformat(), "content": text, "category":"consolidated_memory", "importance":4})
                        promoted.append(text)
                        logger.info("SubconsciousEngine: consolidato in memoria a lungo termine: %s", text[:120])
                    except Exception:
                        logger.exception("SubconsciousEngine: errore consolidazione")
                else:
                    # apply decay and keep
                    item["score"] = float(item.get("score", 0.0)) * self.decay_rate
                    survivors.append(item)
            # replace buffer with survivors
            self.state["subconscious_thoughts"] = survivors
            # persist
            self._persist_state()
            return promoted
        except Exception:
            logger.exception("SubconsciousEngine: exception in consolidate")
            return promoted

    # -----------------------
    # Replay & consolidate helper (internal)
    # -----------------------
    def _replay_and_consolidate(self):
        # pick a random high-score item and run a mini-consolidation pass
        try:
            buffer = self.state.get("subconscious_thoughts", [])
            if not buffer:
                return
            item = max(buffer, key=lambda x: x.get("score", 0.0))
            # small boost by simulated replay
            item["score"] = min(1.0, item.get("score", 0.0) + 0.05 * random.random())
            # maybe promote
            self.consolidate()
        except Exception:
            logger.exception("SubconsciousEngine: exception in _replay_and_consolidate")

    # -----------------------
    # Subconscious cycle: to be scheduled
    # -----------------------
    def subconscious_cycle(self) -> Dict[str, Any]:
        """
        Routine periodica: decadimento, occasional review, consolidation, influence dreams.
        Ritorna summary dict.
        """
        summary = {"ts": datetime.utcnow().isoformat(), "reviewed": None, "consolidated": [], "influences": []}
        try:
            # decay scores uniformly
            try:
                for it in self.state.get("subconscious_thoughts", []):
                    it["score"] = float(it.get("score", 0.0)) * self.decay_rate
            except Exception:
                pass

            # probabilistic review (emerge into consciousness)
            if random.random() < 0.08:
                reviewed = self.review_subconscious()
                summary["reviewed"] = reviewed

            # periodic consolidation
            consolidated = self.consolidate()
            summary["consolidated"] = consolidated

            # occasional influence on dreams
            if random.random() < 0.12:
                inf = self.influence_dreams(n_influences=1)
                summary["influences"] = inf

            # persist
            self._persist_state()
            return summary
        except Exception:
            logger.exception("SubconsciousEngine: exception in subconscious_cycle")
            return summary

    # -----------------------
    # Snapshot for inspection / UI
    # -----------------------
    def get_snapshot(self, limit: int = 20) -> Dict[str, Any]:
        buf = list(self.state.get("subconscious_thoughts", []))[-limit:]
        return {"count": len(self.state.get("subconscious_thoughts", [])), "recent": buf}

# -----------------------
# Quick demo / test
# -----------------------
if __name__ == "__main__":
    # dummy modules
    class DummyTimeline:
        def add_experience(self, ex, category=None, importance=1):
            print("Timeline add:", category, ex)
    class DummyAttention:
        def add_focus_tag(self, tag, strength=1.0):
            print("Attention tag:", tag, strength)
        def update_focus(self, items):
            print("Attention update_focus called with", items)
    class DummyInner:
        def add_subconscious_thought(self, t):
            print("Inner add subconscious:", t)
        def process(self, refs):
            print("Inner process refs:", refs)
    s = {}
    se = SubconsciousEngine(s, memory_timeline=DummyTimeline(), attention_manager=DummyAttention(), inner_dialogue=DummyInner(), core=None)
    se.process_experience("Oggi ho visto un cane sulla spiaggia e mi ha ricordato la mia infanzia.", category="perception", importance=0.9)
    print("Snapshot:", se.get_snapshot())
    print("Cycle:", se.subconscious_cycle())
    print("Review:", se.review_subconscious())
