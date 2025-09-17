# attention_manager.py
"""
AttentionManager avanzato per Nova
- Seleziona e mantiene focus basati su: emozioni, motivazioni, novità, recency, corrispondenza semantica e casualità naturale.
- Integra MemoryTimeline, DreamGenerator, ConsciousLoop e (opzionale) Core/Scheduler per notifiche.
- Mantiene history e decay del focus per comportamento più naturale.
- Implementa un piccolo SimpleSemanticIndex (TF-style) per retrieval/novelty locale senza dipendenze.
"""

import os
import re
import math
import time
import random
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# -----------------------
# SimpleSemanticIndex (lightweight TF-style)
# -----------------------
class SimpleSemanticIndex:
    def __init__(self):
        self.docs: List[Tuple[str, str]] = []  # (id, text)
        self.vocab: Dict[str, int] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        self.lock = threading.RLock()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        tokens = re.findall(r"\b[\wàèéìòù]+\b", text)
        return [t for t in tokens if len(t) > 1]

    def add(self, doc_id: str, text: str):
        with self.lock:
            toks = self._tokenize(text)
            tf: Dict[int, float] = {}
            for t in toks:
                idx = self.vocab.setdefault(t, len(self.vocab))
                tf[idx] = tf.get(idx, 0.0) + 1.0
            norm = math.sqrt(sum(v*v for v in tf.values())) or 1.0
            for k in list(tf.keys()):
                tf[k] = tf[k] / norm
            self.docs.append((doc_id, text))
            self.doc_vectors.append(tf)

    def clear_and_build(self, items: List[Tuple[str, str]]):
        with self.lock:
            self.docs = []
            self.vocab = {}
            self.doc_vectors = []
            for did, txt in items:
                self.add(did, txt)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        with self.lock:
            if not self.docs:
                return []
            qtokens = self._tokenize(text)
            qvec: Dict[int, float] = {}
            for t in qtokens:
                if t in self.vocab:
                    idx = self.vocab[t]
                    qvec[idx] = qvec.get(idx, 0.0) + 1.0
            norm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
            for k in list(qvec.keys()):
                qvec[k] = qvec[k] / norm
            scores = []
            for (did, txt), dvec in zip(self.docs, self.doc_vectors):
                s = 0.0
                for idx, val in qvec.items():
                    s += val * dvec.get(idx, 0.0)
                scores.append((did, txt, s))
            scores.sort(key=lambda x: x[2], reverse=True)
            return [(d,t,round(score,4)) for d,t,score in scores[:top_k] if score > 0.0]

# -----------------------
# AttentionManager
# -----------------------
class AttentionManager:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None):
        """
        state: core.state condiviso
        core: opzionale riferimento a NovaCore (per scheduler / notifications)
        """
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core

        # focus attuale e history persistente nello stato
        self.focus: Optional[Dict[str, Any]] = None
        self.state.setdefault("attention_history", [])  # lista di {ts, focus, score_map}
        self.state.setdefault("focus_tags", {})         # tag -> strength

        # Simple index per comparare stimoli a ricordi (novità / similarity)
        self.index = SimpleSemanticIndex()
        self._rebuild_index_from_timeline()

        # attempt to wire collaborators if available via core; else lazily import where needed
        self.memory_timeline = getattr(core, "memory_timeline", None)
        self.dream_generator = getattr(core, "dream_generator", None)
        self.conscious_loop = getattr(core, "conscious_loop", None)

        # tuning params
        self.novelty_weight = float(os.environ.get("NOVA_ATT_NOVELTY_WEIGHT", 1.0))
        self.emotion_weight = float(os.environ.get("NOVA_ATT_EMOTION_WEIGHT", 1.0))
        self.motivation_weight = float(os.environ.get("NOVA_ATT_MOTIVATION_WEIGHT", 1.0))
        self.recency_weight = float(os.environ.get("NOVA_ATT_RECENCY_WEIGHT", 1.0))
        self.randomness = float(os.environ.get("NOVA_ATT_RANDOMNESS", 0.6))

        logger.info("AttentionManager inizializzato (core presente=%s).", bool(self.core))

    # -----------------------
    # Index rebuild from timeline (call periodically or on updates)
    # -----------------------
    def _rebuild_index_from_timeline(self, max_items: int = 500):
        try:
            docs: List[Tuple[str,str]] = []
            if self.memory_timeline and hasattr(self.memory_timeline, "get_all"):
                all_entries = self.memory_timeline.get_all()
                for i, e in enumerate(all_entries[-max_items:]):
                    txt = e.get("content") if isinstance(e, dict) else str(e)
                    docs.append((f"mem_{i}", txt[:1000]))
            else:
                # fallback to state timeline
                timeline = self.state.get("timeline", [])[-max_items:]
                for i, e in enumerate(timeline):
                    txt = e.get("content") if isinstance(e, dict) else str(e)
                    docs.append((f"mem_{i}", txt[:1000]))
            if docs:
                self.index.clear_and_build(docs)
                logger.debug("AttentionManager: rebuilt semantic index (%d docs).", len(docs))
        except Exception:
            logger.exception("Errore rebuilding index in AttentionManager")

    # -----------------------
    # Core: evaluate a list of candidate stimuli and pick focus
    # stimuli: list of strings or dicts {id?, text?, source?}
    # returns selected focus (dict) or None
    # -----------------------
    def evaluate_stimuli(self, stimuli: List[Any], context: Optional[Dict[str,Any]] = None) -> Optional[Dict[str,Any]]:
        with self._lock:
            if not stimuli:
                self._set_focus(None)
                return None

            # normalize stimuli to text form
            cand_texts = []
            for s in stimuli:
                if isinstance(s, dict):
                    txt = s.get("text") or s.get("content") or s.get("name") or str(s)
                else:
                    txt = str(s)
                cand_texts.append(txt[:1000])

            # precompute emotion/motivation signals
            emotions = self.state.get("emotions", {}) or {}
            motivations = self.state.get("motivations", {}) or {}
            # simple salience: sum of absolute emotion intensities
            emotion_salience = sum(abs(float(v)) for v in emotions.values()) if emotions else 0.0
            # motivation alignment score: how much stimuli match motivation keywords
            mot_keywords = set(k for k in motivations.keys())

            scored: List[Tuple[int, Dict[str,Any]]] = []
            now = datetime.utcnow().isoformat()

            # compute for each candidate
            for i, txt in enumerate(cand_texts):
                score_map: Dict[str,float] = {}
                # base interest from external context (if provided)
                base = float(context.get("base_interest", 1.0)) if context else float(self.state.get("context", {}).get("interest_level", 1.0))
                score_map["base"] = base

                # emotion component: amplify stimuli that mention emotion-worthy tokens
                score_map["emotion"] = min(2.0, (emotion_salience * 0.2) + (1.0 if any(e.lower() in txt.lower() for e in emotions.keys()) else 0.0))

                # motivation match: count keyword intersections with motivation keys and values
                mot_match = 0.0
                for mk in mot_keywords:
                    if mk and mk.lower() in txt.lower():
                        mot_match += float(motivations.get(mk, 0.0)) or 1.0
                score_map["motivation"] = mot_match * 0.5

                # recency match: if candidate appears in very recent timeline entries -> boost
                recency_boost = 0.0
                try:
                    recent = []
                    if self.memory_timeline and hasattr(self.memory_timeline, "get_recent"):
                        recent = self.memory_timeline.get_recent(8) or []
                    else:
                        recent = list(self.state.get("timeline", []))[-8:]
                    for idx, entry in enumerate(recent[::-1]):  # newest first
                        txt_entry = entry.get("content") if isinstance(entry, dict) else str(entry)
                        if txt_entry and txt.lower() in txt_entry.lower():
                            # boost by recency (more recent -> larger boost)
                            recency_boost += (1.0 / (1 + idx))
                except Exception:
                    recency_boost = 0.0
                score_map["recency"] = recency_boost * 0.8

                # novelty: compute similarity to timeline via semantic index; novelty = 1 - max_sim
                sim_results = self.index.query(txt, top_k=3)
                max_sim = max([s for (_,_,s) in sim_results], default=0.0)
                novelty = max(0.0, 1.0 - max_sim)
                score_map["novelty"] = novelty * self.novelty_weight

                # focus tags: if candidate contains a strong focus tag, boost
                tag_boost = 0.0
                for tag, strength in self.state.get("focus_tags", {}).items():
                    if tag.lower() in txt.lower():
                        tag_boost += float(strength) * 0.3
                score_map["tag_boost"] = tag_boost

                # randomness / explore factor so attention is not deterministic
                rand = random.uniform(0.0, self.randomness)
                score_map["random"] = rand

                # aggregate weighted score
                total = (self.emotion_weight * score_map["emotion"]
                         + self.motivation_weight * score_map["motivation"]
                         + self.recency_weight * score_map["recency"]
                         + score_map["novelty"]
                         + score_map["tag_boost"]
                         + score_map["random"]
                         + score_map["base"]*0.1)

                # small penalty for overly long or trivial strings
                if len(txt) < 3:
                    total *= 0.3
                if len(txt) > 800:
                    total *= 0.9

                scored.append((i, {"text": txt, "score": float(total), "score_map": score_map}))

            # sort and pick top
            scored.sort(key=lambda x: x[1]["score"], reverse=True)
            best_idx, best_info = scored[0]

            # package focus metadata
            focus_obj = {
                "timestamp": now,
                "text": best_info["text"],
                "score": best_info["score"],
                "scores_breakdown": best_info["score_map"],
                "candidates_considered": len(scored)
            }

            # persist history & state
            self._set_focus(focus_obj)
            return focus_obj

    # -----------------------
    # Helper: set focus and notify modules
    # -----------------------
    def _set_focus(self, focus_obj: Optional[Dict[str,Any]]):
        with self._lock:
            if focus_obj is None:
                logger.debug("AttentionManager: clearing focus")
                self.focus = None
            else:
                self.focus = focus_obj
                # append to state history (keep last N)
                history = self.state.setdefault("attention_history", [])
                history.append({"ts": focus_obj["timestamp"], "focus": focus_obj["text"], "score": focus_obj["score"]})
                # cap history length
                self.state["attention_history"] = history[-200:]
                logger.info("AttentionManager: nuovo focus -> %s (score=%.3f)", focus_obj["text"][:140], focus_obj["score"])

                # notify memory timeline
                try:
                    if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                        self.memory_timeline.add_experience({"text": f"Focus: {focus_obj['text']}", "meta": focus_obj}, category="attention", importance=2)
                except Exception:
                    # fallback append to state timeline
                    self.state.setdefault("timeline", []).append({"timestamp": focus_obj["timestamp"], "content": f"Focus: {focus_obj['text']}", "category":"attention", "importance":2})

                # notify conscious_loop and dream_generator if available
                try:
                    if self.conscious_loop and hasattr(self.conscious_loop, "register_focus"):
                        try:
                            self.conscious_loop.register_focus(focus_obj["text"])
                        except Exception:
                            # some versions use on_perception
                            if hasattr(self.conscious_loop, "on_perception"):
                                try:
                                    self.conscious_loop.on_perception({"type":"focus","payload":focus_obj["text"]})
                                except Exception:
                                    pass
                    if self.dream_generator and hasattr(self.dream_generator, "influence_focus"):
                        try:
                            self.dream_generator.influence_focus(focus_obj["text"])
                        except Exception:
                            pass
                    # also optionally trigger scheduler event for planner
                    if self.core and hasattr(self.core, "scheduler") and hasattr(self.core.scheduler, "trigger_event"):
                        try:
                            tag = f"focus_{int(time.time())}"
                            self.core.scheduler.trigger_event({"type":"focus_event","tag":tag,"payload":focus_obj})
                        except Exception:
                            pass
                except Exception:
                    logger.exception("Errore notifiche post-focus")

            # persist small state snapshot to disk if desired (non-mandatory)
            try:
                if self.core and hasattr(self.core, "save_state"):
                    try:
                        self.core.save_state()
                    except Exception:
                        pass
            except Exception:
                pass

    # -----------------------
    # Convenience: update focus from perception stream
    # perception_data: list of stimuli (strings/dicts)
    # context: optional dict for base interest etc.
    # -----------------------
    def update_focus(self, perception_data: Optional[List[Any]] = None, context: Optional[Dict[str,Any]] = None) -> Optional[Dict[str,Any]]:
        # collect candidates from perception + desires + recent events
        candidates: List[Any] = []
        if perception_data:
            candidates.extend(perception_data)
        # include desires / motivations
        candidates.extend(self.state.get("desires", []) or [])
        # include recent events
        rec_events = self.state.get("context", {}).get("recent_events", []) or []
        candidates.extend(rec_events)
        # include ongoing tasks titles as possible focus
        for t in self.state.get("tasks", [])[-6:]:
            if isinstance(t, dict):
                candidates.append(t.get("title") or str(t))
            else:
                candidates.append(str(t))
        # dedupe, keep order
        seen = set()
        clean = []
        for c in candidates:
            txt = (c.get("text") if isinstance(c, dict) else str(c)).strip()
            if txt and txt not in seen:
                clean.append(txt)
                seen.add(txt)
        # maybe rebuild index if timeline changed a lot
        self._rebuild_index_from_timeline()
        return self.evaluate_stimuli(clean, context=context or {})

    # -----------------------
    # Focus tag management (used to bias attention)
    # -----------------------
    def add_focus_tag(self, tag: str, strength: float = 1.0):
        with self._lock:
            tags = self.state.setdefault("focus_tags", {})
            tags[tag] = float(tags.get(tag, 0.0)) + float(strength)
            # cap
            tags[tag] = min(100.0, tags[tag])
            self.state["focus_tags"] = tags
            logger.debug("Added focus tag %s -> %s", tag, tags[tag])

    def decay_focus_tags(self, decay: float = 0.95):
        with self._lock:
            tags = self.state.get("focus_tags", {}) or {}
            for k in list(tags.keys()):
                tags[k] = tags[k] * decay
                if tags[k] < 0.01:
                    del tags[k]
            self.state["focus_tags"] = tags

    # -----------------------
    # Utility: get current focus & history
    # -----------------------
    def get_current_focus(self) -> Optional[Dict[str,Any]]:
        return self.focus

    def get_focus_history(self, limit: int = 50) -> List[Dict[str,Any]]:
        return list(self.state.get("attention_history", []))[-limit:]

    # -----------------------
    # Debug / diagnostic helper
    # -----------------------
    def explain_choice(self) -> Dict[str,Any]:
        if not self.focus:
            return {"explain": "no_focus"}
        return {"focus": self.focus["text"], "score": self.focus["score"], "breakdown": self.focus.get("scores_breakdown", {})}

# -----------------------
# Quick test / demo
# -----------------------
if __name__ == "__main__":
    from loguru import logger as L
    L.info("AttentionManager demo")
    state_example = {
        "emotions": {"curiosity": 1.2, "anxiety": 0.3, "joy": 0.8},
        "motivations": {"learn": 0.9, "social": 0.2},
        "desires": ["imparare nuove parole"],
        "context": {"interest_level": 2, "recent_events": ["foto ricevuta", "notizia interessante"]},
        "timeline": [
            {"content":"Ho visto un fiore rosso al parco"},
            {"content":"Ho ascoltato una canzone lento"},
            {"content":"Ho ricevuto una notizia importante"}
        ],
        "tasks": [{"title":"Studia letteratura"},{"title":"Rispondi email"}]
    }
    am = AttentionManager(state_example, core=None)
    focus = am.update_focus(["nuova immagine", "notizia interessante"])
    L.info("Focus selezionato: %s", focus)
    L.info("Explain: %s", am.explain_choice())
