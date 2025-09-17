# desire_engine.py
"""
DesireEngine avanzato per Nova

Scopo:
- Generare e gestire desideri a partire da esperienze (timeline), emozioni e contesto.
- Prioritizzare, deduplicare, suggerire azioni esplorative e integrare con motivational_engine/planner.
- Persistenza atomica dello stato condiviso in internal_state.yaml.
- Uso opzionale di modello locale Gemma (llama_cpp) per formulare desideri/idee testuali più naturali.
"""

import os
import threading
import random
import time
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

# -----------------------
# Helper atomic write
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# Optional local LLM helper (Gemma via llama_cpp) - non obbligatorio
# -----------------------
def _load_llm(model_path: str):
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)
        logger.info("DesireEngine: LLM locale caricato (%s)", model_path)
        def _call(prompt: str, max_tokens: int = 120, temperature: float = 0.8) -> str:
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
                logger.exception("DesireEngine: errore chiamata LLM")
                return ""
        return _call
    except Exception as e:
        logger.debug("DesireEngine: llama_cpp non disponibile (%s). Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 60, temperature: float = 0.5) -> str:
            # fallback: formula un desiderio sintetico basato sul prompt
            seed = (prompt or "")[:120]
            choices = [
                f"Vorrei esplorare: {seed}",
                f"Mi interessa capire meglio: {seed[:80]}",
                f"Proporrei di testare: {seed[:80]}"
            ]
            return random.choice(choices)
        return _fallback

# -----------------------
# DesireEngine
# -----------------------
class DesireEngine:
    def __init__(self,
                 state: Optional[Dict[str,Any]] = None,
                 memory_timeline: Optional[Any] = None,
                 motivational_engine: Optional[Any] = None,
                 emotion_engine: Optional[Any] = None,
                 planner: Optional[Any] = None,
                 identity_manager: Optional[Any] = None,
                 model_path: str = DEFAULT_MODEL_PATH):
        """
        state: se fornito usa il dict condiviso (core.state). Altrimenti carica da STATE_FILE.
        Altri moduli sono opzionali ma consigliati per integrazione completa.
        """
        self._lock = threading.RLock()

        # carica stato condiviso o file
        if state is None:
            self.state = self._load_state_from_file()
        else:
            self.state = state

        # assicura struttura desideri nello stato
        self.state.setdefault("desires", [])  # lista di desideri (dict)
        self.state.setdefault("_desire_history", [])  # storico delle generazioni

        # riferimenti ai moduli
        self.memory_timeline = memory_timeline
        self.motivational_engine = motivational_engine
        self.emotion_engine = emotion_engine
        self.planner = planner
        self.identity_manager = identity_manager

        # llm optional per fraseggio dei desideri
        self.llm = _load_llm(model_path)

        logger.info("DesireEngine inizializzato. Desires=%d", len(self.state.get("desires", [])))

    # -----------------------
    # File load/save
    # -----------------------
    def _load_state_from_file(self) -> Dict[str,Any]:
        try:
            with open(STATE_FILE, "r", encoding="utf8") as f:
                data = yaml.safe_load(f) or {}
                return data
        except FileNotFoundError:
            logger.warning("DesireEngine: state file non trovato (%s). Creo nuovo stato.", STATE_FILE)
            return {}
        except Exception:
            logger.exception("DesireEngine: errore caricamento stato")
            return {}

    def save_state(self) -> None:
        with self._lock:
            try:
                _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
                logger.debug("DesireEngine: stato salvato su %s", STATE_FILE)
            except Exception:
                logger.exception("DesireEngine: errore salvataggio stato")

    # -----------------------
    # Utilities per desideri
    # -----------------------
    def _make_desire(self, content: str, origin: str="internal", priority: float=0.5, metadata: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
        return {
            "id": f"d_{int(time.time()*1000)}_{random.randint(0,9999)}",
            "ts": datetime.utcnow().isoformat(),
            "content": content,
            "origin": origin,
            "priority": float(max(0.0, min(1.0, priority))),
            "metadata": metadata or {},
            "fulfilled": False
        }

    def _normalize_text(self, t: str) -> str:
        return " ".join(t.strip().lower().split())

    def _is_duplicate(self, new_content: str) -> bool:
        n = self._normalize_text(new_content)
        for d in self.state.get("desires", []):
            if self._normalize_text(d.get("content","")) == n:
                return True
        return False

    # -----------------------
    # Public API: gestione desideri
    # -----------------------
    def add_desire(self, content: str, origin: str="manual", priority: float=0.5, metadata: Optional[Dict[str,Any]]=None) -> Optional[Dict[str,Any]]:
        """
        Aggiunge manualmente un desiderio (es. insegnamento umano).
        Ritorna il desiderio creato o None se duplicato.
        """
        with self._lock:
            try:
                if self._is_duplicate(content):
                    logger.debug("DesireEngine: desire duplicato ignorato: %s", content[:80])
                    return None
                d = self._make_desire(content, origin=origin, priority=priority, metadata=metadata)
                self.state.setdefault("desires", []).append(d)
                self.state.setdefault("_desire_history", []).append({"action":"add","desire":d,"ts":d["ts"]})
                logger.info("DesireEngine: desire aggiunto: %s", d["content"][:120])
                # notify collaborators
                self._notify_new_desire(d)
                self.save_state()
                return d
            except Exception:
                logger.exception("DesireEngine: exception in add_desire")
                return None

    def remove_desire(self, desire_id: str) -> bool:
        with self._lock:
            try:
                before = len(self.state.get("desires", []))
                self.state["desires"] = [d for d in self.state.get("desires", []) if d.get("id") != desire_id]
                after = len(self.state.get("desires", []))
                removed = before - after
                if removed:
                    logger.info("DesireEngine: desire rimosso id=%s", desire_id)
                    self.save_state()
                    return True
                return False
            except Exception:
                logger.exception("DesireEngine: exception in remove_desire")
                return False

    def clear_desires(self) -> None:
        with self._lock:
            self.state["desires"] = []
            self.state.setdefault("_desire_history", []).append({"action":"clear_all","ts":datetime.utcnow().isoformat()})
            self.save_state()
            logger.info("DesireEngine: tutti i desideri azzerati.")

    def get_top_desires(self, n: int = 5) -> List[Dict[str,Any]]:
        with self._lock:
            return sorted(self.state.get("desires", []), key=lambda d: d.get("priority",0.0), reverse=True)[:n]

    # -----------------------
    # Generazione desideri da timeline / esperienze
    # -----------------------
    def generate_desires(self, recent_n: int = 30, min_importance: int = 2) -> List[Dict[str,Any]]:
        """
        Scans memory_timeline per trovare esperienze rilevanti e genera desideri.
        La logica usa importanza, emozione dominante e occasionalmente LLM per formulazione.
        """
        with self._lock:
            try:
                # recupera esperienze recenti
                recent = []
                if self.memory_timeline:
                    try:
                        recent = self.memory_timeline.get_recent(recent_n)
                    except Exception:
                        # fallback: try alternative method names
                        try:
                            recent = self.memory_timeline.recent(recent_n)
                        except Exception:
                            recent = []
                else:
                    recent = self.state.get("timeline", [])[-recent_n:]

                new_desires = []
                for exp in recent:
                    # normalizza shape: try to extract content/importance/category
                    content = exp.get("content") if isinstance(exp, dict) else str(exp)
                    importance = exp.get("importance", 1) if isinstance(exp, dict) else 1
                    category = exp.get("category", "misc") if isinstance(exp, dict) else "misc"

                    # skip trivial events
                    if importance < min_importance:
                        continue
                    if category == "task":
                        continue

                    # compute priority base from importance + recency + emotion alignment
                    recency_score = 1.0
                    try:
                        # if exp has timestamp try to compute recency weight
                        ts = exp.get("timestamp") or exp.get("ts") or exp.get("date")
                        if ts:
                            # naive recency (newer => higher)
                            recency_score = max(0.1, 1.0 - ( (time.time() - float(time.mktime(datetime.fromisoformat(ts).timetuple()))) / (60*60*24*30) ))
                    except Exception:
                        recency_score = 1.0

                    emo_boost = 0.0
                    try:
                        if self.emotion_engine and hasattr(self.emotion_engine, "snapshot"):
                            emo = self.emotion_engine.snapshot()
                            # boost if curiosity high
                            if isinstance(emo, dict):
                                curiosity = emo.get("mood", {}).get("valence") or self.state.get("emotions", {}).get("curiosity", 0)
                                emo_boost = float(curiosity) * 0.2
                    except Exception:
                        emo_boost = 0.0

                    base_priority = float(min(1.0, 0.15 * importance + 0.35 * recency_score + emo_boost + random.uniform(0,0.15)))

                    # craft desire text (try LLM)
                    desire_text = self._craft_desire_text(content, category=category)

                    # deduplicate
                    if self._is_duplicate(desire_text):
                        continue

                    d = self._make_desire(desire_text, origin=f"timeline:{category}", priority=base_priority, metadata={"source_exp": exp})
                    self.state.setdefault("desires", []).append(d)
                    self.state.setdefault("_desire_history", []).append({"action":"gen_from_exp","desire":d,"exp":exp,"ts":d["ts"]})
                    new_desires.append(d)
                    logger.info("DesireEngine: generato desire da esperienza: %s (prio=%.2f)", desire_text[:120], base_priority)

                    # notify collaborators
                    self._notify_new_desire(d)

                if new_desires:
                    self.save_state()
                    # notify motivational engine if present
                    try:
                        if self.motivational_engine and hasattr(self.motivational_engine, "add_desire"):
                            for d in new_desires:
                                try:
                                    self.motivational_engine.add_desire(d.get("content"), source="desire_engine", strength=d.get("priority",0.5))
                                except Exception:
                                    logger.debug("MotivationalEngine add_desire failed for %s", d.get("content")[:80])
                    except Exception:
                        logger.exception("DesireEngine: errore notify motivational_engine")

                return new_desires
            except Exception:
                logger.exception("DesireEngine: exception in generate_desires")
                return []

    def _craft_desire_text(self, content: str, category: str = "misc") -> str:
        """
        Prova a creare una breve formulazione naturale del desiderio.
        Usa LLM se disponibile, fallback euristico altrimenti.
        """
        try:
            prompt = f"Da questa esperienza: \"{str(content)[:200]}\" genera in italiano una breve frase che esprima un desiderio di esplorazione o apprendimento correlato."
            text = ""
            if callable(self.llm):
                text = self.llm(prompt, max_tokens=80, temperature=0.8) or ""
            if not text:
                # fallback: heuristic
                short = str(content).strip()
                if len(short) > 140:
                    short = short[:140] + "..."
                text = f"Vorrei esplorare: {short}"
            return text.strip()
        except Exception:
            logger.exception("DesireEngine: exception in _craft_desire_text")
            return f"Vorrei esplorare: {str(content)[:120]}"

    # -----------------------
    # Suggest actions / small experiments for a desire
    # -----------------------
    def suggest_actions_for_desire(self, desire_id: str, max_steps: int = 4) -> List[str]:
        with self._lock:
            try:
                desire = next((d for d in self.state.get("desires", []) if d.get("id")==desire_id), None)
                if not desire:
                    return []
                title = desire.get("content","")
                # try planner or LLM for decomposition
                steps = []
                if self.planner and hasattr(self.planner, "decompose_goal"):
                    try:
                        steps = self.planner.decompose_goal(title, max_steps=max_steps)
                    except Exception:
                        steps = []
                if not steps:
                    # try LLM
                    if callable(self.llm):
                        prompt = f"Obiettivo esplorativo: {title}\nDividi in {max_steps} passi pratici, semplici e non dannosi. Rispondi in italiano, uno step per linea."
                        out = self.llm(prompt, max_tokens=220, temperature=0.7) or ""
                        lines = [l.strip("-• \t") for l in out.splitlines() if l.strip()]
                        steps = lines[:max_steps] if lines else []
                if not steps:
                    # fallback heuristic
                    steps = [
                        f"Raccogli informazioni su: {title}",
                        f"Prova un piccolo esperimento relativo a: {title}",
                        f"Osserva i risultati e annota ciò che impari",
                    ][:max_steps]
                logger.debug("DesireEngine: suggeriti %d passi per desire %s", len(steps), desire_id)
                return steps
            except Exception:
                logger.exception("DesireEngine: exception in suggest_actions_for_desire")
                return []

    # -----------------------
    # Promuove un desire a goal tramite motivational_engine / planner
    # -----------------------
    def promote_desire_to_goal(self, desire_id: str, force: bool = False) -> Optional[Dict[str,Any]]:
        with self._lock:
            try:
                desire = next((d for d in self.state.get("desires", []) if d.get("id")==desire_id), None)
                if not desire:
                    return None
                # try to use motivational_engine
                if self.motivational_engine and hasattr(self.motivational_engine, "promote_desire_to_goal"):
                    try:
                        goal = self.motivational_engine.promote_desire_to_goal(desire, force=force)
                        if goal:
                            logger.info("DesireEngine: desire promosso a goal via MotivationalEngine: %s", desire_id)
                            # mark fulfilled/proposed
                            desire["fulfilled"] = True
                            self.save_state()
                            return goal
                    except Exception:
                        logger.exception("MotivationalEngine promote failed")
                # fallback: create simple goal object and append to state['goals']
                goal = {
                    "id": f"g_{int(time.time()*1000)}",
                    "title": desire.get("content"),
                    "steps": self.suggest_actions_for_desire(desire_id, max_steps=4),
                    "priority": desire.get("priority", 0.5),
                    "origin": "desire_engine"
                }
                self.state.setdefault("goals", []).append(goal)
                desire["fulfilled"] = True
                self.save_state()
                logger.info("DesireEngine: desire promosso a goal (fallback): %s", desire_id)
                return goal
            except Exception:
                logger.exception("DesireEngine: exception in promote_desire_to_goal")
                return None

    # -----------------------
    # Notification hooks: notifica altri moduli della nascita di un desire
    # -----------------------
    def _notify_new_desire(self, desire: Dict[str,Any]) -> None:
        try:
            # memory timeline already called externally, but ensure entry
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": f"Desire generated: {desire.get('content')}", "meta":{"desire_id":desire.get("id")}}, category="desire", importance=2)
            except Exception:
                pass
            # dream generator influence
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    # small extra: tag to dreams
                    pass
            except Exception:
                pass
            # motivational engine: add desire
            try:
                if self.motivational_engine and hasattr(self.motivational_engine, "add_desire"):
                    self.motivational_engine.add_desire(desire.get("content"), source="desire_engine", strength=desire.get("priority",0.5))
            except Exception:
                logger.debug("DesireEngine: impossibile notificare motivational_engine")
        except Exception:
            logger.exception("DesireEngine: exception in _notify_new_desire")

    # -----------------------
    # Debug / snapshots
    # -----------------------
    def snapshot(self) -> Dict[str,Any]:
        with self._lock:
            return {
                "num_desires": len(self.state.get("desires", [])),
                "top_desires": [d.get("content") for d in self.get_top_desires(5)]
            }

# -----------------------
# Demo / test
# -----------------------
if __name__ == "__main__":
    # demo: crea DesireEngine con fallback minimal
    de = DesireEngine(state=None, memory_timeline=None, motivational_engine=None, emotion_engine=None, planner=None)
    # add manual desire
    d = de.add_desire("Voglio capire cosa significa essere gentile con gli altri", origin="manual", priority=0.7)
    print("Added:", d)
    # simulate generate from fake timeline entries
    # if no memory_timeline provided, nothing will be generated; create fake timeline in state:
    de.state.setdefault("timeline", []).append({"content":"Ho visto una persona triste in strada","importance":3,"category":"interaction","timestamp":datetime.utcnow().isoformat()})
    new = de.generate_desires()
    print("Generated:", new)
    print("Snapshot:", de.snapshot())
