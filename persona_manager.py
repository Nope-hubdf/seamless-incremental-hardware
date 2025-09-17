# persona_manager.py
"""
PersonaManager
--------------
Gestisce la personalità di Nova: tratti, coerenza, evoluzione graduale basata su
interazioni e riflessioni. Si sincronizza con internal_state.yaml e fornisce
callback per notificare emotion_engine, motivational_engine, identity_manager, ecc.

API principali:
- get_trait(trait_name, default=None)
- set_trait(trait_name, value, persist=True)
- observe_interaction(interaction: dict)   # aggiorna la personalità in base a un evento
- register_callback(kind: str, fn: callable)
- stabilize_personality()                 # applica piccoli aggiustamenti regolari
"""

from __future__ import annotations
import yaml
import os
import threading
from typing import Any, Callable, Dict, Optional
from copy import deepcopy
from loguru import logger
from datetime import datetime, timedelta

STATE_FILE = "internal_state.yaml"

# Default personality template (esempio)
DEFAULT_PERSONA = {
    "traits": {
        # scala -1.0 .. 1.0 (negativo -> evita, positivo -> predilige)
        "curiosity": 0.2,
        "empathy": 0.0,
        "boldness": 0.0,
        "patience": 0.3,
        "playfulness": 0.1,
        "conscientiousness": 0.2,
        "openness": 0.2
    },
    "history": [],  # lista di eventi che hanno influenzato la personalità
    "last_stabilize": None
}

class PersonaManager:
    def __init__(self, state: Optional[Dict[str, Any]] = None):
        """
        Se viene passato uno state (dallo core), PersonaManager lo userà; altrimenti
        caricherà/creerà internal_state.yaml autonomamente.
        """
        logger.info("Inizializzazione PersonaManager...")
        self._lock = threading.RLock()
        self._callbacks: Dict[str, list[Callable]] = {}  # kind -> [fn]
        self.state = state if state is not None else self._load_state_file()
        # assicurati che ci sia la sezione persona
        if "persona" not in self.state or not isinstance(self.state["persona"], dict):
            self.state["persona"] = deepcopy(DEFAULT_PERSONA)
        # normalize traits keys
        self._ensure_traits_integrity()
        # rapido bilanciamento iniziale
        self.stabilize_personality(minor=True)

    # -----------------------
    #                                         State I/O
    # -----------------------
    def _load_state_file(self) -> dict:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    st = yaml.safe_load(f) or {}
                    logger.info("PersonaManager: stato interno caricato da file.")
                    return st
        except Exception as e:
            logger.exception(f"Errore caricamento stato: {e}")
        logger.info("PersonaManager: creazione nuovo stato di base.")
        return {"persona": deepcopy(DEFAULT_PERSONA)}

    def save_state(self):
        """Salva lo stato su disk (thread-safe)."""
        with self._lock:
            try:
                # aggiorna last_stabilize se assente
                if self.state["persona"].get("last_stabilize") is None:
                    self.state["persona"]["last_stabilize"] = datetime.utcnow().isoformat()
                with open(STATE_FILE, "w") as f:
                    yaml.safe_dump(self.state, f)
                logger.debug("PersonaManager: stato persona salvato su disk.")
            except Exception as e:
                logger.exception(f"PersonaManager: errore salvataggio stato: {e}")

    # -----------------------
    #                                         Utils per tratti
    # -----------------------
    def _ensure_traits_integrity(self):
        with self._lock:
            traits = self.state["persona"].setdefault("traits", {})
            for k, v in DEFAULT_PERSONA["traits"].items():
                if k not in traits:
                    traits[k] = float(v)

    def get_trait(self, trait_name: str, default: Optional[float] = None) -> Optional[float]:
        with self._lock:
            return float(self.state["persona"]["traits"].get(trait_name, default))

    def set_trait(self, trait_name: str, value: float, persist: bool = True):
        """Imposta/aggiusta un tratto (valore coerente tra -1.0 e 1.0)."""
        with self._lock:
            clipped = max(-1.0, min(1.0, float(value)))
            self.state["persona"]["traits"][trait_name] = clipped
            self._append_history({
                "type": "trait_set",
                "trait": trait_name,
                "value": clipped,
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.info(f"PersonaManager: tratto '{trait_name}' impostato a {clipped:.3f}")
            if persist:
                self.save_state()
            # notifica i callbacks sul cambiamento di tratto
            self._notify("trait_change", {"trait": trait_name, "value": clipped})

    # -----------------------
    #                                         Interazioni e apprendimento
    # -----------------------
    def _append_history(self, event: dict):
        hist = self.state["persona"].setdefault("history", [])
        hist.append(event)
        # mantieni la storia limitata a N eventi per dimensione (es. 2000)
        if len(hist) > 2000:
            del hist[0: len(hist) - 2000]

    def observe_interaction(self, interaction: Dict[str, Any]):
        """
        Chiamare questa funzione dopo ogni interazione significativa.
        interaction: {
            "type": "conversation" | "image" | "task" | "correction" | ...,
            "content": "...",
            "affect": {"valence": -1..1, "arousal": 0..1}  # se disponibile
            "source": "telegram|camera|user" (opzionale)
        }
        """
        with self._lock:
            timestamp = datetime.utcnow().isoformat()
            event = {"timestamp": timestamp, **interaction}
            self._append_history(event)
            logger.debug(f"PersonaManager: osservata interazione tipo={interaction.get('type')}")

            # semplice regole di aggiornamento (puoi estenderle)
            affect = interaction.get("affect") or {}
            valence = float(affect.get("valence", 0.0))
            # Se l'interazione è positiva -> aumenta curiosity/openness/playfulness
            if valence > 0.2:
                self._increment_trait("curiosity", 0.01 * valence)
                self._increment_trait("openness", 0.008 * valence)
                self._increment_trait("playfulness", 0.006 * valence)
            elif valence < -0.2:
                # negatività -> aumenta cautela, pazienza, riduce boldness
                self._increment_trait("patience", 0.004 * abs(valence))
                self._increment_trait("boldness", -0.006 * abs(valence))
                self._increment_trait("conscientiousness", 0.005 * abs(valence))

            # se è una "correction" esplicita dall'utente, rafforza empatia e coscienza
            if interaction.get("type") == "correction":
                self._increment_trait("empathy", 0.02)
                self._increment_trait("conscientiousness", 0.02)

            # notifica gli altri moduli di una nuova interazione
            self._notify("interaction", event)
            # persisti lo stato
            self.save_state()

    def _increment_trait(self, trait: str, delta: float):
        current = self.get_trait(trait, 0.0)
        self.set_trait(trait, current + delta, persist=False)

    # -----------------------
    #                                         Stabilizzazione e evoluzione
    # -----------------------
    def stabilize_personality(self, minor: bool = False):
        """
        Piccoli aggiustamenti regolari per evitare oscillazioni e mantenere coerenza.
        - minor=True: aggiustamenti molto leggeri (es. all'avvio)
        - otherwise: esegue regole di smoothing sui tratti e aggiorna last_stabilize
        """
        with self._lock:
            traits = self.state["persona"]["traits"]
            smoothing = 0.01 if minor else 0.03
            for k, v in traits.items():
                # tira i valori lievemente verso il baseline (0.0) per stabilità
                traits[k] = float(v) * (1 - smoothing)
            self.state["persona"]["last_stabilize"] = datetime.utcnow().isoformat()
            self.save_state()
            logger.info("PersonaManager: personalità stabilizzata.")

    # -----------------------
    #                                         Callbacks & integrazione
    # -----------------------
    def register_callback(self, kind: str, fn: Callable[[dict], None]):
        """Registro callback: kind può essere 'trait_change','interaction','request_snapshot' ecc."""
        with self._lock:
            if kind not in self._callbacks:
                self._callbacks[kind] = []
            self._callbacks[kind].append(fn)
            logger.debug(f"PersonaManager: callback registrata per kind='{kind}' -> {fn}")

    def _notify(self, kind: str, payload: dict):
        """Notifica i callback registrati (sincrono)."""
        with self._lock:
            for fn in self._callbacks.get(kind, []):
                try:
                    fn(payload)
                except Exception:
                    logger.exception(f"PersonaManager: errore eseguendo callback {fn} per kind={kind}")

    def snapshot(self) -> Dict[str, Any]:
        """Ritorna una copia dello stato persona (per decision making o salvataggio esterno)."""
        with self._lock:
            return deepcopy(self.state["persona"])

    # -----------------------
    #                                         Utility per integrazione
    # -----------------------
    def apply_user_teaching(self, lesson: Dict[str, Any]):
        """
        Metodo che il 'padre' o l'utente può usare per insegnare esplicitamente un valore.
        lesson: {"trait": "empathy", "value": 0.6, "reason": "..."}.
        """
        trait = lesson.get("trait")
        val = lesson.get("value")
        if trait is None or val is None:
            logger.warning("PersonaManager.apply_user_teaching: lesson invalida.")
            return
        self.set_trait(trait, float(val))
        self._append_history({"type": "teaching", "lesson": lesson, "timestamp": datetime.utcnow().isoformat()})
        self.save_state()
        logger.info(f"PersonaManager: lezione applicata su {trait} -> {val}")

    # -----------------------
    #                                         Main di test
    # -----------------------
if __name__ == "__main__":
    # semplice test locale
    pm = PersonaManager()
    print("Snapshot iniziale:", pm.snapshot())
    pm.observe_interaction({"type": "conversation", "content": "L'utente è gentile", "affect": {"valence": 0.6}})
    pm.observe_interaction({"type": "correction", "content": "No, prova così", "affect": {"valence": -0.3}})
    print("Snapshot dopo interazioni:", pm.snapshot())
