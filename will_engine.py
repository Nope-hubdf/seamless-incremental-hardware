# will_engine.py
"""
Will Engine per Nova
Gestisce la volontà, i goal, la selezione degli obiettivi e la capacità decisionale autonoma.
Progettato per integrarsi con:
 - internal_state.yaml
 - desire_engine (opzionale)
 - emotion_engine (opzionale)
 - planner (opzionale, con metodo plan(goal) o fai_task(task))
 - long_term_memory_manager (opzionale)
 - loguru per logging
"""

import os
import yaml
import uuid
from datetime import datetime
from loguru import logger

STATE_FILE = "internal_state.yaml"

# Import opzionali (se non presenti il modulo rimane funzionante)
try:
    from desire_engine import DesireEngine
except Exception:
    DesireEngine = None

try:
    from emotion_engine import EmotionEngine
except Exception:
    EmotionEngine = None

try:
    from planner import Planner
except Exception:
    Planner = None

try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None


class WillEngine:
    def __init__(self, state=None):
        logger.info("Inizializzazione WillEngine...")
        self.state = state if state is not None else self._load_state()
        # Assicura struttura di stato per will
        if "will" not in self.state:
            self.state["will"] = {
                "goals": [],              # lista goal
                "current_goal_id": None,  # id del goal in corso
                "action_history": [],     # esecuzioni
                "will_power": 1.0         # livello di energia motivazionale [0..10]
            }

        # Integrazioni opzionali
        self.desire_engine = DesireEngine(self.state) if DesireEngine else None
        self.emotion_engine = EmotionEngine(self.state) if EmotionEngine else None
        self.planner = Planner(self.state) if Planner else None
        self.memory = LongTermMemoryManager(self.state) if LongTermMemoryManager else None

    # -------------------------
    # Utility stato persistente
    # -------------------------
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                s = yaml.safe_load(f) or {}
                logger.info("WillEngine: stato caricato da file.")
                return s
        logger.info("WillEngine: nessuno stato trovato, inizializzo nuovo stato.")
        return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            yaml.safe_dump(self.state, f)
        logger.debug("WillEngine: stato salvato.")

    # -------------------------
    # Goal management
    # -------------------------
    def add_goal(self, description, priority=5, deadline=None, metadata=None):
        """Aggiunge un nuovo goal e ritorna l'id"""
        goal_id = str(uuid.uuid4())
        goal = {
            "id": goal_id,
            "description": description,
            "priority": float(priority),  # 0..10
            "deadline": deadline,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending",
            "progress": 0.0  # 0..1
        }
        self.state["will"]["goals"].append(goal)
        logger.info(f"Nuovo goal aggiunto: {description} (id={goal_id})")
        # registra nella memoria a lungo termine se disponibile
        try:
            if self.memory:
                self.memory.add_experience(f"Goal aggiunto: {description}", category="goal", importance=2)
        except Exception:
            logger.exception("WillEngine: errore registrando goal in memory.")
        self._save_state()
        return goal_id

    def get_goal(self, goal_id):
        for g in self.state["will"]["goals"]:
            if g["id"] == goal_id:
                return g
        return None

    def remove_goal(self, goal_id):
        before = len(self.state["will"]["goals"])
        self.state["will"]["goals"] = [g for g in self.state["will"]["goals"] if g["id"] != goal_id]
        after = len(self.state["will"]["goals"])
        logger.info(f"Goal rimosso: {goal_id} ({before}->{after})")
        self._save_state()

    # -------------------------
    # Valutazione e scelta goal
    # -------------------------
    def _score_goal(self, goal):
        """
        Scoring semplice che combina:
          - priority (user assigned)
          - desire influence (se desire_engine disponibile)
          - emotion influence (se emotion_engine disponibile)
          - proximity to deadline (aumenta score)
          - will_power (modula capacità)
        Ritorna uno score numerico (più alto = più importante)
        """
        score = float(goal.get("priority", 5.0))
        # desideri esterni
        try:
            if self.desire_engine:
                desire_score = self.desire_engine.evaluate_goal_desirability(goal)
                score += desire_score
        except Exception:
            logger.debug("WillEngine: desire_engine non disponibile o errore.")

        # stato emotivo
        try:
            if self.emotion_engine:
                mood_influence = self.emotion_engine.get_mood_influence()
                score *= (1 + mood_influence)  # amplifica o riduce
        except Exception:
            logger.debug("WillEngine: emotion_engine non disponibile o errore.")

        # deadline proximity
        try:
            if goal.get("deadline"):
                # semplice: se deadline vicina aggiungi
                # nota: deadline può essere stringa ISO or None
                from dateutil import parser as dateparser  # optional; may fail -> except
                deadline_dt = dateparser.parse(goal["deadline"])
                now = datetime.utcnow()
                delta = (deadline_dt - now).total_seconds()
                if delta >= 0:
                    # più vicino -> maggiore score
                    proximity_boost = max(0.0, (86400.0 - delta) / 86400.0)  # within a day boosts
                    score += proximity_boost * 2.0
        except Exception:
            # se dateutil non presente o parsing fallisce ignoriamo
            pass

        # will_power modulates final score
        will_power = float(self.state["will"].get("will_power", 1.0))
        score *= (0.5 + min(2.0, will_power))  # range ragionevole
        return score

    def evaluate_goals(self):
        """Valuta e ordina i goal, ritorna lista ordinata per score decrescente"""
        goals = [g for g in self.state["will"]["goals"] if g.get("status") in ("pending", "in_progress")]
        scored = []
        for g in goals:
            try:
                s = self._score_goal(g)
            except Exception as e:
                logger.exception(f"Errore scoring goal {g.get('id')}: {e}")
                s = g.get("priority", 0)
            scored.append((s, g))
        scored.sort(key=lambda x: x[0], reverse=True)
        logger.debug(f"WillEngine: valutati {len(scored)} goal.")
        return [g for s, g in scored]

    def select_current_goal(self):
        """Seleziona il goal con score più alto e lo imposta come current_goal"""
        ordered = self.evaluate_goals()
        if not ordered:
            logger.debug("WillEngine: nessun goal disponibile.")
            self.state["will"]["current_goal_id"] = None
            self._save_state()
            return None
        top = ordered[0]
        if self.state["will"].get("current_goal_id") != top["id"]:
            logger.info(f"WillEngine: nuovo goal selezionato: {top['description']} (id={top['id']})")
            self.state["will"]["current_goal_id"] = top["id"]
            top["status"] = "in_progress"
            self._save_state()
        return top

    # -------------------------
    # Pianificazione e esecuzione
    # -------------------------
    def plan_for_goal(self, goal):
        """Chiede al planner di creare un piano; fall back a un piano semplice se planner non presente"""
        if not goal:
            return []
        try:
            if self.planner and hasattr(self.planner, "plan"):
                plan = self.planner.plan(goal)
                logger.info(f"WillEngine: planner esterno ha restituito un piano per goal {goal['id']}")
                return plan
        except Exception:
            logger.exception("WillEngine: planner presente ma errore durante plan().")

        # Fallback: creare piano semplice basato su micro-task
        logger.debug("WillEngine: generazione piano fallback (micro-tasks).")
        desc = goal.get("description", "Task senza descrizione")
        plan = [
            {"id": f"{goal['id']}_step1", "description": f"Analizza: {desc}", "status": "pending"},
            {"id": f"{goal['id']}_step2", "description": f"Esegui azione base su: {desc}", "status": "pending"},
            {"id": f"{goal['id']}_step3", "description": f"Verifica risultati per: {desc}", "status": "pending"},
        ]
        return plan

    def pursue_current_goal(self):
        """Esegue (o richiede al planner di eseguire) il passo successivo del goal corrente"""
        current_id = self.state["will"].get("current_goal_id")
        if not current_id:
            logger.debug("WillEngine: nessun goal in corso da perseguire.")
            return None

        goal = self.get_goal(current_id)
        if not goal:
            logger.warning(f"WillEngine: goal corrente id {current_id} non trovato.")
            self.state["will"]["current_goal_id"] = None
            self._save_state()
            return None

        # Recupera o genera piano
        plan_key = f"plan_for_{goal['id']}"
        plan = goal.get(plan_key)
        if plan is None:
            plan = self.plan_for_goal(goal)
            goal[plan_key] = plan
            self._save_state()

        # Trova primo step pending
        next_step = None
        for step in plan:
            if step.get("status") == "pending":
                next_step = step
                break

        if not next_step:
            # goal completato
            goal["status"] = "completed"
            goal["progress"] = 1.0
            logger.info(f"WillEngine: goal completato: {goal['description']} (id={goal['id']})")
            # registra in memoria
            try:
                if self.memory:
                    self.memory.add_experience(f"Goal completato: {goal['description']}", category="goal", importance=4)
            except Exception:
                logger.debug("WillEngine: errore registrando goal completato in memoria.")
            self.state["will"]["current_goal_id"] = None
            self._save_state()
            return "completed"

        # Esegui step: se planner ha execute/ fai_task proviamo a delegare
        executed = False
        try:
            if self.planner and hasattr(self.planner, "execute_step"):
                res = self.planner.execute_step(next_step)
                executed = True
                next_step["status"] = "done"
                next_step["result"] = res
                logger.info(f"WillEngine: step eseguito tramite planner: {next_step['id']}")
        except Exception:
            logger.exception("WillEngine: planner presente ma execute_step ha fallito.")

        if not executed:
            # fallback: simuliamo esecuzione e segniamo progresso
            logger.info(f"WillEngine: esecuzione fallback per step {next_step['id']}")
            # semplice simulazione: incrementa progress proporzionalmente
            next_step["status"] = "done"
            next_step["result"] = {"simulated": True, "timestamp": datetime.utcnow().isoformat()}

        # Aggiorna progresso del goal
        try:
            done = sum(1 for s in plan if s.get("status") in ("done", "skipped"))
            total = max(1, len(plan))
            goal["progress"] = float(done) / float(total)
        except Exception:
            goal["progress"] = min(1.0, goal.get("progress", 0.0) + 0.2)

        # registra azione nella cronologia del will
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "goal_id": goal["id"],
            "step_id": next_step["id"],
            "description": next_step.get("description"),
            "status": next_step.get("status"),
        }
        self.state["will"]["action_history"].append(rec)

        # Salva stato
        self._save_state()
        return rec

    # -------------------------
    # Loop di aggiornamento (da chiamare periodicamente)
    # -------------------------
    def update(self):
        """
        Metodo centrale da chiamare nel loop principale:
         - valuta goal
         - seleziona current_goal
         - tenta di perseguire un passo del goal
         - adatta will_power in base a successi / fallimenti
        """
        logger.debug("WillEngine: update beginning.")
        try:
            # 1) valuta e seleziona
            self.select_current_goal()

            # 2) prova a perseguire il goal corrente
            result = self.pursue_current_goal()
            logger.debug(f"WillEngine: pursue result: {result}")

            # 3) adattamento will_power (semplice rule-based)
            try:
                last_action = self.state["will"]["action_history"][-1] if self.state["will"]["action_history"] else None
                if last_action and last_action.get("status") == "done":
                    # ricompensa
                    self.state["will"]["will_power"] = min(10.0, self.state["will"].get("will_power", 1.0) + 0.05)
                else:
                    # leggera erosione
                    self.state["will"]["will_power"] = max(0.1, self.state["will"].get("will_power", 1.0) - 0.01)
            except Exception:
                pass

            # 4) salva stato finale
            self._save_state()
            logger.debug("WillEngine: update completed.")
        except Exception as e:
            logger.exception(f"WillEngine: errore in update: {e}")

    # -------------------------
    # API helper
    # -------------------------
    def list_goals(self):
        return self.state["will"]["goals"]

    def get_current_goal(self):
        gid = self.state["will"].get("current_goal_id")
        return self.get_goal(gid) if gid else None


# -------------------------
# Test / Esempio rapido
# -------------------------
if __name__ == "__main__":
    logger.info("Esecuzione test will_engine.py")
    we = WillEngine()
    # aggiunge un goal di test se non ce ne sono
    if not we.list_goals():
        we.add_goal("Imparare a riconoscere immagini", priority=7)
        we.add_goal("Scrivere diario giornaliero", priority=5)
    # ciclo di prova: update 3 volte
    for _ in range(3):
        we.update()
        import time
        time.sleep(1)
    logger.info("Test WillEngine completato.")
