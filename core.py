# core.py
"""
Core orchestrator per Nova (versione con integrazione EthicsEngine natural-teaching).
- Inizializza EthicsEngine(self.state, timeline=self.memory_timeline, motivational_engine=self.motivational_engine)
- Rileva enunciati normativi naturali (es. "Non urlare perché è maleducato") su receive_remote_message
  e li processa con process_natural_statement()
- Registra job di induction etica su Scheduler (NOVA_ETHICS_INDUCTION_INTERVAL, default 86400s)
- Espone should_execute_action() e explain_action() per planner/agent
"""

import os
import time
import yaml
import threading
import signal
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

# Percorsi/config
STATE_FILE = "internal_state.yaml"
CONFIG_FILE = "config.yaml"
LOG_DIR = "nova_logs"

# Caricamento config semplice (env override)
_config = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as _f:
            _config = yaml.safe_load(_f) or {}
    except Exception:
        _config = {}
# overlay env variables that start with NOVA_
for k, v in os.environ.items():
    if k.startswith("NOVA_"):
        _config[k[5:]] = v

LOOP_SLEEP = float(_config.get("LOOP_SLEEP", 1.0))
ETHICS_INDUCTION_INTERVAL = int(_config.get("ETHICS_INDUCTION_INTERVAL", _config.get("ETHICS_IND_INTERVAL", 86400)))  # seconds, default 1 day

# -----------------------
# Import dei moduli (con fallback no-op)
# -----------------------
def _noop(*args, **kwargs):
    return None

class _NoopModule:
    def __init__(self, *args, **kwargs):
        pass
    def update(self, *a, **k): pass
    def cycle(self, *a, **k): pass

try:
    from scheduler_nova import Scheduler
except Exception:
    Scheduler = lambda *a, **k: _NoopModule()

try:
    from planner import Planner
except Exception:
    Planner = lambda *a, **k: _NoopModule()

try:
    from emotion_engine import EmotionEngine
except Exception:
    EmotionEngine = lambda *a, **k: _NoopModule()

try:
    from motivational_engine import MotivationalEngine
except Exception:
    MotivationalEngine = lambda *a, **k: _NoopModule()

try:
    from dream_generator import DreamGenerator
except Exception:
    DreamGenerator = lambda *a, **k: _NoopModule()

try:
    from conscious_loop import ConsciousLoop
except Exception:
    ConsciousLoop = lambda *a, **k: _NoopModule()

try:
    from life_journal import LifeJournal
except Exception:
    LifeJournal = lambda *a, **k: _NoopModule()

try:
    from attention_manager import AttentionManager
except Exception:
    AttentionManager = lambda *a, **k: _NoopModule()

try:
    from identity_manager import IdentityManager
except Exception:
    IdentityManager = lambda *a, **k: _NoopModule()

try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = lambda *a, **k: _NoopModule()

# EthicsEngine (natural-teaching variant)
try:
    from ethics_engine import EthicsEngine
except Exception:
    EthicsEngine = lambda *a, **k: _NoopModule()

# -----------------------
# Helper: scrittura atomica dello stato (temp -> rename)
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# Utility locali (safe init / introspection)
# -----------------------
def _safe_init(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception:
        try:
            return cls(*args)
        except Exception:
            logger.warning("Impossibile inizializzare %s, using noop fallback.", getattr(cls, "__name__", str(cls)))
            return _NoopModule()

def _has_init_args(cls, names):
    try:
        import inspect
        sig = inspect.signature(cls)
        for n in names:
            if n in sig.parameters:
                return True
    except Exception:
        pass
    return False

# -----------------------
# Simple heuristic: detect if text is an ethical/normative statement
# -----------------------
def is_normative_statement(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.lower()
    indicators = ["non ", "evita", "mai", "dovresti", "dovrei", "è meglio", "è maleducato", "perché", "perche", "perchè", "ti prego", "per favore non", "smettila", "non fare"]
    for ind in indicators:
        if ind in t:
            return True
    # short heuristic: sentences containing 'perché' or 'perche' are often explanatory statements
    if "perché" in t or "perche" in t:
        return True
    return False

# -----------------------
# NovaCore
# -----------------------
class NovaCore:
    def __init__(self):
        # Preparazione log
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.add(os.path.join(LOG_DIR, "nova_{time:YYYY-MM-DD}.log"), rotation="1 day", compression="zip")
        logger.info("Inizializzazione NovaCore...")

        # Lock per stato
        self._state_lock = threading.RLock()

        # Carica stato
        self.state = self._load_state()

        # Inizializza moduli (pass state quando possibile)
        try:
            self.scheduler = Scheduler(self.state, core=self) if _has_init_args(Scheduler, ("state", "core")) else Scheduler(core=self)
        except Exception:
            self.scheduler = _safe_init(Scheduler)

        self.planner = _safe_init(Planner, self.state, core=self)
        self.emotion_engine = _safe_init(EmotionEngine, self.state, core=self)
        self.motivational_engine = _safe_init(MotivationalEngine, self.state, core=self)
        self.dream_generator = _safe_init(DreamGenerator, self.state, core=self)
        self.conscious_loop = _safe_init(ConsciousLoop, self.state, core=self)
        self.life_journal = _safe_init(LifeJournal, self.state, core=self)
        self.attention_manager = _safe_init(AttentionManager, self.state, core=self)
        self.identity_manager = _safe_init(IdentityManager, self.state, core=self)
        self.memory_timeline = _safe_init(MemoryTimeline, self.state, core=self)

        # EthicsEngine: pass timeline & motivational_engine so it can store examples and reward
        self.ethics = _safe_init(EthicsEngine, self.state, None, self.memory_timeline, self.motivational_engine)
        # If constructor signature differs, try alternative inits
        try:
            # fallback: EthicsEngine(state=..., timeline=..., motivational_engine=...)
            if isinstance(self.ethics, _NoopModule):
                self.ethics = _safe_init(EthicsEngine, self.state, rules_path=None, timeline=self.memory_timeline, motivational_engine=self.motivational_engine)
        except Exception:
            pass

        # Registrazione dei task ricorrenti (compatibile Scheduler)
        self._register_recurring_tasks()

        # Stato operativo
        self.running = True

        # Gestione segnali per shutdown pulito
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("NovaCore inizializzato correttamente.")

    # -------------------
    # Stato
    # -------------------
    def _load_state(self) -> Dict[str, Any]:
        with self._state_lock:
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, "r", encoding="utf8") as f:
                        data = yaml.safe_load(f) or {}
                    logger.info("Stato interno caricato da file.")
                    return data
                except Exception as e:
                    logger.exception("Errore caricamento stato interno, partire con stato vuoto: %s", e)
                    return {}
            else:
                logger.info("Creazione nuovo stato interno predefinito.")
                return {
                    "emotions": {},
                    "motivations": {},
                    "dreams": [],
                    "tasks": [],
                    "identity": {},
                    "context": {"last_thought": ""},
                    "diary": [],
                    "timeline": [],
                    "ethics_examples": []
                }

    def save_state(self) -> None:
        """Salva lo stato in modo thread-safe e atomico."""
        with self._state_lock:
            try:
                text = yaml.safe_dump(self.state, allow_unicode=True)
                _atomic_write(STATE_FILE, text)
                logger.debug("Stato interno salvato (%s).", datetime.utcnow().isoformat())
            except Exception:
                logger.exception("Errore salvataggio stato interno")

    # -------------------
    # Registrazione task ricorrenti (compatibilità Scheduler)
    # -------------------
    def _register_recurring_tasks(self) -> None:
        task_map = [
            (self.update_emotions, int(_config.get("EMOTION_UPDATE_INTERVAL", 5))),
            (self.update_motivation, int(_config.get("MOTIVATION_UPDATE_INTERVAL", 5))),
            (self.run_conscious_loop, int(_config.get("CONSCIOUS_LOOP_INTERVAL", 2))),
            (self.update_diary, int(_config.get("DIARY_UPDATE_INTERVAL", 60))),
            (self.record_memory, int(_config.get("RECORD_MEMORY_INTERVAL", 10))),
        ]

        # register standard tasks
        for func, interval in task_map:
            try:
                if hasattr(self.scheduler, "add_recurring_task"):
                    self.scheduler.add_recurring_task(func, interval=interval)
                elif hasattr(self.scheduler, "add_job"):
                    try:
                        self.scheduler.add_job(func, interval_seconds=interval)
                    except TypeError:
                        self.scheduler.add_job(func, interval)
                else:
                    logger.warning("Scheduler non espone API conosciute; skipping %s", func.__name__)
            except Exception:
                logger.exception("Errore registrazione job ricorrente per %s", func.__name__)

        # Register ethics induction job (adaptive interval via config)
        try:
            if hasattr(self.ethics, "induction_job"):
                if hasattr(self.scheduler, "add_recurring_task"):
                    self.scheduler.add_recurring_task(lambda: self._safe_run(self.ethics.induction_job), interval=ETHICS_INDUCTION_INTERVAL, tag="ethics_induction")
                else:
                    try:
                        self.scheduler.add_job(lambda: self._safe_run(self.ethics.induction_job), interval_seconds=ETHICS_INDUCTION_INTERVAL)
                    except Exception:
                        # best effort
                        pass
                logger.info("Ethics induction job registrato con intervallo %ss", ETHICS_INDUCTION_INTERVAL)
        except Exception:
            logger.exception("Errore registrazione ethics induction")

    def _safe_run(self, func, *a, **k):
        try:
            return func(*a, **k)
        except Exception:
            logger.exception("Errore in safe_run wrapper per %s", getattr(func, "__name__", str(func)))

    # -------------------
    # Funzioni ricorrenti integrate
    # -------------------
    def update_emotions(self) -> None:
        try:
            logger.debug("update_emotions triggered")
            if hasattr(self.emotion_engine, "update"):
                self.emotion_engine.update()
            self.save_state()
        except Exception:
            logger.exception("Errore in update_emotions")

    def update_motivation(self) -> None:
        try:
            logger.debug("update_motivation triggered")
            if hasattr(self.motivational_engine, "update"):
                self.motivational_engine.update()
            self.save_state()
        except Exception:
            logger.exception("Errore in update_motivation")

    def run_conscious_loop(self) -> None:
        try:
            logger.debug("run_conscious_loop triggered")
            if hasattr(self.conscious_loop, "cycle"):
                self.conscious_loop.cycle()
            if hasattr(self.dream_generator, "generate_if_needed"):
                self.dream_generator.generate_if_needed()
            self.save_state()
        except Exception:
            logger.exception("Errore in run_conscious_loop")

    def update_diary(self) -> None:
        try:
            logger.debug("update_diary triggered")
            if hasattr(self.life_journal, "update"):
                self.life_journal.update()
            self.save_state()
        except Exception:
            logger.exception("Errore in update_diary")

    def record_memory(self) -> None:
        """Registra eventi e riflessioni nella timeline"""
        try:
            last = self.state.get("context", {}).get("last_thought", "")
            content = f"Pensiero corrente: {last}"
            if hasattr(self.memory_timeline, "add_experience"):
                self.memory_timeline.add_experience(content, category="reflection", importance=2)
            else:
                with self._state_lock:
                    self.state.setdefault("timeline", []).append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "content": content,
                        "category": "reflection",
                        "importance": 2
                    })
            self.save_state()
            logger.debug("record_memory salvata")
        except Exception:
            logger.exception("Errore in record_memory")

    # -------------------
    # Core thinking loop (invocato dal run principale)
    # -------------------
    def think(self) -> None:
        try:
            logger.debug("think() start")
            if hasattr(self.attention_manager, "update"):
                self.attention_manager.update()
            # planner may implement fai_task() or handle_pending_tasks()
            if hasattr(self.planner, "fai_task"):
                # if planner can ask core to check ethics, it should call back to should_execute_action()
                self.planner.fai_task()
            elif hasattr(self.planner, "handle_pending_tasks"):
                self.planner.handle_pending_tasks()
            # identity update
            if hasattr(self.identity_manager, "update"):
                self.identity_manager.update()
            self.save_state()
            logger.debug("think() end")
        except Exception:
            logger.exception("Errore in think()")

    # -------------------
    # Remote hooks (chiamabili da remote_comm)
    # -------------------
    def receive_remote_message(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Route inbound text from remote interfaces into planner / timeline, AND detect natural moral teaching statements."""
        try:
            logger.info("Ricevuto messaggio remoto da %s: %s", user_id, (text[:200] if text else ""))
            # store inbound in timeline
            if hasattr(self.memory_timeline, "add_experience"):
                self.memory_timeline.add_experience({"user_id": user_id, "text": text}, category="remote", importance=2)

            # Detect if user is giving a normative statement (natural teaching)
            try:
                if isinstance(text, str) and is_normative_statement(text) and hasattr(self.ethics, "process_natural_statement"):
                    logger.info("Rilevato enunciato etico naturale, processo con EthicsEngine")
                    self.ethics.process_natural_statement(text, teacher_id=user_id, context={"source":"remote_message","metadata":metadata or {}})
                    # save state after learning
                    self.save_state()
            except Exception:
                logger.exception("Errore durante process_natural_statement")

            # Forward to planner for normal handling (planner may call back to core.should_execute_action)
            if hasattr(self.planner, "handle_incoming_message"):
                try:
                    self.planner.handle_incoming_message(user_id, text, metadata=metadata)
                except Exception:
                    logger.exception("Errore planner.handle_incoming_message")
            else:
                # fallback: log in state
                with self._state_lock:
                    self.state.setdefault("inbox", []).append({"from": user_id, "text": text, "metadata": metadata})
                    self.save_state()
        except Exception:
            logger.exception("Errore in receive_remote_message")

    def receive_remote_image(self, user_id: str, image_path: str, analysis: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            logger.info("Ricevuta immagine remota da %s -> %s", user_id, image_path)
            if hasattr(self.planner, "handle_incoming_image"):
                self.planner.handle_incoming_image(user_id, image_path, analysis=analysis, metadata=metadata)
            elif hasattr(self.memory_timeline, "add_experience"):
                self.memory_timeline.add_experience({"user_id": user_id, "image": image_path, "analysis": analysis}, category="remote_image", importance=3)
            self.save_state()
        except Exception:
            logger.exception("Errore in receive_remote_image")

    def receive_remote_voice(self, user_id: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            logger.info("Ricevuto audio remoto da %s -> %s", user_id, payload.get("audio_path"))
            # try to transcribe if speech module exists (planner will be given transcript)
            transcript = payload.get("transcript")
            if not transcript and hasattr(self, "speech_io") and hasattr(self.speech_io, "transcribe"):
                try:
                    transcript = self.speech_io.transcribe(payload.get("audio_path"))
                except Exception:
                    logger.exception("Errore trascrizione audio in receive_remote_voice")
            # save to timeline
            if hasattr(self.memory_timeline, "add_experience"):
                self.memory_timeline.add_experience({"user_id": user_id, "voice": payload, "transcript": transcript}, category="remote_voice", importance=3)
            # forward to planner
            if hasattr(self.planner, "handle_incoming_voice"):
                self.planner.handle_incoming_voice(user_id, {"audio_path": payload.get("audio_path"), "transcript": transcript}, metadata=metadata)
            self.save_state()
        except Exception:
            logger.exception("Errore in receive_remote_voice")

    # -------------------
    # Ethics helpers (exposed to planner/agents)
    # -------------------
    def should_execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Verifica se un'azione proposta è eticamente accettabile.
        action: {"text": "...", "tags": [...], "metadata": {...}}
        Ritorna True se può procedere, False se veto.
        """
        try:
            if hasattr(self.ethics, "evaluate_action"):
                res = self.ethics.evaluate_action(action)
                logger.debug("Ethics evaluate_action result: %s", res)
                if not res.get("allowed", True):
                    # log rejection into timeline and optionally notify planner
                    if hasattr(self.memory_timeline, "add_experience"):
                        self.memory_timeline.add_experience({"action": action, "ethics_verdict": res}, category="ethics_reject", importance=4)
                    return False
                return True
            # default allow if no ethics engine
            return True
        except Exception:
            logger.exception("Errore in should_execute_action")
            return True

    def explain_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if hasattr(self.ethics, "explain_action"):
                return self.ethics.explain_action(action)
            else:
                return {"explain": "no_ethics_engine"}
        except Exception:
            logger.exception("Errore in explain_action")
            return {"error": "exception"}

    # -------------------
    # Loop principale e gestione scheduler
    # -------------------
    def run(self) -> None:
        logger.info("Avvio loop principale di Nova...")
        try:
            while self.running:
                try:
                    if hasattr(self.scheduler, "run_pending"):
                        self.scheduler.run_pending()
                    elif hasattr(self.scheduler, "tick"):
                        self.scheduler.tick()
                except Exception:
                    logger.exception("Errore durante scheduler.run_pending()")
                self.think()
                time.sleep(LOOP_SLEEP)
        except Exception:
            logger.exception("Eccezione non gestita nel run principale")
        finally:
            self._cleanup()

    def stop(self) -> None:
        logger.info("Stop richiesta su NovaCore")
        self.running = False
        try:
            if hasattr(self.scheduler, "stop"):
                self.scheduler.stop()
        except Exception:
            logger.exception("Errore durante stop scheduler")

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Segnale di terminazione ricevuto: %s", signum)
        self.stop()

    def _cleanup(self) -> None:
        logger.info("Cleanup finale NovaCore: salvataggio stato e arresto moduli.")
        try:
            self.save_state()
        except Exception:
            logger.exception("Errore salvataggio stato nel cleanup")
        for mod in (self.conscious_loop, self.dream_generator, self.scheduler):
            try:
                if hasattr(mod, "stop"):
                    mod.stop()
            except Exception:
                logger.exception("Errore stop modulo %s", getattr(mod, "__name__", str(mod)))

# -----------------------
# Esecuzione stand-alone
# -----------------------
if __name__ == "__main__":
    nova = NovaCore()
    try:
        nova.run()
    except KeyboardInterrupt:
        nova.stop()
