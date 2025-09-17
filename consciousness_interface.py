# consciousness_interface.py
"""
ConsciousnessInterface - Interfaccia di alto livello per la coscienza di Nova

Responsabilità:
- Orchestrare i moduli di percezione, pensiero e azione.
- Gestire input esterni (messaggi, voce, visione) e trasformarli in percezioni.
- Eseguire il ciclo di pensiero interno in modo sicuro e controllato.
- Applicare controlli etici prima di compiere azioni esterne.
- Esporre hook/callback per UI, scheduler e supervisione umana.

Nota importante:
Questo modulo non esegue modifiche destructive o auto-propagative. Qualsiasi azione che
implichi operazioni fisiche, transazioni economiche, o diffusione ampia di messaggi richiede
livello di autonomia adeguato nello stato (state['settings']['autonomy_level']) o approvazione manuale.
"""

import time
import threading
import queue
from typing import Optional, Callable, Any, Dict
from loguru import logger

# Import difensivo dei moduli (evita crash se interfacce cambiano)
try:
    from internal_state import InternalState
except Exception:
    InternalState = None

# Moduli del progetto (si presume che esistano; usiamo riferimenti difensivi)
try:
    from dream_generator import DreamGenerator
    from emotion_engine import EmotionEngine
    from motivational_engine import MotivationalEngine
    from planner import Planner
    from inner_dialogue import InnerDialogue
    from remote_comm import RemoteComm
    from speech_io import SpeechIO
    from memory_timeline import MemoryTimeline
    from attention_manager import AttentionManager
except Exception:
    # placeholder None se mancano (il codice userà fallback)
    DreamGenerator = EmotionEngine = MotivationalEngine = Planner = InnerDialogue = RemoteComm = SpeechIO = MemoryTimeline = AttentionManager = None

# Constants
DEFAULT_LOOP_SLEEP = 0.8  # secondi tra iterazioni principali
OUTBOUND_RATE_LIMIT_SECONDS = 2.0  # intervallo minimo tra invii esterni
DEFAULT_AUTONOMY_THRESHOLD = 0.6  # sopra questo valore, Nova può agire senza conferme frequenti


class ConsciousnessInterface:
    def __init__(self, core: Optional[Any] = None, state_file: str = "internal_state.yaml"):
        """
        core: opzionale istanza NovaCore; se fornito useremo i suoi moduli referenziati per wiring
        state_file: percorso dello stato persistente (opzionale se usi InternalState)
        """
        # Stato (wrapper o dict)
        if core and hasattr(core, "state"):
            self.state = core.state
        else:
            if InternalState:
                self.state = InternalState(state_file)
            else:
                # fallback semplice: dict caricato da file se presente
                import yaml, os
                if os.path.exists(state_file):
                    with open(state_file, "r", encoding="utf8") as f:
                        self.state = yaml.safe_load(f) or {}
                else:
                    self.state = {}

        # wiring ai moduli (preferiamo quelli già istanziati in core)
        self.core = core
        self.dream_generator = getattr(core, "dream_generator", None) or (DreamGenerator(self.state) if DreamGenerator else None)
        self.emotion_engine = getattr(core, "emotion_engine", None) or (EmotionEngine(self.state) if EmotionEngine else None)
        self.motivational_engine = getattr(core, "motivational_engine", None) or (MotivationalEngine(self.state) if MotivationalEngine else None)
        self.planner = getattr(core, "planner", None) or (Planner(self.state) if Planner else None)
        self.inner_dialogue = getattr(core, "inner_dialogue", None) or (InnerDialogue(self.state, getattr(self.emotion_engine, "None"), getattr(self.motivational_engine, "None"), getattr(self, "memory_timeline", None), getattr(self.dream_generator, None), getattr(self, "life_journal", None), getattr(self, "attention_manager", None)) if InnerDialogue else None)
        self.memory_timeline = getattr(core, "memory_timeline", None) or (MemoryTimeline() if MemoryTimeline else None)
        self.attention_manager = getattr(core, "attention_manager", None) or (AttentionManager(self.state) if AttentionManager else None)
        self.remote_comm = getattr(core, "remote_comm", None) or (RemoteComm(self.state) if RemoteComm else None)
        self.speech_io = getattr(core, "speech_io", None) or (SpeechIO(self.state) if SpeechIO else None)

        # emergent_self and self_reflection if available in core
        self.emergent_self = getattr(core, "emergent_self", None)
        self.self_reflection = getattr(core, "self_reflection", None)
        self.identity_manager = getattr(core, "identity_manager", None)
        self.ethics = getattr(core, "ethics", None)  # engine etico opzionale

        # runtime internals
        self._running = threading.Event()
        self._running.set()
        self._loop_thread: Optional[threading.Thread] = None
        self._inbound_q = queue.Queue()
        self._last_outbound_ts = 0.0

        # callbacks per UI / monitoraggio: list of callables senza argomenti
        self._callbacks = []

        # Ensure settings defaults
        self.state.setdefault("settings", {})
        self.state["settings"].setdefault("auto_save_interval_seconds", 60)
        self.state["settings"].setdefault("autonomy_level", 0.2)  # 0..1 (0 = manuale, 1 = autonoma)
        self.state["settings"].setdefault("tts_enabled", False)
        self.state["settings"].setdefault("remote_comm_enabled", False)

        logger.info("ConsciousnessInterface inizializzata. Autonomy=%s", self.state["settings"]["autonomy_level"])

    # -----------------------
    # Helpers locali
    # -----------------------
    def _can_act_externally(self, required_level: float = 0.5) -> bool:
        """Controlla se il livello di autonomia permette azioni esterne senza conferma."""
        level = float(self.state.get("settings", {}).get("autonomy_level", 0.0))
        return level >= required_level

    def _rate_limit_outbound(self) -> bool:
        now = time.time()
        if now - self._last_outbound_ts < OUTBOUND_RATE_LIMIT_SECONDS:
            return False
        self._last_outbound_ts = now
        return True

    def _ethics_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Se disponibile, usa l'ethics engine per validare un'azione/testo.
        Restituisce dict {'allowed': bool, 'reason': str, 'verdict': {...}}
        """
        try:
            if self.ethics and hasattr(self.ethics, "evaluate_action"):
                verdict = self.ethics.evaluate_action(payload)
                allowed = verdict.get("allowed", False)
                return {"allowed": allowed, "verdict": verdict}
        except Exception:
            logger.exception("Errore during ethics evaluation")
        # default permissive but mark uncertain
        return {"allowed": True, "verdict": {"note": "no_ethics_engine"}}

    # -----------------------
    # Process internal thoughts (ciclo di coscienza)
    # -----------------------
    def process_internal_thoughts(self):
        """
        Esegue aggiornamenti interni in un ordine consapevole:
        emotion -> motivational -> inner dialogue -> conscious tasks -> emergent_self/dreams
        """
        try:
            logger.debug("process_internal_thoughts: start")

            # 1) emotion update
            if self.emotion_engine and hasattr(self.emotion_engine, "update"):
                self.emotion_engine.update()

            # 2) motivational update
            if self.motivational_engine and hasattr(self.motivational_engine, "update"):
                self.motivational_engine.update()

            # 3) inner dialogue / reflection
            if self.inner_dialogue and hasattr(self.inner_dialogue, "cycle"):
                try:
                    self.inner_dialogue.cycle()
                except Exception:
                    # older inner_dialogue naming fallback
                    try:
                        self.inner_dialogue.run_cycle()
                    except Exception:
                        logger.debug("inner_dialogue cycle fallback failed")

            # 4) emergent self (narrative update) - leggero, non a ogni ciclo
            if self.emergent_self and hasattr(self.emergent_self, "update_narrative"):
                # chiamalo meno frequentemente: solo se curiosità alta o random trial
                curiosity = float(self.state.get("emotions", {}).get("curiosity", 0.0) or 0.0)
                if curiosity > 0.6 or (time.time() % 60) < 1.0:
                    self.emergent_self.update_narrative(use_llm=False)

            # 5) dream generation opportunistica
            if self.dream_generator and hasattr(self.dream_generator, "generate_dream"):
                # genera solo rare volte nel ciclo
                if (time.time() % 120) < 1.0:
                    try:
                        self.dream_generator.generate_dream()
                    except Exception:
                        logger.debug("dream_generator generate_dream fallback failed")

            # 6) update attention
            if self.attention_manager and hasattr(self.attention_manager, "update_focus"):
                try:
                    self.attention_manager.update_focus()
                except Exception:
                    try:
                        self.attention_manager.update()
                    except Exception:
                        logger.debug("attention update fallback failed")

            # Persist occasional state if core/state wrapper offers save()
            try:
                if hasattr(self.state, "save"):
                    self.state.save()
                else:
                    # if it's dict with a state_file we don't auto-save here to avoid IO churn
                    pass
            except Exception:
                logger.exception("Errore salvataggio stato in process_internal_thoughts")

            # run callbacks (e.g., UI)
            for cb in list(self._callbacks):
                try:
                    cb()
                except Exception:
                    logger.exception("Callback hook error")

            logger.debug("process_internal_thoughts: end")
        except Exception:
            logger.exception("Errore generale in process_internal_thoughts")

    # -----------------------
    # Input handling
    # -----------------------
    def receive_remote_message(self) -> Optional[Dict[str, Any]]:
        """
        Riceve un messaggio dalla interfaccia remota (Telegram/WhatsApp) in modo difensivo.
        Restituisce dict {type, text, sender, metadata} o None.
        """
        if not self.state.get("settings", {}).get("remote_comm_enabled", False):
            return None
        if not self.remote_comm:
            return None
        try:
            msg = self.remote_comm.receive()
            if not msg:
                return None
            # normalizza in dict
            if isinstance(msg, dict):
                return msg
            return {"type": "text", "text": str(msg)}
        except Exception:
            logger.exception("Errore ricezione remote message")
            return None

    def receive_voice_input(self) -> Optional[str]:
        """Ritorna testo trascritto dal modulo speech_io (se attivo)"""
        if not self.state.get("settings", {}).get("tts_enabled", False):
            return None
        if not self.speech_io:
            return None
        try:
            txt = self.speech_io.listen()
            return txt
        except Exception:
            logger.exception("Errore ricezione voce")
            return None

    def ingest_perception(self, perception: Dict[str, Any]):
        """
        Ingesta una percezione interna (es. immagine analizzata, testo, sensore).
        Aggiunge alla queue interna per essere processata dal loop.
        """
        try:
            self._inbound_q.put(perception, block=False)
            logger.debug("Perception immessa in queue: %s", str(perception)[:120])
        except queue.Full:
            logger.warning("Queue piena, percezione scartata")

    # -----------------------
    # Outbound: invio messaggi in modo sicuro
    # -----------------------
    def send_outbound(self, text: str, require_confirmation: bool = False, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Invio protetto di messaggi esterni:
         - verifica ethics (se presente)
         - verifica rate limit
         - richiede conferma se autonomy_level basso o require_confirmation True
        """
        try:
            payload = {"text": text, "metadata": metadata or {}, "source": "consciousness_interface"}
            # ethics
            eth = self._ethics_check(payload)
            if not eth.get("allowed", True):
                logger.warning("Outbound bloccato da Ethics: %s", eth.get("verdict"))
                return False

            # autonomy check
            if require_confirmation and not self._can_act_externally(required_level=DEFAULT_AUTONOMY_THRESHOLD):
                logger.info("Outbound require confirmation ma autonomia insufficiente -> blocco")
                return False

            # rate limit
            if not self._rate_limit_outbound():
                logger.warning("Outbound rate limit attivo, messaggio scartato/ritardato")
                return False

            # perform send (remote + tts if enabled)
            sent = False
            if self.remote_comm and self.state.get("settings", {}).get("remote_comm_enabled", False):
                try:
                    self.remote_comm.send(payload)
                    sent = True
                except Exception:
                    logger.exception("Errore invio remote_comm")

            if self.speech_io and self.state.get("settings", {}).get("tts_enabled", False):
                try:
                    # speak non-blocking se implementato così
                    self.speech_io.speak(text)
                    sent = True
                except Exception:
                    logger.exception("Errore TTS speak")

            # log in timeline
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": text, "meta": {"outbound": True}}, category="external", importance=1)
            except Exception:
                pass

            logger.info("Outbound eseguito: %s", text[:120])
            return sent
        except Exception:
            logger.exception("Errore in send_outbound")
            return False

    # -----------------------
    # Main loop: process queue + periodic internal thoughts + incoming remote/voice
    # -----------------------
    def _main_loop(self, sleep_interval: float = DEFAULT_LOOP_SLEEP):
        logger.info("ConsciousnessInterface main loop avviato.")
        try:
            while self._running.is_set():
                # step 1: process internal thought cycle
                try:
                    self.process_internal_thoughts()
                except Exception:
                    logger.exception("process_internal_thoughts fallito nel loop")

                # step 2: process inbound perceptions (max N per ciclo)
                try:
                    for _ in range(6):
                        try:
                            p = self._inbound_q.get(block=False)
                        except queue.Empty:
                            break
                        try:
                            self._handle_perception(p)
                        finally:
                            self._inbound_q.task_done()
                except Exception:
                    logger.exception("Errore processing inbound queue")

                # step 3: check remote messages
                try:
                    rm = self.receive_remote_message()
                    if rm:
                        self._handle_remote_message(rm)
                except Exception:
                    logger.exception("Errore receive_remote_message")

                # step 4: check voice input
                try:
                    vi = self.receive_voice_input()
                    if vi:
                        self._handle_text_input({"type": "voice", "text": vi})
                except Exception:
                    logger.exception("Errore receive_voice_input")

                # small sleep to avoid busy loop
                time.sleep(sleep_interval)
        except Exception:
            logger.exception("Eccezione nel main loop di ConsciousnessInterface")
        logger.info("ConsciousnessInterface main loop terminato.")

    # -----------------------
    # Perception/message handlers
    # -----------------------
    def _handle_perception(self, perception: Dict[str, Any]):
        """
        Perception può essere:
         - {'type':'image', 'image_path':...}
         - {'type':'text', 'text':...}
         - {'type':'sensor', ...}
         Trasforma la percezione in eventi per i moduli (emotion, memory, attention, planner...)
        """
        try:
            t = perception.get("type", "text")
            if t == "text":
                self._handle_text_input({"type": "text", "text": perception.get("text", "")})
            elif t == "image":
                # se esiste un vision module, passalo; altrimenti registra nella timeline
                vision = getattr(self.core, "vision_engine", None) if self.core else None
                if vision and hasattr(vision, "analyze_image"):
                    result = vision.analyze_image(perception.get("image_path"))
                    self.memory_timeline.add_experience({"text": f"Vision analysis: {result}"}, category="perception", importance=2)
                else:
                    self.memory_timeline.add_experience({"text": f"Image perceived: {perception.get('image_path')}"}, category="perception", importance=1)
            else:
                self.memory_timeline.add_experience({"text": f"Perception: {perception}"}, category="perception", importance=1)
            # notify emotion engine for appraisal
            try:
                if self.emotion_engine and hasattr(self.emotion_engine, "process_experience"):
                    self.emotion_engine.process_experience({"content": perception.get("text", str(perception)), "source": "perception", "importance": perception.get("importance", 0.5)})
            except Exception:
                logger.debug("Appraisal fallback failed")
        except Exception:
            logger.exception("Errore _handle_perception")

    def _handle_text_input(self, msg: Dict[str, Any]):
        """
        msg: {'type':'text'|'voice', 'text': str, 'sender':...}
        Pianifica azioni tramite planner e registra nel diario/timeline.
        """
        try:
            text = msg.get("text", "")
            sender = msg.get("sender")
            logger.info("Input testuale ricevuto: %s", text[:140])

            # 1) store in timeline
            try:
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": text, "meta": {"sender": sender}}, category="external_message", importance=2)
            except Exception:
                logger.debug("timeline store failed for text input")

            # 2) pass to self_reflection for learning/insight
            try:
                if self.self_reflection and hasattr(self.self_reflection, "analyze_experience"):
                    self.self_reflection.analyze_experience({"content": text, "type": "interaction", "importance": 0.6})
            except Exception:
                logger.debug("self_reflection analyze failed")

            # 3) Ask planner to interpret and create tasks / replies
            try:
                if self.planner and hasattr(self.planner, "fai_task"):
                    # planner.fai_task should be robust and accept (input_text, state, sender)
                    try:
                        self.planner.fai_task(text, self.state, sender=sender)
                    except TypeError:
                        # fallback: older signature
                        self.planner.fai_task(text)
                else:
                    # simple fallback: append to tasks
                    self.state.setdefault("tasks", []).append({"title": f"Consider: {text}", "timestamp": time.time()})
            except Exception:
                logger.exception("Planner invocation failed")

        except Exception:
            logger.exception("Errore _handle_text_input")

    def _handle_remote_message(self, rm: Dict[str, Any]):
        """
        Gestione messaggi provenienti da remote_comm: routing e risposta automatica condizionale.
        """
        try:
            text = rm.get("text", "") if isinstance(rm, dict) else str(rm)
            sender = rm.get("sender") if isinstance(rm, dict) else None
            logger.info("Remote message from %s: %s", sender, text[:140])

            # Basic auto-reply behaviour: solo se autonomy_level elevato
            if self._can_act_externally(required_level=0.7):
                reply = None
                # try use planner to craft reply
                try:
                    if self.planner and hasattr(self.planner, "generate_reply"):
                        reply = self.planner.generate_reply(text, self.state, sender=sender)
                except Exception:
                    logger.debug("Planner generate_reply failed")
                # fallback simple heuristics: acknowledge
                if not reply:
                    reply = f"Ho ricevuto il tuo messaggio: {text[:80]}"
                self.send_outbound(reply, require_confirmation=False, metadata={"in_reply_to": sender})
            else:
                # store and notify owner (append to journal, optionally request approval)
                if self.memory_timeline and hasattr(self.memory_timeline, "add_experience"):
                    self.memory_timeline.add_experience({"text": f"Remote msg stored (no auto-reply): {text}", "meta":{"sender":sender}}, category="external_message", importance=2)
                logger.info("Autonomia insufficiente per auto-reply. Messaggio archiviato per revisione.")
        except Exception:
            logger.exception("Errore _handle_remote_message")

    # -----------------------
    # Lifecycle controls
    # -----------------------
    def start(self, in_thread: bool = True, sleep_interval: float = DEFAULT_LOOP_SLEEP) -> None:
        """Avvia il loop principale (thread separato se in_thread True)"""
        if self._loop_thread and self._loop_thread.is_alive():
            logger.warning("Main loop già in esecuzione")
            return
        self._running.set()
        if in_thread:
            self._loop_thread = threading.Thread(target=self._main_loop, args=(sleep_interval,), daemon=True)
            self._loop_thread.start()
            logger.info("ConsciousnessInterface: main loop avviato in thread.")
        else:
            self._main_loop(sleep_interval)

    def stop(self, wait: bool = True) -> None:
        """Ferma il loop principale in modo ordinato"""
        logger.info("ConsciousnessInterface: arresto richiesto.")
        self._running.clear()
        if wait and self._loop_thread:
            self._loop_thread.join(timeout=5.0)
            logger.info("Main loop thread terminato.")

    def register_callback(self, cb: Callable[[], None]) -> None:
        """Registra callback da chiamare dopo ogni ciclo di pensiero (UI, dashboard)"""
        if callable(cb):
            self._callbacks.append(cb)

    def unregister_callback(self, cb: Callable[[], None]) -> None:
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

# -----------------------
# Example quick-run (invocare come script solo in ambiente di sviluppo)
# -----------------------
if __name__ == "__main__":
    # Demo: avvia l'interfaccia con default state_dict e simula un input
    ci = ConsciousnessInterface(core=None)
    try:
        ci.start(in_thread=True, sleep_interval=1.0)
        # simula percezione testuale
        ci.ingest_perception({"type": "text", "text": "Ciao Nova, come stai?"})
        time.sleep(3)
        # simula comando esterno da remoto (se remote_comm disabilitato sarà solo salvato)
        ci.ingest_perception({"type": "text", "text": "Non urlare perché è maleducato", "importance": 0.8})
        time.sleep(5)
    finally:
        ci.stop()
