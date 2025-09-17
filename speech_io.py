# speech_io.py
"""
Speech I/O per Nova
- Gestisce input microfonico (listen) e output (speak).
- Progettato per funzionare anche se non è ancora disponibile il TTS definitivo.
- Fornisce un'interfaccia per iniettare un backend TTS custom.
- Notifica moduli registrati (callbacks) quando arriva/parte audio.
"""

import os
import threading
import queue
import time
import uuid
import yaml
from datetime import datetime
from loguru import logger

# Percorsi
STATE_FILE = "internal_state.yaml"
MEDIA_DIR = "media/audio"
os.makedirs(MEDIA_DIR, exist_ok=True)

# Tentativi di import opzionali (evitiamo errori se non installati)
try:
    import sounddevice as sd
    import soundfile as sf
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False
    logger.warning("sounddevice/soundfile non disponibili: input/output audio diretti non funzioneranno.")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except Exception:
    PYTTSX3_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False
    logger.warning("speech_recognition non disponibile: funzionalità di recognizer limitate.")

# Classe principale
class SpeechIO:
    def __init__(self, state_file=STATE_FILE):
        logger.info("Inizializzazione SpeechIO...")
        self.state_file = state_file
        self.state = self._load_state()

        # TTS backend: può essere iniettato tramite set_tts_backend(func)
        self._tts_backend = None
        # fallback a pyttsx3 se disponibile
        if PYTTSX3_AVAILABLE:
            self._init_pyttsx3()

        # Callbacks
        self.on_listen_callbacks = []   # chiamate con signature func(text, metadata)
        self.on_speak_callbacks = []    # chiamate con signature func(text, metadata)

        # Ascolto continuo
        self._listening = False
        self._listen_thread = None
        self._listen_queue = queue.Queue()

        # Oggetti opzionali (import lazy per evitare circolarità)
        self._memory_timeline = None
        self._planner = None

        logger.info("SpeechIO pronto.")

    # -------------------------
    # Stato
    # -------------------------
    def _load_state(self):
        try:
            with open(self.state_file, "r") as f:
                st = yaml.safe_load(f) or {}
                return st
        except FileNotFoundError:
            logger.info("Nessun stato precedente trovato per SpeechIO. Creazione stato vuoto.")
            return {}

    def _save_state(self):
        try:
            with open(self.state_file, "w") as f:
                yaml.safe_dump(self.state, f)
        except Exception as e:
            logger.exception(f"Errore salvataggio stato SpeechIO: {e}")

    # -------------------------
    # TTS backend management
    # -------------------------
    def set_tts_backend(self, speak_func):
        """
        Inietta un backend TTS. speak_func deve accettare (text: str, output_path: str|None) e restituire il path del file audio o None.
        """
        self._tts_backend = speak_func
        logger.info("TTS backend impostato tramite injection.")

    def _init_pyttsx3(self):
        try:
            engine = pyttsx3.init()
            # wrapper
            def _pyttsx3_speak(text, output_path=None):
                if output_path:
                    # pyttsx3 può salvare file su alcune piattaforme con driver specifici; fallback: riprodurre direttamente
                    engine.save_to_file(text, output_path)
                    engine.runAndWait()
                    return output_path
                else:
                    engine.say(text)
                    engine.runAndWait()
                    return None
            self._tts_backend = _pyttsx3_speak
            logger.info("Backend pyttsx3 configurato come fallback TTS.")
        except Exception as e:
            logger.warning(f"Impossibile inizializzare pyttsx3: {e}")

    # -------------------------
    # Speak API
    # -------------------------
    def speak(self, text, persist_audio=True):
        """
        Riproduce testo usando il backend TTS se presente.
        - Se non c'è backend, scrive il testo su logger e nello stato (fallback).
        - Restituisce metadata dict.
        """
        timestamp = datetime.now().isoformat()
        uid = str(uuid.uuid4())
        metadata = {"timestamp": timestamp, "id": uid, "text": text}

        audio_path = None
        try:
            if self._tts_backend:
                # genera file audio temporaneo per persistenza
                if persist_audio:
                    filename = f"tts_{timestamp.replace(':','-')}_{uid}.wav"
                    audio_path = os.path.join(MEDIA_DIR, filename)
                    result = self._tts_backend(text, audio_path)
                    # alcuni backend possono restituire path o None
                    if result:
                        audio_path = result
                else:
                    # riproduzione diretta senza salvataggio
                    self._tts_backend(text, None)
            else:
                # fallback: loggare il testo (utente integrerà TTS)
                logger.info(f"[TTS fallback] {text}")

            # Aggiorna stato interno
            self.state.setdefault("speech", {}).setdefault("spoken", []).append({
                "id": uid,
                "text": text,
                "audio_path": audio_path,
                "timestamp": timestamp
            })
            self._save_state()

            # Notifica callback locali
            for cb in self.on_speak_callbacks:
                try:
                    cb(text, metadata)
                except Exception:
                    logger.exception("Errore in on_speak callback.")

            # Notifica MemoryTimeline/Planner se presenti
            self._notify_post_speak(text, metadata)

            return {"ok": True, "audio_path": audio_path, "metadata": metadata}
        except Exception as e:
            logger.exception(f"Errore in speak(): {e}")
            return {"ok": False, "error": str(e)}

    def _notify_post_speak(self, text, metadata):
        # Lazy import MemoryTimeline e Planner per integrazione non-causale
        if not self._memory_timeline:
            try:
                from memory_timeline import MemoryTimeline
                self._memory_timeline = MemoryTimeline()
            except Exception:
                self._memory_timeline = None

        if not self._planner:
            try:
                from planner import Planner
                self._planner = Planner()
            except Exception:
                self._planner = None

        # Aggiungi evento al diario/timeline
        try:
            if self._memory_timeline:
                self._memory_timeline.add_experience(f"Nova ha parlato: {text}", category="speech_out", importance=2)
        except Exception:
            logger.exception("Errore aggiunta esperienza memory_timeline dopo speak.")

        # Notifica planner (ad es. per azioni consequenziali)
        try:
            if self._planner:
                # planner può avere metodo notify_speech_output(text, metadata)
                if hasattr(self._planner, "notify_speech_output"):
                    self._planner.notify_speech_output(text, metadata)
        except Exception:
            logger.exception("Errore notifica planner dopo speak.")

    # -------------------------
    # Listen API
    # -------------------------
    def listen(self, timeout=5, save_audio=True):
        """
        Ascolta microfono per un tempo (timeout) specificato e prova a convertire in testo.
        - Richiede sounddevice/soundfile per catturare audio.
        - Se speech_recognition è disponibile, usa il recognizer per convertire in testo.
        - Restituisce dict {ok: bool, text: str|None, audio_path: str|None, error: str|None}
        """
        timestamp = datetime.now().isoformat()
        uid = str(uuid.uuid4())
        audio_path = None
        try:
            if not SOUND_AVAILABLE:
                raise RuntimeError("sounddevice/soundfile non disponibili per registrazione audio.")

            # Registrazione
            samplerate = 16000
            duration = float(timeout)
            logger.info(f"Inizio registrazione microfono per {duration}s...")
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()

            if save_audio:
                filename = f"mic_{timestamp.replace(':','-')}_{uid}.wav"
                audio_path = os.path.join(MEDIA_DIR, filename)
                sf.write(audio_path, recording, samplerate)
                logger.info(f"Registrazione salvata in {audio_path}")

            # Recognizer opzionale
            text = None
            if SR_AVAILABLE:
                r = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = r.record(source)
                    try:
                        # default: riconoscimento offline/non specifico — l'utente può cambiare provider
                        text = r.recognize_sphinx(audio_data)
                    except Exception:
                        try:
                            text = r.recognize_google(audio_data)
                        except Exception:
                            text = None

            # Aggiorna stato
            self.state.setdefault("speech", {}).setdefault("heard", []).append({
                "id": uid,
                "text": text,
                "audio_path": audio_path,
                "timestamp": timestamp
            })
            self._save_state()

            # Notifica callback
            metadata = {"timestamp": timestamp, "id": uid, "audio_path": audio_path}
            for cb in self.on_listen_callbacks:
                try:
                    cb(text, metadata)
                except Exception:
                    logger.exception("Errore in on_listen callback.")

            # Integrazione memory_timeline/planner
            try:
                if not self._memory_timeline:
                    from memory_timeline import MemoryTimeline
                    self._memory_timeline = MemoryTimeline()
                if self._memory_timeline:
                    content_preview = (text or "<audio non riconosciuto>")[:200]
                    self._memory_timeline.add_experience(f"Input vocale: {content_preview}", category="speech_in", importance=2)
            except Exception:
                logger.exception("Errore integrazione memory_timeline durante listen.")

            # Notifica planner
            try:
                if not self._planner:
                    from planner import Planner
                    self._planner = Planner()
                if self._planner and hasattr(self._planner, "handle_speech_input"):
                    self._planner.handle_speech_input(text, metadata)
            except Exception:
                logger.exception("Errore notifica planner dopo listen.")

            return {"ok": True, "text": text, "audio_path": audio_path, "metadata": metadata}
        except Exception as e:
            logger.exception(f"Errore in listen(): {e}")
            return {"ok": False, "error": str(e), "audio_path": audio_path}

    # -------------------------
    # Continuous listen (threaded)
    # -------------------------
    def _continuous_listen_worker(self, callback, interval=0.5, chunk_seconds=3):
        """
        Worker che registra in loop e invia trascrizioni al callback.
        callback(text, metadata)
        """
        logger.info("Worker ascolto continuo avviato.")
        while self._listening:
            result = self.listen(timeout=chunk_seconds, save_audio=True)
            if result.get("ok"):
                text = result.get("text")
                metadata = result.get("metadata", {})
                # passa al callback registrato dall'esterno
                try:
                    callback(text, metadata)
                except Exception:
                    logger.exception("Errore callback ascolto continuo.")
            time.sleep(interval)
        logger.info("Worker ascolto continuo terminato.")

    def start_continuous_listen(self, callback, chunk_seconds=3, interval=0.5):
        """
        Avvia ascolto continuo in thread separato. callback(text, metadata).
        """
        if self._listening:
            logger.warning("Ascolto continuo già attivo.")
            return False
        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._continuous_listen_worker,
            args=(callback, interval, chunk_seconds),
            daemon=True
        )
        self._listen_thread.start()
        logger.info("Ascolto continuo avviato.")
        return True

    def stop_listening(self):
        """Ferma l'ascolto continuo."""
        if not self._listening:
            return False
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2.0)
        logger.info("Ascolto continuo fermato.")
        return True

    # -------------------------
    # Callbacks management (altri moduli possono registrarsi qui)
    # -------------------------
    def register_on_listen(self, func):
        """Registra callback per input vocale: func(text, metadata)"""
        self.on_listen_callbacks.append(func)
        logger.info(f"Callback on_listen registrata: {func}")

    def register_on_speak(self, func):
        """Registra callback per output vocale: func(text, metadata)"""
        self.on_speak_callbacks.append(func)
        logger.info(f"Callback on_speak registrata: {func}")

    # -------------------------
    # Utility
    # -------------------------
    def ping(self):
        return {"ok": True, "module": "speech_io", "sound_available": SOUND_AVAILABLE, "tts": bool(self._tts_backend)}

# -------------------------
# Esempio d'uso
# -------------------------
if __name__ == "__main__":
    s = SpeechIO()
    # Esempio: fallback speak
    s.speak("Ciao, sono Nova (test TTS fallback).")

    # Esempio: registrare un callback che stampa ciò che è stato ascoltato
    def demo_cb(text, metadata):
        logger.info(f"Callback esterno - ho sentito: {text} | meta: {metadata}")

    s.register_on_listen(demo_cb)

    # Se sounddevice non disponibile, questo fallirà ma il file rimane integrato e sicuro.
    if SOUND_AVAILABLE:
        logger.info("Inizio ascolto di prova (3s)...")
        res = s.listen(timeout=3)
        logger.info(f"Risultato ascolto di prova: {res}")
    else:
        logger.warning("sounddevice non disponibile: test listen saltato.")
