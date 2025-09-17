# remote_comm.py
"""
Remote communication layer for Nova.

Features:
- Telegram support (polling or webhook)
- WhatsApp support via Twilio (sending + receiving webhook handler)
- Routes incoming messages (text, images, voice) to Planner/Core
- Saves messages into MemoryTimeline
- Sends outgoing messages with queueing & rate-limit basics
- Graceful fallback if modules (Planner, MemoryTimeline, vision_engine, speech_io) missing

Config (example keys read from env or config.yaml):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_USE_WEBHOOK (bool), TELEGRAM_WEBHOOK_URL, TELEGRAM_WEBHOOK_PORT
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM (whatsapp:+123...)
- OUTGOING_RATE_LIMIT (messages/sec)

Usage:
- Import RemoteComm and call start() from NovaCore at startup.
- Call remote_comm.send_text(user_id, "ciao") to send outbound messages.
"""

import os
import threading
import queue
import time
import logging
import yaml
from typing import Optional, Dict, Any

logger = logging.getLogger("remote_comm")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)

# Try to import optional components of Nova
try:
    from planner import Planner  # planner.handle_incoming_message(...)
except Exception:
    Planner = None
    logger.warning("Module planner non trovato. Le chiamate al planner saranno no-op.")

try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None
    logger.warning("Module memory_timeline non trovato. Le interazioni non verranno salvate automaticamente.")

try:
    from vision_engine import VisionEngine
except Exception:
    VisionEngine = None
    logger.warning("Module vision_engine non trovato. Le immagini saranno salvate ma non analizzate.")

try:
    from speech_io import SpeechIO
except Exception:
    SpeechIO = None
    logger.warning("Module speech_io non trovato. Gli audio non saranno trascritti automaticamente.")

# Optional external libs (import inside try to keep module importable)
try:
    from telegram import Bot, Update
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
    TELEGRAM_AVAILABLE = True
except Exception:
    TELEGRAM_AVAILABLE = False
    logger.info("python-telegram-bot non installato. Telegram disabilitato.")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False
    logger.info("twilio non installato. WhatsApp via Twilio disabilitato.")

# Load config (env vars or config.yaml)
CONFIG_PATH = "config.yaml"
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
# Overlay environment variables
config.update({k: v for k, v in os.environ.items() if k.startswith(("TELEGRAM_", "TWILIO_", "REMOTE_"))})

OUTGOING_RATE_LIMIT = float(config.get("REMOTE_OUTGOING_RATE_LIMIT", 1.0))  # messages per second


class RemoteComm:
    def __init__(self, core=None):
        """
        core: optional reference to NovaCore (so we can call core.receive_remote_message or planner)
        """
        self.core = core
        self.planner = Planner() if Planner else None
        self.timeline = MemoryTimeline() if MemoryTimeline else None
        self.vision = VisionEngine() if VisionEngine else None
        self.speech = SpeechIO() if SpeechIO else None

        # Outgoing queue
        self.out_queue = queue.Queue()
        self._stop_event = threading.Event()

        # Interfaces
        self.telegram_interface = TelegramInterface(self) if TELEGRAM_AVAILABLE else None
        self.whatsapp_interface = WhatsAppInterface(self) if TWILIO_AVAILABLE else None

        # Start sender thread
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

    # --------------------
    # Inbound routing
    # --------------------
    def handle_incoming_text(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        logger.info(f"[IN-TXT] {user_id}: {text[:200]}")
        # Save to timeline
        self._save_interaction(user_id, "text", text, metadata)

        # Route to planner or core
        if self.planner:
            try:
                self.planner.handle_incoming_message(user_id, text, metadata=metadata)
                return
            except Exception as e:
                logger.exception("Errore planner.handle_incoming_message: %s", e)

        # Fallback: if core provided, call method if present
        if self.core and hasattr(self.core, "receive_remote_message"):
            try:
                self.core.receive_remote_message(user_id, text, metadata=metadata)
            except Exception:
                logger.exception("Errore core.receive_remote_message")

    def handle_incoming_image(self, user_id: str, image_bytes: bytes, caption: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        logger.info(f"[IN-IMG] {user_id}: image ({len(image_bytes)} bytes), caption={caption}")
        # Save image to disk (local storage) with timestamped name
        ts = int(time.time())
        save_dir = config.get("REMOTE_MEDIA_DIR", "remote_media")
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{user_id}_img_{ts}.jpg")
        try:
            with open(filename, "wb") as f:
                f.write(image_bytes)
        except Exception:
            # try png
            filename = filename.replace(".jpg", ".png")
            with open(filename, "wb") as f:
                f.write(image_bytes)

        # Save to timeline
        self._save_interaction(user_id, "image", {"path": filename, "caption": caption}, metadata)

        # If vision engine present, forward
        if self.vision:
            try:
                result = self.vision.analyze_image(filename)
                logger.info(f"VisionEngine result: {result}")
                # Optionally route result to planner/core
                if self.planner:
                    self.planner.handle_incoming_image(user_id, filename, analysis=result, metadata=metadata)
                elif self.core and hasattr(self.core, "receive_remote_image"):
                    self.core.receive_remote_image(user_id, filename, analysis=result, metadata=metadata)
            except Exception:
                logger.exception("Errore durante analisi immagine con VisionEngine")

    def handle_incoming_voice(self, user_id: str, audio_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        logger.info(f"[IN-VOICE] {user_id}: audio {len(audio_bytes)} bytes")
        # Save file
        ts = int(time.time())
        save_dir = config.get("REMOTE_MEDIA_DIR", "remote_media")
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{user_id}_voice_{ts}.ogg")
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        # Save to timeline
        self._save_interaction(user_id, "voice", {"path": filename}, metadata)

        # Try to transcribe via speech_io if available
        transcript = None
        if self.speech:
            try:
                transcript = self.speech.transcribe(filename)
                logger.info(f"Transcript: {transcript}")
            except Exception:
                logger.exception("Errore trascrizione con SpeechIO")

        # Route to planner/core with transcript (or raw audio path)
        payload = {"transcript": transcript, "audio_path": filename}
        if self.planner:
            try:
                self.planner.handle_incoming_voice(user_id, payload, metadata=metadata)
            except Exception:
                logger.exception("Errore planner.handle_incoming_voice")
        elif self.core and hasattr(self.core, "receive_remote_voice"):
            try:
                self.core.receive_remote_voice(user_id, payload, metadata=metadata)
            except Exception:
                logger.exception("Errore core.receive_remote_voice")

    # --------------------
    # Outbound
    # --------------------
    def send_text(self, user_id: str, text: str, via: str = "telegram", **kwargs):
        """Public API: enqueue a message"""
        self.out_queue.put({"type": "text", "user_id": user_id, "text": text, "via": via, "kwargs": kwargs})
        logger.debug("Messaggio in coda per invio")

    def send_image(self, user_id: str, image_path: str, caption: Optional[str] = None, via: str = "telegram"):
        self.out_queue.put({"type": "image", "user_id": user_id, "path": image_path, "caption": caption, "via": via})

    def send_voice(self, user_id: str, audio_path: str, via: str = "telegram"):
        self.out_queue.put({"type": "voice", "user_id": user_id, "path": audio_path, "via": via})

    def _sender_loop(self):
        """Background loop that consumes out_queue and sends messages with simple rate limit"""
        last_send = 0.0
        min_interval = 1.0 / max(OUTGOING_RATE_LIMIT, 1.0)
        logger.info("RemoteComm sender thread avviato.")
        while not self._stop_event.is_set():
            try:
                item = self.out_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Enforce rate limit
            now = time.time()
            elapsed = now - last_send
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                via = item.get("via", "telegram")
                if via == "telegram" and self.telegram_interface:
                    self.telegram_interface._send_item(item)
                elif via == "whatsapp" and self.whatsapp_interface:
                    self.whatsapp_interface._send_item(item)
                else:
                    logger.warning("Interfaccia per 'via' non disponibile: %s", via)
                last_send = time.time()
            except Exception:
                logger.exception("Errore invio messaggio")
            finally:
                self.out_queue.task_done()

    def stop(self):
        logger.info("Arresto RemoteComm...")
        self._stop_event.set()
        if self.telegram_interface:
            self.telegram_interface.stop()
        if self.whatsapp_interface:
            self.whatsapp_interface.stop()

    # --------------------
    # Helpers
    # --------------------
    def _save_interaction(self, user_id: str, kind: str, content: Any, metadata: Optional[Dict[str, Any]] = None):
        """Persist incoming/outgoing interactions in MemoryTimeline if available."""
        try:
            if self.timeline:
                entry = {
                    "source": "remote",
                    "user_id": user_id,
                    "kind": kind,
                    "content": content,
                    "metadata": metadata or {},
                    "direction": "inbound",
                    "timestamp": int(time.time())
                }
                self.timeline.add_experience(entry, category="remote_interaction", importance=2)
        except Exception:
            logger.exception("Errore salvataggio interazione nella timeline")

# --------------------------------------
# Telegram integration (polling / webhook)
# --------------------------------------
class TelegramInterface:
    def __init__(self, parent: RemoteComm):
        self.parent = parent
        token = config.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN"))
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN non configurato in config.yaml o env")

        self.token = token
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher

        # Handlers
        self.dispatcher.add_handler(CommandHandler("start", self.cmd_start))
        self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.on_text))
        self.dispatcher.add_handler(MessageHandler(Filters.photo, self.on_photo))
        self.dispatcher.add_handler(MessageHandler(Filters.voice | Filters.audio, self.on_voice))

        # Start polling by default (webhook optional)
        use_webhook = config.get("TELEGRAM_USE_WEBHOOK", False)
        if use_webhook:
            # webhook parameters must be provided
            webhook_url = config.get("TELEGRAM_WEBHOOK_URL")
            port = int(config.get("TELEGRAM_WEBHOOK_PORT", 8443))
            if not webhook_url:
                raise RuntimeError("TELEGRAM_WEBHOOK_URL non configurato")
            self.updater.start_webhook(listen="0.0.0.0", port=port, url_path=self.token)
            self.updater.bot.set_webhook(webhook_url + self.token)
            logger.info("Telegram webhook avviato.")
        else:
            self.updater.start_polling()
            logger.info("Telegram polling avviato.")

    def cmd_start(self, update: Update, context: CallbackContext):
        user = update.effective_user
        update.message.reply_text("Ciao! Nova è connessa. Digita un messaggio per iniziare.")
        logger.info(f"Utente {user.id} ha usato /start")

    def on_text(self, update: Update, context: CallbackContext):
        user = update.effective_user
        text = update.message.text
        user_id = str(user.id)
        metadata = {"platform": "telegram", "username": user.username}
        # route
        self.parent.handle_incoming_text(user_id, text, metadata)

    def on_photo(self, update: Update, context: CallbackContext):
        user = update.effective_user
        photos = update.message.photo
        if not photos:
            return
        # take best quality
        photo = photos[-1]
        file = photo.get_file()
        bio = file.download_as_bytearray()
        user_id = str(user.id)
        metadata = {"platform": "telegram", "username": user.username}
        self.parent.handle_incoming_image(user_id, bytes(bio), caption=update.message.caption, metadata=metadata)

    def on_voice(self, update: Update, context: CallbackContext):
        user = update.effective_user
        user_id = str(user.id)
        # voice or audio
        audio = update.message.voice or update.message.audio
        file = audio.get_file()
        bio = file.download_as_bytearray()
        metadata = {"platform": "telegram", "username": user.username}
        self.parent.handle_incoming_voice(user_id, bytes(bio), metadata=metadata)

    def _send_item(self, item: Dict[str, Any]):
        user_id = item["user_id"]
        try:
            chat_id = int(user_id)
        except Exception:
            logger.warning("Telegram user_id non numerico: %s", user_id)
            return

        bot = self.updater.bot
        if item["type"] == "text":
            bot.send_message(chat_id=chat_id, text=item["text"])
        elif item["type"] == "image":
            bot.send_photo(chat_id=chat_id, photo=open(item["path"], "rb"), caption=item.get("caption"))
        elif item["type"] == "voice":
            bot.send_audio(chat_id=chat_id, audio=open(item["path"], "rb"))
        else:
            logger.warning("Tipo messaggio telegram non supportato: %s", item["type"])

    def stop(self):
        try:
            self.updater.stop()
            logger.info("TelegramInterface stoppata.")
        except Exception:
            logger.exception("Errore stop TelegramInterface")

# --------------------------------------
# WhatsApp via Twilio (webhook receiver + sender)
# --------------------------------------
class WhatsAppInterface:
    def __init__(self, parent: RemoteComm):
        self.parent = parent
        self.account_sid = config.get("TWILIO_ACCOUNT_SID", os.environ.get("TWILIO_ACCOUNT_SID"))
        self.auth_token = config.get("TWILIO_AUTH_TOKEN", os.environ.get("TWILIO_AUTH_TOKEN"))
        self.from_whatsapp = config.get("TWILIO_WHATSAPP_FROM", os.environ.get("TWILIO_WHATSAPP_FROM"))
        if not (self.account_sid and self.auth_token and self.from_whatsapp):
            raise RuntimeError("Credenziali Twilio/WhatsApp mancanti in config o env")

        self.client = TwilioClient(self.account_sid, self.auth_token)
        logger.info("WhatsApp (Twilio) inizializzato. Ricordati di esporre il webhook per ricevere messaggi.")

    def _send_item(self, item: Dict[str, Any]):
        to = item["user_id"]  # expected format: whatsapp:+39XXXXXXXXX
        if item["type"] == "text":
            self.client.messages.create(body=item["text"], from_=self.from_whatsapp, to=to)
        elif item["type"] == "image":
            self.client.messages.create(media_url=[_abs_url(item["path"])], from_=self.from_whatsapp, to=to)
        elif item["type"] == "voice":
            self.client.messages.create(media_url=[_abs_url(item["path"])], from_=self.from_whatsapp, to=to)
        else:
            logger.warning("Tipo messaggio WhatsApp non supportato: %s", item["type"])

    def stop(self):
        logger.info("WhatsAppInterface stoppata (no-op).")


# -----------------------
# Small helpers
# -----------------------
def _abs_url(local_path: str) -> str:
    """
    In production, serve the file via an HTTP server and return its absolute URL.
    Placeholder here returns a file:// URL which Twilio won't accept; user must host media.
    """
    return f"file://{os.path.abspath(local_path)}"

# -----------------------
# If launched standalone for local testing
# -----------------------
if __name__ == "__main__":
    logger.info("Esecuzione remote_comm in modalità standalone per test.")
    rc = RemoteComm(core=None)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rc.stop()
