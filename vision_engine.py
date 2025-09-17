# vision_engine.py
"""
VisionEngine per Nova
- Percezione visiva: analisi immagini e stream, riconoscimento oggetti e scene
- Integrazione con memory_timeline, attention_manager, dream_generator, internal_state.yaml
- Modalità "light" (OpenCV/Pillow) e "advanced" (CLIP/transformers embeddings) se disponibili
"""

from __future__ import annotations
import os
import io
import time
import yaml
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

from loguru import logger

# Optional / soft imports
try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
try:
    import numpy as np
except Exception:
    np = None

# Advanced optional imports (CLIP / embeddings)
_ADV_CLIP_AVAILABLE = False
try:
    # Examples: open_clip, sentence_transformers, transformers
    import torch
    from sentence_transformers import SentenceTransformer, util as st_util  # optional
    _ADV_CLIP_AVAILABLE = True
except Exception:
    _ADV_CLIP_AVAILABLE = False

# Paths
STATE_FILE = "internal_state.yaml"
VISION_CACHE_DIR = "nova_vision_cache"
os.makedirs(VISION_CACHE_DIR, exist_ok=True)


# --- Helper: safe imports of project modules (graceful fallback) ---
def try_import(module_name: str):
    try:
        mod = __import__(module_name, fromlist=["*"])
        logger.debug(f"Imported module {module_name}")
        return mod
    except Exception as e:
        logger.warning(f"Modulo opzionale '{module_name}' non trovato: {e}")
        return None


MemoryTimelineMod = try_import("memory_timeline")
AttentionManagerMod = try_import("attention_manager")
DreamGeneratorMod = try_import("dream_generator")
ConsciousLoopMod = try_import("conscious_loop")


# --- VisionEngine class ---
class VisionEngine:
    def __init__(self, state_file: str = STATE_FILE, mode: str = "auto"):
        """
        mode: "auto" | "light" | "advanced"
        - auto: usa advanced se disponibile, altrimenti light
        - light: OpenCV/PIL basic processing
        - advanced: embeddings/CLIP if available
        """
        self.state_file = state_file
        self.mode = mode
        if self.mode == "auto":
            self.mode = "advanced" if _ADV_CLIP_AVAILABLE else "light"
        logger.info(f"VisionEngine inizializzato in modalità: {self.mode}")

        # Optional advanced model
        self.adv_model = None
        if self.mode == "advanced" and _ADV_CLIP_AVAILABLE:
            try:
                # Carica modello leggero per embeddings (se installato)
                self.adv_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Modello avanzato embeddings caricato (SentenceTransformer).")
            except Exception as e:
                logger.warning(f"Non è stato possibile caricare il modello avanzato: {e}")
                self.adv_model = None
                self.mode = "light"

        # Components from other modules (if available)
        self.memory_timeline = None
        if MemoryTimelineMod:
            try:
                self.memory_timeline = MemoryTimelineMod.MemoryTimeline()
            except Exception as e:
                logger.warning(f"Errore inizializzazione MemoryTimeline: {e}")

        self.attention_manager = None
        if AttentionManagerMod:
            try:
                # se AttentionManager ha costruttore
                self.attention_manager = getattr(AttentionManagerMod, "AttentionManager", None)
                if callable(self.attention_manager):
                    self.attention_manager = self.attention_manager()
                else:
                    self.attention_manager = None
            except Exception as e:
                logger.warning(f"Errore inizializzazione AttentionManager: {e}")
                self.attention_manager = None

        self.dream_generator = None
        if DreamGeneratorMod:
            try:
                self.dream_generator = getattr(DreamGeneratorMod, "DreamGenerator", None)
                if callable(self.dream_generator):
                    self.dream_generator = self.dream_generator(self._load_state())
                else:
                    self.dream_generator = None
            except Exception as e:
                logger.warning(f"Errore inizializzazione DreamGenerator: {e}")

        # Camera stream thread
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_running = False

    # --- State helpers ---
    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(self.state_file, "r") as f:
                st = yaml.safe_load(f) or {}
                return st
        except FileNotFoundError:
            logger.warning("File stato interno non trovato. Restituisco stato vuoto.")
            return {}

    def _save_state(self, state: Dict[str, Any]):
        try:
            with open(self.state_file, "w") as f:
                yaml.safe_dump(state, f)
            logger.debug("VisionEngine ha aggiornato lo stato interno.")
        except Exception as e:
            logger.exception(f"Errore salvataggio stato interno: {e}")

    # --- Core API ---
    def analyze_image_bytes(self, image_bytes: bytes, source: str = "upload") -> Dict[str, Any]:
        """Analizza un'immagine data come bytes. Restituisce una descrizione strutturata."""
        try:
            if Image is None:
                raise RuntimeError("Pillow non installato; impossibile elaborare immagini.")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self.analyze_pil_image(image, source=source)
        except Exception as e:
            logger.exception(f"analyze_image_bytes error: {e}")
            return {"ok": False, "error": str(e)}

    def analyze_pil_image(self, pil_img: "Image.Image", source: str = "upload") -> Dict[str, Any]:
        """Analizza un oggetto PIL.Image: rilevamento oggetti base, caption semplificata e integrazione memoria."""
        timestamp = datetime.utcnow().isoformat()
        meta = {"timestamp": timestamp, "source": source}
        logger.info(f"Analisi immagine ricevuta da {source} - {timestamp}")

        # Basic properties
        width, height = pil_img.size
        meta.update({"width": width, "height": height})

        # Save original snapshot
        save_path = os.path.join(VISION_CACHE_DIR, f"img_{int(time.time())}.jpg")
        try:
            pil_img.save(save_path, "JPEG", quality=85)
            meta["saved_path"] = save_path
            logger.debug(f"Snapshot salvato in {save_path}")
        except Exception as e:
            logger.warning(f"Impossibile salvare snapshot: {e}")

        # Light analysis: resize, basic color / edges, simple object heuristics
        analysis: Dict[str, Any] = {"mode": self.mode}
        try:
            if self.mode == "light" or self.adv_model is None:
                # Convert to OpenCV if available
                objects = []
                caption = self._light_caption(pil_img)
                edges = None
                if cv2 is not None and np is not None:
                    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    # Very light heuristic: detect faces with builtin cascade if available
                    face_count = 0
                    try:
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        face_count = len(faces)
                        if face_count > 0:
                            objects.append({"label": "person_like", "count": face_count, "bbox_samples": faces.tolist()})
                    except Exception:
                        pass
                analysis.update({"caption": caption, "objects": objects})
            else:
                # Advanced route: use embeddings & semantic search for caption-like description
                analysis.update(self._advanced_image_analysis(pil_img))
        except Exception as e:
            logger.exception(f"Errore durante analisi immagine: {e}")
            analysis["error"] = str(e)

        meta["analysis"] = analysis

        # Integrations: memory timeline, attention, dreams
        try:
            human_readable = f"Vision: {analysis.get('caption', '')} (w:{width} h:{height})"
            self._record_experience(human_readable, category="vision", importance=analysis.get("importance", 1))
            # Notify attention manager
            if self.attention_manager:
                try:
                    if hasattr(self.attention_manager, "on_new_percept"):
                        self.attention_manager.on_new_percept(analysis)
                    elif hasattr(self.attention_manager, "prioritize"):
                        self.attention_manager.prioritize({"type": "vision", "analysis": analysis})
                except Exception as e:
                    logger.debug(f"Attenzione: errore notifying attention_manager: {e}")
            # Notify dream generator
            if self.dream_generator and hasattr(self.dream_generator, "ingest_percept"):
                try:
                    self.dream_generator.ingest_percept(analysis)
                except Exception as e:
                    logger.debug(f"Errore notifying dream_generator: {e}")
        except Exception as e:
            logger.exception(f"Errore integrazione vision -> timeline/attention/dreams: {e}")

        return {"ok": True, "meta": meta}

    def analyze_file_path(self, path: str, source: str = "file") -> Dict[str, Any]:
        """Analizza immagine dal filesystem."""
        try:
            if Image is None:
                raise RuntimeError("Pillow non installato; impossibile elaborare immagini.")
            img = Image.open(path).convert("RGB")
            return self.analyze_pil_image(img, source=source)
        except Exception as e:
            logger.exception(f"analyze_file_path error: {e}")
            return {"ok": False, "error": str(e)}

    def analyze_stream(self, camera_index: int = 0, fps: int = 3, run_seconds: Optional[int] = None):
        """
        Avvia l'analisi stream da camera. Funziona in un thread separato.
        - camera_index: id camera
        - fps: frame per second processing (non capture fps)
        - run_seconds: durata totale (None = infinito)
        """
        if cv2 is None:
            raise RuntimeError("OpenCV non disponibile: installare opencv-python per usare stream camera.")
        if self._stream_running:
            logger.warning("Stream camera già in esecuzione.")
            return

        def _stream_loop():
            logger.info("Apertura stream camera...")
            cap = cv2.VideoCapture(camera_index)
            last_time = 0
            start_time = time.time()
            while cap.isOpened() and self._stream_running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                now = time.time()
                if now - last_time < 1.0 / max(1, fps):
                    # skip to limit processing rate
                    time.sleep(0.01)
                    continue
                last_time = now
                # Convert BGR to RGB PIL for analysis
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    res = self.analyze_pil_image(pil_img, source="camera_stream")
                    # Optionally save annotated frame
                    annotated = self._annotate_frame(frame, res)
                    timestamp = int(time.time())
                    out_path = os.path.join(VISION_CACHE_DIR, f"stream_{timestamp}.jpg")
                    cv2.imwrite(out_path, annotated)
                except Exception as e:
                    logger.debug(f"Errore process frame: {e}")

                if run_seconds and (time.time() - start_time) > run_seconds:
                    break

            cap.release()
            logger.info("Stream camera terminato.")

        self._stream_running = True
        self._stream_thread = threading.Thread(target=_stream_loop, daemon=True)
        self._stream_thread.start()
        return self._stream_thread

    def stop_stream(self):
        self._stream_running = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2)
        logger.info("Stream camera arrestato.")

    # --- Internal helpers ---
    def _light_caption(self, pil_img: "Image.Image") -> str:
        """Genera una caption semplificata usando statistiche dell'immagine (fallback)."""
        try:
            # Basic heuristics: dominant color, edges, brightness
            arr = np.array(pil_img).reshape(-1, 3)
            dominant = tuple(map(int, np.mean(arr, axis=0)))
            avg_brightness = float(np.mean(arr))
            caption = f"Immagine con colore dominante RGB{dominant}, luminosità media {avg_brightness:.1f}"
            # Quick object heuristics via contours (if opencv available)
            if cv2 is not None:
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_cnt = [c for c in contours if cv2.contourArea(c) > 5000]
                if len(large_cnt) > 0:
                    caption += f", {len(large_cnt)} oggetto/i principali rilevati"
            return caption
        except Exception as e:
            logger.debug(f"_light_caption failed: {e}")
            return "Immagine non descrivibile (fallback)."

    def _advanced_image_analysis(self, pil_img: "Image.Image") -> Dict[str, Any]:
        """Analisi avanzata con embeddings / semantic search se modello avanzato disponibile."""
        if not self.adv_model:
            return {"caption": "Advanced model non disponibile", "objects": []}
        try:
            # Convert to bytes and call embedding model (example using sentence-transformers)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            # NOTE: sentence-transformers doesn't embed images by default; this is an example.
            # In production, install and use a proper image-text model (e.g., CLIP).
            text_candidates = ["a person", "a group", "an animal", "a landscape", "an indoor scene", "a vehicle"]
            # Dummy approach: embed candidates and return highest similarity to a placeholder "image embedding"
            # This is just scaffold logic — replace with a real image model for production.
            text_emb = self.adv_model.encode(text_candidates, convert_to_tensor=True)
            # Fake image embedding: encode the filename or timestamp as proxy
            img_emb = self.adv_model.encode([str(time.time())], convert_to_tensor=True)
            sims = st_util.cos_sim(img_emb, text_emb)[0].cpu().tolist()
            best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
            caption = f"Probabile scena: {text_candidates[best_idx]} (sim score {sims[best_idx]:.3f})"
            return {"caption": caption, "objects": []}
        except Exception as e:
            logger.debug(f"_advanced_image_analysis failed: {e}")
            return {"caption": "Analisi avanzata fallita", "objects": [], "error": str(e)}

    def _record_experience(self, text: str, category: str = "vision", importance: int = 2):
        """Aggiunge un record alla timeline di memoria (se disponibile) e aggiorna lo stato."""
        try:
            if self.memory_timeline:
                try:
                    self.memory_timeline.add_experience(text, category=category, importance=importance)
                except Exception as e:
                    logger.debug(f"Errore aggiunta experience a MemoryTimeline: {e}")
            # Aggiorna file di stato con ultimo_percept
            state = self._load_state()
            if "last_percepts" not in state:
                state["last_percepts"] = []
            state["last_percepts"].append({"time": datetime.utcnow().isoformat(), "text": text, "category": category})
            # Trim to reasonable size
            if len(state["last_percepts"]) > 200:
                state["last_percepts"] = state["last_percepts"][-200:]
            self._save_state(state)
        except Exception as e:
            logger.exception(f"_record_experience failed: {e}")

    def _annotate_frame(self, frame_bgr, analysis: Dict[str, Any]):
        """Annota il frame mettendo testo semplice della caption (OpenCV)."""
        try:
            if cv2 is None:
                return frame_bgr
            text = analysis.get("caption", "vision")
            annotated = frame_bgr.copy()
            cv2.putText(annotated, text[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return annotated
        except Exception as e:
            logger.debug(f"_annotate_frame failed: {e}")
            return frame_bgr

    # Convenience helper for external modules (e.g., Telegram handler)
    def handle_incoming_image_bytes(self, image_bytes: bytes, metadata: Optional[Dict[str, Any]] = None):
        """API semplice per ricevere immagini da bot o UI e processarle."""
        try:
            res = self.analyze_image_bytes(image_bytes, source=(metadata.get("source") if metadata else "external"))
            return res
        except Exception as e:
            logger.exception(f"handle_incoming_image_bytes failed: {e}")
            return {"ok": False, "error": str(e)}


# --- Simple CLI/testing harness ---
if __name__ == "__main__":
    logger.info("Esecuzione test VisionEngine (CLI).")
    ve = VisionEngine(mode="auto")
    test_img_path = None
    # trova un'immagine nella cartella corrente
    for root, _, files in os.walk("."):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                test_img_path = os.path.join(root, f)
                break
        if test_img_path:
            break
    if test_img_path:
        logger.info(f"Analizzo immagine di test: {test_img_path}")
        out = ve.analyze_file_path(test_img_path, source="cli_test")
        logger.info(f"Risultato: {out}")
    else:
        logger.info("Nessuna immagine trovata per test. Avvio stream camera per 10s (se disponibile).")
        try:
            t = ve.analyze_stream(camera_index=0, fps=1, run_seconds=10)
            time.sleep(11)
            ve.stop_stream()
        except Exception as e:
            logger.warning(f"Test stream non riuscito: {e}")
