# context_builder.py
"""
ContextBuilder - costruzione e mantenimento del contesto semantico di Nova

Scopi:
- mappare input (text/image/audio) in concetti, tags e segnali di importanza
- aggiornare timeline / long-term memory / dream generator / attention / emotion
- permettere query semantiche sul contesto e sul long-term memory (se embedding disponibili)
- fornire helper per creare "contesto per il prompt" (finestra contestuale)

Progettato per integrazione difensiva con gli altri moduli del progetto.
"""
import os
import yaml
import threading
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")
DEFAULT_EMBEDDING_MODEL = os.environ.get("NOVA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GEMMA_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")

# optional heavy deps
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except Exception:
    _HAS_LLAMA = False

# try local project imports defensively (if executed within project)
try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None

try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None

try:
    from dream_generator import DreamGenerator
except Exception:
    DreamGenerator = None

try:
    from conscious_loop import ConsciousLoop
except Exception:
    ConsciousLoop = None

try:
    from attention_manager import AttentionManager
except Exception:
    AttentionManager = None

try:
    from emotion_engine import EmotionEngine
except Exception:
    EmotionEngine = None


# -----------------------
# Helpers
# -----------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)


# -----------------------
# Optional utilities
# -----------------------
def _load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    if not _HAS_SBERT:
        logger.debug("ContextBuilder: sentence-transformers non disponibile.")
        return None
    try:
        model = SentenceTransformer(model_name)
        logger.info("ContextBuilder: embedding model caricato: %s", model_name)
        return model
    except Exception:
        logger.exception("ContextBuilder: fallita inizializzazione embedding model")
        return None


def _load_llm(model_path: str = DEFAULT_GEMMA_PATH):
    if not _HAS_LLAMA:
        logger.debug("ContextBuilder: llama_cpp non disponibile per Gemma.")
        return None
    try:
        llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)
        logger.info("ContextBuilder: LLM locale caricato (%s).", model_path)
        def call(prompt: str, max_tokens: int = 120, temperature: float = 0.6) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                # response parsing robusta
                if not resp:
                    return ""
                if isinstance(resp, dict) and resp.get("choices"):
                    return "".join([c.get("text", "") for c in resp.get("choices", [])]).strip()
                if hasattr(resp, "text"):
                    return getattr(resp, "text", "").strip()
                return str(resp).strip()
            except Exception:
                logger.exception("ContextBuilder: errore chiamata LLM")
                return ""
        return call
    except Exception:
        logger.exception("ContextBuilder: impossibile caricare llama_cpp")
        return None


# -----------------------
# ContextBuilder
# -----------------------
class ContextBuilder:
    def __init__(self,
                 state: Optional[Dict[str, Any]] = None,
                 core: Optional[Any] = None,
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 gemma_path: str = DEFAULT_GEMMA_PATH):
        """
        state: passare core.state per condividere lo stesso stato
        core: opzionale, permette wiring con istanze già create (preferred)
        """
        self._lock = threading.RLock()
        # wiring stato
        if core and hasattr(core, "state"):
            self.state = core.state
        elif state is not None:
            self.state = state
        else:
            # carica da file
            self.state = self._load_state()

        # init context structure
        self.state.setdefault("context", {})
        self.state["context"].setdefault("history", [])  # lista di eventi contestuali più recenti
        self.state["context"].setdefault("concept_index", {})  # mapping concept -> occurrences
        self.state["context"].setdefault("last_updated", None)

        # try to wire modules from core if provided
        self.core = core
        self.memory_manager = getattr(core, "long_term_memory_manager", None) or (LongTermMemoryManager(self.state) if LongTermMemoryManager else None)
        self.timeline = getattr(core, "memory_timeline", None) or (MemoryTimeline() if MemoryTimeline else None)
        self.dream_generator = getattr(core, "dream_generator", None) or (DreamGenerator(self.state) if DreamGenerator else None)
        self.conscious_loop = getattr(core, "conscious_loop", None) or (ConsciousLoop(self.state) if ConsciousLoop else None)
        self.attention = getattr(core, "attention_manager", None) or (AttentionManager(self.state) if AttentionManager else None)
        self.emotion = getattr(core, "emotion_engine", None) or (EmotionEngine(self.state) if EmotionEngine else None)

        # optional models
        self._embedding_model = _load_embedding_model(embedding_model_name)
        self._llm = _load_llm(gemma_path)

        # in-memory semantic index (id -> vector) if model present
        self._semantic_index = {}  # maps event_id -> vector (lazy)
        logger.info("ContextBuilder inizializzato. history=%d", len(self.state["context"]["history"]))

    def _load_state(self) -> Dict[str, Any]:
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf8") as f:
                    st = yaml.safe_load(f) or {}
                    return st
        except Exception:
            logger.exception("ContextBuilder: errore caricamento stato da file")
        return {}

    def _save_state(self):
        try:
            self.state["context"]["last_updated"] = _now_iso()
            _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
            logger.debug("ContextBuilder: stato salvato su %s", STATE_FILE)
        except Exception:
            logger.exception("ContextBuilder: errore salvataggio stato")

    # -----------------------
    # Core API
    # -----------------------
    def update_context_from_input(self, input_data: Any, input_type: str = "text", metadata: Optional[Dict[str, Any]] = None):
        """
        Aggiorna il contesto dalla percezione:
        - input_type: 'text'|'image'|'audio'|'vision_object'
        - metadata: dict opzionale (es. sender, image_path, transcription)
        Restituisce il contesto creato come dict.
        """
        with self._lock:
            try:
                event = {
                    "id": f"c_{int(time.time()*1000)}_{random.randint(0,9999)}",
                    "timestamp": _now_iso(),
                    "type": input_type,
                    "raw": input_data,
                    "metadata": metadata or {}
                }

                # Estrai concetti (LLM se possibile, fallback heuristics)
                if input_type == "text":
                    concepts = self._extract_key_concepts_text(input_data)
                    event["summary"] = self._summarize_text_short(input_data)
                elif input_type == "image":
                    concepts = self._extract_key_concepts_image(input_data, metadata)
                    event["summary"] = f"[image] {metadata.get('image_path','image')}" if isinstance(metadata, dict) else "[image]"
                elif input_type == "audio":
                    # audio_data could be raw or transcription in metadata
                    transcript = metadata.get("transcription") if metadata else None
                    if transcript:
                        concepts = self._extract_key_concepts_text(transcript)
                        event["summary"] = transcript[:200]
                    else:
                        concepts = ["audio_event"]
                        event["summary"] = "[audio]"
                else:
                    concepts = ["unknown"]
                    event["summary"] = str(input_data)[:200]

                event["concepts"] = concepts
                # importance heuristic: metadata.importance or infer from keywords
                importance = float(event["metadata"].get("importance", 1.0))
                if any(k in " ".join(concepts).lower() for k in ["importante","critico","emergenza","urgente"]):
                    importance = max(importance, 3.0)
                event["importance"] = importance

                # append to context history (cap to N)
                hist = self.state["context"].setdefault("history", [])
                hist.append(event)
                MAX_H = 300
                if len(hist) > MAX_H:
                    hist[:] = hist[-MAX_H:]

                # update concept_index
                for c in concepts:
                    self.state["context"]["concept_index"].setdefault(c, []).append(event["id"])

                # persist to timeline and long-term memory defensively
                try:
                    if self.timeline and hasattr(self.timeline, "add_experience"):
                        self.timeline.add_experience({"content": event.get("summary") or str(input_data), "importance": int(importance), "category": input_type}, category=input_type, importance=int(max(1, min(5, importance))))
                except Exception:
                    logger.debug("ContextBuilder: timeline add_experience fallback failed")

                try:
                    # try methods of memory_manager: add_experience / add_memory / add_dream
                    if self.memory_manager:
                        if hasattr(self.memory_manager, "add_experience"):
                            self.memory_manager.add_experience(event.get("summary"), category=input_type, tags=concepts, importance=int(max(1, min(5, importance))))
                        elif hasattr(self.memory_manager, "add_memory"):
                            self.memory_manager.add_memory({"type": input_type, "content": event.get("summary"), "concepts": concepts})
                except Exception:
                    logger.debug("ContextBuilder: memory_manager add fallback failed")

                # notify other modules
                try:
                    if self.dream_generator and hasattr(self.dream_generator, "process_new_input"):
                        self.dream_generator.process_new_input(input_data, input_type)
                except Exception:
                    pass
                try:
                    if self.conscious_loop and hasattr(self.conscious_loop, "update_from_context"):
                        self.conscious_loop.update_from_context(self.state)
                except Exception:
                    pass
                try:
                    if self.attention and hasattr(self.attention, "update_focus"):
                        self.attention.update_focus([event.get("summary")])
                except Exception:
                    pass
                try:
                    if self.emotion and hasattr(self.emotion, "process_experience"):
                        self.emotion.process_experience({"content": event.get("summary"), "type": input_type, "importance": importance})
                except Exception:
                    pass

                # semantic indexing (lazy) if embedding model available
                if self._embedding_model:
                    try:
                        vec = self._embedding_model.encode([event.get("summary")], convert_to_tensor=True)[0]
                        self._semantic_index[event["id"]] = vec
                    except Exception:
                        logger.debug("ContextBuilder: embedding compute failed for event %s", event["id"])

                # persist state
                self._save_state()
                logger.info("ContextBuilder: context event aggiunto id=%s type=%s concepts=%s", event["id"], input_type, concepts[:5])
                return event
            except Exception:
                logger.exception("ContextBuilder: exception in update_context_from_input")
                return {}

    # -----------------------
    # concept extraction helpers
    # -----------------------
    def _extract_key_concepts_text(self, text: str, top_k: int = 6) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        # prefer LLM if available for richer concept extraction
        try:
            if self._llm:
                prompt = (
                    "Estrai le principali parole chiave e concetti (max {k}) dal seguente testo in italiano. "
                    "Ritorna una lista separata da virgole, solo parole/short phrases, niente spiegazioni.\n\n"
                    f"Testo:\n{text}\n\nRisposta:"
                ).format(k=top_k)
                out = self._llm(prompt, max_tokens=120, temperature=0.0) or ""
                # parse comma separated or lines
                concepts = []
                for part in out.replace("\n", ",").split(","):
                    p = part.strip()
                    if p:
                        concepts.append(p)
                return list(dict.fromkeys(concepts))[:top_k]
        except Exception:
            logger.debug("ContextBuilder: LLM extraction failed, falling back.")

        # fallback: simple TF-like heuristic: most frequent non-stopwords
        tokens = [t.strip(".,;:!?()[]\"'").lower() for t in text.split()]
        # minimal stopwords italian
        stop = {"e","di","a","da","che","il","la","lo","le","un","una","con","per","in","su","del","della","dei","degli","mi","ti","si"}
        freq = {}
        for tok in tokens:
            if not tok or tok in stop or len(tok) < 3:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        # sort by freq then length
        sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
        concepts = [t for t,_ in sorted_tokens][:top_k]
        # also include some 2-gram heuristics
        if len(concepts) < top_k:
            bigrams = []
            for i in range(len(tokens)-1):
                a, b = tokens[i], tokens[i+1]
                if a not in stop and b not in stop:
                    bigrams.append(f"{a} {b}")
            # freq bigrams
            bg_freq = {}
            for bg in bigrams:
                bg_freq[bg] = bg_freq.get(bg, 0) + 1
            for bg,_ in sorted(bg_freq.items(), key=lambda x:-x[1]):
                if bg not in concepts:
                    concepts.append(bg)
                if len(concepts) >= top_k:
                    break
        return concepts[:top_k]

    def _extract_key_concepts_image(self, image_data: Any, metadata: Optional[Dict[str,Any]] = None) -> List[str]:
        """
        Stub: se hai un vision_engine integra qui.
        metadata può contenere 'image_path' o 'vision_result'
        """
        # try to use vision_engine from core if present
        try:
            vision = getattr(self.core, "vision_engine", None)
            if vision and hasattr(vision, "analyze_image"):
                res = vision.analyze_image(metadata.get("image_path") if metadata else image_data)
                # expected res to be dict with labels/tags
                if isinstance(res, dict):
                    tags = res.get("labels") or res.get("concepts") or []
                    return [str(t) for t in tags][:8]
        except Exception:
            logger.debug("ContextBuilder: vision_engine analyze failed")

        # fallback simple tag
        if metadata and metadata.get("vision_result"):
            return metadata.get("vision_result")[:8]
        return ["immagine","visione","oggetto"]

    def _summarize_text_short(self, text: str, max_len: int = 180) -> str:
        if not text:
            return ""
        # use llm if available
        if self._llm:
            try:
                prompt = f"Fornisci un breve riassunto in italiano (una frase) del seguente testo:\n\n{text}\n\nRiassunto:"
                out = self._llm(prompt, max_tokens=60, temperature=0.2) or ""
                s = out.strip().split("\n")[0]
                if s:
                    return s[:max_len]
            except Exception:
                logger.debug("ContextBuilder: LLM summary failed")
        # fallback: truncate and clean
        clean = " ".join(text.strip().split())
        return (clean[:max_len] + "...") if len(clean) > max_len else clean

    # -----------------------
    # Context utilities
    # -----------------------
    def get_context_window(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Ritorna gli ultimi N eventi contestuali"""
        with self._lock:
            return list(self.state.get("context", {}).get("history", []))[-last_n:]

    def summarize_context(self, last_n: int = 10) -> str:
        """Genera un breve sommario del contesto recente (LLM o heuristica)"""
        events = self.get_context_window(last_n)
        if not events:
            return "Nessun contesto recente."
        aggregated = "\n".join([f"- {e.get('summary') or str(e.get('raw'))}" for e in events])
        if self._llm:
            try:
                prompt = f"Riassumi in italiano i seguenti eventi in 2-3 frasi:\n\n{aggregated}\n\nRisposta:"
                return self._llm(prompt, max_tokens=120, temperature=0.3) or aggregated
            except Exception:
                logger.debug("ContextBuilder: LLM summarize_context fallita")
        # fallback: join first 3 summaries
        first = [e.get("summary") for e in events[:3] if e.get("summary")]
        return " | ".join(first) if first else aggregated[:300]

    def query_semantic_memory(self, query: str, top_k: int = 6, method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Cerca nel long-term memory e nella context history.
        method: 'text'|'fuzzy'|'semantic'|'hybrid'. 'hybrid' usa semantic se disponibile.
        Restituisce lista di risultati (section,item,score?)
        """
        with self._lock:
            results = []
            # first try LTM if available
            try:
                if self.memory_manager and hasattr(self.memory_manager, "search_memory"):
                    # preferire metodo semantico se disponibile
                    if method in ("semantic", "hybrid") and self._embedding_model:
                        res = self.memory_manager.search_memory(query, method="semantic", top_k=top_k)
                        if res:
                            return res
                    # fallback to search_memory hybrid/fuzzy
                    res = self.memory_manager.search_memory(query, method="hybrid", top_k=top_k)
                    if res:
                        return res
            except Exception:
                logger.debug("ContextBuilder: memory_manager search failed")

            # fallback to local fuzzy search on context history
            hist = self.state.get("context", {}).get("history", [])
            q = query.lower().strip()
            scored = []
            for ev in hist:
                txt = (ev.get("summary") or str(ev.get("raw"))).lower()
                score = sum(1 for tok in q.split() if tok and tok in txt)
                if score > 0:
                    scored.append((score, ev))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [{"section": "context_history", "item": ev} for _, ev in scored[:top_k]]
            return results

    def clear_context(self):
        """Svuota la history contestuale (non cancella LTM)"""
        with self._lock:
            self.state["context"]["history"] = []
            self.state["context"]["concept_index"] = {}
            self._semantic_index = {}
            self._save_state()
            logger.info("ContextBuilder: context cleared")

    # -----------------------
    # Debug / snapshots
    # -----------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "history_count": len(self.state.get("context", {}).get("history", [])),
                "concepts_count": len(self.state.get("context", {}).get("concept_index", {})),
                "last_updated": self.state.get("context", {}).get("last_updated")
            }


# Quick demo if run as script
if __name__ == "__main__":
    st = {}
    cb = ContextBuilder(state=st, core=None)
    ev = cb.update_context_from_input("Oggi ho visto un bellissimo tramonto sulla spiaggia e mi ha emozionata.", "text", metadata={"source":"user", "importance":2})
    print("Event:", ev)
    print("Window:", cb.get_context_window(5))
    print("Summary:", cb.summarize_context(5))
