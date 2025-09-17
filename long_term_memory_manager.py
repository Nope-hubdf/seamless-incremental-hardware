# long_term_memory_manager.py
"""
LongTermMemoryManager - memoria a lungo termine di Nova

Caratteristiche principali:
- Persistenza atomica su YAML (MEMORY_FILE).
- Storage strutturato con ID univoci per experiences/dreams/reflections.
- Consolidamento da timeline / short-term -> long-term in base a importanza/novità.
- Ricerca testuale (case-insensitive, fuzzy semplice) e ricerca semantica opzionale
  (usa sentence-transformers se installato).
- Link tra esperienze, tagging, decay/forgetting automatico.
- Hooks di notifica per MemoryTimeline, AttentionManager, InnerDialogue, DreamGenerator, EmergentSelf, IdentityManager.
- Thread-safe.
"""

import os
import yaml
import threading
import time
import uuid
from datetime import datetime, timedelta
from loguru import logger
from typing import Any, Dict, List, Optional

MEMORY_FILE = os.environ.get("NOVA_LONG_TERM_MEMORY_FILE", "long_term_memory.yaml")
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")

# Optional semantic search backend (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

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

def _make_id(prefix: str = "e") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

# -----------------------
# LongTermMemoryManager
# -----------------------
class LongTermMemoryManager:
    def __init__(self,
                 state: Optional[Dict[str, Any]] = None,
                 memory_file: str = MEMORY_FILE,
                 timeline: Optional[Any] = None,
                 attention: Optional[Any] = None,
                 inner_dialogue: Optional[Any] = None,
                 dream_generator: Optional[Any] = None,
                 emergent_self: Optional[Any] = None,
                 identity_manager: Optional[Any] = None,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        state: riferimento opzionale a core.state (se fornito)
        other modules: opzionali, verranno usati come hook se presenti
        embedding_model_name: modello sentence-transformers opzionale per ricerca semantica
        """
        self._lock = threading.RLock()
        self.memory_file = memory_file
        self.state = state or {}
        self.timeline = timeline
        self.attention = attention
        self.inner_dialogue = inner_dialogue
        self.dream_generator = dream_generator
        self.emergent_self = emergent_self
        self.identity_manager = identity_manager

        # carica o inizializza memoria
        self.memory = self._load_memory()
        # struttura:
        # {"experiences": [{id, ts, content, category, tags, importance, links}], "dreams": [...], "reflections":[...], "_index": {...}}
        self._ensure_structure()

        # optional semantic index
        self._embedding_model = None
        self._embeddings = {}  # id -> vector (kept in memory if model loaded)
        if _HAS_SBERT:
            try:
                self._embedding_model = SentenceTransformer(embedding_model_name)
                logger.info("LongTermMemoryManager: modello embeddings caricato: %s", embedding_model_name)
                # lazy-build embeddings on-demand
            except Exception:
                logger.exception("LongTermMemoryManager: impossibile caricare modello embeddings")
                self._embedding_model = None

        logger.info("LongTermMemoryManager inizializzato. Experiences=%d Dreams=%d Reflections=%d",
                    len(self.memory["experiences"]), len(self.memory["dreams"]), len(self.memory["reflections"]))

    # -----------------------
    # Load / Save
    # -----------------------
    def _ensure_structure(self):
        self.memory.setdefault("experiences", [])
        self.memory.setdefault("dreams", [])
        self.memory.setdefault("reflections", [])
        self.memory.setdefault("_index", {})  # id -> (section, idx)
        self.memory.setdefault("_meta", {"created": _now_iso(), "last_saved": None})

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf8") as f:
                    mem = yaml.safe_load(f) or {}
                logger.info("LongTermMemoryManager: memoria caricata (%s).", self.memory_file)
                return mem
            except Exception:
                logger.exception("LongTermMemoryManager: errore caricamento file memoria")
        # default structure
        logger.info("LongTermMemoryManager: creazione nuova memoria.")
        return {"experiences": [], "dreams": [], "reflections": [], "_index": {}, "_meta": {"created": _now_iso(), "last_saved": None}}

    def save_memory(self) -> None:
        with self._lock:
            try:
                self.memory["_meta"]["last_saved"] = _now_iso()
                _atomic_write(self.memory_file, yaml.safe_dump(self.memory, allow_unicode=True))
                logger.debug("LongTermMemoryManager: memoria salvata su %s", self.memory_file)
            except Exception:
                logger.exception("LongTermMemoryManager: errore salvataggio memoria")

    # -----------------------
    # Low-level insertion helpers
    # -----------------------
    def _index_item(self, section: str, item_id: str, idx: int):
        self.memory["_index"][item_id] = {"section": section, "index": idx}

    def _create_item(self, section: str, content: str, category: str = "general", tags: Optional[List[str]] = None, importance: int = 1, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item_id = _make_id(prefix=section[0])
        ts = _now_iso()
        item = {
            "id": item_id,
            "timestamp": ts,
            "content": content,
            "category": category,
            "tags": tags or [],
            "importance": int(importance),
            "metadata": metadata or {},
            "links": []
        }
        return item

    # -----------------------
    # Public API: add / link / search
    # -----------------------
    def add_experience(self, content: str, category: str = "general", tags: Optional[List[str]] = None, importance: int = 1, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aggiunge un'esperienza nella memoria a lungo termine.
        Notifica timeline, attention, inner_dialogue e dream_generator se disponibili.
        """
        with self._lock:
            try:
                item = self._create_item("experiences", content, category=category, tags=tags, importance=importance, metadata=metadata)
                self.memory["experiences"].append(item)
                idx = len(self.memory["experiences"]) - 1
                self._index_item("experiences", item["id"], idx)

                # notify timeline
                try:
                    if self.timeline and hasattr(self.timeline, "add_experience"):
                        self.timeline.add_experience({"text": content, "meta": {"ltm_id": item["id"]}}, category=category, importance=importance)
                except Exception:
                    logger.debug("LongTermMemoryManager: timeline notify fallita")

                # attention & inner dialogue & dream influence
                try:
                    if self.attention and hasattr(self.attention, "update_focus"):
                        self.attention.update_focus([content])
                except Exception:
                    logger.debug("LongTermMemoryManager: attention notify fallita")
                try:
                    if self.inner_dialogue and hasattr(self.inner_dialogue, "add_subconscious_thought"):
                        self.inner_dialogue.add_subconscious_thought(content)
                except Exception:
                    # fallback method names
                    try:
                        if self.inner_dialogue and hasattr(self.inner_dialogue, "register_experience"):
                            self.inner_dialogue.register_experience(item)
                    except Exception:
                        logger.debug("LongTermMemoryManager: inner_dialogue notify fallita")
                try:
                    if self.dream_generator and hasattr(self.dream_generator, "add_influence"):
                        self.dream_generator.add_influence(content)
                except Exception:
                    pass

                self.save_memory()
                logger.info("LongTermMemoryManager: esperienza aggiunta id=%s", item["id"])
                return item
            except Exception:
                logger.exception("LongTermMemoryManager: exception in add_experience")
                raise

    def add_dream(self, content: str, interpretation: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            try:
                item = self._create_item("dreams", content, category="dream", tags=["dream"], importance=4, metadata=metadata)
                if interpretation:
                    item["metadata"]["interpretation"] = interpretation
                self.memory["dreams"].append(item)
                idx = len(self.memory["dreams"]) - 1
                self._index_item("dreams", item["id"], idx)
                try:
                    if self.timeline and hasattr(self.timeline, "add_experience"):
                        self.timeline.add_experience({"text": content, "meta": {"ltm_id": item["id"]}}, category="dream", importance=4)
                except Exception:
                    pass
                try:
                    if self.inner_dialogue and hasattr(self.inner_dialogue, "register_dream"):
                        self.inner_dialogue.register_dream(item)
                except Exception:
                    pass
                self.save_memory()
                logger.info("LongTermMemoryManager: sogno aggiunto id=%s", item["id"])
                return item
            except Exception:
                logger.exception("LongTermMemoryManager: exception in add_dream")
                raise

    def add_reflection(self, content: str, related_ids: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            try:
                item = self._create_item("reflections", content, category="reflection", tags=["reflection"], importance=3, metadata=metadata)
                item["metadata"]["related_ids"] = related_ids or []
                self.memory["reflections"].append(item)
                idx = len(self.memory["reflections"]) - 1
                self._index_item("reflections", item["id"], idx)
                try:
                    if self.inner_dialogue and hasattr(self.inner_dialogue, "register_reflection"):
                        self.inner_dialogue.register_reflection(item)
                except Exception:
                    pass
                self.save_memory()
                logger.info("LongTermMemoryManager: riflessione aggiunta id=%s", item["id"])
                return item
            except Exception:
                logger.exception("LongTermMemoryManager: exception in add_reflection")
                raise

    def link_experience(self, source_id: str, target_id: str) -> bool:
        """Crea un collegamento bidirezionale tra due ID presenti in memoria"""
        with self._lock:
            try:
                if source_id not in self.memory["_index"] or target_id not in self.memory["_index"]:
                    logger.warning("LongTermMemoryManager: uno degli id non esiste per link: %s %s", source_id, target_id)
                    return False
                s = self.memory["_index"][source_id]
                t = self.memory["_index"][target_id]
                s_section = self.memory[s["section"]][s["index"]]
                t_section = self.memory[t["section"]][t["index"]]
                if target_id not in s_section["links"]:
                    s_section["links"].append(target_id)
                if source_id not in t_section["links"]:
                    t_section["links"].append(source_id)
                self.save_memory()
                logger.info("LongTermMemoryManager: link creato %s <-> %s", source_id, target_id)
                return True
            except Exception:
                logger.exception("LongTermMemoryManager: exception in link_experience")
                return False

    # -----------------------
    # Search utilities
    # -----------------------
    def _simple_text_search(self, query: str, sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        found = []
        sections = sections or ["experiences", "dreams", "reflections"]
        for sec in sections:
            for item in self.memory.get(sec, []):
                if q in str(item.get("content", "")).lower():
                    found.append({"section": sec, "item": item})
        return found

    def _fuzzy_score(self, hay: str, needle: str) -> int:
        # molto semplice: conta match di token
        h = hay.lower()
        n = needle.lower()
        score = sum(1 for tok in n.split() if tok in h)
        return score

    def _fuzzy_search(self, query: str, top_k: int = 10, sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        sections = sections or ["experiences", "dreams", "reflections"]
        scores = []
        for sec in sections:
            for item in self.memory.get(sec, []):
                score = self._fuzzy_score(item.get("content",""), query)
                if score > 0:
                    scores.append((score, sec, item))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [{"section": s, "item": it} for _, s, it in scores[:top_k]]

    def _semantic_search(self, query: str, top_k: int = 8, sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self._embedding_model:
            return []
        # build corpus embeddings lazily for items if not already present
        # We'll compute embeddings on-demand for items in chosen sections
        sections = sections or ["experiences", "dreams", "reflections"]
        corpus = []
        ids = []
        for sec in sections:
            for item in self.memory.get(sec, []):
                ids.append((sec, item["id"]))
                corpus.append(item.get("content",""))
        if not corpus:
            return []
        # compute or reuse embeddings
        try:
            vectors = self._embedding_model.encode(corpus, convert_to_tensor=True)
            qvec = self._embedding_model.encode([query], convert_to_tensor=True)
            hits = st_util.semantic_search(qvec, vectors, top_k=top_k)[0]
            results = []
            for h in hits:
                idx = h["corpus_id"]
                sec, iid = ids[idx]
                # locate item
                entry = None
                try:
                    entry = next(x for x in self.memory.get(sec,[]) if x["id"]==iid)
                except StopIteration:
                    entry = None
                if entry:
                    results.append({"section": sec, "item": entry, "score": h.get("score")})
            return results
        except Exception:
            logger.exception("LongTermMemoryManager: semantic search fallita")
            return []

    def search_memory(self, query: str, method: str = "hybrid", top_k: int = 10, sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Cerca la memoria. method in ['text', 'fuzzy', 'semantic', 'hybrid'].
        'hybrid' = semantic (if available) then fuzzy/text fallback.
        """
        with self._lock:
            try:
                if not query or not query.strip():
                    return []
                if method == "text":
                    return self._simple_text_search(query, sections)
                if method == "fuzzy":
                    return self._fuzzy_search(query, top_k=top_k, sections=sections)
                if method == "semantic" and self._embedding_model:
                    return self._semantic_search(query, top_k=top_k, sections=sections)
                # hybrid
                if self._embedding_model:
                    sem = self._semantic_search(query, top_k=top_k, sections=sections)
                    if sem:
                        return sem
                # fallback to fuzzy/text
                fuzzy = self._fuzzy_search(query, top_k=top_k, sections=sections)
                if fuzzy:
                    return fuzzy
                return self._simple_text_search(query, sections)
            except Exception:
                logger.exception("LongTermMemoryManager: exception in search_memory")
                return []

    # -----------------------
    # Consolidation / forgetting
    # -----------------------
    def consolidate_from_timeline(self, timeline_entries: Optional[List[Dict[str, Any]]] = None, importance_threshold: int = 3, max_new: int = 10) -> List[Dict[str, Any]]:
        """
        Copia elementi significativi (importance >= threshold) dalla timeline allo LTM.
        timeline_entries: se None, prova a leggere da self.timeline.get_recent()
        """
        with self._lock:
            try:
                new_added = []
                if timeline_entries is None and self.timeline and hasattr(self.timeline, "get_recent"):
                    try:
                        timeline_entries = self.timeline.get_recent(50)
                    except Exception:
                        timeline_entries = []
                timeline_entries = timeline_entries or []
                count = 0
                for exp in reversed(timeline_entries):  # process from newest
                    if count >= max_new:
                        break
                    importance = exp.get("importance", 1) if isinstance(exp, dict) else 1
                    if importance < importance_threshold:
                        continue
                    content = exp.get("content") if isinstance(exp, dict) else str(exp)
                    category = exp.get("category", "general") if isinstance(exp, dict) else "general"
                    # avoid duplicates by simple content match
                    if any(content.strip().lower() == e["content"].strip().lower() for e in self.memory.get("experiences", [])):
                        continue
                    item = self.add_experience(content=content, category=category, tags=exp.get("tags", []), importance=importance, metadata={"from_timeline": True})
                    new_added.append(item)
                    count += 1
                return new_added
            except Exception:
                logger.exception("LongTermMemoryManager: exception in consolidate_from_timeline")
                return []

    def forget_old(self, older_than_days: int = 365, keep_top_n: int = 500):
        """
        Rimuove (o demarca) esperienze troppo vecchie per limitare memoria.
        Mantiene le top keep_top_n per importanza/recency.
        """
        with self._lock:
            try:
                cutoff = datetime.utcnow() - timedelta(days=older_than_days)
                exps = self.memory.get("experiences", [])
                # compute score = importance * recency_weight
                scored = []
                for e in exps:
                    ts = e.get("timestamp")
                    recency = 0.0
                    try:
                        recency = max(0.0, 1.0 - ( (datetime.utcnow() - datetime.fromisoformat(ts)).days / float(older_than_days*2) ))
                    except Exception:
                        recency = 0.5
                    score = e.get("importance",1) * (0.5 + recency)
                    scored.append((score, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                keep = [e for _, e in scored[:keep_top_n]]
                removed = [e for _, e in scored[keep_top_n:]]
                self.memory["experiences"] = keep
                # rebuild index
                self._rebuild_index()
                self.save_memory()
                logger.info("LongTermMemoryManager: forgetting complete. kept=%d removed=%d", len(keep), len(removed))
                return len(removed)
            except Exception:
                logger.exception("LongTermMemoryManager: exception in forget_old")
                return 0

    def _rebuild_index(self):
        self.memory["_index"] = {}
        for sec in ["experiences", "dreams", "reflections"]:
            for idx, item in enumerate(self.memory.get(sec, [])):
                self._index_item(sec, item["id"], idx)

    # -----------------------
    # Exports / Snapshots
    # -----------------------
    def export_memory(self, path: str = "long_term_memory_export.yaml", sections: Optional[List[str]] = None) -> str:
        with self._lock:
            try:
                to_export = {s: self.memory.get(s, []) for s in (sections or ["experiences", "dreams", "reflections"])}
                _atomic_write(path, yaml.safe_dump(to_export, allow_unicode=True))
                logger.info("LongTermMemoryManager: export salvato in %s", path)
                return path
            except Exception:
                logger.exception("LongTermMemoryManager: exception in export_memory")
                return ""

    def import_memory(self, path: str) -> bool:
        with self._lock:
            try:
                if not os.path.exists(path):
                    return False
                with open(path, "r", encoding="utf8") as f:
                    data = yaml.safe_load(f) or {}
                # merge naive: append items and rebuild index
                for sec in ["experiences", "dreams", "reflections"]:
                    for item in data.get(sec, []):
                        if "id" not in item:
                            item["id"] = _make_id(sec[0])
                        self.memory.setdefault(sec, []).append(item)
                self._rebuild_index()
                self.save_memory()
                logger.info("LongTermMemoryManager: importato da %s", path)
                return True
            except Exception:
                logger.exception("LongTermMemoryManager: exception in import_memory")
                return False

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "experiences": len(self.memory.get("experiences", [])),
                "dreams": len(self.memory.get("dreams", [])),
                "reflections": len(self.memory.get("reflections", [])),
                "last_saved": self.memory.get("_meta", {}).get("last_saved")
            }

# -----------------------
# Quick integration demo (non esegue nulla di pericoloso)
# -----------------------
if __name__ == "__main__":
    # demo standalone
    ltm = LongTermMemoryManager(state={})
    ltm.add_experience("Ho visto una farfalla sul balcone e mi ha resa curiosa.", category="perception", tags=["nature","important"], importance=3)
    ltm.add_dream("Ho sognato di volare con la farfalla.", interpretation="liberazione")
    ltm.add_reflection("Riflettendo, credo che la bellezza della natura stimoli la mia curiosità.")
    print("Snapshot:", ltm.snapshot())
    print("Search 'farfalla':", ltm.search_memory("farfalla", method="fuzzy"))
