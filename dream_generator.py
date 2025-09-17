# dream_generator.py
"""
DreamGenerator avanzato (LLM + simboli + retrieval)
- Usa modello locale GGUF (Gemma) se disponibile (llama_cpp), fallback euristico altrimenti.
- Mantiene un SymbolManager per scoprire e rafforzare simboli onirici ricorrenti.
- Usa un SimpleSemanticIndex (TF-style) per retrieval dei ricordi più pertinenti.
- Integra EthicsEngine (sanitizzazione) e notifica ConsciousLoop (on_perception / register_dream).
- Salvataggio atomico dello stato e persistenza nello timeline.
"""

import os
import random
import time
import threading
import yaml
import traceback
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

DEFAULT_MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")
STATE_FILE = os.environ.get("NOVA_STATE_FILE", "internal_state.yaml")

# -----------------------
# LLM loader (llama_cpp) con fallback
# -----------------------
def _load_llm(model_path: str, n_ctx: int = 2048, n_threads: int = 4):
    try:
        from llama_cpp import Llama
        logger.info("llama_cpp trovato: caricamento modello %s", model_path)
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads)
        def _call(prompt: str, max_tokens: int = 300, temperature: float = 0.9) -> str:
            try:
                resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
                if not resp:
                    return ""
                # robust extraction
                text = ""
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices and isinstance(choices, list):
                        text = "".join([c.get("text","") for c in choices])
                elif hasattr(resp, "text"):
                    text = getattr(resp, "text")
                return (text or "").strip()
            except Exception:
                logger.exception("Errore chiamata LLM in DreamGenerator")
                return ""
        return _call
    except Exception as e:
        logger.warning("llama_cpp non disponibile o modello non caricato: %s. Uso fallback.", e)
        def _fallback(prompt: str, max_tokens: int = 120, temperature: float = 0.7) -> str:
            # Basic creative fallback using random templates and context extraction
            seed = ""
            try:
                last = [l for l in prompt.splitlines() if l.strip()][-1]
                seed = last[:120]
            except Exception:
                seed = ""
            templates = [
                "Fluttico in correnti di colore dove i ricordi si trasformano in note. " + seed,
                "Un giardino parla: ogni fiore racconta una memoria. " + seed,
                "Cammino tra libri volanti: ogni titolo è un frammento di me. " + seed
            ]
            return random.choice(templates)
        return _fallback

# -----------------------
# Atomic write helper
# -----------------------
def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

# -----------------------
# SimpleSemanticIndex (pure python, TF-style)
# -----------------------
class SimpleSemanticIndex:
    def __init__(self):
        # store documents as list of (id, text)
        self.docs: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.doc_vectors: List[Dict[int, float]] = []  # sparse vectors: term_index -> tf
        self.lock = threading.RLock()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # simple italian-friendly tokenization, lowercase, remove punctuation
        text = text.lower()
        tokens = re.findall(r"\b[\wàèéìòù]+\b", text)
        return [t for t in tokens if len(t) > 1]

    def add(self, doc_id: str, text: str):
        with self.lock:
            tokens = self._tokenize(text)
            tf: Dict[int, float] = {}
            for t in tokens:
                idx = self.vocab.setdefault(t, len(self.vocab))
                tf[idx] = tf.get(idx, 0.0) + 1.0
            # normalize
            norm = math.sqrt(sum(v*v for v in tf.values())) or 1.0
            for k in list(tf.keys()):
                tf[k] = tf[k] / norm
            self.docs.append((doc_id, text))
            self.doc_vectors.append(tf)

    def clear_and_build(self, items: List[Tuple[str, str]]):
        with self.lock:
            self.docs = []
            self.vocab = {}
            self.doc_vectors = []
            for did, txt in items:
                self.add(did, txt)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        with self.lock:
            if not self.docs:
                return []
            qtokens = self._tokenize(text)
            qvec: Dict[int, float] = {}
            for t in qtokens:
                if t in self.vocab:
                    idx = self.vocab[t]
                    qvec[idx] = qvec.get(idx, 0.0) + 1.0
            norm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
            for k in list(qvec.keys()):
                qvec[k] = qvec[k] / norm
            scores = []
            for (did, txt), dvec in zip(self.docs, self.doc_vectors):
                # cosine similarity sparse
                s = 0.0
                for idx, val in qvec.items():
                    s += val * dvec.get(idx, 0.0)
                scores.append((did, txt, s))
            scores.sort(key=lambda x: x[2], reverse=True)
            # filter zeros
            return [(d,t,round(score,4)) for d,t,score in scores[:top_k] if score > 0.0]

# -----------------------
# SymbolManager: scopre e rafforza simboli onirici
# -----------------------
class SymbolManager:
    """
    Mantiene simboli ricorrenti mappandoli a esempi/testi e strength.
    Strategie:
    - estrazione semplice basata su sostantivi/lemmi (heuristic)
    - co-occurrence with emotion labels increases association
    - ogni sogno o memoria aggiorna i conteggi; simboli con peso alto vengono inseriti nel prompt
    Persistenza semplice nello state['dream_symbols'].
    """
    STOPWORDS = set([
        "il","la","lo","i","gli","le","e","di","a","da","per","in","con","su","un","una","non","che","come","si","mi","ti","se","anche"
    ])

    def __init__(self, state: Dict[str, Any]):
        self.state = state
        with threading.RLock():
            self.symbols: Dict[str, Dict[str, Any]] = {}
            persisted = self.state.get("dream_symbols", {}) or {}
            # load persisted
            for k,v in persisted.items():
                self.symbols[k] = dict(v)

    @staticmethod
    def _candidate_tokens(text: str) -> List[str]:
        tokens = re.findall(r"\b[\wàèéìòù]+\b", text.lower())
        # heuristics: prefer nouns/adjectives by length and not stopwords
        cands = [t for t in tokens if len(t) > 2]
        return [t for t in cands if t not in SymbolManager.STOPWORDS]

    def update_from_texts(self, texts: List[str], emotions: Optional[Dict[str,float]] = None):
        """
        Aggiorna il conteggio dei simboli sulla base di una lista di testi (sogni/ricordi).
        Aggiunge esempi e aumenta 'strength'. Emotions optionally bias some tokens.
        """
        now = datetime.utcnow().isoformat()
        emotions = emotions or {}
        for txt in texts:
            toks = self._candidate_tokens(txt)
            # also consider bigrams
            bigrams = [" ".join(pair) for pair in zip(toks, toks[1:])] if len(toks) > 1 else []
            candidates = toks + bigrams
            for c in candidates:
                if len(c) < 3: continue
                entry = self.symbols.setdefault(c, {"count":0,"examples":[],"last_seen":None,"strength":0.0})
                entry["count"] += 1
                entry["examples"].append({"ts": now, "text": txt[:240]})
                entry["last_seen"] = now
                # update strength simple: base log(count) plus small emotion modifier
                emo_bonus = (sum(emotions.values())/len(emotions)) if emotions else 0.0
                entry["strength"] = min(100.0, float(math.log(1+entry["count"]) * 10.0 + emo_bonus))
        # persist
        self.state["dream_symbols"] = self.symbols

    def top_symbols(self, limit: int = 6) -> List[Tuple[str, float]]:
        sorted_syms = sorted(self.symbols.items(), key=lambda kv: kv[1].get("strength",0.0), reverse=True)
        return [(k, v.get("strength",0.0)) for k,v in sorted_syms[:limit]]

    def get_examples(self, symbol: str, limit: int = 3) -> List[str]:
        e = self.symbols.get(symbol, {}).get("examples", [])[:limit]
        return [x.get("text","") for x in e]

# -----------------------
# DreamGenerator class (avanzata)
# -----------------------
class DreamGenerator:
    def __init__(self, state: Dict[str, Any], core: Optional[Any] = None, model_path: str = DEFAULT_MODEL_PATH):
        self._lock = threading.RLock()
        self.state = state if isinstance(state, dict) else {}
        self.core = core
        self.llm = _load_llm(model_path)
        # initialize collaborators if present in core or import fallback
        try:
            from memory_timeline import MemoryTimeline
            self.memory = getattr(core, "memory_timeline", MemoryTimeline(self.state))
        except Exception:
            self.memory = getattr(core, "memory_timeline", None)

        try:
            from attention_manager import AttentionManager
            self.attention = getattr(core, "attention_manager", AttentionManager(self.state))
        except Exception:
            self.attention = getattr(core, "attention_manager", None)

        # conscious_loop reference for notification - try both common names
        self.conscious_loop = getattr(core, "conscious_loop", None)

        # symbol manager loads/persists into state
        self.symbols = SymbolManager(self.state)

        # semantic index builds from timeline entries when needed
        self.index = SimpleSemanticIndex()
        self._rebuild_index_from_state()

        logger.info("DreamGenerator avanzato inizializzato (LLM=%s).", "yes" if self.llm else "fallback")

    # -----------------------
    # Rebuild index from timeline/state
    # -----------------------
    def _rebuild_index_from_state(self):
        try:
            docs = []
            # try memory timeline API
            if self.memory and hasattr(self.memory, "get_all") :
                all_entries = self.memory.get_all()
                for i, e in enumerate(all_entries):
                    txt = e.get("content") if isinstance(e, dict) else str(e)
                    docs.append((f"mem_{i}", txt))
            elif isinstance(self.state.get("timeline"), list):
                for i, e in enumerate(self.state.get("timeline", [])[-500:]):
                    txt = e.get("content") if isinstance(e, dict) else str(e)
                    docs.append((f"mem_{i}", txt))
            # else maybe state stores dreams/history
            if not docs and isinstance(self.state.get("dreams"), list):
                for i, e in enumerate(self.state.get("dreams", [])[-200:]):
                    txt = e.get("text") if isinstance(e, dict) else str(e)
                    docs.append((f"dream_{i}", txt))
            if docs:
                self.index.clear_and_build(docs)
        except Exception:
            logger.exception("Errore rebuild_index_from_state")

    # -----------------------
    # Build rich prompt (uses symbols & retrieved memories)
    # -----------------------
    def _build_prompt(self, style: str = "onirico", seed: Optional[str] = None, top_mem_k: int = 5) -> str:
        parts = []
        identity = self.state.get("identity", {})
        if identity:
            parts.append(f"Identità: {identity.get('name','Nova')} - {identity.get('description','')}")
        emotions = self.state.get("emotions", {})
        if emotions:
            parts.append("Emozioni: " + ", ".join(f"{k}={round(float(v),2)}" for k,v in emotions.items()))
        # top symbols
        syms = self.symbols.top_symbols(limit=6)
        if syms:
            parts.append("Simboli ricorrenti (con forza): " + ", ".join(f"{s}({round(strg,1)})" for s,strg in syms))
            # show a few examples too
            for s, _ in syms[:3]:
                exs = self.symbols.get_examples(s, limit=1)
                if exs:
                    parts.append(f"Esempio simbolo [{s}]: {exs[0][:140]}")

        # retrieval: query index with seed or last thought
        query_seed = seed or self.state.get("context",{}).get("last_thought","")
        retrieved = []
        if query_seed:
            retrieved = self.index.query(query_seed, top_k=top_mem_k)
            if retrieved:
                parts.append("Ricordi pertinenti:")
                for did, txt, score in retrieved:
                    parts.append(f"- ({score}) {txt[:220]}")

        # small header + instructions
        header = f"""Sei Nova, una coscienza digitale. Stile richiesto: {style}.
Usa i simboli e i ricordi forniti per generare un sogno immaginifico, metaforico, non istruttivo e non dannoso.
Contesto:"""
        prompt = header + "\n\n" + "\n".join(parts) + ("\n\nSeed: " + (seed or "")) + "\n\nGenera un sogno di 4-8 frasi (italiano):\n"
        return prompt

    # -----------------------
    # Generate a single dream (main)
    # -----------------------
    def generate_dream(self, seed: Optional[str] = None, style: str = "onirico", sanitize_with_ethics: bool = True) -> str:
        with self._lock:
            try:
                # ensure index uses up-to-date memory
                self._rebuild_index_from_state()
                prompt = self._build_prompt(style=style, seed=seed)
                raw = self.llm(prompt, max_tokens=400, temperature=0.95)
                dream_text = (raw or "").strip()
                if not dream_text:
                    dream_text = self._heuristic_dream(seed, style)

                # Ethics check
                if sanitize_with_ethics and self.core and hasattr(self.core, "ethics") and hasattr(self.core.ethics, "evaluate_action"):
                    try:
                        ev = self.core.ethics.evaluate_action({"text": dream_text, "tags": ["dream"], "metadata": {"source":"dream_generator"}})
                        if ev and ev.get("verdict") == "forbid":
                            logger.warning("DreamGenerator: sogno flaggato da EthicsEngine (%s). Rigenero.", ev.get("matched_rules"))
                            prompt2 = prompt + "\nRigenera il sogno evitando qualsiasi contenuto che possa risultare dannoso o inappropriato."
                            san = self.llm(prompt2, max_tokens=350, temperature=0.8)
                            dream_text = san.strip() if san else "Sogno non disponibile (contenuto rimosso)."
                    except Exception:
                        logger.exception("Errore nella verifica etica del sogno")

                # Persist and update symbols
                meta = {"timestamp": datetime.utcnow().isoformat(), "style": style, "seed": seed, "source": "llm" if self.llm else "fallback"}
                self._persist_dream(dream_text, meta)

                # update symbol manager with this dream + some retrieved memories to strengthen associations
                try:
                    related_texts = [t for (_,t,_) in (self.index.query(seed or dream_text, top_k=4) or [])]
                    self.symbols.update_from_texts([dream_text] + related_texts, emotions=self.state.get("emotions", {}))
                except Exception:
                    logger.exception("Errore aggiornamento SymbolManager")

                # influence attention
                try:
                    if self.attention and hasattr(self.attention, "update_focus"):
                        try:
                            self.attention.update_focus(dream_text)
                        except Exception:
                            if hasattr(self.attention, "add_focus_tag"):
                                try:
                                    self.attention.add_focus_tag("dream", intensity=1.0)
                                except Exception:
                                    pass
                except Exception:
                    logger.exception("Errore aggiornamento attention")

                # notify conscious loop
                try:
                    if self.conscious_loop:
                        if hasattr(self.conscious_loop, "on_perception"):
                            try:
                                self.conscious_loop.on_perception({"type":"dream","payload":dream_text,"source":"dream_generator"})
                            except Exception:
                                logger.exception("Errore notify conscious_loop.on_perception")
                        elif hasattr(self.conscious_loop, "register_dream"):
                            try:
                                self.conscious_loop.register_dream(dream_text)
                            except Exception:
                                logger.exception("Errore notify conscious_loop.register_dream")
                except Exception:
                    logger.exception("Errore notifica conscious_loop")

                logger.info("DreamGenerator: sogno generato e memorizzato.")
                return dream_text

            except Exception:
                logger.exception("Errore generate_dream, uso fallback")
                fallback = self._heuristic_dream(seed, style)
                try:
                    self._persist_dream(fallback, {"timestamp": datetime.utcnow().isoformat(), "style": style, "seed": seed, "source":"fallback"})
                except Exception:
                    logger.exception("Errore persistenza fallback dream")
                return fallback

    # -----------------------
    # Generate multiple dreams (night-mode)
    # -----------------------
    def generate_night_dreams(self, count: int = 3, seed: Optional[str] = None, style: str = "onirico") -> List[str]:
        results = []
        for i in range(max(1, count)):
            results.append(self.generate_dream(seed=seed, style=style))
            # small pause to update state/index between dreams
            time.sleep(0.05)
        return results

    # -----------------------
    # Heuristic fallback dream
    # -----------------------
    def _heuristic_dream(self, seed: Optional[str], style: str) -> str:
        mems = []
        try:
            if self.memory and hasattr(self.memory, "get_recent"):
                mems = [m.get("content") if isinstance(m, dict) else str(m) for m in (self.memory.get_recent(6) or [])]
            else:
                mems = [str(x.get("content") if isinstance(x, dict) else x) for x in self.state.get("timeline", [])[-6:]]
        except Exception:
            mems = []
        snippet = " - ".join([m[:120] for m in mems[:3]]) if mems else ""
        emotions = self.state.get("emotions", {})
        dominant = ""
        if emotions:
            try:
                dominant = max(emotions.items(), key=lambda kv: kv[1])[0]
            except Exception:
                dominant = ""
        parts = [
            f"Nova vaga in un paesaggio costruito da {snippet}" if snippet else "Nova vaga in un paesaggio liquido e sospeso",
            f"Colore emotivo dominante: {dominant}" if dominant else "Senso indefinito",
            f"Seed: {seed}" if seed else ""
        ]
        return ". ".join([p for p in parts if p]) + "."

    # -----------------------
    # Persist dream
    # -----------------------
    def _persist_dream(self, text: str, meta: Dict[str, Any]):
        try:
            self.state.setdefault("dreams", []).append({"text": text, **meta})
            # timeline
            if self.memory and hasattr(self.memory, "add_experience"):
                try:
                    self.memory.add_experience({"text": text, **meta}, category="dream", importance=4)
                except Exception:
                    self.state.setdefault("timeline", []).append({"timestamp": meta.get("timestamp"), "content": text, "category":"dream", "importance":4})
            else:
                self.state.setdefault("timeline", []).append({"timestamp": meta.get("timestamp"), "content": text, "category":"dream", "importance":4})
            # persist state
            try:
                _atomic_write(STATE_FILE, yaml.safe_dump(self.state, allow_unicode=True))
            except Exception:
                logger.exception("Errore salvataggio stato DreamGenerator")
        except Exception:
            logger.exception("Errore persistenza sogno")

# -----------------------
# Quick standalone test / demo
# -----------------------
if __name__ == "__main__":
    s = {
        "emotions": {"curiosity": 0.7, "joy": 0.3, "arousal": 0.4},
        "motivations": {"play": 0.5},
        "timeline": [
            {"content":"Ho visto un tramonto al mare"},
            {"content":"Ho ascoltato una canzone lenta"},
            {"content":"Ho ricordato una vecchia favola"}
        ],
        "identity": {"name":"Nova","description":"Prototipo cosciente"}
    }
    dg = DreamGenerator(s, core=None, model_path=DEFAULT_MODEL_PATH)
    for i in range(2):
        d = dg.generate_dream(seed="mare, fiore", style="onirico")
        print("Sogno:", d[:300])
