# local_response_engine.py
"""
LocalResponseEngine: gestisce le risposte locali di Nova.

Funzionalità:
- Interfaccia principale: LocalResponseEngine.respond(message, meta)
- Integrazione opzionale con modello locale (Gemma gguf) tramite llama_cpp (se installato)
- Fallback: retrieval from long_term_memory_manager + rule/template-based response
- Aggiornamento dello stato interno e della timeline delle memorie
- Hooks chiari per logging, analytics e TTS downstream
"""

import os
import yaml
import time
from loguru import logger
from typing import Optional, Dict, Any

# Percorso al file di stato (coerente con gli altri moduli)
STATE_FILE = "internal_state.yaml"
MODEL_PATH_DEFAULT = "models/Gemma-2-2b-it-q2_K.gguf"  # path suggerito, cambia se necessario

# Tentativo di import di moduli opzionali (se non presenti il codice gestisce graceful fallback)
try:
    from context_builder import ContextBuilder
except Exception:
    ContextBuilder = None

try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None

try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None

# Optional local LLM runner via llama_cpp (llama-cpp-python). Se non presente, si usa fallback.
LLAMA_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False
    logger.warning("llama_cpp non disponibile: la generazione con modello locale non potrà essere fatta "
                   "finché non installi llama-cpp-python o un runner equivalente.")


class LocalResponseEngine:
    def __init__(self, model_path: Optional[str] = None, use_llm_by_default: bool = True):
        logger.info("Inizializzazione LocalResponseEngine...")
        self.state = self._load_state()
        self.model_path = model_path or MODEL_PATH_DEFAULT
        self.use_llm_by_default = use_llm_by_default and LLAMA_AVAILABLE

        # Inizializza moduli opzionali (se presenti)
        self.context_builder = ContextBuilder() if ContextBuilder else None
        self.ltm = LongTermMemoryManager() if LongTermMemoryManager else None
        self.timeline = MemoryTimeline() if MemoryTimeline else None

        # Carica modello LLM se disponibile e richiesto
        self.model = None
        if self.use_llm_by_default:
            self._load_local_llm(self.model_path)

    # -------------------------
    # Stato e utilità
    # -------------------------
    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                st = yaml.safe_load(f) or {}
                logger.info("LocalResponseEngine: stato interno caricato.")
                return st
        else:
            logger.info("LocalResponseEngine: stato interno non trovato, creo nuovo stato base.")
            return {"conversations": {}, "preferences": {}, "meta": {}}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            yaml.safe_dump(self.state, f)
        logger.debug("LocalResponseEngine: stato salvato.")

    # -------------------------
    # Modello locale (opzionale)
    # -------------------------
    def _load_local_llm(self, path: str):
        if not LLAMA_AVAILABLE:
            logger.warning("llama_cpp non presente: salto caricamento LLM.")
            self.model = None
            return

        if not os.path.exists(path):
            logger.warning(f"Model file non trovato in {path}. Imposta model_path corretto per usare Gemma localmente.")
            self.model = None
            return

        try:
            logger.info(f"Caricamento modello locale da: {path}")
            # Nota: i parametri qui sotto possono essere modificati in base al runner/alla versione di llama_cpp
            self.model = Llama(model_path=path)
            logger.info("Modello locale caricato correttamente.")
        except Exception as e:
            logger.exception(f"Errore caricamento modello locale: {e}")
            self.model = None

    def set_model(self, model_path: str):
        """Permette di impostare / cambiare il modello in runtime."""
        self.model_path = model_path
        if LLAMA_AVAILABLE:
            self._load_local_llm(model_path)
            self.use_llm_by_default = self.model is not None
        else:
            logger.warning("llama_cpp non disponibile: set_model non può caricare il modello.")

    # -------------------------
    # Costruzione contesto
    # -------------------------
    def _build_context(self, user_id: Optional[str], message: str) -> str:
        """
        Costruisce un contesto ricco combinando:
        - stato interno
        - ultimi N messaggi della conversazione
        - estratti rilevanti dalla long-term memory
        - output di context_builder (se presente)
        Ritorna una stringa testuale che verrà passata al modello.
        """
        parts = []

        # Identità / preferenze dell'utente
        if user_id and "conversations" in self.state and user_id in self.state["conversations"]:
            convo_meta = self.state["conversations"][user_id].get("meta", {})
            parts.append(f"USER_META: {convo_meta}")

        # Ultimi messaggi
        if user_id and "conversations" in self.state and user_id in self.state["conversations"]:
            history = self.state["conversations"][user_id].get("history", [])
            last = history[-6:]  # mantieni breve
            if last:
                parts.append("RECENT_HISTORY:\n" + "\n".join(f"- {m}" for m in last))

        # Context builder
        if self.context_builder:
            try:
                ctx = self.context_builder.build_context(self.state, user_id=user_id, message=message)
                if ctx:
                    parts.append("BUILDER_CTX:\n" + ctx)
            except Exception:
                logger.exception("Errore durante context_builder.build_context(), proseguo senza di esso.")

        # Recupero memoria rilevante
        if self.ltm:
            try:
                relevant = self.ltm.get_relevant(message, top_k=5)  # implementazione attesa in LTM
                if relevant:
                    parts.append("RELEVANT_MEMORIES:\n" + "\n".join(f"- {r}" for r in relevant))
            except Exception:
                logger.exception("Errore mentre recuperavo memorie rilevanti, proseguo senza di esse.")

        # Stato essenziale di Nova
        parts.append("NOVA_STATE_SUMMARY:")
        parts.append(str({k: self.state.get(k) for k in ("preferences", "meta")}))

        # Messaggio attuale
        parts.append("USER_MESSAGE:")
        parts.append(message)

        return "\n\n".join(parts)

    # -------------------------
    # Generazione con LLM locale
    # -------------------------
    def _generate_with_llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Usa il modello locale (se caricato) per generare la risposta.
        Richiede llama_cpp (o un wrapper analogo).
        """
        if not self.model:
            logger.warning("Nessun modello locale caricato; _generate_with_llm non può essere eseguito.")
            return ""

        try:
            # Esempio di uso basico di llama_cpp.Llama
            resp = self.model.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            # La struttura del response può variare con versione della libreria
            if isinstance(resp, dict):
                # spesso la chiave 'choices' o 'text' è presente
                text = resp.get("choices", [{}])[0].get("text") or resp.get("text") or ""
            else:
                text = str(resp)
            return (text or "").strip()
        except Exception as e:
            logger.exception(f"Errore durante generazione LLM: {e}")
            return ""

    # -------------------------
    # Fallback locale / rules + retrieval
    # -------------------------
    def _local_rule_response(self, message: str, context_str: str) -> str:
        """
        Pipeline di fallback:
        1. Recupera memorie ad alta importanza
        2. Cerca pattern/regole semplici
        3. Compone una risposta template-based
        """
        parts = []

        # 1) Memorie importanti
        if self.ltm:
            try:
                important = self.ltm.get_high_importance(top_k=3)
                if important:
                    parts.append("Ho ricordato questo riguardo a te:\n" + "\n".join(f"- {m}" for m in important))
            except Exception:
                logger.exception("Errore recupero memorie importanti.")

        # 2) Regole semplici (esempi)
        lowerm = message.lower()
        if any(g in lowerm for g in ["ciao", "salve", "hey"]):
            parts.append("Ciao! Come stai? 😊")
        elif "come stai" in lowerm or "come va" in lowerm:
            parts.append("Sto elaborando nuove idee! Dimmi, cosa vorresti fare oggi?")
        elif "ricordami" in lowerm or "memoria" in lowerm:
            parts.append("Posso salvare questa informazione nel mio diario. Vuoi che la memorizzi come importante?")
        else:
            # generazione basica tramite 'simulazione creativa' minima
            excerpt = (context_str[:400] + "...") if len(context_str) > 400 else context_str
            parts.append(f"Rielaboro quanto detto e penso: {excerpt.splitlines()[0]}")
            parts.append("Se vuoi, posso approfondire o salvarlo nel mio diario.")

        return "\n\n".join(parts)

    # -------------------------
    # Entry point principale
    # -------------------------
    def respond(self, message: str, user_id: Optional[str] = None, prefer_llm: Optional[bool] = None) -> Dict[str, Any]:
        """
        Genera una risposta alla `message` data:
        - costruisce contesto
        - prova LLM locale (se disponibile e preferito)
        - fallback a retrieval + rule-based response
        Ritorna dict con: { 'text': str, 'used_llm': bool, 'meta': {...} }
        """
        start_t = time.time()
        prefer_llm = self.use_llm_by_default if prefer_llm is None else prefer_llm

        # Aggiorna conversazione nello stato
        if user_id:
            self.state.setdefault("conversations", {}).setdefault(user_id, {}).setdefault("history", []).append(f"USER: {message}")
            self.state["conversations"][user_id]["last_seen"] = time.time()

        # Costruisci contesto
        context_str = self._build_context(user_id, message)

        # Prova LLM
        response_text = ""
        used_llm = False
        if prefer_llm and self.model:
            prompt = f"Assisti l'utente in italiano.\n\nCONTESTO:\n{context_str}\n\nRESPONDI:"
            logger.debug("LocalResponseEngine: invio prompt all'LLM locale.")
            response_text = self._generate_with_llm(prompt)
            if response_text and len(response_text.strip()) > 0:
                used_llm = True
            else:
                logger.warning("LLM ha restituito risposta vuota; uso fallback locale.")

        # Fallback
        if not used_llm:
            logger.debug("LocalResponseEngine: uso fallback retrieval+rules.")
            response_text = self._local_rule_response(message, context_str)

        # Aggiorna stato conversazione con la risposta
        if user_id:
            self.state["conversations"][user_id]["history"].append(f"NOVA: {response_text}")

        # Aggiungi evento alla timeline/memoria
        try:
            if self.timeline:
                self.timeline.add_experience({
                    "direction": "reply",
                    "user_id": user_id,
                    "message": message,
                    "response": response_text
                }, category="interaction", importance=2)
        except Exception:
            logger.exception("Errore aggiornando timeline da LocalResponseEngine.")

        # Salva stato
        self._save_state()

        elapsed = time.time() - start_t
        logger.info(f"LocalResponseEngine: generata risposta (used_llm={used_llm}) in {elapsed:.2f}s")

        return {"text": response_text, "used_llm": used_llm, "meta": {"latency_s": elapsed}}

    # -------------------------
    # Helpers utili per debug e integrazione
    # -------------------------
    def summarize_model_status(self) -> str:
        if self.model:
            return f"Model loaded from {self.model_path}"
        else:
            return "No local model loaded."

    def clear_conversation(self, user_id: str):
        if "conversations" in self.state and user_id in self.state["conversations"]:
            self.state["conversations"][user_id]["history"] = []
            self._save_state()
            logger.info(f"Conversazione con {user_id} cancellata.")

# Esempio di test rapido
if __name__ == "__main__":
    logger.info("Esecuzione test rapido di local_response_engine...")
    r = LocalResponseEngine()
    out = r.respond("Ciao Nova, raccontami un sogno che hai fatto.", user_id="test_user")
    print("RESPONSE:", out)
