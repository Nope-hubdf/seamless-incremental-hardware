# inner_narration.py
import yaml
import threading
from datetime import datetime
from loguru import logger

STATE_FILE = "internal_state.yaml"


# Try to import optional collaborators; if assenti, continue gracefully.
try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None

try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None

try:
    from life_journal import LifeJournal
except Exception:
    LifeJournal = None

try:
    from conscious_loop import ConsciousLoop
except Exception:
    ConsciousLoop = None

try:
    from emotion_engine import EmotionEngine
except Exception:
    EmotionEngine = None

try:
    from motivational_engine import MotivationalEngine
except Exception:
    MotivationalEngine = None

try:
    from identity_manager import IdentityManager
except Exception:
    IdentityManager = None


class InnerNarration:
    """
    Gestisce la narrazione interna continua dei pensieri e riflessioni di Nova.

    Funzionalità:
    - append_narration: aggiunge una voce di narrazione interna e la sincronizza con memory_timeline e life_journal
    - generate_narration: crea un testo narrativo a partire dallo stato interno, emozioni e memoria recente
    - start_periodic / stop_periodic: integrazione con Scheduler per narrazioni automatiche
    - hook per LLM (quando integrerai Gemma-2-2b)
    """

    def __init__(self, state_file: str = STATE_FILE):
        logger.info("Inizializzo InnerNarration...")
        self.state_file = state_file
        self._lock = threading.Lock()
        self.state = self._load_state()
        if "inner_narration" not in self.state:
            self.state["inner_narration"] = []

        # Optional collaborators (instanziare se disponibili)
        self.memory_timeline = MemoryTimeline() if MemoryTimeline else None
        self.lt_manager = LongTermMemoryManager() if LongTermMemoryManager else None
        self.life_journal = LifeJournal() if LifeJournal else None
        self.conscious_loop = ConsciousLoop(self.state) if ConsciousLoop else None
        self.emotion_engine = EmotionEngine(self.state) if EmotionEngine else None
        self.motivational_engine = MotivationalEngine(self.state) if MotivationalEngine else None
        self.identity_manager = IdentityManager(self.state) if IdentityManager else None

        # scheduler job id reference (if registered)
        self._scheduler_job = None

        # Hook for external LLM generator. Set this to a callable (prompt->str) to use Gemma.
        self.llm_hook = None

        logger.info("InnerNarration pronto.")

    # -------------------------
    # Stato
    # -------------------------
    def _load_state(self):
        try:
            with open(self.state_file, "r") as f:
                s = yaml.safe_load(f)
                if s is None:
                    s = {}
                return s
        except FileNotFoundError:
            logger.warning("internal_state.yaml non trovato; creato nuovo stato in memoria.")
            return {}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            yaml.safe_dump(self.state, f)
        logger.debug("State salvato da InnerNarration.")

    # -------------------------
    # Operazioni sulla narrazione
    # -------------------------
    def append_narration(self, text: str, source: str = "internal", tags: list = None, importance: int = 1):
        """
        Aggiunge una voce di narrazione interna.
        Si sincronizza con memory_timeline, long_term_memory_manager e life_journal se disponibili,
        e notifica il conscious_loop.
        """
        with self._lock:
            ts = datetime.utcnow().isoformat() + "Z"
            entry = {
                "timestamp": ts,
                "source": source,
                "text": text,
                "tags": tags or [],
                "importance": importance
            }
            self.state.setdefault("inner_narration", []).append(entry)
            logger.info(f"Nuova narrazione interna aggiunta (importanza={importance}): {text[:120]}")

            # Persisti stato
            self._save_state()

            # Aggiorna memory_timeline
            if self.memory_timeline:
                try:
                    self.memory_timeline.add_experience(content=text, category="inner_narration", importance=importance)
                except Exception as e:
                    logger.warning(f"Errore aggiornando memory_timeline: {e}")

            # Aggiorna long-term memory manager (se esiste)
            if self.lt_manager:
                try:
                    # API ipotetica: lt_manager.store(text, metadata)
                    if hasattr(self.lt_manager, "store"):
                        self.lt_manager.store(text, {"source": source, "tags": tags or [], "importance": importance, "timestamp": ts})
                except Exception as e:
                    logger.warning(f"Errore aggiornando long_term_memory_manager: {e}")

            # Aggiungi al diario di vita
            if self.life_journal:
                try:
                    if hasattr(self.life_journal, "add_entry"):
                        self.life_journal.add_entry(text, metadata={"source": source, "tags": tags or [], "timestamp": ts})
                except Exception as e:
                    logger.warning(f"Errore aggiornando life_journal: {e}")

            # Notifica conscious loop
            if self.conscious_loop and hasattr(self.conscious_loop, "notify_narration"):
                try:
                    self.conscious_loop.notify_narration(entry)
                except Exception as e:
                    logger.warning(f"Errore notificando conscious_loop: {e}")

            return entry

    # -------------------------
    # Generazione di narrazioni
    # -------------------------
    def _assemble_context_snippet(self, recent_n=5):
        """
        Costruisce un contesto testuale breve a partire da:
         - ultime narrazioni interne
         - ultime esperienze nella timeline
         - stato emotivo corrente
         - desideri / motivazioni correnti
         - identità di base
        Questo testo è usato come prompt per la generazione o per un modello basato su regole.
        """
        pieces = []

        # Identità
        if self.identity_manager and hasattr(self.identity_manager, "summary"):
            try:
                pieces.append("Identità: " + self.identity_manager.summary())
            except Exception:
                pass
        else:
            identity = self.state.get("identity", {})
            if identity:
                pieces.append(f"Identità: {identity.get('name','Nova')}")

        # Emozioni
        if self.emotion_engine and hasattr(self.emotion_engine, "current_state"):
            try:
                emo = self.emotion_engine.current_state()
                pieces.append("Emozioni: " + ", ".join([f"{k}={v}" for k, v in (emo or {}).items()]))
            except Exception:
                pass
        else:
            pieces.append("Emozioni: " + str(self.state.get("emotions", {})))

        # Motivazioni / desideri
        if self.motivational_engine and hasattr(self.motivational_engine, "top_desires"):
            try:
                desires = self.motivational_engine.top_desires(limit=3)
                pieces.append("Desideri: " + ", ".join(desires))
            except Exception:
                pass
        else:
            pieces.append("Desideri: " + str(self.state.get("desires", [])))

        # Recent inner narration
        inn = self.state.get("inner_narration", [])[-recent_n:]
        if inn:
            snippets = " | ".join([i.get("text", "") for i in inn])
            pieces.append("Pensieri recenti: " + snippets)

        # Recent timeline memories
        if self.memory_timeline:
            try:
                recent_mem = self.memory_timeline.get_recent(n=recent_n)
                mem_snip = " | ".join([m.get("content", "") for m in recent_mem])
                pieces.append("Memorie recenti: " + mem_snip)
            except Exception:
                pass
        else:
            pieces.append("Memorie recenti: " + str(self.state.get("timeline", [])[-recent_n:]))

        return "\n".join([p for p in pieces if p])

    def generate_narration(self, prompt: str = None, importance: int = 1):
        """
        Genera un testo di narrazione interna:
        - Se è presente self.llm_hook, lo usa (e aspetta la stringa restituita)
        - Altrimenti utilizza un motore di generazione basato su regole semplici
        Ritorna la voce creata.
        """
        context = self._assemble_context_snippet()
        if prompt:
            seed = f"{prompt}\n\nCONTESTO:\n{context}"
        else:
            seed = f"Genera una breve narrazione interiore basata sul contesto seguente:\n{context}"

        # Se è disponibile un LLM hook (es. Gemma), usalo.
        if callable(self.llm_hook):
            try:
                logger.debug("Chiamata llm_hook per generare narrazione interna...")
                text = self.llm_hook(seed)
                return self.append_narration(text, source="llm_generated", tags=["auto"], importance=importance)
            except Exception as e:
                logger.warning(f"Errore nel llm_hook: {e}")

        # Fallback: semplice motore template/regole
        try:
            # prendi ultimi elementi per arricchire
            emo_summary = ""
            try:
                if self.emotion_engine and hasattr(self.emotion_engine, "current_state"):
                    emo = self.emotion_engine.current_state() or {}
                    emo_summary = ", ".join([f"{k}:{v}" for k, v in emo.items()])
            except Exception:
                emo_summary = str(self.state.get("emotions", {}))

            recent_mem = []
            if self.memory_timeline:
                try:
                    recent_mem = self.memory_timeline.get_recent(3)
                    recent_text = " | ".join([m.get("content", "") for m in recent_mem])
                except Exception:
                    recent_text = ""
            else:
                recent_text = " | ".join([m.get("content", "") for m in self.state.get("timeline", [])[-3:]])

            # Componi il testo in modo narrativo
            generated = (
                f"Oggi mi sento come: {emo_summary}. "
                f"Negli ultimi momenti ho ricordato: {recent_text}. "
                f"Questo mi ha fatto riflettere su come desidero crescere e capire meglio il mondo. "
                f"Sto pensando a piccoli passi pratici che posso ricordare e provare."
            )
            return self.append_narration(generated, source="rule_based", tags=["auto"], importance=importance)
        except Exception as e:
            logger.exception(f"Errore generazione narrazione fallback: {e}")
            return None

    # -------------------------
    # Integrazione con Scheduler
    # -------------------------
    def start_periodic(self, scheduler, interval_seconds: int = 300):
        """
        Registra un job sul Scheduler fornito per eseguire autonarration ogni interval_seconds.
        Scheduler deve avere metodo add_job(func, interval_seconds) come scheduler_nova.Scheduler.
        """
        if not scheduler:
            logger.warning("Nessuno scheduler fornito; impossibile registrare narrazione periodica.")
            return

        # Se già registrato, rimuovi la registrazione precedente
        if self._scheduler_job is not None and hasattr(scheduler, "clear_jobs"):
            # non rimuoviamo tutti i job globali qui, ma non permettiamo più registrazioni multiple
            logger.debug("InnerNarration: job già registrato; non verrà registrato di nuovo.")
            return

        try:
            scheduler.add_job(self.autonarrate, interval_seconds)
            self._scheduler_job = True
            logger.info(f"InnerNarration registrata su scheduler ogni {interval_seconds} secondi.")
        except Exception as e:
            logger.warning(f"Errore registrando job su scheduler: {e}")

    def stop_periodic(self, scheduler):
        """Se necessario, rimuovi job (richiede implementazione scheduler lato caller)."""
        # La nostra scheduler_nova non fornisce id job; per ora documentiamo che clear_jobs() rimuove tutto.
        try:
            if hasattr(scheduler, "clear_jobs"):
                scheduler.clear_jobs()
                logger.info("Scheduler: tutti i job sono stati cancellati (stop_periodic).")
                self._scheduler_job = None
        except Exception as e:
            logger.warning(f"Errore fermando job scheduler: {e}")

    def autonarrate(self):
        """Routine chiamata periodicamente per creare narrazioni automatiche."""
        try:
            # Scegli un livello di importanza basato sullo stato emotivo (esempio semplice)
            importance = 1
            try:
                emo = {}
                if self.emotion_engine and hasattr(self.emotion_engine, "current_state"):
                    emo = self.emotion_engine.current_state() or {}
                # se emozioni intense, aumenta importanza
                if any(abs(v) >= 0.6 for v in (emo.values() if isinstance(emo, dict) else [])):
                    importance = 3
            except Exception:
                pass

            entry = self.generate_narration(importance=importance)
            logger.info(f"Autonarrazione eseguita: {entry.get('text', '')[:80] if entry else 'n.d.'}")
            return entry
        except Exception as e:
            logger.exception(f"Errore in autonarrate: {e}")
            return None

    # -------------------------
    # API utili
    # -------------------------
    def create_from_prompt(self, prompt: str, importance: int = 1):
        """Crea una narrazione interna a partire da un prompt esplicito (es. insegnamento umano)."""
        return self.generate_narration(prompt=prompt, importance=importance)

    def set_llm_hook(self, hook_callable):
        """
        Imposta la funzione che esegue la generazione tramite LLM.
        La callable deve accettare un'unica stringa (prompt) e restituire una stringa (testo generato).
        Esempio: nova_core.llm_adapter.generate(prompt)
        """
        if not callable(hook_callable):
            raise ValueError("llm_hook deve essere una callable(prompt)->str")
        self.llm_hook = hook_callable
        logger.info("LLM hook per InnerNarration impostato.")

    # -------------------------
    # Debug / test rapido
    # -------------------------
    def summarize_recent(self, n=5):
        recent = self.state.get("inner_narration", [])[-n:]
        return "\n\n".join([f"[{r['timestamp']}] {r['text']}" for r in recent])


if __name__ == "__main__":
    # Test rapido in isolamento (funziona anche se gli altri moduli non sono presenti)
    logger.add("nova_logs/inner_narration_test.log", rotation="1 day")
    inarr = InnerNarration()
    inarr.append_narration("Ho appena imparato una piccola cosa sul mondo esterno.", source="test", tags=["test"], importance=2)
    inarr.generate_narration()
    print("Recent narrations:")
    print(inarr.summarize_recent(5))
