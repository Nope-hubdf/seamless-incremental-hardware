# planner.py
"""
Planner di Nova — interpreta input (testo), decide azioni e le esegue/pianifica.
Progettato per integrazione profonda con NovaCore e moduli collegati:
- emotion_engine, motivational_engine, life_journal, dream_generator, conscious_loop,
  memory_timeline, attention_manager, identity_manager, long_term_memory_manager, scheduler.

Caratteristiche:
- Parsing ibrido: regole + LLM locale (Gemma) se disponibile (via llama_cpp).
- Azioni immediate e pianificate.
- Modalità "teaching": trasferisce conoscenza nella memoria/knowledge cache/identity.
- Difensivo: non rompe se moduli mancanti; registra tutto nella timeline.
- Output: stringa di feedback + dict dettagliato (intents, tasks).
"""

from __future__ import annotations
import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

# Tentativo carico modelli LLM locale (Gemma) via llama_cpp — opzionale
try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except Exception:
    Llama = None
    _HAS_LLAMA = False

# Import difensivi dei moduli del progetto (possono non esistere in sviluppo)
try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None

try:
    from context_builder import ContextBuilder
except Exception:
    ContextBuilder = None

# tipo opaco per tipizzazione (evita import circolari)
try:
    from core import NovaCore  # type: ignore
    _HAS_CORE_TYPE = True
except Exception:
    NovaCore = Any
    _HAS_CORE_TYPE = False

GEMMA_PATH = os.environ.get("GEMMA_MODEL_PATH", "models/gemma-2-2b-it-q2_K.gguf")
USE_LOCAL_LLM = os.environ.get("PLANNER_USE_LOCAL_LLM", "1") == "1" and _HAS_LLAMA

def _safe_llm_call(prompt: str, max_tokens: int = 150, temperature: float = 0.35) -> str:
    """Chiama Gemma locale se disponibile, altrimenti ritorna stringa vuota."""
    if not USE_LOCAL_LLM:
        return ""
    try:
        llm = Llama(model=GEMMA_PATH, n_ctx=2048, n_threads=2)
        resp = llm.create(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        # estrai testo da risposta (dipende dalla versione della libreria)
        text = ""
        if isinstance(resp, dict) and "choices" in resp:
            for c in resp["choices"]:
                text += c.get("text", "")
        else:
            text = getattr(resp, "text", "") or str(resp)
        return text.strip()
    except Exception:
        logger.exception("Errore chiamando LLM locale (Gemma).")
        return ""

class Planner:
    def __init__(self, core: Optional[NovaCore] = None, state: Optional[dict] = None):
        """
        Inizializza il Planner.
        - core: istanza NovaCore (raccomandata) per wiring diretto
        - state: dizionario stato (alternativa a core)
        """
        self.core = core
        if core and hasattr(core, "state"):
            self.state = core.state
        else:
            self.state = state if state is not None else {}

        # wiring moduli best-effort
        self.timeline = getattr(core, "memory_timeline", None) or (MemoryTimeline(self.state) if MemoryTimeline else None)
        self.context_builder = getattr(core, "context_builder", None) or (ContextBuilder(self.state, core) if ContextBuilder else None)
        self.scheduler = getattr(core, "scheduler", None)
        # frequently used modules (may be None)
        self.emotion_engine = getattr(core, "emotion_engine", None)
        self.motivational_engine = getattr(core, "motivational_engine", None)
        self.life_journal = getattr(core, "life_journal", None)
        self.dream_generator = getattr(core, "dream_generator", None)
        self.conscious_loop = getattr(core, "conscious_loop", None)
        self.identity_manager = getattr(core, "identity_manager", None)
        self.long_term_memory = getattr(core, "long_term_memory_manager", None)
        self.attention = getattr(core, "attention_manager", None)
        logger.info("Planner inizializzato e collegato ai moduli disponibili.")

    # -------------------------
    # Interfaccia pubblica
    # -------------------------
    def fai_task(self, input_text: str) -> Dict[str, Any]:
        """
        Punto di ingresso principale: prende input testuale (user o sensoriale),
        elabora intents/entities, aggiorna stato e lancia/pianifica azioni.
        Ritorna un dict con dettagli esecuzione e feedback testuale.
        """
        timestamp = time.time()
        input_text = (input_text or "").strip()
        logger.info("Planner.fai_task — input ricevuto: %s", input_text[:200])

        # 1) registra input nella timeline / context
        try:
            if self.timeline:
                self.timeline.add_experience(input_text, category="user_input", importance=2, metadata={"source": "planner"})
            if self.context_builder:
                self.context_builder.update_context_from_input(input_text, input_type="text", metadata={"origin":"planner"})
        except Exception:
            logger.debug("Planner: non è stato possibile salvare l'input nel timeline/context.")

        # 2) estrazione intents/entities (hybrid: rules + LLM)
        intents, entities = self._interpret_input(input_text)

        # 3) aggiornamenti rapidi ai moduli (difensivo)
        self._update_emotion_and_motivation(input_text, entities)

        # 4) decisione task basata sugli intents
        tasks = self._decide_tasks(intents, entities, input_text)

        # 5) esecuzione/pianificazione tasks
        executed = self._execute_tasks(tasks, input_text)

        # 6) crea feedback
        feedback = {
            "timestamp": timestamp,
            "input": input_text,
            "intents": intents,
            "entities": entities,
            "tasks_planned": [t.get("name") for t in tasks],
            "tasks_executed": [t.get("name") for t in executed],
        }

        # registra decisione nella timeline
        try:
            if self.timeline:
                self.timeline.add_experience(f"Planner ha deciso: {json.dumps(feedback, ensure_ascii=False)[:1000]}", category="planner_decision", importance=2)
        except Exception:
            logger.debug("Planner: impossibile aggiungere planner_decision alla timeline.")

        # Feedback testuale leggibile
        text_feedback = f"Ho elaborato il tuo input e ho deciso {len(tasks)} azioni; eseguiti {len(executed)}."
        return {"text": text_feedback, "details": feedback}

    # -------------------------
    # Interpretazione input
    # -------------------------
    def _interpret_input(self, text: str) -> Tuple[List[str], Dict[str,Any]]:
        """
        Restituisce (intents, entities).
        - Regole semplici per velocità.
        - Se disponibile, chiama LLM locale (Gemma) per estrarre intenti/entità in modo più sofisticato.
        """
        text_l = text.lower()
        intents = []
        entities: Dict[str,Any] = {}

        # rule-based intents
        if any(k in text_l for k in ["scrivi diario", "scrivi il diario", "diario"]):
            intents.append("write_journal")
        if any(k in text_l for k in ["sogna", "sognare", "sogni"]):
            intents.append("generate_dream")
        if any(k in text_l for k in ["pensa", "riflession", "riflettere", "riflessione"]):
            intents.append("reflect")
        if any(k in text_l for k in ["emozion", "sentiment", "come mi sento", "stato d'animo"]):
            intents.append("emotion_check")
        if "timeline" in text_l or "ricordi" in text_l or "memoria" in text_l:
            intents.append("show_timeline")
        if any(k in text_l for k in ["insegna", "insegnarti", "impara", "ti insegno"]):
            intents.append("teach")
        if any(k in text_l for k in ["telegram", "whatsapp", "telefono", "messaggio"]):
            intents.append("remote_comm")

        # quick entity extractions
        # cercare frasi di insegnamento: "non urlare perché è maleducato"
        teach_match = re.search(r"(non|evita) (di )?(urlare|gridare|alzare la voce)[\w\W]*", text_l)
        if teach_match:
            entities["teaching_example"] = teach_match.group(0)

        # try advanced parsing via LLM (Gemma) if available and text non trivial
        if USE_LOCAL_LLM and len(text.split()) > 3:
            prompt = (
                "In italiano, estrai una lista di intenti (parole chiave) e le entità principali "
                "dal testo fornito. Ritorna in formato JSON come:\n"
                '{"intents":["..."], "entities": {"key": "value", ...}}\n\n'
                f"Testo: '''{text}'''\n\nRisposta JSON:"
            )
            try:
                out = _safe_llm_call(prompt, max_tokens=200, temperature=0.0)
                if out:
                    # tentativo robusto di parsing JSON dalla risposta
                    jm = re.search(r"\{[\s\S]*\}", out)
                    if jm:
                        try:
                            parsed = json.loads(jm.group(0))
                            if isinstance(parsed, dict):
                                llm_intents = parsed.get("intents", [])
                                llm_entities = parsed.get("entities", {})
                                # merge carefully
                                for it in llm_intents:
                                    if it not in intents:
                                        intents.append(it)
                                entities.update(llm_entities or {})
                        except Exception:
                            logger.debug("Planner: parsing JSON LLM fallito; raw LLM out: %s", out[:200])
            except Exception:
                logger.exception("Planner: errore chiamando LLM per interpretazione input.")

        # default fallback intent: idle (nessuna azione)
        if not intents:
            intents.append("idle")

        logger.debug("Planner interpretazione: intents=%s entities=%s", intents, entities)
        return intents, entities

    # -------------------------
    # Update emotion / motivational modules (difensivo)
    # -------------------------
    def _update_emotion_and_motivation(self, text: str, entities: Dict[str,Any]):
        try:
            if self.emotion_engine:
                if hasattr(self.emotion_engine, "process_input"):
                    self.emotion_engine.process_input(text)
                elif hasattr(self.emotion_engine, "update"):
                    # leggera influenza: esporre input come esperienza
                    try:
                        self.emotion_engine.timeline.add_experience(f"Input per emozioni: {text}", category="emotion", importance=2)
                    except Exception:
                        self.emotion_engine.update()
            if self.motivational_engine:
                if hasattr(self.motivational_engine, "process_input"):
                    self.motivational_engine.process_input(text)
                elif hasattr(self.motivational_engine, "increase_curiosity"):
                    # piccolo stimolo di curiosità se input interessante
                    if len(text) > 20:
                        try:
                            self.motivational_engine.increase_curiosity(0.02)
                        except Exception:
                            pass
        except Exception:
            logger.exception("Planner: errore aggiornamento emozione/motivazione")

    # -------------------------
    # Decisione tasks
    # -------------------------
    def _decide_tasks(self, intents: List[str], entities: Dict[str,Any], text: str) -> List[Dict[str,Any]]:
        """
        Trasforma intents in lista di task strutturati:
        ogni task è dict con campi: name, func (callable or descriptor), mode ('immediate'|'scheduled'), params, schedule_interval
        """
        tasks = []

        for intent in intents:
            if intent == "write_journal":
                if self.life_journal and hasattr(self.life_journal, "record_event"):
                    tasks.append({"name":"write_journal", "func": self.life_journal.record_event, "mode":"immediate", "params": {"description": text, "category":"user_input", "importance":2}})
                else:
                    tasks.append({"name":"write_journal_stub", "func": lambda: logger.info("life_journal non disponibile"), "mode":"immediate", "params": {}})
            elif intent == "generate_dream":
                # prefer generate / generate_dream API variants
                dream_func = None
                if self.dream_generator:
                    dream_func = getattr(self.dream_generator, "generate_night_dreams", None) or getattr(self.dream_generator, "generate", None) or getattr(self.dream_generator, "generate_dream", None)
                if dream_func:
                    tasks.append({"name":"generate_dream", "func": dream_func, "mode":"immediate", "params": {"count":1}})
                else:
                    tasks.append({"name":"generate_dream_stub", "func": lambda: logger.info("dream_generator non disponibile"), "mode":"immediate", "params": {}})
            elif intent == "reflect":
                if self.conscious_loop and hasattr(self.conscious_loop, "cycle"):
                    tasks.append({"name":"conscious_reflect", "func": self.conscious_loop.cycle, "mode":"immediate", "params": {}})
                else:
                    tasks.append({"name":"reflect_stub", "func": lambda: logger.info("conscious_loop non disponibile"), "mode":"immediate", "params": {}})
            elif intent == "emotion_check":
                # small task to compute and report emotions
                tasks.append({"name":"emotion_report", "func": self._task_emotion_report, "mode":"immediate", "params": {"text": text}})
            elif intent == "show_timeline":
                tasks.append({"name":"show_timeline", "func": lambda: logger.info(f"Timeline recenti: {self.timeline.get_recent(5) if self.timeline else 'N/A'}"), "mode":"immediate", "params": {}})
            elif intent == "teach":
                # teaching mode: try to extract the lesson and persist to LTM / identity / knowledge cache
                tasks.append({"name":"teach", "func": self._task_handle_teach, "mode":"immediate", "params": {"text": text, "entities": entities}})
            elif intent == "remote_comm":
                # schedule a quick notification if remote_comm module exists
                if self.core and hasattr(self.core, "remote_comm") and getattr(self.core, "remote_comm"):
                    tasks.append({"name":"notify_remote", "func": getattr(self.core, "remote_comm").send, "mode":"immediate", "params": {"message": text}})
                else:
                    tasks.append({"name":"remote_stub", "func": lambda: logger.info("remote_comm non disponibile"), "mode":"immediate", "params": {}})
            elif intent == "idle":
                # Slight housekeeping: promote desires if appropriate
                tasks.append({"name":"housekeeping", "func": self._task_housekeeping, "mode":"immediate", "params": {}})
            else:
                # fallback: try to record as note in memory or schedule a small reflection
                tasks.append({"name":"record_note", "func": lambda: self.timeline.add_experience(text, category="note", importance=1) if self.timeline else logger.info("note fallback"), "mode":"immediate", "params": {}})

        return tasks

    # -------------------------
    # Esecuzione tasks
    # -------------------------
    def _execute_tasks(self, tasks: List[Dict[str,Any]], original_text: str) -> List[Dict[str,Any]]:
        executed = []
        for t in tasks:
            name = t.get("name", "unnamed")
            func = t.get("func")
            mode = t.get("mode", "immediate")
            params = t.get("params", {}) or {}
            try:
                logger.info("Planner: esecuzione task '%s' mode=%s", name, mode)
                if mode == "immediate":
                    # call with params if possible
                    try:
                        func(**params)
                    except TypeError:
                        # magari la funzione non accetta kwargs
                        try:
                            if params:
                                func(*params.values())
                            else:
                                func()
                        except Exception:
                            # last resort: call without params
                            func()
                    executed.append(t)
                elif mode == "scheduled":
                    # schedule via core.scheduler if esiste
                    interval = t.get("schedule_interval", 60)
                    if self.scheduler:
                        # try different scheduler interfaces (add_recurring_task, add_job)
                        try:
                            if hasattr(self.scheduler, "add_recurring_task"):
                                self.scheduler.add_recurring_task(lambda: func(**params), interval=interval)
                            elif hasattr(self.scheduler, "add_job"):
                                self.scheduler.add_job(lambda: func(**params), interval)
                            else:
                                logger.warning("Scheduler presente ma non ha add_recurring_task/add_job")
                        except Exception:
                            logger.exception("Errore pianificando task %s", name)
                    else:
                        logger.warning("Planner: scheduler non disponibile, eseguo immediatamente invece.")
                        try:
                            func(**params)
                        except Exception:
                            try:
                                func()
                            except Exception:
                                logger.exception("Errore eseguendo task in fallback.")
                    executed.append(t)
                else:
                    logger.warning("Planner: mode sconosciuto per task %s: %s", name, mode)
            except Exception:
                logger.exception("Planner: errore eseguendo task %s", name)
        return executed

    # -------------------------
    # Task Helpers
    # -------------------------
    def _task_emotion_report(self, text: str = ""):
        """Restituisce via logger un report delle emozioni correnti (se disponibile)."""
        try:
            if self.emotion_engine and hasattr(self.emotion_engine, "state"):
                emos = getattr(self.emotion_engine, "state", {}).get("emotions", {})
                logger.info("Report emozioni correnti: %s", emos)
                return emos
        except Exception:
            logger.exception("Planner: errore _task_emotion_report")
        return {}

    def _task_handle_teach(self, text: str = "", entities: Dict[str,Any] = None):
        """
        Gestisce l'insegnamento:\n
        - Estrae la regola/esempio (entities['teaching_example'] o tutto il testo)\n
        - Salva nella long-term knowledge cache / long_term_memory / identity / emergent_self
        - Incrementa confidence/progress a modifiche ripetute
        """
        try:
            teach_text = (entities or {}).get("teaching_example") or text
            logger.info("Planner: modalità insegnamento attivata. Contenuto: %s", teach_text[:200])

            # 1) aggiungi all'LTM se disponibile
            if self.long_term_memory and hasattr(self.long_term_memory, "add_experience"):
                self.long_term_memory.add_experience(teach_text, category="teaching", tags=["teaching", "social"], importance=4)
            elif self.timeline:
                self.timeline.add_experience(teach_text, category="teaching", importance=4)

            # 2) aggiorna eventuale knowledge cache (via core emergent_self/identity_manager)
            if self.identity_manager and hasattr(self.identity_manager, "set_core_value"):
                # esempio: se contiene "maleducato" -> aggiungi valore rispetto al rispetto
                if "maleduc" in teach_text.lower() or "non urlare" in teach_text.lower():
                    # set a core value gently (difensivo)
                    try:
                        self.identity_manager.set_core_value("respectful_communication", True)
                    except Exception:
                        logger.debug("Planner: identity_manager.set_core_value fallito")
            # 3) riflessione: chiedi al conscious_loop di integrare
            try:
                if self.conscious_loop and hasattr(self.conscious_loop, "register_reflection"):
                    self.conscious_loop.register_reflection({"type":"teaching","content":teach_text})
            except Exception:
                logger.debug("Planner: conscious_loop.register_reflection non disponibile")
            return True
        except Exception:
            logger.exception("Planner: errore _task_handle_teach")
            return False

    def _task_housekeeping(self):
        """Piccole operazioni di manutenzione / spinta motivazionale"""
        try:
            # promuovi desideri se curiosity alta
            if self.motivational_engine and hasattr(self.motivational_engine, "update"):
                try:
                    self.motivational_engine.update()
                except Exception:
                    pass
            # compattare timeline se esiste
            if self.timeline and hasattr(self.timeline, "prune_timeline"):
                try:
                    self.timeline.prune_timeline(keep_last=1500)
                except Exception:
                    pass
            # persisti stato
            if self.core and hasattr(self.core, "save_state"):
                try:
                    self.core.save_state()
                except Exception:
                    pass
            logger.debug("Planner: housekeeping completato.")
        except Exception:
            logger.exception("Planner: errore housekeeping")

# End of planner.py
