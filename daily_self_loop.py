# daily_self_loop.py
"""
DailySelfLoop
--------------
Loop quotidiano per Nova: coordina aggiornamento della coscienza,
integrazione delle esperienze nella memoria a lungo termine,
generazione sogni notturni, aggiornamento diario, e notifica
degli altri moduli (attention, emotion, motivation, persona, planner).

Design:
- Lavora sullo stato condiviso salvato in `internal_state.yaml`.
- Si integra con i moduli: memory_timeline, long_term_memory_manager,
  dream_generator, life_journal, attention_manager, emotion_engine,
  motivational_engine, self_reflection, persona_manager, planner.
- Funziona sia come singolo run giornaliero che come scheduler integrabile
  (se gli si passa un Scheduler).
"""

import os
import yaml
import traceback
from datetime import datetime, timedelta
from loguru import logger
import importlib
from typing import Optional

STATE_FILE = "internal_state.yaml"

def safe_import(module_name: str):
    """Prova a importare un modulo e ritorna None in caso di errore."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        logger.warning(f"Impossibile importare {module_name}: {e}")
        return None

# Tentativi di import dei moduli "vicini"
memory_timeline_mod = safe_import("memory_timeline")
ltm_mod = safe_import("long_term_memory_manager")
dream_mod = safe_import("dream_generator")
journal_mod = safe_import("life_journal")
attention_mod = safe_import("attention_manager")
emotion_mod = safe_import("emotion_engine")
motivation_mod = safe_import("motivational_engine")
self_reflection_mod = safe_import("self_reflection")
persona_mod = safe_import("persona_manager")
planner_mod = safe_import("planner")

class DailySelfLoop:
    def __init__(self, state_file: str = STATE_FILE, scheduler: Optional[object] = None):
        self.state_file = state_file
        self.state = self._load_state()
        self.scheduler = scheduler  # opzionale: passare scheduler_nova.Scheduler
        logger.info("DailySelfLoop inizializzato.")

        # Connessioni ai componenti (oggetti o moduli)
        self.memory_timeline = getattr(memory_timeline_mod, "MemoryTimeline", None)
        self.long_term_memory_manager = getattr(ltm_mod, "LongTermMemoryManager", None)
        self.dream_generator = getattr(dream_mod, "DreamGenerator", None)
        self.life_journal = getattr(journal_mod, "LifeJournal", None)
        self.attention_manager = getattr(attention_mod, "AttentionManager", None)
        self.emotion_engine = getattr(emotion_mod, "EmotionEngine", None)
        self.motivational_engine = getattr(motivation_mod, "MotivationalEngine", None)
        self.self_reflection = getattr(self_reflection_mod, "SelfReflection", None)
        self.persona_manager = getattr(persona_mod, "PersonaManager", None)
        self.planner = getattr(planner_mod, "Planner", None)

        # Istanze concrete (se le classi sono importabili creiamo istanze condivise)
        self._init_instances()

    def _init_instances(self):
        try:
            # Se la timeline è disponibile, crea un'istanza collegata allo stesso stato
            self.timeline = self.memory_timeline() if self.memory_timeline else None
        except Exception:
            logger.warning("Errore istanziamento MemoryTimeline:\n" + traceback.format_exc())
            self.timeline = None

        try:
            self.ltm = self.long_term_memory_manager(self.state) if self.long_term_memory_manager else None
        except Exception:
            logger.warning("Errore istanziamento LongTermMemoryManager:\n" + traceback.format_exc())
            self.ltm = None

        try:
            self.dreamer = self.dream_generator(self.state) if self.dream_generator else None
        except Exception:
            logger.warning("Errore istanziamento DreamGenerator:\n" + traceback.format_exc())
            self.dreamer = None

        try:
            self.journal = self.life_journal(self.state) if self.life_journal else None
        except Exception:
            logger.warning("Errore istanziamento LifeJournal:\n" + traceback.format_exc())
            self.journal = None

        try:
            self.att_manager = self.attention_manager(self.state) if self.attention_manager else None
        except Exception:
            logger.warning("Errore istanziamento AttentionManager:\n" + traceback.format_exc())
            self.att_manager = None

        try:
            self.emotion = self.emotion_engine(self.state) if self.emotion_engine else None
        except Exception:
            logger.warning("Errore istanziamento EmotionEngine:\n" + traceback.format_exc())
            self.emotion = None

        try:
            self.motivation = self.motivational_engine(self.state) if self.motivational_engine else None
        except Exception:
            logger.warning("Errore istanziamento MotivationalEngine:\n" + traceback.format_exc())
            self.motivation = None

        try:
            self.self_reflector = self.self_reflection(self.state) if self.self_reflection else None
        except Exception:
            logger.warning("Errore istanziamento SelfReflection:\n" + traceback.format_exc())
            self.self_reflector = None

        try:
            self.persona = self.persona_manager(self.state) if self.persona_manager else None
        except Exception:
            logger.warning("Errore istanziamento PersonaManager:\n" + traceback.format_exc())
            self.persona = None

        try:
            self.planner_obj = self.planner(self.state) if self.planner else None
        except Exception:
            logger.warning("Errore istanziamento Planner:\n" + traceback.format_exc())
            self.planner_obj = None

    # ----- Stato -----
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = yaml.safe_load(f) or {}
                    # assicurati che chiavi importanti esistano
                    state.setdefault("timeline", [])
                    state.setdefault("memories", [])
                    state.setdefault("journal", [])
                    state.setdefault("last_daily_run", None)
                    return state
            except Exception:
                logger.exception("Errore caricamento stato interno; creo stato nuovo.")
                return {"timeline": [], "memories": [], "journal": [], "last_daily_run": None}
        else:
            logger.info("Stato interno non trovato. Creazione nuovo stato di base.")
            return {"timeline": [], "memories": [], "journal": [], "last_daily_run": None}

    def save_state(self):
        try:
            with open(self.state_file, "w") as f:
                yaml.safe_dump(self.state, f)
            logger.info("DailySelfLoop: stato salvato.")
        except Exception:
            logger.exception("Errore salvataggio stato interno.")

    # ----- Funzionalità principali -----
    def integrate_recent_experiences(self, lookback: int = 24):
        """
        Integra esperienze recenti dalla timeline nella memoria a lungo termine.
        lookback: ore da considerare recenti.
        """
        logger.info("Inizio integrazione esperienze recenti nella LTM.")
        now = datetime.now()
        cutoff = now - timedelta(hours=lookback)
        # Prendi la timeline dal file o dall'istanza timeline se presente
        entries = []
        try:
            if self.timeline:
                entries = self.timeline.get_recent(100)
            else:
                entries = self.state.get("timeline", [])[-100:]
        except Exception:
            logger.warning("Impossibile ottenere timeline, fallback allo stato interno.")
            entries = self.state.get("timeline", [])[-100:]

        recent = []
        for e in entries:
            try:
                ts = datetime.fromisoformat(e.get("timestamp"))
            except Exception:
                # In caso di timestamp non formattato
                ts = now
            if ts >= cutoff:
                recent.append(e)

        if not recent:
            logger.info("Nessuna esperienza recente da integrare.")
            return

        # Inserimento in LTM se presente
        if self.ltm and hasattr(self.ltm, "store_experience"):
            for e in recent:
                try:
                    self.ltm.store_experience(e)
                except Exception:
                    logger.exception("Errore memorizzazione esperienza in LTM.")
        else:
            # fallback: append diretto allo stato
            for e in recent:
                self.state.setdefault("memories", []).append(e)
            logger.info("Esperienze aggiunte allo stato.memories (fallback LTM).")

        # segnala ai manager
        if self.att_manager and hasattr(self.att_manager, "update_focus_from_memories"):
            try:
                self.att_manager.update_focus_from_memories(recent)
            except Exception:
                logger.warning("AttenzioneManager non è riuscito ad aggiornare il focus.")
        if self.motivation and hasattr(self.motivation, "evaluate_drives"):
            try:
                self.motivation.evaluate_drives(recent)
            except Exception:
                logger.warning("MotivationalEngine: evaluate_drives fallito.")

        # salva stato
        self.save_state()
        logger.info(f"Integrazione completata: {len(recent)} esperienze integrate.")

    def nightly_dream_and_journal(self):
        """Processo notturno: genera sogni a partire dalla timeline / LTM e registra sul diario."""
        logger.info("Esecuzione ciclo onirico notturno.")
        # Prepara input per il dream generator
        inputs = None
        try:
            if self.ltm and hasattr(self.ltm, "retrieve_top_brainstorm"):
                inputs = self.ltm.retrieve_top_brainstorm(limit=20)
            elif self.timeline:
                inputs = self.timeline.get_recent(50)
            else:
                inputs = self.state.get("timeline", [])[-50:]
        except Exception:
            logger.warning("Impossibile ottenere input per il dream generator; fallback minimale.")
            inputs = self.state.get("timeline", [])[-20:]

        generated = None
        if self.dreamer and hasattr(self.dreamer, "generate_dreams"):
            try:
                generated = self.dreamer.generate_dreams(inputs)
            except Exception:
                logger.exception("DreamGenerator.generate_dreams fallito.")
                generated = None
        else:
            # Fallback: creare un dream stub
            generated = [{"timestamp": datetime.now().isoformat(), "content": "Sogno generico da default.", "importance": 2}]

        # Salva sogni nella timeline e nel diario
        for dream in generated:
            entry = {
                "timestamp": dream.get("timestamp", datetime.now().isoformat()),
                "content": dream.get("content", ""),
                "category": "dream",
                "importance": dream.get("importance", 2)
            }
            # aggiungi in timeline e in journal
            try:
                if self.timeline and hasattr(self.timeline, "add_experience"):
                    self.timeline.add_experience(entry["content"], category="dream", importance=entry["importance"])
                else:
                    self.state.setdefault("timeline", []).append(entry)
                if self.journal and hasattr(self.journal, "append_entry"):
                    self.journal.append_entry(entry)
                else:
                    self.state.setdefault("journal", []).append(entry)
            except Exception:
                logger.exception("Errore salvataggio sogno in timeline/journal (fallback).")

        # segnala cambiamenti emotivi / motivazionali
        try:
            if self.emotion and hasattr(self.emotion, "process_dreams"):
                self.emotion.process_dreams(generated)
            if self.motivation and hasattr(self.motivation, "process_dreams"):
                self.motivation.process_dreams(generated)
        except Exception:
            logger.warning("Errori nell'elaborazione sogni da parte di emotion/motivation.")

        self.save_state()
        logger.info(f"Ciclo onirico completato: {len(generated)} sogni generati.")

    def morning_reflection(self):
        """Processo mattutino: riflessione, piani, aggiornamento obiettivi."""
        logger.info("Esecuzione riflessione mattutina.")
        try:
            if self.self_reflector and hasattr(self.self_reflector, "reflect"):
                reflections = self.self_reflector.reflect()
                # salva riflessioni
                for r in (reflections or []):
                    self.state.setdefault("journal", []).append({"timestamp": datetime.now().isoformat(), "content": r, "category": "reflection"})
            else:
                # fallback: semplice sintesi degli ultimi eventi
                recent = (self.timeline.get_recent(10) if self.timeline else self.state.get("timeline", [])[-10:])
                summary = f"Sintesi mattutina: {len(recent)} eventi recenti."
                self.state.setdefault("journal", []).append({"timestamp": datetime.now().isoformat(), "content": summary, "category": "reflection"})
                logger.info("Fallback riflessione mattutina eseguita.")
        except Exception:
            logger.exception("Errore nella riflessione mattutina.")

        # Aggiorna planner / obiettivi
        try:
            if self.planner_obj and hasattr(self.planner_obj, "recalculate_plan"):
                self.planner_obj.recalculate_plan(self.state)
        except Exception:
            logger.warning("Planner: recalculate_plan fallito o non disponibile.")

        # Aggiorna emozioni e motivazione
        try:
            if self.emotion and hasattr(self.emotion, "update"):
                self.emotion.update()
            if self.motivation and hasattr(self.motivation, "update"):
                self.motivation.update()
        except Exception:
            logger.warning("Emotion/Motivation: update fallito in morning_reflection.")

        self.save_state()
        logger.info("Riflessione mattutina completata.")

    def consolidate_long_term_memory(self):
        """Richiama procedure di consolidamento nella LTM (se esistono)."""
        logger.info("Consolidamento memoria a lungo termine.")
        try:
            if self.ltm:
                if hasattr(self.ltm, "consolidate"):
                    self.ltm.consolidate()
                elif hasattr(self.ltm, "run_maintenance"):
                    self.ltm.run_maintenance()
                else:
                    logger.info("LTM presente ma senza metodo di consolidamento definito.")
            else:
                logger.info("LTM non disponibile; skipping consolidamento.")
        except Exception:
            logger.exception("Errore consolidamento LTM.")

    # ----- Metodo unico che esegue il ciclo giornaliero completo -----
    def run_daily_cycle(self):
        """
        Esegue una sequenza completa di operazioni giornaliere:
        1) Integra esperienze recenti in LTM
        2) Consolida la LTM
        3) Esegue riflessione mattutina / piani
        4) (In orario notturno) genera sogni e scrive sul diario
        Nota: la chiamata può essere schedulata esternamente o eseguita manualmente.
        """
        logger.info("Inizio run_daily_cycle.")
        try:
            # 1) integra esperienze
            self.integrate_recent_experiences(lookback=24)

            # 2) consolidamento LTM
            self.consolidate_long_term_memory()

            # 3) riflessione mattutina
            self.morning_reflection()

            # 4) generazione sogni notturni (se è l'ora o se si vuole forzare)
            # Qui non controlliamo l'orario: la funzione è idempotente e può essere
            # schedulata separatamente per la notte. Qui la invochiamo comunque per completezza.
            self.nightly_dream_and_journal()

            # aggiorna last run
            self.state["last_daily_run"] = datetime.now().isoformat()
            self.save_state()
            logger.info("run_daily_cycle completato con successo.")
        except Exception:
            logger.exception("Errore in run_daily_cycle.")

    # ----- Utility per integrazione con Scheduler esterno -----
    def schedule_daily(self, hour: int = 3, minute: int = 0):
        """
        Registra il ciclo giornaliero con lo scheduler passato in __init__.
        Per default la pianifica alle 03:00 (notte).
        """
        if not self.scheduler:
            logger.warning("Nessuno scheduler passato; schedule_daily non può essere completato.")
            return

        try:
            # scheduler deve avere add_daily_job(func, hour, minute)
            if hasattr(self.scheduler, "add_daily_job"):
                self.scheduler.add_daily_job(self.run_daily_cycle, hour, minute)
                logger.info(f"DailySelfLoop schedulato alle {hour:02d}:{minute:02d}.")
            else:
                logger.warning("Lo scheduler fornito non espone add_daily_job().")
        except Exception:
            logger.exception("Errore schedule_daily.")

# Esempio di esecuzione autonoma per test
if __name__ == "__main__":
    logger.info("Esecuzione diretta di DailySelfLoop per test.")
    dsl = DailySelfLoop()
    # Esegui ciclo una volta
    dsl.run_daily_cycle()
    logger.info("Test run_daily_cycle completato.")
