# scheduler_nova.py
"""
Scheduler "vivente" per Nova.

Principi:
- Non solo timer: la frequenza di esecuzione dei job è influenzata dallo stato interno
  (emozioni, motivazioni, attenzione) in modo da creare comportamenti adattivi e non-meccanici.
- Supporta interrupt/eventi che possono far priorizzare certi job (es. messaggi in arrivo).
- Job eseguiti in thread worker separati per non bloccare il loop principale.
- API compatibili con il core (add_recurring_task, add_job, add_daily_job, add_once).
- Se riceve `state` (dict) salva metadati in state['scheduled_jobs'] per debugging/restore.

Nota: lo scheduler non importa NovaCore per evitare circolarità; `core` (o la sua reference) può essere passata al costruttore.
"""

import threading
import time
import random
import schedule
import traceback
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
from loguru import logger
import math

DEFAULT_POLL_INTERVAL = 0.5  # seconds between scheduler checks

# -----------------------
# Helper / fallback
# -----------------------
def _safe_call(func: Callable, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        logger.exception("Errore chiamando %s", getattr(func, "__name__", str(func)))

# -----------------------
# Job dataclass-like
# -----------------------
class AdaptiveJob:
    def __init__(
        self,
        func: Callable,
        base_interval: float,
        tag: Optional[str] = None,
        base_weight: float = 1.0,
        min_interval: float = 0.5,
        max_interval: float = 3600.0,
        metadata: Optional[dict] = None,
        job_type: str = "recurring"  # "recurring", "daily", "once", "micro"
    ):
        self.func = func
        self.tag = tag or f"job_{getattr(func, '__name__', int(time.time()))}_{int(time.time())}"
        self.base_interval = float(base_interval)
        self.base_weight = float(base_weight)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.metadata = metadata or {}
        self.job_type = job_type
        # Next run time: schedule immediately or after base_interval
        self.next_run = time.monotonic() + max(0.0, self.base_interval * random.uniform(0.8, 1.2))
        self.last_run: Optional[float] = None
        # dynamic weight updated each tick
        self.dynamic_weight = float(self.base_weight)

    def compute_effective_interval(self) -> float:
        """
        Convert dynamic_weight into an effective interval.
        Higher dynamic_weight => shorter interval (more often).
        We map weight -> interval via a smooth non-linear transform.
        """
        # avoid division by zero
        w = max(0.0, self.dynamic_weight)
        # mapping: interval = base_interval / (1 + alpha * w) but clamp to [min_interval, max_interval]
        alpha = 1.0
        eff = self.base_interval / (1.0 + alpha * w)
        eff = max(self.min_interval, min(self.max_interval, eff))
        # add a little jitter to avoid perfect periodicity
        eff *= random.uniform(0.95, 1.05)
        return eff

# -----------------------
# Scheduler "vivente"
# -----------------------
class SchedulerNova:
    def __init__(self, state: Optional[dict] = None, core: Optional[Any] = None, poll_interval: float = DEFAULT_POLL_INTERVAL):
        """
        state: optional shared dict (core.state) used to read emotions/motivation/attention and to persist scheduled_jobs meta.
        core: optional reference to NovaCore (used to call hooks or query modules).
        """
        self.state = state
        self.core = core
        self._jobs: Dict[str, AdaptiveJob] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._poll_interval = float(poll_interval or DEFAULT_POLL_INTERVAL)
        self._worker: Optional[threading.Thread] = None
        self._event_queue = []  # simple FIFO list of (timestamp, event_dict)
        logger.info("SchedulerNova inizializzato (core present: %s).", bool(core))
        self.start()

    # -----------------------
    # Start / Stop
    # -----------------------
    def start(self):
        with self._lock:
            if self._worker and self._worker.is_alive():
                logger.debug("Scheduler worker già attivo.")
                return
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._run_loop, name="SchedulerNovaWorker", daemon=True)
            self._worker.start()
            logger.info("SchedulerNova worker avviato.")

    def stop(self, join: bool = False):
        logger.info("SchedulerNova stop richiesto.")
        self._stop_event.set()
        if join and self._worker:
            self._worker.join(timeout=5.0)
            logger.info("SchedulerNova worker terminato.")

    # -----------------------
    # Job registration API (compatibile con core)
    # -----------------------
    def add_recurring_task(self, func: Callable, interval: float, tag: Optional[str] = None, base_weight: float = 1.0, metadata: Optional[dict] = None):
        """Alias usato dal core: recurring every `interval` seconds (base)."""
        return self.add_job(func, interval_seconds=interval, tag=tag, base_weight=base_weight, metadata=metadata)

    def add_job(self, func: Callable, interval_seconds: float, tag: Optional[str] = None, base_weight: float = 1.0, min_interval: float = 0.5, max_interval: float = 3600.0, metadata: Optional[dict] = None):
        """Registra un job adattivo ricorrente."""
        job = AdaptiveJob(func=func, base_interval=interval_seconds, tag=tag, base_weight=base_weight, min_interval=min_interval, max_interval=max_interval, metadata=metadata, job_type="recurring")
        with self._lock:
            self._jobs[job.tag] = job
            self._persist_jobs_meta()
            logger.info("Adaptive job registrato: tag=%s base_interval=%ss base_weight=%s", job.tag, job.base_interval, job.base_weight)
        return job

    def add_daily_job(self, func: Callable, hour: int, minute: int = 0, tag: Optional[str] = None, base_weight: float = 1.0, metadata: Optional[dict] = None):
        """
        Registra un job giornaliero (eseguito quando l'orologio locale raggiunge HH:MM).
        For simplicity we compute next_run as next HH:MM and then schedule as recurring with 24h base_interval.
        """
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        seconds_until = (target - now).total_seconds()
        base_interval = 24 * 3600
        # create a wrapper that will call func and then schedule next run in 24h normally via adaptive logic
        def _daily_wrapper():
            _safe_call(func)
        job = AdaptiveJob(func=_daily_wrapper, base_interval=base_interval, tag=tag, base_weight=base_weight, min_interval=60.0, max_interval=base_interval, metadata=metadata, job_type="daily")
        job.next_run = time.monotonic() + seconds_until
        with self._lock:
            self._jobs[job.tag] = job
            self._persist_jobs_meta()
            logger.info("Daily job registrato: tag=%s next_run=%s", job.tag, datetime.fromtimestamp(time.time() + seconds_until).isoformat())
        return job

    def add_once(self, func: Callable, delay_seconds: float, tag: Optional[str] = None, metadata: Optional[dict] = None):
        """Esegue una sola volta dopo delay_seconds."""
        def _run_and_remove():
            _safe_call(func)
            # removal will be done by remove_job after execution
        job = AdaptiveJob(func=_run_and_remove, base_interval=delay_seconds, tag=tag, base_weight=1.0, min_interval=0.0, max_interval=delay_seconds, metadata=metadata, job_type="once")
        job.next_run = time.monotonic() + delay_seconds
        with self._lock:
            self._jobs[job.tag] = job
            self._persist_jobs_meta()
            logger.info("One-shot job registrato: tag=%s in %ss", job.tag, delay_seconds)
        return job

    def remove_job(self, tag: str) -> bool:
        with self._lock:
            if tag in self._jobs:
                del self._jobs[tag]
                self._persist_jobs_meta()
                logger.info("Job rimosso: tag=%s", tag)
                return True
            logger.debug("remove_job: tag non trovato: %s", tag)
            return False

    def list_jobs(self) -> Dict[str, dict]:
        with self._lock:
            return {
                tag: {
                    "base_interval": j.base_interval,
                    "base_weight": j.base_weight,
                    "next_run_in": max(0.0, j.next_run - time.monotonic()),
                    "job_type": j.job_type,
                    "metadata": j.metadata
                } for tag, j in self._jobs.items()
            }

    def clear(self):
        with self._lock:
            self._jobs.clear()
            self._persist_jobs_meta()
            logger.info("Tutti i job rimossi dallo scheduler adattivo.")

    # -----------------------
    # Event / interrupt API
    # -----------------------
    def trigger_event(self, event: dict):
        """
        Inserisce un evento che può influenzare il comportamento immediato.
        Event dict può contenere keys come: 'type', 'tag', 'priority', 'payload'.
        """
        with self._lock:
            timestamp = time.monotonic()
            self._event_queue.append((timestamp, event))
            logger.debug("Evento triggerato: %s", event)

    # -----------------------
    # Core loop
    # -----------------------
    def _run_loop(self):
        logger.debug("SchedulerNova loop started (poll_interval=%s)", self._poll_interval)
        while not self._stop_event.is_set():
            try:
                now = time.monotonic()
                # 1) process events (highest priority)
                self._process_events()

                # 2) update dynamic weights based on state (emotions, motivations, attention)
                self._update_dynamic_weights()

                # 3) find jobs ready to run
                ready = []
                with self._lock:
                    for tag, job in list(self._jobs.items()):
                        if job.next_run <= now:
                            ready.append(job)

                # 4) execute ready jobs (each in its own thread)
                for job in ready:
                    self._execute_job(job)

                # 5) short sleep
                time.sleep(self._poll_interval)
            except Exception:
                logger.exception("Errore nel loop scheduler:\n%s", traceback.format_exc())
        logger.debug("SchedulerNova loop terminating.")

    def _execute_job(self, job: AdaptiveJob):
        def _run():
            logger.debug("Avvio job: %s", job.tag)
            try:
                _safe_call(job.func)
            except Exception:
                logger.exception("Errore esecuzione job %s", job.tag)
            finally:
                # update metadata and reschedule or remove for once jobs
                with self._lock:
                    job.last_run = time.monotonic()
                    if job.job_type == "once":
                        # remove it
                        if job.tag in self._jobs:
                            del self._jobs[job.tag]
                            logger.debug("One-shot job rimosso post-execution: %s", job.tag)
                    else:
                        # recompute dynamic interval and next_run
                        eff = job.compute_effective_interval()
                        job.next_run = time.monotonic() + eff
                        logger.debug("Job %s ri-schedulato in %ss (dynamic_weight=%s)", job.tag, round(eff, 3), round(job.dynamic_weight, 3))
                        # persist metadata
                        self._persist_jobs_meta()

        t = threading.Thread(target=_run, name=f"JobThread-{job.tag}", daemon=True)
        t.start()

    # -----------------------
    # Dynamic weight adjustments
    # -----------------------
    def _update_dynamic_weights(self):
        """
        Legge lo stato interno e aggiorna `job.dynamic_weight` per ogni job.
        Simple model:
            dynamic_weight = base_weight * (1 + alpha * arousal * motivation) * attention_modifier
        Where arousal/motivation derived from state with safe defaults.
        """
        # read state safely (avoid holding lock long)
        state_copy = {}
        try:
            if isinstance(self.state, dict):
                state_copy = dict(self.state)
        except Exception:
            state_copy = {}

        # extract simple signals with safe defaults
        emotions = state_copy.get("emotions", {}) or {}
        motivations = state_copy.get("motivations", {}) or {}
        attention = state_copy.get("attention", {}) or {}
        # derive numeric signals in [0,1] if present else defaults
        arousal = float(emotions.get("arousal", emotions.get("energy", 0.5) or 0.5))
        valence = float(emotions.get("valence", 0.5) or 0.5)
        motivation_score = float(motivations.get("engagement", motivations.get("drive", 0.5) or 0.5))
        # attention may specify focused_tag or intensity
        focused_tag = attention.get("focus_tag")
        attention_intensity = float(attention.get("intensity", 1.0) or 1.0)

        # clamp to plausible ranges
        arousal = max(0.0, min(2.0, arousal))
        motivation_score = max(0.0, min(2.0, motivation_score))
        attention_intensity = max(0.1, min(5.0, attention_intensity))

        # compute global modifier
        # small non-linear transform to avoid linear explosion
        alpha = 0.6
        global_modifier = 1.0 + alpha * (arousal * motivation_score)  # typically in [1.0, 1+alpha*4] if extremes
        # bias by valence (positive valence increases exploratory behaviors slightly)
        valence_mod = 1.0 + (valence - 0.5) * 0.2

        with self._lock:
            for tag, job in self._jobs.items():
                # base dynamic weight
                w = job.base_weight
                # apply global modifier
                w *= global_modifier * valence_mod
                # if attention focuses on this tag, boost strongly
                if focused_tag and tag == focused_tag:
                    w *= (1.0 + 2.0 * attention_intensity)
                # if job metadata declares 'attention_tag' matching focused_tag, boost moderately
                aj = job.metadata.get("attention_tag")
                if focused_tag and aj and aj == focused_tag:
                    w *= (1.0 + 1.0 * attention_intensity)
                # random small fluctuation to create unpredictability
                w *= random.uniform(0.95, 1.05)
                # clamp
                w = max(0.0, min(100.0, w))
                job.dynamic_weight = w

    # -----------------------
    # Event processing
    # -----------------------
    def _process_events(self):
        """
        Process queued events. Each event can:
        - increase priority of a specific tag
        - schedule an immediate call to a handler
        - create a one-shot high-priority job
        """
        events = []
        with self._lock:
            if not self._event_queue:
                return
            events = list(self._event_queue)
            self._event_queue.clear()

        for ts, event in events:
            try:
                etype = event.get("type", "generic")
                logger.debug("Processing event: %s", event)
                if etype == "focus_tag":
                    tag = event.get("tag")
                    # set attention focus in state (so weight update will boost it)
                    if isinstance(self.state, dict):
                        self.state.setdefault("attention", {})["focus_tag"] = tag
                        self.state["attention"]["intensity"] = float(event.get("intensity", 1.5))
                        logger.debug("State attention focus set to %s", tag)
                elif etype == "immediate_call":
                    handler = event.get("handler")
                    if callable(handler):
                        # spawn thread to run immediately
                        threading.Thread(target=_safe_call, args=(handler,), daemon=True).start()
                elif etype == "high_priority_once":
                    handler = event.get("handler")
                    tag = event.get("tag", f"event_once_{int(time.time())}")
                    # create one-shot that runs immediately
                    self.add_once(handler, delay_seconds=0.1, tag=tag, metadata={"source":"event"})
                else:
                    # generic: if payload contains 'tag', bump its dynamic_weight temporarily
                    tag = event.get("tag")
                    if tag and tag in self._jobs:
                        with self._lock:
                            self._jobs[tag].dynamic_weight *= float(event.get("multiplier", 2.0))
                            logger.debug("Boost applied to job %s by event multiplier %s", tag, event.get("multiplier", 2.0))
            except Exception:
                logger.exception("Errore processing event: %s", event)

    # -----------------------
    # Persistence helper
    # -----------------------
    def _persist_jobs_meta(self):
        if not isinstance(self.state, dict):
            return
        try:
            with self._lock:
                self.state.setdefault("scheduled_jobs", {})
                for tag, job in self._jobs.items():
                    self.state["scheduled_jobs"][tag] = {
                        "base_interval": job.base_interval,
                        "base_weight": job.base_weight,
                        "next_run": float(job.next_run - time.monotonic()),
                        "job_type": job.job_type,
                        "metadata": job.metadata,
                        "last_run": job.last_run
                    }
        except Exception:
            logger.exception("Errore durante persistenza scheduled_jobs nello state")

# -----------------------
# If run directly for a simple demo / test
# -----------------------
if __name__ == "__main__":
    import math

    logger.info("Esecuzione demo SchedulerNova")

    # demo state
    demo_state = {
        "emotions": {"arousal": 0.7, "valence": 0.6},
        "motivations": {"engagement": 0.9},
        "attention": {}
    }

    def quick_think():
        logger.info("quick_think() eseguito.")

    def heavy_job():
        logger.info("heavy_job() eseguito; simulo lavoro pesante")
        time.sleep(2)
        logger.info("heavy_job() completato.")

    sched = SchedulerNova(state=demo_state, core=None, poll_interval=0.5)
    sched.add_job(quick_think, interval_seconds=3, tag="quick_think", base_weight=1.2)
    sched.add_job(heavy_job, interval_seconds=10, tag="heavy_job", base_weight=0.8)
    sched.add_once(lambda: logger.info("one-shot firing now"), delay_seconds=5, tag="oneshot_demo")

    # simulate an external event after 8s that focuses attention to 'heavy_job'
    def trigger_focus():
        time.sleep(8)
        sched.trigger_event({"type":"focus_tag","tag":"heavy_job","intensity":3.0})
    threading.Thread(target=trigger_focus, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sched.stop(join=True)
        logger.info("Demo SchedulerNova terminato.")
