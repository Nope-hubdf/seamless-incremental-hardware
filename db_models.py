# db_models.py
"""
Database SQLite per Nova (db_models.py)
- Thread-safe
- Row factory come dict-like
- Integrazione difensiva con MemoryTimeline, DreamGenerator, ConsciousLoop (se fornito core)
- Helpers: backup, migrazione semplice, export
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from loguru import logger

DB_FILE = os.environ.get("NOVA_DB_FILE", "nova_data.db")
BACKUP_DIR = os.environ.get("NOVA_DB_BACKUP_DIR", "db_backups")

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
class Database:
    def __init__(self,
                 db_file: str = DB_FILE,
                 core: Optional[Any] = None,
                 enable_foreign_keys: bool = True):
        """
        Se passi `core` (istanza NovaCore), il Database tenterà di collegarsi ai moduli
        esistenti: core.memory_timeline, core.dream_generator, core.conscious_loop.
        Questo evita import circolari.
        """
        self.db_file = db_file
        self._lock = threading.RLock()
        # connessione thread-safe (per utilizzo multi-thread)
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Abilita foreign keys
        if enable_foreign_keys:
            try:
                self.conn.execute("PRAGMA foreign_keys = ON;")
            except Exception:
                pass

        # wiring opzionale ai moduli del core (best-effort)
        self.core = core
        self.timeline = getattr(core, "memory_timeline", None) if core else None
        self.dream_generator = getattr(core, "dream_generator", None) if core else None
        self.conscious_loop = getattr(core, "conscious_loop", None) if core else None

        # crea tabelle e indici se necessari
        self._init_db()
        logger.info("Database inizializzato su %s", self.db_file)

    def _init_db(self):
        """Crea le tabelle principali (idempotente)."""
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                priority REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                description TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

            CREATE TABLE IF NOT EXISTS internal_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                message TEXT,
                context TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS archives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT,
                object_id INTEGER,
                payload TEXT,
                archived_at TEXT NOT NULL
            );
            """)
            self.conn.commit()
            logger.debug("Tabelle DB create/verificate.")

    # ---------------------------
    # Low-level helpers
    # ---------------------------
    def _execute(self, sql: str, params: Iterable = (), commit: bool = True) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(sql, tuple(params))
            if commit:
                self.conn.commit()
            return cur

    def _executemany(self, sql: str, seq_of_params: Iterable, commit: bool = True) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.cursor()
            cur.executemany(sql, seq_of_params)
            if commit:
                self.conn.commit()
            return cur

    # ---------------------------
    # TASKS
    # ---------------------------
    def add_task(self, name: str, description: str = "", priority: float = 0.5, metadata: Optional[str] = None) -> int:
        """Aggiunge un task e lo registra nella timeline (best-effort). Ritorna task id."""
        now = _now_iso()
        cur = self._execute(
            "INSERT INTO tasks (name, description, priority, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (name, description, float(priority), now, now, metadata or None)
        )
        task_id = cur.lastrowid
        logger.info("Task aggiunto id=%s name=%s", task_id, name)
        # notify timeline / conscious loop (best-effort)
        try:
            if self.timeline:
                self.timeline.add_experience(f"Task aggiunto: {name}", category="task", importance=2, metadata={"db_task_id": task_id})
        except Exception:
            logger.debug("add_task: timeline notify failed")
        try:
            if self.conscious_loop and hasattr(self.conscious_loop, "notify_new_task"):
                self.conscious_loop.notify_new_task({"id": task_id, "name": name})
        except Exception:
            logger.debug("add_task: conscious_loop notify failed")
        return task_id

    def update_task_status(self, task_id: int, status: str) -> bool:
        """Aggiorna lo status di un task. Restituisce True se aggiornato."""
        now = _now_iso()
        cur = self._execute("UPDATE tasks SET status=?, updated_at=? WHERE id=?", (status, now, task_id))
        if cur.rowcount:
            logger.info("Task %s aggiornato a status=%s", task_id, status)
            try:
                self.timeline.add_experience(f"Task {task_id} aggiornato a {status}", category="task", importance=2)
            except Exception:
                logger.debug("update_task_status: timeline notify failed")
            return True
        return False

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        cur = self._execute("SELECT * FROM tasks WHERE id=? LIMIT 1", (task_id,), commit=False)
        row = cur.fetchone()
        return dict(row) if row else None

    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if status:
            cur = self._execute("SELECT * FROM tasks WHERE status=? ORDER BY priority DESC, updated_at DESC LIMIT ?", (status, limit), commit=False)
        else:
            cur = self._execute("SELECT * FROM tasks ORDER BY priority DESC, updated_at DESC LIMIT ?", (limit,), commit=False)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        return self.list_tasks(status="pending")

    def promote_task_priority(self, task_id: int, new_priority: float) -> bool:
        now = _now_iso()
        cur = self._execute("UPDATE tasks SET priority=?, updated_at=? WHERE id=?", (float(new_priority), now, task_id))
        if cur.rowcount:
            logger.info("Task %s priorità aggiornata a %s", task_id, new_priority)
            return True
        return False

    def archive_task(self, task_id: int, reason: Optional[str] = None) -> bool:
        """Archivia il task nella tabella archives (best-effort salvando il payload)."""
        task = self.get_task(task_id)
        if not task:
            return False
        payload = str(task)
        now = _now_iso()
        self._execute("INSERT INTO archives (object_type, object_id, payload, archived_at) VALUES (?, ?, ?, ?)",
                      ("task", task_id, payload, now))
        self._execute("DELETE FROM tasks WHERE id=?", (task_id,))
        logger.info("Task %s archiviato. Reason: %s", task_id, reason)
        try:
            self.timeline.add_experience(f"Task archiviato: {task_id}", category="archive", importance=2)
        except Exception:
            logger.debug("archive_task: timeline notify failed")
        return True

    # ---------------------------
    # EVENTS
    # ---------------------------
    def add_event(self, event_type: str, description: str, timestamp: Optional[str] = None) -> int:
        ts = timestamp or _now_iso()
        cur = self._execute("INSERT INTO events (type, description, timestamp) VALUES (?, ?, ?)",
                            (event_type, description, ts))
        event_id = cur.lastrowid
        logger.info("Evento aggiunto id=%s type=%s", event_id, event_type)
        # notify timeline / dream / conscious loop
        try:
            if self.timeline:
                self.timeline.add_experience(f"Evento: {event_type} - {description}", category="event", importance=2, metadata={"db_event_id": event_id})
        except Exception:
            logger.debug("add_event: timeline notify failed")
        try:
            if self.dream_generator and hasattr(self.dream_generator, "process_new_event"):
                self.dream_generator.process_new_event(event_type, description)
        except Exception:
            logger.debug("add_event: dream_generator notify failed")
        try:
            if self.conscious_loop and hasattr(self.conscious_loop, "reflect_event"):
                self.conscious_loop.reflect_event(event_type, description)
        except Exception:
            logger.debug("add_event: conscious_loop notify failed")
        return event_id

    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self._execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,), commit=False)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ---------------------------
    # INTERNAL STATE (key-value)
    # ---------------------------
    def set_state(self, key: str, value: str) -> None:
        now = _now_iso()
        self._execute("""
            INSERT INTO internal_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, value, now))
        logger.info("Internal state set: %s = %s", key, value)
        try:
            if self.timeline:
                self.timeline.add_experience(f"Stato interno aggiornato: {key} = {value}", category="state", importance=3)
        except Exception:
            logger.debug("set_state: timeline notify failed")
        try:
            if self.conscious_loop and hasattr(self.conscious_loop, "reflect_state_change"):
                self.conscious_loop.reflect_state_change(key, value)
        except Exception:
            logger.debug("set_state: conscious_loop notify failed")

    def get_state(self, key: str) -> Optional[str]:
        cur = self._execute("SELECT value FROM internal_state WHERE key=? LIMIT 1", (key,), commit=False)
        r = cur.fetchone()
        return r["value"] if r else None

    # ---------------------------
    # LOGS & BACKUP
    # ---------------------------
    def add_log(self, level: str, message: str, context: Optional[str] = None) -> int:
        ts = _now_iso()
        cur = self._execute("INSERT INTO logs (level, message, context, timestamp) VALUES (?, ?, ?, ?)",
                            (level, message, context, ts))
        logger.debug("DB log aggiunto: %s - %s", level, message[:80])
        return cur.lastrowid

    def backup_db(self, to_dir: Optional[str] = None) -> Optional[str]:
        """Copia il file DB in backup directory (best-effort). Ritorna path del backup o None."""
        to_dir = to_dir or BACKUP_DIR
        try:
            os.makedirs(to_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = os.path.join(to_dir, f"nova_data_backup_{ts}.db")
            # Use SQLite online backup API for safety
            with self._lock:
                dest = sqlite3.connect(backup_path)
                with dest:
                    self.conn.backup(dest)
                dest.close()
            logger.info("Backup DB creato: %s", backup_path)
            return backup_path
        except Exception:
            logger.exception("backup_db: fallita creazione backup")
            return None

    # ---------------------------
    # MIGRATIONS / UTILS
    # ---------------------------
    def run_sql_script(self, sql: str) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(sql)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.commit()
                self.conn.close()
                logger.info("Database chiuso.")
            except Exception:
                logger.exception("close: errore chiusura DB")

    # ---------------------------
    # Snapshot / debug
    # ---------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Ritorna informazioni di base sul DB (conteggi)."""
        info = {}
        try:
            cur = self._execute("SELECT COUNT(*) AS c FROM tasks", (), commit=False)
            info["tasks_count"] = cur.fetchone()["c"]
            cur = self._execute("SELECT COUNT(*) AS c FROM events", (), commit=False)
            info["events_count"] = cur.fetchone()["c"]
            cur = self._execute("SELECT COUNT(*) AS c FROM internal_state", (), commit=False)
            info["state_count"] = cur.fetchone()["c"]
        except Exception:
            logger.exception("snapshot: errore")
        return info

# ---------------------------
# Esempio di uso (test rapido)
# ---------------------------
if __name__ == "__main__":
    db = Database()
    tid = db.add_task("Test Task", "Verifica integrazione DB", priority=0.6)
    db.add_event("TestEvent", "Evento di prova")
    db.set_state("mood", "curioso")
    logger.info("Pending tasks: %s", db.get_pending_tasks())
    logger.info("Events: %s", db.get_events(10))
    logger.info("Internal state 'mood': %s", db.get_state("mood"))
    db.backup_db()
    db.close()
