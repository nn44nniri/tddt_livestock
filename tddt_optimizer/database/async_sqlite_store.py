from __future__ import annotations
from pathlib import Path
import queue
import threading
from typing import Any
from .sqlite_store import SQLiteStore


class AsyncSQLiteStore:
    """Single-writer SQLite logger backed by an independent CPU thread.

    The control loop enqueues log records and continues; the writer thread owns
    the SQLite connection and flushes records to disk. This avoids keeping the
    full WORK-OFFLINE report trace in memory and prevents SQLite cross-thread
    connection issues.
    """
    def __init__(self, path: str | Path, max_queue: int = 50000):
        self.path = Path(path)
        self.q: queue.Queue[tuple[str, tuple[Any, ...]] | None] = queue.Queue(maxsize=max_queue)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="tddt-sqlite-writer", daemon=True)
        self.thread.start()

    def _run(self):
        store = SQLiteStore(self.path)
        try:
            while True:
                item = self.q.get()
                if item is None:
                    self.q.task_done()
                    break
                method, args = item
                getattr(store, method)(*args)
                self.q.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                store.close()
            except Exception:
                pass

    def _put(self, method: str, *args: Any):
        if self.error is not None:
            raise RuntimeError("Async SQLite writer failed") from self.error
        self.q.put((method, args))

    def log_inner(self, row: dict):
        self._put("log_inner", row)

    def log_command(self, timestamp: str, mode: str, command: dict, safety_status: str = "OK"):
        self._put("log_command", timestamp, mode, command, safety_status)

    def log_growth(self, row: dict, status: str = "OK"):
        self._put("log_growth", row, status)

    def log_guidance(self, fattening_day: int, guidance: dict):
        self._put("log_guidance", fattening_day, guidance)

    def log_rl_update(self, update):
        self._put("log_rl_update", update)

    def log_sarg_scores(self, rows: list[dict]):
        self._put("log_sarg_scores", rows)

    def log_formal_link(self, relation_type: str, source_key: str, target_key: str, formula_name: str, payload: dict):
        self._put("log_formal_link", relation_type, source_key, target_key, formula_name, payload)

    def close(self):
        self.q.put(None)
        self.q.join()
        self.thread.join(timeout=60)
        if self.error is not None:
            raise RuntimeError("Async SQLite writer failed") from self.error
