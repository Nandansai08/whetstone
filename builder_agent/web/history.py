import json
import sqlite3
import uuid
from datetime import datetime, timezone

from builder_agent import config


class BuildHistory:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.MEMORY_DB_PATH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS builds (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    output_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score INTEGER,
                    artifact TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    build_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    passed BOOLEAN NOT NULL,
                    issues TEXT NOT NULL,
                    exec_output TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(build_id) REFERENCES builds(id)
                )
            """)

    def create_build(self, request: str, output_type: str) -> str:
        build_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO builds (id, request, output_type, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (build_id, request, output_type, "running", created_at)
            )
        return build_id

    def update_build_status(
        self,
        build_id: str,
        status: str,
        score: int | None = None,
        artifact: str | None = None,
    ):
        with self._connect() as conn:
            conn.execute(
                "UPDATE builds SET status = ?, score = ?, artifact = ? "
                "WHERE id = ?",
                (status, score, artifact, build_id)
            )

    def add_attempt(
        self,
        build_id: str,
        subtask_id: str,
        iteration: int,
        code: str,
        score: int,
        passed: bool,
        issues: list[str],
        exec_output: str = "",
    ):
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO attempts (build_id, subtask_id, iteration, "
                "code, score, passed, issues, exec_output, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    build_id,
                    subtask_id,
                    iteration,
                    code,
                    score,
                    int(passed),
                    json.dumps(issues),
                    exec_output,
                    created_at,
                ),
            )

    def get_builds(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM builds ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_build(self, build_id: str) -> dict | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM builds WHERE id = ?", (build_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_attempts(self, build_id: str) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM attempts WHERE build_id = ? "
                "ORDER BY iteration ASC, id ASC",
                (build_id,),
            )
            res = []
            for r in rows:
                d = dict(r)
                d["issues"] = json.loads(d["issues"])
                res.append(d)
            return res

    def clear_history(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM attempts")
            conn.execute("DELETE FROM builds")
