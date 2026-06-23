from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from builder_agent import config
from builder_agent.embedders import Embedder, get_embedder
from builder_agent.schemas import MemoryRecord


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request TEXT NOT NULL,
    output_type TEXT NOT NULL,
    subtask_desc TEXT NOT NULL,
    failures TEXT NOT NULL,
    fix_summary TEXT NOT NULL,
    final_code TEXT NOT NULL,
    embedding TEXT NOT NULL,
    record_type TEXT NOT NULL DEFAULT 'subtask',
    created_at TEXT NOT NULL
)
"""

_MIGRATE_RECORD_TYPE = (
    "ALTER TABLE memory ADD COLUMN record_type TEXT NOT NULL DEFAULT 'subtask'"
)


class Memory:
    def __init__(
        self,
        db_path: str | None = None,
        embedder: Embedder | None = None,
    ):
        self._db_path = db_path or config.MEMORY_DB_PATH
        self._embedder = embedder or get_embedder(config.EMBEDDER)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(memory)").fetchall()
        }
        if "record_type" not in cols:
            conn.execute(_MIGRATE_RECORD_TYPE)

    def store(self, record: MemoryRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory "
                "(request, output_type, subtask_desc, failures, "
                "fix_summary, final_code, embedding, record_type, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.request,
                    record.output_type,
                    record.subtask_desc,
                    json.dumps(record.failures),
                    record.fix_summary,
                    record.final_code,
                    json.dumps(record.embedding),
                    record.record_type,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        record_type: str | None = None,
    ) -> list[MemoryRecord]:
        k = k or config.MEMORY_TOP_K
        query_vec = self._embedder.embed(query)

        with self._connect() as conn:
            if record_type:
                rows = conn.execute(
                    "SELECT request, output_type, subtask_desc, failures, "
                    "fix_summary, final_code, embedding, record_type "
                    "FROM memory WHERE record_type = ?",
                    (record_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT request, output_type, subtask_desc, failures, "
                    "fix_summary, final_code, embedding, record_type "
                    "FROM memory"
                ).fetchall()

        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            embedding = json.loads(row[6])
            sim = _cosine_similarity(query_vec, embedding)
            if sim < config.MEMORY_MIN_SIMILARITY:
                continue
            record = MemoryRecord(
                request=row[0],
                output_type=row[1],
                subtask_desc=row[2],
                failures=json.loads(row[3]),
                fix_summary=row[4],
                final_code=row[5],
                embedding=embedding,
                record_type=row[7],
            )
            scored.append((sim, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def list_records(
        self, record_type: str | None = None,
    ) -> list[dict]:
        with self._connect() as conn:
            if record_type:
                rows = conn.execute(
                    "SELECT id, request, output_type, record_type, "
                    "created_at FROM memory WHERE record_type = ? "
                    "ORDER BY created_at DESC",
                    (record_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, request, output_type, record_type, "
                    "created_at FROM memory ORDER BY created_at DESC"
                ).fetchall()
        return [
            {
                "id": r[0], "request": r[1], "output_type": r[2],
                "record_type": r[3], "created_at": r[4],
            }
            for r in rows
        ]

    def get_record(self, record_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, request, output_type, subtask_desc, failures, "
                "fix_summary, final_code, record_type, created_at "
                "FROM memory WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "request": row[1],
            "output_type": row[2],
            "subtask_desc": row[3],
            "failures": json.loads(row[4]),
            "fix_summary": row[5],
            "final_code": row[6],
            "record_type": row[7],
            "created_at": row[8],
        }

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memory")
            return cursor.rowcount
