"""SQLite-based vector search memory manager storing past build experiences."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

from builder_agent import config
from builder_agent.embedders import Embedder, get_embedder
from builder_agent.schemas import MemoryRecord

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vectorized_cosine_similarity(
    query_vec: list[float], embeddings_matrix: np.ndarray
) -> np.ndarray:
    q = np.array(query_vec)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(embeddings_matrix.shape[0])

    norms = np.linalg.norm(embeddings_matrix, axis=1)
    norms[norms == 0.0] = 1.0

    return np.dot(embeddings_matrix, q) / (norms * q_norm)


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
    """Manager for Whetstone's build-history database and vector embeddings."""

    def __init__(
        self,
        db_path: str | None = None,
        embedder: Embedder | None = None,
    ):
        """Initialize the database and embedding strategy.

        Args:
            db_path: Optional file path path override to the SQLite database.
            embedder: Optional Embedder protocol implementation override.
        """
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
        """Save a build experience record into the memory database.

        Args:
            record: Dataclass representation of the build task metadata.
        """
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
        """Search similar past builds matching the semantic query string.

        Args:
            query: Semantic search query string.
            k: Maximum number of records to return.
            record_type: Filter by task execution type ("subtask" or "plan").

        Returns:
            A list of matching past MemoryRecords, sorted by similarity descending.
        """
        k = k or config.MEMORY_TOP_K
        query_vec = self._embedder.embed(query)

        # Pass 1: Query only id and embedding
        with self._connect() as conn:
            if record_type:
                rows = conn.execute(
                    "SELECT id, embedding FROM memory WHERE record_type = ?",
                    (record_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, embedding FROM memory"
                ).fetchall()

        if not rows:
            return []

        candidate_ids = []
        embeddings = []
        for r_id, emb_json in rows:
            try:
                emb = json.loads(emb_json)
                candidate_ids.append(r_id)
                embeddings.append(emb)
            except Exception:
                continue

        if not candidate_ids:
            return []

        scored_ids: list[tuple[float, int]] = []
        if HAS_NUMPY:
            try:
                emb_matrix = np.array(embeddings)
                similarities = _vectorized_cosine_similarity(query_vec, emb_matrix)
                for idx, sim in enumerate(similarities):
                    if sim >= config.MEMORY_MIN_SIMILARITY:
                        scored_ids.append((float(sim), candidate_ids[idx]))
            except Exception:
                # Fallback to pure Python on unexpected numpy error
                for idx, emb in enumerate(embeddings):
                    sim = _cosine_similarity(query_vec, emb)
                    if sim >= config.MEMORY_MIN_SIMILARITY:
                        scored_ids.append((sim, candidate_ids[idx]))
        else:
            for idx, emb in enumerate(embeddings):
                sim = _cosine_similarity(query_vec, emb)
                if sim >= config.MEMORY_MIN_SIMILARITY:
                    scored_ids.append((sim, candidate_ids[idx]))

        scored_ids.sort(key=lambda x: x[0], reverse=True)
        top_k_candidates = scored_ids[:k]

        if not top_k_candidates:
            return []

        top_ids = [item[1] for item in top_k_candidates]

        # Pass 2: Fetch full records only for the matched top-k IDs
        placeholders = ",".join("?" for _ in top_ids)
        query_sql = f"""
            SELECT request, output_type, subtask_desc, failures,
                   fix_summary, final_code, embedding, record_type, id
            FROM memory WHERE id IN ({placeholders})
        """
        with self._connect() as conn:
            full_rows = conn.execute(query_sql, top_ids).fetchall()

        records_map = {}
        for row in full_rows:
            rec_id = row[8]
            try:
                record = MemoryRecord(
                    request=row[0],
                    output_type=row[1],
                    subtask_desc=row[2],
                    failures=json.loads(row[3]),
                    fix_summary=row[4],
                    final_code=row[5],
                    embedding=json.loads(row[6]),
                    record_type=row[7],
                )
                records_map[rec_id] = record
            except Exception:
                continue

        # Re-sort to preserve exact similarity order from Pass 1
        ordered_records = []
        for _, rec_id in top_k_candidates:
            if rec_id in records_map:
                ordered_records.append(records_map[rec_id])

        return ordered_records

    def list_records(
        self, record_type: str | None = None,
    ) -> list[dict]:
        """List summary info for all records stored in the memory database.

        Args:
            record_type: Filter by record type block.

        Returns:
            A list of dictionary records containing primary metadata key values.
        """
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
        """Fetch a specific memory record by its primary ID.

        Args:
            record_id: Database key ID.

        Returns:
            The memory record dictionary, or None if not found.
        """
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
        """Wipe all stored records from the memory table.

        Returns:
            The count of deleted memory records.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memory")
            return cursor.rowcount
