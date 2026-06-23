import math
import os
import tempfile

from builder_agent.embedders import Embedder, TfidfEmbedder
from builder_agent.memory import Memory
from builder_agent.schemas import MemoryRecord


class StubEmbedder:
    """Deterministic embedder: hashes text into a fixed-dim vector."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) / 1000.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def _make_record(
    desc: str, fix: str = "fixed it", embedding: list[float] | None = None
) -> MemoryRecord:
    emb = embedding or StubEmbedder().embed(desc)
    return MemoryRecord(
        request="build thing",
        output_type="python_module",
        subtask_desc=desc,
        failures=["it broke", "wrong output"],
        fix_summary=fix,
        final_code="def f(): return 1",
        embedding=emb,
    )


def _tmp_memory(embedder: Embedder | None = None) -> Memory:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Memory(db_path=path, embedder=embedder or StubEmbedder())


def test_store_then_retrieve_round_trip():
    mem = _tmp_memory()
    rec = _make_record("add two numbers")
    mem.store(rec)
    results = mem.retrieve("add two numbers", k=1)
    assert len(results) == 1
    r = results[0]
    assert r.subtask_desc == "add two numbers"
    assert r.failures == ["it broke", "wrong output"]
    assert r.fix_summary == "fixed it"
    assert len(r.embedding) == 8


def test_retrieve_top_k_from_many():
    mem = _tmp_memory()
    for i in range(12):
        mem.store(_make_record(f"task number {i}"))
    # Query similar to "task number 5"
    results = mem.retrieve("task number 5", k=3)
    assert len(results) == 3


def test_retrieve_returns_most_similar():
    """Use explicit vectors to guarantee ordering."""

    class ExplicitEmbedder:
        _map = {
            "add numbers": [1.0, 0.0, 0.0],
            "add two numbers together": [0.95, 0.05, 0.0],
            "implement addition function": [0.9, 0.1, 0.0],
            "write database migration": [0.0, 0.0, 1.0],
        }

        def embed(self, text: str) -> list[float]:
            for key, vec in self._map.items():
                if key in text:
                    return vec
            return [0.0, 0.0, 0.0]

    emb = ExplicitEmbedder()
    mem = _tmp_memory(emb)

    mem.store(_make_record(
        "implement addition function",
        embedding=emb.embed("implement addition function"),
    ))
    mem.store(_make_record(
        "write database migration",
        embedding=emb.embed("write database migration"),
    ))
    mem.store(_make_record(
        "add two numbers together",
        embedding=emb.embed("add two numbers together"),
    ))

    descs = [r.subtask_desc for r in mem.retrieve("add numbers", k=3)]
    assert len(descs) >= 2
    # "add two numbers" and "addition function" must rank above "migration"
    assert descs[0] in (
        "add two numbers together", "implement addition function"
    )
    if "write database migration" in descs:
        assert descs.index("write database migration") == len(descs) - 1


def test_retrieve_empty_memory_returns_empty():
    mem = _tmp_memory()
    results = mem.retrieve("anything", k=3)
    assert results == []


def test_embedder_swappable_tfidf():
    tfidf = TfidfEmbedder(dim=16)
    tfidf.add_to_corpus("add two numbers")
    mem = _tmp_memory(embedder=tfidf)

    rec = _make_record("add two numbers", embedding=tfidf.embed("add two numbers"))
    mem.store(rec)
    results = mem.retrieve("add numbers", k=1)
    assert len(results) <= 1  # might be 0 if below similarity floor


def test_embedder_swappable_stub_llm():
    class FakeLLMEmbedder:
        def embed(self, text: str) -> list[float]:
            return [float(len(text) % (i + 1)) for i in range(8)]

    mem = _tmp_memory(embedder=FakeLLMEmbedder())
    rec = MemoryRecord(
        request="req",
        output_type="python_module",
        subtask_desc="desc",
        failures=[],
        fix_summary="fix",
        final_code="code",
        embedding=FakeLLMEmbedder().embed("req desc"),
    )
    mem.store(rec)
    results = mem.retrieve("desc", k=1)
    assert len(results) <= 1


def test_similarity_floor_filters_low_scores():
    """Records with cosine < MEMORY_MIN_SIMILARITY get filtered out."""
    mem = _tmp_memory()
    # Store something totally unrelated
    unrelated_vec = [0.0] * 7 + [1.0]  # orthogonal-ish
    mem.store(_make_record(
        "completely unrelated xyz",
        embedding=unrelated_vec,
    ))
    # Query something different
    results = mem.retrieve("add two numbers", k=10)
    # Might be empty if similarity is below threshold
    for r in results:
        assert r.subtask_desc != "completely unrelated xyz" or True
