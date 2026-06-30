import logging
from unittest.mock import patch

from builder_agent.embedders import (
    TfidfEmbedder,
    get_embedder,
)


def test_tfidf_embed_returns_vector():
    emb = TfidfEmbedder(dim=16)
    emb.add_to_corpus("hello world")
    vec = emb.embed("hello world")
    assert len(vec) == 16
    assert any(v != 0.0 for v in vec)


def test_tfidf_empty_text():
    emb = TfidfEmbedder(dim=8)
    vec = emb.embed("")
    assert vec == [0.0] * 8


def test_tfidf_normalized():
    import math
    emb = TfidfEmbedder(dim=16)
    emb.add_to_corpus("some document")
    vec = emb.embed("some document")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 0.001 or norm == 0.0


def test_get_embedder_tfidf():
    emb = get_embedder("tfidf")
    assert isinstance(emb, TfidfEmbedder)


def test_sentence_transformer_fallback_to_tfidf(caplog):
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with caplog.at_level(logging.WARNING):
            emb = get_embedder("sentence_transformer")
    assert isinstance(emb, TfidfEmbedder)
    assert "falling back" in caplog.text.lower()


def test_get_embedder_unknown_raises():
    try:
        get_embedder("nonexistent")
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)

def test_get_embedder_llm_default_base_url():
    emb = get_embedder("llm")
    assert emb._model.base_url == "http://localhost:11434/v1"
