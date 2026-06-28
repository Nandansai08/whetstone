"""Text embedding generation strategies for past builds retrieval."""

from __future__ import annotations

import logging
import math
from typing import Protocol

from builder_agent.config import ModelConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol defining the interface for all text embedders."""

    def embed(self, text: str) -> list[float]:
        """Generate a numeric embedding vector for the given text.

        Args:
            text: Input string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...


class SentenceTransformerEmbedder:
    """Embedder using local sentence-transformers models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a specific Hugging Face model name."""
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        """Generate SentenceTransformer embedding for text."""
        self._load()
        return self._model.encode(text).tolist()


class TfidfEmbedder:
    """Lightweight, zero-dependency TF-IDF embedder."""

    def __init__(self, dim: int = 128):
        """Initialize the TF-IDF embedder.

        Args:
            dim: Dimension of the generated vector space.
        """
        self._dim = dim
        self._corpus: list[str] = []
        self._idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _refit(self) -> None:
        doc_count = len(self._corpus)
        if doc_count == 0:
            return
        df: dict[str, int] = {}
        for doc in self._corpus:
            seen = set(self._tokenize(doc))
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self._idf = {
            t: math.log((doc_count + 1) / (freq + 1)) + 1
            for t, freq in df.items()
        }

    def add_to_corpus(self, text: str) -> None:
        """Add document text to the TF-IDF corpus statistics and refit."""
        self._corpus.append(text)
        self._refit()

    def embed(self, text: str) -> list[float]:
        """Generate a TF-IDF embedding vector for the text."""
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1.0 / len(tokens)

        weighted: dict[str, float] = {}
        for t, freq in tf.items():
            weighted[t] = freq * self._idf.get(t, 1.0)

        vec = [0.0] * self._dim
        for t, w in weighted.items():
            idx = hash(t) % self._dim
            vec[idx] += w

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class LLMEmbedder:
    """An embedder that generates vectors using an LLM embedding endpoint."""

    def __init__(self, model: ModelConfig):
        """Initialize the LLMEmbedder with model configuration."""
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector using LLM API."""
        from builder_agent.llm import embed
        return embed(text, model=self._model)


def get_embedder(name: str = "sentence_transformer") -> Embedder:
    """Factory function to retrieve an embedder strategy by name.

    Args:
        name: Name of the embedder strategy ("sentence_transformer", "tfidf", "llm").

    Returns:
        An instance implementing the Embedder protocol.
    """
    if name == "sentence_transformer":
        try:
            import sentence_transformers  # noqa: F401
            return SentenceTransformerEmbedder()
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, "
                "falling back to TF-IDF embedder"
            )
            return TfidfEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "llm":
        model = ModelConfig(
            "openai", "nomic-embed-text",
            base_url="http://localhost:11434/v1",
        )
        return LLMEmbedder(model)
    raise ValueError(f"Unknown embedder: {name}")

