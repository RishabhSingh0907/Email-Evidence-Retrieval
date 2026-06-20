"""
embedding_model.py
------------------
Thin wrapper around SentenceTransformer that:
  - reads all config from .env (model name, dim, batch size)
  - applies the correct passage / query prefixes per model family
  - exposes embed_passages() for ingestion and embed_query() for retrieval

Import and use:
    from embedding_model import EmbeddingModel
    em = EmbeddingModel()                      # loads model on first call
    vectors = em.embed_passages(["text1", "text2"])
    qvec    = em.embed_query("who threatened Paul Camera?")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import dotenv
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ── Model-family prefix rules ─────────────────────────────────────────────────
# BGE: no prefix for passages; query gets a task instruction.
# E5:  passages prefixed with "passage: "; queries with "query: ".
# All others: no prefix (fallback).

_PASSAGE_PREFIX: dict[str, str] = {
    "bge": "",
    "e5":  "passage: ",
}

_QUERY_PREFIX: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5":  "query: ",
}


def _detect_family(model_name: str) -> str:
    """Returns 'bge', 'e5', or 'other' based on the model name."""
    lower = model_name.lower()
    if "bge" in lower:
        return "bge"
    if "e5" in lower:
        return "e5"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Configured from .env:
        EMBEDDING_MODEL       HuggingFace model name  (default: BAAI/bge-base-en-v1.5)
        EMBEDDING_DIM         Output dimension         (default: 768)
        EMBEDDING_BATCH_SIZE  Passages per batch       (default: 32)

    The underlying SentenceTransformer is lazy-loaded on first use so that
    importing this module does not trigger a model download.
    """

    def __init__(self) -> None:
        self.model_name:  str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.dim:         int = int(os.getenv("EMBEDDING_DIM", "768"))
        self.batch_size:  int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self._family:     str = _detect_family(self.model_name)
        self._model: Optional[SentenceTransformer] = None

        logger.info(
            "EmbeddingModel configured | model=%s  dim=%d  batch=%d  family=%s",
            self.model_name, self.dim, self.batch_size, self._family,
        )

    # ── Lazy loader ──────────────────────────────────────────────────────────

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading SentenceTransformer: %s", self.model_name)
            t0 = time.perf_counter()
            self._model = SentenceTransformer(self.model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            elapsed = time.perf_counter() - t0
            logger.info(
                "Model loaded in %.2fs | actual_dim=%d (configured=%d)",
                elapsed, actual_dim, self.dim,
            )
            if actual_dim != self.dim:
                logger.warning(
                    "EMBEDDING_DIM mismatch: .env says %d but model outputs %d. "
                    "Update EMBEDDING_DIM in .env to match or the vector index "
                    "will reject embeddings.",
                    self.dim, actual_dim,
                )
        return self._model

    # ── Core encode ──────────────────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encodes a list of texts. Returns Python list[list[float]]."""
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine sim → dot product at query time
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    # ── Public API ───────────────────────────────────────────────────────────

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """
        Embed document/passage texts for ingestion.
        Applies the passage prefix appropriate for the model family.
        Call this in neo4j_ingestor.py.
        """
        prefix = _PASSAGE_PREFIX.get(self._family, "")
        prefixed = [f"{prefix}{t}" if prefix else t for t in texts]
        logger.debug("embed_passages: encoding %d texts", len(prefixed))
        return self._encode(prefixed)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single retrieval query.
        Applies the query prefix appropriate for the model family.
        Call this in the RAG pipeline / retriever.
        """
        prefix = _QUERY_PREFIX.get(self._family, "")
        prefixed = f"{prefix}{query}" if prefix else query
        logger.debug("embed_query: '%s'", query[:80])
        return self._encode([prefixed])[0]

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = "loaded" if self._model else "not loaded"
        return (
            f"EmbeddingModel(model={self.model_name!r}, "
            f"dim={self.dim}, batch={self.batch_size}, {loaded})"
        )