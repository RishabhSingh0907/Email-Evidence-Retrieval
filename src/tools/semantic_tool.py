"""
src/tools/semantic_tool.py
--------------------------
Semantic search tool: HybridCypherRetriever (vector + fulltext) with
VectorCypherRetriever as fallback.

When to use (agent description drives this):
  - Conceptual / topic queries: "emails about IP threats", "fiduciary duty demands"
  - Tone / intent queries: "threatening emails", "ultimatums", "settlement offers"
  - NOT for known entities — use cypher_tools for those

Embedder: EmbeddingModel (BGE/E5, local, free) wrapped as GraphRAGEmbedder.
Indexes: email_embeddings (vector), email_body_fulltext (fulltext)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "chats"))

from agents.graph import driver as neo4j_driver              # noqa: E402
from data_processing.embedding_model import EmbeddingModel                   # noqa: E402
from chat.response_formatter import ResponseFormatter             # noqa: E402
from neo4j_graphrag.embeddings.base import Embedder          # noqa: E402
from neo4j_graphrag.retrievers import (                      # noqa: E402
    HybridCypherRetriever,
    VectorCypherRetriever,
)

logger    = logging.getLogger(__name__)
formatter = ResponseFormatter()

# ─────────────────────────────────────────────────────────────────────────────
# Embedder adapter
# ─────────────────────────────────────────────────────────────────────────────
class _GraphRAGEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        self._em = EmbeddingModel()

    def embed_query(self, text: str) -> list[float]:
        return self._em.embed_query(text)


# ─────────────────────────────────────────────────────────────────────────────
# Graph expansion Cypher — appended to retriever
# ─────────────────────────────────────────────────────────────────────────────
_RETRIEVAL_QUERY = """
OPTIONAL MATCH (sender:Person)-[:SENT]->(node)
OPTIONAL MATCH (node)-[:TO]->(to_p:Person)
OPTIONAL MATCH (node)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (node)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (node)-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (node)-[:REPLIED_TO]->(parent:Email)
RETURN
    node.email_id            AS email_id,
    node.thread_id           AS thread_id,
    node.date                AS date,
    node.subject             AS subject,
    node.body                AS body,
    node.quoted_reply_level  AS reply_level,
    sender.name              AS sender_name,
    sender.email             AS sender_email,
    collect(DISTINCT to_p.email)  AS to_recipients,
    collect(DISTINCT cc_p.email)  AS cc_recipients,
    collect(DISTINCT bcc_p.email) AS bcc_recipients,
    parent.email_id          AS replies_to,
    t.file_name              AS source_file,
    score                    AS similarity_score
"""

# ─────────────────────────────────────────────────────────────────────────────
# Lazy retriever initialisation
# ─────────────────────────────────────────────────────────────────────────────
_embedder:  Optional[_GraphRAGEmbedder]   = None
_hybrid:    Optional[HybridCypherRetriever] = None
_vector:    Optional[VectorCypherRetriever] = None


def _get_embedder() -> _GraphRAGEmbedder:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedder for semantic tool …")
        _embedder = _GraphRAGEmbedder()
    return _embedder


def _get_hybrid() -> Optional[HybridCypherRetriever]:
    global _hybrid
    if _hybrid is None:
        try:
            _hybrid = HybridCypherRetriever(
                driver              = neo4j_driver,
                vector_index_name   = "email_embeddings",
                fulltext_index_name = "email_body_fulltext",
                retrieval_query     = _RETRIEVAL_QUERY,
                embedder            = _get_embedder(),
            )
            logger.info("HybridCypherRetriever ready.")
        except Exception as e:
            logger.warning("HybridCypherRetriever init failed: %s", e)
    return _hybrid


def _get_vector() -> Optional[VectorCypherRetriever]:
    global _vector
    if _vector is None:
        try:
            _vector = VectorCypherRetriever(
                driver          = neo4j_driver,
                index_name      = "email_embeddings",
                retrieval_query = _RETRIEVAL_QUERY,
                embedder        = _get_embedder(),
            )
            logger.info("VectorCypherRetriever ready (fallback).")
        except Exception as e:
            logger.warning("VectorCypherRetriever init failed: %s", e)
    return _vector


def _parse_item(item) -> dict:
    """Normalise a RetrieverResultItem to a plain dict."""
    c = item.content
    if isinstance(c, dict):
        return c
    if hasattr(c, "data"):
        return c.data()
    if hasattr(c, "keys"):
        return dict(zip(c.keys(), c.values()))
    return {"body": str(c), "email_id": "", "thread_id": "",
            "date": "", "subject": "", "sender_email": "", "sender_name": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────────────────────────────────────
@tool
def semantic_email_search(query: str, top_k: int = 10) -> str:
    """
    Search email content semantically using vector similarity combined with
    fulltext keyword matching (hybrid search).

    Use for conceptual, topic-based, or intent-based queries such as:
    - "emails about IP threats"
    - "fiduciary duty demands"
    - "dissolution ultimatums"
    - "hidden agenda accusations"
    - "settlement proposals"
    - "emails mentioning fire sale"

    Do NOT use for queries where you know a specific person's email address,
    a thread_id, or an email_id — use the dedicated Cypher tools for those.

    top_k controls how many results to return (default 10, max 20).
    """
    logger.info("[tool] semantic_email_search query='%s' top_k=%d", query, top_k)
    top_k = min(int(top_k), 20)
    records: list[dict] = []

    # Try hybrid first
    hybrid = _get_hybrid()
    if hybrid:
        try:
            raw = hybrid.search(query_text=query, top_k=top_k)
            records = [_parse_item(item) for item in raw.items]
            logger.info("Hybrid search: %d results", len(records))
        except Exception as e:
            logger.warning("Hybrid search failed: %s", e)
            records = []

    # Vector fallback
    if not records:
        vector = _get_vector()
        if vector:
            try:
                raw = vector.search(query_text=query, top_k=top_k)
                records = [_parse_item(item) for item in raw.items]
                logger.info("Vector fallback: %d results", len(records))
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

    if not records:
        return f"No semantically similar emails found for: {query}"

    emails = formatter.deduplicate(records)
    return formatter.format_email_list(emails, query_context=query).text


SEMANTIC_TOOLS = [semantic_email_search]