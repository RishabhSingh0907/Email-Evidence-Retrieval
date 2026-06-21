"""
app.py  —  Email Evidence Retrieval  |  GraphRAG Chat Interface
---------------------------------------------------------------
Run:   streamlit run src/chats/app.py

Retrieval strategy (layered):
  1. HybridCypherRetriever (vector + fulltext) — semantic + keyword
  2. VectorCypherRetriever  — pure semantic fallback
  3. Direct Cypher fallback — entity queries (email_id / thread_id / person)

Response formatting:
  Uses ResponseFormatter from response_formatter.py for deduplication,
  body cleaning, and citation blocks.

.env keys consumed:
  NEO4J_URI, NEO4J_USER, NEO4J_PASS
  EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import dotenv
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # Agentic_RAG/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "chats"))

dotenv.load_dotenv(ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chat_app")

# ── Neo4j + GraphRAG imports (lazy — validated at connect time) ───────────────
from neo4j import GraphDatabase                                    # noqa: E402
from neo4j_graphrag.embeddings.base import Embedder               # noqa: E402
from neo4j_graphrag.retrievers import (                           # noqa: E402
    HybridCypherRetriever,
    VectorCypherRetriever,
)

from src.data_processing.embedding_model import EmbeddingModel                        # noqa: E402
from src.chat.response_formatter import (                                  # noqa: E402
    ResponseFormatter,
    EmailRecord,
    CitationBlock,
    clean_body,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")

VECTOR_INDEX    = "email_embeddings"
FULLTEXT_INDEX  = "email_body_fulltext"
PERSON_INDEX    = "person_name_fulltext"
TOP_K           = 10

# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingModel → neo4j-graphrag Embedder adapter
# ─────────────────────────────────────────────────────────────────────────────
class GraphRAGEmbedder(Embedder):
    """
    Wraps our EmbeddingModel so it satisfies the neo4j-graphrag Embedder
    interface, which requires a single embed_query(text: str) -> list[float].
    """
    def __init__(self) -> None:
        super().__init__()
        self._em = EmbeddingModel()

    def embed_query(self, text: str) -> list[float]:
        return self._em.embed_query(text)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Cypher — graph expansion appended to both vector and hybrid search
# ─────────────────────────────────────────────────────────────────────────────
# `node` and `score` are in scope from the retriever.
RETRIEVAL_QUERY = """
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
# Full-text + entity Cypher — direct lookups (no vector needed)
# ─────────────────────────────────────────────────────────────────────────────
FT_EMAIL_QUERY = """
CALL db.index.fulltext.queryNodes($index_name, $query)
YIELD node, score
WITH node AS e, score
ORDER BY score DESC LIMIT $top_k
OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
OPTIONAL MATCH (e)-[:TO]->(to_p:Person)
OPTIONAL MATCH (e)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (e)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (e)-[:IN_THREAD]->(t:Thread)
RETURN
    e.email_id            AS email_id,
    e.thread_id           AS thread_id,
    e.date                AS date,
    e.subject             AS subject,
    e.body                AS body,
    e.quoted_reply_level  AS reply_level,
    sender.name           AS sender_name,
    sender.email          AS sender_email,
    collect(DISTINCT to_p.email)  AS to_recipients,
    collect(DISTINCT cc_p.email)  AS cc_recipients,
    collect(DISTINCT bcc_p.email) AS bcc_recipients,
    t.file_name           AS source_file,
    score                 AS similarity_score
"""

THREAD_QUERY = """
MATCH (e:Email {thread_id: $thread_id})-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
OPTIONAL MATCH (e)-[:TO]->(to_p:Person)
OPTIONAL MATCH (e)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (e)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (e)-[:REPLIED_TO]->(parent:Email)
WITH e, t, sender, parent,
     collect(DISTINCT to_p.email)  AS to_r,
     collect(DISTINCT cc_p.email)  AS cc_r,
     collect(DISTINCT bcc_p.email) AS bcc_r
ORDER BY e.date ASC
RETURN
    e.email_id            AS email_id,
    e.thread_id           AS thread_id,
    e.date                AS date,
    e.subject             AS subject,
    e.body                AS body,
    e.quoted_reply_level  AS reply_level,
    sender.name           AS sender_name,
    sender.email          AS sender_email,
    to_r                  AS to_recipients,
    cc_r                  AS cc_recipients,
    bcc_r                 AS bcc_recipients,
    parent.email_id       AS replies_to,
    t.file_name           AS source_file,
    1.0                   AS similarity_score
"""

PERSON_EMAIL_QUERY = """
MATCH (p:Person)
WHERE p.email = toLower($person_email)
   OR toLower(p.name) CONTAINS toLower($person_name)
MATCH (e:Email)
WHERE (p)-[:SENT]->(e)
   OR (e)-[:TO]->(p)
   OR (e)-[:CC]->(p)
   OR (e)-[:BCC]->(p)
OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
OPTIONAL MATCH (e)-[:TO]->(to_p:Person)
OPTIONAL MATCH (e)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (e)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (e)-[:IN_THREAD]->(t:Thread)
WITH DISTINCT e, sender,
     collect(DISTINCT to_p.email)  AS to_r,
     collect(DISTINCT cc_p.email)  AS cc_r,
     collect(DISTINCT bcc_p.email) AS bcc_r,
     t,
     CASE WHEN (p)-[:SENT]->(e)  THEN 'SENDER'
          WHEN (e)-[:BCC]->(p)   THEN 'BCC'
          WHEN (e)-[:CC]->(p)    THEN 'CC'
          ELSE 'TO' END AS role
ORDER BY e.date DESC
RETURN
    e.email_id            AS email_id,
    e.thread_id           AS thread_id,
    e.date                AS date,
    e.subject             AS subject,
    e.body                AS body,
    e.quoted_reply_level  AS reply_level,
    sender.name           AS sender_name,
    sender.email          AS sender_email,
    to_r                  AS to_recipients,
    cc_r                  AS cc_recipients,
    bcc_r                 AS bcc_recipients,
    t.file_name           AS source_file,
    role                  AS role_in_email,
    1.0                   AS similarity_score
LIMIT 50
"""


# ─────────────────────────────────────────────────────────────────────────────
# Query router — detects intent from the user's question
# ─────────────────────────────────────────────────────────────────────────────
_EMAIL_ID_RE  = re.compile(r"\b(\d{2}-\d{2}-\d{4}_\d{4}_\d{4})\b")
_THREAD_ID_RE = re.compile(r"\b(\d{2}-\d{2}-\d{4}_\d{4})\b")
_EMAIL_ADDR   = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]+")


def detect_intent(query: str) -> dict:
    """
    Returns a dict describing how to route the query:
      { "mode": "thread" | "person" | "keyword" | "semantic",
        "thread_id": str | None,
        "email_id":  str | None,
        "person_email": str | None,
        "person_name":  str | None }
    """
    q = query.strip()

    # 1. Explicit email_id pattern
    m = _EMAIL_ID_RE.search(q)
    if m:
        return {"mode": "thread", "email_id": m.group(1),
                "thread_id": "_".join(m.group(1).split("_")[:2]),
                "person_email": None, "person_name": None}

    # 2. Thread id pattern  (date_time without message index)
    m = _THREAD_ID_RE.search(q)
    if m:
        return {"mode": "thread", "thread_id": m.group(1), "email_id": None,
                "person_email": None, "person_name": None}

    # 3. Email address in query
    m = _EMAIL_ADDR.search(q)
    if m:
        return {"mode": "person", "person_email": m.group(0), "person_name": "",
                "thread_id": None, "email_id": None}

    # 4. "emails from/involving/sent by <Name>" patterns
    name_pat = re.search(
        r"(?:from|by|involving|sent by|email[s]? (?:from|by)|communication[s]? (?:from|with))\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        q, re.IGNORECASE
    )
    if name_pat:
        return {"mode": "person", "person_name": name_pat.group(1),
                "person_email": "", "thread_id": None, "email_id": None}

    # 5. Keyword-heavy short queries → fulltext
    words = q.split()
    if len(words) <= 5:
        return {"mode": "keyword", "thread_id": None, "email_id": None,
                "person_email": None, "person_name": None}

    # 6. Default → semantic (hybrid vector + fulltext)
    return {"mode": "semantic", "thread_id": None, "email_id": None,
            "person_email": None, "person_name": None}


# ─────────────────────────────────────────────────────────────────────────────
# Graph connection + retriever setup (cached per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_driver():
    logger.info("Connecting to Neo4j: %s", NEO4J_URI)
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


@st.cache_resource(show_spinner=False)
def get_embedder() -> GraphRAGEmbedder:
    logger.info("Loading embedding model …")
    return GraphRAGEmbedder()


@st.cache_resource(show_spinner=False)
def get_hybrid_retriever() -> Optional[HybridCypherRetriever]:
    """Returns HybridCypherRetriever if both indexes exist, else None."""
    try:
        driver  = get_driver()
        embedder = get_embedder()
        retriever = HybridCypherRetriever(
            driver             = driver,
            vector_index_name  = VECTOR_INDEX,
            fulltext_index_name = FULLTEXT_INDEX,
            retrieval_query    = RETRIEVAL_QUERY,
            embedder           = embedder,
        )
        logger.info("HybridCypherRetriever ready.")
        return retriever
    except Exception as e:
        logger.warning("HybridCypherRetriever unavailable: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def get_vector_retriever() -> Optional[VectorCypherRetriever]:
    """Fallback: vector-only retriever."""
    try:
        driver   = get_driver()
        embedder = get_embedder()
        retriever = VectorCypherRetriever(
            driver          = driver,
            index_name      = VECTOR_INDEX,
            retrieval_query = RETRIEVAL_QUERY,
            embedder        = embedder,
        )
        logger.info("VectorCypherRetriever ready.")
        return retriever
    except Exception as e:
        logger.warning("VectorCypherRetriever unavailable: %s", e)
        return None


def ensure_fulltext_indexes(driver) -> None:
    """Creates fulltext indexes if they don't exist yet."""
    with driver.session() as s:
        s.run("""
        CREATE FULLTEXT INDEX email_body_fulltext IF NOT EXISTS
        FOR (e:Email) ON EACH [e.body, e.subject]
        """)
        s.run("""
        CREATE FULLTEXT INDEX person_name_fulltext IF NOT EXISTS
        FOR (p:Person) ON EACH [p.name]
        """)
    logger.info("Fulltext indexes ensured.")


# ─────────────────────────────────────────────────────────────────────────────
# Core search dispatcher
# ─────────────────────────────────────────────────────────────────────────────
def run_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Routes the query to the appropriate retrieval strategy and returns
    a flat list of raw record dicts.
    """
    driver  = get_driver()
    intent  = detect_intent(query)
    mode    = intent["mode"]
    logger.info("Query intent: %s — '%s'", mode, query)
    records: list[dict] = []

    # ── Thread / single email lookup ──────────────────────────────────────
    if mode == "thread":
        thread_id = intent.get("thread_id")
        if thread_id:
            with driver.session() as s:
                result = s.run(THREAD_QUERY, {"thread_id": thread_id})
                records = [dict(r) for r in result]
                logger.info("Thread query → %d records", len(records))
        return records

    # ── Person lookup ─────────────────────────────────────────────────────
    if mode == "person":
        params = {
            "person_email": intent.get("person_email") or "",
            "person_name":  intent.get("person_name") or "",
        }
        with driver.session() as s:
            result = s.run(PERSON_EMAIL_QUERY, params)
            records = [dict(r) for r in result]
            logger.info("Person query → %d records", len(records))
        return records

    # ── Keyword / semantic — try hybrid first, fall back to vector ────────
    if mode in ("keyword", "semantic"):
        hybrid = get_hybrid_retriever()
        if hybrid:
            try:
                raw = hybrid.search(query_text=query, top_k=top_k)
                records = [dict(item.content) if hasattr(item.content, 'items')
                          else _parse_retriever_item(item)
                          for item in raw.items]
                logger.info("Hybrid search → %d results", len(records))
                if records:
                    return records
            except Exception as e:
                logger.warning("Hybrid search failed: %s", e)

        vector = get_vector_retriever()
        if vector:
            try:
                raw = vector.search(query_text=query, top_k=top_k)
                records = [_parse_retriever_item(item) for item in raw.items]
                logger.info("Vector search → %d results", len(records))
                if records:
                    return records
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        # Last resort: fulltext on body
        logger.info("Falling back to fulltext search")
        with driver.session() as s:
            result = s.run(
                FT_EMAIL_QUERY,
                {"index_name": FULLTEXT_INDEX, "query": query, "top_k": top_k},
            )
            records = [dict(r) for r in result]
            logger.info("Fulltext fallback → %d records", len(records))

    return records


def _parse_retriever_item(item) -> dict:
    """
    Extracts a plain dict from a RetrieverResultItem, which may wrap the
    content as a neo4j Record, a dict, or a string depending on version.
    """
    content = item.content
    if isinstance(content, dict):
        return content
    if hasattr(content, "data"):          # neo4j.Record
        return content.data()
    if hasattr(content, "keys"):          # neo4j.Record alternate
        return dict(zip(content.keys(), content.values()))
    # Fallback: treat as string body only
    return {"body": str(content), "email_id": "", "thread_id": "",
            "date": "", "subject": "", "sender_email": "", "sender_name": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def setup_page():
    st.set_page_config(
        page_title="Email Evidence Retrieval",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    /* ── Base palette ── */
    :root {
        --bg:          #0f1117;
        --surface:     #1a1d27;
        --surface-2:   #22263a;
        --border:      #2e3347;
        --accent:      #4f8ef7;
        --accent-dim:  #1e3a6e;
        --warn:        #f5a623;
        --warn-dim:    #3d2a0a;
        --text:        #d4d8e8;
        --text-dim:    #8890a8;
        --green:       #3ecf6e;
        --green-dim:   #0d2e1c;
        --red:         #e05252;
        --mono:        'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ── Global ── */
    .stApp { background: var(--bg); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: var(--mono);
        font-size: 0.875rem;
    }

    /* ── Email card ── */
    .email-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s;
    }
    .email-card:hover { border-color: var(--accent); }
    .email-card.has-bcc { border-left: 3px solid var(--warn); }

    /* ── Email header row ── */
    .email-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1.5rem;
        font-size: 0.78rem;
        color: var(--text-dim);
        font-family: var(--mono);
        margin-bottom: 0.6rem;
    }
    .email-meta .label { color: var(--accent); font-weight: 600; }
    .bcc-pill {
        background: var(--warn-dim);
        color: var(--warn);
        border: 1px solid var(--warn);
        border-radius: 4px;
        padding: 0 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* ── Subject line ── */
    .email-subject {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.5rem;
    }

    /* ── Body text ── */
    .email-body {
        font-size: 0.83rem;
        color: var(--text);
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
        border-top: 1px solid var(--border);
        padding-top: 0.6rem;
        margin-top: 0.4rem;
    }

    /* ── Citation block ── */
    .citation-block {
        background: var(--green-dim);
        border: 1px solid var(--green);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        font-family: var(--mono);
        font-size: 0.75rem;
        color: var(--green);
    }
    .citation-block.bcc-cite {
        background: var(--warn-dim);
        border-color: var(--warn);
        color: var(--warn);
    }

    /* ── Result summary banner ── */
    .result-banner {
        background: var(--accent-dim);
        border: 1px solid var(--accent);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        color: var(--text);
        margin-bottom: 1rem;
        font-family: var(--mono);
    }

    /* ── Intent badge ── */
    .intent-badge {
        display: inline-block;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1px 8px;
        font-size: 0.7rem;
        color: var(--text-dim);
        font-family: var(--mono);
        margin-bottom: 0.75rem;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-dim);
        font-size: 0.9rem;
    }

    /* Streamlit expander tweak */
    .streamlit-expanderHeader {
        font-family: var(--mono);
        font-size: 0.82rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_email_card(email: EmailRecord, index: int, show_body: bool = True) -> None:
    bcc_class = "has-bcc" if email.has_bcc else ""
    st.markdown(f'<div class="email-card {bcc_class}">', unsafe_allow_html=True)

    # ── BCC warning pill ──
    bcc_pill = ""
    if email.has_bcc:
        bcc_pill = f'<span class="bcc-pill">⚠ BCC</span>'

    # ── Meta row ──
    sender_display = email.sender_name or email.sender_email
    if email.sender_name and email.sender_email and email.sender_name != email.sender_email:
        sender_display = f"{email.sender_name} &lt;{email.sender_email}&gt;"

    to_display = ", ".join(email.to_recipients) if email.to_recipients else "—"
    cc_display = ", ".join(email.cc_recipients) if email.cc_recipients else ""
    bcc_display = ", ".join(email.bcc_recipients) if email.bcc_recipients else ""

    cc_row  = f'<span><span class="label">CC</span> {cc_display}</span>' if cc_display else ""
    bcc_row = f'<span><span class="label">BCC</span> {bcc_display} {bcc_pill}</span>' if bcc_display else ""

    st.markdown(f"""
    <div class="email-meta">
      <span><span class="label">#{index}</span> {email.email_id}</span>
      <span><span class="label">Thread</span> {email.thread_id}</span>
      <span><span class="label">Date</span> {email.date or "—"}</span>
      <span><span class="label">From</span> {sender_display}</span>
      <span><span class="label">To</span> {to_display}</span>
      {cc_row}
      {bcc_row}
    </div>
    <div class="email-subject">{email.subject}</div>
    """, unsafe_allow_html=True)

    # ── Body (collapsible if long) ──
    if show_body and email.body_clean:
        body = email.body_clean
        PREVIEW_LEN = 600
        if len(body) > PREVIEW_LEN:
            preview = body[:PREVIEW_LEN].rsplit(" ", 1)[0] + " …"
            with st.expander("Read full body", expanded=False):
                st.markdown(
                    f'<div class="email-body">{_esc(body)}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="email-body">{_esc(preview)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="email-body">{_esc(body)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_citations(citations: list[CitationBlock]) -> None:
    if not citations:
        return
    with st.expander(f"📎 Evidence citations ({len(citations)})", expanded=False):
        for c in citations:
            bcc_cls = "bcc-cite" if c.bcc_flag else ""
            bcc_note = "  ⚠ BCC INVOLVED" if c.bcc_flag else ""
            st.markdown(
                f'<div class="citation-block {bcc_cls}">'
                f'email_id={c.email_id} &nbsp;|&nbsp; '
                f'thread={c.thread_id} &nbsp;|&nbsp; '
                f'date={c.date} &nbsp;|&nbsp; '
                f'from={c.sender} &nbsp;|&nbsp; '
                f'subject="{c.subject}"'
                f'{bcc_note}'
                f'</div>',
                unsafe_allow_html=True,
            )


def _esc(text: str) -> str:
    """Minimal HTML escaping for inline rendering."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def render_result_banner(summary: str) -> None:
    st.markdown(
        f'<div class="result-banner">🔎 {summary}</div>',
        unsafe_allow_html=True,
    )


def render_intent_badge(mode: str) -> None:
    icons = {
        "semantic": "🧠 semantic",
        "keyword":  "🔤 keyword",
        "thread":   "🧵 thread",
        "person":   "👤 person",
    }
    label = icons.get(mode, mode)
    st.markdown(
        f'<div class="intent-badge">retrieval: {label}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## 🔍 Email Evidence")
        st.markdown("---")

        st.markdown("### Connection")
        uri_display = NEO4J_URI[:30] + "…" if len(NEO4J_URI) > 30 else NEO4J_URI
        st.code(uri_display or "Not configured", language=None)

        # Connection test
        if st.button("Test connection", use_container_width=True):
            try:
                driver = get_driver()
                with driver.session() as s:
                    count = s.run("MATCH (e:Email) RETURN count(e) AS n").single()["n"]
                st.success(f"✓ Connected — {count:,} emails")
            except Exception as ex:
                st.error(f"✗ {ex}")

        st.markdown("---")
        st.markdown("### Search settings")
        top_k = st.slider("Max results", 3, 30, TOP_K)

        show_body = st.toggle("Show email bodies", value=True)

        st.markdown("---")
        st.markdown("### Example queries")
        examples = [
            "Show thread 03-03-2020_1020",
            "All emails from asadushah@gmail.com",
            "Emails threatening legal action",
            "fiduciary duty demands",
            "IP ownership dispute",
            "BCC communications involving Sally",
            "Emails from Paul Camera",
            "dissolution ultimatum",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
                st.session_state["pending_query"] = ex

        st.markdown("---")
        st.markdown("### Actions")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("---")
        st.caption("Email Evidence Retrieval · GraphRAG · Neo4j Aura")

    return {"top_k": top_k, "show_body": show_body}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    setup_page()
    settings = render_sidebar()

    # ── Page header ──
    st.markdown("## Email Evidence Retrieval")
    st.markdown(
        "<span style='color:#8890a8;font-size:0.85rem'>"
        "GraphRAG · Neo4j Aura · Hybrid semantic + fulltext search"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "formatter" not in st.session_state:
        st.session_state["formatter"] = ResponseFormatter()

    # ── Bootstrap fulltext indexes (once per session) ──
    if "indexes_ensured" not in st.session_state:
        try:
            ensure_fulltext_indexes(get_driver())
            st.session_state["indexes_ensured"] = True
        except Exception as e:
            logger.warning("Could not ensure fulltext indexes: %s", e)

    formatter: ResponseFormatter = st.session_state["formatter"]

    # ── Replay history ──
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                # Re-render structured response from stored data
                _replay_response(msg, formatter, settings["show_body"])

    # ── Handle example query injection from sidebar ──
    pending = st.session_state.pop("pending_query", None)

    # ── Chat input ──
    user_input = st.chat_input("Ask about emails, people, threads, or topics …")
    query = pending or user_input

    if query:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run search
        with st.chat_message("assistant"):
            intent = detect_intent(query)
            render_intent_badge(intent["mode"])

            with st.spinner("Searching the graph …"):
                try:
                    raw_records = run_search(query, top_k=settings["top_k"])
                except Exception as ex:
                    st.error(f"Search error: {ex}")
                    logger.error("Search failed", exc_info=True)
                    raw_records = []

            if not raw_records:
                st.markdown(
                    '<div class="empty-state">'
                    "No emails found for this query.<br>"
                    "<small>Try different keywords, an email address, or a thread ID.</small>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "No results.",
                    "emails": [],
                    "citations": [],
                    "summary": "No results found.",
                    "intent_mode": intent["mode"],
                })
            else:
                emails = formatter.deduplicate(raw_records)
                response = formatter.format_email_list(
                    emails, query_context=query
                )

                render_result_banner(response.summary_header)

                for i, email in enumerate(response.emails, start=1):
                    render_email_card(email, i, show_body=settings["show_body"])

                render_citations(response.citations)

                # Store for replay
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response.text,
                    "emails": [_email_to_dict(e) for e in response.emails],
                    "citations": [_citation_to_dict(c) for c in response.citations],
                    "summary": response.summary_header,
                    "intent_mode": intent["mode"],
                })


def _replay_response(msg: dict, formatter: ResponseFormatter, show_body: bool) -> None:
    """Re-renders a stored assistant message from session state."""
    if not msg.get("emails"):
        st.markdown(msg.get("content", ""))
        return

    render_intent_badge(msg.get("intent_mode", ""))
    render_result_banner(msg.get("summary", ""))

    emails = [_dict_to_email(d) for d in msg["emails"]]
    for i, email in enumerate(emails, start=1):
        render_email_card(email, i, show_body=show_body)

    citations = [_dict_to_citation(d) for d in msg.get("citations", [])]
    render_citations(citations)


# ── Serialisation helpers (EmailRecord ↔ plain dict for session state) ───────

def _email_to_dict(e: EmailRecord) -> dict:
    return {
        "email_id":           e.email_id,
        "thread_id":          e.thread_id,
        "date":               e.date,
        "subject":            e.subject,
        "sender_name":        e.sender_name,
        "sender_email":       e.sender_email,
        "to_recipients":      e.to_recipients,
        "cc_recipients":      e.cc_recipients,
        "bcc_recipients":     e.bcc_recipients,
        "reply_level":        e.reply_level,
        "replies_to":         e.replies_to,
        "body_clean":         e.body_clean,
        "source_file":        e.source_file,
        "affiliated_persons": e.affiliated_persons,
    }


def _dict_to_email(d: dict) -> EmailRecord:
    return EmailRecord(
        email_id          = d.get("email_id", ""),
        thread_id         = d.get("thread_id", ""),
        date              = d.get("date", ""),
        subject           = d.get("subject", ""),
        sender_name       = d.get("sender_name", ""),
        sender_email      = d.get("sender_email", ""),
        to_recipients     = d.get("to_recipients", []),
        cc_recipients     = d.get("cc_recipients", []),
        bcc_recipients    = d.get("bcc_recipients", []),
        reply_level       = d.get("reply_level", 0),
        replies_to        = d.get("replies_to"),
        body_clean        = d.get("body_clean", ""),
        source_file       = d.get("source_file"),
        affiliated_persons= d.get("affiliated_persons", []),
    )


def _citation_to_dict(c: CitationBlock) -> dict:
    return {
        "email_id":  c.email_id,
        "thread_id": c.thread_id,
        "date":      c.date,
        "sender":    c.sender,
        "subject":   c.subject,
        "bcc_flag":  c.bcc_flag,
    }


def _dict_to_citation(d: dict) -> CitationBlock:
    return CitationBlock(
        email_id  = d.get("email_id", ""),
        thread_id = d.get("thread_id", ""),
        date      = d.get("date", ""),
        sender    = d.get("sender", ""),
        subject   = d.get("subject", ""),
        bcc_flag  = d.get("bcc_flag", False),
    )


if __name__ == "__main__":
    main()