"""
neo4j_ingestor.py
-----------------
Idempotent ingestion of canonicalized email thread JSON files into Neo4j Aura.
Embeddings are generated on-the-fly during ingestion (no separate step needed).

Pipeline per thread file:
  1. Collect all messages
  2. Batch-embed all message bodies in one model.encode() call
  3. Write Thread + Email + Person + Company nodes and all relationships

Usage:
  python neo4j_ingestor.py
  python neo4j_ingestor.py --dir data/processed/canonicalized_features

Dependencies:
  pip install neo4j sentence-transformers python-dotenv

.env keys consumed:
  NEO4J_URI, NEO4J_USER, NEO4J_PASS
  EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE   (see embedding_model.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import dotenv
from neo4j import GraphDatabase

from ..data_processing.embedding_model import EmbeddingModel

dotenv.load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neo4j_ingestor")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

DEFAULT_DIR = Path("data/processed/canonicalized_features")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_get(data: dict, key: str, default=None):
    return data.get(key, default) if isinstance(data, dict) else default


def make_person_id(name: str, email: str) -> str:
    """
    Canonical merge key for Person nodes.
    Email is preferred (stable, globally unique).
    Falls back to lowercased name if email is absent.
    Returns empty string when both are missing → caller skips.
    """
    if email and email.strip():
        return email.strip().lower()
    if name and name.strip():
        return name.strip().lower()
    return ""


def make_email_id(thread_id: str, message_id: str) -> str:
    """Scoped Email node key: '<thread_id>_<message_id>'."""
    return f"{thread_id}_{message_id}"


def build_embedding_text(msg: dict) -> str:
    """
    Constructs the plain-text passage that will be embedded.
    Subject + body gives the model enough context for semantic search.
    """
    subject = (safe_get(msg, "subject", "") or "").strip()
    body    = (safe_get(msg, "body",    "") or "").strip()
    return f"Subject: {subject}\n\n{body}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Schema — constraints & vector index
# ─────────────────────────────────────────────────────────────────────────────
def create_constraints(session) -> None:
    logger.info("Ensuring constraints …")
    stmts = [
        # Person: merged on person_id (email or name fallback)
        """CREATE CONSTRAINT person_id_unique IF NOT EXISTS
           FOR (p:Person) REQUIRE p.person_id IS UNIQUE""",
        # Email: merged on scoped email_id
        """CREATE CONSTRAINT email_id_unique IF NOT EXISTS
           FOR (e:Email) REQUIRE e.email_id IS UNIQUE""",
        # Thread: merged on thread_id
        """CREATE CONSTRAINT thread_id_unique IF NOT EXISTS
           FOR (t:Thread) REQUIRE t.thread_id IS UNIQUE""",
        # Company: merged on normalised name
        """CREATE CONSTRAINT company_name_unique IF NOT EXISTS
           FOR (c:Company) REQUIRE c.name IS UNIQUE""",
    ]
    for stmt in stmts:
        session.run(stmt)
    logger.info("Constraints ready.")


def create_vector_index(session, dim: int) -> None:
    """
    Creates the vector index on Email.embedding if it doesn't exist.
    dim is read from EMBEDDING_DIM in .env so it always matches the model.
    """
    logger.info("Ensuring vector index (dim=%d) …", dim)
    session.run(
        f"""
        CREATE VECTOR INDEX email_embeddings IF NOT EXISTS
        FOR (e:Email) ON (e.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
    )
    logger.info("Vector index ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Node merges
# ─────────────────────────────────────────────────────────────────────────────
def merge_thread(tx, thread_id: str, file_name: str) -> None:
    tx.run(
        """
        MERGE (t:Thread {thread_id: $thread_id})
        ON CREATE SET t.file_name = $file_name
        """,
        {"thread_id": thread_id, "file_name": file_name},
    )


def merge_email(tx, payload: dict) -> None:
    """
    Uses SET for all content fields so re-ingestion picks up canonicalization
    updates. embedding is always overwritten with the freshly generated vector.
    """
    tx.run(
        """
        MERGE (e:Email {email_id: $email_id})
        SET
            e.thread_id          = $thread_id,
            e.file_name          = $file_name,
            e.subject            = $subject,
            e.body               = $body,
            e.date               = $date,
            e.quoted_reply_level = $quoted_reply_level,
            e.embedding          = $embedding
        """,
        payload,
    )


def merge_person(tx, name: str, email: str) -> str | None:
    """
    Merges a Person node on person_id.
    ON MATCH uses COALESCE so a node created from a BCC entry (name only,
    no email) gets its email filled in if the same person appears later as
    a sender with a full email address.
    Returns person_id, or None if both name and email are empty.
    """
    person_id = make_person_id(name, email)
    if not person_id:
        return None

    tx.run(
        """
        MERGE (p:Person {person_id: $person_id})
        ON CREATE SET
            p.email = $email,
            p.name  = $name
        ON MATCH SET
            p.email = COALESCE(p.email, $email),
            p.name  = COALESCE(p.name,  $name)
        """,
        {
            "person_id": person_id,
            "email":     email.strip().lower() if email else "",
            "name":      name.strip() if name else "",
        },
    )
    return person_id


def merge_company(tx, company_name: str) -> None:
    if not company_name or not company_name.strip():
        return
    tx.run(
        "MERGE (c:Company {name: $name})",
        {"name": company_name.strip()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relationship creation
# ─────────────────────────────────────────────────────────────────────────────
def create_in_thread(tx, email_id: str, thread_id: str) -> None:
    tx.run(
        """
        MATCH (e:Email  {email_id:  $email_id})
        MATCH (t:Thread {thread_id: $thread_id})
        MERGE (e)-[:IN_THREAD]->(t)
        """,
        {"email_id": email_id, "thread_id": thread_id},
    )


def create_sent(tx, person_id: str, email_id: str) -> None:
    if not person_id:
        return
    tx.run(
        """
        MATCH (p:Person {person_id: $person_id})
        MATCH (e:Email  {email_id:  $email_id})
        MERGE (p)-[:SENT]->(e)
        """,
        {"person_id": person_id, "email_id": email_id},
    )


def create_recipient_rel(tx, person_id: str, email_id: str, rel_type: str) -> None:
    """
    rel_type is always one of TO | CC | BCC — never user input —
    so f-string interpolation is safe here.
    """
    if not person_id:
        return
    tx.run(
        f"""
        MATCH (e:Email  {{email_id:  $email_id}})
        MATCH (p:Person {{person_id: $person_id}})
        MERGE (e)-[:{rel_type}]->(p)
        """,
        {"person_id": person_id, "email_id": email_id},
    )


def create_replied_to(
    tx, child_id: str, parent_id: str, level: int
) -> None:
    tx.run(
        """
        MATCH (child:Email  {email_id: $child_id})
        MATCH (parent:Email {email_id: $parent_id})
        MERGE (child)-[r:REPLIED_TO]->(parent)
        SET r.level = $level
        """,
        {"child_id": child_id, "parent_id": parent_id, "level": level},
    )


def create_affiliated_with(
    tx, person_id: str, company_name: str, position: str, location: str
) -> None:
    """
    position and location live on the AFFILIATED_WITH edge, not on Company,
    because the same person can hold different roles at different companies.
    """
    if not person_id or not company_name or not company_name.strip():
        return
    tx.run(
        """
        MATCH (p:Person  {person_id: $person_id})
        MATCH (c:Company {name:      $company_name})
        MERGE (p)-[r:AFFILIATED_WITH]->(c)
        SET
            r.position = $position,
            r.location = $location
        """,
        {
            "person_id":    person_id,
            "company_name": company_name.strip(),
            "position":     position or "",
            "location":     location or "",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thread ingestion
# ─────────────────────────────────────────────────────────────────────────────
def ingest_thread(session, thread_data: dict, em: EmbeddingModel) -> int:
    """
    Ingests one thread file into Neo4j.

    Embedding strategy:
      - Collect all message texts up-front
      - One batched model.encode() call for the entire thread
      - Write each Email node with its pre-computed vector

    Returns the number of messages successfully written.
    """
    thread_id = safe_get(thread_data, "thread_id", "unknown_thread")
    file_name = safe_get(thread_data, "file_name", "unknown_file")
    messages  = thread_data.get("messages", [])

    if not messages:
        logger.warning("Thread %s has no messages — skipping.", thread_id)
        return 0

    logger.info("Thread %s | %d message(s)", thread_id, len(messages))

    # ── 1. Batch-embed all messages in this thread ────────────────────────
    logger.debug("  Generating embeddings …")
    t0    = time.perf_counter()
    texts = [build_embedding_text(msg) for msg in messages]
    vecs  = em.embed_passages(texts)
    logger.info(
        "  Embedded %d messages in %.2fs", len(texts), time.perf_counter() - t0
    )

    # ── 2. Thread node ────────────────────────────────────────────────────
    with session.begin_transaction() as tx:
        merge_thread(tx, thread_id, file_name)
        tx.commit()
        logger.debug("  Thread node merged.")

    # ── 3. Messages — one transaction per message for fault isolation ─────
    written = 0
    for idx, (msg, vec) in enumerate(zip(messages, vecs)):
        message_id  = safe_get(msg, "id", f"auto_{idx}")
        email_id    = make_email_id(thread_id, message_id)
        parent_id   = msg.get("parent_id")
        reply_level = msg.get("quoted_reply_level", 0)

        sender       = msg.get("sender", {})
        sender_name  = (sender.get("name",  "") or "").strip()
        sender_email = (sender.get("email", "") or "").strip().lower()
        company_name = (safe_get(msg, "company",  "") or "").strip()
        position     = (safe_get(msg, "position", "") or "").strip()
        location     = (safe_get(msg, "location", "") or "").strip()

        try:
            with session.begin_transaction() as tx:
                # Email node + embedding
                merge_email(
                    tx,
                    {
                        "email_id":          email_id,
                        "thread_id":         thread_id,
                        "file_name":         file_name,
                        "subject":           safe_get(msg, "subject", "") or "",
                        "body":              safe_get(msg, "body",    "") or "",
                        "date":              safe_get(msg, "date",    "") or "",
                        "quoted_reply_level": reply_level,
                        "embedding":         vec,
                    },
                )
                logger.debug("    [%s] Email node merged.", email_id)

                # Thread membership
                create_in_thread(tx, email_id, thread_id)

                # Sender
                sender_id = merge_person(tx, sender_name, sender_email)
                create_sent(tx, sender_id, email_id)
                logger.debug("    [%s] Sender: %s", email_id, sender_id)

                # Sender company affiliation
                if company_name:
                    merge_company(tx, company_name)
                    create_affiliated_with(tx, sender_id, company_name, position, location)
                    logger.debug(
                        "    [%s] Affiliation: %s → %s", email_id, sender_id, company_name
                    )

                # Recipients
                for rel_type, group_key in [
                    ("TO",  "recipients"),
                    ("CC",  "cc"),
                    ("BCC", "bcc"),
                ]:
                    for person in msg.get(group_key, []):
                        if not isinstance(person, dict):
                            continue
                        pname  = (person.get("name",  "") or "").strip()
                        pemail = (person.get("email", "") or "").strip().lower()
                        pid    = merge_person(tx, pname, pemail)
                        create_recipient_rel(tx, pid, email_id, rel_type)

                # Reply chain
                if parent_id:
                    parent_email_id = make_email_id(thread_id, parent_id)
                    create_replied_to(tx, email_id, parent_email_id, reply_level)
                    logger.debug(
                        "    [%s] REPLIED_TO → %s (level=%d)",
                        email_id, parent_email_id, reply_level,
                    )

                tx.commit()
                written += 1

        except Exception as exc:
            logger.error("    [%s] Failed: %s", email_id, exc, exc_info=True)

    logger.info("  Thread %s done: %d/%d messages written.", thread_id, written, len(messages))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(canonicalized_dir: Path) -> None:
    if not canonicalized_dir.exists():
        logger.error("Directory not found: %s", canonicalized_dir)
        return

    json_files = sorted(canonicalized_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", canonicalized_dir)
        return

    logger.info("Found %d file(s) in %s", len(json_files), canonicalized_dir)

    # Initialise embedding model once — lazy-loads on first embed call
    em     = EmbeddingModel()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    try:
        # Schema setup — run once before any writes
        with driver.session() as session:
            create_constraints(session)
            create_vector_index(session, dim=em.dim)

        total_threads  = 0
        total_messages = 0
        total_failed   = 0

        for json_file in json_files:
            logger.info("─── Processing %s ───", json_file.name)
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    thread_data = json.load(f)

                with driver.session() as session:
                    written = ingest_thread(session, thread_data, em)

                total_threads  += 1
                total_messages += written

            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON in %s: %s", json_file.name, exc)
                total_failed += 1
            except Exception as exc:
                logger.error("Unexpected error in %s: %s", json_file.name, exc, exc_info=True)
                total_failed += 1

        logger.info(
            "═══ Ingestion complete: %d thread(s) | %d message(s) written | %d file(s) failed ═══",
            total_threads, total_messages, total_failed,
        )

    finally:
        driver.close()
        logger.info("Neo4j driver closed.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest canonicalized email JSON files into Neo4j Aura "
                    "(with on-the-fly embedding)"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory of canonicalized JSON files (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args()
    main(args.dir)