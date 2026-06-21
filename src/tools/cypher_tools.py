"""
src/tools/cypher_tools.py
--------------------------
Five deterministic Cypher tools — each maps to one Aura Agent Cypher Template.

These tools run hard-coded, parameterised Cypher.  No LLM is involved.
Results are formatted through ResponseFormatter before being returned as a
string to the agent for synthesis.

Tools:
  get_full_email          — single email by email_id
  get_full_thread         — all emails in a thread, chronological
  get_person_emails       — all emails involving a person (any role incl BCC)
  get_company_emails      — all emails involving anyone affiliated with a company
  trace_reply_chain       — recursive parent chain from a starting email_id
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.tools import tool

# ── path resolution ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "chats"))

from agents.graph import driver                              # noqa: E402
from chat.response_formatter import ResponseFormatter             # noqa: E402

logger    = logging.getLogger(__name__)
formatter = ResponseFormatter()

# ─────────────────────────────────────────────────────────────────────────────
# Shared result builder
# ─────────────────────────────────────────────────────────────────────────────

def _run(cypher: str, params: dict) -> list[dict]:
    with driver.session() as s:
        return [dict(r) for r in s.run(cypher, params)]


def _format(records: list[dict], context: str = "") -> str:
    if not records:
        return f"No results found for: {context}"
    emails = formatter.deduplicate(records)
    return formatter.format_email_list(emails, query_context=context).text


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Get Full Email
# ─────────────────────────────────────────────────────────────────────────────
_GET_EMAIL_CYPHER = """
MATCH (e:Email {email_id: $email_id})
OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
OPTIONAL MATCH (e)-[:TO]->(to_p:Person)
OPTIONAL MATCH (e)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (e)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (e)-[:IN_THREAD]->(t:Thread)
RETURN
    e.email_id            AS email_id,
    e.thread_id           AS thread_id,
    e.subject             AS subject,
    e.date                AS date,
    e.body                AS body,
    e.quoted_reply_level  AS reply_level,
    sender.name           AS sender_name,
    sender.email          AS sender_email,
    collect(DISTINCT to_p.email)  AS to_recipients,
    collect(DISTINCT cc_p.email)  AS cc_recipients,
    collect(DISTINCT bcc_p.email) AS bcc_recipients,
    t.file_name           AS source_file
"""

@tool
def get_full_email(email_id: str) -> str:
    """
    Retrieve the complete content of a single email by its email_id.
    Use when the user asks about a specific email and provides or implies
    an exact email_id (format: thread_id_messageId, e.g. 03-03-2020_1020_0001).
    Returns full body, all recipients including BCC, thread context, and source file.
    Always use this before get_full_thread if only one email is needed.
    """
    logger.info("[tool] get_full_email email_id=%s", email_id)
    records = _run(_GET_EMAIL_CYPHER, {"email_id": email_id})
    return _format(records, context=email_id)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Get Full Thread
# ─────────────────────────────────────────────────────────────────────────────
_GET_THREAD_CYPHER = """
MATCH (e:Email {thread_id: $thread_id})-[:IN_THREAD]->(t:Thread)
OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
OPTIONAL MATCH (e)-[:TO]->(to_p:Person)
OPTIONAL MATCH (e)-[:CC]->(cc_p:Person)
OPTIONAL MATCH (e)-[:BCC]->(bcc_p:Person)
OPTIONAL MATCH (e)-[:REPLIED_TO]->(parent:Email)
WITH e, t, sender, parent,
     collect(DISTINCT to_p.email)  AS to_recipients,
     collect(DISTINCT cc_p.email)  AS cc_recipients,
     collect(DISTINCT bcc_p.email) AS bcc_recipients
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
    to_recipients,
    cc_recipients,
    bcc_recipients,
    parent.email_id       AS replies_to,
    t.file_name           AS source_file,
    1.0                   AS similarity_score
"""

@tool
def get_full_thread(thread_id: str) -> str:
    """
    Retrieve all emails in a thread in chronological order, including sender,
    recipients, CC, BCC, and reply chain structure.
    Use to reconstruct a full conversation timeline.
    thread_id format: date_time, e.g. 03-03-2020_1020 (without the message index).
    Always returns emails oldest-first so the investigator can read the sequence.
    """
    logger.info("[tool] get_full_thread thread_id=%s", thread_id)
    records = _run(_GET_THREAD_CYPHER, {"thread_id": thread_id})
    return _format(records, context=f"Thread {thread_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Get Person Emails
# ─────────────────────────────────────────────────────────────────────────────
_GET_PERSON_CYPHER = """
MATCH (p:Person)
WHERE p.email = toLower($person_email)
   OR (p.email = '' AND toLower(p.name) CONTAINS toLower($person_name))
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
     collect(DISTINCT to_p.email)  AS to_recipients,
     collect(DISTINCT cc_p.email)  AS cc_recipients,
     collect(DISTINCT bcc_p.email) AS bcc_recipients,
     t,
     CASE WHEN (p)-[:SENT]->(e)  THEN 'SENDER'
          WHEN (e)-[:BCC]->(p)   THEN 'BCC'
          WHEN (e)-[:CC]->(p)    THEN 'CC'
          ELSE 'TO' END AS role_in_email
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
    to_recipients,
    cc_recipients,
    bcc_recipients,
    t.file_name           AS source_file,
    role_in_email         AS role_in_email,
    1.0                   AS similarity_score
LIMIT 50
"""

@tool
def get_person_emails(person_email: str = "", person_name: str = "") -> str:
    """
    Find every email where a specific person appears as sender, TO, CC, or BCC recipient.
    Critically important: also surfaces emails where the person was secretly BCC'd.
    Provide person_email if known (more precise). Use person_name as fallback.
    Returns emails newest-first with each email's role (SENDER/TO/CC/BCC) flagged.
    Use this to build a complete communication profile for a suspect or witness.
    """
    logger.info("[tool] get_person_emails email=%s name=%s", person_email, person_name)
    records = _run(_GET_PERSON_CYPHER, {
        "person_email": person_email.lower().strip(),
        "person_name":  person_name.strip(),
    })
    context = person_email or person_name
    return _format(records, context=f"Person: {context}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Get Company Emails
# ─────────────────────────────────────────────────────────────────────────────
_GET_COMPANY_CYPHER = """
MATCH (p:Person)-[:AFFILIATED_WITH]->(c:Company {name: $company_name})
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
WITH DISTINCT e, sender, p, c,
     collect(DISTINCT to_p.email)  AS to_recipients,
     collect(DISTINCT cc_p.email)  AS cc_recipients,
     collect(DISTINCT bcc_p.email) AS bcc_recipients,
     t
ORDER BY e.date DESC
RETURN
    e.email_id                  AS email_id,
    e.thread_id                 AS thread_id,
    e.date                      AS date,
    e.subject                   AS subject,
    e.body                      AS body,
    e.quoted_reply_level        AS reply_level,
    sender.name                 AS sender_name,
    sender.email                AS sender_email,
    to_recipients,
    cc_recipients,
    bcc_recipients,
    t.file_name                 AS source_file,
    p.name                      AS affiliated_person_name,
    p.email                     AS affiliated_person_email,
    c.name                      AS company,
    1.0                         AS similarity_score
LIMIT 50
"""

@tool
def get_company_emails(company_name: str) -> str:
    """
    Find all emails involving anyone affiliated with a specific company.
    Use when the investigator asks about a company's communications rather
    than a specific individual. company_name must match exactly as stored
    (e.g. 'Savvy Commercial Capital', 'WSGR', 'Parkimon').
    The result shows which affiliated person connected each email to the company.
    """
    logger.info("[tool] get_company_emails company=%s", company_name)
    records = _run(_GET_COMPANY_CYPHER, {"company_name": company_name.strip()})
    return _format(records, context=f"Company: {company_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Trace Reply Chain
# ─────────────────────────────────────────────────────────────────────────────
_TRACE_REPLY_CYPHER = """
MATCH path = (e:Email {email_id: $email_id})-[:REPLIED_TO*0..10]->(ancestor:Email)
OPTIONAL MATCH (sender:Person)-[:SENT]->(ancestor)
OPTIONAL MATCH (ancestor)-[:TO]->(to_p:Person)
OPTIONAL MATCH (ancestor)-[:BCC]->(bcc_p:Person)
WITH ancestor, sender, length(path) AS depth,
     collect(DISTINCT to_p.email)  AS to_recipients,
     collect(DISTINCT bcc_p.email) AS bcc_recipients
ORDER BY depth DESC
RETURN
    ancestor.email_id            AS email_id,
    ancestor.thread_id           AS thread_id,
    ancestor.date                AS date,
    ancestor.subject             AS subject,
    ancestor.body                AS body,
    ancestor.quoted_reply_level  AS reply_level,
    sender.name                  AS sender_name,
    sender.email                 AS sender_email,
    to_recipients,
    []                           AS cc_recipients,
    bcc_recipients,
    null                         AS source_file,
    depth                        AS similarity_score
"""

@tool
def trace_reply_chain(email_id: str) -> str:
    """
    Traverse the REPLIED_TO chain upward from a given email_id to reconstruct
    its full ancestry — the sequence of emails that led to it.
    Returns emails ordered from the oldest ancestor down to the starting email.
    Use when you need to understand the context or provenance of a specific email.
    email_id format: thread_id_messageId, e.g. 03-03-2020_1020_0005
    """
    logger.info("[tool] trace_reply_chain email_id=%s", email_id)
    records = _run(_TRACE_REPLY_CYPHER, {"email_id": email_id})
    return _format(records, context=f"Reply chain from {email_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Exported list — consumed by agent.py
# ─────────────────────────────────────────────────────────────────────────────
CYPHER_TOOLS = [
    get_full_email,
    get_full_thread,
    get_person_emails,
    get_company_emails,
    trace_reply_chain,
]