"""
src/tools/text2cypher_tool.py
------------------------------
Text-to-Cypher tool for aggregation, counting, and pattern queries that
none of the deterministic Cypher tools can handle.

Uses GraphCypherQAChain (LangChain-Neo4j) backed by the same Groq LLM.
Few-shot examples in the prompt guide the LLM toward correct Cypher for
our specific schema.

When to use (agent description drives this):
  - "How many emails did X send?"
  - "Which person appears in the most threads?"
  - "List all companies in the graph"
  - "How many BCC relationships exist?"
  - Any counting / grouping / ranking query

When NOT to use:
  - Known entity lookups (use cypher_tools)
  - Semantic / topic search (use semantic_tool)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.llm   import llm    # noqa: E402
from agents.graph import graph  # noqa: E402

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cypher generation prompt with schema + few-shot examples
# ─────────────────────────────────────────────────────────────────────────────
_CYPHER_TEMPLATE = """
You are an expert Neo4j Cypher developer for an email evidence investigation system.
Convert the user question into a valid Cypher query using ONLY the schema below.

SCHEMA:
Nodes:
  (Person  {{person_id, email, name}})
  (Email   {{email_id, thread_id, subject, body, date, quoted_reply_level}})
  (Thread  {{thread_id, file_name}})
  (Company {{name}})

Relationships:
  (Person)-[:SENT]->(Email)
  (Email)-[:TO]->(Person)
  (Email)-[:CC]->(Person)
  (Email)-[:BCC]->(Person)
  (Email)-[:IN_THREAD]->(Thread)
  (Email)-[:REPLIED_TO {{level}}]->(Email)
  (Person)-[:AFFILIATED_WITH {{position, location}}]->(Company)

RULES:
- Never return full Email.body (it is large); return Email.subject, Email.email_id, Email.date only.
- Never return embedding properties.
- Use DISTINCT where cardinality could inflate counts.
- For person lookups, match on p.email = toLower($email) OR toLower(p.name) CONTAINS $name.
- Limit results to 25 unless the question explicitly asks for all.

FEW-SHOT EXAMPLES:

Q: How many emails are in the graph?
Cypher: MATCH (e:Email) RETURN count(e) AS total_emails

Q: Who sent the most emails?
Cypher: MATCH (p:Person)-[:SENT]->(e:Email) RETURN p.name AS sender, count(e) AS sent ORDER BY sent DESC LIMIT 5

Q: How many BCC relationships exist?
Cypher: MATCH (e:Email)-[:BCC]->(p:Person) RETURN count(*) AS bcc_count

Q: List all companies in the graph
Cypher: MATCH (c:Company) RETURN c.name AS company ORDER BY company

Q: Which threads have the most messages?
Cypher: MATCH (e:Email)-[:IN_THREAD]->(t:Thread) RETURN t.thread_id, count(e) AS messages ORDER BY messages DESC LIMIT 10

Q: How many people appear in the graph?
Cypher: MATCH (p:Person) RETURN count(p) AS total_people

Q: Which person is in the most threads?
Cypher: MATCH (p:Person)-[:SENT]->(e:Email)-[:IN_THREAD]->(t:Thread) WITH p, count(DISTINCT t) AS threads ORDER BY threads DESC LIMIT 1 RETURN p.name AS person, threads

Schema: {schema}
Question: {question}
"""

_cypher_prompt = PromptTemplate.from_template(_CYPHER_TEMPLATE)

_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=False,
    cypher_prompt=_cypher_prompt,
    allow_dangerous_requests=True,
    return_intermediate_steps=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────────────────────────────────────
@tool
def aggregation_query(question: str) -> str:
    """
    Use this tool ONLY for aggregation, counting, ranking, or pattern analysis
    questions that cannot be answered by the other tools.

    Examples of when to use:
    - "How many emails did Paul Camera send?"
    - "Which person appears in the most email threads?"
    - "List all companies mentioned across all threads"
    - "How many BCC relationships exist in the graph?"
    - "Which thread has the most messages?"
    - "How many unique senders are there?"

    Do NOT use for: fetching email content, looking up a specific person,
    searching by topic/keyword, or retrieving a thread — use dedicated tools.
    """
    logger.info("[tool] aggregation_query question='%s'", question)
    try:
        result = _chain.invoke({"query": question})
        answer = result.get("result", str(result))
        return str(answer)
    except Exception as e:
        logger.error("Text2Cypher failed: %s", e, exc_info=True)
        return f"Could not generate Cypher for this question: {e}"


TEXT2CYPHER_TOOLS = [aggregation_query]