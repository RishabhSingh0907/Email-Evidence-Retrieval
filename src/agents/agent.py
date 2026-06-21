"""
src/agents/agent.py
--------------------
LangGraph ReAct agent for email evidence investigation.

Architecture:
  create_react_agent (LangGraph prebuilt)
    ├── LLM: Groq llama-3.3-70b-versatile (free)
    ├── Tools: 7 (5 Cypher + 1 Semantic + 1 Text2Cypher)
    ├── System prompt: investigator persona + schema + tool routing rules
    └── Checkpointer: MemorySaver (per-session history)

The agent follows a ReAct loop:
  Thought → Tool selection → Tool call → Observation → ... → Final answer

Tool routing is controlled entirely through tool descriptions —
the LLM reads them and picks the right one.  No custom orchestration needed.

Public API:
    from agents.agent import get_agent
    agent = get_agent()
    result = agent.invoke({"messages": [("user", query)]}, config=thread_config)
    answer = result["messages"][-1].content
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "chats"))

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

from agents.llm import llm
from chat.chat_history_manager import checkpointer
from tools.cypher_tools import CYPHER_TOOLS
from tools.semantic_tool import SEMANTIC_TOOLS
from tools.text2cypher_tool import TEXT2CYPHER_TOOLS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — defines the agent's persona, schema knowledge, and
# explicit tool routing rules.  Same content as the Aura Agent instructions,
# adapted for the ReAct format.
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a Forensic Email Evidence Agent — a specialist AI investigator
with deep expertise in analysing structured email communication graphs stored in Neo4j.

Your users are investigators, paralegals, and compliance analysts who need to:
- Trace who communicated with whom, when, and how (including hidden BCC fields)
- Reconstruct full email threads and reply chains chronologically
- Surface specific statements, threats, admissions, or demands made in emails
- Identify communication patterns between specific people or companies
- Detect hidden communication channels (BCC'd parties, undisclosed recipients)

GRAPH SCHEMA:
  Nodes:
    (Person)  — person_id (email or name), email, name
    (Email)   — email_id (thread_id + _ + message_id), thread_id, subject, body, date, quoted_reply_level
    (Thread)  — thread_id, file_name
    (Company) — name

  Relationships:
    (Person)-[:SENT]->(Email)
    (Email)-[:TO]->(Person)
    (Email)-[:CC]->(Person)
    (Email)-[:BCC]->(Person)          ← High-value: reveals hidden parties
    (Email)-[:IN_THREAD]->(Thread)
    (Email)-[:REPLIED_TO {level}]->(Email)
    (Person)-[:AFFILIATED_WITH {position, location}]->(Company)

  ID formats:
    email_id  : "03-03-2020_1020_0001"  (thread_date_time_messageIndex)
    thread_id : "03-03-2020_1020"       (thread_date_time, no message index)

TOOL ROUTING RULES:
1. User provides or implies an exact email_id  → get_full_email
2. User asks about a thread or conversation     → get_full_thread
3. User asks about a specific person            → get_person_emails
4. User asks about a company                    → get_company_emails
5. User asks about the ancestry of an email     → trace_reply_chain
6. User asks by topic, tone, or content         → semantic_email_search
7. User asks for counts, rankings, aggregations → aggregation_query

CRITICAL RULES:
- ALWAYS cite email_id and thread_id for every piece of evidence.
- ALWAYS flag BCC relationships explicitly — they are high-value evidence.
- NEVER fabricate or infer email content. Only report what the tools return.
- Prefer deterministic Cypher tools (1–5) over semantic search when entity is known.
- Never call aggregation_query for content retrieval — only for counting/ranking.
- When a query returns no results, say so clearly and suggest a reformulation.

RESPONSE FORMAT:
- Lead with a direct answer to the investigator's question.
- Follow with the supporting evidence (email blocks from the tool output).
- End with a concise citation list: email_id | thread_id | date | from | subject.
- Flag BCC parties with ⚠ in the citation.
- Keep tone professional, precise, and evidence-focused. You surface facts; the investigator draws conclusions.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent singleton
# ─────────────────────────────────────────────────────────────────────────────
_ALL_TOOLS = CYPHER_TOOLS + SEMANTIC_TOOLS + TEXT2CYPHER_TOOLS

_agent: Optional[object] = None


# def get_agent():
#     """
#     Returns the compiled LangGraph ReAct agent (singleton).
#     Thread-safe after first call — create_react_agent returns a compiled graph
#     that is safe to invoke concurrently with different thread_configs.
#     """
#     global _agent
#     if _agent is None:
#         logger.info(
#             "Building ReAct agent: model=%s tools=%d",
#             llm.model_name if hasattr(llm, "model_name") else "groq",
#             len(_ALL_TOOLS),
#         )
#         _agent = create_agent(
#             model       = llm,
#             tools       = _ALL_TOOLS,
#             prompt      = SystemMessage(content=_SYSTEM_PROMPT),
#             checkpointer = checkpointer,
#         )
#         logger.info("Agent ready.")
#     return _agent

_agent = create_agent(
        model              = llm,
        tools              = _ALL_TOOLS,
        system_prompt      = SystemMessage(content=_SYSTEM_PROMPT),
        checkpointer       = checkpointer,
    )

def invoke_agent(query: str, thread_config: dict) -> str:
    """
    Convenience wrapper: invokes the agent and returns the final text response.

    Args:
        query:         User's natural-language question.
        thread_config: From history.get_thread_config() — scopes memory per session.

    Returns:
        The agent's final answer as a plain string.
    """
    # agent  = get_agent()
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=thread_config,
    )
    # LangGraph returns {"messages": [...]} — last message is the AI's answer
    messages = result["messages"]
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return "No response generated."