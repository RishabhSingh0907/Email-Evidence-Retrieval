"""
src/chats/history.py
---------------------
Session state and chat history management for the LangGraph agent.

LangGraph's create_react_agent accepts a `checkpointer` that persists
MessagesState across invocations.  We use MemorySaver (in-process, free)
which keeps history for the lifetime of the Streamlit server process.

Each browser tab / user gets its own thread_id from Streamlit's session_state,
so conversations don't bleed across sessions.

Thread ID format: "email_agent_{session_id}"
  - session_id is generated once per Streamlit session and stored in
    st.session_state so it survives reruns but resets on page refresh.

If you want persistent history across server restarts, swap MemorySaver
for SqliteSaver (pip install langgraph-checkpoint-sqlite) — the interface
is identical.

Usage:
    from chats.history import checkpointer, get_thread_config

    agent = create_react_agent(..., checkpointer=checkpointer)
    config = get_thread_config()           # call inside Streamlit context
    result = agent.invoke({"messages": [...]}, config=config)
"""

from __future__ import annotations

import uuid
import logging
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Module-level MemorySaver shared across all sessions in this process
checkpointer = MemorySaver()


def get_session_id() -> str:
    """
    Returns a stable session ID for the current Streamlit browser session.
    Generated once on first call and stored in st.session_state.
    """
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = str(uuid.uuid4())[:8]
        logger.info("New session: %s", st.session_state["agent_session_id"])
    return st.session_state["agent_session_id"]


def get_thread_config() -> dict:
    """
    Returns the LangGraph config dict for the current session.
    Pass this as `config` to agent.invoke() so history is scoped per session.
    """
    session_id = get_session_id()
    thread_id  = f"email_agent_{session_id}"
    return {"configurable": {"thread_id": thread_id}}


def clear_session_history() -> None:
    """
    Resets the current session's conversation history.
    Call on 'Clear chat' button press.
    """
    if "agent_session_id" in st.session_state:
        old = st.session_state.pop("agent_session_id")
        logger.info("History cleared for session %s", old)
    # Force a new session_id on next get_session_id() call
    st.session_state.pop("agent_session_id", None)