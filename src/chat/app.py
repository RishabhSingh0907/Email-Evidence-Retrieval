"""
src/chats/app.py
-----------------
Streamlit chat interface for the Email Evidence Retrieval Agent.

The UI is intentionally thin — it delegates all intelligence to the agent.
Its only jobs are:
  1. Accept a user message
  2. Call invoke_agent() with the session thread config
  3. Render the response with formatting (email cards, citations, BCC flags)
  4. Maintain visual chat history in session state

Run:
  cd Agentic_RAG
  streamlit run src/chats/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# ── Path resolution ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "chats"))

import dotenv
dotenv.load_dotenv(ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── App imports (after path setup) ───────────────────────────────────────────
from agents.agent import invoke_agent, _agent   # noqa: E402
from chat.chat_history_manager import get_thread_config, clear_session_history  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────
def setup_page() -> None:
    st.set_page_config(
        page_title="Email Evidence Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    :root {
        --surface:   #1a1d27;
        --surface2:  #22263a;
        --border:    #2e3347;
        --accent:    #4f8ef7;
        --warn:      #f5a623;
        --warn-dim:  #3d2a0a;
        --green:     #3ecf6e;
        --green-dim: #0d2e1c;
        --text:      #d4d8e8;
        --text-dim:  #8890a8;
        --mono:      'JetBrains Mono', 'Fira Code', monospace;
    }
    .stApp { background: #0f1117; }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    /* ── email card ── */
    .ecard {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .ecard.bcc { border-left: 3px solid var(--warn); }

    /* ── card meta row ── */
    .emeta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem 1.2rem;
        font-size: 0.76rem;
        color: var(--text-dim);
        font-family: var(--mono);
        margin-bottom: 0.5rem;
    }
    .emeta .lbl { color: var(--accent); font-weight: 600; }

    /* ── BCC pill ── */
    .bcc-pill {
        background: var(--warn-dim);
        color: var(--warn);
        border: 1px solid var(--warn);
        border-radius: 4px;
        padding: 0 5px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: .05em;
    }

    .esubject {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.45rem;
    }
    .ebody {
        font-size: 0.82rem;
        color: var(--text);
        line-height: 1.65;
        white-space: pre-wrap;
        word-break: break-word;
        border-top: 1px solid var(--border);
        padding-top: 0.55rem;
        margin-top: 0.35rem;
    }

    /* ── citations ── */
    .cite {
        background: var(--green-dim);
        border: 1px solid var(--green);
        border-radius: 5px;
        padding: 0.35rem 0.65rem;
        margin-bottom: 0.3rem;
        font-family: var(--mono);
        font-size: 0.73rem;
        color: var(--green);
    }
    .cite.bcc-cite {
        background: var(--warn-dim);
        border-color: var(--warn);
        color: var(--warn);
    }

    /* ── banner ── */
    .banner {
        background: #1e3a6e;
        border: 1px solid var(--accent);
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        font-size: 0.8rem;
        color: var(--text);
        margin-bottom: 0.9rem;
        font-family: var(--mono);
    }

    /* ── thinking indicator ── */
    .thinking {
        font-size: 0.8rem;
        color: var(--text-dim);
        font-family: var(--mono);
        padding: 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Response renderer
# Parses the formatted text from ResponseFormatter and renders it as cards.
# Falls back to raw markdown if the response isn't in the structured format.
# ─────────────────────────────────────────────────────────────────────────────
def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_agent_response(response_text: str) -> None:
    """
    Renders the agent's text response.

    The agent synthesises a natural-language answer first, then appends
    tool output (which is already structured by ResponseFormatter).
    We render both: prose as markdown, email blocks as HTML cards.
    """
    if not response_text:
        st.markdown("_(no response)_")
        return

    # ── If structured tool output is present, render as cards ──
    # ResponseFormatter blocks are delimited by ═══ and ─── lines.
    # We split on them and render each email block as a card.

    lines = response_text.splitlines()
    in_email_block = False
    card_lines: list[str] = []
    prose_lines: list[str] = []
    all_citations: list[str] = []
    in_citations = False

    for line in lines:
        stripped = line.strip()

        # Citation section
        if stripped.startswith("═") and "CITATIONS" in "".join(lines[lines.index(line):lines.index(line)+3]):
            in_citations = True
            continue
        if in_citations:
            if stripped.startswith("[CITATION]"):
                all_citations.append(stripped)
            continue

        # Email block delimiter
        if stripped.startswith("─" * 10) or stripped.startswith("═" * 10):
            if in_email_block and card_lines:
                _render_card(card_lines)
                card_lines = []
            in_email_block = stripped.startswith("─")
            continue

        # Banner / result header
        if stripped.startswith("RESULTS:") or stripped.startswith("🔎"):
            st.markdown(
                f'<div class="banner">{_esc(stripped)}</div>',
                unsafe_allow_html=True,
            )
            continue

        if in_email_block:
            card_lines.append(line)
        else:
            prose_lines.append(line)

    # Flush last card
    if card_lines:
        _render_card(card_lines)

    # Prose (agent's synthesised answer)
    prose = "\n".join(prose_lines).strip()
    if prose:
        st.markdown(prose)

    # Citations expander
    if all_citations:
        with st.expander(f"📎 Evidence citations ({len(all_citations)})", expanded=False):
            for c in all_citations:
                is_bcc = "BCC" in c
                cls = "cite bcc-cite" if is_bcc else "cite"
                st.markdown(
                    f'<div class="{cls}">{_esc(c)}</div>',
                    unsafe_allow_html=True,
                )

    # If no structure was found at all, just render as markdown
    if not in_email_block and not all_citations and not prose:
        st.markdown(response_text)


def _render_card(lines: list[str]) -> None:
    """Parse a card_lines block and render an email HTML card."""
    fields: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False

    for line in lines:
        s = line.strip()
        if s == "BODY:":
            in_body = True
            continue
        if in_body:
            body_lines.append(line[4:] if line.startswith("    ") else line)
            continue
        for key in ("email_id", "Thread", "Date", "Subject", "From", "To",
                    "CC", "BCC  ⚠", "Reply Lvl", "Replies→", "Via", "Source", "#"):
            if s.startswith(f"{key}"):
                val = s.split(":", 1)[-1].strip() if ":" in s else s
                fields[key] = val

    bcc_val = fields.get("BCC  ⚠", "")
    has_bcc = bool(bcc_val and bcc_val != "—")
    bcc_cls = "ecard bcc" if has_bcc else "ecard"
    bcc_pill = '<span class="bcc-pill">⚠ BCC</span>' if has_bcc else ""

    html_parts = [f'<div class="{bcc_cls}">']
    html_parts.append('<div class="emeta">')

    if "email_id" in fields or "#" in fields:
        eid = fields.get("email_id", fields.get("#", ""))
        html_parts.append(f'<span><span class="lbl">ID</span> {_esc(eid)}</span>')
    if "Thread" in fields:
        html_parts.append(f'<span><span class="lbl">Thread</span> {_esc(fields["Thread"])}</span>')
    if "Date" in fields:
        html_parts.append(f'<span><span class="lbl">Date</span> {_esc(fields["Date"])}</span>')
    if "From" in fields:
        html_parts.append(f'<span><span class="lbl">From</span> {_esc(fields["From"])}</span>')
    if "To" in fields:
        html_parts.append(f'<span><span class="lbl">To</span> {_esc(fields["To"])}</span>')
    if "CC" in fields:
        html_parts.append(f'<span><span class="lbl">CC</span> {_esc(fields["CC"])}</span>')
    if has_bcc:
        html_parts.append(
            f'<span><span class="lbl">BCC</span> {_esc(bcc_val)} {bcc_pill}</span>'
        )
    html_parts.append("</div>")  # .emeta

    if "Subject" in fields:
        html_parts.append(f'<div class="esubject">{_esc(fields["Subject"])}</div>')

    if body_lines:
        body_text = "\n".join(body_lines).strip()
        PREVIEW = 500
        if len(body_text) > PREVIEW:
            preview = body_text[:PREVIEW].rsplit(" ", 1)[0] + " …"
            html_parts.append(f'<div class="ebody">{_esc(preview)}</div>')
            # Full body in expander handled below
            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)
            with st.expander("Read full body"):
                st.text(body_text)
            return
        else:
            html_parts.append(f'<div class="ebody">{_esc(body_text)}</div>')

    html_parts.append("</div>")  # .ecard
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔍 Email Evidence Agent")
        st.markdown("---")

        st.markdown("### Model")
        st.code("llama-3.3-70b · Groq (free)", language=None)

        st.markdown("### Agent status")
        if st.button("Warm up agent", use_container_width=True):
            with st.spinner("Loading agent and tools …"):
                try:
                    _agent
                    st.success("Agent ready ✓")
                except Exception as e:
                    st.error(f"Failed: {e}")

        st.markdown("---")
        st.markdown("### Example queries")
        examples = [
            "Show me the full thread 03-03-2020_1020",
            "All emails from asadushah@gmail.com",
            "Emails threatening legal action about IP",
            "Find emails mentioning fiduciary duty",
            "Who was secretly BCC'd in any thread?",
            "Emails from Paul Camera to investors",
            "Trace the reply chain of 03-03-2020_1020_0005",
            "How many emails did Paul Camera send?",
            "Which company appears in the most threads?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
                st.session_state["pending_query"] = ex

        st.markdown("---")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = []
            clear_session_history()
            st.rerun()

        st.markdown("---")
        st.caption("GraphRAG · Neo4j Aura · Groq · LangGraph")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    setup_page()
    render_sidebar()

    st.markdown("## Email Evidence Retrieval Agent")
    st.markdown(
        "<span style='color:#8890a8;font-size:0.85rem'>"
        "LangGraph ReAct · Groq llama-3.3-70b · Neo4j Aura · "
        "5 Cypher tools + semantic search + Text2Cypher"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Replay history ──
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_agent_response(msg["content"])

    # ── Handle example query injection from sidebar ──
    pending = st.session_state.pop("pending_query", None)

    # ── Chat input ──
    user_input = st.chat_input("Ask about emails, people, threads, or topics …")
    query = pending or user_input

    if not query:
        return

    # ── Show user message ──
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Run agent ──
    with st.chat_message("assistant"):
        thread_config = get_thread_config()

        with st.spinner("Agent thinking …"):
            try:
                response = invoke_agent(query, thread_config)
            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                response = f"⚠ Agent error: {e}"

        render_agent_response(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()