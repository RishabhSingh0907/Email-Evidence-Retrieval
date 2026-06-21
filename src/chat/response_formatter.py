"""
response_formatter.py
---------------------
Transforms raw Aura Agent tool output into clean, structured, investigator-ready
response objects.

Responsibilities:
  - Deduplicate records that repeat due to multi-person affiliations in Cypher
  - Strip email signature noise (--delimiters, footers, symbol-only lines)
  - Build citation blocks per email (id + thread + date + sender + role)
  - Produce both a rich dict (for Streamlit rendering) and a plain-text
    formatted string (for the agent's LLM answer synthesis)

Usage:
    from response_formatter import ResponseFormatter

    # Inside a tool handler or Streamlit callback:
    raw = neo4j_result["records"]          # list[dict] from Cypher query
    fmt = ResponseFormatter()

    # Deduplicate + clean
    emails = fmt.deduplicate(raw, id_field="email_id")

    # Build full formatted response
    response = fmt.format_email_list(emails, query_context="Savvy Commercial Capital")
    print(response.text)           # plain text for LLM synthesis
    # response.citations            # list of CitationBlock
    # response.emails               # list of EmailRecord (for Streamlit cards)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CitationBlock:
    """
    Minimal evidence identifier for a single email.
    Included in every response so investigators can trace findings back to
    the exact node in the graph.
    """
    email_id:   str
    thread_id:  str
    date:       str
    sender:     str
    subject:    str
    role:       str = ""          # SENDER | TO | CC | BCC | AFFILIATED
    bcc_flag:   bool = False      # True if BCC relationship was involved

    def as_text(self) -> str:
        bcc_note = "  ⚠ BCC RECIPIENT INVOLVED" if self.bcc_flag else ""
        return (
            f"[CITATION] email_id={self.email_id} | thread={self.thread_id} | "
            f"date={self.date} | from={self.sender} | subject={self.subject!r}"
            f"{bcc_note}"
        )


@dataclass
class EmailRecord:
    """
    A single, deduplicated, cleaned email ready for rendering.
    """
    email_id:            str
    thread_id:           str
    date:                str
    subject:             str
    sender_name:         str
    sender_email:        str
    to_recipients:       list[str] = field(default_factory=list)
    cc_recipients:       list[str] = field(default_factory=list)
    bcc_recipients:      list[str] = field(default_factory=list)
    reply_level:         int = 0
    replies_to:          Optional[str] = None        # parent email_id
    body_raw:            str = ""
    body_clean:          str = ""
    source_file:         Optional[str] = None
    affiliated_persons:  list[dict] = field(default_factory=list)  # [{name, email}]

    @property
    def has_bcc(self) -> bool:
        return bool(self.bcc_recipients)

    @property
    def bcc_display(self) -> str:
        return ", ".join(self.bcc_recipients) if self.bcc_recipients else "—"

    def citation(self) -> CitationBlock:
        return CitationBlock(
            email_id=self.email_id,
            thread_id=self.thread_id,
            date=self.date,
            sender=self.sender_email or self.sender_name,
            subject=self.subject,
            bcc_flag=self.has_bcc,
        )


@dataclass
class FormattedResponse:
    """
    Full formatted response returned by ResponseFormatter.format_email_list().
    """
    query_context:  str
    total_emails:   int
    emails:         list[EmailRecord]
    citations:      list[CitationBlock]
    text:           str        # plain-text version for LLM synthesis / display
    summary_header: str        # one-line summary for Streamlit header


# ─────────────────────────────────────────────────────────────────────────────
# Body cleaner
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that mark the start of an email signature / footer block.
# We keep everything BEFORE the first match.
_SIG_DELIMITERS: list[str] = [
    r"\n\s*--\s*\n",                  # RFC 3676 sig separator
    r"\n_{3,}\s*\n",                   # ___ underscores
    r"\n-{3,}\s*\n",                   # --- dashes (3+ on own line)
    r"\n={3,}\s*\n",                   # === equals
    r"\nSent from my iPhone",
    r"\nSent from my Samsung",
    r"\nSent Via Phone",
    r"\nGet Outlook for",
    r"\nThis email was sent",
    r"\nConfidentiality Notice",
    r"\nDISCLAIMER:",
]

# Lines that are entirely decorative noise — remove entirely
_NOISE_LINE = re.compile(
    r"^[•\-\*#=_\|\s]{2,}$"            # lines of only symbols/spaces
)

# Excess blank lines → collapse to max 2
_EXCESS_BLANKS = re.compile(r"\n{3,}")


def clean_body(text: str) -> str:
    """
    Strips email signature blocks, decorative lines, and excess whitespace
    from a raw email body string.
    """
    if not text:
        return ""

    # 1. Split on the first signature delimiter found
    for pattern in _SIG_DELIMITERS:
        parts = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[0]
            break

    # 2. Strip trailing whitespace per line, drop pure-noise lines
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if not _NOISE_LINE.match(line)
    ]

    # 3. Collapse 3+ blank lines → 2
    cleaned = _EXCESS_BLANKS.sub("\n\n", "\n".join(lines))

    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_list(value) -> list[str]:
    """Normalises a field that may be a list, a string, or None."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)] if value else []


# ─────────────────────────────────────────────────────────────────────────────
# Main formatter
# ─────────────────────────────────────────────────────────────────────────────

class ResponseFormatter:
    """
    Stateless formatter. Instantiate once and reuse across calls.

    All public methods accept the raw list[dict] returned by Neo4j Cypher
    (as surfaced by the Aura Agent tool output) and return structured objects.
    """

    # ── Deduplication ────────────────────────────────────────────────────────

    def deduplicate(
        self,
        records: list[dict],
        id_field: str = "email_id",
    ) -> list[EmailRecord]:
        """
        Collapses duplicate rows that arise when a single email is returned
        multiple times because it matched multiple affiliated persons.

        Merges affiliated_person_name / affiliated_person_email into a list
        on the first occurrence, preserving all unique persons.
        """
        seen: dict[str, EmailRecord] = {}

        for raw in records:
            eid = raw.get(id_field, "")
            if not eid:
                continue

            if eid not in seen:
                seen[eid] = EmailRecord(
                    email_id        = eid,
                    thread_id       = raw.get("thread_id", ""),
                    date            = raw.get("date", ""),
                    subject         = raw.get("subject", "") or "(no subject)",
                    sender_name     = raw.get("sender_name", "") or "",
                    sender_email    = raw.get("sender_email", "") or "",
                    to_recipients   = _coerce_list(raw.get("to_recipients")),
                    cc_recipients   = _coerce_list(raw.get("cc_recipients")),
                    bcc_recipients  = _coerce_list(raw.get("bcc_recipients")),
                    reply_level     = int(raw.get("reply_level", 0) or 0),
                    replies_to      = raw.get("replies_to") or raw.get("hops_from_start"),
                    body_raw        = raw.get("body", "") or "",
                    body_clean      = clean_body(raw.get("body", "") or ""),
                    source_file     = raw.get("source_file"),
                    affiliated_persons = [],
                )

            record = seen[eid]

            # Merge affiliated persons (de-duplicate by email)
            p_name  = raw.get("affiliated_person_name", "") or ""
            p_email = raw.get("affiliated_person_email", "") or ""
            if p_email or p_name:
                person = {"name": p_name, "email": p_email}
                if person not in record.affiliated_persons:
                    record.affiliated_persons.append(person)

        # Return sorted by date ascending (nulls last)
        result = list(seen.values())
        result.sort(key=lambda e: e.date or "9999")
        return result

    # ── Single-email block renderer ───────────────────────────────────────────

    def render_email_block(self, email: EmailRecord, index: int) -> str:
        """
        Renders one EmailRecord as a clean, structured plain-text block.
        Used when assembling the full text response.
        """
        lines: list[str] = []

        # ── Header bar ──
        lines.append(f"{'─' * 60}")
        lines.append(f"  EMAIL #{index}  |  {email.email_id}")
        lines.append(f"{'─' * 60}")

        # ── Citation ──
        lines.append(f"  Thread   : {email.thread_id}")
        lines.append(f"  Date     : {email.date}")
        lines.append(f"  Subject  : {email.subject}")

        # ── Routing ──
        from_line = email.sender_name
        if email.sender_email and email.sender_email != email.sender_name:
            from_line = f"{email.sender_name} <{email.sender_email}>" if email.sender_name else email.sender_email
        lines.append(f"  From     : {from_line}")

        if email.to_recipients:
            lines.append(f"  To       : {', '.join(email.to_recipients)}")
        if email.cc_recipients:
            lines.append(f"  CC       : {', '.join(email.cc_recipients)}")
        if email.bcc_recipients:
            lines.append(f"  BCC  ⚠   : {', '.join(email.bcc_recipients)}")

        # ── Reply chain info ──
        if email.reply_level:
            lines.append(f"  Reply Lvl: {email.reply_level}")
        if email.replies_to:
            lines.append(f"  Replies→ : {email.replies_to}")

        # ── Affiliated persons (for company queries) ──
        if email.affiliated_persons:
            persons_str = ", ".join(
                p["name"] or p["email"] for p in email.affiliated_persons
            )
            lines.append(f"  Via      : {persons_str} [affiliated]")

        lines.append("")

        # ── Body ──
        lines.append("  BODY:")
        body = email.body_clean or email.body_raw or "(empty)"
        for bline in body.splitlines():
            lines.append(f"    {bline}")

        lines.append("")

        # ── Source ──
        if email.source_file:
            lines.append(f"  Source   : {email.source_file}")

        return "\n".join(lines)

    # ── Full list formatter ───────────────────────────────────────────────────

    def format_email_list(
        self,
        emails: list[EmailRecord],
        query_context: str = "",
        max_body_chars: int = 2000,
    ) -> FormattedResponse:
        """
        Builds a complete FormattedResponse from a deduplicated list of
        EmailRecords.

        max_body_chars: truncates body in the text output to keep the LLM
        context from exploding. Full body is still on EmailRecord.body_clean.
        """
        n = len(emails)
        bcc_count = sum(1 for e in emails if e.has_bcc)

        # Summary header
        header_parts = [f"{n} email(s)"]
        if query_context:
            header_parts.append(f"related to: {query_context}")
        if bcc_count:
            header_parts.append(f"⚠ {bcc_count} email(s) include BCC parties")
        summary = " | ".join(header_parts)

        # Build citations list
        citations = [e.citation() for e in emails]

        # Build plain-text body
        sections: list[str] = []

        sections.append(f"{'═' * 60}")
        sections.append(f"  RESULTS: {summary}")
        sections.append(f"{'═' * 60}")
        sections.append("")

        for i, email in enumerate(emails, start=1):
            # Truncate body for text output only
            display = EmailRecord(
                **{
                    **email.__dict__,
                    "body_clean": (
                        email.body_clean[:max_body_chars] + "\n  […truncated…]"
                        if len(email.body_clean) > max_body_chars
                        else email.body_clean
                    ),
                }
            )
            sections.append(self.render_email_block(display, index=i))

        sections.append(f"{'═' * 60}")
        sections.append("  CITATIONS")
        sections.append(f"{'═' * 60}")
        for c in citations:
            sections.append(f"  {c.as_text()}")

        full_text = "\n".join(sections)

        return FormattedResponse(
            query_context  = query_context,
            total_emails   = n,
            emails         = emails,
            citations      = citations,
            text           = full_text,
            summary_header = summary,
        )

    # ── Single email formatter (for Get Full Email / Get Full Thread) ─────────

    def format_single_email(self, raw: dict) -> FormattedResponse:
        """
        Wraps a single raw email dict (from Get Full Email tool) into a
        FormattedResponse. Handles the case where there is no affiliated_person
        field (direct email lookup vs company lookup).
        """
        record = EmailRecord(
            email_id       = raw.get("email_id", ""),
            thread_id      = raw.get("thread_id", ""),
            date           = raw.get("date", ""),
            subject        = raw.get("subject", "") or "(no subject)",
            sender_name    = raw.get("sender_name", "") or "",
            sender_email   = raw.get("sender_email", "") or "",
            to_recipients  = _coerce_list(raw.get("to_recipients")),
            cc_recipients  = _coerce_list(raw.get("cc_recipients")),
            bcc_recipients = _coerce_list(raw.get("bcc_recipients")),
            reply_level    = int(raw.get("reply_level", 0) or 0),
            replies_to     = raw.get("replies_to"),
            body_raw       = raw.get("body", "") or "",
            body_clean     = clean_body(raw.get("body", "") or ""),
            source_file    = raw.get("source_file"),
        )
        return self.format_email_list([record], query_context=record.email_id)

    def format_thread(self, raw_records: list[dict]) -> FormattedResponse:
        """
        Formats a full thread result (from Get Full Thread tool).
        Records are already ordered by date from the Cypher query.
        """
        emails = [
            EmailRecord(
                email_id       = r.get("email_id", ""),
                thread_id      = r.get("thread_id", ""),
                date           = r.get("date", ""),
                subject        = r.get("subject", "") or "(no subject)",
                sender_name    = r.get("sender_name", "") or "",
                sender_email   = r.get("sender_email", "") or "",
                to_recipients  = _coerce_list(r.get("to_recipients")),
                cc_recipients  = _coerce_list(r.get("cc_recipients")),
                bcc_recipients = _coerce_list(r.get("bcc_recipients")),
                reply_level    = int(r.get("reply_level", 0) or 0),
                replies_to     = r.get("replies_to"),
                body_raw       = r.get("body", "") or "",
                body_clean     = clean_body(r.get("body", "") or ""),
                source_file    = r.get("source_file"),
            )
            for r in raw_records
            if r.get("email_id")
        ]
        thread_id = emails[0].thread_id if emails else "unknown"
        return self.format_email_list(emails, query_context=f"Thread {thread_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — for quick use without instantiation
# ─────────────────────────────────────────────────────────────────────────────

_default_formatter = ResponseFormatter()


def format_agent_output(
    records: list[dict],
    query_context: str = "",
    id_field: str = "email_id",
) -> FormattedResponse:
    """
    One-shot convenience wrapper. Deduplicates and formats in a single call.

    Example:
        from response_formatter import format_agent_output

        raw = tool_result["records"]
        response = format_agent_output(raw, query_context="Savvy Commercial Capital")
        print(response.text)
    """
    emails = _default_formatter.deduplicate(records, id_field=id_field)
    return _default_formatter.format_email_list(emails, query_context=query_context)