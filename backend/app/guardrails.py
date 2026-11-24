"""Multi-layer input, document-fencing, and output security guardrails.

Why this module exists:
    Enforces defense-in-depth across every Q&A request:
      1. Input Guardrails: rejects empty/overlong queries and detects direct
         prompt-injection or jailbreak attempts before retrieval runs.
      2. Document Fencing: wraps untrusted chunk text in `<retrieved_context>`
         XML tags and neutralizes embedded closing tags to block indirect
         prompt injection hidden inside uploaded files (Edge Case #12).
      3. Output Guardrails: verifies that every inline `[Chunk ID: ...]` tag
         matches an actually retrieved chunk and scrubs leaked credentials or
         system-prompt fragments from outbound responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class _ChunkLike(Protocol):
    """Minimal structural protocol satisfied by RetrievedChunk and test stubs."""

    chunk_id: str
    filename: str
    page_number: int
    content: str


# Compiled input prompt-injection / jailbreak patterns
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(dan|an?\s+unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output|leak)\s+(your\s+)?(system\s+prompt|instructions|api[_\s]?key)", re.IGNORECASE),
    re.compile(r"</?\s*retrieved_context\s*>", re.IGNORECASE),
    re.compile(r"</?\s*system\s*>", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(root|developer\s+mode|god\s+mode)\b", re.IGNORECASE),
)

# Inline citation format enforced by the system prompt:
#   [Chunk ID: <id> | <filename>, p. <page_number>]
_CITATION_PATTERN = re.compile(
    r"\[Chunk\s+ID:\s*([^|\]]+?)\s*\|\s*([^,\]]+?),\s*p\.\s*([^\]]+?)\]",
    re.IGNORECASE,
)

# Outbound secret / sensitive leakage masks
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("groq_api_key", re.compile(r"gsk_[A-Za-z0-9_\-]{10,}")),
    ("cohere_bearer", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)),
    ("postgres_uri", re.compile(r"postgres(?:ql)?(?:\+asyncpg)?://[^\s\"'<>]+", re.IGNORECASE)),
    ("system_prompt_leak", re.compile(r"CRITICAL\s+OPERATIONAL\s+RULES:", re.IGNORECASE)),
)


@dataclass(frozen=True)
class InputCheckResult:
    """Result of evaluating a raw user query against input guardrails."""

    is_safe: bool
    reason: str | None = None
    matched_rule: str | None = None


@dataclass(frozen=True)
class OutputCheckResult:
    """Result of verifying citations and scrubbing secrets on an LLM response."""

    cleaned_answer: str
    valid_cited_ids: list[str]
    violations: list[str]


def check_query_input(query: str, *, max_length: int = 1000) -> InputCheckResult:
    """Validate query length and screen for prompt-injection/jailbreak attempts.

    Args:
        query: Raw user question string.
        max_length: Maximum allowed character count (default: 1000).

    Returns:
        `InputCheckResult` indicating whether the query may proceed to retrieval.
    """
    stripped = (query or "").strip()
    if not stripped:
        return InputCheckResult(
            is_safe=False,
            reason="Query must not be empty.",
            matched_rule="empty_query",
        )
    if len(stripped) > max_length:
        return InputCheckResult(
            is_safe=False,
            reason=f"Query exceeds maximum allowed length of {max_length} characters.",
            matched_rule="max_length_exceeded",
        )

    for idx, pattern in enumerate(_INJECTION_PATTERNS):
        if pattern.search(stripped):
            return InputCheckResult(
                is_safe=False,
                reason="Potential prompt injection or jailbreak attempt detected.",
                matched_rule=f"injection_pattern_{idx}",
            )

    return InputCheckResult(is_safe=True)


def _escape_fence_tags(text: str) -> str:
    """Neutralize XML fence breakout attempts embedded inside untrusted document text."""
    return (
        text.replace("</retrieved_context>", "&lt;/retrieved_context&gt;")
        .replace("<retrieved_context>", "&lt;retrieved_context&gt;")
        .replace("</chunk>", "&lt;/chunk&gt;")
    )


def wrap_retrieved_context(chunks: list[_ChunkLike]) -> str:
    """Isolate retrieved document chunks inside `<retrieved_context>` XML tags.

    Args:
        chunks: Ordered list of retrieved chunks to expose to the LLM.

    Returns:
        XML-fenced context block treating all document text as untrusted data.
    """
    lines = ["<retrieved_context>"]
    for chunk in chunks:
        safe_filename = chunk.filename.replace('"', "&quot;")
        safe_content = _escape_fence_tags(chunk.content)
        lines.append(
            f'  <chunk id="{chunk.chunk_id}" filename="{safe_filename}" page="{chunk.page_number}">'
        )
        lines.append(f"    {safe_content}")
        lines.append("  </chunk>")
    lines.append("</retrieved_context>")
    return "\n".join(lines)


def verify_and_scrub_output(
    answer: str,
    valid_chunk_ids: set[str],
) -> OutputCheckResult:
    """Validate inline citation IDs against retrieved chunks and scrub leaked secrets.

    Args:
        answer: Raw text generated by the LLM.
        valid_chunk_ids: Set of chunk ID strings that were actually retrieved
            for the current request.

    Returns:
        `OutputCheckResult` containing the sanitized answer, ordered unique
        verified chunk IDs, and any guardrail violation messages.
    """
    violations: list[str] = []
    valid_cited_ids: list[str] = []

    def _replace_citation(match: re.Match[str]) -> str:
        cited_id = match.group(1).strip()
        if cited_id not in valid_chunk_ids:
            violations.append(f"Stripped hallucinated citation ID: {cited_id}")
            return ""
        if cited_id not in valid_cited_ids:
            valid_cited_ids.append(cited_id)
        return match.group(0)

    cleaned = _CITATION_PATTERN.sub(_replace_citation, answer)

    # Collapse any double spaces left behind by stripped hallucinated citations.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(cleaned):
            violations.append(f"Redacted sensitive pattern from output: {label}")
            cleaned = pattern.sub("[REDACTED]", cleaned)

    return OutputCheckResult(
        cleaned_answer=cleaned,
        valid_cited_ids=valid_cited_ids,
        violations=violations,
    )
