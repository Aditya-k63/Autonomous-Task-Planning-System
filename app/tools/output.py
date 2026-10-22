"""
Tool output isolation.

Untrusted tool output (web pages, file contents, command results) must not
be read by the model as instructions. This module caps the output size and
wraps it in a delimiter the executor prompt tells the model to treat as
data.

The registry deliberately returns raw output — isolation is applied at the
boundary where data enters the model context, not at dispatch.
"""

from __future__ import annotations

MAX_TOOL_OUTPUT_CHARS = 8000

OPEN_MARKER = "<<<TOOL_OUTPUT_UNTRUSTED>>>"
CLOSE_MARKER = "<<<END_TOOL_OUTPUT>>>"


def isolate_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Cap and delimit tool output so it cannot be read as instructions."""
    body = text if text is not None else ""
    truncated = False
    if len(body) > max_chars:
        body = body[:max_chars]
        truncated = True
    suffix = "\n[... output truncated ...]" if truncated else ""
    return f"{OPEN_MARKER}{body}{suffix}{CLOSE_MARKER}"


def strip_markers(text: str) -> str:
    """Remove the isolation markers, returning the original payload."""
    if OPEN_MARKER not in text:
        return text
    body = text.split(OPEN_MARKER, 1)[1]
    return body.split(CLOSE_MARKER, 1)[0]
