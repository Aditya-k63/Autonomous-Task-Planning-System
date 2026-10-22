"""
Context budget for the agent loop.

Every model call resends the system prompt, the tool schemas, and the
whole conversation so far. On a small or free-tier model that is the
dominant cost: a long task keeps paying for output it already has.

This module trims the message list to a fixed number of recent
tool-call pairs and replaces anything older with a compact note, so the
context stays bounded no matter how many steps a task takes.
"""

from __future__ import annotations

from typing import Any

# How many recent tool-call pairs to keep in full.
DEFAULT_KEEP_PAIRS = 3

# Cap on the characters kept for a single retained tool result.
MAX_RESULT_CHARS = 2000


def _is_tool_message(message: Any) -> bool:
    return type(message).__name__ == "ToolMessage"


def _is_ai_message(message: Any) -> bool:
    return type(message).__name__ == "AIMessage"


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars)"


def trim_messages(
    messages: list[Any], keep_pairs: int = DEFAULT_KEEP_PAIRS
) -> list[Any]:
    """
    Return a bounded copy of the message list.

    Keeps the system message and the most recent ``keep_pairs``
    tool-call exchanges in full. Older exchanges are replaced by a single
    summary line so the model still knows they happened.
    """
    if not messages:
        return list(messages)

    system = [m for m in messages if type(m).__name__ == "SystemMessage"]
    rest = [m for m in messages if type(m).__name__ != "SystemMessage"]

    # Walk backwards collecting complete tool-call exchanges.
    kept: list[Any] = []
    pairs = 0
    index = len(rest) - 1
    while index >= 0:
        message = rest[index]
        if _is_tool_message(message):
            pairs += 1
            if pairs > keep_pairs:
                break
        kept.append(message)
        index -= 1
    kept.reverse()

    dropped = len(rest) - len(kept)
    trimmed: list[Any] = list(system)
    if dropped > 0:
        trimmed.append(_summary_message(dropped))
    trimmed.extend(kept)
    return trimmed


def _summary_message(dropped: int) -> Any:
    """Build the placeholder that stands in for dropped exchanges."""
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=(
            f"[{dropped} earlier tool call(s) were dropped to keep the "
            "context small. Their results are summarised in the task history.]"
        )
    )


def truncate_tool_results(messages: list[Any], limit: int = MAX_RESULT_CHARS) -> list[Any]:
    """Shorten the content of every tool result in the list."""
    out: list[Any] = []
    for message in messages:
        if _is_tool_message(message):
            try:
                message.content = _truncate(str(message.content), limit)
            except Exception:
                pass
        out.append(message)
    return out
