"""
Robust JSON extraction from model output.

Models wrap JSON in code fences, prefix it with prose, or occasionally
double the braces. A shared extractor keeps the planner and the reviewer
from each re-implementing a slightly different version of the same
cleanup.
"""

from __future__ import annotations

import json
from typing import Any


def strip_code_fences(raw: str) -> str:
    """Remove a markdown code fence and any language tag around JSON."""
    text = raw.strip()
    if "```" not in text:
        return text
    parts = text.split("```")
    for part in parts:
        candidate = part.strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        if candidate.startswith("{"):
            return candidate
    return text


def undouble_braces(raw: str) -> str:
    """Collapse doubled braces into single ones."""
    text = raw.strip()
    while text.startswith("{{") and text.endswith("}}"):
        inner = text[1:-1]
        if not inner.strip().startswith("{"):
            break
        text = inner.strip()
    return text


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """
    Pull a JSON object out of a model response.

    Returns the parsed object, or ``None`` when nothing usable is found.
    """
    if not raw:
        return None

    for candidate in (raw, strip_code_fences(raw), undouble_braces(strip_code_fences(raw))):
        candidate = candidate.strip()
        if not candidate:
            continue
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            # One more attempt with the braces collapsed.
            collapsed = snippet.replace("{{", "{").replace("}}", "}")
            try:
                parsed = json.loads(collapsed)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None
