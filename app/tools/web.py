"""Web tools — fetch URL content."""

from __future__ import annotations

import httpx

from .registry import Tool


def _fetch_url(url: str) -> str:
    try:
        resp = httpx.get(url.strip(), timeout=15, follow_redirects=True, headers={"User-Agent": "TaskPlannerAgent/1.0"})
        resp.raise_for_status()
        text = resp.text[:8000]
        return text
    except Exception as e:
        return f"ERROR: {e}"


def get_web_tools() -> list[Tool]:
    return [
        Tool(
            "fetch_url",
            "Fetch a URL and return its text content.",
            _fetch_url,
            safe=True,
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string", "description": "Absolute URL to fetch."}
                },
                "required": ["url"],
            },
        ),
    ]
