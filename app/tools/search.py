"""Search tools — web search via Tavily or fallback."""

from __future__ import annotations

import os
import httpx

from .registry import Tool


def _web_search(query: str) -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return _fallback_search(query)
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query.strip(), "max_results": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No results found"
        lines = []
        for r in results[:5]:
            lines.append(f"**{r.get('title', 'Untitled')}**")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(r.get("content", "")[:300])
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def _fallback_search(query: str) -> str:
    try:
        resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return "No results (TAVILY_API_KEY not set, using Wikipedia fallback)"
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(r.get("snippet", "")[:300])
            lines.append("")
        return "\n".join(lines) + "\n(Note: using Wikipedia fallback — set TAVILY_API_KEY for full web search)"
    except Exception as e:
        return f"Search unavailable: {e}"


def get_search_tools() -> list[Tool]:
    return [
        Tool(
            "web_search",
            "Search the web for information.",
            _web_search,
            safe=True,
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"],
            },
        ),
    ]
