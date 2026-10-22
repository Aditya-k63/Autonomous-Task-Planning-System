"""
LLM factory — provider-agnostic with a runtime model-fallback chain.

Supports OpenRouter (primary, free models), Groq, and xAI/Grok. All heavy
imports are deferred so importing this module stays cheap.

The chain in :mod:`app.llm_models` is walked at call time: each endpoint is
retried a few times for transient failures, then the next endpoint is tried.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.llm_models import ModelSpec, build_model_chain

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

# Substrings that mark a failure as worth retrying on the same endpoint.
TRANSIENT_MARKERS = (
    "rate",
    "limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "overloaded",
    "resourceexhausted",
    "capacity",
    "connection",
    "temporarily unavailable",
    "service unavailable",
)

# Some provider edges reject requests that carry a default library
# User-Agent, so every client identifies itself as a normal HTTP client.
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Longest we will ever sit and wait for a rate limit to clear. Past this
# the endpoint is skipped so the chain can move on to another provider.
MAX_RATE_LIMIT_WAIT = 30.0

# An endpoint that reports a rate limit is quarantined for this long, so
# later calls in the same run skip it instead of paying the round trip.
QUARANTINE_SECONDS = 900.0

# Endpoint label -> monotonic time at which it may be tried again.
_quarantine: dict[str, float] = {}


def quarantine_endpoint(label: str, seconds: float = QUARANTINE_SECONDS) -> None:
    """Mark an endpoint as temporarily unusable."""
    _quarantine[label] = time.monotonic() + seconds
    logger.warning("Quarantined endpoint %s for %ss", label, round(seconds))


def is_quarantined(label: str) -> bool:
    """Return True when the endpoint is inside its quarantine window."""
    until = _quarantine.get(label)
    if until is None:
        return False
    if time.monotonic() >= until:
        _quarantine.pop(label, None)
        return False
    return True


def clear_quarantine() -> None:
    """Forget every quarantined endpoint."""
    _quarantine.clear()


# Endpoints that have reported they cannot do tool calling. They are
# still useful for the planner and the reviewer, which do not need
# tools, so they are tracked separately from the rate-limit quarantine.
_no_tool_support: set[str] = set()


def supports_tool_calling(label: str) -> bool:
    """Return False once an endpoint has said it cannot bind tools."""
    return label not in _no_tool_support


def mark_no_tool_support(label: str) -> None:
    """Record that an endpoint cannot be used for tool calls."""
    _no_tool_support.add(label)
    logger.info("Endpoint %s does not support tool calling", label)


def _is_tool_calling_unsupported(error: Exception) -> bool:
    """Detect the provider error raised when a model cannot bind tools."""
    text = str(error).lower()
    return "tool calling" in text and (
        "not supported" in text or "does not support" in text
    )

# Some providers state the wait explicitly, e.g. "Please try again in
# 1m3.936s" or "in 4m0.624s" or "in 45s". The format is compound, so
# hours, minutes, and seconds can all appear in one value.
_WAIT_PATTERN = re.compile(
    r"try again in\s*"
    r"(?:(\d+)\s*h)?\s*"
    r"(?:(\d+)\s*m)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)


def rate_limit_wait_hint(error: Exception) -> float | None:
    """
    Return the wait a provider asked for, in seconds.

    Handles the compound form providers use, e.g. ``1m3.936s`` is
    63.936 seconds. Returns ``None`` when the error states no wait.
    """
    match = _WAIT_PATTERN.search(str(error))
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    if not (hours or minutes or seconds):
        return None
    try:
        return (
            float(hours or 0) * 3600.0
            + float(minutes or 0) * 60.0
            + float(seconds or 0)
        )
    except ValueError:
        return None


def is_transient(error: Exception) -> bool:
    """Return True when the error looks worth retrying the same endpoint."""
    text = str(error).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _build_client(spec: ModelSpec, settings) -> object:
    """Instantiate the LangChain client for one endpoint."""
    if spec.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model=spec.model,
            temperature=0.3,
            max_tokens=4096,
            default_headers=DEFAULT_HEADERS,
        )
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model=spec.model,
            temperature=0.3,
            max_tokens=4096,
            default_headers=DEFAULT_HEADERS,
        )
    if spec.provider == "xai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
            model=spec.model,
            temperature=0.3,
            max_tokens=4096,
            default_headers=DEFAULT_HEADERS,
        )
    raise ValueError(f"Unknown provider: {spec.provider}")


def _chain(settings) -> list[ModelSpec]:
    return build_model_chain(
        openrouter_api_key=settings.openrouter_api_key,
        groq_api_key=settings.groq_api_key,
        xai_api_key=settings.xai_api_key,
        preferred_model=settings.llm_model,
        preferred_provider=settings.llm_provider,
    )


class AllEndpointsUnavailable(RuntimeError):
    """Raised when every configured endpoint is blocked or rate-limited."""


def invoke_with_retry(llm, messages, max_retries: int = MAX_RETRIES):
    """
    Invoke an LLM, retrying transient failures on the same endpoint.

    A rate limit that says how long to wait is honoured when that wait is
    short. When the wait is longer than :data:`MAX_RATE_LIMIT_WAIT` the
    endpoint is abandoned instead, because sleeping through a daily quota
    reset would stall the run and every other model on the same account
    is blocked by the same limit anyway.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            if not is_transient(e) or attempt >= max_retries - 1:
                raise

            hint = rate_limit_wait_hint(e)
            if hint is not None:
                if hint > MAX_RATE_LIMIT_WAIT:
                    logger.warning(
                        "Rate limit needs %ss — skipping this endpoint "
                        "instead of waiting.", round(hint, 1),
                    )
                    raise
                delay = hint
            else:
                delay = RETRY_DELAY * (attempt + 1)

            logger.warning(
                "LLM transient error (attempt %s/%s): %s. Retrying in %ss...",
                attempt + 1,
                max_retries,
                str(e)[:200],
                round(delay, 1),
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover - defensive


def invoke_with_fallback(
    messages,
    settings=None,
    max_retries: int = MAX_RETRIES,
    tools: list[dict[str, Any]] | None = None,
):
    """
    Invoke the LLM, walking the fallback chain until one endpoint succeeds.

    ``tools`` is an optional list of OpenAI-style function definitions. When
    given, the client is bound to them so the model can emit native tool
    calls. Returns the model response. Raises the last error if every
    endpoint fails, or :class:`AllEndpointsUnavailable` when every
    endpoint is blocked or rate-limited.
    """
    from app.config import get_settings

    settings = settings or get_settings()
    chain = _chain(settings)
    if not chain:
        raise RuntimeError(
            "No LLM provider configured. Set OPENROUTER_API_KEY, GROQ_API_KEY, "
            "or XAI_API_KEY."
        )

    last_error: Exception | None = None
    for spec in chain:
        if is_quarantined(spec.label):
            continue
        if tools and not supports_tool_calling(spec.label):
            continue
        try:
            client = _build_client(spec, settings)
        except Exception as e:  # missing client library for this provider
            logger.warning("Skipping %s: %s", spec.label, e)
            continue
        try:
            if tools:
                try:
                    client = client.bind_tools(tools)
                except Exception as e:
                    logger.warning("Tool binding unsupported on %s: %s", spec.label, e)
            logger.info("Using model %s", spec.label)
            return invoke_with_retry(client, messages, max_retries=max_retries)
        except Exception as e:
            last_error = e
            if _is_tool_calling_unsupported(e):
                mark_no_tool_support(spec.label)
            elif rate_limit_wait_hint(e) is not None:
                quarantine_endpoint(spec.label)
            logger.warning("Model %s failed: %s. Trying next endpoint.", spec.label, str(e)[:200])
    if last_error is None:
        raise AllEndpointsUnavailable(
            "Every configured endpoint is rate-limited or does not support "
            "tool calling. Wait for the quota to reset or add another provider."
        )
    raise RuntimeError(f"All LLM endpoints failed. Last error: {last_error}")


def get_llm():
    """
    Return the primary LLM client for the highest-preference endpoint.

    Kept for callers that need a bound client (e.g. passing ``.bind_tools``).
    Use :func:`invoke_with_fallback` for plain invocations.
    """
    from app.config import get_settings

    settings = get_settings()
    chain = _chain(settings)
    if not chain:
        raise RuntimeError(
            "No LLM provider configured. Set OPENROUTER_API_KEY, GROQ_API_KEY, "
            "or XAI_API_KEY."
        )
    return _build_client(chain[0], settings)
