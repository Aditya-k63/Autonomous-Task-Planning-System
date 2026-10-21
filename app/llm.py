"""
LLM factory — provider-agnostic with a runtime model-fallback chain.

Supports OpenRouter (primary, free models), Groq, and xAI/Grok. All heavy
imports are deferred so importing this module stays cheap.

The chain in :mod:`app.llm_models` is walked at call time: each endpoint is
retried a few times for transient failures, then the next endpoint is tried.
"""

from __future__ import annotations

import logging
import time

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
        )
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model=spec.model,
            temperature=0.3,
            max_tokens=4096,
        )
    if spec.provider == "xai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
            model=spec.model,
            temperature=0.3,
            max_tokens=4096,
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


def invoke_with_retry(llm, messages, max_retries: int = MAX_RETRIES):
    """
    Invoke an LLM, retrying transient failures on the same endpoint.

    Kept for callers that hold a single client. :func:`invoke_with_fallback`
    is the entry point that walks the whole chain.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            if is_transient(e) and attempt < max_retries - 1:
                delay = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    "LLM transient error (attempt %s/%s): %s. Retrying in %ss...",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                raise
    raise last_error  # pragma: no cover - defensive


def invoke_with_fallback(messages, settings=None, max_retries: int = MAX_RETRIES):
    """
    Invoke the LLM, walking the fallback chain until one endpoint succeeds.

    Returns the model response. Raises the last error if every endpoint fails.
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
        try:
            client = _build_client(spec, settings)
        except Exception as e:  # missing client library for this provider
            logger.warning("Skipping %s: %s", spec.label, e)
            continue
        try:
            logger.info("Using model %s", spec.label)
            return invoke_with_retry(client, messages, max_retries=max_retries)
        except Exception as e:
            last_error = e
            logger.warning("Model %s failed: %s. Trying next endpoint.", spec.label, e)
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
