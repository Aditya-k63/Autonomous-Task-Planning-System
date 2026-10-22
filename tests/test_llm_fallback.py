"""Tests for the LLM fallback chain and rate-limit handling."""

import time

import pytest

from app.llm import (
    MAX_RATE_LIMIT_WAIT,
    is_transient,
    rate_limit_wait_hint,
)


def test_transient_errors_are_recognised():
    assert is_transient(Exception("429 Too Many Requests")) is True
    assert is_transient(Exception("Error code: 503 Service Unavailable")) is True
    assert is_transient(Exception("model is overloaded")) is True
    assert is_transient(Exception("invalid api key")) is False


def test_wait_hint_is_parsed_from_the_error():
    error = Exception(
        "Rate limit reached ... Please try again in 1m3.936s. Need more tokens?"
    )
    hint = rate_limit_wait_hint(error)
    assert hint is not None
    assert 63 <= hint <= 65


def test_wait_hint_parses_seconds_and_minutes():
    assert rate_limit_wait_hint(Exception("try again in 45s")) == 45
    assert rate_limit_wait_hint(Exception("try again in 4m0.624s")) == pytest.approx(240.624, abs=0.1)


def test_wait_hint_is_none_when_not_stated():
    assert rate_limit_wait_hint(Exception("some other failure")) is None


def test_long_rate_limit_exceeds_the_wait_ceiling():
    """A daily quota reset must not be waited out."""
    error = Exception("Please try again in 4m0.624s")
    hint = rate_limit_wait_hint(error)
    assert hint is not None
    assert hint > MAX_RATE_LIMIT_WAIT


def test_quarantine_skips_a_blocked_endpoint():
    from app.llm import clear_quarantine, is_quarantined, quarantine_endpoint

    clear_quarantine()
    assert is_quarantined("groq:some-model") is False
    quarantine_endpoint("groq:some-model", seconds=60)
    assert is_quarantined("groq:some-model") is True
    clear_quarantine()
    assert is_quarantined("groq:some-model") is False


def test_quarantine_expires():
    from app.llm import clear_quarantine, is_quarantined, quarantine_endpoint

    clear_quarantine()
    quarantine_endpoint("groq:expiring", seconds=0.01)
    time.sleep(0.05)
    assert is_quarantined("groq:expiring") is False
    clear_quarantine()


def test_unavailable_model_is_recognised():
    from app.llm import _is_model_unavailable

    assert _is_model_unavailable(
        Exception(
            "The model `llama-3.1-8b-instant` does not exist or you do not "
            "have access to it."
        )
    ) is True
    assert _is_model_unavailable(Exception("429 Too Many Requests")) is False


def test_unavailable_model_is_skipped_afterwards():
    from app.llm import mark_model_unavailable, model_is_available

    mark_model_unavailable("groq:retired-model")
    assert model_is_available("groq:retired-model") is False
    assert model_is_available("groq:working-model") is True
