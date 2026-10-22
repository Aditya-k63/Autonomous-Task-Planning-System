"""Tests for tool dispatch — validation, retries, circuit breaker, isolation."""

import pytest

from app.tools.output import OPEN_MARKER, isolate_output, strip_markers
from app.tools.registry import (
    CircuitBreaker,
    Tool,
    ToolRegistry,
    validate_args,
    validate_input,
)


def make_registry(**kwargs):
    return ToolRegistry(retry_base_delay=0.0, **kwargs)


def test_registry_executes_tool():
    reg = make_registry()
    reg.register(Tool("upper", "Uppercase input", lambda x: x.upper()))
    assert reg.execute("upper", "hello") == "HELLO"


def test_registry_rejects_unknown_tool():
    reg = make_registry()
    assert "Unknown tool" in reg.execute("nope", "x")


def test_registry_retries_then_succeeds():
    reg = make_registry(max_retries=2)
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            return "ERROR: transient"
        return "OK"

    reg.register(Tool("flaky", "Flaky", flaky))
    assert reg.execute("flaky", "x") == "OK"
    assert calls["n"] == 3


def test_registry_gives_up_after_retry_budget():
    reg = make_registry(max_retries=1)
    calls = {"n": 0}

    def always_fail(_):
        calls["n"] += 1
        return "ERROR: always"

    reg.register(Tool("bad", "Bad", always_fail))
    assert "ERROR: always" in reg.execute("bad", "x")
    assert calls["n"] == 2


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(threshold=2, cooldown=60)
    breaker.record_failure("t")
    breaker.record_failure("t")
    assert breaker.is_open("t") is True


def test_circuit_breaker_blocks_calls_once_open():
    reg = make_registry(max_retries=0, breaker_threshold=1, breaker_cooldown=60)
    reg.register(Tool("bad", "Bad", lambda _: "ERROR: nope"))
    reg.execute("bad", "x")
    assert "temporarily disabled" in reg.execute("bad", "x")


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=3, cooldown=60)
    breaker.record_failure("t")
    breaker.record_success("t")
    assert breaker.is_open("t") is False


def test_schema_validation_rejects_bad_args():
    """Structured arguments are validated against the tool's schema."""
    schema = {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}
    assert validate_args(schema, {"other": 1}) is not None
    assert validate_args(schema, {"path": "a.txt"}) is None


def test_schema_validation_rejects_malformed_json():
    """A string-typed schema rejects a value of the wrong shape."""
    schema = {"type": "string", "minLength": 3}
    assert validate_input(schema, "ab") is not None
    assert validate_input(schema, "abc") is None


def test_legacy_string_args_skip_object_schema_validation():
    """Object schemas are validated on structured args, not the flat form."""
    schema = {"type": "object", "required": ["path"]}
    assert validate_input(schema, "not json at all") is None


def test_registry_tracks_stats():
    reg = make_registry()
    reg.register(Tool("ok", "Ok", lambda _: "fine"))
    reg.execute("ok", "x")
    stats = reg.snapshot_stats()
    assert stats["ok"]["calls"] == 1
    assert stats["ok"]["success"] == 1


def test_isolate_output_marks_and_caps():
    wrapped = isolate_output("hello world", max_chars=5)
    assert OPEN_MARKER in wrapped
    assert "truncated" in wrapped
    assert strip_markers(wrapped).startswith("hello")


def test_isolate_output_keeps_short_output_intact():
    wrapped = isolate_output("hello", max_chars=50)
    assert "truncated" not in wrapped
    assert strip_markers(wrapped) == "hello"
