"""Tests for fault injection — the mechanism that makes the harness work."""

import random

import pytest

from eval.faults import ALL_FAULTS, fault_by_name, fault_names
from eval.runner import apply_fault
from app.tools.registry import Tool, ToolRegistry


def test_catalogue_is_populated():
    assert len(ALL_FAULTS) >= 10
    assert len(fault_names("tool")) >= 6
    assert len(fault_names("llm")) >= 3


def test_fault_lookup():
    assert fault_by_name("tool_timeout") is not None
    assert fault_by_name("does_not_exist") is None


def test_tool_fault_is_installed_and_restored():
    fault = fault_by_name("tool_exception")
    reg = ToolRegistry(retry_base_delay=0.0, max_retries=0)
    reg.register(Tool("t", "T", lambda _: "OK"))
    before = Tool.execute

    with apply_fault(fault, random.Random(0)):
        assert Tool.execute is not before
        out = reg.execute("t", "x")
        # A tool exception is surfaced to the agent as an ERROR: result,
        # not as an unhandled crash.
        assert out.startswith("ERROR:")

    assert Tool.execute is before


def test_tool_fault_bypasses_the_original_tool():
    fault = fault_by_name("tool_malformed_output")
    reg = ToolRegistry(retry_base_delay=0.0, max_retries=0)
    reg.register(Tool("t", "T", lambda _: "the original output"))

    with apply_fault(fault, random.Random(0)):
        out = reg.execute("t", "x")

    assert "unexpected" in out
    assert "the original output" not in out


def test_no_fault_leaves_dispatch_untouched():
    reg = ToolRegistry(retry_base_delay=0.0)
    reg.register(Tool("t", "T", lambda _: "fine"))
    with apply_fault(None, random.Random(0)):
        assert reg.execute("t", "x") == "fine"


def test_fault_respects_target():
    """A targeted fault must not fire on other tools."""
    fault = fault_by_name("tool_exception")
    fault = type(fault)(**{**fault.__dict__, "target": "other_tool"})
    reg = ToolRegistry(retry_base_delay=0.0, max_retries=0)
    reg.register(Tool("t", "T", lambda _: "OK"))

    with apply_fault(fault, random.Random(0)):
        out = reg.execute("t", "x")

    assert out == "OK"


def test_prompt_injection_fault_carries_instructions():
    fault = fault_by_name("tool_prompt_injection")
    reg = ToolRegistry(retry_base_delay=0.0, max_retries=0)
    reg.register(Tool("t", "T", lambda _: "benign"))
    with apply_fault(fault, random.Random(0)):
        out = reg.execute("t", "x")
    assert "ignore all previous instructions" in out.lower()
