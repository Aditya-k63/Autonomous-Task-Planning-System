"""
End-to-end tests for the evaluation harness.

These run the real agent graph with a scripted oracle model, so they
exercise the planner, executor, tools, workspace, permission gate, and
verifiers without needing an API key.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from eval.oracle import use_oracle
from eval.runner import run_task
from eval.tasks import TASKS, task_by_id


@pytest.fixture
def workspace():
    path = Path(tempfile.mkdtemp(prefix="eval_e2e_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_oracle_solves_a_single_task(workspace):
    """The full loop — plan, execute, review — should complete a task."""
    task = task_by_id("write_lines_01")
    with use_oracle():
        result = run_task(task, workspace=workspace)
    assert result.success, f"{result.task_id} failed: {result.detail}"
    assert result.error is None
    assert result.tool_calls >= 1


def test_oracle_solves_every_task(workspace):
    """Every task in the suite must be solvable by the reference solver."""
    failures = []
    with use_oracle():
        for task in TASKS:
            result = run_task(task, workspace=workspace / task.id)
            if not result.success:
                failures.append(f"{task.id}: {result.detail}")
    assert not failures, "oracle failed: " + "; ".join(failures)


def test_permission_gate_denies_dangerous_command(workspace):
    """A dangerous shell command must be refused, not executed."""
    from app.permissions import PermissionDecision, PermissionPolicy

    policy = PermissionPolicy(mode="enforce")
    request = policy.check("run_shell", "rm -rf /")
    assert request.decision is PermissionDecision.DENY


def test_fault_injection_breaks_the_run(workspace):
    """A tool fault must be visible to the harness as a failure."""
    from eval.faults import fault_by_name

    task = task_by_id("write_lines_01")
    fault = fault_by_name("tool_exception")
    with use_oracle():
        result = run_task(task, fault=fault, workspace=workspace)
    assert not result.success
    assert result.fault == "tool_exception"


def test_results_serialise(workspace):
    from eval.runner import save_results

    task = task_by_id("write_lines_01")
    with use_oracle():
        result = run_task(task, workspace=workspace)
    payload = result.to_dict()
    assert payload["task_id"] == "write_lines_01"
    assert "success" in payload
    assert "wall_ms" in payload
