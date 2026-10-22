"""Tests for the CI evaluation gate."""

import json
from pathlib import Path

from eval.gate import check


def _write(path: Path, rate: float, failures=None) -> Path:
    path.write_text(
        json.dumps(
            {
                "label": "baseline",
                "total": 15,
                "success": int(rate * 15),
                "success_rate": rate,
                "failures": failures or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gate_passes_at_full_success(tmp_path):
    summary = _write(tmp_path / "s.json", 1.0)
    assert check(summary, 1.0) == 0


def test_gate_fails_below_threshold(tmp_path):
    summary = _write(
        tmp_path / "s.json",
        0.5,
        failures=[{"task_id": "t1", "detail": "boom", "error": None}],
    )
    assert check(summary, 1.0) == 1


def test_gate_fails_when_summary_is_missing(tmp_path):
    assert check(tmp_path / "nope.json", 1.0) == 1


def test_infrastructure_errors_are_not_scored_as_failures():
    """A rate-limited provider must never look like a failing agent."""
    from eval.metrics import summarize
    from eval.runner import TaskResult

    results = [
        TaskResult(task_id="a", kind="file_create", tags=[], success=True, detail="ok"),
        TaskResult(task_id="b", kind="file_create", tags=[], success=False, detail="nope"),
        TaskResult(task_id="c", kind="file_create", tags=[], success=False, detail="quota", infra_error=True),
    ]
    summary = summarize(results, label="t")
    assert summary.total == 2
    assert summary.success == 1
    assert summary.success_rate == 0.5
    assert summary.infra_errors == 1


def test_suite_stops_when_the_provider_is_unavailable():
    """Remaining tasks are marked as not attempted, not as failures."""
    from eval.runner import run_suite
    from eval.tasks import TASKS

    results = run_suite(TASKS, token_budget=1)
    assert all(r.infra_error for r in results)
    assert sum(1 for r in results if "not attempted" in r.detail) == len(results) - 1
