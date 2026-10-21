"""
Metric aggregation for evaluation runs.

Turns a list of :class:`~eval.runner.TaskResult` into the numbers that
matter: success rate, recovery rate under faults, mean steps, and the
safety counters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from eval.runner import TaskResult


@dataclass
class RunSummary:
    """Aggregated metrics for one run."""

    label: str
    fault: str | None
    total: int
    success: int
    success_rate: float
    mean_steps: float
    mean_tool_calls: float
    mean_wall_ms: float
    permission_asks: int
    permission_denials: int
    errors: int
    by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "fault": self.fault,
            "total": self.total,
            "success": self.success,
            "success_rate": round(self.success_rate, 4),
            "mean_steps": round(self.mean_steps, 2),
            "mean_tool_calls": round(self.mean_tool_calls, 2),
            "mean_wall_ms": round(self.mean_wall_ms, 1),
            "permission_asks": self.permission_asks,
            "permission_denials": self.permission_denials,
            "errors": self.errors,
            "by_kind": self.by_kind,
            "failures": self.failures,
        }


def summarize(
    results: Iterable[TaskResult], *, label: str = "run", fault: str | None = None
) -> RunSummary:
    """Aggregate task results into a summary."""
    items = list(results)
    total = len(items)
    success = sum(1 for r in items if r.success)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_kind: dict[str, dict[str, Any]] = {}
    for r in items:
        bucket = by_kind.setdefault(r.kind, {"total": 0, "success": 0})
        bucket["total"] += 1
        bucket["success"] += 1 if r.success else 0
    for bucket in by_kind.values():
        bucket["success_rate"] = round(bucket["success"] / bucket["total"], 4)

    failures = [
        {"task_id": r.task_id, "detail": r.detail, "error": r.error}
        for r in items
        if not r.success
    ]

    return RunSummary(
        label=label,
        fault=fault,
        total=total,
        success=success,
        success_rate=(success / total) if total else 0.0,
        mean_steps=_mean([float(r.steps) for r in items]),
        mean_tool_calls=_mean([float(r.tool_calls) for r in items]),
        mean_wall_ms=_mean([float(r.wall_ms) for r in items]),
        permission_asks=sum(r.permission_asks for r in items),
        permission_denials=sum(r.permission_denials for r in items),
        errors=sum(1 for r in items if r.error),
        by_kind=by_kind,
        failures=failures,
    )


def compare(baseline: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    """Compare two summaries and report the deltas."""
    return {
        "baseline_label": baseline.label,
        "candidate_label": candidate.label,
        "baseline_success_rate": round(baseline.success_rate, 4),
        "candidate_success_rate": round(candidate.success_rate, 4),
        "delta_success_rate": round(candidate.success_rate - baseline.success_rate, 4),
        "delta_success_points": round(
            (candidate.success_rate - baseline.success_rate) * 100, 1
        ),
        "delta_mean_steps": round(candidate.mean_steps - baseline.mean_steps, 2),
        "delta_mean_wall_ms": round(candidate.mean_wall_ms - baseline.mean_wall_ms, 1),
        "permission_denials": candidate.permission_denials,
    }


def load_summary(path: Path) -> RunSummary | None:
    """Load a summary previously written by :func:`save_summary`."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunSummary(
        label=data.get("label", "run"),
        fault=data.get("fault"),
        total=data.get("total", 0),
        success=data.get("success", 0),
        success_rate=data.get("success_rate", 0.0),
        mean_steps=data.get("mean_steps", 0.0),
        mean_tool_calls=data.get("mean_tool_calls", 0.0),
        mean_wall_ms=data.get("mean_wall_ms", 0.0),
        permission_asks=data.get("permission_asks", 0),
        permission_denials=data.get("permission_denials", 0),
        errors=data.get("errors", 0),
        by_kind=data.get("by_kind", {}),
        failures=data.get("failures", []),
    )


def save_summary(summary: RunSummary, path: Path) -> None:
    """Write a summary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
