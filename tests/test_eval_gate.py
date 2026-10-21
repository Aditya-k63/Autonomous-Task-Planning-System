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
