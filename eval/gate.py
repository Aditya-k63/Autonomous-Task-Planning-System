"""
CI gate for the evaluation benchmark.

Exits non-zero when the agent's success rate on the benchmark falls
below the required threshold. Used by GitHub Actions to block a
regression in the agent, the tools, or the workspace layer.

Usage::

    python -m eval.gate --min-success-rate 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SUMMARY = Path("benchmarks/results/baseline.json")
REQUIRED_SUCCESS_RATE = 1.0


def check(summary_path: Path, min_success_rate: float) -> int:
    """Return a process exit code: 0 pass, 1 fail."""
    if not summary_path.exists():
        print(f"[gate] no summary at {summary_path} — run the benchmark first")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rate = float(summary.get("success_rate", 0.0))
    total = summary.get("total", 0)
    success = summary.get("success", 0)

    print(f"[gate] {success}/{total} tasks solved ({rate * 100:.1f}%)")

    if rate < min_success_rate:
        print(
            f"[gate] FAIL: success rate {rate * 100:.1f}% is below the "
            f"required {min_success_rate * 100:.1f}%"
        )
        for failure in summary.get("failures", []):
            print(f"  - {failure['task_id']}: {failure['detail']}")
        return 1

    print(f"[gate] PASS: every task solved (threshold {min_success_rate * 100:.0f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.gate")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--min-success-rate", type=float, default=REQUIRED_SUCCESS_RATE)
    args = parser.parse_args(argv)
    return check(Path(args.summary), args.min_success_rate)


if __name__ == "__main__":
    raise SystemExit(main())
