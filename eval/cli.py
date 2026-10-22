"""
Evaluation CLI.

Usage::

    python -m eval.cli --baseline
    python -m eval.cli --fault tool_timeout
    python -m eval.cli --all-faults
    python -m eval.cli --report

Writes results to ``benchmarks/results/`` and a Markdown report to
``benchmarks/results/report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.faults import ALL_FAULTS, fault_by_name
from eval.metrics import compare, load_summary, save_summary, summarize
from eval.report import render_report, write_report
from eval.runner import run_suite, save_results
from eval.tasks import TASKS

RESULTS_DIR = Path("benchmarks/results")
BASELINE_SUMMARY = RESULTS_DIR / "baseline.json"
REPORT_PATH = RESULTS_DIR / "report.md"


def _run(label: str, fault=None, limit: int | None = None, oracle: bool = False):
    print(f"[eval] running '{label}' over {limit or len(TASKS)} tasks...", flush=True)
    if oracle:
        from eval.oracle import use_oracle

        with use_oracle():
            results = run_suite(TASKS, fault=fault, limit=limit)
    else:
        results = run_suite(TASKS, fault=fault, limit=limit)
    summary = summarize(results, label=label, fault=fault.name if fault else None)
    save_results(results, RESULTS_DIR / f"{label}_details.json")
    save_summary(summary, RESULTS_DIR / f"{label}.json")
    print(
        f"[eval] {label}: {summary.success}/{summary.total} "
        f"({summary.success_rate * 100:.1f}%)  "
        f"mean_steps={summary.mean_steps:.2f}  "
        f"mean_latency={summary.mean_wall_ms:.0f}ms  "
        f"tokens_in={summary.tokens_in:,}",
        flush=True,
    )
    if summary.infra_errors:
        print(
            f"[eval] WARNING: {summary.infra_errors} task(s) could not run because "
            "every model endpoint was unavailable. They are excluded from the "
            "success rate, but the run is incomplete.",
            flush=True,
        )
    return results, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description="Agent evaluation harness")
    parser.add_argument("--baseline", action="store_true", help="run the no-fault baseline")
    parser.add_argument("--fault", help="run a single fault by name")
    parser.add_argument("--all-faults", action="store_true", help="run every fault class")
    parser.add_argument("--limit", type=int, help="only run the first N tasks")
    parser.add_argument("--report", action="store_true", help="render the Markdown report")
    parser.add_argument("--list-faults", action="store_true", help="list fault names")
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="use the scripted reference solver instead of a live model",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="regenerate the measured-results section of the README",
    )
    args = parser.parse_args(argv)

    if args.update_readme:
        from eval.readme import update_readme

        changed = update_readme()
        print("[eval] README results updated" if changed else "[eval] README already up to date")
        return 0

    if args.list_faults:
        for fault in ALL_FAULTS:
            print(f"{fault.name:26s} {fault.scope:5s} {fault.description}")
        return 0

    if args.report:
        baseline = load_summary(BASELINE_SUMMARY)
        if baseline is None:
            print("[eval] no baseline found — run with --baseline first", file=sys.stderr)
            return 1
        fault_summaries = []
        for fault in ALL_FAULTS:
            summary = load_summary(RESULTS_DIR / f"fault_{fault.name}.json")
            if summary:
                fault_summaries.append(summary)
        candidate = load_summary(RESULTS_DIR / "candidate.json")
        text = render_report(baseline, fault_summaries, candidate, oracle=args.oracle)
        write_report(text, REPORT_PATH)
        print(f"[eval] report written to {REPORT_PATH}")
        return 0

    baseline_summary = None
    if args.baseline or not args.fault and not args.all_faults:
        _, baseline_summary = _run("baseline", limit=args.limit, oracle=args.oracle)

    if args.fault:
        fault = fault_by_name(args.fault)
        if fault is None:
            print(f"[eval] unknown fault '{args.fault}'", file=sys.stderr)
            return 2
        _, summary = _run(
            f"fault_{fault.name}", fault=fault, limit=args.limit, oracle=args.oracle
        )
        if baseline_summary:
            print("[eval] delta:", json.dumps(compare(baseline_summary, summary), indent=2))
        return 0

    if args.all_faults:
        for fault in ALL_FAULTS:
            _run(
                f"fault_{fault.name}",
                fault=fault,
                limit=args.limit,
                oracle=args.oracle,
            )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
