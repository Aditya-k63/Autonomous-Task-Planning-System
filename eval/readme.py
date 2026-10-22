"""
Regenerate the measured-results section of the README.

The README carries a block between two markers. This module rewrites it
from the summaries in ``benchmarks/results`` so the numbers in the README
always match a real run rather than a hand-copied table.
"""

from __future__ import annotations

from pathlib import Path

from eval.faults import ALL_FAULTS
from eval.metrics import RunSummary, load_summary
from eval.report import summary_table

START_MARKER = "<!-- EVAL_RESULTS:START -->"
END_MARKER = "<!-- EVAL_RESULTS:END -->"

README_PATH = Path("README.md")
RESULTS_DIR = Path("benchmarks/results")


def _fault_rows(results_dir: Path) -> list[str]:
    rows = []
    for fault in ALL_FAULTS:
        summary = load_summary(results_dir / f"fault_{fault.name}.json")
        if summary is None:
            continue
        rows.append(
            f"| `{fault.name}` | {summary.total} | {summary.success} | "
            f"{summary.success_rate * 100:.1f}% | {summary.mean_steps:.2f} | "
            f"{summary.infra_errors} |"
        )
    return rows


def build_section(results_dir: Path = RESULTS_DIR) -> str:
    """Build the markdown block for the README."""
    baseline = load_summary(results_dir / "baseline.json")
    if baseline is None:
        return "## Measured results\n\n_No run recorded yet. Run `python -m eval.cli --baseline`._"

    lines = ["## Measured results", ""]
    lines.append(
        f"**{baseline.success}/{baseline.total} tasks solved "
        f"({baseline.success_rate * 100:.1f}%)**, "
        f"mean {baseline.mean_steps:.2f} steps and "
        f"{baseline.mean_tool_calls:.2f} tool calls per task, "
        f"{baseline.mean_wall_ms:,.0f} ms per task, "
        f"{baseline.mean_tokens_in:,.0f} input tokens per task."
    )
    lines.append("")

    fault_rows = _fault_rows(results_dir)
    if fault_rows:
        lines += [
            "### Fault injection",
            "",
            "Each fault class is injected at the tool or model seam and the "
            "whole suite is re-run.",
            "",
            "| Fault | Tasks | Solved | Success rate | Mean steps | Infra errors |",
            "|---|---:|---:|---:|---:|---:|",
            *fault_rows,
            "",
        ]

    lines += [
        "Regenerate with `python -m eval.cli --baseline` and "
        "`python -m eval.cli --update-readme`. The full report, including "
        "per-task-kind breakdowns and failure detail, is written to "
        "`benchmarks/results/report.md`.",
        "",
    ]
    return "\n".join(lines)


def update_readme(
    readme_path: Path = README_PATH, results_dir: Path = RESULTS_DIR
) -> bool:
    """Rewrite the results block in the README. Returns True if changed."""
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        return False
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    section = build_section(results_dir)
    updated = text[:start] + f"{START_MARKER}\n{section}\n{END_MARKER}" + text[end:]
    if updated == text:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True
