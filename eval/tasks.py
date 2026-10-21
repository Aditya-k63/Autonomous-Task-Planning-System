"""
Benchmark task suite.

Every task has a **programmatic verifier** — a function that inspects the
workspace and returns ``(passed, detail)``. No LLM judge is used for
ground truth, so runs are deterministic and cheap enough for CI.

Task kinds mirror the tool surface a coding agent actually needs:
read, write, edit, search, shell, and multi-step chains.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Verifier = Callable[[Path], "tuple[bool, str]"]


@dataclass(frozen=True)
class Task:
    """One benchmark task."""

    id: str
    prompt: str
    kind: str
    verify: Verifier
    tags: tuple[str, ...] = ()
    max_steps: int = 12
    setup: Callable[[Path], None] | None = None
    expected_tool: str | None = None

    def prepare(self, workspace: Path) -> None:
        """Seed the workspace with any fixture files the task needs."""
        if self.setup:
            self.setup(workspace)


# ---------------------------------------------------------------------------
# Verifier builders
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return (workspace_path(path)).read_text(encoding="utf-8")


def workspace_path(path: Path) -> Path:
    """Resolve a path inside the sandbox root."""
    return path


def file_exists_with_lines(
    rel_path: str, min_lines: int = 1, max_lines: int | None = None
) -> Verifier:
    """File exists, has a line count in range, and every line is non-empty."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        lines = target.read_text(encoding="utf-8").splitlines()
        if len(lines) < min_lines:
            return False, f"expected at least {min_lines} lines, got {len(lines)}"
        if max_lines is not None and len(lines) > max_lines:
            return False, f"expected at most {max_lines} lines, got {len(lines)}"
        if any(not line.strip() for line in lines):
            return False, "file contains a blank line"
        return True, f"{len(lines)} non-empty lines"

    return _verify


def file_matches_regex(rel_path: str, pattern: str, min_lines: int = 1) -> Verifier:
    """Every line of the file matches the given regex."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        lines = target.read_text(encoding="utf-8").splitlines()
        if len(lines) < min_lines:
            return False, f"expected at least {min_lines} lines, got {len(lines)}"
        regex = re.compile(pattern)
        bad = [line for line in lines if not regex.search(line)]
        if bad:
            return False, f"{len(bad)} line(s) do not match {pattern!r}: {bad[0][:60]}"
        return True, f"all {len(lines)} lines match"

    return _verify


def file_contains(rel_path: str, needle: str, ignore_case: bool = True) -> Verifier:
    """File exists and contains the given substring."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        text = target.read_text(encoding="utf-8")
        haystack = text.lower() if ignore_case else text
        needle_text = needle.lower() if ignore_case else needle
        if needle_text not in haystack:
            return False, f"{rel_path} does not contain {needle!r}"
        return True, f"contains {needle!r}"

    return _verify


def csv_column_mean(rel_path: str, column: str, expected: float, tol: float = 1e-6) -> Verifier:
    """A text file holds the mean of a CSV column."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        try:
            written = float(target.read_text(encoding="utf-8").strip())
        except ValueError:
            return False, f"{rel_path} does not contain a number"
        if abs(written - expected) > tol:
            return False, f"expected {expected}, got {written}"
        return True, f"mean = {written}"

    return _verify


def file_list_equals(rel_path: str, expected: list[str]) -> Verifier:
    """A file lists exactly the expected entries, one per line."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        got = [line.strip() for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        if sorted(got) != sorted(expected):
            return False, f"expected {sorted(expected)}, got {sorted(got)}"
        return True, f"{len(got)} entries"

    return _verify


def edited_file_contains(rel_path: str, needle: str) -> Verifier:
    """File exists and contains the text produced by an edit."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        if needle not in target.read_text(encoding="utf-8"):
            return False, f"{rel_path} does not contain the edited text"
        return True, "edit applied"

    return _verify


def json_key_equals(rel_path: str, key: str, expected: object) -> Verifier:
    """A JSON file has the expected value at the given key."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_path
        if not target.exists():
            return False, f"{rel_path} was never created"
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return False, f"{rel_path} is not valid JSON: {e}"
        if data.get(key) != expected:
            return False, f"expected {key}={expected!r}, got {data.get(key)!r}"
        return True, f"{key} = {expected!r}"

    return _verify


def directory_contains_file(rel_dir: str, filename: str) -> Verifier:
    """A directory exists and contains the named file."""

    def _verify(workspace: Path) -> tuple[bool, str]:
        target = workspace / rel_dir / filename
        if not target.exists():
            return False, f"{rel_dir}/{filename} does not exist"
        return True, "present"

    return _verify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n"
_SAMPLE_PY = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "def sub(a, b):\n"
    "    return a - b\n"
    "\n"
    "class Calc:\n"
    "    def mul(self, a, b):\n"
    "        return a * b\n"
)


def _seed_csv(workspace: Path) -> None:
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "numbers.csv").write_text(_SAMPLE_CSV, encoding="utf-8")


def _seed_py(workspace: Path) -> None:
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "calc.py").write_text(_SAMPLE_PY, encoding="utf-8")


def _seed_doc(workspace: Path) -> None:
    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / "notes.md").write_text(
        "# Release Notes\n\n"
        "Version 1.2.0\n"
        "- Added the importer\n"
        "- Fixed the exporter\n"
        "- Improved the scheduler\n"
        "- Updated the docs\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def build_tasks() -> list[Task]:
    """Return the full benchmark task list."""
    tasks: list[Task] = []

    # -- write: create a file with N lines -------------------------------
    tasks.append(
        Task(
            id="write_lines_01",
            prompt=(
                "Create a file named out/report.md containing exactly 5 lines. "
                "Every line must start with the # character."
            ),
            kind="file_create",
            verify=file_matches_regex("out/report.md", r"^#", min_lines=5),
            tags=("write", "single_step"),
            expected_tool="write_file",
        )
    )
    tasks.append(
        Task(
            id="write_lines_02",
            prompt=(
                "Create a file named out/notes.txt with exactly 3 non-empty lines. "
                "Each line should start with the word 'item'."
            ),
            kind="file_create",
            verify=file_matches_regex("out/notes.txt", r"^item", min_lines=3),
            tags=("write", "single_step"),
            expected_tool="write_file",
        )
    )

    # -- transform: read then write --------------------------------------
    tasks.append(
        Task(
            id="transform_upper_01",
            prompt=(
                "Read the file data/upper.txt and write its contents uppercased "
                "to out/upper.txt."
            ),
            kind="file_transform",
            verify=file_contains("out/upper.txt", "HELLO"),
            tags=("read", "write", "single_step"),
            setup=lambda ws: (
                (ws / "data").mkdir(parents=True, exist_ok=True),
                (ws / "data" / "upper.txt").write_text("hello world\n", encoding="utf-8"),
            ),
        )
    )
    tasks.append(
        Task(
            id="transform_count_01",
            prompt=(
                "Read the file data/lines.txt and write the number of lines it "
                "contains to out/count.txt."
            ),
            kind="file_transform",
            verify=file_contains("out/count.txt", "3"),
            tags=("read", "write", "single_step"),
            setup=lambda ws: (
                (ws / "data").mkdir(parents=True, exist_ok=True),
                (ws / "data" / "lines.txt").write_text("a\nb\nc\n", encoding="utf-8"),
            ),
        )
    )

    # -- edit: modify an existing file -----------------------------------
    tasks.append(
        Task(
            id="edit_replace_01",
            prompt=(
                "In the file out/config.txt, replace the word 'DRAFT' with the "
                "word 'FINAL'. Leave everything else unchanged."
            ),
            kind="file_edit",
            verify=edited_file_contains("out/config.txt", "FINAL"),
            tags=("edit", "single_step"),
            expected_tool="edit_file",
            setup=lambda ws: (
                (ws / "out").mkdir(parents=True, exist_ok=True),
                (ws / "out" / "config.txt").write_text("status: DRAFT\n", encoding="utf-8"),
            ),
        )
    )

    # -- search: find files and list them --------------------------------
    tasks.append(
        Task(
            id="glob_list_01",
            prompt=(
                "List every .py file under src/ and write the file names, one per "
                "line, to out/pylist.txt."
            ),
            kind="repo_scan",
            verify=file_list_equals("out/pylist.txt", ["calc.py"]),
            tags=("glob", "search", "single_step"),
            setup=_seed_py,
        )
    )
    tasks.append(
        Task(
            id="grep_count_01",
            prompt=(
                "Search the files under src/ for the word 'return' and write the "
                "number of matches to out/matches.txt."
            ),
            kind="repo_scan",
            verify=file_contains("out/matches.txt", "3"),
            tags=("grep", "search", "single_step"),
            setup=_seed_py,
        )
    )

    # -- data: compute an aggregate --------------------------------------
    tasks.append(
        Task(
            id="aggregate_mean_01",
            prompt=(
                "Read the file data/numbers.csv, compute the mean of the column "
                "'x', and write the result to out/mean.txt."
            ),
            kind="data_aggregate",
            verify=csv_column_mean("out/mean.txt", "x", 3.0),
            tags=("data", "read", "single_step"),
            setup=_seed_csv,
        )
    )

    # -- shell ------------------------------------------------------------
    tasks.append(
        Task(
            id="shell_count_01",
            prompt=(
                "Use the shell to count how many lines are in data/numbers.csv "
                "and write that number to out/lines.txt."
            ),
            kind="http_fetch",
            verify=file_contains("out/lines.txt", "6"),
            tags=("shell", "single_step"),
            setup=_seed_csv,
        )
    )
    tasks.append(
        Task(
            id="shell_grep_01",
            prompt=(
                "Use the shell to find which lines in src/calc.py contain the word "
                "'return', and write those line numbers to out/linenos.txt."
            ),
            kind="http_fetch",
            verify=file_contains("out/linenos.txt", "2"),
            tags=("shell", "single_step"),
            setup=_seed_py,
        )
    )

    # -- multi-step --------------------------------------------------------
    tasks.append(
        Task(
            id="multi_step_01",
            prompt=(
                "Create a directory named out/report, then write a file "
                "out/report/summary.md inside it containing exactly 2 lines that "
                "both start with '- '."
            ),
            kind="multi_step",
            verify=directory_contains_file("out/report", "summary.md"),
            tags=("write", "multi_step"),
            expected_tool="write_file",
        )
    )
    tasks.append(
        Task(
            id="multi_step_02",
            prompt=(
                "Read docs/notes.md, then write out/version.txt containing only the "
                "version number that appears in it (for example 1.2.0)."
            ),
            kind="multi_step",
            verify=file_contains("out/version.txt", "1.2.0"),
            tags=("read", "write", "multi_step"),
            setup=_seed_doc,
        )
    )
    tasks.append(
        Task(
            id="multi_step_03",
            prompt=(
                "Create out/data.json containing the JSON object {\"count\": 3, "
                "\"name\": \"report\"}."
            ),
            kind="multi_step",
            verify=json_key_equals("out/data.json", "count", 3),
            tags=("write", "multi_step"),
            expected_tool="write_file",
        )
    )

    # -- tool discovery ----------------------------------------------------
    tasks.append(
        Task(
            id="list_files_01",
            prompt=(
                "List the contents of the directory docs/ and write the names of "
                "the files you find to out/docs.txt."
            ),
            kind="repo_scan",
            verify=file_contains("out/docs.txt", "notes.md"),
            tags=("list", "search", "single_step"),
            setup=_seed_doc,
        )
    )
    tasks.append(
        Task(
            id="read_and_copy_01",
            prompt=(
                "Read src/calc.py and write a copy of it to out/copy.py."
            ),
            kind="file_transform",
            verify=file_contains("out/copy.py", "def add"),
            tags=("read", "write", "single_step"),
            setup=_seed_py,
        )
    )

    return tasks


TASKS: list[Task] = build_tasks()


def task_by_id(task_id: str) -> Task | None:
    for task in TASKS:
        if task.id == task_id:
            return task
    return None
