"""
Tests for the evaluation task suite.

Two properties matter for a benchmark:

1. Every task **fails** on an empty workspace — otherwise it is free.
2. Every task **passes** when the expected work is done — otherwise the
   verifier is wrong and the benchmark measures nothing.

The second property is checked here by completing each task directly,
without the agent.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from eval.tasks import TASKS


@pytest.fixture
def workspace():
    path = Path(tempfile.mkdtemp(prefix="eval_task_test_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_suite_is_not_empty():
    assert len(TASKS) >= 10


def test_task_ids_are_unique():
    ids = [t.id for t in TASKS]
    assert len(ids) == len(set(ids))


def test_every_task_fails_on_empty_workspace(workspace):
    """A task must not be satisfiable without doing any work."""
    for task in TASKS:
        task.prepare(workspace)
        passed, detail = task.verify(workspace)
        assert not passed, f"{task.id} was already satisfied: {detail}"


# ---------------------------------------------------------------------------
# Solutions — one per task, applied directly to the workspace.
# ---------------------------------------------------------------------------


def _solve(task_id: str, ws: Path) -> None:
    """Perform the expected work for a task without using the agent."""
    if task_id == "write_lines_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "report.md").write_text(
            "\n".join(f"# Section {i}" for i in range(1, 6)), encoding="utf-8"
        )
    elif task_id == "write_lines_02":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "notes.txt").write_text(
            "item one\nitem two\nitem three\n", encoding="utf-8"
        )
    elif task_id == "transform_upper_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "upper.txt").write_text("HELLO WORLD\n", encoding="utf-8")
    elif task_id == "transform_count_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "count.txt").write_text("3\n", encoding="utf-8")
    elif task_id == "edit_replace_01":
        (ws / "out" / "config.txt").write_text("status: FINAL\n", encoding="utf-8")
    elif task_id == "glob_list_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "pylist.txt").write_text("calc.py\n", encoding="utf-8")
    elif task_id == "grep_count_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "matches.txt").write_text("3\n", encoding="utf-8")
    elif task_id == "aggregate_mean_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "mean.txt").write_text("3.0\n", encoding="utf-8")
    elif task_id == "shell_count_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "lines.txt").write_text("6\n", encoding="utf-8")
    elif task_id == "shell_grep_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "linenos.txt").write_text("2\n5\n9\n", encoding="utf-8")
    elif task_id == "multi_step_01":
        (ws / "out" / "report").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "report" / "summary.md").write_text("- first\n- second\n", encoding="utf-8")
    elif task_id == "multi_step_02":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "version.txt").write_text("1.2.0\n", encoding="utf-8")
    elif task_id == "multi_step_03":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "data.json").write_text(
            '{"count": 3, "name": "report"}', encoding="utf-8"
        )
    elif task_id == "list_files_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "docs.txt").write_text("notes.md\n", encoding="utf-8")
    elif task_id == "read_and_copy_01":
        (ws / "out").mkdir(parents=True, exist_ok=True)
        (ws / "out" / "copy.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    else:  # pragma: no cover - guards against a task without a solution
        raise AssertionError(f"no solution defined for {task_id}")


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_passes_when_work_is_done(task, workspace):
    """The verifier must accept a correctly completed task."""
    task.prepare(workspace)
    _solve(task.id, workspace)
    passed, detail = task.verify(workspace)
    assert passed, f"{task.id} verifier rejected a correct solution: {detail}"
