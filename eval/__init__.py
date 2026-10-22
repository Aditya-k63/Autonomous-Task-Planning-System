"""
Evaluation harness for the task-planning agent.

Runs a benchmark of programmatically verified tasks against the agent,
optionally with fault injection, and reports success rate, recovery
behaviour, and safety counters.
"""

from __future__ import annotations

from eval.faults import ALL_FAULTS, Fault, fault_by_name, fault_names
from eval.metrics import RunSummary, compare, load_summary, save_summary, summarize
from eval.report import render_report, write_report
from eval.runner import TaskResult, run_suite, run_task, save_results
from eval.tasks import TASKS, Task, task_by_id

__all__ = [
    "ALL_FAULTS",
    "Fault",
    "RunSummary",
    "TASKS",
    "Task",
    "TaskResult",
    "compare",
    "fault_by_name",
    "fault_names",
    "load_summary",
    "render_report",
    "run_suite",
    "run_task",
    "save_results",
    "save_summary",
    "summarize",
    "task_by_id",
    "write_report",
]
