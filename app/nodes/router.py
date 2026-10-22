"""
Router node — decides what happens next based on current state.

Returns one of: execute, replan, done
"""

from __future__ import annotations

import logging

from app.state import AgentState, TaskStatus, AgentPhase

logger = logging.getLogger(__name__)


def router_node(state: dict) -> dict:
    """Route to next action. Returns state update."""
    tasks_raw = state.get("tasks", [])
    from app.state import Task
    tasks = [Task(**t) if isinstance(t, dict) else t for t in tasks_raw]

    if not tasks:
        logger.info("ROUTER: No tasks — done")
        state["phase"] = "done"
        return state

    # The reviewer reported the whole goal is already satisfied. Stop
    # rather than working through the rest of an over-long plan.
    if state.get("goal_satisfied"):
        logger.info("ROUTER: Goal already satisfied — stopping early")
        for task in tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                task.status = TaskStatus.COMPLETED
                task.result = task.result or "Covered by an earlier task"
        state["tasks"] = [t.model_dump() for t in tasks]
        state["phase"] = "done"
        return state

    # Find next pending task whose dependencies are all completed
    pending = [t for t in tasks if t.status == TaskStatus.PENDING]
    completed_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}

    for task in pending:
        deps_met = all(dep in completed_ids for dep in task.dependencies)
        if deps_met:
            logger.info(f"ROUTER: Next task — {task.description[:50]}")
            state["current_task_id"] = task.id
            state["phase"] = "executing"
            state["retry_count"] = 0
            return state

    # Check if any tasks are in progress
    in_progress = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
    if in_progress:
        logger.info("ROUTER: Tasks still in progress, waiting")
        state["phase"] = "executing"
        return state

    # All tasks resolved
    all_done = all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in tasks)
    if all_done:
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        logger.info(f"ROUTER: All tasks done — {completed} completed, {failed} failed")
        state["phase"] = "done"
        return state

    # Some tasks blocked — try replanning
    logger.info("ROUTER: Some tasks blocked, triggering replan")
    state["phase"] = "planning"
    return state
