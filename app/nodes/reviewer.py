"""
Reviewer node — verifies task completion.

Checks if the task's output satisfies the goal and marks it completed or failed.
"""

from __future__ import annotations

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import AgentState, Task, TaskStatus, AgentPhase, Step
from app.llm import get_llm, invoke_with_fallback
from app.json_utils import extract_json_object

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM = """You are a task reviewer. Evaluate whether a task was completed successfully.

You will receive:
- The original goal
- The task description
- The execution history (tool calls and outputs)

Evaluate: Did the task accomplish what it was supposed to?

RESPOND WITH ONLY valid JSON:
{{
  "pass": true/false,
  "score": 0.0-1.0,
  "reason": "brief explanation",
  "suggestion": "if failed, what should be tried differently",
  "retry": true/false
}}

Be lenient — if the task is substantially complete, mark it as passing.
Only fail if the task clearly did not achieve its purpose.
If a task failed but can be retried with a different approach, set "retry": true."""


def reviewer_node(state: AgentState) -> dict:
    """Review the current task's completion. Returns partial state update."""
    logger.info("REVIEWER: Reviewing task completion")

    # Get current task safely
    tasks_raw = state.get("tasks", [])
    current_task_id = state.get("current_task_id")
    current_task = None
    for t in tasks_raw:
        task = Task(**t) if isinstance(t, dict) else t
        if task.id == current_task_id:
            current_task = task
            break

    if current_task is None:
        return {"phase": AgentPhase.ERROR.value, "error": "No current task to review"}

    # Gather execution history for this task
    task_steps = [s for s in state["execution_history"] if s.get("task_id") == current_task.id]
    steps_text = "\n".join(
        f"  [{s['action']}] tool={s.get('tool', '-')} input={s.get('input', '-')} output={s.get('output', '-')[:200]}"
        for s in task_steps
    )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"""Goal: {state['goal']}

Task: {current_task.description}

Execution history:
{steps_text if steps_text else '(no tool calls — task may not have been executed)'}

Did this task succeed? Respond with JSON only."""),
    ]

    response = invoke_with_fallback(messages)
    from app.nodes.executor import _record_usage

    _record_usage(state, response)
    raw = str(response.content).strip()

    data = extract_json_object(raw)
    if data is None:
        logger.error(f"Reviewer returned invalid JSON: {raw[:300]}")
        passed = False
        score = 0.0
        reason = f"Review parse error: {raw[:300]}"
    else:
        passed = data.get("pass", False)
        score = data.get("score", 0.0)
        reason = data.get("reason", "")

    logger.info(f"REVIEWER: pass={passed}, score={score:.2f}, reason={reason[:100]}")

    # Determine if self-healing retry is needed
    retry_needed = False
    suggestion = ""
    if not passed:
        try:
            retry_needed = data.get("retry", False)
            suggestion = data.get("suggestion", "")
        except Exception:
            pass

    # Update task status
    tasks = state.get("tasks", [])
    for i, t in enumerate(tasks):
        tid = t.get("id") if isinstance(t, dict) else t.id
        if tid == current_task.id:
            if passed:
                tasks[i]["status"] = TaskStatus.COMPLETED.value
                tasks[i]["result"] = reason
                tasks[i]["error"] = None
            else:
                if retry_needed and state["retry_count"] < state["max_retries"]:
                    tasks[i]["status"] = TaskStatus.PENDING.value
                    tasks[i]["error"] = f"{reason} | Retrying (attempt {state['retry_count'] + 1}/{state['max_retries']})"
                    state["retry_count"] = state["retry_count"] + 1
                    logger.info(f"REVIEWER: Self-healing retry for task {current_task.id}")
                else:
                    tasks[i]["status"] = TaskStatus.FAILED.value
                    tasks[i]["error"] = reason
            break

    history = state.get("execution_history", [])
    history.append(Step(
        task_id=current_task.id,
        action="review",
        output=f"pass={passed} score={score:.2f} reason={reason}",
        success=passed,
    ).model_dump())

    if retry_needed and not passed and state["retry_count"] <= state["max_retries"]:
        state["phase"] = AgentPhase.EXECUTING.value
        state["retry_count"] = state["retry_count"] + 1
    else:
        state["tasks"] = tasks

    state["execution_history"] = history
    return state
