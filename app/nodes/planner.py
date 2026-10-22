"""
Planner node — breaks a high-level goal into a dependency DAG of tasks.

LLM receives the goal + available tools, returns a structured task plan.
"""

from __future__ import annotations

import json
import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage

from app.state import AgentState, Task, TaskStatus, TaskPriority, AgentPhase
from app.llm import get_llm, invoke_with_fallback
from app.json_utils import extract_json_object

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a task planner. Given a high-level goal, break it into concrete,
ordered subtasks that an AI agent can execute using available tools.

RULES:
1. Each task must be a single, concrete, actionable step (not vague).
2. Define dependencies explicitly — which tasks must complete before others.
3. Prioritize tasks (low/medium/high/critical).
4. If the goal requires information you don't have, add a task that uses the web_search tool.
5. Keep the plan as SHORT as possible. Most goals need 1-3 tasks. Only use more
   when the goal genuinely has that many independent parts. Never exceed 12 tasks.
6. Do NOT ask clarifying questions — just make reasonable assumptions.

RESPOND WITH ONLY valid JSON (no markdown, no explanation):
{{
  "tasks": [
    {{
      "description": "string — concrete action",
      "priority": "low|medium|high|critical",
      "dependencies": ["task_id_1"],
      "substeps": ["step1", "step2"]
    }}
  ]
}}

Available tools: {tools}
MAX_TASKS: 12. If the LLM returns more than 12 tasks, truncate to the first 12.
"""

PLANNER_RESPONSE_SCHEMA = """{{
  "tasks": [
    {{
      "description": "string",
      "priority": "low|medium|high|critical",
      "dependencies": [],
      "substeps": []
    }}
  ]
}}"""


def planner_node(state: AgentState) -> dict:
    """Break goal into tasks. Returns partial state update."""
    logger.info("PLANNER: Breaking goal into tasks")
    try:
        return _planner_node_inner(state)
    except Exception as e:
        logger.error(f"PLANNER: Unhandled exception: {type(e).__name__}: {e}")
        return {
            "phase": AgentPhase.ERROR.value,
            "error": f"Planner crashed: {type(e).__name__}: {e}",
        }


def _planner_node_inner(state: AgentState) -> dict:
    logger.info("PLANNER: step 1 - resolve model chain")
    logger.info("PLANNER: step 2 - get tools")
    from app.tools.registry import registry
    tools = ", ".join(t.name for t in registry.list_tools())
    logger.info("PLANNER: step 3 - build messages")
    system_msg = PLANNER_SYSTEM.format(tools=tools)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Goal: {state.get('goal', '')}\n\nCreate a task plan. Respond with JSON only."),
    ]

    logger.info("PLANNER: step 4 - invoke llm")
    response = invoke_with_fallback(messages)
    from app.nodes.executor import _record_usage

    _record_usage(state, response)

    logger.info("PLANNER: step 5 - extract content")
    content = response.content
    if isinstance(content, list):
        raw = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    else:
        raw = str(content).strip()
    logger.info(f"PLANNER raw response: {repr(raw[:500])}")

    logger.info("PLANNER: step 6 - extract JSON")
    data = extract_json_object(raw)
    if data is None:
        logger.error("PLANNER: JSON parse failed")
        logger.error("PLANNER: raw content: %r", raw[:500])
        return {
            "phase": AgentPhase.ERROR.value,
            "error": f"Planner returned invalid JSON: {raw[:300]}",
        }

    tasks_raw = data.get("tasks", [])
    logger.info("PLANNER: step 9 - got %s raw tasks", len(tasks_raw))

    logger.info("PLANNER: step 10 - parse tasks")
    tasks: list[Task] = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            logger.warning(f"PLANNER: Skipping non-dict task item: {type(t)}: {str(t)[:100]}")
            continue
        desc = t.get("description", "")
        if not desc or not isinstance(desc, str):
            logger.warning(f"PLANNER: Skipping task with no valid description: {t}")
            continue
        priority_str = t.get("priority", "medium")
        try:
            priority = TaskPriority(priority_str)
        except ValueError:
            priority = TaskPriority.MEDIUM

        task = Task(
            description=desc,
            priority=priority,
            dependencies=[d for d in t.get("dependencies", []) if isinstance(d, str)],
            substeps=[s for s in t.get("substeps", []) if isinstance(s, str)],
            status=TaskStatus.PENDING,
        )
        tasks.append(task)

    # Enforce hard max of 12 tasks
    if len(tasks) > 12:
        logger.warning(f"PLANNER: Truncated {len(tasks)} tasks to 12 max")
        tasks = tasks[:12]

    logger.info(f"PLANNER: step 11 - created {len(tasks)} tasks")

    if not tasks:
        return {
            "phase": AgentPhase.DONE.value,
            "error": "Planner produced no tasks",
        }

    first_task = tasks[0]
    logger.info(f"PLANNER: Created {len(tasks)} tasks, starting with '{first_task.description[:50]}'")

    # Mutate state directly — LangGraph passes state by reference
    state["tasks"] = [t.model_dump() for t in tasks]
    state["current_task_id"] = first_task.id
    state["phase"] = AgentPhase.EXECUTING.value
    state["retry_count"] = 0
    return state
