"""
LangGraph StateGraph — wires all nodes into an agent loop.

Flow:
  START → planner → router → (executor → reviewer → router)* → done
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from app.nodes.planner import planner_node
from app.nodes.executor import executor_node
from app.nodes.reviewer import reviewer_node
from app.nodes.router import router_node

logger = logging.getLogger(__name__)

# Use plain dict as state — LangGraph 1.x handles dicts better than subclasses
StateType = dict[str, Any]


# --- Edge conditions ---

def after_planner(state: StateType) -> str:
    phase = state.get("phase", "")
    logger.info(f"after_planner: phase={phase}")
    if phase == "error":
        return "__end__"
    return "router"


def after_router(state: StateType) -> str:
    phase = state.get("phase", "")
    logger.info(f"after_router: phase={phase}")
    if phase == "done":
        return "__end__"
    if phase == "error":
        return "__end__"
    if phase == "planning":
        return "planner"
    return "executor"


def after_executor(state: StateType) -> str:
    phase = state.get("phase", "")
    logger.info(f"after_executor: phase={phase}")
    if phase == "clarifying":
        return "__end__"
    if phase == "error":
        return "__end__"
    return "reviewer"


def after_reviewer(state: StateType) -> str:
    phase = state.get("phase", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    logger.info(f"after_reviewer: phase={phase}, retry_count={retry_count}/{max_retries}")
    if phase == "error":
        return "__end__"
    # Self-healing: route back to executor for retry instead of router
    if phase == "executing" and retry_count > 0 and retry_count <= max_retries:
        logger.info(f"after_reviewer: self-healing retry to executor")
        return "executor"
    return "router"


# --- Build graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("router", router_node)

    # Edges
    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", after_planner, {"router": "router", "__end__": END})
    graph.add_conditional_edges("router", after_router, {"executor": "executor", "planner": "planner", "__end__": END})
    graph.add_conditional_edges("executor", after_executor, {"reviewer": "reviewer", "__end__": END})
    # "executor" must be listed here: after_reviewer returns it on the
    # self-healing retry path, and a destination missing from this map
    # raises KeyError inside LangGraph the moment a task fails and is
    # retried.
    graph.add_conditional_edges(
        "reviewer",
        after_reviewer,
        {"router": "router", "executor": "executor", "__end__": END},
    )

    return graph


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph
