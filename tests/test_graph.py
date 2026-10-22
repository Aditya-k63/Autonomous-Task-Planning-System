"""
Tests for the LangGraph wiring.

A conditional edge that returns a destination missing from its edge map
raises KeyError inside LangGraph, and only when that branch is taken. The
self-healing retry path is exactly such a branch, so it is exercised here
end to end rather than by inspecting LangGraph internals.
"""

from app.graph import after_executor, after_planner, after_reviewer, after_router, build_graph
from app.state import AgentPhase, TaskStatus


def _state(**kwargs):
    base = {
        "phase": AgentPhase.EXECUTING.value,
        "tasks": [],
        "retry_count": 0,
        "max_retries": 3,
    }
    base.update(kwargs)
    return base


def test_reviewer_can_route_back_to_the_executor():
    """The self-healing retry path must be a declared destination."""
    state = _state(phase=AgentPhase.EXECUTING.value, retry_count=1, max_retries=3)
    assert after_reviewer(state) == "executor"


def test_reviewer_routes_to_router_when_not_retrying():
    state = _state(phase=AgentPhase.EXECUTING.value, retry_count=0, max_retries=3)
    assert after_reviewer(state) == "router"


def test_router_can_route_back_to_the_planner():
    state = _state(phase=AgentPhase.PLANNING.value)
    assert after_router(state) == "planner"


def test_executor_stops_on_clarification():
    state = _state(phase=AgentPhase.CLARIFYING.value)
    assert after_executor(state) == "__end__"


def test_self_healing_retry_reaches_the_executor(monkeypatch):
    """
    A failed task that is about to be retried must not crash the graph.

    The retry destination was missing from the reviewer's edge map, so
    LangGraph raised KeyError on exactly this path.
    """
    from app import graph as graph_module
    from app.state import AgentState

    visited: list[str] = []

    def fake_planner(state):
        state["tasks"] = [
            {"id": "t1", "description": "do the thing", "status": TaskStatus.PENDING.value}
        ]
        state["current_task_id"] = "t1"
        state["phase"] = AgentPhase.EXECUTING.value
        return state

    def fake_executor(state):
        visited.append("executor")
        state["phase"] = AgentPhase.REVIEWING.value
        return state

    def fake_reviewer(state):
        visited.append("reviewer")
        if not state.get("retry_count"):
            # Ask for a self-healing retry, which routes back to the
            # executor. This is the path that used to raise KeyError.
            state["retry_count"] = 1
            state["phase"] = AgentPhase.EXECUTING.value
        else:
            state["phase"] = AgentPhase.REVIEWING.value
        return state

    def fake_router(state):
        visited.append("router")
        # The router picks the next task and sends the graph to the
        # executor. Only stop once the task has been retried.
        if state.get("retry_count"):
            state["phase"] = AgentPhase.DONE.value
        else:
            state["phase"] = AgentPhase.EXECUTING.value
        return state

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "executor_node", fake_executor)
    monkeypatch.setattr(graph_module, "reviewer_node", fake_reviewer)
    monkeypatch.setattr(graph_module, "router_node", fake_router)

    graph = build_graph().compile()
    result = graph.invoke(dict(AgentState(goal="do the thing")), {"recursion_limit": 25})

    assert "executor" in visited, "the retry must reach the executor"
    assert visited.count("executor") >= 2, "the executor must run again on retry"
    assert result.get("phase") == AgentPhase.DONE.value
