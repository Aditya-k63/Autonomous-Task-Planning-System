"""Tests for the LangGraph wiring.

A conditional edge that returns a destination missing from its edge map
raises KeyError inside LangGraph, and only when that branch is taken. The
self-healing retry path is exactly such a branch, so it is exercised here.
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


def _ends(graph, node: str) -> set[str]:
    """The declared destinations for a node's conditional edge."""
    branch = graph.branches[node]
    spec = next(iter(branch.values()))
    return set(spec.ends.keys())


def test_reviewer_can_route_back_to_the_executor():
    """The self-healing retry path must be a declared destination."""
    state = _state(phase=AgentPhase.EXECUTING.value, retry_count=1, max_retries=3)
    assert after_reviewer(state) == "executor"


def test_reviewer_edge_map_declares_every_destination():
    graph = build_graph()
    for destination in ("router", "executor", "__end__"):
        assert destination in _ends(graph, "reviewer")


def test_reviewer_routes_to_router_when_not_retrying():
    state = _state(phase=AgentPhase.EXECUTING.value, retry_count=0, max_retries=3)
    assert after_reviewer(state) == "router"


def test_router_can_route_back_to_the_planner():
    state = _state(phase=AgentPhase.PLANNING.value)
    assert after_router(state) == "planner"


def test_router_edge_map_declares_every_destination():
    graph = build_graph()
    for destination in ("executor", "planner", "__end__"):
        assert destination in _ends(graph, "router")


def test_planner_and_executor_maps_cover_their_destinations():
    graph = build_graph()
    for destination in ("router", "__end__"):
        assert destination in _ends(graph, "planner")
    for destination in ("reviewer", "__end__"):
        assert destination in _ends(graph, "executor")


def test_every_route_the_conditions_can_return_is_declared():
    """A destination missing from the map is a KeyError at runtime."""
    graph = build_graph()
    checks = {
        "planner": (after_planner, ("router", "__end__")),
        "router": (after_router, ("executor", "planner", "__end__")),
        "executor": (after_executor, ("reviewer", "__end__")),
        "reviewer": (after_reviewer, ("router", "executor", "__end__")),
    }
    for node, (_, destinations) in checks.items():
        declared = _ends(graph, node)
        for destination in destinations:
            assert destination in declared, f"{node} cannot route to {destination}"


def test_executor_stops_on_clarification():
    state = _state(phase=AgentPhase.CLARIFYING.value)
    assert after_executor(state) == "__end__"
