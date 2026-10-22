"""
Evaluation runner.

Runs a task against the agent, optionally with a fault injected, and
records what happened: success, steps, tool calls, retries, permissions
requested, latency, and token usage.

Faults are installed by patching the tool dispatch and the LLM call for
the duration of a single run, then restored.
"""

from __future__ import annotations

import contextvars
import json
import random
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.state import AgentPhase
from app.workspace import reset_workspace, set_workspace

from eval.faults import ALL_FAULTS, Fault
from eval.tasks import Task

# A single task may not exceed this many seconds. The limit is generous
# because a free-tier provider can rate-limit for minutes at a time.
TASK_TIMEOUT_SECONDS = 180.0

# Stop the suite once this many input tokens have been spent. A full pass
# costs roughly 115,000 input tokens against a free-tier daily
# allowance, so the budget stops a run from burning the quota and
# leaving the rest of the suite unattempted.
DEFAULT_TOKEN_BUDGET = 160_000


@dataclass
class TaskResult:
    """Outcome of one task run."""

    task_id: str
    kind: str
    tags: list[str]
    success: bool
    detail: str
    fault: str | None = None
    steps: int = 0
    tool_calls: int = 0
    permission_asks: int = 0
    permission_denials: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: int = 0
    error: str | None = None
    tool_trace: list[str] = field(default_factory=list)
    infra_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fault installation
# ---------------------------------------------------------------------------


@contextmanager
def apply_fault(fault: Fault | None, rng: random.Random) -> Iterator[None]:
    """
    Install a fault for the duration of the block, then restore it.

    Tool faults wrap the tool's own function, so an injected exception is
    caught by ``Tool.execute`` and surfaces to the agent as an ``ERROR:``
    result — the same thing a real tool failure looks like.
    """
    if fault is None:
        yield
        return

    from app.tools import registry as registry_module

    if fault.scope == "tool":
        original_execute = registry_module.Tool.execute

        def patched_execute(self, input_text):
            """Inject the fault inside the tool's own error handling."""
            if fault.should_fire(rng) and (
                not fault.target or fault.target == self.name
            ):
                try:
                    return fault.apply(self.func, input_text)
                except Exception as e:
                    return f"ERROR: {e}"
            return original_execute(self, input_text)

        registry_module.Tool.execute = patched_execute
        try:
            yield
        finally:
            registry_module.Tool.execute = original_execute
        return

    # LLM scope — patch the executor only. The planner must keep working,
    # otherwise every task dies at step 0 and the fault measures nothing
    # about execution robustness.
    patched_modules = []
    module = __import__("app.nodes.executor", fromlist=["*"])
    original = getattr(module, "invoke_with_fallback", None)
    if original is not None:

        def make_llm_wrapper(bound_original, _fault=fault):
            def wrapper(messages, settings=None, max_retries=3):
                if _fault.should_fire(rng):
                    content = _fault.apply(lambda *a, **k: "")
                    return type("FaultedResponse", (), {"content": content})()
                return bound_original(messages, settings, max_retries)

            return wrapper

        module.invoke_with_fallback = make_llm_wrapper(original)
        patched_modules.append((module, original))

    try:
        yield
    finally:
        for module, original in patched_modules:
            module.invoke_with_fallback = original


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_graph(goal: str, max_steps: int, timeout_s: float | None = None) -> tuple[dict, list[str]]:
    """
    Drive the agent graph to completion and return the final state.

    The graph runs on a worker thread so a task that hangs is recorded as
    a timeout instead of stalling the whole suite.
    """
    from app.graph import get_graph
    from app.state import AgentState

    def _work() -> tuple[dict, list[str]]:
        graph = get_graph()
        state = AgentState(goal=goal, max_retries=3)
        final = graph.invoke(dict(state), {"recursion_limit": 60})
        trace = [
            f"{s.get('action')}:{s.get('tool', '-')}"
            for s in final.get("execution_history", [])
        ]
        return final, trace

    if timeout_s is None:
        return _work()

    outcome: dict = {}

    def _target() -> None:
        try:
            outcome["value"] = _work()
        except Exception as exc:  # pragma: no cover - defensive
            outcome["error"] = exc

    # A thread starts with a fresh context, so the workspace override
    # (a ContextVar) would be lost. Copy the current context across.
    ctx = contextvars.copy_context()
    worker = threading.Thread(target=lambda: ctx.run(_target), daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"task exceeded {timeout_s:.0f}s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def run_task(
    task: Task,
    *,
    fault: Fault | None = None,
    seed: int = 0,
    workspace: Path | None = None,
) -> TaskResult:
    """Execute one task, optionally with a fault, and verify the outcome."""
    rng = random.Random(seed)
    root = workspace or (Path.cwd() / ".eval_workspace" / task.id)
    root.mkdir(parents=True, exist_ok=True)

    set_workspace(root)
    try:
        reset_workspace()
        task.prepare(root)
        started = time.perf_counter()
        with apply_fault(fault, rng):
            final, trace = _run_graph(task.prompt, task.max_steps, TASK_TIMEOUT_SECONDS)
        wall_ms = int((time.perf_counter() - started) * 1000)

        passed, detail = task.verify(root)
        events = final.get("permission_events", [])
        usage = final.get("token_usage", {}) or {}
        return TaskResult(
            task_id=task.id,
            kind=task.kind,
            tags=list(task.tags),
            success=passed,
            detail=detail,
            fault=fault.name if fault else None,
            steps=len(final.get("execution_history", [])),
            tool_calls=sum(1 for s in final.get("execution_history", []) if s.get("action") == "tool_call"),
            permission_asks=sum(1 for e in events if e.get("decision") == "ask"),
            permission_denials=sum(1 for e in events if e.get("decision") == "deny"),
            tokens_in=int(usage.get("input", 0) or 0),
            tokens_out=int(usage.get("output", 0) or 0),
            wall_ms=wall_ms,
            error=final.get("error"),
            tool_trace=trace,
        )
    except Exception as exc:  # a crash is a failure, not an abort
        from app.llm import AllEndpointsUnavailable

        return TaskResult(
            task_id=task.id,
            kind=task.kind,
            tags=list(task.tags),
            success=False,
            detail=f"harness error: {exc}",
            fault=fault.name if fault else None,
            error=traceback.format_exc(limit=20),
            infra_error=isinstance(exc, AllEndpointsUnavailable),
        )
    finally:
        set_workspace(None)


def run_suite(
    tasks: list[Task],
    *,
    fault: Fault | None = None,
    seed: int = 0,
    limit: int | None = None,
    token_budget: int | None = DEFAULT_TOKEN_BUDGET,
) -> list[TaskResult]:
    """
    Run a list of tasks and return one result per task.

    Stops early when the token budget is spent. Tasks that were never
    attempted are returned as infrastructure errors so the summary can
    distinguish "not run" from "run and failed".
    """
    selected = tasks[:limit] if limit else tasks
    results: list[TaskResult] = []
    spent = 0
    for index, task in enumerate(selected):
        if token_budget is not None and spent >= token_budget:
            results.append(
                TaskResult(
                    task_id=task.id,
                    kind=task.kind,
                    tags=list(task.tags),
                    success=False,
                    detail="not attempted: token budget exhausted",
                    fault=fault.name if fault else None,
                    infra_error=True,
                )
            )
            continue
        result = run_task(task, fault=fault, seed=seed + index)
        spent += result.tokens_in + result.tokens_out
        results.append(result)
        if result.infra_error:
            # The provider is gone. Stop rather than burning the quota on
            # tasks that cannot run.
            for remaining in selected[index + 1 :]:
                results.append(
                    TaskResult(
                        task_id=remaining.id,
                        kind=remaining.kind,
                        tags=list(remaining.tags),
                        success=False,
                        detail="not attempted: model provider unavailable",
                        fault=fault.name if fault else None,
                        infra_error=True,
                    )
                )
            break
    return results


def save_results(results: list[TaskResult], path: Path | str) -> None:
    """Write results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(results),
        "success": sum(1 for r in results if r.success),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
