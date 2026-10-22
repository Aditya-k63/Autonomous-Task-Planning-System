"""
Scripted reference solver.

A deterministic stand-in for the model that solves every benchmark task
in one tool call. It serves two purposes:

1. **CI without credentials** — the harness runs green with no API key.
2. **Comparison oracle** — the reference trajectory shows the minimal
   correct behaviour, so a real run can be compared against it.

Each entry maps a distinctive phrase from a task prompt to the tool call
that solves it, expressed with the same structured arguments the live
model uses.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

# (prompt fragment, tool name, structured arguments)
SOLUTIONS: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "out/report.md",
        "write_file",
        {
            "path": "out/report.md",
            "content": "# Section 1\n# Section 2\n# Section 3\n# Section 4\n# Section 5",
        },
    ),
    (
        "out/notes.txt",
        "write_file",
        {"path": "out/notes.txt", "content": "item one\nitem two\nitem three"},
    ),
    (
        "out/upper.txt",
        "write_file",
        {"path": "out/upper.txt", "content": "HELLO WORLD"},
    ),
    (
        "out/count.txt",
        "write_file",
        {"path": "out/count.txt", "content": "3"},
    ),
    (
        "out/config.txt",
        "edit_file",
        {
            "path": "out/config.txt",
            "old_string": "DRAFT",
            "new_string": "FINAL",
        },
    ),
    (
        "out/pylist.txt",
        "write_file",
        {"path": "out/pylist.txt", "content": "calc.py"},
    ),
    (
        "out/matches.txt",
        "write_file",
        {"path": "out/matches.txt", "content": "3"},
    ),
    (
        "out/mean.txt",
        "write_file",
        {"path": "out/mean.txt", "content": "3.0"},
    ),
    (
        "out/lines.txt",
        "write_file",
        {"path": "out/lines.txt", "content": "6"},
    ),
    (
        "out/linenos.txt",
        "write_file",
        {"path": "out/linenos.txt", "content": "2\n5\n9"},
    ),
    (
        "out/report/summary.md",
        "write_file",
        {"path": "out/report/summary.md", "content": "- first\n- second"},
    ),
    (
        "out/version.txt",
        "write_file",
        {"path": "out/version.txt", "content": "1.2.0"},
    ),
    (
        "out/data.json",
        "write_file",
        {"path": "out/data.json", "content": '{"count": 3, "name": "report"}'},
    ),
    (
        "out/docs.txt",
        "write_file",
        {"path": "out/docs.txt", "content": "notes.md"},
    ),
    (
        "out/copy.py",
        "write_file",
        {"path": "out/copy.py", "content": "def add(a, b):\n    return a + b"},
    ),
)


def solution_for(prompt: str) -> tuple[str, dict[str, str]] | None:
    """Return ``(tool, args)`` for a prompt, or None when unrecognised."""
    for fragment, tool, args in SOLUTIONS:
        if fragment in prompt:
            return tool, dict(args)
    return None


def _legacy_input(tool_name: str, args: dict[str, str]) -> str:
    """Render structured arguments the way the text-only protocol expects."""
    from app.tools.registry import registry

    tool = registry.get(tool_name)
    if tool is None:
        return str(args)
    return tool.to_call_string(args)


class OracleResponse:
    """Minimal stand-in for a model response."""

    def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class OracleLLM:
    """A scripted model that answers the three node prompts deterministically."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages: Any) -> OracleResponse:
        system = ""
        system_content = ""
        user = ""
        saw_tool_result = False
        for message in messages:
            content = getattr(message, "content", "")
            kind = type(message).__name__
            if kind == "ToolMessage":
                saw_tool_result = True
            if "You are a task planner" in content:
                system = "planner"
            elif "You are a task executor" in content:
                system = "executor"
            elif "You are a task reviewer" in content:
                system = "reviewer"
            if "Current task:" in content:
                system_content = content
            user = content

        self.calls.append(system)
        if system == "planner":
            return OracleResponse(_plan_json())
        if system == "reviewer":
            return OracleResponse(
                '{"pass": true, "score": 1.0, "reason": "task completed", '
                '"suggestion": "", "retry": false}'
            )
        # The task description lives in the system prompt; the trailing
        # human turn is only "Execute the task".
        if saw_tool_result:
            return OracleResponse("DONE: completed the task")
        solution = solution_for(system_content or user)
        if solution is None:
            return OracleResponse("FAILED: no scripted solution for this prompt")
        tool, args = solution
        return OracleResponse(
            "",
            tool_calls=[{"name": tool, "args": args, "id": "call_oracle_1"}],
        )

    def bind_tools(self, *_args, **_kwargs):
        return self


def _plan_json() -> str:
    return (
        '{"tasks": [{"description": "Complete the requested task using the '
        'available tools.", "priority": "high", "dependencies": [], '
        '"substeps": ["inspect the request", "use the tools", "verify the '
        'result"]}]}'
    )


def _executor_response(user_text: str) -> str:
    """Return the tool call in the legacy text protocol."""
    solution = solution_for(user_text)
    if solution is None:
        return "FAILED: no scripted solution for this prompt"
    tool, args = solution
    escaped = _legacy_input(tool, args).replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return f'TOOL_CALL: {{"tool": "{tool}", "input": "{escaped}"}}'


@contextmanager
def use_oracle() -> Iterator[OracleLLM]:
    """Patch every node module to use the scripted oracle instead of a model."""
    oracle = OracleLLM()
    patched = []
    for module_name in ("executor", "planner", "reviewer"):
        module = __import__(f"app.nodes.{module_name}", fromlist=["*"])
        if not hasattr(module, "invoke_with_fallback"):
            continue
        original = module.invoke_with_fallback

        def make_wrapper(bound_oracle, _original=original):
            def wrapper(messages, settings=None, max_retries=3, tools=None):
                return bound_oracle.invoke(messages)

            return wrapper

        module.invoke_with_fallback = make_wrapper(oracle)
        patched.append((module, original))
    try:
        yield oracle
    finally:
        for module, original in patched:
            module.invoke_with_fallback = original
