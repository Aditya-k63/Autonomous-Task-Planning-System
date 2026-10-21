"""
Scripted reference solver.

A deterministic stand-in for the model that solves every benchmark task
in one tool call. It serves two purposes:

1. **CI without credentials** — the harness runs green with no API key.
2. **Comparison oracle** — the reference trajectory shows the minimal
   correct behaviour, so a real run can be compared against it.

Each entry maps a distinctive phrase from a task prompt to the tool call
that solves it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

# (prompt fragment, tool name, tool input)
SOLUTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "out/report.md",
        "write_file",
        "out/report.md\n# Section 1\n# Section 2\n# Section 3\n# Section 4\n# Section 5",
    ),
    (
        "out/notes.txt",
        "write_file",
        "out/notes.txt\nitem one\nitem two\nitem three",
    ),
    (
        "out/upper.txt",
        "write_file",
        "out/upper.txt\nHELLO WORLD",
    ),
    (
        "out/count.txt",
        "write_file",
        "out/count.txt\n3",
    ),
    (
        "out/config.txt",
        "edit_file",
        "out/config.txt\nDRAFT\nFINAL",
    ),
    (
        "out/pylist.txt",
        "write_file",
        "out/pylist.txt\ncalc.py",
    ),
    (
        "out/matches.txt",
        "write_file",
        "out/matches.txt\n3",
    ),
    (
        "out/mean.txt",
        "write_file",
        "out/mean.txt\n3.0",
    ),
    (
        "out/lines.txt",
        "write_file",
        "out/lines.txt\n6",
    ),
    (
        "out/linenos.txt",
        "write_file",
        "out/linenos.txt\n2\n5\n9",
    ),
    (
        "out/report/summary.md",
        "write_file",
        "out/report/summary.md\n- first\n- second",
    ),
    (
        "out/version.txt",
        "write_file",
        "out/version.txt\n1.2.0",
    ),
    (
        "out/data.json",
        "write_file",
        'out/data.json\n{"count": 3, "name": "report"}',
    ),
    (
        "out/docs.txt",
        "write_file",
        "out/docs.txt\nnotes.md",
    ),
    (
        "out/copy.py",
        "write_file",
        "out/copy.py\ndef add(a, b):\n    return a + b",
    ),
)


def solution_for(prompt: str) -> tuple[str, str] | None:
    """Return ``(tool, input)`` for a prompt, or None when unrecognised."""
    for fragment, tool, tool_input in SOLUTIONS:
        if fragment in prompt:
            return tool, tool_input
    return None


class OracleResponse:
    """Minimal stand-in for a model response."""

    def __init__(self, content: str) -> None:
        self.content = content


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
        return OracleResponse(_executor_response(system_content or user))

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
    """Return the tool call that solves the task described in the prompt."""
    solution = solution_for(user_text)
    if solution is None:
        return "FAILED: no scripted solution for this prompt"
    tool, tool_input = solution
    escaped = tool_input.replace("\\", "\\\\").replace('"', '\\"')
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
            def wrapper(messages, settings=None, max_retries=3):
                return bound_oracle.invoke(messages)

            return wrapper

        module.invoke_with_fallback = make_wrapper(oracle)
        patched.append((module, original))
    try:
        yield oracle
    finally:
        for module, original in patched:
            module.invoke_with_fallback = original
