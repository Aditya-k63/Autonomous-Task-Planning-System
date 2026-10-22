"""
Tool registry — lazy-loaded, schema-validated, fault-tolerant dispatch.

Responsibilities added over the original registry:

1. **Schema validation** — every tool declares a JSON Schema for its input
   and invalid arguments are rejected before the tool runs.
2. **Retry budget** — transient tool failures are retried with exponential
   backoff, up to a per-tool budget.
3. **Circuit breaker** — a tool that keeps failing is opened and skipped
   for a cooldown period instead of burning the whole step budget.
The registry returns raw tool output. Output isolation is applied by the
executor at the point where data enters the model context.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """A callable tool exposed to the agent."""

    name: str
    description: str
    func: Callable[[str], str]
    safe: bool = True
    requires_confirmation: bool = False
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "safe": self.safe,
            "requires_confirmation": self.requires_confirmation,
            "input_schema": self.input_schema,
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Render the tool as an OpenAI-style function definition."""
        parameters = self.input_schema or {"type": "string"}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_call_string(self, args: dict[str, Any]) -> str:
        """
        Render structured arguments as the string the tool function takes.

        Tools accept a single string argument. Native tool calls arrive as
        a dict, so the declared properties are joined in schema order.
        """
        properties = (self.input_schema or {}).get("properties") or {}
        if not properties:
            return str(args.get("input", "") or args)
        parts = []
        for name in properties:
            if name in args:
                parts.append(str(args[name]))
        if not parts:
            return str(args.get("input", "") or args)
        return "\n".join(parts)

    def execute(self, input_text: str) -> str:
        try:
            result = self.func(input_text)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", self.name, e)
            return f"ERROR: {e}"
        # Tools are contracted to return a string; coerce so a misbehaving
        # tool cannot break the dispatch layer.
        return result if isinstance(result, str) else str(result)


class CircuitBreaker:
    """Opens after repeated failures and blocks calls during cooldown."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        opened = self._opened_at.get(name)
        if opened is None:
            return False
        if time.monotonic() - opened >= self.cooldown:
            self._opened_at.pop(name, None)
            self._failures.pop(name, None)
            return False
        return True

    def record_failure(self, name: str) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if count >= self.threshold:
            self._opened_at[name] = time.monotonic()
            logger.warning("Circuit breaker opened for tool '%s'", name)

    def record_success(self, name: str) -> None:
        self._failures.pop(name, None)
        self._opened_at.pop(name, None)


def validate_args(schema: dict[str, Any], args: Any) -> str | None:
    """
    Validate structured tool arguments against a JSON Schema.

    Returns an error message when invalid, otherwise ``None``.
    """
    if not schema:
        return None
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return f"schema validation failed: {e.message}"
    except jsonschema.SchemaError as e:  # type: ignore[attr-defined]
        return f"schema is invalid: {e.message}"
    return None


def validate_input(schema: dict[str, Any], tool_input: str) -> str | None:
    """
    Validate a legacy string tool argument against a JSON Schema.

    Only applies to string-typed schemas. Object-typed tools receive
    structured arguments, which are validated by :func:`validate_args`
    before they are flattened into the string form.
    """
    if not schema:
        return None
    if schema.get("type") != "string":
        return None
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    try:
        jsonschema.validate(tool_input, schema)
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return f"schema validation failed: {e.message}"
    return None


class ToolRegistry:
    """Holds the tool set and dispatches calls with fault tolerance."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._loaded = False
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.breaker = CircuitBreaker(breaker_threshold, breaker_cooldown)
        self.stats: dict[str, dict[str, int]] = {}

    # -- loading --

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._register_builtins()
        self._loaded = True

    def _register_builtins(self) -> None:
        from .file_ops import get_file_tools
        from .search import get_search_tools
        from .shell import get_shell_tools
        from .web import get_web_tools

        for tool in get_file_tools():
            self.register(tool)
        for tool in get_shell_tools():
            self.register(tool)
        for tool in get_search_tools():
            self.register(tool)
        for tool in get_web_tools():
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    # -- lookup --

    def get(self, name: str) -> Tool | None:
        self._ensure_loaded()
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        self._ensure_loaded()
        return list(self._tools.values())

    def list_schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self.list_tools()]

    def list_prompt_schemas(self) -> list[dict[str, Any]]:
        """
        Compact schemas for the model prompt.

        The full schema carries safety metadata the model does not need,
        and every token is resent on every call. This keeps only what the
        model actually needs to call a tool.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema or {"type": "string"},
            }
            for t in self.list_tools()
        ]

    # -- stats --

    def _stat(self, name: str) -> dict[str, int]:
        return self.stats.setdefault(
            name, {"calls": 0, "success": 0, "failed": 0, "retried": 0, "blocked": 0}
        )

    def snapshot_stats(self) -> dict[str, dict[str, int]]:
        self._ensure_loaded()
        return {name: dict(values) for name, values in self.stats.items()}

    # -- dispatch --

    def execute(
        self, name: str, input_text: str, args: dict[str, Any] | None = None
    ) -> str:
        """
        Validate, then run a tool with retries and a circuit breaker.

        ``args`` carries the structured arguments a native tool call
        supplied. When present they are validated against the tool's JSON
        Schema before the tool runs.
        """
        self._ensure_loaded()
        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: Unknown tool '{name}'"

        stat = self._stat(name)

        if self.breaker.is_open(name):
            stat["blocked"] += 1
            return (
                f"ERROR: Tool '{name}' is temporarily disabled after repeated "
                "failures. Use a different approach or try again later."
            )

        if args is not None:
            schema_error = validate_args(tool.input_schema, args)
        else:
            schema_error = validate_input(tool.input_schema, input_text)
        if schema_error:
            stat["failed"] += 1
            return f"ERROR: {schema_error}"

        stat["calls"] += 1
        last_output = ""
        for attempt in range(self.max_retries + 1):
            output = tool.execute(input_text)
            last_output = output
            if not _is_failure(output):
                self.breaker.record_success(name)
                stat["success"] += 1
                return output
            stat["retried"] += 1
            if attempt < self.max_retries:
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "Tool '%s' failed (attempt %s/%s), retrying in %ss",
                    name,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                )
                time.sleep(delay)

        self.breaker.record_failure(name)
        stat["failed"] += 1
        return last_output


def _is_failure(output: object) -> bool:
    """Treat an output starting with ERROR: as a tool failure."""
    if not isinstance(output, str):
        return False
    return output.lstrip().upper().startswith("ERROR:")


registry = ToolRegistry()
