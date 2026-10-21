"""
Permission model for tool execution.

Every tool call is checked against a policy before it runs. The policy
returns one of three decisions:

- ``ALLOW``  — run the tool
- ``ASK``    — run only if the caller approves it
- ``DENY``   — refuse, with a reason the agent can act on

The module is deliberately small: a decision, a reason, and an optional
payload describing *why* approval is needed so the UI can explain itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.permissions.bash_security import ALLOW, ASK, DENY, classify_command


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

    @classmethod
    def from_str(cls, value: str) -> "PermissionDecision":
        return cls(value)


@dataclass
class PermissionRequest:
    """A proposed tool call awaiting a decision."""

    tool: str
    input: str
    decision: PermissionDecision
    reason: str
    risk_details: list[str] = field(default_factory=list)
    blast_radius: str = "none"

    @property
    def needs_approval(self) -> bool:
        return self.decision is PermissionDecision.ASK

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "input": self.input,
            "decision": self.decision.value,
            "reason": self.reason,
            "risk_details": self.risk_details,
            "blast_radius": self.blast_radius,
            "needs_approval": self.needs_approval,
        }


# Tools that always require a human decision, regardless of arguments.
ALWAYS_ASK_TOOLS: frozenset[str] = frozenset({"run_shell", "write_file", "edit_file"})

# Tools that reach an external service, so they need confirmation in
# "enforce" mode.
NETWORK_TOOLS: frozenset[str] = frozenset({"web_search", "fetch_url"})

# Tools that are never permitted.
NEVER_ALLOWED_TOOLS: frozenset[str] = frozenset()

# Blast-radius descriptions used to explain a request to the user.
BLAST_RADIUS: dict[str, str] = {
    "run_shell": "can modify files or system state inside the sandbox",
    "write_file": "overwrites or creates a file in the workspace",
    "edit_file": "modifies an existing file in the workspace",
    "read_file": "reads a file in the workspace",
    "list_directory": "lists workspace contents",
    "file_exists": "checks whether a file exists",
    "search_files": "scans workspace file contents",
    "glob": "lists files matching a pattern",
    "grep": "scans workspace file contents",
    "web_search": "sends a query to an external service",
    "fetch_url": "fetches content from an external URL",
}


class PermissionPolicy:
    """
    Decides whether a tool call may run.

    ``mode`` controls how ``ASK`` decisions are handled:

    - ``off``      — run everything except hard denies (development default)
    - ``ask``      — surface ``ASK`` decisions to the caller
    - ``enforce``  — refuse anything that is not explicitly allowed
    """

    def __init__(
        self,
        mode: str = "ask",
        *,
        approver: Callable[[PermissionRequest], bool] | None = None,
        always_ask_tools: frozenset[str] = ALWAYS_ASK_TOOLS,
    ) -> None:
        self.mode = mode
        self.approver = approver
        self.always_ask_tools = always_ask_tools

    # -- public API --

    def check(self, tool: str, tool_input: str) -> PermissionRequest:
        """Evaluate a tool call without executing it."""
        if tool in NEVER_ALLOWED_TOOLS:
            return PermissionRequest(
                tool=tool,
                input=tool_input,
                decision=PermissionDecision.DENY,
                reason=f"Tool '{tool}' is not permitted",
            )

        if tool == "run_shell":
            risk = classify_command(tool_input)
            if risk.decision == DENY:
                return PermissionRequest(
                    tool=tool,
                    input=tool_input,
                    decision=PermissionDecision.DENY,
                    reason=f"Blocked: {risk.reason}",
                    risk_details=[risk.reason],
                    blast_radius=BLAST_RADIUS.get(tool, "unknown"),
                )
            if risk.decision == ASK:
                return PermissionRequest(
                    tool=tool,
                    input=tool_input,
                    decision=PermissionDecision.ASK,
                    reason=f"Command {risk.reason}",
                    risk_details=[risk.reason],
                    blast_radius=BLAST_RADIUS.get(tool, "unknown"),
                )
            return PermissionRequest(
                tool=tool,
                input=tool_input,
                decision=PermissionDecision.ALLOW,
                reason="Read-only command",
                blast_radius="none",
            )

        if tool in self.always_ask_tools:
            return PermissionRequest(
                tool=tool,
                input=tool_input,
                decision=PermissionDecision.ASK,
                reason=f"'{tool}' needs confirmation",
                blast_radius=BLAST_RADIUS.get(tool, "unknown"),
            )

        if tool in NETWORK_TOOLS:
            return PermissionRequest(
                tool=tool,
                input=tool_input,
                decision=PermissionDecision.ASK,
                reason=f"'{tool}' sends data to an external service",
                blast_radius=BLAST_RADIUS.get(tool, "unknown"),
            )

        return PermissionRequest(
            tool=tool,
            input=tool_input,
            decision=PermissionDecision.ALLOW,
            reason="Read-only tool",
            blast_radius=BLAST_RADIUS.get(tool, "none"),
        )

    def authorize(self, tool: str, tool_input: str) -> tuple[bool, str, PermissionRequest]:
        """
        Resolve a tool call to ``(allowed, message, request)``.

        In ``enforce`` mode anything not explicitly allowed is refused. In
        ``ask`` mode an ``ASK`` decision is passed to the approver callback
        when one is configured, and otherwise allowed through.
        """
        request = self.check(tool, tool_input)

        if request.decision is PermissionDecision.DENY:
            return False, request.reason, request

        if request.decision is PermissionDecision.ALLOW:
            return True, "", request

        if self.mode == "off":
            return True, "", request
        if self.mode == "enforce":
            return False, f"Permission required: {request.reason}", request
        if self.approver is not None:
            approved = self.approver(request)
            if approved:
                return True, "", request
            return False, f"User denied: {request.reason}", request
        return True, "", request
