"""
Human-readable explanations for permission requests.

When an agent asks for approval, the user needs to know *what* will happen,
not just the tool name. This module turns a :class:`PermissionRequest`
into a short, structured explanation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.permissions.model import PermissionRequest


@dataclass(frozen=True)
class PermissionExplanation:
    """A structured explanation shown to the user before approval."""

    title: str
    what_will_happen: str
    why_it_is_needed: str
    blast_radius: str
    risk_details: tuple[str, ...]

    def render(self) -> str:
        """Render the explanation as plain text for the CLI and API."""
        lines = [
            self.title,
            "",
            f"What will happen: {self.what_will_happen}",
            f"Why it is needed: {self.why_it_is_needed}",
            f"Blast radius: {self.blast_radius}",
        ]
        if self.risk_details:
            lines.append("Risk signals:")
            lines.extend(f"  - {detail}" for detail in self.risk_details)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "what_will_happen": self.what_will_happen,
            "why_it_is_needed": self.why_it_is_needed,
            "blast_radius": self.blast_radius,
            "risk_details": list(self.risk_details),
        }


# What each tool does, in one line.
TOOL_PURPOSE: dict[str, str] = {
    "run_shell": "run a shell command in the sandboxed workspace",
    "write_file": "create or overwrite a file in the workspace",
    "edit_file": "modify part of an existing file in the workspace",
    "read_file": "read a file from the workspace",
    "list_directory": "list the contents of a workspace directory",
    "file_exists": "check whether a path exists in the workspace",
    "search_files": "scan workspace files for a regular expression",
    "glob": "find files matching a glob pattern",
    "grep": "search file contents for a pattern",
    "web_search": "send a query to a web search service",
    "fetch_url": "download a URL and return its text content",
}


def explain(request: PermissionRequest) -> PermissionExplanation:
    """Build a user-facing explanation for a permission request."""
    purpose = TOOL_PURPOSE.get(request.tool, f"invoke the '{request.tool}' tool")
    what = f"The agent wants to {purpose}."
    if request.tool == "run_shell":
        what = f"The agent wants to run: `{request.input.strip()}`"
    why = request.reason or "the agent judged this action was needed to continue."
    return PermissionExplanation(
        title=f"Permission needed: {request.tool}",
        what_will_happen=what,
        why_it_is_needed=why,
        blast_radius=request.blast_radius or "none",
        risk_details=tuple(request.risk_details),
    )
