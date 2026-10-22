"""
Workspace root resolution.

The agent sandbox used to be hard-coded to ``/tmp/task-planner-workspace``,
which breaks on Windows and makes per-task sandboxes impossible. This module
gives every tool a single source of truth and lets the evaluation harness
swap in an isolated workspace per task.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextvars import ContextVar
from pathlib import Path

_DEFAULT_ROOT = Path(
    os.environ.get("AGENT_WORKSPACE_ROOT")
    or (Path(tempfile.gettempdir()) / "agentbench-workspace")
)

_override: ContextVar[Path | None] = ContextVar("agent_workspace_override", default=None)


def current_workspace() -> Path:
    """Return the workspace root for the active context."""
    root = _override.get() or _DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_workspace(path: str | Path | None) -> None:
    """Override the workspace root for the current context (None = reset)."""
    if path is None:
        _override.set(None)
    else:
        _override.set(Path(path))


def reset_workspace() -> None:
    """Delete and recreate the workspace, wiping all task state."""
    root = current_workspace()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_in_workspace(path: str) -> Path:
    """Resolve a user-supplied path inside the workspace, blocking escapes."""
    root = current_workspace().resolve()
    candidate = (root / path.strip().lstrip("/\\")).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"Path escapes workspace: {path}")
    return candidate
