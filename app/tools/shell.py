"""
Shell execution tool — sandboxed, permission-gated, cross-platform.

Resource limits use ``resource`` on POSIX and are skipped on Windows,
where the equivalent limits come from the subprocess timeout and the
output cap instead.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from app.permissions.bash_security import is_workspace_path
from app.tools.registry import Tool
from app.workspace import current_workspace, resolve_in_workspace

TIMEOUT = 30
MAX_OUTPUT_SIZE = 5000

IS_WINDOWS = platform.system() == "Windows"

try:  # POSIX only
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore

MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
CPU_LIMIT_SECONDS = 30
FILE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024


def _apply_sandbox_limits() -> None:
    """Apply POSIX resource limits. No-op where unsupported."""
    if resource is None:
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)
        )
        resource.setrlimit(
            resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS)
        )
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT_BYTES, FILE_SIZE_LIMIT_BYTES)
        )
    except (ValueError, OSError):
        pass


def _pre_exec():
    """preexec_fn is unsupported on Windows."""
    return _apply_sandbox_limits if not IS_WINDOWS else None


def _contains_outside_workspace(command: str) -> str | None:
    """Return a reason when the command reaches outside the workspace."""
    for token in command.replace('"', " ").replace("'", " ").split():
        if token.startswith("/") and not token.startswith("/tmp/"):
            if not is_workspace_path(token):
                return f"path outside workspace: {token}"
    return None


def _run_shell(command: str) -> str:
    cmd = command.strip()
    if not cmd:
        return "ERROR: Empty command"

    workspace = current_workspace()
    workspace.mkdir(parents=True, exist_ok=True)

    escape = _contains_outside_workspace(cmd)
    if escape:
        return f"ERROR: Blocked ({escape})"

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(workspace),
        "TEMP": str(workspace),
        "TMP": str(workspace),
    }

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=str(workspace),
            env=env,
            preexec_fn=_pre_exec(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {TIMEOUT}s"
    except MemoryError:
        return "ERROR: Command exceeded memory limit"
    except ValueError as e:
        return f"ERROR: Command could not be executed: {e}"
    except Exception as e:
        return f"ERROR: {e}"

    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR: {result.stderr}"
    if not output.strip():
        output = "(no output)"
    if len(output) > MAX_OUTPUT_SIZE:
        output = output[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(output)} total chars)"
    return output


def get_shell_tools() -> list[Tool]:
    return [
        Tool(
            "run_shell",
            "Execute a shell command in the sandboxed workspace. Input: shell command string.",
            _run_shell,
            safe=False,
            requires_confirmation=True,
        ),
    ]
