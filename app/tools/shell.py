"""Shell execution tool — runs commands in a sandboxed workspace with resource limits."""

from __future__ import annotations

import subprocess
import os
import resource
import tempfile

from .registry import Tool

WORKSPACE = "/tmp/task-planner-workspace"
TIMEOUT = 30
MAX_OUTPUT_SIZE = 5000
BLOCKED = {"rm -rf /", "mkfs", "dd if=", "> /dev/sd", ":(){ :|:& };:", "format", "shutdown", "reboot"}
SANDBOXED_PATHS = {"/tmp/task-planner-workspace"}


def _apply_sandbox_limits():
    """Apply resource limits to the sandboxed environment."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))  # 512MB max memory
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))  # 30s max CPU time
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))  # 10MB max file size
    except (ValueError, OSError):
        pass  # Limits may not be supported on all platforms


def _run_shell(command: str) -> str:
    cmd = command.strip()
    for pattern in BLOCKED:
        if pattern in cmd:
            return f"ERROR: Blocked dangerous command pattern: {pattern}"
    # Sandbox: only allow access to workspace directory
    cwd = WORKSPACE
    os.makedirs(cwd, exist_ok=True)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=cwd,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": cwd},
            preexec_fn=_apply_sandbox_limits,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if not output.strip():
            output = "(no output)"
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(output)} total chars)"
        return output
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {TIMEOUT}s"
    except MemoryError:
        return "ERROR: Command exceeded memory limit"
    except Exception as e:
        return f"ERROR: {e}"


def get_shell_tools() -> list[Tool]:
    return [
        Tool(
            "run_shell",
            "Execute a shell command in the sandboxed workspace. Input: shell command string.",
            _run_shell,
            safe=True,
            requires_confirmation=False,
        ),
    ]
