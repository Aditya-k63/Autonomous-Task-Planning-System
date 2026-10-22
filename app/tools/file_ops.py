"""
File operation tools — read, write, edit, list, exist, search.

Every path is resolved through :mod:`app.workspace` so operations stay
inside the sandbox and a per-task workspace can be swapped in.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any

from app.tools.registry import Tool
from app.workspace import current_workspace, resolve_in_workspace

MAX_READ_CHARS = 10000
MAX_WRITE_CHARS = 200000


def _schema(**kwargs) -> dict[str, Any]:
    """Build an object schema for a tool."""
    return {"type": "object", "additionalProperties": False, **kwargs}


def _str(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _read_file(path: str) -> str:
    try:
        p = resolve_in_workspace(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: File not found: {path}"
    if not p.is_file():
        return f"ERROR: Not a file: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR: {e}"
    if len(content) > MAX_READ_CHARS:
        return content[:MAX_READ_CHARS] + f"\n... (truncated, {len(content)} total chars)"
    return content


def _write_file(args: str) -> str:
    parts = args.split("\n", 1)
    if len(parts) < 2:
        return "ERROR: Format: <path>\\n<content>"
    path, content = parts[0].strip(), parts[1]
    if len(content) > MAX_WRITE_CHARS:
        return f"ERROR: Content exceeds {MAX_WRITE_CHARS} characters"
    try:
        p = resolve_in_workspace(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: Wrote {len(content)} chars to {path}"


def _edit_file(args: str) -> str:
    """
    Replace an exact string in a file.

    Input format: ``<path>\\n<old_string>\\n<new_string>``
    """
    parts = args.split("\n", 2)
    if len(parts) < 3:
        return "ERROR: Format: <path>\\n<old_string>\\n<new_string>"
    path, old, new = parts[0].strip(), parts[1], parts[2]
    try:
        p = resolve_in_workspace(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: File not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR: {e}"
    if old not in content:
        return f"ERROR: Target text not found in {path}. Read the file first."
    occurrences = content.count(old)
    content = content.replace(old, new, 1)
    p.write_text(content, encoding="utf-8")
    return f"OK: Replaced 1 of {occurrences} occurrence(s) in {path}"


def _list_dir(path: str) -> str:
    target = (path or ".").strip().lstrip("/\\")
    try:
        p = resolve_in_workspace(target)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: Directory not found: {path}"
    if not p.is_dir():
        return f"ERROR: Not a directory: {path}"
    entries = []
    for item in sorted(p.iterdir()):
        prefix = "DIR " if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else 0
        entries.append(f"[{prefix}] {item.name} ({size} bytes)")
    return "\n".join(entries) if entries else "(empty directory)"


def _file_exists(path: str) -> str:
    try:
        p = resolve_in_workspace(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    return f"EXISTS: {p.exists()}"


def _search_in_files(pattern: str) -> str:
    """Regex search across workspace files."""
    try:
        regex = re.compile(pattern.strip(), re.IGNORECASE)
    except re.error as e:
        return f"ERROR: Invalid regex: {e}"

    root = current_workspace()
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fp = pathlib.Path(dirpath) / fname
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{fp.relative_to(root)}:{i}: {line.strip()}")
                        if len(matches) >= 50:
                            return "\n".join(matches)
            except Exception:
                continue
    return "\n".join(matches) if matches else "No matches found"


def _glob(pattern: str) -> str:
    """List files matching a glob pattern, relative to the workspace."""
    root = current_workspace()
    try:
        hits = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.match(pattern.strip())
        )
    except Exception as e:
        return f"ERROR: {e}"
    return "\n".join(hits[:200]) if hits else "No matching files"


def _grep(args: str) -> str:
    """
    Grep-like content search with optional file filter.

    Input format: ``<pattern>`` or ``<pattern>\\n<file_glob>``
    """
    parts = args.split("\n", 1)
    pattern = parts[0].strip()
    file_glob = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "*"
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"ERROR: Invalid regex: {e}"

    root = current_workspace()
    matches = []
    for path in sorted(root.rglob(file_glob)):
        if not path.is_file():
            continue
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
                    if len(matches) >= 50:
                        return "\n".join(matches)
        except Exception:
            continue
    return "\n".join(matches) if matches else "No matches found"


def get_file_tools() -> list[Tool]:
    return [
        Tool(
            "read_file",
            "Read the contents of a file in the workspace.",
            _read_file,
            safe=True,
            input_schema=_schema(
                properties={"path": _str("File path relative to the workspace.")},
                required=["path"],
            ),
        ),
        Tool(
            "write_file",
            "Create a file or overwrite it with new content.",
            _write_file,
            safe=False,
            requires_confirmation=True,
            input_schema=_schema(
                properties={
                    "path": _str("File path relative to the workspace."),
                    "content": _str("Full content to write to the file."),
                },
                required=["path", "content"],
            ),
        ),
        Tool(
            "edit_file",
            "Replace an exact string inside an existing file.",
            _edit_file,
            safe=False,
            requires_confirmation=True,
            input_schema=_schema(
                properties={
                    "path": _str("File path relative to the workspace."),
                    "old_string": _str("Exact text to replace."),
                    "new_string": _str("Replacement text."),
                },
                required=["path", "old_string", "new_string"],
            ),
        ),
        Tool(
            "list_directory",
            "List the files and directories in a workspace directory.",
            _list_dir,
            safe=True,
            input_schema=_schema(
                properties={"path": _str("Directory path, or '.' for the workspace root.")},
                required=["path"],
            ),
        ),
        Tool(
            "file_exists",
            "Check whether a path exists in the workspace.",
            _file_exists,
            safe=True,
            input_schema=_schema(
                properties={"path": _str("Path relative to the workspace.")},
                required=["path"],
            ),
        ),
        Tool(
            "search_files",
            "Regex search across every file in the workspace.",
            _search_in_files,
            safe=True,
            input_schema=_schema(
                properties={"pattern": _str("Regular expression to search for.")},
                required=["pattern"],
            ),
        ),
        Tool(
            "glob",
            "Find files matching a glob pattern, for example '**/*.py'.",
            _glob,
            safe=True,
            input_schema=_schema(
                properties={"pattern": _str("Glob pattern, for example '**/*.py'.")},
                required=["pattern"],
            ),
        ),
        Tool(
            "grep",
            "Search file contents for a pattern, optionally limited to a file glob.",
            _grep,
            safe=True,
            input_schema=_schema(
                properties={
                    "pattern": _str("Regular expression to search for."),
                    "file_glob": _str("Optional glob limiting which files are searched, e.g. '*.py'."),
                },
                required=["pattern"],
            ),
        ),
    ]
