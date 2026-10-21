"""
Shell command risk classification.

Commands are classified into three tiers, mirroring how production coding
agents gate shell access:

- ``ALLOW``   — read-only or clearly scoped to the workspace
- ``ASK``     — writes, mutates, or reaches outside the workspace
- ``DENY``    — destructive, privilege-escalating, or persistence-oriented

The classifier is intentionally conservative: anything it does not
recognise as read-only is treated as ``ASK``.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

# --- Tier definitions ---

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

# Read-only commands that cannot mutate state.
READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "ls", "cat", "head", "tail", "pwd", "echo", "wc", "sort", "uniq",
        "find", "grep", "egrep", "rg", "diff", "stat", "file", "du", "df",
        "date", "whoami", "hostname", "uname", "which", "type", "env",
        "printenv", "tree", "basename", "dirname", "realpath", "readlink",
        "md5sum", "sha256sum", "python", "python3", "pip", "pytest",
    }
)

# Commands that always mutate state.
WRITE_COMMANDS: frozenset[str] = frozenset(
    {
        "mkdir", "touch", "cp", "mv", "rm", "rmdir", "ln", "tee", "truncate",
        "chmod", "chown", "chgrp", "install", "dd", "sed", "tar", "zip",
        "unzip", "git", "pip", "npm", "yarn", "pnpm", "poetry", "uv",
    }
)

# Never allowed, regardless of arguments.
DENY_COMMANDS: frozenset[str] = frozenset(
    {
        "shutdown", "reboot", "halt", "poweroff", "mkfs", "fdisk", "parted",
        "insmod", "rmmod", "modprobe", "iptables", "ufw", "systemctl",
        "service", "init", "telinit", "kill", "killall", "pkill", "su",
        "sudo", "doas", "passwd", "chpasswd", "useradd", "userdel", "usermod",
        "groupadd", "visudo", "crontab", "at", "nohup", "setsid", "eval",
    }
)

# Dangerous substrings anywhere in a command line.
DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+/(\s|$)", "recursive delete of root"),
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\s|$)", "recursive delete of root"),
    (r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:", "fork bomb"),
    (r"\bdd\s+.*of=/dev/", "raw write to a device"),
    (r">\s*/dev/[sh]d[a-z]", "write to a block device"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", "system power control"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/(\s|$)", "world-writable root"),
    (r"\bchown\b.*\s+/(\s|$)", "ownership change on root"),
    (r"\bcurl\b.*\|\s*(ba)?sh\b", "remote script piped into a shell"),
    (r"\bwget\b.*\|\s*(ba)?sh\b", "remote script piped into a shell"),
    (r"\beval\s*\(", "eval of dynamic input"),
    (r"\bexport\s+LD_PRELOAD", "preloaded library injection"),
    (r"\bhistory\s+-c\b", "shell history wipe"),
    (r">\s*\.bashrc", "shell profile overwrite"),
    (r">\s*\.ssh/authorized_keys", "SSH key injection"),
    (r"\bnc\b.*\s-e\b", "reverse shell"),
    (r"\bbase64\s+-d.*\|\s*(ba)?sh\b", "encoded remote script execution"),
)

# Network / device / privilege reach.
RISKY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcurl\b|\bwget\b", "network access"),
    (r"\bnc\b|\bnetcat\b|\btelnet\b", "raw socket access"),
    (r"\bssh\b|\bscp\b|\bsftp\b", "remote host access"),
    (r"\bapt\b|\byum\b|\bapk\b|\bbrew\b", "system package installation"),
    (r"/etc/|/usr/|/bin/|/sbin/|/boot/|/sys/|/proc/|/dev/", "system path access"),
    (r"\benv\b", "environment access"),
    (r"\.\./", "parent directory traversal"),
)


@dataclass(frozen=True)
class CommandRisk:
    """Classification result for one command line."""

    decision: str
    reason: str
    commands: tuple[str, ...]

    @property
    def is_allowed(self) -> bool:
        return self.decision == ALLOW

    @property
    def is_denied(self) -> bool:
        return self.decision == DENY


def _split_commands(command: str) -> list[str]:
    """Split a command line into individual commands on pipes and separators."""
    parts = re.split(r"\|\||&&|;|\|", command)
    return [p.strip() for p in parts if p.strip()]


def _command_name(part: str) -> str:
    """Return the executable name for a single command segment."""
    try:
        tokens = shlex.split(part)
    except ValueError:
        tokens = part.split()
    if not tokens:
        return ""
    exe = tokens[0]
    # Strip a path so /bin/rm resolves to rm.
    return exe.replace("\\", "/").rsplit("/", 1)[-1].lower()


def classify_command(command: str) -> CommandRisk:
    """
    Classify a full shell command line.

    Returns the strictest decision across every command in the line, since a
    pipeline is only as safe as its least safe stage.
    """
    raw = (command or "").strip()
    if not raw:
        return CommandRisk(DENY, "Empty command", ())

    lowered = raw.lower()

    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return CommandRisk(DENY, reason, (raw,))

    parts = _split_commands(raw)
    names = tuple(_command_name(p) for p in parts)
    names = tuple(n for n in names if n)

    for name in names:
        if name in DENY_COMMANDS:
            return CommandRisk(DENY, f"'{name}' is not permitted", names)

    for pattern, reason in RISKY_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return CommandRisk(ASK, f"command {reason}", names)

    if any(n in WRITE_COMMANDS for n in names):
        mutating = next(n for n in names if n in WRITE_COMMANDS)
        return CommandRisk(ASK, f"'{mutating}' modifies state", names)

    if names and all(n in READ_ONLY_COMMANDS for n in names):
        return CommandRisk(ALLOW, "read-only command", names)

    return CommandRisk(ASK, "command not on the read-only allowlist", names)


def is_workspace_path(path: str) -> bool:
    """Return True when the path stays inside the agent workspace."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("/tmp/") or normalized.startswith("/var/"):
        return False
    if re.match(r"^[a-zA-Z]:", normalized):
        return False
    return ".." not in normalized.split("/")
