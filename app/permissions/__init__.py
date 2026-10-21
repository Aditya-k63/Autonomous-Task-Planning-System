"""
Permission layer for tool execution.

Exposes the policy, the decision type, and the human-facing explainer.
"""

from __future__ import annotations

from app.permissions.bash_security import ALLOW, ASK, DENY, CommandRisk, classify_command
from app.permissions.explainer import PermissionExplanation, explain
from app.permissions.model import (
    ALWAYS_ASK_TOOLS,
    PermissionDecision,
    PermissionPolicy,
    PermissionRequest,
)

__all__ = [
    "ALLOW",
    "ASK",
    "DENY",
    "ALWAYS_ASK_TOOLS",
    "CommandRisk",
    "PermissionDecision",
    "PermissionExplanation",
    "PermissionPolicy",
    "PermissionRequest",
    "classify_command",
    "explain",
    "get_policy",
]


def get_policy() -> PermissionPolicy:
    """Build the permission policy from application settings."""
    from app.config import get_settings

    settings = get_settings()
    return PermissionPolicy(mode=settings.permission_mode or "ask")
