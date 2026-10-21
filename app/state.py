from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# --- Enums ---

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CLARIFYING = "clarifying"
    REVIEWING = "reviewing"
    DONE = "done"
    ERROR = "error"


# --- Models ---

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: list[str] = Field(default_factory=list)
    substeps: list[str] = Field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None


class ClarificationOption(BaseModel):
    id: str
    label: str
    description: str


class Clarification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str
    question: str
    context: str
    options: list[ClarificationOption]
    answer: str | None = None
    answered_at: str | None = None


class Step(BaseModel):
    task_id: str
    action: str
    tool: str | None = None
    input: str | None = None
    output: str | None = None
    success: bool = True
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PermissionEvent(BaseModel):
    """A record of one permission decision, used for auditing and metrics."""

    task_id: str
    tool: str
    input: str
    decision: str
    reason: str
    blast_radius: str = "none"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# --- Graph State ---

class AgentState(dict):
    """
    LangGraph state passed between all nodes.

    Uses dict subclass for LangGraph compatibility while providing
    typed access via helper methods.
    """

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "goal": "",
            "tasks": [],
            "current_task_id": None,
            "execution_history": [],
            "messages": [],
            "retry_count": 0,
            "max_retries": 3,
            "pending_clarification": None,
            "clarification_history": [],
            "permission_events": [],
            "phase": AgentPhase.IDLE,
            "error": None,
        }
        defaults.update(kwargs)
        super().__init__(defaults)

    @property
    def goal(self) -> str:
        return self["goal"]

    @property
    def tasks(self) -> list[Task]:
        return [Task(**t) if isinstance(t, dict) else t for t in self["tasks"]]

    @property
    def current_task_id(self) -> str | None:
        return self["current_task_id"]

    @property
    def pending_clarification(self) -> Clarification | None:
        v = self["pending_clarification"]
        if v is None:
            return None
        return Clarification(**v) if isinstance(v, dict) else v

    @property
    def error(self) -> str | None:
        return self.get("error")

    @property
    def phase(self) -> AgentPhase:
        return AgentPhase(self["phase"])

    @property
    def retry_count(self) -> int:
        return self["retry_count"]

    @property
    def execution_history(self) -> list[Step]:
        return [Step(**s) if isinstance(s, dict) else s for s in self["execution_history"]]

    def get_current_task(self) -> Task | None:
        if self.current_task_id is None:
            return None
        for t in self.tasks:
            if t.id == self.current_task_id:
                return t
        return None

    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def all_completed(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in self.tasks)
