"""Base protocols and data classes for the execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ParameterSpec:
    name: str
    required: bool = True
    default: str | None = None
    description: str = ""


@dataclass
class ActionSpec:
    """Raw definition loaded from a YAML catalog entry."""

    name: str
    description: str
    labels: list[str]
    parameters: list[ParameterSpec]
    command: str
    dry_run_command: str | None
    rollback_command: str | None
    timeout_seconds: int
    requires_approval: bool
    safety_level: SafetyLevel
    catalog_file: str = ""  # set by loader — which YAML file it came from


class ActionExecutorProtocol(Protocol):
    """Minimal interface that any action executor must satisfy."""

    async def execute(
        self,
        action: ActionSpec,
        params: dict[str, str],
        *,
        dry_run: bool = False,
    ) -> "ExecutionResult": ...


@dataclass
class ExecutionResult:
    """The outcome of running a single action command."""

    action_name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    rolled_back: bool = False
    rollback_exit_code: int | None = None
    rollback_stdout: str = ""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out
