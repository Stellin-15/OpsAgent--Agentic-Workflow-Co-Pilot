"""Pydantic schemas for the action execution API (Phase 3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParameterSpecOut(BaseModel):
    name: str
    required: bool
    default: str | None
    description: str


class ActionOut(BaseModel):
    """Summary of a single action from the catalog."""

    name: str
    description: str
    labels: list[str]
    parameters: list[ParameterSpecOut]
    has_dry_run: bool
    has_rollback: bool
    timeout_seconds: int
    requires_approval: bool
    safety_level: str
    catalog_file: str


class ExecuteActionRequest(BaseModel):
    """Body for POST /api/actions/{name}/execute."""

    incident_id: str = Field(..., description="The incident this action is being run for")
    params: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter values — keys match the action's parameter names",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, run the safe read-only equivalent command",
    )
    approved_by: str = Field(
        default="api",
        description="User or system that approved this execution",
    )


class ExecutionResultOut(BaseModel):
    """Response schema for an executed action."""

    action_name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    rolled_back: bool
    rollback_exit_code: int | None
    rollback_stdout: str
    succeeded: bool
