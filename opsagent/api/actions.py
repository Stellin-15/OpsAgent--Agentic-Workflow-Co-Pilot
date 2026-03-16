"""
Action execution API — Phase 3.

Endpoints:
    GET  /api/actions                    — list all catalog actions
    GET  /api/actions/{name}             — get action details
    POST /api/actions/{name}/dry-run     — run the safe read-only variant
    POST /api/actions/{name}/execute     — run the real command (approval required)
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.api.deps import get_db
from opsagent.execution.action_catalog import ActionCatalog, ActionDefinition
from opsagent.execution.safe_executor import ExecutionRequest, SafeExecutor
from opsagent.schemas.action import (
    ActionOut,
    ExecuteActionRequest,
    ExecutionResultOut,
    ParameterSpecOut,
)

log = structlog.get_logger(__name__)
router = APIRouter(tags=["actions"])

# Module-level catalog singleton — loaded once at import time.
# The catalog directory is resolved relative to the project root so it works
# both inside Docker (/app/actions) and in local development.
_catalog: ActionCatalog | None = None


def _get_catalog() -> ActionCatalog:
    global _catalog
    if _catalog is None:
        candidates = [
            Path("/app/actions"),
            Path("actions"),
        ]
        catalog_dir = next((p for p in candidates if p.is_dir()), Path("actions"))
        _catalog = ActionCatalog(catalog_dir)
        _catalog.load()
    return _catalog


def _to_action_out(defn: ActionDefinition) -> ActionOut:
    spec = defn.spec
    return ActionOut(
        name=spec.name,
        description=spec.description,
        labels=spec.labels,
        parameters=[
            ParameterSpecOut(
                name=p.name,
                required=p.required,
                default=p.default,
                description=p.description,
            )
            for p in spec.parameters
        ],
        has_dry_run=spec.dry_run_command is not None,
        has_rollback=spec.rollback_command is not None,
        timeout_seconds=spec.timeout_seconds,
        requires_approval=spec.requires_approval,
        safety_level=spec.safety_level.value,
        catalog_file=spec.catalog_file,
    )


# ─────────────────────────────────────────── routes ──────────────────────────


@router.get("/actions", response_model=list[ActionOut])
async def list_actions(label: str | None = None) -> list[ActionOut]:
    """List all available actions. Optionally filter by label."""
    catalog = _get_catalog()
    if label:
        actions = catalog.list_by_label(label)
    else:
        actions = catalog.list_all()
    return [_to_action_out(a) for a in actions]


@router.get("/actions/{name}", response_model=ActionOut)
async def get_action(name: str) -> ActionOut:
    """Get details of a single action by name."""
    catalog = _get_catalog()
    try:
        defn = catalog.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action '{name}' not found")
    return _to_action_out(defn)


@router.post("/actions/{name}/dry-run", response_model=ExecutionResultOut)
async def dry_run_action(
    name: str,
    body: ExecuteActionRequest,
    session: AsyncSession = Depends(get_db),
) -> ExecutionResultOut:
    """
    Execute the safe read-only dry-run variant of an action.
    Does not require approval; always logs to action_log.
    """
    catalog = _get_catalog()
    try:
        defn = catalog.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action '{name}' not found")

    executor = SafeExecutor()
    request = ExecutionRequest(
        action=defn,
        params=body.params,
        incident_id=body.incident_id,
        approved_by=body.approved_by,
        dry_run=True,
    )

    log.info(
        "api.dry_run",
        action=name,
        incident_id=body.incident_id,
        approved_by=body.approved_by,
    )

    result = await executor.run(request, session)
    await session.commit()

    return ExecutionResultOut(
        action_name=result.action_name,
        command=result.command,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        rolled_back=result.rolled_back,
        rollback_exit_code=result.rollback_exit_code,
        rollback_stdout=result.rollback_stdout,
        succeeded=result.succeeded,
    )


@router.post("/actions/{name}/execute", response_model=ExecutionResultOut)
async def execute_action(
    name: str,
    body: ExecuteActionRequest,
    session: AsyncSession = Depends(get_db),
) -> ExecutionResultOut:
    """
    Execute an action for real.

    Actions with ``requires_approval: true`` must supply a non-empty
    ``approved_by`` field. The caller is responsible for verifying that
    approval happened (e.g. via Slack interactive button or web UI).

    This endpoint is intentionally synchronous (runs subprocess inline).
    For very long-running actions use the Celery-based variant in Phase 4.
    """
    catalog = _get_catalog()
    try:
        defn = catalog.get(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action '{name}' not found")

    if defn.requires_approval and (not body.approved_by or body.approved_by == "api"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Action '{name}' requires approval. "
                "Supply 'approved_by' with the approver's identity."
            ),
        )

    executor = SafeExecutor()
    request = ExecutionRequest(
        action=defn,
        params=body.params,
        incident_id=body.incident_id,
        approved_by=body.approved_by,
        dry_run=False,
    )

    log.info(
        "api.execute",
        action=name,
        incident_id=body.incident_id,
        approved_by=body.approved_by,
        safety_level=defn.safety_level.value,
    )

    result = await executor.run(request, session)
    await session.commit()

    return ExecutionResultOut(
        action_name=result.action_name,
        command=result.command,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        rolled_back=result.rolled_back,
        rollback_exit_code=result.rollback_exit_code,
        rollback_stdout=result.rollback_stdout,
        succeeded=result.succeeded,
    )
