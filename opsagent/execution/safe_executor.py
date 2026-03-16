"""Safe subprocess executor.

Runs action commands with:
- Parameter validation and {placeholder} interpolation
- Subprocess isolation (no shell=True for tokenised commands)
- Hard timeout (SIGKILL after timeout_seconds)
- stdout + stderr capture (never printed to console)
- Automatic rollback if exit_code != 0 and rollback_command is defined
- Structured audit entries via action_log_repo

Every execution is append-only logged to the ``action_log`` table.
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from opsagent.execution.action_catalog import ActionDefinition
from opsagent.execution.base import ExecutionResult, SafetyLevel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger(__name__)

# Hard upper bound — catalog can request less but never more.
MAX_TIMEOUT_SECONDS = 300


@dataclass
class ExecutionRequest:
    action: ActionDefinition
    params: dict[str, str]           # raw user-supplied params (may be incomplete)
    incident_id: str
    approved_by: str
    dry_run: bool = False


class SafeExecutor:
    """Runs a single action command safely."""

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    async def run(
        self,
        request: ExecutionRequest,
        session: "AsyncSession",
    ) -> ExecutionResult:
        """
        Validate params → choose command → execute → rollback on failure →
        write action_log row → return result.
        """
        action = request.action
        resolved_params = action.validate_params(request.params)

        if request.dry_run:
            command = action.fill_dry_run_command(resolved_params)
            if command is None:
                # No dry-run available — treat main command as dry-run equivalent
                command = action.fill_command(resolved_params)
            log.info(
                "executor.dry_run",
                action=action.name,
                command=command,
                incident_id=request.incident_id,
            )
        else:
            command = action.fill_command(resolved_params)
            log.info(
                "executor.execute",
                action=action.name,
                command=command,
                incident_id=request.incident_id,
                approved_by=request.approved_by,
            )

        timeout = min(action.spec.timeout_seconds, MAX_TIMEOUT_SECONDS)
        result = await self._run_command(action.name, command, timeout)

        # Auto-rollback on failure (non-dry-run only)
        if not result.succeeded and not request.dry_run:
            result = await self._maybe_rollback(action, resolved_params, result)

        # Persist to audit log
        await self._write_action_log(
            session=session,
            incident_id=request.incident_id,
            result=result,
            approved_by=request.approved_by,
            dry_run=request.dry_run,
        )

        return result

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #

    async def _run_command(
        self,
        action_name: str,
        command: str,
        timeout: int,
    ) -> ExecutionResult:
        tokens = shlex.split(command)
        timed_out = False
        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout),
            )
            exit_code = proc.returncode or 0

        except asyncio.TimeoutError:
            timed_out = True
            exit_code = -1
            stdout_bytes = b""
            stderr_bytes = b"[OpsAgent] Command timed out"
            if proc is not None:
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
            log.warning(
                "executor.timeout",
                action=action_name,
                command=command,
                timeout=timeout,
            )

        except FileNotFoundError as exc:
            # Binary not found on PATH — e.g. kubectl not installed
            exit_code = 127
            stdout_bytes = b""
            stderr_bytes = str(exc).encode()
            log.error("executor.binary_not_found", action=action_name, error=str(exc))

        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()

        if exit_code != 0:
            log.warning(
                "executor.nonzero_exit",
                action=action_name,
                exit_code=exit_code,
                stderr=stderr[:500],
            )
        else:
            log.info(
                "executor.success",
                action=action_name,
                exit_code=exit_code,
                stdout_preview=stdout[:200],
            )

        return ExecutionResult(
            action_name=action_name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------ #
    # Rollback
    # ------------------------------------------------------------------ #

    async def _maybe_rollback(
        self,
        action: ActionDefinition,
        resolved_params: dict[str, str],
        original_result: ExecutionResult,
    ) -> ExecutionResult:
        rollback_cmd = action.fill_rollback_command(resolved_params)
        if rollback_cmd is None:
            log.info(
                "executor.no_rollback_defined",
                action=action.name,
            )
            return original_result

        log.warning(
            "executor.rolling_back",
            action=action.name,
            rollback_command=rollback_cmd,
        )
        rollback_result = await self._run_command(
            f"{action.name}__rollback",
            rollback_cmd,
            timeout=min(action.spec.timeout_seconds, MAX_TIMEOUT_SECONDS),
        )
        original_result.rolled_back = True
        original_result.rollback_exit_code = rollback_result.exit_code
        original_result.rollback_stdout = rollback_result.stdout
        return original_result

    # ------------------------------------------------------------------ #
    # Audit log
    # ------------------------------------------------------------------ #

    async def _write_action_log(
        self,
        session: "AsyncSession",
        incident_id: str,
        result: ExecutionResult,
        approved_by: str,
        dry_run: bool,
    ) -> None:
        # Import here to avoid circular imports at module load time.
        from opsagent.repositories.action_log_repo import ActionLogRepo

        repo = ActionLogRepo(session)
        action_type = f"dry_run:{result.action_name}" if dry_run else result.action_name
        await repo.log_action(
            incident_id=incident_id,
            action_type=action_type,
            command=result.command,
            stdout=f"{result.stdout}\n{result.stderr}".strip(),
            exit_code=result.exit_code,
            approved_by=approved_by,
        )
