"""ActionLog repository — append-only writes only. Never update or delete."""

from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.models.action_log import ActionLog


class ActionLogRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_action(
        self,
        *,
        incident_id: str,
        action_type: str,
        command: str,
        stdout: str | None = None,
        exit_code: int | None = None,
        approved_by: str = "system",
    ) -> ActionLog:
        entry = ActionLog(
            incident_id=incident_id,
            action_type=action_type,
            command=command,
            stdout=stdout,
            exit_code=exit_code,
            approved_by=approved_by,
        )
        self._session.add(entry)
        await self._session.flush()  # get the generated ID without committing
        return entry
