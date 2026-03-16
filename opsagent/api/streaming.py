"""Server-Sent Events (SSE) streaming endpoint (Phase 4).

Streams draft tokens in real-time to the browser/client.

GET /api/incidents/{incident_id}/stream

The client opens an EventSource connection:
    const source = new EventSource('/api/incidents/{id}/stream')
    source.onmessage = (e) => appendToken(e.data)
    source.addEventListener('done', () => source.close())
    source.addEventListener('error', () => source.close())

Events emitted:
    data: <token>           — a chunk of text from the LLM
    event: done\ndata: {}   — stream complete
    event: error\ndata: ... — an error occurred
"""

from __future__ import annotations

import asyncio
import json

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from opsagent.api.deps import get_db, get_rag
from opsagent.repositories.incident_repo import IncidentRepo

log = structlog.get_logger(__name__)
router = APIRouter(tags=["streaming"])


def _sse_event(data: str, event: str | None = None) -> str:
    """Format a single SSE message."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    # Each data line must be prefixed with "data: "
    for line in data.splitlines():
        lines.append(f"data: {line}")
    lines.append("")   # blank line terminates the event
    lines.append("")
    return "\n".join(lines)


@router.get("/incidents/{incident_id}/stream")
async def stream_draft(
    incident_id: str,
    session: AsyncSession = Depends(get_db),
    rag=Depends(get_rag),
) -> StreamingResponse:
    """
    Stream the AI draft for an incident as SSE tokens.

    The RAG pipeline retrieves runbook context then streams LLM tokens.
    If the pipeline is not ready (no API key configured) it returns a
    single 'not_ready' event instead of an error.
    """
    # Verify the incident exists
    repo = IncidentRepo(session)
    incident = await repo.get_by_id(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Support both Phase 1-3 RagService and Phase 4 RagPipeline
    if not rag.is_ready():
        async def _not_ready():
            yield _sse_event(
                json.dumps({"detail": "RAG pipeline not ready"}),
                event="error",
            )
        return StreamingResponse(_not_ready(), media_type="text/event-stream")

    async def _token_generator():
        try:
            # Phase 4 RagPipeline exposes stream_draft()
            if hasattr(rag, "stream_draft"):
                async for token in rag.stream_draft(
                    incident_id=incident_id,
                    alert_name=incident.alert_name,
                    labels=incident.labels or {},
                    description=incident.description,
                ):
                    yield _sse_event(token)
                    await asyncio.sleep(0)  # yield to event loop
            else:
                # Phase 1-3 fallback: generate full draft then emit at once
                draft = await rag.generate_draft(
                    alert_name=incident.alert_name,
                    labels=incident.labels or {},
                    description=incident.description,
                )
                yield _sse_event(draft)

            yield _sse_event("{}", event="done")

        except Exception as exc:
            log.error(
                "sse.stream_error",
                incident_id=incident_id,
                error=str(exc),
            )
            yield _sse_event(
                json.dumps({"detail": str(exc)}),
                event="error",
            )

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
