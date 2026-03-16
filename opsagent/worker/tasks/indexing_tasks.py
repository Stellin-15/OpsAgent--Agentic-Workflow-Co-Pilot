"""
Celery task: rebuild_runbook_index

Scheduled via Celery Beat to run at 2am UTC every day.
Rebuilds the FAISS / pgvector index from the runbooks directory so newly
added runbooks are picked up without restarting the API or workers.

In Phase 4 this task will write embeddings to pgvector instead of FAISS.
"""

import asyncio
import logging

from opsagent.worker.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(
    name="opsagent.tasks.rebuild_runbook_index",
    max_retries=1,
)
def rebuild_runbook_index() -> dict:
    """
    Re-initialise the RAG service to pick up new or changed runbooks.
    Called daily by Celery Beat (see celery_app.py beat_schedule).
    Can also be triggered manually:

        celery -A opsagent.worker.celery_app.celery_app call \\
            opsagent.tasks.rebuild_runbook_index
    """
    return asyncio.run(_rebuild())


async def _rebuild() -> dict:
    from opsagent.config import get_settings
    from opsagent.services.rag_service import RagService
    from opsagent.worker.state import set_worker_rag

    settings = get_settings()
    log.info("rebuild_runbook_index: starting")

    rag = RagService(settings)
    await rag.initialize()
    set_worker_rag(rag)

    log.info("rebuild_runbook_index: complete")
    return {"status": "ok", "ready": rag.is_ready()}
