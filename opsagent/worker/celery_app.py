"""
Celery application factory.

This module is imported by:
  - The API (to enqueue tasks)
  - The Celery worker process (to execute tasks)
  - Celery Beat (to schedule periodic tasks)

Worker initialisation:
  When a worker process starts, the worker_init signal fires.
  We use it to initialise the database connection pool and the RAG service
  so every task has them available without re-initialising per task.
"""

import asyncio
import logging

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_init

log = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    from opsagent.config import get_settings

    settings = get_settings()

    app = Celery(
        "opsagent",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=[
            "opsagent.worker.tasks.incident_tasks",
            "opsagent.worker.tasks.indexing_tasks",
        ],
    )

    app.conf.update(
        # Serialisation
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Reliability
        task_acks_late=True,          # ACK only after task completes (not when received)
        task_reject_on_worker_lost=True,
        task_track_started=True,
        # Timeouts
        task_soft_time_limit=120,     # raises SoftTimeLimitExceeded after 2 min
        task_time_limit=180,          # kills task after 3 min
        # Beat schedule
        beat_schedule={
            "rebuild-runbook-index-daily": {
                "task": "opsagent.worker.tasks.indexing_tasks.rebuild_runbook_index",
                "schedule": crontab(hour=2, minute=0),  # 2am UTC daily
            }
        },
        timezone="UTC",
        enable_utc=True,
    )

    return app


celery_app = create_celery_app()


# ── Worker lifecycle ───────────────────────────────────────────────────────────

@worker_init.connect
def on_worker_init(**kwargs):
    """
    Runs once when each worker process starts.
    Initialises the DB pool and RAG service so tasks don't pay this cost each time.
    """
    from opsagent.config import get_settings
    from opsagent.database import init_db
    from opsagent.services.rag_service import RagService
    from opsagent.worker.state import set_worker_rag

    settings = get_settings()
    log.info("Worker initialising database and RAG service...")

    init_db(settings.database_url)

    rag = RagService(settings)
    asyncio.run(rag.initialize())
    set_worker_rag(rag)

    log.info("Worker ready.")
