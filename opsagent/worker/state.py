"""
Worker-process singletons.

The FastAPI app stores services in app.state (per-process, per-request lifetime).
Celery workers have no FastAPI app, so they use this module instead.

Initialised once via the worker_init Celery signal in celery_app.py.
"""

from opsagent.services.rag_service import RagService

_rag: RagService | None = None


def get_worker_rag() -> RagService:
    if _rag is None:
        raise RuntimeError(
            "Worker RAG service not initialised. "
            "Make sure init_db() and RagService.initialize() ran in worker_init."
        )
    return _rag


def set_worker_rag(rag: RagService) -> None:
    global _rag
    _rag = rag
