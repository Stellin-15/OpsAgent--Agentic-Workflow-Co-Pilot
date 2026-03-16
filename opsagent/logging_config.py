import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

# Context variable holds the correlation ID for the current request/task.
# Any code that calls bind_correlation_id() will have the ID injected into
# every subsequent log line for that request automatically.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Set a correlation ID on the current context. Returns the ID used."""
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    return cid


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output. Call once at startup."""
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # Configure the standard library logger so third-party libs
    # (SQLAlchemy, uvicorn, etc.) also go through structlog.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_int,
    )

    structlog.configure(
        processors=[
            # Merge any context variables (correlation_id, etc.) into the event dict.
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
