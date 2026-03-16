"""OpenTelemetry distributed tracing setup (Phase 5).

A single alert creates a trace spanning:
    webhook_receive → db_write → celery_task → embedding_cache_check →
    retrieval → llm_call → draft_stored → slack_notify

Configure via environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    OTEL_SERVICE_NAME=opsagent-api
    OTEL_TRACES_SAMPLER=parentbased_traceidratio
    OTEL_TRACES_SAMPLER_ARG=1.0   # 100% in dev, lower in prod

Falls back silently if opentelemetry is not installed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import structlog

log = structlog.get_logger(__name__)


def setup_tracing(service_name: str = "opsagent") -> bool:
    """
    Configure OpenTelemetry OTLP exporter.

    Returns True if tracing was successfully configured, False otherwise.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not endpoint:
            log.info("tracing.disabled", reason="OTEL_EXPORTER_OTLP_ENDPOINT not set")
            return False

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        log.info("tracing.initialized", endpoint=endpoint, service=service_name)
        return True

    except ImportError:
        log.info("tracing.disabled", reason="opentelemetry packages not installed")
        return False
    except Exception as exc:
        log.warning("tracing.setup_failed", error=str(exc))
        return False


def get_tracer(name: str = "opsagent"):
    """Return an OpenTelemetry tracer. Returns a no-op if OTel not installed."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


@contextmanager
def span(name: str, attributes: dict | None = None) -> Generator:
    """Convenience context manager for creating a span."""
    tracer = get_tracer()
    try:
        with tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, str(v))
            yield s
    except AttributeError:
        # NoOpTracer doesn't support context manager — just yield
        yield None


class _NoOpTracer:
    """Returned when OpenTelemetry is not installed."""

    def start_as_current_span(self, name: str, **kwargs):
        from contextlib import nullcontext
        return nullcontext()

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    def set_attribute(self, key: str, value) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
