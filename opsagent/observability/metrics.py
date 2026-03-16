"""Prometheus custom metrics for OpsAgent (Phase 5).

All metrics follow the RED method (Rate, Errors, Duration) and map directly
to the Grafana dashboard panels defined in grafana/dashboards/opsagent.json.

Metrics registered here:
    incident_alerts_total           Counter  — alerts received by source/severity
    incident_resolution_seconds     Histogram — FIRING → RESOLVED latency
    incident_status_transitions_total Counter — every state machine transition
    llm_tokens_total                Counter  — by provider, model, direction
    llm_call_duration_seconds       Histogram — per LLM provider
    rag_retrieval_duration_seconds  Histogram — retrieval step only
    action_executions_total         Counter  — by action_type, exit_code bucket
    draft_decisions_total           Counter  — approved vs rejected
    active_incidents_gauge          Gauge    — incidents currently in-flight

Usage:
    from opsagent.observability.metrics import (
        incident_alerts_total,
        llm_call_duration_seconds,
        ...
    )
    incident_alerts_total.labels(source="alertmanager", severity="critical").inc()

    with llm_call_duration_seconds.labels(provider="gemini").time():
        response = await llm.complete(...)
"""

from prometheus_client import Counter, Gauge, Histogram

# ─────────────────────────────────────────── alert ingestion ─────────────────

incident_alerts_total = Counter(
    "opsagent_incident_alerts_total",
    "Total alerts received",
    labelnames=["source", "severity"],
)

incident_status_transitions_total = Counter(
    "opsagent_incident_status_transitions_total",
    "Incident state machine transitions",
    labelnames=["from_status", "to_status"],
)

active_incidents_gauge = Gauge(
    "opsagent_active_incidents",
    "Incidents currently in FIRING or PROCESSING state",
)

# ─────────────────────────────────────────── resolution time ─────────────────

incident_resolution_seconds = Histogram(
    "opsagent_incident_resolution_seconds",
    "Time from FIRING to RESOLVED (seconds)",
    buckets=[30, 60, 120, 300, 600, 1800, 3600, 7200],
)

# ─────────────────────────────────────────── LLM pipeline ────────────────────

llm_tokens_total = Counter(
    "opsagent_llm_tokens_total",
    "LLM tokens consumed",
    labelnames=["provider", "model", "direction"],  # direction: input | output
)

llm_call_duration_seconds = Histogram(
    "opsagent_llm_call_duration_seconds",
    "End-to-end LLM call latency",
    labelnames=["provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

rag_retrieval_duration_seconds = Histogram(
    "opsagent_rag_retrieval_duration_seconds",
    "pgvector + BM25 retrieval latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ─────────────────────────────────────────── execution engine ────────────────

action_executions_total = Counter(
    "opsagent_action_executions_total",
    "Actions executed by the safe execution engine",
    labelnames=["action_name", "safety_level", "outcome"],  # outcome: success|failure|timeout|rollback
)

# ─────────────────────────────────────────── draft quality ───────────────────

draft_decisions_total = Counter(
    "opsagent_draft_decisions_total",
    "Draft approve/reject decisions",
    labelnames=["decision", "reason"],  # decision: approved|rejected; reason: from RejectBody
)

draft_ragas_confidence = Histogram(
    "opsagent_draft_ragas_confidence",
    "RAGAS composite confidence score per draft",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
