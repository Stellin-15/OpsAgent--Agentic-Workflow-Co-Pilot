"""
Integration tests for alert webhook endpoints and the incident workflow.

Requires PostgreSQL at TEST_DATABASE_URL (default: localhost:5432/opsagent_test).
Celery tasks run synchronously (task_always_eager=True set in conftest.py).
"""

import pytest
from httpx import AsyncClient

ALERTMANAGER_PAYLOAD = {
    "version": "4",
    "groupKey": "{}:{alertname='HighCPUUsage'}",
    "status": "firing",
    "receiver": "opsagent",
    "groupLabels": {"alertname": "HighCPUUsage"},
    "commonLabels": {"alertname": "HighCPUUsage", "severity": "critical"},
    "commonAnnotations": {"summary": "CPU usage is above 90%"},
    "externalURL": "http://prometheus:9090",
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "HighCPUUsage",
                "instance": "web-01:9100",
                "severity": "critical",
            },
            "annotations": {
                "summary": "CPU usage is above 90%",
                "description": "Server web-01 has CPU usage of 95% for 10 minutes.",
            },
            "startsAt": "2026-03-17T10:00:00Z",
            "endsAt": "0001-01-01T00:00:00Z",
            "generatorURL": "http://prometheus:9090/graph",
            "fingerprint": "abc123",
        }
    ],
}


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_liveness(self, client: AsyncClient):
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_readiness_returns_checks(self, client: AsyncClient):
        resp = await client.get("/health/ready")
        assert resp.status_code in (200, 503)
        assert "checks" in resp.json()


class TestManualAlert:
    @pytest.mark.asyncio
    async def test_creates_incident_with_processing_then_draft_ready(self, client: AsyncClient):
        """
        With task_always_eager=True, the Celery task runs synchronously inside
        .delay(), so by the time the response arrives the incident is DRAFT_READY.
        """
        payload = {
            "alert_name": "TestHighCPU",
            "severity": "critical",
            "description": "CPU at 95% on web-01",
            "labels": {"instance": "web-01", "env": "production"},
        }
        resp = await client.post("/api/alerts/webhook/manual", json=payload)
        assert resp.status_code == 202

        data = resp.json()
        assert data["alert_name"] == "TestHighCPU"
        assert data["severity"] == "critical"
        # After eager task execution, status = DRAFT_READY
        incident_id = data["id"]

        # Verify via GET
        get_resp = await client.get(f"/api/incidents/{incident_id}")
        assert get_resp.status_code == 200
        detail = get_resp.json()
        assert detail["status"] == "DRAFT_READY"
        assert len(detail["drafts"]) == 1
        assert "Stub draft for TestHighCPU" in detail["drafts"][0]["content"]

    @pytest.mark.asyncio
    async def test_incident_has_ulid_id(self, client: AsyncClient):
        payload = {"alert_name": "ULIDTest", "severity": "warning"}
        resp = await client.post("/api/alerts/webhook/manual", json=payload)
        assert resp.status_code == 202
        incident_id = resp.json()["id"]
        assert len(incident_id) == 26
        assert incident_id.isalnum()

    @pytest.mark.asyncio
    async def test_labels_stored_correctly(self, client: AsyncClient):
        payload = {
            "alert_name": "LabelTest",
            "severity": "info",
            "labels": {"region": "us-east-1", "env": "prod"},
        }
        resp = await client.post("/api/alerts/webhook/manual", json=payload)
        incident_id = resp.json()["id"]
        detail = (await client.get(f"/api/incidents/{incident_id}")).json()
        assert detail["labels"]["region"] == "us-east-1"


class TestAlertManagerWebhook:
    @pytest.mark.asyncio
    async def test_alertmanager_creates_incident(self, client: AsyncClient):
        resp = await client.post(
            "/api/alerts/webhook/alertmanager", json=ALERTMANAGER_PAYLOAD
        )
        assert resp.status_code == 202
        incidents = resp.json()
        assert len(incidents) == 1
        assert incidents[0]["alert_name"] == "HighCPUUsage"
        assert incidents[0]["source"] == "alertmanager"

    @pytest.mark.asyncio
    async def test_resolved_alerts_are_skipped(self, client: AsyncClient):
        payload = {**ALERTMANAGER_PAYLOAD, "status": "resolved"}
        payload["alerts"] = [{**ALERTMANAGER_PAYLOAD["alerts"][0], "status": "resolved"}]
        resp = await client.post("/api/alerts/webhook/alertmanager", json=payload)
        assert resp.status_code == 202
        assert resp.json() == []


class TestGrafanaWebhook:
    @pytest.mark.asyncio
    async def test_grafana_v8_webhook(self, client: AsyncClient):
        payload = {
            "title": "High CPU Usage",
            "state": "alerting",
            "ruleName": "CPU Alert",
            "message": "CPU above threshold",
            "tags": {"severity": "critical"},
        }
        resp = await client.post("/api/alerts/webhook/grafana", json=payload)
        assert resp.status_code == 202
        assert resp.json()["source"] == "grafana"

    @pytest.mark.asyncio
    async def test_grafana_ok_state_skipped(self, client: AsyncClient):
        payload = {"title": "Resolved", "state": "ok", "ruleName": "CPU Alert"}
        resp = await client.post("/api/alerts/webhook/grafana", json=payload)
        assert resp.status_code == 202
        assert resp.json() is None


class TestIncidentWorkflow:
    @pytest.mark.asyncio
    async def test_full_approve_flow(self, client: AsyncClient):
        # 1. Create incident
        create = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "ApproveFlowTest", "severity": "warning"},
        )
        incident_id = create.json()["id"]

        # 2. Approve
        approve = await client.post(f"/api/incidents/{incident_id}/approve")
        assert approve.status_code == 200
        assert approve.json()["status"] == "RESOLVED"

    @pytest.mark.asyncio
    async def test_full_reject_flow(self, client: AsyncClient):
        create = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "RejectFlowTest", "severity": "info"},
        )
        incident_id = create.json()["id"]

        reject = await client.post(
            f"/api/incidents/{incident_id}/reject",
            json={"reason": "hallucination", "notes": "Wrong runbook cited"},
        )
        assert reject.status_code == 200
        assert reject.json()["status"] == "REJECTED"

    @pytest.mark.asyncio
    async def test_cannot_approve_already_resolved(self, client: AsyncClient):
        create = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "DoubleApproveTest", "severity": "warning"},
        )
        incident_id = create.json()["id"]

        await client.post(f"/api/incidents/{incident_id}/approve")
        second = await client.post(f"/api/incidents/{incident_id}/approve")
        assert second.status_code == 409

    @pytest.mark.asyncio
    async def test_list_incidents_with_status_filter(self, client: AsyncClient):
        resp = await client.get("/api/incidents?status_filter=DRAFT_READY")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        for item in data["items"]:
            assert item["status"] == "DRAFT_READY"

    @pytest.mark.asyncio
    async def test_404_for_unknown_incident(self, client: AsyncClient):
        resp = await client.get("/api/incidents/00000000000000000000000000")
        assert resp.status_code == 404
