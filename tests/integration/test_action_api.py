"""Integration tests for the action catalog + execution API (Phase 3).

These tests use the real FastAPI client but stub out subprocess execution so
we don't need kubectl or aws CLI installed.
"""

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


class TestActionCatalogAPI:
    @pytest.mark.asyncio
    async def test_list_actions(self, client: AsyncClient) -> None:
        resp = await client.get("/api/actions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # The test client loads actions/ from the project root
        # In CI the directory may be empty — just assert the endpoint works
        assert all("name" in a for a in data)

    @pytest.mark.asyncio
    async def test_list_actions_with_label_filter(self, client: AsyncClient) -> None:
        resp = await client.get("/api/actions?label=kubernetes")
        assert resp.status_code == 200
        data = resp.json()
        for action in data:
            assert "kubernetes" in action["labels"]

    @pytest.mark.asyncio
    async def test_get_unknown_action_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/actions/nonexistent_action_xyz")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_known_action(self, client: AsyncClient) -> None:
        # First list to find a real action name
        list_resp = await client.get("/api/actions")
        actions = list_resp.json()
        if not actions:
            pytest.skip("No actions in catalog — skipping")

        name = actions[0]["name"]
        resp = await client.get(f"/api/actions/{name}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == name
        assert "safety_level" in data
        assert "parameters" in data


class TestActionDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_low_safety_action(self, client: AsyncClient) -> None:
        # Find a low-safety, no-approval action
        list_resp = await client.get("/api/actions?label=diagnose")
        actions = list_resp.json()
        if not actions:
            pytest.skip("No diagnose actions in catalog")

        action = actions[0]

        # Build minimal params
        params = {p["name"]: "test-value" for p in action["parameters"] if p["required"]}

        # Create a real incident first
        inc_resp = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "DryRunTest", "severity": "warning"},
        )
        incident_id = inc_resp.json()["id"]

        with patch(
            "opsagent.execution.safe_executor.SafeExecutor._run_command",
            new_callable=AsyncMock,
        ) as mock_run:
            from opsagent.execution.base import ExecutionResult
            mock_run.return_value = ExecutionResult(
                action_name=action["name"],
                command="mock command",
                exit_code=0,
                stdout="mock output",
                stderr="",
            )

            resp = await client.post(
                f"/api/actions/{action['name']}/dry-run",
                json={
                    "incident_id": incident_id,
                    "params": params,
                    "dry_run": True,
                    "approved_by": "test",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["action_name"] == action["name"]
        assert "exit_code" in data
        assert "succeeded" in data

    @pytest.mark.asyncio
    async def test_execute_requires_approval_for_high_safety(self, client: AsyncClient) -> None:
        # Find an action that requires_approval
        list_resp = await client.get("/api/actions")
        actions = [a for a in list_resp.json() if a["requires_approval"]]
        if not actions:
            pytest.skip("No approval-required actions in catalog")

        action = actions[0]
        inc_resp = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "ApprovalTest", "severity": "info"},
        )
        incident_id = inc_resp.json()["id"]

        # Execute without proper approved_by → should return 422
        resp = await client.post(
            f"/api/actions/{action['name']}/execute",
            json={
                "incident_id": incident_id,
                "params": {},
                "approved_by": "api",  # default value = not approved
            },
        )
        assert resp.status_code == 422
