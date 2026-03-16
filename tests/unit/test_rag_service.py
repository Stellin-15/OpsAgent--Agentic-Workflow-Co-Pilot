"""
Unit tests for RagService.

These tests do NOT hit the database or make real LLM API calls.
They test the logic of the service in isolation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opsagent.config import Settings
from opsagent.services.rag_service import RagService


def make_settings(**overrides) -> Settings:
    defaults = {
        "database_url": "postgresql+asyncpg://x:x@localhost/x",
        "google_api_key": "test-key",
        "runbooks_path": "/nonexistent",
        "rag_chunk_size": 1000,
        "rag_chunk_overlap": 200,
        "rag_retrieval_k": 3,
        "environment": "test",
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestRagServiceInit:
    def test_not_ready_before_initialize(self):
        svc = RagService(make_settings())
        assert not svc.is_ready()

    @pytest.mark.asyncio
    async def test_initialize_with_missing_path(self, tmp_path):
        """When runbooks path doesn't exist, service stays not-ready but doesn't raise."""
        svc = RagService(make_settings(runbooks_path=str(tmp_path / "missing")))
        await svc.initialize()
        assert not svc.is_ready()

    @pytest.mark.asyncio
    async def test_initialize_with_empty_directory(self, tmp_path):
        """Empty runbooks directory → service not ready, no crash."""
        svc = RagService(make_settings(runbooks_path=str(tmp_path)))
        await svc.initialize()
        assert not svc.is_ready()


class TestRagServiceGenerateDraft:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_not_ready(self):
        svc = RagService(make_settings())
        # Don't call initialize() — chain stays None
        draft = await svc.generate_draft(
            alert_name="TestAlert",
            labels={"instance": "web-01"},
            description="CPU is high",
        )
        assert "not initialised" in draft or "RAG chain" in draft

    @pytest.mark.asyncio
    async def test_generates_draft_when_chain_ready(self):
        svc = RagService(make_settings())
        # Manually inject a mock chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "1. Check CPU usage\n2. Restart service"
        svc._chain = mock_chain

        draft = await svc.generate_draft(
            alert_name="HighCPU",
            labels={"severity": "critical"},
            description="CPU at 95%",
        )

        assert "Check CPU" in draft
        mock_chain.invoke.assert_called_once()
        call_arg = mock_chain.invoke.call_args[0][0]
        assert "HighCPU" in call_arg

    @pytest.mark.asyncio
    async def test_query_includes_alert_name_labels_description(self):
        svc = RagService(make_settings())
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "steps"
        svc._chain = mock_chain

        await svc.generate_draft(
            alert_name="DiskFull",
            labels={"host": "db-01", "severity": "critical"},
            description="Disk at 99%",
        )

        query = mock_chain.invoke.call_args[0][0]
        assert "DiskFull" in query
        assert "db-01" in query
        assert "Disk at 99%" in query
