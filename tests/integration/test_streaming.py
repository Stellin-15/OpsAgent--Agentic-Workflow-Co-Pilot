"""Integration tests for the SSE draft streaming endpoint (Phase 4)."""

import pytest
from httpx import AsyncClient


class TestStreamingEndpoint:
    @pytest.mark.asyncio
    async def test_stream_unknown_incident_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/incidents/00000000000000000000000000/stream")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_returns_sse_content_type(self, client: AsyncClient) -> None:
        # Create an incident first
        inc = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "StreamTest", "severity": "warning"},
        )
        incident_id = inc.json()["id"]

        # The stub RAG is not_ready=False (it is ready — StubRag.is_ready() → True)
        # So streaming should return SSE tokens
        async with client.stream("GET", f"/api/incidents/{incident_id}/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Read a limited amount of the stream
            chunks = []
            async for chunk in resp.aiter_text():
                chunks.append(chunk)
                if len(chunks) >= 5:
                    break

        # Should have received at least one SSE data line or done event
        full = "".join(chunks)
        assert "data:" in full or "event:" in full

    @pytest.mark.asyncio
    async def test_stream_emits_done_event(self, client: AsyncClient) -> None:
        inc = await client.post(
            "/api/alerts/webhook/manual",
            json={"alert_name": "StreamDoneTest", "severity": "info"},
        )
        incident_id = inc.json()["id"]

        # Collect full SSE stream
        full_content = ""
        async with client.stream("GET", f"/api/incidents/{incident_id}/stream") as resp:
            async for chunk in resp.aiter_text():
                full_content += chunk
                if "event: done" in full_content or len(full_content) > 5000:
                    break

        assert "event: done" in full_content
