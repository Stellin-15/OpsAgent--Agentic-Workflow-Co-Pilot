"""Unit tests for the multi-model LLM router (Phase 4)."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from opsagent.ai.llm.base import LLMMessage, LLMResponse
from opsagent.ai.llm.router import LLMRouter


def _make_provider(name: str, model: str = "test-model", *, fail: bool = False):
    """Build a mock LLM provider."""
    provider = MagicMock()
    provider.provider_name = name
    provider.model_id = model

    if fail:
        provider.complete = AsyncMock(side_effect=RuntimeError(f"{name} is down"))
        async def _failing_stream(*args, **kwargs):
            raise RuntimeError(f"{name} stream is down")
            yield  # make it a generator
        provider.stream = _failing_stream
    else:
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content=f"Response from {name}",
                model=model,
                input_tokens=10,
                output_tokens=20,
                provider=name,
            )
        )
        async def _ok_stream(*args, **kwargs):
            yield f"token-from-{name}"
        provider.stream = _ok_stream

    return provider


class TestLLMRouter:
    def test_requires_at_least_one_provider(self) -> None:
        with pytest.raises(ValueError):
            LLMRouter([])

    @pytest.mark.asyncio
    async def test_complete_uses_first_provider(self) -> None:
        p1 = _make_provider("gemini")
        p2 = _make_provider("claude")
        router = LLMRouter([p1, p2])

        messages = [LLMMessage(role="user", content="hello")]
        response = await router.complete(messages, incident_id="fixed-id")

        # Default routing — no A/B — should use p1
        assert response.provider == "gemini"

    @pytest.mark.asyncio
    async def test_falls_back_on_provider_failure(self) -> None:
        p1 = _make_provider("gemini", fail=True)
        p2 = _make_provider("claude")
        router = LLMRouter([p1, p2])

        messages = [LLMMessage(role="user", content="hello")]
        response = await router.complete(messages)

        assert response.provider == "claude"

    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self) -> None:
        p1 = _make_provider("gemini", fail=True)
        p2 = _make_provider("claude", fail=True)
        router = LLMRouter([p1, p2])

        with pytest.raises(RuntimeError):
            await router.complete([LLMMessage(role="user", content="hi")])

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self) -> None:
        p1 = _make_provider("gemini")
        router = LLMRouter([p1])

        tokens = []
        async for token in router.stream([LLMMessage(role="user", content="hi")]):
            tokens.append(token)

        assert tokens == ["token-from-gemini"]

    def test_ab_routing_deterministic(self) -> None:
        """10% of incidents (by hash) should route to the second provider."""
        p1 = _make_provider("gemini")
        p2 = _make_provider("claude")
        router = LLMRouter([p1, p2])

        routed_to_second = sum(
            1
            for i in range(100)
            if router._select_provider(f"incident-{i}") is p2
        )
        # Should be approximately 10 (within tolerance)
        assert 5 <= routed_to_second <= 20


class TestLLMMessage:
    def test_message_fields(self) -> None:
        msg = LLMMessage(role="system", content="You are an SRE assistant")
        assert msg.role == "system"
        assert "SRE" in msg.content
