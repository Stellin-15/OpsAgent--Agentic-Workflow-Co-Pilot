"""Multi-model LLM router.

Strategy: try providers in priority order; on any exception fall through to
the next. If all fail, raises the last exception.

A/B testing: route 10% of traffic to Claude based on hashed incident_id so
the same incident always goes to the same model — deterministic, reproducible.

Usage:
    router = LLMRouter.from_settings(settings)
    response = await router.complete(messages, incident_id="01ABCDEF...")
    async for token in router.stream(messages, incident_id="01ABCDEF..."):
        ...
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, AsyncIterator

import structlog

from opsagent.ai.llm.base import LLMMessage, LLMProvider, LLMResponse

if TYPE_CHECKING:
    from opsagent.config import Settings

log = structlog.get_logger(__name__)


class LLMRouter:
    """Cascading fallback router over multiple LLM providers."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("LLMRouter requires at least one provider")
        self._providers = providers

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_settings(cls, settings: "Settings") -> "LLMRouter":
        """Build a router from application settings, skipping unconfigured providers."""
        providers: list[LLMProvider] = []

        if settings.google_api_key:
            from opsagent.ai.llm.gemini import GeminiProvider
            providers.append(GeminiProvider(api_key=settings.google_api_key))

        if settings.anthropic_api_key:
            from opsagent.ai.llm.anthropic import AnthropicProvider
            providers.append(AnthropicProvider(api_key=settings.anthropic_api_key))

        if settings.openai_api_key:
            from opsagent.ai.llm.openai import OpenAIProvider
            providers.append(OpenAIProvider(api_key=settings.openai_api_key))

        if not providers:
            raise RuntimeError(
                "LLMRouter: no API keys configured. "
                "Set GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )

        return cls(providers)

    # ------------------------------------------------------------------ #
    # Routing logic
    # ------------------------------------------------------------------ #

    def _select_provider(self, incident_id: str | None) -> LLMProvider:
        """
        A/B test: route 10% of traffic (by hash of incident_id) to the second
        provider if available. Otherwise always use the first.
        """
        if incident_id and len(self._providers) > 1:
            digest = int(hashlib.md5(incident_id.encode()).hexdigest(), 16)
            if digest % 10 == 0:  # 10% of traffic
                return self._providers[1]
        return self._providers[0]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        incident_id: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Try providers in order; fall through on exception."""
        primary = self._select_provider(incident_id)
        ordered = [primary] + [p for p in self._providers if p is not primary]

        last_exc: Exception = RuntimeError("No providers available")
        for provider in ordered:
            try:
                log.info(
                    "llm_router.attempt",
                    provider=provider.provider_name,
                    model=provider.model_id,
                )
                response = await provider.complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                log.info(
                    "llm_router.success",
                    provider=provider.provider_name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                return response
            except Exception as exc:
                log.warning(
                    "llm_router.provider_failed",
                    provider=provider.provider_name,
                    error=str(exc),
                )
                last_exc = exc

        raise last_exc

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        incident_id: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Stream tokens; fall through to next provider if streaming fails."""
        primary = self._select_provider(incident_id)
        ordered = [primary] + [p for p in self._providers if p is not primary]

        last_exc: Exception = RuntimeError("No providers available")
        for provider in ordered:
            try:
                log.info(
                    "llm_router.stream_attempt",
                    provider=provider.provider_name,
                    model=provider.model_id,
                )
                async for token in provider.stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    yield token
                return
            except Exception as exc:
                log.warning(
                    "llm_router.stream_provider_failed",
                    provider=provider.provider_name,
                    error=str(exc),
                )
                last_exc = exc

        raise last_exc
