"""Anthropic Claude LLM provider.

Uses the anthropic SDK directly. Falls back in the router chain after Gemini.
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from opsagent.ai.llm.base import LLMMessage, LLMResponse

log = structlog.get_logger(__name__)


class AnthropicProvider:
    """Async wrapper around anthropic.AsyncAnthropic."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._api_key = api_key
        self._model_id = model
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    def _extract_system_and_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str, list[dict]]:
        system = ""
        chat_messages: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})
        return system, chat_messages

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        client = self._get_client()
        system, chat_messages = self._extract_system_and_messages(messages)

        response = await client.messages.create(
            model=self._model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=chat_messages,
        )

        content = response.content[0].text if response.content else ""
        log.debug(
            "llm.anthropic.complete",
            model=self._model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self._model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider="anthropic",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        system, chat_messages = self._extract_system_and_messages(messages)

        async with client.messages.stream(
            model=self._model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
