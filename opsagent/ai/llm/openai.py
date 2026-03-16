"""OpenAI GPT-4o LLM provider.

Last in the fallback chain: Gemini → Claude → GPT-4o.
"""

from __future__ import annotations

from typing import AsyncIterator

import structlog

from opsagent.ai.llm.base import LLMMessage, LLMResponse

log = structlog.get_logger(__name__)


class OpenAIProvider:
    """Async wrapper around openai.AsyncOpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._model_id = model
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model_id,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        log.debug(
            "llm.openai.complete",
            model=self._model_id,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

        return LLMResponse(
            content=content,
            model=self._model_id,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            provider="openai",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self._model_id,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
