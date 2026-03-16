"""Google Gemini LLM provider.

Uses google-generativeai SDK directly (not via LangChain) for more control
over streaming and token counting.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import structlog

from opsagent.ai.llm.base import LLMMessage, LLMResponse

log = structlog.get_logger(__name__)


class GeminiProvider:
    """Async wrapper around google.generativeai."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
    ) -> None:
        self._api_key = api_key
        self._model_id = model
        self._client = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_id)
        return self._client

    def _build_prompt(self, messages: list[LLMMessage]) -> str:
        """Flatten message list into a single prompt string for Gemini."""
        parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"[System]\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"[User]\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"[Assistant]\n{msg.content}")
        return "\n\n".join(parts)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        client = self._get_client()
        prompt = self._build_prompt(messages)

        import google.generativeai as genai
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Run in thread pool to avoid blocking the event loop
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            generation_config=generation_config,
        )

        content = response.text or ""
        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        log.debug(
            "llm.gemini.complete",
            model=self._model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self._model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider="gemini",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        import google.generativeai as genai

        client = self._get_client()
        prompt = self._build_prompt(messages)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Gemini streaming is synchronous — wrap in a queue for async iteration.
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_stream() -> None:
            try:
                for chunk in client.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True,
                ):
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _run_stream)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
