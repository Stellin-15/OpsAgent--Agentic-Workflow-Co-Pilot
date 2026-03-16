"""Base protocol for LLM providers.

Every provider (Gemini, Claude, GPT-4o) implements the same async interface.
The router chains them: Gemini → Claude → GPT-4o → raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Every LLM provider must implement these two methods."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]: ...
