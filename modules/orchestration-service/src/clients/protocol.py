"""
LLM Client Protocol -- the contract all LLM clients must satisfy.

Defines the minimal interface that any LLM client (direct, proxy, etc.)
must implement. Consumers should type-hint against LLMClientProtocol
rather than concrete implementations.
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from clients.models import PromptTranslationResult, StreamChunk


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations.

    All LLM clients (direct, proxy, etc.) must implement this interface.
    """

    async def translate_prompt(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> PromptTranslationResult: ...

    async def translate_prompt_stream(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def connect(self) -> bool: ...

    async def close(self) -> None: ...

    async def health_check(self) -> bool: ...
