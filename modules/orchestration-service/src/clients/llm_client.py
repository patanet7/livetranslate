"""Compatibility shim — `clients.llm_client` is superseded by
`livetranslate_common.llm.client.LLMClient`.

This file used to contain a ~600-line LLMClient implementation that
duplicated logic from `translation/llm_client.py`. Both have been
consolidated into the merged client in `livetranslate_common.llm.client`.

To keep the 5 existing production call sites working without a follow-up
refactor in the same PR (per the consolidation plan), we expose:

- `LLMClient(base_url=..., api_key=..., model=..., proxy_mode=...)` — old
  constructor signature, built on top of the new merged client.
- `create_llm_client(...)` — factory with the same kwargs.

New code MUST import from `livetranslate_common.llm.client` and pass an
`LLMConnection` value object instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from clients.models import PromptTranslationResult, StreamChunk
from livetranslate_common.llm.client import LLMClient as _MergedClient
from livetranslate_common.logging import get_logger
from livetranslate_common.models.llm import LLMConnection

logger = get_logger()


def _infer_engine(base_url: str, default_backend: str = "openai_compatible") -> str:
    lowered = base_url.lower()
    if "api.openai.com" in lowered:
        return "openai"
    if "api.anthropic.com" in lowered:
        return "anthropic"
    if ":11434" in lowered:
        return "ollama"
    if default_backend == "ollama":
        return "ollama"
    return "openai_compatible"


def _normalize_base_url(base_url: str, engine: str) -> str:
    base_url = base_url.rstrip("/")
    if engine in ("openai", "openai_compatible", "vllm") and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


class LLMClient:
    """Legacy adapter over the merged `livetranslate_common.llm.client.LLMClient`.

    Accepts the old (base_url, api_key, model, proxy_mode) kwargs and
    builds an `LLMConnection` internally. Implements only the subset of the
    old protocol still used by production callers (translate_prompt and the
    bare chat surface).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "qwen2.5:3b",
        timeout: float = 120.0,
        default_max_tokens: int = 1024,
        default_temperature: float = 0.3,
        default_backend: str = "openai_compatible",
        proxy_mode: bool = False,
    ) -> None:
        engine = _infer_engine(base_url, default_backend=default_backend)
        normalized = _normalize_base_url(base_url, engine)
        self._connection = LLMConnection(
            engine=engine,  # type: ignore[arg-type]
            base_url=normalized,
            api_key=api_key,
            model=model,
            temperature=default_temperature,
            max_tokens=default_max_tokens,
            timeout_s=timeout,
        )
        self.proxy_mode = proxy_mode
        self._inner: _MergedClient | None = None
        self.base_url = self._connection.base_url
        self.model = self._connection.model
        self.api_key = api_key

    async def connect(self) -> bool:
        if self._inner is None:
            self._inner = _MergedClient(self._connection, proxy_mode=self.proxy_mode)
        return True

    async def close(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
            self._inner = None

    def _client(self) -> _MergedClient:
        if self._inner is None:
            self._inner = _MergedClient(self._connection, proxy_mode=self.proxy_mode)
        return self._inner

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> PromptTranslationResult:
        import time
        from livetranslate_common.models.llm import LLMParameterOverrides
        overrides_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            overrides_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            overrides_kwargs["temperature"] = temperature
        if model is not None:
            overrides_kwargs["model"] = model
        overrides = LLMParameterOverrides(**overrides_kwargs) if overrides_kwargs else None
        start = time.monotonic()
        text = await self._client().complete(messages=messages, overrides=overrides)
        return PromptTranslationResult(
            text=text,
            processing_time_ms=(time.monotonic() - start) * 1000,
            backend_used=self._connection.engine,
            model_used=model or self._connection.model,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        from livetranslate_common.models.llm import LLMParameterOverrides
        overrides_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            overrides_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            overrides_kwargs["temperature"] = temperature
        if model is not None:
            overrides_kwargs["model"] = model
        overrides = LLMParameterOverrides(**overrides_kwargs) if overrides_kwargs else None
        async for delta in self._client().stream(messages=messages, overrides=overrides):
            yield StreamChunk(chunk=delta, done=False)
        yield StreamChunk(done=True)

    async def translate_prompt(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> PromptTranslationResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, max_tokens=max_tokens, temperature=temperature)

    async def translate_prompt_stream(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async for chunk in self.chat_stream(
            messages, max_tokens=max_tokens, temperature=temperature
        ):
            yield chunk

    async def health_check(self) -> bool:
        # Stubbed — callers handle their own probing.
        return True


def create_llm_client(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "",
    model: str = "qwen2.5:3b",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    default_backend: str = "openai_compatible",
    proxy_mode: bool = False,
    timeout: float = 120.0,
) -> LLMClient:
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        default_max_tokens=max_tokens,
        default_temperature=temperature,
        default_backend=default_backend,
        proxy_mode=proxy_mode,
        timeout=timeout,
    )
