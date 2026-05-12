"""Merged LLM HTTP client.

Single client that talks to every LLM backend the project supports:

- `ollama`: native /api/generate path for Qwen3 models (works around the
  OpenAI-compat layer dropping the answer into a `reasoning` field). Other
  Ollama models go through /v1/chat/completions.
- `openai`: hosted OpenAI at https://api.openai.com/v1, Bearer auth.
- `openai_compatible` / `vllm`: any OpenAI-compatible /v1/chat/completions
  endpoint (vLLM, vLLM-MLX, Groq, LM Studio, etc.).
- `anthropic`: /v1/messages with x-api-key + anthropic-version headers.

Folded-in features:
- Auth header (`Authorization: Bearer …` for OpenAI-shaped, `x-api-key` for
  Anthropic).
- Retry with exponential backoff, gated by `LLMConnection.max_retries`.
- Circuit breaker per client instance (configurable threshold + recovery).
- Streaming via SSE for OpenAI-compat / Anthropic; NDJSON for Ollama native.
- Proxy mode (`/api/v3/translate`) for the legacy Translation Service path.
- Per-call sampling overrides via `LLMParameterOverrides`.
- `<think>` block extraction (post-hoc + streaming) via `qwen` helpers.
"""

from __future__ import annotations

import asyncio
import json as _json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from livetranslate_common.llm.qwen import (
    extract_from_reasoning,
    extract_translation_text,
    strip_think_blocks_streaming,
)
from livetranslate_common.logging import get_logger
from livetranslate_common.models.llm import (
    LLMConnection,
    LLMParameterOverrides,
)
from livetranslate_common.tracing import trace_request

logger = get_logger()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a request is fast-failed because the breaker is OPEN."""


# ---------------------------------------------------------------------------
# Circuit breaker (with injectable time source for tests)
# ---------------------------------------------------------------------------


class _CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        time_fn=time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._time_fn = time_fn
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_at: float = 0.0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if self._time_fn() - self._last_failure_at >= self._recovery_seconds:
                self._state = self.HALF_OPEN
        return self._state

    @property
    def is_available(self) -> bool:
        return self.state != self.OPEN

    def record_success(self) -> None:
        self._state = self.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_at = self._time_fn()
        if self._failure_count >= self._failure_threshold:
            self._state = self.OPEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sampling_dict(conn: LLMConnection, overrides: LLMParameterOverrides | None) -> dict[str, Any]:
    """Effective sampling params for a single call — base merged with overrides."""
    eff = conn.merge(overrides) if overrides else conn
    return {
        "temperature": eff.temperature,
        "max_tokens": eff.max_tokens,
        "top_p": eff.top_p,
        "top_k": eff.top_k,
        "repetition_penalty": eff.repetition_penalty,
        "presence_penalty": eff.presence_penalty,
        "model": overrides.model if overrides and overrides.model else conn.model,
        "timeout_s": overrides.timeout_s if overrides and overrides.timeout_s else conn.timeout_s,
    }


def _strip_v1_suffix(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1").rstrip("/")


# ---------------------------------------------------------------------------
# Merged client
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified HTTP client for every supported LLM backend.

    Construct with an immutable `LLMConnection`. Per-call sampling can be
    overridden via `LLMParameterOverrides` passed to `complete()`/`stream()`.

    The client is an async context manager — wrap usage in `async with` to
    ensure the underlying httpx connection pool is released.
    """

    def __init__(
        self,
        connection: LLMConnection,
        *,
        proxy_mode: bool = False,
        breaker_failure_threshold: int = 5,
        breaker_recovery_s: float = 30.0,
        time_fn=time.monotonic,
        enable_qwen_native: bool | None = None,
    ) -> None:
        self._connection = connection
        self._proxy_mode = proxy_mode
        self._breaker = _CircuitBreaker(
            failure_threshold=breaker_failure_threshold,
            recovery_seconds=breaker_recovery_s,
            time_fn=time_fn,
        )
        # Auto-detect Qwen3-on-Ollama unless caller forced it.
        self._use_ollama_native = (
            enable_qwen_native
            if enable_qwen_native is not None
            else connection.is_ollama_qwen3
        )
        # Connection-less httpx client; per-call timeout via `_with_timeout`.
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(connection.timeout_s))

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def connection(self) -> LLMConnection:
        return self._connection

    # -- public API ----------------------------------------------------------

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        overrides: LLMParameterOverrides | None = None,
    ) -> str:
        """Non-streaming completion. Returns the cleaned-up assistant text.

        Emits standardized tracing:
          - llm.request.start    (connection_id, engine, model, prompt_chars,
                                  stream=False)
          - llm.request.complete (+ duration_ms, response_chars)
          - llm.request.failed   (+ duration_ms, error_class, error)
        """
        if not self._breaker.is_available:
            raise CircuitBreakerOpenError(
                f"circuit breaker OPEN for {self._connection.base_url}"
            )

        sampling = _sampling_dict(self._connection, overrides)
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        with trace_request(
            "llm",
            "request",
            connection_id=self._connection.connection_id,
            engine=self._connection.engine,
            model=sampling.get("model", self._connection.model),
            prompt_chars=prompt_chars,
            stream=False,
        ) as t:
            # Engine + proxy dispatch
            if self._proxy_mode:
                result = await self._complete_proxy(messages, sampling)
            elif self._connection.engine == "anthropic":
                result = await self._complete_anthropic(messages, sampling)
            elif self._use_ollama_native:
                result = await self._complete_ollama_native(messages, sampling)
            else:
                result = await self._complete_openai_compat(messages, sampling)
            t.add(response_chars=len(result))
            return result

    async def stream(
        self,
        *,
        messages: list[dict[str, str]],
        overrides: LLMParameterOverrides | None = None,
    ) -> AsyncIterator[str]:
        """Streaming completion. Yields raw text deltas as they arrive.

        Emits the same llm.request.{start,complete,failed} events as
        complete() — duration_ms covers the time from request open to the
        last delta yielded (consumer-paced). response_chars on .complete is
        the sum of all delta bytes that crossed the boundary.
        """
        if not self._breaker.is_available:
            raise CircuitBreakerOpenError(
                f"circuit breaker OPEN for {self._connection.base_url}"
            )
        sampling = _sampling_dict(self._connection, overrides)
        prompt_chars = sum(len(m.get("content", "")) for m in messages)

        with trace_request(
            "llm",
            "request",
            connection_id=self._connection.connection_id,
            engine=self._connection.engine,
            model=sampling.get("model", self._connection.model),
            prompt_chars=prompt_chars,
            stream=True,
        ) as t:
            if self._proxy_mode:
                inner = self._stream_proxy(messages, sampling)
            elif self._connection.engine == "anthropic":
                inner = self._stream_anthropic(messages, sampling)
            elif self._use_ollama_native:
                inner = self._stream_ollama_native(messages, sampling)
            else:
                inner = self._stream_openai_compat(messages, sampling)

            response_chars = 0
            async for chunk in strip_think_blocks_streaming(inner):
                response_chars += len(chunk)
                yield chunk
            t.add(response_chars=response_chars)

    # -- backend: OpenAI-compatible /v1/chat/completions -------------------

    def _openai_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._connection.api_key:
            headers["Authorization"] = f"Bearer {self._connection.api_key}"
        return headers

    def _openai_body(
        self, messages: list[dict[str, str]], sampling: dict[str, Any], *, stream: bool
    ) -> dict[str, Any]:
        body = {
            "model": sampling["model"],
            "messages": messages,
            "temperature": sampling["temperature"],
            "max_tokens": sampling["max_tokens"],
            "top_p": sampling["top_p"],
            "top_k": sampling["top_k"],
            "repetition_penalty": sampling["repetition_penalty"],
            "presence_penalty": sampling["presence_penalty"],
            "stream": stream,
            # Disable Qwen3 thinking on vLLM/SGLang where supported.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return body

    def _openai_compat_url(self) -> str:
        """Resolve the chat-completions URL.

        For `ollama`, users typically configure http://host:11434 (no /v1
        suffix) because Ollama's native API lives at the root and the
        OpenAI-compat layer sits under /v1. For other engines, callers
        include /v1 in `base_url`. Normalize both shapes here.
        """
        base = self._connection.base_url.rstrip("/")
        if self._connection.engine == "ollama" and not base.endswith("/v1"):
            base = base + "/v1"
        return base + "/chat/completions"

    async def _complete_openai_compat(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> str:
        url = self._openai_compat_url()
        body = self._openai_body(messages, sampling, stream=False)
        data = await self._post_json(url, body, sampling["timeout_s"])
        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            # Qwen3-on-OpenAI-compat sometimes drops the answer into reasoning.
            if not content and "reasoning" in message:
                content = extract_from_reasoning(message["reasoning"])
            return extract_translation_text(content)
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected response shape: {e}") from e

    async def _stream_openai_compat(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> AsyncIterator[str]:
        url = self._openai_compat_url()
        body = self._openai_body(messages, sampling, stream=True)
        async for delta in self._sse_text_deltas(url, body, sampling["timeout_s"]):
            yield delta

    # -- backend: Anthropic /v1/messages ----------------------------------

    def _anthropic_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._connection.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _anthropic_body(
        self,
        messages: list[dict[str, str]],
        sampling: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        # Anthropic separates system message from messages array
        system_msg = ""
        user_msgs: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_msgs.append(m)
        body: dict[str, Any] = {
            "model": sampling["model"],
            "max_tokens": sampling["max_tokens"],
            "messages": user_msgs,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "stream": stream,
        }
        if system_msg:
            body["system"] = system_msg
        return body

    async def _complete_anthropic(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> str:
        url = self._connection.base_url.rstrip("/") + "/v1/messages"
        # If base_url already ends with /v1, strip + re-add to avoid /v1/v1/messages
        if url.endswith("/v1/v1/messages"):
            url = url.replace("/v1/v1/messages", "/v1/messages")
        body = self._anthropic_body(messages, sampling, stream=False)
        data = await self._post_json(
            url, body, sampling["timeout_s"], headers=self._anthropic_headers()
        )
        try:
            parts = data.get("content") or []
            text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
            return extract_translation_text(text)
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected anthropic response: {e}") from e

    async def _stream_anthropic(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> AsyncIterator[str]:
        url = self._connection.base_url.rstrip("/") + "/v1/messages"
        if url.endswith("/v1/v1/messages"):
            url = url.replace("/v1/v1/messages", "/v1/messages")
        body = self._anthropic_body(messages, sampling, stream=True)
        try:
            async with self._http.stream(
                "POST",
                url,
                json=body,
                headers=self._anthropic_headers(),
                timeout=sampling["timeout_s"],
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    try:
                        evt = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
            self._breaker.record_success()
        except httpx.HTTPError as e:
            self._breaker.record_failure()
            raise RuntimeError(f"anthropic stream failed: {e}") from e

    # -- backend: Ollama native /api/generate -----------------------------

    def _ollama_native_url(self) -> str:
        return _strip_v1_suffix(self._connection.base_url) + "/api/generate"

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _ollama_native_body(
        self, messages: list[dict[str, str]], sampling: dict[str, Any], *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": sampling["model"],
            "prompt": self._messages_to_prompt(messages),
            "stream": stream,
            "options": {
                "temperature": sampling["temperature"],
                "num_predict": sampling["max_tokens"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "repeat_penalty": sampling["repetition_penalty"],
            },
        }

    async def _complete_ollama_native(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> str:
        url = self._ollama_native_url()
        body = self._ollama_native_body(messages, sampling, stream=False)
        data = await self._post_json(url, body, sampling["timeout_s"])
        response = data.get("response", "")
        if not response and "thinking" in data:
            response = extract_from_reasoning(data["thinking"])
        return extract_translation_text(response)

    async def _stream_ollama_native(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> AsyncIterator[str]:
        url = self._ollama_native_url()
        body = self._ollama_native_body(messages, sampling, stream=True)
        try:
            async with self._http.stream(
                "POST", url, json=body, timeout=sampling["timeout_s"]
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    delta = chunk.get("response", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break
            self._breaker.record_success()
        except httpx.HTTPError as e:
            self._breaker.record_failure()
            raise RuntimeError(f"ollama native stream failed: {e}") from e

    # -- backend: proxy mode (Translation Service V3) ---------------------

    async def _complete_proxy(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> str:
        url = self._connection.base_url.rstrip("/") + "/api/v3/translate"
        body = {
            "messages": messages,
            "model": sampling["model"],
            "temperature": sampling["temperature"],
            "max_tokens": sampling["max_tokens"],
        }
        data = await self._post_json(url, body, sampling["timeout_s"])
        return extract_translation_text(
            data.get("translated_text") or data.get("text") or ""
        )

    async def _stream_proxy(
        self, messages: list[dict[str, str]], sampling: dict[str, Any]
    ) -> AsyncIterator[str]:
        url = self._connection.base_url.rstrip("/") + "/api/v3/translate/stream"
        body = {
            "messages": messages,
            "model": sampling["model"],
            "temperature": sampling["temperature"],
            "max_tokens": sampling["max_tokens"],
        }
        async for delta in self._sse_text_deltas(url, body, sampling["timeout_s"]):
            yield delta

    # -- shared low-level helpers ----------------------------------------

    async def _post_json(
        self,
        url: str,
        body: dict[str, Any],
        timeout_s: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST with retry + breaker bookkeeping. Returns parsed JSON."""
        eff_headers = headers if headers is not None else self._openai_headers()
        attempts = self._connection.max_retries + 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            if not self._breaker.is_available:
                raise CircuitBreakerOpenError(
                    f"circuit breaker OPEN for {url}"
                )
            try:
                resp = await self._http.post(url, json=body, headers=eff_headers, timeout=timeout_s)
                resp.raise_for_status()
                self._breaker.record_success()
                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                self._breaker.record_failure()
                if attempt + 1 < attempts:
                    backoff = 0.2 * (2 ** attempt)
                    logger.warning(
                        "llm_post_retry",
                        url=url,
                        attempt=attempt + 1,
                        error=str(e),
                        backoff_s=backoff,
                    )
                    await asyncio.sleep(backoff)
        raise RuntimeError(f"LLM POST failed after {attempts} attempts: {last_exc}")

    async def _sse_text_deltas(
        self, url: str, body: dict[str, Any], timeout_s: float
    ) -> AsyncIterator[str]:
        """Parse server-sent-events from an OpenAI-compatible streaming endpoint."""
        try:
            async with self._http.stream(
                "POST",
                url,
                json=body,
                headers=self._openai_headers(),
                timeout=timeout_s,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(payload)
                    except _json.JSONDecodeError:
                        continue
                    try:
                        delta = chunk["choices"][0]["delta"].get("content", "")
                    except (KeyError, IndexError, TypeError):
                        continue
                    if delta:
                        yield delta
            self._breaker.record_success()
        except httpx.HTTPError as e:
            self._breaker.record_failure()
            raise RuntimeError(f"sse stream failed: {e}") from e
