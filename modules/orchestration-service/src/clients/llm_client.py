"""
Unified LLM Client

Single client for all LLM interactions. Supports two modes:

- **Direct mode** (proxy_mode=False): Talks directly to any OpenAI-compatible
  /v1/chat/completions endpoint (Ollama, OpenAI, Groq, etc.).

- **Proxy mode** (proxy_mode=True): Talks to the Translation Service V3 API
  at /api/v3/translate, which forwards prompts to an LLM backend.

Both modes expose the same LLMClientProtocol interface, plus direct mode
adds native multi-turn chat() and chat_stream() methods.

Replaces both the former SimpleTranslationClient and DirectLLMClient.
"""

import json
import logging
import time
from collections.abc import AsyncIterator

import aiohttp
from clients.models import CircuitBreaker, PromptTranslationResult, StreamChunk

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client with direct and proxy modes.

    Direct mode (default):
        Sends requests to OpenAI-compatible /v1/chat/completions.
        Supports native multi-turn chat via messages array.

    Proxy mode:
        Sends requests to Translation Service V3 API (/api/v3/translate).
        Suitable when the Translation Service acts as LLM gateway.

    Usage (direct):
        client = LLMClient(
            base_url="http://localhost:11434/v1",
            model="gemma3:4b",
        )
        await client.connect()
        result = await client.chat([
            {"role": "system", "content": "You are a meeting analyst."},
            {"role": "user", "content": "What were the action items?"},
        ])

    Usage (proxy):
        client = LLMClient(
            base_url="http://localhost:5003",
            proxy_mode=True,
        )
        await client.connect()
        result = await client.translate_prompt("Translate to Spanish: Hello world")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "gemma3:4b",
        timeout: float = 120.0,
        default_max_tokens: int = 1024,
        default_temperature: float = 0.3,
        default_backend: str = "ollama",
        proxy_mode: bool = False,
    ):
        """
        Initialize the unified LLM client.

        Args:
            base_url: API base URL. For direct mode, an OpenAI-compatible endpoint.
                      For proxy mode, the Translation Service URL.
            api_key: API key (empty for Ollama, required for OpenAI/Groq).
            model: Model name for direct mode.
            timeout: Request timeout in seconds.
            default_max_tokens: Default max tokens for generation.
            default_temperature: Default temperature for generation.
            default_backend: Default LLM backend name (used in proxy mode).
            proxy_mode: If True, use Translation Service V3 API instead of
                       direct OpenAI-compatible endpoint.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_backend = default_backend
        self.proxy_mode = proxy_mode

        self._session: aiohttp.ClientSession | None = None
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def connect(self) -> bool:
        """Initialize the HTTP session."""
        if self._session is None or self._session.closed:
            if self.proxy_mode:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                )
            else:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=self._headers(),
                )
        mode = "proxy" if self.proxy_mode else "direct"
        logger.info(f"LLMClient connected to {self.base_url} (mode={mode}, model={self.model})")
        return True

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("LLMClient connection closed")

    # =========================================================================
    # Native Multi-Turn Chat (direct mode only)
    # =========================================================================

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> PromptTranslationResult:
        """
        Send a multi-turn conversation to the LLM.

        Only available in direct mode. In proxy mode, falls back to
        translate_prompt with message flattening.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            model: Model override

        Returns:
            PromptTranslationResult with response text and metadata
        """
        if self.proxy_mode:
            # Flatten messages into a single prompt for proxy mode
            prompt, system_prompt = _flatten_messages_to_prompt(messages)
            return await self.translate_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        if not self._circuit_breaker.is_available:
            raise ConnectionError(
                "LLMClient circuit breaker is OPEN - "
                "LLM backend appears unavailable, failing fast"
            )

        if not self._session or self._session.closed:
            await self.connect()

        url = f"{self.base_url}/chat/completions"
        used_model = model or self.model

        payload = {
            "model": used_model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "stream": False,
        }

        start = time.monotonic()
        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._circuit_breaker.record_failure()
                    raise Exception(f"LLM chat failed: HTTP {response.status} - {error_text}")

                data = await response.json()
                elapsed_ms = (time.monotonic() - start) * 1000

                self._circuit_breaker.record_success()

                choice = data["choices"][0]
                text = choice["message"]["content"]
                usage = data.get("usage", {})

                return PromptTranslationResult(
                    text=text,
                    processing_time_ms=elapsed_ms,
                    backend_used="direct_llm",
                    model_used=data.get("model", used_model),
                    tokens_used=usage.get("total_tokens"),
                )

        except aiohttp.ClientError as e:
            self._circuit_breaker.record_failure()
            logger.error(f"LLM chat request failed: {e}")
            raise ConnectionError(f"LLM chat request failed: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a multi-turn conversation response from the LLM.

        Only available in direct mode. In proxy mode, falls back to
        translate_prompt_stream with message flattening.

        Args:
            messages: List of {"role": "...", "content": "..."}
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            model: Model override

        Yields:
            StreamChunk with chunk text or final result
        """
        if self.proxy_mode:
            prompt, system_prompt = _flatten_messages_to_prompt(messages)
            async for chunk in self.translate_prompt_stream(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                yield chunk
            return

        if not self._circuit_breaker.is_available:
            yield StreamChunk(
                error="LLMClient circuit breaker is OPEN - LLM backend unavailable",
                done=True,
            )
            return

        if not self._session or self._session.closed:
            await self.connect()

        url = f"{self.base_url}/chat/completions"
        used_model = model or self.model

        payload = {
            "model": used_model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "stream": True,
        }

        start = time.monotonic()
        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._circuit_breaker.record_failure()
                    yield StreamChunk(
                        error=f"HTTP {response.status} - {error_text}",
                        done=True,
                    )
                    return

                self._circuit_breaker.record_success()

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    if line_str.startswith("data: "):
                        data_str = line_str[6:]

                        if data_str == "[DONE]":
                            elapsed_ms = (time.monotonic() - start) * 1000
                            yield StreamChunk(
                                done=True,
                                processing_time_ms=elapsed_ms,
                                backend_used="direct_llm",
                                model_used=used_model,
                            )
                            return

                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield StreamChunk(
                                    chunk=content,
                                    done=False,
                                    backend_used="direct_llm",
                                    model_used=used_model,
                                )
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue

                # If we reach here without [DONE], emit final chunk
                elapsed_ms = (time.monotonic() - start) * 1000
                yield StreamChunk(
                    done=True,
                    processing_time_ms=elapsed_ms,
                    backend_used="direct_llm",
                    model_used=used_model,
                )

        except aiohttp.ClientError as e:
            self._circuit_breaker.record_failure()
            logger.error(f"LLM streaming failed: {e}")
            yield StreamChunk(error=str(e), done=True)

    # =========================================================================
    # LLMClientProtocol Interface (translate_prompt / translate_prompt_stream)
    # =========================================================================

    async def translate_prompt(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> PromptTranslationResult:
        """
        Send a prompt to the LLM.

        In direct mode: converts prompt/system_prompt into a messages array
        and calls the chat endpoint.

        In proxy mode: sends to Translation Service V3 API at
        {base_url}/api/v3/translate.
        """
        if self.proxy_mode:
            return await self._proxy_translate_prompt(
                prompt=prompt,
                backend=backend,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            )

        # Direct mode: convert to chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def translate_prompt_stream(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a prompt response.

        In direct mode: converts to messages and calls chat_stream.
        In proxy mode: calls Translation Service V3 streaming API.
        """
        if self.proxy_mode:
            async for chunk in self._proxy_translate_prompt_stream(
                prompt=prompt,
                backend=backend,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
            ):
                yield chunk
            return

        # Direct mode: convert to chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async for chunk in self.chat_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield chunk

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """
        Check if the LLM backend is reachable.

        In direct mode: tries /models endpoint (standard OpenAI-compatible).
        In proxy mode: checks /api/health on the Translation Service.
        """
        if not self._session or self._session.closed:
            await self.connect()

        try:
            if self.proxy_mode:
                url = f"{self.base_url}/api/health"
                async with self._session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("status") == "healthy"
                    return False
            else:
                url = f"{self.base_url}/models"
                async with self._session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False

    # =========================================================================
    # Proxy Mode Implementation (Translation Service V3 API)
    # =========================================================================

    async def _proxy_translate_prompt(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> PromptTranslationResult:
        """Send a prompt via Translation Service V3 API."""
        if not self._circuit_breaker.is_available:
            raise ConnectionError(
                "LLMClient circuit breaker is OPEN - "
                "Translation service appears unavailable, failing fast"
            )

        if not self._session or self._session.closed:
            await self.connect()

        url = f"{self.base_url}/api/v3/translate"

        payload = {
            "prompt": prompt,
            "backend": backend or self.default_backend,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._circuit_breaker.record_failure()
                    raise Exception(f"Translation failed: HTTP {response.status} - {error_text}")

                data = await response.json()

                self._circuit_breaker.record_success()
                return PromptTranslationResult(
                    text=data["text"],
                    processing_time_ms=data["processing_time_ms"],
                    backend_used=data["backend_used"],
                    model_used=data["model_used"],
                    tokens_used=data.get("tokens_used"),
                )

        except aiohttp.ClientError as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Translation request failed: {e}")
            raise Exception(f"Translation request failed: {e}") from e

    async def _proxy_translate_prompt_stream(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a prompt response via Translation Service V3 API."""
        if not self._session or self._session.closed:
            await self.connect()

        url = f"{self.base_url}/api/v3/translate/stream"

        payload = {
            "prompt": prompt,
            "backend": backend or self.default_backend,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            async with self._session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamChunk(error=f"HTTP {response.status} - {error_text}", done=True)
                    return

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix

                        try:
                            data = json.loads(data_str)

                            yield StreamChunk(
                                chunk=data.get("chunk"),
                                done=data.get("done", False),
                                processing_time_ms=data.get("processing_time_ms"),
                                backend_used=data.get("backend_used"),
                                model_used=data.get("model_used"),
                                error=data.get("error"),
                            )

                            if data.get("done"):
                                return

                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            logger.error(f"Streaming translation failed: {e}")
            yield StreamChunk(error=str(e), done=True)


# =============================================================================
# Helper Functions
# =============================================================================


def _flatten_messages_to_prompt(
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """
    Flatten a messages array into (prompt, system_prompt) for proxy mode.

    The system message becomes the system_prompt parameter.
    All other messages are concatenated as "ROLE: content" lines.

    Returns:
        Tuple of (prompt_string, system_prompt_string)
    """
    system_prompt = ""
    conversation_lines = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_prompt = content
        elif role == "user":
            conversation_lines.append(f"USER: {content}")
        elif role == "assistant":
            conversation_lines.append(f"ASSISTANT: {content}")

    prompt = "\n".join(conversation_lines)
    return prompt, system_prompt


# =============================================================================
# Factory Function
# =============================================================================


def create_llm_client(
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "",
    model: str = "gemma3:4b",
    timeout: float = 120.0,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    default_backend: str = "ollama",
    proxy_mode: bool = False,
) -> LLMClient:
    """
    Create an LLMClient.

    Args:
        base_url: API base URL
        api_key: API key (empty for Ollama)
        model: Model name (used in direct mode)
        timeout: Request timeout
        max_tokens: Default max tokens
        temperature: Default temperature
        default_backend: Default backend name (used in proxy mode)
        proxy_mode: If True, use Translation Service V3 API

    Returns:
        Configured LLMClient
    """
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        default_max_tokens=max_tokens,
        default_temperature=temperature,
        default_backend=default_backend,
        proxy_mode=proxy_mode,
    )
