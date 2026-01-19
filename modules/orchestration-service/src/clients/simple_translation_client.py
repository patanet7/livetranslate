"""
Simple Translation Client

Minimal client for the V3 translation API - just sends prompts and gets results.

The translation service is DUMB:
- It receives a complete prompt (with context, glossary, language embedded)
- It sends the prompt to the LLM
- It returns the result

All intelligence stays in the orchestration service.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PromptTranslationResult:
    """Result from V3 translation endpoint."""

    text: str
    processing_time_ms: float
    backend_used: str
    model_used: str
    tokens_used: int | None = None


@dataclass
class StreamChunk:
    """Streaming chunk from V3 translation endpoint."""

    chunk: str | None = None
    done: bool = False
    processing_time_ms: float | None = None
    backend_used: str | None = None
    model_used: str | None = None
    error: str | None = None


# =============================================================================
# Simple Translation Client
# =============================================================================


class SimpleTranslationClient:
    """
    Minimal client for the V3 translation API.

    Sends complete prompts to translation service and returns results.
    No session management, no context tracking - just prompt-in, translation-out.

    Usage:
        client = SimpleTranslationClient("http://localhost:5003")
        await client.connect()

        result = await client.translate_prompt(
            prompt="Translate to Spanish: Hello world",
            backend="ollama",
        )
        print(result.text)  # "Hola mundo"

        await client.close()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5003",
        timeout: float = 60.0,
        default_backend: str = "ollama",
        default_max_tokens: int = 256,
        default_temperature: float = 0.3,
    ):
        """
        Initialize the simple translation client.

        Args:
            base_url: Translation service base URL
            timeout: Request timeout in seconds
            default_backend: Default LLM backend (ollama, groq, etc.)
            default_max_tokens: Default max tokens for generation
            default_temperature: Default temperature for generation
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_backend = default_backend
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> bool:
        """
        Initialize the HTTP session.

        Returns:
            True if connected successfully
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            logger.info(f"SimpleTranslationClient connected to {self.base_url}")
        return True

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("SimpleTranslationClient connection closed")

    async def translate_prompt(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> PromptTranslationResult:
        """
        Send a complete prompt to the translation service.

        Args:
            prompt: Complete prompt with context, glossary, language embedded
            backend: LLM backend to use (default: ollama)
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt override

        Returns:
            PromptTranslationResult with translated text and metadata

        Raises:
            ConnectionError: If not connected
            Exception: If translation fails
        """
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
                    raise Exception(f"Translation failed: HTTP {response.status} - {error_text}")

                data = await response.json()

                return PromptTranslationResult(
                    text=data["text"],
                    processing_time_ms=data["processing_time_ms"],
                    backend_used=data["backend_used"],
                    model_used=data["model_used"],
                    tokens_used=data.get("tokens_used"),
                )

        except aiohttp.ClientError as e:
            logger.error(f"Translation request failed: {e}")
            raise Exception(f"Translation request failed: {e}") from e

    async def translate_prompt_stream(
        self,
        prompt: str,
        backend: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream translation results from the translation service.

        Args:
            prompt: Complete prompt with context, glossary, language embedded
            backend: LLM backend to use (default: ollama)
            max_tokens: Max tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt override

        Yields:
            StreamChunk with chunk text or final result

        Raises:
            ConnectionError: If not connected
            Exception: If translation fails
        """
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

                import json

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

    async def health_check(self) -> bool:
        """
        Check if translation service is healthy.

        Returns:
            True if service is healthy
        """
        if not self._session or self._session.closed:
            await self.connect()

        try:
            url = f"{self.base_url}/api/health"
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


# =============================================================================
# Factory Function
# =============================================================================


def create_simple_translation_client(
    base_url: str = "http://localhost:5003",
    backend: str = "ollama",
    timeout: float = 60.0,
) -> SimpleTranslationClient:
    """
    Create a SimpleTranslationClient.

    Args:
        base_url: Translation service URL
        backend: Default LLM backend
        timeout: Request timeout

    Returns:
        Configured SimpleTranslationClient
    """
    return SimpleTranslationClient(
        base_url=base_url,
        default_backend=backend,
        timeout=timeout,
    )
