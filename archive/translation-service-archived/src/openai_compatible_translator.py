#!/usr/bin/env python3
"""
OpenAI-Compatible Translation Backend

Provides translation using any OpenAI-compatible API endpoint.
Supports:
- Local services: Ollama, vLLM, LM Studio, LocalAI, TGI
- Remote services: Groq, Together AI, Fireworks, OpenRouter, Replicate
- OpenAI itself

All these services expose the same OpenAI chat completions API format.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from livetranslate_common.logging import get_logger

logger = get_logger()


@dataclass
class OpenAICompatibleConfig:
    """Configuration for OpenAI-compatible API endpoint"""

    name: str  # Friendly name (e.g., "ollama-local", "groq-cloud")
    base_url: str  # Base URL (e.g., "http://localhost:11434/v1", "https://api.groq.com/openai/v1")
    api_key: str | None = None  # API key (optional for local services like Ollama)
    model: str = "llama3.1:8b"  # Model name
    temperature: float = 0.3  # Lower for more consistent translation
    max_tokens: int = 2048
    timeout: float = 30.0

    # Translation prompt configuration
    system_prompt: str = (
        "You are a professional translator. Translate naturally and semantically, "
        "preserving meaning, tone, and context. Return ONLY the direct translation. "
        "Do not include: explanations, notes, parenthetical comments, alternatives, "
        "translations of your translation, or any other text besides the translation itself."
    )

    # Advanced settings
    streaming: bool = False
    retry_count: int = 3
    retry_delay: float = 1.0

    # Service-specific settings
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleTranslator:
    """
    Generic translator for any OpenAI-compatible API.

    Example configurations:

    # Ollama (local)
    config = OpenAICompatibleConfig(
        name="ollama-local",
        base_url="http://localhost:11434/v1",
        model="llama3.1:8b",
        api_key=None  # Not required for Ollama
    )

    # Groq (remote)
    config = OpenAICompatibleConfig(
        name="groq-cloud",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Together AI (remote)
    config = OpenAICompatibleConfig(
        name="together-ai",
        base_url="https://api.together.xyz/v1",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key=os.getenv("TOGETHER_API_KEY")
    )

    # vLLM Server (local/remote)
    config = OpenAICompatibleConfig(
        name="vllm-server",
        base_url="http://localhost:8000/v1",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key=None
    )
    """

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self.is_ready = False
        self._http_client: httpx.AsyncClient | None = None

        logger.info(f"Initialized OpenAI-compatible translator: {config.name} ({config.base_url})")

    async def initialize(self) -> bool:
        """Initialize and test the connection"""
        try:
            # Create HTTP client
            headers = {
                "Content-Type": "application/json",
            }

            # Add API key if provided
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            # Add extra headers
            headers.update(self.config.extra_headers)

            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout),
            )

            # Test connection with a simple request
            test_result = await self._test_connection()

            if test_result:
                self.is_ready = True
                logger.info(f"✅ {self.config.name} is ready")
                return True
            else:
                logger.warning(f"❌ {self.config.name} connection test failed")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize {self.config.name}: {e}")
            return False

    async def get_available_models(self) -> list[str]:
        """
        Get list of available models from the endpoint.

        Works with:
        - OpenAI-compatible /v1/models endpoint
        - Ollama-specific /api/tags endpoint

        Returns:
            List of model names/IDs
        """
        if not self._http_client:
            return []

        try:
            # Try OpenAI-compatible /v1/models endpoint first
            try:
                response = await self._http_client.get("/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    logger.info(f"📋 {self.config.name} available models (OpenAI format): {models}")
                    return models
            except Exception as e:
                logger.debug(f"OpenAI /v1/models format not supported: {e}")

            # Try Ollama-specific /api/tags endpoint
            try:
                # For Ollama, we need to hit the base URL without /v1
                ollama_base = self.config.base_url.replace("/v1", "")
                logger.info(f"🔍 Fetching Ollama models from: {ollama_base}/api/tags")
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{ollama_base}/api/tags")
                    logger.info(f"🔍 Ollama /api/tags response status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"🔍 Ollama /api/tags raw response: {data}")
                        models = [model["name"] for model in data.get("models", [])]
                        logger.info(
                            f"📋 {self.config.name} available models (Ollama format): {models}"
                        )
                        return models
            except Exception as e:
                logger.warning(f"Ollama /api/tags failed: {e}")

            logger.warning(f"Could not retrieve models list from {self.config.name}")
            return []

        except Exception as e:
            logger.error(f"Failed to get available models from {self.config.name}: {e}")
            return []

    async def validate_model(self, model_name: str) -> bool:
        """
        Check if a specific model is available on this endpoint.

        Args:
            model_name: Model name to check

        Returns:
            True if model is available, False otherwise
        """
        available_models = await self.get_available_models()

        if not available_models:
            # If we can't get the list, assume model might be available
            logger.warning(f"Cannot verify model '{model_name}' availability - will try anyway")
            return True

        # Check exact match
        if model_name in available_models:
            return True

        # Check partial match (e.g., "llama3.1:8b" contains "llama3.1")
        for available_model in available_models:
            if model_name in available_model or available_model in model_name:
                logger.info(
                    f"Model '{model_name}' matched with available model '{available_model}'"
                )
                return True

        logger.warning(f"Model '{model_name}' not found in available models: {available_models}")
        return False

    async def _test_connection(self) -> bool:
        """Test the API connection with a minimal request"""
        try:
            # Try a simple API call to verify connectivity
            # For Ollama, test with a very simple short translation
            test_payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.1,
            }

            response = await self._http_client.post("/chat/completions", json=test_payload)

            if response.status_code == 200:
                logger.info(f"Connection test passed for {self.config.name}")
                return True
            else:
                logger.warning(
                    f"Connection test failed for {self.config.name}: HTTP {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Connection test failed for {self.config.name}: {e}")
            return False

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Translate text using OpenAI-compatible chat completions API.

        Args:
            text: Source text to translate
            source_language: Source language code
            target_language: Target language code
            context: Optional context for better translation

        Returns:
            Dict with translated_text, confidence, metadata
        """
        if not self.is_ready or not self._http_client:
            logger.error(f"{self.config.name} is not ready")
            return None

        start_time = datetime.now()

        try:
            # Build the translation prompt
            prompt = self._build_translation_prompt(text, source_language, target_language, context)

            # Determine which model to use (override takes precedence)
            model_to_use = model_override or self.config.model

            # Prepare the request payload
            payload = {
                "model": model_to_use,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False,  # Non-streaming for now
            }

            # Add extra parameters
            payload.update(self.config.extra_params)

            # Make the API request with retries
            response_data = await self._make_request_with_retry(payload)

            if not response_data:
                return None

            # Extract translation from OpenAI response format
            translated_text = self._extract_translation(response_data)

            if not translated_text:
                logger.error(f"Failed to extract translation from response: {response_data}")
                return None

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "translated_text": translated_text,
                "confidence": self._estimate_confidence(response_data),
                "metadata": {
                    "backend_used": self.config.name,
                    "model_used": model_to_use,
                    "processing_time": processing_time,
                    "api_response": response_data,
                },
            }

        except Exception as e:
            logger.error(f"Translation failed with {self.config.name}: {e}")
            return None

    async def translate_batch(
        self, texts: list[str], source_language: str, target_language: str
    ) -> list[dict[str, Any] | None]:
        """
        Translate multiple texts in parallel.

        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code

        Returns:
            List of translation results
        """
        tasks = [self.translate(text, source_language, target_language) for text in texts]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to None
        return [result if not isinstance(result, Exception) else None for result in results]

    async def generate_from_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Generate text from a raw prompt - NO prompt building, just send directly to LLM.

        This is the simplified "dumb" interface where the caller (orchestration service)
        is responsible for building the complete prompt with context, glossary, etc.

        Args:
            prompt: Complete prompt ready to send to LLM (with all context embedded)
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Temperature for generation (uses config default if None)
            system_prompt: Optional system prompt override

        Returns:
            Dict with text, processing_time_ms, backend_used, model_used
        """
        if not self.is_ready or not self._http_client:
            logger.error(f"{self.config.name} is not ready")
            return None

        start_time = datetime.now()

        try:
            # Build messages - use provided system prompt or default minimal one
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Prepare the request payload
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
                "stream": False,
            }

            # Add extra parameters
            payload.update(self.config.extra_params)

            # Make the API request with retries
            response_data = await self._make_request_with_retry(payload)

            if not response_data:
                return None

            # Extract text from response
            text = self._extract_translation(response_data)

            if not text:
                logger.error(f"Failed to extract text from response: {response_data}")
                return None

            # Calculate processing time in milliseconds
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            return {
                "text": text,
                "processing_time_ms": round(processing_time_ms, 2),
                "backend_used": self.config.name,
                "model_used": self.config.model,
                "tokens_used": response_data.get("usage", {}).get("completion_tokens", 0),
            }

        except Exception as e:
            logger.error(f"Generation failed with {self.config.name}: {e}")
            return None

    async def generate_from_prompt_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ):
        """
        Stream text generation from a raw prompt.

        Yields chunks as they're generated.

        Args:
            prompt: Complete prompt ready to send to LLM
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt override

        Yields:
            Dict with chunk, done, and optionally processing_time_ms when done
        """
        if not self.is_ready or not self._http_client:
            logger.error(f"{self.config.name} is not ready")
            yield {"error": "Service not ready", "done": True}
            return

        start_time = datetime.now()

        try:
            # Build messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            # Prepare the request payload with streaming enabled
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
                "stream": True,
            }

            # Add extra parameters
            payload.update(self.config.extra_params)

            # Make streaming request
            async with self._http_client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                timeout=httpx.Timeout(self.config.timeout * 2),  # Longer timeout for streaming
            ) as response:
                if response.status_code != 200:
                    yield {"error": f"HTTP {response.status_code}", "done": True}
                    return

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            processing_time_ms = (
                                datetime.now() - start_time
                            ).total_seconds() * 1000
                            yield {
                                "done": True,
                                "processing_time_ms": round(processing_time_ms, 2),
                                "backend_used": self.config.name,
                                "model_used": self.config.model,
                            }
                            return

                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield {"chunk": content, "done": False}
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming generation failed with {self.config.name}: {e}")
            yield {"error": str(e), "done": True}

    async def _make_request_with_retry(self, payload: dict) -> dict | None:
        """Make API request with exponential backoff retry"""
        for attempt in range(self.config.retry_count):
            try:
                response = await self._http_client.post("/chat/completions", json=payload)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"{self.config.name} request failed with status {response.status_code}: "
                        f"{response.text}"
                    )

                    # Don't retry on 4xx errors (client errors)
                    if 400 <= response.status_code < 500:
                        return None

            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.config.retry_count - 1:
                await asyncio.sleep(self.config.retry_delay * (2**attempt))

        logger.error(f"All {self.config.retry_count} retry attempts failed for {self.config.name}")
        return None

    def _build_translation_prompt(
        self, text: str, source_language: str, target_language: str, context: str | None = None
    ) -> str:
        """Build the translation prompt - returns ONLY the translated text, no commentary"""
        prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"
        return prompt

    def _extract_translation(self, response_data: dict) -> str | None:
        """Extract translation from OpenAI API response format and clean up commentary"""
        try:
            # Standard OpenAI response format:
            # {
            #   "choices": [
            #     {
            #       "message": {
            #         "content": "translated text here"
            #       }
            #     }
            #   ]
            # }

            choices = response_data.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            content = message.get("content", "")

            # Clean up LLM commentary
            content = content.strip()

            # Remove anything after double newlines (usually explanations)
            if "\n\n" in content:
                content = content.split("\n\n")[0].strip()

            # Remove parenthetical explanations at the end
            import re

            content = re.sub(r"\s*\([^)]*\)\s*$", "", content)

            return content.strip()

        except Exception as e:
            logger.error(f"Failed to extract translation: {e}")
            return None

    def _estimate_confidence(self, response_data: dict) -> float:
        """
        Estimate confidence from API response.

        Some APIs provide logprobs or finish_reason which can indicate quality.
        For now, return a default high confidence for successful responses.
        """
        try:
            finish_reason = response_data.get("choices", [{}])[0].get("finish_reason")

            if finish_reason == "stop":
                return 0.9  # Normal completion
            elif finish_reason == "length":
                return 0.7  # Hit token limit, might be truncated
            else:
                return 0.8  # Default

        except Exception:
            return 0.8  # Default confidence

    async def close(self):
        """Close HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self.is_ready = False
            logger.info(f"Closed {self.config.name} translator")


# =============================================================================
# FACTORY FUNCTIONS FOR COMMON SERVICES
# =============================================================================


def create_ollama_translator(
    model: str = "llama3.1:8b", base_url: str = "http://localhost:11434/v1"
) -> OpenAICompatibleTranslator:
    """Create translator for local Ollama instance"""
    config = OpenAICompatibleConfig(
        name="ollama-local",
        base_url=base_url,
        model=model,
        api_key=None,  # Ollama doesn't require API key
        temperature=0.3,
    )
    return OpenAICompatibleTranslator(config)


def create_groq_translator(
    model: str = "llama-3.1-8b-instant", api_key: str | None = None
) -> OpenAICompatibleTranslator:
    """Create translator for Groq cloud service"""
    api_key = api_key or os.getenv("GROQ_API_KEY")

    config = OpenAICompatibleConfig(
        name="groq-cloud",
        base_url="https://api.groq.com/openai/v1",
        model=model,
        api_key=api_key,
        temperature=0.3,
    )
    return OpenAICompatibleTranslator(config)


def create_together_translator(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", api_key: str | None = None
) -> OpenAICompatibleTranslator:
    """Create translator for Together AI service"""
    api_key = api_key or os.getenv("TOGETHER_API_KEY")

    config = OpenAICompatibleConfig(
        name="together-ai",
        base_url="https://api.together.xyz/v1",
        model=model,
        api_key=api_key,
        temperature=0.3,
    )
    return OpenAICompatibleTranslator(config)


def create_vllm_server_translator(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", base_url: str = "http://localhost:8000/v1"
) -> OpenAICompatibleTranslator:
    """Create translator for vLLM server instance"""
    config = OpenAICompatibleConfig(
        name="vllm-server",
        base_url=base_url,
        model=model,
        api_key=None,  # vLLM server typically doesn't require API key
        temperature=0.3,
    )
    return OpenAICompatibleTranslator(config)


def create_openai_translator(
    model: str = "gpt-4o-mini", api_key: str | None = None
) -> OpenAICompatibleTranslator:
    """Create translator for actual OpenAI API"""
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    config = OpenAICompatibleConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        model=model,
        api_key=api_key,
        temperature=0.3,
    )
    return OpenAICompatibleTranslator(config)


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================


def load_openai_compatible_translators_from_config(
    config_file: str | None = None,
) -> list[OpenAICompatibleTranslator]:
    """
    Load multiple OpenAI-compatible translators from configuration.

    Example config file (YAML or JSON):
    ```yaml
    openai_compatible_backends:
      - name: "ollama-local"
        base_url: "http://localhost:11434/v1"
        model: "llama3.1:8b"
        enabled: true

      - name: "groq-cloud"
        base_url: "https://api.groq.com/openai/v1"
        model: "llama-3.1-8b-instant"
        api_key_env: "GROQ_API_KEY"
        enabled: true

      - name: "together-ai"
        base_url: "https://api.together.xyz/v1"
        model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        api_key_env: "TOGETHER_API_KEY"
        enabled: false
    ```
    """
    translators = []

    # TODO: Implement config file loading
    # For now, return empty list - will be implemented with config integration

    return translators


# =============================================================================
# USAGE EXAMPLE
# =============================================================================


async def example_usage():
    """Example of how to use OpenAI-compatible translator"""

    # Example 1: Local Ollama
    ollama = create_ollama_translator()
    await ollama.initialize()

    if ollama.is_ready:
        result = await ollama.translate(
            text="Hello, how are you?", source_language="en", target_language="es"
        )
        print(f"Ollama translation: {result}")

    # Example 2: Groq Cloud (requires API key)
    if os.getenv("GROQ_API_KEY"):
        groq = create_groq_translator()
        await groq.initialize()

        if groq.is_ready:
            result = await groq.translate(
                text="Good morning", source_language="en", target_language="fr"
            )
            print(f"Groq translation: {result}")

    # Example 3: Custom endpoint
    custom_config = OpenAICompatibleConfig(
        name="my-custom-server",
        base_url="http://my-server.com/v1",
        model="my-model",
        api_key="my-secret-key",
    )
    custom = OpenAICompatibleTranslator(custom_config)
    await custom.initialize()

    if custom.is_ready:
        result = await custom.translate(
            text="Thank you", source_language="en", target_language="de"
        )
        print(f"Custom translation: {result}")

    # Cleanup
    await ollama.close()
    if os.getenv("GROQ_API_KEY"):
        await groq.close()
    await custom.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
