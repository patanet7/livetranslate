"""
Ollama client implementation for easy local model management and inference.
Supports dynamic model loading, GGUF/GGML formats, and simple API.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from .base_client import (
    BackendUnavailableError,
    BaseInferenceClient,
    InferenceBackend,
    InferenceClientError,
    InferenceRequest,
    InferenceResponse,
    InferenceTimeoutError,
    ModelInfo,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


class OllamaClient(BaseInferenceClient):
    """
    Ollama client for easy local LLM inference and model management.

    Supports:
    - Dynamic model loading/unloading
    - GGUF/GGML model formats
    - CPU and GPU inference
    - Model pulling from registry
    - Simple REST API
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        default_model: str = "llama3.1:8b",
        timeout: int = 120,
    ):  # Longer timeout for model loading
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server base URL (default: http://localhost:11434)
            api_key: API key (not typically used with Ollama)
            default_model: Default model name (e.g., "llama3.1:8b")
            timeout: Request timeout in seconds (longer for model operations)
        """
        super().__init__(base_url, api_key, default_model, timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    @property
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        return InferenceBackend.OLLAMA

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using Ollama.

        Args:
            request: Standardized inference request

        Returns:
            Standardized inference response
        """
        start_time = time.time()

        try:
            session = await self._get_session()

            # Ensure model is loaded
            model_name = self.get_model_name(request)
            if not await self._ensure_model_loaded(model_name):
                raise ModelNotFoundError(f"Model {model_name} not available")

            # Build Ollama payload
            payload = self._build_ollama_payload(request)

            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceClientError(
                        f"Ollama request failed with status {response.status}: {error_text}"
                    )

                # Ollama returns streaming by default, so we collect all chunks
                full_response = ""
                response_data = None

                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if chunk.get("response"):
                                full_response += chunk["response"]

                            # Keep the last chunk for metadata
                            response_data = chunk

                            # Check if done
                            if chunk.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000

                return self._parse_ollama_response(
                    full_response, response_data, request, latency_ms
                )

        except TimeoutError as e:
            raise InferenceTimeoutError("Ollama request timed out") from e
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            raise InferenceClientError(f"Ollama inference failed: {e}") from e

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncGenerator[InferenceResponse, None]:
        """
        Generate text with streaming response.

        Args:
            request: Standardized inference request

        Yields:
            Partial InferenceResponse objects
        """
        start_time = time.time()

        try:
            session = await self._get_session()

            # Ensure model is loaded
            model_name = self.get_model_name(request)
            if not await self._ensure_model_loaded(model_name):
                raise ModelNotFoundError(f"Model {model_name} not available")

            # Build Ollama payload
            payload = self._build_ollama_payload(request)
            payload["stream"] = True  # Explicitly enable streaming

            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceClientError(
                        f"Ollama stream request failed with status {response.status}: {error_text}"
                    )

                accumulated_text = ""

                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))

                            if chunk.get("response"):
                                delta_text = chunk["response"]
                                accumulated_text += delta_text

                                # Calculate current latency
                                current_latency = (time.time() - start_time) * 1000

                                yield InferenceResponse(
                                    text=accumulated_text,
                                    model=model_name,
                                    backend=self.backend_type,
                                    latency_ms=current_latency,
                                    raw_response=chunk,
                                )

                            # Check if done
                            if chunk.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

        except TimeoutError as e:
            raise InferenceTimeoutError("Ollama stream request timed out") from e
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(f"Failed to connect to Ollama server: {e}") from e
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise InferenceClientError(f"Ollama streaming failed: {e}") from e

    async def is_healthy(self) -> bool:
        """
        Check if Ollama server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200

        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models on Ollama server.

        Returns:
            List of available model information
        """
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return []

                result = await response.json()
                models = []

                for model_data in result.get("models", []):
                    # Parse model size from details
                    size_gb = None
                    if "size" in model_data:
                        size_bytes = model_data["size"]
                        size_gb = size_bytes / (1024**3)  # Convert to GB

                    model_info = ModelInfo(
                        name=model_data["name"],
                        backend=self.backend_type,
                        size_gb=size_gb,
                        capabilities=["text-generation", "translation", "chat"],
                        is_available=True,
                    )
                    models.append(model_info)

                return models

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def load_model(self, model_name: str) -> bool:
        """
        Load a specific model in Ollama (pull if needed).

        Args:
            model_name: Name of the model to load (e.g., "llama3.1:8b")

        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if model is already available
            models = await self.list_models()
            for model in models:
                if model.name == model_name:
                    return True  # Already available

            # If not available, try to pull it
            session = await self._get_session()

            payload = {"name": model_name}

            async with session.post(f"{self.base_url}/api/pull", json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to pull model {model_name}")
                    return False

                # Wait for pull to complete
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            if chunk.get("status") == "success":
                                logger.info(f"Successfully pulled model {model_name}")
                                return True
                        except json.JSONDecodeError:
                            continue

                return False

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from Ollama.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if successful, False otherwise
        """
        try:
            session = await self._get_session()

            payload = {"name": model_name}

            async with session.delete(f"{self.base_url}/api/delete", json=payload) as response:
                if response.status == 200:
                    logger.info(f"Successfully unloaded model {model_name}")
                    return True
                else:
                    logger.error(f"Failed to unload model {model_name}")
                    return False

        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False

    async def _ensure_model_loaded(self, model_name: str) -> bool:
        """
        Ensure a model is loaded, loading it if necessary.

        Args:
            model_name: Name of the model

        Returns:
            True if model is available, False otherwise
        """
        if not model_name:
            return False

        # Check if model is already loaded
        models = await self.list_models()
        for model in models:
            if model.name == model_name:
                return True

        # Try to load the model
        logger.info(f"Model {model_name} not found, attempting to pull...")
        return await self.load_model(model_name)

    def _build_ollama_payload(self, request: InferenceRequest) -> dict[str, Any]:
        """
        Build Ollama-specific payload.

        Args:
            request: Standardized inference request

        Returns:
            Ollama-compatible payload
        """
        payload = {
            "model": self.get_model_name(request),
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
            },
        }

        # Add stop sequences
        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences

        # Add extra parameters
        if request.extra_params:
            if "options" in request.extra_params:
                payload["options"].update(request.extra_params["options"])
            else:
                payload.update(request.extra_params)

        return payload

    def _parse_ollama_response(
        self,
        text: str,
        response_data: dict[str, Any] | None,
        request: InferenceRequest,
        latency_ms: float,
    ) -> InferenceResponse:
        """
        Parse Ollama response.

        Args:
            text: Generated text
            response_data: Raw response data from Ollama
            request: Original request
            latency_ms: Request latency in milliseconds

        Returns:
            Standardized inference response
        """
        try:
            # Extract performance metrics if available
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            if response_data:
                eval_count = response_data.get("eval_count")
                prompt_eval_count = response_data.get("prompt_eval_count")

                if eval_count:
                    completion_tokens = eval_count
                if prompt_eval_count:
                    prompt_tokens = prompt_eval_count
                if prompt_tokens and completion_tokens:
                    total_tokens = prompt_tokens + completion_tokens

            # Calculate tokens per second
            tokens_per_second = None
            if completion_tokens and latency_ms:
                tokens_per_second = (completion_tokens / latency_ms) * 1000

            return InferenceResponse(
                text=text,
                model=self.get_model_name(request),
                backend=self.backend_type,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                tokens_per_second=tokens_per_second,
                raw_response=response_data,
            )

        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise InferenceClientError(f"Failed to parse Ollama response: {e}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()


# Convenience function for creating Ollama client
def create_ollama_client(
    base_url: str = "http://localhost:11434", default_model: str = "llama3.1:8b"
) -> OllamaClient:
    """
    Create an Ollama client with common configuration.

    Args:
        base_url: Ollama server URL
        default_model: Default model name

    Returns:
        Configured Ollama client
    """
    return OllamaClient(base_url=base_url, default_model=default_model)
