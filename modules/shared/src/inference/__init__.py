"""
Shared inference module for LiveTranslate.
Provides unified access to local inference backends (vLLM, Ollama).
"""

import asyncio
import logging
import os
from typing import Any, Optional

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
from .ollama_client import OllamaClient, create_ollama_client
from .triton_client import TritonInferenceClient, create_triton_client, detect_triton_server
from .vllm_client import VLLMClient, create_vllm_client

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TRITON_URL = "http://localhost:8000"
DEFAULT_MODELS = {
    InferenceBackend.TRITON: "vllm_model",
    InferenceBackend.VLLM: "meta-llama/Llama-3.1-8B-Instruct",
    InferenceBackend.OLLAMA: "llama3.1:8b",
}


async def detect_available_backends() -> dict[InferenceBackend, bool]:
    """
    Detect which inference backends are available.

    Returns:
        Dictionary mapping backends to availability status
    """
    availability = {}

    # Check vLLM availability
    try:
        vllm_client = create_vllm_client(base_url=os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_URL))
        async with vllm_client:
            availability[InferenceBackend.VLLM] = await vllm_client.is_healthy()
    except Exception as e:
        logger.debug(f"vLLM not available: {e}")
        availability[InferenceBackend.VLLM] = False

    # Check Ollama availability
    try:
        ollama_client = create_ollama_client(
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
        )
        async with ollama_client:
            availability[InferenceBackend.OLLAMA] = await ollama_client.is_healthy()
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")
        availability[InferenceBackend.OLLAMA] = False

    return availability


def get_inference_client(
    backend: str | None = None, base_url: str | None = None, model: str | None = None, **kwargs
) -> BaseInferenceClient:
    """
    Get an inference client for the specified or auto-detected backend.

    Args:
        backend: Backend type ('vllm', 'ollama', or None for auto-detect)
        base_url: Custom base URL for the backend
        model: Default model name
        **kwargs: Additional client configuration

    Returns:
        Configured inference client

    Raises:
        BackendUnavailableError: If no backends are available
    """

    # Parse backend parameter
    if backend:
        backend = backend.lower()
        if backend == "vllm":
            backend_enum = InferenceBackend.VLLM
        elif backend == "ollama":
            backend_enum = InferenceBackend.OLLAMA
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    else:
        # Auto-detect backend based on environment or availability
        backend_enum = _auto_detect_backend()

    # Create client based on backend
    if backend_enum == InferenceBackend.VLLM:
        return create_vllm_client(
            base_url=base_url or os.getenv("VLLM_BASE_URL", DEFAULT_VLLM_URL),
            default_model=model
            or os.getenv("VLLM_DEFAULT_MODEL", DEFAULT_MODELS[InferenceBackend.VLLM]),
            **kwargs,
        )
    elif backend_enum == InferenceBackend.OLLAMA:
        return create_ollama_client(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
            default_model=model
            or os.getenv("OLLAMA_DEFAULT_MODEL", DEFAULT_MODELS[InferenceBackend.OLLAMA]),
            **kwargs,
        )
    else:
        raise BackendUnavailableError("No supported inference backend available")


async def get_inference_client_async(
    backend: str | None = None, base_url: str | None = None, model: str | None = None, **kwargs
) -> BaseInferenceClient:
    """
    Async version of get_inference_client with health checking.

    Args:
        backend: Backend type ('vllm', 'ollama', or None for auto-detect)
        base_url: Custom base URL for the backend
        model: Default model name
        **kwargs: Additional client configuration

    Returns:
        Configured and healthy inference client

    Raises:
        BackendUnavailableError: If no healthy backends are available
    """

    if backend:
        # Use specified backend
        client = get_inference_client(backend, base_url, model, **kwargs)
        if await client.is_healthy():
            return client
        else:
            raise BackendUnavailableError(f"Specified backend {backend} is not healthy")

    # Try backends in order of preference
    availability = await detect_available_backends()

    # Prefer Triton for production, then vLLM for performance, fallback to Ollama
    for backend_enum in [InferenceBackend.TRITON, InferenceBackend.VLLM, InferenceBackend.OLLAMA]:
        if availability.get(backend_enum, False):
            try:
                backend_str = backend_enum.value
                client = get_inference_client(backend_str, base_url, model, **kwargs)
                logger.info(f"Using {backend_str} backend for inference")
                return client
            except Exception as e:
                logger.warning(f"Failed to create {backend_str} client: {e}")
                continue

    raise BackendUnavailableError("No healthy inference backends available")


def _auto_detect_backend() -> InferenceBackend:
    """
    Auto-detect the preferred backend based on environment variables.

    Returns:
        Preferred backend enum
    """
    # Check environment preference
    backend_pref = os.getenv("INFERENCE_BACKEND", "").lower()

    if backend_pref == "triton":
        return InferenceBackend.TRITON
    elif backend_pref == "vllm":
        return InferenceBackend.VLLM
    elif backend_pref == "ollama":
        return InferenceBackend.OLLAMA

    # Default preference: Triton for production, fallback to vLLM
    return InferenceBackend.TRITON


async def test_translation(
    text: str = "Hello, world!", target_language: str = "Spanish", backend: str | None = None
) -> InferenceResponse:
    """
    Test translation functionality with available backends.

    Args:
        text: Text to translate
        target_language: Target language for translation
        backend: Specific backend to test (None for auto-detect)

    Returns:
        Translation response
    """
    client = await get_inference_client_async(backend)

    async with client:
        return await client.translate(text=text, target_language=target_language)


# Export key classes and functions
__all__ = [
    "BackendUnavailableError",
    "BaseInferenceClient",
    "InferenceBackend",
    "InferenceClientError",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceTimeoutError",
    "ModelInfo",
    "ModelNotFoundError",
    "OllamaClient",
    "VLLMClient",
    "create_ollama_client",
    "create_vllm_client",
    "detect_available_backends",
    "get_inference_client",
    "get_inference_client_async",
    "test_translation",
]
