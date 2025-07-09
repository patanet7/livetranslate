"""
vLLM client implementation for high-performance local inference.
Provides GPU-accelerated LLM inference with OpenAI-compatible API.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, AsyncGenerator, Any
import aiohttp
import logging

from .base_client import (
    BaseInferenceClient, 
    InferenceRequest, 
    InferenceResponse, 
    InferenceBackend,
    ModelInfo,
    InferenceClientError,
    ModelNotFoundError,
    InferenceTimeoutError,
    BackendUnavailableError
)

logger = logging.getLogger(__name__)


class VLLMClient(BaseInferenceClient):
    """
    vLLM client for high-performance local LLM inference.
    
    Supports:
    - GPU acceleration
    - Batch processing
    - Streaming responses
    - OpenAI-compatible API
    - Model management
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None,
                 timeout: int = 60):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server base URL (default: http://localhost:8000)
            api_key: API key if authentication is enabled
            default_model: Default model name
            timeout: Request timeout in seconds
        """
        super().__init__(base_url, api_key, default_model, timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self._session
    
    @property
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        return InferenceBackend.VLLM
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using vLLM.
        
        Args:
            request: Standardized inference request
            
        Returns:
            Standardized inference response
        """
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Convert to OpenAI-compatible format
            payload = self._build_openai_payload(request)
            
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceClientError(
                        f"vLLM request failed with status {response.status}: {error_text}"
                    )
                
                result = await response.json()
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                return self._parse_openai_response(result, request, latency_ms)
                
        except asyncio.TimeoutError:
            raise InferenceTimeoutError("vLLM request timed out")
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(f"Failed to connect to vLLM server: {e}")
        except Exception as e:
            logger.error(f"vLLM inference error: {e}")
            raise InferenceClientError(f"vLLM inference failed: {e}")
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[InferenceResponse, None]:
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
            
            # Convert to OpenAI-compatible format with streaming
            payload = self._build_openai_payload(request)
            payload["stream"] = True
            
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise InferenceClientError(
                        f"vLLM stream request failed with status {response.status}: {error_text}"
                    )
                
                accumulated_text = ""
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                        
                        if line == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line)
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                
                                if "text" in choice:
                                    delta_text = choice["text"]
                                    accumulated_text += delta_text
                                    
                                    # Calculate current latency
                                    current_latency = (time.time() - start_time) * 1000
                                    
                                    yield InferenceResponse(
                                        text=accumulated_text,
                                        model=self.get_model_name(request),
                                        backend=self.backend_type,
                                        latency_ms=current_latency,
                                        raw_response=chunk
                                    )
                        
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON
                            
        except asyncio.TimeoutError:
            raise InferenceTimeoutError("vLLM stream request timed out")
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(f"Failed to connect to vLLM server: {e}")
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise InferenceClientError(f"vLLM streaming failed: {e}")
    
    async def is_healthy(self) -> bool:
        """
        Check if vLLM server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/health") as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False
    
    async def list_models(self) -> List[ModelInfo]:
        """
        List available models on vLLM server.
        
        Returns:
            List of available model information
        """
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status != 200:
                    return []
                
                result = await response.json()
                models = []
                
                for model_data in result.get("data", []):
                    model_info = ModelInfo(
                        name=model_data["id"],
                        backend=self.backend_type,
                        context_length=model_data.get("context_length"),
                        capabilities=["text-generation", "translation"],
                        is_available=True
                    )
                    models.append(model_info)
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []
    
    async def load_model(self, model_name: str) -> bool:
        """
        Load a specific model (vLLM typically loads one model at startup).
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful (or already loaded), False otherwise
        """
        # vLLM typically loads one model at server startup
        # This method checks if the requested model is available
        models = await self.list_models()
        for model in models:
            if model.name == model_name:
                return True
        return False
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model (not supported by vLLM).
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            False (not supported)
        """
        # vLLM doesn't support dynamic model unloading
        logger.warning("Model unloading is not supported by vLLM")
        return False
    
    def _build_openai_payload(self, request: InferenceRequest) -> Dict[str, Any]:
        """
        Build OpenAI-compatible payload for vLLM.
        
        Args:
            request: Standardized inference request
            
        Returns:
            OpenAI-compatible payload
        """
        payload = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        
        # Add model if specified
        model_name = self.get_model_name(request)
        if model_name:
            payload["model"] = model_name
        
        # Add stop sequences
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        
        # Add extra parameters
        if request.extra_params:
            payload.update(request.extra_params)
        
        return payload
    
    def _parse_openai_response(self, 
                              response: Dict[str, Any], 
                              request: InferenceRequest,
                              latency_ms: float) -> InferenceResponse:
        """
        Parse OpenAI-compatible response from vLLM.
        
        Args:
            response: Raw response from vLLM
            request: Original request
            latency_ms: Request latency in milliseconds
            
        Returns:
            Standardized inference response
        """
        try:
            # Extract text from choices
            choices = response.get("choices", [])
            if not choices:
                raise InferenceClientError("No choices in vLLM response")
            
            text = choices[0].get("text", "")
            
            # Extract usage statistics
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            
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
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Failed to parse vLLM response: {e}")
            raise InferenceClientError(f"Failed to parse vLLM response: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session and not self._session.closed:
            await self._session.close()


# Convenience function for creating vLLM client
def create_vllm_client(base_url: str = "http://localhost:8000",
                       api_key: Optional[str] = None,
                       default_model: Optional[str] = None) -> VLLMClient:
    """
    Create a vLLM client with common configuration.
    
    Args:
        base_url: vLLM server URL
        api_key: API key if authentication enabled
        default_model: Default model name
        
    Returns:
        Configured vLLM client
    """
    return VLLMClient(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model
    ) 