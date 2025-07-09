"""
Base interface for local inference clients (vLLM, Ollama).
Provides a unified API for local LLM inference across LiveTranslate modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum


class InferenceBackend(Enum):
    """Supported local inference backends."""
    TRITON = "triton"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"  # For fallback/comparison


@dataclass
class InferenceRequest:
    """Standard request format for all inference backends."""
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    # Translation-specific parameters
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    
    # Additional backend-specific parameters
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """Standard response format for all inference backends."""
    text: str
    model: str
    backend: InferenceBackend
    
    # Performance metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Timing information
    latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    
    # Confidence and quality metrics
    confidence_score: Optional[float] = None
    
    # Error information
    error: Optional[str] = None
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """Information about available models."""
    name: str
    backend: InferenceBackend
    size_gb: Optional[float] = None
    context_length: Optional[int] = None
    languages: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    is_available: bool = True
    
    # Performance characteristics
    avg_tokens_per_second: Optional[float] = None
    memory_usage_gb: Optional[float] = None


class BaseInferenceClient(ABC):
    """Abstract base class for local inference clients."""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None,
                 timeout: int = 60):
        """
        Initialize the inference client.
        
        Args:
            base_url: Base URL for the inference server
            api_key: API key if required
            default_model: Default model to use for requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using the local inference backend.
        
        Args:
            request: Standardized inference request
            
        Returns:
            Standardized inference response
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, request: InferenceRequest):
        """
        Generate text with streaming response.
        
        Args:
            request: Standardized inference request
            
        Yields:
            Partial InferenceResponse objects
        """
        pass
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """
        Check if the inference backend is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """
        List available models on the backend.
        
        Returns:
            List of available model information
        """
        pass
    
    @abstractmethod
    async def load_model(self, model_name: str) -> bool:
        """
        Load a specific model (if supported by backend).
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model (if supported by backend).
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        pass
    
    def get_model_name(self, request: InferenceRequest) -> str:
        """
        Get the model name to use for a request.
        
        Args:
            request: Inference request
            
        Returns:
            Model name to use
        """
        return request.model or self.default_model
    
    async def translate(self, 
                       text: str, 
                       target_language: str,
                       source_language: Optional[str] = None,
                       model: Optional[str] = None) -> InferenceResponse:
        """
        Convenience method for translation tasks.
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            source_language: Source language (auto-detect if None)
            model: Model to use (default if None)
            
        Returns:
            Translation response
        """
        # Build translation prompt
        if source_language:
            prompt = f"Translate from {source_language} to {target_language}: {text}"
        else:
            prompt = f"Translate to {target_language}: {text}"
        
        request = InferenceRequest(
            prompt=prompt,
            model=model,
            source_language=source_language,
            target_language=target_language,
            max_tokens=len(text) * 2,  # Rough estimation
            temperature=0.3  # Lower temperature for translation
        )
        
        return await self.generate(request)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class InferenceClientError(Exception):
    """Base exception for inference client errors."""
    pass


class ModelNotFoundError(InferenceClientError):
    """Raised when a requested model is not available."""
    pass


class InferenceTimeoutError(InferenceClientError):
    """Raised when inference request times out."""
    pass


class BackendUnavailableError(InferenceClientError):
    """Raised when the inference backend is not available."""
    pass 