"""
Triton Inference Server Client for vLLM Backend

This client provides integration with NVIDIA Triton Inference Server
using the vLLM backend for efficient LLM inference.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any, Union
from dataclasses import dataclass
import aiohttp
import time

from .base_client import BaseInferenceClient, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

@dataclass
class TritonConfig:
    """Configuration for Triton Inference Server"""
    base_url: str = "http://localhost:8000"
    model_name: str = "vllm_model"
    model_version: str = "1"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # vLLM specific parameters
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    
    # Triton specific
    use_http: bool = True  # Use HTTP instead of gRPC for simplicity
    stream: bool = False
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = ["</s>", "<|endoftext|>", "<|im_end|>"]


class TritonInferenceClient(BaseInferenceClient):
    """
    Triton Inference Server client with vLLM backend support
    
    Provides async interface to Triton Server for LLM inference
    with proper error handling, retries, and streaming support.
    """
    
    def __init__(self, config: Optional[TritonConfig] = None):
        """Initialize Triton client with configuration"""
        self.config = config or TritonConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # Triton endpoints
        self.health_url = f"{self.config.base_url}/v2/health"
        self.model_ready_url = f"{self.config.base_url}/v2/models/{self.config.model_name}/ready"
        self.generate_url = f"{self.config.base_url}/v2/models/{self.config.model_name}/generate"
        self.infer_url = f"{self.config.base_url}/v2/models/{self.config.model_name}/infer"
        
        logger.info(f"Triton client initialized for {self.config.base_url}")
    
    async def initialize(self) -> bool:
        """Initialize the client and verify server connection"""
        try:
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Check server health
            health_ok = await self.health_check()
            if not health_ok:
                logger.error("Triton server health check failed")
                return False
            
            # Check model readiness
            model_ready = await self._check_model_ready()
            if not model_ready:
                logger.error(f"Model {self.config.model_name} is not ready")
                return False
            
            self.is_initialized = True
            logger.info("Triton client successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Triton client: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Triton server is healthy"""
        try:
            if not self.session:
                return False
                
            async with self.session.get(self.health_url) as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def _check_model_ready(self) -> bool:
        """Check if the specified model is ready for inference"""
        try:
            if not self.session:
                return False
                
            async with self.session.get(self.model_ready_url) as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"Model readiness check failed: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate text using Triton's generate endpoint
        
        Args:
            request: Inference request with prompt and parameters
            
        Returns:
            Generated text response
        """
        if not self.is_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Prepare request payload for Triton generate endpoint
        payload = {
            "text_input": request.prompt,
            "parameters": {
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "top_k": request.top_k or self.config.top_k,
                "repetition_penalty": getattr(request, 'repetition_penalty', self.config.repetition_penalty),
                "presence_penalty": getattr(request, 'presence_penalty', self.config.presence_penalty),
                "frequency_penalty": getattr(request, 'frequency_penalty', self.config.frequency_penalty),
                "stop": request.stop_sequences or self.config.stop_sequences,
                "stream": request.stream or self.config.stream,
            }
        }
        
        # Remove None values
        payload["parameters"] = {k: v for k, v in payload["parameters"].items() if v is not None}
        
        try:
            async with self.session.post(self.generate_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Triton generate failed: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Extract generated text
                generated_text = result.get("text_output", "")
                
                # Calculate metrics
                processing_time = time.time() - start_time
                
                return InferenceResponse(
                    text=generated_text,
                    model=self.config.model_name,
                    processing_time=processing_time,
                    metadata={
                        "triton_model_version": self.config.model_version,
                        "backend": "triton_vllm",
                        "endpoint": "generate"
                    }
                )
                
        except Exception as e:
            logger.error(f"Triton generate request failed: {e}")
            raise
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """
        Generate streaming text using Triton server
        
        Args:
            request: Inference request with streaming enabled
            
        Yields:
            Text chunks as they are generated
        """
        if not self.is_initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        # Enable streaming in request
        stream_request = InferenceRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
            stream=True
        )
        
        # Prepare streaming payload
        payload = {
            "text_input": stream_request.prompt,
            "parameters": {
                "max_tokens": stream_request.max_tokens or self.config.max_tokens,
                "temperature": stream_request.temperature or self.config.temperature,
                "top_p": stream_request.top_p or self.config.top_p,
                "top_k": stream_request.top_k or self.config.top_k,
                "stop": stream_request.stop_sequences or self.config.stop_sequences,
                "stream": True,
            }
        }
        
        # Remove None values
        payload["parameters"] = {k: v for k, v in payload["parameters"].items() if v is not None}
        
        try:
            async with self.session.post(self.generate_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Triton streaming failed: {response.status} - {error_text}")
                
                # Handle streaming response
                async for line in response.content:
                    if line:
                        try:
                            # Parse streaming JSON response
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                json_str = line_text[6:]  # Remove 'data: ' prefix
                                if json_str == '[DONE]':
                                    break
                                
                                chunk_data = json.loads(json_str)
                                if 'text_output' in chunk_data:
                                    yield chunk_data['text_output']
                                    
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
                        except Exception as e:
                            logger.warning(f"Error processing stream chunk: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Triton streaming request failed: {e}")
            # Fallback to non-streaming
            response = await self.generate(request)
            yield response.text
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            if not self.session:
                return {}
            
            model_info_url = f"{self.config.base_url}/v2/models/{self.config.model_name}"
            async with self.session.get(model_info_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
                    
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return {}
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "type": "triton_vllm",
            "server_url": self.config.base_url,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "supports_streaming": True,
            "supports_batching": True,
            "max_tokens": self.config.max_tokens,
        }
    
    async def close(self):
        """Close the client and cleanup resources"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_initialized = False
            logger.info("Triton client closed")
            
        except Exception as e:
            logger.error(f"Error closing Triton client: {e}")


# Factory function for easy client creation
async def create_triton_client(
    base_url: str = "http://localhost:8000",
    model_name: str = "vllm_model",
    **kwargs
) -> TritonInferenceClient:
    """
    Factory function to create and initialize a Triton client
    
    Args:
        base_url: Triton server URL
        model_name: Name of the model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized TritonInferenceClient
    """
    config = TritonConfig(base_url=base_url, model_name=model_name, **kwargs)
    client = TritonInferenceClient(config)
    
    success = await client.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Triton client")
    
    return client


# Auto-detection function
async def detect_triton_server(base_url: str = "http://localhost:8000") -> bool:
    """
    Detect if a Triton server is available at the given URL
    
    Args:
        base_url: URL to check for Triton server
        
    Returns:
        True if Triton server is detected and healthy
    """
    try:
        async with aiohttp.ClientSession() as session:
            health_url = f"{base_url}/v2/health"
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                return response.status == 200
                
    except Exception:
        return False