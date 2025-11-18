"""
Local translation implementation using vLLM and Ollama inference.
Provides privacy-focused, high-performance translation without external API calls.
"""

import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
import logging

# Import shared inference module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../shared/src"))

from inference import (
    get_inference_client_async,
    InferenceRequest
)

logger = logging.getLogger(__name__)


@dataclass
class LocalTranslationResponse:
    """Response format for local translation operations."""
    translated_text: str
    source_language: str
    target_language: str
    backend: str
    model: str
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    tokens: Optional[Dict[str, int]] = None
    raw_response: Optional[Dict[str, Any]] = None


class LocalTranslationService:
    """
    Local translation service using vLLM and Ollama.
    Provides privacy-focused translation without external API dependencies.
    """
    
    def __init__(self,
                 preferred_backend: Optional[str] = None,
                 fallback_enabled: bool = True,
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        """
        Initialize local translation service.
        
        Args:
            preferred_backend: Preferred backend ('vllm', 'ollama', or None for auto)
            fallback_enabled: Enable fallback between backends
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.preferred_backend = preferred_backend
        self.fallback_enabled = fallback_enabled
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._is_initialized = False
        
        # Language mappings for better prompts
        self.language_codes = {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
            "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
            "th": "Thai", "vi": "Vietnamese", "nl": "Dutch", "sv": "Swedish",
            "da": "Danish", "no": "Norwegian", "fi": "Finnish", "pl": "Polish"
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the local translation service.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Get inference client with preferred backend
            self._client = await get_inference_client_async(
                backend=self.preferred_backend
            )
            
            if self._client:
                self._is_initialized = True
                logger.info(f"Local translation service initialized with {self._client.backend_type.value}")
                return True
            else:
                logger.error("Failed to initialize local inference client")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize local translation service: {e}")
            return False
    
    async def translate(self,
                       text: str,
                       target_language: str,
                       source_language: Optional[str] = None,
                       **kwargs) -> LocalTranslationResponse:
        """
        Translate text using local inference.
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            source_language: Source language (auto-detect if None)
            **kwargs: Additional arguments
            
        Returns:
            Translation response
        """
        if not self._is_initialized:
            raise RuntimeError("Local translation service not initialized")
        
        try:
            start_time = time.time()
            
            # Build translation prompt
            prompt = self._build_translation_prompt(text, target_language, source_language)
            
            # Create inference request
            request = InferenceRequest(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                source_language=source_language,
                target_language=target_language,
                stop_sequences=["</translation>", "\n\n", "Original:"]
            )
            
            # Perform inference
            response = await self._client.generate(request)
            
            # Parse translation from response
            translated_text = self._extract_translation(response.text)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            confidence = self._estimate_confidence(response.text, translated_text)
            
            return LocalTranslationResponse(
                translated_text=translated_text,
                source_language=source_language or "auto",
                target_language=target_language,
                backend=response.backend.value,
                model=response.model,
                confidence=confidence,
                latency_ms=latency_ms,
                tokens={
                    "input": response.prompt_tokens or 0,
                    "output": response.completion_tokens or 0,
                    "total": response.total_tokens or 0
                },
                raw_response={
                    "full_response": response.text,
                    "tokens_per_second": response.tokens_per_second
                }
            )
            
        except Exception as e:
            logger.error(f"Local translation failed: {e}")
            raise
    
    async def translate_stream(self,
                             text: str,
                             target_language: str,
                             source_language: Optional[str] = None,
                             **kwargs) -> AsyncGenerator[LocalTranslationResponse, None]:
        """
        Stream translation using local inference.
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language
            **kwargs: Additional arguments
            
        Yields:
            Partial translation responses
        """
        if not self._is_initialized:
            raise RuntimeError("Local translation service not initialized")
        
        try:
            start_time = time.time()
            
            # Build translation prompt
            prompt = self._build_translation_prompt(text, target_language, source_language)
            
            # Create streaming inference request
            request = InferenceRequest(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                source_language=source_language,
                target_language=target_language,
                stream=True,
                stop_sequences=["</translation>", "\n\n", "Original:"]
            )
            
            accumulated_text = ""
            
            # Stream inference
            async for chunk in self._client.generate_stream(request):
                accumulated_text = chunk.text
                translated_text = self._extract_translation(accumulated_text)
                
                # Only yield if we have meaningful translation text
                if translated_text.strip():
                    current_latency = (time.time() - start_time) * 1000
                    confidence = self._estimate_confidence(accumulated_text, translated_text)
                    
                    yield LocalTranslationResponse(
                        translated_text=translated_text,
                        source_language=source_language or "auto",
                        target_language=target_language,
                        backend=chunk.backend.value,
                        model=chunk.model,
                        confidence=confidence,
                        latency_ms=current_latency,
                        tokens={
                            "input": chunk.prompt_tokens or 0,
                            "output": chunk.completion_tokens or 0,
                            "total": chunk.total_tokens or 0
                        },
                        raw_response={
                            "full_response": accumulated_text,
                            "tokens_per_second": chunk.tokens_per_second
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Local streaming translation failed: {e}")
            raise
    
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages for local inference.
        
        Returns:
            List of supported language codes
        """
        # Return languages based on model capabilities
        return list(self.language_codes.keys())
    
    async def is_healthy(self) -> bool:
        """
        Check if the local translation service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._is_initialized or not self._client:
                return False
            
            return await self._client.is_healthy()
            
        except Exception:
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information
        """
        if not self._is_initialized or not self._client:
            return {}
        
        try:
            models = await self._client.list_models()
            if models:
                model = models[0]  # Get first available model
                return {
                    "name": model.name,
                    "backend": model.backend.value,
                    "size_gb": model.size_gb,
                    "context_length": model.context_length,
                    "languages": model.languages or list(self.language_codes.keys()),
                    "capabilities": model.capabilities
                }
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
        
        return {}
    
    async def shutdown(self):
        """Shutdown the local translation service."""
        try:
            if self._client:
                await self._client.__aexit__(None, None, None)
            
            self._is_initialized = False
            
        except Exception as e:
            logger.error(f"Error during local translation shutdown: {e}")
    
    def _build_translation_prompt(self,
                                text: str,
                                target_language: str,
                                source_language: Optional[str] = None) -> str:
        """
        Build an effective translation prompt for the local model.
        
        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language
            
        Returns:
            Formatted prompt
        """
        # Map language codes to full names
        target_lang_name = self.language_codes.get(target_language.lower(), target_language)
        
        if source_language:
            source_lang_name = self.language_codes.get(source_language.lower(), source_language)
            prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}. Provide only the translation without any explanation or additional text.

Original text: {text}

Translation:"""
        else:
            prompt = f"""Translate the following text to {target_lang_name}. Provide only the translation without any explanation or additional text.

Text: {text}

Translation:"""
        
        return prompt
    
    def _extract_translation(self, response_text: str) -> str:
        """
        Extract the actual translation from the model response.
        
        Args:
            response_text: Full response from the model
            
        Returns:
            Extracted translation text
        """
        # Remove common prefixes and suffixes
        text = response_text.strip()
        
        # Remove common response patterns
        patterns_to_remove = [
            "Translation:", "translation:", "TRANSLATION:",
            "Here is the translation:", "The translation is:",
            "Translated text:", "Result:", "Output:"
        ]
        
        for pattern in patterns_to_remove:
            if text.startswith(pattern):
                text = text[len(pattern):].strip()
        
        # Remove quotes if the entire response is quoted
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Split on double newlines and take the first non-empty part
        parts = text.split('\n\n')
        if parts:
            text = parts[0].strip()
        
        # Take only the first line if there are multiple lines
        lines = text.split('\n')
        if lines:
            text = lines[0].strip()
        
        return text
    
    def _estimate_confidence(self, full_response: str, extracted_translation: str) -> float:
        """
        Estimate confidence in the translation quality.
        
        Args:
            full_response: Full model response
            extracted_translation: Extracted translation
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristics for confidence estimation
        confidence = 1.0
        
        # Penalize very short translations
        if len(extracted_translation.strip()) < 3:
            confidence *= 0.5
        
        # Penalize if the response contains error indicators
        error_indicators = ["error", "cannot", "unable", "sorry", "don't know"]
        if any(indicator in full_response.lower() for indicator in error_indicators):
            confidence *= 0.3
        
        # Penalize if translation is identical to input (for different languages)
        # This is a simplified check
        if extracted_translation.strip().lower() == full_response.strip().lower():
            confidence *= 0.7
        
        # Boost confidence for reasonable length translations
        if 5 <= len(extracted_translation.strip()) <= 500:
            confidence *= 1.1
        
        return min(1.0, max(0.0, confidence))


# Convenience function for creating local translation service
def create_local_translation_service(backend: Optional[str] = None,
                                   temperature: float = 0.3,
                                   max_tokens: int = 1024) -> LocalTranslationService:
    """
    Create a local translation service with common configuration.
    
    Args:
        backend: Preferred backend ('vllm', 'ollama', or None for auto)
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured local translation service
    """
    return LocalTranslationService(
        preferred_backend=backend,
        temperature=temperature,
        max_tokens=max_tokens
    ) 