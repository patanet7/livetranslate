#!/usr/bin/env python3
"""
LiveTranslate Translation Service - FastAPI Version
OpenAI-Compatible Translation with Ollama, Groq, vLLM, etc.
"""

import os
import time
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# Load .env file FIRST
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

# Initialize structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import translation backends
from openai_compatible_translator import OpenAICompatibleTranslator, OpenAICompatibleConfig
from model_manager import RuntimeModelManager, get_model_manager, initialize_model_manager

# ============================================================================
# Pydantic Models
# ============================================================================

class TranslateRequest(BaseModel):
    """Single translation request"""
    text: str = Field(..., description="Text to translate")
    source_language: Optional[str] = Field(None, description="Source language (auto-detect if None)")
    target_language: str = Field(..., description="Target language code")
    model: str = Field("ollama", description="Translation model/backend to use")
    quality: str = Field("balanced", description="Translation quality: fast, balanced, quality")

class MultiLanguageRequest(BaseModel):
    """Multi-language translation request"""
    text: str = Field(..., description="Text to translate")
    source_language: Optional[str] = Field(None, description="Source language (auto-detect if None)")
    target_languages: List[str] = Field(..., description="List of target language codes")
    model: str = Field("ollama", description="Translation model/backend to use")
    quality: str = Field("balanced", description="Translation quality: fast, balanced, quality")

class LanguageDetectionRequest(BaseModel):
    """Language detection request"""
    text: str = Field(..., min_length=1, max_length=10000)

class TranslationResponse(BaseModel):
    """Translation response"""
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    processing_time: float
    model_used: str
    backend_used: str

class MultiLanguageResponse(BaseModel):
    """Multi-language translation response"""
    source_text: str
    source_language: str
    model_requested: str
    quality: str
    total_processing_time: float
    timestamp: str
    translations: Dict[str, Dict[str, Any]]

class LanguageDetectionResponse(BaseModel):
    """Language detection response"""
    language: str
    confidence: float
    alternatives: List[Dict[str, float]]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    backend: str
    version: str
    timestamp: str


# ============================================================================
# Model Management Models
# ============================================================================

class ModelSwitchRequest(BaseModel):
    """Request to switch the active model"""
    model: str = Field(..., description="Model name (e.g., 'llama2:7b', 'mistral:latest')")
    backend: str = Field("ollama", description="Backend to use: ollama, groq, vllm, openai")


class ModelPreloadRequest(BaseModel):
    """Request to preload a model for faster switching"""
    model: str = Field(..., description="Model name to preload")
    backend: str = Field("ollama", description="Backend to use")


class ModelUnloadRequest(BaseModel):
    """Request to unload a cached model"""
    model: str = Field(..., description="Model name to unload")
    backend: str = Field("ollama", description="Backend the model is loaded on")


class ModelSwitchResponse(BaseModel):
    """Response for model switch operation"""
    success: bool
    model: str
    backend: str
    message: str
    cached_models: int


class ModelStatusResponse(BaseModel):
    """Response for model manager status"""
    current_model: Optional[str]
    current_backend: Optional[str]
    is_ready: bool
    cached_models: List[Dict[str, Any]]
    cache_size: int
    supported_backends: List[str]

# ============================================================================
# V3 API Models - Simplified "prompt-in, translation-out" contract
# ============================================================================

class PromptTranslateRequest(BaseModel):
    """
    V3 Simplified translation request - accepts complete prompt directly.

    The orchestration service builds the prompt with:
    - Rolling context windows
    - Glossary terms
    - Speaker information
    - Target language instructions

    This service just sends it to the LLM and returns the result.
    """
    prompt: str = Field(..., description="Complete prompt to send to LLM (with context, glossary embedded)")
    backend: str = Field("ollama", description="Backend to use: ollama, groq, vllm, openai")
    max_tokens: int = Field(256, description="Maximum tokens to generate")
    temperature: float = Field(0.3, description="Temperature for generation (0.0-1.0)")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt override")

class PromptTranslateResponse(BaseModel):
    """V3 Simplified translation response"""
    text: str = Field(..., description="Generated text (the translation)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    backend_used: str = Field(..., description="Backend that processed the request")
    model_used: str = Field(..., description="Model that generated the response")
    tokens_used: Optional[int] = Field(None, description="Tokens used for generation")

# ============================================================================
# Global state
# ============================================================================

ollama_translator: Optional[OpenAICompatibleTranslator] = None
translator_backends: Dict[str, OpenAICompatibleTranslator] = {}

# Runtime model manager for dynamic model switching
model_manager: Optional[RuntimeModelManager] = None

# Active realtime translation sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="LiveTranslate Translation Service",
    description="Multi-language translation with OpenAI-compatible backends (Ollama, Groq, vLLM)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize translation backends on startup"""
    global ollama_translator, translator_backends, model_manager

    logger.info("🚀 Starting LiveTranslate Translation Service (FastAPI)")

    # Initialize RuntimeModelManager for dynamic model switching
    model_manager = get_model_manager()
    logger.info("📦 Initializing RuntimeModelManager...")

    # Initialize Ollama backend
    ollama_enabled = os.getenv("OLLAMA_ENABLE", "true").lower() == "true"

    if ollama_enabled:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.239:11434/v1")
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral:latest")

        logger.info(f"🔌 Initializing Ollama backend: {ollama_base_url}")
        logger.info(f"📦 Ollama model: {ollama_model}")

        ollama_config = OpenAICompatibleConfig(
            name="ollama-local",
            base_url=ollama_base_url,
            model=ollama_model,
            api_key="",  # Ollama doesn't require API key
            timeout=60.0,
        )

        ollama_translator = OpenAICompatibleTranslator(ollama_config)

        # Initialize and test connection
        if await ollama_translator.initialize():
            translator_backends["ollama"] = ollama_translator
            logger.info("✅ Ollama translator ready")

            # Get available models
            models = await ollama_translator.get_available_models()
            logger.info(f"📋 Available Ollama models: {models}")
        else:
            logger.error("❌ Ollama initialization failed")

    # Initialize Groq backend (optional)
    groq_enabled = os.getenv("GROQ_ENABLE", "false").lower() == "true"
    if groq_enabled:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            groq_config = OpenAICompatibleConfig(
                name="groq",
                base_url="https://api.groq.com/openai/v1",
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                api_key=groq_api_key,
                timeout=30.0,
            )
            groq_translator = OpenAICompatibleTranslator(groq_config)
            if await groq_translator.initialize():
                translator_backends["groq"] = groq_translator
                logger.info("✅ Groq translator ready")

    logger.info(f"🎉 Initialized {len(translator_backends)} translator backend(s): {list(translator_backends.keys())}")

    # Initialize RuntimeModelManager with default model
    if await model_manager.initialize_default():
        logger.info(f"✅ RuntimeModelManager ready with model: {model_manager.current_model}")
    else:
        logger.warning("⚠️ RuntimeModelManager initialization failed - will use legacy backends")

# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns service status and available backends
    """
    return HealthResponse(
        status="healthy" if translator_backends else "degraded",
        service="translation",
        backend=",".join(translator_backends.keys()) if translator_backends else "none",
        version="2.0.0",
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/api/device-info", tags=["Info"])
async def get_device_info():
    """Get device information (CPU/GPU status)"""
    return {
        "device": "remote",  # Using remote Ollama
        "backends": list(translator_backends.keys()),
        "available_models": {
            backend: await translator.get_available_models()
            for backend, translator in translator_backends.items()
        }
    }

@app.get("/api/models", tags=["Info"])
async def get_models():
    """
    Get list of available models (orchestration-compatible format)

    Returns models in format expected by orchestration service client.
    """
    models = []
    for backend_name, translator in translator_backends.items():
        try:
            backend_models = await translator.get_available_models()
            for model in backend_models:
                if isinstance(model, str):
                    models.append({"name": model, "backend": backend_name})
                elif isinstance(model, dict):
                    models.append({**model, "backend": backend_name})
        except Exception as e:
            logger.error(f"Failed to get models from {backend_name}: {e}")

    # Include RuntimeModelManager models if available
    if model_manager is not None:
        status = model_manager.get_status()
        if status.get("current_model"):
            models.append({
                "name": status["current_model"],
                "backend": status.get("current_backend", "ollama"),
                "active": True
            })

    return {
        "models": models,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/models/available", tags=["Info"])
async def get_available_models():
    """
    Get list of available models from all backends

    Returns a dict mapping backend names to their available models
    """
    models = {}
    for backend_name, translator in translator_backends.items():
        try:
            backend_models = await translator.get_available_models()
            models[backend_name] = backend_models
        except Exception as e:
            logger.error(f"Failed to get models from {backend_name}: {e}")
            models[backend_name] = []

    return {
        "backends": models,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/languages", tags=["Info"])
async def get_supported_languages():
    """Get list of supported languages"""
    # Common language codes
    languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "it", "name": "Italian"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ar", "name": "Arabic"},
        {"code": "hi", "name": "Hindi"},
    ]

    return {
        "languages": languages,
        "total": len(languages)
    }

# ============================================================================
# Model Management Endpoints - Dynamic Model Switching
# ============================================================================

@app.post("/api/models/switch", response_model=ModelSwitchResponse, tags=["Model Management"])
async def switch_model_runtime(request: ModelSwitchRequest):
    """
    Switch to a different model at runtime (no restart required).

    **Usage:**
    Switch between Ollama models, or even different backends, without restarting the service.
    Previously loaded models are cached for instant switching.

    **Example:**
    ```json
    {
      "model": "llama2:7b",
      "backend": "ollama"
    }
    ```

    **Supported Backends:**
    - `ollama` - Local Ollama instance
    - `groq` - Groq cloud API (requires GROQ_API_KEY)
    - `vllm` - vLLM server
    - `openai` - OpenAI API (requires OPENAI_API_KEY)
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    try:
        success = await model_manager.switch_model(request.model, request.backend)

        status_info = model_manager.get_status()

        if success:
            return ModelSwitchResponse(
                success=True,
                model=request.model,
                backend=request.backend,
                message=f"Successfully switched to {request.model} on {request.backend}",
                cached_models=status_info["cache_size"]
            )
        else:
            return ModelSwitchResponse(
                success=False,
                model=request.model,
                backend=request.backend,
                message=f"Failed to switch to {request.model} on {request.backend}",
                cached_models=status_info["cache_size"]
            )

    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {str(e)}"
        )


@app.post("/api/models/preload", tags=["Model Management"])
async def preload_model(request: ModelPreloadRequest):
    """
    Pre-load a model for faster switching later.

    **Usage:**
    Preload models you'll need soon in the background.
    This makes subsequent switches instant.

    **Example:**
    ```json
    {
      "model": "mistral:latest",
      "backend": "ollama"
    }
    ```
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    try:
        success = await model_manager.preload_model(request.model, request.backend)

        return {
            "success": success,
            "model": request.model,
            "backend": request.backend,
            "message": f"{'Successfully preloaded' if success else 'Failed to preload'} {request.model}",
            "cached_models": model_manager.get_status()["cache_size"]
        }

    except Exception as e:
        logger.error(f"Error preloading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preload model: {str(e)}"
        )


@app.post("/api/models/unload", tags=["Model Management"])
async def unload_model(request: ModelUnloadRequest):
    """
    Unload a cached model to free resources.

    **Note:** Cannot unload the currently active model.
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    try:
        success = await model_manager.unload_model(request.model, request.backend)

        return {
            "success": success,
            "model": request.model,
            "backend": request.backend,
            "message": f"{'Successfully unloaded' if success else 'Failed to unload'} {request.model}",
            "cached_models": model_manager.get_status()["cache_size"]
        }

    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )


@app.get("/api/models/status", response_model=ModelStatusResponse, tags=["Model Management"])
async def get_model_status():
    """
    Get current model manager status.

    Returns information about:
    - Currently active model
    - Cached models
    - Supported backends
    - Usage statistics
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    status_info = model_manager.get_status()

    return ModelStatusResponse(
        current_model=status_info["current_model"],
        current_backend=status_info["current_backend"],
        is_ready=status_info["is_ready"],
        cached_models=status_info["cached_models"],
        cache_size=status_info["cache_size"],
        supported_backends=status_info["supported_backends"]
    )


@app.get("/api/models/list/{backend}", tags=["Model Management"])
async def list_backend_models(backend: str = "ollama"):
    """
    List available models from a specific backend.

    **Usage:**
    Query Ollama, Groq, or other backends for available models.

    **Example:**
    ```
    GET /api/models/list/ollama
    ```
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )

    try:
        models = await model_manager.get_available_models(backend)

        return {
            "backend": backend,
            "models": models,
            "count": len(models),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing models from {backend}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


# ============================================================================
# Translation Endpoints
# ============================================================================

@app.post("/api/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_text(request: TranslateRequest):
    """
    Translate text to a single target language

    **Features:**
    - Auto language detection if source_language not provided
    - Multiple backend support (Ollama, Groq, etc.)
    - Quality settings (fast, balanced, quality)
    - Dynamic model switching via RuntimeModelManager
    """
    backend_name = request.model.lower()

    # Use RuntimeModelManager for dynamic model switching
    translator = None
    if backend_name == "ollama" and model_manager is not None:
        translator = await model_manager.get_current_translator()

    # Fall back to static backends if RuntimeModelManager not available
    if translator is None:
        if backend_name not in translator_backends:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backend '{backend_name}' not available. Available: {list(translator_backends.keys())}"
            )
        translator = translator_backends[backend_name]

    start_time = time.time()

    try:
        result = await translator.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
        )

        processing_time = time.time() - start_time

        # Extract translated text from result dict
        if isinstance(result, dict):
            translated_text = result.get("translated_text", str(result))
            confidence = result.get("confidence", 0.9)
            actual_processing_time = result.get("metadata", {}).get("processing_time", processing_time)
        else:
            translated_text = str(result)
            confidence = 0.9
            actual_processing_time = processing_time

        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language or "auto",
            target_language=request.target_language,
            confidence=confidence,
            processing_time=actual_processing_time,
            model_used=translator.config.model,
            backend_used=backend_name
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )

@app.post("/api/translate/multi", response_model=MultiLanguageResponse, tags=["Translation"])
async def translate_multi_language(request: MultiLanguageRequest):
    """
    Translate text to multiple target languages in a single request

    **Features:**
    - Translate to multiple languages simultaneously
    - Optimized for batch processing
    - Returns individual results for each language

    **Example:**
    ```json
    {
        "text": "Hello, how are you?",
        "target_languages": ["es", "fr", "de"],
        "model": "ollama"
    }
    ```
    """
    backend_name = request.model.lower()

    if backend_name not in translator_backends:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Backend '{backend_name}' not available. Available: {list(translator_backends.keys())}"
        )

    translator = translator_backends[backend_name]

    logger.info(f"Multi-language translation: {len(request.target_languages)} languages for '{request.text[:50]}...'")

    start_time = time.time()
    translations = {}

    # Translate to each target language
    for target_lang in request.target_languages:
        try:
            lang_start = time.time()
            result = await translator.translate(
                text=request.text,
                source_language=request.source_language,
                target_language=target_lang,
            )
            lang_time = time.time() - lang_start

            translations[target_lang] = {
                "translated_text": result,
                "confidence": 0.9,
                "processing_time": lang_time,
                "model_used": translator.config.model,
                "backend_used": backend_name,
            }
        except Exception as e:
            logger.error(f"Translation to {target_lang} failed: {e}")
            translations[target_lang] = {
                "error": str(e),
                "processing_time": 0.0,
            }

    total_time = time.time() - start_time

    logger.info(f"Multi-language translation completed: {len(translations)}/{len(request.target_languages)} successful in {total_time:.3f}s")

    return MultiLanguageResponse(
        source_text=request.text,
        source_language=request.source_language or "auto",
        model_requested=request.model,
        quality=request.quality,
        total_processing_time=total_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
        translations=translations
    )

@app.post("/api/detect", response_model=LanguageDetectionResponse, tags=["Language Detection"])
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of input text

    Uses langdetect library for language detection
    """
    from langdetect import detect, detect_langs

    try:
        # Detect language
        detected_lang = detect(request.text)

        # Get confidence scores for alternatives
        lang_probs = detect_langs(request.text)
        alternatives = [{"lang": str(lp.lang), "confidence": lp.prob} for lp in lang_probs]

        return LanguageDetectionResponse(
            language=detected_lang,
            confidence=alternatives[0]["confidence"] if alternatives else 0.5,
            alternatives=alternatives[:5]  # Top 5 alternatives
        )

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        # Return default
        return LanguageDetectionResponse(
            language="en",
            confidence=0.5,
            alternatives=[{"en": 0.5}]
        )


# ============================================================================
# Batch Translation Endpoint
# ============================================================================

class BatchTranslateRequest(BaseModel):
    """Batch translation request containing multiple translation requests"""
    requests: List[TranslateRequest] = Field(..., min_length=1, max_length=100)


class BatchTranslateResponse(BaseModel):
    """Batch translation response with all results"""
    translations: List[TranslationResponse]
    total_processing_time: float


@app.post("/api/translate/batch", response_model=BatchTranslateResponse, tags=["Translation"])
async def translate_batch(request: BatchTranslateRequest):
    """
    Batch translate multiple texts in a single request.

    **Features:**
    - Process up to 100 translation requests in a batch
    - Optimized for throughput
    - Returns individual results for each translation

    **Example:**
    ```json
    {
        "requests": [
            {"text": "Hello", "target_language": "es", "model": "ollama"},
            {"text": "World", "target_language": "fr", "model": "ollama"}
        ]
    }
    ```
    """
    start_time = time.time()
    translations: List[TranslationResponse] = []

    for req in request.requests:
        backend_name = req.model.lower()

        # Use RuntimeModelManager for dynamic model switching
        translator = None
        if backend_name == "ollama" and model_manager is not None:
            translator = await model_manager.get_current_translator()

        # Fall back to static backends if RuntimeModelManager not available
        if translator is None:
            if backend_name not in translator_backends:
                # Add error response for this item
                translations.append(TranslationResponse(
                    translated_text=f"[ERROR: Backend '{backend_name}' not available]",
                    source_language=req.source_language or "unknown",
                    target_language=req.target_language,
                    confidence=0.0,
                    processing_time=0.0,
                    model_used="none",
                    backend_used="none"
                ))
                continue
            translator = translator_backends[backend_name]

        req_start = time.time()
        try:
            result = await translator.translate(
                text=req.text,
                source_language=req.source_language,
                target_language=req.target_language,
            )
            req_time = time.time() - req_start

            # Extract translated text from result dict
            if isinstance(result, dict):
                translated_text = result.get("translated_text", str(result))
                confidence = result.get("confidence", 0.9)
            else:
                translated_text = str(result)
                confidence = 0.9

            translations.append(TranslationResponse(
                translated_text=translated_text,
                source_language=req.source_language or "auto",
                target_language=req.target_language,
                confidence=confidence,
                processing_time=req_time,
                model_used=translator.config.model,
                backend_used=backend_name
            ))

        except Exception as e:
            logger.error(f"Batch translation item failed: {e}")
            translations.append(TranslationResponse(
                translated_text=f"[ERROR: {str(e)}]",
                source_language=req.source_language or "unknown",
                target_language=req.target_language,
                confidence=0.0,
                processing_time=time.time() - req_start,
                model_used="none",
                backend_used=backend_name
            ))

    total_time = time.time() - start_time
    logger.info(f"Batch translation completed: {len(translations)} items in {total_time:.3f}s")

    return BatchTranslateResponse(
        translations=translations,
        total_processing_time=total_time
    )


# ============================================================================
# Quality Assessment Endpoint
# ============================================================================

class QualityRequest(BaseModel):
    """Quality assessment request"""
    original: str = Field(..., description="Original text")
    translated: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")


class QualityResponse(BaseModel):
    """Quality assessment response"""
    score: float = Field(..., description="Quality score (0.0-1.0)")
    method: str = Field(..., description="Scoring method used")
    original_length: int = Field(..., description="Length of original text")
    translated_length: int = Field(..., description="Length of translated text")
    length_ratio: float = Field(..., description="Ratio of translated to original length")
    timestamp: str = Field(..., description="Assessment timestamp")


@app.post("/api/quality", response_model=QualityResponse, tags=["Translation"])
async def assess_quality(request: QualityRequest):
    """
    Assess the quality of a translation.

    **Scoring Method:**
    Uses SequenceMatcher to compute similarity between original and translated text.
    This is a basic heuristic - higher similarity may indicate under-translation,
    while reasonable deviation indicates proper translation.

    **Note:** This is a basic quality assessment. For production use, consider
    integrating more sophisticated metrics like BLEU, METEOR, or neural quality estimation.

    **Example:**
    ```json
    {
        "original": "Hello world",
        "translated": "Hola mundo",
        "source_language": "en",
        "target_language": "es"
    }
    ```
    """
    original = request.original.strip()
    translated = request.translated.strip()

    # Basic quality scoring using SequenceMatcher
    # Note: For translations, high similarity might indicate under-translation
    # This is inverted for the quality score
    similarity = SequenceMatcher(None, original.lower(), translated.lower()).ratio()

    # Length ratio analysis (translations often change length)
    orig_len = len(original)
    trans_len = len(translated)
    length_ratio = trans_len / orig_len if orig_len > 0 else 0.0

    # Quality heuristic: Good translations should have reasonable length ratio
    # and not be too similar (which might indicate no translation happened)
    if similarity > 0.95:
        # Very similar - might be same language or copy
        score = 0.3
    elif similarity > 0.7:
        # Somewhat similar - might be related languages or partial translation
        score = 0.6
    elif 0.2 <= length_ratio <= 3.0:
        # Reasonable length ratio, different text - likely translated
        score = 0.85 + (1.0 - abs(1.0 - length_ratio) * 0.1)
        score = min(score, 1.0)
    else:
        # Unusual length ratio - might have issues
        score = 0.5

    return QualityResponse(
        score=round(score, 3),
        method="sequence_matcher_heuristic",
        original_length=orig_len,
        translated_length=trans_len,
        length_ratio=round(length_ratio, 3),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


# ============================================================================
# Realtime Translation Session Endpoints
# ============================================================================

class RealtimeSessionConfig(BaseModel):
    """Configuration for a realtime translation session"""
    source_language: Optional[str] = Field(None, description="Source language (auto-detect if None)")
    target_language: str = Field("en", description="Default target language")
    model: str = Field("ollama", description="Translation model/backend")
    quality: str = Field("balanced", description="Translation quality: fast, balanced, quality")


class RealtimeStartResponse(BaseModel):
    """Response for starting a realtime session"""
    session_id: str
    status: str
    created_at: str
    config: Dict[str, Any]


@app.post("/api/realtime/start", response_model=RealtimeStartResponse, tags=["Realtime"])
async def start_realtime_session(config: Optional[RealtimeSessionConfig] = None):
    """
    Start a new realtime translation session.

    **Usage:**
    Creates a session that maintains context for streaming translations.
    The session ID should be used for subsequent translation requests.

    **Example:**
    ```json
    {
        "source_language": "en",
        "target_language": "es",
        "model": "ollama",
        "quality": "balanced"
    }
    ```
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Use default config if none provided
    if config is None:
        config = RealtimeSessionConfig()

    session_data = {
        "session_id": session_id,
        "created_at": created_at.isoformat(),
        "last_activity": created_at.isoformat(),
        "config": {
            "source_language": config.source_language,
            "target_language": config.target_language,
            "model": config.model,
            "quality": config.quality,
        },
        "translation_count": 0,
        "total_characters": 0,
        "context": [],  # Rolling context for better translations
    }

    active_sessions[session_id] = session_data
    logger.info(f"Started realtime session: {session_id}")

    return RealtimeStartResponse(
        session_id=session_id,
        status="started",
        created_at=created_at.isoformat(),
        config=session_data["config"]
    )


class RealtimeTranslateRequest(BaseModel):
    """Request for realtime translation within a session"""
    session_id: str = Field(..., description="Session ID from /api/realtime/start")
    text: str = Field(..., description="Text to translate")
    target_language: Optional[str] = Field(None, description="Override target language for this request")


class RealtimeTranslateResponse(BaseModel):
    """Response for realtime translation"""
    session_id: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    processing_time: float
    translation_count: int


@app.post("/api/realtime/translate", response_model=RealtimeTranslateResponse, tags=["Realtime"])
async def realtime_translate(request: RealtimeTranslateRequest):
    """
    Translate text within an active realtime session.

    **Usage:**
    Uses the session context for consistent translations.
    The target language can be overridden per-request.

    **Example:**
    ```json
    {
        "session_id": "uuid-from-start",
        "text": "Hello, how are you?",
        "target_language": "es"
    }
    ```
    """
    session_id = request.session_id

    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Start a session with /api/realtime/start"
        )

    session = active_sessions[session_id]
    session["last_activity"] = datetime.now(timezone.utc).isoformat()

    # Use request target language or session default
    target_language = request.target_language or session["config"]["target_language"]
    source_language = session["config"]["source_language"]
    backend_name = session["config"]["model"].lower()

    # Use RuntimeModelManager for dynamic model switching
    translator = None
    if backend_name == "ollama" and model_manager is not None:
        translator = await model_manager.get_current_translator()

    # Fall back to static backends if RuntimeModelManager not available
    if translator is None:
        if backend_name not in translator_backends:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backend '{backend_name}' not available. Available: {list(translator_backends.keys())}"
            )
        translator = translator_backends[backend_name]

    start_time = time.time()

    try:
        result = await translator.translate(
            text=request.text,
            source_language=source_language,
            target_language=target_language,
        )
        processing_time = time.time() - start_time

        # Extract translated text from result dict
        if isinstance(result, dict):
            translated_text = result.get("translated_text", str(result))
            confidence = result.get("confidence", 0.9)
        else:
            translated_text = str(result)
            confidence = 0.9

        # Update session statistics
        session["translation_count"] += 1
        session["total_characters"] += len(request.text)

        # Add to rolling context (keep last 5 translations)
        session["context"].append({
            "original": request.text,
            "translated": translated_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if len(session["context"]) > 5:
            session["context"] = session["context"][-5:]

        return RealtimeTranslateResponse(
            session_id=session_id,
            translated_text=translated_text,
            source_language=source_language or "auto",
            target_language=target_language,
            confidence=confidence,
            processing_time=processing_time,
            translation_count=session["translation_count"]
        )

    except Exception as e:
        logger.error(f"Realtime translation failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


class RealtimeStopRequest(BaseModel):
    """Request to stop a realtime session"""
    session_id: str = Field(..., description="Session ID to stop")


class RealtimeStopResponse(BaseModel):
    """Response for stopping a realtime session"""
    session_id: str
    status: str
    duration_seconds: float
    translation_count: int
    total_characters: int


@app.post("/api/realtime/stop", response_model=RealtimeStopResponse, tags=["Realtime"])
async def stop_realtime_session(request: RealtimeStopRequest):
    """
    Stop and clean up a realtime translation session.

    **Usage:**
    Ends the session and returns statistics about the session.

    **Example:**
    ```json
    {
        "session_id": "uuid-from-start"
    }
    ```
    """
    session_id = request.session_id

    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found"
        )

    session = active_sessions.pop(session_id)

    # Calculate session duration
    created_at = datetime.fromisoformat(session["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    duration = (datetime.now(timezone.utc) - created_at).total_seconds()

    logger.info(f"Stopped realtime session {session_id}: {session['translation_count']} translations, {duration:.1f}s duration")

    return RealtimeStopResponse(
        session_id=session_id,
        status="stopped",
        duration_seconds=round(duration, 2),
        translation_count=session["translation_count"],
        total_characters=session["total_characters"]
    )


# ============================================================================
# V3 API Endpoints - Simplified "prompt-in, translation-out" contract
# ============================================================================

@app.post("/api/v3/translate", response_model=PromptTranslateResponse, tags=["V3 Translation"])
async def translate_prompt(request: PromptTranslateRequest):
    """
    V3 Simplified Translation - Send prompt directly to LLM.

    **Design Philosophy:**
    The translation service is DUMB. It receives a complete prompt
    (with context, glossary, target language already embedded) and
    returns the LLM's response.

    All intelligence (context windows, glossary management, speaker tracking)
    stays in the orchestration service.

    **Supports Dynamic Model Switching:**
    Uses RuntimeModelManager for Ollama backend - model can be switched at
    runtime via /api/models/switch without restarting the service.

    **Example Request:**
    ```json
    {
      "prompt": "You are a translator...\\n\\nGlossary:\\n- API = API\\n\\nTranslate to Spanish:\\nHello world\\n\\nTranslation:",
      "backend": "ollama",
      "max_tokens": 256,
      "temperature": 0.3
    }
    ```

    **Example Response:**
    ```json
    {
      "text": "Hola mundo",
      "processing_time_ms": 150.5,
      "backend_used": "ollama-local",
      "model_used": "gemma3:4b"
    }
    ```
    """
    backend_name = request.backend.lower()

    # Use RuntimeModelManager for dynamic model switching
    translator = None
    current_model = None
    if backend_name == "ollama" and model_manager is not None:
        translator = await model_manager.get_current_translator()
        current_model = model_manager.current_model

    # Fall back to static backends if RuntimeModelManager not available
    if translator is None:
        if backend_name not in translator_backends:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backend '{backend_name}' not available. Available: {list(translator_backends.keys())}"
            )
        translator = translator_backends[backend_name]

    try:
        result = await translator.generate_from_prompt(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Translation failed - no result from backend"
            )

        return PromptTranslateResponse(
            text=result["text"],
            processing_time_ms=result["processing_time_ms"],
            backend_used=result["backend_used"],
            model_used=current_model or result["model_used"],  # Use RuntimeModelManager's model
            tokens_used=result.get("tokens_used")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"V3 Translation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@app.post("/api/v3/translate/stream", tags=["V3 Translation"])
async def translate_prompt_stream(request: PromptTranslateRequest):
    """
    V3 Streaming Translation - Stream response chunks via SSE.

    Returns Server-Sent Events with chunks as they're generated.

    **SSE Format:**
    ```
    data: {"chunk": "Hola", "done": false}
    data: {"chunk": " mundo", "done": false}
    data: {"done": true, "processing_time_ms": 150.5, "backend_used": "ollama-local", "model_used": "gemma3:4b"}
    ```
    """
    from fastapi.responses import StreamingResponse
    import json

    backend_name = request.backend.lower()

    # Use RuntimeModelManager for dynamic model switching
    translator = None
    current_model = None
    if backend_name == "ollama" and model_manager is not None:
        translator = await model_manager.get_current_translator()
        current_model = model_manager.current_model

    # Fall back to static backends if RuntimeModelManager not available
    if translator is None:
        if backend_name not in translator_backends:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backend '{backend_name}' not available. Available: {list(translator_backends.keys())}"
            )
        translator = translator_backends[backend_name]

    async def generate():
        try:
            async for chunk_data in translator.generate_from_prompt_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt
            ):
                yield f"data: {json.dumps(chunk_data)}\n\n"
        except Exception as e:
            logger.error(f"V3 Streaming translation failed: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# Test Endpoint
# ============================================================================

@app.get("/api/test", tags=["Testing"])
async def api_test():
    """Simple test endpoint to verify service is running"""
    return {
        "status": "ok",
        "service": "translation",
        "backends": list(translator_backends.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "5003"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"🚀 Starting FastAPI Translation Service on {host}:{port}")
    logger.info(f"📖 API docs available at http://localhost:{port}/docs")
    logger.info(f"📖 ReDoc available at http://localhost:{port}/redoc")

    uvicorn.run(
        "api_server_fastapi:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
