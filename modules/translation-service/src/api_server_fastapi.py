#!/usr/bin/env python3
"""
LiveTranslate Translation Service - FastAPI Version
OpenAI-Compatible Translation with Ollama, Groq, vLLM, etc.
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
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
        timestamp=datetime.utcnow().isoformat()
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
        "timestamp": datetime.utcnow().isoformat()
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
            "timestamp": datetime.utcnow().isoformat()
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
        timestamp=datetime.utcnow().isoformat(),
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
        "timestamp": datetime.utcnow().isoformat()
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
