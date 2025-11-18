#!/usr/bin/env python3
"""
Translation Service API Server

Flask-based REST API with WebSocket support for real-time translation streaming.
Provides endpoints for translation, language detection, and service management.
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid

from translation_service import TranslationService, TranslationRequest, create_translation_service
from prompt_manager import PromptManager, PromptTemplate, PromptPerformanceMetric
from vllm_server_simple import SimpleVLLMTranslationServer
from nllb_translator import get_nllb_translator
from llama_translator import get_llama_translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# SocketIO for real-time streaming
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global translation service instance
translation_service: Optional[TranslationService] = None

# Global prompt manager instance
prompt_manager: Optional[PromptManager] = None

# Active sessions
active_sessions: Dict[str, Dict] = {}

# Global internal vLLM server instance
internal_vllm_server: Optional[SimpleVLLMTranslationServer] = None

# Global NLLB translator instance
nllb_translator = None

# Global Llama translator instance
llama_translator = None

def validate_local_model(model_path: str) -> tuple:
    """
    Validate that a local model directory contains required files
    
    Returns:
        (is_valid, message): Boolean indicating if valid, and descriptive message
    """
    try:
        model_path = Path(model_path)
        
        if not model_path.exists():
            return False, f"Model directory does not exist: {model_path}"
        
        if not model_path.is_dir():
            return False, f"Model path is not a directory: {model_path}"
        
        # Check for required model files
        model_files = [f for f in model_path.iterdir() if f.suffix in ['.bin', '.safetensors']]
        tokenizer_files = [f for f in model_path.iterdir() if 'tokenizer' in f.name.lower()]
        
        missing_files = []
        
        # Check config.json
        if not (model_path / "config.json").exists():
            missing_files.append("config.json")
        
        # Check for model weights
        if not model_files:
            missing_files.append("model weights (.bin or .safetensors files)")
        
        # Check for tokenizer files
        if not tokenizer_files:
            missing_files.append("tokenizer files")
        
        if missing_files:
            return False, f"Missing required files: {', '.join(missing_files)}"
        
        return True, f"Model directory is valid: {len(model_files)} model files, {len(tokenizer_files)} tokenizer files"
        
    except Exception as e:
        return False, f"Error validating model: {str(e)}"

# Global configuration storage
current_config: Dict[str, Any] = {
    "backend": "vllm",
    "model_name": "./models/Llama-3.1-8B-Instruct",  # Local Llama model path
    "temperature": 0.3,
    "max_tokens": 1024,
    "gpu_enable": True,
    "use_internal_vllm": True,
    "internal_port": 8010,
    "version": "1.0.0"
}

async def initialize_internal_vllm() -> Optional[SimpleVLLMTranslationServer]:
    """Initialize and start the internal vLLM server"""
    global internal_vllm_server, current_config
    
    try:
        # Use configuration from global config (with environment variable fallbacks)
        # Default to local Llama model path
        default_model = os.getenv("TRANSLATION_MODEL", current_config.get("model_name", "./models/Llama-3.1-8B-Instruct"))
        logger.info(f"🎯 Using model: {default_model}")
        vllm_port = int(os.getenv("VLLM_INTERNAL_PORT", current_config.get("internal_port", 8010)))
        vllm_host = os.getenv("VLLM_INTERNAL_HOST", "0.0.0.0")
        
        # Validate local model if it's a local path
        if default_model.startswith('./') or default_model.startswith('/'):
            is_valid, validation_message = validate_local_model(default_model)
            if not is_valid:
                logger.error(f"❌ Model validation failed: {validation_message}")
                logger.error(f"💡 Please ensure your model is downloaded to: {default_model}")
                logger.error("💡 The model directory should contain: config.json, tokenizer files, and model weights")
                return None
            else:
                logger.info(f"✅ Model validation passed: {validation_message}")
        
        # Update current config with actual values
        current_config.update({
            "model_name": default_model,
            "internal_port": vllm_port,
            "internal_host": vllm_host
        })
        
        logger.info("🔧 Initializing internal vLLM server...")
        logger.info(f"   Model: {default_model}")
        logger.info(f"   Host: {vllm_host}:{vllm_port}")
        
        # Create internal vLLM server instance
        internal_vllm_server = SimpleVLLMTranslationServer(
            host=vllm_host,
            port=vllm_port,
            model_name=default_model
        )
        
        # The server automatically starts model loading in background thread
        # and starts HTTP/WebSocket servers
        logger.info("✅ Internal vLLM server initialized successfully")
        logger.info(f"   REST API: http://{vllm_host}:{vllm_port}")
        logger.info(f"   Health Check: http://{vllm_host}:{vllm_port}/health")
        logger.info("   Model loading in background...")
        
        return internal_vllm_server
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize internal vLLM server: {e}")
        logger.error("   This may be due to insufficient GPU memory or missing dependencies")
        logger.error("   Will fallback to external services...")
        return None

async def initialize_service():
    """Initialize the translation service with internal vLLM server"""
    global translation_service, prompt_manager
    try:
        logger.info("🚀 Starting LiveTranslate Translation Service with internal vLLM...")
        
        # Get the model path to decide which approach to use
        default_model = os.getenv("TRANSLATION_MODEL", current_config.get("model_name", "./models/Llama-3.1-8B-Instruct"))
        logger.info(f"🎯 Primary model: {default_model}")
        
        # Try direct transformers approach first (more reliable)
        global llama_translator
        if "llama" in default_model.lower() or "meta-llama" in default_model.lower():
            logger.info("🔄 Llama model detected - using direct transformers implementation")
            logger.info("💡 Using transformers.pipeline for better compatibility")
            
            device = "cuda" if current_config.get("gpu_enable") else "cpu"
            try:
                llama_translator = get_llama_translator(default_model, device)
                
                if llama_translator.is_ready:
                    logger.info("✅ Llama translator initialized successfully")
                    # Create a minimal translation service that uses the Llama translator
                    translation_service = await create_translation_service({"backend": "llama_direct"})
                else:
                    logger.error("❌ Failed to initialize Llama translator")
                    llama_translator = None
            except Exception as llama_error:
                logger.error(f"❌ Llama translator error: {llama_error}")
                llama_translator = None
        
        # If Llama transformer failed, try NLLB as fallback (skip vLLM for now)
        if llama_translator is None or not llama_translator.is_ready:
            logger.info("🔄 Llama transformers failed, trying NLLB as fallback...")
            logger.info("💡 Skipping vLLM due to compatibility issues")
            
            # Try NLLB model as fallback
            nllb_model_path = "./models/nllb-200-distilled-1.3B-8bit"
            
            # Try NLLB as fallback translator
            logger.info("🔄 Trying NLLB model as backup translator...")
            logger.info("💡 NLLB models use encoder-decoder architecture which requires transformers")
            
            global nllb_translator
            device = "cuda" if current_config.get("gpu_enable") else "cpu"
            try:
                nllb_translator = get_nllb_translator(nllb_model_path, device)
                if nllb_translator.is_ready:
                    logger.info("✅ NLLB backup translator initialized successfully")
                else:
                    logger.warning("⚠️ NLLB backup translator failed to initialize")
            except Exception as nllb_error:
                logger.warning(f"⚠️ NLLB backup translator error: {nllb_error}")
                nllb_translator = None
            
            # Initialize translation service with available translators
            if nllb_translator and nllb_translator.is_ready:
                logger.info("📱 Using NLLB translator as fallback")
                translation_service = await create_translation_service({"backend": "nllb_direct"})
            else:
                # Try local Ollama as final fallback
                logger.warning("⚠️ Trying local Ollama as final fallback...")
                try:
                    config = {
                        "backend": "ollama",
                        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    }
                    translation_service = await create_translation_service(config)
                    logger.info("✅ Initialized with local Ollama fallback")
                except Exception as ollama_error:
                    logger.error(f"❌ Local Ollama fallback also failed: {ollama_error}")
                    logger.error("💡 All translation backends failed")
                    raise RuntimeError("Translation service initialization failed: No working backends available")
            
        logger.info("Translation service initialized successfully")
        
        # Initialize prompt manager
        prompt_storage_path = os.getenv("PROMPT_STORAGE_PATH", "./prompts")
        prompt_manager = PromptManager(storage_path=prompt_storage_path, enable_analytics=True)
        logger.info("Prompt manager initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Initialize service on startup
with app.app_context():
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialize_service())
    except Exception as e:
        logger.error(f"Failed to initialize service during startup: {e}")

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if translation_service is None:
        return jsonify({"status": "error", "message": "Service not initialized"}), 503
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# API health check endpoint (compatible with other services)
@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check endpoint for service integration"""
    if translation_service is None:
        return jsonify({"status": "error", "message": "Service not initialized"}), 503
    
    return jsonify({
        "status": "healthy",
        "service": "translation",
        "backend": os.getenv("INFERENCE_BACKEND", "auto"),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Service status endpoint
@app.route('/api/status', methods=['GET'])
async def get_status():
    """Get detailed service status"""
    if translation_service is None:
        return jsonify({"status": "error", "message": "Service not initialized"}), 503
    
    try:
        status = asyncio.run(translation_service.get_service_status())
        return jsonify({
            "status": "ok",
            "service": "translation",
            "backend": os.getenv("INFERENCE_BACKEND", "auto"),
            "backends": status,
            "active_sessions": len(active_sessions),
            "endpoints": {
                "translate": "/translate",
                "stream": "/translate/stream",
                "continuity": "/translate/continuity",
                "detect_language": "/detect_language",
                "languages": "/languages",
                "sessions": "/sessions",
                "context": "/sessions/<session_id>/context",
                "health": "/api/health",
                "status": "/api/status"
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Device information endpoint
@app.route('/api/device-info', methods=['GET'])
async def get_device_info():
    """Get current device information (CPU/GPU) and acceleration status"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        # Get service status which includes device information
        status = asyncio.run(translation_service.get_service_status())
        backend = os.getenv("INFERENCE_BACKEND", "auto")
        
        # Determine device type and acceleration based on backend
        device = "unknown"
        device_type = "unknown"
        acceleration = "unknown"
        
        # Check for GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Determine device based on backend and GPU availability
        if backend == "vllm" and gpu_available:
            device = "gpu"
            device_type = "gpu"
            acceleration = "cuda"
        elif backend == "triton" and gpu_available:
            device = "gpu"
            device_type = "gpu"
            acceleration = "cuda"
        elif backend == "ollama":
            if gpu_available:
                device = "gpu"
                device_type = "gpu"
                acceleration = "cuda"
            else:
                device = "cpu"
                device_type = "cpu"
                acceleration = "none"
        else:
            device = "cpu"
            device_type = "cpu"
            acceleration = "none"
        
        # Additional device details
        device_details = {}
        if gpu_available:
            try:
                import torch
                device_details = {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
                }
            except Exception:
                device_details = {"cuda_available": False}
        
        return jsonify({
            "device": device.lower(),
            "device_type": device_type,
            "status": "healthy" if status else "unavailable",
            "acceleration": acceleration,
            "details": {
                "backend": backend,
                "gpu_available": gpu_available,
                "backends_status": status or {},
                **device_details
            },
            "service_info": {
                "version": "1.0.0",
                "backend": backend,
                "active_sessions": len(active_sessions)
            }
        })
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return jsonify({"error": str(e)}), 500

# Configuration endpoints
@app.route('/api/config', methods=['GET'])
async def get_configuration():
    """Get current service configuration"""
    global current_config
    
    try:
        # Add runtime information to configuration
        runtime_config = current_config.copy()
        runtime_config.update({
            "service_status": "running" if translation_service else "initializing",
            "internal_vllm_ready": internal_vllm_server.is_model_ready if internal_vllm_server else False,
            "active_sessions": len(active_sessions),
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "status": "success",
            "configuration": runtime_config
        })
    
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['POST'])
async def update_configuration():
    """Update service configuration"""
    global current_config
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        # Track what changes were made
        changes_applied = {}
        
        # Update allowed configuration parameters
        allowed_params = ["backend", "model_name", "temperature", "max_tokens", "gpu_enable"]
        for key, value in data.items():
            if key in allowed_params:
                old_value = current_config.get(key)
                current_config[key] = value
                changes_applied[key] = {"old": old_value, "new": value}
                logger.info(f"Configuration updated: {key} = {value}")
        
        # Update timestamp
        current_config["last_updated"] = datetime.now().isoformat()
        
        return jsonify({
            "status": "success",
            "message": "Configuration updated successfully",
            "changes_applied": changes_applied,
            "configuration": current_config
        })
    
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return jsonify({"error": str(e)}), 500

# Translation endpoint
@app.route('/translate', methods=['POST'])
async def translate_text():
    """Translate text using the translation service"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        # Create translation request
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=data.get('source_language', 'auto'),
            target_language=data.get('target_language', 'en'),
            session_id=data.get('session_id'),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=data.get('preserve_formatting', True),
            context=data.get('context')
        )
        
        # Check if we should use Llama translator directly
        if llama_translator and llama_translator.is_ready:
            logger.info("Using Llama transformer for direct translation")
            try:
                llama_result = llama_translator.translate(
                    text=translation_request.text,
                    source_lang=translation_request.source_language if translation_request.source_language != 'auto' else 'en',
                    target_lang=translation_request.target_language
                )
                
                if "error" not in llama_result:
                    logger.info(f"Llama translation result: {llama_result['translated_text']}")
                    return jsonify({
                        "translated_text": llama_result['translated_text'],
                        "source_language": llama_result.get('source_language', translation_request.source_language or 'auto'),
                        "target_language": llama_result.get('target_language', translation_request.target_language),
                        "confidence": llama_result.get('confidence_score', 0.9),
                        "confidence_score": llama_result.get('confidence_score', 0.9),
                        "processing_time": llama_result.get('processing_time', 0.0),
                        "backend_used": llama_result.get('backend_used', 'llama_transformers'),
                        "session_id": translation_request.session_id,
                        "timestamp": llama_result.get('timestamp', datetime.utcnow().isoformat()),
                        "model_used": llama_result.get('model_used', 'llama-transformers')
                    })
                else:
                    logger.warning(f"Llama translation error: {llama_result['error']}")
            except Exception as llama_error:
                logger.error(f"Llama translator failed: {llama_error}")
        
        # Check if we should use NLLB translator as fallback
        elif nllb_translator and nllb_translator.is_ready:
            logger.info("Using NLLB translator for direct translation")
            try:
                nllb_result = nllb_translator.translate(
                    text=translation_request.text,
                    source_lang=translation_request.source_language if translation_request.source_language != 'auto' else 'en',
                    target_lang=translation_request.target_language
                )
                
                if "error" not in nllb_result:
                    logger.info(f"NLLB translation result: {nllb_result['translated_text']}")
                    return jsonify({
                        "translated_text": nllb_result['translated_text'],
                        "source_language": nllb_result.get('source_language', translation_request.source_language),
                        "target_language": nllb_result.get('target_language', translation_request.target_language),
                        "confidence_score": nllb_result.get('confidence_score', 0.9),
                        "processing_time": nllb_result.get('processing_time', 0.0),
                        "backend_used": nllb_result.get('backend_used', 'nllb_transformers'),
                        "session_id": translation_request.session_id,
                        "timestamp": nllb_result.get('timestamp', datetime.utcnow().isoformat()),
                        "model_used": nllb_result.get('model_used', current_config.get('model_name', 'nllb'))
                    })
                else:
                    logger.warning(f"NLLB translation error: {nllb_result['error']}")
            except Exception as nllb_error:
                logger.error(f"NLLB translator failed: {nllb_error}")
        
        # Fallback to standard translation service
        result = asyncio.run(translation_service.translate(translation_request))
        
        return jsonify({
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time,
            "backend_used": result.backend_used,
            "session_id": result.session_id,
            "timestamp": result.timestamp
        })
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return jsonify({"error": str(e)}), 500

# Simple test endpoint
@app.route('/api/test', methods=['GET'])
def api_test():
    """Simple test endpoint"""
    return jsonify({
        "status": "ok",
        "service": "translation-service",
        "translation_service_initialized": translation_service is not None,
        "fallback_mode": getattr(translation_service, 'fallback_mode', False) if translation_service else None
    })

# API endpoint for compatibility with orchestration service
@app.route('/api/translate', methods=['POST'])
def api_translate_text():
    """API translate endpoint for orchestration service compatibility"""
    global translation_service
    
    try:
        # Check if service is initialized
        if translation_service is None:
            logger.error("Translation service not initialized")
            return jsonify({"error": "Translation service not initialized"}), 503
            
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        logger.info(f"API translate request: {data}")
        
        # Create translation request
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=data.get('source_language', 'auto'),
            target_language=data.get('target_language', 'en'),
            session_id=data.get('session_id'),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=data.get('preserve_formatting', True),
            context=data.get('context')
        )
        
        # Check if we should use Llama translator directly
        if llama_translator and llama_translator.is_ready:
            logger.info("Using Llama transformer for direct translation")
            try:
                llama_result = llama_translator.translate(
                    text=translation_request.text,
                    source_lang=translation_request.source_language if translation_request.source_language != 'auto' else 'en',
                    target_lang=translation_request.target_language
                )
                
                if "error" not in llama_result:
                    logger.info(f"Llama translation result: {llama_result['translated_text']}")
                    return jsonify({
                        "translated_text": llama_result['translated_text'],
                        "source_language": llama_result.get('source_language', translation_request.source_language or 'auto'),
                        "target_language": llama_result.get('target_language', translation_request.target_language),
                        "confidence": llama_result.get('confidence_score', 0.9),
                        "confidence_score": llama_result.get('confidence_score', 0.9),
                        "processing_time": llama_result.get('processing_time', 0.0),
                        "backend_used": llama_result.get('backend_used', 'llama_transformers'),
                        "session_id": translation_request.session_id,
                        "timestamp": llama_result.get('timestamp', datetime.utcnow().isoformat()),
                        "model_used": llama_result.get('model_used', 'llama-transformers')
                    })
                else:
                    logger.warning(f"Llama translation error: {llama_result['error']}")
            except Exception as llama_error:
                logger.error(f"Llama translator failed: {llama_error}")
        
        # Check if we should use NLLB translator as fallback
        elif nllb_translator and nllb_translator.is_ready:
            logger.info("Using NLLB translator for direct translation")
            try:
                nllb_result = nllb_translator.translate(
                    text=translation_request.text,
                    source_lang=translation_request.source_language if translation_request.source_language != 'auto' else 'en',
                    target_lang=translation_request.target_language
                )
                
                if "error" not in nllb_result:
                    logger.info(f"NLLB translation result: {nllb_result['translated_text']}")
                    return jsonify({
                        "translated_text": nllb_result['translated_text'],
                        "source_language": nllb_result.get('source_language', translation_request.source_language),
                        "target_language": nllb_result.get('target_language', translation_request.target_language),
                        "confidence_score": nllb_result.get('confidence_score', 0.9),
                        "processing_time": nllb_result.get('processing_time', 0.0),
                        "backend_used": nllb_result.get('backend_used', 'nllb_transformers'),
                        "session_id": translation_request.session_id,
                        "timestamp": nllb_result.get('timestamp', datetime.utcnow().isoformat()),
                        "model_used": nllb_result.get('model_used', current_config.get('model_name', 'nllb'))
                    })
                else:
                    logger.warning(f"NLLB translation error: {nllb_result['error']}")
            except Exception as nllb_error:
                logger.error(f"NLLB translator failed: {nllb_error}")
        
        # Fallback to standard translation service
        try:
            result = asyncio.run(translation_service.translate(translation_request))
            logger.info(f"Translation result: {result.translated_text}")
        except Exception as trans_error:
            logger.error(f"Translation execution failed: {trans_error}")
            return jsonify({"error": f"Translation failed: {str(trans_error)}"}), 500
        
        return jsonify({
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time,
            "backend_used": result.backend_used,
            "session_id": result.session_id,
            "timestamp": result.timestamp
        })
        
    except Exception as e:
        logger.error(f"API translate failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Streaming translation endpoint
@app.route('/translate/stream', methods=['POST'])
async def translate_stream():
    """Stream translation results in real-time"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        # Create streaming translation request
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=data.get('source_language', 'auto'),
            target_language=data.get('target_language', 'en'),
            session_id=data.get('session_id'),
            streaming=True,
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=data.get('preserve_formatting', True),
            context=data.get('context')
        )
        
        def generate_stream():
            """Generator for streaming response"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def stream_translation():
                    async for chunk in translation_service.translate_stream(translation_request):
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                for chunk in loop.run_until_complete(stream_translation()):
                    yield chunk
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: {\"done\": true}\n\n"
        
        return Response(
            generate_stream(),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming translation failed: {e}")
        return jsonify({"error": str(e)}), 500

# Translation with continuity and context management
@app.route('/translate/continuity', methods=['POST'])
async def translate_with_continuity():
    """
    Translate with context management and sentence buffering for streaming conversations.
    Receives clean text from whisper service (already deduplicated).
    """
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        if 'session_id' not in data:
            return jsonify({"error": "Missing 'session_id' field for context management"}), 400
        
        # Extract parameters
        clean_text = data['text']  # Already cleaned by whisper service
        session_id = data['session_id']
        target_language = data.get('target_language', 'en')
        source_language = data.get('source_language', 'auto')
        chunk_id = data.get('chunk_id')
        
        logger.info(f"Continuity translation request: session={session_id}, chunk={chunk_id}, clean_text='{clean_text[:50]}...'")
        
        # Process with continuity management (no deduplication needed)
        result = asyncio.run(translation_service.translate_with_continuity(
            text=clean_text,
            session_id=session_id,
            target_language=target_language,
            source_language=source_language,
            chunk_id=chunk_id
        ))
        
        # Add session context info
        context_info = translation_service.get_session_context_info(session_id)
        result['session_context'] = context_info
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Continuity translation failed: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

# Direct clean text processing from whisper service
@app.route('/api/process_clean_text', methods=['POST'])
async def process_clean_text():
    """
    Process clean, deduplicated text from the whisper service.
    This endpoint is called directly by the whisper service.
    """
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        if 'session_id' not in data:
            return jsonify({"error": "Missing 'session_id' field"}), 400
        
        # Extract parameters
        clean_text = data['text']
        session_id = data['session_id']
        target_language = data.get('target_language', 'en')
        source_language = data.get('source_language', 'auto')
        metadata = data.get('metadata', {})
        
        logger.info(f"Processing clean text from whisper service: session={session_id}, text='{clean_text[:50]}...'")
        
        # Process with continuity management
        result = asyncio.run(translation_service.translate_with_continuity(
            text=clean_text,
            session_id=session_id,
            target_language=target_language,
            source_language=source_language,
            chunk_id=metadata.get('inference_number')
        ))
        
        return jsonify({
            "status": "success",
            "translation_result": result,
            "metadata": metadata
        })
        
    except Exception as e:
        logger.error(f"Clean text processing failed: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

# Session context management endpoints
@app.route('/sessions/<session_id>/context', methods=['GET'])
async def get_session_context(session_id: str):
    """Get translation context information for a session"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        context_info = translation_service.get_session_context_info(session_id)
        return jsonify(context_info)
    except Exception as e:
        logger.error(f"Failed to get session context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>/context', methods=['DELETE'])
async def clear_session_context(session_id: str):
    """Clear translation context for a session"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        translation_service.clear_session_context(session_id)
        return jsonify({
            "status": "success",
            "message": f"Context cleared for session {session_id}",
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"Failed to clear session context: {e}")
        return jsonify({"error": str(e)}), 500

# Language detection endpoint
@app.route('/detect_language', methods=['POST'])
async def detect_language():
    """Detect the language of input text"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        language, confidence = asyncio.run(translation_service.detect_language(data['text']))
        
        return jsonify({
            "language": language,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return jsonify({"error": str(e)}), 500

# Supported languages endpoint
@app.route('/languages', methods=['GET'])
async def get_supported_languages():
    """Get list of supported languages"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        languages = asyncio.run(translation_service.get_supported_languages())
        return jsonify({
            "languages": languages,
            "count": len(languages),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        return jsonify({"error": str(e)}), 500

# Session management endpoints
@app.route('/sessions', methods=['POST'])
async def create_session():
    """Create a new translation session"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        session_config = asyncio.run(translation_service.create_session(
            session_id, 
            data.get('config')
        ))
        
        # Track in active sessions
        active_sessions[session_id] = session_config
        
        return jsonify({
            "session_id": session_id,
            "config": session_config,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['GET'])
async def get_session(session_id: str):
    """Get session information"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        session = asyncio.run(translation_service.get_session(session_id))
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify({
            "session_id": session_id,
            "session": session,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['DELETE'])
async def close_session(session_id: str):
    """Close a translation session"""
    if translation_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        session = asyncio.run(translation_service.close_session(session_id))
        
        # Remove from active sessions
        active_sessions.pop(session_id, None)
        
        return jsonify({
            "session_id": session_id,
            "final_stats": session,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        return jsonify({"error": str(e)}), 500

# WebSocket events for real-time streaming
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Join a translation session room"""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        logger.info(f"Client {request.sid} joined session {session_id}")
        emit('joined_session', {'session_id': session_id, 'status': 'joined'})

@socketio.on('leave_session')
def handle_leave_session(data):
    """Leave a translation session room"""
    session_id = data.get('session_id')
    if session_id:
        leave_room(session_id)
        logger.info(f"Client {request.sid} left session {session_id}")
        emit('left_session', {'session_id': session_id, 'status': 'left'})

@socketio.on('translate_stream')
def handle_translate_stream(data):
    """Handle real-time streaming translation via WebSocket"""
    if translation_service is None:
        emit('error', {'message': 'Service not initialized'})
        return
    
    try:
        # Create translation request
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=data.get('source_language', 'auto'),
            target_language=data.get('target_language', 'en'),
            session_id=data.get('session_id'),
            streaming=True,
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=data.get('preserve_formatting', True),
            context=data.get('context')
        )
        
        # Run streaming translation in background
        def run_streaming():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def stream_to_client():
                try:
                    async for chunk in translation_service.translate_stream(translation_request):
                        socketio.emit('translation_chunk', {
                            'chunk': chunk,
                            'session_id': translation_request.session_id
                        }, room=request.sid)
                    
                    # Send completion signal
                    socketio.emit('translation_complete', {
                        'session_id': translation_request.session_id,
                        'timestamp': datetime.now().isoformat()
                    }, room=request.sid)
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    socketio.emit('translation_error', {
                        'error': str(e),
                        'session_id': translation_request.session_id
                    }, room=request.sid)
            
            loop.run_until_complete(stream_to_client())
        
        # Start streaming in background thread
        import threading
        streaming_thread = threading.Thread(target=run_streaming)
        streaming_thread.daemon = True
        streaming_thread.start()
        
    except Exception as e:
        logger.error(f"WebSocket streaming failed: {e}")
        emit('error', {'message': str(e)})

# Prompt Management Endpoints

@app.route('/prompts', methods=['GET'])
def get_prompts():
    """Get all prompt templates"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        filter_active = request.args.get('active', '').lower() == 'true'
        category = request.args.get('category')
        language_pair = request.args.get('language_pair')
        
        if filter_active:
            prompts = prompt_manager.get_active_prompts()
        elif category:
            prompts = prompt_manager.get_prompts_by_category(category)
        elif language_pair:
            source_lang, target_lang = language_pair.split('-') if '-' in language_pair else ('auto', language_pair)
            prompts = prompt_manager.get_prompts_for_language_pair(source_lang, target_lang)
        else:
            prompts = list(prompt_manager.prompts.values())
        
        # Convert to dict for JSON serialization
        prompt_data = []
        for prompt in prompts:
            prompt_dict = {
                'id': prompt.id,
                'name': prompt.name,
                'description': prompt.description,
                'template': prompt.template,
                'system_message': prompt.system_message,
                'language_pairs': prompt.language_pairs,
                'category': prompt.category,
                'version': prompt.version,
                'is_active': prompt.is_active,
                'is_default': prompt.is_default,
                'metadata': prompt.metadata,
                'performance_metrics': prompt.performance_metrics,
                'test_results': prompt.test_results
            }
            prompt_data.append(prompt_dict)
        
        return jsonify({
            "prompts": prompt_data,
            "total_count": len(prompt_data),
            "filters_applied": {
                "active_only": filter_active,
                "category": category,
                "language_pair": language_pair
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get prompts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/<prompt_id>', methods=['GET'])
def get_prompt(prompt_id):
    """Get a specific prompt template"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        prompt = prompt_manager.get_prompt(prompt_id)
        if not prompt:
            return jsonify({"error": "Prompt not found"}), 404
        
        prompt_dict = {
            'id': prompt.id,
            'name': prompt.name,
            'description': prompt.description,
            'template': prompt.template,
            'system_message': prompt.system_message,
            'language_pairs': prompt.language_pairs,
            'category': prompt.category,
            'version': prompt.version,
            'is_active': prompt.is_active,
            'is_default': prompt.is_default,
            'metadata': prompt.metadata,
            'performance_metrics': prompt.performance_metrics,
            'test_results': prompt.test_results
        }
        
        return jsonify(prompt_dict)
        
    except Exception as e:
        logger.error(f"Failed to get prompt {prompt_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts', methods=['POST'])
def create_prompt():
    """Create a new prompt template"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['id', 'name', 'description', 'template']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create prompt template
        prompt = PromptTemplate(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            template=data['template'],
            system_message=data.get('system_message'),
            language_pairs=data.get('language_pairs', ['*']),
            category=data.get('category', 'general'),
            version=data.get('version', '1.0'),
            is_active=data.get('is_active', True),
            is_default=False,  # Only system can create default prompts
            metadata=data.get('metadata', {})
        )
        
        # Create the prompt
        success = prompt_manager.create_prompt(prompt)
        if not success:
            return jsonify({"error": "Prompt with this ID already exists"}), 409
        
        return jsonify({
            "message": "Prompt created successfully",
            "prompt_id": prompt.id
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create prompt: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/<prompt_id>', methods=['PUT'])
def update_prompt(prompt_id):
    """Update an existing prompt template"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Update the prompt
        success = prompt_manager.update_prompt(prompt_id, data)
        if not success:
            return jsonify({"error": "Prompt not found"}), 404
        
        return jsonify({
            "message": "Prompt updated successfully",
            "prompt_id": prompt_id
        })
        
    except Exception as e:
        logger.error(f"Failed to update prompt {prompt_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/<prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id):
    """Delete a prompt template"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        success = prompt_manager.delete_prompt(prompt_id)
        if not success:
            return jsonify({"error": "Prompt not found or cannot be deleted (default prompts are protected)"}), 404
        
        return jsonify({
            "message": "Prompt deleted successfully",
            "prompt_id": prompt_id
        })
        
    except Exception as e:
        logger.error(f"Failed to delete prompt {prompt_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/<prompt_id>/test', methods=['POST'])
async def test_prompt(prompt_id):
    """Test a prompt template with sample data"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    if translation_service is None:
        return jsonify({"error": "Translation service not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        # Get prompt
        prompt = prompt_manager.get_prompt(prompt_id)
        if not prompt:
            return jsonify({"error": "Prompt not found"}), 404
        
        # Build prompt with variables
        variables = {
            'text': data['text'],
            'source_language': data.get('source_language', 'auto'),
            'target_language': data.get('target_language', 'en'),
            'context': data.get('context', ''),
            'style': data.get('style', ''),
            'domain': data.get('domain', ''),
            'preserve_formatting': data.get('preserve_formatting', True)
        }
        
        built_prompt, system_message = prompt_manager.build_prompt(prompt_id, variables)
        if not built_prompt:
            return jsonify({"error": "Failed to build prompt - missing variables"}), 400
        
        # Create translation request with custom prompt
        start_time = time.time()
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=variables['source_language'],
            target_language=variables['target_language'],
            session_id=data.get('session_id'),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=variables['preserve_formatting'],
            context=variables['context'],
            custom_prompt=built_prompt,
            system_message=system_message
        )
        
        # Perform translation using asyncio.run for sync/async compatibility
        result = asyncio.run(translation_service.translate(translation_request))
        processing_time = (time.time() - start_time) * 1000
        
        # Record performance metric
        metric = PromptPerformanceMetric(
            prompt_id=prompt_id,
            quality_score=result.confidence_score,  # Using confidence as quality proxy
            processing_time=processing_time,
            confidence_score=result.confidence_score,
            success=True,
            timestamp=time.time(),
            language_pair=f"{variables['source_language']}-{variables['target_language']}",
            text_length=len(data['text'])
        )
        prompt_manager.record_performance(metric)
        
        return jsonify({
            "test_result": {
                "translated_text": result.translated_text,
                "source_language": result.source_language,
                "target_language": result.target_language,
                "confidence": result.confidence_score,
                "processing_time": processing_time,
                "backend_used": result.backend_used,
                "quality_score": result.confidence_score
            },
            "prompt_used": built_prompt,
            "system_message": system_message,
            "prompt_analysis": f"Test completed successfully. Quality: {result.confidence_score:.2f}, Speed: {processing_time:.0f}ms"
        })
        
    except Exception as e:
        logger.error(f"Failed to test prompt {prompt_id}: {e}")
        
        # Record failed metric
        if prompt_manager and 'data' in locals():
            metric = PromptPerformanceMetric(
                prompt_id=prompt_id,
                quality_score=0.0,
                processing_time=0.0,
                confidence_score=0.0,
                success=False,
                timestamp=time.time(),
                language_pair=f"{data.get('source_language', 'auto')}-{data.get('target_language', 'en')}",
                text_length=len(data.get('text', '')),
                error_message=str(e)
            )
            prompt_manager.record_performance(metric)
        
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/<prompt_id>/performance', methods=['GET'])
def get_prompt_performance(prompt_id):
    """Get performance analysis for a prompt"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        analysis = prompt_manager.get_performance_analysis(prompt_id)
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Failed to get performance analysis for prompt {prompt_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/compare', methods=['POST'])
def compare_prompts():
    """Compare performance of multiple prompts"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'prompt_ids' not in data:
            return jsonify({"error": "Missing 'prompt_ids' field"}), 400
        
        prompt_ids = data['prompt_ids']
        days = data.get('days', 7)
        
        comparison = prompt_manager.compare_prompts(prompt_ids, days)
        return jsonify(comparison)
        
    except Exception as e:
        logger.error(f"Failed to compare prompts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/statistics', methods=['GET'])
def get_prompt_statistics():
    """Get overall prompt management statistics"""
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        stats = prompt_manager.get_prompt_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Failed to get prompt statistics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompts/categories', methods=['GET'])
def get_prompt_categories():
    """Get available prompt categories"""
    categories = [
        {'value': 'general', 'label': 'General Purpose', 'description': 'Standard translation prompts'},
        {'value': 'conversational', 'label': 'Conversational', 'description': 'Natural conversational style'},
        {'value': 'technical', 'label': 'Technical', 'description': 'Technical documentation and manuals'},
        {'value': 'medical', 'label': 'Medical', 'description': 'Medical and healthcare content'},
        {'value': 'legal', 'label': 'Legal', 'description': 'Legal documents and contracts'},
        {'value': 'creative', 'label': 'Creative', 'description': 'Literary and creative content'},
        {'value': 'formal', 'label': 'Formal', 'description': 'Business and formal communications'}
    ]
    
    return jsonify({
        "categories": categories,
        "total_count": len(categories)
    })

@app.route('/prompts/variables', methods=['GET'])
def get_prompt_variables():
    """Get available prompt template variables"""
    variables = [
        {
            'name': 'text',
            'description': 'The text to be translated',
            'type': 'text',
            'required': True,
            'example': 'Hello world!'
        },
        {
            'name': 'source_language',
            'description': 'Source language code or name',
            'type': 'language',
            'required': True,
            'example': 'English'
        },
        {
            'name': 'target_language',
            'description': 'Target language code or name',
            'type': 'language',
            'required': True,
            'example': 'Spanish'
        },
        {
            'name': 'context',
            'description': 'Additional context for translation',
            'type': 'text',
            'required': False,
            'example': 'This is a greeting message'
        },
        {
            'name': 'style',
            'description': 'Translation style (formal, casual, etc.)',
            'type': 'text',
            'required': False,
            'example': 'formal'
        },
        {
            'name': 'domain',
            'description': 'Domain/field (medical, legal, technical)',
            'type': 'text',
            'required': False,
            'example': 'medical'
        },
        {
            'name': 'preserve_formatting',
            'description': 'Whether to preserve text formatting',
            'type': 'boolean',
            'required': False,
            'default_value': True
        }
    ]
    
    return jsonify({
        "variables": variables,
        "usage_example": "Use variables like {text}, {source_language}, {target_language} in your prompt templates"
    })

# Enhanced translation endpoint with prompt support
@app.route('/translate/with_prompt', methods=['POST'])
async def translate_with_custom_prompt():
    """Translate text using a specific prompt template"""
    if translation_service is None:
        return jsonify({"error": "Translation service not initialized"}), 503
    
    if prompt_manager is None:
        return jsonify({"error": "Prompt manager not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        prompt_id = data.get('prompt_id', 'default')
        
        # Build prompt with variables
        variables = {
            'text': data['text'],
            'source_language': data.get('source_language', 'auto'),
            'target_language': data.get('target_language', 'en'),
            'context': data.get('context', ''),
            'style': data.get('style', ''),
            'domain': data.get('domain', ''),
            'preserve_formatting': data.get('preserve_formatting', True)
        }
        
        built_prompt, system_message = prompt_manager.build_prompt(prompt_id, variables)
        if not built_prompt:
            return jsonify({"error": "Failed to build prompt - check prompt template and variables"}), 400
        
        # Create translation request with custom prompt
        start_time = time.time()
        translation_request = TranslationRequest(
            text=data['text'],
            source_language=variables['source_language'],
            target_language=variables['target_language'],
            session_id=data.get('session_id'),
            confidence_threshold=data.get('confidence_threshold', 0.8),
            preserve_formatting=variables['preserve_formatting'],
            context=variables['context'],
            custom_prompt=built_prompt,
            system_message=system_message
        )
        
        # Perform translation using asyncio.run for sync/async compatibility
        result = asyncio.run(translation_service.translate(translation_request))
        processing_time = (time.time() - start_time) * 1000
        
        # Record performance metric
        metric = PromptPerformanceMetric(
            prompt_id=prompt_id,
            quality_score=result.confidence_score,
            processing_time=processing_time,
            confidence_score=result.confidence_score,
            success=True,
            timestamp=time.time(),
            language_pair=f"{variables['source_language']}-{variables['target_language']}",
            text_length=len(data['text'])
        )
        prompt_manager.record_performance(metric)
        
        return jsonify({
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "confidence_score": result.confidence_score,
            "processing_time": processing_time,
            "backend_used": result.backend_used,
            "session_id": result.session_id,
            "timestamp": result.timestamp,
            "prompt_id": prompt_id,
            "prompt_used": built_prompt
        })
        
    except Exception as e:
        logger.error(f"Translation with prompt failed: {e}")
        
        # Record failed metric
        if 'data' in locals() and 'prompt_id' in locals():
            metric = PromptPerformanceMetric(
                prompt_id=prompt_id,
                quality_score=0.0,
                processing_time=0.0,
                confidence_score=0.0,
                success=False,
                timestamp=time.time(),
                language_pair=f"{data.get('source_language', 'auto')}-{data.get('target_language', 'en')}",
                text_length=len(data.get('text', '')),
                error_message=str(e)
            )
            prompt_manager.record_performance(metric)
        
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

def create_app(config: Optional[Dict] = None):
    """Factory function to create Flask app with configuration"""
    if config:
        app.config.update(config)
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Translation Service API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5003, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Translation Service API server on {args.host}:{args.port}")
    
    # Initialize service
    asyncio.run(initialize_service())
    
    # Run server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Disable reloader in production
        allow_unsafe_werkzeug=True  # Allow Werkzeug for development/testing
    ) 