#!/usr/bin/env python3
"""
macOS-Optimized API Server for whisper.cpp Integration

Compatible with orchestration service - mirrors all endpoints from the original 
whisper-service while adding Apple Silicon optimizations with Metal/Core ML.
"""

import os
import sys
import asyncio
import logging
import time
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import soundfile as sf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.whisper_cpp_engine import WhisperCppEngine, WhisperResult, WhisperCppError

logger = logging.getLogger(__name__)

# Global engine instance
whisper_engine: Optional[WhisperCppEngine] = None

# Session storage for orchestration compatibility
active_sessions = {}
stream_configurations = {}

# Create the Flask app instance at module level (needed for route decorators)
app = Flask(__name__)
CORS(app)


def create_app(config: Dict[str, Any] = None):
    """Create and configure Flask application (returns existing app instance)"""
    global app
    
    # Configure app
    if config:
        app.config.update(config)
    
    # Set testing mode if specified
    if os.getenv("TESTING") == "true":
        app.config["TESTING"] = True
    
    # Set start time for uptime calculation
    if "start_time" not in app.config:
        app.config["start_time"] = time.time()
    
    return app


async def initialize_whisper_engine(config: Dict[str, Any] = None):
    """Initialize the whisper.cpp engine"""
    global whisper_engine
    
    try:
        models_dir = config.get("models_dir", "../models") if config else "../models"
        whisper_cpp_path = config.get("whisper_cpp_path", "whisper_cpp") if config else "whisper_cpp"
        
        whisper_engine = WhisperCppEngine(
            models_dir=models_dir,
            whisper_cpp_path=whisper_cpp_path
        )
        
        if whisper_engine.initialize():
            logger.info("✅ whisper.cpp engine initialized successfully")
            
            # Load default model if available
            models = whisper_engine.get_available_models()
            if models:
                default_model = config.get("default_model", "base.en") if config else "base.en"
                # Try to find and load default model
                for model in models:
                    if default_model in model["name"]:
                        whisper_engine.load_model(model["name"])
                        break
                else:
                    # Load first available model
                    whisper_engine.load_model(models[0]["name"])
                    
            return True
        else:
            logger.error("Failed to initialize whisper.cpp engine")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing whisper.cpp engine: {e}")
        return False


# ============================================================================
# CORE COMPATIBILITY ENDPOINTS (Match original whisper-service exactly)
# ============================================================================

@app.route('/favicon.ico')
def favicon():
    """Favicon endpoint"""
    return '', 204


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - CRITICAL for orchestration service"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({
            "status": "unhealthy",
            "error": "whisper.cpp engine not initialized"
        }), 503
    
    try:
        capabilities = whisper_engine.capabilities
        current_model = whisper_engine.current_model
        available_models = len(whisper_engine.available_models)
        
        return jsonify({
            "status": "healthy",
            "service": "whisper-service-mac",
            "version": "1.0.0-mac",
            "engine": "whisper.cpp",
            "uptime": time.time() - app.config.get("start_time", time.time()),
            "timestamp": time.time(),
            "capabilities": capabilities,
            "current_model": current_model["name"] if current_model and isinstance(current_model, dict) else current_model,
            "available_models": available_models,
            "apple_silicon": {
                "metal_enabled": capabilities.get("metal", False),
                "coreml_enabled": capabilities.get("coreml", False),
                "ane_enabled": capabilities.get("ane", False)
            }
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route("/cors-test", methods=['GET', 'POST', 'OPTIONS'])
def cors_test():
    """CORS test endpoint"""
    return jsonify({"status": "ok", "service": "whisper-service-mac"})


@app.route('/models', methods=['GET'])
def list_models():
    """List available models - orchestration compatibility"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "whisper.cpp engine not initialized"}), 503
    
    try:
        models = whisper_engine.get_available_models()
        
        # Format for compatibility with original service
        model_list = []
        for model in models:
            model_list.append({
                "name": model["name"],
                "file_name": model["file_name"],
                "size": model["size"],
                "quantization": model.get("quantization"),
                "coreml_available": model.get("coreml_available", False),
                "format": "ggml"
            })
        
        return jsonify({
            "models": model_list,
            "default_model": whisper_engine.current_model["name"] if whisper_engine.current_model else None,
            "engine": "whisper.cpp"
        })
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def api_list_models():
    """API models endpoint - CRITICAL for orchestration routing"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify([]), 503
    
    try:
        models = whisper_engine.get_available_models()
        
        # Return model names in format expected by orchestration service
        model_names = []
        for model in models:
            # Use consistent naming like "whisper-base"
            name = model["name"]
            if not name.startswith("whisper-"):
                if name.startswith("ggml-"):
                    name = name[5:]  # Remove ggml- prefix
                name = f"whisper-{name}"
            
            model_names.append(name)
        
        # Remove duplicates and sort
        model_names = sorted(list(set(model_names)))
        
        # Return in orchestration-compatible format
        result = {
            "available_models": model_names,
            "current_model": whisper_engine.current_model["name"] if whisper_engine.current_model else None
        }
        
        logger.info(f"Returning models for orchestration: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to get API models: {e}")
        return jsonify([]), 500


@app.route('/api/device-info', methods=['GET'])
def get_device_info():
    """Device info endpoint - CRITICAL for orchestration hardware detection"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        device_info = whisper_engine.get_device_info()
        capabilities = whisper_engine.capabilities
        
        # Format for orchestration service compatibility
        return jsonify({
            "device": "Metal" if capabilities.get("metal") else "CPU",
            "device_type": "Apple Silicon" if device_info["architecture"] == "arm64" else "Intel Mac",
            "acceleration": {
                "metal": capabilities.get("metal", False),
                "coreml": capabilities.get("coreml", False),
                "ane": capabilities.get("coreml", False),  # ANE via Core ML
                "accelerate": capabilities.get("accelerate", False),
                "neon": capabilities.get("neon", False)
            },
            "memory": "Unified Memory" if device_info["architecture"] == "arm64" else "Traditional",
            "platform": device_info["platform"],
            "architecture": device_info["architecture"],
            "capabilities": capabilities,
            "engine": "whisper.cpp",
            "service": "whisper-service-mac"
        })
        
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache"""
    global whisper_engine
    
    try:
        if whisper_engine:
            # Re-scan models
            whisper_engine._scan_models()
            
        return jsonify({
            "status": "success",
            "message": "Model cache cleared"
        })
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TRANSCRIPTION ENDPOINTS (Core functionality for orchestration)
# ============================================================================

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Basic transcription endpoint - orchestration compatibility"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "whisper.cpp engine not initialized"}), 503
    
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get parameters
        language = request.form.get('language', 'auto')
        if language == 'auto':
            language = None
            
        task = request.form.get('task', 'transcribe')
        model_name = request.form.get('model', None)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Transcribe using whisper.cpp
            result = whisper_engine.transcribe(
                audio_data=temp_path,
                model_name=model_name,
                language=language,
                task=task
            )
            
            # Format response for orchestration compatibility
            response = {
                "text": result.text,
                "language": result.language,
                "segments": result.segments,
                "model": result.model_name,
                "processing_time": result.processing_time,
                "engine": "whisper.cpp",
                "service": "whisper-service-mac"
            }
            
            if result.word_timestamps:
                response["word_timestamps"] = result.word_timestamps
            
            return jsonify(response)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except WhisperCppError as e:
        logger.error(f"whisper.cpp transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-chunk', methods=['POST'])
def process_orchestration_chunk():
    """Process audio chunk - CRITICAL for orchestration streaming"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "whisper.cpp engine not initialized"}), 503
    
    try:
        # Parse request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract required fields
        audio_data = data.get('audio_data')
        if not audio_data:
            return jsonify({"error": "No audio_data provided"}), 400
        
        model_name = data.get('model', 'whisper-base')
        language = data.get('language', 'auto')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Convert model name to GGML format
        if model_name.startswith('whisper-'):
            ggml_model = model_name[8:]  # Remove 'whisper-' prefix
        else:
            ggml_model = model_name
        
        if language == 'auto':
            language = None
        
        # Decode audio data (assuming base64 encoded)
        try:
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to numpy array
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Transcribe chunk
                result = whisper_engine.transcribe(
                    audio_data=temp_path,
                    model_name=ggml_model,
                    language=language,
                    word_timestamps=True  # Enable for orchestration
                )
                
                # Format response to match orchestration expectations
                response = {
                    "text": result.text,
                    "language": result.language,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "session_id": session_id,
                    "model": model_name,  # Return original model name
                    "engine": "whisper.cpp",
                    "segments": result.segments
                }
                
                if result.word_timestamps:
                    response["word_timestamps"] = result.word_timestamps
                
                return jsonify(response)
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
        
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe/<model_name>', methods=['POST'])
def transcribe_with_model(model_name: str):
    """Transcribe with specific model"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "whisper.cpp engine not initialized"}), 503
    
    try:
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        
        # Convert model name to GGML format
        if model_name.startswith('whisper-'):
            ggml_model = model_name[8:]  # Remove 'whisper-' prefix
        else:
            ggml_model = model_name
        
        # Get parameters
        language = request.form.get('language', 'auto')
        if language == 'auto':
            language = None
            
        task = request.form.get('task', 'transcribe')
        
        # Save and process file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            result = whisper_engine.transcribe(
                audio_data=temp_path,
                model_name=ggml_model,
                language=language,
                task=task,
                word_timestamps=True
            )
            
            response = {
                "text": result.text,
                "language": result.language,
                "segments": result.segments,
                "model": model_name,  # Return original model name
                "processing_time": result.processing_time,
                "engine": "whisper.cpp"
            }
            
            if result.word_timestamps:
                response["word_timestamps"] = result.word_timestamps
            
            return jsonify(response)
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Model transcription failed: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# STATUS AND MONITORING ENDPOINTS
# ============================================================================

@app.route('/status', methods=['GET'])
def get_status():
    """Service status endpoint"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        perf_stats = whisper_engine.get_performance_stats()
        
        return jsonify({
            "status": "running",
            "service": "whisper-service-mac",
            "engine": "whisper.cpp",
            "uptime": time.time(),  # Simplified uptime
            "performance": perf_stats,
            "sessions": {
                "active": len(active_sessions),
                "total": len(active_sessions)
            },
            "capabilities": whisper_engine.capabilities
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MACOS-SPECIFIC ENDPOINTS (Additional features)
# ============================================================================

@app.route('/api/metal/status', methods=['GET'])
def get_metal_status():
    """Metal GPU utilization status"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "Engine not initialized"}), 503
    
    capabilities = whisper_engine.capabilities
    
    return jsonify({
        "metal_enabled": capabilities.get("metal", False),
        "gpu_type": "Apple Silicon GPU" if capabilities.get("metal") else "Not available",
        "acceleration": "Metal Performance Shaders" if capabilities.get("metal") else None
    })


@app.route('/api/coreml/models', methods=['GET'])
def get_coreml_models():
    """List available Core ML models"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        models = whisper_engine.get_available_models()
        coreml_models = [model for model in models if model.get("coreml_available")]
        
        return jsonify({
            "coreml_models": coreml_models,
            "total_coreml": len(coreml_models),
            "ane_enabled": whisper_engine.capabilities.get("coreml", False)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/word-timestamps', methods=['POST'])
def get_word_timestamps():
    """Get detailed word-level timestamps"""
    global whisper_engine
    
    if not whisper_engine:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['file']
        model_name = request.form.get('model', None)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            result = whisper_engine.transcribe(
                audio_data=temp_path,
                model_name=model_name,
                word_timestamps=True
            )
            
            return jsonify({
                "text": result.text,
                "word_timestamps": result.word_timestamps or [],
                "segments": result.segments,
                "processing_time": result.processing_time
            })
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_mac_app(config: Dict[str, Any] = None):
    """Create Flask app with whisper.cpp engine"""
    
    # Initialize engine
    if not asyncio.run(initialize_whisper_engine(config)):
        logger.error("Failed to initialize whisper.cpp engine")
        return None
    
    # Configure app
    if config:
        app.config.update(config)
    
    logger.info("✅ macOS whisper service ready")
    logger.info(f"🍎 Engine capabilities: {whisper_engine.capabilities}")
    logger.info(f"📊 Available models: {len(whisper_engine.available_models)}")
    
    return app


if __name__ == "__main__":
    # For development/testing
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize and run
    app = create_mac_app()
    if app:
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        logger.error("Failed to create app")
        sys.exit(1)