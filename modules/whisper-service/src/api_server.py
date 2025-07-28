#!/usr/bin/env python3
"""
Whisper Service API Server

Flask-based REST API with WebSocket support for real-time transcription streaming.
Provides endpoints for transcription, model management, session handling, and streaming.
"""

import os
import asyncio
import logging
import io
import tempfile
import time
from typing import Dict, Any, Optional
import json
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.utils import which
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, LifoQueue, PriorityQueue
import weakref
from collections import deque

from whisper_service import WhisperService, TranscriptionRequest, create_whisper_service
from connection_manager import connection_manager, ConnectionState
from error_handler import (
    error_handler, ErrorCategory, ErrorSeverity, ErrorInfo,
    create_connection_error, create_audio_error, create_model_error,
    create_validation_error, create_session_error, create_system_error
)
from heartbeat_manager import heartbeat_manager, HeartbeatState
from message_router import message_router, MessageType, RoutePermission, MessageContext
from simple_auth import simple_auth, auth_middleware, UserRole
from reconnection_manager import reconnection_manager, SessionState, BufferedMessage
from utils.audio_errors import (
    WhisperProcessingBaseError, AudioFormatError, AudioCorruptionError,
    ModelLoadingError, ModelInferenceError, ValidationError as WhisperValidationError,
    ConfigurationError, MemoryError, HardwareError, TimeoutError as WhisperTimeoutError,
    CircuitBreaker, ErrorRecoveryStrategy, ModelRecoveryStrategy, FormatRecoveryStrategy,
    ErrorLogger, error_boundary, default_circuit_breaker, default_error_logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pydub to use local ffmpeg
from pathlib import Path
import pydub.utils

# Set up ffmpeg path for pydub
ffmpeg_path = Path(__file__).parent.parent / "ffmpeg" / "bin"
logger.info(f"[AUDIO] Checking for ffmpeg in: {ffmpeg_path}")

if ffmpeg_path.exists():
    # On Windows, look for ffmpeg.exe
    if os.name == 'nt':
        ffmpeg_exe = ffmpeg_path / "ffmpeg.exe"
        ffprobe_exe = ffmpeg_path / "ffprobe.exe"
    else:
        ffmpeg_exe = ffmpeg_path / "ffmpeg"
        ffprobe_exe = ffmpeg_path / "ffprobe"
    
    logger.info(f"[AUDIO] Looking for ffmpeg at: {ffmpeg_exe}")
    logger.info(f"[AUDIO] ffmpeg exists: {ffmpeg_exe.exists()}")
    
    if ffmpeg_exe.exists():
        # Set both converter and the utils module path
        AudioSegment.converter = str(ffmpeg_exe.absolute())
        pydub.utils.get_player_name = lambda: str(ffmpeg_exe.absolute())
        pydub.utils.get_encoder_name = lambda: str(ffmpeg_exe.absolute())
        logger.info(f"[AUDIO] Successfully configured local ffmpeg: {ffmpeg_exe.absolute()}")
    else:
        logger.warning(f"[AUDIO] ffmpeg.exe not found at {ffmpeg_exe}")
        
    if ffprobe_exe.exists():
        AudioSegment.ffprobe = str(ffprobe_exe.absolute())
        pydub.utils.get_prober_name = lambda: str(ffprobe_exe.absolute())
        logger.info(f"[AUDIO] Successfully configured local ffprobe: {ffprobe_exe.absolute()}")
else:
    logger.warning(f"[AUDIO] Local ffmpeg directory not found at {ffmpeg_path}")

# Performance optimization classes
class AudioProcessingPool:
    """Thread pool for audio processing tasks"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="audio_proc")
        self.queue = PriorityQueue()
        self.results_cache = weakref.WeakValueDictionary()
        self._shutdown = False
    
    def submit_audio_task(self, task_func, priority: int = 5, *args, **kwargs):
        """Submit audio processing task with priority"""
        if self._shutdown:
            return None
        return self.executor.submit(task_func, *args, **kwargs)
    
    def shutdown(self):
        """Shutdown the processing pool"""
        self._shutdown = True
        self.executor.shutdown(wait=True)

class MessageQueue:
    """High-performance message queue with batching"""
    
    def __init__(self, batch_size: int = 10, max_queue_size: int = 1000):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.queue = deque(maxlen=max_queue_size)
        self.priority_queue = PriorityQueue(maxsize=max_queue_size)
        self._lock = threading.RLock()
        self._processing = False
        
    def add_message(self, message: Dict[str, Any], priority: int = 5):
        """Add message to queue with priority"""
        with self._lock:
            if len(self.queue) >= self.max_queue_size:
                # Remove oldest message if queue is full
                self.queue.popleft()
            
            message['timestamp'] = time.time()
            message['priority'] = priority
            self.queue.append(message)
            
            # Also add to priority queue for high-priority messages
            if priority <= 3:  # High priority
                try:
                    self.priority_queue.put_nowait((priority, message))
                except:
                    pass  # Queue full, skip
    
    def get_batch(self, timeout: float = 1.0) -> list:
        """Get a batch of messages for processing"""
        batch = []
        with self._lock:
            # First, get high-priority messages
            while not self.priority_queue.empty() and len(batch) < self.batch_size:
                try:
                    _, message = self.priority_queue.get_nowait()
                    batch.append(message)
                except:
                    break
            
            # Fill remaining batch with regular messages
            while len(self.queue) > 0 and len(batch) < self.batch_size:
                batch.append(self.queue.popleft())
        
        return batch

class PerformanceMonitor:
    """Monitor and optimize performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'audio_processing_times': deque(maxlen=100),
            'transcription_times': deque(maxlen=100),
            'websocket_message_times': deque(maxlen=100),
            'queue_sizes': deque(maxlen=100),
            'memory_usage': deque(maxlen=50),
            'cpu_usage': deque(maxlen=50)
        }
        self._lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric"""
        with self._lock:
            values = self.metrics.get(metric_name, [])
            return sum(values) / len(values) if values else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._lock:
            stats = {}
            for metric_name, values in self.metrics.items():
                if values:
                    stats[metric_name] = {
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'recent': list(values)[-10:]  # Last 10 values
                    }
                else:
                    stats[metric_name] = {
                        'average': 0, 'min': 0, 'max': 0, 'count': 0, 'recent': []
                    }
            return stats

# Global performance optimization instances
audio_pool = AudioProcessingPool(max_workers=int(os.getenv('AUDIO_WORKERS', '4')))
message_queue = MessageQueue(
    batch_size=int(os.getenv('MESSAGE_BATCH_SIZE', '10')),
    max_queue_size=int(os.getenv('MAX_QUEUE_SIZE', '1000'))
)
performance_monitor = PerformanceMonitor()

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# SocketIO for real-time streaming with optimized settings
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    engineio_logger=False,  # Reduce logging overhead
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10**8  # 100MB for large audio chunks
)

# Add favicon endpoint to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

# Global whisper service instance
whisper_service: Optional[WhisperService] = None

# Active streaming sessions
streaming_sessions: Dict[str, Dict] = {}

async def initialize_service():
    """Initialize the whisper service before handling requests"""
    global whisper_service
    try:
        whisper_service = await create_whisper_service()
        
        # Start connection manager
        connection_manager.start()
        
        # Configure and start heartbeat manager
        heartbeat_manager.set_socketio_instance(socketio)
        heartbeat_manager.set_connection_lost_callback(handle_connection_lost)
        heartbeat_manager.set_connection_warning_callback(handle_connection_warning)
        heartbeat_manager.start()
        
        # Start periodic cleanup for expired tokens
        def cleanup_expired_tokens():
            while True:
                try:
                    simple_auth.cleanup_expired_tokens()
                    time.sleep(300)  # Clean up every 5 minutes
                except Exception as e:
                    logger.error(f"Token cleanup error: {e}")
                    time.sleep(60)  # Retry after 1 minute on error
        
        cleanup_thread = threading.Thread(target=cleanup_expired_tokens, daemon=True)
        cleanup_thread.start()
        
        logger.info("Whisper service, connection manager, heartbeat manager, and authentication system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize whisper service: {e}")
        raise

# Initialize service will be called at the end of the file after all functions are defined

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if whisper_service is None:
        return jsonify({"status": "error", "message": "Service not initialized"}), 503
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service_status": whisper_service.get_service_status()
    })

# CORS test endpoint
@app.route("/cors-test", methods=['GET', 'POST', 'OPTIONS'])
def cors_test():
    """Test CORS configuration"""
    return jsonify({"status": "CORS working", "method": request.method})

# Model management endpoints
@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        models = whisper_service.get_available_models()
        status = whisper_service.get_service_status()
        
        return jsonify({
            "available_models": models,
            "loaded_models": status["loaded_models"],
            "default_model": whisper_service.config.get("default_model", "whisper-base"),
            "device": status["device"]
        })
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def api_list_models():
    """List available models (API endpoint for orchestration integration)"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        models = whisper_service.get_available_models()
        status = whisper_service.get_service_status()
        
        return jsonify({
            "models": models,
            "available_models": models,
            "loaded_models": status["loaded_models"],
            "default_model": whisper_service.config.get("default_model", "whisper-base"),
            "device": status["device"],
            "status": "success",
            "service": "whisper"
        })
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/device-info', methods=['GET'])
def get_device_info():
    """Get current device information (CPU/GPU/NPU) and acceleration status"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        status = whisper_service.get_service_status()
        
        # Extract device information from service status
        device = status.get("device", "unknown")
        device_details = status.get("device_details", {})
        
        # Determine device type and acceleration
        device_type = "unknown"
        acceleration = "none"
        
        if device.upper() == "NPU":
            device_type = "npu"
            acceleration = "intel_npu"
        elif device.upper() == "GPU":
            device_type = "gpu"
            acceleration = "cuda" if device_details.get("cuda_available") else "opencl"
        elif device.upper() == "CPU":
            device_type = "cpu"
            acceleration = "none"
        
        return jsonify({
            "device": device.lower(),
            "device_type": device_type,
            "status": "healthy" if status.get("ready", False) else "unavailable",
            "acceleration": acceleration,
            "details": {
                "models_loaded": status.get("loaded_models", []),
                "memory_usage": device_details.get("memory_usage", "unknown"),
                "device_name": device_details.get("device_name", "unknown"),
                "capabilities": device_details.get("capabilities", [])
            },
            "service_info": {
                "version": status.get("version", "unknown"),
                "uptime": status.get("uptime", 0),
                "ready": status.get("ready", False)
            }
        })
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        logger.info("[CACHE] 🗑️ Clearing model cache...")
        whisper_service.clear_cache()
        logger.info("[CACHE] ✅ Model cache cleared successfully")
        return jsonify({"status": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"[CACHE] ❌ Failed to clear cache: {e}")
        return jsonify({"error": str(e)}), 500

# Basic transcription endpoint
@app.route('/transcribe', methods=['POST'])
async def transcribe():
    """Transcribe audio using default model"""
    return await transcribe_with_model(whisper_service.config.get("default_model", "whisper-base"))

# Orchestration-managed chunk processing endpoint
@app.route('/api/process-chunk', methods=['POST'])
async def process_orchestration_chunk():
    """Process audio chunk from orchestration service with metadata"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        logger.info("[WHISPER] 🎯 Orchestration chunk processing request received")
        
        # Parse request data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
        else:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        # Extract required fields
        chunk_id = data.get('chunk_id')
        session_id = data.get('session_id')
        audio_data_b64 = data.get('audio_data')
        chunk_metadata = data.get('chunk_metadata', {})
        model_name = data.get('model_name', whisper_service.config.get("default_model", "whisper-base"))
        
        if not chunk_id:
            return jsonify({"error": "chunk_id is required"}), 400
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not audio_data_b64:
            return jsonify({"error": "audio_data is required"}), 400
        
        # Decode base64 audio data
        import base64
        try:
            audio_bytes = base64.b64decode(audio_data_b64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 audio data: {str(e)}"}), 400
        
        logger.info(f"[WHISPER] 🎯 Processing chunk {chunk_id} for session {session_id}")
        logger.info(f"[WHISPER] 📊 Chunk metadata: {chunk_metadata}")
        logger.info(f"[WHISPER] 📏 Audio data size: {len(audio_bytes)} bytes")
        
        # Create transcription request with chunk metadata
        transcription_request = TranscriptionRequest(
            audio_data=audio_bytes,
            model_name=model_name,
            session_id=session_id,
            streaming=False,  # Process as single chunk
            enhanced=chunk_metadata.get('enable_enhancement', False),
            sample_rate=chunk_metadata.get('sample_rate', 16000),
            enable_vad=chunk_metadata.get('enable_vad', False),  # VAD already applied by orchestration
            timestamp_mode=chunk_metadata.get('timestamp_mode', 'word')
        )
        
        # Process the chunk using orchestration-aware method
        if hasattr(whisper_service, 'process_orchestration_chunk'):
            # Use the new orchestration chunk processing method
            response_data = await whisper_service.process_orchestration_chunk(
                chunk_id=chunk_id,
                session_id=session_id,
                audio_data=audio_bytes,
                chunk_metadata=chunk_metadata,
                model_name=model_name
            )
            
            logger.info(f"[WHISPER] ✅ Chunk {chunk_id} processed via orchestration method")
            if response_data.get("status") == "success":
                logger.info(f"[WHISPER] 📝 Transcription result: {response_data['transcription']['text'][:100]}...")
            else:
                logger.error(f"[WHISPER] ❌ Chunk processing failed: {response_data.get('error', 'Unknown error')}")
        else:
            # Fallback to legacy processing
            logger.warning("[WHISPER] Using legacy processing method - orchestration features unavailable")
            start_time = time.time()
            result = await whisper_service.transcribe(transcription_request)
            processing_time = time.time() - start_time
            
            logger.info(f"[WHISPER] ✅ Chunk {chunk_id} processed in {processing_time:.2f}s (legacy)")
            logger.info(f"[WHISPER] 📝 Transcription result: {result.text[:100]}...")
            
            # Prepare response with chunk context (legacy format)
            response_data = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "status": "success",
                "transcription": {
                    "text": result.text,
                    "language": result.language,
                    "confidence_score": result.confidence_score,
                    "segments": result.segments,
                    "timestamp": result.timestamp
                },
                "processing_info": {
                    "model_used": result.model_used,
                    "device_used": result.device_used,
                    "processing_time": processing_time,
                    "chunk_metadata": chunk_metadata,
                    "service_mode": "legacy"
                },
                "chunk_sequence": chunk_metadata.get('sequence_number', 0),
                "chunk_timing": {
                    "start_time": chunk_metadata.get('start_time', 0.0),
                    "end_time": chunk_metadata.get('end_time', 0.0),
                    "duration": chunk_metadata.get('duration', 0.0)
                }
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"[WHISPER] ❌ Failed to process orchestration chunk: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "chunk_id": chunk_id if 'chunk_id' in locals() else "unknown",
            "session_id": session_id if 'session_id' in locals() else "unknown",
            "status": "error", 
            "error": str(e),
            "error_type": "processing_error"
        }), 500

# Model-specific transcription endpoint
@app.route('/transcribe/<model_name>', methods=['POST'])
async def transcribe_with_model(model_name: str):
    """Transcribe audio using specified model with comprehensive error handling"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    # Generate correlation ID for this request
    correlation_id = f"transcribe_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    with error_boundary(
        correlation_id=correlation_id,
        context={
            "operation": "transcribe_with_model",
            "model_name": model_name,
            "request_method": request.method,
            "content_type": request.content_type
        },
        recovery_strategies=[model_recovery, format_recovery],
        circuit_breaker=default_circuit_breaker
    ) as correlation_id:
        try:
            logger.info(f"[WHISPER] [{correlation_id}] 🎤 Transcription request received for model: {model_name}")
            
            # Enhanced input validation
            await _validate_transcription_request(request, model_name, correlation_id)
        
        # Check if model is loaded or needs loading
        if whisper_service.model_manager and model_name not in whisper_service.model_manager.pipelines:
            logger.info(f"[WHISPER] 🔄 Model {model_name} not loaded, loading now...")
        else:
            logger.info(f"[WHISPER] ✅ Model {model_name} already loaded and ready")
        
        # Handle different content types
        logger.info(f"[WHISPER] 📥 Request content type: {request.content_type}")
        logger.info(f"[WHISPER] 📁 Request files keys: {list(request.files.keys()) if request.files else 'None'}")
        logger.info(f"[WHISPER] 📝 Request form keys: {list(request.form.keys()) if request.form else 'None'}")
        logger.info(f"[WHISPER] 📊 Request data length: {len(request.data) if request.data else 0}")
        
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload
            if 'audio' not in request.files:
                logger.error(f"No 'audio' field in files. Available fields: {list(request.files.keys())}")
                return jsonify({"error": "No audio file provided"}), 400
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read audio data
            audio_data = audio_file.read()
            
        elif request.content_type and 'application/octet-stream' in request.content_type:
            # Raw audio data
            audio_data = request.data
            logger.info(f"Received raw audio data: {len(audio_data)} bytes")
            
        elif request.data and len(request.data) > 0:
            # Fallback: try to use raw data if available
            audio_data = request.data
            logger.info(f"Fallback: using raw request data: {len(audio_data)} bytes")
            
        else:
            logger.error(f"Unsupported request format. Content-Type: {request.content_type}, Files: {bool(request.files)}, Data: {len(request.data) if request.data else 0}")
            return jsonify({"error": "Unsupported content type. Use multipart/form-data or application/octet-stream"}), 400
        
        # Get additional parameters
        language = request.form.get('language') if request.form else None
        session_id = request.form.get('session_id') if request.form else None
        enable_vad = request.form.get('enable_vad', 'false').lower() == 'true' if request.form else False  # Temporarily disable VAD to debug
        
        # Process audio
        logger.info(f"[WHISPER] 🔊 Processing audio data: {len(audio_data)} bytes")
        audio_array = _process_audio_data(audio_data)
        logger.info(f"[WHISPER] 📊 Audio processed to array: {audio_array.shape if hasattr(audio_array, 'shape') else len(audio_array)} samples")
        
        # Audio quality debug info
        if hasattr(audio_array, 'shape') and len(audio_array) > 0:
            duration = len(audio_array) / 16000  # Duration in seconds at 16kHz
            rms_level = np.sqrt(np.mean(audio_array**2))
            max_amplitude = np.max(np.abs(audio_array))
            logger.info(f"[WHISPER] 🎵 Audio duration: {duration:.3f}s, RMS: {rms_level:.6f}, Max amp: {max_amplitude:.3f}")
            logger.info(f"[WHISPER] 🎯 VAD enabled: {enable_vad}")
        else:
            logger.warning(f"[WHISPER] ⚠️ Invalid audio array: {type(audio_array)}")
        
        # Create transcription request
        transcription_request = TranscriptionRequest(
            audio_data=audio_array,
            model_name=model_name,
            language=language,
            session_id=session_id,
            enable_vad=enable_vad,
            sample_rate=16000
        )
        
        # Perform transcription
        logger.info(f"[WHISPER] 🚀 Starting transcription with model: {model_name}")
        start_time = time.time()
        result = await whisper_service.transcribe(transcription_request)
        processing_time = time.time() - start_time
        
        # Improved repetitive text detection - only flag extreme cases
        if result and result.text:
            words = result.text.split()
            if len(words) > 15:  # Only check longer texts
                # Check if text is extremely repetitive
                unique_words = set(words)
                repetition_ratio = len(unique_words) / len(words)
                
                if repetition_ratio < 0.15:  # Less than 15% unique words (was 10%)
                    logger.warning(f"[WHISPER] Detected extreme repetition - ratio: {repetition_ratio:.2f}")
                    logger.warning(f"[WHISPER] Original text: {result.text[:100]}...")
                    
                    # Reduce confidence but don't completely invalidate
                    result.confidence_score = max(0.2, result.confidence_score * 0.5)
                    logger.info(f"[WHISPER] Adjusted confidence to {result.confidence_score:.3f} due to repetition")
        
        logger.info(f"[WHISPER] ✅ Transcription completed successfully")
        logger.info(f"[WHISPER] 📝 Result: '{result.text}'")
        logger.info(f"[WHISPER] ⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"[WHISPER] 🧠 Model used: {result.model_used}")
        logger.info(f"[WHISPER] 💻 Device used: {result.device_used}")
        logger.info(f"[WHISPER] 🌍 Language detected: {result.language}")
        logger.info(f"[WHISPER] 📊 Confidence: {result.confidence_score:.2f}")
        
        return jsonify({
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "confidence": result.confidence_score,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "device_used": result.device_used,
            "timestamp": result.timestamp,
            "session_id": result.session_id
        })
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Enhanced transcription with preprocessing
@app.route('/transcribe/enhanced/<model_name>', methods=['POST'])
async def transcribe_with_enhancement(model_name: str):
    """Transcribe audio with enhanced preprocessing"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        # Handle audio data (same as regular transcribe)
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        # Get parameters
        language = request.form.get('language')
        session_id = request.form.get('session_id')
        
        # Process with enhancement
        audio_array = _process_audio_data(audio_data, enhance=True)
        
        # Create enhanced transcription request
        transcription_request = TranscriptionRequest(
            audio_data=audio_array,
            model_name=model_name,
            language=language,
            session_id=session_id,
            enhanced=True,
            enable_vad=True,
            sample_rate=16000
        )
        
        # Perform transcription
        result = await whisper_service.transcribe(transcription_request)
        
        return jsonify({
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "confidence": result.confidence_score,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "device_used": result.device_used,
            "timestamp": result.timestamp,
            "session_id": result.session_id,
            "enhanced": True
        })
        
    except Exception as e:
        logger.error(f"Enhanced transcription failed: {e}")
        return jsonify({"error": str(e)}), 500

# Streaming endpoints
@app.route('/stream/configure', methods=['POST'])
def configure_streaming():
    """Configure streaming parameters"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Configure streaming session
        streaming_config = {
            "session_id": session_id,
            "model_name": data.get('model_name', 'whisper-base'),
            "language": data.get('language'),
            "buffer_duration": data.get('buffer_duration', 6.0),
            "inference_interval": data.get('inference_interval', 3.0),
            "enable_vad": data.get('enable_vad', True),
            "created_at": datetime.now().isoformat()
        }
        
        # Create session in whisper service
        if whisper_service:
            whisper_service.create_session(session_id, streaming_config)
        
        streaming_sessions[session_id] = streaming_config
        
        return jsonify({
            "status": "configured",
            "session_id": session_id,
            "config": streaming_config
        })
        
    except Exception as e:
        logger.error(f"Failed to configure streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/start', methods=['POST'])
def start_streaming():
    """Start streaming session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if not session_id or session_id not in streaming_sessions:
            return jsonify({"error": "Invalid session_id"}), 400
        
        config = streaming_sessions[session_id]
        config["streaming_active"] = True
        config["started_at"] = datetime.now().isoformat()
        
        return jsonify({
            "status": "streaming_started",
            "session_id": session_id,
            "config": config
        })
        
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/stop', methods=['POST'])
def stop_streaming():
    """Stop streaming session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        if not session_id or session_id not in streaming_sessions:
            return jsonify({"error": "Invalid session_id"}), 400
        
        config = streaming_sessions[session_id]
        config["streaming_active"] = False
        config["stopped_at"] = datetime.now().isoformat()
        
        # Close session in whisper service
        if whisper_service:
            final_stats = whisper_service.close_session(session_id)
            config["final_stats"] = final_stats
        
        return jsonify({
            "status": "streaming_stopped",
            "session_id": session_id,
            "final_stats": config.get("final_stats")
        })
        
    except Exception as e:
        logger.error(f"Failed to stop streaming: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/audio', methods=['POST'])
async def stream_audio_chunk():
    """Stream audio chunk for real-time transcription"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        # Get session ID from header or form data
        session_id = request.headers.get('X-Session-ID') or request.form.get('session_id')
        
        if not session_id or session_id not in streaming_sessions:
            return jsonify({"error": "Invalid session_id"}), 400
        
        # Get audio data
        if request.content_type and 'application/octet-stream' in request.content_type:
            audio_data = request.data
        elif 'audio' in request.files:
            audio_data = request.files['audio'].read()
        else:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Process audio chunk
        audio_array = _process_audio_data(audio_data)
        
        # Add to buffer
        buffer_size = whisper_service.add_audio_chunk(audio_array)
        
        return jsonify({
            "status": "chunk_added",
            "session_id": session_id,
            "buffer_size": buffer_size,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to process audio chunk: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream/transcriptions', methods=['GET'])
def get_rolling_transcriptions():
    """Get recent transcriptions from rolling buffer"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        session_id = request.args.get('session_id')
        limit = int(request.args.get('limit', 10))
        
        if session_id:
            # Get session-specific transcriptions
            session = whisper_service.get_session(session_id)
            if session:
                transcriptions = session.get("transcriptions", [])[-limit:]
            else:
                transcriptions = []
        else:
            # Get global transcription history
            transcriptions = whisper_service.get_transcription_history(limit)
        
        return jsonify({
            "transcriptions": transcriptions,
            "count": len(transcriptions),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get transcriptions: {e}")
        return jsonify({"error": str(e)}), 500

# Session management endpoints
@app.route('/sessions', methods=['POST'])
def create_session():
    """Create a new transcription session"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        session_config = whisper_service.create_session(session_id, data.get('config'))
        
        return jsonify({
            "session_id": session_id,
            "config": session_config,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get session information"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        session = whisper_service.get_session(session_id)
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
def close_session(session_id: str):
    """Close a transcription session"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        session = whisper_service.close_session(session_id)
        
        # Remove from streaming sessions
        streaming_sessions.pop(session_id, None)
        
        return jsonify({
            "session_id": session_id,
            "final_stats": session,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        return jsonify({"error": str(e)}), 500

# Service management endpoints
@app.route('/status', methods=['GET'])
def get_service_status():
    """Get detailed service status"""
    if whisper_service is None:
        return jsonify({"status": "error", "message": "Service not initialized"}), 503
    
    try:
        status = whisper_service.get_service_status()
        connection_stats = connection_manager.get_statistics()
        heartbeat_stats = heartbeat_manager.get_statistics()
        
        # Add detailed model loading information
        model_info = {
            "device": whisper_service.model_manager.device if whisper_service.model_manager else "unknown",
            "loaded_models": list(whisper_service.model_manager.pipelines.keys()) if whisper_service.model_manager else [],
            "available_models": whisper_service.get_available_models(),
            "default_model": whisper_service.config.get("default_model", "whisper-base"),
            "models_directory": whisper_service.config.get("models_dir"),
            "total_loaded": len(whisper_service.model_manager.pipelines) if whisper_service.model_manager else 0
        }
        
        # Log the status request with model info
        logger.info(f"[STATUS] 📊 Service status requested - Device: {model_info['device']}, Loaded: {model_info['loaded_models']}")
        
        return jsonify({
            "status": "ok",
            "service_info": status,
            "model_info": model_info,
            "streaming_sessions": len(streaming_sessions),
            "active_streams": len([s for s in streaming_sessions.values() if s.get("streaming_active", False)]),
            "connection_manager": connection_stats,
            "heartbeat_manager": heartbeat_stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/connections', methods=['GET'])
def get_connections():
    """Get detailed connection information"""
    try:
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        
        if detailed:
            connections = connection_manager.get_detailed_connections()
        else:
            connections = {
                "active_connections": connection_manager.get_connection_count(),
                "active_sessions": connection_manager.get_session_count(),
                "statistics": connection_manager.get_statistics()
            }
        
        return jsonify({
            "connections": connections,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        error_info = create_system_error("Failed to get connections", str(e))
        error_handler.handle_error(error_info)
        response, status_code = error_info.to_http_response()
        return jsonify(response), status_code

@app.route('/errors', methods=['GET'])
def get_error_statistics():
    """Get error statistics and recent errors"""
    try:
        include_recent = request.args.get('recent', 'false').lower() == 'true'
        limit = int(request.args.get('limit', 50))
        
        stats = error_handler.get_error_statistics()
        
        response = {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        if include_recent:
            response["recent_errors"] = error_handler.get_recent_errors(limit)
        
        return jsonify(response)
    except Exception as e:
        error_info = create_system_error("Failed to get error statistics", str(e))
        error_handler.handle_error(error_info)
        response, status_code = error_info.to_http_response()
        return jsonify(response), status_code

@app.route('/heartbeat', methods=['GET'])
def get_heartbeat_statistics():
    """Get heartbeat statistics and connection health"""
    try:
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        
        stats = heartbeat_manager.get_statistics()
        
        response = {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        if detailed:
            all_health = heartbeat_manager.get_all_connections_health()
            response["connection_health"] = {
                conn_id: heartbeat.to_dict() 
                for conn_id, heartbeat in all_health.items()
            }
            
            unhealthy = heartbeat_manager.get_unhealthy_connections()
            response["unhealthy_connections"] = {
                conn_id: heartbeat.to_dict() 
                for conn_id, heartbeat in unhealthy.items()
            }
        
        return jsonify(response)
    except Exception as e:
        error_info = create_system_error("Failed to get heartbeat statistics", str(e))
        error_handler.handle_error(error_info)
        response, status_code = error_info.to_http_response()
        return jsonify(response), status_code

@app.route('/router', methods=['GET'])
def get_router_information():
    """Get message router statistics and documentation"""
    try:
        include_docs = request.args.get('docs', 'false').lower() == 'true'
        
        stats = message_router.get_statistics()
        
        response = {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        if include_docs:
            response["documentation"] = message_router.get_route_documentation()
        
        return jsonify(response)
    except Exception as e:
        error_info = create_system_error("Failed to get router information", str(e))
        error_handler.handle_error(error_info)
        response, status_code = error_info.to_http_response()
        return jsonify(response), status_code

# Authentication endpoints
@app.route('/auth/login', methods=['POST'])
def login():
    """Authenticate user and return token"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400
        
        user_id = data.get('user_id')
        password = data.get('password')
        
        if not user_id or not password:
            return jsonify({"error": "user_id and password required"}), 400
        
        # Authenticate user
        auth_token = simple_auth.authenticate(user_id, password)
        if not auth_token:
            return jsonify({"error": "Invalid credentials"}), 401
        
        return jsonify({
            "status": "success",
            "token": auth_token.token,
            "user_id": auth_token.user_id,
            "role": auth_token.role.value,
            "expires_at": auth_token.expires_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Authentication failed"}), 500

@app.route('/auth/guest', methods=['POST'])
def create_guest_token():
    """Create a guest token for anonymous access"""
    try:
        auth_token = simple_auth.create_guest_token()
        
        return jsonify({
            "status": "success",
            "token": auth_token.token,
            "user_id": auth_token.user_id,
            "role": auth_token.role.value,
            "expires_at": auth_token.expires_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Guest token creation error: {e}")
        return jsonify({"error": "Failed to create guest token"}), 500

@app.route('/auth/validate', methods=['POST'])
def validate_token():
    """Validate an authentication token"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400
        
        token = data.get('token')
        if not token:
            return jsonify({"error": "Token required"}), 400
        
        auth_token = simple_auth.validate_token(token)
        if not auth_token:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        return jsonify({
            "status": "valid",
            "user_id": auth_token.user_id,
            "role": auth_token.role.value,
            "expires_at": auth_token.expires_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return jsonify({"error": "Validation failed"}), 500

@app.route('/auth/logout', methods=['POST'])
def logout():
    """Revoke an authentication token"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400
        
        token = data.get('token')
        if not token:
            return jsonify({"error": "Token required"}), 400
        
        success = simple_auth.revoke_token(token)
        if success:
            return jsonify({
                "status": "success",
                "message": "Token revoked",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Token not found"}), 404
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({"error": "Logout failed"}), 500

@app.route('/auth/stats', methods=['GET'])
def get_auth_statistics():
    """Get authentication statistics"""
    try:
        stats = simple_auth.get_statistics()
        
        return jsonify({
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Auth stats error: {e}")
        return jsonify({"error": "Failed to get statistics"}), 500

@app.route('/reconnection', methods=['GET'])
def get_reconnection_statistics():
    """Get reconnection manager statistics"""
    try:
        stats = reconnection_manager.get_statistics()
        return jsonify({
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get reconnection statistics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>/info', methods=['GET'])
def get_session_details(session_id: str):
    """Get detailed information about a specific session"""
    try:
        session_info = reconnection_manager.get_session(session_id)
        if session_info:
            return jsonify({
                "session": session_info.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        logger.error(f"Failed to get session details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_buffered_messages(session_id: str):
    """Get buffered messages for a session"""
    try:
        pending_messages = reconnection_manager.get_pending_messages(session_id)
        messages_data = [message.to_dict() for message in pending_messages]
        
        return jsonify({
            "session_id": session_id,
            "pending_messages": messages_data,
            "count": len(messages_data),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/performance', methods=['GET'])
def get_performance_statistics():
    """Get comprehensive performance statistics and optimization metrics"""
    try:
        # Get connection manager performance
        conn_stats = connection_manager.get_statistics()
        
        # Get performance monitor statistics
        perf_stats = performance_monitor.get_statistics()
        
        # Get message queue statistics
        queue_stats = {
            "message_queue_size": len(message_queue.queue),
            "priority_queue_size": message_queue.priority_queue.qsize(),
            "max_queue_size": message_queue.max_queue_size,
            "batch_size": message_queue.batch_size
        }
        
        # Get audio processing pool statistics
        pool_stats = {
            "thread_pool_active": not audio_pool._shutdown,
            "max_workers": audio_pool.executor._max_workers if hasattr(audio_pool.executor, '_max_workers') else 'unknown'
        }
        
        # Get system resource usage (optional, requires psutil)
        system_stats = {}
        try:
            import psutil
            process = psutil.Process()
            system_stats = {
                "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "thread_count": process.num_threads(),
                "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 'unknown'
            }
        except ImportError:
            system_stats = {"note": "psutil not available for detailed system metrics"}
        
        return jsonify({
            "performance_overview": {
                "avg_audio_processing_time": perf_stats.get('audio_processing_times', {}).get('average', 0),
                "avg_transcription_time": perf_stats.get('transcription_times', {}).get('average', 0),
                "avg_websocket_message_time": perf_stats.get('websocket_message_times', {}).get('average', 0),
                "active_connections": conn_stats.get('active_connections', 0),
                "pool_efficiency": conn_stats.get('performance_metrics', {}).get('pool_efficiency_percent', 0)
            },
            "connection_manager": conn_stats,
            "performance_metrics": perf_stats,
            "message_queue": queue_stats,
            "audio_processing_pool": pool_stats,
            "system_resources": system_stats,
            "optimization_features": {
                "connection_pooling": conn_stats.get('configuration', {}).get('connection_pooling_enabled', False),
                "async_audio_processing": True,
                "message_batching": True,
                "performance_monitoring": True,
                "memory_optimization": True
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get performance statistics: {e}")
        return jsonify({"error": str(e)}), 500

# Heartbeat callback functions
def handle_connection_lost(connection_id: str):
    """Handle lost connection callback from heartbeat manager"""
    logger.warning(f"Connection lost detected by heartbeat: {connection_id}")
    
    # Handle session disconnection for reconnection support
    session_info = reconnection_manager.handle_disconnection(connection_id)
    
    # Remove from connection manager
    connection_info = connection_manager.remove_connection(connection_id)
    
    # Remove from message router
    message_router.cleanup_connection(connection_id)
    
    # Remove from authentication
    simple_auth.disconnect_connection(connection_id)
    
    # Create error info for tracking
    error_info = ErrorInfo(
        category=ErrorCategory.CONNECTION_TIMEOUT,
        severity=ErrorSeverity.MEDIUM,
        message="Connection lost due to heartbeat timeout",
        connection_id=connection_id,
        suggested_action="Client should reconnect"
    )
    error_handler.handle_error(error_info)
    
    # Emit disconnection event to any remaining session rooms
    try:
        socketio.emit('connection_lost', {
            'connection_id': connection_id,
            'session_id': session_info.session_id if session_info else None,
            'reason': 'heartbeat_timeout',
            'timestamp': datetime.now().isoformat()
        }, room=connection_id)
    except Exception as e:
        logger.error(f"Failed to emit connection_lost event: {e}")

def handle_connection_warning(connection_id: str, heartbeat_info):
    """Handle connection warning callback from heartbeat manager"""
    logger.warning(f"Connection warning for {connection_id}: {heartbeat_info.state.value} "
                  f"(missed: {heartbeat_info.missed_heartbeats})")
    
    # Update connection state in connection manager
    if heartbeat_info.state == HeartbeatState.CRITICAL:
        connection_manager.set_connection_state(connection_id, ConnectionState.ERROR)
    
    # Emit warning to client
    try:
        socketio.emit('connection_warning', {
            'connection_id': connection_id,
            'state': heartbeat_info.state.value,
            'missed_heartbeats': heartbeat_info.missed_heartbeats,
            'timestamp': datetime.now().isoformat()
        }, room=connection_id)
    except Exception as e:
        logger.error(f"Failed to emit connection_warning event: {e}")

# Message router integration
def register_message_routes():
    """Register all WebSocket message handlers with the message router"""
    
    # Connection management routes
    message_router.register_route(
        MessageType.CONNECT,
        handle_connect_message,
        permission=RoutePermission.PUBLIC,
        description="Handle new WebSocket connections"
    )
    
    message_router.register_route(
        MessageType.DISCONNECT,
        handle_disconnect_message,
        permission=RoutePermission.PUBLIC,
        description="Handle WebSocket disconnections"
    )
    
    # Session management routes
    message_router.register_route(
        MessageType.JOIN_SESSION,
        handle_join_session_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=10,  # Max 10 joins per minute
        description="Join a transcription session"
    )
    
    message_router.register_route(
        MessageType.LEAVE_SESSION,
        handle_leave_session_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=10,  # Max 10 leaves per minute
        description="Leave a transcription session"
    )
    
    # Audio streaming routes
    message_router.register_route(
        MessageType.TRANSCRIBE_STREAM,
        handle_transcribe_stream_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=120,  # Max 120 audio chunks per minute (2 per second)
        description="Process real-time audio transcription"
    )
    
    # Heartbeat routes
    message_router.register_route(
        MessageType.PING,
        handle_ping_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=120,  # Max 120 pings per minute
        validate_schema=False,
        description="Handle ping messages for connection health"
    )
    
    message_router.register_route(
        MessageType.PONG,
        handle_pong_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=120,  # Max 120 pongs per minute
        validate_schema=False,
        description="Handle pong responses for connection health"
    )
    
    message_router.register_route(
        MessageType.HEARTBEAT,
        handle_heartbeat_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=60,  # Max 60 heartbeats per minute
        validate_schema=False,
        description="Handle heartbeat messages for connection monitoring"
    )
    
    # Event subscription routes
    message_router.register_route(
        MessageType.SUBSCRIBE_EVENTS,
        handle_subscribe_events_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=10,  # Max 10 subscriptions per minute
        description="Subscribe to server events"
    )
    
    message_router.register_route(
        MessageType.UNSUBSCRIBE_EVENTS,
        handle_unsubscribe_events_message,
        permission=RoutePermission.PUBLIC,
        rate_limit=10,  # Max 10 unsubscriptions per minute
        description="Unsubscribe from server events"
    )

# Message handler functions for router integration
def handle_connect_message(context):
    """Router-compatible connect handler"""
    return handle_connect()

def handle_disconnect_message(context):
    """Router-compatible disconnect handler"""
    return handle_disconnect()

def handle_join_session_message(context):
    """Router-compatible join session handler"""
    return handle_join_session(context.data)

def handle_leave_session_message(context):
    """Router-compatible leave session handler"""
    return handle_leave_session(context.data)

def handle_transcribe_stream_message(context):
    """Router-compatible transcribe stream handler"""
    return handle_transcribe_stream(context.data)

def handle_ping_message(context):
    """Router-compatible ping handler"""
    return handle_ping(context.data)

def handle_pong_message(context):
    """Router-compatible pong handler"""
    return handle_pong(context.data)

def handle_heartbeat_message(context):
    """Router-compatible heartbeat handler"""
    return handle_heartbeat(context.data)

def handle_subscribe_events_message(context):
    """Handle event subscription requests"""
    try:
        event_types = context.data.get('events', [])
        if not isinstance(event_types, list):
            return {"error": "Events must be a list"}
        
        message_router.subscribe_to_events(context.connection_id, event_types)
        
        return {
            "status": "subscribed",
            "events": event_types,
            "connection_id": context.connection_id
        }
    except Exception as e:
        logger.error(f"Error subscribing to events: {e}")
        return {"error": str(e)}

def handle_unsubscribe_events_message(context):
    """Handle event unsubscription requests"""
    try:
        event_types = context.data.get('events')  # None means unsubscribe from all
        
        message_router.unsubscribe_from_events(context.connection_id, event_types)
        
        return {
            "status": "unsubscribed",
            "events": event_types or "all",
            "connection_id": context.connection_id
        }
    except Exception as e:
        logger.error(f"Error unsubscribing from events: {e}")
        return {"error": str(e)}

# Register all routes
register_message_routes()

# Add authentication middleware to message router
message_router.add_global_middleware(auth_middleware)

# WebSocket events for real-time streaming
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    try:
        # Get client information
        client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        user_agent = request.environ.get('HTTP_USER_AGENT', '')
        
        # Add connection to manager
        if connection_manager.add_connection(request.sid, client_ip, user_agent):
            connection_manager.set_connection_state(request.sid, ConnectionState.CONNECTED)
            
            # Add to heartbeat monitoring
            heartbeat_manager.add_connection(request.sid)
            
            # Create session for reconnection handling
            session_id = reconnection_manager.create_session(request.sid)
            
            logger.info(f"Client connected: {request.sid} from {client_ip}, session: {session_id}")
            emit('connected', {
                'status': 'connected', 
                'sid': request.sid,
                'session_id': session_id,
                'heartbeat_enabled': True,
                'ping_interval': heartbeat_manager.ping_interval,
                'auth_required': True,  # Indicate that authentication is required
                'reconnection_enabled': True,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Handle connection limit exceeded
            error_info = ErrorInfo(
                category=ErrorCategory.CONNECTION_LIMIT_EXCEEDED,
                severity=ErrorSeverity.MEDIUM,
                message="Connection limit exceeded",
                details=f"Maximum connections per IP ({connection_manager.max_connections_per_ip}) exceeded",
                connection_id=request.sid,
                suggested_action="Wait for existing connections to close or try from a different IP"
            )
            error_handler.handle_error(error_info)
            
            emit('error', error_info.to_websocket_response()['error'])
            return False
    except Exception as e:
        error_info = create_connection_error(
            "Failed to establish WebSocket connection",
            str(e),
            request.sid
        )
        error_handler.handle_error(error_info)
        emit('error', error_info.to_websocket_response()['error'])
        return False

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    # Handle session disconnection for reconnection support
    session_info = reconnection_manager.handle_disconnection(request.sid)
    
    # Remove from heartbeat monitoring
    heartbeat_manager.remove_connection(request.sid)
    
    # Remove from message router
    message_router.cleanup_connection(request.sid)
    
    # Remove from authentication
    simple_auth.disconnect_connection(request.sid)
    
    # Remove from connection manager
    connection_info = connection_manager.remove_connection(request.sid)
    
    if connection_info:
        duration = connection_info.get_connection_duration()
        if session_info:
            logger.info(f"Client disconnected: {request.sid} (duration: {duration}, session: {session_info.session_id} preserved for reconnection)")
        else:
            logger.info(f"Client disconnected: {request.sid} (duration: {duration})")
    else:
        logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join_session')
def handle_join_session(data):
    """Join a transcription session room"""
    session_id = data.get('session_id')
    if session_id:
        # Update connection manager
        if connection_manager.join_session(request.sid, session_id):
            connection_manager.update_connection_activity(request.sid, messages_received=1)
            
            # Join Flask-SocketIO room
            join_room(session_id)
            
            logger.info(f"Client {request.sid} joined session {session_id}")
            emit('joined_session', {
                'session_id': session_id, 
                'status': 'joined',
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('error', {'message': 'Failed to join session'})

@socketio.on('leave_session')
def handle_leave_session(data):
    """Leave a transcription session room"""
    session_id = data.get('session_id')
    if session_id:
        # Update connection manager
        if connection_manager.leave_session(request.sid):
            connection_manager.update_connection_activity(request.sid, messages_received=1)
            
            # Leave Flask-SocketIO room
            leave_room(session_id)
            
            logger.info(f"Client {request.sid} left session {session_id}")
            emit('left_session', {
                'session_id': session_id, 
                'status': 'left',
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('error', {'message': 'Failed to leave session'})

@socketio.on('transcribe_stream')
def handle_transcribe_stream(data):
    """Handle real-time streaming transcription via WebSocket"""
    if whisper_service is None:
        error_info = ErrorInfo(
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.HIGH,
            message="Whisper service not initialized",
            connection_id=request.sid,
            suggested_action="Wait for service to initialize or contact administrator"
        )
        error_handler.handle_error(error_info)
        emit('error', error_info.to_websocket_response()['error'])
        return
    
    # Update connection activity and state
    connection_manager.update_connection_activity(request.sid, messages_received=1)
    connection_manager.set_connection_state(request.sid, ConnectionState.STREAMING)
    
    try:
        # Validate request data
        if not isinstance(data, dict):
            error_info = create_validation_error("Invalid request format - expected JSON object")
            error_info.connection_id = request.sid
            error_handler.handle_error(error_info)
            emit('error', error_info.to_websocket_response()['error'])
            return
        
        # Create transcription request
        audio_data = data.get('audio_data')  # Base64 encoded audio
        if not audio_data:
            error_info = create_validation_error("No audio data provided", "audio_data")
            error_info.connection_id = request.sid
            error_handler.handle_error(error_info)
            emit('error', error_info.to_websocket_response()['error'])
            return
        
        # Decode and process audio
        try:
            import base64
            audio_bytes = base64.b64decode(audio_data)
            connection_manager.update_connection_activity(request.sid, bytes_received=len(audio_bytes))
            
            # Check audio size limits
            max_audio_size = 10 * 1024 * 1024  # 10MB limit
            if len(audio_bytes) > max_audio_size:
                error_info = ErrorInfo(
                    category=ErrorCategory.AUDIO_TOO_LARGE,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Audio data too large: {len(audio_bytes)} bytes (max: {max_audio_size})",
                    connection_id=request.sid,
                    session_id=data.get('session_id'),
                    suggested_action="Reduce audio chunk size or duration"
                )
                error_handler.handle_error(error_info)
                emit('error', error_info.to_websocket_response()['error'])
                return
            
            audio_array = _process_audio_data(audio_bytes)
        except Exception as e:
            error_info = create_audio_error(
                "Failed to process audio data",
                str(e),
                data.get('session_id')
            )
            error_info.connection_id = request.sid
            error_handler.handle_error(error_info)
            emit('error', error_info.to_websocket_response()['error'])
            return
        
        transcription_request = TranscriptionRequest(
            audio_data=audio_array,
            model_name=data.get('model_name', 'whisper-base'),
            language=data.get('language'),
            session_id=data.get('session_id'),
            streaming=True,
            sample_rate=data.get('sample_rate', 16000),
            enable_vad=data.get('enable_vad', True)
        )
        
        # Run streaming transcription in background
        def run_streaming():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def stream_to_client():
                try:
                    async for result in whisper_service.transcribe_stream(transcription_request):
                        transcription_data = {
                            'text': result.text,
                            'segments': result.segments,
                            'confidence': result.confidence_score,
                            'timestamp': result.timestamp,
                            'session_id': result.session_id
                        }
                        
                        # Try to emit to client
                        try:
                            socketio.emit('transcription_result', transcription_data, room=request.sid)
                        except Exception as emit_error:
                            # If emit fails, try to buffer the message for reconnection
                            logger.warning(f"Failed to emit transcription result to {request.sid}: {emit_error}")
                            session_info = reconnection_manager.get_session_by_connection(request.sid)
                            if session_info:
                                reconnection_manager.buffer_message(
                                    session_info.session_id,
                                    'transcription_result',
                                    transcription_data,
                                    priority=1  # High priority for transcription results
                                )
                    
                    # Send completion signal
                    completion_data = {
                        'session_id': transcription_request.session_id,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    try:
                        socketio.emit('transcription_complete', completion_data, room=request.sid)
                    except Exception as emit_error:
                        # Buffer completion signal if emit fails
                        logger.warning(f"Failed to emit transcription complete to {request.sid}: {emit_error}")
                        session_info = reconnection_manager.get_session_by_connection(request.sid)
                        if session_info:
                            reconnection_manager.buffer_message(
                                session_info.session_id,
                                'transcription_complete',
                                completion_data,
                                priority=2  # High priority for completion signals
                            )
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    socketio.emit('transcription_error', {
                        'error': str(e),
                        'session_id': transcription_request.session_id
                    }, room=request.sid)
            
            loop.run_until_complete(stream_to_client())
        
        # Start streaming in background thread
        streaming_thread = threading.Thread(target=run_streaming)
        streaming_thread.daemon = True
        streaming_thread.start()
        
    except Exception as e:
        error_info = create_system_error(
            "WebSocket streaming failed",
            str(e)
        )
        error_info.connection_id = request.sid
        error_info.session_id = data.get('session_id') if isinstance(data, dict) else None
        error_handler.handle_error(error_info)
        emit('error', error_info.to_websocket_response()['error'])

@socketio.on('pong')
def handle_pong(data):
    """Handle pong response from client"""
    try:
        # Extract ping timestamp if provided
        ping_timestamp = data.get('timestamp') if isinstance(data, dict) else None
        
        # Handle pong in heartbeat manager
        heartbeat_manager.handle_pong(request.sid, ping_timestamp)
        
        logger.debug(f"Received pong from {request.sid}")
        
    except Exception as e:
        logger.error(f"Error handling pong from {request.sid}: {e}")

@socketio.on('heartbeat')
def handle_heartbeat(data):
    """Handle heartbeat message from client (alternative to ping/pong)"""
    try:
        # Handle heartbeat in heartbeat manager
        heartbeat_manager.handle_client_heartbeat(request.sid)
        
        # Update connection activity
        connection_manager.update_connection_activity(request.sid, messages_received=1)
        
        # Send heartbeat acknowledgment
        emit('heartbeat_ack', {
            'timestamp': datetime.now().isoformat(),
            'connection_id': request.sid
        })
        
        logger.debug(f"Received heartbeat from {request.sid}")
        
    except Exception as e:
        logger.error(f"Error handling heartbeat from {request.sid}: {e}")

@socketio.on('ping')
def handle_ping(data):
    """Handle ping from client (respond with pong)"""
    try:
        # Send pong response
        pong_data = {
            'type': 'pong',
            'timestamp': datetime.now().isoformat(),
            'original_timestamp': data.get('timestamp') if isinstance(data, dict) else None
        }
        
        emit('pong', pong_data)
        
        # Update connection activity
        connection_manager.update_connection_activity(request.sid, messages_received=1)
        
        logger.debug(f"Responded to ping from {request.sid}")
        
    except Exception as e:
        logger.error(f"Error handling ping from {request.sid}: {e}")

@socketio.on('route_message')
def handle_route_message(data):
    """Handle messages through the message router"""
    try:
        if not isinstance(data, dict):
            emit('error', {'message': 'Invalid message format'})
            return
        
        message_type = data.get('type')
        message_data = data.get('data', {})
        
        if not message_type:
            emit('error', {'message': 'Message type required'})
            return
        
        # Get client information
        client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        user_agent = request.environ.get('HTTP_USER_AGENT', '')
        
        # Route the message
        result = message_router.route_message(
            connection_id=request.sid,
            message_type=message_type,
            data=message_data,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Send response
        if result:
            emit('route_response', {
                'type': message_type,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error routing message from {request.sid}: {e}")
        emit('error', {'message': f'Routing error: {str(e)}'})

@socketio.on('subscribe_events')
def handle_subscribe_events(data):
    """Handle event subscription via WebSocket"""
    try:
        result = handle_subscribe_events_message(MessageContext(
            connection_id=request.sid,
            message_type=MessageType.SUBSCRIBE_EVENTS,
            data=data
        ))
        emit('subscription_result', result)
    except Exception as e:
        logger.error(f"Error handling event subscription: {e}")
        emit('error', {'message': str(e)})

@socketio.on('unsubscribe_events')
def handle_unsubscribe_events(data):
    """Handle event unsubscription via WebSocket"""
    try:
        result = handle_unsubscribe_events_message(MessageContext(
            connection_id=request.sid,
            message_type=MessageType.UNSUBSCRIBE_EVENTS,
            data=data
        ))
        emit('unsubscription_result', result)
    except Exception as e:
        logger.error(f"Error handling event unsubscription: {e}")
        emit('error', {'message': str(e)})

@socketio.on('authenticate')
def handle_authenticate(data):
    """Handle WebSocket authentication"""
    try:
        if not isinstance(data, dict):
            emit('auth_error', {'message': 'Invalid authentication data format'})
            return
        
        token = data.get('token')
        if not token:
            emit('auth_error', {'message': 'Token required'})
            return
        
        # Associate token with connection
        success = simple_auth.associate_connection(token, request.sid)
        if success:
            auth_token = simple_auth.get_connection_auth(request.sid)
            
            # Update session with user information
            session_info = reconnection_manager.get_session_by_connection(request.sid)
            if session_info:
                reconnection_manager.update_session_data(session_info.session_id, {
                    'user_id': auth_token.user_id,
                    'role': auth_token.role.value
                })
            
            emit('authenticated', {
                'status': 'authenticated',
                'user_id': auth_token.user_id,
                'role': auth_token.role.value,
                'connection_id': request.sid,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"Connection {request.sid} authenticated as {auth_token.user_id}")
        else:
            emit('auth_error', {'message': 'Invalid or expired token'})
            
    except Exception as e:
        logger.error(f"Authentication error for {request.sid}: {e}")
        emit('auth_error', {'message': 'Authentication failed'})

@socketio.on('reconnect_session')
def handle_reconnect_session(data):
    """Handle session reconnection request"""
    try:
        session_id = data.get('session_id')
        user_id = data.get('user_id')  # Optional for verification
        
        if not session_id:
            emit('reconnection_error', {'message': 'Session ID required'})
            return
        
        # Attempt reconnection
        success = reconnection_manager.attempt_reconnection(request.sid, session_id, user_id)
        
        if success:
            # Re-establish connection in other managers
            session_info = reconnection_manager.get_session(session_id)
            if session_info and session_info.session_data.get('user_id'):
                # Re-authenticate if user was previously authenticated
                user_tokens = simple_auth.get_user_tokens(session_info.session_data['user_id'])
                if user_tokens:
                    # Use the first valid token
                    simple_auth.associate_connection(user_tokens[0], request.sid)
            
            # Add back to connection manager
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            user_agent = request.environ.get('HTTP_USER_AGENT', '')
            connection_manager.add_connection(request.sid, client_ip, user_agent)
            connection_manager.set_connection_state(request.sid, ConnectionState.CONNECTED)
            
            # Add back to heartbeat monitoring
            heartbeat_manager.add_connection(request.sid)
            
            # Send pending messages
            pending_messages = reconnection_manager.get_pending_messages(session_id)
            delivered_ids = []
            
            for message in pending_messages:
                try:
                    emit('buffered_message', {
                        'message_id': message.message_id,
                        'type': message.message_type,
                        'data': message.data,
                        'timestamp': message.timestamp.isoformat(),
                        'buffered_at': message.timestamp.isoformat()
                    })
                    delivered_ids.append(message.message_id)
                except Exception as e:
                    logger.error(f"Failed to deliver buffered message {message.message_id}: {e}")
                    reconnection_manager.mark_message_failed(session_id, message.message_id)
            
            # Mark delivered messages
            if delivered_ids:
                reconnection_manager.mark_messages_delivered(session_id, delivered_ids)
            
            emit('reconnection_success', {
                'status': 'reconnected',
                'session_id': session_id,
                'pending_messages_delivered': len(delivered_ids),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Session {session_id} successfully reconnected to {request.sid}")
        else:
            emit('reconnection_error', {'message': 'Failed to reconnect session'})
            
    except Exception as e:
        logger.error(f"Reconnection error: {e}")
        emit('reconnection_error', {'message': 'Reconnection failed'})

@socketio.on('get_session_info')
def handle_get_session_info(data):
    """Get information about the current session"""
    try:
        session_info = reconnection_manager.get_session_by_connection(request.sid)
        if session_info:
            emit('session_info', {
                'session_id': session_info.session_id,
                'state': session_info.state.value,
                'created_at': session_info.created_at.isoformat(),
                'last_activity': session_info.last_activity.isoformat(),
                'buffered_messages': len(session_info.buffered_messages),
                'session_data': session_info.session_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('session_error', {'message': 'No session found for this connection'})
    except Exception as e:
        logger.error(f"Get session info error: {e}")
        emit('session_error', {'message': 'Failed to get session info'})

@socketio.on('buffer_message')
def handle_buffer_message(data):
    """Buffer a message for potential disconnected sessions"""
    try:
        target_session_id = data.get('target_session_id')
        message_type = data.get('message_type', 'custom')
        message_data = data.get('data', {})
        priority = data.get('priority', 0)
        
        if not target_session_id:
            emit('buffer_error', {'message': 'Target session ID required'})
            return
        
        success = reconnection_manager.buffer_message(
            target_session_id, message_type, message_data, priority
        )
        
        if success:
            emit('buffer_success', {
                'status': 'buffered',
                'target_session_id': target_session_id,
                'message_type': message_type,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('buffer_error', {'message': 'Failed to buffer message'})
            
    except Exception as e:
        logger.error(f"Buffer message error: {e}")
        emit('buffer_error', {'message': 'Failed to buffer message'})

# Helper functions
# Audio processing configuration
AUDIO_CONFIG = {
    'default_sample_rate': 16000,
    'resampling_quality': 'kaiser_fast',  # Options: 'kaiser_best', 'kaiser_fast', 'scipy'
    'enable_format_cache': True,
    'max_cache_size': 50,
    'quality_thresholds': {
        'silence_rms': 0.0001,
        'quiet_rms': 0.005,
        'clipping_threshold': 0.99
    }
}

# Format detection cache for improved performance
_format_cache = {}
_format_cache_size = 0

def _detect_audio_format_optimized(audio_data: bytes) -> str:
    """Optimized format detection with smart magic number checking"""
    global _format_cache, _format_cache_size
    
    # Use first 32 bytes as cache key for format detection
    cache_key = audio_data[:32] if len(audio_data) >= 32 else audio_data
    cache_hash = hash(cache_key)
    
    if AUDIO_CONFIG['enable_format_cache'] and cache_hash in _format_cache:
        return _format_cache[cache_hash]
    
    # Enhanced format detection with more precise magic numbers
    format_hint = "unknown"
    
    # WAV format detection (RIFF header)
    if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
        format_hint = "wav"
    # MP4/M4A format detection (ftyp box)
    elif b'ftyp' in audio_data[:32]:
        if b'M4A ' in audio_data[:32] or b'mp41' in audio_data[:32] or b'mp42' in audio_data[:32]:
            format_hint = "mp4"
        elif b'qt  ' in audio_data[:32]:
            format_hint = "mov"
    # MP3 format detection (ID3 tag or frame sync)
    elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb') or audio_data.startswith(b'\xff\xfa'):
        format_hint = "mp3"
    # WebM format detection (EBML header)
    elif audio_data.startswith(b'\x1a\x45\xdf\xa3'):
        format_hint = "webm"
    # OGG format detection
    elif audio_data.startswith(b'OggS'):
        format_hint = "ogg"
    # FLAC format detection
    elif audio_data.startswith(b'fLaC'):
        format_hint = "flac"
    # AAC format detection (ADTS header)
    elif audio_data.startswith(b'\xff\xf1') or audio_data.startswith(b'\xff\xf9'):
        format_hint = "aac"
    
    # Update cache if enabled
    if AUDIO_CONFIG['enable_format_cache']:
        if _format_cache_size >= AUDIO_CONFIG['max_cache_size']:
            # Clear half the cache when full
            _format_cache.clear()
            _format_cache_size = 0
        _format_cache[cache_hash] = format_hint
        _format_cache_size += 1
    
    return format_hint

def _high_quality_resample(audio: np.ndarray, orig_sr: int, target_sr: int, quality: str = 'kaiser_fast') -> np.ndarray:
    """High-quality resampling with configurable quality settings"""
    if orig_sr == target_sr:
        return audio
    
    try:
        if quality == 'kaiser_best':
            # Highest quality resampling for critical applications
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_best')
        elif quality == 'kaiser_fast':
            # Good quality with faster processing
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast')
        elif quality == 'scipy':
            # Use scipy for alternative resampling
            try:
                from scipy import signal
                resample_ratio = target_sr / orig_sr
                num_samples = int(len(audio) * resample_ratio)
                return signal.resample(audio, num_samples).astype(np.float32)
            except ImportError:
                logger.warning("Scipy not available, falling back to librosa for resampling")
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        else:
            # Default librosa resampling
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        logger.warning(f"High-quality resampling failed ({quality}), using default: {e}")
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def _calculate_audio_quality_metrics(audio: np.ndarray, sr: int) -> dict:
    """Calculate comprehensive audio quality metrics"""
    metrics = {}
    
    # Basic statistics
    metrics['duration'] = len(audio) / sr
    metrics['samples'] = len(audio)
    metrics['sample_rate'] = sr
    metrics['channels'] = 1 if len(audio.shape) == 1 else audio.shape[1]
    
    # Amplitude metrics
    metrics['rms'] = float(np.sqrt(np.mean(audio**2)))
    metrics['peak'] = float(np.max(np.abs(audio)))
    metrics['mean'] = float(np.mean(audio))
    metrics['std'] = float(np.std(audio))
    
    # Dynamic range
    metrics['dynamic_range'] = float(metrics['peak'] - np.min(np.abs(audio[audio != 0]))) if np.any(audio != 0) else 0.0
    
    # Zero crossing rate (speech indicator)
    zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
    metrics['zero_crossing_rate'] = zero_crossings / len(audio)
    
    # Spectral centroid (brightness measure)
    try:
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))
    except:
        metrics['spectral_centroid_mean'] = 0.0
        metrics['spectral_centroid_std'] = 0.0
    
    # Quality assessment flags
    metrics['is_silent'] = metrics['rms'] < AUDIO_CONFIG['quality_thresholds']['silence_rms']
    metrics['is_quiet'] = metrics['rms'] < AUDIO_CONFIG['quality_thresholds']['quiet_rms']
    metrics['is_clipped'] = metrics['peak'] >= AUDIO_CONFIG['quality_thresholds']['clipping_threshold']
    
    return metrics

def _process_audio_data(audio_data: bytes, enhance: bool = False, target_sr: int = None, quality: str = None) -> np.ndarray:
    """Optimized audio processing with smart format detection and minimal memory usage"""
    start_time = time.time()
    
    # Use configuration defaults
    target_sr = target_sr or AUDIO_CONFIG['default_sample_rate']
    quality = quality or AUDIO_CONFIG['resampling_quality']
    
    try:
        # Enhanced format detection
        format_hint = _detect_audio_format_optimized(audio_data)
        logger.info(f"[AUDIO] Processing {len(audio_data)} bytes, detected format: {format_hint}")
        
        # Initialize quality tracking
        processing_stages = []
        audio_array = None
        current_sr = None
        
        # Stage 1: Format-specific fast paths for optimal processing
        try:
            if format_hint == "wav":
                # WAV files: Direct soundfile processing (fastest path)
                audio_io = io.BytesIO(audio_data)
                audio_array, current_sr = sf.read(audio_io, dtype=np.float32)
                processing_stages.append("soundfile_direct")
                logger.info(f"[AUDIO] WAV fast path: {len(audio_array)} samples at {current_sr}Hz")
                
            elif format_hint in ["mp3", "flac", "ogg"]:
                # These formats: Try soundfile first, then librosa
                try:
                    audio_io = io.BytesIO(audio_data)
                    audio_array, current_sr = sf.read(audio_io, dtype=np.float32)
                    processing_stages.append("soundfile_direct")
                except Exception:
                    # Fallback to librosa for better format support
                    with tempfile.NamedTemporaryFile(suffix=f'.{format_hint}', delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_file.flush()
                        audio_array, current_sr = librosa.load(tmp_file.name, sr=None, dtype=np.float32)
                        os.unlink(tmp_file.name)
                    processing_stages.append("librosa_file")
                logger.info(f"[AUDIO] {format_hint.upper()} processing: {len(audio_array)} samples at {current_sr}Hz")
                
            elif format_hint in ["mp4", "webm", "mov", "aac"]:
                # Complex formats: Use pydub with ffmpeg or librosa
                ffmpeg_available = bool(which('ffmpeg'))
                
                if ffmpeg_available:
                    # Use pydub for better format support
                    suffix_map = {'mp4': '.mp4', 'webm': '.webm', 'mov': '.mov', 'aac': '.aac'}
                    suffix = suffix_map.get(format_hint, '.mp4')
                    
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_file.flush()
                        
                        audio_segment = AudioSegment.from_file(tmp_file.name)
                        os.unlink(tmp_file.name)
                        
                        # Convert to mono and target sample rate efficiently
                        if audio_segment.channels > 1:
                            audio_segment = audio_segment.set_channels(1)
                        
                        # Get raw audio data
                        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                        
                        # Normalize based on bit depth
                        if audio_segment.sample_width == 2:  # 16-bit
                            audio_array = samples / 32768.0
                        elif audio_segment.sample_width == 4:  # 32-bit
                            audio_array = samples / 2147483648.0
                        else:  # 8-bit or other
                            audio_array = (samples - 128) / 128.0
                        
                        current_sr = audio_segment.frame_rate
                        processing_stages.append("pydub_ffmpeg")
                        
                else:
                    # Fallback to librosa without ffmpeg
                    with tempfile.NamedTemporaryFile(suffix=f'.{format_hint}', delete=False) as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_file.flush()
                        audio_array, current_sr = librosa.load(tmp_file.name, sr=None, dtype=np.float32)
                        os.unlink(tmp_file.name)
                    processing_stages.append("librosa_file")
                    
                logger.info(f"[AUDIO] {format_hint.upper()} processing: {len(audio_array)} samples at {current_sr}Hz")
                
            else:
                # Unknown format: Try all methods in order of preference
                methods = [("soundfile", lambda: sf.read(io.BytesIO(audio_data), dtype=np.float32)),
                          ("librosa_memory", lambda: librosa.load(io.BytesIO(audio_data), sr=None, dtype=np.float32))]
                
                for method_name, method_func in methods:
                    try:
                        audio_array, current_sr = method_func()
                        processing_stages.append(method_name)
                        break
                    except Exception as e:
                        logger.debug(f"[AUDIO] {method_name} failed: {e}")
                        continue
                
                if audio_array is None:
                    raise Exception("All direct processing methods failed")
                    
        except Exception as direct_error:
            logger.warning(f"[AUDIO] Direct processing failed: {direct_error}")
            # Universal fallback: Create temp file and use librosa
            suffix = '.wav' if format_hint == 'unknown' else f'.{format_hint}'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                audio_array, current_sr = librosa.load(tmp_file.name, sr=target_sr, dtype=np.float32)
                os.unlink(tmp_file.name)
            processing_stages.append("librosa_fallback")
            current_sr = target_sr  # librosa already resampled
        
        # Ensure we have mono audio (convert stereo to mono in-place)
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Stage 2: High-quality resampling if needed
        if current_sr != target_sr:
            logger.info(f"[AUDIO] Resampling from {current_sr}Hz to {target_sr}Hz using {quality}")
            audio_array = _high_quality_resample(audio_array, current_sr, target_sr, quality)
            processing_stages.append(f"resample_{quality}")
        
        # Stage 3: Calculate comprehensive quality metrics
        quality_metrics = _calculate_audio_quality_metrics(audio_array, target_sr)
        
        # Stage 4: Apply quality-based processing
        if quality_metrics['is_clipped']:
            logger.warning(f"[AUDIO] Audio clipping detected (peak: {quality_metrics['peak']:.3f})")
            # Apply soft clipping in-place to prevent artifacts
            np.tanh(audio_array * 0.9, out=audio_array)
            audio_array /= 0.9
            processing_stages.append("soft_clipping")
        
        if quality_metrics['is_silent']:
            logger.warning(f"[AUDIO] Silent audio detected (RMS: {quality_metrics['rms']:.6f})")
            # Return minimal silence instead of processing
            return np.zeros(target_sr, dtype=np.float32)  # 1 second of silence
        elif quality_metrics['is_quiet']:
            logger.info(f"[AUDIO] Quiet audio detected (RMS: {quality_metrics['rms']:.6f}) - continuing")
        
        # Stage 5: Optional enhancement
        if enhance:
            audio_array = _enhance_audio_optimized(audio_array)
            processing_stages.append("enhancement")
        
        # Record performance metrics
        processing_time = time.time() - start_time
        performance_monitor.record_metric('audio_processing_times', processing_time)
        
        # Log comprehensive results
        logger.info(f"[AUDIO] Processing complete: {quality_metrics['duration']:.2f}s, "
                   f"RMS: {quality_metrics['rms']:.4f}, Peak: {quality_metrics['peak']:.4f}")
        logger.info(f"[AUDIO] Processing stages: {' -> '.join(processing_stages)}")
        logger.info(f"[AUDIO] Quality metrics: ZCR={quality_metrics['zero_crossing_rate']:.4f}, "
                   f"SC={quality_metrics['spectral_centroid_mean']:.1f}Hz")
        logger.info(f"[AUDIO] Processing time: {processing_time:.3f}s")
        
        return audio_array
        
    except Exception as e:
        logger.error(f"Failed to process audio data: {e}", exc_info=True)
        raise

def _enhance_audio_optimized(audio_array: np.ndarray) -> np.ndarray:
    """Optimized audio enhancement with minimal overhead and in-place operations"""
    try:
        # In-place normalization to prevent unnecessary copies
        max_val = np.abs(audio_array).max()
        if max_val > 0.001:  # Avoid division by very small numbers
            audio_array /= max_val
        
        # Optional: simple high-pass filter for noise reduction
        # Only apply if array is large enough to benefit and not too large to cause slowdown
        if 1000 < len(audio_array) < 500000:  # Between 1000 samples and ~31 seconds at 16kHz
            # Use in-place preemphasis to save memory
            enhanced = librosa.effects.preemphasis(audio_array, coef=0.97)
            # Copy back to original array to maintain in-place operation
            audio_array[:] = enhanced
        
        # Light dynamic range compression for consistent levels
        if len(audio_array) > 0:
            # Simple soft limiting to prevent harsh clipping
            np.clip(audio_array, -0.95, 0.95, out=audio_array)
        
        return audio_array
        
    except Exception as e:
        logger.warning(f"Audio enhancement failed, using unprocessed audio: {e}")
        return audio_array

def _process_audio_async(audio_data: bytes, enhance: bool = False, priority: int = 5):
    """Submit audio processing to thread pool for async handling"""
    future = audio_pool.submit_audio_task(
        _process_audio_data, 
        priority=priority,
        audio_data=audio_data, 
        enhance=enhance
    )
    return future

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

# CORS handlers
@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Session-ID')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.before_request
def handle_preflight():
    """Handle preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({'status': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,X-Session-ID")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS")
        return response

def create_app(config: Optional[Dict] = None):
    """Factory function to create Flask app with configuration"""
    if config:
        app.config.update(config)
    
    return app

# Recovery handlers specific to whisper service
def recover_whisper_device_error(error_info: ErrorInfo):
    """Recovery handler for whisper device errors"""
    global whisper_service
    logger.info("Attempting whisper device error recovery...")
    try:
        if whisper_service:
            # Clear model cache and attempt to reload
            whisper_service.clear_cache()
            logger.info("Cleared whisper model cache for device recovery")
    except Exception as e:
        logger.error(f"Whisper device recovery failed: {e}")

def recover_whisper_memory_error(error_info: ErrorInfo):
    """Recovery handler for whisper memory errors"""
    global whisper_service
    logger.info("Attempting whisper memory error recovery...")
    try:
        if whisper_service:
            # Clear cache and force garbage collection
            whisper_service.clear_cache()
            import gc
            gc.collect()
            logger.info("Cleared whisper cache and forced garbage collection")
    except Exception as e:
        logger.error(f"Whisper memory recovery failed: {e}")

def recover_whisper_connection_error(error_info: ErrorInfo):
    """Recovery handler for whisper connection errors"""
    logger.info("Attempting whisper connection error recovery...")
    try:
        if error_info.connection_id:
            # Clean up connection from manager
            connection_manager.remove_connection(error_info.connection_id)
            logger.info(f"Cleaned up connection {error_info.connection_id}")
    except Exception as e:
        logger.error(f"Whisper connection recovery failed: {e}")

# Register whisper-specific recovery handlers
error_handler.register_recovery_handler(ErrorCategory.DEVICE_ERROR, recover_whisper_device_error)
error_handler.register_recovery_handler(ErrorCategory.OUT_OF_MEMORY, recover_whisper_memory_error)
error_handler.register_recovery_handler(ErrorCategory.CONNECTION_FAILED, recover_whisper_connection_error)
error_handler.register_recovery_handler(ErrorCategory.MODEL_LOAD_FAILED, recover_whisper_device_error)

# Initialize service on startup (for Docker/production usage)
# Skip automatic initialization to avoid event loop conflicts
# Will be initialized when called from main.py

# Configuration and compatibility endpoints
@app.route('/api/config', methods=['GET'])
async def get_whisper_configuration():
    """Get current whisper service configuration and capabilities"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        config_info = {
            "service_mode": "orchestration" if getattr(whisper_service, 'orchestration_mode', False) else "legacy",
            "orchestration_mode": getattr(whisper_service, 'orchestration_mode', False),
            "configuration": {
                "sample_rate": whisper_service.config.get("sample_rate", 16000),
                "buffer_duration": whisper_service.config.get("buffer_duration", 4.0),
                "inference_interval": whisper_service.config.get("inference_interval", 3.0),
                "overlap_duration": whisper_service.config.get("overlap_duration", 0.2),
                "enable_vad": whisper_service.config.get("enable_vad", True),
                "default_model": whisper_service.config.get("default_model", "whisper-base"),
                "max_concurrent_requests": whisper_service.config.get("max_concurrent_requests", 10)
            },
            "capabilities": {
                "internal_chunking": not getattr(whisper_service, 'orchestration_mode', False),
                "orchestration_chunks": getattr(whisper_service, 'orchestration_mode', False),
                "buffer_management": hasattr(whisper_service, 'buffer_manager') and whisper_service.buffer_manager is not None,
                "streaming_support": True,
                "enhanced_processing": True,
                "chunk_metadata_support": True
            },
            "statistics": getattr(whisper_service, 'stats', {}),
            "compatibility": {
                "api_version": "2.0",
                "orchestration_api_version": "1.0",
                "chunk_processing_version": "1.0"
            }
        }
        
        return jsonify(config_info), 200
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        return jsonify({"error": "Failed to retrieve configuration"}), 500


@app.route('/api/compatibility', methods=['GET'])
async def get_compatibility_info():
    """Get compatibility information for orchestration service integration"""
    if whisper_service is None:
        return jsonify({"error": "Service not initialized"}), 503
    
    try:
        compatibility_info = {
            "whisper_service": {
                "version": "2.0",
                "orchestration_ready": True,
                "chunking_compatible": True,
                "metadata_support": True
            },
            "chunking_configuration": {
                "current": {
                    "buffer_duration": whisper_service.config.get("buffer_duration", 4.0),
                    "inference_interval": whisper_service.config.get("inference_interval", 3.0),
                    "overlap_duration": whisper_service.config.get("overlap_duration", 0.2),
                    "sample_rate": whisper_service.config.get("sample_rate", 16000)
                },
                "orchestration_compatible": {
                    "chunk_duration": whisper_service.config.get("inference_interval", 3.0),
                    "overlap_duration": whisper_service.config.get("overlap_duration", 0.2),
                    "processing_interval": whisper_service.config.get("inference_interval", 3.0) * 0.8,
                    "buffer_duration": whisper_service.config.get("buffer_duration", 4.0)
                }
            },
            "endpoints": {
                "orchestration_chunk_processing": "/api/process-chunk",
                "legacy_transcription": "/transcribe",
                "configuration": "/api/config",
                "compatibility": "/api/compatibility",
                "health": "/api/health"
            },
            "migration_notes": [
                "Orchestration mode disables internal audio buffering",
                "Chunk metadata provides processing context", 
                "VAD and enhancement should be applied by orchestration service",
                "Timing information is preserved through chunk metadata",
                "Statistics include both legacy and orchestration metrics"
            ]
        }
        
        return jsonify(compatibility_info), 200
        
    except Exception as e:
        logger.error(f"Failed to get compatibility info: {e}")
        return jsonify({"error": "Failed to retrieve compatibility information"}), 500


# Enhanced validation and error handling functions
async def _validate_transcription_request(flask_request, model_name: str, correlation_id: str):
    """Enhanced transcription request validation"""
    # Validate model name
    if not model_name or not isinstance(model_name, str):
        raise WhisperValidationError(
            "Invalid model name provided",
            correlation_id=correlation_id,
            validation_details={"model_name": model_name}
        )
    
    # Validate audio file presence
    if 'audio' not in flask_request.files:
        raise WhisperValidationError(
            "No audio file provided in request",
            correlation_id=correlation_id,
            validation_details={"files_in_request": list(flask_request.files.keys())}
        )
    
    audio_file = flask_request.files['audio']
    
    # Validate file has content
    if not audio_file.filename:
        raise WhisperValidationError(
            "Audio file has no filename",
            correlation_id=correlation_id,
            validation_details={"file_size": getattr(audio_file, 'content_length', 'unknown')}
        )
    
    # Read and validate audio data
    try:
        audio_data = audio_file.read()
        if len(audio_data) == 0:
            raise AudioCorruptionError(
                "Audio file is empty",
                correlation_id=correlation_id,
                corruption_details={"filename": audio_file.filename}
            )
        
        # Check file size (100MB limit)
        if len(audio_data) > 100 * 1024 * 1024:
            raise WhisperValidationError(
                "Audio file too large (max 100MB)",
                correlation_id=correlation_id,
                validation_details={
                    "file_size": len(audio_data),
                    "max_size": 100 * 1024 * 1024,
                    "filename": audio_file.filename
                }
            )
        
        # Reset file pointer for subsequent reads
        audio_file.seek(0)
        
    except Exception as e:
        if isinstance(e, WhisperProcessingBaseError):
            raise
        raise AudioCorruptionError(
            f"Failed to read audio file: {str(e)}",
            correlation_id=correlation_id,
            corruption_details={
                "filename": audio_file.filename,
                "read_error": str(e)
            }
        )


def _safe_model_loading(model_name: str, correlation_id: str):
    """Safe model loading with error handling"""
    try:
        if whisper_service.model_manager and model_name not in whisper_service.model_manager.pipelines:
            logger.info(f"[WHISPER] [{correlation_id}] 🔄 Model {model_name} not loaded, loading now...")
            
            # Attempt to load model with circuit breaker
            def load_model():
                return whisper_service.load_model(model_name)
            
            success = default_circuit_breaker.call(load_model)
            
            if not success:
                raise ModelLoadingError(
                    f"Failed to load model {model_name}",
                    correlation_id=correlation_id,
                    model_details={"model_name": model_name}
                )
        
        return True
        
    except Exception as e:
        if isinstance(e, WhisperProcessingBaseError):
            raise
        raise ModelLoadingError(
            f"Model loading failed: {str(e)}",
            correlation_id=correlation_id,
            model_details={
                "model_name": model_name,
                "error": str(e)
            }
        )


def _safe_audio_processing(audio_data: bytes, correlation_id: str):
    """Safe audio processing with error handling"""
    try:
        return _process_audio_enhanced(audio_data)
    except Exception as e:
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            raise MemoryError(
                f"Memory error during audio processing: {str(e)}",
                correlation_id=correlation_id,
                memory_details={"audio_size": len(audio_data)}
            )
        elif "cuda" in str(e).lower() or "gpu" in str(e).lower():
            raise HardwareError(
                f"GPU/CUDA error during audio processing: {str(e)}",
                correlation_id=correlation_id,
                hardware_details={"device": "cuda"}
            )
        else:
            raise AudioCorruptionError(
                f"Audio processing failed: {str(e)}",
                correlation_id=correlation_id,
                corruption_details={"processing_error": str(e)}
            )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper Service API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Whisper Service API server on {args.host}:{args.port}")
    
    # Initialize service
    try:
        asyncio.run(initialize_service())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Create a new thread for initialization
            import threading
            import concurrent.futures
            
            def init_service():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(initialize_service())
                finally:
                    loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_service)
                future.result()  # Wait for completion
        else:
            raise
    
    # Run server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Disable reloader in production
        allow_unsafe_werkzeug=True  # Allow Werkzeug in Docker/production
    ) 