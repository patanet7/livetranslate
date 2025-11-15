#!/usr/bin/env python3
"""
Enhanced API Server for Whisper Service with Continuous Streaming

Integrates continuous stream processor, transcript manager, and translation service
for real-time speech-to-text processing with text deduplication and translation.

Features:
- Continuous audio streaming with deduplication
- Complete transcript storage and management
- Integration with translation service
- NPU-optimized processing
- Enterprise WebSocket infrastructure
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import threading

# Import whisper service components
from model_manager import ModelManager
from audio_processor import AudioProcessor
from buffer_manager import RollingBufferManager, BufferConfig
from continuous_stream_processor import ContinuousStreamProcessor, TranslationServiceClient
from transcript_manager import TranscriptManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'whisper-service-secret')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
CORS(app)

# SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global service components
model_manager: Optional[ModelManager] = None
audio_processor: Optional[AudioProcessor] = None
buffer_manager: Optional[RollingBufferManager] = None
transcript_manager: Optional[TranscriptManager] = None
translation_client: Optional[TranslationServiceClient] = None

# Active sessions for continuous streaming
active_sessions: Dict[str, ContinuousStreamProcessor] = {}
session_lock = threading.RLock()

async def initialize_service():
    """Initialize all service components"""
    global model_manager, audio_processor, buffer_manager, transcript_manager, translation_client
    
    try:
        logger.info("Initializing Enhanced Whisper Service...")
        
        # Initialize model manager with NPU optimization
        model_manager = ModelManager()
        default_model = os.getenv('WHISPER_DEFAULT_MODEL', 'whisper-base')
        await model_manager.load_model(default_model)
        logger.info(f"✓ Model manager initialized with {default_model}")
        
        # Initialize audio processor
        audio_processor = AudioProcessor()
        logger.info("✓ Audio processor initialized")
        
        # Initialize buffer manager
        buffer_config = BufferConfig(
            buffer_duration=4.0,  # 4-second sliding windows
            inference_interval=3.0,  # Every 3 seconds
            sample_rate=16000,
            vad_enabled=True,
            enable_speech_enhancement=True
        )
        buffer_manager = RollingBufferManager(buffer_config)
        logger.info("✓ Rolling buffer manager initialized")
        
        # Initialize transcript manager
        storage_dir = os.getenv('TRANSCRIPT_STORAGE_DIR', 'transcripts')
        transcript_manager = TranscriptManager(storage_dir=storage_dir)
        logger.info(f"✓ Transcript manager initialized (storage: {storage_dir})")
        
        # Initialize translation service client
        translation_url = os.getenv('TRANSLATION_SERVICE_URL', 'http://localhost:5003')
        translation_client = TranslationServiceClient(translation_url)
        logger.info(f"✓ Translation client initialized ({translation_url})")
        
        logger.info("🚀 Enhanced Whisper Service fully initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

def transcription_callback(audio_data: np.ndarray, timestamp: float) -> Any:
    """Callback function for transcription processing"""
    try:
        # Use model manager for transcription
        if model_manager and model_manager.pipe:
            result = model_manager.pipe.generate(audio_data)
            return result
        else:
            logger.warning("Model manager not available for transcription")
            return None
    except Exception as e:
        logger.error(f"Transcription callback failed: {e}")
        return None

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with component status"""
    components = {
        'model_manager': model_manager is not None and model_manager.pipe is not None,
        'audio_processor': audio_processor is not None,
        'buffer_manager': buffer_manager is not None,
        'transcript_manager': transcript_manager is not None,
        'translation_client': translation_client is not None
    }
    
    all_healthy = all(components.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'degraded',
        'components': components,
        'active_sessions': len(active_sessions),
        'device': model_manager.device if model_manager else 'unknown',
        'model': model_manager.current_model if model_manager else 'none',
        'timestamp': time.time()
    }), status_code

# API Health check for orchestration service
@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check endpoint for service integration"""
    return health_check()

# Device and hardware info
@app.route('/api/hardware', methods=['GET'])
def get_hardware_info():
    """Get hardware acceleration status"""
    if not model_manager:
        return jsonify({'error': 'Model manager not initialized'}), 503
    
    return jsonify({
        'device': model_manager.device,
        'available_devices': model_manager.available_devices,
        'npu_available': 'NPU' in model_manager.available_devices,
        'gpu_available': 'GPU' in model_manager.available_devices,
        'model_loaded': model_manager.current_model,
        'device_info': model_manager.get_device_info()
    })

# Create continuous streaming session
@app.route('/api/sessions/create', methods=['POST'])
def create_streaming_session():
    """Create a new continuous streaming session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', f"session_{int(time.time())}")
        
        with session_lock:
            if session_id in active_sessions:
                return jsonify({'error': 'Session already exists'}), 400
            
            # Create continuous stream processor
            processor = ContinuousStreamProcessor(session_id=session_id)
            
            # Set up integration components
            processor.transcription_callback = transcription_callback
            processor.translation_client = translation_client
            processor.transcript_manager = transcript_manager
            
            active_sessions[session_id] = processor
            
        logger.info(f"Created streaming session: {session_id}")
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'created_at': time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return jsonify({'error': str(e)}), 500

# Process audio chunk for continuous streaming
@app.route('/api/sessions/<session_id>/process_chunk', methods=['POST'])
def process_audio_chunk(session_id: str):
    """Process audio chunk for continuous streaming"""
    try:
        with session_lock:
            if session_id not in active_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            processor = active_sessions[session_id]
        
        # Get audio data from request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Process audio file
        audio_data = audio_file.read()
        audio_array = audio_processor.process_audio_data(audio_data)
        
        # Add chunk to processor
        result = processor.add_chunk(audio_array, {'timestamp': time.time()})
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to process audio chunk: {e}")
        return jsonify({'error': str(e)}), 500

# Get session transcript
@app.route('/api/sessions/<session_id>/transcript', methods=['GET'])
def get_session_transcript(session_id: str):
    """Get complete transcript for a session"""
    try:
        if not transcript_manager:
            return jsonify({'error': 'Transcript manager not initialized'}), 503
        
        format_type = request.args.get('format', 'json')
        transcript = transcript_manager.get_session_transcript(session_id, format_type)
        
        if transcript is None:
            return jsonify({'error': 'Session not found'}), 404
        
        if format_type == 'json':
            return jsonify({'transcript': transcript})
        else:
            return Response(transcript, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Failed to get transcript: {e}")
        return jsonify({'error': str(e)}), 500

# Get session statistics
@app.route('/api/sessions/<session_id>/stats', methods=['GET'])
def get_session_stats(session_id: str):
    """Get session processing statistics"""
    try:
        stats = {}
        
        # Get processor stats
        with session_lock:
            if session_id in active_sessions:
                processor = active_sessions[session_id]
                stats['processor'] = processor.get_stats()
        
        # Get transcript stats
        if transcript_manager:
            transcript_stats = transcript_manager.get_session_stats(session_id)
            if transcript_stats:
                stats['transcript'] = transcript_stats.__dict__
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Failed to get session stats: {e}")
        return jsonify({'error': str(e)}), 500

# Close streaming session
@app.route('/api/sessions/<session_id>/close', methods=['POST'])
def close_streaming_session(session_id: str):
    """Close a continuous streaming session"""
    try:
        with session_lock:
            if session_id not in active_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            processor = active_sessions[session_id]
            final_stats = processor.get_stats()
            
            # Clean up session
            processor.clear_session()
            del active_sessions[session_id]
        
        logger.info(f"Closed streaming session: {session_id}")
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'final_stats': final_stats,
            'closed_at': time.time()
        })
        
    except Exception as e:
        logger.error(f"Failed to close session: {e}")
        return jsonify({'error': str(e)}), 500

# List active sessions
@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active streaming sessions"""
    try:
        sessions = []
        
        with session_lock:
            for session_id, processor in active_sessions.items():
                stats = processor.get_stats()
                sessions.append({
                    'session_id': session_id,
                    'chunks_received': stats['chunks_received'],
                    'inferences_run': stats['inferences_run'],
                    'text_segments_sent': stats['text_segments_sent'],
                    'buffer_length': stats['buffer_length']
                })
        
        return jsonify({
            'active_sessions': len(sessions),
            'sessions': sessions
        })
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events for real-time streaming
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'sid': request.sid})

@socketio.on('join_streaming_session')
def handle_join_session(data):
    """Join a streaming session"""
    session_id = data.get('session_id')
    if session_id and session_id in active_sessions:
        logger.info(f"Client {request.sid} joined streaming session {session_id}")
        emit('joined_session', {'session_id': session_id, 'status': 'joined'})
    else:
        emit('error', {'message': 'Session not found'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle real-time audio chunk via WebSocket"""
    try:
        session_id = data.get('session_id')
        audio_data = data.get('audio_data')  # Base64 encoded
        
        if not session_id or session_id not in active_sessions:
            emit('error', {'message': 'Invalid session'})
            return
        
        processor = active_sessions[session_id]
        
        # Decode and process audio
        import base64
        audio_bytes = base64.b64decode(audio_data)
        audio_array = audio_processor.process_audio_data(audio_bytes)
        
        # Process through continuous stream processor
        result = processor.add_chunk(audio_array, {'timestamp': time.time()})
        
        # Emit result back to client
        emit('processing_result', result)
        
    except Exception as e:
        logger.error(f"WebSocket audio processing failed: {e}")
        emit('error', {'message': str(e)})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def shutdown_service():
    """Graceful shutdown of all service components"""
    logger.info("Shutting down Enhanced Whisper Service...")
    
    try:
        # Close all active sessions
        with session_lock:
            for session_id, processor in active_sessions.items():
                processor.clear_session()
            active_sessions.clear()
        
        # Shutdown components
        if buffer_manager:
            buffer_manager.shutdown()
        
        if transcript_manager:
            transcript_manager.shutdown()
        
        logger.info("✓ Enhanced Whisper Service shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def create_app(config: Optional[Dict] = None):
    """Factory function to create Flask app with configuration"""
    if config:
        app.config.update(config)
    
    return app

if __name__ == '__main__':
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Enhanced Whisper Service API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        shutdown_service()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting Enhanced Whisper Service on {args.host}:{args.port}")
    
    # Initialize service
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initialize_service())
    
    # Run server
    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False,  # Disable reloader in production
            allow_unsafe_werkzeug=True  # Allow Werkzeug for development
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        shutdown_service()