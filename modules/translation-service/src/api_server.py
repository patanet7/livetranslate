#!/usr/bin/env python3
"""
Translation Service API Server

Flask-based REST API with WebSocket support for real-time translation streaming.
Provides endpoints for translation, language detection, and service management.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid

from translation_service import TranslationService, TranslationRequest, create_translation_service

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

# Active sessions
active_sessions: Dict[str, Dict] = {}

async def initialize_service():
    """Initialize the translation service before handling requests"""
    global translation_service
    try:
        # Initialize translation service based on backend preference
        backend = os.getenv("INFERENCE_BACKEND", "auto").lower()
        
        if backend == "triton":
            # Use Triton backend
            config = {
                "backend": "triton",
                "triton_base_url": os.getenv("TRITON_BASE_URL", "http://localhost:8000"),
                "model_name": os.getenv("MODEL_NAME", "vllm_model")
            }
            translation_service = await create_translation_service(config)
            logger.info("Initialized with Triton backend")
            
        elif backend == "vllm":
            # Use vLLM backend
            config = {
                "backend": "vllm",
                "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000")
            }
            translation_service = await create_translation_service(config)
            logger.info("Initialized with vLLM backend")
            
        elif backend == "ollama":
            # Use Ollama backend  
            config = {
                "backend": "ollama",
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            }
            translation_service = await create_translation_service(config)
            logger.info("Initialized with Ollama backend")
            
        else:
            # Auto-detect backend (Triton -> vLLM -> Ollama)
            translation_service = await create_translation_service()
            logger.info("Initialized with auto-detected backend")
            
        logger.info("Translation service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize translation service: {e}")
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
        status = await translation_service.get_service_status()
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
        
        # Perform translation
        result = await translation_service.translate(translation_request)
        
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
        result = await translation_service.translate_with_continuity(
            text=clean_text,
            session_id=session_id,
            target_language=target_language,
            source_language=source_language,
            chunk_id=chunk_id
        )
        
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
        result = await translation_service.translate_with_continuity(
            text=clean_text,
            session_id=session_id,
            target_language=target_language,
            source_language=source_language,
            chunk_id=metadata.get('inference_number')
        )
        
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
        
        language, confidence = await translation_service.detect_language(data['text'])
        
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
        languages = await translation_service.get_supported_languages()
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
        
        session_config = await translation_service.create_session(
            session_id, 
            data.get('config')
        )
        
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
        session = await translation_service.get_session(session_id)
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
        session = await translation_service.close_session(session_id)
        
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