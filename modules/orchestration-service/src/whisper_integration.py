#!/usr/bin/env python3
"""
Whisper Service Integration for Orchestration Service

Provides convenient endpoints for whisper transcription functionality
integrated through the API gateway.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio
import threading

from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
import requests

logger = logging.getLogger(__name__)

class WhisperIntegration:
    """Integration layer for Whisper service"""
    
    def __init__(self, api_gateway, websocket_manager):
        """Initialize whisper integration with gateway and websocket manager"""
        self.api_gateway = api_gateway
        self.websocket_manager = websocket_manager
        self.active_sessions = {}
        self.session_lock = threading.RLock()
        
        # Default whisper service configuration
        self.service_name = "whisper"
        self.default_model = "whisper-base"
        
        logger.info("Whisper integration initialized")
    
    def add_routes(self, app: Flask, socketio: SocketIO):
        """Add whisper-specific routes to the Flask app"""
        
        # Transcription endpoints
        @app.route('/api/whisper/transcribe', methods=['POST'])
        def transcribe_audio():
            """Transcribe audio using whisper service"""
            return self._handle_transcription_request()
        
        @app.route('/api/whisper/transcribe/<model_name>', methods=['POST'])
        def transcribe_audio_with_model(model_name: str):
            """Transcribe audio using specific model"""
            return self._handle_transcription_request(model_name)
        
        @app.route('/api/whisper/models', methods=['GET'])
        def list_whisper_models():
            """List available whisper models"""
            try:
                return self.api_gateway.route_request(self.service_name, "models", request)
            except Exception as e:
                logger.error(f"Failed to list whisper models: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/health', methods=['GET'])
        def whisper_health():
            """Check whisper service health"""
            try:
                return self.api_gateway.route_request(self.service_name, "health", request)
            except Exception as e:
                logger.error(f"Whisper health check failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/status', methods=['GET'])
        def whisper_status():
            """Get detailed whisper service status"""
            try:
                return self.api_gateway.route_request(self.service_name, "status", request)
            except Exception as e:
                logger.error(f"Whisper status check failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Session management endpoints
        @app.route('/api/whisper/sessions', methods=['POST'])
        def create_whisper_session():
            """Create new whisper transcription session"""
            try:
                data = request.get_json() or {}
                session_id = data.get('session_id', str(uuid.uuid4()))
                
                # Create session through whisper service
                response = self.api_gateway.route_request(
                    self.service_name, 
                    "sessions", 
                    request
                )
                
                # Track session locally
                with self.session_lock:
                    self.active_sessions[session_id] = {
                        "created_at": datetime.now().isoformat(),
                        "status": "active",
                        "transcriptions": []
                    }
                
                return response
                
            except Exception as e:
                logger.error(f"Failed to create whisper session: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/sessions/<session_id>', methods=['GET'])
        def get_whisper_session(session_id: str):
            """Get whisper session information"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    f"sessions/{session_id}", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to get whisper session: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/sessions/<session_id>', methods=['DELETE'])
        def close_whisper_session(session_id: str):
            """Close whisper session"""
            try:
                # Close session in whisper service
                response = self.api_gateway.route_request(
                    self.service_name, 
                    f"sessions/{session_id}", 
                    request
                )
                
                # Remove from local tracking
                with self.session_lock:
                    self.active_sessions.pop(session_id, None)
                
                return response
                
            except Exception as e:
                logger.error(f"Failed to close whisper session: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Real-time streaming endpoints
        @app.route('/api/whisper/stream/configure', methods=['POST'])
        def configure_whisper_streaming():
            """Configure whisper streaming parameters"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    "stream/configure", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to configure whisper streaming: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/stream/start', methods=['POST'])
        def start_whisper_streaming():
            """Start whisper streaming session"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    "stream/start", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to start whisper streaming: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/stream/stop', methods=['POST'])
        def stop_whisper_streaming():
            """Stop whisper streaming session"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    "stream/stop", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to stop whisper streaming: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/stream/audio', methods=['POST'])
        def stream_whisper_audio():
            """Stream audio chunk to whisper service"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    "stream/audio", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to stream audio to whisper: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/api/whisper/stream/transcriptions', methods=['GET'])
        def get_whisper_transcriptions():
            """Get rolling transcriptions from whisper service"""
            try:
                return self.api_gateway.route_request(
                    self.service_name, 
                    "stream/transcriptions", 
                    request
                )
            except Exception as e:
                logger.error(f"Failed to get whisper transcriptions: {e}")
                return jsonify({"error": str(e)}), 500
        
        # WebSocket events for real-time whisper integration
        @socketio.on('whisper_connect')
        def handle_whisper_connect(data):
            """Connect to whisper service via WebSocket"""
            try:
                session_id = data.get('session_id', str(uuid.uuid4()))
                
                # Track connection
                with self.session_lock:
                    if session_id not in self.active_sessions:
                        self.active_sessions[session_id] = {
                            "created_at": datetime.now().isoformat(),
                            "status": "active",
                            "transcriptions": [],
                            "websocket_sid": request.sid
                        }
                    else:
                        self.active_sessions[session_id]["websocket_sid"] = request.sid
                
                emit('whisper_connected', {
                    'session_id': session_id,
                    'status': 'connected',
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Whisper WebSocket connected for session {session_id}")
                
            except Exception as e:
                logger.error(f"Whisper WebSocket connection failed: {e}")
                emit('whisper_error', {'error': str(e)})
        
        @socketio.on('whisper_transcribe_stream')
        def handle_whisper_stream(data):
            """Handle real-time transcription streaming"""
            try:
                session_id = data.get('session_id')
                audio_data = data.get('audio_data')
                
                if not session_id or not audio_data:
                    emit('whisper_error', {'error': 'session_id and audio_data required'})
                    return
                
                # Forward to whisper service via API gateway
                # Note: We'll create a proxy request to the whisper service WebSocket
                # For now, emit a processing status
                emit('whisper_processing', {
                    'session_id': session_id,
                    'status': 'processing',
                    'timestamp': datetime.now().isoformat()
                })
                
                # In a production implementation, you would establish a WebSocket
                # connection to the whisper service and stream data
                # For now, we'll simulate a transcription result
                def simulate_transcription():
                    time.sleep(1)  # Simulate processing time
                    socketio.emit('whisper_result', {
                        'session_id': session_id,
                        'text': 'Simulated transcription result',
                        'confidence': 0.95,
                        'timestamp': datetime.now().isoformat()
                    }, room=request.sid)
                
                # Start simulation in background
                threading.Thread(target=simulate_transcription, daemon=True).start()
                
            except Exception as e:
                logger.error(f"Whisper streaming failed: {e}")
                emit('whisper_error', {'error': str(e)})
        
        @socketio.on('whisper_disconnect')
        def handle_whisper_disconnect(data):
            """Handle whisper WebSocket disconnection"""
            try:
                session_id = data.get('session_id')
                
                if session_id:
                    with self.session_lock:
                        if session_id in self.active_sessions:
                            self.active_sessions[session_id]["status"] = "disconnected"
                
                emit('whisper_disconnected', {
                    'session_id': session_id,
                    'status': 'disconnected',
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Whisper WebSocket disconnected for session {session_id}")
                
            except Exception as e:
                logger.error(f"Whisper WebSocket disconnection failed: {e}")
                emit('whisper_error', {'error': str(e)})
        
        logger.info("Whisper integration routes added to Flask app")
    
    def _handle_transcription_request(self, model_name: Optional[str] = None) -> Response:
        """Handle transcription request through API gateway"""
        try:
            # Validate request
            if not request.files and not request.data:
                return jsonify({"error": "No audio data provided"}), 400
            
            # Determine API path
            if model_name:
                api_path = f"transcribe/{model_name}"
            else:
                api_path = "transcribe"
            
            # Route through API gateway
            response = self.api_gateway.route_request(
                self.service_name, 
                api_path, 
                request
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Transcription request failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    def get_active_sessions(self) -> Dict[str, Any]:
        """Get currently active whisper sessions"""
        with self.session_lock:
            return dict(self.active_sessions)
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self.session_lock:
            return len(self.active_sessions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get whisper integration statistics"""
        with self.session_lock:
            active_count = len([s for s in self.active_sessions.values() if s.get("status") == "active"])
            websocket_count = len([s for s in self.active_sessions.values() if "websocket_sid" in s])
            
            return {
                "total_sessions": len(self.active_sessions),
                "active_sessions": active_count,
                "websocket_connections": websocket_count,
                "service_name": self.service_name,
                "default_model": self.default_model
            }

def add_whisper_routes(app: Flask, socketio: SocketIO, api_gateway, websocket_manager) -> WhisperIntegration:
    """Convenience function to add whisper routes to Flask app"""
    whisper_integration = WhisperIntegration(api_gateway, websocket_manager)
    whisper_integration.add_routes(app, socketio)
    return whisper_integration