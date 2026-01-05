#!/usr/bin/env python3
"""
Pipeline Integration for Whisper Service

Adds real-time pipeline integration endpoints to the Whisper service.
"""

import json
import logging
from datetime import datetime
from flask import request, jsonify
import tempfile
import os

logger = logging.getLogger(__name__)

def add_pipeline_routes(app, whisper_service):
    """
    Add pipeline integration routes to the Whisper service
    
    Args:
        app: Flask application
        whisper_service: WhisperService instance
    """
    
    @app.route('/api/pipeline/transcribe', methods=['POST'])
    async def pipeline_transcribe():
        """Transcribe audio for pipeline processing"""
        try:
            # Get session ID
            session_id = request.form.get('session_id')
            if not session_id:
                return jsonify({"error": "Missing session_id"}), 400
            
            # Get audio file
            if 'file' not in request.files:
                return jsonify({"error": "Missing audio file"}), 400
            
            audio_file = request.files['file']
            
            # Get metadata
            metadata = {}
            if 'metadata' in request.form:
                try:
                    metadata = json.loads(request.form['metadata'])
                except json.JSONDecodeError:
                    metadata = {}
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_file.save(tmp_file.name)
                
                try:
                    # Load audio data
                    import librosa
                    audio_data, sr = librosa.load(tmp_file.name, sr=16000)
                    
                    # Create transcription request
                    from whisper_service import TranscriptionRequest
                    transcription_request = TranscriptionRequest(
                        audio_data=audio_data,
                        model_name=metadata.get('model', 'whisper-base'),
                        language=metadata.get('language'),
                        session_id=session_id,
                        sample_rate=sr,
                        enable_vad=metadata.get('enable_vad', True)
                    )
                    
                    # Perform transcription
                    result = await whisper_service.transcribe(transcription_request)
                    
                    # Return result in pipeline format
                    return jsonify({
                        "text": result.text,
                        "segments": result.segments,
                        "language": result.language,
                        "confidence": result.confidence_score,
                        "processing_time": result.processing_time,
                        "model_used": result.model_used,
                        "device_used": result.device_used,
                        "session_id": result.session_id,
                        "timestamp": result.timestamp
                    })
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            logger.error(f"Pipeline transcription failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/pipeline/stream/start', methods=['POST'])
    async def start_pipeline_stream():
        """Start streaming transcription for pipeline"""
        try:
            data = request.get_json() or {}
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({"error": "Missing session_id"}), 400
            
            # Create session in whisper service
            session_config = whisper_service.create_session(session_id, data.get('config'))
            
            return jsonify({
                "status": "started",
                "session_id": session_id,
                "session_config": session_config
            })
            
        except Exception as e:
            logger.error(f"Failed to start pipeline stream: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/pipeline/stream/stop/<session_id>', methods=['POST'])
    async def stop_pipeline_stream(session_id):
        """Stop streaming transcription for pipeline"""
        try:
            # Close session in whisper service
            session_stats = whisper_service.close_session(session_id)
            
            return jsonify({
                "status": "stopped",
                "session_id": session_id,
                "final_stats": session_stats
            })
            
        except Exception as e:
            logger.error(f"Failed to stop pipeline stream: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/pipeline/status/<session_id>', methods=['GET'])
    def get_pipeline_session_status(session_id):
        """Get pipeline session status"""
        try:
            session = whisper_service.get_session(session_id)
            
            if not session:
                return jsonify({"error": "Session not found"}), 404
            
            # Add service-specific status
            service_status = whisper_service.get_service_status()
            
            return jsonify({
                "session": session,
                "service_status": service_status,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to get pipeline session status: {e}")
            return jsonify({"error": str(e)}), 500
    
    logger.info("Pipeline integration routes added to Whisper service")