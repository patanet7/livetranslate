"""
Audio Service Client

Handles communication with the Whisper-based audio processing service.
Provides methods for transcription, speaker diarization, and audio analysis.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
import aiohttp
import aiofiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""

    language: Optional[str] = None
    task: str = "transcribe"  # transcribe or translate
    enable_diarization: bool = True
    enable_vad: bool = True
    model: str = "base"


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""

    text: str
    language: str
    segments: List[Dict[str, Any]]
    speakers: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    confidence: float


class AudioServiceClient:
    """Client for the Audio/Whisper service"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.base_url = self._get_base_url()
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

    def _get_base_url(self) -> str:
        """Get the audio service base URL from configuration"""
        if self.config_manager:
            return self.config_manager.get_service_url("audio", "http://localhost:5001")
        return "http://localhost:5001"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check audio service health"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "service": "audio",
                        "url": self.base_url,
                        "details": data,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "service": "audio",
                        "url": self.base_url,
                        "error": f"HTTP {response.status}",
                    }
        except Exception as e:
            logger.error(f"Audio service health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "audio",
                "url": self.base_url,
                "error": str(e),
            }

    async def get_models(self) -> List[str]:
        """Get available Whisper models"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", ["base"])
                else:
                    logger.error(f"Failed to get models: HTTP {response.status}")
                    return ["base"]  # fallback
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return ["base"]  # fallback

    async def get_device_info(self) -> Dict[str, Any]:
        """Get current device information (CPU/GPU/NPU) from audio service"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/device-info") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to get device info: HTTP {response.status}")
                    return {"device": "unknown", "status": "error"}
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {"device": "unknown", "status": "error", "error": str(e)}

    async def transcribe_file(
        self, audio_file: Path, request_params: TranscriptionRequest
    ) -> TranscriptionResponse:
        """Transcribe an audio file"""
        try:
            session = await self._get_session()

            # Prepare form data
            data = aiohttp.FormData()
            data.add_field("language", request_params.language or "auto")
            data.add_field("task", request_params.task)
            data.add_field(
                "enable_diarization", str(request_params.enable_diarization).lower()
            )
            data.add_field("enable_vad", str(request_params.enable_vad).lower())
            data.add_field("model", request_params.model)

            # Add file
            async with aiofiles.open(audio_file, "rb") as f:
                file_content = await f.read()
                data.add_field("audio", file_content, filename=audio_file.name)

            async with session.post(
                f"{self.base_url}/api/transcribe", data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return TranscriptionResponse(**result)
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Transcription failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            raise

    async def transcribe_stream(
        self, audio_data: bytes, request_params: TranscriptionRequest
    ) -> TranscriptionResponse:
        """Transcribe audio data from memory"""
        try:
            session = await self._get_session()

            # Prepare form data
            data = aiohttp.FormData()
            data.add_field("language", request_params.language or "auto")
            data.add_field("task", request_params.task)
            data.add_field(
                "enable_diarization", str(request_params.enable_diarization).lower()
            )
            data.add_field("enable_vad", str(request_params.enable_vad).lower())
            data.add_field("model", request_params.model)
            data.add_field("audio", audio_data, filename="stream.wav")

            async with session.post(
                f"{self.base_url}/api/transcribe", data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return TranscriptionResponse(**result)
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Stream transcription failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            raise

    async def start_realtime_session(self, session_config: Dict[str, Any]) -> str:
        """Start a real-time transcription session"""
        try:
            session = await self._get_session()

            async with session.post(
                f"{self.base_url}/api/realtime/start", json=session_config
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["session_id"]
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to start realtime session: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Failed to start realtime session: {e}")
            raise

    async def send_realtime_audio(
        self, session_id: str, audio_chunk: bytes
    ) -> Optional[Dict[str, Any]]:
        """Send audio chunk to real-time session"""
        try:
            session = await self._get_session()

            data = aiohttp.FormData()
            data.add_field("session_id", session_id)
            data.add_field("audio_chunk", audio_chunk, filename="chunk.wav")

            async with session.post(
                f"{self.base_url}/api/realtime/audio", data=data
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 202:
                    # Processing, no result yet
                    return None
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Realtime audio failed: HTTP {response.status} - {error_text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Failed to send realtime audio: {e}")
            return None

    async def stop_realtime_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a real-time transcription session"""
        try:
            session = await self._get_session()

            async with session.post(
                f"{self.base_url}/api/realtime/stop", json={"session_id": session_id}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to stop realtime session: HTTP {response.status} - {error_text}"
                    )
                    return {"status": "error", "message": error_text}

        except Exception as e:
            logger.error(f"Failed to stop realtime session: {e}")
            return {"status": "error", "message": str(e)}

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a real-time session"""
        try:
            session = await self._get_session()

            async with session.get(
                f"{self.base_url}/api/realtime/status/{session_id}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "not_found"}

        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return {"status": "error", "message": str(e)}

    async def analyze_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyze audio for quality metrics"""
        try:
            session = await self._get_session()

            data = aiohttp.FormData()
            data.add_field("audio", audio_data, filename="analyze.wav")

            async with session.post(
                f"{self.base_url}/api/analyze", data=data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Audio analysis failed: HTTP {response.status} - {error_text}"
                    )

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/api/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    # Enhanced pipeline processing methods for orchestration integration
    
    async def process_audio_batch(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Process audio through the enhanced pipeline in batch mode"""
        try:
            session = await self._get_session()
            
            # Prepare the request data for the whisper service
            # The request_data should contain the audio file and configuration
            data = aiohttp.FormData()
            
            # Add audio file if present
            if 'audio_file' in request_data:
                data.add_field("audio", request_data['audio_file'], filename="audio.wav")
            
            # Add pipeline configuration
            if 'config' in request_data:
                data.add_field("config", json.dumps(request_data['config']))
            
            # Add processing options
            data.add_field("pipeline_enabled", "true")
            data.add_field("stage_by_stage", "true")
            data.add_field("request_id", request_id)
            
            # Enhanced pipeline processing endpoint
            async with session.post(f"{self.base_url}/api/process-pipeline", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Audio batch processing completed for request {request_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Audio batch processing failed for request {request_id}: HTTP {response.status} - {error_text}")
                    raise Exception(f"Pipeline processing failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Audio batch processing failed for request {request_id}: {e}")
            raise

    async def start_audio_streaming(self, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Start streaming audio processing"""
        try:
            session = await self._get_session()
            
            # Prepare streaming configuration
            streaming_config = {
                "request_id": request_id,
                "mode": "streaming",
                "pipeline_config": request_data.get('config', {}),
                "chunk_size": request_data.get('chunk_size', 1024),
                "sample_rate": request_data.get('sample_rate', 16000)
            }
            
            async with session.post(f"{self.base_url}/api/start-streaming", json=streaming_config) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Audio streaming started for request {request_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to start audio streaming for request {request_id}: HTTP {response.status} - {error_text}")
                    raise Exception(f"Streaming start failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to start audio streaming for request {request_id}: {e}")
            raise

    async def stream_processing_results(self, session_id: str) -> AsyncGenerator[str, None]:
        """Stream real-time processing results"""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/api/stream-results/{session_id}") as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            yield line.decode('utf-8')
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to stream results for session {session_id}: HTTP {response.status} - {error_text}")
                    raise Exception(f"Results streaming failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to stream results for session {session_id}: {e}")
            raise

    async def process_uploaded_file(self, file_path: str, request_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Process an uploaded audio file"""
        try:
            session = await self._get_session()
            
            # Read the uploaded file
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field("audio", file_content, filename=Path(file_path).name)
            data.add_field("request_id", request_id)
            
            # Add configuration
            if 'config' in request_data:
                data.add_field("config", json.dumps(request_data['config']))
                
            # Add processing options from request
            for key in ['transcription', 'speaker_diarization', 'target_languages']:
                if key in request_data:
                    data.add_field(key, str(request_data[key]))
            
            # Process uploaded file with enhanced pipeline
            async with session.post(f"{self.base_url}/api/upload-process", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"File processing completed for request {request_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"File processing failed for request {request_id}: HTTP {response.status} - {error_text}")
                    raise Exception(f"File processing failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"File processing failed for request {request_id}: {e}")
            raise

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics"""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/api/processing-stats") as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to get processing stats: HTTP {response.status} - {error_text}")
                    # Return basic stats as fallback
                    return {
                        "total_requests": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "average_processing_time": 0,
                        "active_sessions": 0,
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_processing_time": 0,
                "active_sessions": 0,
                "error": str(e)
            }

    async def get_processed_file(self, request_id: str, format: str) -> Dict[str, Any]:
        """Get information about a processed file for download"""
        try:
            session = await self._get_session()
            
            params = {"format": format} if format else {}
            
            async with session.get(f"{self.base_url}/api/download/{request_id}", params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                elif response.status == 404:
                    logger.warning(f"Processed file not found for request {request_id}")
                    return {"error": "File not found", "status": 404}
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get processed file for request {request_id}: HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status} - {error_text}", "status": response.status}
                    
        except Exception as e:
            logger.error(f"Failed to get processed file for request {request_id}: {e}")
            return {"error": str(e), "status": 500}
