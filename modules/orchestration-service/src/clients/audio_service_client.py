"""
Audio Service Client

Handles communication with the Whisper-based audio processing service.
Provides methods for transcription, speaker diarization, and audio analysis.
"""

import logging
import asyncio
import json
import ssl
import mimetypes
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
from pathlib import Path
import aiohttp
import aiofiles
from pydantic import BaseModel
from utils.audio_errors import (
    AudioProcessingBaseError, AudioFormatError, AudioCorruptionError,
    AudioProcessingError, ServiceUnavailableError, ValidationError,
    NetworkError, TimeoutError, CircuitBreaker, RetryManager, RetryConfig,
    ErrorLogger, error_boundary
)

logger = logging.getLogger(__name__)


# Audio format detection and content-type mapping
AUDIO_MIME_TYPES = {
    b'RIFF': ('audio/wav', '.wav'),
    b'ID3': ('audio/mpeg', '.mp3'),
    b'\xff\xfb': ('audio/mpeg', '.mp3'),
    b'\xff\xfa': ('audio/mpeg', '.mp3'),
    b'\xff\xf3': ('audio/mpeg', '.mp3'),
    b'\xff\xf2': ('audio/mpeg', '.mp3'),
    b'OggS': ('audio/ogg', '.ogg'),
    b'fLaC': ('audio/flac', '.flac'),
    b'\x00\x00\x00 ftypM4A': ('audio/mp4', '.m4a'),
    b'\x00\x00\x00 ftypisom': ('audio/mp4', '.mp4'),
    b'\x00\x00\x00 ftypmp41': ('audio/mp4', '.mp4'),
    b'\x00\x00\x00 ftypmp42': ('audio/mp4', '.mp4'),
}

# WebM signature can appear at different offsets
WEBM_SIGNATURES = [
    b'\x1a\x45\xdf\xa3',  # EBML header
    b'webm',  # WebM identifier
]


def detect_audio_format(audio_data: bytes) -> Tuple[str, str]:
    """
    Detect audio format from binary data and return MIME type and file extension.
    
    Args:
        audio_data: Binary audio data
        
    Returns:
        Tuple of (mime_type, file_extension)
    """
    if not audio_data:
        return 'audio/wav', '.wav'  # Default fallback
    
    # Check for WebM format (can have variable header positions)
    audio_start = audio_data[:64]  # Check first 64 bytes for WebM signatures
    for webm_sig in WEBM_SIGNATURES:
        if webm_sig in audio_start:
            return 'audio/webm', '.webm'
    
    # Check standard audio format signatures
    for signature, (mime_type, extension) in AUDIO_MIME_TYPES.items():
        if audio_data.startswith(signature):
            return mime_type, extension
        # Also check at offset 8 for some MP4 variants
        if len(audio_data) > 8 + len(signature) and audio_data[8:8+len(signature)] == signature:
            return mime_type, extension
    
    # Fallback: try to detect MP3 by looking for frame headers anywhere in first 1KB
    mp3_headers = [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3', b'\xff\xf2']
    for i in range(min(1024, len(audio_data) - 1)):
        for header in mp3_headers:
            if audio_data[i:i+len(header)] == header:
                return 'audio/mpeg', '.mp3'
    
    # Final fallback
    logger.warning(f"Unknown audio format, using WAV fallback. First 32 bytes: {audio_data[:32].hex()}")
    return 'audio/wav', '.wav'


def generate_filename(base_name: str, audio_data: bytes = None, original_filename: str = None) -> str:
    """
    Generate appropriate filename based on audio format detection.
    
    Args:
        base_name: Base name for the file (e.g., 'stream', 'chunk', 'analyze')
        audio_data: Binary audio data for format detection
        original_filename: Original filename if available
        
    Returns:
        Generated filename with appropriate extension
    """
    if original_filename:
        # Use original filename if provided
        return original_filename
    
    if audio_data:
        # Detect format from audio data
        _, extension = detect_audio_format(audio_data)
        return f"{base_name}{extension}"
    
    # Fallback to WAV
    return f"{base_name}.wav"


class TranscriptionRequest(BaseModel):
    """Request model for transcription"""

    language: Optional[str] = None
    task: str = "transcribe"  # transcribe or translate
    enable_diarization: bool = True
    enable_vad: bool = True
    model: str = "whisper-base"


class TranscriptionResponse(BaseModel):
    """Response model for transcription"""

    text: str
    language: str
    segments: List[Dict[str, Any]]
    speakers: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    confidence: float


class AudioServiceClient:
    """Client for the Audio/Whisper service with comprehensive error handling"""

    def __init__(self, config_manager=None, settings=None):
        self.config_manager = config_manager
        self.settings = settings
        self.base_url = self._get_base_url()
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        # Create SSL context that doesn't verify certificates for localhost
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize error handling components
        self.circuit_breaker = CircuitBreaker(
            name="audio_service_client",
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        )
        
        self.retry_manager = RetryManager(RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        ))
        
        self.error_logger = ErrorLogger("audio_service_client")

    def _get_base_url(self) -> str:
        """Get the audio service base URL from configuration"""
        # Try settings first, then config_manager, then fallback
        if self.settings:
            return self.settings.services.audio_service_url
        elif self.config_manager:
            return self.config_manager.get_service_url("audio", "http://localhost:5001")
        return "http://localhost:5001"  # Use localhost as default

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            logger.debug(f"Creating new session for base_url: {self.base_url}")
            
            # For HTTP URLs, explicitly disable SSL
            if self.base_url.startswith("http://"):
                logger.debug(f"Using HTTP connection without SSL for audio service at {self.base_url}")
                # Explicitly disable SSL for HTTP connections
                connector = aiohttp.TCPConnector(ssl=False)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=self.timeout
                )
            else:
                logger.debug(f"Using HTTPS connection with SSL context for audio service at {self.base_url}")
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=self.timeout
                )
        return self.session

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self, correlation_id: str = None) -> Dict[str, Any]:
        """Check audio service health with comprehensive error handling"""
        correlation_id = correlation_id or f"health_{asyncio.get_event_loop().time()}"
        
        async with error_boundary(
            correlation_id=correlation_id,
            context={"operation": "health_check", "service": "audio", "url": self.base_url}
        ):
            try:
                return await self.circuit_breaker.call(
                    self._health_check_internal,
                    correlation_id
                )
            except Exception as e:
                self.error_logger.log_error(
                    ServiceUnavailableError(
                        f"Health check failed: {str(e)}",
                        correlation_id=correlation_id,
                        service_name="audio_service"
                    )
                )
                return {
                    "status": "unhealthy",
                    "service": "audio",
                    "url": self.base_url,
                    "error": str(e),
                    "correlation_id": correlation_id
                }

    async def _health_check_internal(self, correlation_id: str) -> Dict[str, Any]:
        """Internal health check implementation"""
        try:
            url = f"{self.base_url}/health"
            logger.info(f"[{correlation_id}] Checking audio service health at: {url}")
            
            session = await self._get_session()
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"[{correlation_id}] Audio service health check successful")
                        return {
                            "status": "healthy",
                            "service": "audio",
                            "url": self.base_url,
                            "details": data,
                            "correlation_id": correlation_id
                        }
                    else:
                        error_text = await response.text()
                        raise ServiceUnavailableError(
                            f"Health check failed with HTTP {response.status}: {error_text}",
                            correlation_id=correlation_id,
                            service_name="audio_service"
                        )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Health check timed out after {self.timeout.total}s",
                    correlation_id=correlation_id,
                    timeout_details={"timeout_seconds": self.timeout.total}
                )
        except aiohttp.ClientConnectorError as e:
            error_msg = str(e)
            if "ssl" in error_msg.lower() and self.base_url.startswith("http://"):
                raise NetworkError(
                    "SSL error on HTTP endpoint - check proxy/redirect configuration",
                    correlation_id=correlation_id,
                    network_details={
                        "error_type": "ssl_on_http",
                        "url": url,
                        "original_error": error_msg
                    }
                )
            else:
                raise NetworkError(
                    f"Connection error to audio service: {error_msg}",
                    correlation_id=correlation_id,
                    network_details={
                        "error_type": "connection_error",
                        "url": url,
                        "original_error": error_msg
                    }
                )
        except Exception as e:
            logger.error(f"Audio service health check failed: {type(e).__name__}: {e}")
            logger.error(f"URL attempted: {url}")
            return {
                "status": "unhealthy",
                "service": "audio",
                "url": self.base_url,
                "error": f"{type(e).__name__}: {str(e)}",
            }

    async def get_models(self) -> List[str]:
        """Get available Whisper models"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", ["whisper-base"])
                else:
                    logger.error(f"Failed to get models: HTTP {response.status}")
                    return ["whisper-base"]  # fallback
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return ["whisper-base"]  # fallback

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
        self, audio_file: Path, request_params: TranscriptionRequest, correlation_id: str = None
    ) -> TranscriptionResponse:
        """Transcribe an audio file with comprehensive error handling"""
        correlation_id = correlation_id or f"transcribe_file_{asyncio.get_event_loop().time()}"
        
        async with error_boundary(
            correlation_id=correlation_id,
            context={
                "operation": "transcribe_file",
                "file_path": str(audio_file),
                "model": request_params.model,
                "language": request_params.language
            }
        ):
            # Validate file exists and is readable
            if not audio_file.exists():
                raise AudioFormatError(
                    f"Audio file not found: {audio_file}",
                    correlation_id=correlation_id,
                    format_details={"file_path": str(audio_file)}
                )
            
            if not audio_file.is_file():
                raise AudioFormatError(
                    f"Path is not a file: {audio_file}",
                    correlation_id=correlation_id,
                    format_details={"file_path": str(audio_file)}
                )
            
            return await self.circuit_breaker.call(
                self.retry_manager.execute_with_retry,
                self._transcribe_file_internal,
                audio_file, request_params, correlation_id,
                correlation_id=correlation_id,
                retryable_exceptions=(NetworkError, TimeoutError, ServiceUnavailableError)
            )

    async def _transcribe_file_internal(
        self, audio_file: Path, request_params: TranscriptionRequest, correlation_id: str
    ) -> TranscriptionResponse:
        """Internal file transcription implementation"""
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

            # Add file with proper content-type detection
            async with aiofiles.open(audio_file, "rb") as f:
                file_content = await f.read()
                # Use original filename but also detect MIME type for content-type header
                mime_type, _ = detect_audio_format(file_content)
                # Use mimetypes as fallback if our detection fails
                if mime_type == 'audio/wav' and not file_content.startswith(b'RIFF'):
                    detected_mime, _ = mimetypes.guess_type(str(audio_file))
                    if detected_mime and detected_mime.startswith('audio/'):
                        mime_type = detected_mime
                data.add_field("audio", file_content, filename=audio_file.name, content_type=mime_type)

            async with session.post(
                f"{self.base_url}/transcribe", data=data
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
            # Detect audio format and set appropriate content-type and filename
            mime_type, _ = detect_audio_format(audio_data)
            filename = generate_filename("stream", audio_data)
            data.add_field("audio", audio_data, filename=filename, content_type=mime_type)

            async with session.post(
                f"{self.base_url}/transcribe", data=data
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
            # Detect audio format and set appropriate content-type and filename
            mime_type, _ = detect_audio_format(audio_chunk)
            filename = generate_filename("chunk", audio_chunk)
            data.add_field("audio_chunk", audio_chunk, filename=filename, content_type=mime_type)

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
            # Detect audio format and set appropriate content-type and filename
            mime_type, _ = detect_audio_format(audio_data)
            filename = generate_filename("analyze", audio_data)
            data.add_field("audio", audio_data, filename=filename, content_type=mime_type)

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

            async with session.get(f"{self.base_url}/performance") as response:
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
                audio_file_data = request_data['audio_file']
                # Detect audio format and set appropriate content-type and filename
                mime_type, _ = detect_audio_format(audio_file_data)
                filename = generate_filename("audio", audio_file_data)
                data.add_field("audio", audio_file_data, filename=filename, content_type=mime_type)
            
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
            
            # Prepare form data for whisper service with proper content-type detection
            data = aiohttp.FormData()
            original_filename = Path(file_path).name
            # Detect MIME type from file content
            mime_type, _ = detect_audio_format(file_content)
            # Use mimetypes as fallback if our detection fails
            if mime_type == 'audio/wav' and not file_content.startswith(b'RIFF'):
                detected_mime, _ = mimetypes.guess_type(file_path)
                if detected_mime and detected_mime.startswith('audio/'):
                    mime_type = detected_mime
            data.add_field("audio", file_content, filename=original_filename, content_type=mime_type)
            
            # Add whisper-specific parameters
            data.add_field("language", "auto")
            data.add_field("task", "transcribe")
            data.add_field("enable_diarization", str(request_data.get('speaker_diarization', True)).lower())
            data.add_field("enable_vad", str(request_data.get('enable_vad', True)).lower())
            data.add_field("model", request_data.get('whisper_model', 'whisper-base'))
                
            # Use the standard whisper service transcribe endpoint
            async with session.post(f"{self.base_url}/transcribe", data=data) as response:
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

    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get audio processing statistics for analytics API"""
        try:
            session = await self._get_session()
            
            # Try the performance endpoint (this is the main statistics endpoint)
            async with session.get(f"{self.base_url}/performance") as response:
                if response.status == 200:
                    result = await response.json()
                    # Normalize the response to match expected format
                    return {
                        "total_requests": result.get("total_requests", 0),
                        "successful_requests": result.get("successful_requests", 0),
                        "failed_requests": result.get("failed_requests", 0),
                        "average_processing_time_ms": result.get("average_processing_time", 0) * 1000,  # Convert to ms
                        "active_sessions": result.get("active_sessions", 0),
                        "transcription_accuracy": result.get("transcription_accuracy", 0.0),
                        "device_utilization": result.get("device_utilization", 0.0),
                        "model_performance": result.get("model_performance", {}),
                        "error_rate": result.get("failed_requests", 0) / max(1, result.get("total_requests", 1)),
                        "throughput_per_minute": result.get("throughput_per_minute", 0),
                        "queue_length": result.get("queue_length", 0),
                        "timestamp": result.get("timestamp", 0)
                    }
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to get processing statistics: HTTP {response.status} - {error_text}")
                    # Return default statistics as fallback
                    return {
                        "total_requests": 0,
                        "successful_requests": 0,
                        "failed_requests": 0,
                        "average_processing_time_ms": 0,
                        "active_sessions": 0,
                        "transcription_accuracy": 0.0,
                        "device_utilization": 0.0,
                        "model_performance": {},
                        "error_rate": 0.0,
                        "throughput_per_minute": 0,
                        "queue_length": 0,
                        "timestamp": 0,
                        "error": f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_processing_time_ms": 0,
                "active_sessions": 0,
                "transcription_accuracy": 0.0,
                "device_utilization": 0.0,
                "model_performance": {},
                "error_rate": 0.0,
                "throughput_per_minute": 0,
                "queue_length": 0,
                "timestamp": 0,
                "error": str(e)
            }
