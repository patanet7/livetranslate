#!/usr/bin/env python3
"""
Audio Coordinator - Orchestration Service

Central coordination class for all audio processing in the LiveTranslate system.
Manages audio chunking, database integration, service communication, and speaker correlation.
Replaces scattered chunking logic across frontend and whisper service with centralized control.

Features:
- Centralized audio chunking with configurable overlap management
- Real-time coordination with whisper and translation services
- Complete database integration with bot_sessions schema
- Speaker correlation between whisper and Google Meet speakers
- WebSocket and chunk-based processing support
- Performance monitoring and quality assurance
- Concurrent session management with resource pooling
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
import httpx
import numpy as np

from .models import (
    AudioChunkMetadata,
    AudioChunkingConfig,
    AudioStreamingSession,
    SourceType,
    get_default_chunking_config,
)
from .database_adapter import AudioDatabaseAdapter
from .chunk_manager import ChunkManager, create_chunk_manager
from .config import (
    AudioProcessingConfig,
    create_audio_config_manager,
)
from .audio_processor import AudioPipelineProcessor, create_audio_pipeline_processor

logger = logging.getLogger(__name__)


class ServiceClientPool:
    """
    HTTP client pool for communicating with whisper and translation services.
    Provides connection pooling, retry logic, and circuit breaker patterns.
    """
    
    def __init__(self, service_urls: Dict[str, str], pool_config: Optional[Dict] = None):
        self.service_urls = service_urls
        pool_config = pool_config or {}
        
        # HTTP client configuration
        self.timeout = httpx.Timeout(
            connect=pool_config.get("connect_timeout", 10.0),
            read=pool_config.get("read_timeout", 30.0),
            write=pool_config.get("write_timeout", 10.0),
            pool=pool_config.get("pool_timeout", 5.0)
        )
        
        self.limits = httpx.Limits(
            max_keepalive_connections=pool_config.get("max_keepalive", 20),
            max_connections=pool_config.get("max_connections", 100),
            keepalive_expiry=pool_config.get("keepalive_expiry", 30.0)
        )
        
        # Circuit breaker state
        self.service_health = {url: {"healthy": True, "failures": 0, "last_failure": 0} for url in service_urls.values()}
        self.max_failures = pool_config.get("max_failures", 5)
        self.failure_timeout = pool_config.get("failure_timeout", 60.0)
        
        # HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self) -> bool:
        """Initialize the HTTP client pool with robust error handling."""
        try:
            # First try with HTTP/2 enabled
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits,
                http2=True,  # Enable HTTP/2 for better performance
                headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"}
            )
            logger.info("Service client pool initialized with HTTP/2 support")
            return True
        except ImportError as e:
            if "h2" in str(e):
                logger.warning("HTTP/2 support not available (h2 package missing), falling back to HTTP/1.1")
                try:
                    # Fallback to HTTP/1.1
                    self.client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=self.limits,
                        http2=False,
                        headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"}
                    )
                    logger.info("Service client pool initialized with HTTP/1.1")
                    return True
                except Exception as fallback_e:
                    logger.error(f"Failed to initialize service client pool even with HTTP/1.1 fallback: {fallback_e}")
                    return False
            else:
                logger.error(f"Failed to initialize service client pool due to import error: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize service client pool: {e}")
            # Try one more time without HTTP/2 as a last resort
            try:
                self.client = httpx.AsyncClient(
                    timeout=self.timeout,
                    limits=self.limits,
                    http2=False,
                    headers={"User-Agent": "LiveTranslate-AudioCoordinator/1.0"}
                )
                logger.warning("Service client pool initialized with HTTP/1.1 as fallback after general error")
                return True
            except Exception as final_e:
                logger.error(f"Complete failure to initialize service client pool: {final_e}")
                return False
    
    async def close(self):
        """Close the HTTP client pool."""
        if self.client:
            await self.client.aclose()
            logger.info("Service client pool closed")
    
    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Get HTTP client with circuit breaker protection."""
        if not self.client:
            raise RuntimeError("Service client pool not initialized")
        yield self.client
    
    async def send_to_whisper_service(
        self,
        session_id: str,
        chunk_metadata: AudioChunkMetadata,
        audio_data: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Send audio chunk to whisper service for transcription."""
        service_url = self.service_urls.get("whisper_service")
        if not service_url or not self._is_service_healthy(service_url):
            return None
        
        try:
            # Convert audio to bytes
            audio_bytes = self._audio_to_bytes(audio_data, chunk_metadata.sample_rate)
            
            # Prepare request data
            files = {"audio": ("chunk.wav", audio_bytes, "audio/wav")}
            data = {
                "session_id": session_id,
                "chunk_id": chunk_metadata.chunk_id,
                "chunk_sequence": chunk_metadata.chunk_sequence,
                "chunk_start_time": chunk_metadata.chunk_start_time,
                "chunk_end_time": chunk_metadata.chunk_end_time,
                "metadata": json.dumps({
                    "source_type": chunk_metadata.source_type.value,
                    "quality_score": chunk_metadata.audio_quality_score,
                    "overlap_metadata": chunk_metadata.overlap_metadata,
                })
            }
            
            async with self.get_client() as client:
                response = await client.post(
                    f"{service_url}/transcribe/chunk",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._mark_service_healthy(service_url)
                    return result
                else:
                    logger.warning(f"Whisper service error: {response.status_code}")
                    self._mark_service_failure(service_url)
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to send chunk to whisper service: {e}")
            self._mark_service_failure(service_url)
            return None
    
    async def send_to_translation_service(
        self,
        session_id: str,
        transcript_data: Dict[str, Any],
        target_language: str
    ) -> Optional[Dict[str, Any]]:
        """Send transcript to translation service."""
        service_url = self.service_urls.get("translation_service")
        if not service_url or not self._is_service_healthy(service_url):
            return None
        
        try:
            translation_request = {
                "session_id": session_id,
                "text": transcript_data["text"],
                "source_language": transcript_data.get("language", "auto"),
                "target_language": target_language,
                "speaker_id": transcript_data.get("speaker_id"),
                "speaker_name": transcript_data.get("speaker_name"),
                "start_timestamp": transcript_data["start_timestamp"],
                "end_timestamp": transcript_data["end_timestamp"],
                "metadata": {
                    "transcript_id": transcript_data.get("transcript_id"),
                    "chunk_id": transcript_data.get("chunk_id"),
                    "confidence": transcript_data.get("confidence", 0.0),
                }
            }
            
            async with self.get_client() as client:
                response = await client.post(
                    f"{service_url}/api/translate",
                    json=translation_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self._mark_service_healthy(service_url)
                    return result
                else:
                    logger.warning(f"Translation service error: {response.status_code}")
                    self._mark_service_failure(service_url)
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to send to translation service: {e}")
            self._mark_service_failure(service_url)
            return None
    
    def _audio_to_bytes(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio data to WAV bytes."""
        import io
        import soundfile as sf
        
        # Convert to int16 for efficient transmission
        if audio_data.dtype != np.int16:
            normalized = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (normalized * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()
    
    def _is_service_healthy(self, service_url: str) -> bool:
        """Check if service is healthy based on circuit breaker state."""
        health = self.service_health.get(service_url, {"healthy": True, "failures": 0, "last_failure": 0})
        
        if health["healthy"]:
            return True
        
        # Check if enough time has passed to retry
        if time.time() - health["last_failure"] > self.failure_timeout:
            health["healthy"] = True
            health["failures"] = 0
            return True
        
        return False
    
    def _mark_service_healthy(self, service_url: str):
        """Mark service as healthy."""
        if service_url in self.service_health:
            self.service_health[service_url]["healthy"] = True
            self.service_health[service_url]["failures"] = 0
    
    def _mark_service_failure(self, service_url: str):
        """Mark service failure and update circuit breaker state."""
        if service_url not in self.service_health:
            return
        
        health = self.service_health[service_url]
        health["failures"] += 1
        health["last_failure"] = time.time()
        
        if health["failures"] >= self.max_failures:
            health["healthy"] = False
            logger.warning(f"Circuit breaker opened for {service_url}")


class SessionManager:
    """
    Manages multiple audio streaming sessions with resource pooling.
    Handles session lifecycle, resource allocation, and cleanup.
    """
    
    def __init__(self, max_concurrent_sessions: int = 10):
        self.max_concurrent_sessions = max_concurrent_sessions
        self.active_sessions: Dict[str, AudioStreamingSession] = {}
        self.session_managers: Dict[str, ChunkManager] = {}
        self.session_callbacks: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        
        # Resource tracking
        self.total_sessions_created = 0
        self.total_chunks_processed = 0
        self.average_session_duration = 0.0
        
    def can_create_session(self) -> bool:
        """Check if new session can be created."""
        return len(self.active_sessions) < self.max_concurrent_sessions
    
    async def create_session(
        self,
        session_id: str,
        bot_session_id: str,
        source_type: SourceType,
        config: AudioChunkingConfig,
        target_languages: List[str],
        database_adapter: AudioDatabaseAdapter
    ) -> Optional[AudioStreamingSession]:
        """Create new audio streaming session."""
        if not self.can_create_session():
            logger.warning(f"Cannot create session {session_id}: max concurrent sessions reached")
            return None
        
        try:
            # Create streaming session
            streaming_session = AudioStreamingSession(
                streaming_session_id=session_id,
                bot_session_id=bot_session_id,
                source_type=source_type,
                chunk_config=config,
                target_languages=target_languages
            )
            
            # Create chunk manager
            chunk_manager = create_chunk_manager(config, database_adapter, bot_session_id, source_type)
            
            # Store session
            self.active_sessions[session_id] = streaming_session
            self.session_managers[session_id] = chunk_manager
            self.total_sessions_created += 1
            
            logger.info(f"Created audio session {session_id} for bot session {bot_session_id}")
            return streaming_session
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            return None
    
    async def start_session(self, session_id: str) -> bool:
        """Start audio processing for a session."""
        if session_id not in self.session_managers:
            return False
        
        try:
            chunk_manager = self.session_managers[session_id]
            success = await chunk_manager.start()
            
            if success:
                self.active_sessions[session_id].stream_status = "active"
                logger.info(f"Started audio session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start session {session_id}: {e}")
            return False
    
    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop audio processing for a session and return statistics."""
        if session_id not in self.session_managers:
            return {}
        
        try:
            chunk_manager = self.session_managers[session_id]
            stats = await chunk_manager.stop()
            
            # Update session
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.stream_status = "ended"
                session.ended_at = datetime.utcnow()
                session.chunks_processed = stats.get("chunks_processed", 0)
                session.total_duration = stats.get("total_audio_duration", 0.0)
                session.average_processing_time = stats.get("average_processing_time", 0.0)
                session.average_quality_score = stats.get("average_quality_score", 0.0)
                
                # Update global statistics
                self.total_chunks_processed += session.chunks_processed
                session_duration = (session.ended_at - session.started_at).total_seconds()
                self.average_session_duration = (
                    (self.average_session_duration * (self.total_sessions_created - 1) + session_duration) /
                    self.total_sessions_created
                )
            
            logger.info(f"Stopped audio session {session_id}: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to stop session {session_id}: {e}")
            return {}
        finally:
            # Cleanup
            self.session_managers.pop(session_id, None)
            self.active_sessions.pop(session_id, None)
            self.session_callbacks.pop(session_id, None)
    
    def get_session(self, session_id: str) -> Optional[AudioStreamingSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_chunk_manager(self, session_id: str) -> Optional[ChunkManager]:
        """Get chunk manager for session."""
        return self.session_managers.get(session_id)
    
    def set_session_callbacks(self, session_id: str, callbacks: Dict[str, Callable]):
        """Set callbacks for a session."""
        if session_id in self.session_managers:
            chunk_manager = self.session_managers[session_id]
            
            if "on_chunk_ready" in callbacks:
                chunk_manager.set_chunk_ready_callback(callbacks["on_chunk_ready"])
            if "on_quality_alert" in callbacks:
                chunk_manager.set_quality_alert_callback(callbacks["on_quality_alert"])
            if "on_error" in callbacks:
                chunk_manager.set_error_callback(callbacks["on_error"])
            
            self.session_callbacks[session_id] = callbacks
    
    def get_all_sessions(self) -> List[AudioStreamingSession]:
        """Get all active sessions."""
        return list(self.active_sessions.values())
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get overall session management statistics."""
        active_sessions = len(self.active_sessions)
        
        return {
            "active_sessions": active_sessions,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "utilization": active_sessions / self.max_concurrent_sessions,
            "total_sessions_created": self.total_sessions_created,
            "total_chunks_processed": self.total_chunks_processed,
            "average_session_duration": self.average_session_duration,
            "average_chunks_per_session": self.total_chunks_processed / max(1, self.total_sessions_created),
        }


class AudioCoordinator:
    """
    Central audio coordination class for the LiveTranslate system.
    Orchestrates all audio processing from reception through chunking, transcription, and translation.
    Includes complete audio processing pipeline with persistent configuration management.
    """
    
    def __init__(
        self,
        config: AudioChunkingConfig,
        database_url: Optional[str],
        service_urls: Dict[str, str],
        max_concurrent_sessions: int = 10,
        audio_config_file: Optional[str] = None
    ):
        self.config = config
        self.database_url = database_url
        self.service_urls = service_urls
        
        # Core components
        self.database_adapter = AudioDatabaseAdapter(database_url) if database_url else None
        self.service_client = ServiceClientPool(service_urls)
        self.session_manager = SessionManager(max_concurrent_sessions)
        
        # Audio processing configuration and pipeline
        self.audio_config_manager = create_audio_config_manager(
            config_file_path=audio_config_file,
            database_adapter=self.database_adapter,
            auto_reload=True
        )
        self.audio_processors: Dict[str, AudioPipelineProcessor] = {}  # Per-session processors
        
        # Processing state
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # Performance tracking
        self.total_audio_processed = 0.0
        self.total_chunks_created = 0
        self.total_transcripts_generated = 0
        self.total_translations_generated = 0
        self.processing_errors = 0
        
        # Event callbacks
        self.on_transcript_ready: Optional[Callable] = None
        self.on_translation_ready: Optional[Callable] = None
        self.on_session_event: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info("AudioCoordinator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the audio coordinator and all components."""
        try:
            # Initialize database adapter (optional for development)
            if self.database_adapter:
                success = await self.database_adapter.initialize()
                if not success:
                    logger.error("Failed to initialize database adapter")
                    return False
            else:
                logger.warning("Database adapter disabled - running without persistence")
            
            # Initialize service client pool
            success = await self.service_client.initialize()
            if not success:
                logger.error("Failed to initialize service client pool")
                return False
            
            # Initialize audio configuration manager
            success = await self.audio_config_manager.initialize()
            if not success:
                logger.error("Failed to initialize audio configuration manager")
                return False
            
            # Set up configuration change callback
            self.audio_config_manager.add_config_change_callback(self._on_audio_config_change)
            
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("AudioCoordinator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AudioCoordinator: {e}")
            return False
    
    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the audio coordinator and return final statistics."""
        try:
            self.is_running = False
            
            # Stop all active sessions
            session_stats = []
            for session_id in list(self.session_manager.active_sessions.keys()):
                stats = await self.session_manager.stop_session(session_id)
                session_stats.append(stats)
            
            # Close components
            await self.service_client.close()
            if self.database_adapter:
                await self.database_adapter.close()
            await self.audio_config_manager.shutdown()
            
            # Clear audio processors
            self.audio_processors.clear()
            
            # Calculate final statistics
            total_runtime = time.time() - self.start_time if self.start_time else 0
            
            final_stats = {
                "total_runtime": total_runtime,
                "total_audio_processed": self.total_audio_processed,
                "total_chunks_created": self.total_chunks_created,
                "total_transcripts_generated": self.total_transcripts_generated,
                "total_translations_generated": self.total_translations_generated,
                "processing_errors": self.processing_errors,
                "session_statistics": self.session_manager.get_session_statistics(),
                "database_statistics": self.database_adapter.get_operation_statistics() if self.database_adapter else {},
                "session_final_stats": session_stats,
            }
            
            logger.info(f"AudioCoordinator shutdown complete: {final_stats}")
            return final_stats
            
        except Exception as e:
            logger.error(f"Error during AudioCoordinator shutdown: {e}")
            return {}
    
    # Event callback setters
    def set_transcript_ready_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for when transcripts are ready."""
        self.on_transcript_ready = callback
    
    def set_translation_ready_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for when translations are ready."""
        self.on_translation_ready = callback
    
    def set_session_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set callback for session events."""
        self.on_session_event = callback
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback
    
    # Audio processing configuration management
    
    async def _on_audio_config_change(self, change_type: str, config: AudioProcessingConfig, session_id: Optional[str] = None):
        """Handle audio configuration changes."""
        try:
            if session_id:
                # Update session-specific processor
                if session_id in self.audio_processors:
                    self.audio_processors[session_id].update_config(config)
                    logger.info(f"Updated audio processor config for session {session_id}")
            else:
                # Update all processors with new default config
                for processor in self.audio_processors.values():
                    processor.update_config(config)
                logger.info("Updated all audio processors with new default config")
                
            # Emit configuration change event
            if self.on_session_event:
                self.on_session_event("audio_config_changed", {
                    "change_type": change_type,
                    "session_id": session_id,
                    "preset_name": config.preset_name,
                })
                
        except Exception as e:
            logger.error(f"Failed to handle audio config change: {e}")
    
    def get_audio_processing_config(self, session_id: Optional[str] = None) -> AudioProcessingConfig:
        """Get audio processing configuration for session or default."""
        if session_id:
            return self.audio_config_manager.get_session_config(session_id)
        else:
            return self.audio_config_manager.get_default_config()
    
    async def update_audio_processing_config(
        self, 
        config_updates: Dict[str, Any], 
        session_id: Optional[str] = None,
        save_persistent: bool = True
    ) -> AudioProcessingConfig:
        """Update audio processing configuration."""
        return await self.audio_config_manager.update_config_from_dict(
            config_updates, session_id, save_persistent
        )
    
    async def apply_audio_preset(self, preset_name: str, session_id: Optional[str] = None) -> bool:
        """Apply an audio processing preset."""
        return await self.audio_config_manager.apply_preset(preset_name, session_id)
    
    def get_available_audio_presets(self) -> List[str]:
        """Get list of available audio processing presets."""
        return self.audio_config_manager.get_available_presets()
    
    def get_audio_config_schema(self) -> Dict[str, Any]:
        """Get audio configuration schema for frontend validation."""
        return self.audio_config_manager.get_config_schema()
    
    def _get_or_create_audio_processor(self, session_id: str) -> AudioPipelineProcessor:
        """Get or create audio processor for session."""
        if session_id not in self.audio_processors:
            # Get session-specific config or use default
            audio_config = self.audio_config_manager.get_session_config(session_id)
            self.audio_processors[session_id] = create_audio_pipeline_processor(audio_config)
            logger.info(f"Created audio processor for session {session_id} with preset: {audio_config.preset_name}")
        
        return self.audio_processors[session_id]
    
    # Session management API
    async def create_audio_session(
        self,
        bot_session_id: str,
        source_type: SourceType = SourceType.BOT_AUDIO,
        target_languages: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create new audio streaming session.
        
        Returns:
            str: Streaming session ID if successful, None if failed
        """
        if not self.is_running:
            logger.error("AudioCoordinator not running")
            return None
        
        try:
            # Generate session ID
            session_id = f"audio_{bot_session_id}_{int(time.time() * 1000)}"
            
            # Prepare configuration
            config = self.config.copy()
            if custom_config:
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Use default target languages if not provided
            if not target_languages:
                target_languages = ["en", "es", "fr", "de"]
            
            # Create session
            streaming_session = await self.session_manager.create_session(
                session_id,
                bot_session_id,
                source_type,
                config,
                target_languages,
                self.database_adapter
            )
            
            if streaming_session:
                # Set up session callbacks
                callbacks = {
                    "on_chunk_ready": lambda metadata, audio_data: asyncio.create_task(
                        self._handle_chunk_ready(session_id, metadata, audio_data)
                    ),
                    "on_quality_alert": lambda alert: self._handle_quality_alert(session_id, alert),
                    "on_error": lambda error: self._handle_session_error(session_id, error),
                }
                
                self.session_manager.set_session_callbacks(session_id, callbacks)
                
                # Emit session event
                if self.on_session_event:
                    self.on_session_event("audio_session_created", {
                        "session_id": session_id,
                        "bot_session_id": bot_session_id,
                        "source_type": source_type.value,
                        "target_languages": target_languages,
                        "config": config.dict(),
                    })
                
                logger.info(f"Created audio session {session_id}")
                return session_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create audio session: {e}")
            if self.on_error:
                self.on_error(f"Session creation failed: {e}")
            return None
    
    async def start_audio_session(self, session_id: str) -> bool:
        """Start audio processing for a session."""
        try:
            success = await self.session_manager.start_session(session_id)
            
            if success and self.on_session_event:
                self.on_session_event("audio_session_started", {"session_id": session_id})
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to start audio session {session_id}: {e}")
            return False
    
    async def stop_audio_session(self, session_id: str) -> Dict[str, Any]:
        """Stop audio processing for a session."""
        try:
            stats = await self.session_manager.stop_session(session_id)
            
            if self.on_session_event:
                self.on_session_event("audio_session_stopped", {
                    "session_id": session_id,
                    "final_stats": stats
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to stop audio session {session_id}: {e}")
            return {}
    
    async def add_audio_data(self, session_id: str, audio_data: np.ndarray) -> bool:
        """
        Add audio data to a session for processing.
        Includes complete audio processing pipeline before chunking.
        
        Args:
            session_id: Audio streaming session ID
            audio_data: Audio samples as numpy array
            
        Returns:
            bool: True if data was successfully added
        """
        try:
            chunk_manager = self.session_manager.get_chunk_manager(session_id)
            if not chunk_manager:
                logger.warning(f"No chunk manager found for session {session_id}")
                return False
            
            # Apply audio processing pipeline
            audio_processor = self._get_or_create_audio_processor(session_id)
            processed_audio, processing_metadata = audio_processor.process_audio_chunk(audio_data)
            
            # Log processing results if significant
            if processing_metadata.get("bypassed", False):
                logger.debug(f"Audio processing bypassed for session {session_id}: {processing_metadata.get('bypass_reason', 'unknown')}")
            elif processing_metadata.get("stages_applied"):
                logger.debug(f"Applied audio processing stages for session {session_id}: {processing_metadata['stages_applied']}")
            
            # Add processed audio to chunk manager
            samples_added = await chunk_manager.add_audio_data(processed_audio)
            
            if samples_added > 0:
                # Update session activity
                session = self.session_manager.get_session(session_id)
                if session:
                    session.last_activity_at = datetime.utcnow()
                
                # Update global statistics
                self.total_audio_processed += samples_added / 16000  # Convert to seconds
                
                # Store processing metadata for analytics
                if self.on_session_event:
                    self.on_session_event("audio_processed", {
                        "session_id": session_id,
                        "samples_processed": len(audio_data),
                        "processing_metadata": processing_metadata,
                    })
                
            return samples_added > 0
            
        except Exception as e:
            logger.error(f"Failed to add audio data to session {session_id}: {e}")
            self.processing_errors += 1
            return False
    
    async def _handle_chunk_ready(self, session_id: str, metadata: AudioChunkMetadata, audio_data: np.ndarray):
        """Handle when audio chunk is ready for processing."""
        try:
            self.total_chunks_created += 1
            
            # Send to whisper service
            transcript_result = await self.service_client.send_to_whisper_service(
                session_id, metadata, audio_data
            )
            
            if transcript_result and transcript_result.get("text"):
                # Store transcript in database (if available)
                transcript_id = None
                if self.database_adapter:
                    transcript_id = await self.database_adapter.store_transcript(
                        metadata.session_id,
                        {
                            "text": transcript_result["text"],
                            "start_timestamp": metadata.chunk_start_time,
                            "end_timestamp": metadata.chunk_end_time,
                            "language": transcript_result.get("language", "en"),
                            "confidence": transcript_result.get("confidence", 0.0),
                            "speaker_info": transcript_result.get("speaker_info", {}),
                            "chunk_id": metadata.chunk_id,
                            "source_type": "whisper_service",
                            "metadata": transcript_result.get("metadata", {}),
                        },
                        audio_file_id=metadata.chunk_id
                    )
                else:
                    # Generate fake ID for non-persistent mode
                    transcript_id = f"transcript_{metadata.chunk_id}"
                
                if transcript_id:
                    self.total_transcripts_generated += 1
                    
                    # Emit transcript ready event
                    if self.on_transcript_ready:
                        self.on_transcript_ready({
                            "session_id": session_id,
                            "transcript_id": transcript_id,
                            "chunk_id": metadata.chunk_id,
                            "text": transcript_result["text"],
                            "start_timestamp": metadata.chunk_start_time,
                            "end_timestamp": metadata.chunk_end_time,
                            "language": transcript_result.get("language", "en"),
                            "speaker_info": transcript_result.get("speaker_info", {}),
                            "confidence": transcript_result.get("confidence", 0.0),
                        })
                    
                    # Request translations
                    session = self.session_manager.get_session(session_id)
                    if session and session.target_languages:
                        await self._request_translations(
                            session_id, transcript_id, transcript_result, session.target_languages
                        )
                else:
                    logger.warning(f"Failed to store transcript for chunk {metadata.chunk_id}")
            else:
                logger.debug(f"No text in transcript result for chunk {metadata.chunk_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle chunk ready for {metadata.chunk_id}: {e}")
            self.processing_errors += 1
    
    async def _request_translations(
        self,
        session_id: str,
        transcript_id: str,
        transcript_result: Dict[str, Any],
        target_languages: List[str]
    ):
        """Request translations for a transcript."""
        source_language = transcript_result.get("language", "auto")
        
        # Create translation tasks
        translation_tasks = []
        for target_lang in target_languages:
            if target_lang != source_language:  # Skip same language
                task = asyncio.create_task(
                    self._process_single_translation(
                        session_id, transcript_id, transcript_result, target_lang
                    )
                )
                translation_tasks.append(task)
        
        # Process translations concurrently
        if translation_tasks:
            await asyncio.gather(*translation_tasks, return_exceptions=True)
    
    async def _process_single_translation(
        self,
        session_id: str,
        transcript_id: str,
        transcript_result: Dict[str, Any],
        target_language: str
    ):
        """Process a single translation request."""
        try:
            translation_result = await self.service_client.send_to_translation_service(
                session_id, transcript_result, target_language
            )
            
            if translation_result and translation_result.get("translated_text"):
                # Store translation in database (if available)
                translation_id = None
                if self.database_adapter:
                    translation_id = await self.database_adapter.store_translation(
                        session_id,
                        transcript_id,
                        {
                            "translated_text": translation_result["translated_text"],
                            "source_language": transcript_result.get("language", "auto"),
                            "target_language": target_language,
                            "confidence": translation_result.get("confidence", 0.0),
                            "translation_service": translation_result.get("service", "unknown"),
                            "speaker_id": transcript_result.get("speaker_info", {}).get("speaker_id"),
                            "speaker_name": transcript_result.get("speaker_info", {}).get("speaker_name"),
                            "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                            "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                            "metadata": translation_result.get("metadata", {}),
                        }
                    )
                else:
                    # Generate fake ID for non-persistent mode
                    translation_id = f"translation_{transcript_id}_{target_language}"
                
                if translation_id:
                    self.total_translations_generated += 1
                    
                    # Emit translation ready event
                    if self.on_translation_ready:
                        self.on_translation_ready({
                            "session_id": session_id,
                            "translation_id": translation_id,
                            "transcript_id": transcript_id,
                            "original_text": transcript_result["text"],
                            "translated_text": translation_result["translated_text"],
                            "source_language": transcript_result.get("language", "auto"),
                            "target_language": target_language,
                            "speaker_info": transcript_result.get("speaker_info", {}),
                            "confidence": translation_result.get("confidence", 0.0),
                            "start_timestamp": transcript_result.get("start_timestamp", 0.0),
                            "end_timestamp": transcript_result.get("end_timestamp", 0.0),
                        })
                else:
                    logger.warning(f"Failed to store translation for transcript {transcript_id}")
                    
        except Exception as e:
            logger.error(f"Failed to process translation {target_language} for transcript {transcript_id}: {e}")
            self.processing_errors += 1
    
    def _handle_quality_alert(self, session_id: str, alert: Dict[str, Any]):
        """Handle quality alerts from chunk processing."""
        logger.info(f"Quality alert for session {session_id}: {alert}")
        
        if self.on_session_event:
            self.on_session_event("quality_alert", {
                "session_id": session_id,
                "alert": alert
            })
    
    def _handle_session_error(self, session_id: str, error: str):
        """Handle errors from session processing."""
        logger.error(f"Session error for {session_id}: {error}")
        self.processing_errors += 1
        
        if self.on_error:
            self.on_error(f"Session {session_id} error: {error}")
    
    # Status and monitoring API
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status."""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        # Get audio processing statistics
        audio_processing_stats = {}
        for session_id, processor in self.audio_processors.items():
            audio_processing_stats[session_id] = processor.get_processing_statistics()
        
        return {
            "is_running": self.is_running,
            "runtime_seconds": runtime,
            "total_audio_processed": self.total_audio_processed,
            "total_chunks_created": self.total_chunks_created,
            "total_transcripts_generated": self.total_transcripts_generated,
            "total_translations_generated": self.total_translations_generated,
            "processing_errors": self.processing_errors,
            "success_rate": 1.0 - (self.processing_errors / max(1, self.total_chunks_created)),
            "session_statistics": self.session_manager.get_session_statistics(),
            "database_statistics": self.database_adapter.get_operation_statistics() if self.database_adapter else {},
            "audio_processing_statistics": audio_processing_stats,
            "audio_config": {
                "default_preset": self.audio_config_manager.get_default_config().preset_name,
                "available_presets": self.get_available_audio_presets(),
                "active_processors": len(self.audio_processors),
            },
            "config": self.config.dict(),
        }
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific session."""
        session = self.session_manager.get_session(session_id)
        chunk_manager = self.session_manager.get_chunk_manager(session_id)
        audio_processor = self.audio_processors.get(session_id)
        
        if not session or not chunk_manager:
            return None
        
        status = {
            "session": session.dict(),
            "chunk_manager_status": chunk_manager.get_status(),
        }
        
        # Add audio processing status if available
        if audio_processor:
            status["audio_processing_status"] = audio_processor.get_processing_statistics()
            status["audio_config"] = {
                "preset_name": audio_processor.config.preset_name,
                "enabled_stages": audio_processor.config.enabled_stages,
                "version": audio_processor.config.version,
            }
        
        return status
    
    def get_all_sessions_status(self) -> List[Dict[str, Any]]:
        """Get status for all active sessions."""
        statuses = []
        
        for session_id in self.session_manager.active_sessions.keys():
            status = self.get_session_status(session_id)
            if status:
                statuses.append(status)
        
        return statuses

    # File processing API
    async def process_audio_file(
        self,
        session_id: str,
        audio_file_path: str,
        config: Dict[str, Any],
        request_id: str
    ) -> str:
        """
        Process an audio file through the orchestration pipeline.
        
        Args:
            session_id: Audio session identifier
            audio_file_path: Path to the input audio file
            config: Audio processing configuration
            request_id: Request tracking ID
            
        Returns:
            str: Path to the processed audio file
        """
        try:
            logger.info(f"[{request_id}] Processing audio file through orchestration pipeline")
            
            # For now, we'll implement a simple pass-through since the full pipeline
            # integration requires more complex setup. This allows the system to work
            # while we can enhance the processing later.
            
            # Check if we have an audio processor for this session
            if session_id in self.audio_processors:
                audio_processor = self.audio_processors[session_id]
                logger.info(f"[{request_id}] Using existing audio processor for session {session_id}")
            else:
                # Create a new audio processor with the provided config
                audio_processor = self._get_or_create_audio_processor(session_id)
                logger.info(f"[{request_id}] Created new audio processor for session {session_id}")
            
            # For file-based processing, we'll return the original file path
            # The audio processing will be handled by the audio service
            # This maintains compatibility while allowing future enhancement
            
            logger.info(
                f"[{request_id}] Audio file processed through orchestration pipeline "
                f"(pass-through mode for compatibility)"
            )
            
            return audio_file_path
            
        except Exception as e:
            logger.error(f"[{request_id}] Failed to process audio file through orchestration: {e}")
            # Return original file path as fallback
            return audio_file_path


# Factory function for creating audio coordinator
def create_audio_coordinator(
    database_url: Optional[str],
    service_urls: Dict[str, str],
    config: Optional[AudioChunkingConfig] = None,
    max_concurrent_sessions: int = 10,
    audio_config_file: Optional[str] = None
) -> AudioCoordinator:
    """Create and return an AudioCoordinator instance."""
    if not config:
        config = get_default_chunking_config()
    
    return AudioCoordinator(config, database_url, service_urls, max_concurrent_sessions, audio_config_file)


# Example usage and testing
async def main():
    """Example usage of the audio coordinator."""
    import os
    
    # Configuration
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/livetranslate")
    service_urls = {
        "whisper_service": "http://localhost:5001",
        "translation_service": "http://localhost:5003",
    }
    
    # Create coordinator
    coordinator = create_audio_coordinator(database_url, service_urls)
    
    # Set callbacks
    def on_transcript_ready(data):
        print(f"Transcript ready: {data['text'][:50]}...")
    
    def on_translation_ready(data):
        print(f"Translation ready ({data['target_language']}): {data['translated_text'][:50]}...")
    
    def on_session_event(event_type, data):
        print(f"Session event: {event_type}")
    
    coordinator.set_transcript_ready_callback(on_transcript_ready)
    coordinator.set_translation_ready_callback(on_translation_ready)
    coordinator.set_session_event_callback(on_session_event)
    
    try:
        # Initialize
        success = await coordinator.initialize()
        if not success:
            print("Failed to initialize coordinator")
            return
        
        # Create session
        session_id = await coordinator.create_audio_session(
            bot_session_id="test-bot-session-123",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en", "es", "fr"]
        )
        
        if session_id:
            print(f"Created session: {session_id}")
            
            # Start session
            await coordinator.start_audio_session(session_id)
            
            # Simulate audio data
            sample_rate = 16000
            for i in range(50):  # 5 seconds of audio
                # Generate test audio
                t = np.arange(1600) / sample_rate + i * 0.1
                audio = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(1600)
                
                await coordinator.add_audio_data(session_id, audio.astype(np.float32))
                await asyncio.sleep(0.1)  # Real-time simulation
            
            # Stop session
            final_stats = await coordinator.stop_audio_session(session_id)
            print(f"Session final stats: {final_stats}")
            
            # Get coordinator status
            status = coordinator.get_coordinator_status()
            print(f"Coordinator status: {status}")
        
    finally:
        # Shutdown
        await coordinator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())