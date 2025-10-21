#!/usr/bin/env python3
"""
Unified Bot Session Repository - Orchestration Service

Consolidates all bot session database operations into a single, cohesive repository
that replaces the 5 separate manager classes:

- AudioFileManager -> audio_* methods
- TranscriptManager -> transcript_* methods  
- TranslationManager -> translation_* methods
- CorrelationManager -> correlation_* methods
- BotSessionDatabaseManager -> session_* methods

This unified approach reduces complexity while maintaining all functionality
through a clean repository pattern with specialized method groups.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid

from .database import DatabaseManager
from .models import (
    BotSession,
    AudioFile,
    Transcript,
    Translation,
)
from audio.models import SpeakerCorrelation  # Pydantic model
from .processing_metrics import ProcessingMetrics

logger = logging.getLogger(__name__)


class UnifiedBotSessionRepository:
    """
    Unified repository for all bot session database operations.
    
    Consolidates functionality from 5 separate managers into organized method groups:
    - session_* methods: Bot session lifecycle management
    - audio_* methods: Audio file storage and retrieval
    - transcript_* methods: Transcript management
    - translation_* methods: Translation handling
    - correlation_* methods: Speaker correlation tracking
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """Initialize with database manager."""
        self.db = database_manager
        self._connection_pool = None
        
        logger.info("UnifiedBotSessionRepository initialized")
    
    async def initialize(self) -> bool:
        """Initialize the repository and database connections."""
        try:
            # Ensure database manager is initialized
            if not await self.db.initialize():
                logger.error("Failed to initialize database manager")
                return False
            
            # Create connection pool if needed
            self._connection_pool = await self.db.get_connection_pool()
            
            logger.info("Unified bot session repository initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize unified bot session repository: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown repository and close connections."""
        try:
            if self.db:
                await self.db.close()
            logger.info("Unified bot session repository shutdown completed")
        except Exception as e:
            logger.error(f"Error during repository shutdown: {e}")
    
    # =============================================================================
    # SESSION MANAGEMENT METHODS (replaces BotSessionDatabaseManager)
    # =============================================================================
    
    async def session_create(self, 
                           session_id: str,
                           meeting_url: str,
                           bot_config: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> Optional[BotSession]:
        """Create a new bot session."""
        try:
            session = BotSession(
                session_id=session_id,
                meeting_url=meeting_url,
                status="initializing",
                bot_config=bot_config,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            async with self.db.get_session() as db_session:
                db_session.add(session)
                await db_session.commit()
                await db_session.refresh(session)
                
            logger.info(f"Created bot session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create bot session {session_id}: {e}")
            return None
    
    async def session_get(self, session_id: str) -> Optional[BotSession]:
        """Get bot session by ID."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM bot_sessions WHERE session_id = ?", (session_id,)
                )
                row = result.fetchone()
                
                if row:
                    return BotSession(**dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get bot session {session_id}: {e}")
            return None
    
    async def session_update_status(self, session_id: str, status: str) -> bool:
        """Update bot session status."""
        try:
            async with self.db.get_session() as db_session:
                await db_session.execute(
                    "UPDATE bot_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                    (status, datetime.utcnow(), session_id)
                )
                await db_session.commit()
                
            logger.debug(f"Updated session {session_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
            return False
    
    async def session_list_active(self) -> List[BotSession]:
        """List all active bot sessions."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM bot_sessions WHERE status IN ('active', 'recording', 'processing')"
                )
                rows = result.fetchall()
                
                return [BotSession(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list active sessions: {e}")
            return []
    
    async def session_cleanup_old(self, older_than_hours: int = 24) -> int:
        """Clean up old completed sessions."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "DELETE FROM bot_sessions WHERE status = 'completed' AND updated_at < ?",
                    (cutoff_time,)
                )
                await db_session.commit()
                
                deleted_count = result.rowcount
                logger.info(f"Cleaned up {deleted_count} old bot sessions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    # =============================================================================
    # AUDIO FILE METHODS (replaces AudioFileManager)
    # =============================================================================
    
    async def audio_store_file(self, 
                              session_id: str,
                              file_path: str,
                              file_type: str = "wav",
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store audio file information in database."""
        try:
            file_id = str(uuid.uuid4())
            
            audio_file = AudioFile(
                file_id=file_id,
                session_id=session_id,
                file_path=file_path,
                file_type=file_type,
                file_size=Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            async with self.db.get_session() as db_session:
                db_session.add(audio_file)
                await db_session.commit()
                
            logger.debug(f"Stored audio file {file_id} for session {session_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store audio file: {e}")
            return None
    
    async def audio_get_files(self, session_id: str) -> List[AudioFile]:
        """Get all audio files for a session."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM audio_files WHERE session_id = ? ORDER BY created_at",
                    (session_id,)
                )
                rows = result.fetchall()
                
                return [AudioFile(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get audio files for session {session_id}: {e}")
            return []
    
    async def audio_delete_file(self, file_id: str) -> bool:
        """Delete audio file record and optionally the file itself."""
        try:
            # Get file info first
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT file_path FROM audio_files WHERE file_id = ?", (file_id,)
                )
                row = result.fetchone()
                
                if row:
                    file_path = row['file_path']
                    
                    # Delete database record
                    await db_session.execute(
                        "DELETE FROM audio_files WHERE file_id = ?", (file_id,)
                    )
                    await db_session.commit()
                    
                    # Optionally delete physical file
                    try:
                        if Path(file_path).exists():
                            Path(file_path).unlink()
                    except Exception as file_e:
                        logger.warning(f"Failed to delete physical file {file_path}: {file_e}")
                    
                    logger.debug(f"Deleted audio file {file_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete audio file {file_id}: {e}")
            return False
    
    # =============================================================================
    # TRANSCRIPT METHODS (replaces TranscriptManager)
    # =============================================================================
    
    async def transcript_store(self, 
                              session_id: str,
                              text: str,
                              speaker_id: Optional[str] = None,
                              timestamp: Optional[datetime] = None,
                              confidence: float = 1.0,
                              metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store transcript segment."""
        try:
            transcript_id = str(uuid.uuid4())
            
            transcript = Transcript(
                transcript_id=transcript_id,
                session_id=session_id,
                text=text,
                speaker_id=speaker_id,
                timestamp=timestamp or datetime.utcnow(),
                confidence=confidence,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            async with self.db.get_session() as db_session:
                db_session.add(transcript)
                await db_session.commit()
                
            logger.debug(f"Stored transcript {transcript_id} for session {session_id}")
            return transcript_id
            
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return None
    
    async def transcript_get_by_session(self, 
                                      session_id: str,
                                      speaker_id: Optional[str] = None) -> List[Transcript]:
        """Get transcripts for a session, optionally filtered by speaker."""
        try:
            query = "SELECT * FROM transcripts WHERE session_id = ?"
            params = [session_id]
            
            if speaker_id:
                query += " AND speaker_id = ?"
                params.append(speaker_id)
            
            query += " ORDER BY timestamp"
            
            async with self.db.get_session() as db_session:
                result = await db_session.execute(query, params)
                rows = result.fetchall()
                
                return [Transcript(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get transcripts for session {session_id}: {e}")
            return []
    
    async def transcript_get_recent(self, 
                                   session_id: str,
                                   minutes: int = 5) -> List[Transcript]:
        """Get recent transcripts from the last N minutes."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM transcripts WHERE session_id = ? AND timestamp > ? ORDER BY timestamp",
                    (session_id, cutoff_time)
                )
                rows = result.fetchall()
                
                return [Transcript(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get recent transcripts: {e}")
            return []
    
    # =============================================================================
    # TRANSLATION METHODS (replaces TranslationManager)
    # =============================================================================
    
    async def translation_store(self, 
                               transcript_id: str,
                               target_language: str,
                               translated_text: str,
                               confidence: float = 1.0,
                               metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store translation for a transcript."""
        try:
            translation_id = str(uuid.uuid4())
            
            translation = Translation(
                translation_id=translation_id,
                transcript_id=transcript_id,
                target_language=target_language,
                translated_text=translated_text,
                confidence=confidence,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            async with self.db.get_session() as db_session:
                db_session.add(translation)
                await db_session.commit()
                
            logger.debug(f"Stored translation {translation_id} for transcript {transcript_id}")
            return translation_id
            
        except Exception as e:
            logger.error(f"Failed to store translation: {e}")
            return None
    
    async def translation_get_by_transcript(self, transcript_id: str) -> List[Translation]:
        """Get all translations for a transcript."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM translations WHERE transcript_id = ? ORDER BY created_at",
                    (transcript_id,)
                )
                rows = result.fetchall()
                
                return [Translation(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get translations for transcript {transcript_id}: {e}")
            return []
    
    async def translation_get_by_language(self, 
                                        session_id: str,
                                        target_language: str) -> List[Dict[str, Any]]:
        """Get all translations for a session in a specific language."""
        try:
            query = """
            SELECT t.*, tr.text as original_text, tr.speaker_id, tr.timestamp
            FROM translations t
            JOIN transcripts tr ON t.transcript_id = tr.transcript_id
            WHERE tr.session_id = ? AND t.target_language = ?
            ORDER BY tr.timestamp
            """
            
            async with self.db.get_session() as db_session:
                result = await db_session.execute(query, (session_id, target_language))
                rows = result.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get translations by language: {e}")
            return []
    
    # =============================================================================
    # CORRELATION METHODS (replaces CorrelationManager)
    # =============================================================================
    
    async def correlation_store(self, 
                               session_id: str,
                               whisper_speaker_id: str,
                               meet_speaker_name: str,
                               confidence: float = 1.0,
                               metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Store speaker correlation between Whisper and Google Meet."""
        try:
            correlation_id = str(uuid.uuid4())
            
            correlation = SpeakerCorrelation(
                correlation_id=correlation_id,
                session_id=session_id,
                whisper_speaker_id=whisper_speaker_id,
                meet_speaker_name=meet_speaker_name,
                confidence=confidence,
                metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            
            async with self.db.get_session() as db_session:
                db_session.add(correlation)
                await db_session.commit()
                
            logger.debug(f"Stored speaker correlation {correlation_id}")
            return correlation_id
            
        except Exception as e:
            logger.error(f"Failed to store speaker correlation: {e}")
            return None
    
    async def correlation_get_by_session(self, session_id: str) -> List[SpeakerCorrelation]:
        """Get all speaker correlations for a session."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT * FROM speaker_correlations WHERE session_id = ? ORDER BY created_at",
                    (session_id,)
                )
                rows = result.fetchall()
                
                return [SpeakerCorrelation(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get correlations for session {session_id}: {e}")
            return []
    
    async def correlation_resolve_speaker(self, 
                                        session_id: str,
                                        whisper_speaker_id: str) -> Optional[str]:
        """Resolve Whisper speaker ID to Google Meet speaker name."""
        try:
            async with self.db.get_session() as db_session:
                result = await db_session.execute(
                    "SELECT meet_speaker_name FROM speaker_correlations WHERE session_id = ? AND whisper_speaker_id = ? ORDER BY confidence DESC LIMIT 1",
                    (session_id, whisper_speaker_id)
                )
                row = result.fetchone()
                
                return row['meet_speaker_name'] if row else None
                
        except Exception as e:
            logger.error(f"Failed to resolve speaker: {e}")
            return None
    
    # =============================================================================
    # UTILITY AND ANALYTICS METHODS
    # =============================================================================
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a session."""
        try:
            stats = {
                "session_id": session_id,
                "audio_files_count": 0,
                "transcripts_count": 0,
                "translations_count": 0,
                "correlations_count": 0,
                "total_duration_seconds": 0,
                "languages": [],
                "speakers": []
            }
            
            async with self.db.get_session() as db_session:
                # Audio files count
                result = await db_session.execute(
                    "SELECT COUNT(*) as count FROM audio_files WHERE session_id = ?",
                    (session_id,)
                )
                stats["audio_files_count"] = result.fetchone()['count']
                
                # Transcripts count
                result = await db_session.execute(
                    "SELECT COUNT(*) as count FROM transcripts WHERE session_id = ?",
                    (session_id,)
                )
                stats["transcripts_count"] = result.fetchone()['count']
                
                # Translations count and languages
                result = await db_session.execute(
                    "SELECT COUNT(*) as count, target_language FROM translations t JOIN transcripts tr ON t.transcript_id = tr.transcript_id WHERE tr.session_id = ? GROUP BY target_language",
                    (session_id,)
                )
                translation_data = result.fetchall()
                stats["translations_count"] = sum(row['count'] for row in translation_data)
                stats["languages"] = [row['target_language'] for row in translation_data]
                
                # Correlations count and speakers
                result = await db_session.execute(
                    "SELECT COUNT(*) as count, meet_speaker_name FROM speaker_correlations WHERE session_id = ? GROUP BY meet_speaker_name",
                    (session_id,)
                )
                correlation_data = result.fetchall()
                stats["correlations_count"] = sum(row['count'] for row in correlation_data)
                stats["speakers"] = [row['meet_speaker_name'] for row in correlation_data]
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the repository."""
        try:
            health_status = {
                "status": "healthy",
                "database_connected": False,
                "tables_accessible": False,
                "last_check": datetime.utcnow().isoformat()
            }
            
            # Check database connection
            async with self.db.get_session() as db_session:
                await db_session.execute("SELECT 1")
                health_status["database_connected"] = True
                
                # Check table accessibility
                tables = ["bot_sessions", "audio_files", "transcripts", "translations", "speaker_correlations"]
                for table in tables:
                    await db_session.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                
                health_status["tables_accessible"] = True
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }


# Global repository instance
_unified_repository: Optional[UnifiedBotSessionRepository] = None


async def get_unified_bot_session_repository() -> UnifiedBotSessionRepository:
    """Get the global unified bot session repository instance."""
    global _unified_repository
    
    if _unified_repository is None:
        from .database import get_database_manager
        db_manager = await get_database_manager()
        _unified_repository = UnifiedBotSessionRepository(db_manager)
        await _unified_repository.initialize()
    
    return _unified_repository
