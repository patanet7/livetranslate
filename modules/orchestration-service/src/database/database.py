"""
Database Configuration and Connection Management

SQLAlchemy async database setup and connection management.
"""

import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event
from contextlib import asynccontextmanager

from .models import Base

logger = logging.getLogger(__name__)


# DatabaseConfig now imported from config.py
from config import DatabaseSettings as DatabaseConfig


class DatabaseManager:
    """Async database manager"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self._initialized = False

    def initialize(self):
        """Initialize database engine and session factory"""
        if self._initialized:
            return

        # Create async engine
        self.engine = create_async_engine(
            self.config.url,
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            # Enable async mode
            future=True,
        )

        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Add event listeners
        self._setup_event_listeners()

        self._initialized = True
        logger.info("Database manager initialized")

    def _setup_event_listeners(self):
        """Setup database event listeners"""

        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for development"""
            if "sqlite" in self.config.url:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        @event.listens_for(self.engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout"""
            logger.debug("Database connection checked out")

        @event.listens_for(self.engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin"""
            logger.debug("Database connection checked in")

    async def create_tables(self):
        """Create all database tables"""
        if not self._initialized:
            self.initialize()

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created")

    async def drop_tables(self):
        """Drop all database tables"""
        if not self._initialized:
            self.initialize()

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        logger.info("Database tables dropped")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager"""
        if not self._initialized:
            self.initialize()

        async with self.session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def get_session_direct(self) -> AsyncSession:
        """Get database session directly (for dependency injection)"""
        if not self._initialized:
            self.initialize()

        return self.session_factory()

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    def get_stats(self) -> dict:
        """Get database connection pool statistics"""
        if not self.engine:
            return {}

        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }


# Global database manager instance
database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global database_manager
    if database_manager is None:
        raise RuntimeError("Database manager not initialized")
    return database_manager


def initialize_database(config: DatabaseConfig):
    """Initialize global database manager"""
    global database_manager
    database_manager = DatabaseManager(config)
    database_manager.initialize()
    return database_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection function for FastAPI"""
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


# Database utility functions
class DatabaseUtils:
    """Database utility functions"""

    @staticmethod
    async def create_test_data(session: AsyncSession):
        """Create test data for development"""
        from .models import BotSession, AudioFile, Transcript, Translation, Participant
        from datetime import datetime
        import uuid

        # Create test session
        test_session = BotSession(
            bot_id="test-bot-123",
            meeting_id="test-meeting-456",
            meeting_title="Test Meeting",
            meeting_uri="https://meet.google.com/test-meeting",
            bot_type="google_meet",
            status="running",
            target_languages=["en", "es"],
            enable_translation=True,
            enable_transcription=True,
            metadata={"test": True},
        )

        session.add(test_session)
        await session.commit()

        # Create test audio file
        test_audio = AudioFile(
            session_id=test_session.session_id,
            filename="test_audio.wav",
            file_path="/tmp/test_audio.wav",
            file_size=1024000,
            file_hash="test_hash_123",
            mime_type="audio/wav",
            duration=60.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )

        session.add(test_audio)
        await session.commit()

        # Create test transcript
        test_transcript = Transcript(
            session_id=test_session.session_id,
            text="This is a test transcript",
            language="en",
            confidence=0.95,
            source="whisper",
            audio_file_id=test_audio.file_id,
            speaker_id="speaker_1",
            speaker_name="Test Speaker",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )

        session.add(test_transcript)
        await session.commit()

        # Create test translation
        test_translation = Translation(
            session_id=test_session.session_id,
            transcript_id=test_transcript.transcript_id,
            original_text="This is a test transcript",
            translated_text="Esta es una transcripcion de prueba",
            source_language="en",
            target_language="es",
            confidence=0.92,
            speaker_id="speaker_1",
            speaker_name="Test Speaker",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            quality_score=0.88,
            word_count=6,
            character_count=25,
        )

        session.add(test_translation)
        await session.commit()

        # Create test participant
        test_participant = Participant(
            session_id=test_session.session_id,
            external_id="participant_123",
            name="Test Speaker",
            email="test@example.com",
            speaker_id="speaker_1",
            joined_at=datetime.utcnow(),
            speaking_time=120.0,
            word_count=100,
        )

        session.add(test_participant)
        await session.commit()

        logger.info("Test data created successfully")
        return test_session.session_id

    @staticmethod
    async def cleanup_old_sessions(session: AsyncSession, days: int = 30):
        """Clean up old session data"""
        from .models import BotSession
        from datetime import datetime, timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Delete old sessions (cascade will handle related data)
        result = await session.execute(
            "DELETE FROM bot_sessions WHERE created_at < :cutoff_date",
            {"cutoff_date": cutoff_date},
        )

        await session.commit()

        logger.info(f"Cleaned up {result.rowcount} old sessions")
        return result.rowcount

    @staticmethod
    async def get_session_statistics(session: AsyncSession, session_id: str) -> dict:
        """Get detailed session statistics"""
        from .models import BotSession, AudioFile, Transcript, Translation, Participant
        from sqlalchemy import func

        # Get basic session info
        bot_session = await session.get(BotSession, session_id)
        if not bot_session:
            return {}

        # Get counts
        audio_count = await session.scalar(
            func.count(AudioFile.file_id).where(AudioFile.session_id == session_id)
        )

        transcript_count = await session.scalar(
            func.count(Transcript.transcript_id).where(
                Transcript.session_id == session_id
            )
        )

        translation_count = await session.scalar(
            func.count(Translation.translation_id).where(
                Translation.session_id == session_id
            )
        )

        participant_count = await session.scalar(
            func.count(Participant.participant_id).where(
                Participant.session_id == session_id
            )
        )

        # Get aggregated metrics
        total_audio_duration = (
            await session.scalar(
                func.sum(AudioFile.duration).where(AudioFile.session_id == session_id)
            )
            or 0
        )

        total_audio_size = (
            await session.scalar(
                func.sum(AudioFile.file_size).where(AudioFile.session_id == session_id)
            )
            or 0
        )

        average_confidence = await session.scalar(
            func.avg(Transcript.confidence).where(Transcript.session_id == session_id)
        )

        return {
            "session_id": session_id,
            "status": bot_session.status,
            "created_at": bot_session.created_at,
            "started_at": bot_session.started_at,
            "ended_at": bot_session.ended_at,
            "duration": (bot_session.ended_at - bot_session.started_at).total_seconds()
            if bot_session.started_at and bot_session.ended_at
            else None,
            "audio_files_count": audio_count,
            "transcripts_count": transcript_count,
            "translations_count": translation_count,
            "participants_count": participant_count,
            "total_audio_duration": total_audio_duration,
            "total_audio_size": total_audio_size,
            "average_confidence": float(average_confidence)
            if average_confidence
            else None,
            "languages_detected": bot_session.target_languages,
        }


# Migration utilities
class MigrationManager:
    """Database migration management"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def run_migrations(self):
        """Run database migrations"""
        # This would implement actual migration logic
        # For now, just create tables
        await self.db_manager.create_tables()
        logger.info("Database migrations completed")

    async def create_indexes(self):
        """Create additional indexes for performance"""
        async with self.db_manager.get_session() as session:
            # Create custom indexes
            await session.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transcripts_text_search 
                ON transcripts USING gin(to_tsvector('english', text))
            """
            )

            await session.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translations_text_search 
                ON translations USING gin(to_tsvector('english', translated_text))
            """
            )

            await session.commit()

        logger.info("Additional indexes created")

    async def optimize_database(self):
        """Optimize database performance"""
        async with self.db_manager.get_session() as session:
            # Analyze tables for query optimization
            await session.execute("ANALYZE")
            await session.commit()

        logger.info("Database optimized")
