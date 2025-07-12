"""
Database base configuration and session management

Async SQLAlchemy setup with PostgreSQL support and session management
"""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import event
from sqlalchemy.engine import Engine

from ..config import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy base class
Base = declarative_base()

# Global variables for engine and session maker
_engine = None
_async_session_maker = None


def _setup_sqlite_pragma(dbapi_connection, connection_record):
    """Setup SQLite pragmas for better performance and WAL mode"""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys=ON")
        # Optimize performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


async def init_database():
    """Initialize database connection and create tables"""
    global _engine, _async_session_maker

    try:
        settings = get_settings()

        # Create async engine
        database_url = settings.get_database_url()

        # Convert PostgreSQL URL to async if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif database_url.startswith("sqlite:///"):
            database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

        logger.info(
            f"Connecting to database: {database_url.split('@')[-1] if '@' in database_url else database_url}"
        )

        # Engine configuration
        engine_kwargs = {
            "echo": settings.database.echo,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # 1 hour
        }

        if "sqlite" in database_url:
            # SQLite specific configuration
            engine_kwargs.update(
                {
                    "poolclass": NullPool,
                    "connect_args": {"check_same_thread": False, "timeout": 30},
                }
            )
        else:
            # PostgreSQL specific configuration
            engine_kwargs.update(
                {
                    "pool_size": settings.database.pool_size,
                    "max_overflow": settings.database.max_overflow,
                    "pool_timeout": 30,
                }
            )

        _engine = create_async_engine(database_url, **engine_kwargs)

        # Setup SQLite pragmas if using SQLite
        if "sqlite" in database_url:
            event.listen(Engine, "connect", _setup_sqlite_pragma)

        # Create session maker
        _async_session_maker = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create all tables
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database initialized successfully")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections"""
    global _engine

    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with automatic cleanup"""
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def get_db_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session"""
    async with get_db_session() as session:
        yield session


class DatabaseHealth:
    """Database health check utilities"""

    @staticmethod
    async def check_connection() -> bool:
        """Check if database connection is healthy"""
        try:
            async with get_db_session() as session:
                # Simple query to test connection
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @staticmethod
    async def get_connection_info() -> dict:
        """Get database connection information"""
        try:
            if _engine is None:
                return {"status": "not_initialized"}

            pool = _engine.pool

            return {
                "status": "connected",
                "pool_size": getattr(pool, "size", "unknown"),
                "checked_out_connections": getattr(pool, "checkedout", "unknown"),
                "overflow_connections": getattr(pool, "overflow", "unknown"),
                "invalid_connections": getattr(pool, "invalidated", "unknown"),
            }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {"status": "error", "error": str(e)}


class DatabaseMigration:
    """Database migration utilities"""

    @staticmethod
    async def create_tables():
        """Create all database tables"""
        if _engine is None:
            raise RuntimeError("Database not initialized")

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created")

    @staticmethod
    async def drop_tables():
        """Drop all database tables (use with caution!)"""
        if _engine is None:
            raise RuntimeError("Database not initialized")

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

        logger.warning("All database tables dropped")

    @staticmethod
    async def reset_database():
        """Reset database by dropping and recreating tables"""
        await DatabaseMigration.drop_tables()
        await DatabaseMigration.create_tables()
        logger.info("Database reset completed")


# Utility functions for testing
async def setup_test_database():
    """Setup test database with in-memory SQLite"""
    global _engine, _async_session_maker

    # Use in-memory SQLite for testing
    database_url = "sqlite+aiosqlite:///:memory:"

    _engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )

    _async_session_maker = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Test database setup completed")


async def cleanup_test_database():
    """Cleanup test database"""
    global _engine

    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Test database cleanup completed")
