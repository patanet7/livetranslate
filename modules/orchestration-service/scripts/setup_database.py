#!/usr/bin/env python3
"""
Database Setup Script for Orchestration Service

This script handles initial database setup, migration execution, and verification.
It provides a single entry point for all database initialization needs.

Usage:
    # Using PDM (recommended)
    DATABASE_URL="postgresql://user:pass@localhost:5432/dbname" pdm run python scripts/setup_database.py

    # Direct execution
    DATABASE_URL="postgresql://user:pass@localhost:5432/dbname" python scripts/setup_database.py

    # With options
    DATABASE_URL="..." python scripts/setup_database.py --create-test-data
    DATABASE_URL="..." python scripts/setup_database.py --verify-only
    DATABASE_URL="..." python scripts/setup_database.py --reset  # WARNING: Drops all tables!
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Example: DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'")
        sys.exit(1)
    return url


def run_alembic_command(command: list[str]) -> tuple[int, str, str]:
    """Run an alembic command and return exit code, stdout, stderr."""
    env = os.environ.copy()
    env["DATABASE_URL"] = get_database_url()

    # Change to the orchestration-service directory
    cwd = Path(__file__).parent.parent

    result = subprocess.run(
        ["pdm", "run", "alembic", *command], cwd=cwd, env=env, capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


def check_database_connection() -> bool:
    """Verify database connection is working."""
    logger.info("Checking database connection...")

    try:
        import psycopg2

        db_url = get_database_url()
        # Parse URL for psycopg2
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result[0] == 1:
            logger.info("Database connection successful")
            return True
        else:
            logger.error("Database connection check failed")
            return False
    except ImportError:
        # Fall back to alembic check
        logger.info("psycopg2 not available, using alembic for connection check")
        code, _stdout, stderr = run_alembic_command(["current"])
        if code == 0:
            logger.info("Database connection successful (via alembic)")
            return True
        else:
            logger.error(f"Database connection failed: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_current_migration() -> str | None:
    """Get the current migration revision."""
    code, stdout, stderr = run_alembic_command(["current"])
    if code != 0:
        logger.error(f"Failed to get current migration: {stderr}")
        return None

    # Parse the output to get revision
    for line in stdout.strip().split("\n"):
        if line and not line.startswith("INFO"):
            return line.strip()

    return None


def run_migrations() -> bool:
    """Run all pending database migrations."""
    logger.info("Running database migrations...")

    code, stdout, stderr = run_alembic_command(["upgrade", "head"])

    if code != 0:
        logger.error(f"Migration failed: {stderr}")
        return False

    # Log the output
    for line in (stdout + stderr).split("\n"):
        if line.strip():
            logger.info(line.strip())

    logger.info("Migrations completed successfully")
    return True


def verify_database_schema() -> bool:
    """Verify database schema matches models."""
    logger.info("Verifying database schema...")

    _code, stdout, stderr = run_alembic_command(["check"])

    output = stdout + stderr

    if "No new upgrade operations detected" in output:
        logger.info("Database schema is in sync with models")
        return True
    else:
        logger.warning("Database schema may be out of sync with models")
        logger.warning(output)
        return False


def list_tables() -> list[str]:
    """List all tables in the database."""
    try:
        import psycopg2

        db_url = get_database_url()
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        return tables
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        return []


async def create_test_data() -> bool:
    """Create test data for development."""
    logger.info("Creating test data...")

    try:
        from config import DatabaseSettings
        from database.database import DatabaseManager, DatabaseUtils

        db_url = get_database_url()
        config = DatabaseSettings(url=db_url)

        db_manager = DatabaseManager(config)
        db_manager.initialize()

        async with db_manager.get_session() as session:
            session_id = await DatabaseUtils.create_test_data(session)
            logger.info(f"Test data created. Session ID: {session_id}")

        await db_manager.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create test data: {e}")
        return False


def reset_database() -> bool:
    """Reset database by running downgrade then upgrade. WARNING: Destructive!"""
    logger.warning("Resetting database - ALL DATA WILL BE LOST!")

    # Downgrade to base
    logger.info("Downgrading database to base...")
    code, _stdout, stderr = run_alembic_command(["downgrade", "base"])
    if code != 0:
        logger.error(f"Downgrade failed: {stderr}")
        return False

    # Upgrade to head
    logger.info("Upgrading database to head...")
    return run_migrations()


def ensure_extensions() -> bool:
    """Ensure required PostgreSQL extensions are installed."""
    logger.info("Checking required PostgreSQL extensions...")

    try:
        import psycopg2

        db_url = get_database_url()
        if db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Check for pg_trgm (needed for full-text search indexes)
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        if not cursor.fetchone():
            logger.info("Creating pg_trgm extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.commit()
            logger.info("pg_trgm extension created")
        else:
            logger.info("pg_trgm extension already installed")

        # Check for uuid-ossp (for UUID generation)
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp'")
        if not cursor.fetchone():
            logger.info("Creating uuid-ossp extension...")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            conn.commit()
            logger.info("uuid-ossp extension created")
        else:
            logger.info("uuid-ossp extension already installed")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Could not check/create extensions: {e}")
        logger.warning("You may need to create extensions manually with superuser privileges")
        return False


def print_status() -> None:
    """Print current database status."""
    print("\n" + "=" * 60)
    print("DATABASE STATUS")
    print("=" * 60)

    # Current migration
    current = get_current_migration()
    print(f"\nCurrent Migration: {current or 'None'}")

    # List tables
    tables = list_tables()
    print(f"\nTables ({len(tables)}):")
    for table in tables:
        print(f"  - {table}")

    # Schema check
    print("\nSchema Status:", end=" ")
    if verify_database_schema():
        print("IN SYNC")
    else:
        print("OUT OF SYNC (run migrations)")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Database setup and management for orchestration service"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify database schema, don't run migrations",
    )
    parser.add_argument(
        "--create-test-data", action="store_true", help="Create test data after setup"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset database (WARNING: Drops all tables!)"
    )
    parser.add_argument("--status", action="store_true", help="Print database status and exit")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ORCHESTRATION SERVICE DATABASE SETUP")
    print("=" * 60 + "\n")

    # Status only
    if args.status:
        if not check_database_connection():
            sys.exit(1)
        print_status()
        sys.exit(0)

    # Step 1: Check connection
    if not check_database_connection():
        logger.error("Cannot proceed without database connection")
        sys.exit(1)

    # Step 2: Ensure extensions
    ensure_extensions()

    # Step 3: Handle reset if requested
    if args.reset:
        confirm = input("\nThis will DELETE ALL DATA. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            if not reset_database():
                sys.exit(1)
        else:
            logger.info("Reset cancelled")
            sys.exit(0)

    # Step 4: Run migrations (unless verify-only)
    if args.verify_only:
        logger.info("Verify-only mode, skipping migrations")
    else:
        if not run_migrations():
            sys.exit(1)

    # Step 5: Verify schema
    if not verify_database_schema() and not args.verify_only:
        logger.warning("Schema verification failed after migrations")

    # Step 6: Create test data if requested
    if args.create_test_data:
        asyncio.run(create_test_data())

    # Print final status
    print_status()

    logger.info("Database setup completed successfully!")


if __name__ == "__main__":
    main()
