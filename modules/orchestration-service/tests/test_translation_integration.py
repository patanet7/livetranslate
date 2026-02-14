#!/usr/bin/env python3
"""
Integration Test for Translation with Database Persistence

Tests the complete flow:
1. Session creation
2. Translation requests with session ID
3. Database persistence
4. Multi-language storage
"""

import asyncio

# Import database components
import socket
import sys
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_settings
from database import (
    BotSession,
    DatabaseConfig,
    DatabaseManager,
    Translation,
)

settings = get_settings()


# Test configuration
ORCHESTRATION_URL = "http://localhost:3000"
TEST_TRANSLATIONS = [
    {
        "text": "Hello, how are you today?",
        "target_lang": "es",
        "expected_lang": "Spanish",
    },
    {
        "text": "Thank you very much for your help!",
        "target_lang": "fr",
        "expected_lang": "French",
    },
    {
        "text": "Good morning, have a nice day!",
        "target_lang": "de",
        "expected_lang": "German",
    },
    {
        "text": "Welcome to our translation service!",
        "target_lang": "zh",
        "expected_lang": "Chinese",
    },
]


async def create_test_session(db: AsyncSession) -> str:
    """Create a test bot session in the database"""
    session = BotSession(
        bot_id=f"test-bot-{uuid.uuid4().hex[:8]}",
        meeting_id=f"test-meeting-{uuid.uuid4().hex[:8]}",
        meeting_title="Integration Test Session",
        meeting_uri="https://meet.google.com/integration-test",
        bot_type="test",
        status="running",
        target_languages=["es", "fr", "de", "zh"],
        enable_translation=True,
        enable_transcription=True,
        session_metadata={"test": True, "integration_test": True},
    )

    db.add(session)
    await db.commit()
    await db.refresh(session)

    print(f"✅ Created test session: {session.session_id}")
    return str(session.session_id)


def _orchestration_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 3000), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _orchestration_reachable(),
    reason="Orchestration service not running on localhost:3000",
)
async def test_translation_with_session(session_id: str, text: str, target_lang: str) -> dict:
    """Test translation API with session ID"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{ORCHESTRATION_URL}/api/translation/",
            json={
                "text": text,
                "target_language": target_lang,
                "source_language": "en",
                "session_id": session_id,
            },
        )

        if response.status_code != 200:
            print(f"❌ Translation request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

        result = response.json()
        print("✅ Translation successful:")
        print(f"   Original: {text}")
        print(f"   Translated: {result['translated_text']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Model: {result['model_used']}")
        print(f"   Backend: {result['backend_used']}")

        return result


async def verify_database_persistence(db: AsyncSession, session_id: str) -> list[Translation]:
    """Verify translations were persisted to database"""
    session_uuid = uuid.UUID(session_id)

    # Query translations for this session
    result = await db.execute(
        select(Translation)
        .where(Translation.session_id == session_uuid)
        .order_by(Translation.start_time)
    )
    translations = result.scalars().all()

    print("\n📊 Database Verification:")
    print(f"   Session ID: {session_id}")
    print(f"   Translations found: {len(translations)}")

    for idx, trans in enumerate(translations, 1):
        print(f"\n   Translation #{idx}:")
        print(f"      ID: {trans.translation_id}")
        print(f"      Original: {trans.original_text[:50]}...")
        print(f"      Translated: {trans.translated_text[:50]}...")
        print(f"      Languages: {trans.source_language} → {trans.target_language}")
        print(f"      Confidence: {trans.confidence}")
        print(f"      Word count: {trans.word_count}")
        print(f"      Char count: {trans.character_count}")
        print(f"      Metadata: {trans.session_metadata}")

    return translations


async def verify_session_statistics(db: AsyncSession, session_id: str):
    """Verify session statistics"""
    session_uuid = uuid.UUID(session_id)

    # Get session
    session = await db.get(BotSession, session_uuid)

    if not session:
        print("❌ Session not found!")
        return

    # Get translation counts
    result = await db.execute(select(Translation).where(Translation.session_id == session_uuid))
    translations = result.scalars().all()

    # Get unique languages
    target_languages = {t.target_language for t in translations}

    print("\n📈 Session Statistics:")
    print(f"   Session: {session.meeting_title}")
    print(f"   Status: {session.status}")
    print(f"   Total translations: {len(translations)}")
    print(f"   Target languages: {sorted(target_languages)}")
    print(f"   Created: {session.created_at}")


async def run_integration_test():
    """Run complete integration test"""
    print("=" * 80)
    print("🚀 Translation Integration Test with Database Persistence")
    print("=" * 80)

    # Initialize database manager
    db_config = DatabaseConfig(
        url=settings.database.url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
        pool_pre_ping=settings.database.pool_pre_ping,
    )

    db_manager = DatabaseManager(db_config)
    db_manager.initialize()

    # Create tables if they don't exist
    try:
        await db_manager.create_tables()
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"⚠️  Database tables may already exist: {e}")

    # Step 1: Create test session
    print("\n" + "=" * 80)
    print("Step 1: Creating test session...")
    print("=" * 80)

    async with db_manager.get_session() as db:
        session_id = await create_test_session(db)

    # Step 2: Test translations with session
    print("\n" + "=" * 80)
    print("Step 2: Testing translations with session ID...")
    print("=" * 80)

    translation_results = []
    for test_case in TEST_TRANSLATIONS:
        print(f"\n🔄 Testing {test_case['expected_lang']} translation...")
        result = await test_translation_with_session(
            session_id=session_id,
            text=test_case["text"],
            target_lang=test_case["target_lang"],
        )
        if result:
            translation_results.append(result)
        await asyncio.sleep(1)  # Small delay between requests

    # Step 3: Verify database persistence
    print("\n" + "=" * 80)
    print("Step 3: Verifying database persistence...")
    print("=" * 80)

    async with db_manager.get_session() as db:
        stored_translations = await verify_database_persistence(db, session_id)

    # Step 4: Verify session statistics
    print("\n" + "=" * 80)
    print("Step 4: Session statistics...")
    print("=" * 80)

    async with db_manager.get_session() as db:
        await verify_session_statistics(db, session_id)

    # Step 5: Validation
    print("\n" + "=" * 80)
    print("Step 5: Test Validation")
    print("=" * 80)

    success_count = len(translation_results)
    expected_count = len(TEST_TRANSLATIONS)
    stored_count = len(stored_translations)

    print("\n✅ Validation Results:")
    print(f"   Expected translations: {expected_count}")
    print(f"   Successful API calls: {success_count}")
    print(f"   Stored in database: {stored_count}")

    # Check if all translations were stored
    if stored_count == expected_count:
        print("\n🎉 SUCCESS: All translations were persisted to database!")
    else:
        print(f"\n⚠️  WARNING: Expected {expected_count} translations but found {stored_count}")

    # Verify data integrity
    print("\n🔍 Data Integrity Checks:")
    for trans in stored_translations:
        has_text = bool(trans.original_text and trans.translated_text)
        has_languages = bool(trans.source_language and trans.target_language)
        has_confidence = trans.confidence is not None
        has_counts = trans.word_count > 0 and trans.character_count > 0

        status = "✅" if all([has_text, has_languages, has_confidence, has_counts]) else "❌"
        print(
            f"   {status} Translation {trans.translation_id}: "
            f"text={has_text}, languages={has_languages}, "
            f"confidence={has_confidence}, counts={has_counts}"
        )

    print("\n" + "=" * 80)
    print("🏁 Integration Test Complete!")
    print("=" * 80)

    # Close database connections
    await db_manager.close()


if __name__ == "__main__":
    asyncio.run(run_integration_test())
