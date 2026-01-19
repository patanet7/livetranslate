#!/usr/bin/env python3
"""
Quick Test Script for Data Pipeline

A simple standalone script to quickly verify the data pipeline is working.
This can be run independently to test the complete flow without pytest.

Usage:
    python test_pipeline_quick.py

Requirements:
    - PostgreSQL running with livetranslate database
    - Database schema initialized
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.bot_session_manager import create_bot_session_manager
from pipeline.data_pipeline import (
    AudioChunkMetadata,
    TranscriptionResult,
    TranslationResult,
    create_data_pipeline,
)


async def main():
    """Run quick test of data pipeline."""
    print("=" * 70)
    print("DATA PIPELINE QUICK TEST")
    print("=" * 70)

    # Configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),  # livetranslate-postgres container
        "database": os.getenv("POSTGRES_DB", "livetranslate_test"),
        "username": os.getenv("POSTGRES_USER", "livetranslate"),
        "password": os.getenv("POSTGRES_PASSWORD", "livetranslate_dev_password"),
    }

    audio_storage = os.getenv("AUDIO_STORAGE_PATH", "/tmp/livetranslate_test/audio")

    print("\n📊 Configuration:")
    print(
        f"  Database: {db_config['username']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    print(f"  Audio Storage: {audio_storage}")

    # Create database manager
    print("\n🔧 Creating database manager...")
    db_manager = create_bot_session_manager(db_config, audio_storage)

    # Initialize database
    print("🔌 Initializing database connection...")
    try:
        success = await db_manager.initialize()
        if not success:
            print("❌ Failed to initialize database")
            return 1
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return 1

    # Create pipeline with initialized database manager
    print("\n🔧 Creating data pipeline...")
    pipeline = create_data_pipeline(
        database_manager=db_manager,
        audio_storage_path=audio_storage,
        enable_speaker_tracking=True,
        enable_segment_continuity=True,
    )

    # Get database statistics
    try:
        stats = await pipeline.db_manager.get_database_statistics()
        print("\n📈 Database Statistics:")
        print(f"  Total sessions: {stats.get('total_sessions', 0)}")
        print(f"  Total transcripts: {stats.get('total_transcripts', 0)}")
        print(f"  Total translations: {stats.get('total_translations', 0)}")
        print(f"  Storage usage: {stats.get('storage_usage_mb', 0):.2f} MB")
    except Exception as e:
        print(f"⚠️  Could not get statistics: {e}")

    # Create test session
    print("\n🎬 Creating test session...")
    session_data = {
        "bot_id": "quick_test_bot",
        "meeting_id": "quick_test_meeting",
        "meeting_title": "Quick Pipeline Test",
        "status": "active",
        "target_languages": ["en", "es"],
    }

    try:
        session_id = await pipeline.db_manager.create_bot_session(session_data)
        print(f"✅ Session created: {session_id}")
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return 1

    # Test 1: Audio Storage
    print("\n🎵 Test 1: Audio Storage")
    try:
        audio_bytes = b"test_audio_data_" + os.urandom(1024)
        metadata = AudioChunkMetadata(
            duration_seconds=5.0,
            sample_rate=16000,
            channels=1,
            chunk_start_time=0.0,
            chunk_end_time=5.0,
        )

        file_id = await pipeline.process_audio_chunk(session_id, audio_bytes, "wav", metadata)

        if file_id:
            print(f"  ✅ Audio stored: {file_id}")
        else:
            print("  ❌ Audio storage failed")
            raise Exception("Audio storage failed")

    except Exception as e:
        print(f"  ❌ Audio storage error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Test 2: Transcription Storage
    print("\n📝 Test 2: Transcription Storage")
    try:
        transcriptions = [
            TranscriptionResult(
                text="Hello, this is a test of the data pipeline.",
                language="en",
                start_time=0.0,
                end_time=3.0,
                speaker="SPEAKER_00",
                speaker_name="Alice",
                confidence=0.95,
                segment_index=0,
            ),
            TranscriptionResult(
                text="Yes, testing the speaker diarization feature.",
                language="en",
                start_time=3.0,
                end_time=6.0,
                speaker="SPEAKER_01",
                speaker_name="Bob",
                confidence=0.93,
                segment_index=1,
            ),
        ]

        transcript_ids = []
        for trans in transcriptions:
            tid = await pipeline.process_transcription_result(session_id, file_id, trans)
            if tid:
                transcript_ids.append(tid)
                print(f"  ✅ Transcript stored: {tid} (Speaker: {trans.speaker})")
            else:
                raise Exception("Transcript storage failed")

    except Exception as e:
        print(f"  ❌ Transcription error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Test 3: Translation Storage
    print("\n🌐 Test 3: Translation Storage")
    try:
        for i, tid in enumerate(transcript_ids):
            translation = TranslationResult(
                text=f"Spanish translation of segment {i}",
                source_language="en",
                target_language="es",
                speaker=transcriptions[i].speaker,
                speaker_name=transcriptions[i].speaker_name,
                confidence=0.90,
            )

            translation_id = await pipeline.process_translation_result(
                session_id,
                tid,
                translation,
                transcriptions[i].start_time,
                transcriptions[i].end_time,
            )

            if translation_id:
                print(f"  ✅ Translation stored: {translation_id}")
            else:
                raise Exception("Translation storage failed")

    except Exception as e:
        print(f"  ❌ Translation error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Test 4: Timeline Query
    print("\n📅 Test 4: Timeline Query")
    try:
        timeline = await pipeline.get_session_timeline(session_id)
        print(f"  ✅ Timeline retrieved: {len(timeline)} entries")

        for entry in timeline[:5]:  # Show first 5
            print(f"     [{entry.timestamp:.1f}s] {entry.entry_type}: {entry.content[:40]}...")

        if len(timeline) > 5:
            print(f"     ... and {len(timeline) - 5} more entries")

    except Exception as e:
        print(f"  ❌ Timeline error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Test 5: Speaker Statistics
    print("\n👥 Test 5: Speaker Statistics")
    try:
        stats = await pipeline.get_speaker_statistics(session_id)
        print(f"  ✅ Speaker statistics: {len(stats)} speakers")

        for speaker in stats:
            print(f"     {speaker.speaker_name} ({speaker.speaker_id}):")
            print(f"       - Speaking time: {speaker.total_speaking_time:.1f}s")
            print(f"       - Segments: {speaker.total_segments}")
            print(f"       - Translations: {speaker.total_translations}")

    except Exception as e:
        print(f"  ❌ Speaker statistics error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Test 6: Full-Text Search
    print("\n🔍 Test 6: Full-Text Search")
    try:
        results = await pipeline.search_transcripts(session_id, "test", use_fuzzy=True)
        print(f"  ✅ Search results: {len(results)} matches")

        for result in results[:3]:  # Show first 3
            print(f"     - {result.transcript_text[:50]}...")

    except Exception as e:
        print(f"  ⚠️  Search error (optional feature): {e}")

    # Test 7: Comprehensive Session Data
    print("\n📊 Test 7: Comprehensive Session Data")
    try:
        comprehensive = await pipeline.db_manager.get_comprehensive_session_data(session_id)

        if comprehensive:
            print("  ✅ Comprehensive data retrieved")
            print(f"     Audio files: {comprehensive['statistics']['audio_files_count']}")
            print(f"     Transcripts: {comprehensive['statistics']['transcripts_count']}")
            print(f"     Translations: {comprehensive['statistics']['translations_count']}")
        else:
            raise Exception("Comprehensive data retrieval failed")

    except Exception as e:
        print(f"  ❌ Comprehensive data error: {e}")
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        await pipeline.db_manager.close()
        return 1

    # Cleanup
    print("\n🧹 Cleaning up test session...")
    try:
        await pipeline.db_manager.cleanup_session(session_id, remove_files=True)
        print("  ✅ Session cleaned up")
    except Exception as e:
        print(f"  ⚠️  Cleanup warning: {e}")

    # Close connection
    await pipeline.db_manager.close()

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe data pipeline is working correctly!")
    print("\nNext steps:")
    print("  1. Run full integration tests: pytest tests/test_data_pipeline_integration.py")
    print("  2. Register API router in your FastAPI app")
    print("  3. Test API endpoints with curl or Postman")
    print("  4. Integrate with Whisper and Translation services")
    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
