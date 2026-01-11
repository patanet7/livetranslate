"""
Integration tests for Translation Persistence

Tests the complete flow:
1. Session creation in database
2. Translation requests with session ID
3. Database persistence verification
4. Multi-language storage

Requirements:
- Orchestration service running on localhost:3000
- Translation service running on localhost:5003
- PostgreSQL running with livetranslate database
"""

import pytest
import uuid
import httpx
import psycopg2


# Test configuration
ORCHESTRATION_URL = "http://localhost:3000"
DATABASE_URL = "postgresql://localhost:5432/livetranslate"


class TestTranslationPersistence:
    """Test translation persistence to database"""

    @pytest.fixture
    def db_connection(self):
        """Provide database connection"""
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        yield conn
        conn.close()

    @pytest.fixture
    def test_session_id(self, db_connection):
        """Create a test bot session and return its ID"""
        cursor = db_connection.cursor()

        session_id = uuid.uuid4()
        cursor.execute(
            """
            INSERT INTO bot_sessions (
                session_id, bot_id, meeting_id, meeting_title,
                bot_type, status, target_languages, enable_translation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                f"test-bot-{uuid.uuid4().hex[:8]}",
                f"test-meeting-{uuid.uuid4().hex[:8]}",
                "Translation Persistence Test Session",
                "test",
                "running",
                ["es", "fr", "de", "zh"],
                True,
            ),
        )
        db_connection.commit()

        yield str(session_id)

        # Cleanup: Delete test session (cascade will delete translations)
        cursor.execute("DELETE FROM bot_sessions WHERE session_id = %s", (session_id,))
        db_connection.commit()
        cursor.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_translation_without_session(self):
        """Test translation works without session ID (no database persistence)"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATION_URL}/api/translation/",
                json={
                    "text": "Hello world",
                    "target_language": "es",
                    "source_language": "en",
                },
            )

            assert response.status_code == 200
            result = response.json()

            assert "translated_text" in result
            assert result["target_language"] == "es"
            assert "model_used" in result
            assert "backend_used" in result
            print(f"✅ Translation without session: {result['translated_text']}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_translation_with_session_persistence(
        self, test_session_id, db_connection
    ):
        """Test translation with session ID persists to database"""
        test_text = "Good morning, how are you today?"
        target_lang = "es"

        # Step 1: Make translation request with session ID
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATION_URL}/api/translation/",
                json={
                    "text": test_text,
                    "target_language": target_lang,
                    "source_language": "en",
                    "session_id": test_session_id,
                },
            )

            assert response.status_code == 200
            result = response.json()

            assert "translated_text" in result
            assert result["target_language"] == target_lang
            print(f"✅ Translation API response: {result['translated_text']}")

        # Step 2: Verify persistence in database
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT translation_id, original_text, translated_text,
                   source_language, target_language, confidence,
                   word_count, character_count, session_metadata
            FROM translations
            WHERE session_id = %s
            ORDER BY start_time DESC
            LIMIT 1
            """,
            (uuid.UUID(test_session_id),),
        )

        row = cursor.fetchone()
        cursor.close()

        # Verify translation was persisted
        assert row is not None, "Translation was not persisted to database!"

        (
            translation_id,
            original,
            translated,
            source_lang,
            target_lang_db,
            confidence,
            word_count,
            char_count,
            metadata,
        ) = row

        assert original == test_text
        assert source_lang == "en"
        assert target_lang_db == target_lang
        assert word_count == len(test_text.split())
        assert char_count == len(test_text)
        assert confidence is not None

        print("✅ Database verification passed:")
        print(f"   ID: {translation_id}")
        print(f"   Original: {original}")
        print(f"   Translated: {translated}")
        print(f"   Confidence: {confidence}")
        print(f"   Metadata: {metadata}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_language_translations(self, test_session_id, db_connection):
        """Test multiple translations in different languages persist correctly"""
        test_cases = [
            {"text": "Hello, welcome!", "target": "es", "lang_name": "Spanish"},
            {"text": "Thank you very much", "target": "fr", "lang_name": "French"},
            {"text": "Good evening", "target": "de", "lang_name": "German"},
        ]

        # Step 1: Make multiple translation requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            for test_case in test_cases:
                response = await client.post(
                    f"{ORCHESTRATION_URL}/api/translation/",
                    json={
                        "text": test_case["text"],
                        "target_language": test_case["target"],
                        "source_language": "en",
                        "session_id": test_session_id,
                    },
                )

                assert response.status_code == 200
                result = response.json()
                print(f"✅ {test_case['lang_name']}: {result['translated_text']}")

        # Step 2: Verify all translations were persisted
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT target_language) as unique_languages
            FROM translations
            WHERE session_id = %s
            """,
            (uuid.UUID(test_session_id),),
        )

        count, unique_languages = cursor.fetchone()

        assert count == len(test_cases), (
            f"Expected {len(test_cases)} translations, found {count}"
        )
        assert unique_languages == len(test_cases), (
            f"Expected {len(test_cases)} languages, found {unique_languages}"
        )

        # Get details of all translations
        cursor.execute(
            """
            SELECT target_language, original_text, translated_text, confidence
            FROM translations
            WHERE session_id = %s
            ORDER BY start_time
            """,
            (uuid.UUID(test_session_id),),
        )

        translations = cursor.fetchall()
        cursor.close()

        print(f"\n✅ Stored {count} translations in {unique_languages} languages:")
        for target_lang, original, translated, confidence in translations:
            print(
                f"   [{target_lang}] {original[:30]}... → {translated[:50]}... (conf: {confidence})"
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_statistics(self, test_session_id, db_connection):
        """Test that session statistics can be queried correctly"""
        # Create some translations
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(3):
                await client.post(
                    f"{ORCHESTRATION_URL}/api/translation/",
                    json={
                        "text": f"Test sentence number {i + 1}",
                        "target_language": "es",
                        "source_language": "en",
                        "session_id": test_session_id,
                    },
                )

        # Query session statistics
        cursor = db_connection.cursor()
        cursor.execute(
            """
            SELECT
                session_id,
                COUNT(*) as translation_count,
                SUM(word_count) as total_words,
                SUM(character_count) as total_chars,
                AVG(confidence) as avg_confidence,
                ARRAY_AGG(DISTINCT target_language) as languages
            FROM translations
            WHERE session_id = %s
            GROUP BY session_id
            """,
            (uuid.UUID(test_session_id),),
        )

        row = cursor.fetchone()
        cursor.close()

        assert row is not None
        session_id, count, total_words, total_chars, avg_conf, languages = row

        assert count == 3
        assert total_words > 0
        assert total_chars > 0
        assert avg_conf is not None
        assert "es" in languages

        print("\n✅ Session Statistics:")
        print(f"   Translations: {count}")
        print(f"   Total words: {total_words}")
        print(f"   Total characters: {total_chars}")
        print(f"   Avg confidence: {avg_conf:.2f}")
        print(f"   Languages: {languages}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_invalid_session_id_graceful_handling(self, db_connection):
        """Test that invalid session ID doesn't break the API"""
        invalid_session_id = str(uuid.uuid4())  # Non-existent session

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ORCHESTRATION_URL}/api/translation/",
                json={
                    "text": "Test with invalid session",
                    "target_language": "es",
                    "source_language": "en",
                    "session_id": invalid_session_id,
                },
            )

            # Should still succeed (just won't persist to DB)
            assert response.status_code == 200
            result = response.json()
            assert "translated_text" in result
            print(
                f"✅ API handles invalid session gracefully: {result['translated_text']}"
            )

        # Verify nothing was persisted for invalid session
        cursor = db_connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM translations WHERE session_id = %s",
            (uuid.UUID(invalid_session_id),),
        )
        count = cursor.fetchone()[0]
        cursor.close()

        assert count == 0, "Translation should not persist for non-existent session"
        print("✅ No data persisted for invalid session (as expected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
