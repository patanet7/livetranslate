"""
Integration tests for translation service optimization features.

Tests the complete flow of:
1. Multi-language translation endpoint
2. Translation result caching
3. Context-aware translations
4. Orchestration to translation service integration

These tests require:
- Translation service running on localhost:5003
- Redis running on localhost:6379
- Orchestration service properly configured
"""

import asyncio
import pytest
import time
import httpx


# Configuration
TRANSLATION_SERVICE_URL = "http://localhost:5003"
ORCHESTRATION_SERVICE_URL = "http://localhost:3000"
REDIS_URL = "redis://localhost:6379/1"


class TestMultiLanguageTranslation:
    """Test multi-language translation endpoint"""

    @pytest.mark.asyncio
    async def test_available_models_endpoint(self):
        """Test that /api/models/available endpoint returns model information"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TRANSLATION_SERVICE_URL}/api/models/available", timeout=10.0
            )

            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}"
            )

            data = response.json()

            # Verify response structure
            assert "models" in data
            assert "default" in data
            assert "recommended" in data

            # Verify models list
            assert isinstance(data["models"], list)
            assert len(data["models"]) > 0, "Should have at least one model available"

            # Verify model structure
            for model in data["models"]:
                assert "name" in model
                assert "display_name" in model
                assert "available" in model
                assert "description" in model

            print(f"Available models: {[m['name'] for m in data['models']]}")
            print(f"Recommended model: {data['recommended']}")

    @pytest.mark.asyncio
    async def test_multi_language_endpoint_exists(self):
        """Test that /api/translate/multi endpoint exists"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TRANSLATION_SERVICE_URL}/api/translate/multi",
                json={"text": "Hello", "target_languages": ["es", "fr"]},
            )
            # Should not return 404
            assert response.status_code != 404, "Multi-language endpoint should exist"

    @pytest.mark.asyncio
    async def test_multi_language_with_model_selection(self):
        """Test multi-language translation with specific model selection"""
        async with httpx.AsyncClient() as client:
            # First, get available models
            models_response = await client.get(
                f"{TRANSLATION_SERVICE_URL}/api/models/available", timeout=10.0
            )

            assert models_response.status_code == 200
            models_data = models_response.json()

            available_models = [
                m["name"] for m in models_data["models"] if m["available"]
            ]

            if len(available_models) == 0:
                pytest.skip("No translation models available")

            # Test with each available model
            for model_name in available_models:
                print(f"\nTesting with model: {model_name}")

                response = await client.post(
                    f"{TRANSLATION_SERVICE_URL}/api/translate/multi",
                    json={
                        "text": "Hello world",
                        "source_language": "en",
                        "target_languages": ["es", "fr"],
                        "model": model_name,
                    },
                    timeout=30.0,
                )

                assert response.status_code == 200, (
                    f"Model {model_name} failed with {response.status_code}"
                )

                data = response.json()

                # Verify model was used
                assert "model_requested" in data
                assert data["model_requested"] == model_name

                # Verify translations
                assert "translations" in data
                assert "es" in data["translations"]
                assert "fr" in data["translations"]

                print(f"✓ Model {model_name} successful")

    @pytest.mark.asyncio
    async def test_multi_language_translation_success(self):
        """Test successful multi-language translation"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TRANSLATION_SERVICE_URL}/api/translate/multi",
                json={
                    "text": "Hello world",
                    "source_language": "en",
                    "target_languages": ["es", "fr", "de"],
                },
                timeout=30.0,
            )

            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )

            data = response.json()

            # Verify response structure
            assert "translations" in data
            assert "source_text" in data
            assert data["source_text"] == "Hello world"

            # Verify all requested languages are present
            translations = data["translations"]
            assert "es" in translations, "Spanish translation missing"
            assert "fr" in translations, "French translation missing"
            assert "de" in translations, "German translation missing"

            # Verify each translation has required fields
            for lang, translation_data in translations.items():
                assert "translated_text" in translation_data, (
                    f"{lang} missing translated_text"
                )
                assert "confidence" in translation_data, f"{lang} missing confidence"
                assert len(translation_data["translated_text"]) > 0, (
                    f"{lang} translation is empty"
                )

    @pytest.mark.asyncio
    async def test_multi_language_performance(self):
        """Test that multi-language is faster than sequential"""
        test_text = "This is a test sentence for performance comparison."
        target_languages = ["es", "fr", "de"]

        async with httpx.AsyncClient() as client:
            # Time multi-language endpoint
            start_multi = time.time()
            response_multi = await client.post(
                f"{TRANSLATION_SERVICE_URL}/api/translate/multi",
                json={
                    "text": test_text,
                    "source_language": "en",
                    "target_languages": target_languages,
                },
                timeout=30.0,
            )
            duration_multi = time.time() - start_multi

            assert response_multi.status_code == 200

            # Time sequential translations
            start_sequential = time.time()
            for lang in target_languages:
                response = await client.post(
                    f"{TRANSLATION_SERVICE_URL}/api/translate",
                    json={
                        "text": test_text,
                        "source_language": "en",
                        "target_language": lang,
                    },
                    timeout=30.0,
                )
                assert response.status_code == 200
            duration_sequential = time.time() - start_sequential

            # Multi-language should be faster (or at least not much slower)
            print(
                f"Multi-language: {duration_multi:.2f}s, Sequential: {duration_sequential:.2f}s"
            )
            assert duration_multi < duration_sequential * 1.2, (
                f"Multi-language ({duration_multi:.2f}s) should be faster than sequential ({duration_sequential:.2f}s)"
            )


class TestTranslationCaching:
    """Test translation result caching"""

    @pytest.mark.asyncio
    async def test_cache_reduces_latency(self):
        """Test that cached translations are faster"""
        test_text = "This text will be cached for testing"
        target_lang = "es"

        async with httpx.AsyncClient() as client:
            # First request (cache miss)
            start_first = time.time()
            response_first = await client.post(
                f"{ORCHESTRATION_SERVICE_URL}/api/translation/translate",
                json={
                    "text": test_text,
                    "source_language": "en",
                    "target_language": target_lang,
                },
                timeout=30.0,
            )
            duration_first = time.time() - start_first

            assert response_first.status_code == 200
            first_result = response_first.json()

            # Second request (should be cached)
            start_second = time.time()
            response_second = await client.post(
                f"{ORCHESTRATION_SERVICE_URL}/api/translation/translate",
                json={
                    "text": test_text,
                    "source_language": "en",
                    "target_language": target_lang,
                },
                timeout=30.0,
            )
            duration_second = time.time() - start_second

            assert response_second.status_code == 200
            second_result = response_second.json()

            # Verify same translation
            assert first_result["translated_text"] == second_result["translated_text"]

            # Cached request should be significantly faster
            print(
                f"First request: {duration_first:.3f}s, Second (cached): {duration_second:.3f}s"
            )
            assert duration_second < duration_first * 0.5, (
                f"Cached request ({duration_second:.3f}s) should be at least 50% faster than first ({duration_first:.3f}s)"
            )

    @pytest.mark.asyncio
    async def test_cache_statistics_endpoint(self):
        """Test that cache statistics are exposed"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ORCHESTRATION_SERVICE_URL}/api/translation/cache/stats", timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()

                # Verify stats structure
                assert "hits" in data or "hit_count" in data
                assert "misses" in data or "miss_count" in data
                assert "hit_rate" in data or "cache_hit_rate" in data
            else:
                pytest.skip("Cache statistics endpoint not yet implemented")


class TestOrchestrationIntegration:
    """Test orchestration service integration with optimized translation"""

    @pytest.mark.asyncio
    async def test_audio_coordinator_uses_multi_language(self):
        """Test that AudioCoordinator uses multi-language translation"""
        # This test requires a full audio processing pipeline
        # We'll test by checking the translation client behavior

        from clients.translation_service_client import TranslationServiceClient

        client = TranslationServiceClient(base_url=TRANSLATION_SERVICE_URL)

        # Test translate_to_multiple_languages method exists
        assert hasattr(client, "translate_to_multiple_languages"), (
            "TranslationServiceClient should have translate_to_multiple_languages method"
        )

        # Test the method works
        result = await client.translate_to_multiple_languages(
            text="Hello world",
            source_language="en",
            target_languages=["es", "fr", "de"],
        )

        assert isinstance(result, dict)
        assert "es" in result
        assert "fr" in result
        assert "de" in result

    @pytest.mark.asyncio
    async def test_audio_coordinator_cache_integration(self):
        """Test that AudioCoordinator integrates with translation cache"""
        # This requires the full AudioCoordinator with cache enabled
        from audio.audio_coordinator import create_audio_coordinator
        from clients.audio_service_client import AudioServiceClient
        from clients.translation_service_client import TranslationServiceClient

        # Create coordinator with cache enabled
        service_urls = {
            "whisper_service": "http://localhost:5001",
            "translation_service": TRANSLATION_SERVICE_URL,
        }

        audio_client = AudioServiceClient(base_url=service_urls["whisper_service"])
        translation_client = TranslationServiceClient(
            base_url=service_urls["translation_service"]
        )

        coordinator = create_audio_coordinator(
            database_url=None,  # Optional for test
            service_urls=service_urls,
            audio_client=audio_client,
            translation_client=translation_client,
        )

        # Check if cache is initialized
        if hasattr(coordinator, "translation_cache"):
            assert coordinator.translation_cache is not None, (
                "Translation cache should be initialized"
            )

            # Test cache stats
            stats = coordinator.translation_cache.get_stats()
            assert "hits" in stats or "hit_count" in stats
        else:
            pytest.skip("Translation cache not yet integrated into AudioCoordinator")


class TestEndToEndOptimization:
    """End-to-end tests for complete optimization pipeline"""

    @pytest.mark.asyncio
    async def test_complete_audio_translation_pipeline(self):
        """Test complete pipeline: audio → transcription → multi-language translation"""
        # This test requires all services running

        async with httpx.AsyncClient() as client:
            # Check all services are healthy
            whisper_health = await client.get(
                "http://localhost:5001/health", timeout=5.0
            )
            translation_health = await client.get(
                f"{TRANSLATION_SERVICE_URL}/api/health", timeout=5.0
            )
            orchestration_health = await client.get(
                f"{ORCHESTRATION_SERVICE_URL}/api/health", timeout=5.0
            )

            if whisper_health.status_code != 200:
                pytest.skip("Whisper service not available")
            if translation_health.status_code != 200:
                pytest.skip("Translation service not available")
            if orchestration_health.status_code != 200:
                pytest.skip("Orchestration service not available")

            # TODO: Complete end-to-end test with actual audio processing
            # This would involve:
            # 1. Upload audio file
            # 2. Wait for transcription
            # 3. Verify translations in multiple languages
            # 4. Verify cache was used for duplicate phrases
            # 5. Verify performance metrics

            pytest.skip("Full pipeline test requires audio file setup")

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test performance with multiple concurrent translation requests"""
        test_texts = [
            "Hello world",
            "How are you?",
            "Thank you very much",
            "Good morning",
            "Hello world",  # Duplicate for cache testing
        ]
        target_languages = ["es", "fr", "de"]

        async with httpx.AsyncClient() as client:
            start_time = time.time()

            # Send all requests concurrently
            tasks = []
            for text in test_texts:
                task = client.post(
                    f"{TRANSLATION_SERVICE_URL}/api/translate/multi",
                    json={
                        "text": text,
                        "source_language": "en",
                        "target_languages": target_languages,
                    },
                    timeout=30.0,
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time

            # Count successful responses
            successful = sum(
                1
                for r in responses
                if not isinstance(r, Exception) and r.status_code == 200
            )

            print(f"Processed {successful}/{len(test_texts)} texts in {duration:.2f}s")
            print(f"Average: {duration / len(test_texts):.3f}s per text")

            assert successful == len(test_texts), (
                f"Expected all requests to succeed, got {successful}/{len(test_texts)}"
            )

            # Performance assertion: should handle at least 2 texts per second
            assert duration < len(test_texts) * 0.5, (
                f"Should process at least 2 texts/second, took {duration:.2f}s for {len(test_texts)} texts"
            )


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
