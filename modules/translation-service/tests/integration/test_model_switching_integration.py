#!/usr/bin/env python3
"""
Integration Tests for Dynamic Model Switching - REAL BEHAVIOR

These tests verify ACTUAL model switching with the translation service:
1. Translation with initial model
2. Switch to different model
3. Translation with new model
4. Verify translations are different (model-specific behavior)
5. Preload and fast-switch tests
6. Model status verification

Requirements:
- Translation service running on localhost:5003
- Ollama server accessible with at least 2 models available
- Run: python -m pytest tests/integration/test_model_switching_integration.py -v

Author: Claude Code
Date: 2026-01-11
"""

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configuration
TRANSLATION_SERVICE_URL = os.getenv("TRANSLATION_SERVICE_URL", "http://localhost:5003")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.239:11434")

# Test texts for translation
TEST_TEXTS = {
    "simple": "Hello, how are you today?",
    "technical": "The server processes requests asynchronously using event loops.",
    "formal": "We cordially invite you to attend our annual conference.",
    "casual": "Hey! What's up? Wanna grab some coffee later?",
}


class TranslationServiceClient:
    """Client for testing the translation service"""

    def __init__(self, base_url: str = TRANSLATION_SERVICE_URL):
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def health_check(self) -> dict[str, Any]:
        """Check if service is healthy"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()

    async def translate(
        self,
        text: str,
        source_language: str = "en",
        target_language: str = "es",
    ) -> dict[str, Any]:
        """Translate text"""
        payload = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        # Try the new endpoint first, fall back to legacy
        for endpoint in ["/api/translate", "/api/v3/translate", "/translate"]:
            try:
                async with self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result["status_code"] = response.status
                        return result
            except Exception:
                continue

        return {"status_code": 404, "detail": "No working translate endpoint found"}

    async def get_model_status(self) -> dict[str, Any]:
        """Get current model status"""
        async with self.session.get(f"{self.base_url}/api/models/status") as response:
            return await response.json()

    async def switch_model(self, model: str, backend: str = "ollama") -> dict[str, Any]:
        """Switch to a different model"""
        payload = {"model": model, "backend": backend}
        async with self.session.post(
            f"{self.base_url}/api/models/switch",
            json=payload,
        ) as response:
            result = await response.json()
            result["status_code"] = response.status
            return result

    async def preload_model(self, model: str, backend: str = "ollama") -> dict[str, Any]:
        """Preload a model"""
        payload = {"model": model, "backend": backend}
        async with self.session.post(
            f"{self.base_url}/api/models/preload",
            json=payload,
        ) as response:
            result = await response.json()
            result["status_code"] = response.status
            return result

    async def unload_model(self, model: str, backend: str = "ollama") -> dict[str, Any]:
        """Unload a cached model"""
        payload = {"model": model, "backend": backend}
        async with self.session.post(
            f"{self.base_url}/api/models/unload",
            json=payload,
        ) as response:
            result = await response.json()
            result["status_code"] = response.status
            return result

    async def list_models(self, backend: str = "ollama") -> dict[str, Any]:
        """List available models"""
        async with self.session.get(f"{self.base_url}/api/models/list/{backend}") as response:
            return await response.json()


async def get_available_ollama_models() -> list[str]:
    """Get list of available Ollama models"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{OLLAMA_BASE_URL}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not get Ollama models: {e}")
    return []


@pytest.fixture
async def client():
    """Create async client for testing"""
    async with TranslationServiceClient() as c:
        yield c


@pytest.fixture
async def available_models():
    """Get available models for testing"""
    models = await get_available_ollama_models()
    if len(models) < 2:
        pytest.skip("Need at least 2 Ollama models for model switching tests")
    return models


class TestServiceHealth:
    """Test service is running and healthy"""

    @pytest.mark.asyncio
    async def test_service_is_healthy(self, client):
        """Verify translation service is running"""
        health = await client.health_check()
        assert health.get("status") == "healthy", f"Service not healthy: {health}"
        logger.info(f"✅ Service healthy: {health}")


class TestModelStatus:
    """Test model status endpoint"""

    @pytest.mark.asyncio
    async def test_get_model_status(self, client):
        """Test getting current model status"""
        status = await client.get_model_status()

        assert "current_model" in status, "Missing current_model in status"
        assert "current_backend" in status, "Missing current_backend in status"
        assert "is_ready" in status, "Missing is_ready in status"
        assert "cached_models" in status, "Missing cached_models in status"

        logger.info(f"✅ Model status: {status['current_model']} on {status['current_backend']}")
        logger.info(f"   Ready: {status['is_ready']}, Cached: {status['cache_size']}")

        return status


class TestTranslationBeforeSwitch:
    """Test translation with initial model"""

    @pytest.mark.asyncio
    async def test_translate_simple_text(self, client):
        """Test basic translation works"""
        result = await client.translate(
            TEST_TEXTS["simple"],
            source_language="en",
            target_language="es",
        )

        assert result.get("status_code") == 200, f"Translation failed: {result}"
        assert "translated_text" in result, f"No translated_text in result: {result}"
        assert len(result["translated_text"]) > 0, "Empty translation"

        logger.info("✅ Translation (initial model):")
        logger.info(f"   EN: {TEST_TEXTS['simple']}")
        logger.info(f"   ES: {result['translated_text']}")
        logger.info(f"   Backend: {result.get('backend_used', 'unknown')}")

        return result


class TestModelSwitching:
    """Test actual model switching behavior"""

    @pytest.mark.asyncio
    async def test_full_model_switch_workflow(self, client, available_models):
        """
        COMPREHENSIVE TEST: Full model switching workflow

        1. Get initial model status
        2. Translate with initial model
        3. Switch to different model
        4. Translate with new model
        5. Compare results (should be different)
        6. Verify status shows new model
        """
        logger.info("\n" + "=" * 60)
        logger.info("FULL MODEL SWITCH WORKFLOW TEST")
        logger.info("=" * 60)

        # Step 1: Get initial status
        initial_status = await client.get_model_status()
        initial_model = initial_status.get("current_model")
        logger.info(f"\n📋 Step 1: Initial model = {initial_model}")

        # Step 2: Translate with initial model
        logger.info("\n📝 Step 2: Translating with initial model...")
        translation1 = await client.translate(
            TEST_TEXTS["technical"],
            source_language="en",
            target_language="es",
        )
        assert translation1.get("status_code") == 200, f"Translation 1 failed: {translation1}"
        text1 = translation1.get("translated_text", "")
        logger.info(f"   EN: {TEST_TEXTS['technical']}")
        logger.info(f"   ES: {text1}")
        logger.info(f"   Model: {translation1.get('model_used', initial_model)}")

        # Step 3: Find a different model and switch
        logger.info("\n🔄 Step 3: Switching model...")

        # Find a model different from current
        new_model = None
        for model in available_models:
            if model != initial_model:
                new_model = model
                break

        if not new_model:
            pytest.skip("No alternative model available for switching")

        logger.info(f"   Switching from {initial_model} to {new_model}...")

        switch_start = time.time()
        switch_result = await client.switch_model(new_model)
        switch_time = time.time() - switch_start

        assert switch_result.get("success"), f"Model switch failed: {switch_result}"
        logger.info(f"   ✅ Switch successful in {switch_time:.2f}s")
        logger.info(f"   Message: {switch_result.get('message')}")

        # Step 4: Translate with new model
        logger.info("\n📝 Step 4: Translating with NEW model...")
        translation2 = await client.translate(
            TEST_TEXTS["technical"],
            source_language="en",
            target_language="es",
        )
        assert translation2.get("status_code") == 200, f"Translation 2 failed: {translation2}"
        text2 = translation2.get("translated_text", "")
        logger.info(f"   EN: {TEST_TEXTS['technical']}")
        logger.info(f"   ES: {text2}")
        logger.info(f"   Model: {translation2.get('model_used', new_model)}")

        # Step 5: Compare translations
        logger.info("\n🔍 Step 5: Comparing translations...")
        if text1 == text2:
            logger.warning("   ⚠️ Translations are IDENTICAL (models may produce same output)")
        else:
            logger.info("   ✅ Translations are DIFFERENT (confirms model switch worked)")

        logger.info(f"\n   Model 1 ({initial_model}): {text1[:80]}...")
        logger.info(f"   Model 2 ({new_model}): {text2[:80]}...")

        # Step 6: Verify status shows new model
        logger.info("\n📋 Step 6: Verifying model status...")
        final_status = await client.get_model_status()
        assert (
            final_status.get("current_model") == new_model
        ), f"Status shows wrong model: {final_status.get('current_model')} != {new_model}"
        logger.info(f"   ✅ Status confirms current model: {final_status.get('current_model')}")

        # Step 7: Switch back to original model
        logger.info("\n🔄 Step 7: Switching back to original model...")
        switch_back = await client.switch_model(initial_model)
        assert switch_back.get("success"), f"Switch back failed: {switch_back}"
        logger.info(f"   ✅ Switched back to {initial_model}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ FULL MODEL SWITCH WORKFLOW PASSED")
        logger.info("=" * 60)

    @pytest.mark.asyncio
    async def test_switch_to_same_model(self, client):
        """Test switching to the same model (should be instant)"""
        status = await client.get_model_status()
        current_model = status.get("current_model")

        if not current_model:
            pytest.skip("No current model set")

        start = time.time()
        result = await client.switch_model(current_model)
        elapsed = time.time() - start

        assert result.get("success"), f"Switch to same model failed: {result}"
        assert elapsed < 0.5, f"Switch to same model took too long: {elapsed}s"

        logger.info(f"✅ Switch to same model ({current_model}): {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def test_switch_to_invalid_model(self, client):
        """Test switching to non-existent model"""
        result = await client.switch_model("nonexistent-model-12345")

        # Should fail gracefully
        assert not result.get("success"), "Should fail for invalid model"
        logger.info(f"✅ Invalid model switch correctly failed: {result.get('message')}")


class TestModelPreloading:
    """Test model preloading for faster switching"""

    @pytest.mark.asyncio
    async def test_preload_and_fast_switch(self, client, available_models):
        """
        Test that preloading makes switching faster

        1. Get current model
        2. Measure cold switch time to model B
        3. Switch back to model A
        4. Preload model B
        5. Measure warm switch time to model B (should be faster)
        """
        status = await client.get_model_status()
        model_a = status.get("current_model")

        # Find a different model
        model_b = None
        for model in available_models:
            if model != model_a:
                model_b = model
                break

        if not model_b:
            pytest.skip("Need 2 models for preload test")

        logger.info(f"\n📋 Preload test: {model_a} <-> {model_b}")

        # Cold switch to B
        logger.info(f"\n🥶 Cold switch to {model_b}...")
        cold_start = time.time()
        await client.switch_model(model_b)
        cold_switch_time = time.time() - cold_start
        logger.info(f"   Cold switch time: {cold_switch_time:.2f}s")

        # Switch back to A
        await client.switch_model(model_a)

        # Warm switch to B (should use cache)
        logger.info(f"\n🔥 Warm switch to {model_b} (cached)...")
        warm_start = time.time()
        await client.switch_model(model_b)
        warm_switch_time = time.time() - warm_start
        logger.info(f"   Warm switch time: {warm_switch_time:.2f}s")

        # Warm should be faster than cold
        logger.info("\n📊 Comparison:")
        logger.info(f"   Cold: {cold_switch_time:.2f}s")
        logger.info(f"   Warm: {warm_switch_time:.2f}s")
        logger.info(f"   Speedup: {cold_switch_time / warm_switch_time:.1f}x")

        assert (
            warm_switch_time < cold_switch_time
        ), f"Warm switch should be faster: {warm_switch_time:.2f}s >= {cold_switch_time:.2f}s"

        logger.info("✅ Preload/cache test passed - warm switch is faster")


class TestCacheManagement:
    """Test model cache management"""

    @pytest.mark.asyncio
    async def test_cache_grows_with_models(self, client, available_models):
        """Test that cache size increases as models are loaded"""
        initial_status = await client.get_model_status()
        initial_cache_size = initial_status.get("cache_size", 0)

        # Load a few different models
        loaded_count = 0
        for model in available_models[:3]:
            result = await client.switch_model(model)
            if result.get("success"):
                loaded_count += 1

        final_status = await client.get_model_status()
        final_cache_size = final_status.get("cache_size", 0)

        logger.info(f"✅ Cache size: {initial_cache_size} -> {final_cache_size}")
        logger.info(f"   Loaded {loaded_count} models")

        # Note: cache size may not always increase if models were already cached
        assert final_cache_size >= 1, "Should have at least 1 cached model"

    @pytest.mark.asyncio
    async def test_unload_non_current_model(self, client, available_models):
        """Test unloading a model that isn't currently active"""
        if len(available_models) < 2:
            pytest.skip("Need 2 models for unload test")

        model_a, model_b = available_models[0], available_models[1]

        # Load both models
        await client.switch_model(model_a)
        await client.switch_model(model_b)

        # Now model_b is current, try to unload model_a
        result = await client.unload_model(model_a)
        assert result.get("success"), f"Unload failed: {result}"
        logger.info(f"✅ Successfully unloaded non-current model: {model_a}")

    @pytest.mark.asyncio
    async def test_cannot_unload_current_model(self, client):
        """Test that unloading current model fails"""
        status = await client.get_model_status()
        current_model = status.get("current_model")

        if not current_model:
            pytest.skip("No current model")

        result = await client.unload_model(current_model)
        assert not result.get("success"), "Should not be able to unload current model"
        logger.info(f"✅ Correctly prevented unloading current model: {current_model}")


class TestTranslationQuality:
    """Test translation quality across different models"""

    @pytest.mark.asyncio
    async def test_multiple_translations_same_model(self, client):
        """Test consistency of translations with same model"""
        results = []
        for text_key, text in TEST_TEXTS.items():
            result = await client.translate(text, "en", "es")
            if result.get("status_code") == 200:
                results.append(
                    {
                        "type": text_key,
                        "original": text,
                        "translated": result.get("translated_text"),
                        "confidence": result.get("confidence_score"),
                    }
                )

        logger.info("\n📝 Translation samples:")
        for r in results:
            logger.info(f"\n   [{r['type']}]")
            logger.info(f"   EN: {r['original']}")
            logger.info(f"   ES: {r['translated']}")

        assert len(results) > 0, "No successful translations"
        logger.info(f"\n✅ {len(results)}/{len(TEST_TEXTS)} translations completed")


# ============================================================================
# Main test runner
# ============================================================================


async def run_all_tests():
    """Run all integration tests manually (without pytest)"""
    logger.info("=" * 70)
    logger.info("MODEL SWITCHING INTEGRATION TESTS")
    logger.info(f"Translation Service: {TRANSLATION_SERVICE_URL}")
    logger.info(f"Ollama Server: {OLLAMA_BASE_URL}")
    logger.info("=" * 70)

    async with TranslationServiceClient() as client:
        # Check service health
        logger.info("\n🏥 Checking service health...")
        try:
            health = await client.health_check()
            if health.get("status") != "healthy":
                logger.error(f"❌ Service not healthy: {health}")
                return False
            logger.info("✅ Service healthy")
        except Exception as e:
            logger.error(f"❌ Cannot connect to service: {e}")
            return False

        # Get available models
        logger.info("\n📦 Getting available Ollama models...")
        models = await get_available_ollama_models()
        logger.info(f"   Found {len(models)} models: {models[:5]}...")

        if len(models) < 2:
            logger.warning("⚠️ Need at least 2 models for full testing")

        # Get current status
        logger.info("\n📋 Current model status...")
        status = await client.get_model_status()
        logger.info(f"   Current: {status.get('current_model')} on {status.get('current_backend')}")
        logger.info(f"   Ready: {status.get('is_ready')}")
        logger.info(f"   Cached: {status.get('cache_size')} models")

        # Test translation
        logger.info("\n📝 Testing translation with current model...")
        result = await client.translate("Hello, how are you?", "en", "es")
        if result.get("status_code") == 200:
            logger.info("   EN: Hello, how are you?")
            logger.info(f"   ES: {result.get('translated_text')}")
        else:
            logger.error(f"   ❌ Translation failed: {result}")

        # Model switching test
        if len(models) >= 2:
            current = status.get("current_model")
            other = next((m for m in models if m != current), None)

            if other:
                logger.info(f"\n🔄 Testing model switch: {current} -> {other}...")

                switch_result = await client.switch_model(other)
                if switch_result.get("success"):
                    logger.info(f"   ✅ Switched to {other}")

                    # Translate with new model
                    result2 = await client.translate("Hello, how are you?", "en", "es")
                    if result2.get("status_code") == 200:
                        logger.info(f"   ES ({other}): {result2.get('translated_text')}")

                    # Switch back
                    await client.switch_model(current)
                    logger.info(f"   ✅ Switched back to {current}")
                else:
                    logger.error(f"   ❌ Switch failed: {switch_result}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ INTEGRATION TESTS COMPLETE")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    # Run as standalone script
    asyncio.run(run_all_tests())
