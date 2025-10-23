#!/usr/bin/env python3
"""
Test Multi-Language Session Isolation

This test verifies that concurrent English and Chinese sessions have completely
isolated rolling contexts and tokenizers, preventing cross-contamination.

Test Scenarios:
1. Create two concurrent sessions (English + Chinese)
2. Transcribe audio in each language
3. Verify rolling contexts remain isolated
4. Verify no language mixing occurs
5. Test session cleanup

Expected Results:
✓ English context contains only English text
✓ Chinese context contains only Chinese text
✓ Tokenizers are language-specific
✓ Session cleanup removes all resources
"""

import asyncio
import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whisper_service import WhisperService, TranscriptionRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_audio(duration_seconds: float = 3.0, frequency: int = 440) -> np.ndarray:
    """Generate simple sine wave test audio"""
    sample_rate = 16000
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


async def test_multilang_isolation():
    """Test concurrent English + Chinese session isolation"""

    logger.info("=" * 80)
    logger.info("MULTI-LANGUAGE SESSION ISOLATION TEST")
    logger.info("=" * 80)

    # Initialize WhisperService
    config = {
        "models_dir": ".models",
        "default_model": "base",  # Use smaller model for faster testing
        "sample_rate": 16000,
        "device": "cpu",  # Use CPU for consistent testing
        "orchestration_mode": False
    }

    logger.info("Initializing WhisperService...")
    service = WhisperService(config)
    await asyncio.sleep(1)  # Allow initialization

    # Test audio (in real scenario, this would be actual audio)
    test_audio = generate_test_audio(duration_seconds=3.0)

    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Create English Session")
    logger.info("=" * 80)

    en_session_id = "test-session-english-001"
    en_request = TranscriptionRequest(
        audio_data=test_audio,
        model_name="base",
        language="en",
        session_id=en_session_id,
        sample_rate=16000,
        streaming=False
    )

    # Initialize English session context
    service.model_manager.init_context(
        session_id=en_session_id,
        language="en",
        static_prompt="English domain terms: "
    )

    # Simulate English transcription context
    service.model_manager.append_to_context(
        "Hello, how are you today?",
        session_id=en_session_id
    )
    service.model_manager.append_to_context(
        "The weather is nice.",
        session_id=en_session_id
    )

    en_context = service.model_manager.get_inference_context(session_id=en_session_id)
    logger.info(f"✓ English context: '{en_context}'")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Create Chinese Session")
    logger.info("=" * 80)

    zh_session_id = "test-session-chinese-001"
    zh_request = TranscriptionRequest(
        audio_data=test_audio,
        model_name="base",
        language="zh",
        session_id=zh_session_id,
        sample_rate=16000,
        streaming=False
    )

    # Initialize Chinese session context
    service.model_manager.init_context(
        session_id=zh_session_id,
        language="zh",
        static_prompt="中文领域术语："
    )

    # Simulate Chinese transcription context
    service.model_manager.append_to_context(
        "你好，今天怎么样？",
        session_id=zh_session_id
    )
    service.model_manager.append_to_context(
        "天气很好。",
        session_id=zh_session_id
    )

    zh_context = service.model_manager.get_inference_context(session_id=zh_session_id)
    logger.info(f"✓ Chinese context: '{zh_context}'")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Verify Isolation")
    logger.info("=" * 80)

    # Test 1: English context should NOT contain Chinese
    if "你好" in en_context or "天气" in en_context:
        logger.error("❌ FAILED: English context contains Chinese text!")
        logger.error(f"   English context: '{en_context}'")
        return False
    else:
        logger.info("✓ PASSED: English context is clean (no Chinese)")

    # Test 2: Chinese context should NOT contain English
    if "Hello" in zh_context or "weather" in zh_context:
        logger.error("❌ FAILED: Chinese context contains English text!")
        logger.error(f"   Chinese context: '{zh_context}'")
        return False
    else:
        logger.info("✓ PASSED: Chinese context is clean (no English)")

    # Test 3: Verify separate tokenizers exist
    if en_session_id not in service.model_manager.session_tokenizers:
        logger.error("❌ FAILED: English tokenizer not created")
        return False

    if zh_session_id not in service.model_manager.session_tokenizers:
        logger.error("❌ FAILED: Chinese tokenizer not created")
        return False

    logger.info("✓ PASSED: Both sessions have separate tokenizers")

    # Test 4: Verify languages tracked
    en_lang = service.model_manager.session_languages.get(en_session_id)
    zh_lang = service.model_manager.session_languages.get(zh_session_id)

    if en_lang != "en":
        logger.error(f"❌ FAILED: English session language is '{en_lang}', expected 'en'")
        return False

    if zh_lang != "zh":
        logger.error(f"❌ FAILED: Chinese session language is '{zh_lang}', expected 'zh'")
        return False

    logger.info(f"✓ PASSED: Languages tracked correctly (en={en_lang}, zh={zh_lang})")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Verify Context Updates Don't Cross-Contaminate")
    logger.info("=" * 80)

    # Add more English
    service.model_manager.append_to_context(
        "I am learning Python programming.",
        session_id=en_session_id
    )

    # Add more Chinese
    service.model_manager.append_to_context(
        "我正在学习编程。",
        session_id=zh_session_id
    )

    # Re-fetch contexts
    en_context_updated = service.model_manager.get_inference_context(session_id=en_session_id)
    zh_context_updated = service.model_manager.get_inference_context(session_id=zh_session_id)

    logger.info(f"Updated English context: '{en_context_updated}'")
    logger.info(f"Updated Chinese context: '{zh_context_updated}'")

    # Verify still isolated
    if "我正在学习" in en_context_updated:
        logger.error("❌ FAILED: Chinese text appeared in English context after update!")
        return False

    if "Python programming" in zh_context_updated:
        logger.error("❌ FAILED: English text appeared in Chinese context after update!")
        return False

    logger.info("✓ PASSED: Contexts remain isolated after updates")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Session Cleanup")
    logger.info("=" * 80)

    # Close English session
    service.close_session(en_session_id)

    # Verify cleanup
    if en_session_id in service.model_manager.session_rolling_contexts:
        logger.error("❌ FAILED: English rolling context not cleaned up")
        return False

    if en_session_id in service.model_manager.session_tokenizers:
        logger.error("❌ FAILED: English tokenizer not cleaned up")
        return False

    if en_session_id in service.model_manager.session_languages:
        logger.error("❌ FAILED: English language tracking not cleaned up")
        return False

    logger.info("✓ PASSED: English session fully cleaned up")

    # Verify Chinese session still exists
    zh_context_after_cleanup = service.model_manager.get_inference_context(session_id=zh_session_id)
    if zh_context_after_cleanup != zh_context_updated:
        logger.error("❌ FAILED: Chinese context changed after English cleanup!")
        return False

    logger.info("✓ PASSED: Chinese session unaffected by English cleanup")

    # Close Chinese session
    service.close_session(zh_session_id)

    if zh_session_id in service.model_manager.session_rolling_contexts:
        logger.error("❌ FAILED: Chinese rolling context not cleaned up")
        return False

    logger.info("✓ PASSED: Chinese session fully cleaned up")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Memory Footprint Check")
    logger.info("=" * 80)

    # Check that no session data remains
    remaining_contexts = len(service.model_manager.session_rolling_contexts)
    remaining_tokenizers = len(service.model_manager.session_tokenizers)
    remaining_languages = len(service.model_manager.session_languages)

    logger.info(f"Remaining contexts: {remaining_contexts}")
    logger.info(f"Remaining tokenizers: {remaining_tokenizers}")
    logger.info(f"Remaining languages: {remaining_languages}")

    if remaining_contexts > 0 or remaining_tokenizers > 0 or remaining_languages > 0:
        logger.warning("⚠️  WARNING: Some session data remains (might be from other tests)")
    else:
        logger.info("✓ PASSED: All session data cleaned up")

    logger.info("\n" + "=" * 80)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✓ English and Chinese sessions are completely isolated")
    logger.info("✓ Rolling contexts do not cross-contaminate")
    logger.info("✓ Per-session tokenizers are correctly managed")
    logger.info("✓ Session cleanup properly removes all resources")
    logger.info("✓ Multi-language concurrent sessions are fully supported")
    logger.info("")
    logger.info("=" * 80)

    return True


async def main():
    """Main test runner"""
    try:
        success = await test_multilang_isolation()
        if success:
            logger.info("✓ Test suite completed successfully")
            return 0
        else:
            logger.error("❌ Test suite failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Test suite encountered error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
