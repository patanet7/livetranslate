#!/usr/bin/env python3
"""
Test Phase 3C: Stability Tracking and Draft/Final Emission

This script tests the word-based stability detection implementation.
"""

import asyncio
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whisper_service import WhisperService, TranscriptionRequest

async def test_stability_tracking():
    """Test Phase 3C stability tracking with simulated streaming"""

    print("=" * 80)
    print("Phase 3C: Stability Tracking Test")
    print("=" * 80)

    # Initialize service
    print("\n[1] Initializing Whisper Service...")
    service = WhisperService()

    # Create a simple audio request (1 second of silence for testing)
    print("\n[2] Creating test audio stream...")
    sample_rate = 16000
    duration = 1.0
    audio_chunk = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Add some noise so it's not pure silence
    audio_chunk += np.random.normal(0, 0.01, audio_chunk.shape).astype(np.float32)

    # Create transcription request
    request = TranscriptionRequest(
        audio_data=audio_chunk,
        model_name="whisper-base",
        language="en",
        session_id="test-stability-session",
        sample_rate=sample_rate,
        streaming=True,
        enable_vad=False  # Disable VAD for testing
    )

    print("\n[3] Starting streaming transcription...")
    print("-" * 80)

    emission_count = 0
    draft_count = 0
    final_count = 0

    try:
        # Start streaming
        async for result in service.transcribe_stream(request):
            emission_count += 1

            print(f"\n[Emission #{emission_count}]")
            print(f"  Text: '{result.text[:80]}...' ({len(result.text)} chars)")
            print(f"  Stable: '{result.stable_text[:50]}...' ({len(result.stable_text)} chars)")
            print(f"  Unstable: '{result.unstable_text[:30]}...' ({len(result.unstable_text)} chars)")
            print(f"  Is Draft: {result.is_draft}")
            print(f"  Is Final: {result.is_final}")
            print(f"  Is Forced: {result.is_forced}")
            print(f"  Should Translate: {result.should_translate}")
            print(f"  Translation Mode: {result.translation_mode}")
            print(f"  Stability Score: {result.stability_score:.3f}")
            print(f"  Confidence: {result.confidence_score:.3f}")

            if result.is_draft:
                draft_count += 1
            if result.is_final:
                final_count += 1

            # Stop after 5 emissions for testing
            if emission_count >= 5:
                print("\n[Test Limit] Reached 5 emissions, stopping...")
                break

    except Exception as e:
        print(f"\n[ERROR] Streaming error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total Emissions: {emission_count}")
    print(f"Draft Emissions: {draft_count}")
    print(f"Final Emissions: {final_count}")
    print("\n✅ Phase 3C: Stability tracking test completed!")
    print("=" * 80)

async def test_helper_methods():
    """Test the helper methods for stability detection"""

    print("\n" + "=" * 80)
    print("Testing Helper Methods")
    print("=" * 80)

    service = WhisperService()

    # Test _find_stable_word_prefix
    print("\n[1] Testing _find_stable_word_prefix...")

    text_history = [
        ("hello world", 1.0),
        ("hello world this", 2.0),
        ("hello world this is", 3.0),
        ("hello world this is a", 4.0),
    ]
    current_text = "hello world this is a test"

    stable_prefix = service._find_stable_word_prefix(text_history, current_text)
    print(f"  Text history: {[txt for txt, _ in text_history]}")
    print(f"  Current text: '{current_text}'")
    print(f"  Stable prefix: '{stable_prefix}'")
    print(f"  Expected: 'hello world this is' (4 words)")

    # Test _calculate_text_stability_score
    print("\n[2] Testing _calculate_text_stability_score...")

    score = service._calculate_text_stability_score(text_history, stable_prefix)
    print(f"  Stability score: {score:.3f}")
    print(f"  Expected: 0.6-0.9 (high consistency)")

    # Test with empty history
    print("\n[3] Testing edge cases...")

    empty_history = []
    empty_prefix = service._find_stable_word_prefix(empty_history, current_text)
    print(f"  Empty history prefix: '{empty_prefix}' (expected: '')")

    single_history = [("hello", 1.0)]
    single_prefix = service._find_stable_word_prefix(single_history, "hello world")
    print(f"  Single history prefix: '{single_prefix}' (expected: '')")

    print("\n✅ Helper methods test completed!")
    print("=" * 80)

async def main():
    """Run all tests"""

    print("\n" + "=" * 80)
    print("PHASE 3C: STABILITY TRACKING - COMPREHENSIVE TEST")
    print("=" * 80)

    # Test 1: Helper methods
    await test_helper_methods()

    # Test 2: Full streaming with stability
    await test_stability_tracking()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
