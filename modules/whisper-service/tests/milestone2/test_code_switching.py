#!/usr/bin/env python3
"""
Milestone 2: Session-Restart Code-Switching Tests

Per FEEDBACK.md lines 171-184:
- Test frame-level LID at 80-120ms hop
- Test sustained detection with hysteresis (P(new)-P(old) > 0.2, ≥6 frames, 250ms dwell)
- Test session switching at VAD boundaries
- Test end-to-end code-switching transcription
- Expected accuracy: 70-85% for inter-sentence code-switching

Test Strategy:
1. Unit tests for LID components
2. Integration tests for session switching
3. End-to-end tests with simulated code-switching audio
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import logging
import time

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from language_id import FrameLevelLID, SustainedLanguageDetector, LIDSmoother
from session_restart import SessionRestartTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_frame_level_lid():
    """
    Test 1: Frame-Level LID Basic Functionality

    Per FEEDBACK.md lines 32-38:
    - 80-120ms hop (we use 100ms = 10Hz)
    - Fast inference for real-time processing
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Frame-Level LID Basic Functionality")
    logger.info("="*80)

    # Create LID detector
    lid = FrameLevelLID(
        hop_ms=100,
        sample_rate=16000,
        target_languages=['en', 'zh'],
        smoothing=True
    )

    # Generate test audio (100ms = 1600 samples at 16kHz)
    audio_frame = np.random.randn(1600).astype(np.float32) * 0.1

    # Run detection
    result = lid.detect(audio_frame, timestamp=0.1, model=None)

    # Verify result
    assert result.language in ['en', 'zh'], f"Invalid language: {result.language}"
    assert 0.0 <= result.confidence <= 1.0, f"Invalid confidence: {result.confidence}"
    assert len(result.probabilities) == 2, f"Expected 2 language probs, got {len(result.probabilities)}"
    assert sum(result.probabilities.values()) > 0, "Probabilities should sum to > 0"

    logger.info(f"✅ LID Detection: language={result.language}, confidence={result.confidence:.3f}")
    logger.info(f"   Probabilities: {result.probabilities}")
    logger.info("✅ TEST 1 PASSED")

    return True


def test_lid_smoothing():
    """
    Test 2: LID Smoothing with HMM/Viterbi

    Per FEEDBACK.md line 37: "Smooth with Viterbi or hysteresis"
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 2: LID Smoothing with HMM/Viterbi")
    logger.info("="*80)

    # Create smoother
    smoother = LIDSmoother(
        languages=['en', 'zh'],
        transition_cost=0.3,
        window_size=5
    )

    # Simulate noisy LID sequence with language flapping
    # Real scenario: en, en, zh (noise), en, en (should smooth to all 'en')
    lid_sequences = [
        {'en': 0.6, 'zh': 0.4},  # en
        {'en': 0.7, 'zh': 0.3},  # en
        {'en': 0.4, 'zh': 0.6},  # zh (noise/false positive)
        {'en': 0.7, 'zh': 0.3},  # en
        {'en': 0.8, 'zh': 0.2},  # en
    ]

    results = []
    for i, probs in enumerate(lid_sequences):
        result = smoother.smooth(probs, timestamp=i * 0.1)
        results.append(result.language)
        logger.info(
            f"Frame {i}: raw={max(probs, key=probs.get)}, "
            f"smoothed={result.language}, "
            f"transition_cost={result.transition_cost:.3f}"
        )

    # Verify smoothing reduced flapping
    # The noisy 'zh' at frame 2 should be smoothed to 'en'
    logger.info(f"Smoothed sequence: {results}")

    # Count transitions
    transitions = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
    logger.info(f"Total transitions: {transitions} (lower is better)")

    # Verify Viterbi reduced flapping compared to raw
    raw_langs = [max(p, key=p.get) for p in lid_sequences]
    raw_transitions = sum(1 for i in range(1, len(raw_langs)) if raw_langs[i] != raw_langs[i-1])
    logger.info(f"Raw transitions: {raw_transitions}")

    assert transitions <= raw_transitions, "Smoothing should reduce or maintain transitions"

    logger.info("✅ TEST 2 PASSED")
    return True


def test_sustained_detection():
    """
    Test 3: Sustained Language Detection with Hysteresis

    Per FEEDBACK.md lines 157-167:
    - Switch only if P(new) - P(old) > 0.2 for ≥6 consecutive frames
    - Minimum dwell: 250ms (= 2.5 frames at 100ms hop, but we require 6 frames = 600ms)
    - Hard stop at VAD boundary
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Sustained Language Detection with Hysteresis")
    logger.info("="*80)

    # Create sustained detector
    detector = SustainedLanguageDetector(
        confidence_margin=0.2,
        min_dwell_frames=6,
        min_dwell_ms=250.0,
        frame_hop_ms=100.0
    )

    # Test Case 3a: Insufficient confidence margin (should NOT switch)
    logger.info("\n--- Test 3a: Insufficient Margin (P(new)-P(old) = 0.15 < 0.2) ---")
    detector.reset()

    for i in range(10):
        # English slightly higher, but margin too small
        result = detector.update({'en': 0.55, 'zh': 0.45}, timestamp=i * 0.1)
        if result:
            logger.info(f"❌ Frame {i}: Unexpected switch to {result.to_language}")
            assert False, "Should NOT switch with insufficient margin"

    logger.info("✅ Test 3a PASSED: No premature switch with insufficient margin")

    # Test Case 3b: Insufficient frames (should NOT switch)
    logger.info("\n--- Test 3b: Insufficient Frames (only 3 frames < 6) ---")
    detector.reset()

    for i in range(3):
        # Good margin but only 3 frames
        result = detector.update({'en': 0.3, 'zh': 0.7}, timestamp=i * 0.1)
        if result:
            logger.info(f"❌ Frame {i}: Unexpected switch to {result.to_language}")
            assert False, "Should NOT switch with insufficient frames"

    logger.info("✅ Test 3b PASSED: No premature switch with insufficient frames")

    # Test Case 3c: Sustained change (SHOULD switch)
    logger.info("\n--- Test 3c: Sustained Change (margin=0.4, 8 frames ≥ 6) ---")
    detector.reset()

    # Initialize with English
    detector.update({'en': 0.9, 'zh': 0.1}, timestamp=0.0)

    switch_detected = False
    for i in range(1, 10):
        # Sustained Chinese with strong margin
        result = detector.update({'en': 0.2, 'zh': 0.8}, timestamp=i * 0.1)
        if result:
            logger.info(
                f"✅ Frame {i}: Switch detected! {result.from_language} → {result.to_language} "
                f"(margin={result.confidence_margin:.3f}, frames={result.dwell_frames}, "
                f"duration={result.dwell_duration_ms:.0f}ms)"
            )
            switch_detected = True
            break

    assert switch_detected, "Should detect sustained language change"
    logger.info("✅ Test 3c PASSED: Sustained change detected correctly")

    # Test Case 3d: Language returns before switch completes (should NOT switch)
    logger.info("\n--- Test 3d: Language Returns Before Switch (false positive prevention) ---")
    detector.reset()

    # Initialize with English
    detector.update({'en': 0.9, 'zh': 0.1}, timestamp=0.0)

    # 4 frames of Chinese (not enough)
    for i in range(1, 5):
        result = detector.update({'en': 0.2, 'zh': 0.8}, timestamp=i * 0.1)
        if result:
            logger.info(f"❌ Frame {i}: Unexpected switch")
            assert False, "Should NOT switch before 6 frames"

    # Then return to English (reset candidate)
    for i in range(5, 8):
        result = detector.update({'en': 0.9, 'zh': 0.1}, timestamp=i * 0.1)
        if result:
            logger.info(f"❌ Frame {i}: Unexpected switch after return to original language")
            assert False, "Should NOT switch after language returns"

    logger.info("✅ Test 3d PASSED: False positive prevented by hysteresis")

    logger.info("\n✅ TEST 3 PASSED: All sustained detection cases work correctly")
    return True


def test_session_switching_simulation():
    """
    Test 4: Session Switching Simulation

    Simulates session-restart code-switching without real Whisper models.
    Tests the logic of detecting switches and managing sessions.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Session Switching Simulation")
    logger.info("="*80)

    # Note: This test would require a full Whisper model setup
    # For now, we'll test the component integration logic

    # Create components
    lid = FrameLevelLID(hop_ms=100, sample_rate=16000, target_languages=['en', 'zh'])
    smoother = LIDSmoother(languages=['en', 'zh'], transition_cost=0.3)
    detector = SustainedLanguageDetector(confidence_margin=0.2, min_dwell_frames=6, min_dwell_ms=250.0)

    # Simulate audio stream with code-switching
    # Scenario: 2 seconds of English, then 2 seconds of Chinese
    logger.info("\nSimulating audio stream: 2s English → 2s Chinese")

    sample_rate = 16000
    hop_samples = int(0.1 * sample_rate)  # 100ms = 1600 samples

    # Generate 4 seconds of audio (40 frames at 10Hz)
    total_frames = 40
    current_language = 'en'

    switch_detected_at = None

    for frame_idx in range(total_frames):
        # Generate frame
        audio_frame = np.random.randn(hop_samples).astype(np.float32) * 0.1
        timestamp = frame_idx * 0.1

        # Simulate LID probabilities
        if frame_idx < 20:
            # First 2 seconds: English
            lid_probs = {'en': 0.85, 'zh': 0.15}
        else:
            # Next 2 seconds: Chinese
            lid_probs = {'en': 0.15, 'zh': 0.85}

        # Run LID detection (mock)
        # In real system: lid_frame = lid.detect(audio_frame, timestamp, model)
        # For test: use simulated probs

        # Apply smoothing
        smoothed = smoother.smooth(lid_probs, timestamp)

        # Check for sustained change
        switch_event = detector.update(smoothed.smoothed_probabilities, timestamp)

        if switch_event:
            switch_detected_at = frame_idx
            logger.info(
                f"🔄 Frame {frame_idx} ({timestamp:.1f}s): Switch detected! "
                f"{switch_event.from_language} → {switch_event.to_language}"
            )
            current_language = switch_event.to_language
            break

    # Verify switch detected around frame 26 (20 frames + 6 frame dwell)
    assert switch_detected_at is not None, "Should detect language switch"
    assert 24 <= switch_detected_at <= 28, f"Switch should occur around frame 26, got {switch_detected_at}"

    logger.info(f"✅ Switch detected at frame {switch_detected_at} (expected ~26)")
    logger.info("✅ TEST 4 PASSED")
    return True


def run_all_tests():
    """Run all Milestone 2 tests"""
    logger.info("\n" + "="*80)
    logger.info("MILESTONE 2: SESSION-RESTART CODE-SWITCHING TESTS")
    logger.info("="*80)

    start_time = time.time()

    tests = [
        ("Frame-Level LID", test_frame_level_lid),
        ("LID Smoothing", test_lid_smoothing),
        ("Sustained Detection", test_sustained_detection),
        ("Session Switching", test_session_switching_simulation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status}: {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    logger.info(f"\nTotal: {passed}/{total} tests passed")
    logger.info(f"Time: {elapsed:.2f}s")

    if passed == total:
        logger.info("\n🎉 ALL MILESTONE 2 TESTS PASSED! 🎉")
        return True
    else:
        logger.error(f"\n❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
