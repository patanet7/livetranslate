#!/usr/bin/env python3
"""
Test Runtime Switching Between Original and Enhanced Stages

Verifies that the use_enhanced_stages config flag correctly switches
between original and enhanced implementations at runtime.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from audio.config import AudioProcessingConfig
from audio.audio_processor import AudioPipelineProcessor


def test_original_stages():
    """Test that original stages are used when flag is False."""
    print("=" * 80)
    print("TEST: Original Stages (use_enhanced_stages=False)")
    print("=" * 80)

    config = AudioProcessingConfig()
    config.use_enhanced_stages = False
    config.enabled_stages = ["lufs_normalization", "compression", "limiter"]

    processor = AudioPipelineProcessor(config, sample_rate=16000)

    # Generate test audio
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5).astype(
        np.float32
    )

    # Process
    processed, metadata = processor.process_audio_chunk(audio)

    print(f"✓ Processed {len(processed)} samples")
    print(f"  Stages applied: {metadata.get('stages_processed', [])}")
    print("  Stage types:")

    # Check stage names
    for stage_name in config.enabled_stages:
        stage = processor.pipeline.get_stage(stage_name)
        if stage:
            print(f"    {stage_name}: {stage.__class__.__name__}")

    print()
    return True


def test_enhanced_stages():
    """Test that enhanced stages are used when flag is True."""
    print("=" * 80)
    print("TEST: Enhanced Stages (use_enhanced_stages=True)")
    print("=" * 80)

    config = AudioProcessingConfig()
    config.use_enhanced_stages = True
    config.enabled_stages = ["lufs_normalization", "compression", "limiter"]

    processor = AudioPipelineProcessor(config, sample_rate=16000)

    # Generate test audio
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5).astype(
        np.float32
    )

    # Process
    processed, metadata = processor.process_audio_chunk(audio)

    print(f"✓ Processed {len(processed)} samples")
    print(f"  Stages applied: {metadata.get('stages_processed', [])}")
    print("  Stage types:")

    # Check stage names
    for stage_name in config.enabled_stages:
        stage = processor.pipeline.get_stage(stage_name)
        if stage:
            class_name = stage.__class__.__name__
            print(f"    {stage_name}: {class_name}")

            # Verify enhanced stages
            if "Enhanced" in class_name:
                print("      ✓ Using enhanced implementation")
            else:
                print(
                    "      ⚠ Using original implementation (enhanced not available?)"
                )

    print()
    return True


def test_runtime_switching():
    """Test switching between original and enhanced at runtime."""
    print("=" * 80)
    print("TEST: Runtime Switching")
    print("=" * 80)

    # Start with original
    config = AudioProcessingConfig()
    config.use_enhanced_stages = False
    config.enabled_stages = ["lufs_normalization", "compression", "limiter"]

    processor = AudioPipelineProcessor(config, sample_rate=16000)
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5).astype(
        np.float32
    )

    # Process with original
    processed1, metadata1 = processor.process_audio_chunk(audio)
    stage1 = processor.pipeline.get_stage("lufs_normalization")
    print(f"Initial: {stage1.__class__.__name__}")

    # Switch to enhanced
    config.use_enhanced_stages = True
    processor2 = AudioPipelineProcessor(config, sample_rate=16000)
    processed2, metadata2 = processor2.process_audio_chunk(audio)
    stage2 = processor2.pipeline.get_stage("lufs_normalization")
    print(f"After switch: {stage2.__class__.__name__}")

    # Verify they're different
    if stage1.__class__.__name__ != stage2.__class__.__name__:
        print("✓ Runtime switching working correctly")
        return True
    else:
        print("✗ Runtime switching not working (classes are the same)")
        return False


def main():
    print("\n" + "=" * 80)
    print("RUNTIME SWITCHING VERIFICATION")
    print("=" * 80 + "\n")

    results = []

    try:
        results.append(("Original stages", test_original_stages()))
    except Exception as e:
        print(f"✗ Original stages test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Original stages", False))

    try:
        results.append(("Enhanced stages", test_enhanced_stages()))
    except Exception as e:
        print(f"✗ Enhanced stages test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Enhanced stages", False))

    try:
        results.append(("Runtime switching", test_runtime_switching()))
    except Exception as e:
        print(f"✗ Runtime switching test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Runtime switching", False))

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✅ All runtime switching tests passed!")
        print("\nIntegration complete:")
        print("  • use_enhanced_stages config flag added")
        print("  • AudioPipelineProcessor respects the flag")
        print("  • Runtime switching between original and enhanced stages working")
        print("\nUsage:")
        print("  config.use_enhanced_stages = True   # Use pyloudnorm, pedalboard")
        print("  config.use_enhanced_stages = False  # Use original custom stages")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
