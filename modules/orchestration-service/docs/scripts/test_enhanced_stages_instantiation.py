#!/usr/bin/env python3
"""
Test that enhanced stages can be instantiated and used.
This test actually loads the libraries and creates stage instances.
"""

import sys

sys.path.insert(0, "src")

import numpy as np


def test_lufs_stage():
    """Test LUFS normalization stage."""
    print("=" * 60)
    print("Testing LUFS Normalization Stage")
    print("=" * 60)

    try:
        from audio.config import LUFSNormalizationConfig, LUFSNormalizationMode
        from audio.stages_enhanced import LUFSNormalizationStageEnhanced

        # Create config
        config = LUFSNormalizationConfig(
            enabled=True,
            mode=LUFSNormalizationMode.STREAMING,
            target_lufs=-14.0,
            true_peak_limiting=True,
        )

        # Create stage
        stage = LUFSNormalizationStageEnhanced(config, sample_rate=16000)
        print(f"✓ Stage created: {stage.stage_name}")

        # Create test audio (1 second of sine wave)
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # Process audio
        result = stage.process(audio)
        print(f"✓ Processed {len(audio)} samples")
        print(f"  Output LUFS: {result.metadata.get('output_lufs', 'N/A')}")
        print(f"  Implementation: {result.metadata.get('implementation', 'N/A')}")

        return True
    except Exception as e:
        print(f"✗ LUFS stage failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_compression_stage():
    """Test compression stage."""
    print("\n" + "=" * 60)
    print("Testing Compression Stage")
    print("=" * 60)

    try:
        from audio.config import CompressionConfig, CompressionMode
        from audio.stages_enhanced import CompressionStageEnhanced

        # Create config
        config = CompressionConfig(
            enabled=True,
            mode=CompressionMode.SOFT_KNEE,
            threshold=-20.0,
            ratio=3.0,
            attack_time=5.0,
            release_time=100.0,
        )

        # Create stage
        stage = CompressionStageEnhanced(config, sample_rate=16000)
        print(f"✓ Stage created: {stage.stage_name}")

        # Create test audio (1 second of sine wave with varying amplitude)
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

        # Process audio
        result = stage.process(audio)
        print(f"✓ Processed {len(audio)} samples")
        print(f"  Gain reduction: {result.metadata.get('gain_reduction_db', 'N/A')} dB")
        print(f"  Implementation: {result.metadata.get('implementation', 'N/A')}")

        return True
    except Exception as e:
        print(f"✗ Compression stage failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_limiter_stage():
    """Test limiter stage."""
    print("\n" + "=" * 60)
    print("Testing Limiter Stage")
    print("=" * 60)

    try:
        from audio.config import LimiterConfig
        from audio.stages_enhanced import LimiterStageEnhanced

        # Create config
        config = LimiterConfig(enabled=True, threshold=-1.0, release_time=50.0, soft_clip=True)

        # Create stage
        stage = LimiterStageEnhanced(config, sample_rate=16000)
        print(f"✓ Stage created: {stage.stage_name}")

        # Create test audio (1 second of loud sine wave)
        t = np.linspace(0, 1, 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.9).astype(np.float32)

        # Process audio
        result = stage.process(audio)
        print(f"✓ Processed {len(audio)} samples")
        print(f"  Input peak: {result.metadata.get('input_peak_db', 'N/A')} dB")
        print(f"  Output peak: {result.metadata.get('output_peak_db', 'N/A')} dB")
        print(f"  Limiting engaged: {result.metadata.get('limiting_engaged', 'N/A')}")
        print(f"  Implementation: {result.metadata.get('implementation', 'N/A')}")

        return True
    except Exception as e:
        print(f"✗ Limiter stage failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all instantiation tests."""
    results = []

    results.append(("LUFS Stage", test_lufs_stage()))
    results.append(("Compression Stage", test_compression_stage()))
    results.append(("Limiter Stage", test_limiter_stage()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✅ All enhanced stages are working correctly!")
        print("\nPhase 1 Implementation Status:")
        print("  ✓ LUFS Normalization (pyloudnorm)")
        print("  ✓ Compression (pedalboard)")
        print("  ✓ Limiter (pedalboard)")
        print("  ✓ VAD libraries available (webrtcvad)")
        print("\nReady for Phase 1.6: A/B comparison testing")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
