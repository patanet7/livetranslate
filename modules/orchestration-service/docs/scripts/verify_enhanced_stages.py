#!/usr/bin/env python3
"""
Simple verification script for enhanced audio stages.
Does not use pytest to avoid macOS code signing issues.
"""

import sys

sys.path.insert(0, "src")


def test_imports():
    """Test that enhanced stages module can be imported."""
    print("=" * 60)
    print("Enhanced Audio Stages Verification")
    print("=" * 60)

    try:
        from audio import stages_enhanced

        print("✓ stages_enhanced module imported")
        print(f"  Version: {stages_enhanced.__version__}")
        print(f"  Phase 1 complete: {stages_enhanced.PHASE_1_COMPLETE}")
        print()

        print("Available features:")
        for feature, available in stages_enhanced.AVAILABLE_FEATURES.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature}: {available}")
        print()

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_configs():
    """Test that config classes can be imported."""
    print("Config classes:")
    try:
        from audio.config import (
            LUFSNormalizationConfig,
            CompressionConfig,
            LimiterConfig,
        )

        print("  ✓ LUFSNormalizationConfig")
        print("  ✓ CompressionConfig")
        print("  ✓ LimiterConfig")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Config import failed: {e}")
        return False


def main():
    """Run all verification tests."""
    results = []

    results.append(("Import test", test_imports()))
    results.append(("Config test", test_configs()))

    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All verification tests passed!")
        print()

        # Check if libraries are actually installed
        from audio import stages_enhanced

        if stages_enhanced.PHASE_1_COMPLETE:
            print("✅ Phase 1 Complete!")
            print("   All enhanced audio libraries are installed and ready.")
            print()
            print("Next steps:")
            print(
                "  • Run 'poetry run python test_enhanced_stages_instantiation.py' for full testing"
            )
            print("  • Proceed to Phase 1.6: A/B comparison testing")
        else:
            print("Note: Libraries are not installed yet.")
            print("Run 'poetry install' to install pyloudnorm and pedalboard.")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
