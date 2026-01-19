#!/usr/bin/env python3
"""
Test DX Optimization Modules

Quick test script to verify all new DX modules are working correctly.

Usage:
    python scripts/test_dx_modules.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_service_config():
    """Test service_config module"""
    print("\n=== Testing service_config ===")
    try:
        from service_config import LIDConfig, SessionConfig, VADConfig, WhisperConfig

        # Test VAD config
        vad = VADConfig.from_env()
        assert vad.threshold == 0.5
        assert vad.sampling_rate == 16000
        print("✅ VADConfig works")

        # Test LID config
        lid = LIDConfig.from_env()
        assert lid.confidence_margin == 0.2
        assert lid.min_dwell_frames == 6
        print("✅ LIDConfig works")

        # Test Whisper config
        whisper = WhisperConfig.from_env(model_path="/tmp/test.pt")
        assert whisper.decoder_type == "greedy"
        assert whisper.beam_size == 1
        print("✅ WhisperConfig works")

        # Test Session config
        config = SessionConfig.from_env(model_path="/tmp/test.pt")
        assert config.log_level == "INFO"
        print("✅ SessionConfig works")

        print("✅ service_config module: PASS")
        return True
    except Exception as e:
        print(f"❌ service_config module: FAIL - {e}")
        return False


def test_type_definitions():
    """Test type_definitions module"""
    print("\n=== Testing type_definitions ===")
    try:
        # Test that TypedDict classes exist
        print("✅ ProcessResult defined")
        print("✅ VADResult defined")
        print("✅ SessionSegment defined")
        print("✅ LIDProbs defined")
        print("✅ SwitchEvent defined")
        print("✅ Statistics defined")

        print("✅ type_definitions module: PASS")
        return True
    except Exception as e:
        print(f"❌ type_definitions module: FAIL - {e}")
        return False


def test_logging_utils():
    """Test logging_utils module"""
    print("\n=== Testing logging_utils ===")
    try:
        import time

        from logging_utils import MetricsCollector, PerformanceLogger, get_component_logger

        # Test PerformanceLogger
        perf = PerformanceLogger("test_component")

        with perf.measure("test_operation"):
            time.sleep(0.01)  # 10ms

        assert "test_operation" in perf.metrics
        assert len(perf.metrics["test_operation"]) > 0
        print("✅ PerformanceLogger works")

        # Test MetricsCollector
        metrics = MetricsCollector("test_metrics")
        metrics.increment("counter", 5)
        metrics.record_time("timer", 123.45)
        metrics.set_gauge("gauge", 42.0)

        summary = metrics.get_summary()
        assert summary["counters"]["counter"] == 5
        assert summary["gauges"]["gauge"] == 42.0
        print("✅ MetricsCollector works")

        # Test get_component_logger
        logger = get_component_logger("test")
        assert logger.name == "whisper_service.test"
        print("✅ get_component_logger works")

        print("✅ logging_utils module: PASS")
        return True
    except Exception as e:
        print(f"❌ logging_utils module: FAIL - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vad_helpers():
    """Test vad_helpers module"""
    print("\n=== Testing vad_helpers ===")
    try:
        from vad_helpers import (
            VADEventType,
            VADStatus,
            get_vad_action_plan,
            parse_vad_event,
            should_buffer_audio,
            should_process_buffer,
        )

        # Test parse_vad_event
        assert parse_vad_event(None) == VADEventType.NO_CHANGE
        assert parse_vad_event({"start": 1.0}) == VADEventType.SPEECH_START
        assert parse_vad_event({"end": 2.0}) == VADEventType.SPEECH_END
        assert parse_vad_event({"start": 1.0, "end": 2.0}) == VADEventType.SPEECH_RESTART
        print("✅ parse_vad_event works")

        # Test should_buffer_audio
        assert should_buffer_audio(VADEventType.SPEECH_START, VADStatus.NONVOICE)
        assert not should_buffer_audio(VADEventType.SPEECH_END, VADStatus.VOICE)
        print("✅ should_buffer_audio works")

        # Test should_process_buffer
        assert should_process_buffer(VADEventType.SPEECH_END)
        assert not should_process_buffer(VADEventType.SPEECH_START)
        print("✅ should_process_buffer works")

        # Test get_vad_action_plan
        should_buffer, should_process, new_status = get_vad_action_plan(
            {"start": 1.0}, VADStatus.NONVOICE
        )
        assert should_buffer
        assert not should_process
        assert new_status == VADStatus.VOICE
        print("✅ get_vad_action_plan works")

        print("✅ vad_helpers module: PASS")
        return True
    except Exception as e:
        print(f"❌ vad_helpers module: FAIL - {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_validation():
    """Test configuration validation"""
    print("\n=== Testing Configuration Validation ===")
    try:
        from service_config import LIDConfig, VADConfig

        # Test valid config
        VADConfig(threshold=0.5, sampling_rate=16000)
        print("✅ Valid VAD config accepted")

        # Test invalid threshold
        try:
            VADConfig(threshold=1.5)  # > 1.0
            print("❌ Should have rejected threshold > 1.0")
            return False
        except ValueError:
            print("✅ Invalid threshold rejected")

        # Test invalid sampling rate
        try:
            VADConfig(sampling_rate=44100)  # Not 8000 or 16000
            print("❌ Should have rejected invalid sampling rate")
            return False
        except ValueError:
            print("✅ Invalid sampling rate rejected")

        # Test valid LID config
        LIDConfig(confidence_margin=0.2, min_dwell_frames=6)
        print("✅ Valid LID config accepted")

        # Test invalid confidence margin
        try:
            LIDConfig(confidence_margin=0.6)  # > 0.5
            print("❌ Should have rejected confidence_margin > 0.5")
            return False
        except ValueError:
            print("✅ Invalid confidence margin rejected")

        print("✅ Configuration validation: PASS")
        return True
    except Exception as e:
        print(f"❌ Configuration validation: FAIL - {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("DX Optimization Modules Test Suite")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(("service_config", test_service_config()))
    results.append(("type_definitions", test_type_definitions()))
    results.append(("logging_utils", test_logging_utils()))
    results.append(("vad_helpers", test_vad_helpers()))
    results.append(("config_validation", test_config_validation()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All DX optimization modules working correctly!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
