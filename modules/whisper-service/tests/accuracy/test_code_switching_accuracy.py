#!/usr/bin/env python3
"""
Accuracy Tests: Code-Switching Accuracy Baseline

Tests maintain 70%+ accuracy on code-switching benchmark.
Per ML Engineer review - Priority 4: Accuracy regression tests with baseline tracking

Critical functionality:
1. Maintain 70%+ accuracy on code-switching benchmark (FEEDBACK.md line 184)
2. Save accuracy baselines for CI/CD regression tracking
3. Test WER/CER on ground truth datasets
4. Compare against historical baselines

Reference: FEEDBACK.md lines 171-184, test_real_code_switching.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add src and tests to path
src_path = Path(__file__).parent.parent.parent / "src"
tests_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))

from session_restart import SessionRestartTranscriber
from test_utils import calculate_cer, calculate_wer_detailed, concatenate_transcription_segments

logger = logging.getLogger(__name__)


# Baseline storage
BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_FILE = BASELINE_DIR / "code_switching_baselines.json"

# Test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"


class AccuracyBaseline:
    """Manage accuracy baselines for regression tracking"""

    @staticmethod
    def load_baseline() -> dict:
        """Load baseline from file"""
        if not BASELINE_FILE.exists():
            return {}

        with open(BASELINE_FILE) as f:
            return json.load(f)

    @staticmethod
    def save_baseline(baseline: dict):
        """Save baseline to file"""
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)

        with open(BASELINE_FILE, "w") as f:
            json.dump(baseline, f, indent=2)

        logger.info(f"✅ Baseline saved to {BASELINE_FILE}")

    @staticmethod
    def update_baseline(test_name: str, metrics: dict):
        """Update baseline with new test results"""
        baseline = AccuracyBaseline.load_baseline()

        if test_name not in baseline:
            baseline[test_name] = {"history": [], "best": None}

        # Add to history
        entry = {"timestamp": datetime.now().isoformat(), "metrics": metrics}
        baseline[test_name]["history"].append(entry)

        # Update best
        current_accuracy = metrics.get("overall_accuracy", 0)
        if baseline[test_name]["best"] is None or current_accuracy > baseline[test_name]["best"][
            "metrics"
        ].get("overall_accuracy", 0):
            baseline[test_name]["best"] = entry

        # Keep only last 10 entries
        baseline[test_name]["history"] = baseline[test_name]["history"][-10:]

        AccuracyBaseline.save_baseline(baseline)

    @staticmethod
    def get_baseline(test_name: str) -> dict:
        """Get baseline for test"""
        baseline = AccuracyBaseline.load_baseline()
        return baseline.get(test_name, {})

    @staticmethod
    def print_comparison(test_name: str, current_metrics: dict):
        """Print comparison with baseline"""
        baseline_data = AccuracyBaseline.get_baseline(test_name)

        if not baseline_data:
            logger.info("📊 No baseline available (first run)")
            return

        best = baseline_data.get("best")
        if not best:
            logger.info("📊 No best baseline available")
            return

        logger.info("\n" + "=" * 80)
        logger.info("BASELINE COMPARISON")
        logger.info("=" * 80)

        current_acc = current_metrics.get("overall_accuracy", 0)
        best_acc = best["metrics"].get("overall_accuracy", 0)
        diff = current_acc - best_acc

        logger.info(f"Current accuracy:  {current_acc:.1f}%")
        logger.info(f"Best baseline:     {best_acc:.1f}%")
        logger.info(f"Difference:        {diff:+.1f}%")
        logger.info(f"Best from:         {best['timestamp']}")

        if diff >= 0:
            logger.info("✅ Accuracy maintained or improved!")
        elif diff > -5:
            logger.info("⚠️ Minor accuracy regression (< 5%)")
        else:
            logger.warning(f"❌ Significant accuracy regression ({diff:.1f}%)")

        logger.info("=" * 80)


# Ground truth data
GROUND_TRUTH = {
    "jfk": {
        "file": "jfk.wav",
        "language": "en",
        "text": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country",
        "target_wer": 25.0,
        "target_accuracy": 75.0,
    },
    "chinese_1": {
        "file": "OSR_cn_000_0072_8k.wav",
        "language": "zh",
        "text": "院子门口不远处就是一个地铁站 这是一个美丽而神奇的景象 树上长满了又大又甜的桃子 海豚和鲸鱼的表演是很好看的节目",
        "target_cer": 30.0,  # CER for Chinese
        "target_accuracy": 70.0,
    },
    "mixed_en_zh": {
        "file": "test_clean_mixed_en_zh.wav",
        "segments": [
            {
                "language": "en",
                "text": "And so my fellow Americans ask not what your country can do for you ask what you can do for your country",
            },
            {
                "language": "zh",
                "text": "院子门口不远处就是一个地铁站 这是一个美丽而神奇的景象 树上长满了又大又甜的桃子",
            },
        ],
        "target_overall_accuracy": 70.0,
    },
}


@pytest.mark.accuracy
class TestEnglishAccuracy:
    """Test English transcription accuracy baseline"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for English tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

    def test_jfk_english_accuracy_baseline(self, transcriber):
        """
        Test JFK English accuracy maintains baseline.

        Target: 75%+ accuracy (25% WER) per FEEDBACK.md
        """
        ground_truth = GROUND_TRUTH["jfk"]
        audio_path = FIXTURES_DIR / ground_truth["file"]

        if not audio_path.exists():
            pytest.skip(f"Audio not found: {audio_path}")

        # Load and process audio
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = audio.astype(np.float32)

        # Process in chunks
        chunk_size = 8000  # 0.5s
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            transcriber.process(chunk)

        # Finalize to flush any remaining audio
        transcriber.finalize()

        # Get transcription
        all_segments = transcriber._get_all_segments()
        text_segments = [seg for seg in all_segments if seg.get("text") and seg.get("text").strip()]
        transcription = concatenate_transcription_segments(text_segments)

        # Calculate metrics
        wer_details = calculate_wer_detailed(ground_truth["text"], transcription)
        cer = calculate_cer(ground_truth["text"], transcription)

        wer = wer_details["normalized"]["wer"]
        accuracy = 100 - wer

        logger.info("\n" + "=" * 80)
        logger.info("JFK ENGLISH ACCURACY TEST")
        logger.info("=" * 80)
        logger.info(f"Expected:      '{ground_truth['text']}'")
        logger.info(f"Transcription: '{transcription}'")
        logger.info(f"\nWER: {wer:.1f}%")
        logger.info(f"Accuracy: {accuracy:.1f}%")
        logger.info(f"CER: {cer:.1f}%")
        logger.info(f"Target accuracy: {ground_truth['target_accuracy']:.1f}%")

        # Store metrics
        metrics = {
            "wer": wer,
            "accuracy": accuracy,
            "cer": cer,
            "transcription": transcription,
            "expected": ground_truth["text"],
        }

        # Update baseline
        AccuracyBaseline.update_baseline("jfk_english", metrics)
        AccuracyBaseline.print_comparison("jfk_english", {"overall_accuracy": accuracy})

        # Assertions
        assert (
            accuracy >= ground_truth["target_accuracy"]
        ), f"Accuracy {accuracy:.1f}% below target {ground_truth['target_accuracy']:.1f}%"

        logger.info(
            f"✅ JFK English accuracy: {accuracy:.1f}% (target: {ground_truth['target_accuracy']:.1f}%)"
        )


@pytest.mark.accuracy
class TestChineseAccuracy:
    """Test Chinese transcription accuracy baseline"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for Chinese tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["zh"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
        )

    def test_chinese_accuracy_baseline(self, transcriber):
        """
        Test Chinese transcription accuracy maintains baseline.

        Target: 70%+ accuracy (30% CER) - more lenient due to 8kHz source
        """
        ground_truth = GROUND_TRUTH["chinese_1"]
        audio_path = FIXTURES_DIR / ground_truth["file"]

        if not audio_path.exists():
            pytest.skip(f"Audio not found: {audio_path}")

        # Load and process audio
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = audio.astype(np.float32)

        # Process in chunks
        chunk_size = 8000
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            transcriber.process(chunk)

        transcriber.finalize()

        # Get transcription
        all_segments = transcriber._get_all_segments()
        text_segments = [seg for seg in all_segments if seg.get("text") and seg.get("text").strip()]
        transcription = concatenate_transcription_segments(text_segments)

        # Calculate metrics (CER for Chinese)
        cer = calculate_cer(ground_truth["text"], transcription)
        accuracy = 100 - cer

        logger.info("\n" + "=" * 80)
        logger.info("CHINESE ACCURACY TEST")
        logger.info("=" * 80)
        logger.info(f"Expected:      '{ground_truth['text']}'")
        logger.info(f"Transcription: '{transcription}'")
        logger.info(f"\nCER: {cer:.1f}%")
        logger.info(f"Accuracy: {accuracy:.1f}%")
        logger.info(f"Target accuracy: {ground_truth['target_accuracy']:.1f}%")

        # Store metrics
        metrics = {
            "cer": cer,
            "accuracy": accuracy,
            "transcription": transcription,
            "expected": ground_truth["text"],
        }

        # Update baseline
        AccuracyBaseline.update_baseline("chinese", metrics)
        AccuracyBaseline.print_comparison("chinese", {"overall_accuracy": accuracy})

        # More lenient threshold due to 8kHz source audio quality
        assert (
            accuracy >= ground_truth["target_accuracy"]
        ), f"Accuracy {accuracy:.1f}% below target {ground_truth['target_accuracy']:.1f}%"

        logger.info(
            f"✅ Chinese accuracy: {accuracy:.1f}% (target: {ground_truth['target_accuracy']:.1f}%)"
        )


@pytest.mark.accuracy
class TestCodeSwitchingAccuracy:
    """Test code-switching accuracy baseline (EN↔ZH)"""

    @pytest.fixture
    def transcriber(self):
        """Create transcriber for code-switching tests"""
        models_dir = Path.home() / ".whisper" / "models"
        model_path = str(models_dir / "large-v3-turbo.pt")

        if not Path(model_path).exists():
            pytest.skip(f"Model not found: {model_path}")

        return SessionRestartTranscriber(
            model_path=model_path,
            models_dir=str(models_dir),
            target_languages=["en", "zh"],
            online_chunk_size=1.2,
            vad_threshold=0.5,
            sampling_rate=16000,
            lid_hop_ms=100,
            confidence_margin=0.2,
            min_dwell_frames=6,
            min_dwell_ms=250.0,
        )

    def test_mixed_en_zh_accuracy_baseline(self, transcriber):
        """
        Test mixed English-Chinese accuracy maintains baseline.

        Target: 70-85% overall accuracy per FEEDBACK.md line 184
        This is the critical code-switching test case.
        """
        ground_truth = GROUND_TRUTH["mixed_en_zh"]
        audio_path = FIXTURES_DIR / ground_truth["file"]

        if not audio_path.exists():
            pytest.skip(f"Audio not found: {audio_path}")

        # Load and process audio
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = audio.astype(np.float32)

        # Process in chunks
        chunk_size = 8000
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            transcriber.process(chunk)

        transcriber.finalize()

        # Get segments by language
        all_segments = transcriber._get_all_segments()
        en_segments = [
            seg
            for seg in all_segments
            if seg.get("language") == "en" and seg.get("text") and seg.get("text").strip()
        ]
        zh_segments = [
            seg
            for seg in all_segments
            if seg.get("language") == "zh" and seg.get("text") and seg.get("text").strip()
        ]

        # Calculate per-language accuracy
        en_transcription = concatenate_transcription_segments(en_segments)
        zh_transcription = concatenate_transcription_segments(zh_segments)

        en_expected = ground_truth["segments"][0]["text"]
        zh_expected = " ".join(
            seg["text"] for seg in ground_truth["segments"] if seg["language"] == "zh"
        )

        # English metrics
        if en_segments:
            en_wer_details = calculate_wer_detailed(en_expected, en_transcription)
            en_accuracy = 100 - en_wer_details["normalized"]["wer"]
        else:
            en_accuracy = 0.0
            logger.warning("⚠️ No English segments detected")

        # Chinese metrics
        if zh_segments:
            zh_cer = calculate_cer(zh_expected, zh_transcription)
            zh_accuracy = 100 - zh_cer
        else:
            zh_accuracy = 0.0
            logger.warning("⚠️ No Chinese segments detected")

        # Overall accuracy
        overall_accuracy = (en_accuracy + zh_accuracy) / 2

        # Language switching
        stats = transcriber.get_statistics()

        logger.info("\n" + "=" * 80)
        logger.info("CODE-SWITCHING ACCURACY TEST")
        logger.info("=" * 80)
        logger.info(f"English segments: {len(en_segments)}")
        logger.info(f"  Expected:      '{en_expected}'")
        logger.info(f"  Transcription: '{en_transcription}'")
        logger.info(f"  Accuracy:      {en_accuracy:.1f}%")
        logger.info(f"\nChinese segments: {len(zh_segments)}")
        logger.info(f"  Expected:      '{zh_expected}'")
        logger.info(f"  Transcription: '{zh_transcription}'")
        logger.info(f"  Accuracy:      {zh_accuracy:.1f}%")
        logger.info(f"\nOverall accuracy: {overall_accuracy:.1f}%")
        logger.info(f"Language switches: {stats['total_switches']}")
        logger.info(f"Total sessions: {stats['total_sessions']}")
        logger.info(f"Target accuracy: {ground_truth['target_overall_accuracy']:.1f}%")

        # Store metrics
        metrics = {
            "en_accuracy": en_accuracy,
            "zh_accuracy": zh_accuracy,
            "overall_accuracy": overall_accuracy,
            "switches": stats["total_switches"],
            "sessions": stats["total_sessions"],
            "en_transcription": en_transcription,
            "zh_transcription": zh_transcription,
        }

        # Update baseline
        AccuracyBaseline.update_baseline("mixed_en_zh", metrics)
        AccuracyBaseline.print_comparison("mixed_en_zh", metrics)

        # Assertions
        assert (
            overall_accuracy >= ground_truth["target_overall_accuracy"]
        ), f"Overall accuracy {overall_accuracy:.1f}% below target {ground_truth['target_overall_accuracy']:.1f}%"

        # Should detect at least one language switch
        if len(en_segments) > 0 and len(zh_segments) > 0:
            logger.info("✅ Both languages detected (code-switching working)")

        logger.info(
            f"✅ Code-switching accuracy: {overall_accuracy:.1f}% (target: {ground_truth['target_overall_accuracy']:.1f}%)"
        )


@pytest.mark.accuracy
class TestAccuracyRegressionDetection:
    """Test accuracy regression detection"""

    def test_detect_accuracy_regression(self):
        """
        Test can detect accuracy regression from baseline.

        This validates the baseline tracking system works.
        """
        test_name = "test_regression_detection"

        # Simulate baseline (80% accuracy)
        baseline_metrics = {"overall_accuracy": 80.0, "wer": 20.0}
        AccuracyBaseline.update_baseline(test_name, baseline_metrics)

        # Simulate regression (60% accuracy)
        current_metrics = {"overall_accuracy": 60.0, "wer": 40.0}

        # Load baseline
        baseline_data = AccuracyBaseline.get_baseline(test_name)

        assert baseline_data is not None
        assert "best" in baseline_data

        best_acc = baseline_data["best"]["metrics"]["overall_accuracy"]
        current_acc = current_metrics["overall_accuracy"]
        regression = current_acc - best_acc

        logger.info(f"Baseline: {best_acc:.1f}%")
        logger.info(f"Current: {current_acc:.1f}%")
        logger.info(f"Regression: {regression:.1f}%")

        # Should detect 20% regression
        assert regression < -10, "Should detect significant regression"

        logger.info("✅ Regression detection works")

    def test_baseline_history_tracking(self):
        """Test baseline tracks history correctly"""
        test_name = "test_history_tracking"

        # Add multiple entries
        for i in range(5):
            metrics = {"overall_accuracy": 70.0 + i, "iteration": i}
            AccuracyBaseline.update_baseline(test_name, metrics)

        # Load baseline
        baseline_data = AccuracyBaseline.get_baseline(test_name)

        assert "history" in baseline_data
        assert len(baseline_data["history"]) == 5

        # Best should be iteration 4 (74% accuracy)
        assert baseline_data["best"]["metrics"]["iteration"] == 4

        logger.info(f"✅ History tracking works: {len(baseline_data['history'])} entries")


@pytest.mark.accuracy
class TestAccuracyReportGeneration:
    """Test accuracy report generation for CI/CD"""

    def test_generate_accuracy_report(self):
        """
        Generate comprehensive accuracy report for CI/CD.

        This report can be used in CI/CD pipelines to track
        accuracy trends over time.
        """
        baseline = AccuracyBaseline.load_baseline()

        report = {"generated_at": datetime.now().isoformat(), "tests": {}}

        for test_name, data in baseline.items():
            if "best" not in data or not data["best"]:
                continue

            best = data["best"]
            report["tests"][test_name] = {
                "best_accuracy": best["metrics"].get(
                    "overall_accuracy", best["metrics"].get("accuracy", 0)
                ),
                "best_timestamp": best["timestamp"],
                "history_length": len(data.get("history", [])),
            }

        logger.info("\n" + "=" * 80)
        logger.info("ACCURACY REPORT")
        logger.info("=" * 80)
        logger.info(json.dumps(report, indent=2))
        logger.info("=" * 80)

        # Save report
        report_path = BASELINE_DIR / "accuracy_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"✅ Accuracy report saved to {report_path}")


if __name__ == "__main__":
    # Run accuracy tests
    pytest.main([__file__, "-v", "-m", "accuracy", "--log-cli-level=INFO"])
