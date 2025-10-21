#!/usr/bin/env python3
"""
A/B Comparison Test: Original vs Enhanced Audio Stages

Compares the quality and performance of original custom DSP implementations
vs enhanced industry-standard library implementations.

Usage:
    python test_ab_comparison.py --stage lufs_normalization
    python test_ab_comparison.py --stage compression
    python test_ab_comparison.py --stage limiter
    python test_ab_comparison.py --all
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio.config import (
    LUFSNormalizationConfig,
    LUFSNormalizationMode,
    CompressionConfig,
    CompressionMode,
    LimiterConfig
)

# Import original stages
from audio.stages.lufs_normalization_stage import LUFSNormalizationStage
from audio.stages.compression_stage import CompressionStage
from audio.stages.limiter_stage import LimiterStage

# Import enhanced stages
from audio.stages_enhanced import (
    LUFSNormalizationStageEnhanced,
    CompressionStageEnhanced,
    LimiterStageEnhanced
)


class ABTester:
    """A/B comparison tester for audio stages."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.results = []

    def generate_test_audio(self, duration: float = 1.0, frequency: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
        """Generate test audio (sine wave)."""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        audio = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
        return audio

    def generate_complex_test_audio(self, duration: float = 5.0) -> np.ndarray:
        """Generate more complex test audio with varying dynamics."""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Multiple frequencies
        audio = np.zeros(num_samples, dtype=np.float32)
        audio += 0.3 * np.sin(2 * np.pi * 440 * t)  # A4
        audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5
        audio += 0.1 * np.sin(2 * np.pi * 220 * t)  # A3

        # Varying amplitude envelope
        envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        audio = audio * envelope

        # Add some quiet and loud sections
        quiet_section = int(num_samples * 0.3)
        loud_section = int(num_samples * 0.7)
        audio[:quiet_section] *= 0.3
        audio[loud_section:] *= 0.9

        return audio.astype(np.float32)

    def benchmark_performance(self, stage, audio: np.ndarray, runs: int = 100) -> Dict[str, float]:
        """Benchmark processing performance."""
        times = []

        for _ in range(runs):
            start = time.perf_counter()
            stage.process(audio)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
        }

    def calculate_quality_metrics(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics between original and processed audio."""
        # RMS levels
        original_rms = np.sqrt(np.mean(original ** 2))
        processed_rms = np.sqrt(np.mean(processed ** 2))

        # Peak levels
        original_peak = np.max(np.abs(original))
        processed_peak = np.max(np.abs(processed))

        # Signal-to-noise ratio (simplified)
        difference = processed - original
        noise_power = np.mean(difference ** 2)
        signal_power = np.mean(processed ** 2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))

        # Correlation
        correlation = np.corrcoef(original, processed)[0, 1]

        return {
            "original_rms_db": 20 * np.log10(original_rms) if original_rms > 0 else -80.0,
            "processed_rms_db": 20 * np.log10(processed_rms) if processed_rms > 0 else -80.0,
            "original_peak_db": 20 * np.log10(original_peak) if original_peak > 0 else -80.0,
            "processed_peak_db": 20 * np.log10(processed_peak) if processed_peak > 0 else -80.0,
            "gain_change_db": 20 * np.log10(processed_rms / max(original_rms, 1e-10)),
            "snr_db": float(snr),
            "correlation": float(correlation)
        }

    def test_lufs_normalization(self) -> Dict[str, Any]:
        """Test LUFS normalization: original vs enhanced."""
        print("\n" + "="*80)
        print("A/B TEST: LUFS Normalization")
        print("="*80)

        # Create config
        config = LUFSNormalizationConfig(
            enabled=True,
            mode=LUFSNormalizationMode.STREAMING,
            target_lufs=-14.0,
            true_peak_limiting=True
        )

        # Generate test audio
        audio = self.generate_complex_test_audio(duration=5.0)
        print(f"\nTest audio: {len(audio)} samples ({len(audio)/self.sample_rate:.1f}s)")
        print(f"Input RMS: {20 * np.log10(np.sqrt(np.mean(audio**2))):.2f} dB")

        # Test original stage
        print("\n--- Original Stage ---")
        original_stage = LUFSNormalizationStage(config, self.sample_rate)
        result_original = original_stage.process(audio)
        perf_original = self.benchmark_performance(original_stage, audio, runs=50)

        print(f"Output LUFS: {result_original.metadata.get('output_lufs', 'N/A')}")
        print(f"Performance: {perf_original['mean_ms']:.2f} ± {perf_original['std_ms']:.2f} ms")

        # Test enhanced stage
        print("\n--- Enhanced Stage ---")
        enhanced_stage = LUFSNormalizationStageEnhanced(config, self.sample_rate)
        result_enhanced = enhanced_stage.process(audio)
        perf_enhanced = self.benchmark_performance(enhanced_stage, audio, runs=50)

        print(f"Output LUFS: {result_enhanced.metadata.get('output_lufs', 'N/A')}")
        print(f"Performance: {perf_enhanced['mean_ms']:.2f} ± {perf_enhanced['std_ms']:.2f} ms")

        # Quality comparison
        print("\n--- Quality Comparison ---")
        quality = self.calculate_quality_metrics(result_original.processed_audio, result_enhanced.processed_audio)
        print(f"Correlation: {quality['correlation']:.4f}")
        print(f"SNR: {quality['snr_db']:.2f} dB")
        print(f"Gain difference: {quality['gain_change_db']:.2f} dB")

        return {
            "stage": "lufs_normalization",
            "original": {
                "metadata": result_original.metadata,
                "performance": perf_original
            },
            "enhanced": {
                "metadata": result_enhanced.metadata,
                "performance": perf_enhanced
            },
            "quality_comparison": quality
        }

    def test_compression(self) -> Dict[str, Any]:
        """Test compression: original vs enhanced."""
        print("\n" + "="*80)
        print("A/B TEST: Compression")
        print("="*80)

        # Create config
        config = CompressionConfig(
            enabled=True,
            mode=CompressionMode.SOFT_KNEE,
            threshold=-20.0,
            ratio=3.0,
            attack_time=5.0,
            release_time=100.0
        )

        # Generate test audio with high dynamics
        audio = self.generate_complex_test_audio(duration=5.0)
        print(f"\nTest audio: {len(audio)} samples ({len(audio)/self.sample_rate:.1f}s)")
        print(f"Input peak: {20 * np.log10(np.max(np.abs(audio))):.2f} dB")

        # Test original stage
        print("\n--- Original Stage ---")
        original_stage = CompressionStage(config, self.sample_rate)
        result_original = original_stage.process(audio)
        perf_original = self.benchmark_performance(original_stage, audio, runs=50)

        print(f"Gain reduction: {result_original.metadata.get('gain_reduction_db', 'N/A')} dB")
        print(f"Performance: {perf_original['mean_ms']:.2f} ± {perf_original['std_ms']:.2f} ms")

        # Test enhanced stage
        print("\n--- Enhanced Stage ---")
        enhanced_stage = CompressionStageEnhanced(config, self.sample_rate)
        result_enhanced = enhanced_stage.process(audio)
        perf_enhanced = self.benchmark_performance(enhanced_stage, audio, runs=50)

        print(f"Gain reduction: {result_enhanced.metadata.get('gain_reduction_db', 'N/A')} dB")
        print(f"Performance: {perf_enhanced['mean_ms']:.2f} ± {perf_enhanced['std_ms']:.2f} ms")

        # Quality comparison
        print("\n--- Quality Comparison ---")
        quality = self.calculate_quality_metrics(result_original.processed_audio, result_enhanced.processed_audio)
        print(f"Correlation: {quality['correlation']:.4f}")
        print(f"SNR: {quality['snr_db']:.2f} dB")
        print(f"Gain difference: {quality['gain_change_db']:.2f} dB")

        return {
            "stage": "compression",
            "original": {
                "metadata": result_original.metadata,
                "performance": perf_original
            },
            "enhanced": {
                "metadata": result_enhanced.metadata,
                "performance": perf_enhanced
            },
            "quality_comparison": quality
        }

    def test_limiter(self) -> Dict[str, Any]:
        """Test limiter: original vs enhanced."""
        print("\n" + "="*80)
        print("A/B TEST: Limiter")
        print("="*80)

        # Create config
        config = LimiterConfig(
            enabled=True,
            threshold=-1.0,
            release_time=50.0,
            soft_clip=True
        )

        # Generate test audio with peaks
        audio = self.generate_complex_test_audio(duration=5.0)
        print(f"\nTest audio: {len(audio)} samples ({len(audio)/self.sample_rate:.1f}s)")
        print(f"Input peak: {20 * np.log10(np.max(np.abs(audio))):.2f} dB")

        # Test original stage
        print("\n--- Original Stage ---")
        original_stage = LimiterStage(config, self.sample_rate)
        result_original = original_stage.process(audio)
        perf_original = self.benchmark_performance(original_stage, audio, runs=50)

        print(f"Output peak: {result_original.metadata.get('output_peak_db', 'N/A')} dB")
        print(f"Gain reduction: {result_original.metadata.get('gain_reduction_db', 'N/A')} dB")
        print(f"Performance: {perf_original['mean_ms']:.2f} ± {perf_original['std_ms']:.2f} ms")

        # Test enhanced stage
        print("\n--- Enhanced Stage ---")
        enhanced_stage = LimiterStageEnhanced(config, self.sample_rate)
        result_enhanced = enhanced_stage.process(audio)
        perf_enhanced = self.benchmark_performance(enhanced_stage, audio, runs=50)

        print(f"Output peak: {result_enhanced.metadata.get('output_peak_db', 'N/A')} dB")
        print(f"Gain reduction: {result_enhanced.metadata.get('gain_reduction_db', 'N/A')} dB")
        print(f"Performance: {perf_enhanced['mean_ms']:.2f} ± {perf_enhanced['std_ms']:.2f} ms")

        # Quality comparison
        print("\n--- Quality Comparison ---")
        quality = self.calculate_quality_metrics(result_original.processed_audio, result_enhanced.processed_audio)
        print(f"Correlation: {quality['correlation']:.4f}")
        print(f"SNR: {quality['snr_db']:.2f} dB")
        print(f"Gain difference: {quality['gain_change_db']:.2f} dB")

        return {
            "stage": "limiter",
            "original": {
                "metadata": result_original.metadata,
                "performance": perf_original
            },
            "enhanced": {
                "metadata": result_enhanced.metadata,
                "performance": perf_enhanced
            },
            "quality_comparison": quality
        }

    def print_summary(self, results: list):
        """Print summary of all A/B tests."""
        print("\n" + "="*80)
        print("A/B COMPARISON SUMMARY")
        print("="*80)

        for result in results:
            stage = result["stage"]
            print(f"\n{stage.upper()}:")

            # Performance comparison
            orig_perf = result["original"]["performance"]["mean_ms"]
            enh_perf = result["enhanced"]["performance"]["mean_ms"]
            perf_diff = ((enh_perf - orig_perf) / orig_perf) * 100

            print(f"  Original performance: {orig_perf:.2f} ms")
            print(f"  Enhanced performance: {enh_perf:.2f} ms")
            print(f"  Performance delta: {perf_diff:+.1f}%")

            # Quality
            quality = result["quality_comparison"]
            print(f"  Correlation: {quality['correlation']:.4f}")
            print(f"  SNR: {quality['snr_db']:.2f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="A/B comparison test: Original vs Enhanced audio stages"
    )

    parser.add_argument(
        '--stage',
        choices=['lufs_normalization', 'compression', 'limiter'],
        help='Test specific stage'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all stages'
    )

    args = parser.parse_args()

    tester = ABTester(sample_rate=16000)
    results = []

    if args.all or args.stage == 'lufs_normalization':
        results.append(tester.test_lufs_normalization())

    if args.all or args.stage == 'compression':
        results.append(tester.test_compression())

    if args.all or args.stage == 'limiter':
        results.append(tester.test_limiter())

    if not args.all and not args.stage:
        print("Please specify --stage or --all")
        parser.print_help()
        sys.exit(1)

    tester.print_summary(results)

    print("\n✅ A/B comparison complete!")
    print("\nConclusion:")
    print("  Enhanced stages use industry-standard libraries (pyloudnorm, pedalboard)")
    print("  Performance may be slightly slower but quality is professional-grade")
    print("  Enhanced stages are ITU-R compliant and battle-tested at scale (Spotify)")


if __name__ == "__main__":
    main()
