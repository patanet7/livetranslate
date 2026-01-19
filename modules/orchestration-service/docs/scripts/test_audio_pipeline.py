#!/usr/bin/env python3
"""
Audio Pipeline Testing CLI

Tests the complete audio processing pipeline with configurable stages.
Saves intermediate outputs after each stage for inspection.

✨ Now using enhanced stages (pyloudnorm, pedalboard) for LUFS, Compression, and Limiter!
   63-99% faster than previous custom implementations.

📚 For complete effects documentation, see: AUDIO_EFFECTS.md

Usage:
    poetry run python test_audio_pipeline.py --config test_config.json
    poetry run python test_audio_pipeline.py --config config_examples/broadcast.json --input ./input/test.wav
    poetry run python test_audio_pipeline.py --list-presets
    poetry run python test_audio_pipeline.py --create-example

Available Presets:
    - default: Balanced processing for general use
    - voice: Voice-optimized (meetings, calls)
    - noisy: Aggressive noise reduction
    - broadcast: Broadcast quality (ITU-R compliant)
    - conference: Conference call optimization
    - minimal: Light processing (low latency)

Output:
    - Saves audio after each stage to ./output/run_TIMESTAMP/
    - Saves processing metadata to JSON files
    - Saves complete test configuration
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio.audio_processor import (
    AudioPipelineProcessor,
    create_audio_pipeline_processor,
)
from audio.config import (
    AudioProcessingConfig,
    create_audio_config_manager,
    get_default_audio_processing_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioPipelineTester:
    """Test runner for audio processing pipeline"""

    def __init__(
        self,
        config_path: str | None = None,
        input_dir: str = "./input",
        output_dir: str = "./output",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path

        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.info("Using default configuration")
            self.config = self._get_default_config()

        # Create timestamp for this test run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory for this run
        self.run_output_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_output_dir.mkdir(exist_ok=True)

        logger.info(f"Output directory: {self.run_output_dir}")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path) as f:
            config_data = json.load(f)

        # Convert to AudioProcessingConfig
        if "preset_name" in config_data:
            # Use preset
            config_manager = create_audio_config_manager()
            return config_manager.get_preset_config(config_data["preset_name"])
        else:
            # Custom config
            return AudioProcessingConfig(**config_data)

    def _get_default_config(self) -> AudioProcessingConfig:
        """Get default audio processing config"""
        return get_default_audio_processing_config()

    def list_available_files(self) -> list[Path]:
        """List all audio files in input directory"""
        audio_extensions = [".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a"]
        files = []
        for ext in audio_extensions:
            files.extend(self.input_dir.glob(f"*{ext}"))
        return sorted(files)

    def load_audio(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Load audio file and return data + sample rate"""
        logger.info(f"Loading audio from: {file_path}")

        # Check if we need to convert format
        file_ext = file_path.suffix.lower()

        if file_ext in [".webm", ".mp4", ".m4a", ".mp3"]:
            # Convert using ffmpeg
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            try:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    str(file_path),
                    "-ac",
                    "1",  # Mono
                    "-ar",
                    "16000",  # 16kHz
                    "-y",
                    temp_wav_path,
                ]

                subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
                audio_data, sample_rate = sf.read(temp_wav_path)

            finally:
                if os.path.exists(temp_wav_path):
                    os.unlink(temp_wav_path)
        else:
            # soundfile can handle this
            audio_data, sample_rate = sf.read(str(file_path))

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        logger.info(f"Loaded: {len(audio_data)} samples at {sample_rate}Hz")
        logger.info(f"Duration: {len(audio_data) / sample_rate:.2f}s")
        logger.info(f"RMS level: {np.sqrt(np.mean(audio_data**2)):.4f}")
        logger.info(f"Peak level: {np.max(np.abs(audio_data)):.4f}")

        return audio_data, sample_rate

    def save_audio(self, audio_data: np.ndarray, sample_rate: int, filename: str):
        """Save audio data to output directory"""
        output_path = self.run_output_dir / filename
        sf.write(str(output_path), audio_data, sample_rate)
        logger.info(f"Saved: {output_path}")
        return output_path

    def save_metadata(self, metadata: dict[str, Any], filename: str):
        """Save processing metadata to JSON"""
        output_path = self.run_output_dir / filename
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {output_path}")
        return output_path

    def run_test(self, input_file: Path | None = None):
        """Run complete audio processing test"""
        logger.info("=" * 80)
        logger.info("AUDIO PIPELINE TEST")
        logger.info("=" * 80)

        # Find input file
        if input_file is None:
            available_files = self.list_available_files()
            if not available_files:
                logger.error(f"No audio files found in {self.input_dir}")
                logger.info(f"Please add audio files to {self.input_dir}")
                return False
            input_file = available_files[0]
            logger.info(f"Using first available file: {input_file}")
        else:
            input_file = Path(input_file)
            if not input_file.exists():
                logger.error(f"Input file not found: {input_file}")
                return False

        # Load audio
        try:
            audio_data, sample_rate = self.load_audio(input_file)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}", exc_info=True)
            return False

        # Save original
        self.save_audio(audio_data, sample_rate, "00_original.wav")

        # Save configuration
        if isinstance(self.config, AudioProcessingConfig):
            config_dict = self.config.dict()
        else:
            config_dict = self.config

        self.save_metadata(
            {
                "input_file": str(input_file),
                "sample_rate": sample_rate,
                "duration_seconds": len(audio_data) / sample_rate,
                "num_samples": len(audio_data),
                "config": config_dict,
                "timestamp": self.timestamp,
            },
            "test_info.json",
        )

        # Create processor
        logger.info("\n" + "=" * 80)
        logger.info("CREATING AUDIO PROCESSOR")
        logger.info("=" * 80)

        processor = create_audio_pipeline_processor(self.config, sample_rate)

        logger.info(f"Enabled stages: {self.config.enabled_stages}")
        logger.info(f"Preset: {self.config.preset_name}")

        # Process audio with stage-by-stage output
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING AUDIO (STAGE BY STAGE)")
        logger.info("=" * 80)

        try:
            # Get the processor to run stage-by-stage
            current_audio = audio_data.copy()
            stage_results = []

            # Run each enabled stage
            for stage_idx, stage_name in enumerate(self.config.enabled_stages, 1):
                logger.info(f"\n--- Stage {stage_idx}: {stage_name} ---")

                # Process through this stage only
                stage_audio, stage_metadata = self._process_single_stage(
                    processor, current_audio, stage_name
                )

                # Calculate metrics
                metrics = self._calculate_metrics(current_audio, stage_audio, sample_rate)

                # Log results
                logger.info(f"  Input RMS:  {metrics['input_rms']:.4f}")
                logger.info(f"  Output RMS: {metrics['output_rms']:.4f}")
                logger.info(f"  Gain change: {metrics['gain_change_db']:.2f} dB")
                logger.info(f"  Peak change: {metrics['peak_change_db']:.2f} dB")

                # Save output
                self.save_audio(stage_audio, sample_rate, f"{stage_idx:02d}_{stage_name}.wav")

                # Save stage metadata
                stage_info = {
                    "stage_name": stage_name,
                    "stage_index": stage_idx,
                    "metrics": metrics,
                    "metadata": stage_metadata,
                }
                stage_results.append(stage_info)

                # Update for next stage
                current_audio = stage_audio

            # Final output
            logger.info("\n" + "=" * 80)
            logger.info("FINAL PROCESSING COMPLETE")
            logger.info("=" * 80)

            final_metrics = self._calculate_metrics(audio_data, current_audio, sample_rate)
            logger.info(f"Overall RMS change: {final_metrics['gain_change_db']:.2f} dB")
            logger.info(f"Overall peak change: {final_metrics['peak_change_db']:.2f} dB")

            self.save_audio(current_audio, sample_rate, "99_final_output.wav")

            # Save complete results
            self.save_metadata(
                {
                    "stage_results": stage_results,
                    "final_metrics": final_metrics,
                    "success": True,
                },
                "processing_results.json",
            )

            logger.info(f"\n✓ Test complete! Results saved to: {self.run_output_dir}")
            return True

        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.save_metadata({"error": str(e), "success": False}, "processing_results.json")
            return False

    def _process_single_stage(
        self, processor: AudioPipelineProcessor, audio_data: np.ndarray, stage_name: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Process audio through a single stage"""
        # Get the stage processor method
        stage_method = getattr(processor, f"_apply_{stage_name}", None)

        if stage_method is None:
            logger.warning(f"Stage {stage_name} not found, skipping")
            return audio_data, {"skipped": True}

        # Run the stage
        try:
            processed_audio = stage_method(audio_data)
            metadata = {"applied": True, "stage": stage_name}
            return processed_audio, metadata
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            return audio_data, {"error": str(e)}

    def _calculate_metrics(
        self, input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int
    ) -> dict[str, Any]:
        """Calculate audio metrics"""
        input_rms = np.sqrt(np.mean(input_audio**2))
        output_rms = np.sqrt(np.mean(output_audio**2))

        input_peak = np.max(np.abs(input_audio))
        output_peak = np.max(np.abs(output_audio))

        # Avoid log(0)
        gain_change_db = 20 * np.log10(output_rms / max(input_rms, 1e-10))
        peak_change_db = 20 * np.log10(output_peak / max(input_peak, 1e-10))

        return {
            "input_rms": float(input_rms),
            "output_rms": float(output_rms),
            "input_peak": float(input_peak),
            "output_peak": float(output_peak),
            "gain_change_db": float(gain_change_db),
            "peak_change_db": float(peak_change_db),
            "sample_rate": sample_rate,
        }


def list_presets():
    """List available audio processing presets"""
    config_manager = create_audio_config_manager()
    presets = config_manager.get_available_presets()

    print("\n" + "=" * 80)
    print("AVAILABLE AUDIO PROCESSING PRESETS")
    print("=" * 80 + "\n")

    for preset in presets:
        print(f"  • {preset}")

    print("\nTo use a preset, create a config file:")
    print('  {"preset_name": "meeting_optimized"}')
    print()


def create_example_config():
    """Create example configuration file"""
    example_config = {
        "preset_name": "meeting_optimized",
        "enabled_stages": [
            "vad",
            "voice_filter",
            "noise_reduction",
            "voice_enhancement",
            "equalizer",
            "lufs_normalization",
            "agc",
            "compression",
            "limiter",
        ],
        "sample_rate": 16000,
        "quality": "high",
    }

    config_path = Path("./test_config.json")
    with open(config_path, "w") as f:
        json.dump(example_config, f, indent=2)

    print(f"Created example config: {config_path}")
    print("Edit this file to customize your test configuration")


def main():
    parser = argparse.ArgumentParser(
        description="Test audio processing pipeline with configurable stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", "-c", help="Path to configuration JSON file", default=None)

    parser.add_argument(
        "--input",
        "-i",
        help="Path to input audio file (default: first file in ./input)",
        default=None,
    )

    parser.add_argument("--input-dir", help="Input directory (default: ./input)", default="./input")

    parser.add_argument(
        "--output-dir", help="Output directory (default: ./output)", default="./output"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available audio processing presets",
    )

    parser.add_argument(
        "--create-example",
        action="store_true",
        help="Create example configuration file",
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_presets:
        list_presets()
        return

    if args.create_example:
        create_example_config()
        return

    # Run test
    tester = AudioPipelineTester(
        config_path=args.config, input_dir=args.input_dir, output_dir=args.output_dir
    )

    success = tester.run_test(input_file=args.input)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
