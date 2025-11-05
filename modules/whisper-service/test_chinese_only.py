#!/usr/bin/env python3
"""
Quick test: Chinese-only audio WITHOUT explicit LID configuration
This mimics test_english_only_no_switch to understand original behavior
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from session_restart import SessionRestartTranscriber

logging.basicConfig(
    level=logging.INFO,  # Back to INFO level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected Chinese text from OSR_cn_000_0072_8k.wav
EXPECTED_CHINESE = '院子门口不远处就是一个地铁站 这是一个美丽而神奇的景象 树上长满了又大又甜的桃子 海豚和鲸鱼的表演是很好看的节目'

# Load Chinese audio
audio_path = Path(__file__).parent / "tests" / "audio" / "OSR_cn_000_0072_8k.wav"
if not audio_path.exists():
    logger.error(f"Audio not found: {audio_path}")
    sys.exit(1)

audio, sr = sf.read(str(audio_path))
logger.info(f"Loaded: {audio_path.name} ({len(audio)/sr:.2f}s, {sr}Hz)")

# Resample to 16kHz
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000
    logger.info(f"Resampled to 16kHz")

# Initialize transcriber WITHOUT explicit LID parameters (like test_english_only_no_switch)
logger.info("\n" + "="*80)
logger.info("TEST: Chinese-only WITHOUT explicit LID config")
logger.info("="*80)

models_dir = Path.home() / ".whisper" / "models"
model_path = str(models_dir / "large-v3-turbo.pt")

logger.info("Creating transcriber WITHOUT explicit LID parameters...")
transcriber = SessionRestartTranscriber(
    model_path=model_path,
    models_dir=str(models_dir),
    target_languages=['en', 'zh'],  # Both languages available
    sampling_rate=16000
    # NO lid_hop_ms, confidence_margin, min_dwell_frames, min_dwell_ms
)

# Stream audio in 500ms chunks
chunk_duration_sec = 0.5
chunk_size_samples = int(chunk_duration_sec * sr)

logger.info(f"\nStreaming audio in 500ms chunks...")
logger.info("="*80)

all_transcriptions = []
for i in range(0, len(audio), chunk_size_samples):
    chunk = audio[i:i + chunk_size_samples]
    timestamp = i / sr

    result = transcriber.process(chunk)

    if result['text']:
        all_transcriptions.append(result['text'])
        logger.info(
            f"[{timestamp:.1f}s] [{result['language']}] "
            f"{'(punctuated)' if result['is_final'] else '(ongoing)'}: {result['text']}"
        )

    if result.get('silence_detected', False):
        logger.info(f"🛑 Sustained silence detected at {timestamp:.1f}s")
        break

# Finalize
transcriber.finalize()

# Get all segments and statistics
all_segments = transcriber._get_all_segments()
stats = transcriber.get_statistics()

logger.info("\n" + "="*80)
logger.info("RESULTS")
logger.info("="*80)

logger.info(f"Total sessions: {stats['total_sessions']}")
logger.info(f"Total switches: {stats['total_switches']}")
logger.info(f"Total segments: {len(all_segments)}")

# Show all segments
logger.info("\nAll segments:")
for seg in all_segments:
    if seg.get('text') and seg.get('text').strip():
        logger.info(f"  [{seg['language']}] {seg['start']:.1f}s: {seg['text']}")

# Full transcription
full_text = ' '.join(seg['text'].strip() for seg in all_segments if seg.get('text'))
logger.info(f"\nFull transcription:\n  '{full_text}'")
logger.info(f"\nExpected:\n  '{EXPECTED_CHINESE}'")

# Check if Chinese was detected
zh_segments = [seg for seg in all_segments if seg.get('language') == 'zh']
en_segments = [seg for seg in all_segments if seg.get('language') == 'en']

logger.info(f"\nLanguage distribution:")
logger.info(f"  Chinese segments: {len(zh_segments)}")
logger.info(f"  English segments: {len(en_segments)}")

if len(zh_segments) > 0:
    logger.info("✅ Chinese segments detected")
else:
    logger.warning("⚠️  NO Chinese segments detected - sessions created with wrong language!")

if len(en_segments) > 0 and len(zh_segments) == 0:
    logger.error("❌ PROBLEM: Only English segments for Chinese audio = WRONG SOT TOKEN!")

logger.info("\n" + "="*80)
