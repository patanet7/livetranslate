#!/usr/bin/env python3
"""
Debug: Why is mixed audio only transcribing Chinese?

This investigates why concatenated Chinese + English audio
only produces Chinese transcription.
"""

import asyncio
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import logging

from whisper_service import TranscriptionRequest, WhisperService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AUDIO_DIR = "tests/audio"


def load_and_resample(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio"""
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


async def debug_mixed_audio():
    """Debug mixed audio transcription"""

    logger.info("=" * 80)
    logger.info("DEBUG: Mixed Audio Transcription Issue")
    logger.info("=" * 80)

    # Load audio
    chinese_audio = load_and_resample(f"{AUDIO_DIR}/OSR_cn_000_0072_8k.wav")
    english_audio = load_and_resample(f"{AUDIO_DIR}/jfk.wav")

    # Create clips
    chinese_clip = chinese_audio[: (5 * 16000)]
    english_clip = english_audio[: (5 * 16000)]
    silence = np.zeros(int(0.5 * 16000), dtype=np.float32)

    # Save individual pieces for inspection
    logger.info("\nSaving individual audio pieces for verification...")
    sf.write("debug_chinese_clip.wav", chinese_clip, 16000)
    sf.write("debug_english_clip.wav", english_clip, 16000)
    sf.write("debug_silence.wav", silence, 16000)

    logger.info(f"✓ Saved debug_chinese_clip.wav ({len(chinese_clip)/16000:.2f}s)")
    logger.info(f"✓ Saved debug_english_clip.wav ({len(english_clip)/16000:.2f}s)")
    logger.info(f"✓ Saved debug_silence.wav ({len(silence)/16000:.2f}s)")

    # Concatenate
    mixed_audio = np.concatenate([chinese_clip, silence, english_clip])
    sf.write("debug_mixed.wav", mixed_audio, 16000)

    logger.info(f"\n✓ Saved debug_mixed.wav ({len(mixed_audio)/16000:.2f}s)")
    logger.info("  Structure:")
    logger.info("    0.0s - 5.0s: Chinese")
    logger.info("    5.0s - 5.5s: Silence")
    logger.info("    5.5s - 10.5s: English")

    # Initialize service
    service = WhisperService(
        {
            "models_dir": ".models",
            "default_model": "base",
            "sample_rate": 16000,
            "orchestration_mode": False,
        }
    )
    await asyncio.sleep(2)

    # Test 1: Transcribe ONLY the English clip
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: English clip ALONE (no code-switching)")
    logger.info("=" * 80)

    request_en_only = TranscriptionRequest(
        audio_data=english_clip,
        model_name="base",
        language="en",
        session_id="test-en-only",
        sample_rate=16000,
        streaming=False,
        enable_code_switching=False,
    )

    result_en_only = await service.transcribe(request_en_only)
    logger.info("\nEnglish clip alone:")
    logger.info(f"  Text: {result_en_only.text}")
    logger.info(
        f"  Segments: {len(result_en_only.segments) if hasattr(result_en_only, 'segments') else 'N/A'}"
    )

    # Test 2: Transcribe mixed audio WITH language pinned to English
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Mixed audio with language='en' (pinned)")
    logger.info("=" * 80)

    request_mixed_en = TranscriptionRequest(
        audio_data=mixed_audio,
        model_name="base",
        language="en",  # Force English
        session_id="test-mixed-en",
        sample_rate=16000,
        streaming=False,
        enable_code_switching=False,
    )

    result_mixed_en = await service.transcribe(request_mixed_en)
    logger.info("\nMixed audio (forced English):")
    logger.info(f"  Text: {result_mixed_en.text}")

    if hasattr(result_mixed_en, "segments") and result_mixed_en.segments:
        logger.info(f"  Segments ({len(result_mixed_en.segments)}):")
        for i, seg in enumerate(result_mixed_en.segments[:10], 1):
            if isinstance(seg, dict):
                logger.info(f"    [{i}] {seg['start']:.1f}s-{seg['end']:.1f}s: {seg['text']}")

    # Test 3: Mixed audio with NO language (auto-detect) but NO code-switching
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Mixed audio with language=None (auto-detect, no code-switching flag)")
    logger.info("=" * 80)

    request_mixed_auto = TranscriptionRequest(
        audio_data=mixed_audio,
        model_name="base",
        language=None,  # Auto-detect
        session_id="test-mixed-auto",
        sample_rate=16000,
        streaming=False,
        enable_code_switching=False,  # No code-switching
    )

    result_mixed_auto = await service.transcribe(request_mixed_auto)
    logger.info("\nMixed audio (auto-detect, no code-switching):")
    logger.info(f"  Text: {result_mixed_auto.text}")

    if hasattr(result_mixed_auto, "segments") and result_mixed_auto.segments:
        logger.info(f"  Segments ({len(result_mixed_auto.segments)}):")
        for i, seg in enumerate(result_mixed_auto.segments[:10], 1):
            if isinstance(seg, dict):
                logger.info(f"    [{i}] {seg['start']:.1f}s-{seg['end']:.1f}s: {seg['text']}")

    # Test 4: Mixed audio WITH code-switching
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Mixed audio with code-switching enabled")
    logger.info("=" * 80)

    request_mixed_cs = TranscriptionRequest(
        audio_data=mixed_audio,
        model_name="base",
        language=None,
        session_id="test-mixed-cs",
        sample_rate=16000,
        streaming=False,
        enable_code_switching=True,  # Code-switching enabled
    )

    result_mixed_cs = await service.transcribe(request_mixed_cs)
    logger.info("\nMixed audio (code-switching enabled):")
    logger.info(f"  Text: {result_mixed_cs.text}")

    if hasattr(result_mixed_cs, "segments") and result_mixed_cs.segments:
        logger.info(f"  Segments ({len(result_mixed_cs.segments)}):")
        for i, seg in enumerate(result_mixed_cs.segments[:10], 1):
            if isinstance(seg, dict):
                text = seg.get("text", "")
                lang = seg.get("detected_language", "N/A")
                conf = seg.get("language_confidence", 0.0)
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                logger.info(f"    [{i}] {start:.1f}s-{end:.1f}s | {lang} ({conf:.2f}): {text}")

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    logger.info("\n🔍 Observations:")

    # Check if English appears in any transcription
    def has_english_words(text):
        return any(
            word in text.lower() for word in ["and", "so", "fellow", "american", "ask", "country"]
        )

    logger.info(
        f"\n1. English clip alone has English words: {has_english_words(result_en_only.text)}"
    )
    logger.info(f"   Text: {result_en_only.text[:100]}")

    logger.info(
        f"\n2. Mixed (forced EN) has English words: {has_english_words(result_mixed_en.text)}"
    )
    logger.info(f"   Text: {result_mixed_en.text[:100]}")

    logger.info(
        f"\n3. Mixed (auto-detect) has English words: {has_english_words(result_mixed_auto.text)}"
    )
    logger.info(f"   Text: {result_mixed_auto.text[:100]}")

    logger.info(
        f"\n4. Mixed (code-switching) has English words: {has_english_words(result_mixed_cs.text)}"
    )
    logger.info(f"   Text: {result_mixed_cs.text[:100]}")

    # Check segment timestamps
    logger.info("\n📊 Segment Timeline Analysis:")

    for name, result in [
        ("Forced EN", result_mixed_en),
        ("Auto-detect", result_mixed_auto),
        ("Code-switching", result_mixed_cs),
    ]:
        if hasattr(result, "segments") and result.segments:
            last_seg = result.segments[-1] if isinstance(result.segments[-1], dict) else None
            if last_seg:
                last_time = last_seg.get("end", 0)
                logger.info(f"  {name}: Last segment ends at {last_time:.1f}s (total audio: 10.5s)")
                if last_time < 6.0:
                    logger.warning(
                        f"    ⚠️  Transcription stopped early! Only {last_time:.1f}s of 10.5s processed"
                    )
                else:
                    logger.info("    ✓ Transcription reached English portion (5.5s+)")

    logger.info("\n💡 HYPOTHESIS:")
    logger.info("  If transcription stops before 5.5s:")
    logger.info("    → Whisper may be detecting end-of-speech too early")
    logger.info("    → The silence gap (5.0s-5.5s) may trigger early termination")
    logger.info("    → VAD or end-of-speech detection is too aggressive")
    logger.info("\n  Possible fixes:")
    logger.info("    1. Reduce silence gap (0.5s → 0.1s)")
    logger.info("    2. Disable VAD for code-switching mode")
    logger.info("    3. Use longer audio context to force full transcription")
    logger.info("    4. Check if 'task=transcribe' is causing early stops")

    # Cleanup
    service.close_session("test-en-only")
    service.close_session("test-mixed-en")
    service.close_session("test-mixed-auto")
    service.close_session("test-mixed-cs")

    logger.info("\n" + "=" * 80)
    logger.info("✅ DEBUG COMPLETE - Check debug_*.wav files for audio verification")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    exit(0 if asyncio.run(debug_mixed_audio()) else 1)
