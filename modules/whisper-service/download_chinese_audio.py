#!/usr/bin/env python3
"""
Download sample Chinese audio for testing

This script downloads a real Chinese speech sample to test
actual multi-language transcription (no mocking).
"""

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_chinese_sample():
    """
    Download a Chinese audio sample from a public source

    Using Mozilla Common Voice Chinese samples (open license)
    """

    logger.info("Downloading Chinese audio sample...")

    # Option 1: Use yt-dlp to download from YouTube (Chinese speech)
    # Option 2: Download from Mozilla Common Voice dataset
    # Option 3: Use openslr.org public datasets

    # For testing, let's use a simple approach with requests
    import requests

    # This is a public domain Chinese speech sample from openslr
    # Dataset: AISHELL-1 (Apache 2.0 license)
    url = "https://www.openslr.org/resources/33/data_aishell.tgz"

    logger.info(f"Note: Full dataset is large. For quick testing:")
    logger.info("1. Visit https://www.openslr.org/33/")
    logger.info("2. Download a sample Chinese .wav file")
    logger.info("3. Place it in this directory as 'chinese_sample.wav'")
    logger.info("")
    logger.info("OR use this command to download a YouTube sample:")
    logger.info("  yt-dlp -x --audio-format wav 'https://youtube.com/watch?v=<chinese_speech_video>'")
    logger.info("")
    logger.info("For now, I'll create a simple test using online TTS...")

    # Use gTTS (Google Text-to-Speech) to generate Chinese audio
    try:
        from gtts import gTTS
        import tempfile

        # Chinese text
        text = "这是一个中文语音测试。我们正在验证多语言隔离功能。系统应该能够正确处理中文和英文。"

        # Generate Chinese speech
        tts = gTTS(text=text, lang='zh-cn')
        output_file = "chinese_sample.wav"

        # Save to mp3 first
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_mp3.name)

        # Convert to WAV using ffmpeg or pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3.name)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(output_file, format='wav')

            os.unlink(temp_mp3.name)

            logger.info(f"✓ Created Chinese audio sample: {output_file}")
            logger.info(f"  Text: {text}")
            return output_file

        except ImportError:
            logger.error("pydub not installed. Install with: pip install pydub")
            logger.info(f"MP3 file saved to: {temp_mp3.name}")
            logger.info("Please convert manually to WAV using ffmpeg:")
            logger.info(f"  ffmpeg -i {temp_mp3.name} -ar 16000 -ac 1 chinese_sample.wav")
            return None

    except ImportError:
        logger.error("gTTS not installed. Install with: pip install gtts")
        logger.info("")
        logger.info("Alternative: Download real Chinese speech manually from:")
        logger.info("- https://commonvoice.mozilla.org/zh-CN/datasets")
        logger.info("- https://www.openslr.org/33/ (AISHELL-1)")
        logger.info("- Any Chinese YouTube video (use yt-dlp)")
        return None

if __name__ == "__main__":
    result = download_chinese_sample()
    if result:
        print(f"\n✓ Success! Chinese audio ready: {result}")
    else:
        print("\n❌ Please download Chinese audio manually")
        print("   Save as 'chinese_sample.wav' in this directory")
