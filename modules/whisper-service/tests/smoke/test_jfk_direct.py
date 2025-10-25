#!/usr/bin/env python3
"""
DIRECT JFK AUDIO FILE TEST

This test bypasses all Socket.IO complexity and directly:
1. Checks JFK file format and properties
2. Loads JFK audio file
3. Calls Whisper transcribe() method directly
4. Compares with streaming results

This will help isolate whether the issue is:
- The JFK audio file itself
- The streaming/chunking pipeline
- The Socket.IO transmission
"""

import wave
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whisper_service import WhisperService

JFK_AUDIO_PATH = "/Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service/jfk.wav"


def check_jfk_file_properties():
    """Check JFK file format and properties"""
    print("\n" + "="*80)
    print("CHECKING JFK FILE PROPERTIES")
    print("="*80)

    if not os.path.exists(JFK_AUDIO_PATH):
        print(f"❌ JFK file not found: {JFK_AUDIO_PATH}")
        return None

    try:
        with wave.open(JFK_AUDIO_PATH, 'rb') as wav_file:
            # Get file properties
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / frame_rate

            print(f"✅ File exists: {JFK_AUDIO_PATH}")
            print(f"   Channels: {n_channels} ({'stereo' if n_channels == 2 else 'mono'})")
            print(f"   Sample Width: {sample_width} bytes ({sample_width * 8} bits)")
            print(f"   Sample Rate: {frame_rate} Hz")
            print(f"   Frames: {n_frames}")
            print(f"   Duration: {duration:.2f} seconds")

            # Read audio data
            audio_bytes = wav_file.readframes(n_frames)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0

            # If stereo, convert to mono
            if n_channels == 2:
                audio_float = audio_float.reshape(-1, 2).mean(axis=1)
                print(f"   Converted stereo to mono")

            # Check audio statistics
            max_amplitude = np.abs(audio_float).max()
            rms = np.sqrt(np.mean(audio_float**2))

            print(f"\n📊 Audio Statistics:")
            print(f"   Max Amplitude: {max_amplitude:.4f}")
            print(f"   RMS Level: {rms:.4f}")
            print(f"   Peak dB: {20 * np.log10(max_amplitude):.1f} dB")
            print(f"   RMS dB: {20 * np.log10(rms):.1f} dB")

            # Check if audio is silent
            if max_amplitude < 0.001:
                print(f"   ⚠️ WARNING: Audio appears to be silent or very quiet!")
            else:
                print(f"   ✅ Audio has good signal level")

            # Check sample rate compatibility
            if frame_rate != 16000:
                print(f"\n⚠️ Sample rate is {frame_rate}Hz, Whisper expects 16kHz")
                print(f"   Will need resampling")
            else:
                print(f"\n✅ Sample rate is correct (16kHz)")

            return audio_float, frame_rate

    except Exception as e:
        print(f"❌ Error reading JFK file: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_direct_transcription(jfk_audio):
    """Test direct Whisper transcription without streaming using jfk_audio fixture"""
    print("\n" + "="*80)
    print("DIRECT TRANSCRIPTION TEST")
    print("="*80)

    # Get audio data and sample rate from fixture
    audio_data, sample_rate = jfk_audio

    try:
        # Initialize Whisper service (model already preloaded)
        print("\n🔧 Initializing Whisper service...")
        service = WhisperService()
        print("✅ Service initialized (model already preloaded)")

        print(f"\n📊 Audio info: {len(audio_data)} samples @ {sample_rate}Hz = {len(audio_data)/sample_rate:.2f}s")

        # JFK audio fixture is already 16kHz, no resampling needed
        if sample_rate != 16000:
            print(f"\n🔄 Resampling from {sample_rate}Hz to 16000Hz...")
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
            sample_rate = 16000
            print(f"✅ Resampled to 16kHz: {len(audio_data)} samples, {len(audio_data)/16000:.2f}s")

        # Transcribe directly using internal safe_inference method
        print("\n🎙️ Transcribing entire audio file directly...")
        print(f"   Audio: {len(audio_data)} samples, {len(audio_data)/sample_rate:.2f}s")
        print(f"   Using: model_manager.safe_inference() (same as streaming)")

        result = service.model_manager.safe_inference(
            model_name="large-v3-turbo",
            audio_data=audio_data,
            beam_size=5,
            initial_prompt=None,
            language="en",
            temperature=0.0,
            streaming_policy="alignatt",
            task="transcribe",
            target_language="en"
        )

        print("\n" + "="*80)
        print("DIRECT TRANSCRIPTION RESULT")
        print("="*80)

        if result:
            print(f"✅ Transcription successful!")

            # Handle dict or object result
            if isinstance(result, dict):
                text = result.get('text', '')
                language = result.get('language', 'unknown')
                segments = result.get('segments', [])
            else:
                text = getattr(result, 'text', '')
                language = getattr(result, 'language', 'unknown')
                segments = getattr(result, 'segments', [])

            print(f"\n📝 Text: '{text}'")
            print(f"   Language: {language}")

            if segments:
                print(f"\n📊 Segments: {len(segments)}")
                for i, seg in enumerate(segments[:5]):  # Show first 5
                    if isinstance(seg, dict):
                        print(f"   {i+1}. [{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] '{seg.get('text', '')}'")
                    else:
                        print(f"   {i+1}. [{getattr(seg, 'start', 0):.1f}s - {getattr(seg, 'end', 0):.1f}s] '{getattr(seg, 'text', '')}'")

            # Compare with expected JFK speech
            expected_keywords = ["fellow", "americans", "ask", "country"]
            found_keywords = [kw for kw in expected_keywords if kw.lower() in text.lower()]

            print(f"\n🔍 Expected JFK Keywords Check:")
            print(f"   Expected: {expected_keywords}")
            print(f"   Found: {found_keywords}")

            if len(found_keywords) >= 2:
                print(f"   ✅ JFK speech detected! Found {len(found_keywords)}/{len(expected_keywords)} keywords")
                return True
            else:
                print(f"   ❌ JFK speech NOT detected. Found only {len(found_keywords)}/{len(expected_keywords)} keywords")
                print(f"   ⚠️ This means the issue is NOT in the streaming pipeline!")
                print(f"   ⚠️ The JFK file itself may not contain the expected speech")
                return False
        else:
            print("❌ Transcription returned None")
            return False

    except Exception as e:
        print(f"❌ Direct transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run direct JFK audio test"""
    print("\n" + "="*80)
    print("JFK AUDIO DIRECT TEST")
    print("Bypassing Socket.IO - Testing file directly")
    print("="*80)

    # Step 1: Check file properties
    result = check_jfk_file_properties()
    if result is None:
        return 1

    audio_data, sample_rate = result

    # Step 2: Test direct transcription
    test_direct_transcription(audio_data, sample_rate)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. If direct transcription works → Issue is in streaming pipeline")
    print("2. If direct transcription fails → Issue is with JFK audio file")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
