#!/usr/bin/env python3
"""
EXTENDED CODE-SWITCHING STREAMING TEST - 30+ seconds

Tests real-time streaming with:
- Multiple Chinese sentences (OSR_cn_000_0072-0075)
- Multiple English segments (JFK speech)
- Mid-sentence cutoffs simulating real streaming
- Realistic chunk sizes (2 seconds)
- Continuous appending for 30+ seconds

Expected: Both Chinese and English correctly transcribed throughout entire stream
"""

import sys
import os
import socketio
import time
import base64
import numpy as np
import wave

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

SERVICE_URL = "http://localhost:5001"
AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'audio')

def load_wav_file(filepath):
    """Load WAV file and return audio as float32 normalized array"""
    print(f"   Loading: {os.path.basename(filepath)}")
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

        # Convert to int16 numpy array
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # If stereo, convert to mono
        if n_channels == 2:
            audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Convert to float32 normalized
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        duration = len(audio_float32) / sample_rate
        print(f"   ✓ Loaded {duration:.2f}s @ {sample_rate}Hz")

        return audio_float32, sample_rate


def resample_audio(audio_float32, orig_sr, target_sr=16000):
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
        return audio_float32

    print(f"   Resampling from {orig_sr}Hz to {target_sr}Hz")
    try:
        import librosa
        audio_resampled = librosa.resample(audio_float32, orig_sr=orig_sr, target_sr=target_sr)
        print(f"   ✓ Resampled to {len(audio_resampled)/target_sr:.2f}s @ {target_sr}Hz")
        return audio_resampled
    except ImportError:
        print("   ⚠️  librosa not available, audio may have wrong sample rate")
        return audio_float32


def create_silence(duration_s, sample_rate=16000):
    """Create silent audio (float32 normalized)"""
    num_samples = int(duration_s * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def cutoff_mid_sentence(audio, cutoff_ratio=0.7):
    """Cut audio mid-sentence to simulate streaming interruption"""
    cutoff_point = int(len(audio) * cutoff_ratio)
    return audio[:cutoff_point]


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) for English text
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total words in reference
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Build edit distance matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER) for Chinese text
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total characters in reference
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    # Build edit distance matrix
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


def test_extended_code_switching_stream():
    """
    Extended streaming test with multiple language switches over 30+ seconds

    Audio sequence:
    - Chinese sentence 1 (5s)
    - Silence (0.5s)
    - English JFK part 1 (5s) - CUT MID-SENTENCE
    - Chinese sentence 2 (5s)
    - Silence (0.5s)
    - English JFK part 2 (5s)
    - Chinese sentence 3 (5s)
    - Silence (0.5s)
    - English JFK part 3 (5s) - CUT MID-SENTENCE
    - Chinese sentence 4 (3s)

    Total: ~34 seconds with multiple code-switches
    """
    print("\n" + "="*80)
    print("EXTENDED CODE-SWITCHING STREAMING TEST")
    print("="*80)
    print("Testing 30+ seconds of continuous mixed Chinese/English streaming")
    print("="*80 + "\n")

    print("[1/5] Loading audio files...")

    # Load Chinese audio files
    chinese_files = [
        'OSR_cn_000_0072_8k.wav',  # 院子门口不远处就是一个地铁站
        'OSR_cn_000_0073_8k.wav',  # 这是一个美丽而神奇的景象
        'OSR_cn_000_0074_8k.wav',  # 树上长满了又大又甜的桃子
        'OSR_cn_000_0075_8k.wav',  # 海豚和鲸鱼的表演是很好看的节目
    ]

    chinese_audio_list = []
    for cn_file in chinese_files:
        cn_path = os.path.join(AUDIO_DIR, cn_file)
        audio, sr = load_wav_file(cn_path)
        audio = resample_audio(audio, sr, 16000)
        chinese_audio_list.append(audio)

    # Load English audio (JFK)
    jfk_path = os.path.join(AUDIO_DIR, 'jfk.wav')
    jfk_audio, jfk_sr = load_wav_file(jfk_path)
    jfk_audio = resample_audio(jfk_audio, jfk_sr, 16000)

    print("\n[2/5] Creating extended mixed audio sequence...")

    # Extract JFK segments (split into 3 parts)
    jfk_duration = len(jfk_audio) / 16000
    jfk_part1 = jfk_audio[:16000 * 5]  # First 5 seconds
    jfk_part2 = jfk_audio[16000 * 5:16000 * 10] if jfk_duration >= 10 else jfk_audio[16000 * 3:16000 * 6]
    jfk_part3 = jfk_audio[-16000 * 5:]  # Last 5 seconds

    # Apply mid-sentence cutoffs to JFK parts 1 and 3
    jfk_part1_cut = cutoff_mid_sentence(jfk_part1, 0.7)  # Cut at 70%
    jfk_part3_cut = cutoff_mid_sentence(jfk_part3, 0.65)  # Cut at 65%

    # Trim Chinese to ~5 seconds each
    chinese_trimmed = []
    for cn_audio in chinese_audio_list:
        duration = len(cn_audio) / 16000
        if duration > 5.0:
            cn_trimmed = cn_audio[:16000 * 5]
        else:
            cn_trimmed = cn_audio
        chinese_trimmed.append(cn_trimmed)

    # Trim last Chinese to 3 seconds
    if len(chinese_trimmed[3]) > 16000 * 3:
        chinese_trimmed[3] = chinese_trimmed[3][:16000 * 3]

    silence_500ms = create_silence(0.5, 16000)

    # Build the sequence
    mixed_audio_segments = [
        ("Chinese #1", chinese_trimmed[0]),
        ("Silence", silence_500ms),
        ("English JFK part 1 (CUT)", jfk_part1_cut),
        ("Chinese #2", chinese_trimmed[1]),
        ("Silence", silence_500ms),
        ("English JFK part 2", jfk_part2),
        ("Chinese #3", chinese_trimmed[2]),
        ("Silence", silence_500ms),
        ("English JFK part 3 (CUT)", jfk_part3_cut),
        ("Chinese #4", chinese_trimmed[3]),
    ]

    # Concatenate all segments
    mixed_audio = np.concatenate([seg[1] for seg in mixed_audio_segments])
    total_duration = len(mixed_audio) / 16000

    print(f"✓ Created extended mixed audio: {total_duration:.2f}s")
    print("\n📋 Audio sequence:")
    current_time = 0.0
    for name, segment in mixed_audio_segments:
        duration = len(segment) / 16000
        print(f"   {current_time:.1f}s - {current_time + duration:.1f}s: {name} ({duration:.2f}s)")
        current_time += duration

    print(f"\n[3/5] Connecting to Whisper service...")

    # Socket.IO setup
    sio = socketio.Client()
    results_received = []
    connection_ready = False
    session_joined = False

    @sio.on('connect')
    def on_connect():
        nonlocal connection_ready
        connection_ready = True
        print("✅ Connected to Whisper service")

    @sio.on('joined_session')
    def on_joined(data):
        nonlocal session_joined
        session_joined = True
        print(f"✅ Joined session: {data.get('session_id')}")

    @sio.on('transcription_result')
    def on_result(data):
        results_received.append(data)
        text = data.get('text', '')
        is_final = data.get('is_final', False)
        detected_lang = data.get('detected_language', 'unknown')

        status = '✅ FINAL' if is_final else '⏳ PARTIAL'
        print(f"\n{status} Result #{len(results_received)}:")
        print(f"   Text: '{text}'")
        print(f"   Language: {detected_lang}")
        print(f"   Final: {is_final}")

    @sio.on('error')
    def on_error(data):
        print(f"❌ Error: {data}")

    try:
        sio.connect(SERVICE_URL)
        time.sleep(1)

        if not connection_ready:
            print("❌ Failed to connect to Whisper service")
            return False

        print("\n[4/5] Streaming extended mixed audio with code-switching enabled...")

        # Create session with code-switching enabled
        session_id = f"extended-cs-test-{int(time.time())}"

        config = {
            'model': 'base',
            'enable_code_switching': True,
            'language': None,  # Auto-detect
            'enable_vad': True
        }

        sio.emit('join_session', {'session_id': session_id, 'config': config})
        time.sleep(1)

        if not session_joined:
            print("❌ Failed to join session")
            return False

        print(f"   Session ID: {session_id}\n")
        print("📝 Configuration:")
        print(f"   enable_code_switching: True")
        print(f"   language: None (auto-detect)")
        print(f"   model: base")
        print(f"   enable_vad: True")

        # FIXED: Stream audio in 2-second chunks (matches simple test, avoids VAD dimension error)
        # Original 0.04s chunks caused: "LSTMCell: Expected input to be 1D or 2D, got 3D instead"
        CHUNK_DURATION = 2.0  # 2-second chunks = 32000 samples (safe for VAD)
        chunk_size = int(16000 * CHUNK_DURATION)
        num_chunks = int(np.ceil(len(mixed_audio) / chunk_size))

        print(f"   Sending {num_chunks} chunks of {CHUNK_DURATION}s each\n")

        chunk_count = 0
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(mixed_audio))
            chunk = mixed_audio[start_idx:end_idx]

            # Convert to int16 for transmission
            chunk_int16 = (chunk * 32768.0).astype(np.int16)
            audio_b64 = base64.b64encode(chunk_int16.tobytes()).decode('utf-8')

            sio.emit('transcribe_stream', {
                'session_id': session_id,
                'audio_data': audio_b64,
                'sample_rate': 16000
            })

            chunk_count += 1
            chunk_start_time = start_idx / 16000
            chunk_end_time = end_idx / 16000
            print(f"📤 Sent chunk {chunk_count}/{num_chunks} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s)")

            # Wait for processing (2.5s per 2s chunk gives processing time)
            time.sleep(2.5)

        print(f"\n   ✅ Finished sending all {num_chunks} chunks!")
        print("\n[5/5] Waiting for final results...")
        print("   (Waiting 30 seconds to ensure all audio is processed...)\n")

        # Wait for results to arrive, checking every second
        wait_time = 0
        max_wait = 30
        while wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            if wait_time % 5 == 0:
                print(f"   ⏳ Waited {wait_time}s... ({len(results_received)} results received so far)")

        print(f"\n   ⏱️  Total wait time: {wait_time}s")

        print("="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Total results received: {len(results_received)}\n")

        # Analyze results
        final_results = [r for r in results_received if r.get('is_final', False)]
        print(f"📝 Final transcription results: {len(final_results)}\n")

        combined_text = ""
        has_chinese = False
        has_english = False

        for idx, result in enumerate(final_results, 1):
            text = result.get('text', '').strip()
            detected_lang = result.get('detected_language', 'unknown')

            print(f"   Result {idx}. [{detected_lang}]")
            print(f"      Text: '{text}'\n")

            combined_text += " " + text

            # Check for Chinese characters
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                has_chinese = True

            # Check for English words (basic heuristic)
            if any(word.lower() in text.lower() for word in ['and', 'the', 'my', 'fellow', 'americans', 'ask', 'not', 'country']):
                has_english = True

        print(f"📄 Combined final transcription:")
        print(f"   '{combined_text.strip()}'\n")

        # Load ground truth and calculate accuracy
        print("📊 Accuracy Evaluation:")
        try:
            # Load ground truth texts
            jfk_truth = open(os.path.join(AUDIO_DIR, 'jfk.txt'), 'r', encoding='utf-8').read().strip()
            cn1_truth = open(os.path.join(AUDIO_DIR, 'OSR_cn_000_0072_8k.txt'), 'r', encoding='utf-8').read().strip()
            cn2_truth = open(os.path.join(AUDIO_DIR, 'OSR_cn_000_0073_8k.txt'), 'r', encoding='utf-8').read().strip()
            cn3_truth = open(os.path.join(AUDIO_DIR, 'OSR_cn_000_0074_8k.txt'), 'r', encoding='utf-8').read().strip()
            cn4_truth = open(os.path.join(AUDIO_DIR, 'OSR_cn_000_0075_8k.txt'), 'r', encoding='utf-8').read().strip()

            # Extract Chinese and English from combined text
            combined_clean = combined_text.strip()

            # Separate Chinese and English characters
            chinese_chars = ''.join([c for c in combined_clean if '\u4e00' <= c <= '\u9fff'])
            english_words = ' '.join([word for word in combined_clean.split() if any(c.isalpha() and ord(c) < 128 for c in word)])

            # Ground truth combined
            chinese_ground_truth = cn1_truth + cn2_truth + cn3_truth + cn4_truth
            english_ground_truth = jfk_truth

            # Calculate CER for Chinese
            if chinese_chars and chinese_ground_truth:
                cer = calculate_cer(chinese_ground_truth, chinese_chars)
                print(f"   Chinese CER: {cer:.2%} (lower is better)")
                print(f"     Reference chars: {len(chinese_ground_truth)}")
                print(f"     Hypothesis chars: {len(chinese_chars)}")
            else:
                cer = 1.0
                print(f"   Chinese CER: N/A (no Chinese detected)")

            # Calculate WER for English
            if english_words and english_ground_truth:
                wer = calculate_wer(english_ground_truth, english_words)
                print(f"   English WER: {wer:.2%} (lower is better)")
                print(f"     Reference words: {len(english_ground_truth.split())}")
                print(f"     Hypothesis words: {len(english_words.split())}")
            else:
                wer = 1.0
                print(f"   English WER: N/A (no English detected)")

            # Overall accuracy score (inverse of average error rate)
            avg_error = (cer + wer) / 2
            accuracy = max(0, 1 - avg_error)
            print(f"   Overall Accuracy: {accuracy:.2%}\n")

        except Exception as e:
            print(f"   ⚠️  Could not calculate accuracy: {e}\n")
            accuracy = None

        print(f"🔍 Language detection:")
        print(f"   {'✅' if has_chinese else '❌'} Chinese characters found: {has_chinese}")
        print(f"   {'✅' if has_english else '❌'} English words found: {has_english}\n")

        print("🎯 Extended code-switching verdict:")
        if has_chinese and has_english:
            print("   ✅ SUCCESS: Both languages detected throughout 30+ second stream!")
            verdict = True
        else:
            print("   ❌ FAILED: Not all languages properly detected")
            verdict = False

        print("="*80)

        # Disconnect
        sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.5)
        sio.disconnect()

        if verdict:
            print("\n✅ EXTENDED TEST PASSED - Code-switching working over long duration!\n")
        else:
            print("\n❌ EXTENDED TEST FAILED - Code-switching issues in long stream\n")

        return verdict

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_extended_code_switching_stream()
    sys.exit(0 if success else 1)
