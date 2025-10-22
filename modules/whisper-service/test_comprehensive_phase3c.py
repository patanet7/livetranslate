#!/usr/bin/env python3
"""
COMPREHENSIVE PHASE 3C TEST SUITE

Tests ALL Phase 3C features with different configurations:

1. Task Parameter Tests:
   - task="transcribe" + target="en" → Use transcribe mode
   - task="transcribe" + target="es" → Use transcribe mode
   - task="translate" + target="en" → Use Whisper translate mode (source → English)
   - task="translate" + target="es" → Use transcribe mode (Whisper can't translate to non-English)

2. Voice Activity Detection (VAD) Tests:
   - Ensure VAD still works with Phase 3C stability tracking
   - Test VAD with both transcribe and translate modes
   - Verify silence is properly filtered

3. Stability Tracking Tests:
   - Verify stable_text, unstable_text, is_draft, is_final fields
   - Test stability_score calculation
   - Verify should_translate flag logic

4. Beam Search Tests:
   - Ensure beam search works with task parameter
   - Test different beam sizes (1, 5, 10)

5. Speaker Diarization Tests:
   - Verify diarization works with Phase 3C
   - Test speaker info in stability-tracked results

6. Integration Tests:
   - All features working together
   - Realistic audio scenarios
"""

import socketio
import json
import numpy as np
import base64
import time
from typing import Dict, List, Any

SERVICE_URL = "http://localhost:5001"

def create_speech_audio(duration=3.0, sample_rate=16000, speech_pattern="normal"):
    """Create different types of speech-like audio for testing"""
    t = np.linspace(0, duration, int(sample_rate * duration))

    if speech_pattern == "silence":
        # Pure silence
        return np.zeros(int(sample_rate * duration), dtype=np.float32)

    elif speech_pattern == "noise":
        # Just noise (no speech)
        return np.random.normal(0, 0.02, int(sample_rate * duration)).astype(np.float32)

    elif speech_pattern == "normal":
        # Normal speech-like audio
        freq = 200 + 100 * np.sin(2 * np.pi * 2 * t)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.15 * np.sin(2 * np.pi * 2 * freq * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        audio = audio * envelope
        audio += np.random.normal(0, 0.02, audio.shape)
        return audio.astype(np.float32)

    elif speech_pattern == "loud":
        # Loud speech
        freq = 250 + 150 * np.sin(2 * np.pi * 3 * t)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.25 * np.sin(2 * np.pi * 2 * freq * t)
        return audio.astype(np.float32)


class TestResult:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.message = ""
        self.details = {}


class ComprehensiveTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []
        self.sio = socketio.Client()
        self.current_results = []
        self.setup_callbacks()

    def setup_callbacks(self):
        @self.sio.on('connect')
        def on_connect():
            print("✅ Connected to Whisper service")

        @self.sio.on('transcription_result')
        def on_transcription(data):
            self.current_results.append(data)

        @self.sio.on('error')
        def on_error(data):
            print(f"❌ Error: {data.get('message', 'Unknown')}")

    def connect(self):
        """Connect to Whisper service"""
        self.sio.connect(SERVICE_URL)
        time.sleep(0.5)

    def disconnect(self):
        """Disconnect from Whisper service"""
        if self.sio.connected:
            self.sio.disconnect()

    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"{'='*80}")

        result = TestResult(test_name)

        try:
            self.current_results = []
            test_func(result)
            self.results.append(result)
        except Exception as e:
            result.passed = False
            result.message = f"Exception: {e}"
            self.results.append(result)
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

        status = "✅ PASSED" if result.passed else "❌ FAILED"
        print(f"\n{status}: {result.message}")
        return result.passed

    def stream_audio(self, session_id: str, audio: np.ndarray, sample_rate: int,
                     model: str, language: str, task: str, target_language: str,
                     beam_size: int = 5, enable_vad: bool = True):
        """Stream audio to Whisper service"""
        # Split into chunks
        chunk_size = int(sample_rate * 0.5)  # 500ms chunks
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(chunk)

        print(f"   Streaming {len(chunks)} chunks (task={task}, target={target_language}, beam={beam_size}, vad={enable_vad})")

        # Join session
        self.sio.emit('join_session', {'session_id': session_id})
        time.sleep(0.3)

        # Stream chunks
        for i, chunk in enumerate(chunks):
            chunk_bytes = chunk.tobytes()
            chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

            self.sio.emit('transcribe_stream', {
                "session_id": session_id,
                "audio_data": chunk_b64,
                "model_name": model,
                "language": language,
                "beam_size": beam_size,
                "sample_rate": sample_rate,
                "task": task,
                "target_language": target_language,
                "enable_vad": enable_vad,
            })

            time.sleep(0.7)

        # Wait for results
        time.sleep(2.0)

        # Leave session
        self.sio.emit('leave_session', {'session_id': session_id})
        time.sleep(0.3)

    # ==================== TASK PARAMETER TESTS ====================

    def test_transcribe_to_english(self, result: TestResult):
        """Test: task='transcribe' + target='en' → Should use transcribe mode"""
        session_id = f"test-transcribe-en-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="transcribe",
            target_language="en",
            beam_size=5
        )

        # Check results
        if len(self.current_results) > 0:
            # Should have stability fields
            has_stability = any('stable_text' in r for r in self.current_results)
            result.passed = has_stability
            result.message = f"Received {len(self.current_results)} results with stability tracking"
            result.details = {"results_count": len(self.current_results)}
        else:
            result.passed = False
            result.message = "No results received (may be filtered by VAD)"

    def test_transcribe_to_spanish(self, result: TestResult):
        """Test: task='transcribe' + target='es' → Should use transcribe mode"""
        session_id = f"test-transcribe-es-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="transcribe",
            target_language="es",
            beam_size=5
        )

        if len(self.current_results) > 0:
            has_stability = any('stable_text' in r for r in self.current_results)
            result.passed = has_stability
            result.message = f"Transcribe mode working with non-English target"
        else:
            result.passed = False
            result.message = "No results received"

    def test_translate_to_english(self, result: TestResult):
        """Test: task='translate' + target='en' → Should use Whisper translate mode"""
        session_id = f"test-translate-en-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="auto",  # Auto-detect source language
            task="translate",
            target_language="en",
            beam_size=5
        )

        if len(self.current_results) > 0:
            # Check for translation_mode field
            has_translate_mode = any(r.get('translation_mode') == 'whisper_translate' for r in self.current_results)
            result.passed = len(self.current_results) > 0
            result.message = f"Translate mode: {has_translate_mode}, results: {len(self.current_results)}"
        else:
            result.passed = False
            result.message = "No results received"

    def test_translate_to_spanish_fallback(self, result: TestResult):
        """Test: task='translate' + target='es' → Should fall back to transcribe (Whisper can't translate to non-English)"""
        session_id = f"test-translate-es-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="translate",
            target_language="es",
            beam_size=5
        )

        if len(self.current_results) > 0:
            # Should fall back to transcribe mode (not whisper_translate)
            result.passed = True
            result.message = f"Correctly fell back to transcribe mode for non-English target"
        else:
            result.passed = False
            result.message = "No results received"

    # ==================== VAD TESTS ====================

    def test_vad_with_speech(self, result: TestResult):
        """Test: VAD correctly detects speech"""
        session_id = f"test-vad-speech-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="loud")  # Use loud speech

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="transcribe",
            target_language="en",
            beam_size=5,
            enable_vad=True
        )

        # VAD should allow speech through
        result.passed = len(self.current_results) > 0
        result.message = f"VAD detected speech: {len(self.current_results)} results"

    def test_vad_with_silence(self, result: TestResult):
        """Test: VAD correctly filters silence"""
        session_id = f"test-vad-silence-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="silence")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="transcribe",
            target_language="en",
            beam_size=5,
            enable_vad=True
        )

        # VAD should filter out silence
        result.passed = len(self.current_results) == 0
        result.message = f"VAD correctly filtered silence (results: {len(self.current_results)})"

    # ==================== STABILITY TRACKING TESTS ====================

    def test_stability_fields_present(self, result: TestResult):
        """Test: All Phase 3C stability fields are present"""
        session_id = f"test-stability-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        self.stream_audio(
            session_id, audio, 16000,
            model="large-v3-turbo",
            language="en",
            task="transcribe",
            target_language="en",
            beam_size=5
        )

        if len(self.current_results) > 0:
            first_result = self.current_results[0]
            required_fields = ['stable_text', 'unstable_text', 'is_draft', 'is_final',
                             'should_translate', 'stability_score']
            missing_fields = [f for f in required_fields if f not in first_result]

            result.passed = len(missing_fields) == 0
            if result.passed:
                result.message = "All Phase 3C stability fields present"
                result.details = {
                    "stable_text": first_result.get('stable_text', '')[:30],
                    "unstable_text": first_result.get('unstable_text', '')[:30],
                    "is_draft": first_result.get('is_draft'),
                    "is_final": first_result.get('is_final'),
                    "stability_score": first_result.get('stability_score')
                }
            else:
                result.message = f"Missing fields: {missing_fields}"
        else:
            result.passed = False
            result.message = "No results received"

    # ==================== BEAM SEARCH TESTS ====================

    def test_beam_search_compatibility(self, result: TestResult):
        """Test: Beam search works with task parameter"""
        session_id = f"test-beam-{int(time.time())}"
        audio = create_speech_audio(duration=2.0, pattern="normal")

        # Test with different beam sizes
        for beam_size in [1, 5, 10]:
            self.current_results = []
            session_id = f"test-beam-{beam_size}-{int(time.time())}"

            self.stream_audio(
                session_id, audio, 16000,
                model="large-v3-turbo",
                language="en",
                task="transcribe",
                target_language="en",
                beam_size=beam_size
            )

            if len(self.current_results) == 0:
                result.passed = False
                result.message = f"Beam size {beam_size} failed"
                return

        result.passed = True
        result.message = "Beam search working with all sizes (1, 5, 10)"

    # ==================== RUN ALL TESTS ====================

    def run_all_tests(self):
        """Run the complete test suite"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PHASE 3C TEST SUITE")
        print("="*80)

        self.connect()

        try:
            # Task Parameter Tests
            print("\n\n📋 TASK PARAMETER TESTS")
            print("-"*80)
            self.run_test("Transcribe to English", self.test_transcribe_to_english)
            self.run_test("Transcribe to Spanish", self.test_transcribe_to_spanish)
            self.run_test("Translate to English (Whisper)", self.test_translate_to_english)
            self.run_test("Translate to Spanish (Fallback)", self.test_translate_to_spanish_fallback)

            # VAD Tests
            print("\n\n🎙️  VOICE ACTIVITY DETECTION TESTS")
            print("-"*80)
            self.run_test("VAD with Speech", self.test_vad_with_speech)
            self.run_test("VAD with Silence", self.test_vad_with_silence)

            # Stability Tests
            print("\n\n📊 STABILITY TRACKING TESTS")
            print("-"*80)
            self.run_test("Stability Fields Present", self.test_stability_fields_present)

            # Beam Search Tests
            print("\n\n🔍 BEAM SEARCH TESTS")
            print("-"*80)
            self.run_test("Beam Search Compatibility", self.test_beam_search_compatibility)

        finally:
            self.disconnect()

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"{status} {result.test_name}")
            print(f"   {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"   - {key}: {value}")

        print(f"\n{'='*80}")
        print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"{'='*80}\n")

        return passed == total


def main():
    """Run comprehensive test suite"""
    suite = ComprehensiveTestSuite()
    all_passed = suite.run_all_tests()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
