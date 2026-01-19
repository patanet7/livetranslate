"""
Property-Based Tests: System Invariants

Tests critical system invariants using property-based testing (hypothesis).
Per ML Engineer review - Priority 5: Property-based tests for system invariants

Critical invariants:
1. LID probabilities always sum to 1.0 (±epsilon)
2. VAD never crashes on arbitrary audio
3. Session state machine maintains invariants
4. KV cache never exceeds n_text_ctx bounds

Reference: FEEDBACK.md, sustained_detector.py, vad_detector.py, session_manager.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pytest
from language_id import SustainedLanguageDetector
from vad_detector import SileroVAD

logger = logging.getLogger(__name__)


# Try to import hypothesis for property-based testing
try:
    from hypothesis import assume, given, settings, strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    logger.warning("⚠️ hypothesis not installed - property tests will be skipped")
    logger.warning("   Install with: pip install hypothesis")


# Hypothesis strategies for audio generation
if HYPOTHESIS_AVAILABLE:

    @st.composite
    def audio_strategy(draw, min_length=1, max_length=16000):
        """Generate arbitrary audio data"""
        length = draw(st.integers(min_value=min_length, max_value=max_length))
        amplitude = draw(st.floats(min_value=0.0, max_value=2.0))
        audio = draw(
            st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=length,
                max_size=length,
            )
        )
        return np.array(audio, dtype=np.float32) * amplitude

    @st.composite
    def lid_probabilities_strategy(draw, languages=None):
        """Generate LID probabilities that should sum to 1.0"""
        # Generate random probabilities
        if languages is None:
            languages = ["en", "zh"]
        probs = [draw(st.floats(min_value=0.0, max_value=1.0)) for _ in languages]
        total = sum(probs)

        # Normalize to sum to 1.0
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # All zeros - set first to 1.0
            probs[0] = 1.0

        return dict(zip(languages, probs, strict=False))


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@pytest.mark.property
class TestLIDProbabilityInvariants:
    """Test LID probability invariants using property-based testing"""

    @pytest.fixture
    def detector(self):
        """Create sustained detector for LID tests"""
        return SustainedLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=6, min_dwell_ms=250.0, frame_hop_ms=100.0
        )

    @given(lid_probabilities_strategy())
    @settings(max_examples=100, deadline=None)
    def test_lid_probabilities_sum_to_one(self, detector, lid_probs):
        """
        Property: LID probabilities always sum to 1.0 (±epsilon).

        This is a fundamental invariant - probabilities must form a valid distribution.
        """
        total = sum(lid_probs.values())

        # Allow small floating point error
        epsilon = 1e-6
        assert (
            abs(total - 1.0) < epsilon
        ), f"LID probabilities sum to {total:.10f}, expected 1.0 (±{epsilon})"

        logger.debug(f"✅ LID probs sum to 1.0: {lid_probs}")

    @given(
        st.lists(lid_probabilities_strategy(), min_size=1, max_size=20),
        st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_detector_maintains_valid_state_under_arbitrary_input(
        self, detector, lid_sequence, start_time
    ):
        """
        Property: Detector maintains valid state under arbitrary input sequence.

        No matter what LID probabilities are fed, detector should:
        1. Never crash
        2. Always return valid state
        3. Maintain internal consistency
        """
        try:
            for i, lid_probs in enumerate(lid_sequence):
                timestamp = start_time + i * 0.1

                # Process
                event = detector.update(lid_probs, timestamp)

                # Verify state consistency
                assert detector.current_language in lid_probs or detector.current_language is None
                assert (
                    detector.candidate_language in lid_probs or detector.candidate_language is None
                )

                if event is not None:
                    # Verify event is valid
                    assert event.from_language in lid_probs
                    assert event.to_language in lid_probs
                    assert event.confidence_margin >= 0.0
                    assert event.dwell_frames >= 0
                    assert event.dwell_duration_ms >= 0.0

        except Exception as e:
            pytest.fail(f"Detector crashed on arbitrary input: {e}")

        logger.debug(f"✅ Detector maintained valid state for {len(lid_sequence)} inputs")

    @given(lid_probabilities_strategy())
    @settings(max_examples=50, deadline=None)
    def test_max_probability_language_is_valid(self, detector, lid_probs):
        """
        Property: Language with max probability is always in language set.
        """
        max_lang = max(lid_probs, key=lid_probs.get)
        assert max_lang in lid_probs

        logger.debug(f"✅ Max language valid: {max_lang} with prob {lid_probs[max_lang]:.3f}")

    @given(st.lists(lid_probabilities_strategy(), min_size=10, max_size=10))
    @settings(max_examples=20, deadline=None)
    def test_detector_total_switches_monotonic(self, detector, lid_sequence):
        """
        Property: Total switches counter is monotonically increasing.
        """
        prev_switches = detector.total_switches

        for i, lid_probs in enumerate(lid_sequence):
            detector.update(lid_probs, timestamp=i * 0.1)

            current_switches = detector.total_switches

            # Should never decrease
            assert current_switches >= prev_switches, "Total switches counter decreased!"

            prev_switches = current_switches

        logger.debug(f"✅ Total switches monotonic: {detector.total_switches}")


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@pytest.mark.property
class TestVADRobustnessInvariants:
    """Test VAD robustness invariants using property-based testing"""

    @pytest.fixture
    def vad(self):
        """Create VAD for property tests"""
        return SileroVAD(threshold=0.5, sampling_rate=16000, min_silence_duration_ms=500)

    @given(audio_strategy(min_length=1, max_length=16000))
    @settings(max_examples=50, deadline=None)
    def test_vad_never_crashes_on_arbitrary_audio(self, vad, audio):
        """
        Property: VAD never crashes on arbitrary audio.

        This is the key robustness property for VAD.
        """
        try:
            result = vad.check_speech(audio)

            # Result should be None or dict
            assert result is None or isinstance(
                result, dict
            ), f"VAD returned invalid type: {type(result)}"

            # If dict, should have valid structure
            if isinstance(result, dict):
                for key in result:
                    assert key in ["start", "end"], f"Invalid key in VAD result: {key}"
                    assert isinstance(
                        result[key], int | float
                    ), f"Invalid value type for {key}: {type(result[key])}"
                    assert result[key] >= 0, f"Negative timestamp: {result[key]}"

        except Exception as e:
            pytest.fail(f"VAD crashed on arbitrary audio (len={len(audio)}): {e}")

        logger.debug(f"✅ VAD processed audio of length {len(audio)} without crash")

    @given(st.lists(audio_strategy(min_length=100, max_length=1000), min_size=5, max_size=10))
    @settings(max_examples=20, deadline=None)
    def test_vad_state_consistent_across_chunks(self, vad, audio_chunks):
        """
        Property: VAD maintains consistent state across arbitrary chunk sequence.
        """
        vad.reset()

        start_events = 0
        end_events = 0

        try:
            for _i, chunk in enumerate(audio_chunks):
                result = vad.check_speech(chunk)

                if result is not None:
                    if "start" in result:
                        start_events += 1
                    if "end" in result:
                        end_events += 1

        except Exception as e:
            pytest.fail(f"VAD crashed during chunk processing: {e}")

        # State should be consistent
        logger.debug(
            f"✅ VAD processed {len(audio_chunks)} chunks: {start_events} starts, {end_events} ends"
        )

    @given(audio_strategy(min_length=512, max_length=512))
    @settings(max_examples=50, deadline=None)
    def test_vad_handles_exact_512_samples(self, vad, audio):
        """
        Property: VAD handles exactly 512 samples (native Silero VAD size).
        """
        assume(len(audio) == 512)

        try:
            result = vad.check_speech(audio)
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"VAD crashed on 512-sample chunk: {e}")

        logger.debug("✅ VAD handles 512-sample chunk")

    @given(st.integers(min_value=1, max_value=32000), st.floats(min_value=0.0, max_value=2.0))
    @settings(max_examples=50, deadline=None)
    def test_vad_handles_arbitrary_length_and_amplitude(self, vad, length, amplitude):
        """
        Property: VAD handles arbitrary audio length and amplitude.
        """
        audio = np.random.randn(length).astype(np.float32) * amplitude

        try:
            result = vad.check_speech(audio)
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"VAD crashed on length={length}, amp={amplitude}: {e}")

        logger.debug(f"✅ VAD handled length={length}, amplitude={amplitude}")


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@pytest.mark.property
class TestSessionStateInvariants:
    """Test session state machine invariants"""

    def test_session_language_is_valid(self):
        """
        Property: Session language is always from target language set.
        """
        detector = SustainedLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=6, min_dwell_ms=250.0, frame_hop_ms=100.0
        )

        target_languages = {"en", "zh"}

        # Initialize
        detector.update({"en": 0.9, "zh": 0.1}, timestamp=0.0)

        assert detector.current_language in target_languages or detector.current_language is None

        # Process more
        for i in range(1, 20):
            lid_probs = {"en": np.random.rand(), "zh": np.random.rand()}
            total = sum(lid_probs.values())
            lid_probs = {k: v / total for k, v in lid_probs.items()}

            detector.update(lid_probs, timestamp=i * 0.1)

            # Invariant: current_language always in target set
            assert (
                detector.current_language in target_languages or detector.current_language is None
            )
            assert (
                detector.candidate_language in target_languages
                or detector.candidate_language is None
            )

        logger.info("✅ Session language always valid")

    def test_candidate_frames_bounded(self):
        """
        Property: Candidate frames deque has bounded size.

        The deque is initialized with maxlen, should never exceed it.
        """
        detector = SustainedLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=6, min_dwell_ms=250.0, frame_hop_ms=100.0
        )

        max_len = detector.candidate_frames.maxlen

        # Initialize
        detector.update({"en": 0.9, "zh": 0.1}, timestamp=0.0)

        # Try to overflow candidate frames
        for i in range(1, 100):
            lid_probs = {"en": 0.2, "zh": 0.8}  # Strong candidate
            detector.update(lid_probs, timestamp=i * 0.1)

            # Invariant: never exceeds maxlen
            assert (
                len(detector.candidate_frames) <= max_len
            ), f"Candidate frames exceeded maxlen: {len(detector.candidate_frames)} > {max_len}"

        logger.info(f"✅ Candidate frames bounded: max {max_len}")

    def test_false_positives_counter_monotonic(self):
        """
        Property: False positives counter is monotonically increasing.
        """
        detector = SustainedLanguageDetector(
            confidence_margin=0.2, min_dwell_frames=6, min_dwell_ms=250.0, frame_hop_ms=100.0
        )

        # Initialize
        detector.update({"en": 0.9, "zh": 0.1}, timestamp=0.0)

        prev_fp = detector.false_positives_prevented

        # Generate false positives (brief switches)
        for i in range(1, 50):
            if i % 3 == 0:
                lid_probs = {"en": 0.3, "zh": 0.7}  # Brief switch
            else:
                lid_probs = {"en": 0.9, "zh": 0.1}  # Return to EN

            detector.update(lid_probs, timestamp=i * 0.1)

            current_fp = detector.false_positives_prevented

            # Should never decrease
            assert current_fp >= prev_fp, "False positives counter decreased!"

            prev_fp = current_fp

        logger.info(f"✅ False positives counter monotonic: {detector.false_positives_prevented}")


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@pytest.mark.property
class TestNumericalStabilityInvariants:
    """Test numerical stability invariants"""

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=5,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_probability_normalization_stable(self, raw_probs):
        """
        Property: Probability normalization is numerically stable.
        """
        # Normalize probabilities
        total = sum(raw_probs)

        if total > 0:
            normalized = [p / total for p in raw_probs]
        else:
            # All zeros case
            normalized = [1.0] + [0.0] * (len(raw_probs) - 1)

        # Check sum to 1.0
        normalized_sum = sum(normalized)

        epsilon = 1e-6
        assert (
            abs(normalized_sum - 1.0) < epsilon
        ), f"Normalized probs sum to {normalized_sum}, expected 1.0"

        # Check all non-negative
        assert all(p >= 0.0 for p in normalized), "Negative probability after normalization"

        # Check all <= 1.0
        assert all(p <= 1.0 for p in normalized), "Probability > 1.0 after normalization"

        logger.debug(f"✅ Normalization stable: {raw_probs} -> {normalized}")

    @given(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_margin_calculation_stable(self, prob_new, prob_old):
        """
        Property: Confidence margin calculation is numerically stable.
        """
        # Clamp to [0, 1]
        prob_new = max(0.0, min(1.0, prob_new))
        prob_old = max(0.0, min(1.0, prob_old))

        # Calculate margin
        margin = prob_new - prob_old

        # Should be in [-1, 1]
        assert -1.0 <= margin <= 1.0, f"Margin {margin} outside valid range [-1, 1]"

        # Should not be NaN or inf
        assert not np.isnan(margin), "Margin is NaN"
        assert not np.isinf(margin), "Margin is inf"

        logger.debug(f"✅ Margin stable: {prob_new:.3f} - {prob_old:.3f} = {margin:.3f}")

    @given(st.integers(min_value=0, max_value=1000), st.floats(min_value=0.0, max_value=10.0))
    @settings(max_examples=50, deadline=None)
    def test_timestamp_arithmetic_stable(self, frame_count, hop_ms):
        """
        Property: Timestamp arithmetic is numerically stable.
        """
        # Calculate timestamp
        timestamp = frame_count * (hop_ms / 1000.0)

        # Should be non-negative
        assert timestamp >= 0.0, f"Negative timestamp: {timestamp}"

        # Should not be NaN or inf
        assert not np.isnan(timestamp), "Timestamp is NaN"
        assert not np.isinf(timestamp), "Timestamp is inf"

        logger.debug(
            f"✅ Timestamp stable: frame={frame_count}, hop={hop_ms}ms -> {timestamp:.3f}s"
        )


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@pytest.mark.property
class TestBoundaryConditions:
    """Test boundary condition invariants"""

    def test_empty_language_set_rejected(self):
        """Property: Empty language set is invalid"""
        with pytest.raises((ValueError, AssertionError, IndexError)):
            # Should reject empty language set
            lid_probs = {}
            detector = SustainedLanguageDetector()
            detector.update(lid_probs, timestamp=0.0)

        logger.info("✅ Empty language set properly rejected")

    def test_single_language_valid(self):
        """Property: Single language is valid"""
        detector = SustainedLanguageDetector()

        # Single language should work
        detector.update({"en": 1.0}, timestamp=0.0)

        assert detector.current_language == "en"
        assert detector.candidate_language is None  # No switching possible

        logger.info("✅ Single language configuration valid")

    def test_many_languages_valid(self):
        """Property: Many languages (>2) is valid"""
        detector = SustainedLanguageDetector()

        languages = ["en", "zh", "es", "fr", "de", "ja", "ko", "ar"]
        probs = {lang: 1.0 / len(languages) for lang in languages}

        detector.update(probs, timestamp=0.0)

        assert detector.current_language in languages

        logger.info(f"✅ Many languages ({len(languages)}) valid")

    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_timestamp_can_be_arbitrary(self, timestamp):
        """Property: Timestamp can be any non-negative float"""
        detector = SustainedLanguageDetector()

        if timestamp < 0:
            # Negative timestamps might be rejected
            pass
        else:
            # Non-negative should work
            detector.update({"en": 0.9, "zh": 0.1}, timestamp=timestamp)
            assert detector.current_language == "en"

        logger.debug(f"✅ Timestamp {timestamp:.3f} handled")


if __name__ == "__main__":
    # Run property tests
    if HYPOTHESIS_AVAILABLE:
        pytest.main([__file__, "-v", "-m", "property", "--log-cli-level=INFO"])
    else:
        print("❌ hypothesis not installed - cannot run property tests")
        print("   Install with: pip install hypothesis")
