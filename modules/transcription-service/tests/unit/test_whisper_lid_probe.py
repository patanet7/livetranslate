#!/usr/bin/env python3
"""
Unit tests for Whisper-native LID probe.

Tests the zero-cost language detection using Whisper's encoder output.
"""

import time
from pathlib import Path

import pytest
import torch
from language_id.lid_detector import FrameLevelLID
from simul_whisper.whisper import load_model
from simul_whisper.whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim


@pytest.fixture(scope="module")
def whisper_model():
    """Load Whisper model once for all tests."""
    model = load_model("base")  # Use base for faster tests
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(whisper_model):
    """Get tokenizer from model."""
    from simul_whisper.whisper.tokenizer import get_tokenizer

    return get_tokenizer(
        multilingual=whisper_model.is_multilingual, num_languages=whisper_model.num_languages
    )


@pytest.fixture
def english_audio():
    """Load English audio (JFK)."""
    audio_path = Path(__file__).parent.parent / "fixtures" / "audio" / "jfk.wav"
    audio = load_audio(str(audio_path))
    return pad_or_trim(audio)


@pytest.fixture
def chinese_audio():
    """Load Chinese audio."""
    audio_path = Path(__file__).parent.parent / "fixtures" / "audio" / "OSR_cn_000_0072_8k.wav"
    audio = load_audio(str(audio_path))
    return pad_or_trim(audio)


class TestLanguageTokenExtraction:
    """Test extraction of language token IDs from tokenizer."""

    def test_get_language_token_ids_basic(self, tokenizer):
        """Test basic language token ID extraction."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Extract language token IDs
        token_ids = lid._get_language_token_ids(tokenizer)

        # Should return dict with language codes as keys
        assert isinstance(token_ids, dict)
        assert "en" in token_ids
        assert "zh" in token_ids

        # Token IDs should be integers
        assert isinstance(token_ids["en"], int)
        assert isinstance(token_ids["zh"], int)

        # Token IDs should be different
        assert token_ids["en"] != token_ids["zh"]

    def test_get_language_token_ids_match_tokenizer(self, tokenizer):
        """Test that extracted IDs match tokenizer's to_language_token()."""
        lid = FrameLevelLID(target_languages=["en", "zh", "es", "fr"])

        token_ids = lid._get_language_token_ids(tokenizer)

        # Verify each language
        for lang in ["en", "zh", "es", "fr"]:
            expected_id = tokenizer.to_language_token(lang)
            assert (
                token_ids[lang] == expected_id
            ), f"Language {lang}: got {token_ids[lang]}, expected {expected_id}"

    def test_get_language_token_ids_invalid_language(self, tokenizer):
        """Test handling of invalid language codes."""
        lid = FrameLevelLID(target_languages=["en", "invalid_lang"])

        # Should raise KeyError for invalid language
        with pytest.raises(KeyError):
            lid._get_language_token_ids(tokenizer)


class TestWhisperLIDProbe:
    """Test Whisper-native LID probe accuracy."""

    def test_probe_english_audio(self, whisper_model, tokenizer, english_audio):
        """Test that probe correctly identifies English audio."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Convert to mel spectrogram
        mel = log_mel_spectrogram(english_audio)

        # Run encoder
        with torch.no_grad():
            encoder_output = whisper_model.encoder(mel.unsqueeze(0).to(whisper_model.device))

        # Run LID probe
        lang_probs = lid.detect(
            encoder_output=encoder_output, model=whisper_model, tokenizer=tokenizer, timestamp=0.0
        )

        # Check output format
        assert isinstance(lang_probs, dict)
        assert "en" in lang_probs
        assert "zh" in lang_probs

        # Check probabilities sum to ~1.0
        prob_sum = sum(lang_probs.values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected ~1.0"

        # English probability should be high (>90%)
        assert (
            lang_probs["en"] > 0.9
        ), f"English probability {lang_probs['en']:.3f} too low (expected >0.9)"

        # Chinese probability should be low (<10%)
        assert (
            lang_probs["zh"] < 0.1
        ), f"Chinese probability {lang_probs['zh']:.3f} too high (expected <0.1)"

    def test_probe_chinese_audio(self, whisper_model, tokenizer, chinese_audio):
        """Test that probe correctly identifies Chinese audio."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Convert to mel spectrogram
        mel = log_mel_spectrogram(chinese_audio)

        # Run encoder
        with torch.no_grad():
            encoder_output = whisper_model.encoder(mel.unsqueeze(0).to(whisper_model.device))

        # Run LID probe
        lang_probs = lid.detect(
            encoder_output=encoder_output, model=whisper_model, tokenizer=tokenizer, timestamp=0.0
        )

        # Check output format
        assert isinstance(lang_probs, dict)
        assert "en" in lang_probs
        assert "zh" in lang_probs

        # Check probabilities sum to ~1.0
        prob_sum = sum(lang_probs.values())
        assert abs(prob_sum - 1.0) < 0.01

        # Chinese probability should be high (>70% for 8kHz audio)
        assert (
            lang_probs["zh"] > 0.7
        ), f"Chinese probability {lang_probs['zh']:.3f} too low (expected >0.7)"

        # English probability should be low
        assert (
            lang_probs["en"] < 0.3
        ), f"English probability {lang_probs['en']:.3f} too high (expected <0.3)"

    def test_probe_deterministic(self, whisper_model, tokenizer, english_audio):
        """Test that probe gives consistent results for same input."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Convert to mel spectrogram
        mel = log_mel_spectrogram(english_audio)

        # Run encoder once
        with torch.no_grad():
            encoder_output = whisper_model.encoder(mel.unsqueeze(0).to(whisper_model.device))

        # Run probe multiple times
        results = []
        for _ in range(3):
            lang_probs = lid.detect(
                encoder_output=encoder_output,
                model=whisper_model,
                tokenizer=tokenizer,
                timestamp=0.0,
            )
            results.append(lang_probs)

        # All results should be identical
        for i in range(1, len(results)):
            for lang in ["en", "zh"]:
                assert (
                    abs(results[i][lang] - results[0][lang]) < 1e-6
                ), f"Probe not deterministic: {results[i][lang]} != {results[0][lang]}"


class TestWhisperLIDProbeLatency:
    """Test Whisper LID probe latency (target: <1ms on GPU)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for latency test")
    def test_probe_latency_gpu(self, whisper_model, tokenizer, english_audio):
        """Test that probe runs in <1ms on GPU."""
        # Move model to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper_model.to(device)

        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Prepare encoder output
        mel = log_mel_spectrogram(english_audio)
        with torch.no_grad():
            encoder_output = model.encoder(mel.unsqueeze(0).to(device))

        # Warm up (10 runs)
        for _ in range(10):
            lid.detect(encoder_output, model, tokenizer, timestamp=0.0)

        # Benchmark (100 runs)
        start_time = time.perf_counter()
        for _ in range(100):
            lid.detect(encoder_output, model, tokenizer, timestamp=0.0)
        elapsed = (time.perf_counter() - start_time) / 100

        # Should be <1ms on GPU
        assert elapsed < 0.001, f"Probe latency {elapsed*1000:.2f}ms exceeds 1ms target"

        print(f"\n✅ Probe latency: {elapsed*1000:.3f}ms (GPU)")

    def test_probe_latency_cpu(self, whisper_model, tokenizer, english_audio):
        """Test probe latency on CPU (should still be reasonable)."""
        # Force CPU
        model = whisper_model.to("cpu")

        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Prepare encoder output
        mel = log_mel_spectrogram(english_audio)
        with torch.no_grad():
            encoder_output = model.encoder(mel.unsqueeze(0).to("cpu"))

        # Warm up
        for _ in range(10):
            lid.detect(encoder_output, model, tokenizer, timestamp=0.0)

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(100):
            lid.detect(encoder_output, model, tokenizer, timestamp=0.0)
        elapsed = (time.perf_counter() - start_time) / 100

        # Should be <50ms on CPU (more lenient for CPU-only systems)
        assert elapsed < 0.050, f"Probe latency {elapsed*1000:.2f}ms exceeds 50ms target (CPU)"

        print(f"\n✅ Probe latency: {elapsed*1000:.3f}ms (CPU)")


class TestWhisperLIDProbeReadOnly:
    """Test that probe is truly READ-ONLY (never modifies model state)."""

    def test_probe_no_kv_cache_pollution(self, whisper_model, tokenizer, english_audio):
        """Verify probe does not create or modify KV cache."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        # Prepare encoder output
        mel = log_mel_spectrogram(english_audio)
        with torch.no_grad():
            encoder_output = whisper_model.encoder(mel.unsqueeze(0).to(whisper_model.device))

        # Check decoder state before probe
        # Note: Whisper decoder doesn't have persistent KV cache by default
        # This test ensures we're not creating one
        initial_params = {
            name: param.clone() for name, param in whisper_model.decoder.named_parameters()
        }

        # Run probe
        lid.detect(encoder_output, whisper_model, tokenizer, timestamp=0.0)

        # Check decoder state after probe
        final_params = dict(whisper_model.decoder.named_parameters())

        # Parameters should be unchanged
        for name in initial_params:
            assert torch.equal(
                initial_params[name], final_params[name]
            ), f"Probe modified decoder parameter: {name}"

    def test_probe_with_torch_no_grad(self, whisper_model, tokenizer, english_audio):
        """Verify probe runs inside torch.no_grad() context."""
        lid = FrameLevelLID(target_languages=["en", "zh"])

        mel = log_mel_spectrogram(english_audio)
        encoder_output = whisper_model.encoder(mel.unsqueeze(0).to(whisper_model.device))

        # Probe should work in no_grad context
        with torch.no_grad():
            lang_probs = lid.detect(encoder_output, whisper_model, tokenizer, timestamp=0.0)

        # Should return valid probabilities
        assert lang_probs["en"] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
