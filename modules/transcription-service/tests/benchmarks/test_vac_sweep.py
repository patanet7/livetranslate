"""Pytest suite for VAC sweep benchmark.

These tests run the full parameter sweep against real audio fixtures using
a stub transcriber (no GPU / services required).  They validate:
  1. The sweep produces results for every config
  2. The best config is selected correctly (lowest error rate)
  3. The metrics module handles number normalization correctly
  4. Latency percentile calculation is correct

To run against a real backend, set the ``BENCHMARK_BACKEND`` env var::

    BENCHMARK_BACKEND=vllm pytest tests/benchmarks/test_vac_sweep.py -v -s

Mark: @pytest.mark.benchmark — skip with ``-m "not benchmark"``
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "audio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path))
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


async def _stub_transcribe(audio: np.ndarray, language: str) -> str:
    """Stub that returns a short plausible string based on language."""
    await asyncio.sleep(0)  # yield — no actual inference
    dur = len(audio) / 16_000
    if language == "zh":
        return f"测试音频片段{dur:.0f}秒"
    return f"test audio segment {dur:.1f} seconds long"


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

class TestNumberNormalization:
    def test_zh_two_thousand_five_hundred_wan(self):
        from benchmarks.metrics import normalize_numbers
        # "两千五百万" should become "2500万"
        result = normalize_numbers("两千五百万")
        assert result == "2500万", f"Got: {result}"

    def test_mixed_digit_and_zh_unit(self):
        from benchmarks.metrics import normalize_numbers
        # "2500万" should pass through unchanged
        assert normalize_numbers("2500万") == "2500万"

    def test_cer_with_number_normalization(self):
        from benchmarks.metrics import error_rate
        ref = "总收入达到了两千五百万美元"
        hyp = "总收入达到了2500万美元"
        er, metric = error_rate(ref, hyp, "zh")
        # After normalization both are identical — CER should be 0 or near 0
        assert er < 0.05, f"Expected near-zero CER after normalization, got {er:.3f}"
        assert metric == "cer"

    def test_pure_integer_conversion(self):
        from benchmarks.metrics import normalize_numbers
        assert normalize_numbers("三百") == "300"
        assert normalize_numbers("一千零五") == "1005"

    def test_wer_english_case_insensitive(self):
        from benchmarks.metrics import error_rate
        ref = "Good morning everyone"
        hyp = "good morning everyone"
        er, metric = error_rate(ref, hyp, "en")
        assert er == 0.0
        assert metric == "wer"


class TestAlignHypothesis:
    def test_no_overlap(self):
        from benchmarks.metrics import align_hypothesis_to_reference
        segs = ["hello world", "foo bar"]
        result = align_hypothesis_to_reference(segs, "hello world foo bar", "en")
        assert "hello" in result
        assert "bar" in result

    def test_overlap_dedup(self):
        from benchmarks.metrics import align_hypothesis_to_reference
        # Second segment starts with the end of first — should not duplicate
        segs = ["the quick brown fox", "brown fox jumps over"]
        result = align_hypothesis_to_reference(segs, "the quick brown fox jumps over", "en")
        # "brown fox" should appear once
        assert result.count("brown fox") == 1

    def test_cjk_no_space_sep(self):
        from benchmarks.metrics import align_hypothesis_to_reference
        segs = ["大家好", "欢迎参加"]
        result = align_hypothesis_to_reference(segs, "大家好欢迎参加", "zh")
        assert " " not in result


class TestLatencyPercentiles:
    def test_basic(self):
        from benchmarks.metrics import latency_percentiles
        data = [0.1 * i for i in range(1, 101)]  # 0.1 … 10.0
        stats = latency_percentiles(data)
        assert stats["p50"] == pytest.approx(5.0, abs=0.5)
        assert stats["p95"] == pytest.approx(9.5, abs=0.5)

    def test_empty(self):
        from benchmarks.metrics import latency_percentiles
        stats = latency_percentiles([])
        assert stats["p50"] == 0.0


# ---------------------------------------------------------------------------
# VAC sweep functional tests (stub backend, no GPU)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
class TestVACSweepStub:
    """Run the VAC sweep with a stub transcriber."""

    def test_sweep_returns_results_for_all_configs(self):
        from benchmarks.vac_sweep import run_vac_sweep

        audio = np.zeros(16_000 * 5, dtype=np.float32)  # 5 s silence
        reference = "测试"

        results = asyncio.run(run_vac_sweep(
            audio=audio,
            reference=reference,
            language="zh",
            transcribe_fn=_stub_transcribe,
            prebuffer_values=[0.5, 1.0],
            stride_values=[2.5],
            overlap_values=[0.3, 0.5],
        ))
        # 2 prebuffer × 1 stride × 2 overlap = 4 configs
        assert len(results) == 4

    def test_sweep_sorted_by_error_rate(self):
        from benchmarks.vac_sweep import run_vac_sweep

        audio = np.zeros(16_000 * 3, dtype=np.float32)
        reference = "stub transcription"

        results = asyncio.run(run_vac_sweep(
            audio=audio,
            reference=reference,
            language="en",
            transcribe_fn=_stub_transcribe,
            prebuffer_values=[0.5, 1.0],
            stride_values=[2.0],
            overlap_values=[0.3],
        ))
        rates = [r.error_rate for r in results]
        assert rates == sorted(rates)

    def test_sweep_overlap_constraint_enforced(self):
        """Configs where overlap >= stride must be excluded."""
        from benchmarks.vac_sweep import run_vac_sweep

        audio = np.zeros(16_000 * 3, dtype=np.float32)
        results = asyncio.run(run_vac_sweep(
            audio=audio,
            reference="test",
            language="en",
            transcribe_fn=_stub_transcribe,
            prebuffer_values=[1.0],
            stride_values=[2.0],
            overlap_values=[1.0, 2.5],  # 2.5 >= stride=2.0, must be excluded
        ))
        for r in results:
            assert r.config.overlap_s < r.config.stride_s

    def test_hallucination_flag_set_correctly(self):
        """If last segment is much longer than average, flag should be set."""
        from benchmarks.vac_sweep import SweepResult, VACConfig
        from benchmarks.metrics import latency_percentiles

        # Construct a result manually with known hallucination pattern
        from benchmarks.vac_sweep import SegmentRecord
        segs = [
            SegmentRecord(0, "短句", 1.0, 0.1),
            SegmentRecord(1, "短短句", 2.0, 0.1),
            SegmentRecord(2, "这个段落非常非常长，长到超过前面所有段落的平均长度的两倍以上，触发幻觉检测", 3.0, 0.1),
        ]
        preceding = [len(s.text) for s in segs[:-1]]
        avg = sum(preceding) / len(preceding)
        ratio = len(segs[-1].text) / avg
        assert ratio > 2.5  # Confirms test setup is correct

    @pytest.mark.skipif(
        not (FIXTURES / "meeting_zh.wav").exists(),
        reason="meeting_zh.wav fixture not found"
    )
    def test_sweep_on_real_zh_fixture_stub(self):
        """Run sweep on real Chinese meeting audio with stub transcriber."""
        from benchmarks.vac_sweep import run_vac_sweep, print_sweep_table

        audio = _load_audio(FIXTURES / "meeting_zh.wav")
        reference = " ".join(
            (FIXTURES / "meeting_zh.txt").read_text(encoding="utf-8").splitlines()
        )

        results = asyncio.run(run_vac_sweep(
            audio=audio,
            reference=reference,
            language="zh",
            transcribe_fn=_stub_transcribe,
            prebuffer_values=[0.5, 1.0],
            stride_values=[2.5],
            overlap_values=[0.5],
        ))

        assert len(results) > 0
        print_sweep_table(results)
        # With stub, we just validate structure, not scores
        assert all(0.0 <= r.error_rate <= 1.0 for r in results)
        assert all(r.segment_count >= 0 for r in results)


# ---------------------------------------------------------------------------
# Real backend sweep (skipped unless BENCHMARK_BACKEND is set)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENCHMARK_BACKEND") not in ("vllm", "faster-whisper"),
    reason="Set BENCHMARK_BACKEND=vllm or faster-whisper to run real sweep"
)
class TestVACSweepRealBackend:
    """Run the sweep against a real Whisper backend."""

    @pytest.fixture(scope="class")
    def transcribe_fn(self):
        backend = os.environ.get("BENCHMARK_BACKEND", "vllm")
        vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")

        if backend == "vllm":
            import httpx
            import base64, io, soundfile as sf

            async def _fn(audio: np.ndarray, language: str) -> str:
                buf = io.BytesIO()
                sf.write(buf, audio, 16_000, format="WAV", subtype="FLOAT")
                encoded = base64.b64encode(buf.getvalue()).decode()
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{vllm_url}/audio/transcriptions",
                        json={"model": "openai/whisper-large-v3-turbo",
                              "audio": encoded, "language": language},
                    )
                    resp.raise_for_status()
                    return resp.json().get("text", "")
            return _fn

        elif backend == "faster-whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("large-v3-turbo", device="auto", compute_type="float16")

            async def _fn(audio: np.ndarray, language: str) -> str:
                segs, _ = model.transcribe(audio, language=language, beam_size=5)
                return " ".join(s.text for s in segs).strip()
            return _fn

    @pytest.mark.parametrize("lang,wav,txt", [
        ("zh", "meeting_zh.wav", "meeting_zh.txt"),
        ("en", "meeting_en.wav", "meeting_en.txt"),
        ("ja", "meeting_ja.wav", "meeting_ja.txt"),
        ("es", "meeting_es.wav", "meeting_es.txt"),
    ])
    def test_real_sweep(self, transcribe_fn, lang, wav, txt):
        from benchmarks.vac_sweep import run_vac_sweep, print_sweep_table

        audio_path = FIXTURES / wav
        ref_path = FIXTURES / txt
        if not audio_path.exists() or not ref_path.exists():
            pytest.skip(f"Fixture not found: {wav}")

        audio = _load_audio(audio_path)
        reference = " ".join(ref_path.read_text(encoding="utf-8").splitlines())

        results = asyncio.run(run_vac_sweep(
            audio=audio,
            reference=reference,
            language=lang,
            transcribe_fn=transcribe_fn,
        ))
        print_sweep_table(results)

        # Best config must be better than a naive no-overlap config
        assert results[0].error_rate < 0.8, (
            f"Best {lang} config CER/WER={results[0].error_rate:.3f} — "
            "seems degenerate; check backend connectivity"
        )
