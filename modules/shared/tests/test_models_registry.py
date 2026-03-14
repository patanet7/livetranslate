"""Behavioral tests for BackendConfig model registry entries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from livetranslate_common.models.registry import BackendConfig


def _english_config() -> BackendConfig:
    """Factory for a valid realtime English backend config."""
    return BackendConfig(
        backend="faster-whisper",
        model="whisper-base",
        compute_type="int8",
        chunk_duration_s=2.0,
        stride_s=1.5,
        overlap_s=0.5,
        vad_threshold=0.5,
        beam_size=5,
        prebuffer_s=0.2,
        batch_profile="realtime",
    )


def _chinese_config() -> BackendConfig:
    """Factory for a valid batch Chinese backend config."""
    return BackendConfig(
        backend="openvino",
        model="whisper-large-v3",
        compute_type="float16",
        chunk_duration_s=5.0,
        stride_s=4.0,
        overlap_s=1.0,
        vad_threshold=0.35,
        beam_size=10,
        prebuffer_s=0.5,
        batch_profile="batch",
    )


class TestBackendConfig:
    def test_english_config(self) -> None:
        cfg = _english_config()
        assert cfg.backend == "faster-whisper"
        assert cfg.model == "whisper-base"
        assert cfg.compute_type == "int8"
        assert cfg.chunk_duration_s == pytest.approx(2.0)
        assert cfg.stride_s == pytest.approx(1.5)
        assert cfg.overlap_s == pytest.approx(0.5)
        assert cfg.vad_threshold == pytest.approx(0.5)
        assert cfg.beam_size == 5
        assert cfg.prebuffer_s == pytest.approx(0.2)
        assert cfg.batch_profile == "realtime"

    def test_chinese_config(self) -> None:
        cfg = _chinese_config()
        assert cfg.backend == "openvino"
        assert cfg.batch_profile == "batch"
        assert cfg.beam_size == 10
        assert cfg.stride_s == pytest.approx(4.0)
        assert cfg.overlap_s == pytest.approx(1.0)

    def test_stride_overlap_consistency_enforced(self) -> None:
        """stride_s != chunk_duration_s - overlap_s must raise ValueError."""
        with pytest.raises(ValueError, match="stride_s"):
            BackendConfig(
                backend="faster-whisper",
                model="whisper-base",
                compute_type="int8",
                chunk_duration_s=2.0,
                stride_s=1.0,   # wrong: should be 1.5
                overlap_s=0.5,
                vad_threshold=0.5,
                beam_size=5,
                prebuffer_s=0.0,
            )

    def test_zero_chunk_duration_rejected(self) -> None:
        """chunk_duration_s must be > 0."""
        with pytest.raises(ValidationError):
            BackendConfig(
                backend="faster-whisper",
                model="whisper-base",
                compute_type="int8",
                chunk_duration_s=0.0,
                stride_s=0.0,
                overlap_s=0.0,
                vad_threshold=0.5,
                beam_size=5,
                prebuffer_s=0.0,
            )

    def test_vad_threshold_out_of_range_rejected(self) -> None:
        """vad_threshold must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            BackendConfig(
                backend="faster-whisper",
                model="whisper-base",
                compute_type="int8",
                chunk_duration_s=2.0,
                stride_s=1.5,
                overlap_s=0.5,
                vad_threshold=1.5,   # out of range
                beam_size=5,
                prebuffer_s=0.0,
            )

    def test_batch_profile_values(self) -> None:
        """batch_profile only accepts 'realtime' or 'batch'."""
        realtime_cfg = _english_config()
        assert realtime_cfg.batch_profile == "realtime"

        batch_cfg = _chinese_config()
        assert batch_cfg.batch_profile == "batch"

        with pytest.raises(ValidationError):
            BackendConfig(
                backend="faster-whisper",
                model="whisper-base",
                compute_type="int8",
                chunk_duration_s=2.0,
                stride_s=1.5,
                overlap_s=0.5,
                vad_threshold=0.5,
                beam_size=5,
                prebuffer_s=0.0,
                batch_profile="streaming",  # type: ignore[arg-type]  # invalid value
            )
