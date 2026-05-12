"""Behavioral tests for WhisperConnection and WhisperParameterOverrides.

Mirror of test_models_llm.py for the Whisper inference side. WhisperConnection is
the canonical "talk to a Whisper endpoint" value object used by:
  - transcription-service VLLMWhisperBackend (remote OpenAI-compatible Whisper)
  - orchestration whisper_resolver
  - dashboard /config/connections UI (future hot-reload)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from livetranslate_common.models.whisper import (
    WhisperConnection,
    WhisperParameterOverrides,
)


# ---------------------------------------------------------------------------
# WhisperConnection — construction & immutability
# ---------------------------------------------------------------------------


class TestWhisperConnectionConstruction:
    def test_construct_minimal(self) -> None:
        """engine + base_url + model are required; other fields default."""
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://localhost:8005",
            model="mlx-community/whisper-large-v3-turbo",
        )
        assert conn.engine == "openai_compatible"
        assert conn.base_url == "http://localhost:8005"
        assert conn.model == "mlx-community/whisper-large-v3-turbo"
        # defaults
        assert conn.api_key == ""
        assert conn.temperature == 0.0
        assert conn.beam_size == 1
        assert conn.no_speech_threshold == 0.6
        assert conn.compression_ratio_threshold == 2.4
        assert conn.language_hint is None
        assert conn.initial_prompt is None
        assert conn.compute_type == "float16"
        assert conn.timeout_s == 30.0
        assert conn.max_retries == 1
        assert conn.connection_id is None

    def test_construct_full(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://100.64.0.2:8089",
            model="mlx-community/whisper-large-v3-turbo",
            api_key="dummy",
            temperature=0.2,
            beam_size=5,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.0,
            language_hint="zh",
            initial_prompt="一段普通话会议录音。",
            compute_type="int8",
            timeout_s=45.0,
            max_retries=2,
            connection_id="conn-whisper-tailscale",
        )
        assert conn.api_key == "dummy"
        assert conn.connection_id == "conn-whisper-tailscale"
        assert conn.language_hint == "zh"
        assert conn.initial_prompt == "一段普通话会议录音。"

    def test_invalid_engine_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WhisperConnection(
                engine="bogus_engine",  # type: ignore[arg-type]
                base_url="http://x",
                model="m",
            )

    def test_is_frozen(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
        )
        with pytest.raises(ValidationError):
            conn.beam_size = 7  # type: ignore[misc]

    def test_local_engines_allow_empty_base_url(self) -> None:
        """mlx_local / faster_whisper_local engines don't need a URL."""
        conn = WhisperConnection(
            engine="mlx_local",
            base_url="",
            model="mlx-community/whisper-large-v3-turbo",
        )
        assert conn.base_url == ""


# ---------------------------------------------------------------------------
# Strict contracts — compute_type Literal + language_hint validation
# ---------------------------------------------------------------------------


class TestStrictContracts:
    def test_compute_type_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            WhisperConnection(
                engine="openai_compatible",
                base_url="http://x",
                model="m",
                compute_type="flot16",  # typo
            )

    def test_compute_type_accepts_known(self) -> None:
        for ct in ("float16", "float32", "int8", "int8_float16"):
            conn = WhisperConnection(
                engine="openai_compatible",
                base_url="http://x",
                model="m",
                compute_type=ct,
            )
            assert conn.compute_type == ct

    def test_language_hint_rejects_unknown_code(self) -> None:
        with pytest.raises(ValidationError):
            WhisperConnection(
                engine="openai_compatible",
                base_url="http://x",
                model="m",
                language_hint="klingon",
            )

    def test_language_hint_accepts_bcp47(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
            language_hint="zh",
        )
        assert conn.language_hint == "zh"

    def test_language_hint_normalizes_case(self) -> None:
        """A case-insensitive validation that returns the canonical lowercase code."""
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
            language_hint="ZH",
        )
        assert conn.language_hint == "zh"

    def test_language_hint_empty_string_becomes_none(self) -> None:
        """Whitespace-only / empty `language_hint` is normalized to None (auto-detect)."""
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
            language_hint="   ",
        )
        assert conn.language_hint is None

    def test_overrides_language_hint_validates(self) -> None:
        from livetranslate_common.models.whisper import WhisperParameterOverrides

        with pytest.raises(ValidationError):
            WhisperParameterOverrides(language_hint="totally-fake")

        # Valid passes
        overrides = WhisperParameterOverrides(language_hint="ja")
        assert overrides.language_hint == "ja"

    def test_whisper_languages_frozenset_size(self) -> None:
        """Sanity: the set has the full 99 BCP-47 codes Whisper supports."""
        from livetranslate_common.models.whisper import WHISPER_LANGUAGES

        assert len(WHISPER_LANGUAGES) == 99
        # Spot-check a few that have caused issues in the past
        for code in ("zh", "en", "ja", "haw", "jw"):
            assert code in WHISPER_LANGUAGES


# ---------------------------------------------------------------------------
# WhisperConnection — derived properties
# ---------------------------------------------------------------------------


class TestRequiresHttp:
    def test_openai_compatible_requires_http(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible", base_url="http://x", model="m"
        )
        assert conn.requires_http is True

    def test_mlx_local_does_not(self) -> None:
        conn = WhisperConnection(engine="mlx_local", base_url="", model="m")
        assert conn.requires_http is False

    def test_faster_whisper_local_does_not(self) -> None:
        conn = WhisperConnection(
            engine="faster_whisper_local", base_url="", model="m"
        )
        assert conn.requires_http is False


# ---------------------------------------------------------------------------
# WhisperParameterOverrides — all optional + range validators
# ---------------------------------------------------------------------------


class TestWhisperParameterOverrides:
    def test_all_fields_optional(self) -> None:
        overrides = WhisperParameterOverrides()
        assert overrides.temperature is None
        assert overrides.beam_size is None
        assert overrides.no_speech_threshold is None
        assert overrides.compression_ratio_threshold is None
        assert overrides.language_hint is None
        assert overrides.initial_prompt is None
        assert overrides.timeout_s is None
        assert overrides.model is None
        assert overrides.connection_id is None

    def test_temperature_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(temperature=-0.01)

    def test_temperature_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(temperature=1.01)

    def test_temperature_valid_range(self) -> None:
        for t in (0.0, 0.2, 0.5, 1.0):
            assert WhisperParameterOverrides(temperature=t).temperature == t

    def test_beam_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(beam_size=0)
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(beam_size=-1)

    def test_no_speech_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(no_speech_threshold=1.1)
        with pytest.raises(ValidationError):
            WhisperParameterOverrides(no_speech_threshold=-0.1)

    def test_model_swap_via_overrides(self) -> None:
        overrides = WhisperParameterOverrides(model="mlx-community/whisper-medium")
        assert overrides.model == "mlx-community/whisper-medium"

    def test_connection_id_via_overrides(self) -> None:
        overrides = WhisperParameterOverrides(connection_id="conn-other")
        assert overrides.connection_id == "conn-other"


# ---------------------------------------------------------------------------
# WhisperConnection.merge — per-session override application
# ---------------------------------------------------------------------------


class TestWhisperConnectionMerge:
    @pytest.fixture
    def base(self) -> WhisperConnection:
        return WhisperConnection(
            engine="openai_compatible",
            base_url="http://localhost:8005",
            model="mlx-community/whisper-large-v3-turbo",
            beam_size=1,
            temperature=0.0,
            no_speech_threshold=0.6,
        )

    def test_overrides_replace(self, base: WhisperConnection) -> None:
        merged = base.merge(WhisperParameterOverrides(beam_size=5))
        assert merged.beam_size == 5
        assert merged.temperature == 0.0
        assert merged.no_speech_threshold == 0.6

    def test_none_preserves_base(self, base: WhisperConnection) -> None:
        merged = base.merge(
            WhisperParameterOverrides(temperature=None, beam_size=3)
        )
        assert merged.temperature == 0.0  # untouched
        assert merged.beam_size == 3

    def test_returns_new_instance(self, base: WhisperConnection) -> None:
        merged = base.merge(WhisperParameterOverrides(beam_size=5))
        assert merged is not base
        assert base.beam_size == 1  # base immutable

    def test_partial_merge(self, base: WhisperConnection) -> None:
        merged = base.merge(WhisperParameterOverrides(language_hint="zh"))
        assert merged.language_hint == "zh"
        assert merged.engine == base.engine
        assert merged.base_url == base.base_url
        assert merged.model == base.model
        assert merged.beam_size == base.beam_size

    def test_merge_with_none_argument(self, base: WhisperConnection) -> None:
        merged = base.merge(None)
        assert merged == base

    def test_connection_id_not_applied_as_field(self, base: WhisperConnection) -> None:
        """connection_id is a routing hint for the resolver, not a field to copy."""
        merged = base.merge(WhisperParameterOverrides(connection_id="other"))
        assert merged == base

    def test_merge_all_overrides(self, base: WhisperConnection) -> None:
        overrides = WhisperParameterOverrides(
            temperature=0.4,
            beam_size=5,
            no_speech_threshold=0.4,
            compression_ratio_threshold=3.0,
            language_hint="ja",
            initial_prompt="A meeting transcript.",
            timeout_s=60.0,
        )
        merged = base.merge(overrides)
        assert merged.temperature == 0.4
        assert merged.beam_size == 5
        assert merged.no_speech_threshold == 0.4
        assert merged.compression_ratio_threshold == 3.0
        assert merged.language_hint == "ja"
        assert merged.initial_prompt == "A meeting transcript."
        assert merged.timeout_s == 60.0
