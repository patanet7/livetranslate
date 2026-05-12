"""Behavioral tests for LLMConnection and LLMParameterOverrides.

These are the canonical value objects for "talk to an LLM endpoint" used by
every service (orchestration translation, intelligence, fireflies, benchmark CLI).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from livetranslate_common.models.llm import LLMConnection, LLMParameterOverrides


# ---------------------------------------------------------------------------
# LLMConnection — construction & immutability
# ---------------------------------------------------------------------------


class TestLLMConnectionConstruction:
    def test_construct_minimal(self) -> None:
        """engine + base_url + model are required; other fields default."""
        conn = LLMConnection(
            engine="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen3:14b",
        )
        assert conn.engine == "ollama"
        assert conn.base_url == "http://localhost:11434/v1"
        assert conn.model == "qwen3:14b"
        # defaults
        assert conn.api_key == ""
        assert conn.temperature == 0.3
        assert conn.max_tokens == 1024
        assert conn.top_p == 0.8
        assert conn.top_k == 20
        assert conn.repetition_penalty == 1.05
        assert conn.presence_penalty == 1.5
        assert conn.timeout_s == 30.0
        assert conn.max_retries == 1
        assert conn.context_length is None
        assert conn.connection_id is None

    def test_construct_full(self) -> None:
        conn = LLMConnection(
            engine="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key="sk-test",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            presence_penalty=0.0,
            timeout_s=60.0,
            max_retries=3,
            context_length=128_000,
            connection_id="conn-openai-prod",
        )
        assert conn.api_key == "sk-test"
        assert conn.connection_id == "conn-openai-prod"
        assert conn.context_length == 128_000

    def test_invalid_engine_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConnection(
                engine="bogus_engine",  # type: ignore[arg-type]
                base_url="http://x",
                model="m",
            )

    def test_is_frozen(self) -> None:
        """Mutation must raise — LLMConnection is an immutable value object."""
        conn = LLMConnection(
            engine="ollama", base_url="http://x", model="qwen3:14b"
        )
        with pytest.raises(ValidationError):
            conn.temperature = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMConnection.is_ollama_qwen3 — replaces brittle "11434 in url" heuristic
# ---------------------------------------------------------------------------


class TestIsOllamaQwen3:
    def test_positive(self) -> None:
        conn = LLMConnection(
            engine="ollama", base_url="http://h:11434/v1", model="qwen3:14b"
        )
        assert conn.is_ollama_qwen3 is True

    def test_positive_non_default_port(self) -> None:
        """Detection must NOT depend on port — that was the original bug."""
        conn = LLMConnection(
            engine="ollama",
            base_url="http://100.64.0.2:8089/v1",
            model="qwen3-4b-vllm-dmr",
        )
        assert conn.is_ollama_qwen3 is True

    def test_wrong_engine(self) -> None:
        """qwen3 model on openai engine is NOT the Ollama-native case."""
        conn = LLMConnection(
            engine="openai",
            base_url="https://api.openai.com/v1",
            model="qwen3:14b",
        )
        assert conn.is_ollama_qwen3 is False

    def test_wrong_model(self) -> None:
        conn = LLMConnection(
            engine="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.1:8b",
        )
        assert conn.is_ollama_qwen3 is False

    def test_case_insensitive(self) -> None:
        conn = LLMConnection(
            engine="ollama", base_url="http://x", model="Qwen3-4B"
        )
        assert conn.is_ollama_qwen3 is True


# ---------------------------------------------------------------------------
# LLMParameterOverrides — all optional + range validators
# ---------------------------------------------------------------------------


class TestLLMParameterOverrides:
    def test_all_fields_optional(self) -> None:
        overrides = LLMParameterOverrides()
        assert overrides.temperature is None
        assert overrides.max_tokens is None
        assert overrides.top_p is None
        assert overrides.top_k is None
        assert overrides.repetition_penalty is None
        assert overrides.presence_penalty is None
        assert overrides.timeout_s is None
        assert overrides.model is None
        assert overrides.connection_id is None

    def test_temperature_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            LLMParameterOverrides(temperature=-0.01)

    def test_temperature_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            LLMParameterOverrides(temperature=2.01)

    def test_temperature_valid_range(self) -> None:
        for t in (0.0, 0.5, 1.0, 1.5, 2.0):
            assert LLMParameterOverrides(temperature=t).temperature == t

    def test_top_p_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            LLMParameterOverrides(top_p=1.5)
        with pytest.raises(ValidationError):
            LLMParameterOverrides(top_p=-0.1)

    def test_top_k_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMParameterOverrides(top_k=-1)

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            LLMParameterOverrides(max_tokens=0)
        with pytest.raises(ValidationError):
            LLMParameterOverrides(max_tokens=-100)

    def test_model_swap_via_overrides(self) -> None:
        overrides = LLMParameterOverrides(model="qwen3:32b")
        assert overrides.model == "qwen3:32b"

    def test_connection_id_via_overrides(self) -> None:
        overrides = LLMParameterOverrides(connection_id="conn-other")
        assert overrides.connection_id == "conn-other"


# ---------------------------------------------------------------------------
# LLMConnection.merge — per-session override application
# ---------------------------------------------------------------------------


class TestLLMConnectionMerge:
    @pytest.fixture
    def base(self) -> LLMConnection:
        return LLMConnection(
            engine="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen3:14b",
            temperature=0.7,
            max_tokens=512,
            top_p=0.8,
        )

    def test_overrides_replace(self, base: LLMConnection) -> None:
        merged = base.merge(LLMParameterOverrides(temperature=0.3))
        assert merged.temperature == 0.3
        # other base fields preserved
        assert merged.max_tokens == 512
        assert merged.top_p == 0.8

    def test_none_preserves_base(self, base: LLMConnection) -> None:
        merged = base.merge(LLMParameterOverrides(temperature=None, top_p=0.9))
        assert merged.temperature == 0.7  # untouched
        assert merged.top_p == 0.9  # overridden

    def test_returns_new_instance(self, base: LLMConnection) -> None:
        merged = base.merge(LLMParameterOverrides(temperature=0.1))
        assert merged is not base
        assert base.temperature == 0.7  # base immutable

    def test_partial_merge(self, base: LLMConnection) -> None:
        """Only specified fields change; others (including non-sampling) preserved."""
        merged = base.merge(LLMParameterOverrides(max_tokens=2048))
        assert merged.max_tokens == 2048
        assert merged.engine == base.engine
        assert merged.base_url == base.base_url
        assert merged.model == base.model
        assert merged.temperature == base.temperature

    def test_model_swap_recomputes_is_ollama_qwen3(self, base: LLMConnection) -> None:
        """Swapping to a non-qwen3 model on Ollama should flip is_ollama_qwen3 to False."""
        merged = base.merge(LLMParameterOverrides(model="llama3.1:8b"))
        assert merged.model == "llama3.1:8b"
        assert merged.is_ollama_qwen3 is False
        # base unchanged
        assert base.is_ollama_qwen3 is True

    def test_merge_with_none_overrides_argument(self, base: LLMConnection) -> None:
        """merge(None) returns base equivalent — convenience for optional overrides."""
        merged = base.merge(None)
        assert merged == base

    def test_merge_all_sampling_fields(self, base: LLMConnection) -> None:
        overrides = LLMParameterOverrides(
            temperature=0.1,
            max_tokens=99,
            top_p=0.42,
            top_k=7,
            repetition_penalty=1.2,
            presence_penalty=0.5,
            timeout_s=15.0,
        )
        merged = base.merge(overrides)
        assert merged.temperature == 0.1
        assert merged.max_tokens == 99
        assert merged.top_p == 0.42
        assert merged.top_k == 7
        assert merged.repetition_penalty == 1.2
        assert merged.presence_penalty == 0.5
        assert merged.timeout_s == 15.0
