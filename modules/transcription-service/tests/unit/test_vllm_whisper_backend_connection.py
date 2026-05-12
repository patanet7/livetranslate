"""Unit tests for VLLMWhisperBackend's new WhisperConnection plumbing.

Verifies the constructor:
  - accepts a WhisperConnection and reads URL/model/api_key/compute_type from it
  - falls back to legacy positional args + env vars when no connection passed
  - prefers VLLM_MLX_API_KEY > LLM_API_KEY for the auth header
  - sets Authorization: Bearer header when api_key is non-empty
  - omits Authorization header when api_key is empty (no auth)

No httpx requests are made — we inspect the httpx.AsyncClient instance after
load_model() is awaited (the client is created lazily there).
"""

from __future__ import annotations

import pytest

from livetranslate_common.models.whisper import WhisperConnection
from backends.vllm_whisper import VLLMWhisperBackend


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in ("VLLM_MLX_URL", "VLLM_MLX_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)


class TestConstructorWithConnection:
    def test_reads_url_from_connection(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://remote:9001",
            model="mlx-community/whisper-large-v3-turbo",
        )
        backend = VLLMWhisperBackend(connection=conn)
        assert backend._base_url == "http://remote:9001"

    def test_reads_api_key_from_connection(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
            api_key="bearer-tok",
        )
        backend = VLLMWhisperBackend(connection=conn)
        assert backend._api_key == "bearer-tok"

    def test_reads_compute_type_from_connection(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="m",
            compute_type="int8",
        )
        backend = VLLMWhisperBackend(connection=conn)
        assert backend._compute_type == "int8"

    def test_connection_model_overrides_positional_arg(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://x",
            model="mlx-community/whisper-medium",
        )
        backend = VLLMWhisperBackend(model_name="large-v3-turbo", connection=conn)
        # connection.model wins
        assert backend._model_name == "mlx-community/whisper-medium"


class TestConstructorEnvFallback:
    def test_url_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_MLX_URL", "http://from-env:8005")
        backend = VLLMWhisperBackend()
        assert backend._base_url == "http://from-env:8005"

    def test_api_key_prefers_vllm_specific(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_MLX_API_KEY", "whisper-tok")
        monkeypatch.setenv("LLM_API_KEY", "llm-tok")
        backend = VLLMWhisperBackend()
        assert backend._api_key == "whisper-tok"

    def test_api_key_falls_back_to_llm_key(self, monkeypatch) -> None:
        """When VLLM_MLX_API_KEY is absent, LLM_API_KEY is used (shared bearer)."""
        monkeypatch.setenv("LLM_API_KEY", "shared-bearer")
        backend = VLLMWhisperBackend()
        assert backend._api_key == "shared-bearer"

    def test_no_api_key_means_empty(self) -> None:
        backend = VLLMWhisperBackend()
        assert backend._api_key == ""


@pytest.mark.asyncio
class TestAuthHeader:
    async def test_authorization_header_set_when_api_key_present(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://localhost:65535",  # unreachable; we never connect
            model="m",
            api_key="secret",
        )
        backend = VLLMWhisperBackend(connection=conn)
        # load_model() builds the httpx client; it also probes /health which
        # will fail — that's fine, we only care about the headers map.
        try:
            await backend.load_model("m")
        except Exception:
            pass
        assert backend._client is not None
        assert backend._client.headers.get("Authorization") == "Bearer secret"

    async def test_no_authorization_header_when_api_key_empty(self) -> None:
        conn = WhisperConnection(
            engine="openai_compatible",
            base_url="http://localhost:65535",
            model="m",
            api_key="",
        )
        backend = VLLMWhisperBackend(connection=conn)
        try:
            await backend.load_model("m")
        except Exception:
            pass
        assert backend._client is not None
        assert "Authorization" not in backend._client.headers
