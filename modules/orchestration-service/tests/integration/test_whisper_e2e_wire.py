"""End-to-end wire test: VLLMWhisperBackend → FakeWhisperServer over real HTTP.

Proves the full Whisper request path without needing live audio or a real GPU:
  1. resolve_whisper_connection(...) yields a WhisperConnection
  2. VLLMWhisperBackend(connection=...) constructs an httpx client with the
     correct Authorization: Bearer header
  3. backend.transcribe(audio) writes a multipart WAV upload to the fake server
  4. The fake server's recorded_requests captures the auth header and form fields
  5. The fake returns a verbose_json response; the backend parses it correctly
  6. The standardized tracing events fire (whisper.request.start/.complete)

This is the CI-friendly version of P10's "live Docker smoke" — same test path,
no infrastructure dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make transcription-service backends importable in this orchestration test.
_TX_SRC = Path(__file__).resolve().parents[3] / "transcription-service" / "src"
if str(_TX_SRC) not in sys.path:
    sys.path.insert(0, str(_TX_SRC))


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_wire_auth_header_flows_to_server(
    fake_whisper_server, whisper_connection_factory
) -> None:
    """Authorization: Bearer X must arrive at the server when the connection
    has an api_key. This is the original bug — VLLMWhisperBackend used to have
    no auth path at all."""
    from backends.vllm_whisper import VLLMWhisperBackend  # noqa: I001

    fake_whisper_server.require_api_key("dummy")
    fake_whisper_server.set_response_text("hello world")

    conn = whisper_connection_factory(
        base_url=fake_whisper_server.base_url,
        api_key="dummy",
        model="mlx-community/whisper-large-v3-turbo",
    )
    backend = VLLMWhisperBackend(connection=conn)
    await backend.load_model(conn.model)

    audio = np.zeros(16000, dtype=np.float32)  # 1s silence
    result = await backend.transcribe(audio, language="en")
    await backend.unload_model()

    assert result.text == "hello world"
    # Verify the auth header reached the server
    recorded = fake_whisper_server.recorded_requests
    transcribe_calls = [
        r for r in recorded if r["path"] == "/v1/audio/transcriptions"
    ]
    assert len(transcribe_calls) >= 1
    auth_header = transcribe_calls[0]["headers"].get("authorization", "")
    assert auth_header == "Bearer dummy"


async def test_wire_no_auth_when_api_key_empty(
    fake_whisper_server, whisper_connection_factory
) -> None:
    """Empty api_key → NO Authorization header (don't send empty Bearer)."""
    from backends.vllm_whisper import VLLMWhisperBackend

    fake_whisper_server.set_response_text("plain text")

    conn = whisper_connection_factory(
        base_url=fake_whisper_server.base_url,
        api_key="",
        model="m",
    )
    backend = VLLMWhisperBackend(connection=conn)
    await backend.load_model(conn.model)
    await backend.transcribe(np.zeros(8000, dtype=np.float32), language="en")
    await backend.unload_model()

    recorded = fake_whisper_server.recorded_requests
    transcribe_calls = [
        r for r in recorded if r["path"] == "/v1/audio/transcriptions"
    ]
    assert "authorization" not in transcribe_calls[0]["headers"]


async def test_wire_form_fields_carry_correct_model_and_language(
    fake_whisper_server, whisper_connection_factory
) -> None:
    """Verify model + language survive the multipart serialization."""
    from backends.vllm_whisper import VLLMWhisperBackend

    fake_whisper_server.set_response_text("一段中文")

    conn = whisper_connection_factory(
        base_url=fake_whisper_server.base_url,
        api_key="dummy",
        model="mlx-community/whisper-medium",
    )
    backend = VLLMWhisperBackend(connection=conn)
    await backend.load_model(conn.model)
    await backend.transcribe(np.zeros(16000, dtype=np.float32), language="zh")
    await backend.unload_model()

    recorded = fake_whisper_server.recorded_requests
    transcribe_calls = [
        r for r in recorded if r["path"] == "/v1/audio/transcriptions"
    ]
    form = transcribe_calls[0]["form_fields"]
    assert form.get("model") == "mlx-community/whisper-medium"
    assert form.get("language") == "zh"
    assert form.get("response_format") == "verbose_json"
    assert transcribe_calls[0]["audio_size"] > 0


async def test_wire_unauthorized_propagates(
    fake_whisper_server, whisper_connection_factory
) -> None:
    """Wrong api_key → server 401s → backend raises (no silent failure)."""
    import httpx

    from backends.vllm_whisper import VLLMWhisperBackend

    fake_whisper_server.require_api_key("expected-key")

    conn = whisper_connection_factory(
        base_url=fake_whisper_server.base_url,
        api_key="wrong-key",
        model="m",
    )
    backend = VLLMWhisperBackend(connection=conn)
    await backend.load_model(conn.model)

    with pytest.raises(httpx.HTTPStatusError):
        await backend.transcribe(np.zeros(8000, dtype=np.float32), language="en")
    await backend.unload_model()


async def test_wire_tracing_emits_start_and_complete(
    fake_whisper_server, whisper_connection_factory, capsys
) -> None:
    """The standardized whisper.request.start/.complete events fire end-to-end."""
    import json

    from livetranslate_common.logging import setup_logging

    from backends.vllm_whisper import VLLMWhisperBackend

    setup_logging(service_name="test-e2e", log_format="json")

    fake_whisper_server.set_response_text("traced text")
    conn = whisper_connection_factory(
        base_url=fake_whisper_server.base_url,
        api_key="",
        model="m",
    )
    backend = VLLMWhisperBackend(connection=conn)
    await backend.load_model(conn.model)
    capsys.readouterr()  # drain pre-transcribe logs
    await backend.transcribe(np.zeros(8000, dtype=np.float32), language="en")
    await backend.unload_model()

    out = capsys.readouterr()
    events = []
    for line in out.err.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    names = [e.get("event") for e in events]
    assert "whisper.request.start" in names
    assert "whisper.request.complete" in names
    complete = next(e for e in events if e.get("event") == "whisper.request.complete")
    assert complete["language_detected"] == "en"
    assert complete["text_chars"] == len("traced text")
    assert complete["engine"] == "openai_compatible"
    assert complete["model"] == "m"
    assert isinstance(complete["duration_ms"], float)
