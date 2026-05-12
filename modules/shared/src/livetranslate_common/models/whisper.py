"""Canonical value objects for talking to a Whisper transcription endpoint.

`WhisperConnection` is the single shape every service uses to identify *which*
Whisper backend to call and *how*. Mirror of `LLMConnection`:

- "openai_compatible" engine covers vllm-mlx, vllm, faster-whisper-server,
  whisper.cpp's server, and OpenAI proper — all speak POST multipart
  /v1/audio/transcriptions and accept Authorization: Bearer auth.
- "mlx_local" / "faster_whisper_local" engines describe in-process libraries
  loaded by the transcription-service backend manager; base_url + api_key are
  inert for those.

`WhisperParameterOverrides` carries per-session tunables (beam_size,
no_speech_threshold, language_hint, initial_prompt, ...) that flow from the
dashboard toolbar to the Whisper request body. Mirror of `LLMParameterOverrides`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

WhisperEngine = Literal["openai_compatible", "mlx_local", "faster_whisper_local"]

# Whisper compute-type enum — accepted by every Whisper backend we support
# (faster-whisper, mlx-whisper, vllm-mlx Whisper). Free-string was too loose:
# a typo like `"flot16"` would parse fine, then fail at backend load time
# with a far less helpful error.
ComputeType = Literal["float16", "float32", "int8", "int8_float16"]

# BCP-47 codes Whisper itself supports — sourced from openai/whisper's
# tokenizer language map. Anything outside this set will produce gibberish
# transcription, so we reject it at the model boundary instead of letting
# the request travel to the backend and fail late.
WHISPER_LANGUAGES: frozenset[str] = frozenset(
    [
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
        "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
        "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
        "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
        "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
        "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
        "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
        "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
        "yi", "yo", "zh",
    ]
)


def _validate_language_code(v: str | None) -> str | None:
    """Field validator — None passes; any other value must be a Whisper BCP-47 code."""
    if v is None:
        return v
    normalized = v.strip().lower()
    if not normalized:
        return None
    if normalized not in WHISPER_LANGUAGES:
        raise ValueError(
            f"language_hint must be one of the {len(WHISPER_LANGUAGES)} BCP-47 "
            f"codes Whisper supports; got {v!r}"
        )
    return normalized


class WhisperConnection(BaseModel):
    """Immutable description of a Whisper endpoint + default decoding parameters.

    The resolver normalizes inputs from any source (DB row, env vars, hard-coded
    defaults) into this shape. `VLLMWhisperBackend` takes one of these in its
    constructor; per-call `WhisperParameterOverrides` are layered via `merge()`.
    """

    # Forward-compatible with future per-model manifests (e.g., custom
    # initial_prompt sets per language, batch_profile hints).
    model_config = ConfigDict(frozen=True, extra="ignore")

    engine: WhisperEngine
    base_url: str
    model: str

    api_key: str = ""

    # Whisper sampling / decoding parameters
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    beam_size: int = Field(default=1, gt=0)
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    compression_ratio_threshold: float = Field(default=2.4, gt=0.0)
    language_hint: str | None = None
    initial_prompt: str | None = None
    compute_type: ComputeType = "float16"

    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=1, ge=0)

    connection_id: str | None = None

    @field_validator("language_hint", mode="before")
    @classmethod
    def _check_language_hint(cls, v: str | None) -> str | None:
        return _validate_language_code(v)

    @property
    def requires_http(self) -> bool:
        """True for engines that go over HTTP (need base_url + auth)."""
        return self.engine == "openai_compatible"

    def merge(self, overrides: WhisperParameterOverrides | None) -> WhisperConnection:
        """Return a new WhisperConnection with non-None override fields applied.

        Pure-functional: never mutates self (frozen=True enforces it anyway).
        Caller pattern:
            effective = self._base_connection.merge(session_overrides)
            await backend.transcribe(audio, **effective.decoding_params())
        """
        if overrides is None:
            return self
        patch = overrides.model_dump(exclude_none=True)
        # connection_id is a routing field for the resolver, not a field
        # to copy onto the current connection.
        patch.pop("connection_id", None)
        if not patch:
            return self
        return self.model_copy(update=patch)


class WhisperParameterOverrides(BaseModel):
    """Per-session, per-call optional overrides for Whisper decoding parameters.

    Flows from the dashboard toolbar → WS ConfigMessage.whisper → SessionConfig →
    VLLMWhisperBackend.transcribe(...). Every field is optional; only specified
    fields override the base WhisperConnection.

    `connection_id` and `model` exist here so the user can swap backend or
    model mid-session without resetting the WebSocket.
    """

    model_config = ConfigDict(extra="ignore")

    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    beam_size: int | None = Field(default=None, gt=0)
    no_speech_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    compression_ratio_threshold: float | None = Field(default=None, gt=0.0)
    language_hint: str | None = None
    initial_prompt: str | None = None
    timeout_s: float | None = Field(default=None, gt=0.0)

    model: str | None = None
    connection_id: str | None = None

    @field_validator("language_hint", mode="before")
    @classmethod
    def _check_language_hint(cls, v: str | None) -> str | None:
        return _validate_language_code(v)


WhisperConnection.model_rebuild()
