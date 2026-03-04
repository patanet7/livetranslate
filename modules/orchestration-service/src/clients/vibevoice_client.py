"""
VibeVoice-ASR vLLM HTTP Client

Thin async HTTP client for communicating with a VibeVoice-ASR model served
via a vLLM-compatible endpoint.

VibeVoice-ASR returns a JSON array of diarized transcription segments:
    [{"speaker": 0, "start": 0.5, "end": 3.2, "text": "Hello there"}, ...]

The client parses this output into the shared ``TranscribeResponse`` model
and exposes a simple async interface for health-checking and transcription.

Reference:
- https://github.com/BerriAI/litellm (vLLM-compatible server)
- VibeVoice-ASR model card
"""

from __future__ import annotations

import json
from typing import Any

import aiohttp
from livetranslate_common.logging import get_logger

from models.diarization import TranscribeResponse, TranscribeSegment

logger = get_logger()


# =============================================================================
# Constants
# =============================================================================

# Wall-clock timeout for a single transcription request.
# A 60-minute audio file can legitimately take 15+ minutes to process on CPU.
TRANSCRIBE_TIMEOUT_SECONDS = 1800

# Short timeout for lightweight health / readiness probes.
HEALTH_CHECK_TIMEOUT_SECONDS = 10

# Default VibeVoice language reported when the model does not emit one.
DEFAULT_LANGUAGE = "en"


# =============================================================================
# Exceptions
# =============================================================================


class VibeVoiceError(Exception):
    """Raised when the VibeVoice-ASR service returns an error or is unreachable.

    Args:
        message: Human-readable description of the error.
        status_code: Optional HTTP status code returned by the server.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# =============================================================================
# Client
# =============================================================================


class VibeVoiceClient:
    """Async HTTP client for the VibeVoice-ASR vLLM inference server.

    Handles audio upload, response parsing, and basic health monitoring.
    The client does **not** manage a persistent ``aiohttp.ClientSession``
    internally; a new session is created per request so that the client
    remains safe to instantiate at module import time without a running
    event loop.

    Args:
        base_url: Base URL of the vLLM-compatible server, e.g.
            ``"http://localhost:8000/v1"``.  Trailing slashes are stripped.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url: str = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_vibevoice_output(
        self,
        raw: str,
        *,
        duration_seconds: float,
        processing_time: float,
    ) -> TranscribeResponse:
        """Parse raw JSON output from VibeVoice-ASR into a ``TranscribeResponse``.

        VibeVoice-ASR returns a JSON array of segment objects.  Each object
        has ``speaker`` (int), ``start`` (float), ``end`` (float), and
        ``text`` (str) fields.  The ``detected_language`` field is not
        currently emitted by the model, so it falls back to ``"en"``.

        On any parse failure (invalid JSON, unexpected schema) the method
        returns an empty ``TranscribeResponse`` rather than raising, so that
        callers can degrade gracefully.

        Args:
            raw: Raw JSON string returned by the VibeVoice-ASR endpoint.
            duration_seconds: Total audio duration in seconds (supplied by
                the caller because the model does not emit it).
            processing_time: Wall-clock seconds spent waiting for the
                inference request to complete.

        Returns:
            A fully populated ``TranscribeResponse`` with parsed segments,
            speaker count, language, duration, and processing time.
        """
        try:
            data: list[dict[str, Any]] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "vibevoice_parse_error",
                reason="invalid_json",
                raw_preview=raw[:200],
            )
            return TranscribeResponse(
                segments=[],
                detected_language=DEFAULT_LANGUAGE,
                num_speakers=0,
                duration_seconds=duration_seconds,
                processing_time_seconds=processing_time,
            )

        if not isinstance(data, list):
            logger.warning(
                "vibevoice_parse_error",
                reason="expected_list",
                got_type=type(data).__name__,
            )
            return TranscribeResponse(
                segments=[],
                detected_language=DEFAULT_LANGUAGE,
                num_speakers=0,
                duration_seconds=duration_seconds,
                processing_time_seconds=processing_time,
            )

        segments: list[TranscribeSegment] = []
        speaker_ids: set[int] = set()

        for item in data:
            try:
                seg = TranscribeSegment(
                    speaker=int(item["speaker"]),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]),
                )
                segments.append(seg)
                speaker_ids.add(seg.speaker)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "vibevoice_segment_parse_error",
                    error=str(exc),
                    item=item,
                )
                continue

        return TranscribeResponse(
            segments=segments,
            detected_language=DEFAULT_LANGUAGE,
            num_speakers=len(speaker_ids),
            duration_seconds=duration_seconds,
            processing_time_seconds=processing_time,
        )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        hotwords: list[str] | None = None,
    ) -> TranscribeResponse:
        """Upload an audio file to VibeVoice-ASR and return parsed results.

        Sends a multipart POST to ``{base_url}/audio/transcriptions`` with
        the audio payload.  The long ``TRANSCRIBE_TIMEOUT_SECONDS`` timeout
        accommodates large audio files that may take many minutes to process.

        Args:
            audio_bytes: Raw audio file content (WAV, MP3, FLAC, …).
            filename: Filename hint sent to the server (affects MIME detection).
            hotwords: Optional list of domain-specific hotwords to bias the
                recognition model (e.g. ``["OpenVINO", "VibeVoice"]``).

        Returns:
            Parsed ``TranscribeResponse`` with diarized segments.

        Raises:
            VibeVoiceError: If the server returns a non-2xx status or the
                request times out.
        """
        import time

        url = f"{self.base_url}/audio/transcriptions"
        timeout = aiohttp.ClientTimeout(total=TRANSCRIBE_TIMEOUT_SECONDS)

        form = aiohttp.FormData()
        form.add_field("file", audio_bytes, filename=filename)
        if hotwords:
            form.add_field("hotwords", " ".join(hotwords))

        logger.info(
            "vibevoice_transcribe_start",
            url=url,
            audio_bytes=len(audio_bytes),
            filename=filename,
            hotwords=hotwords,
        )

        t0 = time.monotonic()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, data=form) as response:
                    elapsed = time.monotonic() - t0
                    body = await response.text()

                    if response.status >= 400:
                        logger.error(
                            "vibevoice_transcribe_error",
                            status=response.status,
                            body_preview=body[:500],
                        )
                        raise VibeVoiceError(
                            f"VibeVoice-ASR returned HTTP {response.status}: {body[:200]}",
                            status_code=response.status,
                        )

                    logger.info(
                        "vibevoice_transcribe_complete",
                        status=response.status,
                        elapsed_seconds=round(elapsed, 2),
                    )
                    return self.parse_vibevoice_output(
                        body,
                        duration_seconds=0.0,  # model does not emit duration
                        processing_time=elapsed,
                    )

            except aiohttp.ClientError as exc:
                raise VibeVoiceError(f"VibeVoice-ASR request failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Check whether the VibeVoice-ASR server is reachable and has models loaded.

        Sends a GET request to ``{base_url}/models`` (the standard vLLM
        health endpoint) and returns ``True`` if the response status is 2xx.

        Returns:
            ``True`` if the server responded successfully, ``False`` otherwise.
        """
        url = f"{self.base_url}/models"
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SECONDS)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    healthy = response.status < 400
                    logger.info(
                        "vibevoice_health_check",
                        url=url,
                        status=response.status,
                        healthy=healthy,
                    )
                    return healthy
        except aiohttp.ClientError as exc:
            logger.warning("vibevoice_health_check_failed", url=url, error=str(exc))
            return False
