"""FirefliesSource — replay a Fireflies transcript through TranslationService.

Skips transcription (Fireflies *is* the transcription) but still routes through
the production translation stack (TranslationService → LLM client → context
store) so the demo exercises the same path as the live mic source.

Dependency-injected so unit tests can pass fixture transcripts and a recording
translator. Production callers use :func:`from_config` which constructs the real
:class:`FirefliesClient` and :class:`TranslationService` from env.
"""
from __future__ import annotations

import asyncio
from hashlib import md5
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from .base import SubtitleSource
from services.pipeline.adapters.source_adapter import CaptionEvent

TranscriptProvider = Callable[[str], Awaitable[dict[str, Any]]]


class _Translator(Protocol):
    async def translate_text(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> str: ...


class FirefliesSource(SubtitleSource):
    def __init__(
        self,
        *,
        meeting_id: str,
        target_language: str,
        replay_speed: float = 1.0,
        source_language: str = "auto",
        transcript_provider: TranscriptProvider,
        translator: _Translator,
    ) -> None:
        self.meeting_id = meeting_id
        self.target_language = target_language
        self.source_language = source_language
        self.replay_speed = float(replay_speed)
        self._provider = transcript_provider
        self._translator = translator

    async def stream(self) -> AsyncIterator[CaptionEvent]:
        transcript = await self._provider(self.meeting_id)
        sentences = transcript.get("sentences") or []
        prev_start: float | None = None
        for s in sentences:
            start = float(s.get("start_time") or 0.0)
            if self.replay_speed > 0.0 and prev_start is not None:
                delta = (start - prev_start) / self.replay_speed
                if delta > 0:
                    await asyncio.sleep(delta)
            prev_start = start
            text = s.get("text") or ""
            speaker_name = s.get("speaker_name")
            speaker_id_raw = s.get("speaker_id")
            speaker_id = (
                str(speaker_id_raw) if speaker_id_raw is not None else None
            )
            translated = await self._translator.translate_text(
                text,
                source_language=self.source_language,
                target_language=self.target_language,
            )
            yield CaptionEvent(
                event_type="added",
                caption_id=_caption_id(self.meeting_id, s),
                text=text,
                translated_text=translated,
                speaker_name=speaker_name,
                speaker_id=speaker_id,
                source_lang=self.source_language,
                target_lang=self.target_language,
                confidence=1.0,
                is_draft=False,
            )

    @classmethod
    def from_config(cls, config) -> "FirefliesSource":
        """Build with real FirefliesClient + real TranslationService.

        Pulled out behind a factory so tests don't pull in HTTP clients.
        Lazy imports because the orchestration deps are heavy.
        """
        import os

        from clients.fireflies_client import FirefliesClient  # type: ignore
        from translation.config import TranslationConfig  # type: ignore
        from translation.service import TranslationService  # type: ignore

        api_key = os.environ.get("FIREFLIES_API_KEY")
        if not api_key:
            raise RuntimeError("FIREFLIES_API_KEY not set")

        client = FirefliesClient(api_key=api_key)

        async def provider(meeting_id: str) -> dict[str, Any]:
            data = await client.get_transcript_detail(meeting_id)
            if not data:
                raise RuntimeError(f"Transcript {meeting_id!r} not found")
            return data

        translator_cfg = TranslationConfig()
        service = TranslationService(translator_cfg)

        async def translate_text(text: str, *, source_language: str, target_language: str) -> str:
            from livetranslate_common.models.translation import TranslationRequest

            req = TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
            )
            resp = await service.translate(req)
            return resp.translated_text

        class _Adapter:
            async def translate_text(self, text, *, source_language, target_language):
                return await translate_text(
                    text, source_language=source_language, target_language=target_language
                )

        return cls(
            meeting_id=config.fireflies_meeting_id,
            target_language=config.target_language,
            source_language=config.source_language,
            replay_speed=config.fireflies_replay_speed,
            transcript_provider=provider,
            translator=_Adapter(),
        )


def _caption_id(meeting_id: str, sentence: dict[str, Any]) -> str:
    """Stable per-(meeting, sentence) id for replay reproducibility."""
    idx = sentence.get("index")
    if idx is None:
        # Fall back to hash of (start_time, text) — still deterministic.
        key = f"{sentence.get('start_time')}|{sentence.get('text','')}"
        return f"ff-{meeting_id}-{md5(key.encode()).hexdigest()[:10]}"
    return f"ff-{meeting_id}-{int(idx):04d}"
