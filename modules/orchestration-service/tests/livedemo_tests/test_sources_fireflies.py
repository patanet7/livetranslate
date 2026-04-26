"""Tests for sources/fireflies.py — GraphQL replay → TranslationService → CaptionEvents.

We use dependency injection to keep tests offline-fast (no real GraphQL, no real
LLM) while exercising the *real* source logic. A fixture dict represents the
real Fireflies transcript shape; a recording translator stand-in implements the
TranslationService callable surface (the source only calls one method).

The truly-behavioral E2E test (real GraphQL + real LLM) lives in
test_fireflies_e2e.py and is skipped when env keys/services are missing.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from livedemo.sources.fireflies import FirefliesSource
from services.pipeline.adapters.source_adapter import CaptionEvent


# Real-shape fixture mirroring Fireflies' transcript(id) GraphQL response.
SAMPLE_TRANSCRIPT: dict[str, Any] = {
    "id": "T-DEMO-1",
    "title": "Q3 Planning Demo",
    "sentences": [
        {
            "index": 0,
            "text": "Hello everyone, welcome to the meeting.",
            "speaker_name": "Alice",
            "speaker_id": 0,
            "start_time": 0.0,
            "end_time": 2.4,
        },
        {
            "index": 1,
            "text": "Thank you, nice to meet you all.",
            "speaker_name": "Bob",
            "speaker_id": 1,
            "start_time": 2.6,
            "end_time": 4.5,
        },
        {
            "index": 2,
            "text": "Let's go through the agenda for today.",
            "speaker_name": "Alice",
            "speaker_id": 0,
            "start_time": 5.0,
            "end_time": 7.2,
        },
    ],
}


class _RecordingTranslator:
    """Stand-in for TranslationService.translate_text — records calls, returns echoes.

    NOT a mock of business logic — the source's contract is "calls
    translate_text once per sentence and consumes the returned string". This
    fake makes the contract testable without a running LLM.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def translate_text(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> str:
        self.calls.append(
            {"text": text, "src": source_language, "tgt": target_language}
        )
        return f"[{target_language}] {text}"


async def _fixture_provider(transcript_id: str) -> dict[str, Any]:
    return SAMPLE_TRANSCRIPT


@pytest.mark.asyncio
async def test_fireflies_source_yields_one_caption_per_sentence():
    src = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="zh",
        replay_speed=0.0,
        transcript_provider=_fixture_provider,
        translator=_RecordingTranslator(),
    )
    out: list[CaptionEvent] = []
    async for evt in src.stream():
        out.append(evt)
    assert len(out) == 3
    assert out[0].text == "Hello everyone, welcome to the meeting."
    assert out[0].speaker_name == "Alice"
    assert out[0].source_lang in ("auto", "en")
    assert out[0].target_lang == "zh"
    assert out[0].translated_text and out[0].translated_text.startswith("[zh]")


@pytest.mark.asyncio
async def test_fireflies_source_routes_through_translation_service():
    """B9 — translator.translate_text called once per sentence with right args."""
    translator = _RecordingTranslator()
    src = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="zh",
        replay_speed=0.0,
        transcript_provider=_fixture_provider,
        translator=translator,
    )
    [_ async for _ in src.stream()]
    assert len(translator.calls) == 3
    assert all(c["tgt"] == "zh" for c in translator.calls)
    assert translator.calls[0]["text"].startswith("Hello")
    assert translator.calls[1]["text"].startswith("Thank")


@pytest.mark.asyncio
async def test_fireflies_source_caption_id_is_stable_per_sentence():
    """Caption IDs must be derivable from (meeting_id, sentence index) so reruns match."""
    translator = _RecordingTranslator()
    src1 = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="zh",
        replay_speed=0.0,
        transcript_provider=_fixture_provider,
        translator=translator,
    )
    src2 = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="zh",
        replay_speed=0.0,
        transcript_provider=_fixture_provider,
        translator=_RecordingTranslator(),
    )
    ids1 = [e.caption_id async for e in src1.stream()]
    ids2 = [e.caption_id async for e in src2.stream()]
    assert ids1 == ids2


@pytest.mark.asyncio
async def test_fireflies_source_replay_speed_compresses_time():
    """replay_speed=10x: ~5s of sentences should finish well under 1s wall-clock."""
    translator = _RecordingTranslator()
    src = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="zh",
        replay_speed=10.0,
        transcript_provider=_fixture_provider,
        translator=translator,
    )
    t0 = time.monotonic()
    [_ async for _ in src.stream()]
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"replay_speed=10x took {elapsed:.2f}s — expected <1s"


@pytest.mark.asyncio
async def test_fireflies_source_speaker_id_stringified():
    """Numeric speaker_id from Fireflies should become string (matches CaptionEvent type)."""
    src = FirefliesSource(
        meeting_id="T-DEMO-1",
        target_language="en",
        replay_speed=0.0,
        transcript_provider=_fixture_provider,
        translator=_RecordingTranslator(),
    )
    out = [e async for e in src.stream()]
    assert all(isinstance(e.speaker_id, str) or e.speaker_id is None for e in out)
    # Speakers 0,1,0 → distinct strings preserved
    assert out[0].speaker_id == out[2].speaker_id
    assert out[0].speaker_id != out[1].speaker_id
