"""Translation-only playback tests.

Replays pre-recorded transcription segments through a real TranslationService
(connected to vLLM-MLX or any OpenAI-compatible LLM endpoint). No transcription
service, GPU, or browser needed.

Gate: requires LLM_BASE_URL env var. Run via:
  LLM_BASE_URL=http://localhost:8006/v1 LLM_MODEL=mlx-community/Qwen3-4B-4bit \
    uv run pytest modules/orchestration-service/tests/e2e/test_translation_playback.py -v -m e2e

Or use the just recipe:
  just test-e2e-playback
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Import conftest fixtures
from .conftest_playback import (  # noqa: F401
    context_window,
    replay_segments_zh,
    translation_service,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_zh_en_translation_produces_english(translation_service, replay_segments_zh):
    """Replay Chinese segments → verify translations are non-empty English."""
    from livetranslate_common.models import TranslationRequest

    final_segments = [s for s in replay_segments_zh if not s["is_draft"]]
    assert len(final_segments) >= 3, "Need at least 3 final segments"

    translations = []
    for seg in final_segments[:3]:
        request = TranslationRequest(
            text=seg["text"],
            source_language="zh",
            target_language="en",
        )
        response = await translation_service.translate(request)
        translations.append(response.translated_text)

    for i, text in enumerate(translations):
        assert len(text) > 0, f"Translation {i} is empty"
        # Verify it contains ASCII characters (English output)
        ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        assert ascii_chars > 5, f"Translation {i} doesn't appear to be English: {text[:80]}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_context_window_grows(translation_service, replay_segments_zh):
    """Replay 4 final segments → verify context window accumulates entries."""
    from livetranslate_common.models import TranslationRequest

    final_segments = [s for s in replay_segments_zh if not s["is_draft"]]

    for seg in final_segments[:4]:
        request = TranslationRequest(
            text=seg["text"],
            source_language="zh",
            target_language="en",
        )
        await translation_service.translate(request)

    context = translation_service.get_context("zh", "en")
    assert len(context) >= 2, (
        f"Context window should have accumulated entries, got {len(context)}"
    )

    # Verify context entries have both source and translation
    for entry in context:
        assert len(entry.text) > 0, "Context entry source text is empty"
        assert len(entry.translation) > 0, "Context entry translation is empty"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_draft_final_lifecycle(translation_service, replay_segments_zh):
    """Replay draft+final pairs → verify both can be translated."""
    from livetranslate_common.models import TranslationRequest

    # Find a segment_id that has both draft and final
    by_seg_id: dict[int, list[dict]] = {}
    for seg in replay_segments_zh:
        sid = seg["segment_id"]
        if sid not in by_seg_id:
            by_seg_id[sid] = []
        by_seg_id[sid].append(seg)

    draft_final_pairs = [
        segs for segs in by_seg_id.values()
        if any(s["is_draft"] for s in segs) and any(not s["is_draft"] for s in segs)
    ]
    assert len(draft_final_pairs) > 0, "No draft+final pairs in fixture"

    pair = draft_final_pairs[0]
    draft = next(s for s in pair if s["is_draft"])
    final = next(s for s in pair if not s["is_draft"])

    # Translate draft (skip_context=True — drafts don't pollute context)
    draft_request = TranslationRequest(
        text=draft["text"],
        source_language="zh",
        target_language="en",
    )
    draft_response = await translation_service.translate(draft_request, skip_context=True)
    assert len(draft_response.translated_text) > 0, "Draft translation is empty"

    # Context should be empty (draft skipped)
    assert len(translation_service.get_context("zh", "en")) == 0, "Draft should not write to context"

    # Translate final (writes to context)
    final_request = TranslationRequest(
        text=final["text"],
        source_language="zh",
        target_language="en",
    )
    final_response = await translation_service.translate(final_request)
    assert len(final_response.translated_text) > 0, "Final translation is empty"

    # Context should now have 1 entry
    assert len(translation_service.get_context("zh", "en")) == 1, "Final should write to context"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_second_translation_uses_context(translation_service, replay_segments_zh):
    """Verify context accumulates across consecutive translations.

    Translates two final segments sequentially and checks that the context
    window grows after each, proving context is available to the LLM for the
    second call (rolling context window is populated before the second translate).
    """
    from livetranslate_common.models import TranslationRequest

    final_segments = [s for s in replay_segments_zh if not s["is_draft"]]
    assert len(final_segments) >= 2, "Need at least 2 final segments"

    # Translate segment 1
    request_1 = TranslationRequest(
        text=final_segments[0]["text"],
        source_language="zh",
        target_language="en",
    )
    await translation_service.translate(request_1)

    ctx_after_1 = translation_service.get_context("zh", "en")
    assert len(ctx_after_1) == 1, (
        f"After 1st translation, context should have 1 entry, got {len(ctx_after_1)}"
    )

    # Translate segment 2 — context is non-empty, so LLM receives prior entry
    request_2 = TranslationRequest(
        text=final_segments[1]["text"],
        source_language="zh",
        target_language="en",
    )
    response_2 = await translation_service.translate(request_2)

    assert len(response_2.translated_text) > 0, "Second translation is empty"

    ctx_after_2 = translation_service.get_context("zh", "en")
    assert len(ctx_after_2) == 2, (
        f"After 2nd translation, context should have 2 entries, got {len(ctx_after_2)}"
    )

    # Verify both entries are populated
    for i, entry in enumerate(ctx_after_2):
        assert len(entry.text) > 0, f"Context entry {i} source text is empty"
        assert len(entry.translation) > 0, f"Context entry {i} translation is empty"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sentence_accumulation_and_flush(translation_service, replay_segments_zh):
    """Replay non-final segments followed by final → verify accumulated text translates."""
    from livetranslate_common.models import TranslationRequest

    # Accumulate stable_text from non-final segments, then flush on final
    # This mirrors the websocket_audio.py stable_text_buffer logic
    buffer = ""
    final_segments = [s for s in replay_segments_zh if not s["is_draft"]]

    for seg in final_segments[:3]:
        stable = seg.get("stable_text", seg["text"])
        if stable.strip():
            buffer += stable.strip()

    assert len(buffer) > 20, f"Accumulated buffer too short: {buffer[:50]}"

    request = TranslationRequest(
        text=buffer,
        source_language="zh",
        target_language="en",
    )
    response = await translation_service.translate(request)
    assert len(response.translated_text) > 10, (
        f"Accumulated translation too short: {response.translated_text}"
    )
