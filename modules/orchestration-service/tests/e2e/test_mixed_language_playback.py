"""B1.3 / B1.4 / B1.5: Mixed-language playback and context isolation tests.

B1.3: Replay mixed zh/en transcription segments and verify that ZH segments
      are routed to English translation while EN segments are skipped (same
      as the effective target language).

B1.4: Translate 3 ZH→EN segments, then 2 EN→ZH segments WITHOUT clearing
      context. Assert both directional windows retain their entries, proving
      DirectionalContextStore isolation without requiring clear_context().

B1.5: Translate English segments with target_language="zh" and verify the
      output contains CJK characters.

Gate: requires LLM_BASE_URL env var.

Run via:
  LLM_BASE_URL=http://localhost:8006/v1 \\
    uv run pytest modules/orchestration-service/tests/e2e/test_mixed_language_playback.py -v -m e2e
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Import shared fixtures
from .conftest_playback import (  # noqa: F401
    context_window,
    translation_service,
)

TRANSCRIPTION_FIXTURES = (
    Path(__file__).parent.parent.parent.parent
    / "transcription-service"
    / "tests"
    / "fixtures"
    / "audio"
)

# Mixed language replay fixture — either a pre-generated JSON replay or built
# inline from the reference text file.
MIXED_REPLAY_PATH = TRANSCRIPTION_FIXTURES / "meeting_mixed_zh_en_transcription_replay_v2.json"
MIXED_REFERENCE_TXT = TRANSCRIPTION_FIXTURES / "meeting_mixed_zh_en.txt"


def _skip_if_no_llm() -> None:
    if not os.getenv("LLM_BASE_URL"):
        pytest.skip("LLM_BASE_URL not set — skip mixed language playback tests")


def _load_mixed_segments() -> list[dict]:
    """Load mixed-language replay fixture.

    Prefers the JSON replay fixture. Falls back to parsing the reference TXT
    file and constructing minimal segment dicts (final-only, no draft/stable).
    """
    if MIXED_REPLAY_PATH.exists():
        with open(MIXED_REPLAY_PATH) as f:
            data = json.load(f)
        return data["segments"]

    if not MIXED_REFERENCE_TXT.exists():
        pytest.skip(
            f"Mixed language fixture not found: {MIXED_REFERENCE_TXT} "
            f"(and no replay JSON at {MIXED_REPLAY_PATH})"
        )

    # Build minimal segments from the reference TXT lines.
    # Lines 1-5 are Chinese, lines 6-9 mix both (line 6 switches mid-sentence).
    # We construct separate segments per line to keep language detection clean.
    segments = []
    with open(MIXED_REFERENCE_TXT) as f:
        lines = [line.strip() for line in f if line.strip()]

    seg_id = 1
    for line in lines:
        # Heuristic: if line contains a majority of CJK characters, tag as zh;
        # otherwise tag as en.
        cjk_count = sum(1 for c in line if "\u4e00" <= c <= "\u9fff")
        total_alpha = sum(1 for c in line if c.isalpha())
        lang = "zh" if total_alpha > 0 and cjk_count / total_alpha > 0.4 else "en"

        segments.append(
            {
                "segment_id": seg_id,
                "text": line,
                "language": lang,
                "is_draft": False,
                "stable_text": line,
            }
        )
        seg_id += 1

    return segments


@pytest.fixture
def replay_segments_mixed() -> list[dict]:
    """Fixture: mixed zh/en transcription segments."""
    return _load_mixed_segments()


# ---------------------------------------------------------------------------
# B1.3: Mixed language segment routing
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mixed_segments_route_to_correct_target(
    translation_service, replay_segments_mixed
) -> None:
    """B1.3: ZH segments produce EN translations; EN segments are skipped.

    Replays a mixed zh/en segment sequence and simulates the routing logic
    from websocket_audio.py:
    - ZH segment: translate to EN (effective_target = "en")
    - EN segment: effective_target = "en" == segment language → skip
    """
    _skip_if_no_llm()

    from livetranslate_common.models import TranslationRequest

    target_language = "en"

    zh_translations: list[str] = []
    en_skipped_count: int = 0

    for seg in replay_segments_mixed:
        text = seg["text"].strip()
        if not text:
            continue

        seg_lang = seg.get("language", "zh")

        if seg_lang == target_language:
            # Same language as target — skip (would not translate in live pipeline)
            en_skipped_count += 1
            continue

        if seg_lang == "zh":
            request = TranslationRequest(
                text=text,
                source_language="zh",
                target_language="en",
            )
            response = await translation_service.translate(request)
            zh_translations.append(response.translated_text)

    # Must have translated at least 2 ZH segments
    assert len(zh_translations) >= 2, (
        f"Expected at least 2 ZH→EN translations, got {len(zh_translations)}. "
        f"Skipped EN segments: {en_skipped_count}"
    )

    # All ZH translations must be non-empty with ASCII content (English output)
    for i, text in enumerate(zh_translations):
        assert len(text) > 0, f"ZH translation {i} is empty"
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        assert ascii_alpha > 3, (
            f"ZH translation {i} doesn't appear to be English output: {text[:80]}"
        )

    # Must have encountered at least 1 EN segment to skip
    assert en_skipped_count >= 1, (
        "Expected at least 1 EN segment in the mixed fixture "
        "(used to verify skip-same-language logic)"
    )


# ---------------------------------------------------------------------------
# B1.4: Context stays clean after language switch (DirectionalContextStore isolation)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_context_stays_clean_after_language_switch(translation_service) -> None:
    """B1.4: Translate 3 ZH→EN then 2 EN→ZH with NO clear_context() between them.

    Production (post-architect review) does NOT call clear_context() on direction
    flip. DirectionalContextStore provides per-direction isolation automatically.

    Asserts:
    - get_context("zh","en") has 3 entries (still there after direction switch)
    - get_context("en","zh") has 2 entries (accumulated in new direction)
    """
    _skip_if_no_llm()

    from livetranslate_common.models import TranslationRequest

    zh_texts = [
        "欢迎参加今天的会议",
        "我们第三季度的总收入达到了两千五百万美元",
        "亚太地区表现最好，增长了百分之二十二",
    ]

    en_texts = [
        "Translation is working with Ollama locally",
        "The quarterly results exceeded all expectations",
    ]

    # Phase 1: translate 3 ZH→EN segments, building zh->en context
    for text in zh_texts:
        request = TranslationRequest(
            text=text,
            source_language="zh",
            target_language="en",
        )
        await translation_service.translate(request)

    ctx_after_zh = translation_service.get_context("zh", "en")
    assert len(ctx_after_zh) == 3, (
        f"After 3 ZH translations, zh->en context should have 3 entries, got {len(ctx_after_zh)}"
    )

    # Phase 2: direction flip — NO clear_context() (production behavior)
    # Translate 2 EN→ZH segments in the new direction
    for text in en_texts:
        en_request = TranslationRequest(
            text=text,
            source_language="en",
            target_language="zh",
        )
        en_response = await translation_service.translate(en_request)
        assert len(en_response.translated_text) > 0, f"EN→ZH translation is empty for: {text}"

    # zh->en must STILL have 3 entries — DirectionalContextStore isolation
    ctx_zh_en = translation_service.get_context("zh", "en")
    assert len(ctx_zh_en) == 3, (
        f"zh->en context must retain 3 entries after direction switch (no clear), "
        f"got {len(ctx_zh_en)}"
    )

    # en->zh must have 2 entries — new direction accumulated independently
    ctx_en_zh = translation_service.get_context("en", "zh")
    assert len(ctx_en_zh) == 2, (
        f"en->zh context must have 2 entries, got {len(ctx_en_zh)}"
    )


# ---------------------------------------------------------------------------
# B1.5: EN→ZH translation produces Chinese output
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_en_zh_translation_produces_chinese(translation_service) -> None:
    """B1.5: Translate 2 English segments with target_language="zh".

    Asserts that output contains a meaningful number of CJK characters,
    proving the LLM is producing Chinese output (not passthrough or garbled).
    """
    _skip_if_no_llm()

    from livetranslate_common.models import TranslationRequest

    en_texts = [
        "Welcome to today's meeting, everyone",
        "Our third quarter revenue reached twenty-five million dollars",
    ]

    for text in en_texts:
        request = TranslationRequest(
            text=text,
            source_language="en",
            target_language="zh",
        )
        response = await translation_service.translate(request)
        translated = response.translated_text

        assert len(translated) > 0, f"EN→ZH translation is empty for: {text}"

        cjk_count = sum(1 for c in translated if "\u4e00" <= c <= "\u9fff")
        assert cjk_count > 3, (
            f"EN→ZH output must contain CJK characters (got {cjk_count}): {translated[:80]}"
        )
