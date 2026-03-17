"""B1.3 / B1.4: Mixed-language playback and context isolation tests.

B1.3: Replay mixed zh/en transcription segments and verify that ZH segments
      are routed to English translation while EN segments are skipped (same
      as the effective target language).

B1.4: After translating ZH segments, clear context and translate an EN
      segment — verify context is clean with only the EN entry.

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
# B1.4: Context stays clean after language switch
# ---------------------------------------------------------------------------

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_context_stays_clean_after_language_switch(translation_service) -> None:
    """B1.4: Translate 3 ZH texts, verify context=3, clear, translate 1 EN text,
    verify context=1 with correct content.

    Mirrors the direction-flip behavior in websocket_audio.py where clear_context()
    is called before the first segment of the new direction is translated.
    """
    _skip_if_no_llm()

    from livetranslate_common.models import TranslationRequest

    zh_texts = [
        "欢迎参加今天的会议",
        "我们第三季度的总收入达到了两千五百万美元",
        "亚太地区表现最好，增长了百分之二十二",
    ]

    # Phase 1: translate 3 ZH→EN segments, building context
    for text in zh_texts:
        request = TranslationRequest(
            text=text,
            source_language="zh",
            target_language="en",
        )
        await translation_service.translate(request)

    ctx_after_zh = translation_service.get_context("zh", "en")
    assert len(ctx_after_zh) == 3, (
        f"After 3 ZH translations, context should have 3 entries, got {len(ctx_after_zh)}"
    )

    # Phase 2: direction flip — clear context (as websocket_audio.py does)
    translation_service.clear_context()
    ctx_after_clear = translation_service.get_context("zh", "en")
    assert len(ctx_after_clear) == 0, (
        f"Context should be empty after clear, got {len(ctx_after_clear)}"
    )

    # Phase 3: translate 1 EN→ZH segment in the new direction
    en_text = "Translation is working with Ollama locally"
    en_request = TranslationRequest(
        text=en_text,
        source_language="en",
        target_language="zh",
    )
    en_response = await translation_service.translate(en_request)

    assert len(en_response.translated_text) > 0, "EN→ZH translation is empty"

    # Context must have exactly 1 entry — only the EN segment
    ctx_after_en = translation_service.get_context("en", "zh")
    assert len(ctx_after_en) == 1, (
        f"After 1 EN translation, context should have 1 entry, got {len(ctx_after_en)}"
    )

    # The single context entry must be the EN source, not any ZH source
    assert ctx_after_en[0].text == en_text, (
        f"Context entry should be the EN source text '{en_text}', "
        f"got '{ctx_after_en[0].text}'"
    )
    assert len(ctx_after_en[0].translation) > 0, "Context entry translation must be non-empty"
