"""Real-network behavioral E2E for FirefliesSource.

Exercises the *real* GraphQL endpoint and the *real* TranslationService against
a live Ollama/vLLM-MLX. Skipped when env keys/services unavailable so this
remains runnable on dev laptops that haven't set up the stack.

Run conditions:
- FIREFLIES_API_KEY env var present
- LIVEDEMO_FIREFLIES_TEST_MEETING_ID env var = a real transcript ID with ≥1 sentence
- LLM endpoint (LLM_BASE_URL, default http://localhost:8006/v1) reachable
"""
from __future__ import annotations

import os

import pytest

from livedemo.config import LiveDemoConfig
from livedemo.sources.fireflies import FirefliesSource


_REQUIRED_VARS = ["FIREFLIES_API_KEY", "LIVEDEMO_FIREFLIES_TEST_MEETING_ID"]


def _missing() -> str | None:
    for v in _REQUIRED_VARS:
        if not os.environ.get(v):
            return v
    return None


@pytest.mark.e2e
@pytest.mark.skipif(_missing() is not None, reason=f"E2E env not configured: missing {_missing()}")
@pytest.mark.asyncio
async def test_fireflies_e2e_real_transcript_yields_at_least_one_caption():
    cfg = LiveDemoConfig(
        meeting_url="https://meet.google.com/aaa-bbbb-ccc",
        source="fireflies",
        fireflies_meeting_id=os.environ["LIVEDEMO_FIREFLIES_TEST_MEETING_ID"],
        fireflies_replay_speed=0.0,  # instant replay
        target_language="en",
    )
    src = FirefliesSource.from_config(cfg)
    out = []
    async for evt in src.stream():
        out.append(evt)
        if len(out) >= 3:
            break  # one is enough — don't burn LLM tokens on a smoke
    assert len(out) >= 1
    first = out[0]
    assert first.text  # non-empty original
    assert first.translated_text  # non-empty translation
    assert first.target_lang == "en"
