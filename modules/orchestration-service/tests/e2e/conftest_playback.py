"""Fixtures for translation-only playback tests.

Loads transcription replay fixtures (pre-recorded segment sequences) and
instantiates real TranslationService + RollingContextWindow for testing
translation behavior without needing the transcription service or browser.

Requires an OpenAI-compatible LLM endpoint (vLLM-MLX, Ollama, etc.).
Gate: tests skip if LLM_BASE_URL env var is not set.

IMPORTANT: On Apple Silicon with split vLLM-MLX inference (just dev),
running these tests concurrently with the orchestration service can cause
Metal GPU contention crashes. Run standalone:
  just dev-llm  # start LLM only in one terminal
  just test-e2e-playback  # run tests in another
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio

# Add orchestration src to path
_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Fixture paths
TRANSCRIPTION_FIXTURES = (
    Path(__file__).parent.parent.parent.parent
    / "transcription-service"
    / "tests"
    / "fixtures"
    / "audio"
)


def _skip_if_no_llm():
    """Skip test if LLM_BASE_URL is not configured."""
    if not os.getenv("LLM_BASE_URL"):
        pytest.skip("LLM_BASE_URL not set — skip translation playback tests")


@pytest.fixture
def replay_segments_zh():
    """Load Chinese transcription replay fixture (v2 with draft/final fields)."""
    fixture_path = TRANSCRIPTION_FIXTURES / "meeting_zh_transcription_replay_v2.json"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    with open(fixture_path) as f:
        data = json.load(f)
    return data["segments"]


@pytest_asyncio.fixture
async def translation_service():
    """Create a real TranslationService connected to the LLM endpoint."""
    _skip_if_no_llm()

    from translation.config import TranslationConfig
    from translation.service import TranslationService

    config = TranslationConfig()  # reads LLM_* env vars
    svc = TranslationService(config)
    yield svc
    try:
        await svc.close()
    except RuntimeError:
        pass  # "Event loop is closed" — safe to ignore during teardown


@pytest_asyncio.fixture
async def context_window():
    """Create a fresh RollingContextWindow."""
    from translation.context import RollingContextWindow

    return RollingContextWindow(max_entries=5, max_tokens=500)
