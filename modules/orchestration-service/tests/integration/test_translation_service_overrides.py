"""TranslationService honors LLMParameterOverrides per call.

Real fake LLM server captures request bodies so we can verify that:
- Per-call overrides land in the outgoing request body
- The base connection is never mutated
- Two parallel translates with different overrides produce two distinct bodies
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make tests/ importable so `from integration.fakes import ...` works.
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from livetranslate_common.models import TranslationRequest  # noqa: E402
from livetranslate_common.models.llm import LLMParameterOverrides  # noqa: E402

from translation.config import TranslationConfig  # noqa: E402
from translation.service import TranslationService  # noqa: E402


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_two_parallel_translates_distinct_temperatures(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("OK")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        temperature=0.7,
    )
    svc = TranslationService(conn, TranslationConfig())
    try:
        await asyncio.gather(
            svc.translate(
                TranslationRequest(text="a", source_language="zh", target_language="en"),
                overrides=LLMParameterOverrides(temperature=0.2),
            ),
            svc.translate(
                TranslationRequest(text="b", source_language="zh", target_language="en"),
                overrides=LLMParameterOverrides(temperature=0.9),
            ),
        )
        temps = sorted(r["json"]["temperature"] for r in fake_openai_server.recorded_requests)
        assert temps == [0.2, 0.9]
    finally:
        await svc.close()


async def test_base_connection_immutable_after_overrides(
    fake_openai_server, llm_connection_factory
) -> None:
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
        temperature=0.7,
    )
    fake_openai_server.set_response_text("OK")
    svc = TranslationService(conn, TranslationConfig())
    try:
        await svc.translate(
            TranslationRequest(text="a", source_language="zh", target_language="en"),
            overrides=LLMParameterOverrides(temperature=0.1),
        )
        assert svc.base_connection.temperature == 0.7
    finally:
        await svc.close()


async def test_model_swap_via_overrides(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("OK")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="base-model",
    )
    svc = TranslationService(conn, TranslationConfig())
    try:
        resp = await svc.translate(
            TranslationRequest(text="x", source_language="zh", target_language="en"),
            overrides=LLMParameterOverrides(model="override-model"),
        )
        assert resp.model_used == "override-model"
        # base connection unchanged
        assert svc.base_connection.model == "base-model"
        # request body sent override-model
        body = fake_openai_server.recorded_requests[-1]["json"]
        assert body["model"] == "override-model"
    finally:
        await svc.close()


async def test_translate_draft_uses_draft_max_tokens(
    fake_openai_server, llm_connection_factory
) -> None:
    fake_openai_server.set_response_text("OK")
    conn = llm_connection_factory(
        engine="openai_compatible",
        base_url=fake_openai_server.base_url + "/v1",
        model="m",
    )
    behavioral = TranslationConfig()  # draft_max_tokens=256
    svc = TranslationService(conn, behavioral)
    try:
        await svc.translate_draft("hello", "zh", "en")
        body = fake_openai_server.recorded_requests[-1]["json"]
        assert body["max_tokens"] == behavioral.draft_max_tokens
    finally:
        await svc.close()
