"""Pure-function tests for Qwen3 / LLM response cleanup helpers.

These helpers are lifted out of `modules/orchestration-service/src/translation/llm_client.py`
and `routers/audio/websocket_audio.py:_strip_think_and_stream` so they're
reusable by the merged client + benchmark tool + any future caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from livetranslate_common.llm.qwen import (
    extract_from_reasoning,
    extract_translation_text,
    strip_think_blocks_streaming,
)


# ---------------------------------------------------------------------------
# extract_translation_text — synchronous post-hoc cleanup
# ---------------------------------------------------------------------------


class TestExtractTranslationText:
    def test_strips_think_single_line(self) -> None:
        assert extract_translation_text("<think>reasoning</think>Hello") == "Hello"

    def test_strips_multiline_think(self) -> None:
        raw = "<think>\nstep 1\nstep 2\n</think>\nHello world"
        assert extract_translation_text(raw) == "Hello world"

    def test_strips_unclosed_think(self) -> None:
        """max_tokens cutoff can leave the </think> off the end."""
        raw = "<think>this got cut off because"
        assert extract_translation_text(raw) == ""

    def test_strips_ascii_double_quotes(self) -> None:
        assert extract_translation_text('"Hello"') == "Hello"

    def test_strips_curly_double_quotes(self) -> None:
        assert extract_translation_text("“Hello”") == "Hello"

    def test_strips_single_quotes(self) -> None:
        assert extract_translation_text("'Hello'") == "Hello"

    def test_strips_curly_single_quotes(self) -> None:
        assert extract_translation_text("‘Hello’") == "Hello"

    def test_strips_english_prefix_label(self) -> None:
        assert extract_translation_text("Translation: Hello") == "Hello"
        assert extract_translation_text("Output: Hello") == "Hello"
        assert extract_translation_text("Result: Hello") == "Hello"

    def test_strips_chinese_prefix_label(self) -> None:
        assert extract_translation_text("翻译：你好世界") == "你好世界"
        assert extract_translation_text("译文: 你好世界") == "你好世界"

    def test_passthrough_when_no_prefix_or_quotes(self) -> None:
        assert extract_translation_text("Just text") == "Just text"

    def test_empty_input(self) -> None:
        assert extract_translation_text("") == ""

    def test_only_whitespace(self) -> None:
        assert extract_translation_text("   \n\t  ") == ""

    def test_strips_think_then_prefix_then_quotes(self) -> None:
        """All three cleanup layers applied in order."""
        raw = "<think>foo</think>Translation: \"Hello\""
        assert extract_translation_text(raw) == "Hello"


# ---------------------------------------------------------------------------
# strip_think_blocks_streaming — async iterator filter
# ---------------------------------------------------------------------------


async def _from_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


async def _collect(stream: AsyncIterator[str]) -> str:
    out: list[str] = []
    async for chunk in stream:
        out.append(chunk)
    return "".join(out)


class TestStripThinkBlocksStreaming:
    @pytest.mark.asyncio
    async def test_single_chunk_with_think(self) -> None:
        result = await _collect(
            strip_think_blocks_streaming(_from_chunks(["<think>secret</think>Hello"]))
        )
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_split_open_tag(self) -> None:
        """Open tag arrives split across two chunks."""
        result = await _collect(
            strip_think_blocks_streaming(_from_chunks(["<thi", "nk>secret</think>OK"]))
        )
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_split_close_tag(self) -> None:
        result = await _collect(
            strip_think_blocks_streaming(_from_chunks(["<think>secret</thi", "nk>OK"]))
        )
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_no_think_block(self) -> None:
        result = await _collect(
            strip_think_blocks_streaming(_from_chunks(["Hello ", "world"]))
        )
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_unclosed_think_drops_everything(self) -> None:
        """If we enter a <think> but never see </think>, the content is dropped."""
        result = await _collect(
            strip_think_blocks_streaming(_from_chunks(["<think>never closes"]))
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_stream(self) -> None:
        result = await _collect(strip_think_blocks_streaming(_from_chunks([])))
        assert result == ""


# ---------------------------------------------------------------------------
# extract_from_reasoning — Qwen3-on-Ollama "reasoning" field salvage
# ---------------------------------------------------------------------------


class TestExtractFromReasoning:
    def test_final_decision_label(self) -> None:
        reasoning = "let me think about this...\nFinal Decision: Hello world"
        assert extract_from_reasoning(reasoning) == "Hello world"

    def test_final_output_label(self) -> None:
        reasoning = "reasoning blah\n**Final Output Generation:** Hello there"
        assert extract_from_reasoning(reasoning) == "Hello there"

    def test_quoted_translation(self) -> None:
        reasoning = 'thinking. The most natural English: "Hello world."'
        assert extract_from_reasoning(reasoning) == "Hello world"

    def test_fallback_last_capital_starting_line(self) -> None:
        reasoning = "reasoning step\nanother thought\nHello world is the translation"
        # last line that starts with capital, has space, isn't meta
        assert "Hello" in extract_from_reasoning(reasoning)

    def test_empty_input_returns_empty(self) -> None:
        assert extract_from_reasoning("") == ""

    def test_only_meta_returns_empty(self) -> None:
        """If everything looks like reasoning meta-text, return empty."""
        assert extract_from_reasoning("let me think\ni need to consider\nthinking step") == ""
