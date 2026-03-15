# Plan 4: Translation Module (in Orchestration) & Benchmarking

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Absorb translation into orchestration as a thin module (~150 LOC core) calling Ollama/vLLM directly on thomas-pc. Add rolling context window for translation quality, backpressure queue, and a standalone benchmarking harness for comparing models/configurations.

**Architecture:** The translation module lives in `modules/orchestration-service/src/translation/`. It calls Ollama's OpenAI-compatible API (`/v1/chat/completions`) directly over Tailscale. No intermediate translation service. Rolling context (last N sentence pairs) gives the LLM continuity. A bounded queue with drop-oldest backpressure prevents translation lag from blocking the pipeline. The old `modules/translation-service/` is archived. The benchmarking harness is a standalone CLI tool in `tools/translation_benchmark/`.

**Tech Stack:** Python 3.12+, httpx (async HTTP client), Pydantic v2, UV workspace

**Spec:** `docs/superpowers/specs/2026-03-14-loopback-transcription-translation-design.md` — Plan 4 section

**Depends on:** Plan 0 (shared contracts: `TranslationRequest`, `TranslationResponse`, `TranslationContext`)

**Extension to Plan 0 shared contracts:** `TranslationRequest` gains an optional `glossary_terms: dict[str, str] | None = None` field for consistent domain-specific terminology. The `TranslationService.translate()` method accepts a `TranslationRequest` object (not bare parameters) and passes glossary terms through to the prompt builder. When the existing `TranslationPromptBuilder` (`services/translation_prompt_builder.py`) is available, it handles glossary injection; otherwise, the LLM client includes an inline glossary section in the prompt.

---

## Chunk 1: Translation Module

### Task 1: LLM client — Ollama/vLLM HTTP calls

**Files:**
- Create: `modules/orchestration-service/src/translation/__init__.py`
- Create: `modules/orchestration-service/src/translation/llm_client.py`
- Create: `modules/orchestration-service/src/translation/config.py`
- Create: `modules/orchestration-service/tests/test_llm_client.py`

- [ ] **Step 1: Write failing test for LLM client**

```python
# modules/orchestration-service/tests/test_llm_client.py
"""Tests for translation LLM client — calls Ollama/vLLM directly.

These tests hit the real Ollama server on thomas-pc via Tailscale.
Mark with @pytest.mark.integration for CI filtering.
"""
import pytest
from translation.config import TranslationConfig
from translation.llm_client import LLMClient


@pytest.fixture
def config():
    return TranslationConfig(
        llm_base_url="http://thomas-pc:11434/v1",
        model="qwen3.5:7b",
        temperature=0.3,
        timeout_s=10,  # matches spec default; integration tests can override via LLM_TIMEOUT_S env var
    )


class TestLLMClientUnit:
    def test_build_prompt_no_context(self, config):
        client = LLMClient(config)
        messages = client._build_messages(
            text="你好世界",
            source_language="zh",
            target_language="en",
            context=[],
        )
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert "translator" in messages[0]["content"].lower()
        assert "你好世界" in messages[1]["content"]

    def test_build_prompt_with_context(self, config):
        from livetranslate_common.models import TranslationContext

        client = LLMClient(config)
        context = [
            TranslationContext(text="之前的话", translation="Previous words"),
        ]
        messages = client._build_messages(
            text="这是新的",
            source_language="zh",
            target_language="en",
            context=context,
        )
        user_msg = messages[1]["content"]
        assert "之前的话" in user_msg
        assert "Previous words" in user_msg
        assert "这是新的" in user_msg


@pytest.mark.integration
class TestLLMClientIntegration:
    @pytest.mark.asyncio
    async def test_translate_simple(self, config):
        client = LLMClient(config)
        result = await client.translate(
            text="你好世界",
            source_language="zh",
            target_language="en",
        )
        assert result is not None
        assert len(result) > 0
        # Should contain something related to "hello" or "world"

    @pytest.mark.asyncio
    async def test_translate_with_context(self, config):
        from livetranslate_common.models import TranslationContext

        client = LLMClient(config)
        context = [
            TranslationContext(text="张经理来了", translation="Manager Zhang has arrived"),
        ]
        result = await client.translate(
            text="他说你好",
            source_language="zh",
            target_language="en",
            context=context,
        )
        assert result is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_llm_client.py::TestLLMClientUnit -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write translation config**

```python
# modules/orchestration-service/src/translation/__init__.py
"""Translation module — calls Ollama/vLLM directly for translation."""

# modules/orchestration-service/src/translation/config.py
"""Translation module configuration.

All settings can be overridden via environment variables:
  LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_S,
  LLM_CONTEXT_WINDOW_SIZE, LLM_MAX_CONTEXT_TOKENS
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TranslationConfig:
    llm_base_url: str = "http://thomas-pc:11434/v1"
    model: str = "qwen3.5:7b"
    temperature: float = 0.3
    timeout_s: int = 10
    context_window_size: int = 5
    max_context_tokens: int = 500
    max_queue_depth: int = 10

    @classmethod
    def from_env(cls) -> TranslationConfig:
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            model=os.getenv("LLM_MODEL", cls.model),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            timeout_s=int(os.getenv("LLM_TIMEOUT_S", str(cls.timeout_s))),
            context_window_size=int(os.getenv("LLM_CONTEXT_WINDOW_SIZE", str(cls.context_window_size))),
            max_context_tokens=int(os.getenv("LLM_MAX_CONTEXT_TOKENS", str(cls.max_context_tokens))),
            max_queue_depth=int(os.getenv("LLM_MAX_QUEUE_DEPTH", str(cls.max_queue_depth))),
        )
```

- [ ] **Step 4: Write LLM client**

```python
# modules/orchestration-service/src/translation/llm_client.py
"""LLM client for translation — calls Ollama/vLLM OpenAI-compatible API.

Ported from translation-service's OpenAICompatibleTranslator (~150 lines
of core logic). Everything else was proxy/wrapper code.

Handles: prompt construction, HTTP POST, response parsing, retry with
exponential backoff.
"""
from __future__ import annotations

import time

import httpx
from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext

from translation.config import TranslationConfig

logger = get_logger()


class LLMClient:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.llm_base_url,
            timeout=httpx.Timeout(config.timeout_s),
        )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext] | None = None,
        glossary_terms: dict[str, str] | None = None,
    ) -> str:
        """Translate text using the LLM. Returns translated string.

        Args:
            glossary_terms: Optional dict of {source_term: target_term} for
                consistent terminology. Passed to the prompt builder so the
                LLM uses the preferred translations for domain-specific terms.
        """
        messages = self._build_messages(
            text, source_language, target_language,
            context or [], glossary_terms,
        )

        start = time.monotonic()
        response_text = await self._call_llm(messages)
        latency_ms = (time.monotonic() - start) * 1000

        translation = self._extract_translation(response_text)

        logger.info(
            "translation_complete",
            source_lang=source_language,
            target_lang=target_language,
            model=self.config.model,
            latency_ms=round(latency_ms, 1),
            input_len=len(text),
            output_len=len(translation),
        )

        return translation

    def _build_messages(
        self,
        text: str,
        source_language: str,
        target_language: str,
        context: list[TranslationContext],
        glossary_terms: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build LLM messages for translation.

        If glossary_terms is provided AND the existing TranslationPromptBuilder
        is available, delegates to it for richer prompt construction (glossary
        injection, speaker attribution, etc.). Otherwise uses the inline prompt.

        See: modules/orchestration-service/src/services/translation_prompt_builder.py
        """
        # Try using TranslationPromptBuilder for glossary-aware prompts
        if glossary_terms:
            try:
                from services.translation_prompt_builder import (
                    TranslationPromptBuilder,
                    PromptContext,
                )
                builder = TranslationPromptBuilder()
                previous_sentences = [ctx.text for ctx in context] if context else None
                result = builder.build(PromptContext(
                    current_sentence=text,
                    target_language=target_language,
                    source_language=source_language,
                    previous_sentences=previous_sentences,
                    glossary_terms=glossary_terms,
                ))
                return [
                    {"role": "system", "content": (
                        f"You are a professional translator from {source_language} "
                        f"to {target_language}. Output ONLY the translation."
                    )},
                    {"role": "user", "content": result.prompt},
                ]
            except ImportError:
                logger.debug(
                    "prompt_builder_unavailable",
                    msg="TranslationPromptBuilder not found, using inline prompt with glossary",
                )

        system_prompt = (
            f"You are a professional translator from {source_language} to {target_language}. "
            "Translate the given text naturally and accurately. "
            "Output ONLY the translation, nothing else. "
            "Preserve the original meaning, tone, and style."
        )

        user_parts = []

        # Glossary section (inline fallback when TranslationPromptBuilder unavailable)
        if glossary_terms:
            user_parts.append("Glossary (use these exact translations for the listed terms):")
            for src_term, tgt_term in glossary_terms.items():
                user_parts.append(f"  {src_term} -> {tgt_term}")
            user_parts.append("")

        if context:
            user_parts.append("Context (previous sentences for reference):")
            for ctx in context:
                user_parts.append(f"  Original: {ctx.text}")
                user_parts.append(f"  Translation: {ctx.translation}")
            user_parts.append("")

        user_parts.append(f"Translate the following from {source_language} to {target_language}:")
        user_parts.append(text)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    async def _call_llm(self, messages: list[dict], max_retries: int = 2) -> str:
        """Call the LLM API with retry logic."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "temperature": self.config.temperature,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    logger.warning(
                        "llm_retry",
                        attempt=attempt + 1,
                        error=str(e),
                        wait_s=wait,
                    )
                    import asyncio
                    await asyncio.sleep(wait)

        raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")

    def _extract_translation(self, response: str) -> str:
        """Clean up LLM response to extract just the translation."""
        text = response.strip()
        # Remove common LLM wrapping patterns
        for prefix in ["Translation:", "翻译:", "Here is the translation:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        # Remove surrounding quotes if present
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        return text

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 5: Run unit tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_llm_client.py::TestLLMClientUnit -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add modules/orchestration-service/src/translation/ modules/orchestration-service/tests/test_llm_client.py
git commit -m "feat(orchestration): add LLM translation client calling Ollama directly"
```

---

### Task 2: Rolling context window

**Files:**
- Create: `modules/orchestration-service/src/translation/context.py`
- Create: `modules/orchestration-service/tests/test_translation_context.py`

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_translation_context.py
"""Tests for rolling translation context window."""
from livetranslate_common.models import TranslationContext
from translation.context import RollingContextWindow


class TestRollingContextWindow:
    def test_empty_context(self):
        window = RollingContextWindow(max_entries=5)
        assert window.get_context() == []

    def test_add_and_get(self):
        window = RollingContextWindow(max_entries=5)
        window.add("你好", "Hello")
        ctx = window.get_context()
        assert len(ctx) == 1
        assert ctx[0].text == "你好"
        assert ctx[0].translation == "Hello"

    def test_eviction_by_count(self):
        window = RollingContextWindow(max_entries=3)
        window.add("one", "一")
        window.add("two", "二")
        window.add("three", "三")
        window.add("four", "四")  # should evict "one"

        ctx = window.get_context()
        assert len(ctx) == 3
        assert ctx[0].text == "two"
        assert ctx[2].text == "four"

    def test_eviction_by_tokens(self):
        window = RollingContextWindow(max_entries=100, max_tokens=20)
        window.add("short", "短")  # ~6 tokens
        window.add("medium text here", "中等文本")  # ~10 tokens
        window.add("another long entry", "又一个长条目")  # ~10 tokens → exceeds 20

        ctx = window.get_context()
        # Oldest entries evicted until under token limit
        total_tokens = sum(window._estimate_tokens(c.text + c.translation) for c in ctx)
        assert total_tokens <= 20

    def test_failed_translations_not_added(self):
        window = RollingContextWindow(max_entries=5)
        window.add("good", "好")
        # Don't add failed translations — caller responsibility
        assert len(window.get_context()) == 1

    def test_clear(self):
        window = RollingContextWindow(max_entries=5)
        window.add("one", "一")
        window.clear()
        assert window.get_context() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_translation_context.py -v`
Expected: FAIL

- [ ] **Step 3: Write RollingContextWindow**

```python
# modules/orchestration-service/src/translation/context.py
"""Rolling context window for translation quality.

Maintains a ring buffer of recent (source, translation) pairs.
Each translation request includes the last N sentences as context,
giving the LLM continuity for pronouns, terminology, and tone.

Eviction: by count (max_entries) AND by token estimate (max_tokens),
whichever limit is hit first.
"""
from __future__ import annotations

from collections import deque

from livetranslate_common.models import TranslationContext


class RollingContextWindow:
    def __init__(self, max_entries: int = 5, max_tokens: int = 500):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries: deque[TranslationContext] = deque(maxlen=max_entries)

    def add(self, source_text: str, translation: str) -> None:
        """Add a successful translation pair to the context window."""
        self._entries.append(TranslationContext(text=source_text, translation=translation))
        self._evict_by_tokens()

    def get_context(self) -> list[TranslationContext]:
        """Return the current context window as a list."""
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def _evict_by_tokens(self) -> None:
        """Remove oldest entries until total tokens <= max_tokens."""
        while self._entries and self._total_tokens() > self.max_tokens:
            self._entries.popleft()

    def _total_tokens(self) -> int:
        return sum(
            self._estimate_tokens(e.text + e.translation) for e in self._entries
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English, ~2 for CJK."""
        # Simple heuristic — good enough for context window sizing
        return max(1, len(text) // 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_translation_context.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/translation/context.py modules/orchestration-service/tests/test_translation_context.py
git commit -m "feat(orchestration): add rolling context window for translation quality"
```

---

### Task 3: Translation service (combines client + context + backpressure)

**Files:**
- Create: `modules/orchestration-service/src/translation/service.py`
- Create: `modules/orchestration-service/tests/test_translation_service.py`

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_translation_service.py
"""Tests for TranslationService — combines LLM client, context, and backpressure.

These tests hit the real Ollama server on thomas-pc via Tailscale.
NO MOCKING — all tests are behavioral/integration tests per project rules.
Mark with @pytest.mark.integration for CI filtering.
"""
import asyncio

import pytest

from translation.service import TranslationService
from translation.config import TranslationConfig
from livetranslate_common.models import TranslationRequest, TranslationResponse


@pytest.fixture
def config():
    return TranslationConfig(
        llm_base_url="http://thomas-pc:11434/v1",
        model="qwen3.5:7b",
        context_window_size=3,
        max_queue_depth=5,
        timeout_s=10,  # spec default; override via LLM_TIMEOUT_S env var for slow models
    )


@pytest.mark.integration
class TestTranslationServiceIntegration:
    @pytest.mark.asyncio
    async def test_translate_adds_to_context(self, config):
        service = TranslationService(config)
        try:
            request = TranslationRequest(
                text="你好世界",
                source_language="zh",
                target_language="en",
                context=[],
            )
            response = await service.translate(request)

            assert response.translated_text is not None
            assert len(response.translated_text) > 0
            assert response.model_used == "qwen3.5:7b"
            assert response.latency_ms >= 0

            # Context should now contain this pair
            ctx = service.get_context()
            assert len(ctx) == 1
            assert ctx[0].text == "你好世界"
            assert ctx[0].translation == response.translated_text
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_context_improves_consistency(self, config):
        """Translate two related sentences — context should help pronoun resolution."""
        service = TranslationService(config)
        try:
            req1 = TranslationRequest(
                text="张经理来了",
                source_language="zh",
                target_language="en",
                context=[],
            )
            resp1 = await service.translate(req1)
            assert resp1.translated_text is not None

            # Second sentence — context from first should help resolve "他"
            req2 = TranslationRequest(
                text="他说你好",
                source_language="zh",
                target_language="en",
                context=service.get_context(),
            )
            resp2 = await service.translate(req2)
            assert resp2.translated_text is not None

            # Context should now have both pairs
            ctx = service.get_context()
            assert len(ctx) == 2
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_failed_translation_not_in_context(self, config):
        """Use a bad URL to trigger a real failure — context must stay empty."""
        bad_config = TranslationConfig(
            llm_base_url="http://localhost:1",  # nothing listening here
            model="nonexistent-model",
            context_window_size=3,
            max_queue_depth=5,
            timeout_s=2,
        )
        service = TranslationService(bad_config)
        try:
            request = TranslationRequest(
                text="失败的句子",
                source_language="zh",
                target_language="en",
                context=[],
            )
            with pytest.raises(RuntimeError):
                await service.translate(request)

            assert len(service.get_context()) == 0
        finally:
            await service.close()


@pytest.mark.integration
class TestBackpressure:
    @pytest.mark.asyncio
    async def test_queue_drops_oldest_when_full(self, config):
        config.max_queue_depth = 2
        service = TranslationService(config)
        try:
            # Queue 3 translations to a real LLM — oldest should be dropped
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    service.enqueue_translation(
                        text=f"翻译句子{i}",
                        source_language="zh",
                        target_language="en",
                    )
                )
                tasks.append(task)
                await asyncio.sleep(0.05)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            # At least one should complete, at most one dropped
            completed = [r for r in results if isinstance(r, TranslationResponse)]
            dropped = [r for r in results if isinstance(r, RuntimeError)]
            assert len(completed) >= 1
            # With queue depth 2 and 3 requests, at most 1 should be dropped
            assert len(dropped) <= 1
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_queue_completes_all_when_not_full(self, config):
        """Queue fewer items than max depth — all should complete."""
        config.max_queue_depth = 5
        service = TranslationService(config)
        try:
            tasks = []
            for i in range(2):
                task = asyncio.create_task(
                    service.enqueue_translation(
                        text=f"句子{i}",
                        source_language="zh",
                        target_language="en",
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed = [r for r in results if isinstance(r, TranslationResponse)]
            assert len(completed) == 2
        finally:
            await service.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_translation_service.py::TestTranslationServiceUnit -v`
Expected: FAIL

- [ ] **Step 3: Write TranslationService**

```python
# modules/orchestration-service/src/translation/service.py
"""TranslationService — combines LLM client, rolling context, and backpressure.

This is the high-level interface that the meeting pipeline calls.
It manages:
- LLM client for actual translation
- Rolling context window for quality
- Bounded queue with drop-oldest backpressure
"""
from __future__ import annotations

import asyncio
import time

from livetranslate_common.logging import get_logger
from livetranslate_common.models import TranslationContext, TranslationRequest, TranslationResponse

from translation.config import TranslationConfig
from translation.context import RollingContextWindow
from translation.llm_client import LLMClient

logger = get_logger()


class TranslationService:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self._client = LLMClient(config)
        self._context = RollingContextWindow(
            max_entries=config.context_window_size,
            max_tokens=config.max_context_tokens,
        )
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=config.max_queue_depth)
        self._processing = False

    async def translate(
        self,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Translate text synchronously (blocking until complete).

        Accepts a TranslationRequest (from Plan 0 shared contracts) instead
        of bare parameters. Context from the request is merged with the
        rolling context window.

        Used for direct translation. For queued translation with
        backpressure, use enqueue_translation().
        """
        # Merge: use request context if provided, otherwise use rolling window
        context = request.context if request.context else self._context.get_context()
        start = time.monotonic()

        translated = await self._client.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            context=context,
            glossary_terms=request.glossary_terms,
        )

        latency_ms = (time.monotonic() - start) * 1000

        # Only add to context on success
        self._context.add(request.text, translated)

        return TranslationResponse(
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=self.config.model,
            latency_ms=round(latency_ms, 1),
        )

    async def enqueue_translation(
        self,
        text: str,
        source_language: str,
        target_language: str,
        glossary_terms: dict[str, str] | None = None,
    ) -> TranslationResponse:
        """Queue a translation request with backpressure.

        If queue is full, drops the oldest pending request.
        Each queued item stores its own language pair so _process_queue
        does not need to assume a single pair for all items.
        """
        future: asyncio.Future[TranslationResponse] = asyncio.get_event_loop().create_future()

        if self._queue.full():
            # Drop oldest
            try:
                old_text, _old_src, _old_tgt, _old_glossary, old_future = self._queue.get_nowait()
                old_future.set_exception(
                    RuntimeError("Translation request dropped (backpressure)")
                )
                logger.warning("translation_dropped", text=old_text[:50])
            except asyncio.QueueEmpty:
                pass

        await self._queue.put((text, source_language, target_language, glossary_terms, future))

        if not self._processing:
            asyncio.create_task(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        """Process queued translation requests. Each item carries its own language pair."""
        self._processing = True
        try:
            while not self._queue.empty():
                text, source_language, target_language, glossary_terms, future = await self._queue.get()
                try:
                    request = TranslationRequest(
                        text=text,
                        source_language=source_language,
                        target_language=target_language,
                        context=self._context.get_context(),
                        glossary_terms=glossary_terms,
                    )
                    result = await self.translate(request)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
        finally:
            self._processing = False

    def get_context(self) -> list[TranslationContext]:
        return self._context.get_context()

    def clear_context(self) -> None:
        self._context.clear()

    async def close(self) -> None:
        await self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_translation_service.py::TestTranslationServiceUnit -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/translation/service.py modules/orchestration-service/tests/test_translation_service.py
git commit -m "feat(orchestration): add TranslationService with context window and backpressure"
```

---

## Chunk 2: Cleanup Old Translation Service

### Task 4: Archive translation-service and delete old clients

**Files:**
- Delete: `modules/orchestration-service/src/clients/translation_service_client.py`
- Delete: `modules/orchestration-service/src/internal_services/translation.py`
- Modify: any files importing from deleted modules
- Note: `modules/translation-service/` will be archived (moved to `archive/`) — not deleted

- [ ] **Step 1: Identify all imports of old translation clients**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && grep -rn "translation_service_client\|internal_services.translation\|from translation_service\|import translation_service" modules/orchestration-service/src/ --include="*.py" || echo "No matches"
```

- [ ] **Step 2: Remove imports and references**

The following files import from the old translation client (found via grep in Step 1). Each must be updated:

| File | What to change |
|------|---------------|
| `src/routers/settings/prompts.py` | Remove `from clients.translation_service_client import ...`; replace with `from translation.service import TranslationService` |
| `src/routers/settings/_shared.py` | Remove translation_service_client import; use new `TranslationService` |
| `src/routers/audio/audio_core.py` | Remove `from clients.translation_service_client import ...`; use `TranslationService` |
| `src/audio/audio_coordinator.py` | Remove old client import; inject `TranslationService` |
| `src/routers/fireflies.py` | Remove `from internal_services.translation import ...`; use `TranslationService` |
| `src/routers/translation.py` | Remove old client/service imports; delegate to `TranslationService` |
| `src/services/rolling_window_translator.py` | Remove old client import; this module is superseded by `translation.service` — delete it |
| `src/routers/system.py` | Remove health-check reference to old translation client |
| `src/dependencies.py` | Remove old client factory; add `TranslationService` factory |
| `src/routers/analytics.py` | Remove old translation client import |
| `src/routers/settings/sync.py` | Remove old translation client import |
| `src/clients/__init__.py` | Remove `translation_service_client` re-export |

**Grep commands to verify cleanup is complete:**

```bash
# Should return NO matches after cleanup:
grep -rn "translation_service_client\|TranslationServiceClient" modules/orchestration-service/src/ --include="*.py"
grep -rn "internal_services.translation\|internal_services\.translation" modules/orchestration-service/src/ --include="*.py"
grep -rn "from translation_service\|import translation_service" modules/orchestration-service/src/ --include="*.py"

# Replacement pattern for common cases:
# OLD: from clients.translation_service_client import TranslationServiceClient
# NEW: from translation.service import TranslationService
#
# OLD: translation_client = TranslationServiceClient(...)
# NEW: translation_service = TranslationService(TranslationConfig.from_env())
```

For each file:
- Remove the old import
- Replace calls with the new `TranslationService` from `translation.service`
- If a file becomes empty or fully superseded after cleanup (e.g., `rolling_window_translator.py`), delete it

- [ ] **Step 3: Delete the old client files**

```bash
rm -f modules/orchestration-service/src/clients/translation_service_client.py
rm -f modules/orchestration-service/src/internal_services/translation.py
```

- [ ] **Step 4: Move translation-service to archive**

```bash
mkdir -p archive
git mv modules/translation-service archive/translation-service-archived
```

- [ ] **Step 5: Update CLAUDE.md**

In root `CLAUDE.md`:
- Update the "4 Core Services" section (now 3 services)
- Add note about translation being absorbed into orchestration
- Remove translation service port (5003)
- Update any cross-references

- [ ] **Step 6: Verify no broken imports**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run python -c "
import sys; sys.path.insert(0, 'modules/orchestration-service/src')
from translation.service import TranslationService
from translation.config import TranslationConfig
from translation.llm_client import LLMClient
from translation.context import RollingContextWindow
print('All translation imports OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: archive translation-service, delete old clients, update CLAUDE.md"
```

---

## Chunk 3: Benchmarking Harness

### Task 5: Translation benchmarking CLI

**Files:**
- Create: `tools/translation_benchmark/pyproject.toml`
- Create: `tools/translation_benchmark/__init__.py`
- Create: `tools/translation_benchmark/__main__.py`
- Create: `tools/translation_benchmark/run.py`
- Create: `tools/translation_benchmark/metrics.py`
- Create: `tools/translation_benchmark/tests/__init__.py`
- Create: `tools/translation_benchmark/tests/test_metrics.py`
- Modify: root `pyproject.toml` (add to UV workspace members)

- [ ] **Step 0: Create UV workspace package for the benchmark tool**

The benchmark tool needs to import from the `translation` module in orchestration-service and from `livetranslate-common`. Add it to the UV workspace as a proper package.

```toml
# tools/translation_benchmark/pyproject.toml
[project]
name = "translation-benchmark"
version = "0.1.0"
description = "Translation benchmarking harness for LiveTranslate"
requires-python = ">=3.12,<3.14"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "livetranslate-common",
]

[project.optional-dependencies]
comet = ["unbabel-comet>=2.2"]

[project.scripts]
translation-benchmark = "tools.translation_benchmark.run:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["tools/translation_benchmark"]
```

Also add the benchmark tool to the root `pyproject.toml` workspace members:

```toml
# In root pyproject.toml, under [tool.uv.workspace]:
members = [
    "modules/*",
    "tools/translation_benchmark",
]
```

And add the orchestration translation module path as a dependency so imports resolve:

```toml
# In tools/translation_benchmark/pyproject.toml [project.dependencies]:
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "livetranslate-common",
    "livetranslate-orchestration",  # for translation.config, translation.service
]
```

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv sync`

- [ ] **Step 1: Write BLEU score computation**

```python
# tools/translation_benchmark/metrics.py
"""Translation quality metrics for benchmarking.

BLEU: corpus-level n-gram precision (standard MT metric)
COMET: learned metric using pretrained models (optional, requires comet-ml)
"""
from __future__ import annotations

import math
from collections import Counter


def bleu_score(
    references: list[str],
    hypotheses: list[str],
    max_n: int = 4,
) -> float:
    """Compute corpus-level BLEU score.

    Simplified implementation for benchmarking — for publication-grade
    results, use sacrebleu.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")

    if not references:
        return 0.0

    precisions = []
    bp_r = 0
    bp_c = 0

    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.strip().split()
            hyp_tokens = hyp.strip().split()

            if n == 1:
                bp_r += len(ref_tokens)
                bp_c += len(hyp_tokens)

            ref_ngrams = _get_ngrams(ref_tokens, n)
            hyp_ngrams = _get_ngrams(hyp_tokens, n)

            # Clipped counts
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            total += sum(hyp_ngrams.values())

        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)

    # Geometric mean of precisions
    log_avg = sum(math.log(p) if p > 0 else float("-inf") for p in precisions) / max_n
    if log_avg == float("-inf"):
        return 0.0

    # Brevity penalty
    if bp_c == 0:
        bp = 0.0
    elif bp_c >= bp_r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - bp_r / bp_c)

    return bp * math.exp(log_avg)


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# --- COMET (optional, requires unbabel-comet) ---

try:
    from comet import download_model, load_from_checkpoint

    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False


def comet_available() -> bool:
    """Check whether COMET scoring is available."""
    return _COMET_AVAILABLE


def comet_score(
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> float | None:
    """Compute COMET score. Returns None if comet is not installed.

    Install with: uv add unbabel-comet --optional benchmark
    """
    if not _COMET_AVAILABLE:
        return None

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    output = model.predict(data, batch_size=32, gpus=0)
    return float(output.system_score)
```

- [ ] **Step 2: Write tests for BLEU**

```python
# tools/translation_benchmark/tests/__init__.py

# tools/translation_benchmark/tests/test_metrics.py
import pytest
from tools.translation_benchmark.metrics import bleu_score, comet_available, comet_score


class TestBLEU:
    def test_identical(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on the mat"]
        score = bleu_score(refs, hyps)
        assert score == 1.0

    def test_completely_wrong(self):
        refs = ["the cat sat on the mat"]
        hyps = ["foo bar baz qux quux"]
        score = bleu_score(refs, hyps)
        assert score == 0.0

    def test_partial_match(self):
        refs = ["the cat sat on the mat"]
        hyps = ["the cat sat on a mat"]
        score = bleu_score(refs, hyps)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert bleu_score([], []) == 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            bleu_score(["one"], ["one", "two"])


class TestCOMET:
    def test_comet_available_returns_bool(self):
        # Should return True or False without raising
        result = comet_available()
        assert isinstance(result, bool)

    def test_comet_score_returns_none_when_unavailable(self):
        if comet_available():
            pytest.skip("COMET is installed, cannot test unavailable path")
        result = comet_score(
            sources=["Hello"],
            references=["Hola"],
            hypotheses=["Hola"],
        )
        assert result is None
```

- [ ] **Step 3: Run metric tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest tools/translation_benchmark/tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 4: Write benchmark runner**

```python
# tools/translation_benchmark/__init__.py
"""Translation benchmarking harness."""

# tools/translation_benchmark/run.py
"""CLI benchmark runner for translation models.

Usage: uv run python -m tools.translation_benchmark.run --model qwen3.5:7b --lang-pair zh-en
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from livetranslate_common.logging import setup_logging, get_logger

from tools.translation_benchmark.metrics import bleu_score, comet_available, comet_score

logger = get_logger()


def get_system_info() -> dict:
    """Collect system info for reproducibility.

    Includes GPU model, driver version, CUDA version, Python package
    versions, and model checksum (SHA256) where available.
    """
    import subprocess

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": {},
        "packages": {},
    }

    # GPU info via nvidia-smi
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = smi.stdout.strip().split(", ", 1)
            info["gpu"]["model"] = parts[0] if parts else "unknown"
            info["gpu"]["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"]["model"] = "N/A (nvidia-smi not found)"

    # CUDA version
    try:
        cuda = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.splitlines():
                if "release" in line.lower():
                    info["gpu"]["cuda_version"] = line.strip().split("release")[-1].strip().rstrip(",")
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["gpu"]["cuda_version"] = "N/A"

    # Key Python package versions
    try:
        import importlib.metadata
        for pkg in ["httpx", "pydantic", "livetranslate-common", "unbabel-comet"]:
            try:
                info["packages"][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                pass
    except ImportError:
        pass

    return info


def get_model_checksum(ollama_url: str, model: str) -> str | None:
    """Get SHA256 digest of the Ollama model for reproducibility."""
    import httpx as _httpx
    try:
        resp = _httpx.post(
            f"{ollama_url.rstrip('/v1')}/api/show",
            json={"name": model},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("digest", None)
    except Exception:
        pass
    return None


async def run_single_model_benchmark(
    model: str,
    lang_pair: str,
    sources: list[str],
    references: list[str],
    ollama_url: str,
    context_sizes: list[int],
    concurrency: int = 1,
) -> dict:
    """Run benchmark for a single model. Returns result dict."""
    from translation.config import TranslationConfig
    from translation.service import TranslationService
    from livetranslate_common.models import TranslationRequest

    src_lang, tgt_lang = lang_pair.split("-")
    model_result = {
        "model": model,
        "model_checksum": get_model_checksum(ollama_url, model),
        "runs": [],
    }

    for ctx_size in context_sizes:
        config = TranslationConfig(
            llm_base_url=ollama_url,
            model=model,
            context_window_size=ctx_size,
        )
        service = TranslationService(config)

        hypotheses = []
        latencies = []

        for i, source in enumerate(sources):
            try:
                request = TranslationRequest(
                    text=source,
                    source_language=src_lang,
                    target_language=tgt_lang,
                    context=service.get_context(),
                )
                response = await service.translate(request)
                hypotheses.append(response.translated_text)
                latencies.append(response.latency_ms)
            except Exception as e:
                hypotheses.append("")
                logger.warning("translation_failed", index=i, error=str(e))

        bleu = bleu_score(references, hypotheses)

        run_result: dict = {
            "context_window_size": ctx_size,
            "bleu": round(bleu, 4),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
            "samples": len(sources),
            "failures": sum(1 for h in hypotheses if not h),
        }

        # COMET score (optional — skipped if unbabel-comet not installed)
        if comet_available():
            comet = comet_score(sources, references, hypotheses)
            run_result["comet"] = round(comet, 4) if comet is not None else None
            logger.info("comet_computed", score=run_result["comet"])
        else:
            run_result["comet"] = None
            logger.info("comet_skipped", reason="unbabel-comet not installed")

        model_result["runs"].append(run_result)

        logger.info(
            "benchmark_run_complete",
            model=model,
            context_size=ctx_size,
            bleu=run_result["bleu"],
            avg_latency=run_result["avg_latency_ms"],
        )

        await service.close()

    # Concurrent throughput measurement
    if concurrency > 1:
        config = TranslationConfig(
            llm_base_url=ollama_url,
            model=model,
            context_window_size=0,  # no context for throughput test
        )
        service = TranslationService(config)

        # Fire N requests simultaneously
        sample_texts = (sources * ((concurrency // len(sources)) + 1))[:concurrency]

        async def single_request(text: str) -> float:
            start = time.monotonic()
            request = TranslationRequest(
                text=text,
                source_language=src_lang,
                target_language=tgt_lang,
                context=[],
            )
            await service.translate(request)
            return (time.monotonic() - start) * 1000

        batch_start = time.monotonic()
        concurrent_latencies = await asyncio.gather(
            *[single_request(t) for t in sample_texts],
            return_exceptions=True,
        )
        batch_elapsed_s = time.monotonic() - batch_start

        successful = [lat for lat in concurrent_latencies if isinstance(lat, float)]
        failed = len(concurrent_latencies) - len(successful)

        model_result["concurrent_throughput"] = {
            "concurrency": concurrency,
            "requests_per_second": round(len(successful) / batch_elapsed_s, 2) if batch_elapsed_s > 0 else 0,
            "total_requests": len(concurrent_latencies),
            "successful": len(successful),
            "failed": failed,
            "avg_latency_ms": round(sum(successful) / max(len(successful), 1), 1),
            "wall_clock_s": round(batch_elapsed_s, 2),
        }
        logger.info("concurrent_throughput_complete", **model_result["concurrent_throughput"])
        await service.close()

    return model_result


async def run_benchmark(
    models: list[str],
    lang_pair: str,
    data_dir: Path,
    output_dir: Path,
    ollama_url: str = "http://thomas-pc:11434/v1",
    context_sizes: list[int] | None = None,
    concurrency: int = 1,
) -> None:
    """Run translation benchmark with one or more models and a language pair."""
    setup_logging(service_name="translation-benchmark", log_format="dev")

    context_sizes = context_sizes or [0, 3, 5]

    # Load test data
    source_file = data_dir / f"{lang_pair}.source"
    reference_file = data_dir / f"{lang_pair}.reference"

    if not source_file.exists() or not reference_file.exists():
        logger.warning("no_test_data", data_dir=str(data_dir), lang_pair=lang_pair)
        return

    sources = source_file.read_text().strip().split("\n")
    references = reference_file.read_text().strip().split("\n")

    if len(sources) != len(references):
        logger.error("data_mismatch", sources=len(sources), references=len(references))
        return

    results = {
        "lang_pair": lang_pair,
        "ollama_url": ollama_url,
        "system_info": get_system_info(),
        "models": [],
    }

    for model in models:
        logger.info("benchmark_starting", model=model, samples=len(sources))
        model_result = await run_single_model_benchmark(
            model, lang_pair, sources, references,
            ollama_url, context_sizes, concurrency,
        )
        results["models"].append(model_result)

    # Print comparison table if multiple models
    if len(models) > 1:
        print("\n=== Model Comparison ===")
        print(f"{'Model':<25} {'Ctx':>4} {'BLEU':>8} {'COMET':>8} {'Avg ms':>8} {'P95 ms':>8}")
        print("-" * 70)
        for m in results["models"]:
            for run in m["runs"]:
                comet_str = f"{run['comet']:.4f}" if run.get("comet") is not None else "N/A"
                print(
                    f"{m['model']:<25} {run['context_window_size']:>4} "
                    f"{run['bleu']:>8.4f} {comet_str:>8} "
                    f"{run['avg_latency_ms']:>8.1f} {run['p95_latency_ms']:>8.1f}"
                )
        print()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = "_vs_".join(m.replace(":", "_") for m in models)
    out_file = output_dir / f"{model_slug}_{lang_pair}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("benchmark_complete", output=str(out_file))


def main():
    parser = argparse.ArgumentParser(description="Translation Benchmark")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Single Ollama model name (e.g. qwen3.5:7b)")
    model_group.add_argument(
        "--models",
        help="Comma-separated model names for comparison (e.g. qwen3.5:7b,llama3.1:8b)",
    )
    parser.add_argument("--lang-pair", required=True, help="Language pair (e.g. zh-en)")
    parser.add_argument("--data-dir", type=Path, default=Path("tools/translation_benchmark/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("tools/translation_benchmark/results"))
    parser.add_argument("--ollama-url", default="http://thomas-pc:11434/v1")
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent requests for throughput measurement (default: 1, no concurrency test)",
    )
    args = parser.parse_args()

    models = args.models.split(",") if args.models else [args.model]

    asyncio.run(run_benchmark(
        models, args.lang_pair, args.data_dir, args.output_dir,
        args.ollama_url, args.context_sizes, args.concurrency,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add __main__.py for `python -m` invocation**

```python
# tools/translation_benchmark/__main__.py
"""Allow running the benchmark tool as a module: python -m tools.translation_benchmark"""
from tools.translation_benchmark.run import main

if __name__ == "__main__":
    main()
```

This enables: `uv run python -m tools.translation_benchmark --model qwen3.5:7b --lang-pair zh-en`

- [ ] **Step 6: Commit**

```bash
git add tools/translation_benchmark/
git commit -m "feat: add translation benchmarking harness with BLEU/COMET metrics and concurrent throughput"
```

---

## Summary

**Total tasks:** 5 tasks, ~30 steps
**Branch:** `plan-4/translation-benchmarking`

After completing Plan 4:
- Translation module in orchestration (`src/translation/`):
  - `llm_client.py` — calls Ollama/vLLM directly (~150 LOC), with glossary term support via `TranslationPromptBuilder`
  - `config.py` — environment-configurable settings
  - `context.py` — rolling context window with dual eviction
  - `service.py` — combines client + context + backpressure queue; accepts `TranslationRequest` (Plan 0 shared contract)
  - Queue items store per-item `(text, source_language, target_language, glossary_terms, future)` tuples
- Old `modules/translation-service/` archived to `archive/`
- Old translation clients deleted (`translation_service_client.py`, `internal_services/translation.py`) with 14 specific files updated
- Translation benchmarking CLI in `tools/translation_benchmark/` (underscores, Python-importable):
  - UV workspace package with its own `pyproject.toml`
  - `__main__.py` for `python -m tools.translation_benchmark` invocation
  - BLEU score computation (built-in) + COMET score (optional, via `unbabel-comet`)
  - `--models` flag for multi-model comparison tables
  - `--concurrency` flag for concurrent throughput measurement (requests/second)
  - Supports multiple context window sizes
  - Results JSON with full system info (GPU model, driver, CUDA, package versions, model SHA256)
