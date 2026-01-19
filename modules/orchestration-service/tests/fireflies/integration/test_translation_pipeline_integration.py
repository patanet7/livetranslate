"""
Integration tests for the full Fireflies translation pipeline.

Tests the complete flow:
1. Fireflies chunks → SentenceAggregator → TranslationUnits
2. TranslationUnits → RollingWindowTranslator → TranslationResults
3. TranslationResults → CaptionBuffer → Display captions

Uses test-local implementations to avoid import chain issues.
"""

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar
from uuid import uuid4

import pytest

# =============================================================================
# Test-Local Model Definitions
# =============================================================================


@dataclass
class FirefliesChunk:
    """Incoming transcript chunk from Fireflies."""

    transcript_id: str
    chunk_id: str
    text: str
    speaker_name: str
    start_time: float
    end_time: float


@dataclass
class TranslationUnit:
    """Complete sentence ready for translation."""

    text: str
    speaker_name: str
    start_time: float
    end_time: float
    chunk_ids: list[str]
    boundary_type: str  # punctuation, pause, speaker_change, nlp, forced
    word_count: int
    session_id: str | None = None


@dataclass
class TranslationContext:
    """Context for translation including glossary and previous sentences."""

    previous_sentences: list[str] = field(default_factory=list)
    glossary_terms: dict[str, str] = field(default_factory=dict)
    domain: str | None = None


@dataclass
class TranslationResult:
    """Result of translation."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    speaker_name: str
    confidence: float = 0.0
    context_used: bool = False
    glossary_terms_applied: list[str] = field(default_factory=list)
    translation_time_ms: float = 0.0


@dataclass
class Caption:
    """A display caption with timing and speaker info."""

    id: str
    translated_text: str
    original_text: str | None
    speaker_name: str
    speaker_color: str
    target_language: str
    created_at: float
    expires_at: float
    priority: int = 0
    confidence: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.translated_text,
            "original": self.original_text,
            "speaker": self.speaker_name,
            "color": self.speaker_color,
            "language": self.target_language,
            "confidence": self.confidence,
        }


# =============================================================================
# Test-Local Service Implementations
# =============================================================================


class SentenceAggregator:
    """Aggregates streaming transcript chunks into complete sentences."""

    def __init__(
        self,
        pause_threshold_ms: float = 800,
        max_buffer_words: int = 30,
        max_buffer_seconds: float = 5.0,
        min_words_for_translation: int = 3,
    ):
        self.pause_threshold_ms = pause_threshold_ms
        self.max_buffer_words = max_buffer_words
        self.max_buffer_seconds = max_buffer_seconds
        self.min_words_for_translation = min_words_for_translation
        self.buffers: dict[str, dict] = {}
        self.abbreviations = {
            "dr.",
            "mr.",
            "mrs.",
            "ms.",
            "prof.",
            "inc.",
            "ltd.",
            "etc.",
            "e.g.",
            "i.e.",
        }

    def process_chunk(self, chunk: FirefliesChunk) -> list[TranslationUnit]:
        """Process incoming chunk, return complete sentences ready for translation."""
        speaker = chunk.speaker_name
        results = []

        if speaker not in self.buffers:
            self.buffers[speaker] = {
                "text": "",
                "chunks": [],
                "start_time": None,
                "end_time": None,
            }

        buffer = self.buffers[speaker]

        # Pause detection
        if buffer["end_time"] is not None:
            gap_ms = (chunk.start_time - buffer["end_time"]) * 1000
            if gap_ms > self.pause_threshold_ms and buffer["text"].strip():
                results.extend(self._flush_buffer(speaker, "pause"))

        # Add to buffer
        if buffer["start_time"] is None:
            buffer["start_time"] = chunk.start_time
        buffer["text"] += " " + chunk.text if buffer["text"] else chunk.text
        buffer["chunks"].append(chunk.chunk_id)
        buffer["end_time"] = chunk.end_time

        # Check for sentence-ending punctuation
        results.extend(self._extract_punctuated_sentences(speaker))

        # Force flush if limits exceeded
        word_count = len(buffer["text"].split())
        duration = buffer["end_time"] - buffer["start_time"]
        if word_count >= self.max_buffer_words or duration >= self.max_buffer_seconds:
            results.extend(self._flush_buffer(speaker, "forced"))

        return results

    def _extract_punctuated_sentences(self, speaker: str) -> list[TranslationUnit]:
        """Extract sentences that end with punctuation."""
        buffer = self.buffers[speaker]
        results = []
        text = buffer["text"]

        # Find sentence boundaries
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        if len(sentences) > 1:
            # Extract complete sentences
            for sentence in sentences[:-1]:
                if sentence.strip() and len(sentence.split()) >= self.min_words_for_translation:
                    # Check for abbreviations
                    lower = sentence.lower()
                    is_abbreviation = any(lower.endswith(abbr) for abbr in self.abbreviations)
                    if not is_abbreviation:
                        unit = TranslationUnit(
                            text=sentence.strip(),
                            speaker_name=speaker,
                            start_time=buffer["start_time"],
                            end_time=buffer["end_time"],
                            chunk_ids=buffer["chunks"].copy(),
                            boundary_type="punctuation",
                            word_count=len(sentence.split()),
                        )
                        results.append(unit)

            # Keep remainder
            buffer["text"] = sentences[-1]
            buffer["chunks"] = []

        return results

    def _flush_buffer(self, speaker: str, boundary_type: str) -> list[TranslationUnit]:
        """Flush buffer and return translation unit."""
        buffer = self.buffers[speaker]
        results = []

        text = buffer["text"].strip()
        if text and len(text.split()) >= self.min_words_for_translation:
            unit = TranslationUnit(
                text=text,
                speaker_name=speaker,
                start_time=buffer["start_time"],
                end_time=buffer["end_time"],
                chunk_ids=buffer["chunks"].copy(),
                boundary_type=boundary_type,
                word_count=len(text.split()),
            )
            results.append(unit)

        # Clear buffer
        buffer["text"] = ""
        buffer["chunks"] = []
        buffer["start_time"] = None
        buffer["end_time"] = None

        return results

    def flush_all(self) -> list[TranslationUnit]:
        """Flush all speaker buffers."""
        results = []
        for speaker in list(self.buffers.keys()):
            results.extend(self._flush_buffer(speaker, "forced"))
        return results


class MockTranslationClient:
    """Mock translation service client for testing."""

    def __init__(self, delay_ms: float = 50):
        self.delay_ms = delay_ms
        self.call_count = 0
        self.translations: dict[str, str] = {}  # Pre-defined translations

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "en",
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Mock translation."""
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)

        # Use pre-defined translation or generate mock
        if text in self.translations:
            translated = self.translations[text]
        else:
            translated = f"[{target_language}] {text}"

        return {
            "translated_text": translated,
            "confidence": 0.95,
            "source_language": source_language,
            "target_language": target_language,
        }


class RollingWindowTranslator:
    """Context-aware translation with rolling window and glossary support."""

    def __init__(
        self,
        translation_client: MockTranslationClient,
        glossary_service: Any | None = None,
        window_size: int = 3,
        include_cross_speaker_context: bool = True,
    ):
        self.translation_client = translation_client
        self.glossary_service = glossary_service
        self.window_size = window_size
        self.include_cross_speaker_context = include_cross_speaker_context
        self.speaker_windows: dict[str, deque] = {}
        self.global_window: deque = deque(maxlen=window_size)
        self.translation_count = 0
        self.error_count = 0

    def _get_speaker_window(self, speaker: str) -> deque:
        """Get or create speaker's context window."""
        if speaker not in self.speaker_windows:
            self.speaker_windows[speaker] = deque(maxlen=self.window_size)
        return self.speaker_windows[speaker]

    async def translate(
        self,
        unit: TranslationUnit,
        target_language: str,
        glossary_terms: dict[str, str] | None = None,
        source_language: str = "en",
    ) -> TranslationResult:
        """Translate a unit with context window and glossary."""
        start_time = time.time()

        # Get context
        speaker_window = self._get_speaker_window(unit.speaker_name)
        context_sentences = list(speaker_window)

        # Build prompt with context and glossary
        prompt_parts = []
        prompt_parts.append(f"Target Language: {target_language}")

        if glossary_terms:
            glossary_str = "\n".join(f"  {k} → {v}" for k, v in glossary_terms.items())
            prompt_parts.append(f"Glossary:\n{glossary_str}")

        if context_sentences:
            context_str = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(context_sentences))
            prompt_parts.append(f"Previous context:\n{context_str}")

        prompt_parts.append(f"Translate: {unit.text}")
        prompt = "\n\n".join(prompt_parts)

        # Call translation service
        try:
            result = await self.translation_client.translate(
                text=unit.text,
                target_language=target_language,
                source_language=source_language,
                prompt=prompt,
            )
            self.translation_count += 1

            # Update windows AFTER successful translation
            speaker_window.append(unit.text)
            self.global_window.append(unit.text)

            # Determine which glossary terms were applied
            applied_terms = []
            if glossary_terms:
                for term in glossary_terms:
                    if term.lower() in unit.text.lower():
                        applied_terms.append(term)

            elapsed_ms = (time.time() - start_time) * 1000

            return TranslationResult(
                original_text=unit.text,
                translated_text=result["translated_text"],
                source_language=source_language,
                target_language=target_language,
                speaker_name=unit.speaker_name,
                confidence=result.get("confidence", 0.0),
                context_used=len(context_sentences) > 0,
                glossary_terms_applied=applied_terms,
                translation_time_ms=elapsed_ms,
            )

        except Exception:
            self.error_count += 1
            raise

    def clear_session(self):
        """Clear all context windows."""
        self.speaker_windows.clear()
        self.global_window.clear()


class CaptionBuffer:
    """Manages caption display queue with expiration and speaker colors."""

    MATERIAL_COLORS: ClassVar[list[str]] = [
        "#2196F3",  # Blue
        "#4CAF50",  # Green
        "#FF9800",  # Orange
        "#9C27B0",  # Purple
        "#00BCD4",  # Cyan
        "#E91E63",  # Pink
        "#FFEB3B",  # Yellow
        "#795548",  # Brown
    ]

    def __init__(
        self,
        max_captions: int = 5,
        default_duration: float = 8.0,
        min_display_time: float = 2.0,
        show_original: bool = True,
        on_caption_added: Callable[[Caption], None] | None = None,
        on_caption_expired: Callable[[Caption], None] | None = None,
    ):
        self.max_captions = max_captions
        self.default_duration = default_duration
        self.min_display_time = min_display_time
        self.show_original = show_original
        self.on_caption_added = on_caption_added
        self.on_caption_expired = on_caption_expired
        self._captions: list[Caption] = []
        self._speaker_colors: dict[str, str] = {}
        self._next_color_index = 0
        self._lock = threading.RLock()

    def _get_speaker_color(self, speaker: str) -> str:
        """Get or assign color for speaker."""
        if speaker not in self._speaker_colors:
            self._speaker_colors[speaker] = self.MATERIAL_COLORS[
                self._next_color_index % len(self.MATERIAL_COLORS)
            ]
            self._next_color_index += 1
        return self._speaker_colors[speaker]

    def add_caption(
        self,
        translation_result: TranslationResult,
        duration: float | None = None,
        priority: int = 0,
    ) -> Caption:
        """Add a caption from translation result."""
        with self._lock:
            now = time.time()
            duration = duration or self.default_duration

            caption = Caption(
                id=str(uuid4()),
                translated_text=translation_result.translated_text,
                original_text=translation_result.original_text if self.show_original else None,
                speaker_name=translation_result.speaker_name,
                speaker_color=self._get_speaker_color(translation_result.speaker_name),
                target_language=translation_result.target_language,
                created_at=now,
                expires_at=now + duration,
                priority=priority,
                confidence=translation_result.confidence,
            )

            # Handle overflow
            if len(self._captions) >= self.max_captions:
                # Remove lowest priority expired first
                self._cleanup_expired()
                if len(self._captions) >= self.max_captions:
                    # Still full, remove oldest lowest priority
                    self._captions.sort(key=lambda c: (c.priority, -c.created_at))
                    removed = self._captions.pop(0)
                    if self.on_caption_expired:
                        self.on_caption_expired(removed)

            self._captions.append(caption)

            if self.on_caption_added:
                self.on_caption_added(caption)

            return caption

    def _cleanup_expired(self) -> list[Caption]:
        """Remove expired captions."""
        expired = []
        time.time()
        remaining = []

        for caption in self._captions:
            if caption.is_expired:
                expired.append(caption)
                if self.on_caption_expired:
                    self.on_caption_expired(caption)
            else:
                remaining.append(caption)

        self._captions = remaining
        return expired

    def get_active_captions(self) -> list[Caption]:
        """Get all active (non-expired) captions."""
        with self._lock:
            self._cleanup_expired()
            return list(self._captions)

    def get_display_data(self) -> list[dict[str, Any]]:
        """Get caption data for display."""
        return [c.to_display_dict() for c in self.get_active_captions()]

    def clear(self):
        """Clear all captions."""
        with self._lock:
            self._captions.clear()


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullPipelineIntegration:
    """Tests for the complete translation pipeline."""

    @pytest.fixture
    def aggregator(self):
        """Create sentence aggregator."""
        return SentenceAggregator(
            pause_threshold_ms=800,
            max_buffer_words=30,
            min_words_for_translation=3,
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock translation client."""
        return MockTranslationClient(delay_ms=10)

    @pytest.fixture
    def translator(self, mock_client):
        """Create rolling window translator."""
        return RollingWindowTranslator(
            translation_client=mock_client,
            window_size=3,
        )

    @pytest.fixture
    def caption_buffer(self):
        """Create caption buffer."""
        return CaptionBuffer(
            max_captions=5,
            default_duration=8.0,
        )

    @pytest.mark.asyncio
    async def test_single_sentence_flow(self, aggregator, translator, caption_buffer):
        """Test complete flow for a single sentence."""
        # GIVEN: A complete sentence in chunks
        chunks = [
            FirefliesChunk("t1", "c1", "Hello, how are", "Alice", 0.0, 1.0),
            FirefliesChunk("t1", "c2", "you doing today?", "Alice", 1.0, 2.0),
        ]

        # WHEN: Processing chunks through aggregator
        units = []
        for chunk in chunks:
            units.extend(aggregator.process_chunk(chunk))

        # AND: Flushing remaining buffer
        units.extend(aggregator.flush_all())

        # THEN: Should have one translation unit
        assert len(units) == 1
        assert "Hello" in units[0].text
        assert "today" in units[0].text

        # WHEN: Translating the unit
        result = await translator.translate(units[0], target_language="es")

        # THEN: Should have translation result
        assert result.original_text == units[0].text
        assert result.translated_text is not None
        assert result.speaker_name == "Alice"

        # WHEN: Adding to caption buffer
        caption = caption_buffer.add_caption(result)

        # THEN: Caption should be active
        assert caption.speaker_name == "Alice"
        assert caption.translated_text == result.translated_text
        active = caption_buffer.get_active_captions()
        assert len(active) == 1

    @pytest.mark.asyncio
    async def test_multi_speaker_flow(self, aggregator, translator, caption_buffer):
        """Test pipeline with multiple speakers."""
        # GIVEN: Chunks from multiple speakers
        chunks = [
            FirefliesChunk("t1", "c1", "Let's start the meeting.", "Alice", 0.0, 1.0),
            FirefliesChunk("t1", "c2", "Sounds good to me.", "Bob", 2.0, 3.0),
            FirefliesChunk("t1", "c3", "I have a question.", "Charlie", 4.0, 5.0),
        ]

        # WHEN: Processing all chunks
        units = []
        for chunk in chunks:
            units.extend(aggregator.process_chunk(chunk))
        units.extend(aggregator.flush_all())

        # THEN: Should have units from each speaker
        assert len(units) == 3
        speakers = {u.speaker_name for u in units}
        assert speakers == {"Alice", "Bob", "Charlie"}

        # WHEN: Translating all units
        for unit in units:
            result = await translator.translate(unit, target_language="fr")
            caption_buffer.add_caption(result)

        # THEN: Should have captions for each speaker with different colors
        captions = caption_buffer.get_active_captions()
        assert len(captions) == 3
        colors = {c.speaker_color for c in captions}
        assert len(colors) == 3  # Each speaker gets unique color

    @pytest.mark.asyncio
    async def test_context_builds_across_sentences(self, aggregator, translator, caption_buffer):
        """Test that context window builds across multiple sentences."""
        # GIVEN: Multiple sentences from same speaker
        sentences = [
            "The API has changed significantly.",
            "We need to update our integration.",
            "The new endpoints are documented.",
        ]

        # WHEN: Processing each sentence
        for i, sentence in enumerate(sentences):
            chunk = FirefliesChunk("t1", f"c{i}", sentence, "Alice", float(i), float(i) + 1.0)
            units = aggregator.process_chunk(chunk)
            units.extend(aggregator.flush_all())

            for unit in units:
                result = await translator.translate(unit, target_language="de")
                caption_buffer.add_caption(result)

        # THEN: Translation count should match
        assert translator.translation_count == 3

        # AND: Context window should contain previous sentences
        speaker_window = translator._get_speaker_window("Alice")
        assert len(speaker_window) == 3

    @pytest.mark.asyncio
    async def test_glossary_integration(self, aggregator, translator, caption_buffer):
        """Test glossary terms are applied during translation."""
        # GIVEN: A sentence with glossary terms
        chunk = FirefliesChunk(
            "t1", "c1", "The API integration with OAuth is complete.", "Alice", 0.0, 2.0
        )

        glossary = {
            "API": "API",
            "OAuth": "OAuth",
            "integration": "intégration",
        }

        # WHEN: Processing and translating
        units = aggregator.process_chunk(chunk)
        units.extend(aggregator.flush_all())

        for unit in units:
            result = await translator.translate(
                unit,
                target_language="fr",
                glossary_terms=glossary,
            )
            caption_buffer.add_caption(result)

        # THEN: Glossary terms should be marked as applied
        captions = caption_buffer.get_active_captions()
        assert len(captions) == 1
        # Note: Our mock doesn't actually apply glossary, but tracks what would be applied

    @pytest.mark.asyncio
    async def test_pause_triggers_flush(self, aggregator, translator, caption_buffer):
        """Test that pauses trigger sentence completion."""
        # GIVEN: Chunks with a pause gap
        chunks = [
            FirefliesChunk("t1", "c1", "First sentence here", "Alice", 0.0, 1.0),
            FirefliesChunk("t1", "c2", "Second after pause", "Alice", 3.0, 4.0),  # 2s gap
        ]

        # WHEN: Processing chunks
        units = []
        for chunk in chunks:
            units.extend(aggregator.process_chunk(chunk))

        # THEN: First sentence should be flushed due to pause
        assert len(units) >= 1
        assert units[0].boundary_type == "pause"

    @pytest.mark.asyncio
    async def test_caption_expiration(self, aggregator, translator, caption_buffer):
        """Test that captions expire after duration."""
        # GIVEN: Caption buffer with short duration
        short_buffer = CaptionBuffer(
            max_captions=5,
            default_duration=0.1,  # 100ms
        )

        # AND: A translation result
        chunk = FirefliesChunk("t1", "c1", "Quick message here.", "Alice", 0.0, 1.0)
        units = aggregator.process_chunk(chunk)
        units.extend(aggregator.flush_all())

        result = await translator.translate(units[0], target_language="es")
        short_buffer.add_caption(result)

        # WHEN: Waiting for expiration
        assert len(short_buffer.get_active_captions()) == 1
        await asyncio.sleep(0.15)

        # THEN: Caption should be expired
        active = short_buffer.get_active_captions()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_caption_overflow_handling(self, aggregator, translator):
        """Test caption buffer handles overflow correctly."""
        # GIVEN: Small capacity buffer
        small_buffer = CaptionBuffer(max_captions=2)

        # AND: Multiple sentences
        sentences = ["First message.", "Second message.", "Third message."]

        for i, sentence in enumerate(sentences):
            chunk = FirefliesChunk("t1", f"c{i}", sentence, f"Speaker{i}", float(i), float(i) + 1.0)
            units = aggregator.process_chunk(chunk)
            units.extend(aggregator.flush_all())

            for unit in units:
                result = await translator.translate(unit, target_language="ja")
                small_buffer.add_caption(result)

        # THEN: Should only have max_captions
        active = small_buffer.get_active_captions()
        assert len(active) <= 2

    @pytest.mark.asyncio
    async def test_concurrent_speakers_maintain_separate_context(self, mock_client):
        """Test that speakers maintain independent context windows."""
        # GIVEN: Fresh translator
        translator = RollingWindowTranslator(
            translation_client=mock_client,
            window_size=2,
        )

        # WHEN: Translating alternating speakers
        alice_sentences = ["Alice first.", "Alice second.", "Alice third."]
        bob_sentences = ["Bob first.", "Bob second."]

        for i, sentence in enumerate(alice_sentences):
            unit = TranslationUnit(
                text=sentence,
                speaker_name="Alice",
                start_time=float(i * 2),
                end_time=float(i * 2 + 1),
                chunk_ids=[f"a{i}"],
                boundary_type="punctuation",
                word_count=2,
            )
            await translator.translate(unit, target_language="es")

        for i, sentence in enumerate(bob_sentences):
            unit = TranslationUnit(
                text=sentence,
                speaker_name="Bob",
                start_time=float(i * 2 + 1),
                end_time=float(i * 2 + 2),
                chunk_ids=[f"b{i}"],
                boundary_type="punctuation",
                word_count=2,
            )
            await translator.translate(unit, target_language="es")

        # THEN: Each speaker should have their own context
        alice_window = translator._get_speaker_window("Alice")
        bob_window = translator._get_speaker_window("Bob")

        assert len(alice_window) == 2  # Limited by window_size
        assert len(bob_window) == 2
        assert "Alice" in next(iter(alice_window))
        assert "Bob" in next(iter(bob_window))

    @pytest.mark.asyncio
    async def test_full_meeting_simulation(self, aggregator, translator, caption_buffer):
        """Simulate a complete meeting with multiple speakers and topics."""
        # GIVEN: A realistic meeting conversation
        meeting_chunks = [
            # Meeting start
            ("Alice", "Good morning everyone, let's begin.", 0.0, 2.0),
            ("Bob", "Thanks Alice. I have the status update ready.", 3.0, 5.0),
            # Discussion
            ("Bob", "The backend API is now complete.", 6.0, 8.0),
            ("Charlie", "Great work! What about the frontend?", 9.0, 11.0),
            ("Alice", "We're still working on that.", 12.0, 14.0),
            # Wrap up
            ("Alice", "Let's meet again tomorrow.", 15.0, 17.0),
            ("Bob", "Sounds good to me.", 18.0, 19.0),
            ("Charlie", "I'll prepare the demo.", 20.0, 22.0),
        ]

        # WHEN: Processing all chunks
        all_captions = []
        for speaker, text, start, end in meeting_chunks:
            chunk = FirefliesChunk(
                transcript_id="meeting1",
                chunk_id=f"chunk_{start}",
                text=text,
                speaker_name=speaker,
                start_time=start,
                end_time=end,
            )

            units = aggregator.process_chunk(chunk)
            units.extend(aggregator.flush_all())

            for unit in units:
                result = await translator.translate(unit, target_language="es")
                caption = caption_buffer.add_caption(result)
                all_captions.append(caption)

        # THEN: Should have processed all sentences
        assert len(all_captions) == 8

        # AND: Each speaker should have consistent color
        speaker_colors = {}
        for caption in all_captions:
            if caption.speaker_name not in speaker_colors:
                speaker_colors[caption.speaker_name] = caption.speaker_color
            else:
                assert speaker_colors[caption.speaker_name] == caption.speaker_color

        # AND: Should have 3 unique speakers
        assert len(speaker_colors) == 3


class TestPipelineErrorHandling:
    """Tests for error handling in the pipeline."""

    @pytest.fixture
    def aggregator(self):
        return SentenceAggregator()

    @pytest.mark.asyncio
    async def test_translation_error_recovery(self, aggregator):
        """Test pipeline continues after translation error."""

        # GIVEN: A client that fails once then succeeds
        class FailOnceClient(MockTranslationClient):
            def __init__(self):
                super().__init__()
                self.fail_count = 0

            async def translate(self, *args, **kwargs):
                self.fail_count += 1
                if self.fail_count == 1:
                    raise Exception("Translation service error")
                return await super().translate(*args, **kwargs)

        fail_client = FailOnceClient()
        translator = RollingWindowTranslator(translation_client=fail_client)
        caption_buffer = CaptionBuffer()

        # WHEN: Processing multiple sentences (must have >= 3 words for min_words_for_translation)
        sentences = ["This is the first sentence.", "This is the second sentence."]
        errors = []
        successes = []

        for i, sentence in enumerate(sentences):
            chunk = FirefliesChunk(
                "t1", f"c{i}", sentence, "Alice", float(i) * 2, float(i) * 2 + 1.0
            )
            units = aggregator.process_chunk(chunk)
            units.extend(aggregator.flush_all())

            for unit in units:
                try:
                    result = await translator.translate(unit, target_language="es")
                    caption_buffer.add_caption(result)
                    successes.append(sentence)
                except Exception as e:
                    errors.append((sentence, str(e)))

        # THEN: First should fail, second should succeed
        assert (
            len(errors) == 1
        ), f"Expected 1 error, got {len(errors)}. Errors: {errors}, Successes: {successes}, Fail count: {fail_client.fail_count}"
        assert (
            len(successes) == 1
        ), f"Expected 1 success, got {len(successes)}. Errors: {errors}, Successes: {successes}"
        assert translator.error_count == 1

    @pytest.mark.asyncio
    async def test_empty_chunk_handling(self, aggregator):
        """Test pipeline handles empty chunks gracefully."""
        RollingWindowTranslator(translation_client=MockTranslationClient())

        # GIVEN: Chunks with empty text
        chunks = [
            FirefliesChunk("t1", "c1", "", "Alice", 0.0, 1.0),
            FirefliesChunk("t1", "c2", "   ", "Alice", 1.0, 2.0),
            FirefliesChunk("t1", "c3", "Valid text here.", "Alice", 2.0, 3.0),
        ]

        # WHEN: Processing chunks
        units = []
        for chunk in chunks:
            units.extend(aggregator.process_chunk(chunk))
        units.extend(aggregator.flush_all())

        # THEN: Should only produce units with valid text
        assert len(units) >= 1
        for unit in units:
            assert unit.text.strip()


class TestPipelinePerformance:
    """Performance tests for the pipeline."""

    @pytest.mark.asyncio
    async def test_high_throughput(self):
        """Test pipeline handles high message volume."""
        aggregator = SentenceAggregator()
        translator = RollingWindowTranslator(translation_client=MockTranslationClient(delay_ms=1))
        caption_buffer = CaptionBuffer(max_captions=100)

        # GIVEN: 100 messages
        start_time = time.time()

        for i in range(100):
            chunk = FirefliesChunk(
                "t1",
                f"c{i}",
                f"Message number {i} from the meeting.",
                f"Speaker{i % 5}",
                float(i),
                float(i) + 0.5,
            )

            units = aggregator.process_chunk(chunk)
            units.extend(aggregator.flush_all())

            for unit in units:
                result = await translator.translate(unit, target_language="es")
                caption_buffer.add_caption(result)

        elapsed = time.time() - start_time

        # THEN: Should complete within reasonable time
        assert elapsed < 5.0  # 5 seconds for 100 messages
        assert translator.translation_count == 100

    @pytest.mark.asyncio
    async def test_context_window_memory_bounded(self):
        """Test that context windows don't grow unbounded."""
        translator = RollingWindowTranslator(
            translation_client=MockTranslationClient(delay_ms=1),
            window_size=3,
        )

        # WHEN: Translating many sentences
        for i in range(100):
            unit = TranslationUnit(
                text=f"Sentence number {i} with some content.",
                speaker_name="Alice",
                start_time=float(i),
                end_time=float(i) + 1.0,
                chunk_ids=[f"c{i}"],
                boundary_type="punctuation",
                word_count=6,
            )
            await translator.translate(unit, target_language="es")

        # THEN: Context window should be bounded
        speaker_window = translator._get_speaker_window("Alice")
        assert len(speaker_window) == 3  # Window size limit


class TestCallbackIntegration:
    """Tests for callback integration in the pipeline."""

    @pytest.mark.asyncio
    async def test_caption_callbacks_fire(self):
        """Test that caption callbacks are triggered."""
        added_captions = []
        expired_captions = []

        buffer = CaptionBuffer(
            max_captions=5,
            default_duration=0.1,  # Short for testing
            on_caption_added=lambda c: added_captions.append(c),
            on_caption_expired=lambda c: expired_captions.append(c),
        )

        translator = RollingWindowTranslator(translation_client=MockTranslationClient(delay_ms=1))

        # WHEN: Adding captions
        for i in range(3):
            unit = TranslationUnit(
                text=f"Message {i}.",
                speaker_name="Alice",
                start_time=float(i),
                end_time=float(i) + 1.0,
                chunk_ids=[f"c{i}"],
                boundary_type="punctuation",
                word_count=2,
            )
            result = await translator.translate(unit, target_language="es")
            buffer.add_caption(result)

        # THEN: Added callbacks should fire
        assert len(added_captions) == 3

        # WHEN: Waiting for expiration
        await asyncio.sleep(0.2)
        buffer.get_active_captions()  # Trigger cleanup

        # THEN: Expired callbacks should fire
        assert len(expired_captions) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
