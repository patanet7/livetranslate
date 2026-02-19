"""
Sentence Aggregator Service

Aggregates streaming transcript chunks into complete sentences using hybrid
boundary detection:
1. Speaker change detection
2. Pause detection (timing gaps > threshold)
3. Punctuation boundary detection
4. NLP sentence boundary detection (spaCy)
5. Buffer limits (max words / max seconds)

Produces TranslationUnit objects ready for the RollingWindowTranslator.

Reference: docs/archive/root-reports/status-plans/FIREFLIES_ADAPTATION_PLAN.md
Section "Sentence Aggregation System"
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime

from livetranslate_common.logging import get_logger
from models.fireflies import (
    FirefliesChunk,
    FirefliesSessionConfig,
    SpeakerBuffer,
    TranslationUnit,
)

logger = get_logger()


# =============================================================================
# Constants
# =============================================================================

# Common abbreviations that should NOT trigger sentence boundaries
ABBREVIATIONS = frozenset(
    {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "jr.",
        "inc.",
        "ltd.",
        "corp.",
        "co.",
        "etc.",
        "e.g.",
        "i.e.",
        "vs.",
        "v.",
        "a.m.",
        "p.m.",
        "est.",
        "pst.",
        "u.s.",
        "u.k.",
        "no.",
        "st.",
        "ave.",
        "blvd.",
        "dept.",
        "fig.",
        "vol.",
        "ch.",
        "pp.",
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "sept.",
        "oct.",
        "nov.",
        "dec.",
    }
)

# Sentence-ending punctuation (including Unicode variants)
SENTENCE_ENDINGS = frozenset(
    {
        ".",
        "!",
        "?",
        "\u3002",  # Chinese/Japanese period
        "\uff01",  # Chinese/Japanese exclamation
        "\uff1f",  # Chinese/Japanese question
        "¿",  # Spanish opening question (pair marker)
        "¡",  # Spanish opening exclamation (pair marker)
    }
)

# Pattern for decimal numbers (should not split on period)
DECIMAL_PATTERN = re.compile(r"\d+\.\d+")


# =============================================================================
# NLP Loader (Lazy Loading)
# =============================================================================


class NLPLoader:
    """Lazy loader for spaCy NLP model"""

    _nlp = None
    _load_attempted = False

    @classmethod
    def get_nlp(cls):
        """
        Get spaCy NLP model with lazy loading.
        Returns None if spaCy is not available.
        """
        if cls._load_attempted:
            return cls._nlp

        cls._load_attempted = True
        try:
            import spacy

            # Try small model first (fastest)
            for model_name in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]:
                try:
                    cls._nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    return cls._nlp
                except OSError:
                    continue

            logger.warning(
                "No spaCy English model found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
        except ImportError:
            logger.warning("spaCy not installed. NLP boundary detection disabled.")

        return cls._nlp


# =============================================================================
# Sentence Aggregator
# =============================================================================


class SentenceAggregator:
    """
    Aggregates streaming transcript chunks into complete sentences.

    Uses multiple signals for boundary detection:
    - Speaker change: Always flush on speaker change
    - Pause detection: Gap > threshold indicates natural boundary
    - Punctuation: Sentence-ending punctuation (with abbreviation handling)
    - NLP: spaCy sentence boundary detection for mid-buffer detection
    - Limits: Force flush when buffer exceeds word/time limits

    Thread-safe: Each instance maintains its own buffer state.
    """

    def __init__(
        self,
        session_id: str,
        transcript_id: str,
        config: FirefliesSessionConfig | None = None,
        on_sentence_ready: Callable[[TranslationUnit], None] | None = None,
    ):
        """
        Initialize the sentence aggregator.

        Args:
            session_id: Internal session ID
            transcript_id: Fireflies transcript ID
            config: Session configuration (uses defaults if None)
            on_sentence_ready: Callback when a sentence is ready for translation
        """
        self.session_id = session_id
        self.transcript_id = transcript_id

        # Configuration
        self.pause_threshold_ms = config.pause_threshold_ms if config else 800.0
        self.max_buffer_words = config.max_buffer_words if config else 30
        self.max_buffer_seconds = config.max_buffer_seconds if config else 5.0
        self.min_words_for_translation = config.min_words_for_translation if config else 3
        self.use_nlp = config.use_nlp_boundary_detection if config else True
        self.nlp_threshold_words = 10  # Only run NLP when buffer has this many words

        # Per-speaker buffers
        self.buffers: dict[str, SpeakerBuffer] = {}

        # Callback for sentence output
        self.on_sentence_ready = on_sentence_ready

        # Statistics
        self.chunks_processed = 0
        self.sentences_produced = 0
        self.last_speaker: str | None = None

        logger.info(
            f"SentenceAggregator initialized: session={session_id}, "
            f"pause_threshold={self.pause_threshold_ms}ms, "
            f"max_buffer={self.max_buffer_words} words / {self.max_buffer_seconds}s"
        )

    def process_chunk(self, chunk: FirefliesChunk) -> list[TranslationUnit]:
        """
        Process an incoming transcript chunk.

        Returns a list of complete sentences ready for translation.
        May return 0, 1, or multiple TranslationUnits depending on
        boundary detection results.

        Args:
            chunk: Incoming Fireflies chunk

        Returns:
            List of TranslationUnit objects ready for translation
        """
        self.chunks_processed += 1
        results: list[TranslationUnit] = []
        speaker = chunk.speaker_name

        # Get or create buffer for this speaker
        if speaker not in self.buffers:
            self.buffers[speaker] = SpeakerBuffer(speaker_name=speaker)
            logger.debug(f"Created buffer for new speaker: {speaker}")

        buffer = self.buffers[speaker]

        # 1. Speaker change detection
        if self.last_speaker is not None and self.last_speaker != speaker:
            # Flush previous speaker's buffer on speaker change
            prev_buffer = self.buffers.get(self.last_speaker)
            if prev_buffer and prev_buffer.chunks:
                flushed = self._flush_buffer(prev_buffer, "speaker_change")
                results.extend(flushed)

        self.last_speaker = speaker

        # 2. Pause detection (timing gap > threshold)
        if buffer.chunks and self._is_pause(buffer.last_chunk, chunk):
            flushed = self._flush_buffer(buffer, "pause")
            results.extend(flushed)

        # 3. Add chunk to buffer
        buffer.add(chunk)
        logger.debug(
            f"Added chunk to buffer: speaker={speaker}, "
            f"words={buffer.word_count}, text='{chunk.text[:50]}...'"
        )

        # 4. Check for punctuation boundaries
        punct_sentences = self._extract_punctuated_sentences(buffer)
        results.extend(punct_sentences)

        # 5. NLP boundary detection (if buffer is large enough)
        if self.use_nlp and buffer.word_count >= self.nlp_threshold_words:
            nlp_sentences = self._extract_nlp_sentences(buffer)
            results.extend(nlp_sentences)

        # 6. Force flush if limits exceeded
        if self._exceeds_limits(buffer):
            forced = self._flush_buffer(buffer, "forced")
            results.extend(forced)

        # Emit results via callback if registered
        for unit in results:
            self.sentences_produced += 1
            if self.on_sentence_ready:
                self.on_sentence_ready(unit)

        return results

    def flush_all(self) -> list[TranslationUnit]:
        """
        Flush all speaker buffers (e.g., on session end).

        Returns all remaining text as TranslationUnits.
        """
        results: list[TranslationUnit] = []
        for _speaker, buffer in self.buffers.items():
            if buffer.chunks:
                flushed = self._flush_buffer(buffer, "session_end")
                results.extend(flushed)

        for unit in results:
            self.sentences_produced += 1
            if self.on_sentence_ready:
                self.on_sentence_ready(unit)

        return results

    def get_stats(self) -> dict:
        """Get aggregator statistics"""
        return {
            "chunks_processed": self.chunks_processed,
            "sentences_produced": self.sentences_produced,
            "active_buffers": len([b for b in self.buffers.values() if b.chunks]),
            "speakers": list(self.buffers.keys()),
        }

    # =========================================================================
    # Boundary Detection Methods
    # =========================================================================

    def _is_pause(
        self,
        last_chunk: FirefliesChunk | None,
        new_chunk: FirefliesChunk,
    ) -> bool:
        """
        Check if there's a pause between chunks.

        A pause is detected when the gap between the end of the last chunk
        and the start of the new chunk exceeds the threshold.
        """
        if last_chunk is None:
            return False

        gap_ms = (new_chunk.start_time - last_chunk.end_time) * 1000
        is_pause = gap_ms >= self.pause_threshold_ms

        if is_pause:
            logger.debug(f"Pause detected: gap={gap_ms:.0f}ms")

        return is_pause

    def _extract_punctuated_sentences(self, buffer: SpeakerBuffer) -> list[TranslationUnit]:
        """
        Extract complete sentences ending with sentence punctuation.

        Handles abbreviations by checking if the period is part of
        a known abbreviation. Extracts sentences iteratively.
        """
        results: list[TranslationUnit] = []
        text = buffer.get_text()

        # Find all sentence boundaries
        boundaries = []
        for i, char in enumerate(text):
            if char in SENTENCE_ENDINGS and self._is_real_sentence_boundary(text, i):
                boundaries.append(i)

        if not boundaries:
            return results

        # Extract sentences for all boundaries except possibly the last one
        # (which might be the end of the buffer and we want to wait for more)
        start_pos = 0
        extracted_any = False

        for boundary in boundaries:
            sentence_text = text[start_pos : boundary + 1].strip()

            # Check if this sentence meets minimum requirements
            if len(sentence_text.split()) >= self.min_words_for_translation:
                unit = self._create_translation_unit(buffer, sentence_text, "punctuation")
                results.append(unit)
                start_pos = boundary + 1
                extracted_any = True

        # Update buffer with remainder after last extracted sentence
        if extracted_any:
            remainder_text = text[start_pos:].strip()
            if remainder_text:
                last_chunk = buffer.last_chunk
                buffer.reset_to(
                    remainder_text,
                    last_chunk.start_time if last_chunk else 0.0,
                    last_chunk.end_time if last_chunk else 0.0,
                    self.transcript_id,
                )
            else:
                buffer.clear()

        return results

    def _is_real_sentence_boundary(self, text: str, position: int) -> bool:
        """
        Determine if a punctuation mark at position is a real sentence boundary.

        Returns False for:
        - Abbreviations (Dr., Mr., etc.)
        - Decimal numbers (3.50)
        - Ellipsis (...)
        - Mid-word punctuation
        """
        if position >= len(text):
            return False

        char = text[position]

        # Check for ellipsis (part of ... sequence)
        if char == ".":
            # Check if this is part of an ellipsis
            # Look for other periods nearby
            is_ellipsis = False
            if position > 0 and text[position - 1] == ".":
                is_ellipsis = True
            if position + 1 < len(text) and text[position + 1] == ".":
                is_ellipsis = True
            if is_ellipsis:
                return False

        # Get word containing this position
        start = position
        while start > 0 and text[start - 1].isalnum():
            start -= 1

        end = position + 1
        while end < len(text) and text[end].isalnum():
            end += 1

        word_with_punct = text[start:end].lower()

        # Check for abbreviation
        if char == "." and word_with_punct in ABBREVIATIONS:
            return False

        # Check for decimal number - look for digit before AND after the period
        if char == ".":
            has_digit_before = position > 0 and text[position - 1].isdigit()
            has_digit_after = position + 1 < len(text) and text[position + 1].isdigit()
            if has_digit_before and has_digit_after:
                return False

        # Check if followed by lowercase letter (likely not sentence end)
        return not (position + 1 < len(text) and text[position + 1].islower())

    def _extract_nlp_sentences(self, buffer: SpeakerBuffer) -> list[TranslationUnit]:
        """
        Use spaCy NLP to detect sentence boundaries within the buffer.

        Only runs when buffer has >= nlp_threshold_words.
        Extracts complete sentences found by spaCy, keeps remainder.
        """
        nlp = NLPLoader.get_nlp()
        if nlp is None:
            return []

        results: list[TranslationUnit] = []
        text = buffer.get_text()

        # Process with spaCy
        doc = nlp(text)
        sentences = list(doc.sents)

        # If only one sentence or no complete sentences, return empty
        if len(sentences) <= 1:
            return results

        # Extract all complete sentences except the last (which may be incomplete)
        complete_sentences = sentences[:-1]
        remainder = str(sentences[-1]).strip()

        for sent in complete_sentences:
            sent_text = str(sent).strip()
            if len(sent_text.split()) >= self.min_words_for_translation:
                unit = self._create_translation_unit(buffer, sent_text, "nlp")
                results.append(unit)

        # Update buffer with remainder
        if remainder:
            last_chunk = buffer.last_chunk
            buffer.reset_to(
                remainder,
                last_chunk.start_time if last_chunk else 0.0,
                last_chunk.end_time if last_chunk else 0.0,
                self.transcript_id,
            )
        else:
            buffer.clear()

        if results:
            logger.debug(
                f"NLP extracted {len(results)} sentences, remainder: '{remainder[:30]}...'"
            )

        return results

    def _exceeds_limits(self, buffer: SpeakerBuffer) -> bool:
        """Check if buffer exceeds word or time limits"""
        if buffer.word_count >= self.max_buffer_words:
            logger.debug(
                f"Buffer exceeds word limit: {buffer.word_count} >= {self.max_buffer_words}"
            )
            return True

        if buffer.duration_seconds >= self.max_buffer_seconds:
            logger.debug(
                f"Buffer exceeds time limit: {buffer.duration_seconds:.1f}s >= {self.max_buffer_seconds}s"
            )
            return True

        return False

    def _flush_buffer(self, buffer: SpeakerBuffer, boundary_type: str) -> list[TranslationUnit]:
        """
        Flush the entire buffer contents as a TranslationUnit.

        For 'forced' flushes, attempts to find a natural break point.
        """
        results: list[TranslationUnit] = []
        text = buffer.get_text().strip()

        if not text or len(text.split()) < self.min_words_for_translation:
            buffer.clear()
            return results

        # For forced flushes, try to find a good break point
        if boundary_type == "forced" and buffer.word_count > 15:
            # Try to break at a comma, semicolon, or conjunction
            break_patterns = [", ", "; ", " and ", " but ", " or ", " - "]
            best_break = -1

            for pattern in break_patterns:
                idx = text.rfind(pattern)
                if idx > len(text) // 3:  # Don't break too early
                    best_break = idx + len(pattern.rstrip())
                    break

            if best_break > 0:
                first_part = text[:best_break].strip()
                remainder = text[best_break:].strip()

                if len(first_part.split()) >= self.min_words_for_translation:
                    unit = self._create_translation_unit(buffer, first_part, "forced_break")
                    results.append(unit)

                    if remainder:
                        last_chunk = buffer.last_chunk
                        buffer.reset_to(
                            remainder,
                            last_chunk.start_time if last_chunk else 0.0,
                            last_chunk.end_time if last_chunk else 0.0,
                            self.transcript_id,
                        )
                    else:
                        buffer.clear()
                    return results

        # Default: flush entire buffer
        unit = self._create_translation_unit(buffer, text, boundary_type)
        results.append(unit)
        buffer.clear()

        logger.debug(f"Flushed buffer: type={boundary_type}, text='{text[:50]}...'")

        return results

    def _create_translation_unit(
        self,
        buffer: SpeakerBuffer,
        text: str,
        boundary_type: str,
    ) -> TranslationUnit:
        """Create a TranslationUnit from buffer contents"""
        return TranslationUnit(
            text=text,
            speaker_name=buffer.speaker_name,
            start_time=buffer.buffer_start_time or 0.0,
            end_time=buffer.get_end_time() or 0.0,
            session_id=self.session_id,
            transcript_id=self.transcript_id,
            chunk_ids=[c.chunk_id for c in buffer.chunks],
            boundary_type=boundary_type,
            created_at=datetime.now(UTC),
        )
