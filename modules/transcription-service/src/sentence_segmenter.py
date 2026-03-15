"""Language-universal sentence segmenter.

Splits transcription text on sentence-ending punctuation for both
Latin (. ! ?) and CJK (。！？) scripts. Additional scripts (Arabic,
Thai, Devanagari) can be added via SENTENCE_ENDINGS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Sentence-ending punctuation across scripts
SENTENCE_ENDINGS = re.compile(
    r"([.!?]"             # Latin
    r"|[。！？]"           # CJK fullwidth
    r"|[।॥]"              # Devanagari (future-proofing)
    r")"
)

# Terminal characters for is_sentence_end() compatibility
_TERMINALS = set(".!?\u3002\uff01\uff1f\u061f\u0964\u0965")

# Trailing quote/bracket chars to strip before terminal check
_TRAILING_CHARS = set('"\'\u201c\u201d\u2018\u2019`\u300d\u300f\uff09\u3011\u3015])\u007d\u203a\u00bb')


@dataclass
class SegmentResult:
    """Result of sentence segmentation."""
    sentences: list[str]    # Completed sentences (including trailing punctuation)
    remainder: str          # Incomplete tail (no sentence-ending punctuation yet)


class SentenceSegmenter:
    """Splits streaming text into sentences at punctuation boundaries.

    Designed for real-time transcription: accumulates text and emits
    completed sentences while holding back the incomplete remainder.
    """

    def segment(self, text: str) -> SegmentResult:
        """Segment text into completed sentences and a remainder.

        Args:
            text: Input text, possibly containing multiple sentences.

        Returns:
            SegmentResult with completed sentences and the trailing remainder.
        """
        sentences: list[str] = []
        remaining = text

        while remaining:
            match = SENTENCE_ENDINGS.search(remaining)
            if match is None:
                break

            # Include the punctuation mark in the sentence
            end_pos = match.end()
            sentence = remaining[:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining[end_pos:].lstrip()

        return SegmentResult(sentences=sentences, remainder=remaining)

    def is_sentence_end(self, text: str) -> bool:
        """Check if text ends with a sentence terminal.

        Handles trailing punctuation, quotes, and parentheses:
        - "Hello world." → True
        - 'Hello world."' → True (period inside quote)
        - "Hello world?)" → True (question mark inside paren)
        - 'Hello world"' → False (just quote, no terminal)

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence terminal
        """
        text = text.rstrip()
        if not text:
            return False

        # Strip trailing quotes, parentheses, brackets
        while text and text[-1] in _TRAILING_CHARS:
            text = text[:-1]

        if not text:
            return False

        return text[-1] in _TERMINALS
