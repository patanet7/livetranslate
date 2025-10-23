#!/usr/bin/env python3
"""
Sentence Segmenter - Multi-language sentence boundary detection

Based on SimulStreaming reference implementation with extended language support.
Reference: SimulStreaming/translate/sentence_segmenter.py

Supports:
- Latin scripts: English, Spanish, French, German, Italian, Portuguese, etc. (!?.)
- Japanese: 。！？
- Chinese: 。！？
- Arabic: ؟ (Arabic question mark)
- Indic scripts: । (Devanagari danda - Hindi, Sanskrit, Marathi, Nepali)
- Korean: Uses Latin period (.)
- And many more via Unicode properties

Returns: List of strings, where each string is a sentence.
Spaces following punctuation are preserved.
Total character count: output == input (lossless)
"""

import regex
from functools import lru_cache
from typing import List


class SentenceSegmenter:
    """
    Regex-based sentence splitter for 50+ languages.

    Based on sacrebleu TokenizerV14International(BaseTokenizer).

    Usage:
        segmenter = SentenceSegmenter()
        sentences = segmenter("Hello world. How are you?")
        # Result: ["Hello world. ", "How are you?"]

    Features:
    - Preserves whitespace after terminals
    - Handles multiple scripts (Latin, CJK, Arabic, Indic)
    - LRU cache for performance (16K entries)
    - Character-preserving (len(input) == sum(len(s) for s in output))
    """

    # Unique separator (won't appear in real text)
    sep = 'ŽžŽžSentenceSeparatorŽžŽž'

    # Sentence terminals by script family
    latin_terminals = '!?.'           # English, Romance, Germanic languages
    jap_zh_terminals = '。！？'        # Japanese, Chinese
    arabic_terminals = '؟'            # Arabic question mark (period already in latin)
    indic_terminals = '।'             # Devanagari danda (Hindi, Sanskrit, Marathi, Nepali)

    # Combined terminal set (covers 50+ languages)
    terminals = latin_terminals + jap_zh_terminals + arabic_terminals + indic_terminals

    def __init__(self):
        """
        Initialize sentence segmenter with regex patterns.

        Pattern logic:
        1. Split when terminal preceded by non-digit, append trailing whitespace
        2. Split when terminal followed by non-digit (BUT NOT QUOTES!)

        Uses Unicode properties:
        - \\P{N}: Non-digit (language-agnostic)
        - \\p{Z}: Whitespace/separator (any script)

        CRITICAL FIX: Don't split on terminal+quote patterns!
        - "Hello." She → Keep quotes with sentence
        - "Hello."She → Keep quotes with sentence
        """
        terminals = self.terminals

        # All quote characters (Latin, CJK, typographic)
        # Include: " ' " " ' ' « » 「 」 『 』 ‹ › etc.
        quotes = r'''"\'"\'`""''«»「」『』‹›〈〉《》【】〔〕（）()\[\]'''

        self._re = [
            # Rule 1: terminal + optional quotes + optional whitespace → separator
            # Handles: "Hello." → split, "Hello.\" " → split, "Hello.\"" → split
            # Pattern: (non-digit)(terminal)(quotes*)(whitespace*)
            (regex.compile(r'(\P{N})([' + terminals + r'])([' + quotes + r']*)(\p{Z}*)'),
             r'\1\2\3\4' + self.sep),

            # Rule 2: terminal + optional quotes + non-digit-non-quote → separator
            # Handles: "Hello."She → split (but not "Hello.\"")
            # Pattern: (terminal)(quotes*)(non-digit-non-quote-non-whitespace)
            # CRITICAL: Use negative lookahead to exclude quotes and whitespace
            (regex.compile(r'([' + terminals + r'])([' + quotes + r']*)([^\P{N}' + quotes + r'\p{Z}])'),
             r'\1\2' + self.sep + r'\3'),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line: str) -> List[str]:
        """
        Segment text into sentences.

        Args:
            line: Input text (any language)

        Returns:
            List of sentences with preserved whitespace

        Example:
            >>> segmenter = SentenceSegmenter()
            >>> segmenter("Hello world. How are you?")
            ["Hello world. ", "How are you?"]

            >>> segmenter("こんにちは。元気ですか？")
            ["こんにちは。", "元気ですか？"]
        """
        # Apply regex substitutions
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)

        # Split on separator and filter empty strings
        return [t for t in line.split(self.sep) if t != '']

    def is_sentence_end(self, text: str) -> bool:
        """
        Check if text ends with a sentence terminal.

        Handles trailing punctuation, quotes, and parentheses:
        - "Hello world." → True
        - "Hello world.\"" → True (period inside quote)
        - "Hello world?)" → True (question mark inside paren)
        - "Hello world\"" → False (just quote, no terminal)

        Args:
            text: Text to check

        Returns:
            True if text ends with sentence terminal

        Example:
            >>> segmenter.is_sentence_end("Hello world.")
            True
            >>> segmenter.is_sentence_end("Hello world")
            False
            >>> segmenter.is_sentence_end('He said "Hello world."')
            True
        """
        text = text.rstrip()  # Remove trailing whitespace
        if not text:
            return False

        # Strip trailing quotes, parentheses, brackets
        # Common patterns: ."  ?"  !)  .』 etc.
        trailing_chars = '"\'"\'`」』）】〕])}›»'
        while text and text[-1] in trailing_chars:
            text = text[:-1]

        if not text:
            return False

        return text[-1] in self.terminals

    def get_last_sentence(self, text: str) -> str:
        """
        Extract the last complete sentence from text.

        Useful for incremental processing - get the most recent
        complete sentence without splitting incomplete ones.

        Args:
            text: Input text (may contain multiple sentences)

        Returns:
            Last complete sentence, or empty string if none

        Example:
            >>> segmenter.get_last_sentence("First. Second. Third")
            "Second. "
        """
        sentences = self(text)
        if not sentences:
            return ""

        # If last sentence ends with terminal, return it
        if self.is_sentence_end(sentences[-1]):
            return sentences[-1]

        # Otherwise return second-to-last (if exists)
        if len(sentences) >= 2:
            return sentences[-2]

        return ""

    def count_complete_sentences(self, text: str) -> int:
        """
        Count complete sentences (ending with terminals).

        Args:
            text: Input text

        Returns:
            Number of complete sentences

        Example:
            >>> segmenter.count_complete_sentences("First. Second. Third")
            2  # "Third" is incomplete
        """
        sentences = self(text)
        return sum(1 for s in sentences if self.is_sentence_end(s))


def test_sentence_segmenter():
    """Test sentence segmenter with multiple languages."""
    segmenter = SentenceSegmenter()

    test_cases = [
        # English
        ("Hello world. How are you?", ["Hello world. ", "How are you?"]),
        ("First! Second? Third.", ["First! ", "Second? ", "Third."]),

        # Spanish
        ("¡Hola! ¿Cómo estás?", ["¡Hola! ", "¿Cómo estás?"]),

        # Japanese
        ("こんにちは。元気ですか？", ["こんにちは。", "元気ですか？"]),

        # Chinese
        ("你好。你好吗？", ["你好。", "你好吗？"]),

        # Mixed (incomplete sentence)
        ("Complete sentence. Incomplete", ["Complete sentence. ", "Incomplete"]),

        # Edge cases
        ("", []),
        ("No punctuation", ["No punctuation"]),
        ("Multiple   spaces.  After  period.", ["Multiple   spaces.  ", "After  period."]),
    ]

    print("Testing SentenceSegmenter:")
    print("=" * 80)

    all_passed = True
    for text, expected in test_cases:
        result = segmenter(text)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✅" if passed else "❌"
        print(f"\n{status} Input: '{text}'")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")

        # Test character preservation
        if text:
            char_count_input = len(text)
            char_count_output = sum(len(s) for s in result)
            char_preserved = char_count_input == char_count_output
            if not char_preserved:
                print(f"   ⚠️  Character count mismatch: {char_count_input} -> {char_count_output}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

    return all_passed


if __name__ == "__main__":
    test_sentence_segmenter()
