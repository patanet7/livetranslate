#!/usr/bin/env python3
"""
UTF-8 Boundary Fixer for Streaming Text

Fixes incomplete UTF-8 multi-byte characters at streaming chunk boundaries.

Problem:
When streaming text is split at arbitrary boundaries, multi-byte UTF-8
characters (like Chinese, Japanese, emoji) can be split, creating invalid
byte sequences that decode to � (U+FFFD replacement character).

Example:
- Chinese "处" = 3 bytes: E5 A4 84
- If chunk ends after E5 A4, the incomplete sequence → �
- Next chunk starts with 84... which is also invalid → �

Solution:
Remove � characters that appear at chunk boundaries (start/end of text),
as they represent incomplete multi-byte characters from the split.
"""

import logging

logger = logging.getLogger(__name__)

# Unicode replacement character (used when UTF-8 decoding fails)
REPLACEMENT_CHAR = "\ufffd"  # �


class UTF8BoundaryFixer:
    """
    Fixes incomplete UTF-8 characters at streaming text boundaries.

    Usage:
        fixer = UTF8BoundaryFixer()

        # Process chunk 1
        clean_text1 = fixer.fix_boundaries("院子门口不远�")  # Returns "院子门口不远"

        # Process chunk 2
        clean_text2 = fixer.fix_boundaries("�就是一个地铁站")  # Returns "就是一个地铁站"

    The fixer automatically removes � characters from:
    - Start of text (incomplete character from previous chunk)
    - End of text (incomplete character split at boundary)
    """

    def __init__(self):
        self.previous_had_trailing_replacement = False
        logger.info("UTF8BoundaryFixer initialized")

    def fix_boundaries(self, text: str) -> str:
        """
        Remove incomplete UTF-8 characters (�) from chunk boundaries.

        Args:
            text: Decoded text that may contain � at boundaries

        Returns:
            Clean text with boundary � characters removed
        """
        if not text:
            return text

        original_text = text
        removed_start = 0
        removed_end = 0

        # Remove � from START of text (incomplete from previous chunk)
        # Strip ALL leading � characters
        while text.startswith(REPLACEMENT_CHAR):
            text = text[1:]
            removed_start += 1

        # Remove � from END of text (incomplete in current chunk)
        # Strip ALL trailing � characters
        while text.endswith(REPLACEMENT_CHAR):
            text = text[:-1]
            removed_end += 1

        if removed_start > 0 or removed_end > 0:
            logger.info(
                f"[UTF8_FIX] Cleaned chunk boundaries: "
                f"removed {removed_start} leading + {removed_end} trailing � chars"
            )
            logger.debug(f"[UTF8_FIX] Before: '{original_text[:50]}...'")
            logger.debug(f"[UTF8_FIX] After:  '{text[:50]}...'")

        # Track if this chunk ended with � (for next chunk's context)
        self.previous_had_trailing_replacement = removed_end > 0

        return text

    def has_replacement_chars(self, text: str) -> bool:
        """
        Check if text contains any � characters.

        Args:
            text: Text to check

        Returns:
            True if text contains �, False otherwise
        """
        return REPLACEMENT_CHAR in text

    def count_replacement_chars(self, text: str) -> int:
        """
        Count number of � characters in text.

        Args:
            text: Text to check

        Returns:
            Number of � characters found
        """
        return text.count(REPLACEMENT_CHAR)

    def reset(self):
        """
        Reset state for new session.

        Call this when starting a new transcription session or after
        segment boundaries where context is reset.
        """
        logger.debug("[UTF8_FIX] Resetting boundary fixer state")
        self.previous_had_trailing_replacement = False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 80)
    print("UTF8BoundaryFixer - Example Usage")
    print("=" * 80)

    fixer = UTF8BoundaryFixer()

    # Scenario 1: Chinese text with boundary artifacts
    print("\n--- Scenario 1: Chinese boundary artifacts ---")
    chunk1 = "院子门口不远�"  # Ends with incomplete char
    chunk2 = "�就是一个地铁站"  # Starts with incomplete char

    print(f"Chunk 1 (original): '{chunk1}'")
    clean1 = fixer.fix_boundaries(chunk1)
    print(f"Chunk 1 (cleaned):  '{clean1}'")
    print()

    print(f"Chunk 2 (original): '{chunk2}'")
    clean2 = fixer.fix_boundaries(chunk2)
    print(f"Chunk 2 (cleaned):  '{clean2}'")
    print()

    print(f"Combined: '{clean1}{clean2}'")

    # Scenario 2: Multiple replacement chars
    print("\n--- Scenario 2: Multiple replacement chars ---")
    fixer.reset()

    text_with_multiple = "��Hello��World��"
    print(f"Original: '{text_with_multiple}'")
    cleaned = fixer.fix_boundaries(text_with_multiple)
    print(f"Cleaned:  '{cleaned}'")
    print(f"Removed: {len(text_with_multiple) - len(cleaned)} chars")

    # Scenario 3: No replacement chars
    print("\n--- Scenario 3: Clean text (no fixes needed) ---")
    fixer.reset()

    clean_text = "This is clean English text"
    print(f"Original: '{clean_text}'")
    result = fixer.fix_boundaries(clean_text)
    print(f"Result:   '{result}'")
    print(f"Changed:  {clean_text != result}")

    # Scenario 4: Detection
    print("\n--- Scenario 4: Detection methods ---")
    test_texts = [
        "Clean text",
        "Text with � in middle",
        "�Starts with replacement",
        "Ends with replacement�",
    ]

    for text in test_texts:
        has_repl = fixer.has_replacement_chars(text)
        count = fixer.count_replacement_chars(text)
        print(f"'{text[:30]}...'")
        print(f"  Has �: {has_repl}, Count: {count}")

    print("\n" + "=" * 80)
    print("✅ UTF8BoundaryFixer examples complete")
    print("=" * 80)
