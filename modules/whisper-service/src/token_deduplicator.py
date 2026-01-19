#!/usr/bin/env python3
"""
Token De-duplicator for Streaming Whisper

Prevents token repetition at chunk boundaries during streaming transcription.

Problem:
When Whisper processes audio in chunks with stateful decoding (KV cache),
tokens at chunk boundaries can be repeated or incomplete:
- Chunk 1 ends: [..., token_a, token_b]
- Chunk 2 starts: [token_b, token_c, ...]
  Result: "...word_a word_b word_b word_c..." (duplication!)

Or worse, incomplete UTF-8:
- Chunk 1 ends: [..., 院, 子, incomplete_token]
- Chunk 2 starts: [incomplete_token, 门, ...]
  Result: "...院子�门..." (� artifact!)

Solution:
Track recent tokens and deduplicate overlaps at chunk boundaries.

Reference: Common pattern in streaming ASR systems
"""

import logging

logger = logging.getLogger(__name__)


class TokenDeduplicator:
    """
    Tracks recent tokens to prevent duplication at streaming chunk boundaries.

    Usage:
        deduper = TokenDeduplicator(lookback_tokens=10)

        # Process chunk 1
        tokens1 = [1, 2, 3, 4, 5]
        clean_tokens1 = deduper.deduplicate(tokens1)  # Returns [1, 2, 3, 4, 5]

        # Process chunk 2 (with overlap)
        tokens2 = [4, 5, 6, 7, 8]  # Overlaps with chunk 1
        clean_tokens2 = deduper.deduplicate(tokens2)  # Returns [6, 7, 8]

    Parameters:
        lookback_tokens (int): How many tokens to track from previous chunk
                              Default 10 (handles up to ~5 word overlap)
    """

    def __init__(self, lookback_tokens: int = 10):
        self.lookback_tokens = lookback_tokens
        self.previous_tokens: list[int] = []

        logger.info(f"TokenDeduplicator initialized (lookback={lookback_tokens} tokens)")

    def deduplicate(self, new_tokens: list[int]) -> list[int]:
        """
        Deduplicate tokens by removing overlap with previous chunk.

        Algorithm:
        1. If no previous tokens, return new tokens as-is (first chunk)
        2. Check for overlap: Does new_tokens START with any suffix of previous_tokens?
        3. Find longest overlap (e.g., if previous ends [a,b,c] and new starts [b,c,d],
           overlap is [b,c], length 2)
        4. Remove overlap from new_tokens (return [d])
        5. Store last N tokens from new_tokens for next iteration

        Args:
            new_tokens: Token IDs from current chunk

        Returns:
            Deduplicated token IDs (with overlap removed)
        """
        if not new_tokens:
            logger.debug("[DEDUP] Empty token list, returning as-is")
            return new_tokens

        if not self.previous_tokens:
            # First chunk - no deduplication needed
            logger.debug(f"[DEDUP] First chunk: {len(new_tokens)} tokens, no dedup needed")
            self._update_previous(new_tokens)
            return new_tokens

        # Find overlap between end of previous_tokens and start of new_tokens
        overlap_length = self._find_overlap_length(self.previous_tokens, new_tokens)

        if overlap_length > 0:
            logger.info(f"[DEDUP] Found {overlap_length} token overlap at chunk boundary")
            logger.debug(f"[DEDUP] Previous suffix: {self.previous_tokens[-overlap_length:]}")
            logger.debug(f"[DEDUP] New prefix:      {new_tokens[:overlap_length]}")
            logger.debug(f"[DEDUP] Removing {overlap_length} tokens from new chunk")

            # Remove overlapping prefix from new tokens
            deduplicated = new_tokens[overlap_length:]

            logger.info(
                f"[DEDUP] Deduplicated: {len(new_tokens)} → {len(deduplicated)} tokens "
                f"({overlap_length} removed)"
            )
        else:
            logger.debug(f"[DEDUP] No overlap detected ({len(new_tokens)} tokens)")
            deduplicated = new_tokens

        # Update previous tokens for next iteration
        self._update_previous(new_tokens)

        return deduplicated

    def _find_overlap_length(self, prev: list[int], new: list[int]) -> int:
        """
        Find longest overlap where prev's suffix matches new's prefix.

        Example:
            prev = [1, 2, 3, 4, 5]
            new  = [4, 5, 6, 7]
            overlap = 2 (prev ends with [4,5], new starts with [4,5])

        Args:
            prev: Previous token sequence
            new: New token sequence

        Returns:
            Length of overlap (0 if no overlap)
        """
        max_overlap = min(len(prev), len(new), self.lookback_tokens)

        # Check from longest possible overlap down to 1
        for overlap_len in range(max_overlap, 0, -1):
            prev_suffix = prev[-overlap_len:]
            new_prefix = new[:overlap_len]

            if prev_suffix == new_prefix:
                logger.debug(
                    f"[DEDUP] Overlap found: {overlap_len} tokens "
                    f"(prev suffix = new prefix = {prev_suffix})"
                )
                return overlap_len

        return 0

    def _update_previous(self, tokens: list[int]):
        """
        Update stored previous tokens with last N tokens from current chunk.

        Args:
            tokens: Current chunk's tokens
        """
        if len(tokens) > self.lookback_tokens:
            self.previous_tokens = tokens[-self.lookback_tokens :]
        else:
            self.previous_tokens = tokens.copy()

        logger.debug(f"[DEDUP] Updated previous tokens: {len(self.previous_tokens)} tokens stored")

    def reset(self):
        """
        Reset deduplicator state.

        Call this at:
        - Segment boundaries (e.g., after VAD detected silence)
        - Session resets
        - Language changes (if SOT reset occurred)
        """
        logger.info("[DEDUP] Resetting deduplicator state")
        self.previous_tokens = []

    def get_state(self) -> dict:
        """
        Get current deduplicator state (for debugging).

        Returns:
            Dictionary with current state
        """
        return {
            "lookback_tokens": self.lookback_tokens,
            "previous_tokens_count": len(self.previous_tokens),
            "previous_tokens": self.previous_tokens,
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 80)
    print("TokenDeduplicator - Example Usage")
    print("=" * 80)

    deduper = TokenDeduplicator(lookback_tokens=10)

    # Scenario 1: Exact overlap (2 tokens)
    print("\n--- Scenario 1: 2-token overlap ---")
    chunk1 = [1, 2, 3, 4, 5]
    chunk2 = [4, 5, 6, 7, 8]

    print(f"Chunk 1: {chunk1}")
    clean1 = deduper.deduplicate(chunk1)
    print(f"Clean 1: {clean1}\n")

    print(f"Chunk 2: {chunk2}")
    clean2 = deduper.deduplicate(chunk2)
    print(f"Clean 2: {clean2} (removed {len(chunk2) - len(clean2)} overlapping tokens)")

    # Scenario 2: No overlap
    print("\n--- Scenario 2: No overlap ---")
    deduper.reset()
    chunk3 = [10, 11, 12]
    chunk4 = [20, 21, 22]

    print(f"Chunk 3: {chunk3}")
    clean3 = deduper.deduplicate(chunk3)
    print(f"Clean 3: {clean3}\n")

    print(f"Chunk 4: {chunk4}")
    clean4 = deduper.deduplicate(chunk4)
    print(f"Clean 4: {clean4} (no overlap)")

    # Scenario 3: Large overlap (5 tokens)
    print("\n--- Scenario 3: 5-token overlap ---")
    deduper.reset()
    chunk5 = [100, 101, 102, 103, 104, 105, 106]
    chunk6 = [102, 103, 104, 105, 106, 107, 108, 109]

    print(f"Chunk 5: {chunk5}")
    clean5 = deduper.deduplicate(chunk5)
    print(f"Clean 5: {clean5}\n")

    print(f"Chunk 6: {chunk6}")
    clean6 = deduper.deduplicate(chunk6)
    print(f"Clean 6: {clean6} (removed {len(chunk6) - len(clean6)} overlapping tokens)")

    print("\n" + "=" * 80)
    print("✅ TokenDeduplicator examples complete")
    print("=" * 80)
