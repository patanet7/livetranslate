#!/usr/bin/env python3
"""
TokenBuffer - Context Management for Rolling Context System

Following SimulStreaming reference implementation:
- token_buffer.py: Context management with static/rolling prompts
- simul_whisper/simul_whisper.py lines 151-195: Context initialization and trimming

This implements a two-tier context system:
1. Static prompt: Domain-specific terminology (never trimmed)
2. Rolling context: Recent transcriptions (FIFO word-level trimming)

Target: +25-40% quality improvement on long-form content
Reference: SimulStreaming/token_buffer.py
"""

from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TokenBuffer:
    """
    TokenBuffer manages context text with automatic tokenization

    Features:
    - Store text with optional tokenizer for token ID conversion
    - FIFO word-level trimming with static prefix preservation
    - Append completed segments to context
    - Convert between text and token IDs

    Usage:
        # Create empty buffer
        buffer = TokenBuffer.empty(tokenizer=tokenizer)

        # Create from text
        buffer = TokenBuffer.from_text("Medical terms: hypertension", tokenizer=tokenizer)

        # Append new segment
        new_tokens = tokenizer.encode(" diabetes mellitus")
        buffer.append_token_ids(new_tokens)

        # Trim old context (preserve static prefix)
        buffer.trim_words(num=2, after=len("Medical terms: "))
    """

    def __init__(
        self,
        text: str = "",
        tokenizer: Optional[Any] = None,
        prefix_token_ids: Optional[List[int]] = None
    ):
        """
        Initialize TokenBuffer

        Args:
            text: Initial text content
            tokenizer: Tokenizer for converting text to token IDs (tiktoken or Whisper tokenizer)
            prefix_token_ids: Optional prefix token IDs (e.g., <|sot_prev|> token)
        """
        self._text = text
        self.tokenizer = tokenizer
        self.prefix_token_ids = prefix_token_ids or []

    @classmethod
    def empty(cls, tokenizer: Optional[Any] = None) -> "TokenBuffer":
        """
        Create an empty TokenBuffer

        Args:
            tokenizer: Optional tokenizer for token ID conversion

        Returns:
            Empty TokenBuffer instance
        """
        return cls(text="", tokenizer=tokenizer)

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: Optional[Any] = None,
        prefix_token_ids: Optional[List[int]] = None
    ) -> "TokenBuffer":
        """
        Create TokenBuffer from text

        Args:
            text: Initial text content
            tokenizer: Optional tokenizer for token ID conversion
            prefix_token_ids: Optional prefix token IDs

        Returns:
            TokenBuffer instance with text content
        """
        return cls(text=text, tokenizer=tokenizer, prefix_token_ids=prefix_token_ids)

    def is_empty(self) -> bool:
        """
        Check if buffer is empty

        Returns:
            True if buffer has no text content
        """
        return len(self._text) == 0

    @property
    def text(self) -> str:
        """
        Get current text content

        Returns:
            Text content as string
        """
        return self._text

    @text.setter
    def text(self, value: str):
        """
        Set text content

        Args:
            value: New text content
        """
        self._text = value

    def as_text(self) -> str:
        """
        Get text content as string

        Returns:
            Text content
        """
        return self._text

    def as_token_ids(self) -> List[int]:
        """
        Convert text to token IDs using tokenizer

        Returns:
            List of token IDs (includes prefix tokens if set)

        Raises:
            ValueError: If no tokenizer is set
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set - cannot convert text to token IDs")

        # Encode text to token IDs
        text_token_ids = self.tokenizer.encode(self._text)

        # Prepend prefix tokens if set
        if self.prefix_token_ids:
            return self.prefix_token_ids + text_token_ids

        return text_token_ids

    def append_token_ids(self, token_ids: List[int]):
        """
        Append token IDs to buffer (decode and append as text)

        Args:
            token_ids: Token IDs to append

        Raises:
            ValueError: If no tokenizer is set
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set - cannot decode token IDs")

        # Decode token IDs to text
        new_text = self.tokenizer.decode(token_ids)

        # Append to existing text
        self._text += new_text

    def trim_words(self, num: int, after: int = 0) -> int:
        """
        Trim words from beginning (FIFO word-level trimming)

        Following SimulStreaming behavior:
        - Removes oldest words first (FIFO)
        - Preserves static prefix (text before 'after' position)
        - Operates at word boundaries

        Args:
            num: Number of words to trim from beginning
            after: Character position after which to trim (preserves static prefix)

        Returns:
            Number of tokens removed

        Example:
            buffer.text = "Medical terms: patient has symptoms"
            buffer.trim_words(num=1, after=len("Medical terms: "))
            # Result: "Medical terms: has symptoms"
        """
        if self.is_empty():
            return 0

        # Split into static prefix and rolling context
        static_prefix = self._text[:after]
        rolling_context = self._text[after:]

        # Calculate tokens before trimming
        tokens_before = 0
        if self.tokenizer is not None:
            try:
                tokens_before = len(self.tokenizer.encode(rolling_context))
            except Exception:
                # If tokenizer fails, just proceed with word trimming
                pass

        # Split rolling context into words
        words = rolling_context.split()

        # Trim requested number of words from beginning
        if num >= len(words):
            # Trim all words
            trimmed_words = []
        else:
            # Trim first 'num' words
            trimmed_words = words[num:]

        # Reconstruct text
        new_rolling_context = " ".join(trimmed_words)
        self._text = static_prefix + new_rolling_context

        # Calculate tokens removed
        tokens_after = 0
        if self.tokenizer is not None:
            try:
                tokens_after = len(self.tokenizer.encode(new_rolling_context))
            except Exception:
                # If tokenizer fails, estimate based on words removed
                return min(num, len(words))

        tokens_removed = tokens_before - tokens_after
        return max(0, tokens_removed)

    def __repr__(self) -> str:
        """String representation for debugging"""
        text_preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
        return f"TokenBuffer(text='{text_preview}', length={len(self._text)})"

    def __len__(self) -> int:
        """Get length of text content"""
        return len(self._text)
