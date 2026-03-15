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

from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()


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
        tokenizer: Any | None = None,
        device: Any | None = None,
        prefix_token_ids: list[int] | None = None,
    ):
        """
        Initialize TokenBuffer

        Args:
            text: Initial text content
            tokenizer: Tokenizer for converting text to token IDs (tiktoken or Whisper tokenizer)
            device: Device for tensor operations (e.g., 'cuda', 'cpu')
            prefix_token_ids: Optional prefix token IDs (e.g., <|sot_prev|> token)
        """
        self._text = text
        self.tokenizer = tokenizer
        self.device = device
        self.prefix_token_ids = prefix_token_ids or []

    @classmethod
    def empty(cls, *args, **kwargs) -> "TokenBuffer":
        """
        Create an empty TokenBuffer

        Args:
            *args, **kwargs: Arguments to pass to TokenBuffer constructor

        Returns:
            Empty TokenBuffer instance
        """
        return cls(*args, **kwargs)

    @classmethod
    def from_text(cls, text: str, *args, **kwargs) -> "TokenBuffer":
        """
        Create TokenBuffer from text

        Args:
            text: Initial text content
            *args, **kwargs: Additional arguments to pass to TokenBuffer constructor

        Returns:
            TokenBuffer instance with text content
        """
        return cls(*args, text=text, **kwargs)

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

    def as_token_ids(self) -> list[int]:
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

    def as_tensor_beam(self, beam_size: int, device: Any):
        """
        Convert text to token tensor repeated for beam search

        This method is called by SimulWhisper to prepare context tokens
        for beam search decoding. The context tokens are repeated beam_size
        times to match the beam search batch dimension.

        Args:
            beam_size: Number of beams in beam search
            device: Torch device for tensor (e.g., 'cuda', 'cpu', 'mps')

        Returns:
            Torch tensor of shape (beam_size, num_tokens) containing context token IDs

        Raises:
            ValueError: If no tokenizer is set

        Example:
            buffer = TokenBuffer.from_text("Hello world", tokenizer=tokenizer)
            tensor = buffer.as_tensor_beam(beam_size=5, device='cuda')
            # Result: tensor([[50258, 50259, ...], ...]) shape (5, num_tokens)
        """
        import torch

        # Get token IDs
        token_ids = self.as_token_ids()

        # Convert to tensor
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Repeat for beam search (beam_size copies)
        return token_tensor.repeat_interleave(beam_size, dim=0)

    def append_token_ids(self, token_ids: list[int]):
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

        CRITICAL FIX: Match SimulStreaming's tokenizer-based word splitting!
        Reference: token_buffer.py lines 47-62

        Following SimulStreaming behavior:
        - Uses tokenizer.split_to_word_tokens() to preserve tokenization boundaries
        - Removes oldest words first (FIFO)
        - Preserves static prefix (text before 'after' position)
        - Operates at Whisper token word boundaries (not Python str.split())

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
        if self.tokenizer is None:
            logger.warning("Tokenizer not set - cannot trim words properly")
            return 0

        if self.is_empty():
            return 0

        # Encode rolling context (after static prefix)
        # Reference: token_buffer.py line 55
        ids = self.tokenizer.encode(self._text[after:])

        # Split into word tokens using Whisper's tokenizer
        # Reference: token_buffer.py line 56
        words, wids = self.tokenizer.split_to_word_tokens(ids)

        if not words:
            return 0

        # Reconstruct text: static prefix + remaining words
        # Reference: token_buffer.py line 61
        self._text = self._text[:after] + "".join(words[num:])

        # Calculate tokens removed
        # Reference: token_buffer.py line 62
        return sum(len(wi) for wi in wids[:num])

    def __repr__(self) -> str:
        """String representation for debugging"""
        text_preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
        return f"TokenBuffer(text='{text_preview}', length={len(self._text)})"

    def __len__(self) -> int:
        """Get length of text content"""
        return len(self._text)
