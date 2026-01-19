#!/usr/bin/env python3
"""
Stability Tracking System for Incremental Streaming Transcription

Based on SimulStreaming's incremental update pattern - tracks which tokens
are "stable" (confirmed, won't change) vs "unstable" (might still change).

This enables:
- Draft/final emission protocol (emit only when stable prefix grows)
- Translation optimization (only translate stable tokens)
- UI visualization (black=stable, grey=unstable)

Key Concept from SimulStreaming:
- stable_prefix: Tokens confident won't change → send to MT
- unstable_tail: Tokens that might change → hold back

Reference: modules/whisper-service/buffer_plan.md Phase 2 (lines 69-257)
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """
    Configuration for stability detection.

    Tunable based on use case:
    - Low Latency (news, sports): threshold=0.75, min_hold=0.1s, max_latency=1.0s
    - High Quality (medical, legal): threshold=0.95, min_hold=0.5s, max_latency=3.0s
    - Balanced (conversation): threshold=0.85, min_hold=0.3s, max_latency=2.0s
    """

    stability_threshold: float = 0.85  # Min confidence for stable token
    min_stable_words: int = 2  # Min words before emitting
    min_hold_time: float = 0.3  # Min time to observe token (seconds)
    max_latency: float = 2.0  # Max delay before forcing emit
    word_boundary_bonus: float = 0.1  # Confidence boost at word boundaries


@dataclass
class TokenState:
    """
    State of a single token for stability tracking.

    Tracks:
    - Token identity (ID and text)
    - Confidence metrics (logprob, update count, age)
    - Temporal information (first seen, last updated)
    - Linguistic info (word boundary detection)
    """

    token_id: int
    text: str
    logprob: float
    first_seen: float  # Timestamp when first generated
    last_updated: float  # Timestamp of last appearance
    update_count: int  # How many times it's been consistent
    confidence: float  # Current confidence score (0.0-1.0)
    is_word_boundary: bool = False  # True if ends a word


class StabilityTracker:
    """
    Tracks token stability for incremental MT updates.

    Pattern from SimulStreaming:
    - Maintains stable_prefix (confirmed tokens) and unstable_tail (uncertain)
    - Detects when tokens become stable based on consistency + confidence
    - Provides draft/final emission decisions

    Display Visualization:
    - Black text = stable_prefix (confirmed, sent to MT)
    - Grey text = unstable_tail (might change, held back)

    Usage:
        tracker = StabilityTracker(config, tokenizer)

        # Each inference cycle
        new_stable, should_force = tracker.update(tokens, logprobs, time.time())

        if new_stable:
            # DRAFT: Emit new stable tokens
            yield draft_result(new_stable)

        if should_force:
            # FINAL: Force emission due to max latency
            yield final_result(tracker.finalize_segment())
    """

    def __init__(self, config: StabilityConfig, tokenizer: Any):
        """
        Initialize stability tracker.

        Args:
            config: Stability detection parameters
            tokenizer: Whisper tokenizer for text decoding
        """
        self.config = config
        self.tokenizer = tokenizer

        # Token tracking
        self.stable_prefix: list[TokenState] = []
        self.unstable_tail: list[TokenState] = []

        # Stability detection state
        self.last_emit_time = time.time()

        logger.info(
            f"[StabilityTracker] Initialized with threshold={config.stability_threshold}, "
            f"min_hold={config.min_hold_time}s, max_latency={config.max_latency}s"
        )

    def update(
        self,
        new_tokens: list[int],
        logprobs: list[float] | None = None,
        timestamp: float | None = None,
    ) -> tuple[list[TokenState], bool]:
        """
        Update with new tokens from decoder.

        Args:
            new_tokens: Token IDs from latest inference
            logprobs: Log probabilities for each token (optional)
            timestamp: Current timestamp (defaults to time.time())

        Returns:
            (new_stable_tokens, is_forced_emit)
            - new_stable_tokens: Tokens that just became stable
            - is_forced_emit: True if max_latency exceeded
        """
        if timestamp is None:
            timestamp = time.time()

        if logprobs is None:
            # Default to high confidence if no logprobs provided
            logprobs = [0.0] * len(new_tokens)

        # Align new tokens with existing unstable tail
        aligned = self._align_tokens(new_tokens, logprobs, timestamp)

        # Detect stable prefix boundary
        stable_idx = self._find_stable_boundary(aligned, timestamp)

        # Extract new stable tokens
        new_stable = aligned[:stable_idx]

        # Update state
        if new_stable:
            self.stable_prefix.extend(new_stable)
            self.unstable_tail = aligned[stable_idx:]
            self.last_emit_time = timestamp

            logger.info(
                f"[StabilityTracker] 📝 {len(new_stable)} new stable tokens: "
                f"'{self._decode_tokens(new_stable)}' "
                f"(stable_prefix={len(self.stable_prefix)}, unstable_tail={len(self.unstable_tail)})"
            )
        else:
            self.unstable_tail = aligned

        # Check if we should force emit due to max latency
        should_force = self._should_force_emit(timestamp)

        if should_force:
            logger.info(
                f"[StabilityTracker] ⏰ Force emit triggered (latency={timestamp - self.last_emit_time:.2f}s > {self.config.max_latency}s)"
            )

        return new_stable, should_force

    def _align_tokens(
        self, new_tokens: list[int], logprobs: list[float], timestamp: float
    ) -> list[TokenState]:
        """
        Align new tokens with existing unstable tail.

        Strategy:
        - Find longest common prefix with existing unstable tokens
        - Increment update_count for matched tokens (consistency tracking)
        - Create new TokenState for unmatched tokens

        This is key to stability detection: tokens that appear consistently
        across multiple inference cycles get higher update_count → higher confidence.
        """
        matched = 0

        # Find matching prefix
        for i, (old, new_tok) in enumerate(zip(self.unstable_tail, new_tokens, strict=False)):
            if old.token_id == new_tok:
                matched = i + 1
                # Token matched - increment consistency counter
                old.update_count += 1
                old.last_updated = timestamp
                old.confidence = self._compute_confidence(old, logprobs[i])

                logger.debug(
                    f"[StabilityTracker] Token {i} matched: '{old.text}' "
                    f"(update_count={old.update_count}, confidence={old.confidence:.3f})"
                )
            else:
                # Mismatch - stop here
                break

        # Keep matched tokens
        result = self.unstable_tail[:matched]

        # Create new TokenState for unmatched tokens
        new_states = [
            TokenState(
                token_id=tok,
                text=self.tokenizer.decode([tok]) if self.tokenizer else f"<{tok}>",
                logprob=lp,
                first_seen=timestamp,
                last_updated=timestamp,
                update_count=1,
                confidence=math.exp(lp) if lp < 0 else lp,
                is_word_boundary=self._is_word_boundary(tok),
            )
            for tok, lp in zip(new_tokens[matched:], logprobs[matched:], strict=False)
        ]

        if new_states:
            logger.debug(
                f"[StabilityTracker] {len(new_states)} new tokens: "
                f"'{self._decode_tokens(new_states)}'"
            )

        return result + new_states

    def _compute_confidence(self, token: TokenState, new_logprob: float) -> float:
        """
        Compute confidence based on:
        - Current logprob (model confidence)
        - Update count (consistency across inferences)
        - Time observed (age bonus)

        Formula: confidence = base_conf + consistency_bonus + time_bonus
        """
        base_conf = math.exp(new_logprob) if new_logprob < 0 else new_logprob

        # Consistency bonus: more appearances = more confident
        consistency_bonus = min(token.update_count * 0.05, 0.2)

        # Time bonus: older tokens = more stable
        age = time.time() - token.first_seen
        time_bonus = min(age * 0.1, 0.1)

        return min(base_conf + consistency_bonus + time_bonus, 1.0)

    def _find_stable_boundary(self, tokens: list[TokenState], timestamp: float) -> int:
        """
        Find index where tokens transition from stable to unstable.

        Stability Criteria (ALL must be met):
        1. Seen at least 3 times (update_count >= 3)
        2. High confidence (>= stability_threshold)
        3. Observed long enough (>= min_hold_time)
        4. Bonus: word boundaries get stability boost

        Returns:
            Index of first unstable token (everything before is stable)
        """
        for i, token in enumerate(tokens):
            age = timestamp - token.first_seen

            # Check stability criteria
            if token.update_count < 3:
                logger.debug(
                    f"[StabilityTracker] Token {i} unstable: update_count={token.update_count} < 3"
                )
                return i

            if token.confidence < self.config.stability_threshold:
                logger.debug(
                    f"[StabilityTracker] Token {i} unstable: confidence={token.confidence:.3f} < {self.config.stability_threshold}"
                )
                return i

            if age < self.config.min_hold_time:
                logger.debug(
                    f"[StabilityTracker] Token {i} unstable: age={age:.3f}s < {self.config.min_hold_time}s"
                )
                return i

            # Bonus: word boundaries are natural stable points
            if token.is_word_boundary:
                # Check if next few tokens are also stable
                if self._check_word_stability(tokens[i + 1 : i + 4], timestamp):
                    logger.debug(
                        f"[StabilityTracker] Word boundary detected at token {i}: '{token.text}'"
                    )
                    return i + 1

        # All tokens are stable
        return len(tokens)

    def _check_word_stability(self, tokens: list[TokenState], timestamp: float) -> bool:
        """
        Check if a word (sequence of tokens) is stable.

        Slightly relaxed criteria for lookahead tokens after word boundary.
        """
        if not tokens:
            return True

        return all(
            t.update_count >= 2 and t.confidence >= self.config.stability_threshold * 0.9
            for t in tokens[:3]
        )

    def _is_word_boundary(self, token_id: int) -> bool:
        """
        Check if token represents word boundary.

        Heuristics:
        - Token starts with space (Whisper tokens like " hello")
        - Token ends with punctuation
        """
        if not self.tokenizer:
            return False

        text = self.tokenizer.decode([token_id])

        # Starts with space = word boundary
        if text.startswith(" "):
            return True

        # Ends with punctuation = sentence boundary
        return bool(text.rstrip() != text or text.endswith((".", "!", "?", ",", ";", ":")))

    def _should_force_emit(self, timestamp: float) -> bool:
        """
        Check if we should force emit due to max latency.

        Prevents unbounded delays when tokens don't stabilize quickly.
        """
        elapsed = timestamp - self.last_emit_time
        return elapsed >= self.config.max_latency

    def _decode_tokens(self, tokens: list[TokenState]) -> str:
        """Decode list of TokenState to text."""
        if not tokens:
            return ""
        return "".join(t.text for t in tokens)

    def get_stable_text(self) -> str:
        """Get text from stable prefix (black in UI)."""
        return self._decode_tokens(self.stable_prefix)

    def get_unstable_text(self) -> str:
        """Get text from unstable tail (grey in UI)."""
        return self._decode_tokens(self.unstable_tail)

    def get_full_text(self) -> str:
        """Get full text (stable + unstable)."""
        return self.get_stable_text() + self.get_unstable_text()

    def finalize_segment(self) -> list[TokenState]:
        """
        Mark all tokens as stable at segment boundary.

        Called when segment is complete (silence, punctuation, max length).
        Returns all tokens and resets state for next segment.
        """
        all_stable = self.stable_prefix + self.unstable_tail

        logger.info(
            f"[StabilityTracker] 🏁 Finalizing segment: {len(all_stable)} tokens "
            f"('{self._decode_tokens(all_stable)}')"
        )

        # Reset state for next segment
        self.stable_prefix = []
        self.unstable_tail = []
        self.last_emit_time = time.time()

        return all_stable

    def reset(self):
        """Reset tracker state (for new session)."""
        self.stable_prefix = []
        self.unstable_tail = []
        self.last_emit_time = time.time()
        logger.info("[StabilityTracker] ♻️  State reset")

    def get_stats(self) -> dict[str, Any]:
        """Get current tracker statistics."""
        return {
            "stable_count": len(self.stable_prefix),
            "unstable_count": len(self.unstable_tail),
            "stable_text": self.get_stable_text(),
            "unstable_text": self.get_unstable_text(),
            "time_since_emit": time.time() - self.last_emit_time,
            "avg_stable_confidence": sum(t.confidence for t in self.stable_prefix)
            / len(self.stable_prefix)
            if self.stable_prefix
            else 0.0,
            "avg_unstable_confidence": sum(t.confidence for t in self.unstable_tail)
            / len(self.unstable_tail)
            if self.unstable_tail
            else 0.0,
        }
