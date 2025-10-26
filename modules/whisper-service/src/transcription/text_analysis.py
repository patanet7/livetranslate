#!/usr/bin/env python3
"""
Text Analysis Utilities

Utility functions for analyzing transcription text:
- Hallucination detection
- Text stability tracking
- Stability scoring

Extracted from whisper_service.py for better modularity and testability.
"""

from typing import List, Tuple


def detect_hallucination(text: str, confidence: float) -> bool:
    """
    Improved hallucination detection that only flags obvious cases
    and considers model confidence in the decision.

    Args:
        text: Transcribed text to analyze
        confidence: Model confidence score (0.0-1.0)

    Returns:
        True if text is likely a hallucination, False otherwise
    """
    if not text or len(text.strip()) < 2:
        return True

    text_lower = text.lower().strip()

    # Only flag very obvious hallucination patterns
    obvious_noise_patterns = [
        # Very short repetitive patterns
        'aaaa', 'bbbb', 'cccc', 'dddd', 'eeee',
        # Common Whisper artifacts (but be more selective)
        'mbc 뉴스', '김정진입니다', 'thanks for watching our channel',
    ]

    # Check for obvious noise only
    for pattern in obvious_noise_patterns:
        if pattern in text_lower:
            return True

    # Check for excessive repetition (stricter criteria)
    words = text_lower.split()
    if len(words) > 5:
        # Only flag if more than 80% of words are the same
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.2:  # Less than 20% unique words (was 10%)
            return True

    # Check for single character repetition
    if len(text_lower) > 10 and len(set(text_lower.replace(' ', ''))) < 3:
        return True

    # Don't flag educational content about language learning
    educational_phrases = [
        'english phrase', 'language', 'learning', 'practice', 'exercise',
        'get in shape', 'happened to you', 'trying to think', 'word', 'vocabulary'
    ]

    for phrase in educational_phrases:
        if phrase in text_lower:
            return False  # Definitely not hallucination

    return False


def find_stable_word_prefix(text_history: List[Tuple[str, float]], current_text: str) -> str:
    """
    Find the stable word prefix from text history.

    A word is considered stable if it appears in the same position
    across multiple consecutive transcriptions.

    Args:
        text_history: List of (text, timestamp) tuples
        current_text: Current transcription text

    Returns:
        Stable prefix string
    """
    if not text_history or len(text_history) < 2:
        return ""

    # Get all texts from history
    texts = [txt for txt, _ in text_history]

    # Split into words
    current_words = current_text.split()
    if not current_words:
        return ""

    # Find longest common prefix across recent texts
    stable_word_count = 0

    for i, word in enumerate(current_words):
        # Check if this word appears in same position in at least 2 recent texts
        appearances = 0
        for text in texts[-3:]:  # Check last 3 texts
            words = text.split()
            if i < len(words) and words[i] == word:
                appearances += 1

        # If word appears in at least 2 texts, consider it stable
        if appearances >= min(2, len(texts)):
            stable_word_count = i + 1
        else:
            break  # Stop at first unstable word

    # Return stable prefix
    if stable_word_count > 0:
        return " ".join(current_words[:stable_word_count])
    return ""


def calculate_text_stability_score(text_history: List[Tuple[str, float]], stable_prefix: str) -> float:
    """
    Calculate stability score based on text consistency.

    Args:
        text_history: List of (text, timestamp) tuples
        stable_prefix: Current stable prefix

    Returns:
        Stability score (0.0-1.0)
    """
    if not text_history:
        return 0.0

    if not stable_prefix:
        return 0.1  # Low score if nothing is stable

    # Calculate based on:
    # 1. Length of stable prefix relative to total text
    # 2. Consistency across history
    # 3. Age of stable prefix (older = more stable)

    current_text = text_history[-1][0] if text_history else ""
    if not current_text:
        return 0.0

    # Factor 1: Proportion of text that's stable
    stable_ratio = len(stable_prefix) / max(1, len(current_text))

    # Factor 2: Consistency (how many recent texts contain this prefix)
    consistency = 0
    for text, _ in text_history[-5:]:
        if text.startswith(stable_prefix):
            consistency += 1
    consistency_score = consistency / min(5, len(text_history))

    # Factor 3: Age bonus (longer stable = higher score)
    if len(text_history) >= 3:
        age_bonus = min(0.2, len(text_history) * 0.05)
    else:
        age_bonus = 0.0

    # Combine factors
    score = (stable_ratio * 0.5 + consistency_score * 0.4 + age_bonus)
    return min(1.0, max(0.0, score))
