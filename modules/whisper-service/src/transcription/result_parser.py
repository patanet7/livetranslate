#!/usr/bin/env python3
"""
Whisper Result Parser

Parses Whisper model results from different formats (PyTorch dict, OpenVINO, etc.)
and extracts text, segments, language, and confidence scores.

Extracted from whisper_service.py for better modularity and testability.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_whisper_result(result: Any) -> tuple[str, list, str, float]:
    """
    Parse Whisper model result and extract text, segments, language, and confidence.

    Handles multiple result formats:
    - PyTorch Whisper: dict with 'text', 'segments', 'language'
    - OpenVINO: object with 'texts', 'chunks' attributes
    - Fallback: object with 'text' attribute or string conversion

    Args:
        result: Raw result from Whisper model (dict or object)

    Returns:
        Tuple of (text, segments, language, confidence_score)
    """
    text = ""
    segments = []
    language = "unknown"
    confidence_score = 0.8  # Default for successful transcription

    # CRITICAL FIX: Check for dict FIRST before checking hasattr
    if isinstance(result, dict):
        # PyTorch Whisper returns dict with 'text' and 'segments' keys
        text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "unknown")
        logger.info(
            f"[WHISPER] 📝 Dict result - text: '{text[:60]}', segments: {len(segments)}, lang: {language}"
        )

        # Extract confidence from segments
        if segments:
            avg_logprobs = [
                seg.get("avg_logprob", -1.0) for seg in segments if "avg_logprob" in seg
            ]
            if avg_logprobs:
                avg_logprob = sum(avg_logprobs) / len(avg_logprobs)
                confidence_score = min(1.0, max(0.0, (avg_logprob + 1.0)))
                logger.info(
                    f"[WHISPER] 🎯 Calculated confidence from {len(avg_logprobs)} segments: {confidence_score:.3f}"
                )

    # Handle OpenVINO WhisperDecodedResults structure
    elif hasattr(result, "texts") and result.texts:
        # OpenVINO returns 'texts' (plural) - get the first text
        text = result.texts[0] if result.texts else ""
        logger.info(f"[WHISPER] 📝 Text extracted from 'texts': '{text}'")

        # Try to get segments from chunks and extract confidence
        chunk_confidences = []

        if hasattr(result, "chunks") and result.chunks:
            segments = result.chunks
            logger.info(f"[WHISPER] 📋 Chunks/segments count: {len(segments)}")

            # Extract confidence from all chunks
            for i, chunk in enumerate(segments):
                chunk_confidence = _extract_chunk_confidence(chunk)

                if chunk_confidence is not None:
                    # Ensure confidence is in valid range
                    chunk_confidence = max(0.0, min(1.0, chunk_confidence))
                    chunk_confidences.append(chunk_confidence)
                    if i == 0:  # Log first chunk for debugging
                        logger.info(f"[WHISPER] 🎯 Chunk {i} confidence: {chunk_confidence:.3f}")

        # Calculate overall confidence
        if chunk_confidences:
            # Use weighted average of chunk confidences
            confidence_score = sum(chunk_confidences) / len(chunk_confidences)
            logger.info(
                f"[WHISPER] 🎯 Calculated confidence from {len(chunk_confidences)} chunks: {confidence_score:.3f}"
            )
        else:
            # Try to extract overall confidence from result object
            confidence_score = _extract_result_confidence(result, confidence_score)

    elif hasattr(result, "text"):
        # Fallback to 'text' attribute
        text = result.text
        segments = getattr(result, "segments", [])
        logger.info(f"[WHISPER] 📝 Text extracted from 'text': '{text}'")
        logger.info(f"[WHISPER] 📋 Segments count: {len(segments)}")

    else:
        # Last resort: string conversion
        text = str(result)
        segments = []
        logger.info(f"[WHISPER] ⚠️ Using string conversion: '{text}'")

    # Enhanced language detection
    language = _detect_language(result, text, language)

    # Ensure confidence is in valid range
    confidence_score = max(0.0, min(1.0, confidence_score))

    return text, segments, language, confidence_score


def _extract_chunk_confidence(chunk: Any) -> float:
    """
    Extract confidence score from a single chunk/segment.

    Tries multiple attributes in order:
    - confidence, score, probability, prob
    - avg_logprob (converted from log space)
    - no_speech_prob (inverted)

    Args:
        chunk: Chunk/segment object

    Returns:
        Confidence score (0.0-1.0) or None if not found
    """
    # Try different confidence attributes
    if hasattr(chunk, "confidence"):
        return chunk.confidence
    elif hasattr(chunk, "score"):
        return chunk.score
    elif hasattr(chunk, "probability"):
        return chunk.probability
    elif hasattr(chunk, "prob"):
        return chunk.prob
    elif hasattr(chunk, "avg_logprob"):
        # Convert log probability to confidence (0-1 range)
        # avg_logprob is typically negative, closer to 0 is better
        return min(1.0, max(0.0, (chunk.avg_logprob + 1.0)))
    elif hasattr(chunk, "no_speech_prob"):
        # Convert no-speech probability to confidence
        return 1.0 - chunk.no_speech_prob

    return None


def _extract_result_confidence(result: Any, default: float = 0.8) -> float:
    """
    Extract confidence score from overall result object.

    Args:
        result: Result object
        default: Default confidence if none found

    Returns:
        Confidence score (0.0-1.0)
    """
    if hasattr(result, "confidence"):
        confidence = result.confidence
        logger.info(f"[WHISPER] 🎯 Found result confidence: {confidence:.3f}")
        return confidence
    elif hasattr(result, "avg_logprob"):
        # Convert log probability to confidence
        confidence = min(1.0, max(0.0, (result.avg_logprob + 1.0)))
        logger.info(f"[WHISPER] 🎯 Calculated confidence from result avg_logprob: {confidence:.3f}")
        return confidence
    elif hasattr(result, "no_speech_prob"):
        confidence = 1.0 - result.no_speech_prob
        logger.info(f"[WHISPER] 🎯 Calculated confidence from no_speech_prob: {confidence:.3f}")
        return confidence
    elif hasattr(result, "scores") and result.scores:
        try:
            # Get average score
            avg_score = sum(result.scores) / len(result.scores)
            confidence = max(0.0, min(1.0, avg_score))
            logger.info(f"[WHISPER] 🎯 Average result score: {confidence:.3f}")
            return confidence
        except Exception:
            logger.info("[WHISPER] ⚠️ Failed to calculate average score - using default")
            return default
    else:
        logger.info(f"[WHISPER] ⚠️ No confidence attributes found - using default: {default:.3f}")
        return default


def _detect_language(result: Any, text: str, current_language: str = "unknown") -> str:
    """
    Detect language from result object or text content.

    Tries multiple methods:
    1. Check result.language or result.lang attributes
    2. Check first chunk for language info
    3. Simple character-based detection from text

    Args:
        result: Result object
        text: Transcribed text
        current_language: Current language value (may be from dict result)

    Returns:
        Detected language code
    """
    language = current_language

    # Only detect if not already set
    if language != "unknown":
        return language

    # Method 1: Check result attributes (for non-dict results)
    if hasattr(result, "language"):
        language = result.language
        logger.info(f"[WHISPER] 🌍 Found language attribute: {language}")
        return language
    elif hasattr(result, "lang"):
        language = result.lang
        logger.info(f"[WHISPER] 🌍 Found lang attribute: {language}")
        return language

    # Method 2: Check chunks for language info
    if hasattr(result, "chunks") and result.chunks:
        try:
            first_chunk = result.chunks[0]
            if hasattr(first_chunk, "language"):
                language = first_chunk.language
                logger.info(f"[WHISPER] 🌍 Found language in chunk: {language}")
                return language
            elif hasattr(first_chunk, "lang"):
                language = first_chunk.lang
                logger.info(f"[WHISPER] 🌍 Found lang in chunk: {language}")
                return language
        except Exception as e:
            logger.debug(f"[WHISPER] Could not extract language from chunks: {e}")

    # Method 3: Simple language detection from text content
    if text:
        # Detect Chinese characters
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            language = "zh"
            logger.info(f"[WHISPER] 🌍 Detected Chinese from text content: {language}")
            return language
        # Detect other common patterns
        elif any(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in text):
            language = "en"
            logger.info(f"[WHISPER] 🌍 Detected English from text content: {language}")
            return language
        else:
            language = "auto"
            logger.info(f"[WHISPER] 🌍 Auto-detected language: {language}")
            return language

    return "unknown"
