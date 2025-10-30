#!/usr/bin/env python3
"""
Test Utilities - Reusable Metrics and Helpers

Provides WER/CER calculation and other test utilities for transcription testing.
Extracted from milestone1/test_baseline_transcription.py for reusability.
"""

import re
import string
import numpy as np
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for fair comparison:
    - Remove punctuation
    - Lowercase
    - Normalize whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for WER/CER comparison
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Normalize whitespace (multiple spaces to single space)
    text = re.sub(r'\s+', ' ', text)
    # Strip
    text = text.strip()
    return text


def calculate_wer_detailed(reference: str, hypothesis: str) -> Dict:
    """
    Calculate detailed Word Error Rate (WER) with alignment information

    WER = (S + D + I) / N
    where:
    - S = substitutions
    - D = deletions
    - I = insertions
    - N = number of words in reference

    Returns both raw and normalized WER with detailed error breakdown.

    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate

    Returns:
        Dict with:
        - raw: Raw WER metrics (with punctuation)
        - normalized: Normalized WER metrics (without punctuation)
        - ref_normalized: Normalized reference text
        - hyp_normalized: Normalized hypothesis text
    """
    # Calculate raw WER (with punctuation)
    ref_words_raw = reference.lower().split()
    hyp_words_raw = hypothesis.lower().split()

    # Calculate normalized WER (without punctuation)
    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)
    ref_words = ref_normalized.split()
    hyp_words = hyp_normalized.split()

    # Levenshtein distance with backtracking for alignment
    def levenshtein_with_alignment(ref_words: List[str], hyp_words: List[str]):
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

        # Initialize
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        # Fill matrix
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        # Backtrack to get alignment
        i, j = len(ref_words), len(hyp_words)
        substitutions = []
        deletions = []
        insertions = []

        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
                # Match
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                # Substitution
                substitutions.append((ref_words[i-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                # Insertion
                insertions.append(hyp_words[j-1])
                j -= 1
            else:
                # Deletion
                deletions.append(ref_words[i-1])
                i -= 1

        edit_distance = d[len(ref_words)][len(hyp_words)]
        wer = (edit_distance / len(ref_words) * 100) if len(ref_words) > 0 else 0

        return {
            'wer': wer,
            'edit_distance': int(edit_distance),
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'num_words': len(ref_words)
        }

    # Calculate both metrics
    raw_result = levenshtein_with_alignment(ref_words_raw, hyp_words_raw)
    normalized_result = levenshtein_with_alignment(ref_words, hyp_words)

    return {
        'raw': raw_result,
        'normalized': normalized_result,
        'ref_normalized': ref_normalized,
        'hyp_normalized': hyp_normalized
    }


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) - normalized

    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate

    Returns:
        CER as percentage (0-100)
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Levenshtein distance for characters
    d = np.zeros((len(ref_norm) + 1, len(hyp_norm) + 1), dtype=int)

    for i in range(len(ref_norm) + 1):
        d[i][0] = i
    for j in range(len(hyp_norm) + 1):
        d[0][j] = j

    for i in range(1, len(ref_norm) + 1):
        for j in range(1, len(hyp_norm) + 1):
            if ref_norm[i-1] == hyp_norm[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    cer = d[len(ref_norm)][len(hyp_norm)] / len(ref_norm) if len(ref_norm) > 0 else 0
    return cer * 100


def print_wer_results(reference: str, hypothesis: str, target_wer: float = 25.0):
    """
    Calculate and print detailed WER/CER results with formatting

    Args:
        reference: Ground truth text
        hypothesis: Transcribed text to evaluate
        target_wer: Target WER threshold (default 25% = 75% accuracy)
    """
    # Calculate metrics
    wer_details = calculate_wer_detailed(reference, hypothesis)
    cer = calculate_cer(reference, hypothesis)

    # Extract key metrics
    raw_wer = wer_details['raw']['wer']
    normalized_wer = wer_details['normalized']['wer']
    normalized_accuracy = 100 - normalized_wer

    print("="*80)
    print("TRANSCRIPTION COMPARISON")
    print("="*80)
    print(f"Reference (raw):")
    print(f"  '{reference}'")
    print(f"\nTranscription (raw):")
    print(f"  '{hypothesis}'\n")

    print("="*80)
    print("NORMALIZED TEXT (for fair comparison)")
    print("="*80)
    print(f"Reference (normalized):")
    print(f"  '{wer_details['ref_normalized']}'")
    print(f"\nTranscription (normalized):")
    print(f"  '{wer_details['hyp_normalized']}'")
    print()

    print("="*80)
    print("ACCURACY METRICS")
    print("="*80)
    print(f"📊 Word Error Rate (WER):")
    print(f"   Raw WER (with punctuation):        {raw_wer:.1f}%")
    print(f"   Normalized WER (no punctuation):   {normalized_wer:.1f}%")
    print(f"   Normalized Accuracy:               {normalized_accuracy:.1f}%")
    print(f"\n📊 Character Error Rate (CER):")
    print(f"   Normalized CER:                    {cer:.1f}%")
    print(f"\n🎯 Target: ≥{100-target_wer:.0f}% accuracy (≤{target_wer:.0f}% WER)")
    print()

    # Show detailed error analysis if there are errors
    if wer_details['normalized']['edit_distance'] > 0:
        print("="*80)
        print("DETAILED ERROR ANALYSIS (Normalized)")
        print("="*80)
        norm = wer_details['normalized']
        print(f"Total errors: {norm['edit_distance']} / {norm['num_words']} words")

        if norm['substitutions']:
            print(f"\n❌ Substitutions ({len(norm['substitutions'])}):")
            for ref_word, hyp_word in norm['substitutions']:
                print(f"   '{ref_word}' → '{hyp_word}'")

        if norm['deletions']:
            print(f"\n❌ Deletions ({len(norm['deletions'])}):")
            for word in norm['deletions']:
                print(f"   Missing: '{word}'")

        if norm['insertions']:
            print(f"\n❌ Insertions ({len(norm['insertions'])}):")
            for word in norm['insertions']:
                print(f"   Extra: '{word}'")
        print()
    else:
        print("="*80)
        print("🎉 PERFECT TRANSCRIPTION - ZERO ERRORS!")
        print("="*80)
        print("All words match exactly (after normalization)")
        print("Differences are only in punctuation/formatting")
        print()

    return {
        'wer': normalized_wer,
        'accuracy': normalized_accuracy,
        'cer': cer,
        'passed': normalized_wer <= target_wer
    }


def concatenate_transcription_segments(segments: List[Dict]) -> str:
    """
    Concatenate all transcription segments into single text

    Args:
        segments: List of segment dicts with 'text' field

    Returns:
        Concatenated text
    """
    return ' '.join(seg.get('text', '').strip() for seg in segments if seg.get('text'))
