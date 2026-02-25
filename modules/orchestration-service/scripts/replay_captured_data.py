#!/usr/bin/env python3
"""
Replay captured Fireflies JSONL through SentenceAggregator.

Feeds finalized chunks (deduped to last version per chunk_id) through the
aggregator and reports sentence statistics: count, avg length, boundary
types, fragments, etc.

Usage:
    uv run python scripts/replay_captured_data.py [--pause-ms 600] [--max-words 25]
    uv run python scripts/replay_captured_data.py --compare
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Setup import path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from models.fireflies import FirefliesChunk, FirefliesSessionConfig, TranslationUnit
from services.sentence_aggregator import SentenceAggregator

DEFAULT_JSONL = root / "captured_data" / "20260224_190411_fireflies_raw_capture.jsonl"


def load_finalized_chunks(jsonl_path: Path) -> list[dict]:
    """Load JSONL and deduplicate to final version per chunk_id."""
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Filter to transcript events, deduplicate by chunk_id (last wins)
    final_by_id: dict[str, dict] = {}
    for e in events:
        if e["event"] not in ("transcription.broadcast", "transcript"):
            continue
        data = e.get("data", {})
        payload = data.get("payload", data.get("data", data))
        if isinstance(payload, dict) and payload.get("chunk_id"):
            final_by_id[payload["chunk_id"]] = payload

    # Sort by start_time
    return sorted(final_by_id.values(), key=lambda c: c.get("start_time", 0))


def replay(chunks: list[dict], config: FirefliesSessionConfig) -> list[TranslationUnit]:
    """Feed chunks through SentenceAggregator, collect produced sentences."""
    sentences: list[TranslationUnit] = []

    def on_sentence(unit: TranslationUnit):
        sentences.append(unit)

    aggregator = SentenceAggregator(
        session_id="replay",
        transcript_id="replay",
        config=config,
        on_sentence_ready=on_sentence,
    )

    for c in chunks:
        chunk = FirefliesChunk(
            transcript_id="replay",
            chunk_id=str(c["chunk_id"]),
            text=c.get("text", ""),
            speaker_name=c.get("speaker_name", "Unknown"),
            start_time=float(c.get("start_time", 0)),
            end_time=float(c.get("end_time", 0)),
        )
        aggregator.process_chunk(chunk)

    # Flush remaining
    remaining = aggregator.flush_all()
    sentences.extend(remaining)

    return sentences


def report(sentences: list[TranslationUnit], label: str):
    """Print stats report for a set of sentences."""
    if not sentences:
        print(f"\n=== {label}: No sentences produced ===")
        return

    lengths = [len(s.text.split()) for s in sentences]
    boundaries = Counter(s.boundary_type for s in sentences)
    speakers = Counter(s.speaker_name for s in sentences)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Sentences produced:   {len(sentences)}")
    print(f"  Avg words/sentence:   {sum(lengths) / len(lengths):.1f}")
    print(f"  Min words:            {min(lengths)}")
    print(f"  Max words:            {max(lengths)}")
    print(f"  Boundary types:       {dict(boundaries)}")
    print(f"  Speakers:             {dict(speakers)}")
    print(f"  Short (<3 words):     {sum(1 for l in lengths if l < 3)}")
    print(f"  Long (>20 words):     {sum(1 for l in lengths if l > 20)}")
    print()
    for i, s in enumerate(sentences):
        words = len(s.text.split())
        print(
            f"  [{i + 1:3}] ({s.boundary_type:15}) "
            f"[{s.speaker_name:>20}] ({words:2}w) \"{s.text}\""
        )


def main():
    parser = argparse.ArgumentParser(
        description="Replay Fireflies data through SentenceAggregator"
    )
    parser.add_argument(
        "--jsonl", type=Path, default=DEFAULT_JSONL, help="JSONL file to replay"
    )
    parser.add_argument(
        "--pause-ms", type=float, default=None, help="Override pause_threshold_ms"
    )
    parser.add_argument(
        "--max-words", type=int, default=None, help="Override max_buffer_words"
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="Override max_buffer_seconds"
    )
    parser.add_argument(
        "--min-words", type=int, default=None, help="Override min_words_for_translation"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both current and proposed thresholds",
    )
    args = parser.parse_args()

    chunks = load_finalized_chunks(args.jsonl)
    print(f"Loaded {len(chunks)} finalized chunks from {args.jsonl}")

    if args.compare:
        # Current thresholds
        current_config = FirefliesSessionConfig(
            api_key="",
            transcript_id="replay",
            pause_threshold_ms=800,
            max_buffer_words=30,
            max_buffer_seconds=5.0,
            min_words_for_translation=3,
        )
        current_sentences = replay(chunks, current_config)
        report(current_sentences, "CURRENT THRESHOLDS (800ms / 30w / 5s / min3)")

        # Proposed thresholds
        proposed_config = FirefliesSessionConfig(
            api_key="",
            transcript_id="replay",
            pause_threshold_ms=600,
            max_buffer_words=25,
            max_buffer_seconds=4.0,
            min_words_for_translation=2,
        )
        proposed_sentences = replay(chunks, proposed_config)
        report(proposed_sentences, "PROPOSED THRESHOLDS (600ms / 25w / 4s / min2)")
    else:
        config = FirefliesSessionConfig(
            api_key="",
            transcript_id="replay",
            pause_threshold_ms=args.pause_ms or 800,
            max_buffer_words=args.max_words or 30,
            max_buffer_seconds=args.max_seconds or 5.0,
            min_words_for_translation=args.min_words or 3,
        )
        sentences = replay(chunks, config)
        report(
            sentences,
            f"Thresholds: {config.pause_threshold_ms}ms / "
            f"{config.max_buffer_words}w / {config.max_buffer_seconds}s / "
            f"min{config.min_words_for_translation}",
        )


if __name__ == "__main__":
    main()
