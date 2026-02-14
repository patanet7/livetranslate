"""
Pipeline Configuration

Shared configuration for TranscriptionPipelineCoordinator.
All sources use the same config structure, with source-specific
metadata stored in the source_metadata field.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PipelineConfig:
    """
    Configuration for TranscriptionPipelineCoordinator.

    This config is source-agnostic - it works for Fireflies, Google Meet,
    or any other transcript source. Source-specific settings go in source_metadata.

    Attributes:
        session_id: Internal session identifier
        source_type: Source identifier ("fireflies", "google_meet", etc.)
        transcript_id: External transcript/meeting identifier
        source_language: Source language code (default: "en")
        target_languages: Target languages for translation
        pause_threshold_ms: Pause duration that indicates sentence boundary
        max_words_per_sentence: Maximum words before forcing sentence break
        max_time_per_sentence_ms: Maximum time before forcing sentence break
        min_words_for_translation: Minimum words required to translate
        use_nlp_boundary_detection: Enable spaCy sentence detection
        speaker_context_window: Context sentences per speaker
        global_context_window: Cross-speaker context sentences
        glossary_id: Optional glossary for term injection
        domain: Domain for glossary filtering
        caption_duration_seconds: How long captions stay visible
        speaker_aggregation_window_seconds: Window for aggregating same-speaker text
        enable_websocket: Enable WebSocket caption output
        enable_obs: Enable OBS text source output
        source_metadata: Source-specific metadata (e.g., Fireflies API key)
    """

    # Session identification
    session_id: str
    source_type: str  # "fireflies", "google_meet", "whisper"
    transcript_id: str = ""

    # Translation settings
    source_language: str = "en"
    target_languages: list[str] = field(default_factory=lambda: ["es"])

    # Sentence aggregation
    pause_threshold_ms: float = 800.0
    max_words_per_sentence: int = 30
    max_time_per_sentence_ms: float = 5000.0
    min_words_for_translation: int = 3
    use_nlp_boundary_detection: bool = True

    # Context windows
    speaker_context_window: int = 3
    global_context_window: int = 10
    include_cross_speaker_context: bool = True

    # Glossary
    glossary_id: str | None = None
    domain: str = "general"

    # Caption display
    caption_duration_seconds: float = 4.0
    speaker_aggregation_window_seconds: float = 5.0

    # Output
    enable_websocket: bool = True
    enable_obs: bool = False

    # Meeting Intelligence - Auto-Notes
    enable_auto_notes: bool = False
    auto_notes_interval: int = 10  # Generate auto-note every N sentences
    auto_notes_template: str = "auto_note"
    intelligence_llm_backend: str = ""  # Empty = use session default

    # Source-specific metadata (adapter uses this)
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Statistics tracked by the pipeline."""

    chunks_received: int = 0
    sentences_produced: int = 0
    translations_completed: int = 0
    translations_failed: int = 0
    captions_displayed: int = 0
    errors: int = 0
    auto_notes_generated: int = 0
    speakers_seen: int = 0
    speaker_names: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_chunk_at: datetime | None = None
    last_translation_at: datetime | None = None
    average_translation_time_ms: float = 0.0

    def record_translation(self, translation_time_ms: float) -> None:
        """Record a successful translation and update average time."""
        n = self.translations_completed
        self.average_translation_time_ms = (
            self.average_translation_time_ms * n + translation_time_ms
        ) / (n + 1)
        self.translations_completed += 1
        self.last_translation_at = datetime.now(UTC)

    def record_speaker(self, speaker_name: str) -> None:
        """Record a speaker if not already seen."""
        if speaker_name and speaker_name not in self.speaker_names:
            self.speaker_names.append(speaker_name)
            self.speakers_seen = len(self.speaker_names)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunks_received": self.chunks_received,
            "sentences_produced": self.sentences_produced,
            "translations_completed": self.translations_completed,
            "translations_failed": self.translations_failed,
            "captions_displayed": self.captions_displayed,
            "errors": self.errors,
            "auto_notes_generated": self.auto_notes_generated,
            "speakers_seen": self.speakers_seen,
            "speaker_names": self.speaker_names,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_chunk_at": self.last_chunk_at.isoformat() if self.last_chunk_at else None,
            "last_translation_at": (
                self.last_translation_at.isoformat() if self.last_translation_at else None
            ),
            "average_translation_time_ms": round(self.average_translation_time_ms, 2),
            "running_seconds": (
                (datetime.now(UTC) - self.started_at).total_seconds() if self.started_at else 0
            ),
        }
