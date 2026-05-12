"""Shared dataclasses for the lang_detect harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FrameTrace:
    """One Whisper inference frame as captured by Stage 1.

    Mirrors the (language, confidence, text, ...) state that arrives at
    api.py:_run_inference per audio chunk in production.
    """

    t_ms: float
    chunk_dur_s: float
    language: str
    confidence: float
    text: str
    no_speech_prob: float | None = None
    audio_rms: float | None = None


@dataclass
class GroundTruthSegment:
    """One labelled span of audio. Multiple segments per fixture for transitions."""

    t_ms_start: float
    t_ms_end: float
    language: str


@dataclass
class FixtureTrace:
    """A captured trace + its ground-truth labels."""

    fixture_id: str
    wav_path: Path
    frames: list[FrameTrace]
    ground_truth: list[GroundTruthSegment]
    total_duration_ms: float


@dataclass
class DetectorParams:
    """All sweepable parameters. Defaults match production WhisperLanguageDetector."""

    confidence_margin: float = 0.2
    min_dwell_frames: int = 4
    min_dwell_ms: float = 10_000.0
    # Proposed variants (default off to reproduce current behavior):
    initial_confidence_threshold: float = 0.0  # 0.0 = accept any first detection
    script_tiebreaker_enabled: bool = False
    script_tiebreaker_min_ratio: float = 0.3  # 30% script chars triggers override
    script_tiebreaker_max_confidence: float = 0.7  # only override if LID is unsure


@dataclass
class RunResult:
    """Scored output of one (fixture × params) run.

    Switches are classified into distinct buckets so the ranking can punish
    the harmful kind (flaps — breaking a correct state) more than the helpful
    kind (corrections — recovering from a wrong initial lock).
    """

    fixture_id: str
    params: DetectorParams

    # Correctness
    correct_at_end: bool
    time_to_correct_lang_ms: float | None  # None if never reached
    final_language: str | None

    # Switch classification
    correct_initial: bool        # first lock landed on the right language
    wrong_initial: bool          # first lock landed on the wrong language
    correction_switches: int     # wrong → right (helpful, recovers from bad start)
    flap_switches: int           # right → wrong (harmful, breaks a good state)
    wrong_recovery_switches: int # wrong → still wrong (different wrong language)
    transitions_caught: int      # switches that match a real ground-truth transition
    missed_transitions: int      # real transitions the detector never caught

    # Latency + impact
    switch_latency_ms_median: float | None  # over transitions_caught
    frames_total: int
    frames_with_wrong_lang: int             # cumulative wrong-state frames

    # Diagnostic
    notes: list[str] = field(default_factory=list)
