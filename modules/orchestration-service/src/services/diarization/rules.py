"""
Diarization Auto-Trigger Rules Engine

Evaluates meeting metadata against a DiarizationRules configuration to decide
whether an offline diarization job should be automatically enqueued.

Pattern matching uses fnmatch (glob-style) and is case-insensitive throughout.
"""

from __future__ import annotations

import fnmatch
from typing import Any

from models.diarization import DiarizationRules


def evaluate_rules(meeting: dict[str, Any], rules: DiarizationRules) -> dict[str, Any] | None:
    """Evaluate a meeting against diarization auto-trigger rules.

    Applies the configured DiarizationRules to a meeting metadata dict and
    returns a match descriptor when the meeting qualifies for automatic
    diarization, or None when it does not.

    Args:
        meeting: Mapping of meeting metadata with the following keys:

            - ``title`` (str): Meeting title.
            - ``participants`` (list[str]): Participant email addresses.
            - ``duration`` (int): Meeting length in seconds.
            - ``sentence_count`` (int, optional): Number of transcript
              sentences; omit or set to a positive value when sentences
              exist.

        rules: Configured DiarizationRules instance to evaluate against.

    Returns:
        A dict with the following keys if the meeting matches a rule:

        - ``match_type`` (str): Either ``"participant"`` or ``"title"``.
        - ``matched_value`` (str): The concrete value that triggered the
          match (the participant email or the meeting title).
        - ``matched_pattern`` (str): The glob pattern that produced the
          match.

        Returns ``None`` when the meeting does not qualify.

    Example:
        >>> from models.diarization import DiarizationRules
        >>> rules = DiarizationRules(
        ...     enabled=True,
        ...     participant_patterns=["*@company.com"],
        ... )
        >>> meeting = {
        ...     "title": "Weekly Sync",
        ...     "participants": ["alice@company.com"],
        ...     "duration": 600,
        ... }
        >>> result = evaluate_rules(meeting, rules)
        >>> result["match_type"]
        'participant'
    """
    # ------------------------------------------------------------------
    # Gate 1: rules must be enabled
    # ------------------------------------------------------------------
    if not rules.enabled:
        return None

    # ------------------------------------------------------------------
    # Gate 2: meeting must meet the minimum duration threshold
    # ------------------------------------------------------------------
    duration: int = meeting.get("duration", 0)
    min_seconds: int = rules.min_duration_minutes * 60
    if duration < min_seconds:
        return None

    # ------------------------------------------------------------------
    # Gate 3: optionally skip meetings with no transcript content
    # ------------------------------------------------------------------
    if rules.exclude_empty:
        sentence_count = meeting.get("sentence_count")
        if sentence_count is not None and sentence_count == 0:
            return None

    # ------------------------------------------------------------------
    # Check 1: participant patterns (evaluated before title patterns)
    # ------------------------------------------------------------------
    participants: list[str] = meeting.get("participants", [])
    for pattern in rules.participant_patterns:
        pattern_lower = pattern.lower()
        for participant in participants:
            if fnmatch.fnmatch(participant.lower(), pattern_lower):
                return {
                    "match_type": "participant",
                    "matched_value": participant,
                    "matched_pattern": pattern,
                }

    # ------------------------------------------------------------------
    # Check 2: title patterns
    # ------------------------------------------------------------------
    title: str = meeting.get("title", "")
    title_lower = title.lower()
    for pattern in rules.title_patterns:
        if fnmatch.fnmatch(title_lower, pattern.lower()):
            return {
                "match_type": "title",
                "matched_value": title,
                "matched_pattern": pattern,
            }

    # No patterns matched
    return None
