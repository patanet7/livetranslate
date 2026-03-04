"""
Behavioral tests for the diarization auto-trigger rules engine.

Tests evaluate real DiarizationRules instances against meeting metadata dicts
using the production evaluate_rules function — no mocks.
"""

from models.diarization import DiarizationRules
from services.diarization.rules import evaluate_rules


def test_no_match_when_disabled():
    """Rules engine returns None immediately when rules are disabled."""
    rules = DiarizationRules(enabled=False, participant_patterns=["*"])
    meeting = {"title": "anything", "participants": ["eric@test.com"], "duration": 60}
    assert evaluate_rules(meeting, rules) is None


def test_match_participant_pattern():
    """A matching participant pattern returns a result with match_type 'participant'."""
    rules = DiarizationRules(enabled=True, participant_patterns=["eric@*"], title_patterns=[])
    meeting = {
        "title": "standup",
        "participants": ["eric@company.com", "alice@company.com"],
        "duration": 600,
    }
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["match_type"] == "participant"
    assert "eric@company.com" in result["matched_value"]


def test_match_title_pattern():
    """A matching title pattern returns a result with match_type 'title'."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=[], title_patterns=["dev weekly*"]
    )
    meeting = {"title": "Dev Weekly Sync - March", "participants": [], "duration": 600}
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["match_type"] == "title"


def test_title_case_insensitive():
    """Title pattern matching is case-insensitive."""
    rules = DiarizationRules(enabled=True, participant_patterns=[], title_patterns=["1:1*"])
    meeting = {"title": "1:1 with Eric", "participants": [], "duration": 600}
    assert evaluate_rules(meeting, rules) is not None


def test_skip_short_meetings():
    """Meetings shorter than min_duration_minutes are skipped regardless of patterns."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["eric@*"], min_duration_minutes=5
    )
    meeting = {"title": "quick", "participants": ["eric@co.com"], "duration": 180}
    assert evaluate_rules(meeting, rules) is None


def test_skip_empty_meetings():
    """Meetings with sentence_count == 0 are skipped when exclude_empty is True."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["eric@*"], exclude_empty=True
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 600,
        "sentence_count": 0,
    }
    assert evaluate_rules(meeting, rules) is None


def test_no_rules_no_match():
    """No patterns configured means no match is ever returned."""
    rules = DiarizationRules(enabled=True, participant_patterns=[], title_patterns=[])
    meeting = {"title": "standup", "participants": ["eric@co.com"], "duration": 600}
    assert evaluate_rules(meeting, rules) is None


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_result_includes_matched_pattern():
    """Result dict includes 'matched_pattern' key reflecting which glob was used."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=[], title_patterns=["*weekly*"]
    )
    meeting = {"title": "Engineering Weekly", "participants": [], "duration": 600}
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["matched_pattern"] == "*weekly*"


def test_participant_pattern_case_insensitive():
    """Participant pattern matching is case-insensitive."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["ERIC@*"], title_patterns=[]
    )
    meeting = {
        "title": "sync",
        "participants": ["eric@company.com"],
        "duration": 600,
    }
    assert evaluate_rules(meeting, rules) is not None


def test_non_zero_sentence_count_not_excluded():
    """Meetings with sentence_count > 0 are not excluded by exclude_empty."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["*@co.com"], exclude_empty=True
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 600,
        "sentence_count": 5,
    }
    assert evaluate_rules(meeting, rules) is not None


def test_exclude_empty_false_allows_zero_sentence_count():
    """When exclude_empty is False, meetings with zero sentences still qualify."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["*@co.com"], exclude_empty=False
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 600,
        "sentence_count": 0,
    }
    assert evaluate_rules(meeting, rules) is not None


def test_exact_min_duration_boundary_passes():
    """A meeting exactly at the min duration threshold is not skipped."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["*@co.com"], min_duration_minutes=5
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 300,  # exactly 5 minutes
    }
    assert evaluate_rules(meeting, rules) is not None


def test_one_second_below_min_duration_fails():
    """A meeting one second below the threshold is skipped."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["*@co.com"], min_duration_minutes=5
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 299,
    }
    assert evaluate_rules(meeting, rules) is None


def test_participant_checked_before_title():
    """Participant patterns are evaluated before title patterns."""
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        title_patterns=["standup*"],
    )
    meeting = {
        "title": "standup daily",
        "participants": ["eric@co.com"],
        "duration": 600,
    }
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["match_type"] == "participant"


def test_missing_sentence_count_not_excluded():
    """When sentence_count key is absent, exclude_empty does not block the meeting."""
    rules = DiarizationRules(
        enabled=True, participant_patterns=["*@co.com"], exclude_empty=True
    )
    meeting = {
        "title": "standup",
        "participants": ["eric@co.com"],
        "duration": 600,
        # no 'sentence_count' key
    }
    assert evaluate_rules(meeting, rules) is not None
