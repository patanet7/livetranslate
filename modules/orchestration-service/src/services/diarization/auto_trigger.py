"""Auto-trigger: evaluate diarization rules after Fireflies sync.

Called from the Fireflies sync flow after a meeting is persisted.
Checks rules and queues diarization if a match is found.
"""

from typing import Any

from livetranslate_common.logging import get_logger
from models.diarization import DiarizationRules
from services.diarization.rules import evaluate_rules

logger = get_logger()


async def maybe_trigger_diarization(
    meeting: dict[str, Any],
    rules: DiarizationRules,
    pipeline: Any,
) -> dict[str, Any] | None:
    """Evaluate rules and queue diarization if meeting matches.

    Args:
        meeting: Dict with title, participants, duration, sentence_count.
        rules: Current diarization rules config.
        pipeline: DiarizationPipeline instance.

    Returns:
        Job dict if queued, None if no match.
    """
    match = evaluate_rules(meeting, rules)
    if match is None:
        return None

    logger.info(
        "diarization_auto_triggered",
        meeting_id=meeting.get("id"),
        match_type=match["match_type"],
        matched_pattern=match["matched_pattern"],
    )

    job = await pipeline.create_job(
        meeting_id=meeting["id"],
        triggered_by="auto_rule",
        rule_matched=match,
    )
    return job
