"""
Meeting Intelligence Router

API endpoints for the Meeting Intelligence system:
- Notes: Real-time manual + LLM-analyzed notes
- Insights: Post-meeting template-based analysis
- Templates: CRUD for configurable prompt templates
- Agent: Scaffolded chat interface for Q&A about transcripts

All endpoints are prefixed with /api/intelligence (set in main_fastapi.py).
"""

import logging
import uuid

from database.models import BotSession, Transcript
from dependencies import get_database_manager
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from models.insights import (
    AgentConversationCreateRequest,
    AgentConversationResponse,
    AgentMessageRequest,
    AgentMessageResponse,
    InsightGenerateRequest,
    InsightGenerateResponse,
    InsightListResponse,
    InsightResponse,
    NoteAnalyzeRequest,
    NoteCreateRequest,
    NoteListResponse,
    NoteResponse,
    SuggestedQueriesResponse,
    TemplateCreateRequest,
    TemplateListResponse,
    TemplateResponse,
    TemplateUpdateRequest,
)
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Meeting Intelligence"])


# =============================================================================
# Helpers
# =============================================================================


def _validate_uuid(value: str, name: str = "ID") -> str:
    """Validate that a string is a valid UUID, raise 400 if not."""
    try:
        uuid.UUID(value)
        return value
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name}: '{value}' is not a valid UUID",
        ) from e


# =============================================================================
# Dependency: MeetingIntelligenceService
# =============================================================================


def get_intelligence_service():
    """Get the MeetingIntelligenceService singleton."""
    from dependencies import get_meeting_intelligence_service

    return get_meeting_intelligence_service()


# =============================================================================
# Notes Endpoints
# =============================================================================


@router.get(
    "/sessions/{session_id}/notes",
    response_model=NoteListResponse,
    summary="List notes for a session",
)
async def list_notes(
    session_id: str,
    note_type: str | None = Query(default=None, description="Filter: auto, manual, annotation"),
    service=Depends(get_intelligence_service),
):
    """Get all notes for a session, optionally filtered by type."""
    _validate_uuid(session_id, "session_id")
    notes = await service.get_notes(session_id, note_type=note_type)
    return NoteListResponse(
        notes=[NoteResponse(**n) for n in notes],
        count=len(notes),
    )


@router.post(
    "/sessions/{session_id}/notes",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manual note",
)
async def create_note(
    session_id: str,
    request: NoteCreateRequest,
    service=Depends(get_intelligence_service),
):
    """Create a plain-text manual note for a session."""
    _validate_uuid(session_id, "session_id")
    note = await service.create_manual_note(
        session_id=session_id,
        content=request.content,
        speaker_name=request.speaker_name,
    )
    return NoteResponse(**note)


@router.post(
    "/sessions/{session_id}/notes/analyze",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an LLM-analyzed note",
)
async def analyze_note(
    session_id: str,
    request: NoteAnalyzeRequest,
    service=Depends(get_intelligence_service),
):
    """Create an LLM-analyzed note using a custom prompt."""
    _validate_uuid(session_id, "session_id")
    try:
        note = await service.create_analyzed_note(
            session_id=session_id,
            prompt=request.prompt,
            context_sentences=request.context_sentences,
            speaker_name=request.speaker_name,
            llm_backend=request.llm_backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return NoteResponse(**note)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e


@router.delete(
    "/notes/{note_id}",
    summary="Delete a note",
)
async def delete_note(
    note_id: str,
    service=Depends(get_intelligence_service),
):
    """Delete a note by ID."""
    _validate_uuid(note_id, "note_id")
    deleted = await service.delete_note(note_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Note {note_id} not found",
        )
    return {"success": True, "message": f"Note {note_id} deleted"}


# =============================================================================
# Insights Endpoints
# =============================================================================


@router.post(
    "/sessions/{session_id}/insights/generate",
    response_model=InsightGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate insights from templates",
)
async def generate_insights(
    session_id: str,
    request: InsightGenerateRequest,
    service=Depends(get_intelligence_service),
):
    """
    Generate insight(s) from template(s) for a session.

    Requires transcript data to be available in the database for this session.
    """
    _validate_uuid(session_id, "session_id")
    try:
        # Get transcript text from database
        transcript_text, speakers, duration = await _get_session_transcript(session_id)

        if not transcript_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No transcript data found for this session. Import or record a transcript first.",
            )

        insights = await service.generate_all_insights(
            session_id=session_id,
            template_names=request.template_names,
            transcript_text=transcript_text,
            speakers=speakers,
            duration=duration,
            custom_instructions=request.custom_instructions,
            llm_backend=request.llm_backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Separate successes from errors
        successful = [InsightResponse(**i) for i in insights if "error" not in i]
        failed = [i for i in insights if "error" in i]
        total_time = sum(i.get("processing_time_ms", 0) for i in insights if "error" not in i)

        message = f"Generated {len(successful)} insight(s)"
        if failed:
            failed_names = [f.get("template_name", "unknown") for f in failed]
            message += f"; {len(failed)} failed: {', '.join(failed_names)}"

        return InsightGenerateResponse(
            insights=successful,
            total_processing_time_ms=total_time,
            message=message,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception(f"Insight generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Insight generation failed: {e!s}",
        ) from e


@router.get(
    "/sessions/{session_id}/insights",
    response_model=InsightListResponse,
    summary="List insights for a session",
)
async def list_insights(
    session_id: str,
    service=Depends(get_intelligence_service),
):
    """Get all insights for a session."""
    _validate_uuid(session_id, "session_id")
    insights = await service.get_insights(session_id)
    return InsightListResponse(
        insights=[InsightResponse(**i) for i in insights],
        count=len(insights),
    )


@router.get(
    "/insights/{insight_id}",
    response_model=InsightResponse,
    summary="Get a specific insight",
)
async def get_insight(
    insight_id: str,
    service=Depends(get_intelligence_service),
):
    """Get a specific insight by ID."""
    _validate_uuid(insight_id, "insight_id")
    insight = await service.get_insight(insight_id)
    if not insight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight {insight_id} not found",
        )
    return InsightResponse(**insight)


@router.delete(
    "/insights/{insight_id}",
    summary="Delete an insight",
)
async def delete_insight(
    insight_id: str,
    service=Depends(get_intelligence_service),
):
    """Delete an insight by ID."""
    _validate_uuid(insight_id, "insight_id")
    deleted = await service.delete_insight(insight_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insight {insight_id} not found",
        )
    return {"success": True, "message": f"Insight {insight_id} deleted"}


# =============================================================================
# Template Endpoints
# =============================================================================


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="List all templates",
)
async def list_templates(
    category: str | None = Query(default=None, description="Filter by category"),
    active_only: bool = Query(default=True, description="Only show active templates"),
    service=Depends(get_intelligence_service),
):
    """List all insight prompt templates."""
    templates = await service.get_templates(category=category, active_only=active_only)
    return TemplateListResponse(
        templates=[TemplateResponse(**t) for t in templates],
        count=len(templates),
    )


@router.get(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Get a specific template",
)
async def get_template(
    template_id: str,
    service=Depends(get_intelligence_service),
):
    """Get a template by ID or name (accepts UUID or name string)."""
    template = await service.get_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return TemplateResponse(**template)


@router.post(
    "/templates",
    response_model=TemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom template",
)
async def create_template(
    request: TemplateCreateRequest,
    service=Depends(get_intelligence_service),
):
    """Create a new custom insight prompt template."""
    try:
        template = await service.create_template(request.model_dump())
        return TemplateResponse(**template)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.put(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="Update a template",
)
async def update_template(
    template_id: str,
    request: TemplateUpdateRequest,
    service=Depends(get_intelligence_service),
):
    """Update an existing template."""
    _validate_uuid(template_id, "template_id")
    update_data = request.model_dump(exclude_none=True)
    template = await service.update_template(template_id, update_data)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_id} not found",
        )
    return TemplateResponse(**template)


@router.delete(
    "/templates/{template_id}",
    summary="Delete a template",
)
async def delete_template(
    template_id: str,
    service=Depends(get_intelligence_service),
):
    """Delete a custom template. Built-in templates cannot be deleted."""
    _validate_uuid(template_id, "template_id")
    deleted = await service.delete_template(template_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_id} not found or is a built-in template",
        )
    return {"success": True, "message": f"Template {template_id} deleted"}


# =============================================================================
# Agent Endpoints (Scaffolding)
# =============================================================================


@router.post(
    "/sessions/{session_id}/agent/conversations",
    response_model=AgentConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start an agent conversation",
)
async def create_conversation(
    session_id: str,
    request: AgentConversationCreateRequest | None = None,
    service=Depends(get_intelligence_service),
):
    """Start a new agent conversation for a session."""
    _validate_uuid(session_id, "session_id")
    # Try to get transcript for context
    transcript_text, _, _ = await _get_session_transcript(session_id)

    title = request.title if request else None
    conv = await service.create_conversation(
        session_id=session_id,
        title=title,
        transcript_text=transcript_text,
    )
    return AgentConversationResponse(**conv)


@router.get(
    "/agent/conversations/{conversation_id}",
    response_model=AgentConversationResponse,
    summary="Get conversation with history",
)
async def get_conversation(
    conversation_id: str,
    service=Depends(get_intelligence_service),
):
    """Get a conversation with its full message history."""
    _validate_uuid(conversation_id, "conversation_id")
    conv = await service.get_conversation_history(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return AgentConversationResponse(**conv)


@router.post(
    "/agent/conversations/{conversation_id}/messages",
    response_model=AgentMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message and get LLM response",
)
async def send_message(
    conversation_id: str,
    request: AgentMessageRequest,
    service=Depends(get_intelligence_service),
):
    """Send a message in an agent conversation. Returns an LLM-powered response."""
    _validate_uuid(conversation_id, "conversation_id")
    try:
        msg = await service.send_message(
            conversation_id=conversation_id,
            content=request.content,
        )
        return AgentMessageResponse(**msg)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.post(
    "/agent/conversations/{conversation_id}/messages/stream",
    summary="Send a message with streaming response (SSE)",
)
async def send_message_stream(
    conversation_id: str,
    request: AgentMessageRequest,
    service=Depends(get_intelligence_service),
):
    """Stream an LLM response token-by-token via Server-Sent Events."""
    _validate_uuid(conversation_id, "conversation_id")
    return StreamingResponse(
        service.send_message_stream(conversation_id, request.content),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/sessions/{session_id}/agent/suggestions",
    response_model=SuggestedQueriesResponse,
    summary="Get suggested queries for a session",
)
async def get_suggestions(
    session_id: str,
    service=Depends(get_intelligence_service),
):
    """Get suggested queries for the agent chat."""
    _validate_uuid(session_id, "session_id")
    queries = await service.get_suggested_queries(session_id)
    return SuggestedQueriesResponse(queries=queries)


# =============================================================================
# Helpers
# =============================================================================


async def _get_session_transcript(
    session_id: str,
) -> tuple[str | None, list[str] | None, str | None]:
    """
    Retrieve transcript text, speakers, and duration for a session from the database.

    Returns:
        Tuple of (transcript_text, speakers_list, duration_string)
    """
    try:
        db_manager = get_database_manager()
        async with db_manager.get_session() as session:
            # Get session info
            result = await session.execute(
                select(BotSession).where(BotSession.session_id == uuid.UUID(session_id))
            )
            bot_session = result.scalar_one_or_none()

            # Get transcripts
            result = await session.execute(
                select(Transcript)
                .where(Transcript.session_id == uuid.UUID(session_id))
                .order_by(Transcript.start_time.asc())
            )
            transcripts = result.scalars().all()

            if not transcripts:
                return None, None, None

            # Build transcript text
            lines = []
            speakers = set()
            for t in transcripts:
                speaker = t.speaker_name or "Unknown"
                speakers.add(speaker)
                lines.append(f"{speaker}: {t.text}")

            transcript_text = "\n".join(lines)
            speakers_list = sorted(speakers)

            # Calculate duration
            duration = None
            if bot_session and bot_session.started_at and bot_session.ended_at:
                delta = bot_session.ended_at - bot_session.started_at
                minutes = int(delta.total_seconds() // 60)
                duration = f"{minutes} minutes"

            return transcript_text, speakers_list, duration

    except Exception as e:
        logger.warning(f"Could not retrieve transcript for session {session_id}: {e}")
        return None, None, None
