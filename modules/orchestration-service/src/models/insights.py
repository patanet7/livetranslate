"""
Pydantic models for the Meeting Intelligence system.

Request/response models for notes, insights, templates, and agent conversations.
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from .base import BaseModel, ResponseMixin

# =============================================================================
# Note Models
# =============================================================================


class NoteCreateRequest(BaseModel):
    """Request to create a manual note."""

    content: str = Field(..., min_length=1, description="Note content")
    speaker_name: str | None = Field(default=None, description="Optional speaker name")


class NoteAnalyzeRequest(BaseModel):
    """Request to create an LLM-analyzed note."""

    prompt: str = Field(..., min_length=1, description="Analysis prompt")
    context_sentences: list[str] | None = Field(
        default=None, description="Optional transcript sentences as context"
    )
    speaker_name: str | None = Field(default=None, description="Optional speaker filter")
    llm_backend: str | None = Field(default=None, description="LLM backend override")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class NoteResponse(BaseModel):
    """Response for a single note."""

    note_id: str
    session_id: str
    note_type: str
    content: str
    prompt_used: str | None = None
    context_sentences: list[str] | None = None
    speaker_name: str | None = None
    transcript_range_start: float | None = None
    transcript_range_end: float | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    processing_time_ms: float | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class NoteListResponse(ResponseMixin):
    """Response for listing notes."""

    notes: list[NoteResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Insight Models
# =============================================================================


class InsightGenerateRequest(BaseModel):
    """Request to generate insight(s) from template(s)."""

    template_names: list[str] = Field(
        ..., min_length=1, description="Template names to generate insights from"
    )
    custom_instructions: str | None = Field(
        default=None, description="Additional instructions to append to template"
    )
    llm_backend: str | None = Field(default=None, description="LLM backend override")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class InsightResponse(BaseModel):
    """Response for a single insight."""

    insight_id: str
    session_id: str
    template_id: str | None = None
    insight_type: str
    title: str
    content: str
    prompt_used: str | None = None
    transcript_length: int | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    processing_time_ms: float | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class InsightListResponse(ResponseMixin):
    """Response for listing insights."""

    insights: list[InsightResponse] = Field(default_factory=list)
    count: int = 0


class InsightGenerateResponse(ResponseMixin):
    """Response for insight generation."""

    insights: list[InsightResponse] = Field(default_factory=list)
    total_processing_time_ms: float = 0.0


# =============================================================================
# Template Models
# =============================================================================


class TemplateCreateRequest(BaseModel):
    """Request to create a custom template."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    category: str = Field(default="custom", description="Template category")
    prompt_template: str = Field(
        ..., min_length=1, description="Prompt template with {transcript}, {speakers}, etc."
    )
    system_prompt: str | None = None
    expected_output_format: str = Field(default="markdown")
    default_llm_backend: str | None = None
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1024, ge=1, le=8192)
    metadata: dict[str, Any] | None = None


class TemplateUpdateRequest(BaseModel):
    """Request to update a template."""

    name: str | None = Field(default=None, max_length=255)
    description: str | None = None
    category: str | None = None
    prompt_template: str | None = None
    system_prompt: str | None = None
    expected_output_format: str | None = None
    default_llm_backend: str | None = None
    default_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    default_max_tokens: int | None = Field(default=None, ge=1, le=8192)
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class TemplateResponse(BaseModel):
    """Response for a single template."""

    template_id: str
    name: str
    description: str | None = None
    category: str
    prompt_template: str
    system_prompt: str | None = None
    expected_output_format: str
    default_llm_backend: str | None = None
    default_temperature: float
    default_max_tokens: int
    is_builtin: bool
    is_active: bool
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TemplateListResponse(ResponseMixin):
    """Response for listing templates."""

    templates: list[TemplateResponse] = Field(default_factory=list)
    count: int = 0


# =============================================================================
# Agent Conversation Models
# =============================================================================


class AgentConversationCreateRequest(BaseModel):
    """Request to create an agent conversation."""

    title: str | None = Field(default=None, description="Conversation title")


class AgentMessageRequest(BaseModel):
    """Request to send a message in agent chat."""

    content: str = Field(..., min_length=1, description="Message content")


class AgentMessageResponse(BaseModel):
    """Response for a single agent message."""

    message_id: str
    conversation_id: str
    role: str
    content: str
    llm_backend: str | None = None
    llm_model: str | None = None
    processing_time_ms: float | None = None
    suggested_queries: list[str] | None = None
    created_at: datetime | None = None


class AgentConversationResponse(BaseModel):
    """Response for an agent conversation."""

    conversation_id: str
    session_id: str
    title: str | None = None
    status: str
    messages: list[AgentMessageResponse] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SuggestedQueriesResponse(ResponseMixin):
    """Response for suggested queries."""

    queries: list[str] = Field(default_factory=list)
