"""Pydantic models for business insights chat API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ProviderInfo(BaseModel):
    name: str
    configured: bool
    healthy: bool | None = None


class ModelInfoResponse(BaseModel):
    id: str
    name: str
    provider: str
    context_window: int | None = None


class ChatSettingsRequest(BaseModel):
    active_model: str = ""  # Prefixed model ID e.g. "home-gpu/qwen3.5:4b"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=128000)


class ChatSettingsResponse(BaseModel):
    active_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str | None = None
    provider: str | None = None
    model: str | None = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    provider: str | None = None  # Override per-message
    model: str | None = None  # Override per-message


class ToolCallInfo(BaseModel):
    tool_name: str
    arguments: dict
    result: str | None = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str | None = None
    tool_calls: list[ToolCallInfo] | None = None
    model: str | None = None
    provider: str | None = None
    tokens_used: int | None = None
    created_at: datetime


class SuggestedQueriesResponse(BaseModel):
    suggestions: list[str]
