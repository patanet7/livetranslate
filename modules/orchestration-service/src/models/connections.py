"""Pydantic models for the unified AI connections API."""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")
_VALID_ENGINES = {"ollama", "openai", "anthropic", "openai_compatible"}


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:64]


class AIConnectionCreate(BaseModel):
    id: str | None = None  # Auto-generated from name if omitted
    name: str = Field(min_length=1, max_length=200)
    engine: str
    url: str
    api_key: str = ""
    prefix: str = ""
    enabled: bool = True
    context_length: int | None = None
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    max_retries: int = Field(default=3, ge=0, le=10)
    priority: int = Field(default=0, ge=0)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        if v not in _VALID_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_ENGINES}")
        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str | None) -> str | None:
        if v is not None and not _SLUG_RE.match(v):
            raise ValueError("id must be lowercase alphanumeric + hyphens, max 64 chars")
        return v


class AIConnectionUpdate(BaseModel):
    name: str | None = None
    engine: str | None = None
    url: str | None = None
    api_key: str | None = None
    prefix: str | None = None
    enabled: bool | None = None
    context_length: int | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    priority: int | None = None

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_ENGINES}")
        return v


class AIConnectionResponse(BaseModel):
    id: str
    name: str
    engine: str
    url: str
    has_api_key: bool  # Never expose the actual key
    prefix: str
    enabled: bool
    context_length: int | None
    timeout_ms: int
    max_retries: int
    priority: int

    model_config = {"from_attributes": True}


class VerifyResult(BaseModel):
    status: str  # "connected" | "error"
    message: str
    version: str | None = None
    models: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class AggregatedModel(BaseModel):
    id: str  # "prefix/model_name"
    name: str
    connection_id: str
    connection_name: str
    prefix: str
    engine: str


class AggregateModelsResponse(BaseModel):
    models: list[AggregatedModel] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class FeaturePreference(BaseModel):
    active_model: str = ""
    fallback_model: str = ""
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
