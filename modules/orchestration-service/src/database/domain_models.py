#!/usr/bin/env python3
"""
Database Models for In-Domain Prompting System
Phase 2: SimulStreaming Innovation

Stores domain-specific terminology and prompts for Whisper Large-v3
Target: -40-60% domain error reduction

Reference: SimulStreaming paper - In-Domain Prompts section
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    Boolean,
    DateTime,
    ARRAY,
    Index,
    ForeignKey,
    JSON as JSONB,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB as PostgresJSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from .base import Base


class DomainCategory(Base):
    """
    Domain categories for in-domain prompting
    Examples: medical, legal, technical, education, business
    """

    __tablename__ = "domain_categories"

    domain_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Statistics
    usage_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(Integer, default=0, nullable=False)  # 0-100%

    # Relationships
    terminology = relationship(
        "DomainTerminology", back_populates="category", cascade="all, delete-orphan"
    )
    prompts = relationship(
        "DomainPrompt", back_populates="category", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_domain_categories_active", "is_active"),
        Index("ix_domain_categories_usage", "usage_count"),
    )


class DomainTerminology(Base):
    """
    Domain-specific terminology for in-domain prompting

    Per SimulStreaming paper:
    - Store domain-specific terms that commonly get mis-transcribed
    - Inject into Whisper's initial_prompt for better accuracy
    - Target: -40-60% reduction in domain-specific errors
    """

    __tablename__ = "domain_terminology"

    term_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain_id = Column(
        UUID(as_uuid=True), ForeignKey("domain_categories.domain_id"), nullable=False
    )

    # Term details
    term = Column(String(255), nullable=False, index=True)
    normalized_term = Column(String(255), nullable=False)  # Lowercase, stripped
    phonetic = Column(
        String(255), nullable=True
    )  # Phonetic spelling for better matching

    # Context
    common_context = Column(Text, nullable=True)  # Example usage
    alternatives = Column(ARRAY(String), nullable=True)  # Alternative spellings

    # Metadata
    importance = Column(
        Integer, default=50, nullable=False
    )  # 0-100, higher = more important
    frequency = Column(Integer, default=0, nullable=False)  # How often used
    accuracy_improvement = Column(
        Integer, default=0, nullable=True
    )  # % improvement when used

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    category = relationship("DomainCategory", back_populates="terminology")

    __table_args__ = (
        Index("ix_domain_terminology_term", "term"),
        Index("ix_domain_terminology_normalized", "normalized_term"),
        Index("ix_domain_terminology_importance", "importance"),
        Index("ix_domain_terminology_domain_importance", "domain_id", "importance"),
    )


class DomainPrompt(Base):
    """
    Pre-formatted domain prompts for Whisper initial_prompt parameter

    Per SimulStreaming paper (Section 4.2):
    - Scrolling context window: 448 tokens max
    - Include domain terminology + recent output context
    - Format: domain_terms + previous_context (223 tokens max)
    """

    __tablename__ = "domain_prompts"

    prompt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain_id = Column(
        UUID(as_uuid=True), ForeignKey("domain_categories.domain_id"), nullable=False
    )

    # Prompt details
    name = Column(String(255), nullable=False)
    template = Column(Text, nullable=False)  # Prompt template with placeholders

    # Example: "The following medical consultation discusses {topics}.
    # Common terms: {terminology}. Previous context: {context}"

    # Metadata
    is_default = Column(Boolean, default=False, nullable=False)
    max_tokens = Column(Integer, default=448, nullable=False)  # SimulStreaming limit

    # Performance
    usage_count = Column(Integer, default=0, nullable=False)
    average_quality_score = Column(Integer, default=0, nullable=True)  # 0-100%

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    category = relationship("DomainCategory", back_populates="prompts")

    __table_args__ = (
        Index("ix_domain_prompts_domain", "domain_id"),
        Index("ix_domain_prompts_default", "is_default"),
        Index("ix_domain_prompts_usage", "usage_count"),
    )


class UserDomainPreference(Base):
    """
    User-specific domain preferences and custom terminology
    Allows users to create custom domain dictionaries
    """

    __tablename__ = "user_domain_preferences"

    preference_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False)
    domain_id = Column(
        UUID(as_uuid=True), ForeignKey("domain_categories.domain_id"), nullable=False
    )

    # Custom settings
    custom_terminology = Column(PostgresJSONB, nullable=True)  # User-added terms
    # Format: [{"term": "xyz", "context": "...", "importance": 80}, ...]

    custom_prompt_template = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Usage
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_user_domain_prefs_user", "user_id"),
        Index("ix_user_domain_prefs_domain", "domain_id"),
        Index("ix_user_domain_prefs_user_domain", "user_id", "domain_id", unique=True),
    )


class DomainUsageLog(Base):
    """
    Logs domain prompt usage for analytics and optimization
    Tracks effectiveness of different domains and prompts
    """

    __tablename__ = "domain_usage_logs"

    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.session_id"),
        nullable=True,
    )
    domain_id = Column(
        UUID(as_uuid=True), ForeignKey("domain_categories.domain_id"), nullable=False
    )
    prompt_id = Column(
        UUID(as_uuid=True), ForeignKey("domain_prompts.prompt_id"), nullable=True
    )

    # Usage details
    user_id = Column(String(255), nullable=True)
    model_used = Column(String(100), nullable=False)  # "whisper-large-v3"

    # Quality metrics
    transcription_quality = Column(Integer, nullable=True)  # 0-100% confidence
    error_count = Column(Integer, default=0, nullable=False)
    terminology_matches = Column(Integer, default=0, nullable=False)

    # Performance
    processing_time_ms = Column(Integer, nullable=True)

    # Context
    prompt_tokens_used = Column(Integer, nullable=True)
    context_tokens_used = Column(Integer, nullable=True)

    # Timestamp
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        Index("ix_domain_usage_domain", "domain_id"),
        Index("ix_domain_usage_session", "session_id"),
        Index("ix_domain_usage_created", "created_at"),
        Index("ix_domain_usage_domain_date", "domain_id", "created_at"),
    )


# Pre-populated domain data initialization
PREDEFINED_DOMAINS = [
    {
        "name": "medical",
        "display_name": "Medical & Healthcare",
        "description": "Medical consultations, diagnoses, procedures, and healthcare terminology",
        "is_active": True,
    },
    {
        "name": "legal",
        "display_name": "Legal & Compliance",
        "description": "Legal proceedings, contracts, compliance, and legal terminology",
        "is_active": True,
    },
    {
        "name": "technical",
        "display_name": "Technical & Engineering",
        "description": "Software development, engineering, IT, and technical terminology",
        "is_active": True,
    },
    {
        "name": "business",
        "display_name": "Business & Finance",
        "description": "Business meetings, financial discussions, corporate terminology",
        "is_active": True,
    },
    {
        "name": "education",
        "display_name": "Education & Academic",
        "description": "Lectures, academic discussions, educational content",
        "is_active": True,
    },
    {
        "name": "general",
        "display_name": "General Conversation",
        "description": "General purpose conversations without specific domain",
        "is_active": True,
    },
]

# Medical terminology examples (expanded from SimulStreaming paper)
MEDICAL_TERMINOLOGY = [
    {"term": "diagnosis", "importance": 90, "context": "The diagnosis indicates..."},
    {"term": "symptoms", "importance": 90, "context": "Patient reports symptoms of..."},
    {
        "term": "prescription",
        "importance": 85,
        "context": "Writing a prescription for...",
    },
    {"term": "hypertension", "importance": 80, "context": "Patient has hypertension"},
    {"term": "diabetes", "importance": 80, "context": "Type 2 diabetes management"},
    {
        "term": "cardiovascular",
        "importance": 75,
        "context": "Cardiovascular examination",
    },
    {"term": "antibiotic", "importance": 75, "context": "Prescribing antibiotics"},
    {"term": "inflammation", "importance": 70, "context": "Signs of inflammation"},
    {"term": "consultation", "importance": 70, "context": "Medical consultation"},
    {"term": "examination", "importance": 70, "context": "Physical examination"},
]

# Technical terminology examples
TECHNICAL_TERMINOLOGY = [
    {
        "term": "Kubernetes",
        "importance": 90,
        "context": "Deploying to Kubernetes cluster",
    },
    {
        "term": "microservices",
        "importance": 85,
        "context": "Microservices architecture",
    },
    {"term": "Docker", "importance": 85, "context": "Docker containerization"},
    {"term": "CI/CD", "importance": 80, "context": "CI/CD pipeline"},
    {"term": "API", "importance": 80, "context": "REST API endpoint"},
    {"term": "database", "importance": 75, "context": "PostgreSQL database"},
    {"term": "authentication", "importance": 75, "context": "User authentication"},
    {"term": "deployment", "importance": 70, "context": "Production deployment"},
    {"term": "scalability", "importance": 70, "context": "System scalability"},
    {"term": "latency", "importance": 70, "context": "Reducing latency"},
]

# Legal terminology examples
LEGAL_TERMINOLOGY = [
    {"term": "plaintiff", "importance": 90, "context": "The plaintiff alleges..."},
    {"term": "defendant", "importance": 90, "context": "The defendant argues..."},
    {"term": "litigation", "importance": 85, "context": "Ongoing litigation"},
    {"term": "jurisdiction", "importance": 80, "context": "Court jurisdiction"},
    {"term": "compliance", "importance": 80, "context": "Regulatory compliance"},
    {"term": "contract", "importance": 75, "context": "Contract terms"},
    {"term": "liability", "importance": 75, "context": "Legal liability"},
    {"term": "statute", "importance": 70, "context": "Statutory requirements"},
    {"term": "testimony", "importance": 70, "context": "Witness testimony"},
    {"term": "precedent", "importance": 70, "context": "Legal precedent"},
]
