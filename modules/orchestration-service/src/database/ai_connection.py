"""SQLAlchemy model for unified AI connections."""

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer, Text, func

from .base import Base


class AIConnection(Base):
    """A configured AI backend connection (Ollama, OpenAI, Anthropic, etc.)."""

    __tablename__ = "ai_connections"
    __table_args__ = (
        CheckConstraint(
            "engine IN ('ollama', 'openai', 'anthropic', 'openai_compatible')",
            name="ck_ai_connections_engine",
        ),
    )

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    engine = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    api_key = Column(Text, nullable=False, server_default="")
    prefix = Column(Text, nullable=False, server_default="")
    enabled = Column(Boolean, nullable=False, server_default="true")
    context_length = Column(Integer, nullable=True)
    timeout_ms = Column(Integer, nullable=False, server_default="30000")
    max_retries = Column(Integer, nullable=False, server_default="3")
    priority = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
