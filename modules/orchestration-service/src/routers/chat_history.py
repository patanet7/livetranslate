"""
Chat History Router

FastAPI router for conversation history management.
Provides CRUD endpoints for conversations, messages, and full-text search.
"""

import logging
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from database import (
    get_db_session,
    User,
    APIToken,
    ConversationSession,
    ChatMessage,
    ConversationStatistics,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# Request/Response Models
# ============================================================================

class CreateUserRequest(BaseModel):
    """Request to create a new user"""
    user_id: str
    email: str
    name: Optional[str] = None
    image_url: Optional[str] = None
    preferred_language: str = "en"
    max_concurrent_sessions: int = 10


class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    email: str
    name: Optional[str]
    image_url: Optional[str]
    preferred_language: Optional[str]
    created_at: datetime
    last_active_at: Optional[datetime]
    is_active: bool


class CreateSessionRequest(BaseModel):
    """Request to create new conversation session"""
    user_id: str
    session_type: str = "user_chat"
    session_title: Optional[str] = None
    enable_translation: bool = False
    target_languages: Optional[List[str]] = None


class SessionResponse(BaseModel):
    """Conversation session response"""
    session_id: str
    user_id: str
    session_type: str
    session_title: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    last_message_at: Optional[datetime]
    message_count: int
    enable_translation: bool
    target_languages: Optional[List[str]]


class CreateMessageRequest(BaseModel):
    """Request to add message to session"""
    session_id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    original_language: Optional[str] = "en"
    translated_content: Optional[dict] = None
    confidence: Optional[int] = None
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None


class MessageResponse(BaseModel):
    """Chat message response"""
    message_id: str
    session_id: str
    sequence_number: int
    role: str
    content: str
    original_language: Optional[str]
    translated_content: Optional[dict]
    timestamp: datetime
    confidence: Optional[int]
    speaker_id: Optional[str]
    speaker_name: Optional[str]


# ============================================================================
# User Endpoints
# ============================================================================

@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new user"""
    try:
        # Check if user already exists
        result = await db.execute(
            select(User).where(User.user_id == request.user_id)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with ID {request.user_id} already exists"
            )

        # Create new user
        user = User(
            user_id=request.user_id,
            email=request.email,
            name=request.name,
            image_url=request.image_url,
            preferred_language=request.preferred_language,
            max_concurrent_sessions=request.max_concurrent_sessions,
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        logger.info(f"Created user: {user.user_id}")

        return UserResponse(**user.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get user by ID"""
    result = await db.execute(
        select(User).where(User.user_id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    return UserResponse(**user.to_dict())


# ============================================================================
# Session Endpoints
# ============================================================================

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Create new conversation session"""
    try:
        # Verify user exists
        result = await db.execute(
            select(User).where(User.user_id == request.user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {request.user_id} not found"
            )

        # Create session
        session = ConversationSession(
            user_id=request.user_id,
            session_type=request.session_type,
            session_title=request.session_title,
            enable_translation=request.enable_translation,
            target_languages=request.target_languages,
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        logger.info(f"Created conversation session: {session.session_id}")

        return SessionResponse(**session.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get session by ID"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    result = await db.execute(
        select(ConversationSession)
        .options(selectinload(ConversationSession.messages))
        .where(ConversationSession.session_id == session_uuid)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return SessionResponse(**session.to_dict())


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session_type: Optional[str] = None,
    limit: int = Query(50, le=1000),
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session)
):
    """List user's conversation sessions with pagination and filtering"""
    # Build query
    query = select(ConversationSession).where(
        ConversationSession.user_id == user_id
    )

    if start_date:
        query = query.where(ConversationSession.started_at >= start_date)
    if end_date:
        query = query.where(ConversationSession.started_at <= end_date)
    if session_type:
        query = query.where(ConversationSession.session_type == session_type)

    # Add ordering and pagination
    query = query.order_by(ConversationSession.started_at.desc())
    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    sessions = result.scalars().all()

    return [SessionResponse(**s.to_dict()) for s in sessions]


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Delete conversation session (cascade deletes messages)"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    result = await db.execute(
        select(ConversationSession).where(
            ConversationSession.session_id == session_uuid
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    await db.delete(session)
    await db.commit()

    logger.info(f"Deleted conversation session: {session_id}")

    return None


# ============================================================================
# Message Endpoints
# ============================================================================

@router.post("/messages", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def create_message(
    request: CreateMessageRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """Add message to conversation session"""
    try:
        # Parse session UUID
        try:
            session_uuid = uuid.UUID(request.session_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID format"
            )

        # Verify session exists
        result = await db.execute(
            select(ConversationSession).where(
                ConversationSession.session_id == session_uuid
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {request.session_id} not found"
            )

        # Create message (sequence_number auto-incremented by trigger)
        message = ChatMessage(
            session_id=session_uuid,
            role=request.role,
            content=request.content,
            original_language=request.original_language,
            translated_content=request.translated_content,
            confidence=request.confidence,
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
        )

        db.add(message)
        await db.commit()
        await db.refresh(message)

        logger.info(f"Added message to session {request.session_id}: {message.message_id}")

        return MessageResponse(**message.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create message: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/messages/{session_id}", response_model=List[MessageResponse])
async def get_messages(
    session_id: str,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session)
):
    """Get all messages in a session"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    query = select(ChatMessage).where(
        ChatMessage.session_id == session_uuid
    ).order_by(
        ChatMessage.sequence_number
    ).limit(limit).offset(offset)

    result = await db.execute(query)
    messages = result.scalars().all()

    return [MessageResponse(**m.to_dict()) for m in messages]


@router.get("/search", response_model=List[MessageResponse])
async def search_messages(
    user_id: str,
    query: str,
    limit: int = Query(50, le=500),
    db: AsyncSession = Depends(get_db_session)
):
    """Search messages using full-text search"""
    # Build search query
    search_query = (
        select(ChatMessage)
        .join(ConversationSession)
        .where(ConversationSession.user_id == user_id)
        .where(ChatMessage.content.ilike(f"%{query}%"))
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit)
    )

    result = await db.execute(search_query)
    messages = result.scalars().all()

    return [MessageResponse(**m.to_dict()) for m in messages]


# ============================================================================
# Export Endpoints
# ============================================================================

@router.get("/export/{session_id}")
async def export_session(
    session_id: str,
    format: str = Query("json", pattern="^(json|txt)$"),
    db: AsyncSession = Depends(get_db_session)
):
    """Export conversation session in various formats"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    # Get session
    session_result = await db.execute(
        select(ConversationSession).where(
            ConversationSession.session_id == session_uuid
        )
    )
    session = session_result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    # Get messages
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_uuid)
        .order_by(ChatMessage.sequence_number)
    )
    messages = messages_result.scalars().all()

    if format == "json":
        return {
            "session": session.to_dict(),
            "messages": [m.to_dict() for m in messages]
        }

    elif format == "txt":
        # Plain text export
        lines = [f"Conversation: {session.session_title or session.session_id}"]
        lines.append(f"Started: {session.started_at}")
        lines.append("=" * 80)
        lines.append("")

        for msg in messages:
            speaker = msg.speaker_name or msg.role.upper()
            lines.append(f"[{msg.timestamp}] {speaker}:")
            lines.append(f"  {msg.content}")
            lines.append("")

        return JSONResponse(
            content={"content": "\n".join(lines), "format": "text/plain"}
        )


# ============================================================================
# Statistics Endpoints
# ============================================================================

@router.get("/statistics/{session_id}")
async def get_session_statistics(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get conversation statistics for a session"""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format"
        )

    # Get session
    result = await db.execute(
        select(ConversationSession)
        .options(selectinload(ConversationSession.messages))
        .where(ConversationSession.session_id == session_uuid)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    # Calculate statistics
    total_messages = session.message_count
    user_messages = len([m for m in session.messages if m.role == "user"])
    assistant_messages = len([m for m in session.messages if m.role == "assistant"])

    # Calculate average confidence
    confidences = [m.confidence for m in session.messages if m.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    # Calculate total characters and words
    total_chars = sum(len(m.content) for m in session.messages)
    total_words = sum(len(m.content.split()) for m in session.messages)

    return {
        "session_id": str(session.session_id),
        "total_messages": total_messages,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "average_confidence": avg_confidence,
        "total_characters": total_chars,
        "total_words": total_words,
        "session_duration": (session.ended_at - session.started_at).total_seconds()
        if session.ended_at
        else None,
    }
