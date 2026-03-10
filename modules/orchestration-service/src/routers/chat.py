"""
Business Insights Chat Router

Provides a general-purpose chat interface with LLM providers
and tool-calling for business data queries.
"""

import json
from datetime import UTC, datetime

from database import get_db_session
from database.models import ChatConversation, ChatMessageModel, SystemConfig
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from livetranslate_common.logging import get_logger
from models.chat import (
    ChatSettingsRequest,
    ChatSettingsResponse,
    ConversationCreateRequest,
    ConversationResponse,
    MessageRequest,
    MessageResponse,
    ModelInfoResponse,
    ProviderInfo,
    SuggestedQueriesResponse,
    ToolCallInfo,
)
from services.chat_tools import register_all_tools
from services.llm.adapter import ChatMessage
from services.llm.registry import get_registry
from services.llm.tool_executor import ToolExecutor
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = get_logger()
router = APIRouter(tags=["Business Chat"])

# System prompt for business insights
SYSTEM_PROMPT = """You are a business insights assistant for LiveTranslate, a real-time speech translation platform.
You have access to tools that query meeting data, translation statistics, speaker analytics, and diarization information.
Use these tools to answer questions about meetings, translations, speakers, and usage patterns.
Be concise and data-driven in your responses. When presenting data, use clear formatting."""


def _get_tool_executor() -> ToolExecutor:
    """Get a configured tool executor with all business insight tools."""
    executor = ToolExecutor()
    register_all_tools(executor)
    return executor


# -- Provider & Model endpoints ------------------------------------------------


@router.get("/providers")
async def list_providers() -> list[ProviderInfo]:
    """List available LLM providers with configuration status."""
    registry = get_registry()
    providers = registry.list_providers()
    health = await registry.health_check_all()
    return [
        ProviderInfo(
            name=p["name"],
            configured=p["configured"],
            healthy=health.get(p["name"]),
        )
        for p in providers
    ]


@router.get("/providers/{provider}/models")
async def list_provider_models(provider: str) -> list[ModelInfoResponse]:
    """List available models for a provider."""
    registry = get_registry()
    try:
        models = await registry.list_provider_models(provider)
        return [
            ModelInfoResponse(
                id=m.id,
                name=m.name,
                provider=m.provider,
                context_window=m.context_window,
            )
            for m in models
        ]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# -- Settings endpoints --------------------------------------------------------


@router.get("/settings")
async def get_chat_settings(
    db: AsyncSession = Depends(get_db_session),
) -> ChatSettingsResponse:
    """Get current chat settings."""
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "chat_settings")
    )
    row = result.scalar_one_or_none()
    if row and row.value:
        data = json.loads(row.value) if isinstance(row.value, str) else row.value
        return ChatSettingsResponse(**data)
    return ChatSettingsResponse(provider="ollama")


@router.put("/settings")
async def update_chat_settings(
    request: ChatSettingsRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ChatSettingsResponse:
    """Update chat settings and configure the provider."""
    registry = get_registry()

    # Build provider config
    config: dict = {}
    if request.api_key:
        config["api_key"] = request.api_key
    if request.base_url:
        config["base_url"] = request.base_url
    if request.model:
        config["default_model"] = request.model

    # Configure the provider
    try:
        registry.configure(request.provider, config)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to configure provider: {e}"
        )

    # Save settings to DB
    settings_data = {
        "provider": request.provider,
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "has_api_key": bool(request.api_key),
        "base_url": request.base_url,
    }
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "chat_settings")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = json.dumps(settings_data)
        row.updated_at = datetime.now(UTC)
    else:
        row = SystemConfig(key="chat_settings", value=json.dumps(settings_data))
        db.add(row)
    await db.commit()

    return ChatSettingsResponse(**settings_data)


# -- Conversation endpoints ----------------------------------------------------


@router.post("/conversations", status_code=201)
async def create_conversation(
    request: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ConversationResponse:
    """Create a new chat conversation."""
    conv = ChatConversation(title=request.title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return _conv_to_response(conv, msg_count=0)


@router.get("/conversations")
async def list_conversations(
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
) -> list[ConversationResponse]:
    """List conversations ordered by most recent."""
    result = await db.execute(
        select(ChatConversation)
        .order_by(ChatConversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    convs = result.scalars().all()
    return [_conv_to_response(c) for c in convs]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Get a conversation with all its messages."""
    result = await db.execute(
        select(ChatConversation)
        .where(ChatConversation.id == conversation_id)
        .options(selectinload(ChatConversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation": _conv_to_response(conv),
        "messages": [_msg_to_response(m) for m in conv.messages],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a conversation and all its messages."""
    result = await db.execute(
        select(ChatConversation).where(ChatConversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()
    return {"success": True}


# -- Message endpoints ---------------------------------------------------------


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    request: MessageRequest,
    db: AsyncSession = Depends(get_db_session),
) -> MessageResponse:
    """Send a message and get LLM response with tool-calling."""
    # Load conversation
    result = await db.execute(
        select(ChatConversation)
        .where(ChatConversation.id == conversation_id)
        .options(selectinload(ChatConversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Determine provider/model
    registry = get_registry()
    provider_name = request.provider or conv.provider or "ollama"
    model = request.model or conv.model

    try:
        adapter = registry.get_adapter(provider_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build message history
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for m in conv.messages:
        messages.append(
            ChatMessage(
                role=m.role,
                content=m.content,
                tool_call_id=m.tool_call_id,
                name=m.tool_name,
            )
        )
    messages.append(ChatMessage(role="user", content=request.content))

    # Save user message
    user_msg = ChatMessageModel(
        conversation_id=conv.id,
        role="user",
        content=request.content,
    )
    db.add(user_msg)

    # Run chat with tools
    executor = _get_tool_executor()
    try:
        response, _ = await executor.run_chat_with_tools(
            adapter=adapter,
            messages=messages,
            model=model,
            db=db,
        )
    except Exception as e:
        logger.error("chat_error", error=str(e), conversation_id=conversation_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    # Save assistant response
    tool_calls_json = None
    if response.tool_calls:
        tool_calls_json = [
            {"tool_name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ]

    assistant_msg = ChatMessageModel(
        conversation_id=conv.id,
        role="assistant",
        content=response.content,
        tool_calls=tool_calls_json,
        model=response.model,
        provider=provider_name,
        tokens_used=response.usage.total_tokens if response.usage else None,
    )
    db.add(assistant_msg)

    # Update conversation metadata
    conv.updated_at = datetime.now(UTC)
    if not conv.provider:
        conv.provider = provider_name
    if not conv.model and response.model:
        conv.model = response.model
    if not conv.title and request.content:
        conv.title = request.content[:100]

    await db.commit()
    await db.refresh(assistant_msg)

    return _msg_to_response(assistant_msg)


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    request: MessageRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Send a message and stream the LLM response via SSE."""
    # Load conversation
    result = await db.execute(
        select(ChatConversation)
        .where(ChatConversation.id == conversation_id)
        .options(selectinload(ChatConversation.messages))
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    registry = get_registry()
    provider_name = request.provider or conv.provider or "ollama"
    model = request.model or conv.model

    try:
        adapter = registry.get_adapter(provider_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build messages
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for m in conv.messages:
        messages.append(ChatMessage(role=m.role, content=m.content))
    messages.append(ChatMessage(role="user", content=request.content))

    # Save user message
    user_msg = ChatMessageModel(
        conversation_id=conv.id, role="user", content=request.content
    )
    db.add(user_msg)
    await db.commit()

    async def generate():
        full_content = ""
        try:
            async for chunk in adapter.chat_stream(messages=messages, model=model):
                if chunk.delta_content:
                    full_content += chunk.delta_content
                    yield f"data: {json.dumps({'content': chunk.delta_content})}\n\n"
                if chunk.finish_reason:
                    yield f"data: {json.dumps({'finish_reason': chunk.finish_reason})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Save assistant response after streaming
        assistant_msg = ChatMessageModel(
            conversation_id=conv.id,
            role="assistant",
            content=full_content,
            provider=provider_name,
            model=model,
        )
        db.add(assistant_msg)
        conv.updated_at = datetime.now(UTC)
        if not conv.title and request.content:
            conv.title = request.content[:100]
        await db.commit()
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# -- Suggestions endpoint ------------------------------------------------------


@router.get("/conversations/{conversation_id}/suggestions")
async def get_suggestions(
    conversation_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SuggestedQueriesResponse:
    """Get contextual query suggestions."""
    result = await db.execute(
        select(ChatConversation).where(ChatConversation.id == conversation_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

    return SuggestedQueriesResponse(
        suggestions=[
            "How many meetings happened this week?",
            "Show me translation statistics by language",
            "Who are the most active speakers?",
            "What's the diarization status?",
            "Search transcripts for 'project update'",
            "Show usage trends for the last 30 days",
        ]
    )


# -- Helpers -------------------------------------------------------------------


def _conv_to_response(conv: ChatConversation, msg_count: int | None = None) -> ConversationResponse:
    if msg_count is None:
        try:
            msg_count = len(conv.messages) if conv.messages else 0
        except Exception:
            msg_count = 0
    return ConversationResponse(
        id=str(conv.id),
        title=conv.title,
        provider=conv.provider,
        model=conv.model,
        message_count=msg_count,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def _msg_to_response(msg: ChatMessageModel) -> MessageResponse:
    tool_calls = None
    if msg.tool_calls:
        tool_calls = [ToolCallInfo(**tc) for tc in msg.tool_calls]
    return MessageResponse(
        id=str(msg.id),
        conversation_id=str(msg.conversation_id),
        role=msg.role,
        content=msg.content,
        tool_calls=tool_calls,
        model=msg.model,
        provider=msg.provider,
        tokens_used=msg.tokens_used,
        created_at=msg.created_at,
    )
