"""Anthropic LLM adapter."""

import json
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from livetranslate_common.logging import get_logger

from ..adapter import (
    ChatMessage,
    ChatResponse,
    LLMAdapter,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    UsageInfo,
)

logger = get_logger()

# Known Anthropic models for list_models since the API doesn't have a list endpoint.
_KNOWN_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        context_window=200000,
    ),
    ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="anthropic",
        context_window=200000,
    ),
    ModelInfo(
        id="claude-haiku-35-20241022",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        context_window=200000,
    ),
]


class AnthropicAdapter(LLMAdapter):
    """LLM adapter for the Anthropic Messages API."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-20250514",
    ):
        self.default_model = default_model
        self._client = AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
    ) -> ChatResponse:
        model = model or self.default_model
        system_msg, formatted_msgs = self._split_system(messages)
        kwargs: dict = {
            "model": model,
            "messages": formatted_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [self._format_tool(t) for t in tools]
        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.default_model
        system_msg, formatted_msgs = self._split_system(messages)
        kwargs: dict = {
            "model": model,
            "messages": formatted_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [self._format_tool(t) for t in tools]

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            yield StreamChunk(delta_content=delta.text)
                        elif hasattr(delta, "partial_json"):
                            # Tool use input delta — skip for now
                            pass
                    elif event.type == "message_delta":
                        yield StreamChunk(
                            finish_reason=event.delta.stop_reason
                            if hasattr(event.delta, "stop_reason")
                            else None,
                        )

    async def list_models(self) -> list[ModelInfo]:
        return list(_KNOWN_MODELS)

    async def health_check(self) -> bool:
        try:
            # Minimal API call to verify connectivity.
            await self._client.messages.create(
                model=self.default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _split_system(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict]]:
        """Extract system message and format remaining messages for Anthropic."""
        system_msg: str | None = None
        formatted: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
                continue

            if msg.role == "assistant" and msg.tool_calls:
                # Anthropic represents tool calls as content blocks.
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                formatted.append({"role": "assistant", "content": content_blocks})
            elif msg.role == "tool":
                # Anthropic uses tool_result content blocks in a user message.
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
            else:
                formatted.append(
                    {"role": msg.role, "content": msg.content or ""}
                )

        return system_msg, formatted

    @staticmethod
    def _format_tool(tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    @staticmethod
    def _parse_response(response) -> ChatResponse:
        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text = (content_text or "") + block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                        if isinstance(block.input, dict)
                        else json.loads(block.input),
                    )
                )

        usage = UsageInfo(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens
            + response.usage.output_tokens,
        )

        return ChatResponse(
            content=content_text,
            tool_calls=tool_calls,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
        )
