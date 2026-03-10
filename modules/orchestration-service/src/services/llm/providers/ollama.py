"""Ollama LLM adapter using OpenAI-compatible API."""

import json
from typing import AsyncIterator

import httpx
from livetranslate_common.logging import get_logger
from openai import AsyncOpenAI

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


class OllamaAdapter(LLMAdapter):
    """LLM adapter for local Ollama instances via OpenAI-compatible API."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1", api_key="ollama"
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
    ) -> ChatResponse:
        model = model or self.default_model
        kwargs = self._build_kwargs(messages, model, temperature, max_tokens, tools)
        response = await self._client.chat.completions.create(**kwargs)
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
        kwargs = self._build_kwargs(messages, model, temperature, max_tokens, tools)
        kwargs["stream"] = True
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                yield StreamChunk(
                    delta_content=delta.content if delta else None,
                    finish_reason=choice.finish_reason,
                )

    async def list_models(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/api/tags", timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    ModelInfo(id=m["name"], name=m["name"], provider="ollama")
                    for m in data.get("models", [])
                ]
        except Exception as e:
            logger.warning("ollama_list_models_failed", error=str(e))
            return []

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/api/tags", timeout=5
                )
                return resp.status_code == 200
        except Exception:
            return False

    def _build_kwargs(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[ToolDefinition] | None,
    ) -> dict:
        formatted = [self._format_message(m) for m in messages]
        kwargs: dict = {
            "model": model,
            "messages": formatted,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [self._format_tool(t) for t in tools]
        return kwargs

    @staticmethod
    def _format_message(msg: ChatMessage) -> dict:
        d: dict = {"role": msg.role, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.role == "tool" and msg.name:
            d["name"] = msg.name
        return d

    @staticmethod
    def _format_tool(tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _parse_response(self, response) -> ChatResponse:
        choice = response.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(args)
                        if isinstance(args, str)
                        else args,
                    )
                )
        usage = UsageInfo(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens
            if response.usage
            else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
        return ChatResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason,
        )
