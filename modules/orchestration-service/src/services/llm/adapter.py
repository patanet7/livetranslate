"""Abstract LLM adapter interface and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class ToolDefinition:
    """JSON Schema tool definition for LLM function calling."""

    name: str
    description: str
    parameters: dict  # JSON Schema object


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages
    name: str | None = None  # Tool name for tool responses


@dataclass
class UsageInfo:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Complete chat response from LLM."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: UsageInfo = field(default_factory=UsageInfo)
    finish_reason: str | None = None


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    delta_content: str | None = None
    delta_tool_call: ToolCall | None = None
    finish_reason: str | None = None


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    provider: str
    context_window: int | None = None


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    All providers (Ollama, OpenAI, Anthropic, etc.) implement this interface.
    """

    provider_name: str = "unknown"

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
    ) -> ChatResponse:
        """Send a chat request and get a complete response."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send a chat request and stream the response."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models for this provider."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable and working."""
        ...
