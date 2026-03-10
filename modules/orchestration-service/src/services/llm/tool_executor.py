"""Tool executor — orchestrates LLM chat with tool-calling."""

import json
from typing import Any, Callable

from livetranslate_common.logging import get_logger

from .adapter import ChatMessage, ChatResponse, LLMAdapter, ToolCall, ToolDefinition

logger = get_logger()

MAX_TOOL_ROUNDS = 3


class ToolExecutor:
    """Manages tool definitions and executes tool calls from LLM responses."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
    ) -> None:
        """Register a tool with its handler function."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._handlers[name] = handler

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Get all registered tool definitions for LLM."""
        return list(self._tools.values())

    async def execute_tool(self, tool_call: ToolCall, **extra_kwargs: Any) -> str:
        """Execute a single tool call and return the result as a string."""
        handler = self._handlers.get(tool_call.name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_call.name}"})
        try:
            result = await handler(**tool_call.arguments, **extra_kwargs)
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_call.name, error=str(e))
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    async def run_chat_with_tools(
        self,
        adapter: LLMAdapter,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **tool_kwargs: Any,
    ) -> tuple[ChatResponse, list[ChatMessage]]:
        """Run a chat with tool-calling loop.

        Returns the final response and the full message history.
        Tool calls are automatically executed up to MAX_TOOL_ROUNDS times.
        """
        tools = self.get_tool_definitions()
        all_messages = list(messages)

        for round_num in range(MAX_TOOL_ROUNDS):
            response = await adapter.chat(
                messages=all_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools if tools else None,
            )

            # If no tool calls, we're done
            if not response.tool_calls:
                return response, all_messages

            logger.info(
                "tool_calls_received",
                round=round_num + 1,
                count=len(response.tool_calls),
                tools=[tc.name for tc in response.tool_calls],
            )

            # Add assistant message with tool calls
            all_messages.append(ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))

            # Execute each tool call and add results
            for tc in response.tool_calls:
                result = await self.execute_tool(tc, **tool_kwargs)
                all_messages.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Max rounds reached, do final call without tools
        logger.warning("max_tool_rounds_reached", max_rounds=MAX_TOOL_ROUNDS)
        response = await adapter.chat(
            messages=all_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response, all_messages
