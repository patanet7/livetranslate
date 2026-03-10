"""OpenAI-compatible LLM adapter for third-party providers."""

from typing import AsyncIterator

from livetranslate_common.logging import get_logger
from openai import AsyncOpenAI

from ..adapter import (
    ChatMessage,
    ModelInfo,
    StreamChunk,
    ToolDefinition,
)
from .ollama import OllamaAdapter

logger = get_logger()


class OpenAICompatAdapter(OllamaAdapter):
    """Generic adapter for any OpenAI-compatible API endpoint.

    Works with vLLM, Groq, Together AI, LM Studio, etc.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        default_model: str = "default",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1", api_key=api_key or "none"
        )

    async def list_models(self) -> list[ModelInfo]:
        try:
            models = await self._client.models.list()
            return [
                ModelInfo(
                    id=m.id, name=m.id, provider=self.provider_name
                )
                for m in models.data
            ]
        except Exception as e:
            logger.warning(
                "openai_compat_list_models_failed", error=str(e)
            )
            return []

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
