"""Canonical value objects for talking to an LLM endpoint.

`LLMConnection` is the single shape every service uses to identify *which*
LLM to call and *how*. It replaces three pre-existing config systems:

- env-driven `TranslationConfig` (live translation hot path)
- env-driven `MeetingIntelligenceSettings.direct_llm_*` (intelligence)
- DB-driven `AIConnection` (dashboard Connections UI)

`LLMParameterOverrides` carries per-session sampling tunables that flow from
the dashboard toolbar to the LLM request body.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LLMEngine = Literal["ollama", "openai", "openai_compatible", "anthropic", "vllm"]


class LLMConnection(BaseModel):
    """Immutable description of an LLM endpoint + default sampling parameters.

    The resolver normalizes inputs from any source (DB row, env vars, hard-coded
    defaults) into this shape. The merged `LLMClient` takes one of these in its
    constructor; per-call `LLMParameterOverrides` are layered via `merge()`.
    """

    # extra="ignore" — silently drop unknown fields so future per-model manifest
    # additions (e.g., system prompt overrides, tool schemas, modality hints)
    # don't break older callers or stored connection rows mid-rollout.
    model_config = ConfigDict(frozen=True, extra="ignore")

    engine: LLMEngine
    base_url: str
    model: str

    api_key: str = ""

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    repetition_penalty: float = Field(default=1.05, ge=0.0)
    presence_penalty: float = Field(default=1.5, ge=-2.0, le=2.0)

    timeout_s: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=1, ge=0)

    context_length: int | None = None
    connection_id: str | None = None

    @property
    def is_ollama_qwen3(self) -> bool:
        """True when this connection is Qwen3 on Ollama.

        Replaces the brittle `"11434" in url` port heuristic that broke when
        Ollama ran on non-default ports (e.g., Tailscale-hosted vLLM proxy
        at 100.64.0.2:8089). The Qwen3 on Ollama OpenAI-compat layer drops the
        response into a "reasoning" field instead of "content", so callers need
        the native /api/generate path. This property gates that fallback.
        """
        if self.engine != "ollama":
            return False
        return "qwen3" in self.model.lower()

    def merge(self, overrides: LLMParameterOverrides | None) -> LLMConnection:
        """Return a new LLMConnection with non-None override fields applied.

        Pure-functional: never mutates self (frozen=True enforces it anyway).
        Caller pattern in the translation hot path is:
            effective = self._base_connection.merge(session_overrides)
            await client.chat(messages, sampling=effective)
        """
        if overrides is None:
            return self
        patch = overrides.model_dump(exclude_none=True)
        # connection_id is a routing field for the resolver, not a sampling
        # override applied to the current connection; drop it here.
        patch.pop("connection_id", None)
        if not patch:
            return self
        return self.model_copy(update=patch)


class LLMParameterOverrides(BaseModel):
    """Per-session, per-call optional overrides for LLM sampling parameters.

    Flows from the dashboard toolbar → WS ConfigMessage.llm → SessionConfig →
    TranslationService.translate(...). Every field is optional; only specified
    fields override the base LLMConnection.

    `connection_id` and `model` exist here so the user can swap backend or
    model mid-session without resetting the WebSocket.
    """

    model_config = ConfigDict(extra="ignore")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, ge=0.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    timeout_s: float | None = Field(default=None, gt=0.0)

    model: str | None = None
    connection_id: str | None = None


# Allow forward reference resolution for merge's type hint.
LLMConnection.model_rebuild()
