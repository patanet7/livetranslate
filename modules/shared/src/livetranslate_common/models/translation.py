"""Translation-related shared Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranslationContext(BaseModel):
    """A single prior exchange used as rolling context for translation.

    Args:
        text: The original source text.
        translation: The corresponding translation.
    """

    text: str
    translation: str


class TranslationRequest(BaseModel):
    """Request payload sent to the translation service.

    Args:
        text: Source text to translate.
        source_language: BCP-47 code of the source language.
        target_language: BCP-47 code of the target language.
        context: Prior source/translation pairs for context conditioning.
        context_window_size: Maximum number of context pairs to include.
        max_context_tokens: Token budget for context injection.
        glossary_terms: Domain-specific term overrides {source: target}.
        speaker_name: Optional speaker name for personalised prompts.
    """

    text: str
    source_language: str
    target_language: str
    context: list[TranslationContext] = Field(default_factory=list)
    context_window_size: int = 5
    max_context_tokens: int = 500
    glossary_terms: dict[str, str] = Field(default_factory=dict)
    speaker_name: str | None = None


class TranslationResponse(BaseModel):
    """Response returned by the translation service.

    Args:
        translated_text: The translated output text.
        source_language: BCP-47 code of the detected/specified source language.
        target_language: BCP-47 code of the target language.
        model_used: Identifier of the model that produced this translation.
        latency_ms: Wall-clock time taken for the translation in milliseconds.
        quality_score: Optional quality estimate in [0.0, 1.0].
    """

    translated_text: str
    source_language: str
    target_language: str
    model_used: str
    latency_ms: float
    quality_score: float | None = None
