"""
Rolling Window Translator

Context-aware translation service that maintains rolling context windows
for each speaker, enabling more accurate translation through:
- Previous sentence context (resolves pronouns, references)
- Glossary term injection (consistent technical terminology)
- Speaker-specific context tracking

The key insight: provide context for understanding, but only translate
the current sentence. This avoids re-translating previous content while
maintaining coherent translations.

Reference: FIREFLIES_ADAPTATION_PLAN.md Section "Rolling Window Translation"
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from clients.simple_translation_client import (
    SimpleTranslationClient,
)
from clients.translation_service_client import (
    TranslationRequest,
    TranslationServiceClient,
)
from models.fireflies import (
    TranslationContext,
    TranslationResult,
    TranslationUnit,
)
from services.translation_prompt_builder import (
    PromptContext,
    TranslationPromptBuilder,
    create_prompt_builder,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SpeakerContext:
    """Context window for a single speaker."""

    speaker_name: str
    sentences: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    translation_count: int = 0
    last_translation_time: datetime | None = None

    def add(self, sentence: str) -> None:
        """Add a sentence to this speaker's context."""
        self.sentences.append(sentence)
        self.translation_count += 1
        self.last_translation_time = datetime.now(UTC)

    def get_context(self, max_sentences: int = 3) -> list[str]:
        """Get recent context sentences."""
        return list(self.sentences)[-max_sentences:]

    def clear(self) -> None:
        """Clear the context window."""
        self.sentences.clear()


@dataclass
class SessionTranslationState:
    """Translation state for a Fireflies session."""

    session_id: str
    speaker_contexts: dict[str, SpeakerContext] = field(default_factory=dict)
    global_context: deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=10)
    )  # (speaker, sentence)
    total_translations: int = 0
    total_errors: int = 0
    average_translation_time_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_speaker_context(self, speaker_name: str) -> SpeakerContext:
        """Get or create context for a speaker."""
        if speaker_name not in self.speaker_contexts:
            self.speaker_contexts[speaker_name] = SpeakerContext(speaker_name=speaker_name)
        return self.speaker_contexts[speaker_name]

    def add_to_global_context(self, speaker: str, sentence: str) -> None:
        """Add to the global cross-speaker context."""
        self.global_context.append((speaker, sentence))

    def get_cross_speaker_context(self, exclude_speaker: str, max_sentences: int = 3) -> list[str]:
        """Get recent sentences from other speakers."""
        other_sentences = [
            f"[{speaker}] {sentence}"
            for speaker, sentence in self.global_context
            if speaker != exclude_speaker
        ]
        return other_sentences[-max_sentences:]

    def update_stats(self, translation_time_ms: float, success: bool) -> None:
        """Update translation statistics."""
        if success:
            n = self.total_translations
            self.average_translation_time_ms = (
                self.average_translation_time_ms * n + translation_time_ms
            ) / (n + 1)
            self.total_translations += 1
        else:
            self.total_errors += 1


# =============================================================================
# Rolling Window Translator
# =============================================================================


class RollingWindowTranslator:
    """
    Context-aware translator with rolling window for coherent translations.

    This translator maintains separate context windows for each speaker
    and optionally includes cross-speaker context. It injects glossary
    terms into translation prompts to ensure consistent terminology.

    Usage:
        translator = RollingWindowTranslator(
            translation_client=translation_client,
            glossary_service=glossary_service,  # Optional
            window_size=3,
        )

        result = await translator.translate(
            unit=translation_unit,
            target_language="es",
            glossary_id=glossary_uuid,
        )
    """

    def __init__(
        self,
        translation_client: TranslationServiceClient,
        glossary_service=None,
        window_size: int = 3,
        include_cross_speaker_context: bool = True,
        max_cross_speaker_sentences: int = 2,
        use_prompt_based_translation: bool = True,
        simple_client: SimpleTranslationClient | None = None,
        prompt_builder: TranslationPromptBuilder | None = None,
    ):
        """
        Initialize the rolling window translator.

        Args:
            translation_client: Client for the translation service (legacy)
            glossary_service: Optional GlossaryService for term injection
            window_size: Number of previous sentences to include in context
            include_cross_speaker_context: Include other speakers' sentences
            max_cross_speaker_sentences: Max other-speaker sentences to include
            use_prompt_based_translation: Use LLM prompt with context (vs direct)
            simple_client: Optional SimpleTranslationClient for V3 API (recommended)
            prompt_builder: Optional PromptBuilder for building prompts
        """
        self.translation_client = translation_client
        self.glossary_service = glossary_service
        self.window_size = window_size
        self.include_cross_speaker_context = include_cross_speaker_context
        self.max_cross_speaker_sentences = max_cross_speaker_sentences
        self.use_prompt_based_translation = use_prompt_based_translation

        # V3 API components (prompt-based translation)
        self.simple_client = simple_client
        self.prompt_builder = prompt_builder or create_prompt_builder()

        # Session state tracking
        self._sessions: dict[str, SessionTranslationState] = {}

        logger.info(
            f"RollingWindowTranslator initialized: "
            f"window_size={window_size}, "
            f"cross_speaker={include_cross_speaker_context}, "
            f"v3_api={'enabled' if simple_client else 'disabled'}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    async def translate(
        self,
        unit: TranslationUnit,
        target_language: str,
        glossary_id: UUID | None = None,
        domain: str | None = None,
        source_language: str = "en",
    ) -> TranslationResult:
        """
        Translate a sentence with rolling context window.

        Args:
            unit: TranslationUnit containing the sentence to translate
            target_language: Target language code (e.g., 'es', 'fr')
            glossary_id: Optional glossary ID for term injection
            domain: Optional domain for glossary filtering
            source_language: Source language code (default: 'en')

        Returns:
            TranslationResult with original, translated text, and metadata
        """
        start_time = time.time()

        # Get or create session state
        session_state = self._get_session_state(unit.session_id)
        speaker_context = session_state.get_speaker_context(unit.speaker_name)

        try:
            # Build translation context
            context = await self._build_context(
                unit=unit,
                session_state=session_state,
                speaker_context=speaker_context,
                target_language=target_language,
                glossary_id=glossary_id,
                domain=domain,
                source_language=source_language,
            )

            # Perform translation
            if self.use_prompt_based_translation:
                translated_text, confidence = await self._translate_with_prompt(
                    text=unit.text,
                    context=context,
                )
            else:
                translated_text, confidence = await self._translate_direct(
                    text=unit.text,
                    target_language=target_language,
                    source_language=source_language,
                )

            # Calculate translation time
            translation_time_ms = (time.time() - start_time) * 1000

            # Update context windows AFTER successful translation
            speaker_context.add(unit.text)
            session_state.add_to_global_context(unit.speaker_name, unit.text)
            session_state.update_stats(translation_time_ms, success=True)

            # Find which glossary terms were applied
            glossary_terms_applied = self._find_applied_glossary_terms(
                original=unit.text,
                translated=translated_text,
                glossary_terms=context.glossary,
            )

            result = TranslationResult(
                original=unit.text,
                translated=translated_text,
                speaker_name=unit.speaker_name,
                source_language=source_language,
                target_language=target_language,
                confidence=confidence,
                context_sentences_used=len(context.previous_sentences),
                glossary_terms_applied=glossary_terms_applied,
                translation_time_ms=translation_time_ms,
                session_id=unit.session_id,
                translation_unit_id=f"{unit.transcript_id}_{unit.start_time}",
            )

            logger.debug(
                f"Translated: '{unit.text[:50]}...' -> '{translated_text[:50]}...' "
                f"({translation_time_ms:.1f}ms, {len(glossary_terms_applied)} terms)"
            )

            return result

        except Exception as e:
            translation_time_ms = (time.time() - start_time) * 1000
            session_state.update_stats(translation_time_ms, success=False)

            logger.error(f"Translation failed: {e}", exc_info=True)

            # Return error result with original text
            return TranslationResult(
                original=unit.text,
                translated=f"[Translation Error: {str(e)[:50]}]",
                speaker_name=unit.speaker_name,
                source_language=source_language,
                target_language=target_language,
                confidence=0.0,
                context_sentences_used=0,
                glossary_terms_applied=[],
                translation_time_ms=translation_time_ms,
                session_id=unit.session_id,
            )

    async def translate_batch(
        self,
        units: list[TranslationUnit],
        target_language: str,
        glossary_id: UUID | None = None,
        domain: str | None = None,
        source_language: str = "en",
        max_concurrency: int = 5,
    ) -> list[TranslationResult]:
        """
        Translate multiple units, maintaining order for context.

        Note: Units are translated sequentially to maintain proper context
        window state. For parallel translation use translate_batch_parallel.

        Args:
            units: List of TranslationUnits to translate
            target_language: Target language code
            glossary_id: Optional glossary ID
            domain: Optional domain filter
            source_language: Source language code
            max_concurrency: Not used in sequential mode

        Returns:
            List of TranslationResults in same order as input
        """
        results = []
        for unit in units:
            result = await self.translate(
                unit=unit,
                target_language=target_language,
                glossary_id=glossary_id,
                domain=domain,
                source_language=source_language,
            )
            results.append(result)
        return results

    async def translate_to_multiple_languages(
        self,
        unit: TranslationUnit,
        target_languages: list[str],
        glossary_id: UUID | None = None,
        domain: str | None = None,
        source_language: str = "en",
    ) -> dict[str, TranslationResult]:
        """
        Translate a single sentence to multiple target languages.

        Args:
            unit: TranslationUnit to translate
            target_languages: List of target language codes
            glossary_id: Optional glossary ID
            domain: Optional domain filter
            source_language: Source language code

        Returns:
            Dict mapping language code to TranslationResult
        """
        results: dict[str, TranslationResult] = {}

        # Translate to each language (can be parallelized since context
        # is only updated once per source sentence)
        tasks = [
            self.translate(
                unit=unit,
                target_language=lang,
                glossary_id=glossary_id,
                domain=domain,
                source_language=source_language,
            )
            for lang in target_languages
        ]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for lang, result in zip(target_languages, completed, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Translation to {lang} failed: {result}")
                results[lang] = TranslationResult(
                    original=unit.text,
                    translated=f"[Error: {str(result)[:50]}]",
                    speaker_name=unit.speaker_name,
                    source_language=source_language,
                    target_language=lang,
                    confidence=0.0,
                    session_id=unit.session_id,
                )
            else:
                results[lang] = result

        return results

    def get_session_stats(self, session_id: str) -> dict | None:
        """
        Get translation statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dict with stats or None if session not found
        """
        if session_id not in self._sessions:
            return None

        state = self._sessions[session_id]
        return {
            "session_id": session_id,
            "total_translations": state.total_translations,
            "total_errors": state.total_errors,
            "error_rate": (
                state.total_errors / max(1, state.total_translations + state.total_errors)
            ),
            "average_translation_time_ms": state.average_translation_time_ms,
            "speakers": list(state.speaker_contexts.keys()),
            "speaker_translation_counts": {
                name: ctx.translation_count for name, ctx in state.speaker_contexts.items()
            },
            "created_at": state.created_at.isoformat(),
        }

    def clear_session(self, session_id: str) -> bool:
        """
        Clear all context for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            True if cleared, False if session not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session context: {session_id}")
            return True
        return False

    def clear_speaker_context(self, session_id: str, speaker_name: str) -> bool:
        """
        Clear context for a specific speaker in a session.

        Args:
            session_id: Session ID
            speaker_name: Speaker name to clear

        Returns:
            True if cleared, False if not found
        """
        if session_id not in self._sessions:
            return False

        state = self._sessions[session_id]
        if speaker_name in state.speaker_contexts:
            state.speaker_contexts[speaker_name].clear()
            logger.info(f"Cleared speaker context: session={session_id}, speaker={speaker_name}")
            return True
        return False

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_session_state(self, session_id: str) -> SessionTranslationState:
        """Get or create session translation state."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionTranslationState(session_id=session_id)
        return self._sessions[session_id]

    async def _build_context(
        self,
        unit: TranslationUnit,
        session_state: SessionTranslationState,
        speaker_context: SpeakerContext,
        target_language: str,
        glossary_id: UUID | None,
        domain: str | None,
        source_language: str,
    ) -> TranslationContext:
        """Build translation context with previous sentences and glossary."""
        # Get speaker's previous sentences
        previous_sentences = speaker_context.get_context(self.window_size)

        # Optionally add cross-speaker context
        if self.include_cross_speaker_context:
            cross_speaker = session_state.get_cross_speaker_context(
                exclude_speaker=unit.speaker_name,
                max_sentences=self.max_cross_speaker_sentences,
            )
            # Prepend cross-speaker context (older context first)
            previous_sentences = cross_speaker + previous_sentences

        # Get glossary terms
        glossary_terms: dict[str, str] = {}
        if self.glossary_service and glossary_id:
            try:
                glossary_terms = self.glossary_service.get_glossary_terms(
                    glossary_id=glossary_id,
                    target_language=target_language,
                    domain=domain,
                    include_default=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load glossary terms: {e}")

        return TranslationContext(
            previous_sentences=previous_sentences,
            glossary=glossary_terms,
            target_language=target_language,
            source_language=source_language,
        )

    async def _translate_with_prompt(
        self,
        text: str,
        context: TranslationContext,
    ) -> tuple[str, float]:
        """
        Translate using LLM prompt with context.

        Uses the V3 API if simple_client is available, otherwise falls back
        to legacy API (which doesn't actually use the prompt).

        Returns:
            Tuple of (translated_text, confidence)
        """
        # Build the prompt using PromptBuilder
        prompt_context = PromptContext(
            current_sentence=text,
            target_language=context.target_language,
            source_language=context.source_language,
            previous_sentences=context.previous_sentences,
            glossary_terms=context.glossary,
        )

        built_prompt = self.prompt_builder.build(prompt_context)

        logger.debug(
            f"Built prompt: template={built_prompt.template_used}, "
            f"context_sentences={built_prompt.context_sentence_count}, "
            f"glossary_terms={built_prompt.glossary_term_count}"
        )

        # Use V3 API if available (sends the actual prompt)
        if self.simple_client:
            try:
                result = await self.simple_client.translate_prompt(
                    prompt=built_prompt.prompt,
                    backend="ollama",
                )
                # V3 API doesn't return confidence, estimate based on success
                confidence = 0.9 if result.text else 0.5
                return result.text, confidence
            except Exception as e:
                logger.warning(f"V3 API failed, falling back to legacy: {e}")
                # Fall through to legacy API

        # Legacy fallback - NOTE: this still doesn't send the prompt!
        # This is kept for backwards compatibility but the V3 API should be used
        logger.warning(
            "Using legacy translation API - prompt context/glossary will NOT be applied! "
            "Configure simple_client to use V3 API for full functionality."
        )

        request = TranslationRequest(
            text=text,
            source_language=context.source_language,
            target_language=context.target_language,
            quality="balanced",
        )

        response = await self.translation_client.translate(request)

        return response.translated_text, response.confidence

    async def _translate_direct(
        self,
        text: str,
        target_language: str,
        source_language: str,
    ) -> tuple[str, float]:
        """
        Translate directly without context injection.

        Returns:
            Tuple of (translated_text, confidence)
        """
        request = TranslationRequest(
            text=text,
            source_language=source_language,
            target_language=target_language,
            quality="balanced",
        )

        response = await self.translation_client.translate(request)

        return response.translated_text, response.confidence

    def _find_applied_glossary_terms(
        self,
        original: str,
        translated: str,
        glossary_terms: dict[str, str],
    ) -> list[str]:
        """
        Find which glossary terms were likely applied.

        We check if the source term was in the original text and
        the target term appears in the translation.

        Args:
            original: Original text
            translated: Translated text
            glossary_terms: Dict of source -> target terms

        Returns:
            List of source terms that were applied
        """
        applied = []
        original_lower = original.lower()
        translated_lower = translated.lower()

        for source_term, target_term in glossary_terms.items():
            # Check if source term is in original
            if source_term.lower() in original_lower:
                # Check if target term is in translation
                if target_term.lower() in translated_lower:
                    applied.append(source_term)

        return applied


# =============================================================================
# Factory Function
# =============================================================================


def create_rolling_window_translator(
    translation_client: TranslationServiceClient,
    glossary_service=None,
    config: dict | None = None,
    simple_client: SimpleTranslationClient | None = None,
    prompt_builder: TranslationPromptBuilder | None = None,
) -> RollingWindowTranslator:
    """
    Factory function to create a RollingWindowTranslator with configuration.

    Args:
        translation_client: TranslationServiceClient instance (legacy, fallback)
        glossary_service: Optional GlossaryService instance
        config: Optional configuration dict with keys:
            - window_size: int (default: 3)
            - include_cross_speaker_context: bool (default: True)
            - max_cross_speaker_sentences: int (default: 2)
            - use_prompt_based_translation: bool (default: True)
        simple_client: Optional SimpleTranslationClient for V3 API (recommended)
        prompt_builder: Optional TranslationPromptBuilder for building prompts

    Returns:
        Configured RollingWindowTranslator instance

    Note:
        For full context/glossary support, provide a simple_client configured
        to use the V3 translation API. Without it, prompts are built but not
        actually sent to the LLM.
    """
    config = config or {}

    return RollingWindowTranslator(
        translation_client=translation_client,
        glossary_service=glossary_service,
        window_size=config.get("window_size", 3),
        include_cross_speaker_context=config.get("include_cross_speaker_context", True),
        max_cross_speaker_sentences=config.get("max_cross_speaker_sentences", 2),
        use_prompt_based_translation=config.get("use_prompt_based_translation", True),
        simple_client=simple_client,
        prompt_builder=prompt_builder,
    )
