"""
Translation Prompt Builder

Responsible for building complete LLM prompts with:
- Rolling context windows (previous sentences)
- Glossary term injection (consistent terminology)
- Target language instructions
- Speaker attribution

The built prompt is sent directly to the translation service,
which just forwards it to the LLM without modification.

Reference: API Contract - Translation Service is DUMB
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

# Full context prompt - includes glossary, context window, speaker info
TRANSLATION_PROMPT_TEMPLATE = """You are a professional real-time translator.

Target Language: {target_language}

{glossary_section}

Previous context (DO NOT translate, only use for understanding references):
{context_window}

---

Translate ONLY the following sentence to {target_language}:
{current_sentence}

Translation:"""

# Simple prompt - no context or glossary
SIMPLE_TRANSLATION_PROMPT = """Translate to {target_language}:
{current_sentence}

Translation:"""

# Minimal prompt - just the text and language
MINIMAL_PROMPT = """Translate to {target_language}: {current_sentence}"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PromptContext:
    """Context for building a translation prompt."""

    current_sentence: str
    target_language: str
    source_language: str = "auto"
    previous_sentences: Optional[List[str]] = None
    glossary_terms: Optional[Dict[str, str]] = None
    speaker_name: Optional[str] = None


@dataclass
class BuiltPrompt:
    """Result of building a translation prompt."""

    prompt: str
    has_context: bool
    has_glossary: bool
    context_sentence_count: int
    glossary_term_count: int
    template_used: str


# =============================================================================
# Translation Prompt Builder
# =============================================================================

class TranslationPromptBuilder:
    """
    Builds complete LLM prompts for translation.

    This class is responsible for:
    1. Formatting glossary terms
    2. Formatting context windows
    3. Choosing appropriate template
    4. Building the final prompt

    The built prompt contains ALL the intelligence - the translation service
    just sends it to the LLM as-is.

    Usage:
        builder = TranslationPromptBuilder()

        result = builder.build(PromptContext(
            current_sentence="Hello world",
            target_language="es",
            previous_sentences=["Good morning", "How are you?"],
            glossary_terms={"API": "API", "endpoint": "punto de acceso"},
        ))

        # result.prompt is ready to send to translation service
    """

    def __init__(
        self,
        full_template: str = TRANSLATION_PROMPT_TEMPLATE,
        simple_template: str = SIMPLE_TRANSLATION_PROMPT,
        minimal_template: str = MINIMAL_PROMPT,
        use_minimal_when_empty: bool = False,
    ):
        """
        Initialize the prompt builder.

        Args:
            full_template: Template with context and glossary
            simple_template: Template without context
            minimal_template: Most minimal template
            use_minimal_when_empty: Use minimal template when no context
        """
        self.full_template = full_template
        self.simple_template = simple_template
        self.minimal_template = minimal_template
        self.use_minimal_when_empty = use_minimal_when_empty

    def build(self, context: PromptContext) -> BuiltPrompt:
        """
        Build a complete translation prompt from context.

        Args:
            context: PromptContext with sentence, language, and optional context/glossary

        Returns:
            BuiltPrompt with the ready-to-send prompt
        """
        previous = context.previous_sentences or []
        glossary = context.glossary_terms or {}

        has_context = len(previous) > 0
        has_glossary = len(glossary) > 0

        # Choose template and build prompt
        if has_context or has_glossary:
            prompt = self._build_full_prompt(context, previous, glossary)
            template_used = "full"
        elif self.use_minimal_when_empty:
            prompt = self._build_minimal_prompt(context)
            template_used = "minimal"
        else:
            prompt = self._build_simple_prompt(context)
            template_used = "simple"

        return BuiltPrompt(
            prompt=prompt,
            has_context=has_context,
            has_glossary=has_glossary,
            context_sentence_count=len(previous),
            glossary_term_count=len(glossary),
            template_used=template_used,
        )

    def _build_full_prompt(
        self,
        context: PromptContext,
        previous_sentences: List[str],
        glossary_terms: Dict[str, str],
    ) -> str:
        """Build prompt with context and glossary."""
        # Format glossary section
        glossary_section = ""
        if glossary_terms:
            glossary_lines = [
                f"- {source} = {target}"
                for source, target in glossary_terms.items()
            ]
            glossary_section = (
                "Glossary (use these exact translations):\n"
                + "\n".join(glossary_lines)
            )

        # Format context window
        context_window = self._format_context_window(previous_sentences)

        return self.full_template.format(
            target_language=context.target_language,
            glossary_section=glossary_section,
            context_window=context_window,
            current_sentence=context.current_sentence,
        )

    def _build_simple_prompt(self, context: PromptContext) -> str:
        """Build simple prompt without context."""
        return self.simple_template.format(
            target_language=context.target_language,
            current_sentence=context.current_sentence,
        )

    def _build_minimal_prompt(self, context: PromptContext) -> str:
        """Build minimal prompt."""
        return self.minimal_template.format(
            target_language=context.target_language,
            current_sentence=context.current_sentence,
        )

    def _format_context_window(self, sentences: List[str]) -> str:
        """Format previous sentences as context window."""
        if not sentences:
            return "(no previous context)"

        # Format each sentence, numbered for clarity
        formatted = []
        for i, sentence in enumerate(sentences, 1):
            formatted.append(f"{i}. {sentence}")

        return "\n".join(formatted)


# =============================================================================
# Factory Functions
# =============================================================================

def create_prompt_builder(
    config: Optional[Dict] = None,
) -> TranslationPromptBuilder:
    """
    Create a TranslationPromptBuilder with optional configuration.

    Args:
        config: Optional dict with:
            - full_template: Custom full template
            - simple_template: Custom simple template
            - minimal_template: Custom minimal template
            - use_minimal_when_empty: bool

    Returns:
        Configured TranslationPromptBuilder
    """
    config = config or {}

    return TranslationPromptBuilder(
        full_template=config.get("full_template", TRANSLATION_PROMPT_TEMPLATE),
        simple_template=config.get("simple_template", SIMPLE_TRANSLATION_PROMPT),
        minimal_template=config.get("minimal_template", MINIMAL_PROMPT),
        use_minimal_when_empty=config.get("use_minimal_when_empty", False),
    )
