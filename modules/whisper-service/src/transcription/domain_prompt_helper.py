#!/usr/bin/env python3
"""
Domain Prompt Helper

Prepares domain-specific prompts for Whisper transcription.
Extracted from whisper_service.py for better modularity and testability.
"""

from livetranslate_common.logging import get_logger

logger = get_logger()


def prepare_domain_prompt(
    domain: str | None = None,
    custom_terms: list | None = None,
    previous_context: str | None = None,
    initial_prompt: str | None = None,
) -> str | None:
    """
    Prepare domain-specific prompt for Whisper transcription.

    Handles:
    - Provided initial prompts (highest priority)
    - Domain-based prompt generation
    - Custom terminology integration
    - Previous context carryover

    Args:
        domain: Domain type (e.g., 'medical', 'legal', 'technical')
        custom_terms: List of custom terminology to include
        previous_context: Previous transcription context for continuity
        initial_prompt: Pre-defined initial prompt (overrides generation)

    Returns:
        Prepared prompt string, or None if no prompting needed
    """
    # Quick exit if nothing to do
    if not (domain or custom_terms or previous_context or initial_prompt):
        return None

    try:
        # If explicit initial_prompt provided, use it directly
        if initial_prompt:
            logger.info("[DOMAIN] Using provided initial prompt")
            return initial_prompt

        # Otherwise, generate from domain/terms/context
        from domain_prompt_manager import DomainPromptManager

        # Create domain prompt manager
        domain_mgr = DomainPromptManager()

        # Generate prompt from domain, terms, and context
        generated_prompt = domain_mgr.create_domain_prompt(
            domain=domain, custom_terms=custom_terms, previous_context=previous_context
        )

        logger.info(f"[DOMAIN] Generated prompt: {len(generated_prompt)} chars, domain={domain}")
        return generated_prompt

    except Exception as e:
        logger.warning(f"[DOMAIN] Failed to create prompt: {e}")
        # Fall back to basic initial_prompt if provided
        return initial_prompt
