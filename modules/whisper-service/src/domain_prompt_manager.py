#!/usr/bin/env python3
"""
Domain Prompt Manager for Whisper Large-v3
Phase 2: SimulStreaming Innovation - In-Domain Prompting

Manages domain-specific prompts and terminology for improved transcription accuracy
Target: -40-60% reduction in domain-specific errors

Reference: SimulStreaming paper (IWSLT 2025) - Section 4.2
- Scrolling context window: 448 tokens max
- Domain terminology injection
- Previous context carryover: 223 tokens max
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class DomainPromptConfig:
    """Configuration for domain prompt generation"""
    max_total_tokens: int = 448  # SimulStreaming paper: 448 tokens max
    max_context_tokens: int = 223  # SimulStreaming paper: 223 tokens for previous context
    max_terminology_tokens: int = 225  # Remaining tokens for domain terminology
    separator: str = ". "  # Separator between terms


class DomainPromptManager:
    """
    Manages domain-specific prompts for Whisper Large-v3

    Per SimulStreaming paper (Section 4.2):
    - Uses scrolling context window of 448 tokens maximum
    - Combines domain terminology (225 tokens) + previous context (223 tokens)
    - Reduces domain-specific errors by 40-60%

    Features:
    - Database-backed domain dictionaries (medical, legal, technical, etc.)
    - Custom terminology injection
    - Context carryover from previous transcriptions
    - Token-aware prompt construction
    """

    def __init__(self, config: Optional[DomainPromptConfig] = None, db_session=None, glossary_api_url: Optional[str] = None):
        """
        Initialize domain prompt manager

        Args:
            config: Domain prompt configuration
            db_session: DEPRECATED - no longer used (kept for backwards compatibility)
            glossary_api_url: URL of orchestration service for glossary API (e.g., "http://localhost:3000")
        """
        self.config = config or DomainPromptConfig()
        self.db_session = db_session  # Kept for backwards compatibility but not used

        # Glossary API URL for fetching terms from orchestration service
        import os
        self._glossary_api_url = glossary_api_url or os.getenv("ORCHESTRATION_SERVICE_URL", "http://localhost:3000")

        # Built-in domain dictionaries (fallback if glossary API unavailable)
        self.builtin_domains = self._load_builtin_domains()

        # Scrolling context buffer
        self.context_buffer: List[str] = []

        logger.info(f"DomainPromptManager initialized: max_tokens={self.config.max_total_tokens}, "
                   f"max_context={self.config.max_context_tokens}, glossary_api={self._glossary_api_url}")

    def _load_builtin_domains(self) -> Dict[str, List[str]]:
        """Load built-in domain dictionaries as fallback"""
        return {
            "medical": [
                "diagnosis", "symptoms", "prescription", "patient", "consultation",
                "hypertension", "diabetes", "cardiovascular", "antibiotic", "inflammation",
                "examination", "treatment", "medication", "dosage", "healthcare"
            ],
            "legal": [
                "plaintiff", "defendant", "litigation", "jurisdiction", "compliance",
                "contract", "liability", "statute", "testimony", "precedent",
                "attorney", "court", "evidence", "ruling", "legal"
            ],
            "technical": [
                "Kubernetes", "microservices", "Docker", "API", "database",
                "CI/CD", "deployment", "scalability", "authentication", "latency",
                "infrastructure", "architecture", "repository", "endpoint", "service"
            ],
            "business": [
                "revenue", "stakeholder", "quarter", "metrics", "strategy",
                "budget", "forecast", "analysis", "investment", "ROI",
                "partnership", "acquisition", "growth", "market", "client"
            ],
            "education": [
                "curriculum", "pedagogy", "assessment", "student", "lecture",
                "syllabus", "assignment", "exam", "research", "thesis",
                "academic", "scholar", "university", "degree", "course"
            ]
        }

    def get_domain_terminology(self, domain: str, limit: int = 15) -> List[str]:
        """
        Get domain-specific terminology from external glossary service or built-in dictionary

        Args:
            domain: Domain name (e.g., "medical", "legal", "technical")
            limit: Maximum number of terms to return

        Returns:
            List of domain-specific terms
        """
        # Try to fetch from orchestration service's glossary API
        if self._glossary_api_url:
            try:
                import httpx

                # Fetch glossary entries for the domain
                response = httpx.get(
                    f"{self._glossary_api_url}/api/glossaries",
                    params={"domain": domain, "limit": limit},
                    timeout=5.0
                )

                if response.status_code == 200:
                    glossaries = response.json()
                    if glossaries:
                        # Get entries from matching glossary
                        glossary_id = glossaries[0].get("glossary_id")
                        if glossary_id:
                            entries_response = httpx.get(
                                f"{self._glossary_api_url}/api/glossaries/{glossary_id}/entries",
                                params={"limit": limit},
                                timeout=5.0
                            )
                            if entries_response.status_code == 200:
                                entries = entries_response.json()
                                terms = [e.get("source_term") for e in entries if e.get("source_term")]
                                if terms:
                                    logger.info(f"[DOMAIN] Loaded {len(terms)} terms from glossary API for '{domain}'")
                                    return terms[:limit]
            except Exception as e:
                logger.debug(f"Failed to load terms from glossary API: {e}")

        # Fallback to built-in dictionaries
        if domain in self.builtin_domains:
            terms = self.builtin_domains[domain][:limit]
            logger.info(f"[DOMAIN] Using {len(terms)} built-in terms for '{domain}'")
            return terms

        logger.warning(f"[DOMAIN] Unknown domain '{domain}', no terminology available")
        return []

    def create_domain_prompt(
        self,
        domain: Optional[str] = None,
        custom_terms: Optional[List[str]] = None,
        previous_context: Optional[str] = None
    ) -> str:
        """
        Create domain-specific prompt for Whisper initial_prompt parameter

        Per SimulStreaming paper:
        - Max 448 tokens total
        - Domain terminology: ~225 tokens
        - Previous context: ~223 tokens

        Args:
            domain: Domain name ("medical", "legal", "technical", etc.)
            custom_terms: Additional custom terms to inject
            previous_context: Previous transcription output for continuity

        Returns:
            Formatted prompt string (max 448 tokens)
        """
        prompt_parts = []
        total_tokens = 0

        # Part 1: Domain terminology (max ~225 tokens)
        if domain or custom_terms:
            terminology = []

            # Add domain-specific terms
            if domain:
                terminology.extend(self.get_domain_terminology(domain, limit=15))

            # Add custom terms
            if custom_terms:
                terminology.extend(custom_terms)

            # Remove duplicates while preserving order
            terminology = list(dict.fromkeys(terminology))

            # Format terminology
            if terminology:
                terms_text = self.config.separator.join(terminology)
                terms_tokens = self._estimate_tokens(terms_text)

                # Trim if exceeds max terminology tokens
                if terms_tokens > self.config.max_terminology_tokens:
                    terms_text = self._trim_to_tokens(terms_text, self.config.max_terminology_tokens)
                    terms_tokens = self.config.max_terminology_tokens

                prompt_parts.append(terms_text)
                total_tokens += terms_tokens

                logger.info(f"[DOMAIN] Added {len(terminology)} terms (~{terms_tokens} tokens)")

        # Part 2: Previous context (max ~223 tokens)
        if previous_context:
            context_tokens = self._estimate_tokens(previous_context)

            # Trim if exceeds max context tokens
            if context_tokens > self.config.max_context_tokens:
                previous_context = self._trim_to_tokens(previous_context, self.config.max_context_tokens)
                context_tokens = self.config.max_context_tokens

            prompt_parts.append(previous_context)
            total_tokens += context_tokens

            logger.info(f"[DOMAIN] Added previous context (~{context_tokens} tokens)")

        # Combine parts
        if not prompt_parts:
            return ""

        final_prompt = self.config.separator.join(prompt_parts)

        # Final token check (should be <= 448)
        final_tokens = self._estimate_tokens(final_prompt)
        if final_tokens > self.config.max_total_tokens:
            logger.warning(f"[DOMAIN] Prompt exceeds max tokens ({final_tokens} > {self.config.max_total_tokens}), trimming...")
            final_prompt = self._trim_to_tokens(final_prompt, self.config.max_total_tokens)
            final_tokens = self.config.max_total_tokens

        logger.info(f"[DOMAIN] Generated prompt: {final_tokens} tokens, {len(final_prompt)} chars")
        logger.debug(f"[DOMAIN] Prompt content: '{final_prompt[:100]}...'")

        return final_prompt

    def update_context(self, new_output: str):
        """
        Update scrolling context window with new transcription output

        Args:
            new_output: Latest transcription text
        """
        if not new_output:
            return

        # Add to buffer
        self.context_buffer.append(new_output)

        # Keep buffer within token limit (use last few outputs)
        combined = self.config.separator.join(self.context_buffer)
        tokens = self._estimate_tokens(combined)

        # Trim oldest outputs if exceeds limit
        while tokens > self.config.max_context_tokens and len(self.context_buffer) > 1:
            self.context_buffer.pop(0)  # Remove oldest
            combined = self.config.separator.join(self.context_buffer)
            tokens = self._estimate_tokens(combined)

        logger.debug(f"[DOMAIN] Context buffer updated: {len(self.context_buffer)} segments, ~{tokens} tokens")

    def get_current_context(self) -> str:
        """
        Get current context from scrolling window

        Returns:
            Combined context string (max 223 tokens)
        """
        if not self.context_buffer:
            return ""

        context = self.config.separator.join(self.context_buffer)
        tokens = self._estimate_tokens(context)

        # Trim if needed
        if tokens > self.config.max_context_tokens:
            context = self._trim_to_tokens(context, self.config.max_context_tokens)

        return context

    def clear_context(self):
        """Clear context buffer"""
        self.context_buffer.clear()
        logger.info("[DOMAIN] Context buffer cleared")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Whisper uses GPT-2 tokenizer, rough estimate: 1 token ≈ 4 characters
        More accurate: use tiktoken library

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple estimation: ~4 chars per token
        # This is conservative; actual tokenization may vary
        return len(text) // 4

    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Trim text to maximum token count

        Args:
            text: Input text
            max_tokens: Maximum tokens allowed

        Returns:
            Trimmed text
        """
        if not text:
            return ""

        current_tokens = self._estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate approximate character limit
        max_chars = max_tokens * 4

        # Trim to character limit, try to break at sentence boundary
        trimmed = text[:max_chars]

        # Try to break at last complete sentence
        last_period = trimmed.rfind('. ')
        if last_period > max_chars * 0.7:  # Keep at least 70% of content
            trimmed = trimmed[:last_period + 1]

        return trimmed.strip()

    def get_prompt_template(self, domain: str) -> Optional[str]:
        """
        Get pre-defined prompt template for domain from glossary API

        Args:
            domain: Domain name

        Returns:
            Prompt template string or None
        """
        # Try to fetch from orchestration service's glossary API
        if self._glossary_api_url:
            try:
                import httpx

                # Fetch glossary that has description/notes for the domain
                response = httpx.get(
                    f"{self._glossary_api_url}/api/glossaries",
                    params={"domain": domain},
                    timeout=5.0
                )

                if response.status_code == 200:
                    glossaries = response.json()
                    if glossaries and len(glossaries) > 0:
                        # Use the glossary description as a prompt template if available
                        glossary = glossaries[0]
                        description = glossary.get("description")
                        if description:
                            logger.info(f"[DOMAIN] Loaded prompt template for '{domain}' from glossary API")
                            return description

            except Exception as e:
                logger.debug(f"Failed to load prompt template from glossary API: {e}")

        # Built-in templates as fallback
        builtin_templates = {
            "medical": "Medical transcription with clinical terminology, diagnoses, and treatment discussions.",
            "legal": "Legal transcription with court proceedings, contracts, and legal terminology.",
            "technical": "Technical discussion with software, infrastructure, and engineering terms.",
            "business": "Business meeting with financial, strategic, and organizational discussions.",
            "education": "Educational content with academic, pedagogical, and research terminology."
        }

        if domain in builtin_templates:
            logger.info(f"[DOMAIN] Using built-in template for '{domain}'")
            return builtin_templates[domain]

        return None

    def log_usage(
        self,
        domain: str,
        session_id: Optional[str] = None,
        quality_score: Optional[int] = None,
        processing_time_ms: Optional[int] = None
    ):
        """
        Log domain prompt usage for analytics

        Args:
            domain: Domain name used
            session_id: Session ID
            quality_score: Transcription quality (0-100)
            processing_time_ms: Processing time in milliseconds
        """
        # Log locally for now - orchestration service can collect via API if needed
        logger.info(
            f"[DOMAIN_USAGE] domain={domain}, session={session_id}, "
            f"quality={quality_score}, time_ms={processing_time_ms}"
        )

        # Optionally send to orchestration service for centralized logging
        if self._glossary_api_url:
            try:
                import httpx

                response = httpx.post(
                    f"{self._glossary_api_url}/api/analytics/domain-usage",
                    json={
                        "domain": domain,
                        "session_id": session_id,
                        "quality_score": quality_score,
                        "processing_time_ms": processing_time_ms,
                        "model_used": "whisper-large-v3"
                    },
                    timeout=2.0
                )

                if response.status_code == 200:
                    logger.debug(f"[DOMAIN] Logged usage to orchestration service")

            except Exception as e:
                # Don't fail on analytics logging errors
                logger.debug(f"Failed to log usage to orchestration service: {e}")


# Convenience function
def create_domain_prompt(
    domain: Optional[str] = None,
    custom_terms: Optional[List[str]] = None,
    previous_context: Optional[str] = None,
    db_session=None
) -> str:
    """
    Quick function to create domain prompt without instantiating manager

    Args:
        domain: Domain name
        custom_terms: Custom terminology list
        previous_context: Previous transcription for context carryover
        db_session: Database session (optional)

    Returns:
        Formatted prompt string
    """
    manager = DomainPromptManager(db_session=db_session)
    return manager.create_domain_prompt(domain, custom_terms, previous_context)


if __name__ == "__main__":
    # Test domain prompt manager
    print("Domain Prompt Manager - Phase 2 Implementation")
    print("=" * 60)

    manager = DomainPromptManager()

    # Test medical domain
    print("\n1. MEDICAL DOMAIN PROMPT:")
    medical_prompt = manager.create_domain_prompt(
        domain="medical",
        custom_terms=["COVID-19", "vaccination"],
        previous_context="The patient reports feeling unwell for three days."
    )
    print(f"Tokens: ~{manager._estimate_tokens(medical_prompt)}")
    print(f"Content: {medical_prompt[:200]}...")

    # Test technical domain
    print("\n2. TECHNICAL DOMAIN PROMPT:")
    tech_prompt = manager.create_domain_prompt(
        domain="technical",
        custom_terms=["FastAPI", "PostgreSQL"]
    )
    print(f"Tokens: ~{manager._estimate_tokens(tech_prompt)}")
    print(f"Content: {tech_prompt[:200]}...")

    # Test context carryover
    print("\n3. CONTEXT CARRYOVER:")
    manager.update_context("We discussed the deployment strategy.")
    manager.update_context("The team decided on Kubernetes for orchestration.")
    context = manager.get_current_context()
    print(f"Context: {context}")
    print(f"Tokens: ~{manager._estimate_tokens(context)}")
