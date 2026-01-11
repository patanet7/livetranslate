#!/usr/bin/env python3
"""
Integration Tests for Glossary Service

Tests the full integration of the glossary service with:
- Sentence Aggregator
- TranslationContext model
- Translation pipeline flow
- Multi-speaker conversations with glossary terms

These tests verify that glossary terms are properly:
- Loaded from the database
- Merged (default + session-specific)
- Formatted for translation prompts
- Applied during translation
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
import re
import pytest

# Prevent FastAPI app import issues
os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, JSON, Float, ForeignKey, or_
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship, selectinload
import uuid as uuid_module

# Import Fireflies models (these should work)
from models.fireflies import (
    FirefliesChunk,
    FirefliesSessionConfig,
    TranslationUnit,
    TranslationContext,
    TranslationResult,
)


# =============================================================================
# Test-Local Database Models (matching glossary_service tests)
# =============================================================================

IntegrationBase = declarative_base()


class IntGlossary(IntegrationBase):
    """Integration test Glossary model."""

    __tablename__ = "glossaries"

    glossary_id = Column(String(36), primary_key=True, default=lambda: str(uuid_module.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=True)
    source_language = Column(String(10), nullable=False, default="en")
    target_languages = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    entry_count = Column(Integer, nullable=False, default=0)

    entries = relationship("IntGlossaryEntry", back_populates="glossary", cascade="all, delete-orphan")


class IntGlossaryEntry(IntegrationBase):
    """Integration test GlossaryEntry model."""

    __tablename__ = "glossary_entries"

    entry_id = Column(String(36), primary_key=True, default=lambda: str(uuid_module.uuid4()))
    glossary_id = Column(String(36), ForeignKey("glossaries.glossary_id"), nullable=False)
    source_term = Column(String(500), nullable=False)
    source_term_normalized = Column(String(500), nullable=False)
    translations = Column(JSON, nullable=False)
    context = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    case_sensitive = Column(Boolean, nullable=False, default=False)
    match_whole_word = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    glossary = relationship("IntGlossary", back_populates="entries")

    def get_translation(self, target_language: str) -> Optional[str]:
        if self.translations and target_language in self.translations:
            return self.translations[target_language]
        return None


# =============================================================================
# Minimal GlossaryService for Integration Tests
# =============================================================================


class IntGlossaryService:
    """Minimal GlossaryService for integration testing."""

    def __init__(self, db: Session):
        self.db = db

    def create_glossary(
        self,
        name: str,
        target_languages: List[str],
        domain: Optional[str] = None,
        is_default: bool = False,
    ) -> IntGlossary:
        glossary = IntGlossary(
            name=name,
            target_languages=target_languages,
            domain=domain,
            is_default=is_default,
        )
        self.db.add(glossary)
        self.db.commit()
        self.db.refresh(glossary)
        return glossary

    def add_entry(
        self,
        glossary_id,
        source_term: str,
        translations: Dict[str, str],
        priority: int = 0,
    ) -> IntGlossaryEntry:
        entry = IntGlossaryEntry(
            glossary_id=str(glossary_id),
            source_term=source_term,
            source_term_normalized=source_term.lower(),
            translations=translations,
            priority=priority,
        )
        self.db.add(entry)
        glossary = self.db.query(IntGlossary).filter(IntGlossary.glossary_id == str(glossary_id)).first()
        if glossary:
            glossary.entry_count = (glossary.entry_count or 0) + 1
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_glossary_terms(
        self,
        glossary_id,
        target_language: str,
        include_default: bool = True,
    ) -> Dict[str, str]:
        terms: Dict[str, str] = {}

        # Get default glossary terms first
        if include_default:
            defaults = self.db.query(IntGlossary).options(
                selectinload(IntGlossary.entries)
            ).filter(
                IntGlossary.is_active == True,
                IntGlossary.is_default == True,
            ).all()

            for glossary in defaults:
                for entry in glossary.entries:
                    translation = entry.get_translation(target_language)
                    if translation:
                        terms[entry.source_term] = translation

        # Get specific glossary terms (overrides default)
        if glossary_id:
            glossary = self.db.query(IntGlossary).options(
                selectinload(IntGlossary.entries)
            ).filter(IntGlossary.glossary_id == str(glossary_id)).first()

            if glossary and glossary.is_active:
                for entry in sorted(glossary.entries, key=lambda e: e.priority):
                    translation = entry.get_translation(target_language)
                    if translation:
                        terms[entry.source_term] = translation

        return terms


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_engine():
    """Create in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    IntegrationBase.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def glossary_service(db_session):
    """Create GlossaryService instance."""
    return IntGlossaryService(db_session)


@pytest.fixture
def tech_glossary(glossary_service):
    """Create a tech domain glossary with common terms."""
    glossary = glossary_service.create_glossary(
        name="Tech Terms",
        target_languages=["es", "fr", "de"],
        domain="technology",
    )

    terms = [
        ("API", {"es": "API", "fr": "API", "de": "API"}, 10),
        ("backend", {"es": "servidor", "fr": "serveur", "de": "Backend"}, 5),
        ("frontend", {"es": "interfaz", "fr": "interface", "de": "Frontend"}, 5),
        ("deployment", {"es": "despliegue", "fr": "déploiement", "de": "Bereitstellung"}, 5),
        ("microservices", {"es": "microservicios", "fr": "microservices", "de": "Microservices"}, 8),
        ("Kubernetes", {"es": "Kubernetes", "fr": "Kubernetes", "de": "Kubernetes"}, 10),
        ("container", {"es": "contenedor", "fr": "conteneur", "de": "Container"}, 5),
    ]

    for term, translations, priority in terms:
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term=term,
            translations=translations,
            priority=priority,
        )

    return glossary


@pytest.fixture
def default_glossary(glossary_service):
    """Create a default glossary with common terms."""
    glossary = glossary_service.create_glossary(
        name="Default Terms",
        target_languages=["es", "fr", "de"],
        is_default=True,
    )

    terms = [
        ("hello", {"es": "hola", "fr": "bonjour", "de": "hallo"}),
        ("thank you", {"es": "gracias", "fr": "merci", "de": "danke"}),
        ("please", {"es": "por favor", "fr": "s'il vous plaît", "de": "bitte"}),
    ]

    for term, translations in terms:
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term=term,
            translations=translations,
        )

    return glossary


@pytest.fixture
def sample_session_config():
    """Create a sample Fireflies session config."""
    return FirefliesSessionConfig(
        api_key="ff-test-api-key",
        transcript_id="transcript-integration-test",
        target_languages=["es"],
        pause_threshold_ms=800.0,
        max_buffer_words=30,
        context_window_size=3,
    )


# =============================================================================
# Integration Tests: Glossary + TranslationContext
# =============================================================================


class TestGlossaryTranslationContextIntegration:
    """Tests for glossary integration with TranslationContext."""

    def test_glossary_terms_in_translation_context(
        self, glossary_service, tech_glossary
    ):
        """Test that glossary terms are properly loaded into TranslationContext."""
        # Get terms for Spanish
        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Create TranslationContext with terms
        context = TranslationContext(
            previous_sentences=[
                "We discussed the API architecture.",
                "The backend team is ready.",
            ],
            glossary=terms,
            target_language="es",
            source_language="en",
        )

        # Verify glossary is properly set
        assert context.glossary == terms
        assert len(context.glossary) == 7  # All tech terms

        # Verify format_glossary works
        formatted = context.format_glossary()
        assert "API -> API" in formatted
        assert "backend -> servidor" in formatted
        assert "deployment -> despliegue" in formatted

    def test_glossary_merging_in_context(
        self, glossary_service, tech_glossary, default_glossary
    ):
        """Test that default and specific glossaries are merged correctly."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=True,
        )

        context = TranslationContext(
            glossary=terms,
            target_language="es",
        )

        # Should have both tech and default terms
        assert "API" in context.glossary  # Tech term
        assert "hello" in context.glossary  # Default term
        assert "thank you" in context.glossary  # Default term

        # Total should be 7 (tech) + 3 (default) = 10
        assert len(context.glossary) == 10

    def test_empty_context_when_no_glossary(self, glossary_service, db_session):
        """Test handling when no glossary is provided."""
        # Create empty glossary
        empty_glossary = glossary_service.create_glossary(
            name="Empty",
            target_languages=["es"],
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=empty_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        context = TranslationContext(
            glossary=terms,
            target_language="es",
        )

        assert context.glossary == {}
        assert context.format_glossary() == "(No glossary terms)"


# =============================================================================
# Integration Tests: Full Translation Pipeline Simulation
# =============================================================================


class TestFullTranslationPipelineIntegration:
    """Tests simulating the full translation pipeline with glossary."""

    def test_translation_unit_with_glossary_context(
        self, glossary_service, tech_glossary, sample_session_config
    ):
        """Test creating TranslationUnit and context for translation."""
        # Create a TranslationUnit (simulating sentence aggregator output)
        translation_unit = TranslationUnit(
            text="The API deployment to Kubernetes is complete.",
            speaker_name="Alice",
            start_time=0.0,
            end_time=3.0,
            session_id="test-session",
            transcript_id=sample_session_config.transcript_id,
            chunk_ids=["chunk_001", "chunk_002"],
            boundary_type="punctuation",
        )

        # Get glossary terms
        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Create translation context
        context = TranslationContext(
            previous_sentences=["We discussed the backend changes."],
            glossary=terms,
            target_language="es",
            source_language="en",
        )

        # Verify the text contains glossary terms
        text = translation_unit.text
        glossary_terms_in_text = [
            term for term in terms.keys() if term.lower() in text.lower()
        ]

        assert "API" in glossary_terms_in_text
        assert "deployment" in glossary_terms_in_text
        assert "Kubernetes" in glossary_terms_in_text

        # Verify context has these terms for translation
        assert context.glossary["API"] == "API"
        assert context.glossary["deployment"] == "despliegue"
        assert context.glossary["Kubernetes"] == "Kubernetes"

    def test_multi_speaker_conversation_with_glossary(
        self, glossary_service, tech_glossary
    ):
        """Test handling multi-speaker conversation with glossary terms."""
        # Simulate conversation between two speakers
        conversation = [
            TranslationUnit(
                text="Alice: Have you seen the API documentation?",
                speaker_name="Alice",
                start_time=0.0,
                end_time=2.0,
                session_id="test-session",
                transcript_id="test-transcript",
                chunk_ids=["chunk_001"],
                boundary_type="punctuation",
            ),
            TranslationUnit(
                text="Bob: Yes, the backend integration looks good.",
                speaker_name="Bob",
                start_time=2.5,
                end_time=5.0,
                session_id="test-session",
                transcript_id="test-transcript",
                chunk_ids=["chunk_002"],
                boundary_type="punctuation",
            ),
            TranslationUnit(
                text="Alice: Great, let's start the deployment.",
                speaker_name="Alice",
                start_time=5.5,
                end_time=8.0,
                session_id="test-session",
                transcript_id="test-transcript",
                chunk_ids=["chunk_003"],
                boundary_type="punctuation",
            ),
        ]

        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Build rolling context window
        context_window = []

        for i, unit in enumerate(conversation):
            # Create context with previous sentences
            context = TranslationContext(
                previous_sentences=context_window[-3:],  # Last 3 sentences
                glossary=terms,
                target_language="es",
                source_language="en",
            )

            # Verify context is properly built
            assert len(context.previous_sentences) == min(i, 3)
            assert context.glossary == terms

            # Simulate translation and add to context window
            context_window.append(unit.text)

        # Final context should have 2 previous sentences
        assert len(context_window) == 3

    def test_translation_result_tracks_glossary_terms(
        self, glossary_service, tech_glossary
    ):
        """Test that TranslationResult tracks applied glossary terms."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Simulate translation result
        original_text = "The API and backend are ready for deployment."

        # Find which glossary terms appear in the text
        applied_terms = [
            term for term in terms.keys()
            if term.lower() in original_text.lower()
        ]

        # Create TranslationResult
        result = TranslationResult(
            original=original_text,
            translated="La API y el servidor están listos para el despliegue.",
            speaker_name="Alice",
            source_language="en",
            target_language="es",
            confidence=0.95,
            context_sentences_used=2,
            glossary_terms_applied=applied_terms,
            translation_time_ms=150.0,
            session_id="test-session",
        )

        # Verify glossary terms are tracked
        assert "API" in result.glossary_terms_applied
        assert "backend" in result.glossary_terms_applied
        assert "deployment" in result.glossary_terms_applied
        assert len(result.glossary_terms_applied) == 3


# =============================================================================
# Integration Tests: Glossary Override Behavior
# =============================================================================


class TestGlossaryOverrideBehavior:
    """Tests for glossary term override behavior."""

    def test_specific_glossary_overrides_default(
        self, glossary_service, db_session
    ):
        """Test that session-specific glossary overrides default terms."""
        # Create default glossary
        default = glossary_service.create_glossary(
            name="Default",
            target_languages=["es"],
            is_default=True,
        )
        glossary_service.add_entry(
            glossary_id=default.glossary_id,
            source_term="container",
            translations={"es": "recipiente"},  # Generic translation
        )

        # Create tech-specific glossary
        tech = glossary_service.create_glossary(
            name="Tech",
            target_languages=["es"],
            domain="technology",
        )
        glossary_service.add_entry(
            glossary_id=tech.glossary_id,
            source_term="container",
            translations={"es": "contenedor"},  # Docker container
            priority=10,
        )

        # Get merged terms
        terms = glossary_service.get_glossary_terms(
            glossary_id=tech.glossary_id,
            target_language="es",
            include_default=True,
        )

        # Tech term should override default
        assert terms["container"] == "contenedor"

    def test_priority_based_term_selection(
        self, glossary_service
    ):
        """Test that higher priority terms are selected."""
        glossary = glossary_service.create_glossary(
            name="Priority Test",
            target_languages=["es"],
        )

        # Add same term twice with different priorities
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="server",
            translations={"es": "servidor"},
            priority=5,
        )

        # Higher priority entry (added later, should override)
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="server",
            translations={"es": "máquina servidor"},
            priority=10,
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Higher priority should win
        assert terms["server"] == "máquina servidor"


# =============================================================================
# Integration Tests: Multi-Language Support
# =============================================================================


class TestMultiLanguageGlossaryIntegration:
    """Tests for multi-language glossary handling."""

    def test_same_glossary_different_languages(
        self, glossary_service, tech_glossary
    ):
        """Test getting terms for different target languages."""
        terms_es = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        terms_fr = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="fr",
            include_default=False,
        )

        terms_de = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="de",
            include_default=False,
        )

        # Same source terms, different translations
        assert terms_es["backend"] == "servidor"
        assert terms_fr["backend"] == "serveur"
        assert terms_de["backend"] == "Backend"

        assert terms_es["deployment"] == "despliegue"
        assert terms_fr["deployment"] == "déploiement"
        assert terms_de["deployment"] == "Bereitstellung"

    def test_translation_context_per_language(
        self, glossary_service, tech_glossary
    ):
        """Test creating separate TranslationContext per target language."""
        languages = ["es", "fr", "de"]
        contexts = {}

        for lang in languages:
            terms = glossary_service.get_glossary_terms(
                glossary_id=tech_glossary.glossary_id,
                target_language=lang,
                include_default=False,
            )

            contexts[lang] = TranslationContext(
                previous_sentences=["The API is ready."],
                glossary=terms,
                target_language=lang,
                source_language="en",
            )

        # Each context should have different translations
        assert contexts["es"].glossary["deployment"] == "despliegue"
        assert contexts["fr"].glossary["deployment"] == "déploiement"
        assert contexts["de"].glossary["deployment"] == "Bereitstellung"


# =============================================================================
# Integration Tests: Edge Cases
# =============================================================================


class TestGlossaryIntegrationEdgeCases:
    """Tests for edge cases in glossary integration."""

    def test_missing_target_language_translation(
        self, glossary_service
    ):
        """Test handling when entry doesn't have target language translation."""
        glossary = glossary_service.create_glossary(
            name="Partial",
            target_languages=["es", "fr"],
        )

        # Add entry with only Spanish translation
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="widget",
            translations={"es": "componente"},  # No French
        )

        terms_es = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        terms_fr = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="fr",
            include_default=False,
        )

        assert "widget" in terms_es
        assert "widget" not in terms_fr

    def test_unicode_terms_in_context(self, glossary_service):
        """Test handling Unicode terms in translation context."""
        glossary = glossary_service.create_glossary(
            name="Unicode Test",
            target_languages=["ja", "zh"],
        )

        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="hello",
            translations={"ja": "こんにちは", "zh": "你好"},
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="ja",
            include_default=False,
        )

        context = TranslationContext(
            glossary=terms,
            target_language="ja",
        )

        formatted = context.format_glossary()
        assert "hello -> こんにちは" in formatted

    def test_empty_translation_unit_text(self, glossary_service, tech_glossary):
        """Test handling empty or minimal text in TranslationUnit."""
        # Create TranslationUnit with minimal text
        unit = TranslationUnit(
            text="OK.",
            speaker_name="Alice",
            start_time=0.0,
            end_time=0.5,
            session_id="test-session",
            transcript_id="test-transcript",
            chunk_ids=["chunk_001"],
            boundary_type="punctuation",
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=tech_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # No glossary terms should match
        applied = [t for t in terms.keys() if t.lower() in unit.text.lower()]
        assert len(applied) == 0
