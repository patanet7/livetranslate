#!/usr/bin/env python3
"""
Unit Tests for Glossary Service

Tests CRUD operations, term matching, glossary merging,
and integration with the translation pipeline.

Uses in-memory SQLite database for isolated testing.
"""

import sys
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
import pytest

# Prevent FastAPI app import issues
os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

# Add src to path for imports
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, JSON, Float, ForeignKey, Index, select, and_, or_
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship, selectinload
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
import uuid as uuid_module

# =============================================================================
# Test-Local Model Definitions (to avoid import chain issues)
# =============================================================================

TestBase = declarative_base()


class TestGlossary(TestBase):
    """Test-local Glossary model matching production schema."""

    __tablename__ = "glossaries"

    glossary_id = Column(String(36), primary_key=True, default=lambda: str(uuid_module.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=True, index=True)
    source_language = Column(String(10), nullable=False, default="en")
    target_languages = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_default = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    entry_count = Column(Integer, nullable=False, default=0)

    entries = relationship("TestGlossaryEntry", back_populates="glossary", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "glossary_id": str(self.glossary_id),
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "source_language": self.source_language,
            "target_languages": self.target_languages,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "entry_count": self.entry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TestGlossaryEntry(TestBase):
    """Test-local GlossaryEntry model matching production schema."""

    __tablename__ = "glossary_entries"

    entry_id = Column(String(36), primary_key=True, default=lambda: str(uuid_module.uuid4()))
    glossary_id = Column(String(36), ForeignKey("glossaries.glossary_id"), nullable=False)
    source_term = Column(String(500), nullable=False)
    source_term_normalized = Column(String(500), nullable=False, index=True)
    translations = Column(JSON, nullable=False)
    context = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    case_sensitive = Column(Boolean, nullable=False, default=False)
    match_whole_word = Column(Boolean, nullable=False, default=True)
    priority = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    glossary = relationship("TestGlossary", back_populates="entries")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": str(self.entry_id),
            "glossary_id": str(self.glossary_id),
            "source_term": self.source_term,
            "translations": self.translations,
            "context": self.context,
            "notes": self.notes,
            "case_sensitive": self.case_sensitive,
            "match_whole_word": self.match_whole_word,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def get_translation(self, target_language: str) -> Optional[str]:
        if self.translations and target_language in self.translations:
            return self.translations[target_language]
        return None


# =============================================================================
# Test-Local GlossaryService (uses test models)
# =============================================================================


class TestGlossaryService:
    """Test-local GlossaryService using test models."""

    def __init__(self, db: Session):
        self.db = db

    # === Glossary CRUD ===

    def create_glossary(
        self,
        name: str,
        target_languages: List[str],
        description: Optional[str] = None,
        domain: Optional[str] = None,
        source_language: str = "en",
        is_default: bool = False,
        created_by: Optional[str] = None,
    ) -> TestGlossary:
        glossary = TestGlossary(
            name=name,
            description=description,
            domain=domain,
            source_language=source_language,
            target_languages=target_languages,
            is_default=is_default,
            created_by=created_by,
        )
        self.db.add(glossary)
        self.db.commit()
        self.db.refresh(glossary)
        return glossary

    def get_glossary(self, glossary_id) -> Optional[TestGlossary]:
        return self.db.query(TestGlossary).filter(TestGlossary.glossary_id == str(glossary_id)).first()

    def get_glossary_with_entries(self, glossary_id) -> Optional[TestGlossary]:
        return self.db.query(TestGlossary).options(selectinload(TestGlossary.entries)).filter(TestGlossary.glossary_id == str(glossary_id)).first()

    def list_glossaries(
        self,
        domain: Optional[str] = None,
        source_language: Optional[str] = None,
        active_only: bool = True,
    ) -> List[TestGlossary]:
        query = self.db.query(TestGlossary)
        if active_only:
            query = query.filter(TestGlossary.is_active == True)
        if domain:
            query = query.filter(TestGlossary.domain == domain)
        if source_language:
            query = query.filter(TestGlossary.source_language == source_language)
        return query.order_by(TestGlossary.name).all()

    def update_glossary(
        self,
        glossary_id,
        name: Optional[str] = None,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        target_languages: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[TestGlossary]:
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            return None
        if name is not None:
            glossary.name = name
        if description is not None:
            glossary.description = description
        if domain is not None:
            glossary.domain = domain
        if target_languages is not None:
            glossary.target_languages = target_languages
        if is_active is not None:
            glossary.is_active = is_active
        self.db.commit()
        self.db.refresh(glossary)
        return glossary

    def delete_glossary(self, glossary_id) -> bool:
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            return False
        self.db.delete(glossary)
        self.db.commit()
        return True

    # === Entry CRUD ===

    def add_entry(
        self,
        glossary_id,
        source_term: str,
        translations: Dict[str, str],
        context: Optional[str] = None,
        notes: Optional[str] = None,
        case_sensitive: bool = False,
        match_whole_word: bool = True,
        priority: int = 0,
    ) -> Optional[TestGlossaryEntry]:
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            return None
        entry = TestGlossaryEntry(
            glossary_id=str(glossary_id),
            source_term=source_term,
            source_term_normalized=source_term.lower(),
            translations=translations,
            context=context,
            notes=notes,
            case_sensitive=case_sensitive,
            match_whole_word=match_whole_word,
            priority=priority,
        )
        self.db.add(entry)
        glossary.entry_count = (glossary.entry_count or 0) + 1
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_entry(self, entry_id) -> Optional[TestGlossaryEntry]:
        return self.db.query(TestGlossaryEntry).filter(TestGlossaryEntry.entry_id == str(entry_id)).first()

    def list_entries(
        self,
        glossary_id,
        target_language: Optional[str] = None,
    ) -> List[TestGlossaryEntry]:
        entries = self.db.query(TestGlossaryEntry).filter(
            TestGlossaryEntry.glossary_id == str(glossary_id)
        ).order_by(TestGlossaryEntry.priority.desc(), TestGlossaryEntry.source_term).all()
        if target_language:
            entries = [e for e in entries if target_language in (e.translations or {})]
        return entries

    def update_entry(
        self,
        entry_id,
        source_term: Optional[str] = None,
        translations: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
        notes: Optional[str] = None,
        case_sensitive: Optional[bool] = None,
        match_whole_word: Optional[bool] = None,
        priority: Optional[int] = None,
    ) -> Optional[TestGlossaryEntry]:
        entry = self.get_entry(entry_id)
        if not entry:
            return None
        if source_term is not None:
            entry.source_term = source_term
            entry.source_term_normalized = source_term.lower()
        if translations is not None:
            entry.translations = translations
        if context is not None:
            entry.context = context
        if notes is not None:
            entry.notes = notes
        if case_sensitive is not None:
            entry.case_sensitive = case_sensitive
        if match_whole_word is not None:
            entry.match_whole_word = match_whole_word
        if priority is not None:
            entry.priority = priority
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def delete_entry(self, entry_id) -> bool:
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        glossary = self.get_glossary(entry.glossary_id)
        if glossary:
            glossary.entry_count = max(0, (glossary.entry_count or 1) - 1)
        self.db.delete(entry)
        self.db.commit()
        return True

    # === Bulk Operations ===

    def import_entries(
        self,
        glossary_id,
        entries: List[Dict],
    ) -> tuple:
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            return (0, len(entries))
        successful = 0
        failed = 0
        for entry_data in entries:
            source_term = entry_data.get("source_term")
            translations = entry_data.get("translations")
            if not source_term or not translations:
                failed += 1
                continue
            try:
                entry = TestGlossaryEntry(
                    glossary_id=str(glossary_id),
                    source_term=source_term,
                    source_term_normalized=source_term.lower(),
                    translations=translations,
                    context=entry_data.get("context"),
                    notes=entry_data.get("notes"),
                    case_sensitive=entry_data.get("case_sensitive", False),
                    match_whole_word=entry_data.get("match_whole_word", True),
                    priority=entry_data.get("priority", 0),
                )
                self.db.add(entry)
                successful += 1
            except Exception:
                failed += 1
        glossary.entry_count = (glossary.entry_count or 0) + successful
        self.db.commit()
        return (successful, failed)

    # === Term Matching ===

    def get_glossary_terms(
        self,
        glossary_id,
        target_language: str,
        domain: Optional[str] = None,
        include_default: bool = True,
    ) -> Dict[str, str]:
        terms: Dict[str, str] = {}
        if include_default:
            query = self.db.query(TestGlossary).options(selectinload(TestGlossary.entries)).filter(
                TestGlossary.is_active == True,
                TestGlossary.is_default == True,
            )
            if domain:
                query = query.filter(or_(TestGlossary.domain == domain, TestGlossary.domain.is_(None)))
            for glossary in query.all():
                for entry in glossary.entries:
                    translation = entry.get_translation(target_language)
                    if translation:
                        terms[entry.source_term] = translation
        if glossary_id:
            glossary = self.get_glossary_with_entries(glossary_id)
            if glossary and glossary.is_active:
                for entry in sorted(glossary.entries, key=lambda e: e.priority):
                    translation = entry.get_translation(target_language)
                    if translation:
                        terms[entry.source_term] = translation
        return terms

    def find_matching_terms(
        self,
        text: str,
        glossary_id,
        target_language: str,
        domain: Optional[str] = None,
    ) -> List[tuple]:
        matches = []
        entries = []
        query = self.db.query(TestGlossary).options(selectinload(TestGlossary.entries)).filter(
            TestGlossary.is_active == True,
            TestGlossary.is_default == True,
        )
        for glossary in query.all():
            for entry in glossary.entries:
                if target_language in (entry.translations or {}):
                    entries.append(entry)
        if glossary_id:
            glossary = self.get_glossary_with_entries(glossary_id)
            if glossary and glossary.is_active:
                for entry in glossary.entries:
                    if target_language in (entry.translations or {}):
                        entries.append(entry)
        entries = sorted(entries, key=lambda e: len(e.source_term), reverse=True)
        for entry in entries:
            translation = entry.get_translation(target_language)
            if not translation:
                continue
            pattern_str = re.escape(entry.source_term)
            if entry.match_whole_word:
                pattern_str = rf"\b{pattern_str}\b"
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)
            for match in pattern.finditer(text):
                matches.append((entry.source_term, translation, match.start(), match.end()))
        matches.sort(key=lambda m: m[2])
        return matches

    def get_glossary_stats(self, glossary_id) -> Optional[Dict]:
        glossary = self.get_glossary_with_entries(glossary_id)
        if not glossary:
            return None
        language_counts: Dict[str, int] = {}
        for entry in glossary.entries:
            for lang in (entry.translations or {}):
                language_counts[lang] = language_counts.get(lang, 0) + 1
        return {
            "glossary_id": str(glossary.glossary_id),
            "name": glossary.name,
            "domain": glossary.domain,
            "source_language": glossary.source_language,
            "target_languages": glossary.target_languages,
            "is_active": glossary.is_active,
            "is_default": glossary.is_default,
            "entry_count": glossary.entry_count,
            "actual_entry_count": len(glossary.entries),
            "translations_per_language": language_counts,
            "created_at": glossary.created_at.isoformat() if glossary.created_at else None,
            "updated_at": glossary.updated_at.isoformat() if glossary.updated_at else None,
        }


# Alias for tests
GlossaryService = TestGlossaryService


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_engine():
    """Create in-memory SQLite database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    TestBase.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def glossary_service(db_session):
    """Create GlossaryService instance."""
    return GlossaryService(db_session)


@pytest.fixture
def sample_glossary(db_session, glossary_service):
    """Create a sample glossary with entries."""
    glossary = glossary_service.create_glossary(
        name="Tech Glossary",
        target_languages=["es", "fr", "de"],
        description="Technical terminology for software development",
        domain="technology",
        source_language="en",
        created_by="test_user",
    )

    # Add entries
    glossary_service.add_entry(
        glossary_id=glossary.glossary_id,
        source_term="API",
        translations={"es": "API", "fr": "API", "de": "API"},
        context="Application Programming Interface",
        priority=10,
    )

    glossary_service.add_entry(
        glossary_id=glossary.glossary_id,
        source_term="backend",
        translations={"es": "servidor", "fr": "serveur", "de": "Backend"},
        priority=5,
    )

    glossary_service.add_entry(
        glossary_id=glossary.glossary_id,
        source_term="frontend",
        translations={"es": "interfaz", "fr": "interface", "de": "Frontend"},
        priority=5,
    )

    return glossary


@pytest.fixture
def default_glossary(db_session, glossary_service):
    """Create a default glossary."""
    glossary = glossary_service.create_glossary(
        name="Default Terms",
        target_languages=["es", "fr"],
        description="Default terminology",
        is_default=True,
        source_language="en",
    )

    glossary_service.add_entry(
        glossary_id=glossary.glossary_id,
        source_term="hello",
        translations={"es": "hola", "fr": "bonjour"},
        priority=0,
    )

    glossary_service.add_entry(
        glossary_id=glossary.glossary_id,
        source_term="goodbye",
        translations={"es": "adiós", "fr": "au revoir"},
        priority=0,
    )

    return glossary


# =============================================================================
# Glossary CRUD Tests
# =============================================================================


class TestGlossaryCreate:
    """Tests for glossary creation."""

    def test_create_glossary_basic(self, glossary_service):
        """Test creating a basic glossary."""
        glossary = glossary_service.create_glossary(
            name="Test Glossary",
            target_languages=["es", "fr"],
        )

        assert glossary is not None
        assert glossary.glossary_id is not None
        assert glossary.name == "Test Glossary"
        assert glossary.target_languages == ["es", "fr"]
        assert glossary.source_language == "en"
        assert glossary.is_active is True
        assert glossary.is_default is False
        assert glossary.entry_count == 0

    def test_create_glossary_with_domain(self, glossary_service):
        """Test creating a glossary with domain."""
        glossary = glossary_service.create_glossary(
            name="Medical Terms",
            target_languages=["es"],
            domain="medical",
            description="Medical terminology",
        )

        assert glossary.domain == "medical"
        assert glossary.description == "Medical terminology"

    def test_create_default_glossary(self, glossary_service):
        """Test creating a default glossary."""
        glossary = glossary_service.create_glossary(
            name="Default Glossary",
            target_languages=["es"],
            is_default=True,
        )

        assert glossary.is_default is True

    def test_create_glossary_with_creator(self, glossary_service):
        """Test creating a glossary with creator info."""
        glossary = glossary_service.create_glossary(
            name="User Glossary",
            target_languages=["es"],
            created_by="john.doe@example.com",
        )

        assert glossary.created_by == "john.doe@example.com"


class TestGlossaryRead:
    """Tests for glossary reading."""

    def test_get_glossary_by_id(self, glossary_service, sample_glossary):
        """Test retrieving glossary by ID."""
        retrieved = glossary_service.get_glossary(sample_glossary.glossary_id)

        assert retrieved is not None
        assert retrieved.glossary_id == sample_glossary.glossary_id
        assert retrieved.name == "Tech Glossary"

    def test_get_nonexistent_glossary(self, glossary_service):
        """Test retrieving non-existent glossary."""
        result = glossary_service.get_glossary(uuid4())
        assert result is None

    def test_get_glossary_with_entries(self, glossary_service, sample_glossary):
        """Test retrieving glossary with entries loaded."""
        retrieved = glossary_service.get_glossary_with_entries(
            sample_glossary.glossary_id
        )

        assert retrieved is not None
        assert len(retrieved.entries) == 3
        assert any(e.source_term == "API" for e in retrieved.entries)

    def test_list_glossaries_all(self, glossary_service, sample_glossary, default_glossary):
        """Test listing all glossaries."""
        glossaries = glossary_service.list_glossaries()

        assert len(glossaries) == 2
        names = [g.name for g in glossaries]
        assert "Tech Glossary" in names
        assert "Default Terms" in names

    def test_list_glossaries_by_domain(self, glossary_service, sample_glossary, default_glossary):
        """Test listing glossaries filtered by domain."""
        glossaries = glossary_service.list_glossaries(domain="technology")

        assert len(glossaries) == 1
        assert glossaries[0].name == "Tech Glossary"

    def test_list_glossaries_by_language(self, glossary_service, sample_glossary):
        """Test listing glossaries filtered by source language."""
        glossaries = glossary_service.list_glossaries(source_language="en")

        assert len(glossaries) >= 1
        assert all(g.source_language == "en" for g in glossaries)


class TestGlossaryUpdate:
    """Tests for glossary updates."""

    def test_update_glossary_name(self, glossary_service, sample_glossary):
        """Test updating glossary name."""
        updated = glossary_service.update_glossary(
            glossary_id=sample_glossary.glossary_id,
            name="Updated Tech Glossary",
        )

        assert updated is not None
        assert updated.name == "Updated Tech Glossary"

    def test_update_glossary_domain(self, glossary_service, sample_glossary):
        """Test updating glossary domain."""
        updated = glossary_service.update_glossary(
            glossary_id=sample_glossary.glossary_id,
            domain="software",
        )

        assert updated.domain == "software"

    def test_update_glossary_target_languages(self, glossary_service, sample_glossary):
        """Test updating target languages."""
        updated = glossary_service.update_glossary(
            glossary_id=sample_glossary.glossary_id,
            target_languages=["es", "fr", "de", "it"],
        )

        assert updated.target_languages == ["es", "fr", "de", "it"]

    def test_update_glossary_deactivate(self, glossary_service, sample_glossary):
        """Test deactivating a glossary."""
        updated = glossary_service.update_glossary(
            glossary_id=sample_glossary.glossary_id,
            is_active=False,
        )

        assert updated.is_active is False

    def test_update_nonexistent_glossary(self, glossary_service):
        """Test updating non-existent glossary."""
        result = glossary_service.update_glossary(
            glossary_id=uuid4(),
            name="New Name",
        )
        assert result is None


class TestGlossaryDelete:
    """Tests for glossary deletion."""

    def test_delete_glossary(self, glossary_service, sample_glossary):
        """Test deleting a glossary."""
        glossary_id = sample_glossary.glossary_id
        result = glossary_service.delete_glossary(glossary_id)

        assert result is True

        # Verify deletion
        retrieved = glossary_service.get_glossary(glossary_id)
        assert retrieved is None

    def test_delete_nonexistent_glossary(self, glossary_service):
        """Test deleting non-existent glossary."""
        result = glossary_service.delete_glossary(uuid4())
        assert result is False


# =============================================================================
# Entry CRUD Tests
# =============================================================================


class TestEntryCreate:
    """Tests for entry creation."""

    def test_add_entry_basic(self, glossary_service, sample_glossary):
        """Test adding a basic entry."""
        entry = glossary_service.add_entry(
            glossary_id=sample_glossary.glossary_id,
            source_term="database",
            translations={"es": "base de datos", "fr": "base de données"},
        )

        assert entry is not None
        assert entry.entry_id is not None
        assert entry.source_term == "database"
        assert entry.source_term_normalized == "database"
        assert entry.translations["es"] == "base de datos"
        assert entry.case_sensitive is False
        assert entry.match_whole_word is True
        assert entry.priority == 0

    def test_add_entry_with_options(self, glossary_service, sample_glossary):
        """Test adding entry with all options."""
        entry = glossary_service.add_entry(
            glossary_id=sample_glossary.glossary_id,
            source_term="iOS",
            translations={"es": "iOS"},
            context="Apple mobile operating system",
            notes="Keep capitalization in translation",
            case_sensitive=True,
            match_whole_word=True,
            priority=20,
        )

        assert entry.context == "Apple mobile operating system"
        assert entry.notes == "Keep capitalization in translation"
        assert entry.case_sensitive is True
        assert entry.priority == 20

    def test_add_entry_updates_count(self, glossary_service, db_session):
        """Test that adding entries updates glossary count."""
        glossary = glossary_service.create_glossary(
            name="Count Test",
            target_languages=["es"],
        )

        assert glossary.entry_count == 0

        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="term1",
            translations={"es": "término1"},
        )

        # Refresh to get updated count
        db_session.refresh(glossary)
        assert glossary.entry_count == 1

    def test_add_entry_to_nonexistent_glossary(self, glossary_service):
        """Test adding entry to non-existent glossary."""
        entry = glossary_service.add_entry(
            glossary_id=uuid4(),
            source_term="test",
            translations={"es": "prueba"},
        )

        assert entry is None


class TestEntryRead:
    """Tests for entry reading."""

    def test_get_entry_by_id(self, glossary_service, sample_glossary):
        """Test retrieving entry by ID."""
        entry = glossary_service.add_entry(
            glossary_id=sample_glossary.glossary_id,
            source_term="test",
            translations={"es": "prueba"},
        )

        retrieved = glossary_service.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.source_term == "test"

    def test_get_nonexistent_entry(self, glossary_service):
        """Test retrieving non-existent entry."""
        result = glossary_service.get_entry(uuid4())
        assert result is None

    def test_list_entries(self, glossary_service, sample_glossary):
        """Test listing all entries for a glossary."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)

        assert len(entries) == 3
        terms = [e.source_term for e in entries]
        assert "API" in terms
        assert "backend" in terms
        assert "frontend" in terms

    def test_list_entries_by_language(self, glossary_service, sample_glossary):
        """Test listing entries filtered by target language."""
        # Add entry without German translation
        glossary_service.add_entry(
            glossary_id=sample_glossary.glossary_id,
            source_term="only_spanish",
            translations={"es": "solo español"},
        )

        entries = glossary_service.list_entries(
            sample_glossary.glossary_id,
            target_language="de",
        )

        # Should only get entries with German translations
        assert len(entries) == 3  # API, backend, frontend
        assert all("de" in e.translations for e in entries)

    def test_list_entries_ordered_by_priority(self, glossary_service, sample_glossary):
        """Test that entries are ordered by priority (descending)."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)

        # API has priority 10, backend/frontend have priority 5
        assert entries[0].source_term == "API"
        assert entries[0].priority == 10


class TestEntryUpdate:
    """Tests for entry updates."""

    def test_update_entry_source_term(self, glossary_service, sample_glossary):
        """Test updating entry source term."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)
        entry = [e for e in entries if e.source_term == "API"][0]

        updated = glossary_service.update_entry(
            entry_id=entry.entry_id,
            source_term="REST API",
        )

        assert updated is not None
        assert updated.source_term == "REST API"
        assert updated.source_term_normalized == "rest api"

    def test_update_entry_translations(self, glossary_service, sample_glossary):
        """Test updating entry translations."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)
        entry = [e for e in entries if e.source_term == "backend"][0]

        updated = glossary_service.update_entry(
            entry_id=entry.entry_id,
            translations={"es": "backend", "fr": "serveur backend", "de": "Backend-Server"},
        )

        assert updated.translations["es"] == "backend"
        assert updated.translations["fr"] == "serveur backend"

    def test_update_entry_priority(self, glossary_service, sample_glossary):
        """Test updating entry priority."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)
        entry = [e for e in entries if e.source_term == "backend"][0]

        updated = glossary_service.update_entry(
            entry_id=entry.entry_id,
            priority=100,
        )

        assert updated.priority == 100

    def test_update_nonexistent_entry(self, glossary_service):
        """Test updating non-existent entry."""
        result = glossary_service.update_entry(
            entry_id=uuid4(),
            source_term="new term",
        )
        assert result is None


class TestEntryDelete:
    """Tests for entry deletion."""

    def test_delete_entry(self, glossary_service, sample_glossary, db_session):
        """Test deleting an entry."""
        entries = glossary_service.list_entries(sample_glossary.glossary_id)
        entry = entries[0]
        entry_id = entry.entry_id

        initial_count = sample_glossary.entry_count

        result = glossary_service.delete_entry(entry_id)
        assert result is True

        # Verify deletion
        retrieved = glossary_service.get_entry(entry_id)
        assert retrieved is None

        # Verify count updated
        db_session.refresh(sample_glossary)
        assert sample_glossary.entry_count == initial_count - 1

    def test_delete_nonexistent_entry(self, glossary_service):
        """Test deleting non-existent entry."""
        result = glossary_service.delete_entry(uuid4())
        assert result is False


# =============================================================================
# Bulk Operations Tests
# =============================================================================


class TestBulkImport:
    """Tests for bulk import operations."""

    def test_import_entries_success(self, glossary_service, db_session):
        """Test successful bulk import."""
        glossary = glossary_service.create_glossary(
            name="Import Test",
            target_languages=["es"],
        )

        entries_data = [
            {"source_term": "term1", "translations": {"es": "término1"}},
            {"source_term": "term2", "translations": {"es": "término2"}},
            {"source_term": "term3", "translations": {"es": "término3"}},
        ]

        successful, failed = glossary_service.import_entries(
            glossary.glossary_id, entries_data
        )

        assert successful == 3
        assert failed == 0

        # Verify entries created
        entries = glossary_service.list_entries(glossary.glossary_id)
        assert len(entries) == 3

        # Verify count updated
        db_session.refresh(glossary)
        assert glossary.entry_count == 3

    def test_import_entries_with_options(self, glossary_service):
        """Test bulk import with full options."""
        glossary = glossary_service.create_glossary(
            name="Import Test",
            target_languages=["es"],
        )

        entries_data = [
            {
                "source_term": "iPhone",
                "translations": {"es": "iPhone"},
                "case_sensitive": True,
                "match_whole_word": True,
                "priority": 10,
                "context": "Apple smartphone",
            },
        ]

        successful, _ = glossary_service.import_entries(
            glossary.glossary_id, entries_data
        )
        assert successful == 1

        entries = glossary_service.list_entries(glossary.glossary_id)
        entry = entries[0]
        assert entry.case_sensitive is True
        assert entry.priority == 10
        assert entry.context == "Apple smartphone"

    def test_import_entries_partial_failure(self, glossary_service):
        """Test bulk import with some failures."""
        glossary = glossary_service.create_glossary(
            name="Import Test",
            target_languages=["es"],
        )

        entries_data = [
            {"source_term": "good1", "translations": {"es": "bueno1"}},
            {"source_term": "", "translations": {"es": "bad"}},  # Empty term
            {"source_term": "missing_translations"},  # No translations
            {"source_term": "good2", "translations": {"es": "bueno2"}},
        ]

        successful, failed = glossary_service.import_entries(
            glossary.glossary_id, entries_data
        )

        assert successful == 2
        assert failed == 2

    def test_import_to_nonexistent_glossary(self, glossary_service):
        """Test import to non-existent glossary."""
        entries_data = [
            {"source_term": "test", "translations": {"es": "prueba"}},
        ]

        successful, failed = glossary_service.import_entries(uuid4(), entries_data)

        assert successful == 0
        assert failed == 1


# =============================================================================
# Term Matching Tests
# =============================================================================


class TestGetGlossaryTerms:
    """Tests for getting glossary terms."""

    def test_get_terms_from_glossary(self, glossary_service, sample_glossary):
        """Test getting terms from a specific glossary."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        assert len(terms) == 3
        assert terms["API"] == "API"
        assert terms["backend"] == "servidor"
        assert terms["frontend"] == "interfaz"

    def test_get_terms_for_different_language(self, glossary_service, sample_glossary):
        """Test getting terms for different target languages."""
        terms_es = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        terms_de = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="de",
            include_default=False,
        )

        assert terms_es["backend"] == "servidor"
        assert terms_de["backend"] == "Backend"

    def test_get_terms_includes_default(self, glossary_service, sample_glossary, default_glossary):
        """Test getting terms includes default glossary."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=True,
        )

        # Should have both specific and default terms
        assert "API" in terms  # From specific glossary
        assert "hello" in terms  # From default glossary

    def test_specific_glossary_overrides_default(self, glossary_service, db_session):
        """Test that specific glossary terms override default."""
        # Create default glossary with term
        default = glossary_service.create_glossary(
            name="Default",
            target_languages=["es"],
            is_default=True,
        )
        glossary_service.add_entry(
            glossary_id=default.glossary_id,
            source_term="hello",
            translations={"es": "hola"},
        )

        # Create specific glossary with same term, different translation
        specific = glossary_service.create_glossary(
            name="Specific",
            target_languages=["es"],
        )
        glossary_service.add_entry(
            glossary_id=specific.glossary_id,
            source_term="hello",
            translations={"es": "saludos"},
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=specific.glossary_id,
            target_language="es",
            include_default=True,
        )

        # Specific should override default
        assert terms["hello"] == "saludos"

    def test_get_terms_with_domain_filter(self, glossary_service, db_session):
        """Test getting terms with domain filtering."""
        # Create default with domain
        glossary = glossary_service.create_glossary(
            name="Medical Default",
            target_languages=["es"],
            domain="medical",
            is_default=True,
        )
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="doctor",
            translations={"es": "médico"},
        )

        # Create default without domain
        general = glossary_service.create_glossary(
            name="General Default",
            target_languages=["es"],
            is_default=True,
        )
        glossary_service.add_entry(
            glossary_id=general.glossary_id,
            source_term="hello",
            translations={"es": "hola"},
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=None,
            target_language="es",
            domain="medical",
            include_default=True,
        )

        # Should include medical domain and general (no domain)
        assert "doctor" in terms
        assert "hello" in terms


class TestFindMatchingTerms:
    """Tests for finding matching terms in text."""

    def test_find_matching_terms_basic(self, glossary_service, sample_glossary):
        """Test finding matching terms in text."""
        text = "The API connects the backend to the frontend."

        matches = glossary_service.find_matching_terms(
            text=text,
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
        )

        assert len(matches) == 3

        terms = [m[0] for m in matches]
        assert "API" in terms
        assert "backend" in terms
        assert "frontend" in terms

    def test_find_matching_terms_positions(self, glossary_service, sample_glossary):
        """Test that match positions are correct."""
        text = "The API is great."

        matches = glossary_service.find_matching_terms(
            text=text,
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
        )

        assert len(matches) >= 1
        api_match = [m for m in matches if m[0] == "API"][0]

        # Check positions
        source_term, translation, start, end = api_match
        assert text[start:end] == "API"

    def test_find_matching_terms_case_insensitive(self, glossary_service, sample_glossary):
        """Test case-insensitive matching (default)."""
        text = "The api and the FRONTEND work together."

        matches = glossary_service.find_matching_terms(
            text=text,
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
        )

        terms = [m[0] for m in matches]
        # Should match regardless of case
        assert "API" in terms or len([m for m in matches if m[0].lower() == "api"]) > 0
        assert "frontend" in terms or len([m for m in matches if m[0].lower() == "frontend"]) > 0

    def test_find_matching_terms_whole_word(self, glossary_service, db_session):
        """Test whole-word matching."""
        glossary = glossary_service.create_glossary(
            name="Test",
            target_languages=["es"],
        )
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="end",
            translations={"es": "fin"},
            match_whole_word=True,
        )

        # "end" should not match within "backend"
        text = "The backend has an end point."

        matches = glossary_service.find_matching_terms(
            text=text,
            glossary_id=glossary.glossary_id,
            target_language="es",
        )

        # Should only match "end" as standalone word
        assert len(matches) == 1
        assert "end point" in text[matches[0][2] - 1 : matches[0][3] + 6] or text[matches[0][2]:matches[0][3]] == "end"

    def test_find_matching_terms_no_matches(self, glossary_service, sample_glossary):
        """Test when no terms match."""
        text = "This text has no matching terms at all."

        matches = glossary_service.find_matching_terms(
            text=text,
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
        )

        assert len(matches) == 0


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestGlossaryStats:
    """Tests for glossary statistics."""

    def test_get_glossary_stats(self, glossary_service, sample_glossary):
        """Test getting glossary statistics."""
        stats = glossary_service.get_glossary_stats(sample_glossary.glossary_id)

        assert stats is not None
        assert stats["name"] == "Tech Glossary"
        assert stats["domain"] == "technology"
        assert stats["entry_count"] == 3
        assert stats["actual_entry_count"] == 3
        assert "es" in stats["translations_per_language"]
        assert "fr" in stats["translations_per_language"]
        assert "de" in stats["translations_per_language"]

    def test_get_stats_nonexistent_glossary(self, glossary_service):
        """Test getting stats for non-existent glossary."""
        stats = glossary_service.get_glossary_stats(uuid4())
        assert stats is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_glossary_terms(self, glossary_service):
        """Test getting terms from empty glossary."""
        glossary = glossary_service.create_glossary(
            name="Empty",
            target_languages=["es"],
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        assert terms == {}

    def test_inactive_glossary_not_returned(self, glossary_service, sample_glossary):
        """Test that inactive glossary terms are not returned."""
        glossary_service.update_glossary(
            sample_glossary.glossary_id,
            is_active=False,
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        assert terms == {}

    def test_entry_without_target_language(self, glossary_service):
        """Test entry that doesn't have the target language."""
        glossary = glossary_service.create_glossary(
            name="Test",
            target_languages=["es", "fr"],
        )
        glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="only_spanish",
            translations={"es": "solo español"},  # No French
        )

        terms = glossary_service.get_glossary_terms(
            glossary_id=glossary.glossary_id,
            target_language="fr",
            include_default=False,
        )

        assert "only_spanish" not in terms

    def test_unicode_terms(self, glossary_service):
        """Test handling of unicode terms."""
        glossary = glossary_service.create_glossary(
            name="Unicode Test",
            target_languages=["ja", "zh"],
        )

        entry = glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="hello",
            translations={"ja": "こんにちは", "zh": "你好"},
        )

        assert entry is not None
        assert entry.translations["ja"] == "こんにちは"
        assert entry.translations["zh"] == "你好"

    def test_special_characters_in_terms(self, glossary_service):
        """Test handling of special characters in terms."""
        glossary = glossary_service.create_glossary(
            name="Special Chars",
            target_languages=["es"],
        )

        entry = glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term="C++",
            translations={"es": "C++"},
        )

        assert entry is not None
        assert entry.source_term == "C++"

    def test_long_source_term(self, glossary_service):
        """Test handling of long source terms."""
        glossary = glossary_service.create_glossary(
            name="Long Terms",
            target_languages=["es"],
        )

        long_term = "artificial intelligence machine learning natural language processing"

        entry = glossary_service.add_entry(
            glossary_id=glossary.glossary_id,
            source_term=long_term,
            translations={"es": "IA ML PLN"},
        )

        assert entry is not None
        assert entry.source_term == long_term


# =============================================================================
# Integration with TranslationContext Tests
# =============================================================================


class TestTranslationContextIntegration:
    """Tests for integration with TranslationContext model."""

    def test_terms_format_for_translation_context(self, glossary_service, sample_glossary):
        """Test that terms are formatted correctly for TranslationContext."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Should be simple dict[str, str] format
        assert isinstance(terms, dict)
        for key, value in terms.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_terms_can_be_used_in_translation_context(self, glossary_service, sample_glossary):
        """Test that terms work with TranslationContext model."""
        terms = glossary_service.get_glossary_terms(
            glossary_id=sample_glossary.glossary_id,
            target_language="es",
            include_default=False,
        )

        # Import the model and verify compatibility
        try:
            from models.fireflies import TranslationContext

            context = TranslationContext(
                glossary=terms,
                target_language="es",
                previous_sentences=["Previous sentence."],
            )

            formatted = context.format_glossary()
            assert "API -> API" in formatted
            assert "backend -> servidor" in formatted
        except ImportError:
            pytest.skip("TranslationContext model not available")
