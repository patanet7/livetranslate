"""
Glossary Service

Provides business logic for managing translation glossaries and applying
glossary terms during translation. Integrates with the Fireflies translation
pipeline to ensure consistent translation of domain-specific terms.

Key Features:
- CRUD operations for glossaries and entries
- Global vs session-specific glossary merging
- Domain-based filtering
- Term matching with whole-word and case-sensitivity options
- Priority-based conflict resolution

Reference: FIREFLIES_ADAPTATION_PLAN.md Section "Glossary & Context System"
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session as DBSession, selectinload

from database.models import Glossary, GlossaryEntry

logger = logging.getLogger(__name__)


# =============================================================================
# Glossary Service
# =============================================================================


class GlossaryService:
    """
    Service for managing translation glossaries.

    Handles CRUD operations for glossaries and entries, and provides
    term matching utilities for the translation pipeline.

    Usage:
        service = GlossaryService(db_session)

        # Get glossary terms for translation
        terms = service.get_glossary_terms(
            glossary_id=config.glossary_id,
            target_language="es",
            domain="medical"
        )

        # Apply to TranslationContext
        context = TranslationContext(
            glossary=terms,
            target_language="es"
        )
    """

    def __init__(self, db: DBSession):
        """
        Initialize the glossary service.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    # =========================================================================
    # Glossary CRUD Operations
    # =========================================================================

    def create_glossary(
        self,
        name: str,
        target_languages: List[str],
        description: Optional[str] = None,
        domain: Optional[str] = None,
        source_language: str = "en",
        is_default: bool = False,
        created_by: Optional[str] = None,
    ) -> Glossary:
        """
        Create a new glossary.

        Args:
            name: Glossary name
            target_languages: List of target language codes
            description: Optional description
            domain: Optional domain (e.g., 'medical', 'legal', 'tech')
            source_language: Source language code (default: 'en')
            is_default: Whether this is the default glossary
            created_by: User who created the glossary

        Returns:
            Created Glossary object
        """
        glossary = Glossary(
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

        logger.info(
            f"Created glossary: id={glossary.glossary_id}, "
            f"name={name}, domain={domain}"
        )

        return glossary

    def get_glossary(self, glossary_id: UUID) -> Optional[Glossary]:
        """
        Get a glossary by ID.

        Args:
            glossary_id: Glossary UUID

        Returns:
            Glossary object or None if not found
        """
        return self.db.get(Glossary, glossary_id)

    def get_glossary_with_entries(self, glossary_id: UUID) -> Optional[Glossary]:
        """
        Get a glossary with all its entries loaded.

        Args:
            glossary_id: Glossary UUID

        Returns:
            Glossary object with entries loaded, or None if not found
        """
        stmt = (
            select(Glossary)
            .options(selectinload(Glossary.entries))
            .where(Glossary.glossary_id == glossary_id)
        )
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()

    def list_glossaries(
        self,
        domain: Optional[str] = None,
        source_language: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Glossary]:
        """
        List glossaries with optional filters.

        Args:
            domain: Filter by domain
            source_language: Filter by source language
            active_only: Only return active glossaries

        Returns:
            List of matching glossaries
        """
        conditions = []

        if active_only:
            conditions.append(Glossary.is_active == True)

        if domain:
            conditions.append(Glossary.domain == domain)

        if source_language:
            conditions.append(Glossary.source_language == source_language)

        stmt = select(Glossary)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(Glossary.name)

        result = self.db.execute(stmt)
        return list(result.scalars().all())

    def update_glossary(
        self,
        glossary_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        target_languages: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Glossary]:
        """
        Update a glossary.

        Args:
            glossary_id: Glossary UUID
            name: New name (optional)
            description: New description (optional)
            domain: New domain (optional)
            target_languages: New target languages (optional)
            is_active: New active status (optional)

        Returns:
            Updated Glossary or None if not found
        """
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

        logger.info(f"Updated glossary: id={glossary_id}")

        return glossary

    def delete_glossary(self, glossary_id: UUID) -> bool:
        """
        Delete a glossary and all its entries.

        Args:
            glossary_id: Glossary UUID

        Returns:
            True if deleted, False if not found
        """
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            return False

        self.db.delete(glossary)
        self.db.commit()

        logger.info(f"Deleted glossary: id={glossary_id}")

        return True

    # =========================================================================
    # Entry CRUD Operations
    # =========================================================================

    def add_entry(
        self,
        glossary_id: UUID,
        source_term: str,
        translations: Dict[str, str],
        context: Optional[str] = None,
        notes: Optional[str] = None,
        case_sensitive: bool = False,
        match_whole_word: bool = True,
        priority: int = 0,
    ) -> Optional[GlossaryEntry]:
        """
        Add an entry to a glossary.

        Args:
            glossary_id: Parent glossary UUID
            source_term: Source term to translate
            translations: Dict of target_language -> translation
            context: Optional usage context/example
            notes: Optional internal notes
            case_sensitive: Whether matching is case-sensitive
            match_whole_word: Whether to match whole words only
            priority: Priority for conflict resolution (higher = more important)

        Returns:
            Created GlossaryEntry or None if glossary not found
        """
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            logger.warning(f"Cannot add entry: glossary {glossary_id} not found")
            return None

        entry = GlossaryEntry(
            glossary_id=glossary_id,
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

        # Update entry count
        glossary.entry_count = (glossary.entry_count or 0) + 1

        self.db.commit()
        self.db.refresh(entry)

        logger.debug(
            f"Added glossary entry: '{source_term}' -> {translations} "
            f"(glossary={glossary_id})"
        )

        return entry

    def get_entry(self, entry_id: UUID) -> Optional[GlossaryEntry]:
        """
        Get a glossary entry by ID.

        Args:
            entry_id: Entry UUID

        Returns:
            GlossaryEntry or None if not found
        """
        return self.db.get(GlossaryEntry, entry_id)

    def list_entries(
        self,
        glossary_id: UUID,
        target_language: Optional[str] = None,
    ) -> List[GlossaryEntry]:
        """
        List entries for a glossary.

        Args:
            glossary_id: Glossary UUID
            target_language: Optional filter for entries with this target language

        Returns:
            List of matching entries
        """
        stmt = (
            select(GlossaryEntry)
            .where(GlossaryEntry.glossary_id == glossary_id)
            .order_by(GlossaryEntry.priority.desc(), GlossaryEntry.source_term)
        )

        result = self.db.execute(stmt)
        entries = list(result.scalars().all())

        # Filter by target language if specified
        if target_language:
            entries = [
                e for e in entries
                if target_language in (e.translations or {})
            ]

        return entries

    def update_entry(
        self,
        entry_id: UUID,
        source_term: Optional[str] = None,
        translations: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
        notes: Optional[str] = None,
        case_sensitive: Optional[bool] = None,
        match_whole_word: Optional[bool] = None,
        priority: Optional[int] = None,
    ) -> Optional[GlossaryEntry]:
        """
        Update a glossary entry.

        Args:
            entry_id: Entry UUID
            source_term: New source term (optional)
            translations: New translations dict (optional)
            context: New context (optional)
            notes: New notes (optional)
            case_sensitive: New case sensitivity (optional)
            match_whole_word: New whole word setting (optional)
            priority: New priority (optional)

        Returns:
            Updated GlossaryEntry or None if not found
        """
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

        logger.debug(f"Updated glossary entry: id={entry_id}")

        return entry

    def delete_entry(self, entry_id: UUID) -> bool:
        """
        Delete a glossary entry.

        Args:
            entry_id: Entry UUID

        Returns:
            True if deleted, False if not found
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        glossary = self.get_glossary(entry.glossary_id)
        if glossary:
            glossary.entry_count = max(0, (glossary.entry_count or 1) - 1)

        self.db.delete(entry)
        self.db.commit()

        logger.debug(f"Deleted glossary entry: id={entry_id}")

        return True

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def import_entries(
        self,
        glossary_id: UUID,
        entries: List[Dict],
    ) -> Tuple[int, int]:
        """
        Bulk import entries into a glossary.

        Args:
            glossary_id: Target glossary UUID
            entries: List of entry dicts with keys:
                - source_term: str (required)
                - translations: Dict[str, str] (required)
                - context: str (optional)
                - notes: str (optional)
                - case_sensitive: bool (optional, default False)
                - match_whole_word: bool (optional, default True)
                - priority: int (optional, default 0)

        Returns:
            Tuple of (successful_count, failed_count)
        """
        glossary = self.get_glossary(glossary_id)
        if not glossary:
            logger.warning(f"Cannot import: glossary {glossary_id} not found")
            return (0, len(entries))

        successful = 0
        failed = 0

        for entry_data in entries:
            try:
                source_term = entry_data.get("source_term")
                translations = entry_data.get("translations")

                if not source_term or not translations:
                    failed += 1
                    continue

                entry = GlossaryEntry(
                    glossary_id=glossary_id,
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

            except Exception as e:
                logger.warning(f"Failed to import entry: {e}")
                failed += 1

        # Update entry count
        glossary.entry_count = (glossary.entry_count or 0) + successful

        self.db.commit()

        logger.info(
            f"Imported entries to glossary {glossary_id}: "
            f"{successful} successful, {failed} failed"
        )

        return (successful, failed)

    # =========================================================================
    # Term Matching and Translation Support
    # =========================================================================

    def get_glossary_terms(
        self,
        glossary_id: Optional[UUID],
        target_language: str,
        domain: Optional[str] = None,
        include_default: bool = True,
    ) -> Dict[str, str]:
        """
        Get glossary terms for use in translation.

        This method retrieves all applicable terms and returns them as a
        simple source_term -> translation dict for use in TranslationContext.

        Merges:
        1. Default glossary (if include_default=True)
        2. Specified glossary (if glossary_id provided)

        Later entries override earlier ones (specified glossary wins).

        Args:
            glossary_id: Specific glossary to use (optional)
            target_language: Target language code
            domain: Filter by domain (optional)
            include_default: Include default glossary terms

        Returns:
            Dict mapping source_term -> translation for the target language
        """
        terms: Dict[str, str] = {}

        # Get default glossary first
        if include_default:
            default_terms = self._get_default_glossary_terms(
                target_language, domain
            )
            terms.update(default_terms)

        # Get specified glossary (overrides default)
        if glossary_id:
            specific_terms = self._get_glossary_terms_by_id(
                glossary_id, target_language
            )
            terms.update(specific_terms)

        logger.debug(
            f"Retrieved {len(terms)} glossary terms for "
            f"target_language={target_language}, domain={domain}"
        )

        return terms

    def _get_default_glossary_terms(
        self,
        target_language: str,
        domain: Optional[str] = None,
    ) -> Dict[str, str]:
        """Get terms from default glossaries."""
        conditions = [
            Glossary.is_active == True,
            Glossary.is_default == True,
        ]

        if domain:
            # Include matching domain OR no domain (global defaults)
            conditions.append(
                or_(
                    Glossary.domain == domain,
                    Glossary.domain.is_(None),
                )
            )

        stmt = (
            select(Glossary)
            .options(selectinload(Glossary.entries))
            .where(and_(*conditions))
        )

        result = self.db.execute(stmt)
        glossaries = result.scalars().all()

        terms: Dict[str, str] = {}
        for glossary in glossaries:
            for entry in glossary.entries:
                translation = entry.get_translation(target_language)
                if translation:
                    terms[entry.source_term] = translation

        return terms

    def _get_glossary_terms_by_id(
        self,
        glossary_id: UUID,
        target_language: str,
    ) -> Dict[str, str]:
        """Get terms from a specific glossary."""
        glossary = self.get_glossary_with_entries(glossary_id)
        if not glossary or not glossary.is_active:
            return {}

        terms: Dict[str, str] = {}
        for entry in sorted(glossary.entries, key=lambda e: e.priority):
            translation = entry.get_translation(target_language)
            if translation:
                terms[entry.source_term] = translation

        return terms

    def find_matching_terms(
        self,
        text: str,
        glossary_id: Optional[UUID],
        target_language: str,
        domain: Optional[str] = None,
    ) -> List[Tuple[str, str, int, int]]:
        """
        Find glossary terms that match within the given text.

        Returns matches with their positions for potential highlighting
        or verification that terms were applied.

        Args:
            text: Text to search for terms
            glossary_id: Glossary to use (optional)
            target_language: Target language
            domain: Domain filter (optional)

        Returns:
            List of (source_term, translation, start_pos, end_pos) tuples
        """
        matches: List[Tuple[str, str, int, int]] = []

        # Get all applicable entries
        entries = self._get_all_applicable_entries(
            glossary_id, target_language, domain
        )

        # Sort by length (longest first) to match longer terms first
        entries = sorted(entries, key=lambda e: len(e.source_term), reverse=True)

        text_lower = text.lower()

        for entry in entries:
            translation = entry.get_translation(target_language)
            if not translation:
                continue

            # Build regex pattern
            pattern_str = re.escape(entry.source_term)
            if entry.match_whole_word:
                pattern_str = rf"\b{pattern_str}\b"

            flags = 0 if entry.case_sensitive else re.IGNORECASE

            pattern = re.compile(pattern_str, flags)

            for match in pattern.finditer(text):
                matches.append((
                    entry.source_term,
                    translation,
                    match.start(),
                    match.end(),
                ))

        # Sort by position
        matches.sort(key=lambda m: m[2])

        return matches

    def _get_all_applicable_entries(
        self,
        glossary_id: Optional[UUID],
        target_language: str,
        domain: Optional[str] = None,
    ) -> List[GlossaryEntry]:
        """Get all entries from applicable glossaries."""
        entries: List[GlossaryEntry] = []

        # Default glossaries
        conditions = [
            Glossary.is_active == True,
            Glossary.is_default == True,
        ]

        if domain:
            conditions.append(
                or_(
                    Glossary.domain == domain,
                    Glossary.domain.is_(None),
                )
            )

        stmt = (
            select(Glossary)
            .options(selectinload(Glossary.entries))
            .where(and_(*conditions))
        )

        result = self.db.execute(stmt)
        for glossary in result.scalars():
            for entry in glossary.entries:
                if target_language in (entry.translations or {}):
                    entries.append(entry)

        # Specific glossary
        if glossary_id:
            glossary = self.get_glossary_with_entries(glossary_id)
            if glossary and glossary.is_active:
                for entry in glossary.entries:
                    if target_language in (entry.translations or {}):
                        entries.append(entry)

        return entries

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_glossary_stats(self, glossary_id: UUID) -> Optional[Dict]:
        """
        Get statistics for a glossary.

        Args:
            glossary_id: Glossary UUID

        Returns:
            Dict with stats or None if glossary not found
        """
        glossary = self.get_glossary_with_entries(glossary_id)
        if not glossary:
            return None

        # Count translations per language
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
