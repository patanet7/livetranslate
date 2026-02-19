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

Reference: docs/archive/root-reports/status-plans/FIREFLIES_ADAPTATION_PLAN.md
Section "Glossary & Context System"
"""

import re
from uuid import UUID

from database.models import Glossary, GlossaryEntry
from livetranslate_common.logging import get_logger
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = get_logger()


# =============================================================================
# Glossary Service (Async)
# =============================================================================


class GlossaryService:
    """
    Service for managing translation glossaries.

    Handles CRUD operations for glossaries and entries, and provides
    term matching utilities for the translation pipeline.

    Usage:
        service = GlossaryService(db_session)

        # Get glossary terms for translation
        terms = await service.get_glossary_terms(
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

    def __init__(self, db: AsyncSession):
        """
        Initialize the glossary service.

        Args:
            db: SQLAlchemy async database session
        """
        self.db = db

    # =========================================================================
    # Glossary CRUD Operations
    # =========================================================================

    async def create_glossary(
        self,
        name: str,
        target_languages: list[str],
        description: str | None = None,
        domain: str | None = None,
        source_language: str = "en",
        is_default: bool = False,
        created_by: str | None = None,
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
            is_active=True,  # Explicitly set
            created_by=created_by,
        )

        self.db.add(glossary)
        await self.db.commit()
        await self.db.refresh(glossary)

        logger.info(
            f"Created glossary: id={glossary.glossary_id}, " f"name={name}, domain={domain}"
        )

        return glossary

    async def get_glossary(self, glossary_id: UUID) -> Glossary | None:
        """
        Get a glossary by ID.

        Args:
            glossary_id: Glossary UUID

        Returns:
            Glossary object or None if not found
        """
        result = await self.db.execute(select(Glossary).where(Glossary.glossary_id == glossary_id))
        return result.scalar_one_or_none()

    async def get_glossary_with_entries(self, glossary_id: UUID) -> Glossary | None:
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
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_glossaries(
        self,
        domain: str | None = None,
        source_language: str | None = None,
        active_only: bool = True,
    ) -> list[Glossary]:
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
            conditions.append(Glossary.is_active)

        if domain:
            conditions.append(Glossary.domain == domain)

        if source_language:
            conditions.append(Glossary.source_language == source_language)

        stmt = select(Glossary)
        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by(Glossary.name)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_glossary(
        self,
        glossary_id: UUID,
        name: str | None = None,
        description: str | None = None,
        domain: str | None = None,
        target_languages: list[str] | None = None,
        is_active: bool | None = None,
    ) -> Glossary | None:
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
        glossary = await self.get_glossary(glossary_id)
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

        await self.db.commit()
        await self.db.refresh(glossary)

        logger.info(f"Updated glossary: id={glossary_id}")
        return glossary

    async def delete_glossary(self, glossary_id: UUID) -> bool:
        """
        Delete a glossary and all its entries.

        Args:
            glossary_id: Glossary UUID

        Returns:
            True if deleted, False if not found
        """
        glossary = await self.get_glossary(glossary_id)
        if not glossary:
            return False

        await self.db.delete(glossary)
        await self.db.commit()

        logger.info(f"Deleted glossary: id={glossary_id}")
        return True

    async def get_glossary_stats(self, glossary_id: UUID) -> dict | None:
        """
        Get statistics for a glossary.

        Args:
            glossary_id: Glossary UUID

        Returns:
            Statistics dict or None if glossary not found
        """
        glossary = await self.get_glossary(glossary_id)
        if not glossary:
            return None

        # Count entries
        count_result = await self.db.execute(
            select(func.count(GlossaryEntry.entry_id)).where(
                GlossaryEntry.glossary_id == glossary_id
            )
        )
        entry_count = count_result.scalar() or 0

        # Get language coverage
        entries_result = await self.db.execute(
            select(GlossaryEntry.translations).where(GlossaryEntry.glossary_id == glossary_id)
        )
        all_translations = entries_result.scalars().all()

        languages_with_translations = set()
        for translations in all_translations:
            if translations:
                languages_with_translations.update(translations.keys())

        return {
            "glossary_id": str(glossary_id),
            "name": glossary.name,
            "entry_count": entry_count,
            "target_languages": glossary.target_languages or [],
            "languages_with_translations": list(languages_with_translations),
            "coverage": len(languages_with_translations) / len(glossary.target_languages)
            if glossary.target_languages
            else 0,
        }

    # =========================================================================
    # Entry CRUD Operations
    # =========================================================================

    async def add_entry(
        self,
        glossary_id: UUID,
        source_term: str,
        translations: dict[str, str],
        context: str | None = None,
        notes: str | None = None,
        priority: int = 0,
        case_sensitive: bool = False,
        match_whole_word: bool = True,
    ) -> GlossaryEntry | None:
        """
        Add a new entry to a glossary.

        Args:
            glossary_id: Parent glossary UUID
            source_term: Source term to translate
            translations: Dict mapping language codes to translations
            context: Optional usage context
            notes: Optional internal notes
            priority: Priority for conflict resolution (higher = more important)
            case_sensitive: Whether matching should be case-sensitive
            match_whole_word: Whether to match whole words only

        Returns:
            Created GlossaryEntry or None if glossary not found
        """
        glossary = await self.get_glossary(glossary_id)
        if not glossary:
            return None

        entry = GlossaryEntry(
            glossary_id=glossary_id,
            source_term=source_term,
            source_term_normalized=source_term.lower(),
            translations=translations,
            context=context,
            notes=notes,
            priority=priority,
            case_sensitive=case_sensitive,
            match_whole_word=match_whole_word,
        )

        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)

        logger.info(
            f"Added entry to glossary {glossary_id}: "
            f"'{source_term}' -> {len(translations)} translations"
        )

        return entry

    async def get_entry(self, entry_id: UUID) -> GlossaryEntry | None:
        """Get an entry by ID."""
        result = await self.db.execute(
            select(GlossaryEntry).where(GlossaryEntry.entry_id == entry_id)
        )
        return result.scalar_one_or_none()

    async def list_entries(
        self,
        glossary_id: UUID,
        target_language: str | None = None,
    ) -> list[GlossaryEntry]:
        """
        List entries for a glossary.

        Args:
            glossary_id: Glossary UUID
            target_language: Optional filter for entries with this language

        Returns:
            List of matching entries
        """
        stmt = select(GlossaryEntry).where(GlossaryEntry.glossary_id == glossary_id)

        # Note: Filtering by target_language would require JSON operations
        # For now, return all and filter in Python if needed

        stmt = stmt.order_by(GlossaryEntry.source_term)

        result = await self.db.execute(stmt)
        entries = list(result.scalars().all())

        # Filter by target language if specified
        if target_language:
            entries = [e for e in entries if e.translations and target_language in e.translations]

        return entries

    async def update_entry(
        self,
        entry_id: UUID,
        translations: dict[str, str] | None = None,
        definition: str | None = None,
        context: str | None = None,
        priority: int | None = None,
        case_sensitive: bool | None = None,
        whole_word: bool | None = None,
    ) -> GlossaryEntry | None:
        """
        Update a glossary entry.

        Args:
            entry_id: Entry UUID
            translations: New translations dict (optional)
            definition: New definition (optional)
            context: New context (optional)
            priority: New priority (optional)
            case_sensitive: New case sensitivity setting (optional)
            whole_word: New whole word setting (optional)

        Returns:
            Updated entry or None if not found
        """
        entry = await self.get_entry(entry_id)
        if not entry:
            return None

        if translations is not None:
            entry.translations = translations
        if definition is not None:
            entry.definition = definition
        if context is not None:
            entry.context = context
        if priority is not None:
            entry.priority = priority
        if case_sensitive is not None:
            entry.case_sensitive = case_sensitive
        if whole_word is not None:
            entry.whole_word = whole_word

        await self.db.commit()
        await self.db.refresh(entry)

        logger.info(f"Updated entry: id={entry_id}")
        return entry

    async def delete_entry(self, entry_id: UUID) -> bool:
        """
        Delete a glossary entry.

        Args:
            entry_id: Entry UUID

        Returns:
            True if deleted, False if not found
        """
        entry = await self.get_entry(entry_id)
        if not entry:
            return False

        await self.db.delete(entry)
        await self.db.commit()

        logger.info(f"Deleted entry: id={entry_id}")
        return True

    async def bulk_add_entries(
        self,
        glossary_id: UUID,
        entries: list[dict],
    ) -> tuple[int, int, list[str]]:
        """
        Bulk add entries to a glossary.

        Args:
            glossary_id: Glossary UUID
            entries: List of entry dicts with source_term, translations, etc.

        Returns:
            Tuple of (added_count, skipped_count, errors)
        """
        glossary = await self.get_glossary(glossary_id)
        if not glossary:
            return 0, 0, ["Glossary not found"]

        added = 0
        skipped = 0
        errors = []

        for entry_data in entries:
            try:
                source_term = entry_data.get("source_term")
                translations = entry_data.get("translations", {})

                if not source_term:
                    skipped += 1
                    errors.append("Missing source_term")
                    continue

                if not translations:
                    skipped += 1
                    errors.append(f"No translations for '{source_term}'")
                    continue

                entry = GlossaryEntry(
                    glossary_id=glossary_id,
                    source_term=source_term,
                    source_term_normalized=source_term.lower(),
                    translations=translations,
                    context=entry_data.get("context"),
                    notes=entry_data.get("notes"),
                    priority=entry_data.get("priority", 0),
                    case_sensitive=entry_data.get("case_sensitive", False),
                    match_whole_word=entry_data.get("match_whole_word", True),
                )
                self.db.add(entry)
                added += 1

            except Exception as e:
                skipped += 1
                errors.append(str(e))

        if added > 0:
            await self.db.commit()

        logger.info(f"Bulk import to glossary {glossary_id}: " f"added={added}, skipped={skipped}")

        return added, skipped, errors

    # =========================================================================
    # Term Matching & Lookup
    # =========================================================================

    async def get_glossary_terms(
        self,
        target_language: str,
        glossary_id: UUID | None = None,
        domain: str | None = None,
        include_default: bool = True,
    ) -> dict[str, str]:
        """
        Get glossary terms formatted for use in translation.

        Args:
            target_language: Target language code
            glossary_id: Specific glossary to use (optional)
            domain: Domain to filter by (optional)
            include_default: Include default glossaries (default: True)

        Returns:
            Dict mapping source terms to translations
        """
        terms = {}

        # Get specific glossary if provided
        if glossary_id:
            glossary = await self.get_glossary_with_entries(glossary_id)
            if glossary and glossary.entries:
                for entry in glossary.entries:
                    if entry.translations and target_language in entry.translations:
                        terms[entry.source_term] = entry.translations[target_language]

        # Get default glossaries
        if include_default:
            conditions = [Glossary.is_default, Glossary.is_active]
            if domain:
                conditions.append(Glossary.domain == domain)

            stmt = select(Glossary).options(selectinload(Glossary.entries)).where(and_(*conditions))
            result = await self.db.execute(stmt)
            default_glossaries = result.scalars().all()

            for glossary in default_glossaries:
                if glossary.entries:
                    for entry in glossary.entries:
                        if entry.translations and target_language in entry.translations:
                            # Don't override specific glossary terms
                            if entry.source_term not in terms:
                                terms[entry.source_term] = entry.translations[target_language]

        return terms

    async def find_matching_terms(
        self,
        text: str,
        glossary_id: UUID | None = None,
        domain: str | None = None,
        target_language: str | None = None,
    ) -> list[dict]:
        """
        Find glossary terms that match within the given text.

        Args:
            text: Text to search for terms
            glossary_id: Specific glossary to search
            domain: Domain to filter by
            target_language: Target language for translations

        Returns:
            List of match dicts with term, position, and translation info
        """
        matches = []

        # Build query for entries
        if glossary_id:
            entries = await self.list_entries(glossary_id)
        else:
            # Get entries from all active glossaries
            conditions = [Glossary.is_active]
            if domain:
                conditions.append(Glossary.domain == domain)

            stmt = select(Glossary).options(selectinload(Glossary.entries)).where(and_(*conditions))
            result = await self.db.execute(stmt)
            glossaries = result.scalars().all()

            entries = []
            for g in glossaries:
                if g.entries:
                    entries.extend(g.entries)

        # Find matches
        for entry in entries:
            pattern = entry.source_term
            if not entry.case_sensitive:
                pattern = re.escape(pattern)
                flags = re.IGNORECASE
            else:
                pattern = re.escape(pattern)
                flags = 0

            if entry.whole_word:
                pattern = r"\b" + pattern + r"\b"

            for match in re.finditer(pattern, text, flags):
                match_info = {
                    "term": entry.source_term,
                    "start": match.start(),
                    "end": match.end(),
                    "matched_text": match.group(),
                    "entry_id": str(entry.entry_id),
                    "priority": entry.priority,
                }

                if target_language and entry.translations:
                    match_info["translation"] = entry.translations.get(target_language)

                matches.append(match_info)

        # Sort by position, then priority
        matches.sort(key=lambda m: (m["start"], -m["priority"]))

        return matches
