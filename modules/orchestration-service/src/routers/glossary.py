"""
Glossary Router

FastAPI router for managing translation glossaries.
Provides REST API endpoints for:
- Glossary CRUD operations
- Entry management
- Bulk import/export
- Term lookup

Integrates with the GlossaryService for business logic.
"""

from uuid import UUID

from database.database import DatabaseManager
from dependencies import get_database_manager
from fastapi import APIRouter, Depends, HTTPException, status
from livetranslate_common.logging import get_logger
from pydantic import BaseModel, Field
from services.glossary_service import GlossaryService
from sqlalchemy.orm import Session as DBSession

logger = get_logger()

router = APIRouter(prefix="/glossaries", tags=["glossaries"])


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session(
    db_manager: DatabaseManager = Depends(get_database_manager),
):
    """Get database session for request scope."""
    session = await db_manager.get_session_direct()
    try:
        yield session
    finally:
        await session.close()


async def get_glossary_service(
    db: DBSession = Depends(get_db_session),
) -> GlossaryService:
    """Get GlossaryService instance with database session."""
    return GlossaryService(db)


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateGlossaryRequest(BaseModel):
    """Request to create a new glossary."""

    name: str = Field(..., min_length=1, max_length=255, description="Glossary name")
    description: str | None = Field(None, description="Glossary description")
    domain: str | None = Field(
        None, max_length=100, description="Domain (e.g., 'medical', 'legal', 'tech')"
    )
    source_language: str = Field("en", description="Source language code")
    target_languages: list[str] = Field(
        ..., min_length=1, description="List of target language codes"
    )
    is_default: bool = Field(False, description="Whether this is a default glossary")


class UpdateGlossaryRequest(BaseModel):
    """Request to update a glossary."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    domain: str | None = None
    target_languages: list[str] | None = None
    is_active: bool | None = None


class GlossaryResponse(BaseModel):
    """Glossary response model."""

    glossary_id: str
    name: str
    description: str | None
    domain: str | None
    source_language: str
    target_languages: list[str]
    is_active: bool
    is_default: bool
    entry_count: int
    created_at: str | None
    updated_at: str | None


class CreateEntryRequest(BaseModel):
    """Request to create a glossary entry."""

    source_term: str = Field(..., min_length=1, max_length=500, description="Source term")
    translations: dict[str, str] = Field(..., description="Translations by target language")
    context: str | None = Field(None, description="Usage context or example")
    notes: str | None = Field(None, description="Internal notes")
    case_sensitive: bool = Field(False, description="Case-sensitive matching")
    match_whole_word: bool = Field(True, description="Match whole words only")
    priority: int = Field(0, ge=0, description="Priority (higher = more important)")


class UpdateEntryRequest(BaseModel):
    """Request to update a glossary entry."""

    source_term: str | None = None
    translations: dict[str, str] | None = None
    context: str | None = None
    notes: str | None = None
    case_sensitive: bool | None = None
    match_whole_word: bool | None = None
    priority: int | None = None


class EntryResponse(BaseModel):
    """Glossary entry response model."""

    entry_id: str
    glossary_id: str
    source_term: str
    translations: dict[str, str]
    context: str | None
    notes: str | None
    case_sensitive: bool
    match_whole_word: bool
    priority: int
    created_at: str | None
    updated_at: str | None


class BulkImportRequest(BaseModel):
    """Request to bulk import entries."""

    entries: list[CreateEntryRequest] = Field(
        ..., min_length=1, description="List of entries to import"
    )


class BulkImportResponse(BaseModel):
    """Response from bulk import."""

    successful: int
    failed: int
    total: int


class TermLookupRequest(BaseModel):
    """Request to look up terms in text."""

    text: str = Field(..., min_length=1, description="Text to search for terms")
    target_language: str = Field(..., description="Target language for translations")
    domain: str | None = Field(None, description="Domain filter")


class TermMatch(BaseModel):
    """A matched term in text."""

    source_term: str
    translation: str
    start_pos: int
    end_pos: int


class TermLookupResponse(BaseModel):
    """Response from term lookup."""

    matches: list[TermMatch]
    match_count: int


# =============================================================================
# Glossary Endpoints
# =============================================================================


@router.get("", response_model=list[GlossaryResponse])
async def list_glossaries(
    domain: str | None = None,
    source_language: str | None = None,
    active_only: bool = True,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    List all glossaries with optional filters.

    - **domain**: Filter by domain (e.g., 'medical', 'legal')
    - **source_language**: Filter by source language code
    - **active_only**: Only return active glossaries (default: true)
    """
    glossaries = await service.list_glossaries(
        domain=domain,
        source_language=source_language,
        active_only=active_only,
    )

    return [
        GlossaryResponse(
            glossary_id=str(g.glossary_id),
            name=g.name,
            description=g.description,
            domain=g.domain,
            source_language=g.source_language,
            target_languages=g.target_languages or [],
            is_active=g.is_active,
            is_default=g.is_default,
            entry_count=g.entry_count or 0,
            created_at=g.created_at.isoformat() if g.created_at else None,
            updated_at=g.updated_at.isoformat() if g.updated_at else None,
        )
        for g in glossaries
    ]


@router.post("", response_model=GlossaryResponse, status_code=status.HTTP_201_CREATED)
async def create_glossary(
    request: CreateGlossaryRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Create a new glossary.

    Glossaries can be:
    - **Default**: Applied to all translations in a domain
    - **Session-specific**: Applied only when explicitly referenced
    """
    glossary = await service.create_glossary(
        name=request.name,
        description=request.description,
        domain=request.domain,
        source_language=request.source_language,
        target_languages=request.target_languages,
        is_default=request.is_default,
    )

    return GlossaryResponse(
        glossary_id=str(glossary.glossary_id),
        name=glossary.name,
        description=glossary.description,
        domain=glossary.domain,
        source_language=glossary.source_language,
        target_languages=glossary.target_languages or [],
        is_active=glossary.is_active,
        is_default=glossary.is_default,
        entry_count=glossary.entry_count or 0,
        created_at=glossary.created_at.isoformat() if glossary.created_at else None,
        updated_at=glossary.updated_at.isoformat() if glossary.updated_at else None,
    )


@router.get("/{glossary_id}", response_model=GlossaryResponse)
async def get_glossary(
    glossary_id: UUID,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Get a glossary by ID."""
    glossary = await service.get_glossary(glossary_id)
    if not glossary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )

    return GlossaryResponse(
        glossary_id=str(glossary.glossary_id),
        name=glossary.name,
        description=glossary.description,
        domain=glossary.domain,
        source_language=glossary.source_language,
        target_languages=glossary.target_languages or [],
        is_active=glossary.is_active,
        is_default=glossary.is_default,
        entry_count=glossary.entry_count or 0,
        created_at=glossary.created_at.isoformat() if glossary.created_at else None,
        updated_at=glossary.updated_at.isoformat() if glossary.updated_at else None,
    )


@router.patch("/{glossary_id}", response_model=GlossaryResponse)
async def update_glossary(
    glossary_id: UUID,
    request: UpdateGlossaryRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Update a glossary."""
    glossary = await service.update_glossary(
        glossary_id=glossary_id,
        name=request.name,
        description=request.description,
        domain=request.domain,
        target_languages=request.target_languages,
        is_active=request.is_active,
    )

    if not glossary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )

    return GlossaryResponse(
        glossary_id=str(glossary.glossary_id),
        name=glossary.name,
        description=glossary.description,
        domain=glossary.domain,
        source_language=glossary.source_language,
        target_languages=glossary.target_languages or [],
        is_active=glossary.is_active,
        is_default=glossary.is_default,
        entry_count=glossary.entry_count or 0,
        created_at=glossary.created_at.isoformat() if glossary.created_at else None,
        updated_at=glossary.updated_at.isoformat() if glossary.updated_at else None,
    )


@router.delete("/{glossary_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_glossary(
    glossary_id: UUID,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Delete a glossary and all its entries.

    This operation is irreversible.
    """
    deleted = await service.delete_glossary(glossary_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )


@router.get("/{glossary_id}/stats")
async def get_glossary_stats(
    glossary_id: UUID,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Get statistics for a glossary."""
    stats = await service.get_glossary_stats(glossary_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )

    return stats


# =============================================================================
# Entry Endpoints
# =============================================================================


@router.get("/{glossary_id}/entries", response_model=list[EntryResponse])
async def list_entries(
    glossary_id: UUID,
    target_language: str | None = None,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    List entries for a glossary.

    - **target_language**: Filter entries that have a translation for this language
    """
    # Verify glossary exists
    glossary = await service.get_glossary(glossary_id)
    if not glossary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )

    entries = await service.list_entries(
        glossary_id=glossary_id,
        target_language=target_language,
    )

    return [
        EntryResponse(
            entry_id=str(e.entry_id),
            glossary_id=str(e.glossary_id),
            source_term=e.source_term,
            translations=e.translations or {},
            context=e.context,
            notes=e.notes,
            case_sensitive=e.case_sensitive,
            match_whole_word=e.match_whole_word,
            priority=e.priority,
            created_at=e.created_at.isoformat() if e.created_at else None,
            updated_at=e.updated_at.isoformat() if e.updated_at else None,
        )
        for e in entries
    ]


@router.post(
    "/{glossary_id}/entries",
    response_model=EntryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_entry(
    glossary_id: UUID,
    request: CreateEntryRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Add a new entry to a glossary."""
    entry = await service.add_entry(
        glossary_id=glossary_id,
        source_term=request.source_term,
        translations=request.translations,
        context=request.context,
        notes=request.notes,
        case_sensitive=request.case_sensitive,
        match_whole_word=request.match_whole_word,
        priority=request.priority,
    )

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Glossary {glossary_id} not found",
        )

    return EntryResponse(
        entry_id=str(entry.entry_id),
        glossary_id=str(entry.glossary_id),
        source_term=entry.source_term,
        translations=entry.translations or {},
        context=entry.context,
        notes=entry.notes,
        case_sensitive=entry.case_sensitive,
        match_whole_word=entry.match_whole_word,
        priority=entry.priority,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
        updated_at=entry.updated_at.isoformat() if entry.updated_at else None,
    )


@router.patch("/{glossary_id}/entries/{entry_id}", response_model=EntryResponse)
async def update_entry(
    glossary_id: UUID,
    entry_id: UUID,
    request: UpdateEntryRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Update a glossary entry."""
    # Verify entry belongs to glossary
    entry = await service.get_entry(entry_id)
    if not entry or entry.glossary_id != glossary_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found in glossary {glossary_id}",
        )

    updated = await service.update_entry(
        entry_id=entry_id,
        source_term=request.source_term,
        translations=request.translations,
        context=request.context,
        notes=request.notes,
        case_sensitive=request.case_sensitive,
        match_whole_word=request.match_whole_word,
        priority=request.priority,
    )

    return EntryResponse(
        entry_id=str(updated.entry_id),
        glossary_id=str(updated.glossary_id),
        source_term=updated.source_term,
        translations=updated.translations or {},
        context=updated.context,
        notes=updated.notes,
        case_sensitive=updated.case_sensitive,
        match_whole_word=updated.match_whole_word,
        priority=updated.priority,
        created_at=updated.created_at.isoformat() if updated.created_at else None,
        updated_at=updated.updated_at.isoformat() if updated.updated_at else None,
    )


@router.delete("/{glossary_id}/entries/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_entry(
    glossary_id: UUID,
    entry_id: UUID,
    service: GlossaryService = Depends(get_glossary_service),
):
    """Delete a glossary entry."""
    # Verify entry belongs to glossary
    entry = await service.get_entry(entry_id)
    if not entry or entry.glossary_id != glossary_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entry {entry_id} not found in glossary {glossary_id}",
        )

    await service.delete_entry(entry_id)


# =============================================================================
# Bulk Operations
# =============================================================================


@router.post("/{glossary_id}/import", response_model=BulkImportResponse)
async def bulk_import_entries(
    glossary_id: UUID,
    request: BulkImportRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Bulk import entries into a glossary.

    Imports multiple entries in a single transaction.
    Returns count of successful and failed imports.
    """
    # Convert request entries to dicts
    entries_data = [
        {
            "source_term": e.source_term,
            "translations": e.translations,
            "context": e.context,
            "notes": e.notes,
            "case_sensitive": e.case_sensitive,
            "match_whole_word": e.match_whole_word,
            "priority": e.priority,
        }
        for e in request.entries
    ]

    added, skipped, _errors = await service.bulk_add_entries(
        glossary_id=glossary_id,
        entries=entries_data,
    )
    successful = added
    failed = skipped

    if successful == 0 and failed == len(request.entries):
        # Check if glossary exists
        glossary = await service.get_glossary(glossary_id)
        if not glossary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Glossary {glossary_id} not found",
            )

    return BulkImportResponse(
        successful=successful,
        failed=failed,
        total=len(request.entries),
    )


# =============================================================================
# Term Lookup
# =============================================================================


@router.post("/{glossary_id}/lookup", response_model=TermLookupResponse)
async def lookup_terms(
    glossary_id: UUID,
    request: TermLookupRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Find glossary terms that match within the given text.

    Returns all matching terms with their positions and translations.
    Useful for highlighting terms or verifying glossary coverage.
    """
    matches = await service.find_matching_terms(
        text=request.text,
        glossary_id=glossary_id,
        target_language=request.target_language,
        domain=request.domain,
    )

    return TermLookupResponse(
        matches=[
            TermMatch(
                source_term=m[0],
                translation=m[1],
                start_pos=m[2],
                end_pos=m[3],
            )
            for m in matches
        ],
        match_count=len(matches),
    )


@router.post("/lookup", response_model=TermLookupResponse)
async def lookup_terms_global(
    request: TermLookupRequest,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Find glossary terms across all applicable glossaries.

    Searches default glossaries and optionally domain-filtered glossaries.
    Does not require a specific glossary_id.
    """
    matches = await service.find_matching_terms(
        text=request.text,
        glossary_id=None,  # Search all applicable
        target_language=request.target_language,
        domain=request.domain,
    )

    return TermLookupResponse(
        matches=[
            TermMatch(
                source_term=m[0],
                translation=m[1],
                start_pos=m[2],
                end_pos=m[3],
            )
            for m in matches
        ],
        match_count=len(matches),
    )


# =============================================================================
# Translation Helper
# =============================================================================


@router.get("/terms/{target_language}")
async def get_terms_for_translation(
    target_language: str,
    glossary_id: UUID | None = None,
    domain: str | None = None,
    include_default: bool = True,
    service: GlossaryService = Depends(get_glossary_service),
):
    """
    Get glossary terms formatted for use in translation.

    Returns a dict mapping source terms to their translations
    for the specified target language. This is the format
    expected by the RollingWindowTranslator.

    - **target_language**: Target language code (required)
    - **glossary_id**: Specific glossary to use (optional)
    - **domain**: Filter by domain (optional)
    - **include_default**: Include default glossary terms (default: true)
    """
    terms = await service.get_glossary_terms(
        glossary_id=glossary_id,
        target_language=target_language,
        domain=domain,
        include_default=include_default,
    )

    return {
        "target_language": target_language,
        "term_count": len(terms),
        "terms": terms,
    }
