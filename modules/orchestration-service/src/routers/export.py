"""
Meeting Export Router

Download meeting transcripts and translations in various formats.
"""

import io
import zipfile
from typing import Literal

from database import get_db_session
from database.models import Meeting, MeetingSentence
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from livetranslate_common.logging import get_logger
from services.export_service import to_json, to_pdf, to_srt, to_txt, to_vtt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = get_logger()
router = APIRouter(tags=["Export"])

ExportFormat = Literal["srt", "vtt", "txt", "json", "pdf"]

MIME_TYPES: dict[str, str] = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "json": "application/json",
    "pdf": "application/pdf",
}


async def _get_meeting_with_data(
    db: AsyncSession, meeting_id: str, *, load_translations: bool = False
):
    """Fetch meeting with sentences and optionally translations."""
    stmt = select(Meeting).where(Meeting.id == meeting_id)
    if load_translations:
        stmt = stmt.options(
            selectinload(Meeting.sentences).selectinload(MeetingSentence.translations)
        )
    else:
        stmt = stmt.options(selectinload(Meeting.sentences))
    result = await db.execute(stmt)
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
    return meeting


def _extract_sentences(meeting) -> list[dict]:
    """Extract sorted sentence dicts from a meeting."""
    return [
        {
            "text": s.text,
            "speaker_name": s.speaker_name,
            "start_time": s.start_time,
            "end_time": s.end_time,
        }
        for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0)
    ]


@router.get("/meetings/{meeting_id}/transcript")
async def export_transcript(
    meeting_id: str,
    format: ExportFormat = Query(default="srt"),
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting transcript in the specified format."""
    meeting = await _get_meeting_with_data(db, meeting_id)
    sentences = _extract_sentences(meeting)

    if format == "pdf":
        content = to_pdf(meeting, sentences, [])
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{meeting.title or meeting_id}.pdf"'
            },
        )

    converters = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}
    if format == "json":
        content = to_json(meeting, sentences, [])
    else:
        content = converters[format](sentences)

    filename = f"{meeting.title or meeting_id}.{format}"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=MIME_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/meetings/{meeting_id}/translations")
async def export_translations(
    meeting_id: str,
    lang: str = Query(description="Target language code (e.g., 'es')"),
    format: ExportFormat = Query(default="srt"),
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting translations for a specific language."""
    meeting = await _get_meeting_with_data(db, meeting_id, load_translations=True)
    sentences = []
    for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0):
        translation = next(
            (t for t in s.translations if t.target_language == lang),
            None,
        )
        if translation:
            sentences.append(
                {
                    "text": translation.translated_text,
                    "speaker_name": s.speaker_name,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                }
            )

    if not sentences:
        raise HTTPException(
            status_code=404, detail=f"No translations found for language '{lang}'"
        )

    converters = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}
    if format in converters:
        content = converters[format](sentences)
    elif format == "json":
        import json

        content = json.dumps(sentences, default=str, indent=2)
    else:
        content = to_txt(sentences)

    filename = f"{meeting.title or meeting_id}_{lang}.{format}"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=MIME_TYPES.get(format, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/meetings/{meeting_id}/archive")
async def export_archive(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting as ZIP archive with transcript, translations, and metadata."""
    meeting = await _get_meeting_with_data(db, meeting_id, load_translations=True)
    sentences = _extract_sentences(meeting)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("transcript.srt", to_srt(sentences))
        zf.writestr("transcript.txt", to_txt(sentences))
        zf.writestr("transcript.vtt", to_vtt(sentences))
        zf.writestr("metadata.json", to_json(meeting, sentences, []))

        # Add translations per language
        languages: set[str] = set()
        for s in meeting.sentences:
            for t in s.translations:
                languages.add(t.target_language)
        for lang in sorted(languages):
            lang_sentences = []
            for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0):
                translation = next(
                    (t for t in s.translations if t.target_language == lang), None
                )
                if translation:
                    lang_sentences.append(
                        {
                            "text": translation.translated_text,
                            "speaker_name": s.speaker_name,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                        }
                    )
            if lang_sentences:
                zf.writestr(f"translations/{lang}.srt", to_srt(lang_sentences))

    buf.seek(0)
    filename = f"{meeting.title or meeting_id}_archive.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
