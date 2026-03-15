"""Compatibility shim — re-exports unified model aliases from models.py.

The meeting pipeline was initially written against this module with names
``MeetingSession``, ``MeetingTranscript``, and ``SessionTranslation``.
All ORM classes now live in ``database.models`` and map to the canonical
tables (meetings, meeting_chunks, meeting_translations).

Import from here or from ``database.models`` — both work identically.
"""

from .models import (  # noqa: F401
    Meeting as MeetingSession,
    MeetingChunk as MeetingTranscript,
    MeetingTranslation as SessionTranslation,
)

__all__ = ["MeetingSession", "MeetingTranscript", "SessionTranslation"]
