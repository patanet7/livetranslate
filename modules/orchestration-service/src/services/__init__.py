"""
Services module for Fireflies integration.

Contains business logic services:
- SentenceAggregator: Aggregates transcript chunks into complete sentences
- RollingWindowTranslator: Context-aware translation with glossary
- GlossaryService: CRUD for translation glossaries
- CaptionBuffer: Display queue management for captions
"""

from .sentence_aggregator import SentenceAggregator
from .glossary_service import GlossaryService
from .rolling_window_translator import (
    RollingWindowTranslator,
    create_rolling_window_translator,
)
from .caption_buffer import (
    CaptionBuffer,
    Caption,
    SessionCaptionManager,
    create_caption_buffer,
)

__all__ = [
    "SentenceAggregator",
    "GlossaryService",
    "RollingWindowTranslator",
    "create_rolling_window_translator",
    "CaptionBuffer",
    "Caption",
    "SessionCaptionManager",
    "create_caption_buffer",
]
