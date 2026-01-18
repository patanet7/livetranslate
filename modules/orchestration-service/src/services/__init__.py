"""
Services module for Fireflies integration.

Contains business logic services:
- SentenceAggregator: Aggregates transcript chunks into complete sentences
- RollingWindowTranslator: Context-aware translation with glossary
- GlossaryService: CRUD for translation glossaries
- CaptionBuffer: Display queue management for captions
"""

from .caption_buffer import (
    Caption,
    CaptionBuffer,
    SessionCaptionManager,
    create_caption_buffer,
)
from .glossary_service import GlossaryService
from .rolling_window_translator import (
    RollingWindowTranslator,
    create_rolling_window_translator,
)
from .sentence_aggregator import SentenceAggregator

__all__ = [
    "Caption",
    "CaptionBuffer",
    "GlossaryService",
    "RollingWindowTranslator",
    "SentenceAggregator",
    "SessionCaptionManager",
    "create_caption_buffer",
    "create_rolling_window_translator",
]
