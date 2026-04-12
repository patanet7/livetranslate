#!/usr/bin/env python3
"""
Virtual Webcam Generator for Google Meet Bot

Generates a virtual webcam stream that displays real-time translations
for Google Meet sessions. Integrated into the orchestration service
for centralized bot management.

Features:
- Real-time translation display overlay
- Multi-language support with customizable layouts
- Speaker identification and attribution
- Translation confidence indicators
- Customizable themes and styling
- Performance optimization for low latency
"""

import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from livetranslate_common.logging import get_logger
from livetranslate_common.theme import (
    SPEAKER_COLORS,
    DisplayMode,
    ThemeColors,
    get_theme_colors,
    hex_to_rgb,
)
from PIL import Image, ImageDraw, ImageFont

from bot.text_wrapper import wrap_text

logger = get_logger()


class Theme(Enum):
    """Visual themes for translation display."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    MINIMAL = "minimal"
    CORPORATE = "corporate"


@dataclass
class WebcamConfig:
    """Virtual webcam configuration."""

    width: int = 1280
    height: int = 720
    fps: int = 30
    format: str = "RGB24"
    device_name: str = "LiveTranslate Virtual Camera"
    display_mode: DisplayMode = DisplayMode.SUBTITLE
    theme: Theme = Theme.DARK
    max_translations_displayed: int = 5
    translation_duration_seconds: float = 10.0
    font_size: int = 24
    background_opacity: float = 0.8
    show_speaker_names: bool = True
    show_confidence: bool = True
    show_timestamps: bool = False


@dataclass
class TranslationDisplay:
    """Translation display item."""

    translation_id: str
    text: str
    source_language: str
    target_language: str
    speaker_name: str | None
    confidence: float
    timestamp: datetime
    display_position: tuple[int, int] = (0, 0)
    expires_at: datetime | None = None


@dataclass
class SpeakerInfo:
    """Speaker display information."""

    speaker_id: str
    speaker_name: str
    color: tuple[int, int, int]
    position: tuple[int, int]
    last_active: datetime


class VirtualWebcamManager:
    """
    Manages virtual webcam generation for displaying translations.
    """

    def __init__(self, config: WebcamConfig, bot_manager=None):
        self.config = config
        self.bot_manager = bot_manager

        # Display state
        self.is_streaming = False
        self.current_translations = deque(maxlen=config.max_translations_displayed)
        self.speakers = {}  # speaker_id -> SpeakerInfo
        self.speaker_colors = [hex_to_rgb(c) for c in SPEAKER_COLORS]
        self.next_speaker_color = 0

        # Rendering
        self.current_frame = None
        self.frame_lock = threading.RLock()
        self.stream_thread = None

        # Fonts and styling
        self.fonts = {}
        self._load_fonts()

        # Metrics
        self.frames_generated = 0
        self.start_time = None
        self.last_translation_time = None

        # Callbacks
        self.on_frame_generated = None
        self.on_error = None

        logger.info("Virtual Webcam Manager initialized")
        logger.info(f"  Resolution: {config.width}x{config.height}@{config.fps}fps")
        logger.info(f"  Display mode: {config.display_mode.value}")
        logger.info(f"  Theme: {config.theme.value}")

    def _load_fonts(self):
        """Load fonts for text rendering. Prioritizes CJK-capable fonts."""
        try:
            # CJK-capable fonts first, then Latin fallbacks.
            # Container (Dockerfile): fonts-unifont covers CJK.
            # macOS: STHeiti, Hiragino, PingFang cover CJK.
            font_paths = [
                # Container (Linux) — Pan-CJK + Latin capable
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/unifont/unifont.ttf",
                # macOS — Pan-Unicode (CJK + Latin + accented chars)
                "/Library/Fonts/Arial Unicode.ttf",  # Full Unicode coverage (best)
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS 10.15+
                "/System/Library/Fonts/PingFang.ttc",  # CJK + some Latin
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Korean + CJK (limited Latin)
                "/System/Library/Fonts/Hiragino Sans GB.ttc",  # Chinese + Japanese
                "/System/Library/Fonts/STHeiti Medium.ttc",  # Chinese only (fallback)
                # Windows — Pan-CJK
                "C:/Windows/Fonts/arial.ttf",  # Good Latin support
                "C:/Windows/Fonts/malgun.ttf",  # Korean
                "C:/Windows/Fonts/msyh.ttc",  # Chinese
                "C:/Windows/Fonts/meiryo.ttc",  # Japanese
                # Latin fallbacks (no CJK)
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Arial.ttf",
            ]

            font_loaded = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.fonts["regular"] = ImageFont.truetype(font_path, self.config.font_size)
                        self.fonts["bold"] = ImageFont.truetype(
                            font_path, self.config.font_size + 4
                        )
                        self.fonts["small"] = ImageFont.truetype(
                            font_path, self.config.font_size - 6
                        )
                        font_loaded = True
                        logger.info("font_loaded", path=font_path, size=self.config.font_size)
                        break
                    except Exception as e:
                        logger.debug("font_load_failed", path=font_path, error=str(e))

            if not font_loaded:
                self.fonts["regular"] = ImageFont.load_default()
                self.fonts["bold"] = ImageFont.load_default()
                self.fonts["small"] = ImageFont.load_default()
                logger.warning("font_fallback_default", hint="CJK text will render as boxes")

        except Exception as e:
            logger.error("font_load_error", error=str(e))
            self.fonts["regular"] = ImageFont.load_default()
            self.fonts["bold"] = ImageFont.load_default()
            self.fonts["small"] = ImageFont.load_default()

    def get_theme_colors(self) -> dict[str, tuple[int, int, int]]:
        """Get color scheme for current theme."""
        theme_colors: ThemeColors = get_theme_colors(self.config.theme.value)
        # Derive overlay_bg: dark themes use black, light themes use white
        is_dark = theme_colors.background[0] < 128
        overlay_bg = (0, 0, 0) if is_dark else (255, 255, 255)
        return {
            "background": theme_colors.background,
            "text_primary": theme_colors.text_primary,
            "text_secondary": theme_colors.text_secondary,
            "accent": theme_colors.accent,
            "border": theme_colors.border,
            "overlay_bg": overlay_bg,
        }

    async def start_stream(self, session_id: str) -> bool:
        """Start virtual webcam streaming."""
        try:
            if self.is_streaming:
                logger.warning("Virtual webcam already streaming")
                return False

            self.session_id = session_id
            self.start_time = time.time()
            self.is_streaming = True

            # Initialize frame
            self._initialize_frame()

            # Start streaming thread
            self.stream_thread = threading.Thread(
                target=self._stream_loop,
                daemon=True,
                name=f"VirtualWebcam-{session_id}",
            )
            self.stream_thread.start()

            logger.info(f"Started virtual webcam stream for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start virtual webcam: {e}")
            if self.on_error:
                self.on_error(f"Webcam start failed: {e}")
            return False

    async def stop_stream(self) -> bool:
        """Stop virtual webcam streaming."""
        try:
            if not self.is_streaming:
                logger.warning("Virtual webcam not streaming")
                return False

            self.is_streaming = False

            # Wait for stream thread
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5.0)

            # Calculate final stats
            duration = time.time() - self.start_time if self.start_time else 0
            fps = self.frames_generated / duration if duration > 0 else 0

            logger.info("Stopped virtual webcam stream")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Frames generated: {self.frames_generated}")
            logger.info(f"  Average FPS: {fps:.1f}")

            return True

        except Exception as e:
            logger.error(f"Error stopping virtual webcam: {e}")
            if self.on_error:
                self.on_error(f"Webcam stop failed: {e}")
            return False

    def add_translation(self, translation_data: dict[str, Any]):
        """Add a new translation to display."""
        try:
            # Determine display text and type
            is_original = translation_data.get("is_original_transcription", False)
            display_text = translation_data["translated_text"]

            # Format speaker name with diarization info if available
            speaker_name = translation_data.get("speaker_name")
            speaker_id = translation_data.get("speaker_id")

            # Enhanced speaker name formatting
            if speaker_name and speaker_id:
                # Check if speaker_id contains diarization info (e.g., "SPEAKER_00", "diarized_speaker_1")
                if "SPEAKER_" in speaker_id or "diarized" in speaker_id.lower():
                    formatted_speaker = f"{speaker_name} ({speaker_id})"
                else:
                    formatted_speaker = speaker_name
            elif speaker_id:
                formatted_speaker = speaker_id
            elif speaker_name:
                formatted_speaker = speaker_name
            else:
                formatted_speaker = "Unknown Speaker"

            # Create translation display item
            translation = TranslationDisplay(
                translation_id=translation_data.get("translation_id", str(uuid.uuid4())),
                text=display_text,
                source_language=translation_data.get("source_language", "auto"),
                target_language=translation_data.get("target_language", "en"),
                speaker_name=formatted_speaker,
                confidence=translation_data.get(
                    "translation_confidence", 0.0
                ),  # Default to 0.0 instead of 1.0
                timestamp=datetime.now(UTC),
                expires_at=datetime.now(UTC)
                + timedelta(seconds=self.config.translation_duration_seconds),
            )

            # Add speaker if new
            if speaker_id and speaker_id not in self.speakers:
                self._add_speaker(speaker_id, formatted_speaker)

            # Update speaker activity
            if speaker_id in self.speakers:
                self.speakers[speaker_id].last_active = datetime.now(UTC)

            # Add to display queue
            with self.frame_lock:
                self.current_translations.append(translation)
                self.last_translation_time = time.time()

            # Log with appropriate prefix
            prefix = "TRANSCRIPTION" if is_original else "TRANSLATION"
            lang_info = f"[{translation_data.get('source_language', 'auto')} → {translation_data.get('target_language', 'en')}]"
            logger.info(f"{prefix} {lang_info} {formatted_speaker}: {display_text[:100]}...")

        except Exception as e:
            logger.error(f"Error adding translation: {e}")

    def _add_speaker(self, speaker_id: str, speaker_name: str | None):
        """Add a new speaker to tracking."""
        color = self.speaker_colors[self.next_speaker_color % len(self.speaker_colors)]
        self.next_speaker_color += 1

        self.speakers[speaker_id] = SpeakerInfo(
            speaker_id=speaker_id,
            speaker_name=speaker_name or f"Speaker {len(self.speakers) + 1}",
            color=color,
            position=(0, 0),  # Will be calculated during layout
            last_active=datetime.now(UTC),
        )

        logger.info(f"Added speaker: {speaker_name} ({speaker_id})")

    def _initialize_frame(self):
        """Initialize the base frame."""
        with self.frame_lock:
            # All canonical display modes use solid backgrounds
            colors = self.get_theme_colors()
            bg_color = colors["background"]
            self.current_frame = np.full(
                (self.config.height, self.config.width, 3), bg_color, dtype=np.uint8
            )

    def _stream_loop(self):
        """Main streaming loop with frame-paced timer."""
        logger.info("Virtual webcam stream loop started")

        frame_interval = 1.0 / self.config.fps
        next_frame_time = time.monotonic()

        try:
            while self.is_streaming:
                next_frame_time += frame_interval

                # Generate new frame
                self._generate_frame()
                self.frames_generated += 1

                # Callback for frame generation
                if self.on_frame_generated and self.current_frame is not None:
                    self.on_frame_generated(self.current_frame.copy())

                # Clean up expired translations
                self._cleanup_expired_translations()

                # Sleep until next frame is due
                sleep_time = next_frame_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Frame took too long — skip to next slot
                    next_frame_time = time.monotonic()

        except Exception as e:
            logger.error(f"Error in webcam stream loop: {e}")
            if self.on_error:
                self.on_error(f"Stream loop error: {e}")

        logger.info("Virtual webcam stream loop ended")

    def _generate_frame(self):
        """Generate a single frame with current translations."""
        try:
            with self.frame_lock:
                if not self.current_translations:
                    # No translations to display - show waiting message
                    self._render_waiting_frame()
                else:
                    # Render translations based on canonical display mode
                    if self.config.display_mode == DisplayMode.SUBTITLE:
                        self._render_banner_frame()
                    elif self.config.display_mode == DisplayMode.SPLIT:
                        self._render_sidebar_frame()
                    elif self.config.display_mode == DisplayMode.INTERPRETER:
                        self._render_fullscreen_frame()

        except Exception as e:
            logger.error(f"Error generating frame: {e}")

    def _render_waiting_frame(self):
        """Render frame when no translations are available."""
        self._initialize_frame()
        colors = self.get_theme_colors()

        # Convert to PIL for text rendering (all canonical modes use RGB)
        img = Image.fromarray(self.current_frame, "RGB")

        draw = ImageDraw.Draw(img)

        # Draw waiting message
        text = "Waiting for translations..."
        font = self.fonts.get("regular", ImageFont.load_default())

        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.config.width - text_width) // 2
        y = (self.config.height - text_height) // 2

        draw.text((x, y), text, fill=colors["text_secondary"], font=font)

        # Convert back to numpy
        self.current_frame = np.array(img)

    def _render_overlay_frame(self):
        """Render overlay mode with floating translation boxes."""
        self._initialize_frame()
        colors = self.get_theme_colors()

        # Convert to PIL for text rendering
        img = Image.fromarray(self.current_frame, "RGB")
        draw = ImageDraw.Draw(img)

        # Group translations by speaker and sort by timestamp
        active_translations = []
        for translation in self.current_translations:
            if not (translation.expires_at and datetime.now(UTC) > translation.expires_at):
                active_translations.append(translation)

        # Sort by timestamp (newest first)
        active_translations.sort(key=lambda t: t.timestamp, reverse=True)

        # Render each translation with improved spacing
        y_offset = 30
        max_display = min(len(active_translations), self.config.max_translations_displayed)

        for translation in active_translations[:max_display]:
            self._draw_translation_box(draw, translation, y_offset, colors)
            y_offset += 100  # Increased space for enhanced boxes

        # Add session info header if there are translations
        if active_translations:
            self._draw_session_header(draw, colors)

        # Convert back to numpy
        self.current_frame = np.array(img)

    def _render_sidebar_frame(self):
        """Render sidebar mode with translations in side panel."""
        self._initialize_frame()
        colors = self.get_theme_colors()

        img = Image.fromarray(self.current_frame, "RGB")
        draw = ImageDraw.Draw(img)

        # Draw sidebar background
        sidebar_width = self.config.width // 3
        draw.rectangle(
            [
                (self.config.width - sidebar_width, 0),
                (self.config.width, self.config.height),
            ],
            fill=colors["overlay_bg"],
            outline=colors["border"],
            width=2,
        )

        # Render translations in sidebar
        y_offset = 20
        sidebar_x = self.config.width - sidebar_width + 10

        for translation in self.current_translations:
            if translation.expires_at and datetime.now(UTC) > translation.expires_at:
                continue

            self._draw_sidebar_translation(
                draw, translation, sidebar_x, y_offset, sidebar_width - 20, colors
            )
            y_offset += 100

        self.current_frame = np.array(img)

    def _render_banner_frame(self):
        """Render bottom banner mode with scrolling translations."""
        self._initialize_frame()
        colors = self.get_theme_colors()

        img = Image.fromarray(self.current_frame, "RGB")
        draw = ImageDraw.Draw(img)

        # Draw banner background
        banner_height = 120
        banner_y = self.config.height - banner_height
        draw.rectangle(
            [(0, banner_y), (self.config.width, self.config.height)],
            fill=colors["overlay_bg"],
            outline=colors["border"],
            width=2,
        )

        # Render latest translation in banner
        if self.current_translations:
            latest = self.current_translations[-1]
            self._draw_banner_translation(draw, latest, banner_y + 10, colors)

        self.current_frame = np.array(img)

    def _render_floating_frame(self):
        """Render floating mode with positioned translation bubbles."""
        # Similar to overlay but with more dynamic positioning
        self._render_overlay_frame()

    def _render_fullscreen_frame(self):
        """Render fullscreen mode showing only translations."""
        self._initialize_frame()
        colors = self.get_theme_colors()

        img = Image.fromarray(self.current_frame, "RGB")
        draw = ImageDraw.Draw(img)

        # Center the translations
        if self.current_translations:
            total_height = len(self.current_translations) * 100
            start_y = (self.config.height - total_height) // 2

            for i, translation in enumerate(self.current_translations):
                if translation.expires_at and datetime.now(UTC) > translation.expires_at:
                    continue

                y_pos = start_y + (i * 100)
                self._draw_centered_translation(draw, translation, y_pos, colors)

        self.current_frame = np.array(img)

    def _draw_translation_box(
        self,
        draw: ImageDraw.Draw,
        translation: TranslationDisplay,
        y_offset: int,
        colors: dict[str, tuple[int, int, int]],
    ):
        """Draw a translation box with text and metadata."""
        # Box dimensions
        box_width = min(self.config.width - 100, 800)
        box_height = 90  # Increased height for more info
        x = 50
        y = y_offset

        # Determine if this is original transcription or translation
        is_original = translation.source_language == translation.target_language

        # Choose border color based on type
        border_color = colors["accent"] if is_original else colors["border"]
        border_width = 2 if is_original else 1

        # Background (RGB mode — opacity not applicable, use solid overlay_bg)
        overlay_color = colors["overlay_bg"]
        draw.rectangle(
            [(x, y), (x + box_width, y + box_height)],
            fill=overlay_color,
            outline=border_color,
            width=border_width,
        )

        # Type indicator (left side)
        type_text = "🎤 TRANSCRIPTION" if is_original else "🌐 TRANSLATION"
        draw.text(
            (x + 10, y + 5),
            type_text,
            fill=colors["accent"] if is_original else colors["text_secondary"],
            font=self.fonts.get("small", ImageFont.load_default()),
        )

        # Speaker info with enhanced formatting
        if self.config.show_speaker_names and translation.speaker_name:
            speaker_text = f"👤 {translation.speaker_name}"
            # Use speaker color if available
            speaker_color = colors["text_primary"]

            draw.text(
                (x + 10, y + 25),
                speaker_text,
                fill=speaker_color,
                font=self.fonts.get("bold", ImageFont.load_default()),
            )

        # Main text with multi-script wrapping
        main_text = translation.text
        lines = wrap_text(main_text, max_chars=60, max_lines=2)
        for i, line in enumerate(lines):
            draw.text(
                (x + 10, y + 45 + (i * 18)),
                line,
                fill=colors["text_primary"],
                font=self.fonts.get("regular", ImageFont.load_default()),
            )

        # Confidence indicator (top right)
        if self.config.show_confidence:
            confidence_text = f"📊 {translation.confidence:.1%}"
            confidence_color = (
                colors["accent"] if translation.confidence > 0.8 else colors["text_secondary"]
            )
            draw.text(
                (x + box_width - 80, y + 5),
                confidence_text,
                fill=confidence_color,
                font=self.fonts.get("small", ImageFont.load_default()),
            )

        # Language indicator (bottom right)
        if not is_original:
            lang_text = f"🔄 {translation.source_language} → {translation.target_language}"
            draw.text(
                (x + box_width - 120, y + box_height - 20),
                lang_text,
                fill=colors["text_secondary"],
                font=self.fonts.get("small", ImageFont.load_default()),
            )

        # Timestamp (bottom right corner)
        if self.config.show_timestamps:
            timestamp_text = translation.timestamp.strftime("%H:%M:%S")
            draw.text(
                (x + box_width - 60, y + box_height - 20),
                timestamp_text,
                fill=colors["text_secondary"],
                font=self.fonts.get("small", ImageFont.load_default()),
            )

    def _draw_sidebar_translation(
        self,
        draw: ImageDraw.Draw,
        translation: TranslationDisplay,
        x: int,
        y: int,
        width: int,
        colors: dict[str, tuple[int, int, int]],
    ):
        """Draw translation in sidebar format."""
        # Similar to box but adapted for sidebar layout
        if translation.speaker_name:
            draw.text(
                (x, y),
                f"{translation.speaker_name}:",
                fill=colors["accent"],
                font=self.fonts.get("small", ImageFont.load_default()),
            )
            y += 20

        # Wrap text for sidebar
        lines = wrap_text(translation.text, max_chars=30, max_lines=3)

        for line in lines:
            draw.text(
                (x, y),
                line,
                fill=colors["text_primary"],
                font=self.fonts.get("regular", ImageFont.load_default()),
            )
            y += 20

    def _draw_banner_translation(
        self,
        draw: ImageDraw.Draw,
        translation: TranslationDisplay,
        y: int,
        colors: dict[str, tuple[int, int, int]],
    ):
        """Draw translation in banner format with text wrapping."""
        text = translation.text
        font = self.fonts.get("regular", ImageFont.load_default())

        # Draw speaker name on first line if present
        line_y = y + 10
        if translation.speaker_name:
            speaker_text = f"{translation.speaker_name}:"
            bbox = draw.textbbox((0, 0), speaker_text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (self.config.width - text_width) // 2
            draw.text((x, line_y), speaker_text, fill=colors["accent"], font=font)
            line_y += 30

        # Wrap main text to fit banner width (max ~80 chars for 1280px at default font)
        lines = wrap_text(text, max_chars=80, max_lines=2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (self.config.width - text_width) // 2
            draw.text((x, line_y), line, fill=colors["text_primary"], font=font)
            line_y += 28

    def _draw_centered_translation(
        self,
        draw: ImageDraw.Draw,
        translation: TranslationDisplay,
        y: int,
        colors: dict[str, tuple[int, int, int]],
    ):
        """Draw translation centered for fullscreen mode."""
        font = self.fonts.get("bold", ImageFont.load_default())

        # Speaker name
        if translation.speaker_name:
            speaker_text = translation.speaker_name
            bbox = draw.textbbox(
                (0, 0),
                speaker_text,
                font=self.fonts.get("small", ImageFont.load_default()),
            )
            text_width = bbox[2] - bbox[0]
            x = (self.config.width - text_width) // 2
            draw.text(
                (x, y),
                speaker_text,
                fill=colors["accent"],
                font=self.fonts.get("small", ImageFont.load_default()),
            )
            y += 30

        # Translation text
        bbox = draw.textbbox((0, 0), translation.text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (self.config.width - text_width) // 2
        draw.text((x, y), translation.text, fill=colors["text_primary"], font=font)

    def _draw_session_header(self, draw: ImageDraw.Draw, colors: dict[str, tuple[int, int, int]]):
        """Draw session information header."""
        try:
            session_id = getattr(self, "session_id", "Unknown")
            active_speakers = len(self.speakers)

            header_text = f"📹 LiveTranslate Virtual Webcam | Session: {session_id[:8]}... | 👥 {active_speakers} speakers"

            # Header background
            header_height = 25
            draw.rectangle(
                [(0, 0), (self.config.width, header_height)],
                fill=colors["overlay_bg"],
                outline=colors["border"],
                width=1,
            )

            # Header text
            draw.text(
                (10, 5),
                header_text,
                fill=colors["text_primary"],
                font=self.fonts.get("small", ImageFont.load_default()),
            )

        except Exception as e:
            logger.error(f"Error drawing session header: {e}")

    def _cleanup_expired_translations(self):
        """Remove expired translations from display."""
        current_time = datetime.now(UTC)
        with self.frame_lock:
            # Filter out expired translations
            active_translations = deque()
            for translation in self.current_translations:
                if not translation.expires_at or current_time <= translation.expires_at:
                    active_translations.append(translation)

            self.current_translations = active_translations

    def get_current_frame_base64(self) -> str | None:
        """Get current frame as base64 encoded image."""
        try:
            with self.frame_lock:
                if self.current_frame is None:
                    return None

                # Convert to PIL Image
                if self.current_frame.shape[2] == 4:  # RGBA
                    img = Image.fromarray(self.current_frame, "RGBA")
                else:  # RGB
                    img = Image.fromarray(self.current_frame, "RGB")

                # Convert to JPEG bytes
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)

                # Encode as base64
                return base64.b64encode(buffer.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return None

    def get_webcam_stats(self) -> dict[str, Any]:
        """Get comprehensive webcam statistics."""
        duration = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_generated / duration if duration > 0 else 0

        return {
            "is_streaming": self.is_streaming,
            "session_id": getattr(self, "session_id", None),
            "frames_generated": self.frames_generated,
            "duration_seconds": duration,
            "average_fps": fps,
            "current_translations_count": len(self.current_translations),
            "speakers_count": len(self.speakers),
            "config": asdict(self.config),
            "last_translation_time": self.last_translation_time,
        }


# Factory functions
def create_virtual_webcam(config: WebcamConfig, bot_manager=None) -> VirtualWebcamManager:
    """Create a virtual webcam manager instance."""
    return VirtualWebcamManager(config, bot_manager)


def create_default_webcam_config(
    display_mode: DisplayMode = DisplayMode.SUBTITLE,
    theme: Theme = Theme.DARK,
    resolution: tuple[int, int] = (1280, 720),
) -> WebcamConfig:
    """Create a default webcam configuration."""
    return WebcamConfig(
        width=resolution[0],
        height=resolution[1],
        display_mode=display_mode,
        theme=theme,
    )


# Example usage
async def main():
    """Example usage of virtual webcam."""
    config = create_default_webcam_config(display_mode=DisplayMode.SUBTITLE, theme=Theme.DARK)

    webcam = create_virtual_webcam(config)

    # Start streaming
    success = await webcam.start_stream("test-session-123")
    if success:
        print("Virtual webcam started")

        # Simulate translations
        test_translations = [
            {
                "translation_id": "trans_1",
                "translated_text": "Hello, how are you doing today?",
                "source_language": "zh",
                "target_language": "en",
                "speaker_name": "Alice",
                "speaker_id": "speaker_1",
                "translation_confidence": 0.95,
            },
            {
                "translation_id": "trans_2",
                "translated_text": "I am doing great, thank you for asking!",
                "source_language": "en",
                "target_language": "es",
                "speaker_name": "Bob",
                "speaker_id": "speaker_2",
                "translation_confidence": 0.87,
            },
        ]

        # Add translations with delay
        for i, translation in enumerate(test_translations):
            await asyncio.sleep(2)
            webcam.add_translation(translation)
            print(f"Added translation {i + 1}")

        # Run for 20 seconds
        await asyncio.sleep(20)

        # Stop streaming
        await webcam.stop_stream()

        # Get stats
        stats = webcam.get_webcam_stats()
        print(f"Webcam stats: {json.dumps(stats, indent=2, default=str)}")
    else:
        print("Failed to start virtual webcam")


if __name__ == "__main__":
    asyncio.run(main())
