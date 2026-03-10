"""Meeting transcript export converters.

Supports: SRT, VTT, TXT, JSON, PDF formats.
"""

import io
import json
from datetime import datetime


def format_timecode_srt(seconds: float | None) -> str:
    """Convert float seconds to SRT timecode: HH:MM:SS,mmm"""
    if seconds is None:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timecode_vtt(seconds: float | None) -> str:
    """Convert float seconds to VTT timecode: HH:MM:SS.mmm"""
    if seconds is None:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def to_srt(sentences: list[dict]) -> str:
    """Convert sentences to SubRip (SRT) format."""
    lines = []
    for i, s in enumerate(sentences, 1):
        start = format_timecode_srt(s.get("start_time"))
        end = format_timecode_srt(s.get("end_time"))
        speaker = s.get("speaker_name", "")
        text = s.get("text", "")
        display = f"{speaker}: {text}" if speaker else text
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(display)
        lines.append("")
    return "\n".join(lines)


def to_vtt(sentences: list[dict]) -> str:
    """Convert sentences to WebVTT format."""
    lines = ["WEBVTT", ""]
    for s in sentences:
        start = format_timecode_vtt(s.get("start_time"))
        end = format_timecode_vtt(s.get("end_time"))
        speaker = s.get("speaker_name", "")
        text = s.get("text", "")
        display = f"<v {speaker}>{text}" if speaker else text
        lines.append(f"{start} --> {end}")
        lines.append(display)
        lines.append("")
    return "\n".join(lines)


def to_txt(sentences: list[dict], include_timestamps: bool = True) -> str:
    """Convert sentences to plain text with speaker attribution."""
    lines = []
    for s in sentences:
        speaker = s.get("speaker_name", "Unknown")
        text = s.get("text", "")
        if include_timestamps and s.get("start_time") is not None:
            ts = format_timecode_vtt(s["start_time"])
            lines.append(f"[{ts}] {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def to_json(meeting, sentences: list[dict], translations: list[dict]) -> str:
    """Full structured JSON export."""
    title = getattr(meeting, "title", None) or str(getattr(meeting, "id", "unknown"))
    meeting_date = getattr(meeting, "date", None) or getattr(meeting, "created_at", None)

    data = {
        "meeting": {
            "title": title,
            "date": meeting_date.isoformat() if isinstance(meeting_date, datetime) else str(meeting_date) if meeting_date else None,
        },
        "sentences": sentences,
        "translations": translations,
        "exported_at": datetime.utcnow().isoformat(),
    }
    return json.dumps(data, default=str, indent=2, ensure_ascii=False)


def to_pdf(meeting, sentences: list[dict], translations: list[dict]) -> bytes:
    """Generate PDF transcript. Requires reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        raise RuntimeError("reportlab is required for PDF export. Install with: uv add reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = getattr(meeting, "title", None) or "Meeting Transcript"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Speaker style
    speaker_style = ParagraphStyle("Speaker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10)
    text_style = ParagraphStyle("Text", parent=styles["Normal"], fontSize=10, leftIndent=20)

    for s in sentences:
        speaker = s.get("speaker_name", "Unknown")
        text = s.get("text", "")
        ts = ""
        if s.get("start_time") is not None:
            ts = f" [{format_timecode_vtt(s['start_time'])}]"
        story.append(Paragraph(f"{speaker}{ts}", speaker_style))
        story.append(Paragraph(text, text_style))
        story.append(Spacer(1, 6))

    if translations:
        story.append(Spacer(1, 18))
        story.append(Paragraph("Translations", styles["Heading2"]))
        story.append(Spacer(1, 6))
        for t in translations:
            lang = t.get("language", "")
            text = t.get("text", "")
            story.append(Paragraph(f"[{lang}] {text}", text_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()
