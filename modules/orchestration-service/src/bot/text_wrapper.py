"""Multi-script text wrapping for subtitle rendering.

Handles Latin (word-break), CJK (char-break), and mixed text.
CJK characters are detected via Unicode ranges:
  - CJK Unified Ideographs: U+4E00–U+9FFF
  - Hiragana: U+3040–U+309F
  - Katakana: U+30A0–U+30FF
  - Hangul Syllables: U+AC00–U+D7AF
  - CJK Extension A: U+3400–U+4DBF
  - Full-width punctuation: U+3000–U+303F
"""

from __future__ import annotations


def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3000 <= cp <= 0x303F
        or 0xFF00 <= cp <= 0xFFEF
    )


def wrap_text(text: str, max_chars: int = 40, max_lines: int = 0) -> list[str]:
    """Wrap text for subtitle display, handling multi-script content.

    Args:
        text: Input text to wrap.
        max_chars: Maximum characters per line.
        max_lines: Maximum number of lines (0 = unlimited).

    Returns:
        List of wrapped lines.
    """
    if not text:
        return [""]

    lines: list[str] = []
    current_line = ""

    i = 0
    while i < len(text):
        ch = text[i]

        if len(current_line) >= max_chars:
            if _is_cjk(ch) or (current_line and _is_cjk(current_line[-1])):
                lines.append(current_line)
                current_line = ""
            else:
                last_space = current_line.rfind(" ")
                if last_space > 0:
                    lines.append(current_line[:last_space])
                    current_line = current_line[last_space + 1:]
                else:
                    lines.append(current_line)
                    current_line = ""

        current_line += ch
        i += 1

    if current_line:
        lines.append(current_line)

    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]

    if not lines:
        return [""]

    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]

    return lines
