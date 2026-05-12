"""Pure cleanup helpers for LLM responses.

These exist because:

1. Qwen3 models like to emit `<think>...</think>` reasoning blocks even in
   /nothink mode. Callers want just the translation.

2. The Ollama OpenAI-compat layer drops Qwen3 responses into a separate
   `reasoning` field instead of `content`, so the merged client salvages the
   final answer from the reasoning text via labelled markers, quoted strings,
   or a "last capital-starting line" fallback.

3. The streaming filter is needed because `<think>` markers can straddle
   chunk boundaries — a naïve `re.sub` on each token would miss them.

All three are pure functions (no I/O, no state) — easy to unit-test and
reusable from any client implementation.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Synchronous post-hoc cleanup
# ---------------------------------------------------------------------------


_PREFIX_PATTERN = re.compile(
    r'^(?:translation|translated text|output|result|the translation is|翻译|译文)\s*[:：]\s*',
    flags=re.IGNORECASE,
)
_THINK_BLOCK = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
_THINK_UNCLOSED = re.compile(r'<think>.*$', flags=re.DOTALL)

_QUOTE_PAIRS = (
    ('"', '"'),
    ('“', '”'),  # curly double
    ("'", "'"),
    ('‘', '’'),  # curly single
)


def extract_translation_text(response: str) -> str:
    """Clean an LLM response down to just the translated text.

    Strips, in order:
    1. Closed `<think>...</think>` blocks (any depth, any newlines)
    2. Unclosed `<think>` blocks (max_tokens cutoff case)
    3. Common prefix labels (EN + ZH): "Translation:", "翻译：", etc.
    4. Surrounding matched quote pairs (ASCII or Unicode curly)
    """
    text = response.strip()
    text = _THINK_BLOCK.sub('', text).strip()
    text = _THINK_UNCLOSED.sub('', text).strip()
    text = _PREFIX_PATTERN.sub('', text, count=1)
    text = text.strip()
    if len(text) >= 2:
        for open_q, close_q in _QUOTE_PAIRS:
            if text[0] == open_q and text[-1] == close_q:
                text = text[1:-1]
                break
    return text.strip()


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------


_BUFFER_LIMIT = 30  # bytes-ish; enough to detect "<think>" prefix


async def strip_think_blocks_streaming(
    raw_stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Filter out `<think>...</think>` blocks from a token stream.

    Buffers the first ~30 characters to detect a leading `<think>`; after that
    boundary (or once the open tag has been ruled out) switches to passthrough.
    Open or close tags split across chunks are handled correctly because the
    buffer is appended to, not flushed, until we either find `</think>` or
    decide there's no think block.

    Lifted from `routers/audio/websocket_audio.py:_strip_think_and_stream`.
    """
    buffer = ""
    in_think = False

    async for delta in raw_stream:
        if in_think:
            buffer += delta
            end_idx = buffer.find("</think>")
            if end_idx != -1:
                remainder = buffer[end_idx + 8:]
                buffer = ""
                in_think = False
                if remainder:
                    yield remainder
            continue

        if len(buffer) < _BUFFER_LIMIT:
            buffer += delta
            if buffer.startswith("<think>"):
                in_think = True
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    remainder = buffer[end_idx + 8:]
                    buffer = ""
                    in_think = False
                    if remainder:
                        yield remainder
                continue
            if len(buffer) >= _BUFFER_LIMIT:
                yield buffer
                buffer = ""
        else:
            yield delta

    # Flush the tail: any partial buffer that wasn't a think block.
    # If we're still mid-think (unclosed), drop everything — that's safer than
    # leaking the model's chain-of-thought to the user.
    if buffer and not in_think:
        yield buffer


# ---------------------------------------------------------------------------
# Reasoning-field salvage (Qwen3 on Ollama bug workaround)
# ---------------------------------------------------------------------------


_META_PHRASES = (
    "let me", "i should", "i need", "thinking", "step",
    "first", "second", "third", "finally", "therefore",
    "so ", "thus", "hence", "because", "since", "as ",
    "wait", "actually", "hmm", "ok", "okay", "note",
    "polished", "alternative", "literal", "meaning",
    "analysis", "breakdown", "context", "constraint",
    "translate the", "translation task", "going with",
    "check", "double-check", "verify", "review",
)


def _is_meta(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _META_PHRASES)


def _clean_marker_result(result: str) -> str:
    """Strip markdown, quote chars, trailing parens/qualifiers from a marker hit."""
    result = re.sub(r'^[\"\'\*\s\-]+|[\"\'\*\s\-]+$', '', result)
    result = re.sub(r'\s*\([^)]+\)\s*$', '', result)
    result = re.sub(r'\s+(?:or similar|or just|Let\'s go).*$', '', result, flags=re.IGNORECASE)
    return result.strip()


def extract_from_reasoning(reasoning: str) -> str:
    """Salvage a translation from Qwen3's "reasoning" field.

    Ollama's OpenAI-compat layer drops Qwen3's actual answer into a `reasoning`
    field when content is empty. The model emits chain-of-thought followed by
    a clearly-marked final answer (or a quoted complete sentence). This
    function probes for the answer using a cascade of patterns, falling back
    to "the last capital-starting sentence that isn't meta-text."

    Returns "" when nothing answer-shaped is found.
    """
    if not reasoning:
        return ""

    # Priority 1: explicit "Final Output Generation:" / "Draft Output:" sections
    for section_marker in (
        r'\*{0,2}Final Output(?:\s+Generation)?[:\*]{0,2}\s*[:\s]*(.+?)(?:\n|$)',
        r'\*{0,2}Draft Output[:\*]{0,2}\s*[:\s]*(.+?)(?:\n|$)',
    ):
        m = re.search(section_marker, reasoning, re.IGNORECASE)
        if m:
            result = _clean_marker_result(m.group(1).strip())
            if result and len(result) > 2 and not _is_meta(result):
                return result

    # Priority 2a: "Most standard:" / "common:" / "natural:" labels
    m = re.search(
        r'(?:Most\s+)?(?:standard|common|natural)[^:]*:\s*["\']?([A-Z][^"\'.\n]+)["\']?',
        reasoning,
        re.IGNORECASE,
    )
    if m:
        result = re.sub(r'^[\"\'\s]+|[\"\'\s]+$', '', m.group(1).strip())
        if result and len(result) > 2 and not _is_meta(result):
            return result

    # Priority 2b: "Final Decision/Response/Output/Answer:" / "Going with"
    for pattern in (
        r"Final Decision:\s*[\"']?([^\"'\n]+)[\"']?",
        r"Final Response:\s*[\"']?([^\"'\n]+)[\"']?",
        r"Final Output:\s*[\"']?([^\"'\n]+)[\"']?",
        r"Final Answer:\s*[\"']?([^\"'\n]+)[\"']?",
        r"(?:Let's go with|Going with)[:\s]*[\"']?([^\"'\n]+)[\"']?",
    ):
        m = re.search(pattern, reasoning, re.IGNORECASE)
        if m:
            result = re.sub(r'[\.\,]$', '', m.group(1).strip()).strip()
            if result and len(result) > 2 and not _is_meta(result):
                return result

    # Priority 3: "Full sentence:" / "Full translation:" with trailing "."/"or"
    m = re.search(
        r'Full\s+(?:sentence|translation)[:\s]*["\']?([A-Z][^"]+?)["\']?\s*(?:\.|or\s)',
        reasoning,
        re.IGNORECASE,
    )
    if m:
        result = re.sub(r'^[\"\'\s]+|[\"\'\s]+$', '', m.group(1).strip())
        if result and len(result) > 3 and not _is_meta(result):
            return result

    # Priority 4a: double-quoted capital-starting sentences
    dquoted = re.findall(r'"([A-Z][^"]{3,})"', reasoning)
    if dquoted:
        valid = [q for q in dquoted if not _is_meta(q)]
        if valid:
            # Prefer "complete sentences" — end with period or contain a space
            for q in valid:
                if q.endswith('.') or ' ' in q:
                    return q.rstrip('.')
            longest = max(valid, key=len)
            if len(longest) > 5:
                return longest.rstrip('.')

    # Priority 4b: "Option N: translation" patterns
    m = re.search(r'Option \d+:\s*[\""]?([A-Z][^\""\n\.]+)[\""]?', reasoning)
    if m:
        result = m.group(1).strip()
        if not _is_meta(result):
            return result

    # Priority 5: last capital-starting non-meta sentence line
    lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
    for line in reversed(lines):
        if _is_meta(line):
            continue
        if line.startswith(("*", "[", "-", "(", "#")):
            continue
        if line.endswith((":", "?")):
            continue
        if len(line) > 5 and line[0].isupper() and " " in line:
            return line

    return ""
