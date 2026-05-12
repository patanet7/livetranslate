"""LLM message construction for translation requests.

Lifted from the old `translation/llm_client.py:_build_messages` so the
merged `LLMClient` (which only knows raw messages) can be driven by the
high-level `TranslationService`.
"""

from __future__ import annotations

from livetranslate_common.models import TranslationContext


_LANG_NAMES = {
    "zh": "Chinese", "en": "English", "es": "Spanish",
    "ja": "Japanese", "ko": "Korean", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
}

_EXTRA_INSTRUCTIONS = {
    ("en", "zh"): " Use simplified characters.",
}


def build_messages(
    text: str,
    source_language: str,
    target_language: str,
    context: list[TranslationContext] | None = None,
    glossary_terms: dict[str, str] | None = None,
    cross_context: list[TranslationContext] | None = None,
) -> list[dict]:
    """Build [system, user] messages for a translation call.

    System prompt has two variants — with-context includes a "never repeat
    context" guard; without-context is shorter for lower draft latency.

    Glossary terms are truncated to prevent prompt injection via oversized
    entries (50 entries × 100 chars each).
    """
    if glossary_terms:
        glossary_terms = {
            k[:100]: v[:100]
            for k, v in list(glossary_terms.items())[:50]
        }

    src_name = _LANG_NAMES.get(source_language, source_language)
    tgt_name = _LANG_NAMES.get(target_language, target_language)
    extra = _EXTRA_INSTRUCTIONS.get((source_language, target_language), "")

    context = context or []
    has_context = bool(context)

    if has_context:
        system_prompt = (
            f"Translate {src_name} speech to {tgt_name}.{extra} "
            f"Output ONLY the {tgt_name} translation. "
            f"Never repeat context."
        )
    else:
        system_prompt = (
            f"Translate {src_name} speech to {tgt_name}.{extra} "
            f"Output ONLY the {tgt_name} translation."
        )

    user_parts: list[str] = []

    if glossary_terms:
        sanitized = {
            k.replace("\n", " "): v.replace("\n", " ")
            for k, v in glossary_terms.items()
        }
        terms = ", ".join(f"{k}={v}" for k, v in sanitized.items())
        user_parts.append(f"Terms: {terms}")
        user_parts.append("")

    if context:
        user_parts.append("[Prior:]")
        for ctx in context:
            src = ctx.text.replace("\n", " ")
            tgt = ctx.translation.replace("\n", " ")
            user_parts.append(f"[{src_name}] {src}")
            user_parts.append(f"[{tgt_name}] {tgt}")
        user_parts.append("")

    if cross_context:
        user_parts.append("[Recent context (other speaker):]")
        for ctx in cross_context:
            src = ctx.text.replace("\n", " ")
            tgt = ctx.translation.replace("\n", " ")
            user_parts.append(f"[{tgt_name}] {src}")
            user_parts.append(f"[{src_name}] {tgt}")
        user_parts.append("")

    if has_context:
        user_parts.append("[New:]")
        user_parts.append(text)
    else:
        user_parts.append(f"Translate: {text}")

    user_content = "\n".join(user_parts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
