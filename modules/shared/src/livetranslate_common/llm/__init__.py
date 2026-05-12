"""LLM client primitives shared across services.

`LLMConnection` / `LLMParameterOverrides` value objects live in
`livetranslate_common.models.llm`. This subpackage adds:

- `qwen`: pure cleanup helpers (think-block stripping, reasoning extraction)
- `client`: merged HTTP client (added in Phase 5)
"""

from livetranslate_common.llm.client import CircuitBreakerOpenError, LLMClient
from livetranslate_common.llm.qwen import (
    extract_from_reasoning,
    extract_translation_text,
    strip_think_blocks_streaming,
)

__all__ = [
    "CircuitBreakerOpenError",
    "LLMClient",
    "extract_from_reasoning",
    "extract_translation_text",
    "strip_think_blocks_streaming",
]
