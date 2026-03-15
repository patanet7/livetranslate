# Translation Service (Legacy)

**Status**: This standalone service is being superseded. Translation is now handled by the **orchestration service** calling **Ollama** directly with rolling context + glossary.

## Current Architecture

The orchestration service coordinates translation via Ollama's OpenAI-compatible API at `:11434`. Shared contracts (`TranslationRequest`, `TranslationResponse`, `TranslationContext`) are defined in `livetranslate_common.models.translation`.

## Legacy Code

This directory contains older translation backends (vLLM, Triton, NLLB, local LLM) that may be useful as reference but are not actively used in the current pipeline.

Key files:
- `src/api_server.py` — Flask API (legacy)
- `src/local_translation.py` — vLLM integration
- `src/openai_compatible_translator.py` — OpenAI-compatible API (closest to current Ollama approach)
- `src/prompt_manager.py` — Prompt templates for translation

## Tests

```bash
uv run pytest modules/translation-service/tests/ -v
```
