# Whisper Service (Legacy — being renamed to transcription-service)

**Status**: Being renamed to `modules/transcription-service/` with pluggable backend architecture (Plan 1).

## Important: `is_final` Flag

The `is_final` flag in Whisper segments means **"ends with punctuation"**, NOT "final transcription".

- `is_final=True` — Segment ends with period, comma, etc. (pause/punctuation boundary)
- `is_final=False` — Segment is ongoing, no punctuation yet

**Common mistake**: Filtering by `is_final=True` loses most of the transcription. Always collect ALL segments with text content.

```python
# WRONG — loses most text
final_segments = [seg for seg in all_segments if seg.get('is_final')]

# CORRECT — collect all text
text_segments = [seg for seg in all_segments if seg.get('text', '').strip()]
```

## Commands

```bash
uv run pytest modules/whisper-service/tests/ -v
```
