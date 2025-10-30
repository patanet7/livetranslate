# Whisper Service - Context for Claude Code

## Important: Understanding `is_final` Flag

**CRITICAL CLARIFICATION**: The `is_final` flag in Whisper transcription segments does NOT mean "final vs draft" or "complete vs incomplete".

### What `is_final` Actually Means
- `is_final=True`: Segment ends with punctuation (period, comma, etc.) - marks a **pause/punctuation boundary**
- `is_final=False`: Segment is ongoing, no punctuation yet - just a **word fragment**

### Common Mistake to Avoid
❌ **WRONG**: Filter segments by `is_final=True` to get "final transcription"
```python
# DON'T DO THIS - you'll lose most of the transcription!
final_segments = [seg for seg in all_segments if seg.get('is_final')]
```

✅ **CORRECT**: Collect ALL segments that have text content
```python
# Collect all segments with actual text
text_segments = [seg for seg in all_segments if seg.get('text') and seg.get('text').strip()]
```

### Why This Matters
Whisper's SimulStreaming produces many small segments:
```
Output: And so          (is_final=False)
Output: , my            (is_final=False)
Output: fellow          (is_final=False)
Output: Americans...    (is_final=True)   # Punctuation mark!
```

If you filter by `is_final=True`, you'll only get "Americans..." and lose "And so, my fellow"!

### When to Use `is_final`
- **Display purposes**: Show punctuated text with `is_final=True` for better UX
- **Logging**: Indicate punctuation boundaries with labels like "(punctuated)" vs "(ongoing)"
- **Streaming UI**: Update display when `is_final=True` to show complete phrases

### When NOT to Use `is_final`
- **Accuracy validation**: Always use ALL segments for WER/CER calculations
- **Full transcription**: Concatenate ALL segments to get complete text
- **Testing**: Never filter by `is_final` when validating transcription quality
