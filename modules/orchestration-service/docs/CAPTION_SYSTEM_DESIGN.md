# Caption System Design

## Overview

Real-time caption display system for live translation overlay in OBS/streaming applications.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Caption Flow                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Fireflies/Whisper → Translation → CaptionBuffer → WebSocket    │
│                                           │                      │
│                                           ▼                      │
│                                    HTML Overlay                  │
│                                    (Browser Source)              │
│                                           │                      │
│                                           ▼                      │
│                                    OBS Rendering                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Caption Display Style: Roll-Up (CEA-608 Style)

Based on industry standards for live captioning:

- **Roll-Up Mode**: Text appears line by line, pushing previous lines up
- **Speaker Aggregation**: Same speaker's text accumulates in one caption block
- **Time-Based Expiration**: Captions fade after duration (default 8s)
- **Max Lines**: Limit visible captions (default 3-5)

### 2. Speaker Aggregation Logic

```python
# When new text arrives from a speaker:
1. Check if speaker has active (non-expired) caption
2. If YES: Append text to existing caption, extend expiration
3. If NO: Create new caption, remove oldest if at max
4. Broadcast update event to all WebSocket clients
```

**Aggregation Window**: Text from same speaker within the expiration window is combined.

### 3. Multi-Worker Solution

**Problem**: Uvicorn workers have separate memory - caption buffers not shared.

**Solutions** (in order of preference):

1. **Single Worker for Development** (current)
   - Set `DEBUG=true` or `workers=1`
   - Simple, works for testing

2. **Redis Pub/Sub for Production**
   - Store caption state in Redis
   - Publish updates via Redis pub/sub
   - Workers subscribe and broadcast to their WebSocket clients

3. **Sticky Sessions**
   - Route same session to same worker
   - Requires load balancer support

### 4. WebSocket Event Protocol

```json
// New caption
{"event": "caption_added", "caption": {...}}

// Caption text updated (aggregation)
{"event": "caption_updated", "caption": {...}}

// Caption expired
{"event": "caption_expired", "caption_id": "..."}

// Session cleared
{"event": "session_cleared"}
```

### 5. Caption Data Structure

```json
{
  "id": "uuid",
  "translated_text": "Translated text content",
  "original_text": "Original text (optional)",
  "speaker_name": "Alice",
  "speaker_color": "#4CAF50",
  "target_language": "es",
  "confidence": 0.95,
  "created_at": "ISO timestamp",
  "expires_at": "ISO timestamp",
  "duration_seconds": 8.0,
  "time_remaining_seconds": 5.2
}
```

### 6. HTML Overlay Rendering

**Caption Box Lifecycle**:
1. **Appear**: Fade in with slide animation (0.3s)
2. **Update**: Update text content smoothly
3. **Expire**: Fade out (configurable, default 2s)
4. **Remove**: Remove from DOM after fade

**Visual Design**:
- Semi-transparent background (rgba black)
- Speaker name with color indicator
- Original text (smaller, italic, optional)
- Translated text (larger, bold)
- Shadow for readability on any background

### 7. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxCaptions` | 3 | Max visible captions |
| `defaultDuration` | 8.0s | Display duration per caption |
| `fadeTime` | 2.0s | Fade out animation time |
| `aggregateSpeaker` | true | Combine same-speaker text |
| `showOriginal` | false | Show original text |
| `showSpeaker` | true | Show speaker names |
| `fontSize` | 32px | Main text size |
| `position` | bottom | top/center/bottom |

## Implementation Checklist

### Backend (CaptionBuffer)

- [x] Basic add/get/remove captions
- [x] Speaker color assignment
- [x] Time-based expiration
- [x] Speaker text aggregation
- [x] Return (caption, was_updated) tuple
- [ ] On-update callback for broadcasts
- [ ] Redis integration for multi-worker

### WebSocket Router

- [x] Connection management
- [x] Session-based routing
- [x] Language filtering
- [x] Broadcast to session
- [ ] Handle caption_updated event distinctly
- [ ] Cleanup stale connections

### HTML Overlay

- [x] WebSocket connection
- [x] Caption rendering
- [x] Fade animations
- [x] Configuration via URL params
- [ ] Handle caption_updated (update text in place)
- [ ] Proper expiration timer based on time_remaining
- [ ] Reconnection with state sync

### Test Script

- [x] Basic POST caption
- [x] Demo mode with multiple speakers
- [ ] Test speaker aggregation
- [ ] Test expiration timing
- [ ] Stress test concurrent speakers

## Testing Commands

```bash
# Start server (single worker)
DEBUG=true poetry run python -m main_fastapi

# Test caption POST
curl -X POST http://localhost:3000/api/captions/test \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "speaker_name": "Alice", "duration_seconds": 10}'

# Run demo
poetry run python docs/scripts/test_caption_overlay.py --demo 30

# View overlay
# http://localhost:3000/static/captions.html?session=test&showStatus=true
```

## References

- [CEA-608 Caption Standard](https://en.wikipedia.org/wiki/EIA-608)
- [W3C Roll-Up Captions](https://www.w3.org/community/texttracks/wiki/RollupCaptions)
- [WebVTT Specification](https://www.w3.org/TR/webvtt1/)
