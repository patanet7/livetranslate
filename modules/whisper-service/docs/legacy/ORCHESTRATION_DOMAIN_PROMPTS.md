# Implementing Domain Prompts in Orchestration Service

## Overview

The Whisper service now supports domain-specific prompts that improve transcription accuracy by 40-60% for specialized terminology (medical, legal, technical, etc.).

**Current Status:**
- ✅ Whisper service: Domain prompts fully implemented and tested
- ❌ Orchestration service: Does NOT pass domain prompts to Whisper yet
- ⏳ Frontend: Needs UI for domain prompt configuration

This guide shows how to add domain prompt support to the orchestration service.

---

## Required Changes

### 1. Add Domain Prompt Fields to Audio Upload Endpoint

**File**: `modules/orchestration-service/src/routers/audio/audio_core.py`

**Location**: `upload_audio_file` function (line ~225)

**Add these Form parameters:**

```python
async def upload_audio_file(
    audio: UploadFile = File(..., alias="audio"),
    config: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    chunk_id: Optional[str] = Form(None),
    target_languages: Optional[str] = Form(None),
    enable_transcription: Optional[str] = Form("true"),
    enable_translation: Optional[str] = Form("false"),
    enable_diarization: Optional[str] = Form("true"),
    whisper_model: Optional[str] = Form("whisper-tiny"),
    translation_quality: Optional[str] = Form("balanced"),
    audio_processing: Optional[str] = Form("true"),
    noise_reduction: Optional[str] = Form("false"),
    speech_enhancement: Optional[str] = Form("true"),

    # ✅ ADD THESE NEW FIELDS:
    domain: Optional[str] = Form(None, description="Domain name: medical, legal, technical, etc."),
    custom_terms: Optional[str] = Form(None, description="JSON array of custom terminology"),
    initial_prompt: Optional[str] = Form(None, description="Custom initial prompt text"),
    previous_context: Optional[str] = Form(None, description="Rolling context from previous segments"),

    audio_coordinator=Depends(get_audio_coordinator),
    config_sync_manager=Depends(get_config_sync_manager),
    audio_client=Depends(get_audio_service_client),
    event_publisher=Depends(get_event_publisher),
) -> Dict[str, Any]:
```

**In the function body, parse domain fields:**

```python
# Parse domain prompts
domain_prompts = {}
if domain:
    domain_prompts['domain'] = domain
if custom_terms:
    try:
        domain_prompts['custom_terms'] = json.loads(custom_terms)
    except json.JSONDecodeError:
        logger.warning(f"Invalid custom_terms JSON: {custom_terms}")
if initial_prompt:
    domain_prompts['initial_prompt'] = initial_prompt
if previous_context:
    domain_prompts['previous_context'] = previous_context
```

---

### 2. Pass Domain Prompts to Whisper Service

**File**: `modules/orchestration-service/src/clients/audio_service_client.py`

**OR** (depending on your architecture):

**File**: `modules/orchestration-service/src/websocket_whisper_client.py`

**In the method that sends audio to Whisper, add domain fields:**

```python
# Example for WebSocket client
request_data = {
    "session_id": session_id,
    "audio_data": audio_base64,
    "model_name": model_name,
    "language": language,
    "beam_size": beam_size,
    "sample_rate": sample_rate,
    "task": task,
    "target_language": target_language,
    "enable_vad": enable_vad,

    # ✅ ADD DOMAIN PROMPTS:
    **domain_prompts  # Unpack domain prompt dict
}

sio.emit('transcribe_stream', request_data)
```

---

### 3. Update AudioProcessingRequest Model (Optional)

**File**: `modules/orchestration-service/src/models/audio.py`

**Add domain prompt fields to the model:**

```python
class AudioProcessingRequest(BaseModel):
    """Audio processing request"""

    # ... existing fields ...

    # Domain prompting support (Phase 2: In-Domain Prompting)
    domain: Optional[str] = Field(default=None, description="Domain name: medical, legal, technical, etc.")
    custom_terms: Optional[List[str]] = Field(default=None, description="Custom terminology list")
    initial_prompt: Optional[str] = Field(default=None, description="Custom initial prompt text")
    previous_context: Optional[str] = Field(default=None, description="Rolling context from previous segments")
```

---

## Testing

Once implemented, test with:

```bash
cd modules/whisper-service
python test_jfk_via_orchestration.py
```

This will:
1. Upload JFK audio to orchestration service
2. Pass political domain prompts
3. Verify improved transcription accuracy

Expected results:
- ✅ "Americans" recognized correctly
- ✅ "country" recognized correctly
- ✅ "fellow" recognized correctly

---

## Frontend Integration

After orchestration service is updated, add UI in frontend:

**File**: `modules/frontend-service/src/pages/Settings/TranscriptionSettings.tsx`

**Add domain prompt configuration:**

```tsx
<FormControl fullWidth margin="normal">
  <InputLabel>Domain</InputLabel>
  <Select
    value={domain}
    onChange={(e) => setDomain(e.target.value)}
  >
    <MenuItem value="">None</MenuItem>
    <MenuItem value="medical">Medical</MenuItem>
    <MenuItem value="legal">Legal</MenuItem>
    <MenuItem value="technical">Technical</MenuItem>
    <MenuItem value="business">Business</MenuItem>
    <MenuItem value="education">Education</MenuItem>
  </Select>
</FormControl>

<TextField
  fullWidth
  margin="normal"
  label="Custom Terms (comma-separated)"
  value={customTerms}
  onChange={(e) => setCustomTerms(e.target.value)}
  placeholder="hypertension, cardiomyopathy, antibiotic"
  helperText="Add domain-specific terms to improve accuracy"
/>

<TextField
  fullWidth
  margin="normal"
  label="Initial Prompt"
  value={initialPrompt}
  onChange={(e) => setInitialPrompt(e.target.value)}
  placeholder="Medical consultation about cardiovascular health"
  helperText="Provide context for better transcription"
  multiline
  rows={2}
/>
```

---

## Architecture Flow

```
Frontend (Settings UI)
    ↓ domain, custom_terms, initial_prompt
Orchestration Service (/api/audio/upload)
    ↓ WebSocket transcribe_stream
Whisper Service (api_server.py)
    ↓ Domain Prompt Manager
SimulWhisper (PaddedAlignAttWhisper)
    ↓ Context Window (448 tokens)
Improved Transcription ✅
```

---

## Benefits

- **40-60% reduction in domain-specific errors** (SimulStreaming paper)
- **Automatic context preservation** (static prompts never trimmed)
- **Rolling context support** (previous transcriptions carried forward)
- **Language-agnostic** (works for all Whisper-supported languages)

---

## References

- Whisper implementation: `modules/whisper-service/DOMAIN_PROMPTS.md`
- SimulStreaming paper: Section 4.2 "In-Domain Prompting"
- Test file: `modules/whisper-service/test_domain_prompts.py`
- Orchestration test: `modules/whisper-service/test_jfk_via_orchestration.py`
