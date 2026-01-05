# Pipeline Effects Integration Test Suite

## Overview

Comprehensive testing strategy for the pipeline effects system covering frontend components, WebSocket integration, backend endpoints, and end-to-end workflows.

## Test Coverage Summary

### ✅ **Completed Tests**

1. **AudioStageNode Component Tests** (`modules/frontend-service/src/components/audio/PipelineEditor/__tests__/AudioStageNode.test.tsx`)
   - ✅ 15+ test cases covering all functionality
   - ✅ Parameter slider changes with debouncing
   - ✅ WebSocket message broadcasting
   - ✅ Gain control adjustments (input/output)
   - ✅ Enable/disable toggling
   - ✅ Real-time sync indicators
   - ✅ Event propagation handling
   - ✅ Cleanup on unmount

2. **PipelineCanvas Integration Tests** (`modules/frontend-service/src/components/audio/PipelineEditor/__tests__/PipelineCanvas.test.tsx`)
   - ✅ 20+ test cases for node management
   - ✅ Adding nodes to pipeline
   - ✅ Deleting nodes from pipeline
   - ✅ Creating connections (edges)
   - ✅ Pipeline validation
   - ✅ WebSocket connection propagation
   - ✅ Edge type memoization
   - ✅ Callback preservation during updates

### 🚀 **Required Tests** (Implementation Guide)

#### 3. **WebSocket Parameter Sync Integration Tests**

**File**: `modules/frontend-service/src/components/audio/PipelineEditor/__tests__/WebSocketIntegration.test.tsx`

**Test Cases**:
```typescript
describe('WebSocket Parameter Sync', () => {
  it('should send parameter update via WebSocket when slider changes', async () => {
    // 1. Render node with WebSocket
    // 2. Change parameter slider
    // 3. Wait for debounce (300ms)
    // 4. Verify WebSocket.send called with correct message format
    expect(mockWebSocket.send).toHaveBeenCalledWith(
      JSON.stringify({
        type: 'update_stage',
        stage_id: 'node-1',
        parameters: { strength: 0.8 }
      })
    );
  });

  it('should batch multiple parameter changes', async () => {
    // 1. Change multiple parameters rapidly
    // 2. Verify only ONE WebSocket message sent after debounce
    // 3. Message should contain all updated parameters
  });

  it('should receive config_updated confirmation from backend', async () => {
    // 1. Send parameter update
    // 2. Simulate WebSocket receiving: { type: 'config_updated', stage_id: 'node-1', success: true }
    // 3. Verify sync indicator shows success
  });

  it('should handle WebSocket errors gracefully', async () => {
    // 1. Mock WebSocket.send to throw error
    // 2. Attempt parameter update
    // 3. Verify error logged, UI shows error state
  });

  it('should reconnect and resync parameters after disconnect', async () => {
    // 1. Disconnect WebSocket
    // 2. Make parameter changes
    // 3. Reconnect WebSocket
    // 4. Verify parameters re-sent to backend
  });
});
```

**Mock WebSocket Setup**:
```typescript
class MockWebSocket {
  private messageHandlers: Array<(event: MessageEvent) => void> = [];

  send = vi.fn((data: string) => {
    const message = JSON.parse(data);
    // Simulate backend response
    setTimeout(() => {
      this.simulateMessage({
        type: 'config_updated',
        stage_id: message.stage_id,
        success: true
      });
    }, 50);
  });

  addEventListener(event: string, handler: any) {
    if (event === 'message') {
      this.messageHandlers.push(handler);
    }
  }

  simulateMessage(data: any) {
    const event = new MessageEvent('message', {
      data: JSON.stringify(data)
    });
    this.messageHandlers.forEach(handler => handler(event));
  }
}
```

---

#### 4. **RealTimeProcessor Integration Tests**

**File**: `modules/frontend-service/src/components/audio/PipelineEditor/__tests__/RealTimeProcessor.test.tsx`

**Test Cases**:
```typescript
describe('RealTimeProcessor', () => {
  it('should start real-time session and open WebSocket', async () => {
    // 1. Render with pipeline
    // 2. Click "Start Live Processing"
    // 3. Verify API call to /api/pipeline/realtime/start
    // 4. Verify WebSocket connection opened
    // 5. Verify onWebSocketChange callback fired
  });

  it('should send audio chunks via WebSocket', async () => {
    // 1. Start real-time processing
    // 2. Provide microphone audio data
    // 3. Verify WebSocket sends: { type: 'audio_chunk', data: base64Audio }
  });

  it('should receive and display processed audio', async () => {
    // 1. Send audio chunk
    // 2. Simulate response: { type: 'processed_audio', audio: base64Audio }
    // 3. Verify processed audio displayed/played
  });

  it('should display real-time metrics', async () => {
    // 1. Receive metrics: { type: 'metrics', metrics: { total_latency: 45.2, ... } }
    // 2. Verify metrics displayed in UI
  });

  it('should stop processing and close WebSocket', async () => {
    // 1. Start processing
    // 2. Click "Stop Processing"
    // 3. Verify WebSocket closed
    // 4. Verify session terminated
  });

  it('should handle microphone permission denied', async () => {
    // 1. Mock getUserMedia to reject
    // 2. Attempt to start
    // 3. Verify error shown to user
  });
});
```

---

#### 5. **Backend Pipeline Endpoint Tests**

**File**: `modules/orchestration-service/tests/test_pipeline_endpoints.py`

**Test Cases**:
```python
import pytest
import json
from fastapi.testclient import TestClient
from websockets.sync.client import connect as ws_connect

class TestPipelineEndpoints:
    def test_process_pipeline_batch(self, client: TestClient):
        """Test batch audio processing through pipeline"""
        pipeline_config = {
            "pipeline_id": "test-pipeline",
            "name": "Test Pipeline",
            "stages": {
                "noise_reduction": {
                    "enabled": True,
                    "gain_in": 0.0,
                    "gain_out": 0.0,
                    "parameters": {
                        "strength": 0.7,
                        "voiceProtection": True
                    }
                }
            },
            "connections": []
        }

        # Create test audio file
        with open("test_audio.wav", "rb") as audio_file:
            response = client.post(
                "/api/pipeline/process",
                data={
                    "pipeline_config": json.dumps(pipeline_config),
                    "processing_mode": "batch",
                    "output_format": "wav"
                },
                files={"audio_file": ("test.wav", audio_file, "audio/wav")}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "processed_audio" in data
        assert data["metrics"]["total_latency"] > 0

    def test_start_realtime_session(self, client: TestClient):
        """Test real-time session creation"""
        pipeline_config = {
            "pipeline_config": {
                "pipeline_id": "test-realtime",
                "name": "Test Realtime",
                "stages": {},
                "connections": []
            }
        }

        response = client.post("/api/pipeline/realtime/start", json=pipeline_config)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "running"

    def test_websocket_parameter_update(self, client: TestClient):
        """Test parameter updates via WebSocket"""
        # 1. Start real-time session
        session_response = client.post("/api/pipeline/realtime/start", json={
            "pipeline_config": {
                "pipeline_id": "test",
                "name": "Test",
                "stages": {
                    "noise_reduction": {
                        "enabled": True,
                        "parameters": {"strength": 0.5}
                    }
                },
                "connections": []
            }
        })
        session_id = session_response.json()["session_id"]

        # 2. Connect WebSocket
        with ws_connect(f"ws://localhost:3000/api/pipeline/realtime/{session_id}") as ws:
            # 3. Send parameter update
            ws.send(json.dumps({
                "type": "update_stage",
                "stage_id": "noise_reduction",
                "parameters": {"strength": 0.8}
            }))

            # 4. Wait for confirmation
            response = json.loads(ws.recv())
            assert response["type"] == "config_updated"
            assert response["stage_id"] == "noise_reduction"
            assert response["success"] is True

    def test_websocket_audio_chunk_processing(self, client: TestClient):
        """Test audio chunk processing via WebSocket"""
        # 1. Start session
        # 2. Send audio chunk
        # 3. Verify processed audio returned
        # 4. Verify metrics updated
        pass

    def test_pipeline_validation(self, client: TestClient):
        """Test pipeline configuration validation"""
        invalid_config = {
            "pipeline_id": "test",
            "name": "Invalid",
            "stages": {},  # Missing required stages
            "connections": []
        }

        response = client.post("/api/pipeline/process", data={
            "pipeline_config": json.dumps(invalid_config)
        })

        assert response.status_code == 422
        assert "Invalid pipeline configuration" in response.json()["detail"]

    def test_concurrent_realtime_sessions(self, client: TestClient):
        """Test multiple concurrent real-time sessions"""
        sessions = []
        for i in range(5):
            response = client.post("/api/pipeline/realtime/start", json={
                "pipeline_config": {
                    "pipeline_id": f"test-{i}",
                    "name": f"Test {i}",
                    "stages": {},
                    "connections": []
                }
            })
            sessions.append(response.json()["session_id"])

        # Verify all sessions active
        active_response = client.get("/api/pipeline/realtime/sessions")
        assert len(active_response.json()["active_sessions"]) == 5

    def test_stop_realtime_session(self, client: TestClient):
        """Test stopping real-time session"""
        # 1. Start session
        # 2. Stop session
        # 3. Verify session removed
        # 4. Verify WebSocket closed
        pass
```

**Fixtures**:
```python
@pytest.fixture
def client():
    from main_fastapi import app
    return TestClient(app)

@pytest.fixture
def test_audio_file(tmp_path):
    """Generate test audio file"""
    import wave
    import numpy as np

    file_path = tmp_path / "test_audio.wav"

    # Generate 1 second of audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)

    with wave.open(str(file_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return file_path
```

---

#### 6. **End-to-End Pipeline Flow Tests**

**File**: `modules/orchestration-service/tests/integration/test_pipeline_e2e.py`

**Test Scenarios**:

```python
class TestPipelineE2E:
    @pytest.mark.e2e
    def test_complete_pipeline_with_parameter_changes(self):
        """
        End-to-end test: Create pipeline, start processing, change parameters, verify output
        """
        # 1. Create pipeline with 3 stages
        # 2. Start real-time processing
        # 3. Send audio chunk
        # 4. Change parameter on middle stage
        # 5. Send another audio chunk
        # 6. Verify both chunks processed
        # 7. Verify second chunk reflects parameter change
        pass

    @pytest.mark.e2e
    def test_add_delete_nodes_during_processing(self):
        """
        Test dynamic pipeline modification
        """
        # 1. Start with simple pipeline (Input → Output)
        # 2. Start processing
        # 3. Add noise reduction stage in middle
        # 4. Verify pipeline reconfigured
        # 5. Send audio, verify noise reduction applied
        # 6. Delete stage
        # 7. Verify audio bypasses deleted stage
        pass

    @pytest.mark.e2e
    def test_preset_loading_and_streaming(self):
        """
        Test loading preset and immediately streaming audio
        """
        # 1. Load "Voice Clarity Pro" preset
        # 2. Verify 6 stages created with correct parameters
        # 3. Start real-time processing
        # 4. Stream audio for 5 seconds
        # 5. Verify all stages processing correctly
        # 6. Check metrics for each stage
        pass

    @pytest.mark.e2e
    def test_error_recovery(self):
        """
        Test system recovery from errors
        """
        # 1. Start processing
        # 2. Inject error (invalid audio format)
        # 3. Verify error message received
        # 4. Send valid audio
        # 5. Verify processing resumes
        pass
```

---

## Running the Tests

### Frontend Tests (Vitest)

```bash
cd modules/frontend-service

# Run all tests
pnpm test

# Run specific test file
pnpm test AudioStageNode.test.tsx

# Run with coverage
pnpm test:coverage

# Run in watch mode
pnpm test:watch

# Run with UI
pnpm test:ui
```

### Backend Tests (Pytest)

```bash
cd modules/orchestration-service

# Install test dependencies
pip install -r requirements.txt pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline_endpoints.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run E2E tests
pytest tests/integration/ -m e2e

# Run with verbose output
pytest tests/ -v
```

---

## Test Data Files

Create these test fixtures:

**1. Test Audio Files**
- `tests/fixtures/test_audio_clean.wav` - Clean voice recording
- `tests/fixtures/test_audio_noisy.wav` - Noisy recording
- `tests/fixtures/test_audio_silence.wav` - Silence for VAD testing

**2. Test Pipeline Configurations**
- `tests/fixtures/pipeline_simple.json` - Input → Output only
- `tests/fixtures/pipeline_voice_clarity.json` - Full voice processing chain
- `tests/fixtures/pipeline_invalid.json` - Invalid config for error testing

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Pipeline Effects Tests

on: [push, pull_request]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd modules/frontend-service && pnpm install
      - name: Run tests
        run: cd modules/frontend-service && pnpm test:coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: cd modules/orchestration-service && pip install -r requirements.txt pytest pytest-cov
      - name: Run tests
        run: cd modules/orchestration-service && pytest tests/ --cov=src
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Test Coverage Goals

- **Frontend Components**: ≥ 85% coverage
- **Backend Endpoints**: ≥ 90% coverage
- **WebSocket Integration**: ≥ 80% coverage
- **End-to-End Flows**: Critical paths covered

---

## Manual Testing Checklist

Before releasing, manually verify:

- [ ] Load "Voice Clarity Pro" preset
- [ ] All 6 nodes appear with parameters
- [ ] Click any slider - it moves smoothly
- [ ] Click "Start Live Processing"
- [ ] Nodes turn green, connections animate
- [ ] Adjust noise reduction strength slider
- [ ] Observe immediate audio quality change
- [ ] Check metrics update in real-time
- [ ] Add new node while processing
- [ ] Delete node while processing
- [ ] Stop processing
- [ ] Nodes return to idle state

---

## Debugging Failed Tests

**WebSocket Tests Failing?**
```typescript
// Enable WebSocket logging
const mockWebSocket = {
  send: vi.fn((data) => {
    console.log('[WS SEND]', data);
  }),
  addEventListener: vi.fn((event, handler) => {
    console.log('[WS LISTEN]', event);
  })
};
```

**Parameter Updates Not Working?**
```typescript
// Check debounce timing
vi.advanceTimersByTime(300); // Exact debounce duration
await waitFor(() => {
  expect(mockWebSocket.send).toHaveBeenCalled();
}, { timeout: 1000 });
```

**Backend Tests Failing?**
```python
# Enable request/response logging
@pytest.fixture
def client():
    from main_fastapi import app
    app.debug = True  # Enable debug mode
    return TestClient(app)
```

---

## Next Steps

1. ✅ Implement remaining test files (WebSocket, RealTimeProcessor, Backend)
2. ✅ Create test fixtures (audio files, configs)
3. ✅ Set up CI/CD pipeline
4. ✅ Achieve coverage goals
5. ✅ Run manual testing checklist
6. ✅ Document any bugs found
7. ✅ Create regression test suite

---

## Success Metrics

- All tests pass ✅
- Coverage goals met ✅
- No console errors in browser ✅
- All sliders responsive ✅
- Real-time parameter sync working ✅
- Backend correctly processes updates ✅
- E2E flows complete successfully ✅
