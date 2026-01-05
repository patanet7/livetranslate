# Level 2: Container Architecture

## Container Overview

LiveTranslate consists of **4 primary containers** (services), each optimized for specific hardware:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LiveTranslate System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Frontend   │───▶│ Orchestration│───▶│   Whisper    │      │
│  │   :5173      │    │    :3000     │    │    :5001     │      │
│  │  [Browser]   │    │   [CPU]      │    │   [NPU]      │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                              │                                   │
│                              │                                   │
│                              ▼                                   │
│                     ┌──────────────┐                            │
│                     │ Translation  │                            │
│                     │    :5003     │                            │
│                     │   [GPU]      │                            │
│                     └──────────────┘                            │
│                                                                   │
│                     ┌──────────────┐                            │
│                     │  PostgreSQL  │                            │
│                     │    :5432     │                            │
│                     └──────────────┘                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Services

### 1. Frontend Service (:5173)
**Technology**: React 18 + TypeScript + Vite
**Hardware**: Browser-optimized
**Purpose**: User interface, audio capture, visualization

**Key Responsibilities**:
- Capture audio from browser/microphone
- Upload audio chunks to Orchestration
- Display real-time transcriptions and translations
- Manage bot lifecycle (Google Meet)
- Settings synchronization

**Communication**:
- HTTP REST → Orchestration (:3000)
- WebSocket → Orchestration (real-time updates)

---

### 2. Orchestration Service (:3000)
**Technology**: Python + FastAPI + AsyncIO
**Hardware**: CPU-optimized
**Purpose**: Central coordinator, API gateway

**Key Responsibilities**:
- Route audio to Whisper service
- Coordinate transcription + translation
- Manage Google Meet bots
- Configuration synchronization
- Database persistence
- Health monitoring

**Communication**:
- HTTP REST ← Frontend
- HTTP REST → Whisper (:5001)
- HTTP REST → Translation (:5003)
- PostgreSQL → Database (:5432)
- WebSocket ← → Frontend

---

### 3. Whisper Service (:5001)
**Technology**: Python + OpenVINO + Whisper
**Hardware**: NPU-optimized (Intel), GPU/CPU fallback
**Purpose**: Speech-to-text transcription

**Key Responsibilities**:
- Audio transcription (Whisper models)
- Speaker diarization
- Voice activity detection (VAD)
- Language identification
- Session management

**Communication**:
- HTTP REST ← Orchestration
- Local model files (OpenVINO IR)

**Hardware Acceleration**:
- **Primary**: Intel NPU via OpenVINO
- **Fallback**: GPU (CUDA) → CPU

---

### 4. Translation Service (:5003)
**Technology**: Python + Transformers + LLMs
**Hardware**: GPU-optimized (NVIDIA)
**Purpose**: Multi-language translation

**Key Responsibilities**:
- Text translation (100+ languages)
- Multi-backend support (Ollama, OpenAI, vLLM)
- Quality scoring
- Translation continuity management

**Communication**:
- HTTP REST ← Orchestration
- HTTP REST → External LLM services (Ollama, OpenAI)

**Hardware Acceleration**:
- **Primary**: NVIDIA GPU (CUDA)
- **Fallback**: CPU

---

### 5. PostgreSQL Database (:5432)
**Technology**: PostgreSQL 14+
**Purpose**: Data persistence

**Stored Data**:
- Bot sessions
- Transcriptions
- Translations
- Speaker mappings
- Audio file metadata
- Analytics data

**Communication**:
- PostgreSQL protocol ← Orchestration

---

## Communication Patterns

### Synchronous (HTTP REST)
```
Frontend → Orchestration → Whisper
                 ↓
            Translation
```

**Use Cases**:
- Audio upload/processing
- Service health checks
- Configuration updates
- Database queries

**Latency**: 50-500ms per request

---

### Asynchronous (WebSocket)
```
Frontend ←→ Orchestration
```

**Use Cases**:
- Real-time transcription updates
- Translation streaming
- Status notifications
- Progress updates

**Latency**: < 50ms per message

---

### Data Storage (PostgreSQL)
```
Orchestration → PostgreSQL
```

**Use Cases**:
- Session persistence
- Transcript archive
- Analytics queries
- Bot session management

---

## Technology Stack

| Service | Language | Framework | Key Libraries |
|---------|----------|-----------|---------------|
| Frontend | TypeScript | React 18 + Vite | Material-UI, Redux Toolkit |
| Orchestration | Python 3.11 | FastAPI | AsyncIO, SQLAlchemy, httpx |
| Whisper | Python 3.11 | FastAPI/Flask | OpenVINO, librosa, pyannote |
| Translation | Python 3.11 | FastAPI | Transformers, Ollama SDK, vLLM |
| Database | SQL | PostgreSQL 14+ | - |

---

## Deployment

### Development (Single Machine)
```bash
# Terminal 1
cd modules/orchestration-service && python src/main.py

# Terminal 2
cd modules/whisper-service && python src/main.py

# Terminal 3
cd modules/translation-service && python src/api_server_fastapi.py

# Terminal 4
cd modules/frontend-service && npm run dev
```

### Docker Compose
```bash
docker-compose up
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

See [Deployment Architecture](./deployment-architecture.md) for details.

---

## Scalability

### Horizontal Scaling
- **Orchestration**: 3+ replicas behind load balancer
- **Whisper**: Scale based on CPU/NPU availability
- **Translation**: Scale based on GPU availability
- **Frontend**: CDN + multiple replicas

### Vertical Scaling
- **Whisper**: More NPU/GPU memory
- **Translation**: Larger GPU instances
- **Database**: More storage, faster disks

### Bottlenecks
1. **NPU/GPU availability**: Limited by hardware
2. **Database**: Write-heavy workloads
3. **Network**: WebSocket connections (10k+ limit)

---

## Related Documentation

- **Communication Details**: [Communication Patterns](./communication-patterns.md)
- **Data Flow**: [Data Flow](./data-flow.md)
- **Hardware Strategy**: [Hardware Optimization](./hardware-optimization.md)
- **Deployment**: [Deployment Architecture](./deployment-architecture.md)
