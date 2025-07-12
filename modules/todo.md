# Centralized Audio Chunking & Database Integration TODO

## Overview
This TODO list tracks the implementation of centralized audio chunking coordination in the orchestration service with comprehensive database integration for bot audio processing.

## Project Context
- **Problem**: Audio chunking is scattered across services (Frontend: 3s chunks, Whisper: 4s buffer + 3s intervals + 0.2s overlap) with no coordination
- **Solution**: Centralize all audio chunking in orchestration service with proper overlap coordination, speaker correlation, and database integration
- **Database**: Comprehensive `bot_sessions` PostgreSQL schema already exists with tables for sessions, audio_files, transcripts, translations, correlations, participants, events, and session_statistics

## Implementation Plan

### Phase 1: Core Audio Coordination Infrastructure

#### 1. AudioCoordinator Class (HIGH PRIORITY - IN PROGRESS)
**File**: `modules/orchestration-service/src/audio/audio_coordinator.py`
**Status**: In Progress
**Description**: Central coordination class that manages all audio chunking and processing
**Key Features**:
- Receive audio streams from bots/frontend
- Create chunks with configurable overlap (default: 3s chunks, 0.5s overlap)
- Coordinate timing between all services
- Manage database integration for all audio processing
- Handle both WebSocket streaming and chunked processing
- Track chunk lineage and processing state

**Dependencies**: DatabaseAdapter, ChunkManager, SpeakerCorrelationManager, TimingCoordinator
**Database Integration**: Insert into `bot_sessions.audio_files`, `bot_sessions.events`
**Configuration**: AudioChunkingConfig with configurable parameters

#### 2. ChunkManager Implementation (HIGH PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/chunk_manager.py`
**Status**: Pending
**Description**: Handles audio chunking logic with database persistence
**Key Features**:
- Intelligent chunking with overlap management
- Audio quality analysis per chunk
- File storage with metadata
- Chunk sequence tracking
- Memory-efficient buffer management

**Dependencies**: DatabaseAdapter, AudioChunkMetadata model
**Database Integration**: Store chunk metadata in `bot_sessions.audio_files`
**Models**: AudioChunkMetadata dataclass with full chunk information

#### 3. SpeakerCorrelationManager (HIGH PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/speaker_correlator.py`
**Status**: Pending
**Description**: Correlate whisper speaker IDs with Google Meet speakers
**Key Features**:
- Temporal alignment analysis
- Text similarity scoring
- Speaker mapping maintenance
- Correlation confidence tracking
- Database persistence of correlations

**Dependencies**: DatabaseAdapter, correlation queries
**Database Integration**: Store correlations in `bot_sessions.correlations`, update `bot_sessions.transcripts` with speaker mapping
**Algorithm**: Use time windows + text similarity + historical patterns

#### 4. TimingCoordinator (HIGH PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/timing_coordinator.py`
**Status**: Pending
**Description**: Coordinate timestamps and alignment across all data sources
**Key Features**:
- Timestamp synchronization between services
- Overlap window management
- Temporal correlation tracking
- Processing time tracking
- Database timestamp consistency

**Dependencies**: DatabaseAdapter
**Database Integration**: Update timestamp fields in all tables, store timing metadata
**Precision**: Sub-second timing accuracy for proper correlation

#### 5. DatabaseAdapter for Audio Pipeline (HIGH PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/database_adapter.py`
**Status**: Pending
**Description**: Specialized database operations for audio processing pipeline
**Key Features**:
- Audio file operations (insert, update, query)
- Transcript storage with speaker correlation
- Translation storage with lineage tracking
- Correlation management
- Batch operations for performance
- Analytics queries for session statistics

**Dependencies**: Existing bot_sessions database schema
**Database Operations**: All CRUD operations for audio pipeline tables
**Performance**: Optimized queries with proper indexing

#### 6. Enhanced Data Models (HIGH PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/models.py`
**Status**: Pending
**Description**: Pydantic models for audio processing pipeline
**Key Models**:
- `AudioChunkMetadata`: Complete chunk information
- `SpeakerCorrelation`: Speaker mapping data
- `ProcessingResult`: Unified processing result format
- `AudioChunkingConfig`: Configuration parameters
- `ChunkLineage`: Track processing lineage
- `QualityMetrics`: Audio quality assessment

**Dependencies**: Pydantic, existing database models
**Validation**: Comprehensive validation for all audio processing data
**Serialization**: JSON serialization for API responses and database storage

### Phase 2: API Integration

#### 7. Audio Coordination API Endpoints (MEDIUM PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/routers/audio_coordination.py`
**Status**: Pending
**Description**: FastAPI endpoints for coordinated audio processing
**Endpoints**:
- `POST /api/audio/stream/start` - Start audio streaming session
- `POST /api/audio/stream/chunk` - Submit audio chunk for processing
- `GET /api/audio/stream/{session_id}/results` - Get processing results
- `POST /api/audio/stream/stop` - Stop audio streaming session
- `GET /api/audio/sessions/{session_id}/analytics` - Session analytics
- `GET /api/audio/config` - Get current audio processing configuration
- `POST /api/audio/config` - Update audio processing configuration

**Dependencies**: AudioCoordinator, DatabaseAdapter, authentication
**Response Models**: Structured responses with processing status and results
**WebSocket Support**: Real-time streaming for bot audio

### Phase 3: Service Integration

#### 8. Whisper Service Updates (MEDIUM PRIORITY - PENDING)
**Files**: `modules/whisper-service/src/api_server.py`, `modules/whisper-service/src/enhanced_api_server.py`
**Status**: Pending
**Description**: Update whisper service to work with orchestration-managed chunks
**Changes**:
- Remove internal chunking logic from `buffer_manager.py`
- Add orchestration-aware processing endpoints
- Support chunk metadata in requests/responses
- Add speaker information to response format
- Stateless processing mode for coordinated chunks

**Dependencies**: Enhanced API models with chunk metadata
**Database**: Remove whisper-specific chunking, focus on processing
**Performance**: Optimized for orchestration-coordinated processing

#### 9. Bot Audio Integration (MEDIUM PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/bot/audio_capture.py`
**Status**: Pending
**Description**: Update bot audio capture to stream to orchestration coordinator
**Changes**:
- Remove bot-specific chunking logic
- Stream audio directly to AudioCoordinator
- Use orchestration session management
- Integrate with speaker correlation system
- Remove duplicate database operations

**Dependencies**: AudioCoordinator, enhanced bot session management
**Database**: Use centralized audio storage through orchestration
**Quality**: Maintain audio quality monitoring through coordinator

#### 10. Frontend Integration (MEDIUM PRIORITY - PENDING)
**File**: `modules/frontend-service/src/pages/MeetingTest/index.tsx`
**Status**: Pending
**Description**: Update Meeting Test Dashboard to use orchestration chunking
**Changes**:
- Remove frontend chunking logic (3-second chunks)
- Stream audio directly to orchestration service
- Receive coordinated transcription/translation results
- Update visualization to show orchestration-managed chunks
- Add real-time processing status display

**Dependencies**: Updated API endpoints for audio coordination
**UI Updates**: Show chunk processing status, speaker correlations, timing coordination
**Performance**: Improved performance through centralized processing

### Phase 4: Database & Configuration

#### 11. Database Schema Enhancements (LOW PRIORITY - PENDING)
**File**: Update existing `scripts/bot-sessions-schema.sql`
**Status**: Pending
**Description**: Minor enhancements to support chunk correlation tracking
**Changes**:
- Add `chunk_sequence`, `overlap_metadata`, `processing_pipeline_version` to `audio_files`
- Add `whisper_speaker_id`, `speaker_correlation_confidence` to `transcripts`
- Add `chunk_lineage`, `processing_pipeline_version` to `translations`
- Create new indexes for chunk-based queries
- Add views for chunk processing analytics

**Dependencies**: Existing comprehensive database schema
**Migration**: Backward-compatible additions only
**Performance**: New indexes for efficient chunk querying

#### 12. Audio Configuration System (LOW PRIORITY - PENDING)
**File**: `modules/orchestration-service/src/audio/config.py`
**Status**: Pending
**Description**: Comprehensive configuration system for audio processing
**Features**:
- Hot-reloadable configuration
- Environment variable support
- Configuration validation with Pydantic
- Per-session configuration overrides
- A/B testing support for different chunking strategies
- Performance monitoring integration

**Configuration Categories**:
- Chunking parameters (duration, overlap, intervals)
- Database integration settings
- Speaker correlation configuration
- Quality thresholds and monitoring
- Service coordination timeouts
- Data retention policies

## Key Technical Details

### Data Flow Architecture
```
Bot Audio / Frontend Audio
     ↓
AudioCoordinator (orchestration-service)
  ├── ChunkManager: Create chunks with overlap
  ├── DatabaseAdapter: Store audio_files immediately
  ├── Send chunks to Whisper Service
     ↓
Whisper Service (simplified processing)
  ├── Process chunk (no internal chunking)
  ├── Return transcript + speakers
     ↓
AudioCoordinator (continued)
  ├── SpeakerCorrelationManager: Map whisper → Google Meet speakers
  ├── DatabaseAdapter: Store transcripts with speaker correlation
  ├── TimingCoordinator: Align timestamps
  ├── Send to Translation Service
     ↓
Translation Service
  ├── Translate with speaker context
     ↓
AudioCoordinator (final)
  ├── DatabaseAdapter: Store translations with lineage
  ├── Update session statistics
  ├── Emit real-time results
```

### Database Integration Points
- **audio_files**: Store every chunk with metadata immediately
- **transcripts**: Store whisper results with speaker correlation
- **translations**: Store translations with full lineage tracking
- **correlations**: Track speaker correlation and timing alignment
- **events**: Log all processing events for debugging
- **session_statistics**: Real-time analytics updates

### Configuration Parameters
```yaml
audio_chunking:
  chunk_duration: 3.0        # 3 second chunks
  overlap_duration: 0.5      # 0.5 second overlap
  processing_interval: 2.5   # Process every 2.5 seconds
  buffer_duration: 10.0      # 10 second rolling buffer
  
speaker_correlation:
  enabled: true
  confidence_threshold: 0.7
  temporal_window: 2.0       # 2 second window for correlation
  
database_integration:
  store_audio_files: true
  store_transcripts: true
  store_translations: true
  store_correlations: true
  track_chunk_lineage: true
```

## Success Criteria
- [ ] All audio chunking centralized in orchestration service
- [ ] Proper overlap coordination between all services
- [ ] Speaker correlation between whisper and Google Meet working
- [ ] Complete database integration with lineage tracking
- [ ] Configurable chunking parameters with hot-reload
- [ ] Bot audio processing using centralized coordination
- [ ] Frontend test interface using orchestration chunking
- [ ] Comprehensive analytics and debugging support
- [ ] Performance meets or exceeds current implementation
- [ ] All data properly tagged and tracked in database

## Testing Strategy
- Unit tests for each component (AudioCoordinator, ChunkManager, etc.)
- Integration tests for database operations
- End-to-end tests for bot audio processing
- Performance tests for chunking coordination
- Load tests for concurrent audio streams
- Database consistency tests for correlation tracking

## Dependencies
- Existing `bot_sessions` PostgreSQL database schema
- FastAPI orchestration service framework
- Pydantic for data validation
- asyncio for concurrent processing
- httpx for service communication
- PostgreSQL with JSONB support for metadata

This comprehensive TODO list provides the roadmap for implementing centralized audio chunking coordination with full database integration while maintaining compatibility with the existing bot session management system.