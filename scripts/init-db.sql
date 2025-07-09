-- LiveTranslate Database Initialization Script
-- This script sets up the basic database structure for LiveTranslate

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS transcription;
CREATE SCHEMA IF NOT EXISTS translation;
CREATE SCHEMA IF NOT EXISTS sessions;
CREATE SCHEMA IF NOT EXISTS speakers;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create basic tables for transcription
CREATE TABLE IF NOT EXISTS transcription.audio_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    duration_seconds DECIMAL(10,2),
    sample_rate INTEGER,
    channels INTEGER,
    format VARCHAR(50),
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending',
    session_id UUID,
    chunk_sequence INTEGER DEFAULT 0,
    start_timestamp DECIMAL(10,3),
    end_timestamp DECIMAL(10,3),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS transcription.transcripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audio_file_id UUID REFERENCES transcription.audio_files(id),
    session_id UUID,
    text TEXT NOT NULL,
    language VARCHAR(10),
    confidence DECIMAL(5,4),
    segments JSONB,
    model_used VARCHAR(100),
    processing_time_ms INTEGER,
    speaker_id VARCHAR(50),
    speaker_confidence DECIMAL(5,4),
    start_timestamp DECIMAL(10,3),
    end_timestamp DECIMAL(10,3),
    is_complete_sentence BOOLEAN DEFAULT TRUE,
    segment_type VARCHAR(50) DEFAULT 'transcription',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create basic tables for translation
CREATE TABLE IF NOT EXISTS translation.translations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transcript_id UUID REFERENCES transcription.transcripts(id),
    session_id UUID,
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    confidence DECIMAL(5,4),
    model_used VARCHAR(100),
    processing_time_ms INTEGER,
    speaker_id VARCHAR(50),
    context_used TEXT,
    translation_quality_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create session management tables
CREATE TABLE IF NOT EXISTS sessions.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS sessions.live_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_token VARCHAR(255) REFERENCES sessions.user_sessions(session_token),
    room_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    source_language VARCHAR(10),
    target_languages TEXT[],
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    participant_count INTEGER DEFAULT 0,
    meeting_title VARCHAR(500),
    meeting_type VARCHAR(50) DEFAULT 'conversation',
    settings JSONB
);

-- Create comprehensive speaker management tables
CREATE TABLE IF NOT EXISTS speakers.speaker_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    speaker_id VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    full_name VARCHAR(500),
    email VARCHAR(255),
    organization VARCHAR(255),
    role VARCHAR(100),
    voice_embedding BYTEA,
    voice_characteristics JSONB,
    language_preferences TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE,
    total_speaking_time INTEGER DEFAULT 0,
    total_sessions INTEGER DEFAULT 0,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS speakers.session_participants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    speaker_id VARCHAR(50) REFERENCES speakers.speaker_profiles(speaker_id),
    participant_name VARCHAR(255),
    participant_role VARCHAR(100),
    join_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    leave_time TIMESTAMP WITH TIME ZONE,
    speaking_time_seconds INTEGER DEFAULT 0,
    utterance_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    microphone_info JSONB,
    connection_quality VARCHAR(50),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS speakers.speaker_segments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    speaker_id VARCHAR(50) REFERENCES speakers.speaker_profiles(speaker_id),
    audio_file_id UUID REFERENCES transcription.audio_files(id),
    transcript_id UUID REFERENCES transcription.transcripts(id),
    start_timestamp DECIMAL(10,3) NOT NULL,
    end_timestamp DECIMAL(10,3) NOT NULL,
    duration_seconds DECIMAL(10,3),
    confidence DECIMAL(5,4),
    voice_activity_level DECIMAL(5,4),
    embedding_distance DECIMAL(8,6),
    segment_type VARCHAR(50) DEFAULT 'speech',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS speakers.speaker_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    speaker_from VARCHAR(50) REFERENCES speakers.speaker_profiles(speaker_id),
    speaker_to VARCHAR(50) REFERENCES speakers.speaker_profiles(speaker_id),
    interaction_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS speakers.speaker_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    speaker_id VARCHAR(50) REFERENCES speakers.speaker_profiles(speaker_id),
    total_speaking_time INTEGER NOT NULL,
    total_utterances INTEGER NOT NULL,
    average_utterance_length DECIMAL(8,3),
    interruption_count INTEGER DEFAULT 0,
    interruption_received_count INTEGER DEFAULT 0,
    silence_gaps_count INTEGER DEFAULT 0,
    average_confidence DECIMAL(5,4),
    words_per_minute DECIMAL(8,2),
    language_switches INTEGER DEFAULT 0,
    dominant_languages TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analysis_period_start TIMESTAMP WITH TIME ZONE,
    analysis_period_end TIMESTAMP WITH TIME ZONE
);

-- Create enhanced session tables for meeting management
CREATE TABLE IF NOT EXISTS sessions.meeting_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID UNIQUE NOT NULL,
    meeting_title VARCHAR(500),
    meeting_description TEXT,
    meeting_type VARCHAR(50) DEFAULT 'conversation',
    organizer_email VARCHAR(255),
    platform VARCHAR(50),
    platform_meeting_id VARCHAR(255),
    scheduled_start TIMESTAMP WITH TIME ZONE,
    scheduled_end TIMESTAMP WITH TIME ZONE,
    actual_start TIMESTAMP WITH TIME ZONE,
    actual_end TIMESTAMP WITH TIME ZONE,
    participant_count INTEGER DEFAULT 0,
    recording_enabled BOOLEAN DEFAULT FALSE,
    translation_enabled BOOLEAN DEFAULT FALSE,
    target_languages TEXT[],
    meeting_status VARCHAR(50) DEFAULT 'scheduled',
    privacy_level VARCHAR(50) DEFAULT 'private',
    retention_policy VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions.session_timeline (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    speaker_id VARCHAR(50),
    event_data JSONB,
    description TEXT,
    importance_level INTEGER DEFAULT 1
);

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS monitoring.service_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS monitoring.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB
);

-- Create indexes for better performance

-- Audio and transcription indexes
CREATE INDEX IF NOT EXISTS idx_audio_files_status ON transcription.audio_files(status);
CREATE INDEX IF NOT EXISTS idx_audio_files_uploaded_at ON transcription.audio_files(uploaded_at);
CREATE INDEX IF NOT EXISTS idx_audio_files_session_id ON transcription.audio_files(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_files_chunk_sequence ON transcription.audio_files(session_id, chunk_sequence);
CREATE INDEX IF NOT EXISTS idx_audio_files_timestamps ON transcription.audio_files(start_timestamp, end_timestamp);

CREATE INDEX IF NOT EXISTS idx_transcripts_audio_file_id ON transcription.transcripts(audio_file_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_session_id ON transcription.transcripts(session_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_language ON transcription.transcripts(language);
CREATE INDEX IF NOT EXISTS idx_transcripts_speaker_id ON transcription.transcripts(speaker_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_timestamps ON transcription.transcripts(start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_transcripts_created_at ON transcription.transcripts(created_at);

-- Translation indexes
CREATE INDEX IF NOT EXISTS idx_translations_transcript_id ON translation.translations(transcript_id);
CREATE INDEX IF NOT EXISTS idx_translations_session_id ON translation.translations(session_id);
CREATE INDEX IF NOT EXISTS idx_translations_languages ON translation.translations(source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_translations_speaker_id ON translation.translations(speaker_id);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON sessions.user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON sessions.user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_live_sessions_room_id ON sessions.live_sessions(room_id);
CREATE INDEX IF NOT EXISTS idx_meeting_sessions_session_id ON sessions.meeting_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_meeting_sessions_platform ON sessions.meeting_sessions(platform, platform_meeting_id);
CREATE INDEX IF NOT EXISTS idx_meeting_sessions_status ON sessions.meeting_sessions(meeting_status);
CREATE INDEX IF NOT EXISTS idx_session_timeline_session_id ON sessions.session_timeline(session_id);
CREATE INDEX IF NOT EXISTS idx_session_timeline_timestamp ON sessions.session_timeline(event_timestamp);

-- Speaker indexes
CREATE INDEX IF NOT EXISTS idx_speaker_profiles_speaker_id ON speakers.speaker_profiles(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_profiles_email ON speakers.speaker_profiles(email);
CREATE INDEX IF NOT EXISTS idx_speaker_profiles_last_seen ON speakers.speaker_profiles(last_seen);

CREATE INDEX IF NOT EXISTS idx_session_participants_session_id ON speakers.session_participants(session_id);
CREATE INDEX IF NOT EXISTS idx_session_participants_speaker_id ON speakers.session_participants(speaker_id);
CREATE INDEX IF NOT EXISTS idx_session_participants_active ON speakers.session_participants(session_id, is_active);

CREATE INDEX IF NOT EXISTS idx_speaker_segments_session_id ON speakers.speaker_segments(session_id);
CREATE INDEX IF NOT EXISTS idx_speaker_segments_speaker_id ON speakers.speaker_segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_segments_timestamps ON speakers.speaker_segments(start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_speaker_segments_audio_file ON speakers.speaker_segments(audio_file_id);

CREATE INDEX IF NOT EXISTS idx_speaker_interactions_session_id ON speakers.speaker_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_speaker_interactions_speakers ON speakers.speaker_interactions(speaker_from, speaker_to);
CREATE INDEX IF NOT EXISTS idx_speaker_interactions_timestamp ON speakers.speaker_interactions(timestamp);

CREATE INDEX IF NOT EXISTS idx_speaker_analytics_session_id ON speakers.speaker_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_speaker_analytics_speaker_id ON speakers.speaker_analytics(speaker_id);
CREATE INDEX IF NOT EXISTS idx_speaker_analytics_period ON speakers.speaker_analytics(analysis_period_start, analysis_period_end);

-- Monitoring indexes
CREATE INDEX IF NOT EXISTS idx_service_health_service ON monitoring.service_health(service_name);
CREATE INDEX IF NOT EXISTS idx_service_health_checked_at ON monitoring.service_health(checked_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON monitoring.performance_metrics(service_name, metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON monitoring.performance_metrics(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW transcription.recent_transcripts AS
SELECT 
    t.id,
    t.session_id,
    t.text,
    t.language,
    t.confidence,
    t.speaker_id,
    t.speaker_confidence,
    t.start_timestamp,
    t.end_timestamp,
    t.created_at,
    af.filename,
    af.duration_seconds,
    sp.display_name as speaker_name,
    sp.role as speaker_role
FROM transcription.transcripts t
JOIN transcription.audio_files af ON t.audio_file_id = af.id
LEFT JOIN speakers.speaker_profiles sp ON t.speaker_id = sp.speaker_id
WHERE t.created_at >= NOW() - INTERVAL '24 hours'
ORDER BY t.created_at DESC;

CREATE OR REPLACE VIEW speakers.session_speaker_summary AS
SELECT 
    p.session_id,
    p.speaker_id,
    sp.display_name,
    sp.role,
    p.speaking_time_seconds,
    p.utterance_count,
    p.join_time,
    p.leave_time,
    p.is_active,
    CASE WHEN p.leave_time IS NULL THEN 
        EXTRACT(EPOCH FROM (NOW() - p.join_time))
    ELSE 
        EXTRACT(EPOCH FROM (p.leave_time - p.join_time))
    END as session_duration_seconds
FROM speakers.session_participants p
LEFT JOIN speakers.speaker_profiles sp ON p.speaker_id = sp.speaker_id
ORDER BY p.session_id, p.join_time;

CREATE OR REPLACE VIEW sessions.active_meeting_summary AS
SELECT 
    ms.session_id,
    ms.meeting_title,
    ms.meeting_type,
    ms.platform,
    ms.actual_start,
    ms.participant_count,
    ls.status,
    ls.source_language,
    ls.target_languages,
    COUNT(sp.id) as active_participants,
    SUM(sp.speaking_time_seconds) as total_speaking_time
FROM sessions.meeting_sessions ms
JOIN sessions.live_sessions ls ON ms.session_id::text = ls.room_id
LEFT JOIN speakers.session_participants sp ON ms.session_id = sp.session_id AND sp.is_active = true
WHERE ms.meeting_status = 'active' AND ls.status = 'active'
GROUP BY ms.session_id, ms.meeting_title, ms.meeting_type, ms.platform, 
         ms.actual_start, ms.participant_count, ls.status, ls.source_language, ls.target_languages
ORDER BY ms.actual_start DESC;

CREATE OR REPLACE VIEW speakers.speaker_analytics_summary AS
SELECT 
    sa.speaker_id,
    sp.display_name,
    COUNT(DISTINCT sa.session_id) as total_sessions,
    SUM(sa.total_speaking_time) as total_speaking_time,
    SUM(sa.total_utterances) as total_utterances,
    AVG(sa.average_confidence) as avg_confidence,
    AVG(sa.words_per_minute) as avg_words_per_minute,
    SUM(sa.interruption_count) as total_interruptions,
    MAX(sa.created_at) as last_analysis
FROM speakers.speaker_analytics sa
LEFT JOIN speakers.speaker_profiles sp ON sa.speaker_id = sp.speaker_id
GROUP BY sa.speaker_id, sp.display_name
ORDER BY total_speaking_time DESC;

CREATE OR REPLACE VIEW monitoring.service_status_summary AS
SELECT 
    service_name,
    status,
    COUNT(*) as count,
    AVG(response_time_ms) as avg_response_time,
    MAX(checked_at) as last_check
FROM monitoring.service_health
WHERE checked_at >= NOW() - INTERVAL '1 hour'
GROUP BY service_name, status
ORDER BY service_name, status;

-- Insert initial data
INSERT INTO monitoring.service_health (service_name, status, response_time_ms, error_message) VALUES
('whisper-service', 'healthy', 0, NULL),
('translation-service', 'healthy', 0, NULL),
('speaker-service', 'healthy', 0, NULL),
('frontend-service', 'healthy', 0, NULL)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA transcription TO livetranslate;
GRANT USAGE ON SCHEMA translation TO livetranslate;
GRANT USAGE ON SCHEMA sessions TO livetranslate;
GRANT USAGE ON SCHEMA speakers TO livetranslate;
GRANT USAGE ON SCHEMA monitoring TO livetranslate;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA transcription TO livetranslate;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA translation TO livetranslate;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA sessions TO livetranslate;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA speakers TO livetranslate;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO livetranslate;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA transcription TO livetranslate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA translation TO livetranslate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA sessions TO livetranslate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA speakers TO livetranslate;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO livetranslate;

-- Log completion
INSERT INTO monitoring.service_health (service_name, status, response_time_ms, error_message) VALUES
('database-init', 'completed', 0, 'Database initialization completed successfully');

COMMIT; 