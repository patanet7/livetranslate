-- Bot Sessions Database Schema
-- Comprehensive schema for managing Google Meet bot sessions with all related data

-- Create bot_sessions schema
CREATE SCHEMA IF NOT EXISTS bot_sessions;

-- Bot Sessions Table
-- Central table for tracking bot session lifecycle
CREATE TABLE IF NOT EXISTS bot_sessions.sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    bot_id VARCHAR(100) NOT NULL,
    meeting_id VARCHAR(255) NOT NULL,
    meeting_title VARCHAR(500),
    meeting_uri TEXT,
    google_meet_space_id VARCHAR(255),
    conference_record_id VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'spawning',
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    participant_count INTEGER DEFAULT 0,
    target_languages JSONB DEFAULT '[]'::JSONB,
    session_metadata JSONB DEFAULT '{}'::JSONB,
    performance_stats JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Audio Files Table
-- Storage metadata for all audio files captured during sessions
CREATE TABLE IF NOT EXISTS bot_sessions.audio_files (
    file_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_name VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    file_format VARCHAR(20) NOT NULL,
    duration_seconds REAL,
    sample_rate INTEGER,
    channels INTEGER,
    chunk_start_time REAL,
    chunk_end_time REAL,
    audio_quality_score REAL,
    processing_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    file_hash VARCHAR(128),
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Transcripts Table
-- Both Google Meet and in-house transcription results
CREATE TABLE IF NOT EXISTS bot_sessions.transcripts (
    transcript_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL, -- 'google_meet', 'whisper_service', 'manual'
    transcript_text TEXT NOT NULL,
    language_code VARCHAR(10) NOT NULL,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL NOT NULL,
    speaker_id VARCHAR(100),
    speaker_name VARCHAR(255),
    confidence_score REAL,
    segment_index INTEGER DEFAULT 0,
    audio_file_id VARCHAR(100) REFERENCES bot_sessions.audio_files (file_id) ON DELETE SET NULL,
    google_transcript_entry_id VARCHAR(255),
    processing_metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Translations Table
-- Translation results for all transcribed content
CREATE TABLE IF NOT EXISTS bot_sessions.translations (
    translation_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    source_transcript_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.transcripts (transcript_id) ON DELETE CASCADE,
    translated_text TEXT NOT NULL,
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    translation_confidence REAL,
    translation_service VARCHAR(100) NOT NULL,
    speaker_id VARCHAR(100),
    speaker_name VARCHAR(255),
    start_timestamp REAL NOT NULL,
    end_timestamp REAL NOT NULL,
    processing_metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Correlations Table
-- Time correlation between Google Meet captions and in-house transcriptions
CREATE TABLE IF NOT EXISTS bot_sessions.correlations (
    correlation_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    google_transcript_id VARCHAR(100) REFERENCES bot_sessions.transcripts (transcript_id) ON DELETE SET NULL,
    inhouse_transcript_id VARCHAR(100) REFERENCES bot_sessions.transcripts (transcript_id) ON DELETE SET NULL,
    correlation_confidence REAL NOT NULL DEFAULT 0.0,
    timing_offset REAL NOT NULL DEFAULT 0.0,
    correlation_type VARCHAR(50) NOT NULL, -- 'exact', 'interpolated', 'inferred'
    correlation_method VARCHAR(100) NOT NULL,
    speaker_id VARCHAR(100),
    start_timestamp REAL NOT NULL,
    end_timestamp REAL NOT NULL,
    correlation_metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Meeting Participants Table
-- Track participants throughout the meeting
CREATE TABLE IF NOT EXISTS bot_sessions.participants (
    participant_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    google_participant_id VARCHAR(255),
    display_name VARCHAR(255),
    email VARCHAR(255),
    join_time TIMESTAMP,
    leave_time TIMESTAMP,
    total_speaking_time REAL DEFAULT 0.0,
    participant_metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Session Events Table
-- Log of all significant events during bot sessions
CREATE TABLE IF NOT EXISTS bot_sessions.events (
    event_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_subtype VARCHAR(100),
    event_data JSONB DEFAULT '{}'::JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    source_component VARCHAR(100), -- 'bot_manager', 'google_meet_api', 'whisper_service', etc.
    severity VARCHAR(20) DEFAULT 'info' -- 'debug', 'info', 'warning', 'error', 'critical'
);

-- Session Statistics Table
-- Aggregated statistics for completed sessions
CREATE TABLE IF NOT EXISTS bot_sessions.session_statistics (
    session_id VARCHAR(100) PRIMARY KEY REFERENCES bot_sessions.sessions (session_id) ON DELETE CASCADE,
    total_duration REAL,
    total_participants INTEGER,
    unique_speakers INTEGER,
    total_audio_files INTEGER,
    total_audio_duration REAL,
    total_audio_size BIGINT,
    total_transcripts INTEGER,
    total_transcript_words INTEGER,
    total_translations INTEGER,
    translation_languages TEXT [],
    correlation_success_rate REAL,
    average_correlation_confidence REAL,
    quality_metrics JSONB DEFAULT '{}'::JSONB,
    computed_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for Performance
-- Session-based queries
CREATE INDEX IF NOT EXISTS idx_sessions_bot_id ON bot_sessions.sessions (bot_id);
CREATE INDEX IF NOT EXISTS idx_sessions_meeting_id ON bot_sessions.sessions (meeting_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON bot_sessions.sessions (status);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON bot_sessions.sessions (start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON bot_sessions.sessions (created_at);

-- Audio file queries
CREATE INDEX IF NOT EXISTS idx_audio_files_session_id ON bot_sessions.audio_files (session_id);
CREATE INDEX IF NOT EXISTS idx_audio_files_processing_status ON bot_sessions.audio_files (processing_status);
CREATE INDEX IF NOT EXISTS idx_audio_files_created_at ON bot_sessions.audio_files (created_at);

-- Transcript queries
CREATE INDEX IF NOT EXISTS idx_transcripts_session_id ON bot_sessions.transcripts (session_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_source_type ON bot_sessions.transcripts (source_type);
CREATE INDEX IF NOT EXISTS idx_transcripts_speaker_id ON bot_sessions.transcripts (speaker_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_timestamps ON bot_sessions.transcripts (start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_transcripts_language ON bot_sessions.transcripts (language_code);

-- Translation queries
CREATE INDEX IF NOT EXISTS idx_translations_session_id ON bot_sessions.translations (session_id);
CREATE INDEX IF NOT EXISTS idx_translations_source_transcript ON bot_sessions.translations (source_transcript_id);
CREATE INDEX IF NOT EXISTS idx_translations_languages ON bot_sessions.translations (source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_translations_speaker ON bot_sessions.translations (speaker_id);
CREATE INDEX IF NOT EXISTS idx_translations_timestamps ON bot_sessions.translations (start_timestamp, end_timestamp);

-- Correlation queries
CREATE INDEX IF NOT EXISTS idx_correlations_session_id ON bot_sessions.correlations (session_id);
CREATE INDEX IF NOT EXISTS idx_correlations_google_transcript ON bot_sessions.correlations (google_transcript_id);
CREATE INDEX IF NOT EXISTS idx_correlations_inhouse_transcript ON bot_sessions.correlations (inhouse_transcript_id);
CREATE INDEX IF NOT EXISTS idx_correlations_type ON bot_sessions.correlations (correlation_type);
CREATE INDEX IF NOT EXISTS idx_correlations_confidence ON bot_sessions.correlations (correlation_confidence);

-- Participant queries
CREATE INDEX IF NOT EXISTS idx_participants_session_id ON bot_sessions.participants (session_id);
CREATE INDEX IF NOT EXISTS idx_participants_google_id ON bot_sessions.participants (google_participant_id);
CREATE INDEX IF NOT EXISTS idx_participants_join_time ON bot_sessions.participants (join_time);

-- Event queries
CREATE INDEX IF NOT EXISTS idx_events_session_id ON bot_sessions.events (session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON bot_sessions.events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON bot_sessions.events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON bot_sessions.events (severity);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_transcripts_session_time ON bot_sessions.transcripts (session_id, start_timestamp);
CREATE INDEX IF NOT EXISTS idx_translations_session_lang ON bot_sessions.translations (session_id, target_language);
CREATE INDEX IF NOT EXISTS idx_correlations_session_confidence ON bot_sessions.correlations (
    session_id, correlation_confidence
);

-- JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_sessions_metadata_gin ON bot_sessions.sessions USING gin (session_metadata);
CREATE INDEX IF NOT EXISTS idx_audio_metadata_gin ON bot_sessions.audio_files USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_transcripts_metadata_gin ON bot_sessions.transcripts USING gin (processing_metadata);
CREATE INDEX IF NOT EXISTS idx_translations_metadata_gin ON bot_sessions.translations USING gin (processing_metadata);
CREATE INDEX IF NOT EXISTS idx_correlations_metadata_gin ON bot_sessions.correlations USING gin (correlation_metadata);

-- Views for Common Queries
-- Complete session overview with statistics
CREATE OR REPLACE VIEW bot_sessions.session_overview AS
SELECT
    s.session_id,
    s.bot_id,
    s.meeting_id,
    s.meeting_title,
    s.status,
    s.start_time,
    s.end_time,
    s.participant_count,
    s.target_languages,
    EXTRACT(EPOCH FROM (COALESCE(s.end_time, NOW()) - s.start_time)) AS duration_seconds,
    COUNT(DISTINCT af.file_id) AS audio_files_count,
    COALESCE(SUM(af.file_size), 0) AS total_audio_size,
    COUNT(DISTINCT t.transcript_id) AS transcripts_count,
    COUNT(DISTINCT tr.translation_id) AS translations_count,
    COUNT(DISTINCT c.correlation_id) AS correlations_count,
    COUNT(DISTINCT p.participant_id) AS participants_count
FROM bot_sessions.sessions AS s
LEFT JOIN bot_sessions.audio_files AS af ON s.session_id = af.session_id
LEFT JOIN bot_sessions.transcripts AS t ON s.session_id = t.session_id
LEFT JOIN bot_sessions.translations AS tr ON s.session_id = tr.session_id
LEFT JOIN bot_sessions.correlations AS c ON s.session_id = c.session_id
LEFT JOIN bot_sessions.participants AS p ON s.session_id = p.session_id
GROUP BY
    s.session_id, s.bot_id, s.meeting_id, s.meeting_title, s.status,
    s.start_time, s.end_time, s.participant_count, s.target_languages;

-- Speaker statistics per session
CREATE OR REPLACE VIEW bot_sessions.speaker_statistics AS
SELECT
    t.session_id,
    t.speaker_id,
    t.speaker_name,
    COUNT(*) AS transcript_segments,
    SUM(t.end_timestamp - t.start_timestamp) AS total_speaking_time,
    AVG(t.confidence_score) AS avg_confidence,
    COUNT(DISTINCT tr.target_language) AS languages_translated_to,
    COUNT(DISTINCT tr.translation_id) AS total_translations
FROM bot_sessions.transcripts AS t
LEFT JOIN bot_sessions.translations AS tr ON t.transcript_id = tr.source_transcript_id
WHERE t.speaker_id IS NOT NULL
GROUP BY t.session_id, t.speaker_id, t.speaker_name;

-- Translation quality metrics
CREATE OR REPLACE VIEW bot_sessions.translation_quality AS
SELECT
    tr.session_id,
    tr.source_language,
    tr.target_language,
    tr.translation_service,
    COUNT(*) AS translation_count,
    AVG(tr.translation_confidence) AS avg_confidence,
    MIN(tr.translation_confidence) AS min_confidence,
    MAX(tr.translation_confidence) AS max_confidence,
    COUNT(DISTINCT tr.speaker_id) AS speakers_translated
FROM bot_sessions.translations AS tr
GROUP BY tr.session_id, tr.source_language, tr.target_language, tr.translation_service;

-- Correlation effectiveness
CREATE OR REPLACE VIEW bot_sessions.correlation_effectiveness AS
SELECT
    c.session_id,
    c.correlation_type,
    c.correlation_method,
    COUNT(*) AS correlation_count,
    AVG(c.correlation_confidence) AS avg_confidence,
    COUNT(CASE WHEN c.correlation_confidence >= 0.8 THEN 1 END) AS high_confidence_count,
    AVG(ABS(c.timing_offset)) AS avg_timing_offset
FROM bot_sessions.correlations AS c
GROUP BY c.session_id, c.correlation_type, c.correlation_method;

-- Functions for Common Operations
-- Function to calculate session duration
CREATE OR REPLACE FUNCTION bot_sessions.get_session_duration(session_id_param VARCHAR)
RETURNS INTERVAL AS $$
DECLARE
    session_record RECORD;
BEGIN
    SELECT start_time, end_time INTO session_record
    FROM bot_sessions.sessions
    WHERE session_id = session_id_param;

    IF session_record.end_time IS NOT NULL THEN
        RETURN session_record.end_time - session_record.start_time;
    ELSE
        RETURN NOW() - session_record.start_time;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get session word count
CREATE OR REPLACE FUNCTION bot_sessions.get_session_word_count(session_id_param VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    total_words INTEGER;
BEGIN
    SELECT SUM(array_length(string_to_array(transcript_text, ' '), 1))
    INTO total_words
    FROM bot_sessions.transcripts
    WHERE session_id = session_id_param;

    RETURN COALESCE(total_words, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to update session statistics
CREATE OR REPLACE FUNCTION bot_sessions.update_session_statistics(session_id_param VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    stats_record RECORD;
BEGIN
    -- Calculate comprehensive statistics
    SELECT
        EXTRACT(EPOCH FROM bot_sessions.get_session_duration(session_id_param)) as total_duration,
        (SELECT participant_count FROM bot_sessions.sessions WHERE session_id = session_id_param) as total_participants,
        COUNT(DISTINCT t.speaker_id) as unique_speakers,
        COUNT(DISTINCT af.file_id) as total_audio_files,
        COALESCE(SUM(af.duration_seconds), 0) as total_audio_duration,
        COALESCE(SUM(af.file_size), 0) as total_audio_size,
        COUNT(DISTINCT t.transcript_id) as total_transcripts,
        bot_sessions.get_session_word_count(session_id_param) as total_transcript_words,
        COUNT(DISTINCT tr.translation_id) as total_translations,
        array_agg(DISTINCT tr.target_language) FILTER (WHERE tr.target_language IS NOT NULL) as translation_languages,
        COALESCE(AVG(c.correlation_confidence), 0) as average_correlation_confidence,
        CASE
            WHEN COUNT(c.correlation_id) > 0 THEN
                COUNT(CASE WHEN c.correlation_confidence >= 0.7 THEN 1 END)::REAL / COUNT(c.correlation_id)::REAL
            ELSE 0
        END as correlation_success_rate
    INTO stats_record
    FROM bot_sessions.sessions s
    LEFT JOIN bot_sessions.audio_files af ON s.session_id = af.session_id
    LEFT JOIN bot_sessions.transcripts t ON s.session_id = t.session_id
    LEFT JOIN bot_sessions.translations tr ON s.session_id = tr.session_id
    LEFT JOIN bot_sessions.correlations c ON s.session_id = c.session_id
    WHERE s.session_id = session_id_param
    GROUP BY s.session_id;

    -- Insert or update statistics
    INSERT INTO bot_sessions.session_statistics (
        session_id, total_duration, total_participants, unique_speakers,
        total_audio_files, total_audio_duration, total_audio_size,
        total_transcripts, total_transcript_words, total_translations,
        translation_languages, correlation_success_rate, average_correlation_confidence
    ) VALUES (
        session_id_param, stats_record.total_duration, stats_record.total_participants, stats_record.unique_speakers,
        stats_record.total_audio_files, stats_record.total_audio_duration, stats_record.total_audio_size,
        stats_record.total_transcripts, stats_record.total_transcript_words, stats_record.total_translations,
        stats_record.translation_languages, stats_record.correlation_success_rate, stats_record.average_correlation_confidence
    )
    ON CONFLICT (session_id) DO UPDATE SET
        total_duration = EXCLUDED.total_duration,
        total_participants = EXCLUDED.total_participants,
        unique_speakers = EXCLUDED.unique_speakers,
        total_audio_files = EXCLUDED.total_audio_files,
        total_audio_duration = EXCLUDED.total_audio_duration,
        total_audio_size = EXCLUDED.total_audio_size,
        total_transcripts = EXCLUDED.total_transcripts,
        total_transcript_words = EXCLUDED.total_transcript_words,
        total_translations = EXCLUDED.total_translations,
        translation_languages = EXCLUDED.translation_languages,
        correlation_success_rate = EXCLUDED.correlation_success_rate,
        average_correlation_confidence = EXCLUDED.average_correlation_confidence,
        computed_at = NOW();

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic statistics updates
CREATE OR REPLACE FUNCTION bot_sessions.trigger_update_session_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update statistics for the affected session
    IF TG_OP = 'DELETE' THEN
        PERFORM bot_sessions.update_session_statistics(OLD.session_id);
        RETURN OLD;
    ELSE
        PERFORM bot_sessions.update_session_statistics(NEW.session_id);
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to relevant tables
DROP TRIGGER IF EXISTS trigger_audio_files_stats ON bot_sessions.audio_files;
CREATE TRIGGER trigger_audio_files_stats
AFTER INSERT OR UPDATE OR DELETE ON bot_sessions.audio_files
FOR EACH ROW EXECUTE FUNCTION bot_sessions.trigger_update_session_stats();

DROP TRIGGER IF EXISTS trigger_transcripts_stats ON bot_sessions.transcripts;
CREATE TRIGGER trigger_transcripts_stats
AFTER INSERT OR UPDATE OR DELETE ON bot_sessions.transcripts
FOR EACH ROW EXECUTE FUNCTION bot_sessions.trigger_update_session_stats();

DROP TRIGGER IF EXISTS trigger_translations_stats ON bot_sessions.translations;
CREATE TRIGGER trigger_translations_stats
AFTER INSERT OR UPDATE OR DELETE ON bot_sessions.translations
FOR EACH ROW EXECUTE FUNCTION bot_sessions.trigger_update_session_stats();

DROP TRIGGER IF EXISTS trigger_correlations_stats ON bot_sessions.correlations;
CREATE TRIGGER trigger_correlations_stats
AFTER INSERT OR UPDATE OR DELETE ON bot_sessions.correlations
FOR EACH ROW EXECUTE FUNCTION bot_sessions.trigger_update_session_stats();

-- Cleanup procedures
-- Procedure to archive old sessions
CREATE OR REPLACE FUNCTION bot_sessions.archive_old_sessions(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Move old completed sessions to archive (implementation depends on archival strategy)
    -- For now, just count what would be archived
    SELECT COUNT(*)
    INTO archived_count
    FROM bot_sessions.sessions
    WHERE status IN ('ended', 'error')
    AND created_at < NOW() - INTERVAL '1 day' * days_old;

    -- TODO: Implement actual archival logic

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA bot_sessions TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA bot_sessions TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bot_sessions TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bot_sessions TO postgres;

-- Sample data for testing (optional)
-- INSERT INTO bot_sessions.sessions (
--     session_id, bot_id, meeting_id, meeting_title, status, start_time, target_languages
-- )
-- VALUES (
--     'test_session_001', 'bot_001', 'meeting_001', 'Test Meeting',
--     'active', NOW(), '["en", "es", "fr"]'::JSONB
-- );

COMMENT ON SCHEMA bot_sessions IS 'Schema for Google Meet bot sessions, audio, transcripts, and translations';
COMMENT ON TABLE bot_sessions.sessions IS 'Central table for bot session lifecycle management';
COMMENT ON TABLE bot_sessions.audio_files IS 'Metadata for audio files captured during sessions';
COMMENT ON TABLE bot_sessions.transcripts IS 'Transcription results from all sources (Google Meet and in-house)';
COMMENT ON TABLE bot_sessions.translations IS 'Translation results for transcribed content';
COMMENT ON TABLE bot_sessions.correlations IS 'Time correlation between external and internal transcription sources';
COMMENT ON TABLE bot_sessions.participants IS 'Meeting participant tracking';
COMMENT ON TABLE bot_sessions.events IS 'Session event logging for debugging and analytics';
COMMENT ON TABLE bot_sessions.session_statistics IS 'Aggregated statistics for completed sessions';

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Bot sessions database schema created successfully!';
    RAISE NOTICE 'Schema includes: sessions, audio_files, transcripts, translations, correlations, participants, events, session_statistics';
    RAISE NOTICE 'Indexes, views, functions, and triggers have been created for optimal performance';
END
$$;
