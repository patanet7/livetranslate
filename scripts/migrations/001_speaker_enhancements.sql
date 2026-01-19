-- Migration 001: Speaker Enhancements and Full-Text Search
-- For EXISTING databases that already have the base bot-sessions-schema.sql
--
-- This migration adds:
-- 1. Speaker identity tracking (SPEAKER_00 → participant mapping)
-- 2. Full-text search capabilities (tsvector)
-- 3. Segment continuity tracking
--
-- Version: 1.0
-- Created: 2025-11-05
-- Safe to run multiple times (idempotent)

BEGIN;

-- ============================================================================
-- ENABLE REQUIRED EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity searches

-- ============================================================================
-- 1. SPEAKER IDENTITY TRACKING
-- ============================================================================

-- Create speaker_identities table if it doesn't exist
CREATE TABLE IF NOT EXISTS bot_sessions.speaker_identities (
    identity_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions (
        session_id
    ) ON DELETE CASCADE,
    speaker_label VARCHAR(100) NOT NULL,  -- 'SPEAKER_00', 'SPEAKER_01', etc.
    participant_id VARCHAR(100) REFERENCES bot_sessions.participants (
        participant_id
    ),
    identified_name VARCHAR(255),
    -- 'manual', 'voice_print', 'correlation', 'google_meet'
    identification_method VARCHAR(50) NOT NULL,
    identification_confidence REAL DEFAULT 0.0,
    identification_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Ensure unique speaker labels per session
    UNIQUE (session_id, speaker_label)
);

-- Add indexes for speaker identity queries
CREATE INDEX IF NOT EXISTS idx_speaker_identities_session
ON bot_sessions.speaker_identities (session_id);

CREATE INDEX IF NOT EXISTS idx_speaker_identities_label
ON bot_sessions.speaker_identities (speaker_label);

CREATE INDEX IF NOT EXISTS idx_speaker_identities_participant
ON bot_sessions.speaker_identities (participant_id);

CREATE INDEX IF NOT EXISTS idx_speaker_identities_session_label
ON bot_sessions.speaker_identities (session_id, speaker_label);

-- ============================================================================
-- 2. FULL-TEXT SEARCH CAPABILITIES
-- ============================================================================

-- Add search_vector column to transcripts if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bot_sessions'
        AND table_name = 'transcripts'
        AND column_name = 'search_vector'
    ) THEN
        ALTER TABLE bot_sessions.transcripts ADD COLUMN search_vector TSVECTOR;
    END IF;
END $$;

-- Add search_vector column to translations if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bot_sessions'
        AND table_name = 'translations'
        AND column_name = 'search_vector'
    ) THEN
        ALTER TABLE bot_sessions.translations ADD COLUMN search_vector TSVECTOR;
    END IF;
END $$;

-- Create function to update transcript search vectors
CREATE OR REPLACE FUNCTION bot_sessions.update_transcript_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.transcript_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for transcript search vectors
DROP TRIGGER IF EXISTS trigger_transcript_search_vector ON bot_sessions.transcripts;
CREATE TRIGGER trigger_transcript_search_vector
BEFORE INSERT OR UPDATE OF transcript_text ON bot_sessions.transcripts
FOR EACH ROW EXECUTE FUNCTION bot_sessions.update_transcript_search_vector();

-- Create function to update translation search vectors
CREATE OR REPLACE FUNCTION bot_sessions.update_translation_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.translated_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for translation search vectors
DROP TRIGGER IF EXISTS trigger_translation_search_vector ON bot_sessions.translations;
CREATE TRIGGER trigger_translation_search_vector
BEFORE INSERT OR UPDATE OF translated_text ON bot_sessions.translations
FOR EACH ROW EXECUTE FUNCTION bot_sessions.update_translation_search_vector();

-- Add indexes for full-text search
CREATE INDEX IF NOT EXISTS idx_transcripts_search_vector
ON bot_sessions.transcripts USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_transcripts_text_trgm
ON bot_sessions.transcripts USING gin (transcript_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_translations_search_vector
ON bot_sessions.translations USING gin (search_vector);

-- Populate search vectors for existing records
UPDATE bot_sessions.transcripts
SET search_vector = TO_TSVECTOR('english', COALESCE(transcript_text, ''))
WHERE search_vector IS NULL;

UPDATE bot_sessions.translations
SET search_vector = TO_TSVECTOR('english', COALESCE(translated_text, ''))
WHERE search_vector IS NULL;

-- ============================================================================
-- 3. SEGMENT CONTINUITY TRACKING
-- ============================================================================

-- Add segment continuity columns to transcripts if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bot_sessions'
        AND table_name = 'transcripts'
        AND column_name = 'previous_segment_id'
    ) THEN
        ALTER TABLE bot_sessions.transcripts ADD COLUMN previous_segment_id VARCHAR(100);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bot_sessions'
        AND table_name = 'transcripts'
        AND column_name = 'next_segment_id'
    ) THEN
        ALTER TABLE bot_sessions.transcripts ADD COLUMN next_segment_id VARCHAR(100);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'bot_sessions'
        AND table_name = 'transcripts'
        AND column_name = 'is_segment_boundary'
    ) THEN
        ALTER TABLE bot_sessions.transcripts ADD COLUMN is_segment_boundary BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Add indexes for segment continuity
CREATE INDEX IF NOT EXISTS idx_transcripts_previous_segment
ON bot_sessions.transcripts (previous_segment_id);

CREATE INDEX IF NOT EXISTS idx_transcripts_next_segment
ON bot_sessions.transcripts (next_segment_id);

CREATE INDEX IF NOT EXISTS idx_transcripts_segment_index
ON bot_sessions.transcripts (session_id, segment_index);

-- ============================================================================
-- 4. ENHANCED AUDIO FILE INDEXES
-- ============================================================================

-- Add index for audio chunk time queries
CREATE INDEX IF NOT EXISTS idx_audio_files_chunk_times
ON bot_sessions.audio_files (chunk_start_time, chunk_end_time);

-- ============================================================================
-- 5. UPDATE SPEAKER STATISTICS VIEW
-- ============================================================================

-- Recreate speaker statistics view with identity resolution
CREATE OR REPLACE VIEW bot_sessions.speaker_statistics AS
SELECT
    t.session_id,
    t.speaker_id,
    si.identification_method,
    si.identification_confidence,
    COALESCE(si.identified_name, t.speaker_name) AS speaker_name,
    COUNT(*) AS transcript_segments,
    SUM(t.end_timestamp - t.start_timestamp) AS total_speaking_time,
    AVG(t.confidence_score) AS avg_confidence,
    COUNT(DISTINCT tr.target_language) AS languages_translated_to,
    COUNT(DISTINCT tr.translation_id) AS total_translations
FROM bot_sessions.transcripts AS t
LEFT JOIN bot_sessions.speaker_identities AS si
    ON t.session_id = si.session_id AND t.speaker_id = si.speaker_label
LEFT JOIN bot_sessions.translations AS tr
    ON t.transcript_id = tr.source_transcript_id
WHERE t.speaker_id IS NOT NULL
GROUP BY
    t.session_id, t.speaker_id, si.identified_name, t.speaker_name,
    si.identification_method, si.identification_confidence;

-- ============================================================================
-- 6. ADD COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE bot_sessions.speaker_identities IS
'Maps anonymous speaker IDs (SPEAKER_00) to identified participants with confidence scores';

COMMENT ON COLUMN bot_sessions.transcripts.search_vector IS
'Full-text search vector automatically updated from transcript_text';

COMMENT ON COLUMN bot_sessions.transcripts.previous_segment_id IS
'Reference to the previous transcript segment for continuity tracking';

COMMENT ON COLUMN bot_sessions.transcripts.next_segment_id IS
'Reference to the next transcript segment for continuity tracking';

COMMENT ON COLUMN bot_sessions.transcripts.is_segment_boundary IS
'Indicates if this segment marks a boundary (e.g., speaker change, long pause)';

-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Migration 001 completed successfully!';
    RAISE NOTICE '';
    RAISE NOTICE '🆕 New features added:';
    RAISE NOTICE '  - Speaker identity tracking (speaker_identities table)';
    RAISE NOTICE '  - Full-text search (search_vector columns + triggers)';
    RAISE NOTICE '  - Segment continuity tracking (previous/next segment fields)';
    RAISE NOTICE '';
    RAISE NOTICE '📊 Updated components:';
    RAISE NOTICE '  - Enhanced speaker_statistics view with identity resolution';
    RAISE NOTICE '  - Additional indexes for performance optimization';
    RAISE NOTICE '  - Existing data search vectors populated';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Database ready for enhanced speaker tracking and search!';
END
$$;

COMMIT;
