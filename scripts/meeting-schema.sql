-- Meeting Persistence Database Schema
-- Stores meeting data captured from Fireflies.ai real-time transcription,
-- including transcript chunks, aggregated sentences, translations, AI insights,
-- and speaker metadata. Designed for the Fireflies Real-Time Enhancement project.
--
-- NOTE: This file uses meeting_data_insights (not meeting_insights) to avoid
-- collision with the bot-sessions meeting_insights table managed by Alembic
-- migration 004. The meeting_insights table (migration 004) is linked to
-- bot_sessions via session_id and stores LLM-generated structured text insights.
-- The meeting_data_insights table stores Fireflies AI insights as JSONB content
-- linked to meetings via meeting_id.
--
-- This schema is idempotent: safe to re-apply with CREATE TABLE IF NOT EXISTS.
-- However, production schema changes MUST go through Alembic migrations.
-- See: modules/orchestration-service/alembic/versions/005_add_fireflies_meeting_persistence.py

-- Enable uuid-ossp extension for UUID generation (idempotent)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- meetings: Core meeting record
-- One row per meeting session, tracks lifecycle from live through completed.
-- The fireflies_transcript_id links back to the Fireflies.ai transcript.
-- ============================================================================
CREATE TABLE IF NOT EXISTS meetings (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fireflies_transcript_id TEXT,
    title                   TEXT,
    meeting_link            TEXT,
    organizer_email         TEXT,
    participants            JSONB NOT NULL DEFAULT '[]'::JSONB,
    start_time              TIMESTAMPTZ,
    end_time                TIMESTAMPTZ,
    duration                INTEGER,                          -- duration in seconds
    source                  TEXT NOT NULL DEFAULT 'fireflies',
    status                  TEXT NOT NULL DEFAULT 'live',      -- live, completed, error, archived
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- meeting_chunks: Raw transcript chunks (deduplicated)
-- Individual transcript segments as received from the real-time stream.
-- The (meeting_id, chunk_id) UNIQUE constraint prevents duplicate ingestion.
-- ============================================================================
CREATE TABLE IF NOT EXISTS meeting_chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id    UUID NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    chunk_id      TEXT NOT NULL,
    text          TEXT NOT NULL,
    speaker_name  TEXT,
    start_time    REAL,
    end_time      REAL,
    is_command    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (meeting_id, chunk_id)
);

-- ============================================================================
-- meeting_sentences: Aggregated sentences
-- Chunks assembled into coherent sentences using sentence-boundary detection.
-- chunk_ids tracks which raw chunks contributed to each sentence.
-- ============================================================================
CREATE TABLE IF NOT EXISTS meeting_sentences (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id      UUID NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    speaker_name    TEXT,
    start_time      REAL,
    end_time        REAL,
    boundary_type   TEXT,                                     -- e.g. 'period', 'question', 'pause', 'timeout'
    chunk_ids       JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- meeting_translations: Translations of sentences
-- Each sentence can have multiple translations (one per target language).
-- Tracks translation performance and model provenance.
-- ============================================================================
CREATE TABLE IF NOT EXISTS meeting_translations (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sentence_id         UUID NOT NULL REFERENCES meeting_sentences (id) ON DELETE CASCADE,
    translated_text     TEXT NOT NULL,
    target_language     TEXT NOT NULL,
    source_language     TEXT NOT NULL DEFAULT 'en',
    confidence          REAL NOT NULL DEFAULT 1.0,
    translation_time_ms REAL NOT NULL DEFAULT 0.0,
    model_used          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- meeting_data_insights: AI-generated insights with JSONB content
-- Deliberately schema-flexible via JSONB to accommodate the many types of
-- insights Fireflies produces (summaries, action items, sentiment, etc.)
-- without requiring schema migrations for each new insight type.
--
-- Named meeting_data_insights (not meeting_insights) to avoid collision with
-- the existing meeting_insights table from Alembic migration 004 which is
-- linked to bot_sessions and stores LLM-generated structured text.
--
-- insight_type values:
--   'summary', 'action_items', 'keywords', 'topics', 'sentiment',
--   'speaker_analytics', 'ai_filters', 'attendance', 'media',
--   'outline', 'questions', 'decisions', 'custom'
-- ============================================================================
CREATE TABLE IF NOT EXISTS meeting_data_insights (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id    UUID NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    insight_type  TEXT NOT NULL,
    content       JSONB NOT NULL,
    source        TEXT NOT NULL DEFAULT 'fireflies',
    model_used    TEXT,
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- meeting_speakers: Speaker metadata per meeting
-- Aggregated speaker-level analytics. The (meeting_id, speaker_name) UNIQUE
-- constraint ensures one row per speaker per meeting for upsert patterns.
-- ============================================================================
CREATE TABLE IF NOT EXISTS meeting_speakers (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meeting_id          UUID NOT NULL REFERENCES meetings (id) ON DELETE CASCADE,
    speaker_name        TEXT NOT NULL,
    email               TEXT,
    talk_time_seconds   REAL NOT NULL DEFAULT 0,
    word_count          INTEGER NOT NULL DEFAULT 0,
    sentiment_score     REAL,
    analytics           JSONB,
    UNIQUE (meeting_id, speaker_name)
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- meetings indexes
CREATE INDEX IF NOT EXISTS idx_meetings_ff_id
    ON meetings (fireflies_transcript_id);
CREATE INDEX IF NOT EXISTS idx_meetings_status
    ON meetings (status);
CREATE INDEX IF NOT EXISTS idx_meetings_source
    ON meetings (source);

-- meeting_chunks indexes
CREATE INDEX IF NOT EXISTS idx_chunks_meeting
    ON meeting_chunks (meeting_id);

-- meeting_sentences indexes
CREATE INDEX IF NOT EXISTS idx_sentences_meeting
    ON meeting_sentences (meeting_id);

-- meeting_translations indexes
CREATE INDEX IF NOT EXISTS idx_mtrans_sentence
    ON meeting_translations (sentence_id);

-- meeting_data_insights indexes
CREATE INDEX IF NOT EXISTS idx_insights_meeting
    ON meeting_data_insights (meeting_id);
CREATE INDEX IF NOT EXISTS idx_insights_type
    ON meeting_data_insights (insight_type);

-- meeting_speakers indexes
CREATE INDEX IF NOT EXISTS idx_speakers_meeting
    ON meeting_speakers (meeting_id);

-- Full-text search indexes for transcript content
CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON meeting_chunks USING gin(to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS idx_sentences_text_search
    ON meeting_sentences USING gin(to_tsvector('english', text));

-- ============================================================================
-- Trigger: auto-update updated_at on meetings table
-- ============================================================================
CREATE OR REPLACE FUNCTION update_meetings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_meetings_updated_at ON meetings;
CREATE TRIGGER trigger_meetings_updated_at
    BEFORE UPDATE ON meetings
    FOR EACH ROW
    EXECUTE FUNCTION update_meetings_updated_at();

-- ============================================================================
-- Table and column comments
-- ============================================================================
COMMENT ON TABLE meetings IS 'Core meeting record — one row per Fireflies meeting session';
COMMENT ON TABLE meeting_chunks IS 'Raw transcript chunks deduplicated by (meeting_id, chunk_id)';
COMMENT ON TABLE meeting_sentences IS 'Aggregated sentences assembled from transcript chunks';
COMMENT ON TABLE meeting_translations IS 'Translations of aggregated sentences into target languages';
COMMENT ON TABLE meeting_data_insights IS 'AI-generated insights stored as flexible JSONB content (named meeting_data_insights to avoid collision with migration 004 meeting_insights)';
COMMENT ON TABLE meeting_speakers IS 'Speaker-level metadata and analytics per meeting';

COMMENT ON COLUMN meetings.fireflies_transcript_id IS 'Links to the Fireflies.ai transcript ID';
COMMENT ON COLUMN meetings.duration IS 'Meeting duration in seconds';
COMMENT ON COLUMN meetings.source IS 'Data source identifier (e.g. fireflies, manual)';
COMMENT ON COLUMN meetings.status IS 'Meeting lifecycle status: live, completed, error, archived';
COMMENT ON COLUMN meeting_chunks.chunk_id IS 'Unique chunk identifier from the real-time stream (dedup key)';
COMMENT ON COLUMN meeting_chunks.is_command IS 'Whether this chunk is a voice command rather than speech';
COMMENT ON COLUMN meeting_sentences.boundary_type IS 'Sentence boundary detection type: period, question, pause, timeout';
COMMENT ON COLUMN meeting_sentences.chunk_ids IS 'JSON array of chunk IDs that compose this sentence';
COMMENT ON COLUMN meeting_translations.translation_time_ms IS 'Time taken to produce this translation in milliseconds';
COMMENT ON COLUMN meeting_data_insights.insight_type IS 'Type of insight: summary, action_items, keywords, topics, sentiment, speaker_analytics, ai_filters, attendance, media, outline, questions, decisions, custom';
COMMENT ON COLUMN meeting_data_insights.content IS 'Flexible JSONB payload — structure varies by insight_type';
COMMENT ON COLUMN meeting_speakers.analytics IS 'Flexible JSONB for speaker-level analytics from Fireflies';

-- ============================================================================
-- Success notice
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Meeting persistence schema created successfully!';
    RAISE NOTICE 'Tables: meetings, meeting_chunks, meeting_sentences, meeting_translations, meeting_data_insights, meeting_speakers';
    RAISE NOTICE 'Indexes: 10 B-tree indexes + 2 full-text search (GIN) indexes';
    RAISE NOTICE 'Trigger: auto-update updated_at on meetings';
END
$$;
