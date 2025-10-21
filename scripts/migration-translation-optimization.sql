-- Migration: Translation Optimization Features
-- Adds database support for translation caching, model selection, and performance tracking
-- Run this after bot-sessions-schema.sql

-- ============================================================================
-- TRANSLATION OPTIMIZATION TABLES
-- ============================================================================

-- Translation Cache Statistics Table
-- Tracks cache performance for translation results
CREATE TABLE IF NOT EXISTS bot_sessions.translation_cache_stats (
    cache_stat_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES bot_sessions.sessions(session_id) ON DELETE CASCADE,

    -- Cache operation details
    cache_key_hash VARCHAR(128) NOT NULL,  -- MD5 hash of text + language pair
    source_text_preview VARCHAR(200),      -- First 200 chars for debugging
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,

    -- Cache performance
    was_cache_hit BOOLEAN NOT NULL DEFAULT false,
    cache_latency_ms REAL,                 -- Latency for cache lookup
    translation_latency_ms REAL,           -- Latency for actual translation (if miss)

    -- Model information
    model_used VARCHAR(100),               -- llama, nllb, etc.
    translation_service VARCHAR(100),      -- Which backend service

    -- Quality metrics
    confidence_score REAL,
    quality_score REAL,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Translation Batch Metadata Table
-- Tracks multi-language translation batches for performance analysis
CREATE TABLE IF NOT EXISTS bot_sessions.translation_batches (
    batch_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions(session_id) ON DELETE CASCADE,

    -- Batch details
    source_text TEXT NOT NULL,
    source_language VARCHAR(10) NOT NULL,
    target_languages TEXT[] NOT NULL,      -- Array of target languages
    num_languages INTEGER NOT NULL,

    -- Performance metrics
    total_processing_time_ms REAL NOT NULL,
    avg_translation_time_ms REAL,
    cache_hit_count INTEGER DEFAULT 0,
    cache_miss_count INTEGER DEFAULT 0,
    cache_hit_rate REAL,                   -- Calculated: hits / total

    -- Model information
    model_requested VARCHAR(100),
    models_used JSONB,                     -- {"es": "llama", "fr": "nllb", ...}

    -- Request details
    request_type VARCHAR(50) DEFAULT 'multi_language',  -- multi_language, batch, sequential
    api_endpoint VARCHAR(200),

    -- Results
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    translation_results JSONB,             -- Complete results for analytics

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Model Performance Tracking Table
-- Aggregate performance metrics by model and language pair
CREATE TABLE IF NOT EXISTS bot_sessions.model_performance (
    performance_id BIGSERIAL PRIMARY KEY,

    -- Model identification
    model_name VARCHAR(100) NOT NULL,
    model_backend VARCHAR(100),            -- llama_transformers, nllb_transformers, etc.

    -- Language pair
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,

    -- Performance aggregates (rolling window)
    total_translations INTEGER NOT NULL DEFAULT 0,
    successful_translations INTEGER NOT NULL DEFAULT 0,
    failed_translations INTEGER NOT NULL DEFAULT 0,

    -- Latency metrics (milliseconds)
    avg_latency_ms REAL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    p99_latency_ms REAL,
    min_latency_ms REAL,
    max_latency_ms REAL,

    -- Quality metrics
    avg_confidence REAL,
    avg_quality_score REAL,

    -- Cache effectiveness
    cache_hit_rate REAL,

    -- Time window for aggregation
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,

    -- Last updated
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Unique constraint on model + language pair + time window
    CONSTRAINT unique_model_lang_window UNIQUE (model_name, source_language, target_language, window_start)
);

-- Translation Session Context Table
-- Stores context history for context-aware translations
CREATE TABLE IF NOT EXISTS bot_sessions.translation_context (
    context_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL REFERENCES bot_sessions.sessions(session_id) ON DELETE CASCADE,

    -- Context window
    target_language VARCHAR(10) NOT NULL,
    context_window_size INTEGER DEFAULT 5,

    -- Recent translations (FIFO queue)
    recent_translations JSONB DEFAULT '[]'::jsonb,  -- Array of {source, translation, timestamp}

    -- Terminology tracking
    custom_terminology JSONB DEFAULT '{}'::jsonb,   -- {source_term: target_term}

    -- Style preferences
    style_preferences JSONB DEFAULT '{}'::jsonb,    -- {formality, gender, etc.}

    -- Last activity
    last_activity TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- ALTER EXISTING TABLES
-- ============================================================================

-- Add new columns to translations table for optimization tracking
ALTER TABLE bot_sessions.translations
    ADD COLUMN IF NOT EXISTS model_name VARCHAR(100),
    ADD COLUMN IF NOT EXISTS model_backend VARCHAR(100),
    ADD COLUMN IF NOT EXISTS batch_id VARCHAR(100) REFERENCES bot_sessions.translation_batches(batch_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS was_cached BOOLEAN DEFAULT false,
    ADD COLUMN IF NOT EXISTS cache_latency_ms REAL,
    ADD COLUMN IF NOT EXISTS translation_latency_ms REAL,
    ADD COLUMN IF NOT EXISTS optimization_metadata JSONB DEFAULT '{}'::jsonb;

-- Add comment to new column
COMMENT ON COLUMN bot_sessions.translations.optimization_metadata IS 'Stores cache hits, model selection, batch info, etc.';

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Cache statistics indexes
CREATE INDEX IF NOT EXISTS idx_cache_stats_session_id ON bot_sessions.translation_cache_stats(session_id);
CREATE INDEX IF NOT EXISTS idx_cache_stats_cache_key ON bot_sessions.translation_cache_stats(cache_key_hash);
CREATE INDEX IF NOT EXISTS idx_cache_stats_hit ON bot_sessions.translation_cache_stats(was_cache_hit);
CREATE INDEX IF NOT EXISTS idx_cache_stats_languages ON bot_sessions.translation_cache_stats(source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_cache_stats_created_at ON bot_sessions.translation_cache_stats(created_at);

-- Batch metadata indexes
CREATE INDEX IF NOT EXISTS idx_batches_session_id ON bot_sessions.translation_batches(session_id);
CREATE INDEX IF NOT EXISTS idx_batches_model ON bot_sessions.translation_batches(model_requested);
CREATE INDEX IF NOT EXISTS idx_batches_hit_rate ON bot_sessions.translation_batches(cache_hit_rate);
CREATE INDEX IF NOT EXISTS idx_batches_created_at ON bot_sessions.translation_batches(created_at);

-- Model performance indexes
CREATE INDEX IF NOT EXISTS idx_model_perf_model ON bot_sessions.model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_perf_languages ON bot_sessions.model_performance(source_language, target_language);
CREATE INDEX IF NOT EXISTS idx_model_perf_window ON bot_sessions.model_performance(window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_model_perf_updated ON bot_sessions.model_performance(last_updated);

-- Context indexes
CREATE INDEX IF NOT EXISTS idx_context_session_id ON bot_sessions.translation_context(session_id);
CREATE INDEX IF NOT EXISTS idx_context_target_lang ON bot_sessions.translation_context(target_language);
CREATE INDEX IF NOT EXISTS idx_context_activity ON bot_sessions.translation_context(last_activity);

-- New indexes on translations table
CREATE INDEX IF NOT EXISTS idx_translations_model ON bot_sessions.translations(model_name);
CREATE INDEX IF NOT EXISTS idx_translations_cached ON bot_sessions.translations(was_cached);
CREATE INDEX IF NOT EXISTS idx_translations_batch_id ON bot_sessions.translations(batch_id);

-- JSONB indexes for optimization metadata
CREATE INDEX IF NOT EXISTS idx_translations_opt_metadata_gin ON bot_sessions.translations USING GIN (optimization_metadata);

-- ============================================================================
-- VIEWS FOR ANALYTICS
-- ============================================================================

-- Cache performance overview
CREATE OR REPLACE VIEW bot_sessions.cache_performance AS
SELECT
    session_id,
    source_language,
    target_language,
    model_used,
    COUNT(*) as total_lookups,
    SUM(CASE WHEN was_cache_hit THEN 1 ELSE 0 END) as cache_hits,
    SUM(CASE WHEN NOT was_cache_hit THEN 1 ELSE 0 END) as cache_misses,
    ROUND((SUM(CASE WHEN was_cache_hit THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0) * 100), 2) as hit_rate_percent,
    ROUND(AVG(CASE WHEN was_cache_hit THEN cache_latency_ms ELSE NULL END), 2) as avg_cache_hit_latency_ms,
    ROUND(AVG(CASE WHEN NOT was_cache_hit THEN translation_latency_ms ELSE NULL END), 2) as avg_translation_latency_ms,
    ROUND(AVG(confidence_score), 3) as avg_confidence
FROM bot_sessions.translation_cache_stats
GROUP BY session_id, source_language, target_language, model_used;

-- Multi-language batch efficiency
CREATE OR REPLACE VIEW bot_sessions.batch_efficiency AS
SELECT
    session_id,
    model_requested,
    COUNT(*) as total_batches,
    ROUND(AVG(num_languages), 1) as avg_languages_per_batch,
    ROUND(AVG(total_processing_time_ms), 2) as avg_total_time_ms,
    ROUND(AVG(avg_translation_time_ms), 2) as avg_per_language_ms,
    ROUND(AVG(cache_hit_rate) * 100, 2) as avg_cache_hit_rate_percent,
    SUM(cache_hit_count) as total_cache_hits,
    SUM(cache_miss_count) as total_cache_misses,
    ROUND(SUM(success_count)::numeric / NULLIF(SUM(success_count + error_count), 0) * 100, 2) as success_rate_percent
FROM bot_sessions.translation_batches
GROUP BY session_id, model_requested;

-- Model performance comparison
CREATE OR REPLACE VIEW bot_sessions.model_comparison AS
SELECT
    model_name,
    source_language,
    target_language,
    SUM(total_translations) as total_translations,
    ROUND(AVG(avg_latency_ms), 2) as avg_latency_ms,
    ROUND(AVG(p95_latency_ms), 2) as p95_latency_ms,
    ROUND(AVG(avg_confidence), 3) as avg_confidence,
    ROUND(AVG(cache_hit_rate) * 100, 2) as cache_hit_rate_percent,
    ROUND(SUM(successful_translations)::numeric / NULLIF(SUM(total_translations), 0) * 100, 2) as success_rate_percent
FROM bot_sessions.model_performance
WHERE window_end > NOW() - INTERVAL '7 days'  -- Last 7 days
GROUP BY model_name, source_language, target_language;

-- Session translation summary with optimization metrics
CREATE OR REPLACE VIEW bot_sessions.session_translation_summary AS
SELECT
    t.session_id,
    COUNT(DISTINCT t.translation_id) as total_translations,
    COUNT(DISTINCT t.target_language) as unique_target_languages,
    COUNT(DISTINCT t.model_name) as models_used,
    SUM(CASE WHEN t.was_cached THEN 1 ELSE 0 END) as cached_translations,
    ROUND((SUM(CASE WHEN t.was_cached THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0) * 100), 2) as cache_hit_rate_percent,
    ROUND(AVG(t.translation_confidence), 3) as avg_confidence,
    ROUND(AVG(CASE WHEN t.was_cached THEN t.cache_latency_ms ELSE t.translation_latency_ms END), 2) as avg_latency_ms,
    COUNT(DISTINCT t.batch_id) as batches_used,
    ARRAY_AGG(DISTINCT t.model_name) FILTER (WHERE t.model_name IS NOT NULL) as models_list
FROM bot_sessions.translations t
GROUP BY t.session_id;

-- ============================================================================
-- FUNCTIONS FOR OPERATIONS
-- ============================================================================

-- Function to record cache statistic
CREATE OR REPLACE FUNCTION bot_sessions.record_cache_stat(
    p_session_id VARCHAR,
    p_cache_key_hash VARCHAR,
    p_source_text VARCHAR,
    p_source_language VARCHAR,
    p_target_language VARCHAR,
    p_was_cache_hit BOOLEAN,
    p_cache_latency_ms REAL,
    p_translation_latency_ms REAL DEFAULT NULL,
    p_model_used VARCHAR DEFAULT NULL,
    p_translation_service VARCHAR DEFAULT NULL,
    p_confidence REAL DEFAULT NULL,
    p_quality REAL DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    stat_id BIGINT;
BEGIN
    INSERT INTO bot_sessions.translation_cache_stats (
        session_id, cache_key_hash, source_text_preview, source_language, target_language,
        was_cache_hit, cache_latency_ms, translation_latency_ms,
        model_used, translation_service, confidence_score, quality_score
    ) VALUES (
        p_session_id, p_cache_key_hash, LEFT(p_source_text, 200), p_source_language, p_target_language,
        p_was_cache_hit, p_cache_latency_ms, p_translation_latency_ms,
        p_model_used, p_translation_service, p_confidence, p_quality
    )
    RETURNING cache_stat_id INTO stat_id;

    RETURN stat_id;
END;
$$ LANGUAGE plpgsql;

-- Function to record translation batch
CREATE OR REPLACE FUNCTION bot_sessions.record_translation_batch(
    p_batch_id VARCHAR,
    p_session_id VARCHAR,
    p_source_text TEXT,
    p_source_language VARCHAR,
    p_target_languages TEXT[],
    p_total_time_ms REAL,
    p_cache_hits INTEGER,
    p_cache_misses INTEGER,
    p_model_requested VARCHAR DEFAULT NULL,
    p_success_count INTEGER DEFAULT 0,
    p_error_count INTEGER DEFAULT 0,
    p_results JSONB DEFAULT '{}'::jsonb
)
RETURNS VARCHAR AS $$
DECLARE
    num_langs INTEGER;
    hit_rate REAL;
    avg_time REAL;
BEGIN
    num_langs := array_length(p_target_languages, 1);
    hit_rate := CASE WHEN (p_cache_hits + p_cache_misses) > 0
                THEN p_cache_hits::REAL / (p_cache_hits + p_cache_misses)
                ELSE 0 END;
    avg_time := CASE WHEN num_langs > 0 THEN p_total_time_ms / num_langs ELSE 0 END;

    INSERT INTO bot_sessions.translation_batches (
        batch_id, session_id, source_text, source_language, target_languages, num_languages,
        total_processing_time_ms, avg_translation_time_ms,
        cache_hit_count, cache_miss_count, cache_hit_rate,
        model_requested, success_count, error_count, translation_results
    ) VALUES (
        p_batch_id, p_session_id, p_source_text, p_source_language, p_target_languages, num_langs,
        p_total_time_ms, avg_time,
        p_cache_hits, p_cache_misses, hit_rate,
        p_model_requested, p_success_count, p_error_count, p_results
    );

    RETURN p_batch_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update model performance metrics
CREATE OR REPLACE FUNCTION bot_sessions.update_model_performance(
    p_model_name VARCHAR,
    p_model_backend VARCHAR,
    p_source_language VARCHAR,
    p_target_language VARCHAR,
    p_latency_ms REAL,
    p_success BOOLEAN,
    p_confidence REAL DEFAULT NULL,
    p_was_cached BOOLEAN DEFAULT false
)
RETURNS BOOLEAN AS $$
DECLARE
    window_start TIMESTAMP;
    window_end TIMESTAMP;
BEGIN
    -- Use 1-hour windows
    window_start := date_trunc('hour', NOW());
    window_end := window_start + INTERVAL '1 hour';

    -- Insert or update performance record
    INSERT INTO bot_sessions.model_performance (
        model_name, model_backend, source_language, target_language,
        total_translations, successful_translations, failed_translations,
        avg_latency_ms, min_latency_ms, max_latency_ms,
        avg_confidence, cache_hit_rate,
        window_start, window_end
    ) VALUES (
        p_model_name, p_model_backend, p_source_language, p_target_language,
        1,
        CASE WHEN p_success THEN 1 ELSE 0 END,
        CASE WHEN NOT p_success THEN 1 ELSE 0 END,
        p_latency_ms, p_latency_ms, p_latency_ms,
        p_confidence,
        CASE WHEN p_was_cached THEN 1.0 ELSE 0.0 END,
        window_start, window_end
    )
    ON CONFLICT (model_name, source_language, target_language, window_start) DO UPDATE SET
        total_translations = bot_sessions.model_performance.total_translations + 1,
        successful_translations = bot_sessions.model_performance.successful_translations +
            CASE WHEN p_success THEN 1 ELSE 0 END,
        failed_translations = bot_sessions.model_performance.failed_translations +
            CASE WHEN NOT p_success THEN 1 ELSE 0 END,
        avg_latency_ms = (bot_sessions.model_performance.avg_latency_ms * bot_sessions.model_performance.total_translations + p_latency_ms) /
            (bot_sessions.model_performance.total_translations + 1),
        min_latency_ms = LEAST(bot_sessions.model_performance.min_latency_ms, p_latency_ms),
        max_latency_ms = GREATEST(bot_sessions.model_performance.max_latency_ms, p_latency_ms),
        avg_confidence = CASE
            WHEN p_confidence IS NOT NULL THEN
                (COALESCE(bot_sessions.model_performance.avg_confidence, 0) * bot_sessions.model_performance.total_translations + p_confidence) /
                (bot_sessions.model_performance.total_translations + 1)
            ELSE bot_sessions.model_performance.avg_confidence
        END,
        cache_hit_rate = (bot_sessions.model_performance.cache_hit_rate * bot_sessions.model_performance.total_translations +
            CASE WHEN p_was_cached THEN 1.0 ELSE 0.0 END) / (bot_sessions.model_performance.total_translations + 1),
        last_updated = NOW();

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA bot_sessions TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bot_sessions TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bot_sessions TO postgres;

-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Translation optimization schema migration completed!';
    RAISE NOTICE '';
    RAISE NOTICE 'New tables created:';
    RAISE NOTICE '  - bot_sessions.translation_cache_stats (cache performance tracking)';
    RAISE NOTICE '  - bot_sessions.translation_batches (multi-language batch metadata)';
    RAISE NOTICE '  - bot_sessions.model_performance (model performance aggregates)';
    RAISE NOTICE '  - bot_sessions.translation_context (context-aware translation)';
    RAISE NOTICE '';
    RAISE NOTICE 'Enhanced existing tables:';
    RAISE NOTICE '  - bot_sessions.translations (added cache and model columns)';
    RAISE NOTICE '';
    RAISE NOTICE 'New views created:';
    RAISE NOTICE '  - bot_sessions.cache_performance';
    RAISE NOTICE '  - bot_sessions.batch_efficiency';
    RAISE NOTICE '  - bot_sessions.model_comparison';
    RAISE NOTICE '  - bot_sessions.session_translation_summary';
    RAISE NOTICE '';
    RAISE NOTICE 'Helper functions:';
    RAISE NOTICE '  - record_cache_stat()';
    RAISE NOTICE '  - record_translation_batch()';
    RAISE NOTICE '  - update_model_performance()';
END
$$;
