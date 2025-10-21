-- Chat History Database Schema
-- PostgreSQL schema for user conversation persistence
-- Based on Vexa patterns with user-centric scoping
--
-- Features:
-- - Multi-tenant user isolation
-- - Full-text search on message content
-- - JSONB for flexible metadata and translations
-- - Efficient indexing for date range queries
-- - Automatic timestamp management
-- - Cascading deletes for data integrity

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For full-text search

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    -- Primary key
    user_id VARCHAR(255) PRIMARY KEY,

    -- User information
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100),
    image_url TEXT,

    -- Configuration
    max_concurrent_sessions INTEGER NOT NULL DEFAULT 10,
    preferred_language VARCHAR(10) DEFAULT 'en',

    -- User preferences (flexible JSONB storage)
    preferences JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP,

    -- Account status
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Indexes for users table
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
CREATE INDEX IF NOT EXISTS ix_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS ix_users_last_active ON users(last_active_at);
CREATE INDEX IF NOT EXISTS ix_users_preferences_gin ON users USING gin(preferences);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_users_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_users_updated_at();

-- ============================================================================
-- API TOKENS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_tokens (
    -- Primary key
    token_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Token information
    token VARCHAR(255) NOT NULL UNIQUE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Token metadata
    name VARCHAR(100),
    scopes JSONB NOT NULL DEFAULT '["read", "write"]'::jsonb,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Indexes for api_tokens table
CREATE INDEX IF NOT EXISTS ix_api_tokens_token ON api_tokens(token);
CREATE INDEX IF NOT EXISTS ix_api_tokens_user_id ON api_tokens(user_id);
CREATE INDEX IF NOT EXISTS ix_api_tokens_expires_at ON api_tokens(expires_at);

-- ============================================================================
-- CONVERSATION SESSIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversation_sessions (
    -- Primary key
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- User scoping (CRITICAL for multi-tenant isolation)
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Session metadata
    session_type VARCHAR(50) NOT NULL DEFAULT 'user_chat',
    session_title VARCHAR(500),

    -- Session lifecycle
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP,
    last_message_at TIMESTAMP,

    -- Session configuration
    target_languages JSONB,
    enable_translation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Session statistics (denormalized for performance)
    message_count INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,

    -- Session metadata (flexible JSONB storage)
    session_metadata JSONB
);

-- Indexes for conversation_sessions table
CREATE INDEX IF NOT EXISTS ix_conv_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS ix_conv_sessions_started_at ON conversation_sessions(started_at);
CREATE INDEX IF NOT EXISTS ix_conv_sessions_last_message_at ON conversation_sessions(last_message_at);
CREATE INDEX IF NOT EXISTS ix_conv_sessions_session_type ON conversation_sessions(session_type);
CREATE INDEX IF NOT EXISTS ix_conv_sessions_user_started ON conversation_sessions(user_id, started_at);
CREATE INDEX IF NOT EXISTS ix_conv_sessions_metadata_gin ON conversation_sessions USING gin(session_metadata);

-- ============================================================================
-- CHAT MESSAGES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    -- Primary key
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Session scoping
    session_id UUID NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    -- Message ordering (auto-incrementing within session)
    sequence_number INTEGER NOT NULL,

    -- Message role
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    -- Message content
    content TEXT NOT NULL,
    original_language VARCHAR(10) DEFAULT 'en',

    -- Translations (JSONB for flexibility)
    -- Format: {"es": "Hola mundo", "fr": "Bonjour le monde"}
    translated_content JSONB,

    -- Message metadata
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    edited_at TIMESTAMP,

    -- Quality metrics
    confidence INTEGER CHECK (confidence >= 0 AND confidence <= 100),
    token_count INTEGER,

    -- Metadata (flexible JSONB storage)
    message_metadata JSONB,

    -- Constraint: unique sequence number per session
    CONSTRAINT unique_session_sequence UNIQUE (session_id, sequence_number)
);

-- Indexes for chat_messages table
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS ix_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS ix_chat_messages_role ON chat_messages(role);
CREATE INDEX IF NOT EXISTS ix_chat_messages_sequence ON chat_messages(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS ix_chat_messages_content_fulltext ON chat_messages USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_chat_messages_translated_gin ON chat_messages USING gin(translated_content);

-- Trigger to auto-increment sequence_number
CREATE OR REPLACE FUNCTION set_message_sequence_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.sequence_number IS NULL THEN
        SELECT COALESCE(MAX(sequence_number), 0) + 1
        INTO NEW.sequence_number
        FROM chat_messages
        WHERE session_id = NEW.session_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_message_sequence
    BEFORE INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION set_message_sequence_number();

-- Trigger to update session statistics on message insert
CREATE OR REPLACE FUNCTION update_session_on_message_insert()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversation_sessions
    SET
        message_count = message_count + 1,
        last_message_at = NEW.timestamp,
        total_tokens = total_tokens + COALESCE(NEW.token_count, 0)
    WHERE session_id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_session_on_message
    AFTER INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_session_on_message_insert();

-- ============================================================================
-- CONVERSATION STATISTICS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversation_statistics (
    -- Primary key
    statistics_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL UNIQUE REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,

    -- Message statistics
    total_messages INTEGER NOT NULL DEFAULT 0,
    user_messages INTEGER NOT NULL DEFAULT 0,
    assistant_messages INTEGER NOT NULL DEFAULT 0,

    -- Content statistics
    total_characters INTEGER NOT NULL DEFAULT 0,
    total_words INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,

    -- Language statistics
    languages_used JSONB,
    translation_count INTEGER NOT NULL DEFAULT 0,

    -- Timing statistics
    duration_seconds INTEGER,
    avg_response_time INTEGER,

    -- Quality metrics
    average_confidence INTEGER CHECK (average_confidence >= 0 AND average_confidence <= 100),

    -- Timestamps
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for conversation_statistics table
CREATE INDEX IF NOT EXISTS ix_conv_stats_session_id ON conversation_statistics(session_id);
CREATE INDEX IF NOT EXISTS ix_conv_stats_calculated_at ON conversation_statistics(calculated_at);

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View for active user sessions
CREATE OR REPLACE VIEW active_user_sessions AS
SELECT
    u.user_id,
    u.email,
    u.name,
    cs.session_id,
    cs.session_title,
    cs.started_at,
    cs.last_message_at,
    cs.message_count
FROM users u
JOIN conversation_sessions cs ON u.user_id = cs.user_id
WHERE cs.ended_at IS NULL
ORDER BY cs.last_message_at DESC;

-- View for recent conversations (last 30 days)
CREATE OR REPLACE VIEW recent_conversations AS
SELECT
    cs.session_id,
    cs.user_id,
    cs.session_title,
    cs.started_at,
    cs.ended_at,
    cs.message_count,
    COUNT(cm.message_id) as actual_message_count,
    MAX(cm.timestamp) as last_message_time
FROM conversation_sessions cs
LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
WHERE cs.started_at >= NOW() - INTERVAL '30 days'
GROUP BY cs.session_id
ORDER BY cs.started_at DESC;

-- View for user activity summary
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT
    u.user_id,
    u.email,
    u.name,
    COUNT(DISTINCT cs.session_id) as total_sessions,
    COUNT(cm.message_id) as total_messages,
    MAX(cs.last_message_at) as last_active,
    SUM(cs.message_count) as total_message_count
FROM users u
LEFT JOIN conversation_sessions cs ON u.user_id = cs.user_id
LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
GROUP BY u.user_id, u.email, u.name
ORDER BY last_active DESC;

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to calculate conversation statistics
CREATE OR REPLACE FUNCTION calculate_conversation_statistics(p_session_id UUID)
RETURNS void AS $$
DECLARE
    v_total_messages INTEGER;
    v_user_messages INTEGER;
    v_assistant_messages INTEGER;
    v_total_characters BIGINT;
    v_total_words BIGINT;
    v_total_tokens INTEGER;
    v_avg_confidence INTEGER;
    v_duration INTEGER;
    v_languages_used JSONB;
BEGIN
    -- Calculate message counts
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE role = 'user'),
        COUNT(*) FILTER (WHERE role = 'assistant')
    INTO v_total_messages, v_user_messages, v_assistant_messages
    FROM chat_messages
    WHERE session_id = p_session_id;

    -- Calculate content statistics
    SELECT
        SUM(LENGTH(content)),
        SUM(array_length(regexp_split_to_array(content, '\s+'), 1)),
        SUM(COALESCE(token_count, 0)),
        AVG(confidence)::INTEGER
    INTO v_total_characters, v_total_words, v_total_tokens, v_avg_confidence
    FROM chat_messages
    WHERE session_id = p_session_id;

    -- Calculate duration
    SELECT EXTRACT(EPOCH FROM (ended_at - started_at))::INTEGER
    INTO v_duration
    FROM conversation_sessions
    WHERE session_id = p_session_id AND ended_at IS NOT NULL;

    -- Get unique languages used
    SELECT jsonb_agg(DISTINCT original_language)
    INTO v_languages_used
    FROM chat_messages
    WHERE session_id = p_session_id AND original_language IS NOT NULL;

    -- Insert or update statistics
    INSERT INTO conversation_statistics (
        session_id,
        total_messages,
        user_messages,
        assistant_messages,
        total_characters,
        total_words,
        total_tokens,
        average_confidence,
        duration_seconds,
        languages_used,
        calculated_at
    ) VALUES (
        p_session_id,
        v_total_messages,
        v_user_messages,
        v_assistant_messages,
        v_total_characters,
        v_total_words,
        v_total_tokens,
        v_avg_confidence,
        v_duration,
        v_languages_used,
        NOW()
    )
    ON CONFLICT (session_id)
    DO UPDATE SET
        total_messages = EXCLUDED.total_messages,
        user_messages = EXCLUDED.user_messages,
        assistant_messages = EXCLUDED.assistant_messages,
        total_characters = EXCLUDED.total_characters,
        total_words = EXCLUDED.total_words,
        total_tokens = EXCLUDED.total_tokens,
        average_confidence = EXCLUDED.average_confidence,
        duration_seconds = EXCLUDED.duration_seconds,
        languages_used = EXCLUDED.languages_used,
        calculated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to search messages by content (full-text search)
CREATE OR REPLACE FUNCTION search_messages(
    p_user_id VARCHAR(255),
    p_search_term TEXT,
    p_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    message_id UUID,
    session_id UUID,
    content TEXT,
    role VARCHAR(20),
    timestamp TIMESTAMP,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cm.message_id,
        cm.session_id,
        cm.content,
        cm.role,
        cm.timestamp,
        similarity(cm.content, p_search_term) as similarity
    FROM chat_messages cm
    JOIN conversation_sessions cs ON cm.session_id = cs.session_id
    WHERE cs.user_id = p_user_id
        AND cm.content % p_search_term  -- Trigram similarity
    ORDER BY similarity DESC, cm.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Create sample user
INSERT INTO users (user_id, email, name, preferred_language)
VALUES ('test_user_123', 'test@example.com', 'Test User', 'en')
ON CONFLICT (user_id) DO NOTHING;

-- Create sample conversation session
DO $$
DECLARE
    v_session_id UUID;
BEGIN
    INSERT INTO conversation_sessions (user_id, session_type, session_title)
    VALUES ('test_user_123', 'user_chat', 'Test Conversation')
    RETURNING session_id INTO v_session_id;

    -- Add sample messages
    INSERT INTO chat_messages (session_id, role, content, original_language)
    VALUES
        (v_session_id, 'user', 'Hello, how are you?', 'en'),
        (v_session_id, 'assistant', 'I''m doing well, thank you! How can I help you today?', 'en'),
        (v_session_id, 'user', 'Can you help me with database design?', 'en'),
        (v_session_id, 'assistant', 'Of course! I''d be happy to help with database design.', 'en');
END $$;

-- ============================================================================
-- GRANTS (adjust as needed for your environment)
-- ============================================================================

-- Grant permissions to application role (adjust role name as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO livetranslate_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO livetranslate_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO livetranslate_app;

-- ============================================================================
-- SCHEMA COMPLETE
-- ============================================================================

-- Verify installation
SELECT 'Chat history schema installed successfully!' as status;

-- Show table counts
SELECT
    'users' as table_name,
    COUNT(*) as row_count
FROM users
UNION ALL
SELECT 'conversation_sessions', COUNT(*) FROM conversation_sessions
UNION ALL
SELECT 'chat_messages', COUNT(*) FROM chat_messages
UNION ALL
SELECT 'api_tokens', COUNT(*) FROM api_tokens
UNION ALL
SELECT 'conversation_statistics', COUNT(*) FROM conversation_statistics;
