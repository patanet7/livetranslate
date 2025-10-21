# Translation Optimization Database Setup

## Overview

The orchestration service database has been enhanced to support translation optimization features including caching, model selection, batch tracking, and performance metrics.

---

## 📁 Files Created

### 1. Database Migration Script
**File**: `scripts/migration-translation-optimization.sql`

Run this migration to add translation optimization support:

```bash
# Connect to your PostgreSQL database
psql -U postgres -d livetranslate -f scripts/migration-translation-optimization.sql
```

### 2. Python Database Adapter
**File**: `modules/orchestration-service/src/database/translation_optimization_adapter.py`

Python class to interact with the optimization schema.

---

## 🗄️ New Database Tables

### 1. `bot_sessions.translation_cache_stats`
**Purpose**: Track cache performance per translation request

**Key Columns**:
- `cache_key_hash` - MD5 hash of text + language pair
- `was_cache_hit` - Whether cache hit occurred
- `cache_latency_ms` - Latency for cache lookup
- `translation_latency_ms` - Latency for actual translation (if miss)
- `model_used` - Which model was used
- `confidence_score` - Translation confidence
- `quality_score` - Translation quality

**Indexes**:
- By session_id, cache_key, hit/miss status, languages, created_at

---

### 2. `bot_sessions.translation_batches`
**Purpose**: Track multi-language translation batch operations

**Key Columns**:
- `batch_id` - Unique batch identifier
- `source_text` - Original text
- `target_languages` - Array of target languages
- `num_languages` - Count of languages
- `total_processing_time_ms` - Total batch processing time
- `cache_hit_count` / `cache_miss_count` - Cache performance
- `cache_hit_rate` - Calculated hit rate
- `model_requested` - Requested model
- `models_used` - JSONB map of {language: model}
- `success_count` / `error_count` - Success metrics
- `translation_results` - Complete results JSON

**Indexes**:
- By session_id, model, hit_rate, created_at

---

### 3. `bot_sessions.model_performance`
**Purpose**: Aggregate performance metrics by model and language pair

**Key Columns**:
- `model_name` - Model identifier (llama, nllb, etc.)
- `model_backend` - Backend service
- `source_language` / `target_language` - Language pair
- `total_translations` - Count of translations
- `successful_translations` / `failed_translations` - Success tracking
- `avg_latency_ms`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms` - Latency percentiles
- `avg_confidence` - Average confidence score
- `cache_hit_rate` - Cache effectiveness
- `window_start` / `window_end` - Time window for aggregation (1-hour windows)

**Indexes**:
- By model_name, languages, time window, last_updated

**Time Windows**: Metrics are aggregated in 1-hour windows for trending

---

### 4. `bot_sessions.translation_context`
**Purpose**: Store translation context for context-aware translations

**Key Columns**:
- `session_id` - Bot session ID
- `target_language` - Target language code
- `context_window_size` - Number of recent translations to keep (default: 5)
- `recent_translations` - JSONB array of recent {source, translation, timestamp}
- `custom_terminology` - JSONB map of custom term translations
- `style_preferences` - JSONB preferences (formality, gender, etc.)
- `last_activity` - Last context update timestamp

**Indexes**:
- By session_id, target_language, last_activity

---

## 📊 New Database Views

### 1. `bot_sessions.cache_performance`
**Purpose**: Cache hit rate and latency by session/model/language

**Columns**:
- session_id, source_language, target_language, model_used
- total_lookups, cache_hits, cache_misses
- hit_rate_percent
- avg_cache_hit_latency_ms, avg_translation_latency_ms
- avg_confidence

**Usage**:
```sql
-- Get cache performance for a session
SELECT * FROM bot_sessions.cache_performance
WHERE session_id = 'session_123';

-- Get overall cache performance by model
SELECT model_used, SUM(total_lookups) as total, AVG(hit_rate_percent) as avg_hit_rate
FROM bot_sessions.cache_performance
GROUP BY model_used;
```

---

### 2. `bot_sessions.batch_efficiency`
**Purpose**: Multi-language batch translation efficiency metrics

**Columns**:
- session_id, model_requested
- total_batches, avg_languages_per_batch
- avg_total_time_ms, avg_per_language_ms
- avg_cache_hit_rate_percent
- total_cache_hits, total_cache_misses
- success_rate_percent

**Usage**:
```sql
-- Get batch efficiency for a session
SELECT * FROM bot_sessions.batch_efficiency
WHERE session_id = 'session_123';

-- Compare batch efficiency by model
SELECT model_requested, avg_cache_hit_rate_percent, avg_per_language_ms
FROM bot_sessions.batch_efficiency
ORDER BY avg_per_language_ms ASC;
```

---

### 3. `bot_sessions.model_comparison`
**Purpose**: Compare model performance across language pairs

**Columns**:
- model_name, source_language, target_language
- total_translations
- avg_latency_ms, p95_latency_ms
- avg_confidence, cache_hit_rate_percent
- success_rate_percent

**Usage**:
```sql
-- Compare models for a specific language pair
SELECT * FROM bot_sessions.model_comparison
WHERE source_language = 'en' AND target_language = 'es'
ORDER BY avg_latency_ms ASC;

-- Find best performing model overall
SELECT model_name, AVG(avg_latency_ms) as overall_latency, AVG(avg_confidence) as overall_confidence
FROM bot_sessions.model_comparison
GROUP BY model_name
ORDER BY overall_latency ASC;
```

---

### 4. `bot_sessions.session_translation_summary`
**Purpose**: Session-level translation summary with optimization metrics

**Columns**:
- session_id
- total_translations, unique_target_languages, models_used
- cached_translations, cache_hit_rate_percent
- avg_confidence, avg_latency_ms
- batches_used, models_list

**Usage**:
```sql
-- Get summary for a session
SELECT * FROM bot_sessions.session_translation_summary
WHERE session_id = 'session_123';
```

---

## 🔧 Database Functions

### 1. `record_cache_stat(...)`
Records a single cache lookup statistic.

**Parameters**:
- p_session_id, p_cache_key_hash, p_source_text, p_source_language, p_target_language
- p_was_cache_hit, p_cache_latency_ms, p_translation_latency_ms
- p_model_used, p_translation_service, p_confidence, p_quality

**Returns**: cache_stat_id

**Example**:
```sql
SELECT bot_sessions.record_cache_stat(
    'session_123',
    'abc123def456...',
    'Hello world',
    'en',
    'es',
    true,  -- was cache hit
    2.5,   -- cache latency ms
    NULL,  -- no translation latency (was cached)
    'llama',
    'llama_transformers',
    0.95,  -- confidence
    0.92   -- quality
);
```

---

### 2. `record_translation_batch(...)`
Records a multi-language translation batch.

**Parameters**:
- p_batch_id, p_session_id, p_source_text, p_source_language
- p_target_languages (array), p_total_time_ms
- p_cache_hits, p_cache_misses, p_model_requested
- p_success_count, p_error_count, p_results (JSONB)

**Returns**: batch_id

**Example**:
```sql
SELECT bot_sessions.record_translation_batch(
    'batch_abc123',
    'session_123',
    'Hello world',
    'en',
    ARRAY['es', 'fr', 'de'],
    150.0,  -- total time ms
    2,      -- cache hits
    1,      -- cache misses
    'llama',
    3,      -- success count
    0,      -- error count
    '{"es": {"text": "Hola mundo", "confidence": 0.95}, ...}'::jsonb
);
```

---

### 3. `update_model_performance(...)`
Updates aggregated model performance metrics.

**Parameters**:
- p_model_name, p_model_backend, p_source_language, p_target_language
- p_latency_ms, p_success, p_confidence, p_was_cached

**Returns**: boolean (success)

**Example**:
```sql
SELECT bot_sessions.update_model_performance(
    'llama',
    'llama_transformers',
    'en',
    'es',
    120.5,  -- latency ms
    true,   -- success
    0.95,   -- confidence
    false   -- was not cached
);
```

**Note**: This function automatically aggregates into 1-hour windows

---

## 🔄 Enhanced Existing Tables

### Updates to `bot_sessions.translations`

**New Columns Added**:
- `model_name` VARCHAR(100) - Model used for translation
- `model_backend` VARCHAR(100) - Backend service
- `batch_id` VARCHAR(100) - Reference to translation_batches
- `was_cached` BOOLEAN - Whether result was cached
- `cache_latency_ms` REAL - Cache lookup latency
- `translation_latency_ms` REAL - Translation latency
- `optimization_metadata` JSONB - Additional optimization data

**New Indexes**:
- `idx_translations_model` - On model_name
- `idx_translations_cached` - On was_cached
- `idx_translations_batch_id` - On batch_id
- `idx_translations_opt_metadata_gin` - GIN index on optimization_metadata

---

## 🐍 Python Usage

### Initialize Adapter

```python
from database.database import DatabaseManager
from database.translation_optimization_adapter import TranslationOptimizationAdapter

# Initialize database manager
db_manager = DatabaseManager(database_url="postgresql://...")
await db_manager.initialize()

# Initialize optimization adapter
opt_adapter = TranslationOptimizationAdapter(db_manager)
await opt_adapter.initialize()
```

### Record Cache Statistics

```python
# Record cache hit
await opt_adapter.record_cache_stat(
    session_id="session_123",
    text="Hello world",
    source_language="en",
    target_language="es",
    was_cache_hit=True,
    cache_latency_ms=2.5,
    model_used="llama",
    translation_service="llama_transformers",
    confidence=0.95
)

# Record cache miss
await opt_adapter.record_cache_stat(
    session_id="session_123",
    text="New phrase",
    source_language="en",
    target_language="fr",
    was_cache_hit=False,
    cache_latency_ms=1.2,
    translation_latency_ms=150.0,
    model_used="nllb",
    confidence=0.88
)
```

### Record Translation Batch

```python
await opt_adapter.record_translation_batch(
    session_id="session_123",
    source_text="Hello world",
    source_language="en",
    target_languages=["es", "fr", "de"],
    total_time_ms=180.0,
    cache_hits=2,
    cache_misses=1,
    model_requested="llama",
    success_count=3,
    error_count=0,
    results={
        "es": {"translated_text": "Hola mundo", "confidence": 0.95},
        "fr": {"translated_text": "Bonjour le monde", "confidence": 0.93},
        "de": {"translated_text": "Hallo Welt", "confidence": 0.94}
    }
)
```

### Update Model Performance

```python
await opt_adapter.update_model_performance(
    model_name="llama",
    model_backend="llama_transformers",
    source_language="en",
    target_language="es",
    latency_ms=125.0,
    success=True,
    confidence=0.95,
    was_cached=False
)
```

### Get Analytics

```python
# Get cache performance
cache_stats = await opt_adapter.get_cache_performance(session_id="session_123")
for stat in cache_stats:
    print(f"Lang: {stat['source_language']}→{stat['target_language']}")
    print(f"Hit rate: {stat['hit_rate_percent']}%")
    print(f"Avg latency: {stat['avg_cache_hit_latency_ms']}ms")

# Get batch efficiency
batch_stats = await opt_adapter.get_batch_efficiency(session_id="session_123")
for stat in batch_stats:
    print(f"Model: {stat['model_requested']}")
    print(f"Avg languages per batch: {stat['avg_languages_per_batch']}")
    print(f"Cache hit rate: {stat['avg_cache_hit_rate_percent']}%")

# Get model comparison
model_stats = await opt_adapter.get_model_comparison(
    source_language="en",
    target_language="es"
)
for stat in model_stats:
    print(f"Model: {stat['model_name']}")
    print(f"Avg latency: {stat['avg_latency_ms']}ms")
    print(f"Confidence: {stat['avg_confidence']}")
    print(f"Success rate: {stat['success_rate_percent']}%")

# Get session summary
summary = await opt_adapter.get_session_translation_summary("session_123")
print(f"Total translations: {summary['total_translations']}")
print(f"Cache hit rate: {summary['cache_hit_rate_percent']}%")
print(f"Models used: {summary['models_list']}")
```

---

## 📊 Example Queries

### Find Sessions with Low Cache Hit Rate

```sql
SELECT session_id, cache_hit_rate_percent, total_translations
FROM bot_sessions.session_translation_summary
WHERE cache_hit_rate_percent < 50
ORDER BY total_translations DESC;
```

### Identify Slowest Model/Language Combinations

```sql
SELECT model_name, source_language, target_language, avg_latency_ms, p95_latency_ms
FROM bot_sessions.model_comparison
ORDER BY p95_latency_ms DESC
LIMIT 10;
```

### Analyze Batch Performance Over Time

```sql
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    AVG(cache_hit_rate) as avg_hit_rate,
    AVG(total_processing_time_ms) as avg_processing_time,
    COUNT(*) as batch_count
FROM bot_sessions.translation_batches
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
```

### Find Most Translated Phrases (Cache Key Analysis)

```sql
SELECT
    cache_key_hash,
    source_text_preview,
    source_language,
    target_language,
    COUNT(*) as lookup_count,
    SUM(CASE WHEN was_cache_hit THEN 1 ELSE 0 END) as hit_count,
    ROUND(AVG(cache_latency_ms), 2) as avg_latency
FROM bot_sessions.translation_cache_stats
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY cache_key_hash, source_text_preview, source_language, target_language
HAVING COUNT(*) > 10
ORDER BY lookup_count DESC
LIMIT 20;
```

---

## 🔒 Permissions

The migration script automatically grants permissions to the `postgres` user. For production, create specific service accounts:

```sql
-- Create service account
CREATE USER orchestration_service WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT USAGE ON SCHEMA bot_sessions TO orchestration_service;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA bot_sessions TO orchestration_service;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA bot_sessions TO orchestration_service;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bot_sessions TO orchestration_service;
```

---

## 🧪 Testing

### Verify Migration Success

```sql
-- Check all new tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'bot_sessions'
AND table_name IN (
    'translation_cache_stats',
    'translation_batches',
    'model_performance',
    'translation_context'
);

-- Check new columns in translations table
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'bot_sessions'
AND table_name = 'translations'
AND column_name IN (
    'model_name', 'model_backend', 'batch_id',
    'was_cached', 'cache_latency_ms', 'translation_latency_ms',
    'optimization_metadata'
);

-- Check views exist
SELECT table_name
FROM information_schema.views
WHERE table_schema = 'bot_sessions'
AND table_name IN (
    'cache_performance',
    'batch_efficiency',
    'model_comparison',
    'session_translation_summary'
);

-- Check functions exist
SELECT routine_name
FROM information_schema.routines
WHERE routine_schema = 'bot_sessions'
AND routine_name IN (
    'record_cache_stat',
    'record_translation_batch',
    'update_model_performance'
);
```

### Test Database Operations

```sql
-- Insert test cache stat
SELECT bot_sessions.record_cache_stat(
    'test_session', 'test_hash', 'test text', 'en', 'es',
    true, 2.5, NULL, 'llama', 'test_service', 0.95, 0.90
);

-- Insert test batch
SELECT bot_sessions.record_translation_batch(
    'test_batch', 'test_session', 'test text', 'en', ARRAY['es', 'fr'],
    150.0, 1, 1, 'llama', 2, 0, '{}'::jsonb
);

-- Update test model performance
SELECT bot_sessions.update_model_performance(
    'llama', 'transformers', 'en', 'es', 120.0, true, 0.95, false
);

-- Query test results
SELECT * FROM bot_sessions.cache_performance WHERE session_id = 'test_session';
SELECT * FROM bot_sessions.batch_efficiency WHERE session_id = 'test_session';
SELECT * FROM bot_sessions.model_comparison WHERE model_name = 'llama';
```

---

## 📈 Performance Monitoring

### Key Metrics to Track

1. **Cache Hit Rate**: Target > 60%
   ```sql
   SELECT AVG(cache_hit_rate_percent) FROM bot_sessions.cache_performance;
   ```

2. **Average Latency**: Target < 200ms
   ```sql
   SELECT AVG(avg_latency_ms) FROM bot_sessions.session_translation_summary;
   ```

3. **Batch Efficiency**: Monitor languages per batch
   ```sql
   SELECT AVG(avg_languages_per_batch) FROM bot_sessions.batch_efficiency;
   ```

4. **Model Success Rate**: Target > 95%
   ```sql
   SELECT model_name, AVG(success_rate_percent)
   FROM bot_sessions.model_comparison
   GROUP BY model_name;
   ```

---

## 🔄 Migration Steps

### Step-by-Step Setup

1. **Backup existing database** (if in production):
   ```bash
   pg_dump -U postgres livetranslate > backup_$(date +%Y%m%d).sql
   ```

2. **Run optimization migration**:
   ```bash
   psql -U postgres -d livetranslate -f scripts/migration-translation-optimization.sql
   ```

3. **Verify migration**:
   ```bash
   psql -U postgres -d livetranslate -c "SELECT * FROM bot_sessions.cache_performance LIMIT 1;"
   ```

4. **Update application code**:
   - Import `TranslationOptimizationAdapter`
   - Initialize in application startup
   - Update `AudioCoordinator` to use adapter

5. **Monitor initial data**:
   ```sql
   -- Check if data is being recorded
   SELECT COUNT(*) FROM bot_sessions.translation_cache_stats;
   SELECT COUNT(*) FROM bot_sessions.translation_batches;
   ```

---

## 🚧 Future Enhancements

- [ ] Add materialized views for faster analytics
- [ ] Implement automatic data retention policies (archive old records)
- [ ] Add alerting for low cache hit rates
- [ ] Create Grafana dashboard for real-time monitoring
- [ ] Implement ML-based model selection based on historical performance

---

*Last Updated: 2025-01-20*
*Schema Version: 1.0*
