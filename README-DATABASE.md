# LiveTranslate Database Setup

This guide explains how to set up and manage the database services for LiveTranslate.

## Quick Start

### Windows (PowerShell)
```powershell
# Start database services
./scripts/start-database.ps1

# Start with clean data
./scripts/start-database.ps1 -Clean

# Start and show logs
./scripts/start-database.ps1 -ShowLogs
```

### Linux/macOS (Bash)
```bash
# Start database services
./scripts/start-database.sh

# Start with clean data
./scripts/start-database.sh --clean

# Start and show logs
./scripts/start-database.sh --logs
```

## Services Included

### PostgreSQL Database
- **Port**: 5432
- **Database**: livetranslate
- **Username**: livetranslate
- **Password**: livetranslate_dev_password (development)

### Redis Cache
- **Port**: 6379
- **Purpose**: Session storage, caching, pub/sub messaging

### pgAdmin Web Interface
- **URL**: http://localhost:8080
- **Email**: admin@livetranslate.local
- **Password**: admin (development)

## Database Schema

The database includes comprehensive schemas for:

### 1. Core Transcription (`transcription` schema)
- `audio_files` - Audio file metadata and storage references
- `transcripts` - Transcription results from Whisper and other services

### 2. Translation Services (`translation` schema)  
- `translations` - Multi-language translation results

### 3. Session Management (`sessions` schema)
- `user_sessions` - User authentication and session tracking
- `live_sessions` - Active meeting/conversation sessions
- `meeting_sessions` - Meeting metadata and lifecycle
- `session_timeline` - Event timeline for sessions

### 4. Speaker Management (`speakers` schema)
- `speaker_profiles` - Speaker identification and voice profiles
- `session_participants` - Per-session participant tracking
- `speaker_segments` - Speaker diarization results
- `speaker_interactions` - Speaker interaction analytics
- `speaker_analytics` - Comprehensive speaker statistics

### 5. Bot Sessions (`bot_sessions` schema)
- `sessions` - Google Meet bot session lifecycle
- `audio_files` - Bot-captured audio metadata
- `transcripts` - Bot transcription results
- `translations` - Bot translation results
- `correlations` - Time correlation between sources
- `participants` - Meeting participant tracking
- `events` - Session event logging
- `session_statistics` - Aggregated session metrics

### 6. Monitoring (`monitoring` schema)
- `service_health` - Service health monitoring
- `performance_metrics` - System performance tracking

## Connection Information

### Direct Database Connection
```bash
# Using psql
docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate

# Using environment variables
psql postgresql://livetranslate:livetranslate_dev_password@localhost:5432/livetranslate
```

### Redis Connection
```bash
# Using redis-cli
docker exec -it livetranslate-redis redis-cli

# Test connection
redis-cli -h localhost -p 6379 ping
```

### Application Connection Strings
```bash
# PostgreSQL
DATABASE_URL=postgresql://livetranslate:livetranslate_dev_password@localhost:5432/livetranslate

# Redis
REDIS_URL=redis://localhost:6379/0
```

## Common Operations

### Database Management
```bash
# View all databases
docker exec -it livetranslate-postgres psql -U livetranslate -c "\l"

# View all schemas
docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate -c "\dn"

# View tables in a schema
docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate -c "\dt transcription.*"

# Run SQL file
docker exec -i livetranslate-postgres psql -U livetranslate -d livetranslate < your-script.sql
```

### Container Management
```bash
# View container status
docker-compose -f docker-compose.database.yml ps

# View logs
docker-compose -f docker-compose.database.yml logs -f postgres
docker-compose -f docker-compose.database.yml logs -f redis

# Restart services
docker-compose -f docker-compose.database.yml restart

# Stop services
docker-compose -f docker-compose.database.yml down

# Stop and remove volumes (DESTRUCTIVE)
docker-compose -f docker-compose.database.yml down -v
```

### Backup and Restore
```bash
# Create backup
docker exec livetranslate-postgres pg_dump -U livetranslate -d livetranslate > backup.sql

# Restore from backup
docker exec -i livetranslate-postgres psql -U livetranslate -d livetranslate < backup.sql

# Backup with compression
docker exec livetranslate-postgres pg_dump -U livetranslate -d livetranslate | gzip > backup.sql.gz
```

## Environment Configuration

### Development Environment
- Uses `.env.database` file for configuration
- Default passwords and settings
- All services on localhost
- pgAdmin enabled for easy management

### Production Environment
- Set `MODE=prod` when starting
- Change default passwords in environment variables
- Consider external database services for scaling
- Disable pgAdmin or secure with proper authentication

### Environment Variables
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=livetranslate
POSTGRES_USER=livetranslate
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Connection URLs
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db
```

## Performance Tuning

### PostgreSQL Optimization
The database is pre-configured with optimized settings:
- `shared_buffers = 256MB`
- `effective_cache_size = 1GB`
- `work_mem = 4MB`
- `maintenance_work_mem = 64MB`

### Redis Optimization
- `maxmemory = 256mb`
- `maxmemory-policy = allkeys-lru`
- Persistence enabled with AOF

### Monitoring
- Built-in health checks for all services
- Performance metrics collection
- Log aggregation and rotation

## Troubleshooting

### Database Won't Start
```bash
# Check container logs
docker logs livetranslate-postgres

# Check if port is in use
netstat -tulpn | grep :5432

# Reset everything
./scripts/start-database.ps1 -Clean
```

### Connection Issues
```bash
# Test database connectivity
docker exec livetranslate-postgres pg_isready -U livetranslate

# Check network connectivity
docker network ls | grep livetranslate
```

### Performance Issues
```bash
# Check database connections
docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate -c "SELECT * FROM pg_stat_activity;"

# Check Redis memory usage
docker exec livetranslate-redis redis-cli info memory
```

## Security Considerations

### Development
- Default passwords are used for convenience
- All services accessible on localhost
- No SSL/TLS encryption

### Production
- Change all default passwords
- Use environment variables for secrets
- Enable SSL/TLS connections
- Restrict network access
- Regular security updates
- Implement proper backup encryption

## Integration with LiveTranslate Services

The database integrates with all LiveTranslate microservices:

- **Whisper Service**: Stores transcription results and audio metadata
- **Translation Service**: Stores translation results and quality metrics
- **Orchestration Service**: Manages sessions, bot data, and service coordination
- **Frontend Service**: Session management and user preferences

Each service connects using the provided connection strings and follows the established schema patterns for data consistency and performance.