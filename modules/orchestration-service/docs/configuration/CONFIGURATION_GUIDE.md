# Configuration Guide

This guide explains how to configure the LiveTranslate Orchestration Service using environment variables.

## ✅ NO HARDCODED CREDENTIALS

All sensitive configuration values (passwords, secrets, etc.) are loaded from environment variables or `.env` files. **Never commit credentials to the repository.**

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your values:**
   ```bash
   nano .env  # or use your favorite editor
   ```

3. **Key values to set:**
   - `BOT_DATABASE_PASSWORD` - If using database persistence
   - `JWT_SECRET` - For production deployments
   - Service URLs if running on non-default ports

4. **Start the service:**
   ```bash
   python src/main.py
   ```
   The service will automatically load `.env` file

## Configuration Sections

### 🖥️ Server Configuration

```bash
HOST=0.0.0.0              # Server bind address
PORT=3000                 # Server port
WORKERS=4                 # Number of worker processes
DEBUG=false               # Enable debug mode
ENVIRONMENT=development   # Environment name
```

### 🗄️ Database Configuration

```bash
DATABASE_URL=postgresql://localhost:5432/livetranslate
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
DATABASE_ECHO=false       # Set true to log SQL queries
```

### 🔴 Redis Configuration

```bash
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10
REDIS_SOCKET_TIMEOUT=5
```

**Note:** Redis is required for bot command communication. Make sure Redis is running:
```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis if not running
docker run -d -p 6379:6379 redis:latest
```

### 🤖 Bot Management Configuration

#### Docker Settings
```bash
BOT_DOCKER_IMAGE=livetranslate-bot:latest
BOT_DOCKER_NETWORK=livetranslate_default
```

#### Database Settings (Optional)
```bash
BOT_ENABLE_DATABASE=false              # Enable database persistence
BOT_DATABASE_HOST=localhost
BOT_DATABASE_PORT=5432
BOT_DATABASE_NAME=livetranslate
BOT_DATABASE_USER=postgres
BOT_DATABASE_PASSWORD=                 # SET THIS if enabling database
```

**Important:**
- Database is **disabled by default** for local development
- Enable only if you need persistent bot session storage
- Never commit `BOT_DATABASE_PASSWORD` to version control

#### Storage Settings
```bash
BOT_AUDIO_STORAGE_PATH=/tmp/livetranslate/audio
```

### 🔒 Security Configuration

```bash
JWT_SECRET=your-secret-key-here        # CHANGE IN PRODUCTION!
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600                    # Token lifetime in seconds

CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Security Best Practices:**
- Generate a strong random JWT_SECRET for production:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```
- Never use default secrets in production
- Restrict CORS_ORIGINS to your actual frontend domains

### 📝 Logging Configuration

```bash
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                        # json or text
LOG_FILE_ENABLED=true
LOG_FILE_PATH=logs/orchestration.log
LOG_MAX_FILE_SIZE=10485760            # 10MB
LOG_BACKUP_COUNT=5
```

### 📊 Monitoring Configuration

```bash
ENABLE_METRICS=true
METRICS_PORT=8000
HEALTH_CHECK_ENDPOINT=/health
```

### 🎛️ Feature Flags

```bash
ENABLE_AUDIO_PROCESSING=true
ENABLE_BOT_MANAGEMENT=true
ENABLE_TRANSLATION=true
ENABLE_ANALYTICS=true
```

## Environment-Specific Configuration

### Development
```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
BOT_ENABLE_DATABASE=false              # Keep disabled for faster startup
```

### Staging
```bash
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
BOT_ENABLE_DATABASE=true
```

### Production
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
BOT_ENABLE_DATABASE=true
JWT_SECRET=<use-strong-random-secret>
CORS_ORIGINS=https://yourdomain.com
```

## Configuration Priority

Configuration values are loaded in this order (later values override earlier):

1. Default values in `config.py`
2. Environment variables
3. `.env` file (if present)
4. Command-line arguments (if applicable)

## Verifying Configuration

After starting the service, verify configuration:

```bash
# Check health endpoint
curl http://localhost:3000/api/health

# Check debug health (shows configuration status)
curl http://localhost:3000/debug/health

# View all registered routes
curl http://localhost:3000/debug/routes | jq
```

## Common Issues

### Issue: Bot manager can't connect to Redis
```
ERROR: Failed to initialize Redis client: Error 8 connecting to redis:6379
```

**Solution:**
1. Check Redis is running: `docker ps | grep redis`
2. Verify REDIS_URL in `.env`: `REDIS_URL=redis://localhost:6379/0`
3. Start Redis if needed: `docker run -d -p 6379:6379 redis:latest`

### Issue: Database connection errors
```
ERROR: Failed to initialize database
```

**Solution:**
1. If you don't need database persistence, set `BOT_ENABLE_DATABASE=false`
2. If you do need it:
   - Verify database is running
   - Check BOT_DATABASE_* settings in `.env`
   - Ensure BOT_DATABASE_PASSWORD is set

### Issue: Docker client not available
```
WARNING: Docker SDK not available - running in mock mode
```

**Solution:**
1. Install Docker SDK: `poetry add docker`
2. Verify Docker is running: `docker ps`
3. Restart the service

## Security Checklist

Before deploying to production:

- [ ] Changed `JWT_SECRET` to a strong random value
- [ ] Set `DEBUG=false`
- [ ] Updated `CORS_ORIGINS` to your actual domains
- [ ] Set `BOT_DATABASE_PASSWORD` if using database
- [ ] Reviewed all default values
- [ ] Ensured `.env` file is in `.gitignore`
- [ ] Enabled HTTPS in production
- [ ] Set up proper firewall rules
- [ ] Configured log rotation
- [ ] Set up monitoring and alerts

## Example Production `.env`

```bash
# Production configuration
ENVIRONMENT=production
DEBUG=false

# Server
HOST=0.0.0.0
PORT=3000
WORKERS=4

# Security
JWT_SECRET=<generated-with-secrets-token_urlsafe-32>
CORS_ORIGINS=https://app.yourdomain.com

# Database
DATABASE_URL=postgresql://dbuser:dbpass@db.internal:5432/livetranslate
REDIS_URL=redis://redis.internal:6379/0

# Bot Management
BOT_ENABLE_DATABASE=true
BOT_DATABASE_HOST=db.internal
BOT_DATABASE_PASSWORD=<secure-password>
BOT_DOCKER_IMAGE=your-registry/livetranslate-bot:v1.0.0

# Logging
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE_ENABLED=true
```

## Getting Help

- **Configuration issues:** Check this guide and `.env.example`
- **Service not starting:** Check logs and `debug/health` endpoint
- **Bot issues:** See `BOT_TESTING_GUIDE.md`
- **Security concerns:** Review security checklist above

---

**Remember:** Never commit `.env` files or credentials to version control!
