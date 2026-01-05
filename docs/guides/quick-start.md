# Quick Start Guide

Get LiveTranslate running in 5 minutes.

## Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (optional, for persistence)

## Start Services (3 Terminals)

### Terminal 1: Orchestration
```bash
cd modules/orchestration-service
python src/main.py
```

### Terminal 2: Whisper
```bash
cd modules/whisper-service
python src/main.py
```

### Terminal 3: Translation
```bash
cd modules/translation-service
pip install python-dotenv fastapi uvicorn httpx langdetect structlog
python src/api_server_fastapi.py
```

## Test Translation

### Option 1: Chinese → English Subtitles
```bash
python simple_cn_to_en_subtitles.py
```

### Option 2: Full-Stack Test
```bash
python test_loopback_fullstack.py
```

## Next Steps
- [Database Setup](./database-setup.md) - Configure PostgreSQL
- [Translation Testing](./translation-testing.md) - Advanced testing
- [Architecture](../02-containers/README.md) - Understand the system
