# Quick Start: Database Initialization

**Estimated Time**: 10-15 minutes
**Prerequisites**: Docker running, PostgreSQL container available

---

## Quick Setup (Recommended)

### Using Shared PostgreSQL Container (Default)

If you already have PostgreSQL running (like `kyc_postgres_dev`):

```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate
chmod +x scripts/quick_db_setup.sh
./scripts/quick_db_setup.sh
```

**Uses:**
- Container: `kyc_postgres_dev` (existing)
- Database: `livetranslate` (isolated)
- Password: `password123`

### Create New Dedicated Container

For a separate LiveTranslate PostgreSQL:

```bash
./scripts/quick_db_setup.sh --new-container
```

**Creates:**
- Container: `livetranslate-postgres` (new)
- Database: `livetranslate`
- Password: `livetranslate`

### With Git Commits

```bash
./scripts/quick_db_setup.sh --git-commits
```

---

## What Gets Created

**This will automatically**:
1. ✅ Verify/start PostgreSQL container
2. ✅ Initialize schema (608 SQL lines, 9 tables)
3. ✅ Apply speaker enhancements migration (255 SQL lines)
4. ✅ Update .env configuration with correct password
5. ✅ Test database connection
6. ✅ Run data pipeline tests (13-15 tests passing)
7. ✅ Update documentation

---

## Verify Setup

```bash
# For shared container:
docker exec kyc_postgres_dev psql -U postgres -d livetranslate -c "\dt bot_sessions.*"

# For new container:
docker exec livetranslate-postgres psql -U postgres -d livetranslate -c "\dt bot_sessions.*"

# Should show 9 tables
```

---

## Done!

**Total Commits**: 9 git commits
**Database**: PostgreSQL 15 with 9 tables, 40+ indexes, 4 views
**Status**: Ready for Priority 2 (Translation Service GPU Optimization)

---

## If Something Goes Wrong

See `DATABASE_SETUP_GUIDE.md` for troubleshooting (477 lines of detailed help).

**Common Issues**:
- **Docker not running**: Start Docker Desktop
- **Port 5432 in use**: Stop existing PostgreSQL: `docker stop livetranslate-postgres`
- **Tests fail**: Check database connection in `.env` file

---

## What Happens Next

After database initialization:
1. **Priority 2**: Translation Service GPU Optimization (8-12 hours)
2. **Priority 3**: End-to-End Integration Testing (4-6 hours)
3. **Priority 4**: Whisper Session State Persistence (6-8 hours)

See `modules/orchestration-service/plan.md` for details.
