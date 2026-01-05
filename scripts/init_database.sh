#!/bin/bash
#
# Database Initialization Script
#
# This script helps initialize or migrate the LiveTranslate database
# for the data pipeline.
#
# Usage:
#   ./init_database.sh [fresh|migrate]
#
# Options:
#   fresh   - Initialize a fresh database (WARNING: drops existing data)
#   migrate - Apply migration to existing database (safe, idempotent)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from environment or defaults
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-livetranslate}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-livetranslate}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# SQL files
INIT_SQL="${SCRIPT_DIR}/database-init-complete.sql"
MIGRATE_SQL="${SCRIPT_DIR}/migrations/001_speaker_enhancements.sql"

# Function to print colored output
print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if psql is installed
check_psql() {
    if ! command -v psql &> /dev/null; then
        print_error "psql command not found. Please install PostgreSQL client."
        exit 1
    fi
}

# Function to check database connection
check_connection() {
    print_header "Checking database connection..."

    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" > /dev/null 2>&1; then
        print_success "Connected to database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
        return 0
    else
        print_error "Cannot connect to database: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
        print_warning "Please check your connection settings and ensure PostgreSQL is running."
        return 1
    fi
}

# Function to check if schema exists
check_schema() {
    print_header "Checking for existing schema..."

    SCHEMA_EXISTS=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -tAc \
        "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'bot_sessions');")

    if [ "$SCHEMA_EXISTS" = "t" ]; then
        print_warning "Schema 'bot_sessions' already exists"
        return 0
    else
        print_success "Schema 'bot_sessions' does not exist (fresh installation)"
        return 1
    fi
}

# Function to initialize fresh database
init_fresh() {
    print_header "═══════════════════════════════════════════════════════════"
    print_header "  FRESH DATABASE INITIALIZATION"
    print_header "═══════════════════════════════════════════════════════════"
    echo ""

    if ! check_connection; then
        exit 1
    fi

    if check_schema; then
        echo ""
        print_warning "WARNING: This will drop all existing data in the bot_sessions schema!"
        print_warning "Are you sure you want to continue? (yes/no)"
        read -r CONFIRM

        if [ "$CONFIRM" != "yes" ]; then
            print_warning "Aborted."
            exit 0
        fi

        print_header "Dropping existing schema..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "DROP SCHEMA IF EXISTS bot_sessions CASCADE;" > /dev/null
        print_success "Existing schema dropped"
    fi

    echo ""
    print_header "Running initialization script: $INIT_SQL"

    if [ ! -f "$INIT_SQL" ]; then
        print_error "Initialization script not found: $INIT_SQL"
        exit 1
    fi

    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$INIT_SQL"

    echo ""
    print_success "Database initialized successfully!"
    echo ""
    print_header "Created components:"
    echo "  - Schema: bot_sessions"
    echo "  - Tables: 9 tables (sessions, audio_files, transcripts, etc.)"
    echo "  - Indexes: 40+ performance indexes"
    echo "  - Views: 4 pre-computed views"
    echo "  - Functions: Statistics computation and triggers"
    echo "  - Features: Full-text search, speaker tracking, segment continuity"
    echo ""
}

# Function to apply migration
migrate_db() {
    print_header "═══════════════════════════════════════════════════════════"
    print_header "  DATABASE MIGRATION"
    print_header "═══════════════════════════════════════════════════════════"
    echo ""

    if ! check_connection; then
        exit 1
    fi

    if ! check_schema; then
        print_error "Schema 'bot_sessions' does not exist!"
        print_warning "Please run: $0 fresh"
        exit 1
    fi

    echo ""
    print_header "Running migration script: $MIGRATE_SQL"

    if [ ! -f "$MIGRATE_SQL" ]; then
        print_error "Migration script not found: $MIGRATE_SQL"
        exit 1
    fi

    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$MIGRATE_SQL"

    echo ""
    print_success "Migration applied successfully!"
    echo ""
    print_header "Migration added:"
    echo "  - Table: speaker_identities"
    echo "  - Column: transcripts.search_vector"
    echo "  - Column: translations.search_vector"
    echo "  - Columns: segment continuity fields"
    echo "  - Triggers: Automatic search vector updates"
    echo "  - Indexes: Full-text search indexes"
    echo "  - View: Enhanced speaker_statistics"
    echo ""
    print_warning "Migration is idempotent - safe to run multiple times"
    echo ""
}

# Function to show database status
show_status() {
    print_header "═══════════════════════════════════════════════════════════"
    print_header "  DATABASE STATUS"
    print_header "═══════════════════════════════════════════════════════════"
    echo ""

    if ! check_connection; then
        exit 1
    fi

    echo ""
    print_header "Configuration:"
    echo "  Host: $DB_HOST"
    echo "  Port: $DB_PORT"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo ""

    if check_schema; then
        print_header "Tables in bot_sessions schema:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'bot_sessions' ORDER BY tablename;"

        echo ""
        print_header "Row counts:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c \
            "SELECT
                'sessions' as table_name, COUNT(*) as rows FROM bot_sessions.sessions
            UNION ALL
            SELECT 'audio_files', COUNT(*) FROM bot_sessions.audio_files
            UNION ALL
            SELECT 'transcripts', COUNT(*) FROM bot_sessions.transcripts
            UNION ALL
            SELECT 'translations', COUNT(*) FROM bot_sessions.translations
            UNION ALL
            SELECT 'speaker_identities', COUNT(*) FROM bot_sessions.speaker_identities
            ORDER BY table_name;"
    else
        print_warning "Schema 'bot_sessions' not found"
        echo ""
        print_header "To initialize database, run:"
        echo "  $0 fresh"
    fi

    echo ""
}

# Main script
case "${1:-}" in
    fresh)
        init_fresh
        ;;
    migrate)
        migrate_db
        ;;
    status)
        show_status
        ;;
    *)
        echo "Database Initialization Script"
        echo ""
        echo "Usage: $0 [fresh|migrate|status]"
        echo ""
        echo "Commands:"
        echo "  fresh   - Initialize fresh database (WARNING: drops existing data)"
        echo "  migrate - Apply migration to existing database (safe, idempotent)"
        echo "  status  - Show current database status"
        echo ""
        echo "Environment variables:"
        echo "  POSTGRES_HOST     - Database host (default: localhost)"
        echo "  POSTGRES_PORT     - Database port (default: 5432)"
        echo "  POSTGRES_DB       - Database name (default: livetranslate)"
        echo "  POSTGRES_USER     - Database user (default: postgres)"
        echo "  POSTGRES_PASSWORD - Database password (default: livetranslate)"
        echo ""
        echo "Examples:"
        echo "  # Initialize fresh database"
        echo "  $0 fresh"
        echo ""
        echo "  # Apply migration to existing database"
        echo "  $0 migrate"
        echo ""
        echo "  # Show database status"
        echo "  $0 status"
        echo ""
        exit 1
        ;;
esac
