#!/bin/bash
set -euo pipefail

# LiveTranslate Database Startup Script (Linux/macOS)
# Usage: ./start-database.sh [mode] [options]

# Default values
MODE="${1:-dev}"
CLEAN=false
SHOW_LOGS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [dev|prod|test] [--clean] [--logs] [--help]"
            echo ""
            echo "Options:"
            echo "  dev|prod|test  Environment mode (default: dev)"
            echo "  --clean        Clean existing data volumes"
            echo "  --logs         Show real-time logs after startup"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 dev"
            echo "  $0 prod --clean"
            echo "  $0 dev --logs"
            exit 0
            ;;
        dev|prod|test)
            MODE="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}$1${NC}"
}

log_warning() {
    echo -e "${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${RED}$1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "❌ Docker not found. Please install Docker."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "❌ Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
}

# Test if database is running
test_database() {
    docker exec livetranslate-postgres pg_isready -U livetranslate -d livetranslate &> /dev/null
}

# Wait for database to be ready
wait_for_database() {
    local max_attempts=30
    local delay=2

    log_warning "🔄 Waiting for database to be ready..."

    for ((i=1; i<=max_attempts; i++)); do
        if test_database; then
            log_success "✅ Database is ready!"
            return 0
        fi

        echo "   Attempt $i/$max_attempts - waiting..."
        sleep $delay
    done

    log_error "❌ Database failed to start within expected time"
    return 1
}

# Show database information
show_database_info() {
    log_info "\n📊 Database Service Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check container status
    local containers=("livetranslate-postgres" "livetranslate-redis" "livetranslate-pgadmin")

    for container in "${containers[@]}"; do
        if docker ps --filter "name=$container" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "$container"; then
            docker ps --filter "name=$container" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep "$container"
        else
            log_error "❌ $container: Not running"
        fi
    done

    echo ""
    echo "🔗 Connection Information:"
    echo "   PostgreSQL: localhost:5432"
    echo "   Database: livetranslate"
    echo "   Username: livetranslate"
    echo "   Redis: localhost:6379"
    echo "   pgAdmin: http://localhost:8080"
    echo ""
}

# Show useful commands
show_useful_commands() {
    log_info "\n🛠️  Useful Commands"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Check logs:        docker-compose -f docker-compose.database.yml logs -f"
    echo "   Connect to DB:     docker exec -it livetranslate-postgres psql -U livetranslate -d livetranslate"
    echo "   Connect to Redis:  docker exec -it livetranslate-redis redis-cli"
    echo "   Stop services:     docker-compose -f docker-compose.database.yml down"
    echo "   Clean volumes:     docker-compose -f docker-compose.database.yml down -v"
    echo "   Restart service:   docker-compose -f docker-compose.database.yml restart postgres"
    echo ""
}

# Create configuration files
create_config_files() {
    # Create directories
    mkdir -p docker/postgres docker/redis docker/pgadmin

    # PostgreSQL configuration
    cat > docker/postgres/postgresql.conf << 'EOF'
# PostgreSQL Configuration for LiveTranslate
listen_addresses = '*'
port = 5432
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 4
effective_io_concurrency = 2
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB

# Logging
log_statement = 'all'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Performance monitoring
shared_preload_libraries = 'pg_stat_statements'
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all
EOF

    # pg_hba.conf
    cat > docker/postgres/pg_hba.conf << 'EOF'
# PostgreSQL Client Authentication Configuration
local   all             all                                     scram-sha-256
host    all             all             127.0.0.1/32            scram-sha-256
host    all             all             ::1/128                 scram-sha-256
host    all             all             172.20.0.0/16           scram-sha-256
host    all             all             0.0.0.0/0               scram-sha-256
EOF

    # Redis configuration
    cat > docker/redis/redis.conf << 'EOF'
# Redis Configuration for LiveTranslate
bind 0.0.0.0
port 6379
protected-mode no
timeout 300
tcp-keepalive 60
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
EOF

    # pgAdmin servers configuration
    cat > docker/pgadmin/servers.json << 'EOF'
{
    "Servers": {
        "1": {
            "Name": "LiveTranslate Local",
            "Group": "Servers",
            "Host": "postgres",
            "Port": 5432,
            "MaintenanceDB": "livetranslate",
            "Username": "livetranslate",
            "SSLMode": "prefer",
            "Comment": "LiveTranslate PostgreSQL Database"
        }
    }
}
EOF
}

# Main execution
main() {
    log_success "🚀 Starting LiveTranslate Database Services"
    log_info "Mode: $MODE"

    # Check Docker installation
    check_docker

    # Navigate to project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_ROOT"

    log_info "📁 Working directory: $PROJECT_ROOT"

    # Create configuration files
    create_config_files

    # Set environment variables based on mode
    if [[ "$MODE" == "prod" ]]; then
        export POSTGRES_PASSWORD="secure_production_password_change_me"
        export PGADMIN_PASSWORD="secure_admin_password"
    else
        export POSTGRES_PASSWORD="livetranslate_dev_password"
        export PGADMIN_PASSWORD="admin"
    fi

    export PGADMIN_EMAIL="admin@livetranslate.local"

    # Clean volumes if requested
    if [[ "$CLEAN" == true ]]; then
        log_warning "🧹 Cleaning existing data volumes..."
        docker-compose -f docker-compose.database.yml down -v --remove-orphans 2>/dev/null || true

        # Remove named volumes explicitly
        local volumes=("livetranslate_postgres_data" "livetranslate_redis_data" "livetranslate_pgadmin_data")
        for volume in "${volumes[@]}"; do
            docker volume rm "$volume" 2>/dev/null || true
            log_warning "   Removed volume: $volume"
        done
    fi

    # Start services
    log_warning "🐳 Starting Docker containers..."
    if docker-compose -f docker-compose.database.yml up -d --remove-orphans; then
        log_success "✅ Docker containers started successfully"
    else
        log_error "❌ Failed to start Docker containers"
        exit 1
    fi

    # Wait for database to be ready
    if ! wait_for_database; then
        log_error "❌ Database startup failed"
        log_warning "💡 Try running with --clean flag to reset data"
        exit 1
    fi

    # Give services a moment to fully initialize
    sleep 5

    # Show service information
    show_database_info
    show_useful_commands

    log_success "🎉 Database services started successfully!"
    log_info "💡 Use pgAdmin at http://localhost:8080 to manage the database"

    # Show logs if requested
    if [[ "$SHOW_LOGS" == true ]]; then
        log_info "\n📋 Showing real-time logs (Ctrl+C to exit)..."
        docker-compose -f docker-compose.database.yml logs -f
    fi

    log_success "\n✅ Database startup complete!"
}

# Run main function
main "$@"
