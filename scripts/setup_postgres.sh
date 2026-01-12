#!/bin/bash
#
# LiveTranslate PostgreSQL Setup Script
# Creates a dedicated PostgreSQL container with both main and test databases
#
# Usage:
#   ./scripts/setup_postgres.sh              # Create container on port 5433
#   ./scripts/setup_postgres.sh --port 5434  # Use custom port
#   ./scripts/setup_postgres.sh --reset      # Remove and recreate container
#

set -e

# Configuration
CONTAINER_NAME="livetranslate-postgres"
POSTGRES_USER="livetranslate"
POSTGRES_PASSWORD="livetranslate_dev_password"
POSTGRES_DB="livetranslate"
POSTGRES_TEST_DB="livetranslate_test"
POSTGRES_PORT=5433
RESET=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            POSTGRES_PORT="$2"
            shift 2
            ;;
        --reset)
            RESET=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT    Use custom port (default: 5433)"
            echo "  --reset        Remove and recreate container"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}LiveTranslate PostgreSQL Setup${NC}"
echo "================================"
echo "Container: $CONTAINER_NAME"
echo "Port: $POSTGRES_PORT"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Reset if requested
if [ "$RESET" = true ]; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}Container already running${NC}"
    else
        echo "Starting existing container..."
        docker start "$CONTAINER_NAME"
    fi
else
    echo "Creating new PostgreSQL container..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e POSTGRES_DB="$POSTGRES_DB" \
        -e POSTGRES_USER="$POSTGRES_USER" \
        -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        -p "${POSTGRES_PORT}:5432" \
        postgres:15-alpine

    echo "Waiting for PostgreSQL to start..."
    sleep 3
fi

# Wait for ready
echo "Checking PostgreSQL is ready..."
for i in {1..30}; do
    if docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" &>/dev/null; then
        echo -e "${GREEN}PostgreSQL is ready${NC}"
        break
    fi
    sleep 1
done

# Create test database
echo "Creating test database..."
docker exec "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "CREATE DATABASE $POSTGRES_TEST_DB;" 2>/dev/null || echo "Test database already exists"

# Run init scripts
if [ -f "$SCRIPT_DIR/init-db.sql" ]; then
    echo "Running init-db.sql on main database..."
    docker exec -i "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$SCRIPT_DIR/init-db.sql" > /dev/null 2>&1

    echo "Running init-db.sql on test database..."
    docker exec -i "$CONTAINER_NAME" psql -U "$POSTGRES_USER" -d "$POSTGRES_TEST_DB" < "$SCRIPT_DIR/init-db.sql" > /dev/null 2>&1

    echo -e "${GREEN}Database schemas initialized${NC}"
fi

# Print connection info
echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo "==============="
echo ""
echo "Connection strings:"
echo "  Main DB:  postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
echo "  Test DB:  postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_TEST_DB}"
echo ""
echo "Environment variables to add to .env:"
echo "  DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
echo "  TEST_DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_TEST_DB}"
echo ""
echo "To connect manually:"
echo "  docker exec -it $CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB"
