#!/bin/bash
#
# Database Setup Script for Orchestration Service
#
# This script provides a simple wrapper around setup_database.py
# with convenient defaults for development.
#
# Usage:
#   ./scripts/setup_database.sh                    # Setup with default local DB
#   ./scripts/setup_database.sh --reset            # Reset database
#   ./scripts/setup_database.sh --status           # Show status only
#   ./scripts/setup_database.sh --create-test-data # Create test data
#
# Environment Variables:
#   DATABASE_URL - PostgreSQL connection string (default: local dev instance)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default database URL for local development
# Override with DATABASE_URL environment variable
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="postgresql://livetranslate:livetranslate_dev_password@localhost:5433/livetranslate"
    echo "Using default DATABASE_URL for local development"
    echo "Set DATABASE_URL environment variable to use a different database"
fi

# Change to project directory
cd "$PROJECT_DIR"

echo ""
echo "============================================================"
echo "DATABASE SETUP - ORCHESTRATION SERVICE"
echo "============================================================"
echo ""
echo "Database: ${DATABASE_URL##*@}"  # Only show host/db part
echo ""

# Check if PDM is available
if ! command -v pdm &> /dev/null; then
    echo "ERROR: PDM is not installed. Please install it first:"
    echo "  pip install pdm"
    exit 1
fi

# Run the Python setup script with all arguments passed through
pdm run python scripts/setup_database.py "$@"
