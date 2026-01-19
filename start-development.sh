#!/usr/bin/env bash
#
# LiveTranslate Development Environment Startup Script
#
# This script starts all services needed for development:
# - Frontend Service (React + Vite) on port 5173
# - Orchestration Service (FastAPI) on port 3000
#
# Usage:
#   ./start-development.sh [OPTIONS]
#
# Options:
#   --no-docker       Skip Docker services (PostgreSQL, Redis)
#   --service=NAME    Start only specific service (frontend, backend, all)
#   --help            Show this help message
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

FRONTEND_PORT=5173
BACKEND_PORT=3000
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"

# Default options
SKIP_DOCKER=false
SERVICE="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${CYAN}${BOLD}$1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
LiveTranslate Development Environment Startup Script

Usage:
    ./start-development.sh [OPTIONS]

Options:
    --no-docker       Skip Docker services (PostgreSQL, Redis)
    --service=NAME    Start only specific service:
                        frontend  - Start only frontend service
                        backend   - Start only backend service
                        all       - Start all services (default)
    --help, -h        Show this help message

Examples:
    ./start-development.sh                      # Start all services with Docker
    ./start-development.sh --no-docker          # Start without Docker services
    ./start-development.sh --service=frontend   # Start only frontend
    ./start-development.sh --service=backend    # Start only backend

Service URLs:
    Frontend:     http://localhost:5173
    Backend API:  http://localhost:3000
    API Docs:     http://localhost:3000/docs
    Health Check: http://localhost:3000/api/health

EOF
    exit 0
}

cleanup() {
    echo ""
    print_header "Stopping all services..."

    if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        print_info "Stopping backend service (PID: ${BACKEND_PID})..."
        kill "${BACKEND_PID}" 2>/dev/null || true
        wait "${BACKEND_PID}" 2>/dev/null || true
    fi

    if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        print_info "Stopping frontend service (PID: ${FRONTEND_PID})..."
        kill "${FRONTEND_PID}" 2>/dev/null || true
        wait "${FRONTEND_PID}" 2>/dev/null || true
    fi

    print_success "All services stopped"
    echo ""
    echo -e "${CYAN}Thank you for using LiveTranslate!${NC}"
    exit 0
}

check_command() {
    local cmd="$1"
    local name="${2:-$1}"
    local install_hint="${3:-}"

    if command -v "${cmd}" &> /dev/null; then
        local version
        version=$("${cmd}" --version 2>&1 | head -n1) || version="installed"
        print_success "${name}: ${version}"
        return 0
    else
        print_error "${name} not found."
        if [[ -n "${install_hint}" ]]; then
            echo -e "    ${install_hint}"
        fi
        return 1
    fi
}

check_port_available() {
    local port="$1"
    if lsof -Pi ":${port}" -sTCP:LISTEN -t &>/dev/null; then
        return 1
    fi
    return 0
}

wait_for_service() {
    local url="$1"
    local name="$2"
    local max_attempts="${3:-30}"
    local delay="${4:-2}"

    print_info "Waiting for ${name} to be ready..."

    for ((i=1; i<=max_attempts; i++)); do
        if curl -s -o /dev/null -w "%{http_code}" "${url}" | grep -q "^[23]"; then
            print_success "${name} is ready!"
            return 0
        fi
        echo -n "."
        sleep "${delay}"
    done

    echo ""
    print_warning "${name} health check failed, but continuing..."
    return 1
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --service=*)
            SERVICE="${1#*=}"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate service option
case "${SERVICE}" in
    frontend|backend|all)
        ;;
    *)
        print_error "Invalid service: ${SERVICE}"
        echo "Valid options: frontend, backend, all"
        exit 1
        ;;
esac

# =============================================================================
# Main Script
# =============================================================================

# Set up cleanup trap
trap cleanup SIGINT SIGTERM EXIT

print_header "Starting LiveTranslate Development Environment"
echo "=================================================="

# Check if we're in the right directory
if [[ ! -d "${PROJECT_ROOT}/modules" ]]; then
    print_error "Please run this script from the project root directory"
    echo "Expected to find 'modules' directory"
    exit 1
fi

# Display architecture overview
echo ""
print_header "Service Architecture:"
echo "------------------------------------------------------------"
echo "  Frontend Service (React + Vite)"
echo "    Port: ${FRONTEND_PORT}"
echo "    Features: Audio testing, Bot management, Dashboard"
echo "    Proxy: API calls -> Backend (Port ${BACKEND_PORT})"
echo ""
echo "  Backend Service (FastAPI + Python)"
echo "    Port: ${BACKEND_PORT}"
echo "    Features: API endpoints, WebSocket, Service coordination"
echo "    Connections: Whisper service, Translation service"
echo "------------------------------------------------------------"
echo ""

# Service URLs
print_header "Service URLs:"
echo "  Frontend:     ${FRONTEND_URL}"
echo "  Backend API:  ${BACKEND_URL}"
echo "  API Docs:     ${BACKEND_URL}/docs"
echo "  Health Check: ${BACKEND_URL}/api/health"
echo ""

# Check prerequisites
print_header "Checking prerequisites..."

PREREQ_FAILED=false

# Check Node.js
if ! check_command "node" "Node.js" "Install from https://nodejs.org/"; then
    PREREQ_FAILED=true
fi

# Check PDM (for Python dependency management)
if ! check_command "pdm" "PDM" "Install with: pip install pdm"; then
    # Fallback to Poetry if PDM not available
    if check_command "poetry" "Poetry (fallback)" "Install from https://python-poetry.org/"; then
        print_warning "Using Poetry as fallback. Consider installing PDM: pip install pdm"
    else
        PREREQ_FAILED=true
    fi
fi

# Check pnpm
if ! command -v pnpm &> /dev/null; then
    print_warning "pnpm not found. Installing pnpm..."
    npm install -g pnpm
    print_success "pnpm installed: $(pnpm --version)"
else
    print_success "pnpm: $(pnpm --version)"
fi

# Check Docker (if not skipping)
if [[ "${SKIP_DOCKER}" == "false" ]]; then
    if ! check_command "docker" "Docker" "Install Docker Desktop from https://docker.com"; then
        print_warning "Docker not found. Use --no-docker to skip Docker services"
    fi
fi

if [[ "${PREREQ_FAILED}" == "true" ]]; then
    print_error "Some prerequisites are missing. Please install them and try again."
    exit 1
fi

echo ""

# Check port availability
print_header "Checking port availability..."

if [[ "${SERVICE}" == "backend" ]] || [[ "${SERVICE}" == "all" ]]; then
    if ! check_port_available "${BACKEND_PORT}"; then
        print_error "Port ${BACKEND_PORT} is already in use"
        echo "    Run: lsof -i :${BACKEND_PORT} to see what's using it"
        exit 1
    fi
    print_success "Port ${BACKEND_PORT} is available"
fi

if [[ "${SERVICE}" == "frontend" ]] || [[ "${SERVICE}" == "all" ]]; then
    if ! check_port_available "${FRONTEND_PORT}"; then
        print_error "Port ${FRONTEND_PORT} is already in use"
        echo "    Run: lsof -i :${FRONTEND_PORT} to see what's using it"
        exit 1
    fi
    print_success "Port ${FRONTEND_PORT} is available"
fi

echo ""

# Start Docker services if not skipped
if [[ "${SKIP_DOCKER}" == "false" ]] && [[ "${SERVICE}" != "frontend" ]]; then
    if command -v docker &> /dev/null && [[ -f "${PROJECT_ROOT}/docker-compose.database.yml" ]]; then
        print_header "Starting Docker services..."

        # Create required directories
        mkdir -p "${PROJECT_ROOT}/docker/postgres"
        mkdir -p "${PROJECT_ROOT}/docker/redis"
        mkdir -p "${PROJECT_ROOT}/docker/pgadmin"

        # Start database services
        if docker compose -f "${PROJECT_ROOT}/docker-compose.database.yml" up -d; then
            print_success "Docker services started"

            # Wait for PostgreSQL to be ready
            print_info "Waiting for PostgreSQL to be ready..."
            for i in {1..30}; do
                if docker exec livetranslate-postgres pg_isready -U livetranslate -d livetranslate &>/dev/null; then
                    print_success "PostgreSQL is ready"
                    break
                fi
                sleep 1
            done
        else
            print_warning "Failed to start Docker services. Continuing without database..."
        fi
        echo ""
    fi
fi

# Start backend service
if [[ "${SERVICE}" == "backend" ]] || [[ "${SERVICE}" == "all" ]]; then
    print_header "Starting backend service..."

    BACKEND_DIR="${PROJECT_ROOT}/modules/orchestration-service"

    if [[ ! -d "${BACKEND_DIR}" ]]; then
        print_error "Backend directory not found: ${BACKEND_DIR}"
        exit 1
    fi

    cd "${BACKEND_DIR}"

    # Install dependencies
    print_info "Installing backend dependencies..."
    if command -v pdm &> /dev/null; then
        pdm install --no-self 2>/dev/null || pdm install
    elif command -v poetry &> /dev/null; then
        poetry install --no-root
    else
        print_error "Neither PDM nor Poetry found for dependency management"
        exit 1
    fi

    # Start backend server
    print_info "Starting uvicorn server..."
    if command -v pdm &> /dev/null; then
        pdm run uvicorn src.main_fastapi:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload &
    elif command -v poetry &> /dev/null; then
        poetry run uvicorn src.main_fastapi:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload &
    fi
    BACKEND_PID=$!

    cd "${PROJECT_ROOT}"

    # Wait for backend to be ready
    sleep 3
    wait_for_service "${BACKEND_URL}/api/health" "Backend" 15 2 || true

    echo ""
fi

# Install and start frontend
if [[ "${SERVICE}" == "frontend" ]] || [[ "${SERVICE}" == "all" ]]; then
    print_header "Starting frontend service..."

    FRONTEND_DIR="${PROJECT_ROOT}/modules/frontend-service"

    if [[ ! -d "${FRONTEND_DIR}" ]]; then
        print_error "Frontend directory not found: ${FRONTEND_DIR}"
        exit 1
    fi

    cd "${FRONTEND_DIR}"

    # Install dependencies
    if [[ ! -d "node_modules" ]]; then
        print_info "Installing frontend dependencies..."
        pnpm install
        print_success "Frontend dependencies installed"
    else
        print_success "Frontend dependencies already installed"
    fi

    echo ""
    print_header "Development Environment Ready!"
    echo ""
    echo -e "${CYAN}Available Services:${NC}"
    echo "  Frontend:      ${FRONTEND_URL}"
    echo "  Backend API:   ${BACKEND_URL}"
    echo "  API Docs:      ${BACKEND_URL}/docs"
    echo "  Health Check:  ${BACKEND_URL}/api/health"
    echo ""
    echo -e "${CYAN}Features:${NC}"
    echo "  - Audio Testing & Recording Interface"
    echo "  - Bot Management & Analytics Dashboard"
    echo "  - Real-time System Monitoring"
    echo "  - Settings & Configuration Management"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    echo ""

    # Start frontend (this will block)
    pnpm dev &
    FRONTEND_PID=$!

    # Wait for processes
    wait
fi

# If only backend was started, keep running
if [[ "${SERVICE}" == "backend" ]]; then
    echo ""
    print_header "Backend Service Running"
    echo ""
    echo "  Backend API:   ${BACKEND_URL}"
    echo "  API Docs:      ${BACKEND_URL}/docs"
    echo "  Health Check:  ${BACKEND_URL}/api/health"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
    echo ""

    wait "${BACKEND_PID}"
fi
