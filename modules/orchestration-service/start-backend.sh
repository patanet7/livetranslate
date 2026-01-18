#!/usr/bin/env bash
#
# Backend Service Startup Script
#
# Starts the FastAPI orchestration service with uvicorn.
#
# Usage:
#   ./start-backend.sh [--port PORT] [--reload]
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}"

BACKEND_PORT="${PORT:-3000}"
BACKEND_HOST="${HOST:-0.0.0.0}"
RELOAD="--reload"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

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
Backend Service Startup Script

Usage:
    ./start-backend.sh [OPTIONS]

Options:
    --port PORT     Set the server port (default: 3000)
    --host HOST     Set the server host (default: 0.0.0.0)
    --no-reload     Disable auto-reload
    --help, -h      Show this help message

Environment Variables:
    PORT            Server port (default: 3000)
    HOST            Server host (default: 0.0.0.0)
    LOG_LEVEL       Logging level (default: INFO)

Examples:
    ./start-backend.sh                    # Start with defaults
    ./start-backend.sh --port 8000        # Start on port 8000
    ./start-backend.sh --no-reload        # Start without auto-reload

EOF
    exit 0
}

check_command() {
    local cmd="$1"
    local name="${2:-$1}"

    if command -v "${cmd}" &> /dev/null; then
        local version
        version=$("${cmd}" --version 2>&1 | head -n1) || version="installed"
        print_success "${name}: ${version}"
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --host)
            BACKEND_HOST="$2"
            shift 2
            ;;
        --no-reload)
            RELOAD=""
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

# =============================================================================
# Main Script
# =============================================================================

print_header "Starting Orchestration Service Backend (FastAPI)"
echo "===================================================="

# Navigate to backend service directory
cd "${BACKEND_DIR}"
print_info "Working directory: ${BACKEND_DIR}"

# Check prerequisites
print_header "Checking prerequisites..."

# Check Python
if check_command "python3" "Python"; then
    PYTHON_CMD="python3"
elif check_command "python" "Python"; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.9+ first."
    exit 1
fi

echo ""

# Set up Python environment
print_header "Setting up Python environment..."

# Check for PDM first, then Poetry, then fallback to venv
if command -v pdm &> /dev/null; then
    print_success "Using PDM for dependency management"

    # Install dependencies
    print_info "Installing dependencies with PDM..."
    pdm install --no-self 2>/dev/null || pdm install
    print_success "Dependencies installed"

    # Set the run command
    RUN_PREFIX="pdm run"

elif command -v poetry &> /dev/null; then
    print_warning "PDM not found, using Poetry as fallback"
    print_success "Poetry: $(poetry --version)"

    # Install dependencies
    print_info "Installing dependencies with Poetry..."
    poetry install --no-root
    print_success "Dependencies installed"

    # Set the run command
    RUN_PREFIX="poetry run"

else
    print_warning "Neither PDM nor Poetry found. Using venv..."

    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]] && [[ ! -d ".venv" ]]; then
        print_info "Creating virtual environment..."
        ${PYTHON_CMD} -m venv venv
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    if [[ -d ".venv" ]]; then
        VENV_DIR=".venv"
    else
        VENV_DIR="venv"
    fi

    print_info "Activating virtual environment..."
    source "${VENV_DIR}/bin/activate"
    print_success "Virtual environment activated"

    # Install dependencies
    print_info "Installing dependencies with pip..."
    pip install -r requirements.txt 2>/dev/null || true
    if [[ -f "requirements-database.txt" ]]; then
        pip install -r requirements-database.txt 2>/dev/null || true
    fi
    if [[ -f "requirements-google-meet.txt" ]]; then
        pip install -r requirements-google-meet.txt 2>/dev/null || true
    fi
    print_success "Dependencies installed"

    # Set the run command
    RUN_PREFIX=""
fi

echo ""

# Display service information
BACKEND_URL="http://localhost:${BACKEND_PORT}"

print_header "Service Information:"
echo "  Backend API:     ${BACKEND_URL}"
echo "  API Docs:        ${BACKEND_URL}/docs"
echo "  ReDoc:           ${BACKEND_URL}/redoc"
echo "  Health Check:    ${BACKEND_URL}/api/health"
echo "  Technology:      FastAPI + Python + Async"
echo ""

print_header "Features Available:"
echo "  - RESTful API Endpoints"
echo "  - WebSocket Real-time Communication"
echo "  - Audio Processing API"
echo "  - Bot Management API"
echo "  - System Health Monitoring"
echo "  - Service Coordination"
echo ""

print_header "Frontend Service:"
echo "  Frontend will connect to this backend on port ${BACKEND_PORT}"
echo "  Start frontend: cd ../frontend-service && ./start-frontend.sh"
echo ""

print_info "Starting FastAPI backend..."
echo -e "${YELLOW}Press Ctrl+C to stop the backend service${NC}"
echo ""

# Determine the entry point
if [[ -f "src/main_fastapi.py" ]]; then
    ENTRY_POINT="src.main_fastapi:app"
elif [[ -f "src/main.py" ]]; then
    ENTRY_POINT="src.main:app"
elif [[ -f "backend/main.py" ]]; then
    cd backend
    ENTRY_POINT="main:app"
else
    print_error "No backend entry point found"
    echo "Expected: src/main_fastapi.py, src/main.py, or backend/main.py"
    exit 1
fi

# Set environment variables
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONPATH="${BACKEND_DIR}:${PYTHONPATH:-}"

# Start the server
if [[ -n "${RUN_PREFIX}" ]]; then
    ${RUN_PREFIX} uvicorn "${ENTRY_POINT}" --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" ${RELOAD}
else
    uvicorn "${ENTRY_POINT}" --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" ${RELOAD}
fi
