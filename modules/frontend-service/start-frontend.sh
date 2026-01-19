#!/usr/bin/env bash
#
# Frontend Service Startup Script
#
# Starts the React frontend development server with Vite.
#
# Usage:
#   ./start-frontend.sh
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${SCRIPT_DIR}"

FRONTEND_PORT="${VITE_PORT:-5173}"
BACKEND_PORT="${BACKEND_PORT:-3000}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
BACKEND_URL="http://localhost:${BACKEND_PORT}"

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

check_command() {
    local cmd="$1"
    local name="${2:-$1}"

    if command -v "${cmd}" &> /dev/null; then
        local version
        version=$("${cmd}" --version 2>&1 | head -n1) || version="installed"
        print_success "${name}: ${version}"
        return 0
    else
        print_error "${name} not found"
        return 1
    fi
}

# =============================================================================
# Main Script
# =============================================================================

print_header "Starting Frontend Service (React + Vite)"
echo "============================================="

# Navigate to frontend service directory
cd "${FRONTEND_DIR}"
print_info "Working directory: ${FRONTEND_DIR}"

# Check prerequisites
print_header "Checking prerequisites..."

if ! check_command "node" "Node.js"; then
    print_error "Node.js not found. Please install Node.js 18+ first."
    exit 1
fi

# Check pnpm
if ! command -v pnpm &> /dev/null; then
    print_warning "pnpm not found. Installing pnpm..."
    npm install -g pnpm
    print_success "pnpm installed: $(pnpm --version)"
else
    print_success "pnpm: $(pnpm --version)"
fi

echo ""

# Install dependencies
print_header "Installing frontend dependencies..."

if [[ ! -d "node_modules" ]]; then
    pnpm install
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

echo ""

# Display service information
print_header "Service Information:"
echo "  Frontend URL: ${FRONTEND_URL}"
echo "  Backend API:  ${BACKEND_URL}"
echo "  Technology:   React 18 + TypeScript + Vite"
echo ""

print_header "Features Available:"
echo "  - Audio Testing Interface"
echo "  - Bot Management Dashboard"
echo "  - Real-time System Monitoring"
echo "  - Settings & Configuration"
echo ""

print_warning "Note: Backend must be running on port ${BACKEND_PORT}"
echo "  Run backend with: cd ../orchestration-service && ./start-backend.sh"
echo ""

print_info "Starting frontend development server..."
echo -e "${YELLOW}Press Ctrl+C to stop the frontend service${NC}"
echo ""

# Set environment variables
export VITE_API_BASE_URL="${BACKEND_URL}"
export VITE_WS_BASE_URL="ws://localhost:${BACKEND_PORT}"

# Start frontend development server
pnpm dev
