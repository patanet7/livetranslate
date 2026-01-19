#!/usr/bin/env bash
#
# Docker Environment Check Script
#
# Verifies that Docker is properly configured for the LiveTranslate project.
#
# Usage:
#   ./check_docker_env.sh [OPTIONS]
#
# Options:
#   --fix           Attempt to fix common issues
#   --verbose       Show detailed information
#   --help          Show this help message
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Required Docker images
REQUIRED_IMAGES=(
    "postgres:15-alpine"
    "redis:7-alpine"
    "dpage/pgadmin4:latest"
)

# Compose files to validate
COMPOSE_FILES=(
    "docker-compose.database.yml"
    "docker-compose.dev.yml"
    "docker-compose.minimal.yml"
)

# Minimum resource requirements
MIN_MEMORY_GB=4
MIN_CPU_CORES=2
MIN_DISK_GB=10

# Default options
VERBOSE=false
FIX_ISSUES=false

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
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

show_help() {
    cat << EOF
Docker Environment Check Script

Usage:
    ./check_docker_env.sh [OPTIONS]

Options:
    --fix           Attempt to fix common issues (pull missing images)
    --verbose, -v   Show detailed information
    --help, -h      Show this help message

Checks performed:
    - Docker daemon is running
    - Required images exist or can be pulled
    - Docker Compose configuration is valid
    - Memory allocation is sufficient
    - CPU allocation is sufficient
    - Disk space is sufficient

Examples:
    ./check_docker_env.sh                 # Basic check
    ./check_docker_env.sh --verbose       # Detailed output
    ./check_docker_env.sh --fix           # Fix issues automatically

EOF
    exit 0
}

bytes_to_human() {
    local bytes=$1
    if [[ $bytes -ge 1073741824 ]]; then
        echo "$(( bytes / 1073741824 ))GB"
    elif [[ $bytes -ge 1048576 ]]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        echo "${bytes}B"
    fi
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)
            FIX_ISSUES=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
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
# Check Functions
# =============================================================================

check_docker_installed() {
    print_header "Checking Docker Installation..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "    Install Docker Desktop from https://docker.com"
        return 1
    fi

    local docker_version
    docker_version=$(docker --version)
    print_success "Docker installed: ${docker_version}"

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        echo "    Docker Compose should be included with Docker Desktop"
        return 1
    fi

    local compose_version
    if docker compose version &> /dev/null; then
        compose_version=$(docker compose version --short 2>/dev/null || docker compose version)
        print_success "Docker Compose (v2): ${compose_version}"
    else
        compose_version=$(docker-compose --version)
        print_success "Docker Compose (v1): ${compose_version}"
    fi

    return 0
}

check_docker_running() {
    print_header "Checking Docker Daemon..."

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        echo "    Start Docker Desktop or run: sudo systemctl start docker"
        return 1
    fi

    print_success "Docker daemon is running"

    if [[ "${VERBOSE}" == "true" ]]; then
        echo ""
        echo "Docker Info:"
        docker info 2>/dev/null | grep -E "Server Version|Operating System|Total Memory|CPUs|Docker Root Dir" | sed 's/^/    /'
    fi

    return 0
}

check_docker_resources() {
    print_header "Checking Docker Resources..."

    local issues=0

    # Get Docker system info
    local memory_bytes
    local cpus

    # Try to get memory from docker info
    memory_bytes=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
    cpus=$(docker info --format '{{.NCPU}}' 2>/dev/null || echo "0")

    # Check memory
    local memory_gb=$(( memory_bytes / 1073741824 ))
    if [[ $memory_gb -ge $MIN_MEMORY_GB ]]; then
        print_success "Memory: ${memory_gb}GB (minimum: ${MIN_MEMORY_GB}GB)"
    else
        if [[ $memory_gb -gt 0 ]]; then
            print_warning "Memory: ${memory_gb}GB (recommended: ${MIN_MEMORY_GB}GB+)"
            ((issues++))
        else
            print_info "Memory: Unable to determine (running in limited mode?)"
        fi
    fi

    # Check CPUs
    if [[ $cpus -ge $MIN_CPU_CORES ]]; then
        print_success "CPUs: ${cpus} (minimum: ${MIN_CPU_CORES})"
    else
        if [[ $cpus -gt 0 ]]; then
            print_warning "CPUs: ${cpus} (recommended: ${MIN_CPU_CORES}+)"
            ((issues++))
        else
            print_info "CPUs: Unable to determine"
        fi
    fi

    # Check disk space (on Docker root directory)
    local docker_root
    docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "/var/lib/docker")

    if command -v df &> /dev/null; then
        local disk_available_kb
        disk_available_kb=$(df -k "${docker_root}" 2>/dev/null | tail -1 | awk '{print $4}')
        local disk_available_gb=$(( disk_available_kb / 1048576 ))

        if [[ $disk_available_gb -ge $MIN_DISK_GB ]]; then
            print_success "Disk space: ${disk_available_gb}GB available (minimum: ${MIN_DISK_GB}GB)"
        else
            print_warning "Disk space: ${disk_available_gb}GB available (recommended: ${MIN_DISK_GB}GB+)"
            ((issues++))
        fi
    else
        print_info "Disk space: Unable to determine"
    fi

    return $issues
}

check_required_images() {
    print_header "Checking Required Images..."

    local missing_images=()

    for image in "${REQUIRED_IMAGES[@]}"; do
        if docker image inspect "${image}" &> /dev/null; then
            print_success "Image exists: ${image}"
        else
            print_warning "Image missing: ${image}"
            missing_images+=("${image}")
        fi
    done

    if [[ ${#missing_images[@]} -gt 0 ]]; then
        if [[ "${FIX_ISSUES}" == "true" ]]; then
            echo ""
            print_info "Pulling missing images..."
            for image in "${missing_images[@]}"; do
                echo "    Pulling ${image}..."
                if docker pull "${image}"; then
                    print_success "Pulled: ${image}"
                else
                    print_error "Failed to pull: ${image}"
                fi
            done
        else
            echo ""
            print_info "Run with --fix to pull missing images automatically"
            print_info "Or pull manually: docker pull <image>"
        fi
        return 1
    fi

    return 0
}

check_compose_configs() {
    print_header "Checking Docker Compose Configurations..."

    local issues=0

    for compose_file in "${COMPOSE_FILES[@]}"; do
        local full_path="${PROJECT_ROOT}/${compose_file}"

        if [[ ! -f "${full_path}" ]]; then
            if [[ "${VERBOSE}" == "true" ]]; then
                print_info "Compose file not found: ${compose_file} (optional)"
            fi
            continue
        fi

        # Validate compose file
        if docker compose -f "${full_path}" config &> /dev/null; then
            print_success "Valid: ${compose_file}"

            if [[ "${VERBOSE}" == "true" ]]; then
                echo "    Services:"
                docker compose -f "${full_path}" config --services 2>/dev/null | sed 's/^/      - /'
            fi
        else
            print_error "Invalid: ${compose_file}"
            docker compose -f "${full_path}" config 2>&1 | head -5 | sed 's/^/    /'
            ((issues++))
        fi
    done

    return $issues
}

check_network_conflicts() {
    print_header "Checking Network Configuration..."

    local issues=0

    # Check if livetranslate network exists
    if docker network inspect livetranslate-network &> /dev/null; then
        print_success "LiveTranslate network exists"

        if [[ "${VERBOSE}" == "true" ]]; then
            echo "    Subnet: $(docker network inspect livetranslate-network --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}')"
        fi
    else
        print_info "LiveTranslate network will be created on first run"
    fi

    # Check for port conflicts
    local ports_to_check=("5432" "6379" "8080" "3000" "5173" "5001" "5003")

    echo ""
    print_info "Checking common ports..."

    for port in "${ports_to_check[@]}"; do
        if lsof -i ":${port}" -sTCP:LISTEN &> /dev/null; then
            local process
            process=$(lsof -i ":${port}" -sTCP:LISTEN 2>/dev/null | tail -1 | awk '{print $1}')
            print_warning "Port ${port} in use by: ${process}"
            ((issues++)) || true
        else
            if [[ "${VERBOSE}" == "true" ]]; then
                print_success "Port ${port} available"
            fi
        fi
    done

    if [[ $issues -eq 0 ]]; then
        print_success "No port conflicts detected"
    fi

    return 0
}

check_running_containers() {
    print_header "Checking Running Containers..."

    local containers
    containers=$(docker ps --filter "name=livetranslate" --format "{{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null)

    if [[ -n "${containers}" ]]; then
        print_info "LiveTranslate containers running:"
        echo "${containers}" | while IFS=$'\t' read -r name status ports; do
            echo -e "    ${GREEN}${name}${NC}: ${status}"
            if [[ -n "${ports}" ]]; then
                echo "        Ports: ${ports}"
            fi
        done
    else
        print_info "No LiveTranslate containers running"
    fi

    return 0
}

# =============================================================================
# Main Script
# =============================================================================

echo ""
print_header "LiveTranslate Docker Environment Check"
echo "========================================"
echo ""

TOTAL_ISSUES=0

# Run all checks
check_docker_installed || ((TOTAL_ISSUES++))
echo ""

check_docker_running || ((TOTAL_ISSUES++))
echo ""

check_docker_resources || true  # Resources are warnings, not failures
echo ""

check_required_images || ((TOTAL_ISSUES++))
echo ""

check_compose_configs || ((TOTAL_ISSUES++))
echo ""

check_network_conflicts
echo ""

check_running_containers
echo ""

# Summary
print_header "Summary"
echo "========"

if [[ $TOTAL_ISSUES -eq 0 ]]; then
    print_success "All checks passed! Docker environment is ready."
    echo ""
    echo "To start database services:"
    echo "    docker compose -f docker-compose.database.yml up -d"
    echo ""
    echo "To start development environment:"
    echo "    ./start-development.sh"
    exit 0
else
    print_warning "${TOTAL_ISSUES} issue(s) found"
    echo ""
    echo "Fix the issues above and run this script again."
    if [[ "${FIX_ISSUES}" == "false" ]]; then
        echo "Or run with --fix to attempt automatic fixes."
    fi
    exit 1
fi
