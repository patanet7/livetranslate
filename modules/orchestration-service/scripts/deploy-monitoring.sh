#!/bin/bash

# LiveTranslate Orchestration Service - Monitoring Stack Deployment Script
# Deploys the integrated monitoring infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.monitoring.yml"
MONITORING_DIR="$PROJECT_ROOT/monitoring"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."

    # Create log directory
    mkdir -p "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/logs"

    # Create config directory
    mkdir -p "$PROJECT_ROOT/config"

    log_success "Directories created"
}

# Check configuration files
check_configuration() {
    log_info "Checking monitoring configuration..."

    local config_files=(
        "$MONITORING_DIR/prometheus/prometheus.yml"
        "$MONITORING_DIR/prometheus/rules/livetranslate-alerts.yml"
        "$MONITORING_DIR/alertmanager/alertmanager.yml"
        "$MONITORING_DIR/grafana/provisioning/datasources/datasources.yml"
        "$MONITORING_DIR/loki/loki.yml"
        "$MONITORING_DIR/loki/promtail.yml"
    )

    for config_file in "${config_files[@]}"; do
        if [[ ! -f "$config_file" ]]; then
            log_error "Missing configuration file: $config_file"
            exit 1
        fi
    done

    log_success "Configuration files verified"
}

# Create Docker networks
create_networks() {
    log_info "Creating Docker networks..."

    # Create livetranslate network if it doesn't exist
    if ! docker network ls | grep -q "livetranslate"; then
        docker network create livetranslate
        log_success "Created livetranslate network"
    else
        log_info "livetranslate network already exists"
    fi

    # Create monitoring network if it doesn't exist
    if ! docker network ls | grep -q "monitoring"; then
        docker network create monitoring
        log_success "Created monitoring network"
    else
        log_info "monitoring network already exists"
    fi
}

# Pull Docker images
pull_images() {
    log_info "Pulling Docker images..."

    docker-compose -f "$COMPOSE_FILE" pull

    log_success "Docker images pulled"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."

    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d

    log_success "Monitoring stack deployed"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to become healthy..."

    local services=("prometheus" "grafana" "loki" "alertmanager")
    local max_wait=120
    local wait_time=0

    for service in "${services[@]}"; do
        log_info "Waiting for $service to be ready..."

        while [[ $wait_time -lt $max_wait ]]; do
            if docker-compose -f "$COMPOSE_FILE" ps | grep -q "$service.*healthy\|Up"; then
                log_success "$service is ready"
                break
            fi

            sleep 5
            wait_time=$((wait_time + 5))

            if [[ $wait_time -ge $max_wait ]]; then
                log_warning "$service is taking longer than expected to start"
                break
            fi
        done
    done
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    # Check service endpoints
    local endpoints=(
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3001/api/health:Grafana"
        "http://localhost:3100/ready:Loki"
        "http://localhost:9093/-/healthy:AlertManager"
    )

    for endpoint in "${endpoints[@]}"; do
        IFS=':' read -r url name <<< "$endpoint"

        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "$name is responding"
        else
            log_warning "$name is not responding at $url"
        fi
    done
}

# Show access information
show_access_info() {
    log_info "Deployment completed! Access information:"
    echo ""
    echo -e "${GREEN}Orchestration Service:${NC} http://localhost:3000"
    echo -e "${GREEN}Grafana Dashboards:${NC}   http://localhost:3001 (admin/livetranslate2023)"
    echo -e "${GREEN}Prometheus:${NC}           http://localhost:9090"
    echo -e "${GREEN}AlertManager:${NC}         http://localhost:9093"
    echo -e "${GREEN}Loki:${NC}                 http://localhost:3100"
    echo ""
    echo -e "${YELLOW}Logs:${NC} docker-compose -f $COMPOSE_FILE logs -f"
    echo -e "${YELLOW}Stop:${NC} docker-compose -f $COMPOSE_FILE down"
}

# Cleanup function
cleanup() {
    log_info "Stopping services..."
    docker-compose -f "$COMPOSE_FILE" down
    log_success "Services stopped"
}

# Main deployment function
main() {
    local action="${1:-deploy}"

    case "$action" in
        "deploy")
            log_info "Starting LiveTranslate Orchestration Service monitoring deployment..."
            check_prerequisites
            create_directories
            check_configuration
            create_networks
            pull_images
            deploy_monitoring
            wait_for_services
            verify_deployment
            show_access_info
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            sleep 5
            main deploy
            ;;
        "status")
            docker-compose -f "$COMPOSE_FILE" ps
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" logs -f
            ;;
        *)
            echo "Usage: $0 {deploy|stop|restart|status|logs}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the monitoring stack"
            echo "  stop     - Stop all monitoring services"
            echo "  restart  - Stop and restart all services"
            echo "  status   - Show service status"
            echo "  logs     - Show service logs"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap cleanup INT TERM

# Run main function
main "$@"
