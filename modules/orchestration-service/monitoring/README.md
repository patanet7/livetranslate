# Orchestration Service - Integrated Monitoring Stack

## Overview

The Orchestration Service includes a comprehensive monitoring stack that has been consolidated from the standalone monitoring service. This integrated approach provides enterprise-grade observability for the entire LiveTranslate system while maintaining the CPU-optimized focus of the orchestration service.

## Monitoring Components

### Prometheus (Port 9090)
- **Purpose**: Metrics collection and time-series database
- **Configuration**: `monitoring/prometheus/prometheus.yml`
- **Alert Rules**: `monitoring/prometheus/rules/livetranslate-alerts.yml`
- **Features**: 
  - 30-day retention
  - Orchestration-specific metrics
  - Auto-discovery of services
  - 80+ production-ready alert rules

### Grafana (Port 3001)
- **Purpose**: Data visualization and dashboards
- **Configuration**: `monitoring/grafana/provisioning/`
- **Dashboards**: `monitoring/grafana/dashboards/`
- **Features**:
  - Pre-configured datasources (Prometheus, Loki, AlertManager)
  - Orchestration service dashboards
  - Real-time system metrics
  - Alert management integration

### AlertManager (Port 9093)
- **Purpose**: Alert routing and notification management
- **Configuration**: `monitoring/alertmanager/alertmanager.yml`
- **Features**:
  - Smart alert grouping
  - Notification routing
  - Silence management
  - Integration with external systems

### Loki (Port 3100)
- **Purpose**: Log aggregation and analysis
- **Configuration**: `monitoring/loki/loki.yml`
- **Features**:
  - 7-day log retention
  - Structured log parsing
  - Integration with Grafana
  - Efficient storage and querying

### Promtail
- **Purpose**: Log collection and shipping to Loki
- **Configuration**: `monitoring/loki/promtail.yml`
- **Features**:
  - Container log collection
  - Service-specific log parsing
  - Automatic labeling
  - Real-time log streaming

## Deployment

### Integrated Deployment (Recommended)
```bash
# Deploy orchestration service with monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### Individual Service Access
```bash
# Orchestration Service
http://localhost:3000

# Grafana Dashboards
http://localhost:3001
# Default login: admin/livetranslate2023

# Prometheus
http://localhost:9090

# AlertManager
http://localhost:9093

# Loki
http://localhost:3100
```

## Service Integration

### Metrics Endpoints
The monitoring stack automatically discovers and scrapes metrics from:
- **Orchestration Service**: `http://orchestration-service:3000/api/metrics`
- **Audio Service**: `http://audio-service:5001/api/metrics`
- **Translation Service**: `http://translation-service:5003/api/metrics`

### Log Collection
Logs are automatically collected from:
- Container logs (Docker runtime)
- Service-specific log files in `/app/logs/`
- System logs from `/var/log/`

### Health Monitoring
All services are monitored for:
- Service availability (up/down status)
- Response time and latency
- Error rates and failure patterns
- Resource utilization (CPU, memory, disk)
- Custom business metrics

## Alert Configuration

### Critical Alerts
- Service downtime (>30 seconds)
- High error rates (>10%)
- Resource exhaustion (CPU >80%, Memory >85%, Disk >90%)
- WebSocket connection overload (>9500 connections)

### Warning Alerts
- Performance degradation
- High latency (>1 second for API Gateway)
- Resource pressure
- Quality score degradation

### Custom Orchestration Alerts
- WebSocket connection limits
- API Gateway circuit breaker status
- Service health monitor failures
- Session management issues

## Configuration Management

### Environment Variables
```bash
# Monitoring configuration
PROMETHEUS_METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=10
LOG_FORMAT=json
LOG_FILE=/app/logs/orchestration.log

# Grafana configuration
GF_SECURITY_ADMIN_PASSWORD=livetranslate2023
GF_USERS_ALLOW_SIGN_UP=false
```

### Monitoring Labels
All services are automatically labeled for:
- `service`: Service type (orchestration, audio, translation)
- `environment`: Deployment environment
- `version`: Service version
- `cluster`: LiveTranslate cluster identifier

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Services
```bash
# Check service discovery
curl http://localhost:9090/api/v1/targets

# Verify service health endpoints
curl http://localhost:3000/api/health
curl http://localhost:5001/api/health
curl http://localhost:5003/api/health
```

#### Grafana Datasource Issues
```bash
# Check datasource connectivity
curl http://localhost:3001/api/datasources

# Test Prometheus connection
curl http://localhost:9090/-/healthy

# Test Loki connection
curl http://localhost:3100/ready
```

#### Log Collection Problems
```bash
# Check Promtail status
curl http://localhost:9080/metrics

# Verify log file permissions
ls -la /app/logs/

# Check Loki ingestion
curl http://localhost:3100/loki/api/v1/labels
```

### Performance Optimization

#### Prometheus Storage
- Default retention: 30 days
- Configurable in `docker-compose.monitoring.yml`
- Monitor disk usage for `/prometheus` volume

#### Loki Storage
- Default retention: 7 days
- Configurable in `monitoring/loki/loki.yml`
- Automatic compaction and cleanup

#### Resource Allocation
```yaml
# Recommended resource limits
prometheus:
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1'

grafana:
  deploy:
    resources:
      limits:
        memory: 512M
        cpus: '0.5'

loki:
  deploy:
    resources:
      limits:
        memory: 1G
        cpus: '0.5'
```

## Maintenance

### Backup Strategy
```bash
# Backup Prometheus data
docker run --rm -v prometheus-data:/data busybox tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /data

# Backup Grafana dashboards
docker run --rm -v grafana-data:/data busybox tar czf /backup/grafana-$(date +%Y%m%d).tar.gz /data

# Backup alert rules
tar czf /backup/alert-rules-$(date +%Y%m%d).tar.gz monitoring/prometheus/rules/
```

### Updates and Maintenance
```bash
# Update monitoring stack
docker-compose -f docker-compose.monitoring.yml pull
docker-compose -f docker-compose.monitoring.yml up -d

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload

# Reload AlertManager configuration
curl -X POST http://localhost:9093/-/reload
```

This integrated monitoring stack provides comprehensive observability for the LiveTranslate orchestration service while maintaining the performance and efficiency requirements of the CPU-optimized architecture.
