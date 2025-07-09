# Grafana Dashboards

This directory contains pre-configured Grafana dashboards for the LiveTranslate monitoring system.

## Available Dashboards

### System Dashboards
- **system-overview.json** - Overall system health and resource usage
- **docker-containers.json** - Container metrics and health
- **infrastructure.json** - Redis, networking, and storage metrics

### Application Dashboards
- **livetranslate-overview.json** - High-level application metrics
- **whisper-service.json** - Transcription service metrics
- **translation-service.json** - Translation service metrics
- **speaker-service.json** - Speaker diarization metrics
- **postprocessing-service.json** - Post-processing service metrics
- **frontend-service.json** - Frontend application metrics

### Business Dashboards
- **translation-quality.json** - Translation accuracy and quality metrics
- **session-analytics.json** - User session and engagement metrics
- **performance-overview.json** - End-to-end performance metrics

## Dashboard Installation

Dashboards are automatically provisioned when the monitoring stack starts. To add new dashboards:

1. Place the JSON file in this directory
2. Restart the Grafana container
3. The dashboard will be automatically imported

## Dashboard Development

To create new dashboards:

1. Create the dashboard in Grafana UI
2. Export the dashboard JSON
3. Place the JSON file in this directory
4. Commit to version control

## Variables and Templating

Most dashboards use these standard variables:
- `$datasource` - Prometheus datasource
- `$service` - Service name filter
- `$instance` - Instance/container filter
- `$interval` - Time interval for queries

## Alerts Integration

Dashboards include alert panels that show:
- Current firing alerts
- Alert history
- Alert rule status
- Notification status 