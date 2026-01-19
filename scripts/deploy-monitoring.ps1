#!/usr/bin/env pwsh
# LiveTranslate Monitoring Stack Deployment Script
# This script deploys the complete monitoring infrastructure

param(
    [switch]$Force,
    [switch]$SkipValidation,
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"

Write-Host "LiveTranslate Monitoring Stack Deployment" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if monitoring directory exists
if (-not (Test-Path "monitoring")) {
    Write-Host "[ERROR] Monitoring directory not found. Please run from project root." -ForegroundColor Red
    exit 1
}

# Validate configuration files
if (-not $SkipValidation) {
    Write-Host "Validating configuration files..." -ForegroundColor Yellow

    $configFiles = @(
        "monitoring/docker-compose.monitoring.yml",
        "monitoring/prometheus/prometheus.yml",
        "monitoring/loki/loki-config.yml",
        "monitoring/promtail/promtail-config.yml",
        "monitoring/alertmanager/alertmanager.yml",
        "monitoring/prometheus/rules/livetranslate-alerts.yml",
        "monitoring/grafana/provisioning/datasources/datasources.yml"
    )

    foreach ($file in $configFiles) {
        if (-not (Test-Path $file)) {
            Write-Host "[ERROR] Missing configuration file: $file" -ForegroundColor Red
            exit 1
        }
        Write-Host "[OK] Found: $file" -ForegroundColor Green
    }
}

# Create required directories
Write-Host "Creating required directories..." -ForegroundColor Yellow
$directories = @(
    "monitoring/grafana/data",
    "monitoring/prometheus/data",
    "monitoring/loki/data",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[OK] Created directory: $dir" -ForegroundColor Green
    }
}

# Set proper permissions for Grafana
if ($IsLinux -or $IsMacOS) {
    Write-Host "Setting Grafana permissions..." -ForegroundColor Yellow
    sudo chown -R 472:472 monitoring/grafana/data
}

# Stop existing monitoring stack if running
Write-Host "Stopping existing monitoring stack..." -ForegroundColor Yellow
try {
    docker-compose -f monitoring/docker-compose.monitoring.yml down --remove-orphans 2>$null
    Write-Host "[OK] Stopped existing stack" -ForegroundColor Green
} catch {
    Write-Host "[INFO] No existing stack to stop" -ForegroundColor Blue
}

# Pull latest images
Write-Host "Pulling latest monitoring images..." -ForegroundColor Yellow
docker-compose -f monitoring/docker-compose.monitoring.yml pull

# Deploy monitoring stack
Write-Host "Deploying monitoring stack..." -ForegroundColor Yellow
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Wait for services to be ready
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Health check function
function Test-ServiceHealth {
    param($ServiceName, $Url, $ExpectedStatus = 200)

    try {
        $response = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq $ExpectedStatus) {
            Write-Host "[OK] $ServiceName is healthy" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARNING] $ServiceName returned status $($response.StatusCode)" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "[ERROR] $ServiceName is not responding: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Check service health
Write-Host "Checking service health..." -ForegroundColor Yellow
$services = @(
    @{ Name = "Prometheus"; Url = "http://localhost:9090/-/healthy" },
    @{ Name = "Grafana"; Url = "http://localhost:3001/api/health" },
    @{ Name = "Loki"; Url = "http://localhost:3100/ready" },
    @{ Name = "Alertmanager"; Url = "http://localhost:9093/-/healthy" }
)

$allHealthy = $true
foreach ($service in $services) {
    $healthy = Test-ServiceHealth -ServiceName $service.Name -Url $service.Url
    if (-not $healthy) {
        $allHealthy = $false
    }
}

# Display service URLs
Write-Host "`nMonitoring Services:" -ForegroundColor Cyan
Write-Host "  Grafana:      http://localhost:3001 (admin/admin)" -ForegroundColor White
Write-Host "  Prometheus:   http://localhost:9090" -ForegroundColor White
Write-Host "  Loki:        http://localhost:3100" -ForegroundColor White
Write-Host "  Alertmanager: http://localhost:9093" -ForegroundColor White
Write-Host "  Node Exporter: http://localhost:9100/metrics" -ForegroundColor White
Write-Host "  cAdvisor:     http://localhost:8080" -ForegroundColor White

# Show container status
Write-Host "`nContainer Status:" -ForegroundColor Cyan
docker-compose -f monitoring/docker-compose.monitoring.yml ps

if ($allHealthy) {
    Write-Host "`n[SUCCESS] Monitoring stack deployed successfully!" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Access Grafana at http://localhost:3001 (admin/admin)" -ForegroundColor White
    Write-Host "  2. Import dashboards for LiveTranslate services" -ForegroundColor White
    Write-Host "  3. Configure alert notification channels" -ForegroundColor White
    Write-Host "  4. Test alert rules with sample conditions" -ForegroundColor White
} else {
    Write-Host "`n[WARNING] Some services may not be fully ready yet." -ForegroundColor Yellow
    Write-Host "   Please wait a few more minutes and check the logs:" -ForegroundColor White
    Write-Host "   docker-compose -f monitoring/docker-compose.monitoring.yml logs" -ForegroundColor Gray
}

Write-Host "`nMonitoring deployment complete!" -ForegroundColor Green
