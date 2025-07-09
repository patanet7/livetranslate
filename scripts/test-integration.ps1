#!/usr/bin/env pwsh
# LiveTranslate Integration Test Script
# Tests all services and their communication paths

param(
    [switch]$Verbose,
    [switch]$SkipBuild,
    [string]$Environment = "development"
)

$ErrorActionPreference = "Stop"

Write-Host "LiveTranslate Integration Test Suite" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow

# Test configuration
$services = @(
    @{
        Name = "Translation Service"
        Url = "http://localhost:5003/api/health"
        Port = 5003
        Container = "livetranslate-translation-service"
        ExpectedStatus = 200
    },
    @{
        Name = "Whisper Service"
        Url = "http://localhost:5001/api/health"
        Port = 5001
        Container = "livetranslate-whisper-service"
        ExpectedStatus = 200
    },
    @{
        Name = "Frontend Service"
        Url = "http://localhost:3000/api/health"
        Port = 3000
        Container = "frontend-service-frontend-1"
        ExpectedStatus = 200
    },
    @{
        Name = "Speaker Service"
        Url = "http://localhost:5002/api/health"
        Port = 5002
        Container = "livetranslate-speaker-service"
        ExpectedStatus = 200
    }
)

$monitoringServices = @(
    @{
        Name = "Prometheus"
        Url = "http://localhost:9090/-/healthy"
        Port = 9090
        Container = "livetranslate-prometheus"
    },
    @{
        Name = "Grafana"
        Url = "http://localhost:3001/api/health"
        Port = 3001
        Container = "livetranslate-grafana"
    },
    @{
        Name = "Loki"
        Url = "http://localhost:3100/ready"
        Port = 3100
        Container = "livetranslate-loki"
    },
    @{
        Name = "Alertmanager"
        Url = "http://localhost:9093/-/healthy"
        Port = 9093
        Container = "livetranslate-alertmanager"
    }
)

function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [string]$Url,
        [int]$ExpectedStatus = 200,
        [int]$TimeoutSeconds = 30
    )
    
    try {
        $response = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSeconds -UseBasicParsing
        if ($response.StatusCode -eq $ExpectedStatus) {
            Write-Host "[OK] $ServiceName is healthy (HTTP $($response.StatusCode))" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARN] $ServiceName returned HTTP $($response.StatusCode), expected $ExpectedStatus" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "[ERROR] $ServiceName is not responding: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Test-ServiceCommunication {
    Write-Host "`nTesting service-to-service communication..." -ForegroundColor Cyan
    
    # Test Translation Service Redis connection
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5003/api/status" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "[OK] Translation Service Redis connection working" -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] Translation Service Redis status check failed" -ForegroundColor Yellow
    }
    
    # Test Whisper Service model loading
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5001/api/status" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "[OK] Whisper Service model status working" -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] Whisper Service model status check failed" -ForegroundColor Yellow
    }
    
    # Test Frontend to Backend communication
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/api/services" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "[OK] Frontend service discovery working" -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] Frontend service discovery check failed" -ForegroundColor Yellow
    }
}

function Test-NetworkConnectivity {
    Write-Host "`nTesting Docker network connectivity..." -ForegroundColor Cyan
    
    $networks = @(
        "livetranslate-frontend",
        "livetranslate-backend", 
        "livetranslate-data",
        "livetranslate-monitoring"
    )
    
    foreach ($network in $networks) {
        try {
            $networkInfo = docker network inspect $network 2>$null | ConvertFrom-Json
            if ($networkInfo) {
                $containerCount = $networkInfo.Containers.Count
                Write-Host "[OK] Network '$network' exists with $containerCount containers" -ForegroundColor Green
            }
        } catch {
            Write-Host "[ERROR] Network '$network' not found or inaccessible" -ForegroundColor Red
        }
    }
}

function Test-VolumeIntegrity {
    Write-Host "`nTesting Docker volume integrity..." -ForegroundColor Cyan
    
    $volumes = @(
        "livetranslate-postgres-data",
        "livetranslate-redis-data",
        "livetranslate-models-whisper",
        "livetranslate-models-speaker",
        "livetranslate-models-translation",
        "livetranslate-sessions",
        "livetranslate-audio-uploads",
        "livetranslate-prometheus-data",
        "livetranslate-grafana-data",
        "livetranslate-loki-data",
        "livetranslate-logs"
    )
    
    foreach ($volume in $volumes) {
        try {
            $volumeInfo = docker volume inspect $volume 2>$null | ConvertFrom-Json
            if ($volumeInfo) {
                Write-Host "[OK] Volume '$volume' exists and accessible" -ForegroundColor Green
            }
        } catch {
            Write-Host "[ERROR] Volume '$volume' not found or inaccessible" -ForegroundColor Red
        }
    }
}

# Main test execution
Write-Host "`n=== DOCKER INFRASTRUCTURE TESTS ===" -ForegroundColor Magenta
Test-NetworkConnectivity
Test-VolumeIntegrity

Write-Host "`n=== CORE SERVICE HEALTH TESTS ===" -ForegroundColor Magenta
$coreServicesHealthy = $true
foreach ($service in $services) {
    $healthy = Test-ServiceHealth -ServiceName $service.Name -Url $service.Url -ExpectedStatus $service.ExpectedStatus
    if (-not $healthy) {
        $coreServicesHealthy = $false
    }
}

Write-Host "`n=== MONITORING SERVICE HEALTH TESTS ===" -ForegroundColor Magenta
$monitoringHealthy = $true
foreach ($service in $monitoringServices) {
    $healthy = Test-ServiceHealth -ServiceName $service.Name -Url $service.Url
    if (-not $healthy) {
        $monitoringHealthy = $false
    }
}

Write-Host "`n=== SERVICE COMMUNICATION TESTS ===" -ForegroundColor Magenta
Test-ServiceCommunication

Write-Host "`n=== INTEGRATION TEST SUMMARY ===" -ForegroundColor Magenta
if ($coreServicesHealthy) {
    Write-Host "[SUCCESS] All core services are healthy and responding" -ForegroundColor Green
} else {
    Write-Host "[FAILURE] Some core services are not healthy" -ForegroundColor Red
}

if ($monitoringHealthy) {
    Write-Host "[SUCCESS] All monitoring services are healthy and responding" -ForegroundColor Green
} else {
    Write-Host "[FAILURE] Some monitoring services are not healthy" -ForegroundColor Red
}

Write-Host "`nService URLs:" -ForegroundColor Yellow
Write-Host "  Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host "  Whisper API:  http://localhost:5001" -ForegroundColor White
Write-Host "  Speaker API:  http://localhost:5002" -ForegroundColor White
Write-Host "  Translation:  http://localhost:5003" -ForegroundColor White
Write-Host "  Prometheus:   http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana:      http://localhost:3001 (admin/admin)" -ForegroundColor White
Write-Host "  Loki:         http://localhost:3100" -ForegroundColor White
Write-Host "  Alertmanager: http://localhost:9093" -ForegroundColor White

if ($coreServicesHealthy -and $monitoringHealthy) {
    Write-Host "`nIntegration test completed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nIntegration test completed with failures!" -ForegroundColor Red
    exit 1
} 