#!/usr/bin/env pwsh
<#
.SYNOPSIS
Backend Service startup script for FastAPI orchestration service

.DESCRIPTION
This script starts the FastAPI backend service with all necessary
dependencies and health checks.

.EXAMPLE
./start-backend.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting Orchestration Service Backend (FastAPI)" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Gray

# Navigate to orchestration service directory
$backendPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendPath

Write-Host "📁 Working directory: $backendPath" -ForegroundColor Gray

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create virtual environment if it doesn't exist
Write-Host "🐍 Setting up Python virtual environment..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    & .\venv\Scripts\Activate.ps1
} else {
    & source venv/bin/activate
}
Write-Host "✅ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if (Test-Path "requirements-database.txt") {
    pip install -r requirements-database.txt
}
if (Test-Path "requirements-google-meet.txt") {
    pip install -r requirements-google-meet.txt
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green

Write-Host ""

# Display service information
$BACKEND_URL = "http://localhost:3000"
$FRONTEND_URL = "http://localhost:5173"

Write-Host "🚀 Starting Backend Service..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Service Information:" -ForegroundColor Cyan
Write-Host "  Backend API:     $BACKEND_URL" -ForegroundColor White
Write-Host "  API Docs:        $BACKEND_URL/docs" -ForegroundColor White
Write-Host "  ReDoc:           $BACKEND_URL/redoc" -ForegroundColor White
Write-Host "  Health Check:    $BACKEND_URL/api/health" -ForegroundColor White
Write-Host "  Technology:      FastAPI + Python + Async" -ForegroundColor White
Write-Host ""

Write-Host "Features Available:" -ForegroundColor Cyan
Write-Host "  🌐 RESTful API Endpoints" -ForegroundColor White
Write-Host "  🔄 WebSocket Real-time Communication" -ForegroundColor White
Write-Host "  🎙️ Audio Processing API" -ForegroundColor White
Write-Host "  🤖 Bot Management API" -ForegroundColor White
Write-Host "  📊 System Health Monitoring" -ForegroundColor White
Write-Host "  🔧 Service Coordination" -ForegroundColor White
Write-Host ""

Write-Host "Frontend Service:" -ForegroundColor Cyan
Write-Host "  Frontend will connect to this backend on port 3000" -ForegroundColor White
Write-Host "  Start frontend: cd ../frontend-service && ./start-frontend.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "Press Ctrl+C to stop the backend service" -ForegroundColor Yellow
Write-Host ""

# Start backend server
try {
    # Check if we have src/main_fastapi.py (new FastAPI backend)
    if (Test-Path "src/main_fastapi.py") {
        Write-Host "🚀 Starting FastAPI backend..." -ForegroundColor Green
        python src/main_fastapi.py
    } elseif (Test-Path "backend/main.py") {
        Write-Host "🚀 Starting FastAPI backend (legacy location)..." -ForegroundColor Green
        Set-Location backend
        python main.py
    } elseif (Test-Path "src/orchestration_service.py") {
        Write-Host "🚀 Starting orchestration service..." -ForegroundColor Green
        python src/orchestration_service.py
    } else {
        Write-Host "❌ No backend entry point found" -ForegroundColor Red
        Write-Host "Expected: src/main_fastapi.py, backend/main.py, or src/orchestration_service.py" -ForegroundColor Gray
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ Backend service failed to start" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Gray
    exit 1
}