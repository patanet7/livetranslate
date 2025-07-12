#!/usr/bin/env pwsh
<#
.SYNOPSIS
Development environment startup script for LiveTranslate

.DESCRIPTION
This script starts both the backend orchestration service and frontend service
concurrently for development. It provides health checks and coordination between services.

.EXAMPLE
./start-development.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "🌟 Starting LiveTranslate Development Environment" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Gray

# Check if we're in the right directory
if (!(Test-Path "modules")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    Write-Host "Expected to find 'modules' directory" -ForegroundColor Gray
    exit 1
}

# Display architecture overview
Write-Host ""
Write-Host "🏗️ Service Architecture:" -ForegroundColor Cyan
Write-Host "┌─────────────────────────────────────────────────────────┐" -ForegroundColor Gray
Write-Host "│                  Development Setup                     │" -ForegroundColor Gray
Write-Host "├─────────────────────────────────────────────────────────┤" -ForegroundColor Gray
Write-Host "│  Frontend Service (React + Vite)                       │" -ForegroundColor White
Write-Host "│  ├─ Port: 5173                                          │" -ForegroundColor White
Write-Host "│  ├─ Features: Audio testing, Bot management, Dashboard  │" -ForegroundColor White
Write-Host "│  └─ Proxy: API calls → Backend (Port 3000)             │" -ForegroundColor White
Write-Host "│                          ↓                             │" -ForegroundColor Gray
Write-Host "│  Backend Service (FastAPI + Python)                    │" -ForegroundColor White
Write-Host "│  ├─ Port: 3000                                          │" -ForegroundColor White
Write-Host "│  ├─ Features: API endpoints, WebSocket, Service coord  │" -ForegroundColor White
Write-Host "│  └─ Connections: Whisper service, Translation service  │" -ForegroundColor White
Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Gray
Write-Host ""

# Service URLs
$FRONTEND_URL = "http://localhost:5173"
$BACKEND_URL = "http://localhost:3000"

Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "  Frontend:     $FRONTEND_URL" -ForegroundColor White
Write-Host "  Backend API:  $BACKEND_URL" -ForegroundColor White
Write-Host "  API Docs:     $BACKEND_URL/docs" -ForegroundColor White
Write-Host "  Health Check: $BACKEND_URL/api/health" -ForegroundColor White
Write-Host ""

# Check prerequisites
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 18+ first." -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Check pnpm
try {
    $pnpmVersion = pnpm --version 2>$null
    Write-Host "✅ pnpm: $pnpmVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️ pnpm not found. Installing pnpm..." -ForegroundColor Yellow
    npm install -g pnpm
    $pnpmVersion = pnpm --version
    Write-Host "✅ pnpm installed: $pnpmVersion" -ForegroundColor Green
}

Write-Host ""

# Start backend in background
Write-Host "🚀 Starting backend service..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    Set-Location "modules/orchestration-service"
    
    # Create virtual environment if needed
    if (!(Test-Path "venv")) {
        python -m venv venv
    }
    
    # Activate virtual environment
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        & .\venv\Scripts\Activate.ps1
    }
    
    # Install dependencies
    pip install -r requirements.txt -q
    if (Test-Path "requirements-database.txt") {
        pip install -r requirements-database.txt -q
    }
    if (Test-Path "requirements-google-meet.txt") {
        pip install -r requirements-google-meet.txt -q
    }
    
    # Start backend
    if (Test-Path "backend/main.py") {
        Set-Location backend
        python main.py
    } elseif (Test-Path "src/orchestration_service.py") {
        python src/orchestration_service.py
    }
}

# Give backend time to start
Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-RestMethod -Uri "$BACKEND_URL/api/health" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✅ Backend service started successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Backend health check failed, but continuing..." -ForegroundColor Yellow
    Write-Host "   Backend may still be starting up" -ForegroundColor Gray
}

Write-Host ""

# Install frontend dependencies
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location "modules/frontend-service"
if (!(Test-Path "node_modules")) {
    pnpm install
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Frontend dependencies already installed" -ForegroundColor Green
}

# Start browser opener in background
$openBrowser = {
    Start-Sleep -Seconds 8
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Start-Process $using:FRONTEND_URL
    } elseif ($IsMacOS) {
        open $using:FRONTEND_URL
    } else {
        xdg-open $using:FRONTEND_URL
    }
}
$browserJob = Start-Job -ScriptBlock $openBrowser

Write-Host ""
Write-Host "🎨 Starting frontend service..." -ForegroundColor Yellow
Write-Host ""
Write-Host "🌟 Development Environment Ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Available Services:" -ForegroundColor Cyan
Write-Host "  🎨 Frontend:      $FRONTEND_URL" -ForegroundColor White
Write-Host "  🚀 Backend API:   $BACKEND_URL" -ForegroundColor White
Write-Host "  📚 API Docs:      $BACKEND_URL/docs" -ForegroundColor White
Write-Host "  🔍 Health Check:  $BACKEND_URL/api/health" -ForegroundColor White
Write-Host ""
Write-Host "Features:" -ForegroundColor Cyan
Write-Host "  🎙️ Audio Testing & Recording Interface" -ForegroundColor White
Write-Host "  🤖 Bot Management & Analytics Dashboard" -ForegroundColor White
Write-Host "  📊 Real-time System Monitoring" -ForegroundColor White
Write-Host "  ⚙️ Settings & Configuration Management" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Start frontend (this will block)
try {
    pnpm dev
} finally {
    # Cleanup: Stop backend job
    Write-Host ""
    Write-Host "🛑 Stopping all services..." -ForegroundColor Yellow
    
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    
    Stop-Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job $browserJob -ErrorAction SilentlyContinue
    
    Write-Host "✅ All services stopped" -ForegroundColor Green
    Write-Host ""
    Write-Host "Thank you for using LiveTranslate! 🎉" -ForegroundColor Cyan
}