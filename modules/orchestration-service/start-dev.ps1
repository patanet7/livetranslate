#!/usr/bin/env pwsh
<#
.SYNOPSIS
Development startup script for React frontend + FastAPI backend

.DESCRIPTION
This script starts both the React frontend (Vite dev server) and FastAPI backend
concurrently for development. It also provides health checks and helpful output.

.EXAMPLE
./start-dev.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting LiveTranslate Orchestration Service Development Environment" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Gray

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

# Install frontend dependencies
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
if (!(Test-Path "node_modules")) {
    pnpm install
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Frontend dependencies already installed" -ForegroundColor Green
}
Set-Location ..

# Install backend dependencies
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment and install dependencies
if ($IsWindows) {
    & .\venv\Scripts\Activate.ps1
} else {
    & source venv/bin/activate
}

pip install -r requirements.txt
Write-Host "✅ Backend dependencies installed" -ForegroundColor Green

Write-Host ""
Write-Host "🎯 Starting services..." -ForegroundColor Yellow

# Define service URLs
$FRONTEND_URL = "http://localhost:5173"
$BACKEND_URL = "http://localhost:3000"

Write-Host "Frontend (React + Vite): $FRONTEND_URL" -ForegroundColor Cyan
Write-Host "Backend (FastAPI): $BACKEND_URL" -ForegroundColor Cyan
Write-Host "API Documentation: $BACKEND_URL/docs" -ForegroundColor Cyan

Write-Host ""

# Start backend in background
Write-Host "🔧 Starting FastAPI backend..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    if ($IsWindows) {
        & .\venv\Scripts\Activate.ps1
    } else {
        & source venv/bin/activate
    }

    Set-Location backend
    python main.py
}

# Give backend time to start
Start-Sleep -Seconds 3

# Check if backend started successfully
try {
    $response = Invoke-RestMethod -Uri "$BACKEND_URL/api/health" -Method GET -TimeoutSec 5
    Write-Host "✅ Backend started successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Backend health check failed, but continuing..." -ForegroundColor Yellow
}

# Start frontend
Write-Host "🎨 Starting React frontend..." -ForegroundColor Yellow
Set-Location frontend

# Function to open browser after frontend starts
$openBrowser = {
    Start-Sleep -Seconds 5
    if ($IsWindows) {
        Start-Process $FRONTEND_URL
    } elseif ($IsMacOS) {
        open $FRONTEND_URL
    } else {
        xdg-open $FRONTEND_URL
    }
}

# Start browser opener in background
$browserJob = Start-Job -ScriptBlock $openBrowser

Write-Host ""
Write-Host "🌟 Development Environment Ready!" -ForegroundColor Green
Write-Host "Frontend: $FRONTEND_URL" -ForegroundColor Cyan
Write-Host "Backend: $BACKEND_URL" -ForegroundColor Cyan
Write-Host "API Docs: $BACKEND_URL/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Start frontend (this will block)
try {
    pnpm dev
} finally {
    # Cleanup: Stop backend job
    Write-Host ""
    Write-Host "🛑 Stopping services..." -ForegroundColor Yellow

    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue

    Stop-Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job $browserJob -ErrorAction SilentlyContinue

    Write-Host "✅ All services stopped" -ForegroundColor Green
}
