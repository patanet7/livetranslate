#!/usr/bin/env pwsh
<#
.SYNOPSIS
Frontend Service startup script for React development server

.DESCRIPTION
This script starts the React frontend service with Vite dev server,
including dependency installation and health checks.

.EXAMPLE
./start-frontend.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "🎨 Starting Frontend Service (React + Vite)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Gray

# Navigate to frontend service directory
$frontendPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $frontendPath

Write-Host "📁 Working directory: $frontendPath" -ForegroundColor Gray

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

# Install dependencies
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
if (!(Test-Path "node_modules")) {
    pnpm install
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

Write-Host ""

# Display service information
$FRONTEND_URL = "http://localhost:5173"
$BACKEND_URL = "http://localhost:3000"

Write-Host "🚀 Starting Frontend Service..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Service Information:" -ForegroundColor Cyan
Write-Host "  Frontend URL: $FRONTEND_URL" -ForegroundColor White
Write-Host "  Backend API:  $BACKEND_URL" -ForegroundColor White
Write-Host "  Technology:   React 18 + TypeScript + Vite" -ForegroundColor White
Write-Host ""

Write-Host "Features Available:" -ForegroundColor Cyan
Write-Host "  🎙️ Audio Testing Interface" -ForegroundColor White
Write-Host "  🤖 Bot Management Dashboard" -ForegroundColor White
Write-Host "  📊 Real-time System Monitoring" -ForegroundColor White
Write-Host "  ⚙️ Settings & Configuration" -ForegroundColor White
Write-Host ""

Write-Host "Note: Backend must be running on port 3000" -ForegroundColor Yellow
Write-Host "Run backend with: cd ../orchestration-service && ./start-backend.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "Press Ctrl+C to stop the frontend service" -ForegroundColor Yellow
Write-Host ""

# Start frontend development server
try {
    pnpm dev
} catch {
    Write-Host ""
    Write-Host "❌ Frontend service failed to start" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Gray
    exit 1
}