#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting LiveTranslate Orchestration Service with Poetry" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Gray

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
}
catch {
    Write-Host "❌ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Check Poetry
try {
    $poetryVersion = poetry --version 2>$null
    Write-Host "✅ Poetry: $poetryVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Poetry not found. Installing Poetry..." -ForegroundColor Yellow
    try {
        python -m pip install poetry
        $poetryVersion = poetry --version 2>$null
        Write-Host "✅ Poetry installed: $poetryVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install Poetry. Please install manually:" -ForegroundColor Red
        Write-Host "   curl -sSL https://install.python-poetry.org | python3 -" -ForegroundColor Gray
        exit 1
    }
}

Write-Host ""

# Install dependencies using Poetry
Write-Host "📦 Installing dependencies with Poetry..." -ForegroundColor Yellow
try {
    Write-Host "  📦 Installing core dependencies..." -ForegroundColor Gray
    poetry install --only main
    
    Write-Host "  🔧 Installing development dependencies..." -ForegroundColor Gray
    poetry install --with dev
    
    Write-Host "  🎵 Installing audio dependencies (may skip on Windows)..." -ForegroundColor Gray
    try {
        poetry install --with audio
        Write-Host "  ✅ Audio dependencies installed" -ForegroundColor Green
    }
    catch {
        Write-Host "  ⚠️ Audio dependencies skipped (not required for basic functionality)" -ForegroundColor Yellow
    }
    
    Write-Host "  📊 Installing monitoring dependencies..." -ForegroundColor Gray
    try {
        poetry install --with monitoring
        Write-Host "  ✅ Monitoring dependencies installed" -ForegroundColor Green
    }
    catch {
        Write-Host "  ⚠️ Some monitoring dependencies skipped" -ForegroundColor Yellow
    }
    
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Gray
    Write-Host $_.Exception.Message -ForegroundColor Gray
    
    Write-Host "🔄 Trying fallback installation..." -ForegroundColor Yellow
    try {
        poetry install --only main --no-dev
        Write-Host "✅ Basic dependencies installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Even basic installation failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Display service information
$BACKEND_URL = "http://localhost:3000"
$FRONTEND_URL = "http://localhost:5173"

Write-Host "🚀 Starting Complete Orchestration Service Backend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Service Information:" -ForegroundColor Cyan
Write-Host "  Backend API:         $BACKEND_URL" -ForegroundColor White
Write-Host "  API Documentation:   $BACKEND_URL/docs" -ForegroundColor White
Write-Host "  ReDoc:              $BACKEND_URL/redoc" -ForegroundColor White
Write-Host "  Health Check:       $BACKEND_URL/api/health" -ForegroundColor White
Write-Host "  Frontend Connection: $FRONTEND_URL" -ForegroundColor White
Write-Host ""

Write-Host "Technology Stack:" -ForegroundColor Cyan
Write-Host "  🐍 Python with Poetry dependency management" -ForegroundColor White
Write-Host "  🚀 FastAPI with async/await support" -ForegroundColor White
Write-Host "  🔌 WebSocket real-time communication" -ForegroundColor White
Write-Host "  📊 Pydantic data validation" -ForegroundColor White
Write-Host "  🔧 Uvicorn ASGI server" -ForegroundColor White
Write-Host ""

Write-Host "Backend Features:" -ForegroundColor Cyan
Write-Host "  🌐 RESTful API Endpoints" -ForegroundColor White
Write-Host "  🔄 WebSocket Real-time Communication" -ForegroundColor White
Write-Host "  🎙️ Audio Processing API Integration" -ForegroundColor White
Write-Host "  🤖 Bot Management System" -ForegroundColor White
Write-Host "  📊 System Health Monitoring" -ForegroundColor White
Write-Host "  🔧 Service Coordination and Load Balancing" -ForegroundColor White
Write-Host "  🗄️ Database Integration (PostgreSQL)" -ForegroundColor White
Write-Host "  📈 Monitoring and Analytics" -ForegroundColor White
Write-Host ""

Write-Host "Service Integration:" -ForegroundColor Cyan
Write-Host "  🎤 Whisper Service (Port 5001) - NPU-optimized transcription" -ForegroundColor White
Write-Host "  🌍 Translation Service (Port 5003) - GPU-optimized translation" -ForegroundColor White
Write-Host "  🎨 Frontend Service (Port 5173) - Modern React interface" -ForegroundColor White
Write-Host "  📊 Monitoring Stack - Prometheus, Grafana, Loki" -ForegroundColor White
Write-Host ""

Write-Host "Development Commands:" -ForegroundColor Cyan
Write-Host "  poetry run start-backend     # Start backend service" -ForegroundColor Gray
Write-Host "  poetry run pytest           # Run tests" -ForegroundColor Gray
Write-Host "  poetry run black src/        # Format code" -ForegroundColor Gray
Write-Host "  poetry run flake8 src/       # Lint code" -ForegroundColor Gray
Write-Host "  poetry shell                 # Activate virtual environment" -ForegroundColor Gray
Write-Host ""

Write-Host "Frontend Service:" -ForegroundColor Cyan
Write-Host "  Start frontend: cd ../frontend-service && ./start-frontend.ps1" -ForegroundColor White
Write-Host "  Frontend will automatically connect to this backend on port 3000" -ForegroundColor Gray
Write-Host ""

Write-Host "Press Ctrl+C to stop the orchestration service" -ForegroundColor Yellow
Write-Host ""

# Start the complete orchestration service using Poetry
try {
    Write-Host "🚀 Starting complete orchestration service backend..." -ForegroundColor Green
    Write-Host "🔧 Using Poetry virtual environment..." -ForegroundColor Gray
    
    # Use Poetry to run the main application
    poetry run python src/main.py
}
catch {
    Write-Host ""
    Write-Host "❌ Orchestration service failed to start" -ForegroundColor Red
    Write-Host "Error details:" -ForegroundColor Gray
    Write-Host $_.Exception.Message -ForegroundColor Gray
    
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if all dependencies are installed: poetry install" -ForegroundColor Gray
    Write-Host "  2. Verify Python version: python --version (should be 3.9+)" -ForegroundColor Gray
    Write-Host "  3. Check Poetry status: poetry env info" -ForegroundColor Gray
    Write-Host "  4. Run health check: poetry run python -c import src.main print(OK)" -ForegroundColor Gray
    Write-Host ""
    
    exit 1
}