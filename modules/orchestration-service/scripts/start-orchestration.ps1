# Start Orchestration Service
# This script starts the orchestration service with proper Python environment

param(
    [string]$PythonPath = ""
)

Write-Host "Starting Livetranslate Orchestration Service..." -ForegroundColor Green

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: conda is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Determine Python executable
$PY_BIN = "python"
if ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $PY_BIN = "$env:CONDA_PREFIX\python.exe"
    Write-Host "Using conda environment: $env:CONDA_PREFIX" -ForegroundColor Green
} elseif ($PythonPath -and (Test-Path $PythonPath)) {
    $PY_BIN = $PythonPath
    Write-Host "Using specified Python: $PythonPath" -ForegroundColor Green
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PY_BIN = "python3"
    Write-Host "Using system python3" -ForegroundColor Yellow
} else {
    Write-Host "Error: No suitable Python executable found" -ForegroundColor Red
    exit 1
}

# Check Python version
try {
    $PY_VERSION = & $PY_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    $PY_MAJOR = & $PY_BIN -c "import sys; print(sys.version_info.major)" 2>$null
    $PY_MINOR = & $PY_BIN -c "import sys; print(sys.version_info.minor)" 2>$null

    if ([int]$PY_MAJOR -lt 3 -or ([int]$PY_MAJOR -eq 3 -and [int]$PY_MINOR -lt 8)) {
        Write-Host "Error: Python >= 3.8 required. Current: $PY_VERSION" -ForegroundColor Red
        Write-Host "Please activate the conda environment: conda activate livetranslate-orchestration" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "Python version: $PY_VERSION" -ForegroundColor Green
} catch {
    Write-Host "Error checking Python version" -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "src\main_fastapi.py")) {
    Write-Host "Error: main_fastapi.py not found. Please run this script from the orchestration-service directory" -ForegroundColor Red
    exit 1
}

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Green
try {
    & $PY_BIN -c "import fastapi, uvicorn, websockets, requests, httpx, pydantic, yaml, numpy, soundfile, scipy, librosa, pydub, aiosqlite" 2>$null
} catch {
    Write-Host "Some dependencies missing. Installing..." -ForegroundColor Yellow
    if (Test-Path "requirements.txt") {
        & $PY_BIN -m pip install -r requirements.txt
    } else {
        Write-Host "requirements.txt not found" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Starting orchestration service on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Green

# Start the service
& $PY_BIN -m uvicorn src.main_fastapi:app --host 0.0.0.0 --port 8000 --reload
