#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start Translation Service with GPU acceleration
    
.DESCRIPTION
    Starts the LiveTranslate Translation Service with optimal configuration for
    GPU-accelerated inference using vLLM, Ollama, or Triton backends.
    
.PARAMETER Backend
    Inference backend to use (auto, vllm, ollama, triton)
    
.PARAMETER Port
    Port to run the service on (default: 5003)
    
.PARAMETER LogLevel
    Logging level (DEBUG, INFO, WARNING, ERROR)
    
.PARAMETER GPU
    Enable GPU acceleration (default: true)
    
.PARAMETER Model
    Model name to load (default: auto-detect)
    
.EXAMPLE
    .\start-translation-service.ps1
    Start with auto-detection
    
.EXAMPLE
    .\start-translation-service.ps1 -Backend vllm -GPU $true
    Start with vLLM backend and GPU acceleration
#>

param(
    [string]$Backend = "auto",
    [int]$Port = 5003,
    [string]$LogLevel = "INFO",
    [bool]$GPU = $true,
    [string]$Model = "auto",
    [string]$ApiServer = "fastapi"  # fastapi (default) or flask
)

# Color output functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }

Write-Info "🚀 Starting LiveTranslate Translation Service"
Write-Info "================================================"

# Check if we're in the correct directory
$expectedDir = "translation-service"
$currentDir = Split-Path -Leaf (Get-Location)

if ($currentDir -ne $expectedDir) {
    Write-Warning "⚠️  Current directory: $currentDir"
    Write-Warning "⚠️  Expected directory: $expectedDir"
    Write-Info "💡 Attempting to navigate to translation service directory..."
    
    # Try to find and navigate to the translation service directory
    $possiblePaths = @(
        ".\modules\translation-service",
        "..\translation-service",
        ".\translation-service",
        "..\..\modules\translation-service"
    )
    
    $found = $false
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            Set-Location $path
            Write-Info "✅ Navigated to: $(Get-Location)"
            $found = $true
            break
        }
    }
    
    if (-not $found) {
        Write-Error "❌ Could not find translation-service directory"
        Write-Info "💡 Please run this script from the translation-service directory"
        exit 1
    }
}

# Verify required files exist based on API server choice
if ($ApiServer -eq "fastapi") {
    $requiredFiles = @(
        "src/api_server_fastapi.py",
        "requirements.txt"
    )
} else {
    $requiredFiles = @(
        "src/api_server.py",
        "src/translation_service.py",
        "requirements.txt"
    )
}

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "❌ Required file not found: $file"
        exit 1
    }
}

Write-Info "✅ All required files found (API: $ApiServer)"

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Info "🐍 Python version: $pythonVersion"
} catch {
    Write-Error "❌ Python not found. Please install Python 3.8+ and add it to PATH"
    exit 1
}

# Check if virtual environment exists
$venvPath = "venv"
if (-not (Test-Path $venvPath)) {
    Write-Info "📦 Creating Python virtual environment..."
    python -m venv $venvPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to create virtual environment"
        exit 1
    }
}

# Activate virtual environment
Write-Info "🔧 Activating virtual environment..."
if ($IsWindows -or $env:OS -like "*Windows*") {
    & "$venvPath\Scripts\Activate.ps1"
} else {
    . "$venvPath/bin/activate"
}

# Check if requirements are installed
Write-Info "📋 Checking dependencies..."
$pipList = pip list 2>&1

if ($pipList -notmatch "fastapi" -or $pipList -notmatch "uvicorn") {
    Write-Info "📦 Installing required dependencies..."
    
    # Install base requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "⚠️  Basic requirements installation had issues, continuing..."
    }
    
    # Try to install optional GPU dependencies if GPU is enabled
    if ($GPU) {
        Write-Info "🚀 Installing GPU dependencies..."
        
        # Check if CUDA is available
        try {
            $cudaVersion = nvidia-smi 2>&1
            if ($cudaVersion -match "CUDA Version: (\d+\.\d+)") {
                $detectedCuda = $matches[1]
                Write-Info "🎮 CUDA detected: $detectedCuda"
                
                # Install PyTorch with CUDA support
                Write-Info "📦 Installing PyTorch with CUDA support..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                
                # Try to install vLLM if not present
                if ($pipList -notmatch "vllm") {
                    Write-Info "📦 Installing vLLM..."
                    pip install vllm
                }
            }
        } catch {
            Write-Warning "⚠️  CUDA not detected, will use CPU fallback"
        }
    }
}

# Set environment variables
Write-Info "⚙️  Setting environment variables..."

$env:PORT = $Port
$env:LOG_LEVEL = $LogLevel
$env:INFERENCE_BACKEND = $Backend
$env:GPU_ENABLE = if ($GPU) { "true" } else { "false" }

if ($Model -ne "auto") {
    $env:TRANSLATION_MODEL = $Model
}

# Additional GPU optimizations
if ($GPU) {
    $env:CUDA_VISIBLE_DEVICES = "0"  # Use first GPU
    $env:VLLM_TENSOR_PARALLEL_SIZE = "1"
    $env:GPU_MEMORY_UTILIZATION = "0.85"
}

# Display configuration
Write-Info "🔧 Service Configuration:"
Write-Info "   API Server: $ApiServer"
Write-Info "   Backend: $Backend"
Write-Info "   Port: $Port"
Write-Info "   Log Level: $LogLevel"
Write-Info "   GPU Enabled: $GPU"
Write-Info "   Model: $Model"

# Check if port is available
try {
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($connection) {
        Write-Warning "⚠️  Port $Port appears to be in use"
        Write-Info "💡 The service will attempt to start anyway, but may fail"
    }
} catch {
    # Test-NetConnection might not be available on all systems
    Write-Info "💡 Cannot check port availability, proceeding..."
}

Write-Info ""
Write-Info "🚀 Starting Translation Service..."
Write-Info "   API URL: http://localhost:$Port"
Write-Info "   Health Check: http://localhost:$Port/api/health"
Write-Info "   Documentation: http://localhost:$Port/docs"
Write-Info ""
Write-Info "📝 Logs will appear below. Press Ctrl+C to stop the service."
Write-Info "================================================"

# Start the service
try {
    if ($ApiServer -eq "fastapi") {
        Write-Info "🚀 Starting FastAPI server..."
        python src/api_server_fastapi.py
    } else {
        Write-Info "🚀 Starting Flask server..."
        python src/api_server.py
    }
} catch {
    Write-Error "❌ Service failed to start: $_"
    exit 1
} finally {
    Write-Info ""
    Write-Info "🛑 Translation Service stopped"
    Write-Info "================================================"
}