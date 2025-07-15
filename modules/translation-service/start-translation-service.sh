#!/bin/bash
#
# Start Translation Service with GPU acceleration
#
# This script starts the LiveTranslate Translation Service with optimal
# configuration for GPU-accelerated inference using vLLM, Ollama, or Triton.

set -euo pipefail

# Default configuration
BACKEND="${BACKEND:-auto}"
PORT="${PORT:-5003}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
GPU_ENABLE="${GPU_ENABLE:-true}"
MODEL="${MODEL:-auto}"

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}$1${NC}"
}

log_warning() {
    echo -e "${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${RED}$1${NC}"
}

log_info "🚀 Starting LiveTranslate Translation Service"
log_info "================================================"

# Check if we're in the correct directory
expected_dir="translation-service"
current_dir=$(basename "$PWD")

if [[ "$current_dir" != "$expected_dir" ]]; then
    log_warning "⚠️  Current directory: $current_dir"
    log_warning "⚠️  Expected directory: $expected_dir"
    log_info "💡 Attempting to navigate to translation service directory..."
    
    # Try to find and navigate to the translation service directory
    possible_paths=(
        "./modules/translation-service"
        "../translation-service"
        "./translation-service"
        "../../modules/translation-service"
    )
    
    found=false
    for path in "${possible_paths[@]}"; do
        if [[ -d "$path" ]]; then
            cd "$path"
            log_info "✅ Navigated to: $(pwd)"
            found=true
            break
        fi
    done
    
    if [[ "$found" == false ]]; then
        log_error "❌ Could not find translation-service directory"
        log_info "💡 Please run this script from the translation-service directory"
        exit 1
    fi
fi

# Verify required files exist
required_files=(
    "src/api_server.py"
    "src/translation_service.py"
    "requirements.txt"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        log_error "❌ Required file not found: $file"
        exit 1
    fi
done

log_info "✅ All required files found"

# Check Python installation
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    log_error "❌ Python not found. Please install Python 3.8+ and add it to PATH"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

python_version=$($PYTHON_CMD --version 2>&1)
log_info "🐍 Python version: $python_version"

# Check if virtual environment exists
venv_path="venv"
if [[ ! -d "$venv_path" ]]; then
    log_info "📦 Creating Python virtual environment..."
    $PYTHON_CMD -m venv "$venv_path"
    
    if [[ $? -ne 0 ]]; then
        log_error "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
log_info "🔧 Activating virtual environment..."
source "$venv_path/bin/activate"

# Check if requirements are installed
log_info "📋 Checking dependencies..."
pip_list=$(pip list 2>&1)

if ! echo "$pip_list" | grep -q "fastapi" || ! echo "$pip_list" | grep -q "uvicorn"; then
    log_info "📦 Installing required dependencies..."
    
    # Install base requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [[ $? -ne 0 ]]; then
        log_warning "⚠️  Basic requirements installation had issues, continuing..."
    fi
    
    # Try to install optional GPU dependencies if GPU is enabled
    if [[ "$GPU_ENABLE" == "true" ]]; then
        log_info "🚀 Installing GPU dependencies..."
        
        # Check if CUDA is available
        if command -v nvidia-smi &> /dev/null; then
            cuda_output=$(nvidia-smi 2>&1 || true)
            if echo "$cuda_output" | grep -q "CUDA Version"; then
                detected_cuda=$(echo "$cuda_output" | grep -o "CUDA Version: [0-9.]*" | cut -d' ' -f3)
                log_info "🎮 CUDA detected: $detected_cuda"
                
                # Install PyTorch with CUDA support
                log_info "📦 Installing PyTorch with CUDA support..."
                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                
                # Try to install vLLM if not present
                if ! echo "$pip_list" | grep -q "vllm"; then
                    log_info "📦 Installing vLLM..."
                    pip install vllm || log_warning "⚠️  vLLM installation failed, will use fallback"
                fi
            fi
        else
            log_warning "⚠️  CUDA not detected, will use CPU fallback"
        fi
    fi
fi

# Set environment variables
log_info "⚙️  Setting environment variables..."

export PORT="$PORT"
export LOG_LEVEL="$LOG_LEVEL"
export INFERENCE_BACKEND="$BACKEND"
export GPU_ENABLE="$GPU_ENABLE"

if [[ "$MODEL" != "auto" ]]; then
    export TRANSLATION_MODEL="$MODEL"
fi

# Additional GPU optimizations
if [[ "$GPU_ENABLE" == "true" ]]; then
    export CUDA_VISIBLE_DEVICES="0"  # Use first GPU
    export VLLM_TENSOR_PARALLEL_SIZE="1"
    export GPU_MEMORY_UTILIZATION="0.85"
fi

# Display configuration
log_info "🔧 Service Configuration:"
log_info "   Backend: $BACKEND"
log_info "   Port: $PORT"
log_info "   Log Level: $LOG_LEVEL"
log_info "   GPU Enabled: $GPU_ENABLE"
log_info "   Model: $MODEL"

# Check if port is available
if command -v lsof &> /dev/null; then
    if lsof -i :$PORT &> /dev/null; then
        log_warning "⚠️  Port $PORT appears to be in use"
        log_info "💡 The service will attempt to start anyway, but may fail"
    fi
elif command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
        log_warning "⚠️  Port $PORT appears to be in use"
        log_info "💡 The service will attempt to start anyway, but may fail"
    fi
fi

echo
log_info "🚀 Starting Translation Service..."
log_info "   API URL: http://localhost:$PORT"
log_info "   Health Check: http://localhost:$PORT/api/health"
log_info "   Documentation: http://localhost:$PORT/docs"
echo
log_info "📝 Logs will appear below. Press Ctrl+C to stop the service."
log_info "================================================"

# Start the service
trap 'log_info "\n🛑 Translation Service stopped\n================================================"' EXIT

$PYTHON_CMD src/api_server.py