#!/bin/bash
#
# Start Translation Service with existing conda environment
# Uses the vllm-cuda conda environment that's already set up
#

set -e

# Color output functions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info "🚀 Starting LiveTranslate Translation Service (Local)"
info "Using existing conda environment: vllm-cuda"
info "================================================"

# Check if we're in the right directory
if [ ! -f "src/api_server.py" ]; then
    error "src/api_server.py not found. Please run from translation-service directory"
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "vllm-cuda"; then
    error "Conda environment 'vllm-cuda' not found"
    error "Please create it first or use the full setup script"
    exit 1
fi

# Activate conda environment
info "🔧 Activating conda environment: vllm-cuda"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-cuda

# Set environment variables for internal vLLM
export INFERENCE_BACKEND="vllm"
export PORT=5003
export LOG_LEVEL="INFO"
export GPU_ENABLE="true"
export TRANSLATION_MODEL="./models/Llama-3.1-8B-Instruct"
export VLLM_INTERNAL_PORT="8010"
export CUDA_VISIBLE_DEVICES="0"

info "🔧 Configuration:"
info "   Port: 5003"
info "   Backend: Internal vLLM (auto-start)"
info "   Model: ./models/Llama-3.1-8B-Instruct (local)"
info "   Internal vLLM Port: 8010"
info "   GPU: Enabled"

echo ""
info "🚀 Starting Translation Service..."
info "   API: http://localhost:5003"
info "   Health: http://localhost:5003/api/health"
info "   Config: http://localhost:5003/api/config"
info "   Internal vLLM: http://localhost:8010"
echo ""
info "📝 Press Ctrl+C to stop"
info "================================================"

# Start the service
python src/api_server.py
