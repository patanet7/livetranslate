#!/bin/bash
# Development startup script for macOS-optimized Whisper Service

set -e

echo "🍎 Starting macOS-Optimized Whisper Service"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo "❌ Error: Please run this script from the whisper-service-mac directory"
    exit 1
fi

# Check for macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "⚠️  Warning: This service is optimized for macOS"
    echo "It may work on other platforms but without Apple Silicon optimizations"
fi

# Check for virtual environment
VENV_DIR="./venv"
if [ -d "$VENV_DIR" ]; then
    echo "🐍 Activating local virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  No virtual environment found"
    echo "💡 Run setup first: ./setup-env.sh"
    
    # Check for Python as fallback
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: Python 3 is required but not installed"
        echo "Install with: brew install python"
        exit 1
    fi
fi

# Check for whisper.cpp binary
if [ ! -f "whisper-cli" ] && [ ! -f "whisper_cpp/build/bin/whisper-cli" ]; then
    echo "❌ Error: whisper.cpp not built"
    echo "Please run: ./build-scripts/build-whisper-cpp.sh"
    exit 1
fi

# Set up environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export WHISPER_SERVICE_TYPE="mac"
export WHISPER_ENGINE="whisper.cpp"

# Default configuration
PORT=${PORT:-5002}
HOST=${HOST:-"0.0.0.0"}
MODEL=${MODEL:-"base.en"}
DEBUG=${DEBUG:-"false"}
METAL=${METAL:-"true"}
COREML=${COREML:-"true"}
THREADS=${THREADS:-4}

echo "📊 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Model: $MODEL"
echo "   Metal: $METAL"
echo "   Core ML: $COREML"
echo "   Threads: $THREADS"
echo "   Debug: $DEBUG"

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."
python3 -c "
import sys
try:
    import flask
    print('✅ Flask available:', flask.__version__)
except ImportError:
    print('❌ Flask not available')
    sys.exit(1)

try:
    import numpy
    print('✅ NumPy available:', numpy.__version__)
except ImportError:
    print('❌ NumPy not available')
    sys.exit(1)

try:
    import soundfile
    print('✅ SoundFile available:', soundfile.__version__)
except ImportError:
    print('❌ SoundFile not available')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "💡 Install missing dependencies with:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check for whisper.cpp capabilities
echo ""
echo "🔍 Checking whisper.cpp capabilities..."
if [ -f "whisper-cli" ]; then
    WHISPER_CLI="./whisper-cli"
elif [ -f "whisper_cpp/build/bin/whisper-cli" ]; then
    WHISPER_CLI="./whisper_cpp/build/bin/whisper-cli"
fi

if [ -n "$WHISPER_CLI" ]; then
    echo "✅ whisper.cpp binary found: $WHISPER_CLI"
    
    # Try to get system info (this will show capabilities)
    if timeout 5 $WHISPER_CLI --help > /dev/null 2>&1; then
        echo "✅ whisper.cpp binary is functional"
    else
        echo "⚠️  whisper.cpp binary may have issues"
    fi
else
    echo "❌ whisper.cpp binary not found"
    exit 1
fi

# Check for models
echo ""
echo "📁 Checking for models..."
MODELS_DIR="../models/ggml"
if [ -d "$MODELS_DIR" ]; then
    MODEL_COUNT=$(find "$MODELS_DIR" -name "*.bin" | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "✅ Found $MODEL_COUNT GGML models in $MODELS_DIR"
        echo "Available models:"
        find "$MODELS_DIR" -name "*.bin" -exec basename {} \; | sort
    else
        echo "⚠️  No GGML models found in $MODELS_DIR"
        echo "💡 Download models with: ./scripts/download-models.sh"
    fi
else
    echo "⚠️  Models directory not found: $MODELS_DIR"
    echo "💡 Download models with: ./scripts/download-models.sh"
fi

# Check for Core ML models (Apple Silicon only)
if [ "$(uname -m)" = "arm64" ]; then
    echo ""
    echo "🧠 Checking for Core ML models..."
    COREML_DIR="../models/cache/coreml"
    if [ -d "$COREML_DIR" ]; then
        COREML_COUNT=$(find "$COREML_DIR" -name "*.mlmodelc" | wc -l)
        if [ "$COREML_COUNT" -gt 0 ]; then
            echo "✅ Found $COREML_COUNT Core ML models (Apple Neural Engine acceleration)"
        else
            echo "💡 No Core ML models found - generate with: ./scripts/generate-coreml-models.sh"
        fi
    else
        echo "💡 Generate Core ML models for ANE acceleration: ./scripts/generate-coreml-models.sh"
    fi
fi

echo ""
echo "🎯 Starting macOS whisper service..."
echo "   Access at: http://$HOST:$PORT"
echo "   Health: http://$HOST:$PORT/health"
echo "   Models: http://$HOST:$PORT/api/models"
echo "   Device Info: http://$HOST:$PORT/api/device-info"
echo ""
echo "🔗 Compatible with orchestration service routing"
echo ""

# Build command line arguments
ARGS=""
if [ "$DEBUG" = "true" ]; then
    ARGS="$ARGS --debug"
fi

if [ "$METAL" = "true" ]; then
    ARGS="$ARGS --metal"
fi

if [ "$COREML" = "false" ]; then
    ARGS="$ARGS --no-coreml"
fi

# Start the service
exec python3 src/main.py \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --threads "$THREADS" \
    $ARGS