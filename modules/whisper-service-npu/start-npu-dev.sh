#!/bin/bash
# Development startup script for NPU-optimized Whisper Service

set -e

echo "🚀 Starting NPU-Optimized Whisper Service"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo "❌ Error: Please run this script from the whisper-service-npu directory"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check for NPU drivers (Linux)
if [ -d "/dev/accel" ]; then
    echo "✅ NPU device files detected: $(ls /dev/accel/)"
else
    echo "⚠️  NPU device files not found - will fallback to GPU/CPU"
fi

# Set up environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export WHISPER_SERVICE_TYPE="npu"
export OPENVINO_LOG_LEVEL="2"

# Default configuration
PORT=${PORT:-5001}
HOST=${HOST:-"0.0.0.0"}
DEVICE=${DEVICE:-"auto"}
POWER_PROFILE=${POWER_PROFILE:-"balanced"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "📊 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Device: $DEVICE"
echo "   Power Profile: $POWER_PROFILE"
echo "   Log Level: $LOG_LEVEL"

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."
python3 -c "
import sys
try:
    import openvino
    print('✅ OpenVINO available:', openvino.__version__)
except ImportError:
    print('❌ OpenVINO not available')
    sys.exit(1)

try:
    import openvino_genai
    print('✅ OpenVINO GenAI available')
except ImportError:
    print('❌ OpenVINO GenAI not available')
    sys.exit(1)

try:
    import flask
    print('✅ Flask available:', flask.__version__)
except ImportError:
    print('❌ Flask not available')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "💡 Install missing dependencies with:"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Create models directory if it doesn't exist
if [ ! -d "../models" ]; then
    echo ""
    echo "📁 Creating models directory..."
    mkdir -p ../models/{openvino,ggml,base,phenome,cache}
    echo "✅ Models directory structure created"
fi

echo ""
echo "🎯 Starting NPU service..."
echo "   Access at: http://$HOST:$PORT"
echo "   Health: http://$HOST:$PORT/health"
echo "   Device Info: http://$HOST:$PORT/api/device-info"
echo ""

# Start the service
exec python3 src/main.py \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --power-profile "$POWER_PROFILE" \
    ${LOG_LEVEL:+--debug}