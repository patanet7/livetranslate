#!/bin/bash
# Quick start script for translation testing

echo "=================================="
echo "🚀 STARTING TRANSLATION TEST SERVICES"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "test_loopback_translation.py" ]; then
    echo "❌ Error: Run this from the livetranslate root directory"
    exit 1
fi

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "Starting services in separate terminal windows..."
echo ""

# Function to start a service in a new terminal (macOS)
start_service_mac() {
    local service_name=$1
    local service_dir=$2
    local service_cmd=$3

    echo -e "${YELLOW}Starting ${service_name}...${NC}"

    osascript <<EOF
tell application "Terminal"
    do script "cd '${PWD}/${service_dir}' && echo '🚀 Starting ${service_name}...' && ${service_cmd}"
    set custom title of front window to "${service_name}"
end tell
EOF
}

# 1. Start Translation Service (with Ollama backend)
start_service_mac "Translation Service" \
    "modules/translation-service" \
    "python3 src/api_server_fastapi.py"

echo "   ✅ Translation service starting on port 5003"
sleep 2

# 2. Start Whisper Service
start_service_mac "Whisper Service" \
    "modules/whisper-service" \
    "python3 src/main.py"

echo "   ✅ Whisper service starting on port 5001"
sleep 2

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start (10 seconds)..."
sleep 10

echo ""
echo "${GREEN}=================================="
echo "✅ SERVICES STARTED"
echo "==================================${NC}"
echo ""
echo "📋 Service URLs:"
echo "   Translation: http://localhost:5003/docs"
echo "   Whisper: http://localhost:5001/health"
echo ""
echo "🎯 Next steps:"
echo "   1. Make sure you have a loopback device (BlackHole, Soundflower)"
echo "   2. Run the test: python3 test_loopback_translation.py"
echo "   3. Play some audio/video on your system"
echo "   4. Watch live translations appear!"
echo ""
echo "💡 To install BlackHole loopback device:"
echo "   brew install blackhole-2ch"
echo ""
echo "🛑 To stop services: Close the terminal windows or press Ctrl+C in each"
echo ""
