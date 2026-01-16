#!/bin/bash
#
# Start Translation Service with Ollama backend (fastest startup)
# Uses local Ollama server - no model download required
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Translation Service with Ollama backend${NC}"

cd "$(dirname "$0")"

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Ollama not running. Starting Ollama...${NC}"
    ollama serve &
    sleep 3
fi

# Show available models
echo -e "${GREEN}Available Ollama models:${NC}"
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print('  ' + '\n  '.join(m['name'] for m in models))"

# Set environment for Ollama-first startup
export TRANSLATION_MODEL="ollama"
export OLLAMA_ENABLE=true
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export SKIP_LOCAL_MODEL=true

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $OLLAMA_MODEL"
echo "  Base URL: $OLLAMA_BASE_URL"
echo "  Port: 5003"

echo -e "${GREEN}Starting service...${NC}"
python src/api_server.py
