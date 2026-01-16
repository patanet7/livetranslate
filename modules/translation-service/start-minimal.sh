#!/bin/bash
#
# Minimal Translation Service startup using FastAPI version
# Skips heavy model loading - uses Ollama directly
#

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting Minimal Translation Service (FastAPI + Ollama)${NC}"

cd "$(dirname "$0")"

# Set minimal config
export OLLAMA_ENABLE=true
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434/v1}"
export PORT=5003
export LOG_LEVEL=INFO

echo -e "${GREEN}Config: Ollama model=$OLLAMA_MODEL, Port=$PORT${NC}"

# Use FastAPI version if available (faster startup)
if [ -f "src/api_server_fastapi.py" ]; then
    echo -e "${GREEN}Using FastAPI server (faster)${NC}"
    python -m uvicorn src.api_server_fastapi:app --host 0.0.0.0 --port 5003 --reload
else
    echo -e "${GREEN}Using Flask server${NC}"
    python src/api_server.py
fi
