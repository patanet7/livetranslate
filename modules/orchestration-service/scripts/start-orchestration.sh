#!/bin/bash

# Start Orchestration Service
# This script starts the orchestration service with proper Python environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Livetranslate Orchestration Service...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

# Determine Python executable
PY_BIN="python"
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY_BIN="$CONDA_PREFIX/bin/python"
    echo -e "${GREEN}Using conda environment: $CONDA_PREFIX${NC}"
elif command -v python3 &> /dev/null; then
    PY_BIN="python3"
    echo -e "${YELLOW}Using system python3${NC}"
else
    echo -e "${RED}Error: No suitable Python executable found${NC}"
    exit 1
fi

# Check Python version
PY_VERSION=$($PY_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PY_BIN -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PY_BIN -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python >= 3.8 required. Current: $PY_VERSION${NC}"
    echo -e "${YELLOW}Please activate the conda environment: conda activate livetranslate-orchestration${NC}"
    exit 1
fi

echo -e "${GREEN}Python version: $PY_VERSION${NC}"

# Check if we're in the right directory
if [ ! -f "src/main_fastapi.py" ]; then
    echo -e "${RED}Error: main_fastapi.py not found. Please run this script from the orchestration-service directory${NC}"
    exit 1
fi

# Check if dependencies are installed
echo -e "${GREEN}Checking dependencies...${NC}"
if ! $PY_BIN -c "import fastapi, uvicorn, websockets, requests, httpx, pydantic, yaml, numpy, soundfile, scipy, librosa, pydub, aiosqlite" 2>/dev/null; then
    echo -e "${YELLOW}Some dependencies missing. Installing...${NC}"
    if [ -f "requirements.txt" ]; then
        $PY_BIN -m pip install -r requirements.txt
    else
        echo -e "${RED}requirements.txt not found${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Starting orchestration service on http://localhost:8000${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}"

# Start the service
$PY_BIN -m uvicorn src.main_fastapi:app --host 0.0.0.0 --port 8000 --reload
