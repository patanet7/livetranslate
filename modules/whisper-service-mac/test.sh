#!/bin/bash
# Quick test script for whisper-service-mac

set -e

echo "🧪 Running whisper-service-mac tests"
echo "====================================="

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ] && [ -d "./venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the test runner
python3 run_tests.py "$@"