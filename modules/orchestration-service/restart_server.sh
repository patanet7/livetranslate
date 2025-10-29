#!/bin/bash
echo "🔄 Restarting orchestration service..."

# Find and kill existing process
PID=$(ps aux | grep "main_fastapi.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PID" ]; then
    echo "Stopping existing process (PID: $PID)..."
    kill $PID
    sleep 2
fi

# Start new process with Poetry
echo "Starting orchestration service with Poetry environment..."
poetry run python src/main_fastapi.py
