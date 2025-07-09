#!/bin/bash
# Startup script for Triton-based Translation Service
# Starts both Triton Inference Server and the Translation Service API

set -e

echo "Starting Triton-based Translation Service..."

# Set default environment variables
export MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.9"}
export ENFORCE_EAGER=${ENFORCE_EAGER:-"false"}

# Triton server configuration
export TRITON_MODEL_REPOSITORY=${TRITON_MODEL_REPOSITORY:-"/app/model_repository"}
export TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-"8000"}
export TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-"8001"}
export TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-"8002"}

# Translation service configuration
export TRANSLATION_SERVICE_PORT=${TRANSLATION_SERVICE_PORT:-"5003"}
export TRITON_BASE_URL=${TRITON_BASE_URL:-"http://localhost:${TRITON_HTTP_PORT}"}
export INFERENCE_BACKEND=${INFERENCE_BACKEND:-"triton"}

echo "Configuration:"
echo "  Model: ${MODEL_NAME}"
echo "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "  Max Model Length: ${MAX_MODEL_LEN}"
echo "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  Triton HTTP Port: ${TRITON_HTTP_PORT}"
echo "  Translation Service Port: ${TRANSLATION_SERVICE_PORT}"

# Function to wait for Triton to be ready
wait_for_triton() {
    echo "Waiting for Triton server to be ready..."
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://localhost:${TRITON_HTTP_PORT}/v2/health" >/dev/null 2>&1; then
            echo "Triton server is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Triton not ready yet, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Triton server failed to start within expected time"
    return 1
}

# Function to wait for model to be ready
wait_for_model() {
    echo "Waiting for vLLM model to be ready..."
    local max_attempts=120  # vLLM model loading can take a while
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://localhost:${TRITON_HTTP_PORT}/v2/models/vllm_model/ready" >/dev/null 2>&1; then
            echo "vLLM model is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Model not ready yet, waiting..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: vLLM model failed to load within expected time"
    return 1
}

# Function to start Triton server
start_triton() {
    echo "Starting Triton Inference Server..."
    
    # Start Triton server in background
    tritonserver \
        --model-repository="${TRITON_MODEL_REPOSITORY}" \
        --http-port="${TRITON_HTTP_PORT}" \
        --grpc-port="${TRITON_GRPC_PORT}" \
        --metrics-port="${TRITON_METRICS_PORT}" \
        --allow-http=true \
        --allow-grpc=true \
        --allow-metrics=true \
        --log-verbose=1 \
        --log-info=true \
        --log-warning=true \
        --log-error=true \
        --strict-model-config=false \
        --strict-readiness=false \
        --exit-timeout-secs=30 &
    
    # Store Triton PID
    TRITON_PID=$!
    echo "Triton server started with PID: ${TRITON_PID}"
    
    # Wait for Triton to be ready
    if ! wait_for_triton; then
        echo "Failed to start Triton server"
        kill $TRITON_PID 2>/dev/null || true
        exit 1
    fi
    
    # Wait for model to be ready
    if ! wait_for_model; then
        echo "Failed to load vLLM model"
        kill $TRITON_PID 2>/dev/null || true
        exit 1
    fi
}

# Function to start translation service
start_translation_service() {
    echo "Starting Translation Service API..."
    
    # Start translation service in background
    cd /app
    python src/api_server.py \
        --host 0.0.0.0 \
        --port "${TRANSLATION_SERVICE_PORT}" \
        --backend triton &
    
    # Store Translation Service PID
    TRANSLATION_PID=$!
    echo "Translation service started with PID: ${TRANSLATION_PID}"
    
    # Wait a moment for service to start
    sleep 5
    
    # Check if translation service is healthy
    if curl -f "http://localhost:${TRANSLATION_SERVICE_PORT}/api/health" >/dev/null 2>&1; then
        echo "Translation service is ready!"
    else
        echo "WARNING: Translation service may not be ready yet"
    fi
}

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    
    if [ ! -z "$TRANSLATION_PID" ]; then
        echo "Stopping translation service (PID: $TRANSLATION_PID)"
        kill $TRANSLATION_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$TRITON_PID" ]; then
        echo "Stopping Triton server (PID: $TRITON_PID)"
        kill $TRITON_PID 2>/dev/null || true
        # Give Triton time to shutdown gracefully
        sleep 10
        kill -9 $TRITON_PID 2>/dev/null || true
    fi
    
    echo "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start services
start_triton
start_translation_service

echo "All services started successfully!"
echo "Triton Inference Server: http://localhost:${TRITON_HTTP_PORT}"
echo "Translation Service API: http://localhost:${TRANSLATION_SERVICE_PORT}"
echo "Metrics: http://localhost:${TRITON_METRICS_PORT}/metrics"

# Keep the script running and monitor processes
while true; do
    # Check if Triton is still running
    if ! kill -0 $TRITON_PID 2>/dev/null; then
        echo "ERROR: Triton server has stopped unexpectedly"
        cleanup
    fi
    
    # Check if Translation service is still running
    if ! kill -0 $TRANSLATION_PID 2>/dev/null; then
        echo "ERROR: Translation service has stopped unexpectedly"
        cleanup
    fi
    
    sleep 30
done