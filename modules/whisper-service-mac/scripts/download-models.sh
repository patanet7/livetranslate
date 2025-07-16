#!/bin/bash
# Download GGML models for whisper.cpp on macOS

set -e

echo "📥 Downloading GGML models for whisper.cpp"
echo "==========================================="

# Check if we're in the right directory
if [ ! -d "whisper_cpp" ]; then
    echo "❌ Error: whisper_cpp directory not found"
    echo "Please run this script from the whisper-service-mac directory"
    exit 1
fi

# Create models directory using the unified structure
MODELS_DIR="../models"
GGML_DIR="$MODELS_DIR/ggml"

echo "📁 Setting up models directory..."
mkdir -p "$GGML_DIR"
mkdir -p "$MODELS_DIR/cache/coreml"

# Default models to download (all multilingual versions)
DEFAULT_MODELS=("tiny" "base" "small" "medium" "large-v3")
MODELS_TO_DOWNLOAD=("${@:-${DEFAULT_MODELS[@]}}")

echo "🎯 Models to download: ${MODELS_TO_DOWNLOAD[*]}"
echo ""

# Function to download a model
download_model() {
    local model_name=$1
    local ggml_file="ggml-${model_name}.bin"
    local target_path="$GGML_DIR/$ggml_file"
    
    if [ -f "$target_path" ]; then
        echo "✅ $model_name already exists, skipping..."
        return 0
    fi
    
    echo "📥 Downloading $model_name..."
    
    # Use whisper.cpp's download script
    cd whisper_cpp
    
    if [ -f "models/download-ggml-model.sh" ]; then
        bash models/download-ggml-model.sh "$model_name"
        
        # Move to unified models directory
        if [ -f "models/$ggml_file" ]; then
            mv "models/$ggml_file" "../$target_path"
            echo "✅ $model_name downloaded and moved to unified models directory"
        else
            echo "❌ Error: Failed to download $model_name"
            return 1
        fi
    else
        echo "❌ Error: download-ggml-model.sh not found in whisper.cpp"
        return 1
    fi
    
    cd ..
}

# Function to get model info
get_model_info() {
    local model_name=$1
    local ggml_file="ggml-${model_name}.bin"
    local model_path="$GGML_DIR/$ggml_file"
    
    if [ -f "$model_path" ]; then
        local size=$(du -h "$model_path" | cut -f1)
        echo "   📊 $model_name: $size"
    fi
}

# Download models
echo "🚀 Starting downloads..."
for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    download_model "$model"
done

echo ""
echo "📊 Downloaded models:"
for model in "${MODELS_TO_DOWNLOAD[@]}"; do
    get_model_info "$model"
done

# Check for Apple Silicon for Core ML generation
if [ "$(uname -m)" = "arm64" ]; then
    echo ""
    echo "🍎 Apple Silicon detected!"
    echo "💡 You can generate Core ML models for faster inference:"
    echo "   ./scripts/generate-coreml-models.sh"
fi

echo ""
echo "✅ Model download complete!"
echo ""
echo "🎯 Test transcription with:"
echo "   ./whisper-cli -m $GGML_DIR/ggml-base.bin -f audio.wav"
echo ""
echo "📁 Models location: $GGML_DIR"
echo "   Use these models with the Mac service API"