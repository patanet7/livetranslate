#!/bin/bash
# Generate Core ML models for Apple Neural Engine acceleration

set -e

echo "🧠 Generating Core ML models for Apple Neural Engine"
echo "===================================================="

# Check for Apple Silicon
if [ "$(uname -m)" != "arm64" ]; then
    echo "⚠️  Warning: Core ML optimization is designed for Apple Silicon"
    echo "This script will still work but benefits will be limited on Intel Macs"
fi

# Check if we're in the right directory
if [ ! -d "whisper_cpp" ]; then
    echo "❌ Error: whisper_cpp directory not found"
    exit 1
fi

# Check for required Python packages
echo "🔍 Checking Python dependencies..."
python3 -c "
import sys
required_packages = ['ane_transformers', 'openai-whisper', 'coremltools']
missing = []

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f'✅ {package} available')
    except ImportError:
        missing.append(package)
        print(f'❌ {package} missing')

if missing:
    print(f'\\n💡 Install missing packages with:')
    print(f'   pip3 install {\" \".join(missing)}')
    sys.exit(1)
else:
    print('\\n✅ All required packages available')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Models to generate Core ML for
MODELS_TO_GENERATE=("${@:-base.en small.en}")
MODELS_DIR="../models"
COREML_CACHE_DIR="$MODELS_DIR/cache/coreml"

echo ""
echo "🎯 Generating Core ML models for: ${MODELS_TO_GENERATE[*]}"
echo "📁 Output directory: $COREML_CACHE_DIR"

mkdir -p "$COREML_CACHE_DIR"

cd whisper_cpp

# Function to generate Core ML model
generate_coreml_model() {
    local model_name=$1
    local coreml_dir="ggml-${model_name}-encoder.mlmodelc"
    local target_path="../$COREML_CACHE_DIR/$coreml_dir"
    
    if [ -d "$target_path" ]; then
        echo "✅ Core ML model for $model_name already exists, skipping..."
        return 0
    fi
    
    echo "🧠 Generating Core ML model for $model_name..."
    
    # Check if the generation script exists
    if [ -f "models/generate-coreml-model.sh" ]; then
        bash models/generate-coreml-model.sh "$model_name"
        
        # Move to unified models directory
        if [ -d "models/$coreml_dir" ]; then
            mv "models/$coreml_dir" "$target_path"
            echo "✅ Core ML model for $model_name generated and moved"
        else
            echo "❌ Error: Failed to generate Core ML model for $model_name"
            return 1
        fi
    else
        echo "❌ Error: generate-coreml-model.sh not found"
        return 1
    fi
}

# Generate Core ML models
echo "🚀 Starting Core ML generation..."
for model in "${MODELS_TO_GENERATE[@]}"; do
    generate_coreml_model "$model"
done

cd ..

echo ""
echo "📊 Generated Core ML models:"
for model in "${MODELS_TO_GENERATE[@]}"; do
    local coreml_dir="ggml-${model}-encoder.mlmodelc"
    local model_path="$COREML_CACHE_DIR/$coreml_dir"
    
    if [ -d "$model_path" ]; then
        local size=$(du -sh "$model_path" | cut -f1)
        echo "   🧠 $model: $size"
    fi
done

echo ""
echo "✅ Core ML generation complete!"
echo ""
echo "🎯 Test Core ML acceleration with:"
echo "   ./whisper-cli -m ../models/ggml/ggml-base.en.bin -f audio.wav"
echo "   (Core ML models will be automatically detected and used)"
echo ""
echo "⚡ Expected performance improvement:"
echo "   • 3-5x faster encoder inference on Apple Neural Engine"
echo "   • Lower power consumption"
echo "   • Better thermal management"
echo ""
echo "📁 Core ML models location: $COREML_CACHE_DIR"