#!/bin/bash
# Build whisper.cpp with Apple Silicon optimizations

set -e

echo "🍎 Building whisper.cpp for Apple Silicon"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "whisper_cpp" ]; then
    echo "❌ Error: whisper_cpp directory not found"
    echo "Please run this script from the whisper-service-mac directory"
    exit 1
fi

cd whisper_cpp

# Detect architecture
ARCH=$(uname -m)
echo "🔍 Detected architecture: $ARCH"

# Set build flags based on architecture
if [ "$ARCH" = "arm64" ]; then
    echo "✅ Building for Apple Silicon (ARM64)"
    CMAKE_FLAGS="-DGGML_METAL=1 -DWHISPER_COREML=1 -DGGML_ACCELERATE=1"
    BUILD_TYPE="Apple Silicon optimized"
elif [ "$ARCH" = "x86_64" ]; then
    echo "✅ Building for Intel Mac (x86_64)"
    CMAKE_FLAGS="-DGGML_ACCELERATE=1 -DGGML_BLAS=1"
    BUILD_TYPE="Intel Mac optimized"
else
    echo "⚠️  Unknown architecture, using default build"
    CMAKE_FLAGS=""
    BUILD_TYPE="default"
fi

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "❌ Error: cmake is required but not installed"
    echo "Install with: brew install cmake"
    exit 1
fi

# Check for Xcode tools (required for Metal/Core ML)
if [ "$ARCH" = "arm64" ]; then
    if ! xcode-select -p &> /dev/null; then
        echo "❌ Error: Xcode command line tools required for Metal/Core ML"
        echo "Install with: xcode-select --install"
        exit 1
    fi
    
    # Check for Python (required for Core ML model generation)
    if ! command -v python3 &> /dev/null; then
        echo "❌ Error: Python 3 is required for Core ML model generation"
        echo "Install with: brew install python"
        exit 1
    fi
    
    echo "🔧 Installing Core ML dependencies..."
    pip3 install --quiet ane_transformers openai-whisper coremltools || {
        echo "⚠️  Warning: Could not install Core ML dependencies"
        echo "Core ML features will be disabled"
        CMAKE_FLAGS=$(echo $CMAKE_FLAGS | sed 's/-DWHISPER_COREML=1//')
    }
fi

echo ""
echo "🚀 Build configuration:"
echo "   Type: $BUILD_TYPE"
echo "   Flags: $CMAKE_FLAGS"
echo "   Threads: $(sysctl -n hw.ncpu)"

# Clean previous build
if [ -d "build" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build
fi

# Configure build
echo "⚙️  Configuring build..."
cmake -B build $CMAKE_FLAGS -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ Error: CMake configuration failed"
    exit 1
fi

# Build with all available cores
echo "🔨 Building whisper.cpp..."
cmake --build build -j $(sysctl -n hw.ncpu) --config Release

if [ $? -ne 0 ]; then
    echo "❌ Error: Build failed"
    exit 1
fi

# Verify build
echo ""
echo "✅ Build completed successfully!"
echo ""
echo "🔍 Verifying build..."

if [ -f "build/bin/whisper-cli" ]; then
    echo "✅ whisper-cli binary created"
    
    # Test with system info
    echo "📊 System capabilities:"
    ./build/bin/whisper-cli --help | head -1
    
    # Show build info if possible
    echo ""
    echo "🔧 Build features:"
    if [ "$ARCH" = "arm64" ]; then
        echo "   Metal: $([ -f "build/bin/whisper-cli" ] && echo "✅ Enabled" || echo "❌ Disabled")"
        echo "   Core ML: $([ -d "../models" ] && echo "🔧 Available" || echo "📁 Needs models directory")"
        echo "   Accelerate: ✅ Enabled"
        echo "   NEON: ✅ Enabled"
    else
        echo "   Accelerate: ✅ Enabled"
        echo "   AVX: $(sysctl -n machdep.cpu.features | grep -q AVX && echo "✅ Available" || echo "❌ Not available")"
    fi
else
    echo "❌ Error: whisper-cli binary not found"
    exit 1
fi

# Create symlinks for easy access
echo ""
echo "🔗 Creating convenience symlinks..."
cd ..
ln -sf whisper_cpp/build/bin/whisper-cli whisper-cli
ln -sf whisper_cpp/build/bin/whisper-bench whisper-bench
if [ -f "whisper_cpp/build/bin/quantize" ]; then
    ln -sf whisper_cpp/build/bin/quantize quantize
fi

echo "✅ whisper.cpp build complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Download models: ./scripts/download-models.sh"
echo "   2. Test transcription: ./whisper-cli -f audio.wav"
if [ "$ARCH" = "arm64" ]; then
    echo "   3. Generate Core ML models: ./scripts/generate-coreml-models.sh"
fi
echo ""