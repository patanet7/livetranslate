#!/bin/bash
# Setup local Python environment for whisper-service-mac

set -e

echo "🐍 Setting up local Python environment for whisper-service-mac"
echo "=============================================================="

# Find the best Python version available
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
    if command -v "$cmd" &> /dev/null; then
        VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        if [[ "$VERSION" != "0.0" ]] && python3 -c "import sys; sys.exit(0 if tuple(map(int, '$VERSION'.split('.'))) >= (3, 8) else 1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            PYTHON_VERSION="$VERSION"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.8+ is required but not found"
    echo "Available Python versions:"
    for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
            echo "   $cmd: $VERSION"
        fi
    done
    echo ""
    echo "💡 You have Python 3.12.2 available. Try:"
    echo "   which python3.12"
    echo "   python3.12 --version"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD (version $PYTHON_VERSION)"

# Create virtual environment in the project directory
VENV_DIR="./venv"
if [ -d "$VENV_DIR" ]; then
    echo "📁 Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " recreate
    if [[ $recreate =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "📦 Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check for system dependencies
echo "🔍 Checking system dependencies..."

# Check for cmake (required for whisper.cpp)
if ! command -v cmake &> /dev/null; then
    echo "⚠️  cmake not found - required for building whisper.cpp"
    echo "📦 Install with: brew install cmake"
    CMAKE_MISSING=1
fi

# Check for git (for submodules)
if ! command -v git &> /dev/null; then
    echo "⚠️  git not found - required for whisper.cpp submodule"
    GIT_MISSING=1
fi

# Summary
echo ""
echo "📋 Setup Summary"
echo "================"
echo "✅ Virtual environment: $VENV_DIR"
echo "✅ Python dependencies: Installed"

if [ -n "$CMAKE_MISSING" ] || [ -n "$GIT_MISSING" ]; then
    echo ""
    echo "⚠️  Missing system dependencies:"
    [ -n "$CMAKE_MISSING" ] && echo "   - cmake (install with: brew install cmake)"
    [ -n "$GIT_MISSING" ] && echo "   - git (install with: brew install git)"
    echo ""
    echo "Please install missing dependencies before building whisper.cpp"
else
    echo "✅ System dependencies: Ready"
fi

echo ""
echo "🚀 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Install cmake if missing: brew install cmake"
echo "3. Build whisper.cpp: ./build-scripts/build-whisper-cpp.sh"
echo "4. Download models: ./scripts/download-models.sh"
echo "5. Start the service: python src/main.py"
echo ""
echo "💡 To deactivate the environment later: deactivate"