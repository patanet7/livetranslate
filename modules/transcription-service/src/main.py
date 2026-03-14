#!/usr/bin/env python3
"""
Main entry point for the Whisper Service

This service provides NPU-optimized speech-to-text transcription with real-time streaming.
It coordinates with the orchestration service and provides both REST API and WebSocket endpoints.

Features:
- NPU/GPU/CPU acceleration with automatic fallback
- Real-time streaming transcription
- Voice Activity Detection (VAD)
- Session management and persistence
- WebSocket communication for real-time updates
- Integration with LiveTranslate orchestration service
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from livetranslate_common.logging import get_logger, setup_logging

# Configure structured logging (must happen before any get_logger calls)
setup_logging(service_name="whisper")

logger = get_logger()


def check_dependencies():
    """Check if all required dependencies are available"""
    required_imports = [
        ("openvino", "OpenVINO"),
        ("openvino_genai", "OpenVINO GenAI"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("webrtcvad", "WebRTC VAD"),
        ("flask", "Flask"),
        ("flask_socketio", "Flask-SocketIO"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]

    missing_deps = []
    for module_name, display_name in required_imports:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name} available")
        except ImportError:
            missing_deps.append(display_name)
            logger.error(f"✗ {display_name} not available")

    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install required dependencies with: pip install -r requirements.txt")
        return False

    logger.info("✓ All dependencies available")
    return True


def detect_hardware():
    """Detect available hardware acceleration"""
    try:
        import openvino as ov

        core = ov.Core()
        available_devices = core.available_devices

        logger.info(f"Available OpenVINO devices: {available_devices}")

        if "NPU" in available_devices:
            logger.info("🚀 Intel NPU detected! Using NPU acceleration")
            return "NPU"
        elif "GPU" in available_devices:
            logger.info("⚡ GPU detected! Using GPU acceleration")
            return "GPU"
        else:
            logger.info("💻 Using CPU fallback")
            return "CPU"

    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")
        return "CPU"


def setup_environment():
    """Setup environment variables and paths"""
    # Set up models directory
    models_dir = os.getenv("WHISPER_MODELS_DIR")
    if not models_dir:
        # Check for local models directory in whisper-service first
        local_models = Path(__file__).parent.parent / "models"
        if local_models.exists():
            models_dir = str(local_models)
            os.environ["WHISPER_MODELS_DIR"] = models_dir
            logger.info(f"Using local OpenVINO models directory: {models_dir}")
        else:
            # Check legacy location as fallback
            legacy_models = (
                Path(__file__).parent.parent.parent
                / "legacy"
                / "python"
                / "whisper-npu-server"
                / "models"
            )
            if legacy_models.exists():
                models_dir = str(legacy_models)
                os.environ["WHISPER_MODELS_DIR"] = models_dir
                logger.info(f"Using legacy models directory: {models_dir}")
            else:
                # Fall back to home directory
                models_dir = os.path.expanduser("~/.whisper/models")
                os.environ["WHISPER_MODELS_DIR"] = models_dir
                logger.info(f"Using default models directory: {models_dir}")

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Set up session data directory
    session_dir = os.getenv("SESSION_DIR", str(Path(__file__).parent.parent / "session_data"))
    os.makedirs(session_dir, exist_ok=True)
    os.environ["SESSION_DIR"] = session_dir

    # Set up logs directory
    logs_dir = str(Path(__file__).parent.parent / "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Detect and set optimal device
    env_device = os.getenv("OPENVINO_DEVICE")
    if env_device:
        logger.info(f"Using OPENVINO_DEVICE from environment: {env_device}")
    else:
        optimal_device = detect_hardware()
        os.environ["OPENVINO_DEVICE"] = optimal_device
        logger.info(f"Auto-detected and set OPENVINO_DEVICE to {optimal_device}")

    # Set default configurations if not provided
    default_env = {
        "WHISPER_DEFAULT_MODEL": "whisper-base",
        "SAMPLE_RATE": "16000",
        "BUFFER_DURATION": "6.0",
        "INFERENCE_INTERVAL": "3.0",
        "ENABLE_VAD": "true",
        "MIN_INFERENCE_INTERVAL": "0.2",
        "MAX_CONCURRENT_REQUESTS": "10",
        "LOG_LEVEL": "INFO",
        "HOST": "0.0.0.0",
        "PORT": "5001",
    }

    for key, default_value in default_env.items():
        if not os.getenv(key):
            os.environ[key] = default_value


def list_available_models():
    """List available models in the models directory"""
    models_dir = os.getenv("WHISPER_MODELS_DIR")
    if not models_dir or not os.path.exists(models_dir):
        logger.warning("Models directory not found")
        return []

    models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            # Check if it looks like a valid model directory
            if any(f.endswith(".xml") for f in os.listdir(model_path)):
                models.append(item)

    if models:
        logger.info(f"Available models: {', '.join(models)}")
    else:
        logger.warning(
            "No OpenVINO models found. Please download models or mount models directory."
        )
        logger.info("The service will work in simulation mode without real transcription.")

    return models


async def start_service():
    """Start the whisper service"""
    logger.info("🎤 Starting LiveTranslate Whisper Service...")

    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False

    # Setup environment
    setup_environment()

    # List available models
    models = list_available_models()

    # Import and start the API server
    try:
        logger.info("Starting API server...")
        from api_server import app, initialize_service, socketio

        # Initialize the service
        try:
            await initialize_service()
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

        # Get configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "5001"))
        debug = os.getenv("DEBUG", "false").lower() == "true"

        logger.info(f"🚀 Whisper Service starting on {host}:{port}")
        logger.info(f"📊 Hardware: {os.getenv('OPENVINO_DEVICE')}")
        logger.info(f"🧠 Default model: {os.getenv('WHISPER_DEFAULT_MODEL')}")
        logger.info(f"📁 Models directory: {os.getenv('WHISPER_MODELS_DIR')}")
        logger.info(f"📂 Session directory: {os.getenv('SESSION_DIR')}")

        if models:
            logger.info(f"✅ Found {len(models)} OpenVINO models")
        else:
            logger.warning("⚠️  No models found - running in simulation mode")

        logger.info("🌐 Service endpoints:")
        logger.info(f"   Health: http://{host}:{port}/health")
        logger.info(f"   Models: http://{host}:{port}/models")
        logger.info(f"   Transcribe: http://{host}:{port}/transcribe")
        logger.info(f"   WebSocket: ws://{host}:{port}/ws")

        # Start the server
        socketio.run(
            app, host=host, port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True
        )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="LiveTranslate Whisper Service - NPU-optimized speech-to-text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with auto-detected hardware
  python main.py --device npu      # Force NPU usage
  python main.py --device gpu      # Force GPU usage
  python main.py --device cpu      # Force CPU usage
  python main.py --port 5002       # Start on different port
  python main.py --debug           # Enable debug mode
        """,
    )

    parser.add_argument("--host", default=None, help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: 5001)")
    parser.add_argument(
        "--device", choices=["auto", "npu", "gpu", "cpu"], help="Device to use for inference"
    )
    parser.add_argument("--model", default=None, help="Default model to use")
    parser.add_argument("--models-dir", default=None, help="Directory containing models")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")

    args = parser.parse_args()

    # Set environment variables from arguments
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.device:
        os.environ["OPENVINO_DEVICE"] = args.device.upper()
    if args.model:
        os.environ["WHISPER_DEFAULT_MODEL"] = args.model
    if args.models_dir:
        os.environ["WHISPER_MODELS_DIR"] = args.models_dir
    if args.debug:
        os.environ["DEBUG"] = "true"
        logging.getLogger().setLevel(logging.DEBUG)  # lower stdlib root for third-party libs
    if args.workers:
        os.environ["WORKERS"] = str(args.workers)

    # Handle special commands
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)

    if args.list_models:
        setup_environment()
        models = list_available_models()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model}")
        sys.exit(0)

    # Start the service
    try:
        # Try to get existing event loop or create new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop in thread
                import concurrent.futures

                def run_service():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(start_service())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_service)
                    success = future.result()
            else:
                success = loop.run_until_complete(start_service())
        except RuntimeError:
            # No event loop exists, create one
            success = asyncio.run(start_service())

        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
