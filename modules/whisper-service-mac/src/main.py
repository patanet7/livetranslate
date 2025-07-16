#!/usr/bin/env python3
"""
macOS-Optimized Whisper Service - Main Entry Point

Native whisper.cpp integration with Apple Silicon optimizations for maximum
performance and compatibility with the LiveTranslate orchestration service.

Features:
- Apple Silicon Metal GPU + ANE acceleration
- Complete API compatibility with original whisper-service
- Word-level timestamps with whisper.cpp
- Real-time streaming transcription
- GGML model support with Core ML optimization
"""

import os
import sys
import asyncio
import logging
import argparse
import signal
from pathlib import Path
from typing import Optional, Dict

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper-service-mac.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def check_mac_dependencies():
    """Check if macOS-specific dependencies are available"""
    required_imports = [
        ('soundfile', 'SoundFile'),
        ('numpy', 'NumPy'),
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
    ]
    
    missing_deps = []
    available_deps = []
    
    for module_name, display_name in required_imports:
        try:
            __import__(module_name)
            available_deps.append(display_name)
            logger.info(f"✓ {display_name} available")
        except ImportError:
            missing_deps.append(display_name)
            logger.error(f"✗ {display_name} not available")
    
    if missing_deps:
        logger.error(f"Missing macOS dependencies: {', '.join(missing_deps)}")
        logger.error("Please install required dependencies with: pip install -r requirements.txt")
        return False
    
    logger.info(f"✓ All macOS dependencies available ({len(available_deps)} modules)")
    return True

def detect_mac_hardware():
    """Detect macOS hardware and capabilities"""
    import platform
    
    hardware_info = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "apple_silicon": platform.machine() == "arm64",
        "macos_version": platform.mac_ver()[0] if platform.system() == "Darwin" else None,
        "capabilities": {}
    }
    
    # Check for Apple Silicon specific features
    if hardware_info["apple_silicon"]:
        logger.info("🍎 Apple Silicon Mac detected!")
        hardware_info["capabilities"].update({
            "metal": True,  # Assume Metal available on Apple Silicon
            "coreml": True,  # Assume Core ML available
            "ane": True,     # Apple Neural Engine
            "unified_memory": True,
            "neon": True     # ARM NEON
        })
    else:
        logger.info("💻 Intel Mac detected")
        hardware_info["capabilities"].update({
            "metal": False,
            "coreml": True,  # Core ML available on Intel too
            "ane": False,
            "unified_memory": False,
            "avx": True      # Assume AVX on Intel
        })
    
    # Check whisper.cpp availability
    whisper_cpp_path = Path("whisper_cpp/build/bin/whisper-cli")
    hardware_info["whisper_cpp_available"] = whisper_cpp_path.exists()
    
    if hardware_info["whisper_cpp_available"]:
        logger.info("✅ whisper.cpp binary found")
    else:
        logger.warning("⚠️  whisper.cpp binary not found - run ./build-scripts/build-whisper-cpp.sh")
    
    logger.info(f"Hardware info: {hardware_info}")
    return hardware_info

def setup_mac_environment():
    """Setup macOS-specific environment variables and paths"""
    environment_info = {}
    
    # Set up models directory with macOS structure
    models_dir = os.getenv("WHISPER_MODELS_DIR")
    if not models_dir:
        # Check for unified models directory
        unified_models = Path(__file__).parent.parent.parent.parent / "models"
        if unified_models.exists():
            models_dir = str(unified_models)
            logger.info(f"Using unified models directory: {models_dir}")
        else:
            # Create local models directory
            local_models = Path(__file__).parent.parent / "models"
            local_models.mkdir(exist_ok=True)
            models_dir = str(local_models)
            logger.info(f"Created local models directory: {models_dir}")
    
    os.environ["WHISPER_MODELS_DIR"] = models_dir
    environment_info["models_dir"] = models_dir
    
    # Set up GGML and Core ML paths
    ggml_dir = Path(models_dir) / "ggml"
    coreml_cache_dir = Path(models_dir) / "cache" / "coreml"
    
    ggml_dir.mkdir(parents=True, exist_ok=True)
    coreml_cache_dir.mkdir(parents=True, exist_ok=True)
    
    environment_info["ggml_dir"] = str(ggml_dir)
    environment_info["coreml_cache_dir"] = str(coreml_cache_dir)
    
    # macOS-specific environment variables
    mac_env = {
        "WHISPER_ENGINE": "whisper.cpp",
        "WHISPER_MAC_METAL": os.getenv("WHISPER_MAC_METAL", "true"),
        "WHISPER_MAC_COREML": os.getenv("WHISPER_MAC_COREML", "true"),
        "WHISPER_MAC_THREADS": os.getenv("WHISPER_MAC_THREADS", "4"),
        "WHISPER_SERVICE_TYPE": "mac"
    }
    
    for key, value in mac_env.items():
        os.environ[key] = value
        environment_info[key] = value
    
    logger.info(f"macOS environment configured: {environment_info}")
    return environment_info

def load_mac_config(config_path: Optional[str] = None) -> Dict:
    """Load macOS-specific configuration"""
    if not config_path:
        config_path = Path(__file__).parent.parent / "config" / "mac_config.yaml"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded macOS configuration from {config_path}")
        return config
    except ImportError:
        logger.warning("PyYAML not available, using default configuration")
    except Exception as e:
        logger.warning(f"Could not load macOS config from {config_path}: {e}")
    
    # Default configuration
    logger.info("Using default macOS configuration")
    return {
        "apple_silicon": {
            "metal_enabled": True,
            "coreml_enabled": True,
            "ane_enabled": True,
            "threads": 4
        },
        "models": {
            "default_model": "base.en",
            "auto_download": True
        },
        "api": {
            "host": "0.0.0.0",
            "port": 5002,
            "workers": 1
        },
        "streaming": {
            "buffer_duration": 6.0,
            "inference_interval": 2.0,
            "word_timestamps": True
        }
    }

async def initialize_mac_service(config: Dict, hardware_info: Dict):
    """Initialize the macOS-optimized Whisper service"""
    logger.info("Initializing macOS-optimized Whisper service...")
    
    try:
        # Import macOS components
        from api.api_server import create_mac_app
        
        # Prepare configuration for the app
        app_config = {
            "models_dir": config.get("models", {}).get("models_dir") or "../models",
            "whisper_cpp_path": "whisper_cpp",
            "default_model": config.get("models", {}).get("default_model", "base.en"),
            "hardware_info": hardware_info,
            "mac_config": config
        }
        
        # Create Flask application with whisper.cpp engine
        app = create_mac_app(app_config)
        
        if not app:
            raise Exception("Failed to create Flask application")
        
        logger.info("✅ macOS-optimized Whisper service initialized successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to initialize macOS service: {e}")
        raise

async def start_mac_service():
    """Start the macOS-optimized Whisper service"""
    parser = argparse.ArgumentParser(description="macOS-Optimized Whisper Service")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--port", type=int, default=5002, help="Service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--model", type=str, help="Default model to use")
    parser.add_argument("--metal", action="store_true", help="Force enable Metal acceleration")
    parser.add_argument("--no-coreml", action="store_true", help="Disable Core ML acceleration")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for whisper.cpp")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Check dependencies
    if not check_mac_dependencies():
        sys.exit(1)
    
    # Detect hardware
    hardware_info = detect_mac_hardware()
    
    # Check if whisper.cpp is built
    if not hardware_info["whisper_cpp_available"]:
        logger.error("whisper.cpp not found. Please run:")
        logger.error("  ./build-scripts/build-whisper-cpp.sh")
        sys.exit(1)
    
    # Setup environment
    env_info = setup_mac_environment()
    
    # Load configuration
    config = load_mac_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.setdefault("models", {})["default_model"] = args.model
    
    if args.metal:
        config.setdefault("apple_silicon", {})["metal_enabled"] = True
    
    if args.no_coreml:
        config.setdefault("apple_silicon", {})["coreml_enabled"] = False
    
    if args.threads:
        config.setdefault("apple_silicon", {})["threads"] = args.threads
    
    config.setdefault("api", {}).update({
        "host": args.host,
        "port": args.port
    })
    
    try:
        # Initialize and start service
        app = await initialize_mac_service(config, hardware_info)
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(shutdown_service())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the service
        host = config["api"]["host"]
        port = config["api"]["port"]
        
        logger.info(f"🚀 Starting macOS-optimized Whisper service on {host}:{port}")
        logger.info(f"🍎 Hardware: {hardware_info['architecture']} ({hardware_info['platform']})")
        logger.info(f"⚡ Acceleration: Metal={hardware_info['capabilities'].get('metal', False)}, Core ML={hardware_info['capabilities'].get('coreml', False)}")
        logger.info(f"📁 Models Directory: {env_info['models_dir']}")
        logger.info(f"🔗 Compatible with orchestration service")
        
        # Check for available models
        ggml_dir = Path(env_info["ggml_dir"])
        ggml_models = list(ggml_dir.glob("*.bin"))
        if ggml_models:
            logger.info(f"📊 Found {len(ggml_models)} GGML models")
        else:
            logger.warning("⚠️  No GGML models found. Download with: ./scripts/download-models.sh")
        
        # Run the Flask app
        app.run(host=host, port=port, debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

async def shutdown_service():
    """Gracefully shutdown the macOS service"""
    logger.info("Shutting down macOS-optimized Whisper service...")
    
    try:
        # Any cleanup needed here
        logger.info("✅ macOS service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        asyncio.run(start_mac_service())
    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()