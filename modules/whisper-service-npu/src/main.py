#!/usr/bin/env python3
"""
NPU-Optimized Whisper Service - Main Entry Point

This service provides Intel NPU-optimized speech-to-text transcription with:
- Intel NPU acceleration via OpenVINO
- Automatic hardware fallback (NPU → GPU → CPU)  
- Power management and thermal optimization
- Real-time streaming with NPU optimization
- Model conversion and management

Hardware Target: Intel NPU (Core Ultra series) with fallback support
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
import structlog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper-service-npu.log', encoding='utf-8')
    ]
)

logger = structlog.get_logger(__name__)

def check_npu_dependencies():
    """Check if NPU-specific dependencies are available"""
    required_imports = [
        ('openvino', 'OpenVINO Runtime'),
        ('openvino_genai', 'OpenVINO GenAI'),
        ('librosa', 'Librosa'),
        ('soundfile', 'SoundFile'),
        ('webrtcvad', 'WebRTC VAD'),
        ('flask', 'Flask'),
        ('flask_socketio', 'Flask-SocketIO'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('yaml', 'PyYAML'),
        ('redis', 'Redis')
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
        logger.error(f"Missing NPU dependencies: {', '.join(missing_deps)}")
        logger.error("Please install required dependencies with: pip install -r requirements-npu.txt")
        return False
    
    logger.info(f"✓ All NPU dependencies available ({len(available_deps)} modules)")
    return True

def detect_npu_hardware():
    """Detect Intel NPU and available hardware acceleration"""
    hardware_info = {
        "npu": False,
        "gpu": False,
        "cpu": True,
        "selected_device": "CPU",
        "available_devices": [],
        "npu_details": None
    }
    
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        
        hardware_info["available_devices"] = available_devices
        logger.info(f"OpenVINO available devices: {available_devices}")
        
        # Check for NPU
        if "NPU" in available_devices:
            hardware_info["npu"] = True
            hardware_info["selected_device"] = "NPU"
            logger.info("🚀 Intel NPU detected! Using NPU acceleration")
            
            # Get NPU details
            try:
                npu_properties = core.get_property("NPU", "SUPPORTED_PROPERTIES")
                hardware_info["npu_details"] = str(npu_properties)
                logger.info(f"NPU properties: {npu_properties}")
            except Exception as e:
                logger.warning(f"Could not get NPU details: {e}")
                
        elif "GPU" in available_devices:
            hardware_info["gpu"] = True
            hardware_info["selected_device"] = "GPU"
            logger.info("⚡ GPU detected! Using GPU acceleration (NPU fallback)")
        else:
            logger.info("💻 Using CPU fallback (no NPU/GPU detected)")
            
    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")
        logger.info("💻 Defaulting to CPU-only mode")
    
    return hardware_info

def setup_npu_environment():
    """Setup NPU-specific environment variables and paths"""
    environment_info = {}
    
    # Set up models directory with NPU structure
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
    
    # Set up NPU-specific OpenVINO cache
    cache_dir = os.getenv("OPENVINO_CACHE_DIR") or str(Path(models_dir) / "cache" / "openvino")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["OPENVINO_CACHE_DIR"] = cache_dir
    environment_info["cache_dir"] = cache_dir
    
    # NPU-specific environment variables
    npu_env = {
        "OPENVINO_LOG_LEVEL": os.getenv("OPENVINO_LOG_LEVEL", "2"),  # INFO level
        "OPENVINO_DEVICE": os.getenv("OPENVINO_DEVICE", "AUTO"),     # Auto-detect
        "NPU_COMPILER_TYPE": os.getenv("NPU_COMPILER_TYPE", "AUTO"), # NPU compiler
        "WHISPER_NPU_PRECISION": os.getenv("WHISPER_NPU_PRECISION", "FP16"),
        "WHISPER_NPU_POWER_PROFILE": os.getenv("WHISPER_NPU_POWER_PROFILE", "balanced")
    }
    
    for key, value in npu_env.items():
        os.environ[key] = value
        environment_info[key] = value
    
    logger.info(f"NPU environment configured: {environment_info}")
    return environment_info

def load_npu_config(config_path: Optional[str] = None) -> Dict:
    """Load NPU-specific configuration"""
    if not config_path:
        config_path = Path(__file__).parent.parent / "config" / "npu_config.yaml"
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded NPU configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Could not load NPU config from {config_path}: {e}")
        logger.info("Using default NPU configuration")
        return {
            "npu": {
                "device_priority": ["NPU", "GPU", "CPU"],
                "precision": "FP16",
                "power_profile": "balanced"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 5001,
                "workers": 1
            }
        }

async def initialize_npu_service(config: Dict, hardware_info: Dict):
    """Initialize the NPU-optimized Whisper service"""
    logger.info("Initializing NPU-optimized Whisper service...")
    
    try:
        # Import extracted NPU components
        from core.npu_model_manager import ModelManager
        from core.npu_engine import WhisperService, create_whisper_service
        from api.api_server import app
        
        # Initialize model manager with NPU settings
        models_dir = config.get("models", {}).get("models_dir", "./models")
        device = hardware_info.get("selected_device", "CPU")
        
        model_manager = ModelManager(
            models_dir=models_dir,
            device=device,
            default_model=config.get("models", {}).get("default_model", "whisper-base")
        )
        
        # Initialize whisper service with NPU optimization
        whisper_service = await create_whisper_service(config={
            "model_manager": model_manager,
            "device": device,
            "hardware_info": hardware_info
        })
        
        # Configure Flask app with whisper service
        app.config["WHISPER_SERVICE"] = whisper_service
        app.config["NPU_CONFIG"] = config
        app.config["HARDWARE_INFO"] = hardware_info
        
        logger.info("✅ NPU-optimized Whisper service initialized successfully")
        return app, whisper_service
        
    except Exception as e:
        logger.error(f"Failed to initialize NPU service: {e}")
        raise

async def start_npu_service():
    """Start the NPU-optimized Whisper service"""
    parser = argparse.ArgumentParser(description="NPU-Optimized Whisper Service")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--port", type=int, default=5001, help="Service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--device", type=str, choices=["auto", "npu", "gpu", "cpu"], 
                       default="auto", help="Force specific device")
    parser.add_argument("--power-profile", type=str, 
                       choices=["performance", "balanced", "power_saver"], 
                       default="balanced", help="NPU power profile")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Check dependencies
    if not check_npu_dependencies():
        sys.exit(1)
    
    # Detect hardware
    hardware_info = detect_npu_hardware()
    
    # Override device if specified
    if args.device != "auto":
        hardware_info["selected_device"] = args.device.upper()
        logger.info(f"Device override: using {args.device.upper()}")
    
    # Setup environment
    env_info = setup_npu_environment()
    
    # Load configuration
    config = load_npu_config(args.config)
    
    # Override config with command line arguments
    if args.power_profile:
        config.setdefault("npu", {})["power_profile"] = args.power_profile
    
    config.setdefault("api", {}).update({
        "host": args.host,
        "port": args.port
    })
    
    try:
        # Initialize and start service
        app, npu_engine = await initialize_npu_service(config, hardware_info)
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(shutdown_service(npu_engine))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the service
        host = config["api"]["host"]
        port = config["api"]["port"]
        
        logger.info(f"🚀 Starting NPU-optimized Whisper service on {host}:{port}")
        logger.info(f"📊 Hardware: {hardware_info['selected_device']}")
        logger.info(f"⚡ Power Profile: {config.get('npu', {}).get('power_profile', 'balanced')}")
        logger.info(f"📁 Models Directory: {env_info['models_dir']}")
        
        # Run the Flask app
        app.run(host=host, port=port, debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

async def shutdown_service(npu_engine):
    """Gracefully shutdown the NPU service"""
    logger.info("Shutting down NPU-optimized Whisper service...")
    
    try:
        if npu_engine:
            await npu_engine.shutdown()
        logger.info("✅ NPU service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    try:
        asyncio.run(start_npu_service())
    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()