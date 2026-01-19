#!/usr/bin/env python3
"""
Start Whisper Service Integration

This script starts both the whisper service and orchestration service
for testing the new whisper integration.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manage whisper and orchestration services"""

    def __init__(self):
        self.services = {}
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping services...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)

    def start_whisper_service(self):
        """Start the whisper service"""
        logger.info("Starting Whisper Service...")

        whisper_dir = Path(__file__).parent / "modules" / "whisper-service"
        main_script = whisper_dir / "src" / "main.py"

        if not main_script.exists():
            logger.error(f"Whisper service main script not found: {main_script}")
            return None

        # Set environment variables for whisper service
        env = os.environ.copy()
        env.update(
            {
                "OPENVINO_DEVICE": "AUTO",  # Auto-detect NPU/GPU/CPU
                "WHISPER_DEFAULT_MODEL": "whisper-base",
                "PORT": "5001",
                "HOST": "0.0.0.0",
                "LOG_LEVEL": "INFO",
                "ENABLE_VAD": "true",
            }
        )

        try:
            process = subprocess.Popen(
                [sys.executable, str(main_script), "--port", "5001"],
                cwd=str(whisper_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.services["whisper"] = process
            logger.info(f"Whisper service started with PID {process.pid}")

            # Start log monitoring in background
            def monitor_whisper_logs():
                for line in process.stdout:
                    if self.running:
                        logger.info(f"[WHISPER] {line.strip()}")

            log_thread = threading.Thread(target=monitor_whisper_logs, daemon=True)
            log_thread.start()

            return process

        except Exception as e:
            logger.error(f"Failed to start whisper service: {e}")
            return None

    def start_orchestration_service(self):
        """Start the orchestration service"""
        logger.info("Starting Orchestration Service...")

        orch_dir = Path(__file__).parent / "modules" / "orchestration-service"
        main_script = orch_dir / "src" / "orchestration_service.py"

        if not main_script.exists():
            logger.error(f"Orchestration service main script not found: {main_script}")
            return None

        # Set environment variables for orchestration service
        env = os.environ.copy()
        env.update({"PORT": "3000", "HOST": "0.0.0.0", "LOG_LEVEL": "INFO"})

        try:
            process = subprocess.Popen(
                [sys.executable, str(main_script), "--port", "3000"],
                cwd=str(orch_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.services["orchestration"] = process
            logger.info(f"Orchestration service started with PID {process.pid}")

            # Start log monitoring in background
            def monitor_orch_logs():
                for line in process.stdout:
                    if self.running:
                        logger.info(f"[ORCHESTRATION] {line.strip()}")

            log_thread = threading.Thread(target=monitor_orch_logs, daemon=True)
            log_thread.start()

            return process

        except Exception as e:
            logger.error(f"Failed to start orchestration service: {e}")
            return None

    def check_service_health(self, name, port):
        """Check if a service is responding"""
        try:
            import requests

            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ {name} service is healthy")
                return True
            else:
                logger.warning(f"⚠️ {name} service returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"❌ {name} service health check failed: {e}")
            return False

    def wait_for_services(self):
        """Wait for both services to be ready"""
        logger.info("Waiting for services to start...")

        max_wait = 60  # Maximum wait time in seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            whisper_ready = self.check_service_health("Whisper", 5001)
            orch_ready = self.check_service_health("Orchestration", 3000)

            if whisper_ready and orch_ready:
                logger.info("🚀 Both services are ready!")
                logger.info("📊 Orchestration Dashboard: http://localhost:3000")
                logger.info("🎤 Whisper Service API: http://localhost:5001")
                logger.info("🔍 Whisper Service Health: http://localhost:5001/health")
                logger.info("📈 Integration Status: http://localhost:3000/api/health")
                return True

            if not self.running:
                return False

            time.sleep(2)

        logger.error("❌ Services did not start within the timeout period")
        return False

    def stop_all_services(self):
        """Stop all running services"""
        for name, process in self.services.items():
            if process and process.poll() is None:
                logger.info(f"Stopping {name} service...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                    logger.info(f"✅ {name} service stopped")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name} service...")
                    process.kill()
                    process.wait()
                except Exception as e:
                    logger.error(f"Error stopping {name} service: {e}")

    def monitor_services(self):
        """Monitor running services"""
        while self.running:
            for name, process in list(self.services.items()):
                if process.poll() is not None:
                    logger.error(
                        f"❌ {name} service has stopped unexpectedly (exit code: {process.returncode})"
                    )
                    self.running = False
                    break
            time.sleep(5)

    def run(self):
        """Main run method"""
        logger.info("🎤 Starting LiveTranslate Whisper Integration")
        logger.info("=" * 50)

        # Check if Python modules exist
        required_paths = [
            Path(__file__).parent / "modules" / "whisper-service" / "src" / "main.py",
            Path(__file__).parent
            / "modules"
            / "orchestration-service"
            / "src"
            / "orchestration_service.py",
        ]

        for path in required_paths:
            if not path.exists():
                logger.error(f"❌ Required file not found: {path}")
                logger.error("Make sure you're running this from the project root directory")
                return 1

        try:
            # Start services
            whisper_process = self.start_whisper_service()
            if not whisper_process:
                return 1

            # Wait a bit for whisper service to initialize
            time.sleep(5)

            orch_process = self.start_orchestration_service()
            if not orch_process:
                self.stop_all_services()
                return 1

            # Wait for services to be ready
            if not self.wait_for_services():
                self.stop_all_services()
                return 1

            logger.info("🎉 All services started successfully!")
            logger.info("")
            logger.info("To test the integration:")
            logger.info("1. Open http://localhost:3000 in your browser")
            logger.info("2. Check the 'Service Health' panel for whisper service status")
            logger.info("3. Try uploading an audio file for transcription")
            logger.info("4. Check if NPU is detected in the whisper service logs")
            logger.info("")
            logger.info("Press Ctrl+C to stop all services")

            # Monitor services
            self.monitor_services()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop_all_services()

        return 0


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("LiveTranslate Whisper Integration Starter")
        print("")
        print("This script starts both whisper-service and orchestration-service")
        print("for testing the whisper integration with NPU auto-detection.")
        print("")
        print("Usage: python start_whisper_integration.py")
        print("")
        print("Services will be available at:")
        print("  - Orchestration Dashboard: http://localhost:3000")
        print("  - Whisper Service API: http://localhost:5001")
        print("")
        print("Press Ctrl+C to stop all services")
        return 0

    manager = ServiceManager()
    return manager.run()


if __name__ == "__main__":
    sys.exit(main())
