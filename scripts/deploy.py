#!/usr/bin/env python3
"""
LiveTranslate Intelligent Deployment Script

This script detects the operating system and available hardware acceleration
(NPU, GPU, CPU) and automatically configures Docker Compose services accordingly.

Features:
- OS detection (Windows, Linux, macOS)
- NPU detection via OpenVINO and device files
- GPU detection via nvidia-smi and PyTorch
- Automatic fallback chain configuration (NPU → GPU → CPU)
- Environment file generation
- Docker Compose profile selection
- Hardware-specific optimizations
"""

import os
import sys
import platform
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class HardwareDetector:
    """Detect available hardware acceleration capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detected_hardware = {
            'npu': False,
            'gpu': False,
            'cpu': True,  # Always available
            'os': platform.system().lower(),
            'architecture': platform.machine(),
            'python_version': platform.python_version()
        }
    
    def detect_os(self) -> str:
        """Detect operating system."""
        os_name = platform.system().lower()
        self.logger.info(f"Detected OS: {os_name}")
        
        if os_name == 'windows':
            return 'windows'
        elif os_name == 'linux':
            return 'linux'
        elif os_name == 'darwin':
            return 'macos'
        else:
            self.logger.warning(f"Unsupported OS detected: {os_name}")
            return 'unknown'
    
    def detect_npu(self) -> bool:
        """Detect Intel NPU availability."""
        self.logger.info("Checking for Intel NPU...")
        
        # Method 1: Check for NPU device files (Linux)
        npu_device_paths = [
            '/dev/accel/accel0',
            '/dev/dri/renderD*',
            '/sys/class/drm/card*/device/vendor'
        ]
        
        device_found = False
        for device_path in npu_device_paths:
            if '*' in device_path:
                # Handle glob patterns
                import glob
                matches = glob.glob(device_path)
                if matches:
                    device_found = True
                    self.logger.info(f"Found NPU-related device: {matches}")
                    break
            else:
                if Path(device_path).exists():
                    device_found = True
                    self.logger.info(f"Found NPU device: {device_path}")
                    break
        
        # Method 2: Check OpenVINO NPU support
        openvino_npu = False
        try:
            import openvino as ov
            core = ov.Core()
            available_devices = core.available_devices
            self.logger.info(f"OpenVINO available devices: {available_devices}")
            
            if 'NPU' in available_devices:
                openvino_npu = True
                self.logger.info("✓ OpenVINO NPU support detected!")
            else:
                self.logger.info("✗ No OpenVINO NPU support found")
                
        except ImportError:
            self.logger.warning("OpenVINO not installed, cannot check NPU support")
        except Exception as e:
            self.logger.warning(f"Error checking OpenVINO NPU: {e}")
        
        # Method 3: Check Intel drivers (Windows)
        intel_driver = False
        if self.detected_hardware['os'] == 'windows':
            try:
                result = subprocess.run(
                    ['driverquery', '/fo', 'csv'],
                    capture_output=True, text=True, timeout=10
                )
                if 'intel' in result.stdout.lower() and 'npu' in result.stdout.lower():
                    intel_driver = True
                    self.logger.info("✓ Intel NPU driver detected on Windows")
            except Exception as e:
                self.logger.warning(f"Error checking Windows drivers: {e}")
        
        npu_available = device_found or openvino_npu or intel_driver
        self.detected_hardware['npu'] = npu_available
        
        if npu_available:
            self.logger.info("🎯 NPU acceleration available!")
        else:
            self.logger.info("⚠ NPU not detected")
            
        return npu_available
    
    def detect_gpu(self) -> bool:
        """Detect NVIDIA GPU availability."""
        self.logger.info("Checking for NVIDIA GPU...")
        
        # Method 1: nvidia-smi command
        nvidia_smi = False
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                gpus = result.stdout.strip().split('\n')
                nvidia_smi = True
                self.logger.info(f"✓ NVIDIA GPUs detected: {gpus}")
            else:
                self.logger.info("✗ No NVIDIA GPUs found via nvidia-smi")
        except FileNotFoundError:
            self.logger.info("✗ nvidia-smi not found")
        except Exception as e:
            self.logger.warning(f"Error running nvidia-smi: {e}")
        
        # Method 2: PyTorch CUDA support
        pytorch_cuda = False
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                pytorch_cuda = True
                self.logger.info(f"✓ PyTorch CUDA support: {gpu_count} GPU(s) - {gpu_names}")
            else:
                self.logger.info("✗ PyTorch CUDA not available")
        except ImportError:
            self.logger.warning("PyTorch not installed, cannot check CUDA support")
        except Exception as e:
            self.logger.warning(f"Error checking PyTorch CUDA: {e}")
        
        # Method 3: Check CUDA installation
        cuda_installed = False
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                cuda_installed = True
                self.logger.info("✓ CUDA toolkit installed")
        except FileNotFoundError:
            self.logger.info("✗ CUDA toolkit not found")
        except Exception as e:
            self.logger.warning(f"Error checking CUDA: {e}")
        
        gpu_available = nvidia_smi or pytorch_cuda
        self.detected_hardware['gpu'] = gpu_available
        
        if gpu_available:
            self.logger.info("🚀 GPU acceleration available!")
        else:
            self.logger.info("⚠ GPU not detected")
            
        return gpu_available
    
    def get_optimal_profile(self) -> str:
        """Determine the optimal Docker Compose profile based on detected hardware."""
        if self.detected_hardware['npu']:
            return 'npu'
        elif self.detected_hardware['gpu']:
            return 'gpu'
        else:
            return 'cpu'
    
    def get_fallback_chain(self) -> List[str]:
        """Get the fallback chain based on available hardware."""
        chain = []
        if self.detected_hardware['npu']:
            chain.append('npu')
        if self.detected_hardware['gpu']:
            chain.append('gpu')
        chain.append('cpu')  # Always available
        return chain
    
    def detect_all(self) -> Dict:
        """Run comprehensive hardware detection."""
        self.logger.info("Starting hardware detection...")
        
        # Detect all hardware
        self.detect_npu()
        self.detect_gpu()
        
        # Add derived information
        self.detected_hardware['optimal_profile'] = self.get_optimal_profile()
        self.detected_hardware['fallback_chain'] = self.get_fallback_chain()
        
        self.logger.info("Hardware detection complete!")
        return self.detected_hardware


class EnvironmentGenerator:
    """Generate environment files and configurations."""
    
    def __init__(self, hardware_info: Dict, project_root: Path):
        self.hardware_info = hardware_info
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
    
    def generate_env_file(self) -> Path:
        """Generate .env file with detected hardware capabilities."""
        env_file = self.project_root / '.env'
        
        env_content = f"""# LiveTranslate Environment Configuration
# Generated automatically by deploy.py

# Hardware Detection Results
DETECTED_OS={self.hardware_info['os']}
DETECTED_ARCHITECTURE={self.hardware_info['architecture']}
NPU_AVAILABLE={str(self.hardware_info['npu']).lower()}
GPU_AVAILABLE={str(self.hardware_info['gpu']).lower()}
OPTIMAL_PROFILE={self.hardware_info['optimal_profile']}

# Docker Compose Configuration
COMPOSE_PROJECT_NAME=livetranslate
DOCKER_BUILDKIT=1

# User/Group IDs for non-root containers
UID={os.getuid() if hasattr(os, 'getuid') else 1000}
GID={os.getgid() if hasattr(os, 'getgid') else 1000}

# Database Configuration
POSTGRES_PASSWORD=livetranslate_secure_pass_$(openssl rand -hex 8)
POSTGRES_DB=livetranslate
POSTGRES_USER=livetranslate

# Monitoring Configuration
GRAFANA_PASSWORD=admin_$(openssl rand -hex 6)

# Service URLs (Internal Docker Network)
WHISPER_NPU_URL=http://whisper-npu-server:5000
WHISPER_GPU_URL=http://whisper-gpu-server:5000
WHISPER_CPU_URL=http://whisper-cpu-server:5000

# Hardware-Specific Optimizations
"""
        
        if self.hardware_info['npu']:
            env_content += """
# NPU Optimizations
OPENVINO_DEVICE=NPU
OPENVINO_LOG_LEVEL=1
NPU_MEMORY_OPTIMIZATION=true
"""
        
        if self.hardware_info['gpu']:
            env_content += """
# GPU Optimizations
CUDA_VISIBLE_DEVICES=all
NVIDIA_VISIBLE_DEVICES=all
CUDA_MEMORY_FRACTION=0.8
"""
        
        # OS-specific configurations
        if self.hardware_info['os'] == 'windows':
            env_content += """
# Windows-specific settings
DOCKER_HOST=npipe:////./pipe/docker_engine
"""
        elif self.hardware_info['os'] == 'linux':
            env_content += """
# Linux-specific settings
DOCKER_HOST=unix:///var/run/docker.sock
"""
        
        env_file.write_text(env_content)
        self.logger.info(f"Generated environment file: {env_file}")
        return env_file
    
    def generate_compose_override(self) -> Path:
        """Generate docker-compose.override.yml for hardware-specific optimizations."""
        override_file = self.project_root / 'docker-compose.override.yml'
        
        override_config = {
            'version': '3.8',
            'services': {}
        }
        
        # NPU-specific overrides
        if self.hardware_info['npu']:
            override_config['services']['whisper-npu-server'] = {
                'environment': [
                    'OPENVINO_DEVICE=NPU',
                    'NPU_OPTIMIZATION=true'
                ],
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '6G',
                            'cpus': '3.0'
                        }
                    }
                }
            }
        
        # GPU-specific overrides
        if self.hardware_info['gpu']:
            override_config['services']['whisper-gpu-server'] = {
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '10G',
                            'cpus': '6.0'
                        }
                    }
                }
            }
            override_config['services']['translation-server'] = {
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '12G',
                            'cpus': '8.0'
                        }
                    }
                }
            }
        
        # CPU-only optimizations
        if not self.hardware_info['npu'] and not self.hardware_info['gpu']:
            override_config['services']['whisper-cpu-server'] = {
                'deploy': {
                    'resources': {
                        'limits': {
                            'memory': '8G',
                            'cpus': '6.0'
                        }
                    }
                }
            }
        
        with open(override_file, 'w') as f:
            import yaml
            yaml.dump(override_config, f, default_flow_style=False)
        
        self.logger.info(f"Generated Docker Compose override: {override_file}")
        return override_file


class DockerDeployer:
    """Handle Docker Compose deployment operations."""
    
    def __init__(self, project_root: Path, hardware_info: Dict):
        self.project_root = project_root
        self.hardware_info = hardware_info
        self.logger = logging.getLogger(__name__)
    
    def get_compose_command(self, action: str, extra_args: List[str] = None) -> List[str]:
        """Build Docker Compose command with appropriate profiles."""
        profile = self.hardware_info['optimal_profile']
        
        cmd = ['docker-compose']
        
        # Use comprehensive compose file
        cmd.extend(['-f', 'docker-compose.comprehensive.yml'])
        
        # Add override file if it exists
        override_file = self.project_root / 'docker-compose.override.yml'
        if override_file.exists():
            cmd.extend(['-f', str(override_file)])
        
        # Add profile
        cmd.extend(['--profile', profile])
        
        # Add action
        cmd.append(action)
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        return cmd
    
    def deploy(self, services: List[str] = None, detached: bool = True) -> bool:
        """Deploy the application stack."""
        self.logger.info(f"Deploying with profile: {self.hardware_info['optimal_profile']}")
        
        try:
            # Build images first
            build_cmd = self.get_compose_command('build')
            if services:
                build_cmd.extend(services)
            
            self.logger.info(f"Building images: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, cwd=self.project_root)
            if result.returncode != 0:
                self.logger.error("Failed to build images")
                return False
            
            # Start services
            up_cmd = self.get_compose_command('up')
            if detached:
                up_cmd.append('-d')
            if services:
                up_cmd.extend(services)
            
            self.logger.info(f"Starting services: {' '.join(up_cmd)}")
            result = subprocess.run(up_cmd, cwd=self.project_root)
            if result.returncode != 0:
                self.logger.error("Failed to start services")
                return False
            
            self.logger.info("✅ Deployment successful!")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False
    
    def status(self) -> bool:
        """Check service status."""
        try:
            status_cmd = self.get_compose_command('ps')
            result = subprocess.run(status_cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Failed to check status: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop all services."""
        try:
            stop_cmd = self.get_compose_command('down')
            result = subprocess.run(stop_cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Failed to stop services: {e}")
            return False


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('deploy.log')
        ]
    )


def main():
    """Main deployment script entry point."""
    parser = argparse.ArgumentParser(
        description='LiveTranslate Intelligent Deployment Script'
    )
    parser.add_argument(
        '--action', 
        choices=['detect', 'deploy', 'status', 'stop', 'full'],
        default='full',
        help='Action to perform'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--services',
        nargs='*',
        help='Specific services to deploy'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Detect hardware
        detector = HardwareDetector()
        hardware_info = detector.detect_all()
        
        logger.info("=" * 60)
        logger.info("HARDWARE DETECTION RESULTS")
        logger.info("=" * 60)
        for key, value in hardware_info.items():
            logger.info(f"{key.upper()}: {value}")
        logger.info("=" * 60)
        
        if args.action in ['detect']:
            print(json.dumps(hardware_info, indent=2))
            return 0
        
        # Generate environment files
        env_gen = EnvironmentGenerator(hardware_info, args.project_root)
        env_file = env_gen.generate_env_file()
        override_file = env_gen.generate_compose_override()
        
        if args.action in ['deploy', 'full']:
            # Deploy services
            deployer = DockerDeployer(args.project_root, hardware_info)
            success = deployer.deploy(args.services)
            if not success:
                return 1
            
            # Show status
            logger.info("\nDeployment Status:")
            deployer.status()
            
        elif args.action == 'status':
            deployer = DockerDeployer(args.project_root, hardware_info)
            deployer.status()
            
        elif args.action == 'stop':
            deployer = DockerDeployer(args.project_root, hardware_info)
            deployer.stop()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 