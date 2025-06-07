#!/usr/bin/env python3
"""
Whisper NPU Server - Docker Startup
Runs NPU server in containerized environment
"""

import subprocess
import sys
import time
import requests
import os

def print_status(message, success=True):
    """Print colored status messages"""
    color = '\033[92m' if success else '\033[91m'  # Green or Red
    reset = '\033[0m'
    print(f"{color}{message}{reset}")

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"✓ Docker found: {result.stdout.strip()}")
            return True
        else:
            print_status("✗ Docker not available", False)
            return False
    except FileNotFoundError:
        print_status("✗ Docker not installed", False)
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status(f"✓ Docker Compose found: {result.stdout.strip()}")
            return True
        else:
            # Try docker compose (newer syntax)
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True)
            if result.returncode == 0:
                print_status(f"✓ Docker Compose found: {result.stdout.strip()}")
                return True
            else:
                print_status("✗ Docker Compose not available", False)
                return False
    except FileNotFoundError:
        print_status("✗ Docker Compose not installed", False)
        return False

def build_image():
    """Build the NPU Docker image"""
    print_status("Building NPU Docker image...")
    
    result = subprocess.run(['docker', 'build', '-f', 'Dockerfile.npu', '-t', 'whisper-npu', '.'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print_status("✓ Docker image built successfully")
        return True
    else:
        print_status(f"✗ Docker build failed: {result.stderr}", False)
        return False

def start_containers():
    """Start Docker containers"""
    print_status("Starting NPU containers with Docker Compose...")
    
    # Try docker-compose first, then docker compose
    commands = [
        ['docker-compose', '-f', 'docker-compose.npu.yml', 'up', '-d'],
        ['docker', 'compose', '-f', 'docker-compose.npu.yml', 'up', '-d']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print_status("✓ Containers started successfully")
                return True
        except FileNotFoundError:
            continue
    
    print_status("✗ Failed to start containers", False)
    return False

def stop_containers():
    """Stop Docker containers"""
    print_status("Stopping NPU containers...")
    
    commands = [
        ['docker-compose', '-f', 'docker-compose.npu.yml', 'down'],
        ['docker', 'compose', '-f', 'docker-compose.npu.yml', 'down']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print_status("✓ Containers stopped")
                return True
        except FileNotFoundError:
            continue
    
    print_status("⚠ Could not stop containers gracefully", False)
    return False

def test_server(port=8009):
    """Test if Docker server is responding"""
    print("Testing Docker server connection...")
    
    for attempt in range(30):  # Docker takes longer to start
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print_status(f"✓ Docker server healthy! Device: {data.get('device', 'Unknown')}")
                print_status(f"✓ Models available: {data.get('models_available', 0)}")
                return True
        except:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print()
    print_status("✗ Docker server not responding", False)
    return False

def show_logs():
    """Show container logs"""
    print_status("Recent container logs:")
    
    commands = [
        ['docker-compose', '-f', 'docker-compose.npu.yml', 'logs', '--tail=20'],
        ['docker', 'compose', '-f', 'docker-compose.npu.yml', 'logs', '--tail=20']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                return
        except FileNotFoundError:
            continue

def main():
    print_status("=== Whisper NPU Server - Docker Startup ===")
    print_status("This mode runs in isolated Docker containers")
    print()
    
    # Check Docker availability
    if not check_docker():
        print_status("Please install Docker first: https://docs.docker.com/get-docker/", False)
        return 1
    
    if not check_docker_compose():
        print_status("Please install Docker Compose", False)
        return 1
    
    # Build image
    if not build_image():
        return 1
    
    # Start containers
    if start_containers():
        # Wait a bit for containers to initialize
        time.sleep(5)
        
        if test_server():
            print()
            print_status("🚀 Docker containers started successfully!")
            print_status("🌐 Access: http://localhost:8009")
            print_status("🧠 NPU acceleration in container")
            print()
            print("Press Ctrl+C to stop containers")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print()
                print_status("Stopping containers...")
                stop_containers()
                return 0
        else:
            print_status("✗ Container health check failed", False)
            print_status("Showing recent logs...", False)
            show_logs()
            return 1
    else:
        print_status("✗ Container startup failed", False)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 