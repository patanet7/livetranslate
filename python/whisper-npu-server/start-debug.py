#!/usr/bin/env python3
"""
Debug startup script for Whisper NPU Server
Includes comprehensive logging and error handling
"""

import subprocess
import sys
import time
import requests
import os
import threading
import json
from pathlib import Path

def log(message, level='INFO'):
    """Enhanced logging"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def check_port(port, host='localhost'):
    """Check if a port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0  # True if port is in use
    except Exception as e:
        log(f"Error checking port {port}: {e}", 'ERROR')
        return False

def start_backend_server():
    """Start the backend server with logging"""
    log("Starting backend server...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, 'server.py')
    
    if not os.path.exists(server_path):
        log(f"Server file not found: {server_path}", 'ERROR')
        return None
    
    # Check if port 5000 is already in use
    if check_port(5000):
        log("Port 5000 is already in use - stopping existing server", 'WARN')
        try:
            # Try to stop existing server
            requests.post('http://localhost:5000/shutdown', timeout=5)
            time.sleep(2)
        except:
            log("Could not gracefully stop existing server", 'WARN')
    
    # Start server
    try:
        if os.name == 'nt':  # Windows
            cmd = [sys.executable, server_path]
            process = subprocess.Popen(cmd, cwd=script_dir, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
        else:  # Linux/Mac
            cmd = [sys.executable, server_path]
            process = subprocess.Popen(cmd, cwd=script_dir,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
        
        log(f"Started backend server with PID: {process.pid}")
        return process
        
    except Exception as e:
        log(f"Failed to start backend server: {e}", 'ERROR')
        return None

def start_frontend_server():
    """Start the frontend server with logging"""
    log("Starting frontend server...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_server_path = os.path.join(script_dir, 'serve-frontend-alt.py')
    
    if not os.path.exists(frontend_server_path):
        log(f"Frontend server file not found: {frontend_server_path}", 'ERROR')
        return None
    
    # Check if port 3000 is already in use
    if check_port(3000):
        log("Port 3000 is already in use", 'WARN')
    
    # Start frontend server
    try:
        if os.name == 'nt':  # Windows
            cmd = [sys.executable, frontend_server_path]
            process = subprocess.Popen(cmd, cwd=script_dir,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
        else:  # Linux/Mac
            cmd = [sys.executable, frontend_server_path]
            process = subprocess.Popen(cmd, cwd=script_dir,
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
        
        log(f"Started frontend server with PID: {process.pid}")
        return process
        
    except Exception as e:
        log(f"Failed to start frontend server: {e}", 'ERROR')
        return None

def test_backend():
    """Test backend server connectivity"""
    log("Testing backend server...")
    
    for attempt in range(20):
        try:
            response = requests.get("http://localhost:5000/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                log(f"✓ Backend server healthy! Device: {data.get('device', 'Unknown')}")
                log(f"✓ Models available: {data.get('models_available', 0)}")
                return True
            else:
                log(f"Backend server returned status {response.status_code}", 'WARN')
        except requests.exceptions.ConnectionError:
            log(f"Attempt {attempt + 1}/20: Backend server not ready...", 'DEBUG')
        except Exception as e:
            log(f"Backend test error: {e}", 'ERROR')
        
        time.sleep(1)
    
    log("✗ Backend server not responding", 'ERROR')
    return False

def test_frontend():
    """Test frontend server connectivity"""
    log("Testing frontend server...")
    
    for attempt in range(10):
        try:
            response = requests.get("http://localhost:3000/", timeout=3)
            if response.status_code == 200:
                log("✓ Frontend server responding")
                return True
            else:
                log(f"Frontend server returned status {response.status_code}", 'WARN')
        except requests.exceptions.ConnectionError:
            log(f"Attempt {attempt + 1}/10: Frontend server not ready...", 'DEBUG')
        except Exception as e:
            log(f"Frontend test error: {e}", 'ERROR')
        
        time.sleep(1)
    
    log("✗ Frontend server not responding", 'ERROR')
    return False

def test_api_endpoints():
    """Test specific API endpoints"""
    log("Testing API endpoints...")
    
    endpoints = [
        ('/health', 'Health check'),
        ('/models', 'Models list'),
        ('/settings', 'Settings')
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:5000{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                log(f"✓ {description}: {len(str(data))} bytes")
            else:
                log(f"✗ {description}: HTTP {response.status_code}", 'WARN')
        except Exception as e:
            log(f"✗ {description}: {e}", 'ERROR')

def monitor_processes(backend_process, frontend_process):
    """Monitor server processes"""
    log("Monitoring server processes...")
    
    try:
        while True:
            # Check backend
            if backend_process and backend_process.poll() is not None:
                log("Backend server process terminated!", 'ERROR')
                stdout, stderr = backend_process.communicate()
                if stdout:
                    log(f"Backend stdout: {stdout}", 'DEBUG')
                if stderr:
                    log(f"Backend stderr: {stderr}", 'ERROR')
                break
            
            # Check frontend
            if frontend_process and frontend_process.poll() is not None:
                log("Frontend server process terminated!", 'ERROR')
                stdout, stderr = frontend_process.communicate()
                if stdout:
                    log(f"Frontend stdout: {stdout}", 'DEBUG')
                if stderr:
                    log(f"Frontend stderr: {stderr}", 'ERROR')
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        log("Shutting down servers...")
        
        # Graceful shutdown
        if backend_process:
            try:
                requests.post('http://localhost:5000/shutdown', timeout=3)
            except:
                pass
            backend_process.terminate()
            backend_process.wait()
        
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
        
        log("Servers stopped")

def main():
    log("=== Whisper NPU Server Debug Startup ===")
    
    # Check environment
    log(f"Python: {sys.executable}")
    log(f"Working directory: {os.getcwd()}")
    log(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Start servers
    backend_process = start_backend_server()
    if not backend_process:
        log("Failed to start backend server", 'ERROR')
        return 1
    
    # Wait for backend to be ready
    if not test_backend():
        log("Backend server failed health check", 'ERROR')
        backend_process.terminate()
        return 1
    
    # Test API endpoints
    test_api_endpoints()
    
    # Start frontend
    frontend_process = start_frontend_server()
    if not frontend_process:
        log("Failed to start frontend server", 'ERROR')
        backend_process.terminate()
        return 1
    
    # Wait for frontend to be ready
    if not test_frontend():
        log("Frontend server failed health check", 'ERROR')
        frontend_process.terminate()
        backend_process.terminate()
        return 1
    
    # Success!
    log("🚀 All servers started successfully!")
    log("🌐 Frontend: http://localhost:3000/")
    log("🔧 Backend: http://localhost:5000/")
    log("🐛 Debug: Check console for detailed logs")
    log("")
    log("Press Ctrl+C to stop servers")
    
    # Monitor processes
    monitor_processes(backend_process, frontend_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 