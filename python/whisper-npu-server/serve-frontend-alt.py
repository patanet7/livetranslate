#!/usr/bin/env python3
"""
Simple HTTP server to serve the NPU frontend
Serves static files from the frontend directory
Enhanced with better error handling and port management
"""

import http.server
import socketserver
import os
import sys
import time
import socket
from pathlib import Path

def log(message, level='INFO'):
    """Enhanced logging"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [Frontend-{level}] {message}")

def find_available_port(start_port=3000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                log(f"Found available port: {port}")
                return port
        except OSError as e:
            log(f"Port {port} unavailable: {e}", 'DEBUG')
            continue
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Set the directory to serve from (frontend directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_dir = os.path.join(script_dir, 'frontend')
        
        if not os.path.exists(frontend_dir):
            log(f"Frontend directory not found: {frontend_dir}", 'WARN')
            log("Creating fallback to serve original frontend...", 'INFO')
            # Fallback to script directory
            super().__init__(*args, directory=script_dir, **kwargs)
        else:
            log(f"Serving from frontend directory: {frontend_dir}")
            super().__init__(*args, directory=frontend_dir, **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        # Redirect root to index.html
        if self.path == '/':
            self.path = '/index.html'
        elif self.path == '/npu-frontend.html':
            # Redirect old URL to new modular frontend
            self.path = '/index.html'
        
        log(f"Serving: {self.path}", 'DEBUG')
        return super().do_GET()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def log_message(self, format, *args):
        # Custom logging format
        log(f"{self.address_string()} - {format % args}", 'DEBUG')

def test_server_response(port, max_attempts=5):
    """Test if the server is responding correctly"""
    import urllib.request
    import urllib.error
    
    for attempt in range(max_attempts):
        try:
            log(f"Testing server response (attempt {attempt + 1}/{max_attempts})...")
            with urllib.request.urlopen(f'http://localhost:{port}/', timeout=5) as response:
                if response.getcode() == 200:
                    log("✓ Server is responding correctly")
                    return True
                else:
                    log(f"Server returned status {response.getcode()}", 'WARN')
        except urllib.error.URLError as e:
            log(f"Server test failed: {e}", 'WARN')
        except Exception as e:
            log(f"Unexpected error during server test: {e}", 'ERROR')
        
        if attempt < max_attempts - 1:
            time.sleep(1)
    
    log("✗ Server is not responding correctly", 'ERROR')
    return False

def main():
    try:
        log("🌐 Starting Whisper NPU Frontend Server")
        
        # Check if frontend directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        frontend_dir = os.path.join(script_dir, 'frontend')
        
        if os.path.exists(frontend_dir):
            log(f"✓ Serving modular frontend from: {frontend_dir}")
            log(f"📁 Frontend structure:")
            try:
                for root, dirs, files in os.walk(frontend_dir):
                    level = root.replace(frontend_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    log(f"{indent}{os.path.basename(root)}/", 'INFO')
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        log(f"{subindent}{file}", 'INFO')
            except Exception as e:
                log(f"Error listing frontend structure: {e}", 'WARN')
        else:
            log(f"⚠ Frontend directory not found, serving from: {script_dir}", 'WARN')
        
        # Find available port
        port = find_available_port(3000)
        log(f"Using port: {port}")
        
        # Create and configure server
        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True
        
        with ReusableTCPServer(("", port), CustomHTTPRequestHandler) as httpd:
            log(f"🚀 Frontend server started successfully!")
            log(f"📡 Server details:")
            log(f"   • Address: 0.0.0.0:{port}")
            log(f"   • URLs:")
            log(f"     - http://localhost:{port}/")
            log(f"     - http://127.0.0.1:{port}/")
            log(f"     - http://localhost:{port}/index.html")
            log(f"     - http://localhost:{port}/settings.html")
            log(f"     - http://localhost:{port}/test-api.html")
            log(f"")
            log(f"📖 Features:")
            log(f"   • Modular JavaScript architecture")
            log(f"   • Audio test functionality (record 1s, play back)")
            log(f"   • Server status in header (no status card)")
            log(f"   • Enhanced audio visualization")
            log(f"   • Chinese audio transcription optimized")
            log(f"   • Comprehensive API testing page")
            log(f"")
            log("Press Ctrl+C to stop the server")
            
            # Test server after brief startup delay
            import threading
            def delayed_test():
                time.sleep(2)
                test_server_response(port)
            
            test_thread = threading.Thread(target=delayed_test, daemon=True)
            test_thread.start()
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                log("\n🛑 Server stopped gracefully")
                return 0
                
    except Exception as e:
        log(f"❌ Failed to start server: {e}", 'ERROR')
        log(f"Error details: {type(e).__name__}: {str(e)}", 'ERROR')
        return 1

if __name__ == "__main__":
    sys.exit(main()) 