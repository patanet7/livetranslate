#!/usr/bin/env python3
"""
Simple HTTP server to serve the NPU frontend
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

def main():
    # Use current working directory instead of changing it
    current_dir = os.getcwd()
    print(f"📁 Working directory: {current_dir}")
    
    # Check if npu-frontend.html exists in current directory
    html_file = os.path.join(current_dir, "npu-frontend.html")
    if not os.path.exists(html_file):
        print(f"❌ Error: npu-frontend.html not found in {current_dir}")
        print("Please run this script from the external/whisper-npu-server directory")
        return
    
    PORT = 8081  # Changed from 8080 to avoid conflict with nginx
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers for cross-origin requests to the NPU server
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 Serving NPU Frontend at http://localhost:{PORT}")
        print(f"📁 Serving from: {current_dir}")
        print(f"🔗 Open: http://localhost:{PORT}/npu-frontend.html")
        print(f"🧠 NPU Server should be running on port 8009")
        print(f"⚠️  Note: Port 8080 is used by main LiveTranslate frontend")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            # Try to open the browser automatically
            webbrowser.open(f'http://localhost:{PORT}/npu-frontend.html')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main() 