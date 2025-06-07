#!/usr/bin/env python3
"""
Whisper NPU Server - Native Python Startup
Always runs on NPU with simplified configuration
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def print_status(message, success=True):
    """Print colored status messages"""
    color = '\033[92m' if success else '\033[91m'  # Green or Red
    reset = '\033[0m'
    print(f"{color}{message}{reset}")

def check_dependencies():
    """Quick dependency check"""
    print("Checking dependencies...")
    
    required_packages = ['flask', 'librosa', 'openvino', 'numpy', 'pydub', 'soundfile']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print_status(f"Failed to install dependencies: {result.stderr}", False)
            return False
        print_status("✓ Dependencies installed")
    
    return True

def check_ffmpeg():
    """Check if local ffmpeg is available"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(script_dir, "..", "ffmpeg", "bin", "ffmpeg.exe")
    
    if os.path.exists(local_ffmpeg):
        print_status(f"✓ Local ffmpeg found: {local_ffmpeg}")
        return True
    else:
        print_status("⚠ Local ffmpeg not found, will try system PATH", False)
        return False

def check_frontend_structure():
    """Check if the new modular frontend exists"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(script_dir, 'frontend')
    
    if os.path.exists(frontend_dir):
        print_status(f"✓ Modular frontend found: {frontend_dir}")
        
        # Check for key files
        key_files = ['index.html', 'css/styles.css', 'js/main.js', 'js/api.js', 'js/audio.js', 'js/ui.js']
        missing_files = []
        
        for file in key_files:
            file_path = os.path.join(frontend_dir, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file}")
                missing_files.append(file)
        
        if missing_files:
            print_status(f"⚠ Missing frontend files: {missing_files}", False)
            return False
        else:
            print_status("✓ All frontend files present")
            return True
    else:
        print_status(f"✗ Modular frontend not found: {frontend_dir}", False)
        return False

def start_backend():
    """Start the NPU backend server"""
    print_status("Starting NPU backend server on port 5000...")
    
    # Get the script directory to run server.py from the correct location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, 'server.py')
    
    # Start server in background
    if os.name == 'nt':  # Windows
        cmd = ['start', '/B', 'python', server_path]
        subprocess.Popen(cmd, shell=True, cwd=script_dir)
    else:  # Linux/Mac
        subprocess.Popen(['python', server_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=script_dir)
    
    # Wait for server to start
    time.sleep(3)
    return test_server()

def start_frontend():
    """Start the frontend server"""
    print_status("Starting modular frontend server on port 3000...")
    
    # Get the script directory to run frontend from the correct location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(script_dir, 'serve-frontend-alt.py')
    
    if os.name == 'nt':  # Windows
        cmd = ['start', '/B', 'python', frontend_path]
        subprocess.Popen(cmd, shell=True, cwd=script_dir)
    else:  # Linux/Mac
        subprocess.Popen(['python', frontend_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=script_dir)
    
    time.sleep(2)
    return True

def test_server():
    """Test if server is responding"""
    print("Testing server connection...")
    
    for attempt in range(15):
        try:
            response = requests.get("http://localhost:5000/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print_status(f"✓ Server healthy! Device: {data.get('device', 'Unknown')}")
                print_status(f"✓ Models available: {data.get('models_available', 0)}")
                return True
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print()
    print_status("✗ Server not responding", False)
    return False

def main():
    print_status("=== Whisper NPU Server - Native Startup ===")
    print_status("Enhanced modular frontend with audio testing + Speaker Diarization")
    print()
    print_status("🔧 Latest Fixes & Features:")
    print_status("   • ✅ Server status moved to header (status card removed)")
    print_status("   • ✅ Audio test: Record 1s, play back for verification") 
    print_status("   • ✅ Modular JavaScript architecture (main.js, api.js, audio.js, ui.js)")
    print_status("   • ✅ Ultra-lenient audio detection for Chinese speech")
    print_status("   • ✅ Enhanced WebM audio processing with local FFmpeg")
    print_status("   • ✅ Better error handling and NPU memory management")
    print()
    print_status("🎤 NEW: Advanced Speaker Diarization System:")
    print_status("   • 🔥 Real-time speaker identification with continuity tracking")
    print_status("   • 🔥 6-second sliding windows with 2-second overlap")
    print_status("   • 🔥 Configurable speaker count (auto-detect or specify 2-10)")
    print_status("   • 🔥 Speech enhancement with noise reduction")
    print_status("   • 🔥 Punctuation-aligned segmentation to reduce artifacts")
    print_status("   • 🔥 Advanced clustering (HDBSCAN, Agglomerative)")
    print_status("   • 🔥 Multiple embedding methods (Resemblyzer, PyAnnote, Spectral)")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check ffmpeg
    check_ffmpeg()
    
    # Check frontend structure
    frontend_ok = check_frontend_structure()
    if not frontend_ok:
        print_status("⚠ Will serve original frontend as fallback", False)
    
    # Start servers
    if start_backend():
        if start_frontend():
            print()
            print_status("🚀 All servers started successfully!")
            print_status("🌐 Frontend URLs:")
            if frontend_ok:
                print_status("   • http://localhost:3000/ (modular frontend)")
                print_status("   • http://localhost:3000/index.html (main interface)")
            else:
                print_status("   • http://localhost:3000/npu-frontend.html (original frontend)")
            print_status("🔧 Backend API: http://localhost:5000")
            print_status("🧠 NPU acceleration active")
            print_status("🎵 Audio test available in visualization panel")
            print_status("🗣️ Optimized for Chinese speech transcription")
            print()
            print_status("🎤 Speaker Diarization API Endpoints:")
            print_status("   • POST /diarization/configure - Configure speaker settings")
            print_status("   • GET /diarization/status - Get diarization status")
            print_status("   • POST /transcribe/enhanced/<model> - Enhanced transcription")
            print_status("   • GET /speaker/history - Get speaker transcription history")
            print()
            print_status("📖 New Features to Try:")
            print_status("   1. Use 'Test Audio' button to verify microphone working")
            print_status("   2. Server status now shown in top header bar")
            print_status("   3. Better audio sensitivity for quiet Chinese speech")
            print_status("   4. Improved WebM audio format processing")
            print_status("   5. 🔥 Configure speaker diarization for multi-speaker scenarios")
            print_status("   6. 🔥 Enhanced transcription with speech improvement")
            print_status("   7. 🔥 Real-time speaker tracking with 6s windows")
            print()
            print("Press Ctrl+C to stop servers")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print()
                print_status("Stopping servers...")
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                                 capture_output=True)
                else:
                    subprocess.run(['pkill', '-f', 'python.*server'], 
                                 capture_output=True)
                print_status("✓ Servers stopped")
                return 0
        else:
            print_status("✗ Frontend startup failed", False)
            return 1
    else:
        print_status("✗ Backend startup failed", False)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 