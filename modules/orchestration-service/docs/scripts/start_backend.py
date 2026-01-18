#!/usr/bin/env python3
"""
Startup script for the orchestration service backend
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Change to src directory
os.chdir(src_dir)

# Now import and run the app
if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting LiveTranslate Orchestration Service Backend...")
    print("📍 WebSocket endpoint: ws://localhost:3000/ws")
    print("📍 API documentation: http://localhost:3000/docs")
    print("📍 Main dashboard: http://localhost:3000")

    # Run the server
    uvicorn.run(
        "main_fastapi:app",
        host="0.0.0.0",
        port=3000,
        reload=True,
        workers=1,
        log_level="info",
        access_log=True,
    )
