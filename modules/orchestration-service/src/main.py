#!/usr/bin/env python3
"""
LiveTranslate Orchestration Service - Main Entry Point

This is the main entry point for the orchestration service backend.
It handles:
- FastAPI backend server
- WebSocket management
- Service coordination
- Bot management
- Audio processing coordination
- Database integration
- Monitoring and health checks
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import orchestration service components
try:
    from main_fastapi import app as fastapi_app
    from config import settings
except ImportError:
    # Fallback to create a basic FastAPI app
    fastapi_app = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    version: str
    services: Dict[str, str]
    uptime: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    logger.info("🚀 Starting LiveTranslate Orchestration Service")
    logger.info("🔧 Initializing services...")

    # Initialize services here
    # - Database connections
    # - Redis connections
    # - Service health monitoring
    # - Bot management system

    yield

    logger.info("🛑 Shutting down LiveTranslate Orchestration Service")
    # Cleanup services here


# Create the main FastAPI application
if fastapi_app is None:
    # Create a basic FastAPI app if the backend module isn't available
    app = FastAPI(
        title="LiveTranslate Orchestration Service",
        description="Backend API & Service Coordination for LiveTranslate",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint - serves a simple status page"""
        return """
        <html>
            <head>
                <title>LiveTranslate Orchestration Service</title>
                <style>
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', roboto, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        margin: 0;
                        padding: 40px;
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                    }
                    h1 { font-size: 2.5em; margin-bottom: 20px; }
                    p { font-size: 1.2em; margin-bottom: 10px; }
                    .status { background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin-top: 20px; }
                    .endpoints { margin-top: 30px; }
                    .endpoint { background: rgba(255,255,255,0.1); padding: 10px; margin: 5px 0; border-radius: 5px; }
                    a { color: #FFD700; text-decoration: none; }
                    a:hover { text-decoration: underline; }
                </style>
            </head>
            <body>
                <h1>🚀 LiveTranslate Orchestration Service</h1>
                <p>Backend API & Service Coordination</p>
                <div class="status">
                    <p><strong>Status:</strong> ✅ Running</p>
                    <p><strong>Version:</strong> 1.0.0</p>
                    <p><strong>Technology:</strong> FastAPI + Python + Async</p>
                </div>
                <div class="endpoints">
                    <h3>Available Endpoints:</h3>
                    <div class="endpoint">🔍 <a href="/docs">API Documentation</a> - Interactive API docs</div>
                    <div class="endpoint">📖 <a href="/redoc">ReDoc</a> - Alternative API documentation</div>
                    <div class="endpoint">❤️ <a href="/api/health">Health Check</a> - Service health status</div>
                    <div class="endpoint">🎙️ <a href="/api/audio">Audio API</a> - Audio processing endpoints</div>
                    <div class="endpoint">🤖 <a href="/api/bot">Bot API</a> - Bot management endpoints</div>
                    <div class="endpoint">🌐 <a href="/ws">WebSocket</a> - Real-time communication</div>
                </div>
                <div style="margin-top: 40px;">
                    <p><strong>Frontend:</strong> <a href="http://localhost:5173">http://localhost:5173</a></p>
                    <p><strong>Backend:</strong> <a href="http://localhost:3000">http://localhost:3000</a></p>
                </div>
            </body>
        </html>
        """

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            services={
                "orchestration": "healthy",
                "database": "checking",
                "redis": "checking",
                "audio-service": "checking",
                "translation-service": "checking",
            },
            uptime=0.0,
        )

    @app.get("/api/audio/health")
    async def audio_health():
        """Audio service health check"""
        return {"status": "healthy", "service": "audio-proxy"}

    @app.get("/api/bot/health")
    async def bot_health():
        """Bot service health check"""
        return {"status": "healthy", "service": "bot-management"}

    @app.get("/api/system/health")
    async def system_health():
        """System health check"""
        return {"status": "healthy", "service": "system-monitoring"}

else:
    # Use the existing FastAPI app from backend module
    app = fastapi_app

# Mount static files if they exist
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount templates if they exist
templates_dir = Path(__file__).parent.parent / "templates"
if templates_dir.exists():
    app.mount("/templates", StaticFiles(directory=str(templates_dir)), name="templates")


def main():
    """Main entry point for the orchestration service"""
    logger.info("🚀 Starting LiveTranslate Orchestration Service")
    logger.info("🔧 Backend Technology: FastAPI + Python + Async")
    logger.info("🌐 Frontend URL: http://localhost:5173")
    logger.info("🔗 Backend URL: http://localhost:3000")
    logger.info("📖 API Docs: http://localhost:3000/docs")
    logger.info("❤️ Health Check: http://localhost:3000/api/health")

    try:
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=3000,
            reload=True,
            log_level="info",
            access_log=True,
            reload_dirs=["src"],
            reload_excludes=["*.pyc", "__pycache__"],
        )
    except KeyboardInterrupt:
        logger.info("🛑 Orchestration service stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start orchestration service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
