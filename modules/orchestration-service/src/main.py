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
from main_fastapi import app as fastapi_app
from config import get_settings

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
    logger.info(" Starting LiveTranslate Orchestration Service")
    logger.info(" Initializing services...")

    # Initialize services here
    # - Database connections
    # - Redis connections
    # - Service health monitoring
    # - Bot management system

    yield

    logger.info("🛑 Shutting down LiveTranslate Orchestration Service")
    # Cleanup services here


# Use the FastAPI app from main_fastapi module (no fallback)
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
    logger.info("Starting LiveTranslate Orchestration Service")
    logger.info(" Backend Technology: FastAPI + Python + Async")
    logger.info("Frontend URL: http://localhost:5173")
    logger.info("Backend URL: http://localhost:3000")
    logger.info("API Docs: http://localhost:3000/docs")
    logger.info("Health Check: http://localhost:3000/api/health")

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
        logger.info(" Orchestration service stopped by user")
    except Exception as e:
        logger.error(f" Failed to start orchestration service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
