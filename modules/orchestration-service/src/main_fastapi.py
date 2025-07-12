#!/usr/bin/env python3
"""
FastAPI Backend for Orchestration Service

Modern async/await backend with enhanced API endpoints, automatic documentation,
and improved performance over the legacy Flask implementation.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

try:
    from routers import (
        audio_router,
        bot_router,
        websocket_router,
        system_router,
        settings_router,
    )
    from models import SystemStatus, ServiceHealth, ConfigUpdate, ErrorResponse
    from dependencies import (
        get_config_manager,
        get_websocket_manager,
        get_health_monitor,
        get_bot_manager,
        get_database_manager,
    )
    from middleware import (
        SecurityMiddleware,
        LoggingMiddleware,
        ErrorHandlingMiddleware,
    )
    from config import get_settings
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for testing
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global managers (initialized in lifespan)
config_manager = None
websocket_manager = None
health_monitor = None
bot_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global config_manager, websocket_manager, health_monitor, bot_manager

    logger.info("🚀 Starting FastAPI Orchestration Service...")

    try:
        # Initialize managers
        settings = get_settings()

        # Initialize dependencies
        from dependencies import initialize_dependencies

        await initialize_dependencies()

        logger.info("✅ All managers started successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    finally:
        # Shutdown managers
        logger.info("🛑 Shutting down FastAPI Orchestration Service...")

        # Cleanup dependencies
        from dependencies import cleanup_dependencies

        await cleanup_dependencies()

        logger.info("✅ Shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="LiveTranslate Orchestration Service",
    description="Modern FastAPI backend for orchestrating audio processing, translation, and bot management services",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Mount static files for React build
static_path = Path(__file__).parent.parent / "frontend" / "dist"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routers
app.include_router(audio_router, prefix="/api/audio", tags=["Audio"])
app.include_router(bot_router, prefix="/api/bot", tags=["Bot Management"])
app.include_router(websocket_router, prefix="/api/websocket", tags=["WebSocket"])
app.include_router(system_router, prefix="/api/system", tags=["System"])
app.include_router(settings_router, prefix="/api/settings", tags=["Settings"])


# Add direct WebSocket endpoint for frontend compatibility
@app.websocket("/ws")
async def websocket_endpoint_direct(websocket: WebSocket):
    """
    Direct WebSocket endpoint for frontend compatibility
    Provides the same functionality as /api/websocket/connect
    """
    import json
    import logging
    from datetime import datetime
    from dependencies import get_websocket_manager
    from models.websocket import WebSocketResponse, MessageType
    from utils.rate_limiting import RateLimiter

    logger = logging.getLogger(__name__)
    connection_id = None

    try:
        # Get dependencies manually
        websocket_manager = get_websocket_manager()
        ws_rate_limiter = RateLimiter()

        # Accept WebSocket connection
        await websocket.accept()
        logger.info(f"WebSocket connection attempt from {websocket.client}")

        # Register connection (don't double-accept)
        client_ip = str(websocket.client.host) if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")

        # Create connection ID and register manually since we already accepted
        from uuid import uuid4
        import time

        # Import the proper ConnectionInfo class
        from managers.websocket_manager import ConnectionInfo, ConnectionState

        connection_id = str(uuid4())

        # Create proper ConnectionInfo object
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            client_ip=client_ip,
            user_agent=user_agent,
            connected_at=time.time(),
            last_ping=time.time(),
            session_id=None,
            user_id=None,
            state=ConnectionState.CONNECTED,
            metadata={},
        )

        # Register connection manually (bypassing the manager's connect method to avoid double messages)
        websocket_manager.connections[connection_id] = connection_info
        websocket_manager.stats["total_connections"] += 1

        logger.info(f"WebSocket connection registered: {connection_id}")

        # Send welcome message in frontend-expected format
        current_time = datetime.utcnow()
        welcome_message = {
            "type": "connection:established",
            "data": {
                "connectionId": connection_id,  # camelCase to match frontend
                "serverTime": int(current_time.timestamp() * 1000),  # number timestamp
            },
            "timestamp": int(current_time.timestamp() * 1000),
            "messageId": f"msg-{int(current_time.timestamp() * 1000)}-{connection_id[:8]}",
        }

        await websocket.send_text(json.dumps(welcome_message))

        # Message handling loop
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_text()

                # Rate limiting check
                if not await ws_rate_limiter.is_allowed(
                    client_ip, "websocket", 100, 60
                ):
                    error_response = {
                        "type": "error:rate_limit",
                        "data": {
                            "error": "Rate limit exceeded",
                            "error_code": "RATE_LIMIT",
                        },
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "messageId": f"rate-limit-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue

                # Parse message
                try:
                    message_data = json.loads(raw_message)
                    message_type = message_data.get("type", "")

                    # Handle frontend message types
                    if message_type == "connection:ping":
                        # Respond with pong in frontend-expected format
                        pong_response = {
                            "type": "connection:pong",
                            "data": {
                                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                "server_time": datetime.utcnow().isoformat(),
                            },
                            "timestamp": int(datetime.utcnow().timestamp() * 1000),
                            "messageId": f"pong-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                            "correlationId": message_data.get("messageId"),
                        }
                        await websocket.send_text(json.dumps(pong_response))
                    elif message_type in ["ping", "pong"]:
                        # Handle basic ping/pong
                        pong_response = {
                            "type": "pong",
                            "data": {"timestamp": datetime.utcnow().isoformat()},
                            "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        }
                        await websocket.send_text(json.dumps(pong_response))
                    elif message_type == "system:health_request":
                        # Get health data and send it
                        try:
                            from dependencies import get_health_monitor
                            health_monitor = get_health_monitor()
                            health_data = await health_monitor.get_system_health()
                            
                            health_response = {
                                "type": "system:health_update",
                                "data": health_data,
                                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                "messageId": f"health-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                                "correlationId": message_data.get("messageId"),
                            }
                            await websocket.send_text(json.dumps(health_response))
                        except Exception as e:
                            logger.error(f"Failed to get health data: {e}")
                            error_response = {
                                "type": "error:health_data",
                                "data": {"error": str(e)},
                                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                                "messageId": f"error-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                                "correlationId": message_data.get("messageId"),
                            }
                            await websocket.send_text(json.dumps(error_response))
                    else:
                        # Echo back other messages for now with proper format
                        echo_response = {
                            "type": "message:echo",
                            "data": {
                                "original_message": message_data,
                                "echo_from": "orchestration-service",
                            },
                            "timestamp": int(datetime.utcnow().timestamp() * 1000),
                            "messageId": f"echo-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                            "correlationId": message_data.get("messageId"),
                        }
                        await websocket.send_text(json.dumps(echo_response))

                except (json.JSONDecodeError, ValueError) as e:
                    error_response = {
                        "type": "error:message_format",
                        "data": {
                            "error": f"Invalid message format: {e}",
                            "error_code": "INVALID_FORMAT",
                        },
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "messageId": f"error-{int(datetime.utcnow().timestamp() * 1000)}-{connection_id[:8]}",
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue

            except Exception as e:
                error_code = getattr(e, "code", None)
                if error_code in [1000, 1001]:  # Normal close codes
                    logger.info(f"WebSocket closed normally: {error_code}")
                else:
                    logger.error(f"Error handling WebSocket message: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

    finally:
        # Unregister connection manually (don't use manager's disconnect to avoid double-close)
        if connection_id and connection_id in websocket_manager.connections:
            # Just remove from connections dict without trying to close again
            del websocket_manager.connections[connection_id]
            logger.info(f"WebSocket connection unregistered: {connection_id}")


# Root endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    """Serve React application"""
    index_path = Path(__file__).parent.parent / "frontend" / "dist" / "index.html"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LiveTranslate Orchestration Service</title>
                <style>
                    body { 
                        font-family: 'Inter', sans-serif; 
                        margin: 40px; 
                        background: #1a1a1a; 
                        color: #e0e0e0; 
                        text-align: center; 
                    }
                    .container { 
                        max-width: 600px; 
                        margin: 0 auto; 
                        padding: 40px; 
                        background: #2d2d2d; 
                        border-radius: 12px; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    }
                    h1 { color: #4ECDC4; margin-bottom: 20px; }
                    .status { color: #96CEB4; margin: 20px 0; }
                    .links a { 
                        color: #4ECDC4; 
                        text-decoration: none; 
                        margin: 0 15px; 
                        padding: 10px 20px; 
                        border: 1px solid #4ECDC4; 
                        border-radius: 6px; 
                        transition: all 0.3s ease;
                    }
                    .links a:hover { 
                        background: #4ECDC4; 
                        color: #1a1a1a; 
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎙️ LiveTranslate Orchestration Service</h1>
                    <div class="status">✅ FastAPI Backend Running</div>
                    <p>Modern async/await backend with enhanced API endpoints</p>
                    <div class="links">
                        <a href="/docs">📚 API Documentation</a>
                        <a href="/redoc">📖 ReDoc</a>
                        <a href="/api/system/health">🔍 Health Check</a>
                    </div>
                </div>
            </body>
            </html>
            """,
            status_code=200,
        )


@app.get("/api/health")
async def health_check(
    health_monitor=Depends(get_health_monitor),
    database_manager=Depends(get_database_manager),
):
    """Comprehensive health check endpoint"""
    try:
        # Get overall health status
        overall_health = health_monitor.get_overall_health()

        # Check database health
        db_health = await database_manager.health_check()

        return {
            "status": overall_health["status"],
            "version": "2.0.0",
            "services": {
                "orchestration": "healthy",
                "database": "healthy" if db_health else "unhealthy",
                "websocket": "healthy",
                **overall_health["services"],
            },
            "uptime": overall_health.get("uptime_percentage", 0.0),
            "healthy_services": overall_health["healthy_services"],
            "total_services": overall_health["total_services"],
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed",
        )


@app.get("/api/services/status")
async def get_services_status(
    health_monitor=Depends(get_health_monitor),
    config_manager=Depends(get_config_manager),
):
    """Get status of all managed services"""
    try:
        # Get configured services
        services = {}
        for service_name, service_config in config_manager.config.services.items():
            service_health = health_monitor.get_service_health(service_name)
            services[service_name] = {
                "status": service_health.status.value if service_health else "unknown",
                "url": service_config.url,
                "last_check": service_health.last_check if service_health else None,
                "response_time": service_health.response_time
                if service_health
                else None,
                "error_count": service_health.error_count if service_health else 0,
            }

        # Add default services if not configured
        default_services = {
            "frontend-service": {"status": "checking", "url": "http://localhost:5173"}
        }

        for service_name, service_info in default_services.items():
            if service_name not in services:
                services[service_name] = service_info

        return services
    except Exception as e:
        logger.error(f"Services status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Services status check failed",
        )


@app.post("/api/config/update", response_model=Dict[str, Any])
async def update_configuration(
    config_update: ConfigUpdate, config_manager=Depends(get_config_manager)
) -> Dict[str, Any]:
    """Update system configuration"""
    try:
        # For now, just return success with current config
        current_config = config_manager.config
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_keys": list(config_update.dict().keys()),
            "timestamp": current_config.loaded_at.isoformat(),
        }
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration update failed: {str(e)}",
        )


@app.get("/api/websocket/stats")
async def get_websocket_stats(
    websocket_manager=Depends(get_websocket_manager),
) -> Dict[str, Any]:
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"WebSocket stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve WebSocket statistics",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, status_code=exc.status_code, path=str(request.url.path)
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error", status_code=500, path=str(request.url.path)
        ).dict(),
    )


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="LiveTranslate Orchestration API",
        version="2.0.0",
        description="""
        ## LiveTranslate Orchestration Service API
        
        Modern FastAPI backend for orchestrating audio processing, translation, and bot management services.
        
        ### Features
        - 🎙️ **Audio Processing**: Real-time audio capture and processing
        - 🤖 **Bot Management**: Google Meet bot lifecycle management
        - 🌐 **WebSocket**: Real-time communication with connection pooling
        - ⚙️ **System Management**: Health monitoring and configuration
        - 📊 **Analytics**: Performance metrics and monitoring
        
        ### Authentication
        - Bearer token authentication for protected endpoints
        - WebSocket authentication via token parameter
        
        ### Rate Limiting
        - API endpoints: 100 requests/minute
        - WebSocket connections: 10 connections/IP
        
        ### Error Handling
        - Consistent error response format
        - Detailed error messages for development
        - Proper HTTP status codes
        """,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer"}
    }

    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    """Run the FastAPI server"""
    settings = get_settings()

    uvicorn.run(
        "main_fastapi:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.logging.level.lower(),
        access_log=True,
        server_header=False,
        date_header=False,
    )
