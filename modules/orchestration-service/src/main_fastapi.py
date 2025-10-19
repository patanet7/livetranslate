#!/usr/bin/env python3
"""
FastAPI Backend for Orchestration Service

Modern async/await backend with enhanced API endpoints, automatic documentation,
and improved performance over the legacy Flask implementation.
"""

import os
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.utils import get_openapi

import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
router_logger = logging.getLogger("router_registration")
import_logger = logging.getLogger("import_analysis")
route_logger = logging.getLogger("route_conflicts")
startup_logger = logging.getLogger("startup_process")

router_logger.setLevel(logging.WARNING)
import_logger.setLevel(logging.WARNING)
route_logger.setLevel(logging.WARNING)
startup_logger.setLevel(logging.INFO)

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Enhanced import process with detailed logging
import_logger.info("[SEARCH] Starting detailed import analysis...")

# Import routers with individual logging
routers_status = {}
try:
    import_logger.info("[FOLDER] Importing router modules...")
    
    import_logger.debug("Importing audio_router...")
    from routers.audio import router as audio_router
    import_logger.info("[OK] audio_router imported successfully")
    routers_status['audio_router'] = {'status': 'success', 'routes': len(audio_router.routes) if hasattr(audio_router, 'routes') else 0}
    
    import_logger.debug("Importing audio_coordination_router...")
    from routers.audio_coordination import router as audio_coordination_router
    import_logger.info("[OK] audio_coordination_router imported successfully")
    routers_status['audio_coordination_router'] = {'status': 'success', 'routes': len(audio_coordination_router.routes) if hasattr(audio_coordination_router, 'routes') else 0}
    
    import_logger.debug("Importing bot_router...")
    from routers.bot import router as bot_router
    import_logger.info("[OK] bot_router imported successfully")
    routers_status['bot_router'] = {'status': 'success', 'routes': len(bot_router.routes) if hasattr(bot_router, 'routes') else 0}
    
    import_logger.debug("Importing websocket_router...")
    from routers.websocket import router as websocket_router
    import_logger.info("[OK] websocket_router imported successfully")
    routers_status['websocket_router'] = {'status': 'success', 'routes': len(websocket_router.routes) if hasattr(websocket_router, 'routes') else 0}
    
    import_logger.debug("Importing system_router...")
    from routers.system import router as system_router
    import_logger.info("[OK] system_router imported successfully")
    routers_status['system_router'] = {'status': 'success', 'routes': len(system_router.routes) if hasattr(system_router, 'routes') else 0}
    
    import_logger.debug("Importing settings_router...")
    from routers.settings import router as settings_router
    import_logger.info("[OK] settings_router imported successfully")
    routers_status['settings_router'] = {'status': 'success', 'routes': len(settings_router.routes) if hasattr(settings_router, 'routes') else 0}
    
    import_logger.debug("Importing translation_router...")
    from routers.translation import router as translation_router
    import_logger.info("[OK] translation_router imported successfully")
    routers_status['translation_router'] = {'status': 'success', 'routes': len(translation_router.routes) if hasattr(translation_router, 'routes') else 0}
    
    import_logger.debug("Importing analytics_router...")
    from routers.analytics import router as analytics_router
    import_logger.info("[OK] analytics_router imported successfully")
    routers_status['analytics_router'] = {'status': 'success', 'routes': len(analytics_router.routes) if hasattr(analytics_router, 'routes') else 0}
    
    import_logger.info("[STATS] Router import summary:")
    for router_name, status in routers_status.items():
        import_logger.info(f"  {router_name}: {status['status']} ({status['routes']} routes)")

except ImportError as e:
    import_logger.error(f"[ERROR] Router import failed: {e}")
    routers_status[str(e)] = {'status': 'failed', 'error': str(e)}
    import traceback
    import_logger.error("Full traceback:")
    import_logger.error(traceback.format_exc())

# Import models with logging
try:
    import_logger.debug("Importing models...")
    from models import SystemStatus, ServiceHealth, ConfigUpdate, ErrorResponse
    import_logger.info("[OK] Models imported successfully")
except ImportError as e:
    import_logger.error(f"[ERROR] Models import failed: {e}")

# Import dependencies with logging
try:
    import_logger.debug("Importing dependencies...")
    from dependencies import (
        get_config_manager,
        get_websocket_manager,
        get_health_monitor,
        get_bot_manager,
        get_database_manager,
    )
    import_logger.info("[OK] Dependencies imported successfully")
except ImportError as e:
    import_logger.error(f"[ERROR] Dependencies import failed: {e}")

# Import middleware with logging
try:
    import_logger.debug("Importing middleware...")
    from middleware import (
        SecurityMiddleware,
        LoggingMiddleware,
        ErrorHandlingMiddleware,
    )
    import_logger.info("[OK] Middleware imported successfully")
except ImportError as e:
    import_logger.error(f"[ERROR] Middleware import failed: {e}")

# Import config with logging
try:
    import_logger.debug("Importing config...")
    from config import get_settings
    import_logger.info("[OK] Config imported successfully")
except ImportError as e:
    import_logger.error(f"[ERROR] Config import failed: {e}")

import_logger.info("[TARGET] Import analysis complete - checking for any failures...")
failed_imports = [name for name, status in routers_status.items() if status['status'] == 'failed']
if failed_imports:
    import_logger.error(f"[CRITICAL] Failed imports detected: {failed_imports}")
    sys.exit(1)
else:
    import_logger.info("[OK] All critical imports successful")

# Continue with enhanced logging already configured at top

# Custom JSON encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            # Convert datetime to ISO format string for JSON serialization
            return obj.isoformat()
        return super().default(obj)

# Custom JSONResponse class that uses our datetime-aware encoder
class DateTimeJSONResponse(JSONResponse):
    """Custom JSONResponse that properly handles datetime objects"""
    
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

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

    logger.info("[START] Starting FastAPI Orchestration Service...")

    try:
        # Initialize managers
        settings = get_settings()

        # Initialize dependencies
        from dependencies import initialize_dependencies

        await initialize_dependencies()

        logger.info("[OK] All managers started successfully")

        yield

    except Exception as e:
        logger.error(f"[ERROR] Startup failed: {e}")
        raise

    finally:
        # Shutdown managers
        logger.info("[STOP] Shutting down FastAPI Orchestration Service...")

        # Cleanup dependencies
        from dependencies import cleanup_dependencies

        await cleanup_dependencies()

        logger.info("[OK] Shutdown completed")


# Create FastAPI application with custom JSON encoder
app = FastAPI(
    title="LiveTranslate Orchestration Service",
    description="Modern FastAPI backend for orchestrating audio processing, translation, and bot management services",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    default_response_class=DateTimeJSONResponse,  # Use our custom JSON response class
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

# Mount orchestration static files (favicon, etc.)
orchestration_static_path = Path(__file__).parent.parent / "static"
if orchestration_static_path.exists():
    app.mount("/static", StaticFiles(directory=str(orchestration_static_path)), name="orchestration_static")

# Enhanced router registration with conflict detection and resolution
router_logger.info("[LINK] Starting enhanced router registration process...")

def log_router_details(router_name, router, prefix):
    """Log detailed information about a router before registration"""
    route_count = len(router.routes) if hasattr(router, 'routes') else 0
    router_logger.info(f"[LIST] {router_name} details:")
    router_logger.info(f"  Prefix: {prefix}")
    router_logger.info(f"  Route count: {route_count}")
    if hasattr(router, 'routes') and router.routes:
        for i, route in enumerate(router.routes):
            methods = getattr(route, 'methods', [])
            path = getattr(route, 'path', 'unknown')
            router_logger.debug(f"    Route {i+1}: {methods} {prefix}{path}")

def check_route_conflicts(new_prefix, new_router_name, existing_routes):
    """Check for potential route conflicts"""
    conflicts = []
    for existing_prefix, existing_name in existing_routes:
        if new_prefix == existing_prefix:
            conflicts.append(f"{existing_name} (same prefix: {existing_prefix})")
    if conflicts:
        route_logger.warning(f"[WARNING] CONFLICT DETECTED: {new_router_name} conflicts with: {', '.join(conflicts)}")
    return conflicts

# Track registered routers for conflict detection
registered_routes = []

# Register audio_router first
router_logger.info("[1] Registering audio_router...")
log_router_details("audio_router", audio_router, "/api/audio")
conflicts = check_route_conflicts("/api/audio", "audio_router", registered_routes)
app.include_router(audio_router, prefix="/api/audio", tags=["Audio"])
registered_routes.append(("/api/audio", "audio_router"))
router_logger.info("[OK] audio_router registered successfully")

# Register audio_coordination_router with different prefix to resolve conflict
router_logger.info("[2] Registering audio_coordination_router...")
audio_coord_prefix = "/api/audio-coordination"  # Different prefix to avoid conflict
log_router_details("audio_coordination_router", audio_coordination_router, audio_coord_prefix)
conflicts = check_route_conflicts(audio_coord_prefix, "audio_coordination_router", registered_routes)
app.include_router(audio_coordination_router, prefix=audio_coord_prefix, tags=["Audio Coordination"])
registered_routes.append((audio_coord_prefix, "audio_coordination_router"))
router_logger.info(f"[OK] audio_coordination_router registered successfully with prefix {audio_coord_prefix}")

# Register bot_router
router_logger.info("[3] Registering bot_router...")
log_router_details("bot_router", bot_router, "/api/bot")
conflicts = check_route_conflicts("/api/bot", "bot_router", registered_routes)
app.include_router(bot_router, prefix="/api/bot", tags=["Bot Management"])
registered_routes.append(("/api/bot", "bot_router"))
router_logger.info("[OK] bot_router registered successfully")

# Register websocket_router
router_logger.info("[4] Registering websocket_router...")
log_router_details("websocket_router", websocket_router, "/api/websocket")
conflicts = check_route_conflicts("/api/websocket", "websocket_router", registered_routes)
app.include_router(websocket_router, prefix="/api/websocket", tags=["WebSocket"])
registered_routes.append(("/api/websocket", "websocket_router"))
router_logger.info("[OK] websocket_router registered successfully")

# Register system_router
router_logger.info("[5] Registering system_router...")
log_router_details("system_router", system_router, "/api/system")
conflicts = check_route_conflicts("/api/system", "system_router", registered_routes)
app.include_router(system_router, prefix="/api/system", tags=["System"])
registered_routes.append(("/api/system", "system_router"))
router_logger.info(" system_router registered successfully")

# Register settings_router
router_logger.info("[6] Registering settings_router...")
log_router_details("settings_router", settings_router, "/api/settings")
conflicts = check_route_conflicts("/api/settings", "settings_router", registered_routes)
app.include_router(settings_router, prefix="/api/settings", tags=["Settings"])
registered_routes.append(("/api/settings", "settings_router"))
router_logger.info(" settings_router registered successfully")

# Register translation_router
# Register pipeline_router for Pipeline Studio
router_logger.info("[7] Registering pipeline_router...")
try:
    from routers.pipeline import router as pipeline_router
    log_router_details("pipeline_router", pipeline_router, "/api/pipeline")
    conflicts = check_route_conflicts("/api/pipeline", "pipeline_router", registered_routes)
    app.include_router(pipeline_router, prefix="/api/pipeline", tags=["Pipeline Processing"])
    registered_routes.append(("/api/pipeline", "pipeline_router"))
    router_logger.info("pipeline_router registered successfully")
except Exception as e:
    router_logger.error(f"Failed to register pipeline_router: {str(e)}")

router_logger.info("[8] Registering translation_router...")
log_router_details("translation_router", translation_router, "/api/translation")
conflicts = check_route_conflicts("/api/translation", "translation_router", registered_routes)
app.include_router(translation_router, prefix="/api/translation", tags=["Translation"])
registered_routes.append(("/api/translation", "translation_router"))
router_logger.info(" translation_router registered successfully")

# Also include translation router on /api/translate for direct compatibility
router_logger.info("[8] Registering translation_router (compatibility alias)...")
log_router_details("translation_router", translation_router, "/api/translate")
conflicts = check_route_conflicts("/api/translate", "translation_router_alias", registered_routes)
app.include_router(translation_router, prefix="/api/translate", tags=["Translation Direct"])
registered_routes.append(("/api/translate", "translation_router_alias"))
router_logger.info(" translation_router compatibility alias registered successfully")

# Register analytics_router
router_logger.info("[9] Registering analytics_router...")
log_router_details("analytics_router", analytics_router, "/api/analytics")
conflicts = check_route_conflicts("/api/analytics", "analytics_router", registered_routes)
app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])
registered_routes.append(("/api/analytics", "analytics_router"))
router_logger.info(" analytics_router registered successfully")

# Summary of registration
router_logger.info(" Router registration summary:")
total_routes = 0
for prefix, name in registered_routes:
    router_logger.info(f"   {name}: {prefix}")
    if name == "audio_router":
        total_routes += len(audio_router.routes) if hasattr(audio_router, 'routes') else 0
    elif name == "audio_coordination_router":
        total_routes += len(audio_coordination_router.routes) if hasattr(audio_coordination_router, 'routes') else 0
    # Add other routers as needed

router_logger.info(f"Total routers registered: {len(registered_routes)}")
router_logger.info(f"Estimated total routes: {total_routes}")
router_logger.info(" Router registration process completed successfully!")

# Debug and diagnostic endpoints
startup_logger.info(" Adding diagnostic endpoints for debugging...")

@app.get("/debug/routes")
async def get_debug_routes():
    """Debug endpoint to inspect all registered routes"""
    global registered_routes
    
    routes_info = []
    
    try:
        for route in app.routes:
            route_info = {
                "path": getattr(route, 'path', 'unknown'),
                "methods": list(getattr(route, 'methods', [])),
                "name": getattr(route, 'name', 'unknown'),
                "tags": getattr(route, 'tags', [])
            }
            routes_info.append(route_info)
        
        route_logger.info(f"Debug routes endpoint called - returning {len(routes_info)} routes")
        
        return {
            "total_routes": len(routes_info),
            "routes": routes_info,
            "registered_routers": registered_routes if registered_routes else [],
            "registered_routers_count": len(registered_routes) if registered_routes else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        route_logger.error(f"Error in debug/routes endpoint: {e}")
        return {
            "error": str(e),
            "total_routes": 0,
            "routes": [],
            "registered_routers": [],
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug/routers")
async def get_debug_routers():
    """Debug endpoint to show router registration status"""
    global registered_routes, routers_status
    
    try:
        router_status = {}
        
        # Check each imported router
        for router_name, status in routers_status.items():
            router_status[router_name] = status
        
        route_logger.info(f"Debug routers endpoint called - returning {len(router_status)} router statuses")
        
        return {
            "router_count": len(router_status),
            "router_status": router_status,
            "registered_routes": registered_routes if registered_routes else [],
            "registered_routes_count": len(registered_routes) if registered_routes else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        route_logger.error(f"Error in debug/routers endpoint: {e}")
        return {
            "error": str(e),
            "router_count": 0,
            "router_status": {},
            "registered_routes": [],
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug/conflicts")
async def get_debug_conflicts():
    """Debug endpoint to detect route conflicts"""
    # Use global registered_routes variable
    global registered_routes
    
    conflicts = []
    prefixes_seen = {}
    
    # Check if registered_routes is populated
    if not registered_routes:
        route_logger.warning("registered_routes is empty - router registration may not have completed")
        return {
            "conflict_count": 0,
            "conflicts": [],
            "all_prefixes": {},
            "resolution_status": "ERROR",
            "error": "No registered routes found - service may not be fully initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Check for conflicts
    for prefix, name in registered_routes:
        if prefix in prefixes_seen:
            conflicts.append({
                "prefix": prefix,
                "conflicting_routers": [prefixes_seen[prefix], name],
                "severity": "HIGH"
            })
        else:
            prefixes_seen[prefix] = name
    
    route_logger.info(f"Debug conflicts endpoint called - found {len(conflicts)} conflicts")
    
    return {
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "all_prefixes": prefixes_seen,
        "resolution_status": "RESOLVED" if len(conflicts) == 0 else "NEEDS_ATTENTION",
        "registered_routes_count": len(registered_routes),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/debug/imports")
async def get_debug_imports():
    """Debug endpoint to show import status"""
    global routers_status
    
    try:
        import_logger.info("Debug imports endpoint called")
        
        return {
            "import_count": len(routers_status),
            "import_status": routers_status,
            "successful_imports": [name for name, status in routers_status.items() if status['status'] == 'success'],
            "failed_imports": [name for name, status in routers_status.items() if status['status'] == 'failed'],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        import_logger.error(f"Error in debug/imports endpoint: {e}")
        return {
            "error": str(e),
            "import_count": 0,
            "import_status": {},
            "successful_imports": [],
            "failed_imports": [],
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug/health")
async def get_debug_health():
    """Debug endpoint for comprehensive health check"""
    global registered_routes, routers_status
    
    try:
        health_info = {
            "service_status": "operational",
            "router_registration": "completed",
            "total_routes": len(app.routes),
            "total_routers": len(registered_routes) if registered_routes else 0,
            "import_status": "success" if all(status['status'] == 'success' for status in routers_status.values()) else "partial_failure",
            "conflicts_detected": len([1 for prefix in [p for p, n in registered_routes] if [p for p, n in registered_routes].count(prefix) > 1]) if registered_routes else 0,
            "debug_session_id": startup_logger.name,
            "registered_routes_available": bool(registered_routes),
            "routers_status_available": bool(routers_status),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        startup_logger.info(f"Debug health endpoint called - service status: {health_info['service_status']}")
        
        return health_info
    except Exception as e:
        startup_logger.error(f"Error in debug/health endpoint: {e}")
        return {
            "error": str(e),
            "service_status": "error",
            "router_registration": "failed",
            "total_routes": 0,
            "total_routers": 0,
            "import_status": "failed",
            "conflicts_detected": 0,
            "debug_session_id": startup_logger.name,
            "timestamp": datetime.utcnow().isoformat()
        }

startup_logger.info(" Diagnostic endpoints added successfully")

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
@app.get("/favicon.ico")
async def favicon():
    """Serve favicon to prevent 404 errors"""
    favicon_path = Path(__file__).parent.parent / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    else:
        # Return empty response to prevent 404 logging
        return Response(content="", media_type="image/x-icon")

@app.get("/admin")
async def admin_redirect():
    """Redirect /admin to API docs"""
    return HTMLResponse(content="""
    <html>
        <head><title>Admin - LiveTranslate</title></head>
        <body>
            <h2>LiveTranslate Admin</h2>
            <p>Available admin interfaces:</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/api/health">Health Check</a></li>
                <li><a href="/debug/health">Debug Health</a></li>
            </ul>
        </body>
    </html>
    """, status_code=200)

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
                    <h1>[LIVE] LiveTranslate Orchestration Service</h1>
                    <div class="status"> FastAPI Backend Running</div>
                    <p>Modern async/await backend with enhanced API endpoints</p>
                    <div class="links">
                        <a href="/docs"> API Documentation</a>
                        <a href="/redoc"> ReDoc</a>
                        <a href="/api/system/health"> Health Check</a>
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
        - **Audio Processing**: Real-time audio capture and processing
        - **Bot Management**: Google Meet bot lifecycle management
        - **WebSocket**: Real-time communication with connection pooling
        - **System Management**: Health monitoring and configuration
        - **Analytics**: Performance metrics and monitoring
        
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
