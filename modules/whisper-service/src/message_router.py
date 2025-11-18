#!/usr/bin/env python3
"""
WebSocket Message Router

Advanced message routing system for real-time audio streaming WebSocket server.
Provides organized message handling, middleware support, route registration,
and extensible message processing pipeline.
"""

import logging
import inspect
import time
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from error_handler import (
    error_handler, ErrorCategory, ErrorSeverity, ErrorInfo,
    create_validation_error, create_system_error
)

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    
    # Session management
    JOIN_SESSION = "join_session"
    LEAVE_SESSION = "leave_session"
    
    # Audio streaming
    TRANSCRIBE_STREAM = "transcribe_stream"
    AUDIO_CHUNK = "audio_chunk"
    
    # Heartbeat
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    
    # Control messages
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    PAUSE_RECORDING = "pause_recording"
    RESUME_RECORDING = "resume_recording"
    
    # Configuration
    SET_CONFIG = "set_config"
    GET_CONFIG = "get_config"
    
    # Status and monitoring
    GET_STATUS = "get_status"
    SUBSCRIBE_EVENTS = "subscribe_events"
    UNSUBSCRIBE_EVENTS = "unsubscribe_events"

class RoutePermission(Enum):
    """Route permission levels"""
    PUBLIC = "public"           # No authentication required
    AUTHENTICATED = "authenticated"  # Basic authentication required
    ADMIN = "admin"            # Admin privileges required

@dataclass
class RouteInfo:
    """Information about a registered route"""
    message_type: MessageType
    handler: Callable
    permission: RoutePermission = RoutePermission.PUBLIC
    rate_limit: Optional[int] = None  # Max requests per minute
    validate_schema: bool = True
    middleware: List[Callable] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for documentation"""
        return {
            "message_type": self.message_type.value,
            "permission": self.permission.value,
            "rate_limit": self.rate_limit,
            "validate_schema": self.validate_schema,
            "middleware_count": len(self.middleware),
            "description": self.description,
            "handler_name": self.handler.__name__ if self.handler else None
        }

@dataclass
class MessageContext:
    """Context information for message processing"""
    connection_id: str
    message_type: MessageType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Processing metadata
    route_info: Optional[RouteInfo] = None
    middleware_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "connection_id": self.connection_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "processing_time_ms": self.processing_time * 1000,
            "middleware_data": self.middleware_data
        }

class MessageRouter:
    """Advanced WebSocket message router"""
    
    def __init__(self):
        self.routes: Dict[MessageType, RouteInfo] = {}
        self.global_middleware: List[Callable] = []
        self.rate_limiters: Dict[str, Dict[str, List[float]]] = {}  # connection_id -> message_type -> timestamps
        
        # Statistics
        self.message_counts: Dict[MessageType, int] = {}
        self.error_counts: Dict[MessageType, int] = {}
        self.total_processing_time: Dict[MessageType, float] = {}
        
        # Event subscribers
        self.event_subscribers: Dict[str, List[str]] = {}  # event_type -> [connection_ids]
        
        logger.info("Message router initialized")
    
    def register_route(self, 
                      message_type: MessageType,
                      handler: Callable,
                      permission: RoutePermission = RoutePermission.PUBLIC,
                      rate_limit: Optional[int] = None,
                      validate_schema: bool = True,
                      middleware: Optional[List[Callable]] = None,
                      description: str = ""):
        """Register a message route"""
        route_info = RouteInfo(
            message_type=message_type,
            handler=handler,
            permission=permission,
            rate_limit=rate_limit,
            validate_schema=validate_schema,
            middleware=middleware or [],
            description=description
        )
        
        self.routes[message_type] = route_info
        logger.info(f"Registered route: {message_type.value} -> {handler.__name__}")
    
    def add_global_middleware(self, middleware: Callable):
        """Add global middleware that runs for all messages"""
        self.global_middleware.append(middleware)
        logger.info(f"Added global middleware: {middleware.__name__}")
    
    def route_message(self, connection_id: str, message_type: str, data: Any, **kwargs) -> Any:
        """Route a message to the appropriate handler"""
        start_time = time.time()
        
        try:
            # Convert string to MessageType enum
            try:
                msg_type = MessageType(message_type)
            except ValueError:
                error_info = create_validation_error(f"Unknown message type: {message_type}")
                error_info.connection_id = connection_id
                error_handler.handle_error(error_info)
                return {"error": error_info.to_websocket_response()["error"]}
            
            # Create message context
            context = MessageContext(
                connection_id=connection_id,
                message_type=msg_type,
                data=data,
                client_ip=kwargs.get('client_ip'),
                user_agent=kwargs.get('user_agent'),
                session_id=data.get('session_id') if isinstance(data, dict) else None
            )
            
            # Check if route exists
            route_info = self.routes.get(msg_type)
            if not route_info:
                error_info = create_validation_error(f"No handler registered for message type: {message_type}")
                error_info.connection_id = connection_id
                error_handler.handle_error(error_info)
                return {"error": error_info.to_websocket_response()["error"]}
            
            context.route_info = route_info
            
            # Check rate limiting
            if not self._check_rate_limit(connection_id, msg_type, route_info.rate_limit):
                error_info = ErrorInfo(
                    category=ErrorCategory.RATE_LIMIT_EXCEEDED,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Rate limit exceeded for {message_type}",
                    connection_id=connection_id,
                    suggested_action="Reduce message frequency"
                )
                error_handler.handle_error(error_info)
                return {"error": error_info.to_websocket_response()["error"]}
            
            # Run global middleware
            for middleware in self.global_middleware:
                try:
                    result = middleware(context)
                    if result is False:  # Middleware rejected the message
                        return {"error": "Message rejected by middleware"}
                except Exception as e:
                    logger.error(f"Global middleware error: {e}")
                    error_info = create_system_error("Middleware processing failed", str(e))
                    error_info.connection_id = connection_id
                    error_handler.handle_error(error_info)
                    return {"error": error_info.to_websocket_response()["error"]}
            
            # Run route-specific middleware
            for middleware in route_info.middleware:
                try:
                    result = middleware(context)
                    if result is False:
                        return {"error": "Message rejected by route middleware"}
                except Exception as e:
                    logger.error(f"Route middleware error: {e}")
                    error_info = create_system_error("Route middleware processing failed", str(e))
                    error_info.connection_id = connection_id
                    error_handler.handle_error(error_info)
                    return {"error": error_info.to_websocket_response()["error"]}
            
            # Validate schema if required
            if route_info.validate_schema:
                validation_result = self._validate_message_schema(msg_type, data)
                if validation_result is not True:
                    error_info = create_validation_error(validation_result)
                    error_info.connection_id = connection_id
                    error_handler.handle_error(error_info)
                    return {"error": error_info.to_websocket_response()["error"]}
            
            # Call the handler
            try:
                # Check handler signature and call appropriately
                sig = inspect.signature(route_info.handler)
                if len(sig.parameters) == 1:
                    # Handler expects only context
                    result = route_info.handler(context)
                else:
                    # Handler expects traditional parameters
                    result = route_info.handler(data)
                
                # Update statistics
                self.message_counts[msg_type] = self.message_counts.get(msg_type, 0) + 1
                processing_time = time.time() - start_time
                self.total_processing_time[msg_type] = self.total_processing_time.get(msg_type, 0) + processing_time
                context.processing_time = processing_time
                
                logger.debug(f"Routed message {message_type} from {connection_id} in {processing_time*1000:.2f}ms")
                return result
                
            except Exception as e:
                self.error_counts[msg_type] = self.error_counts.get(msg_type, 0) + 1
                logger.error(f"Handler error for {message_type}: {e}")
                error_info = create_system_error(f"Handler failed for {message_type}", str(e))
                error_info.connection_id = connection_id
                error_handler.handle_error(error_info)
                return {"error": error_info.to_websocket_response()["error"]}
        
        except Exception as e:
            logger.error(f"Router error: {e}")
            error_info = create_system_error("Message routing failed", str(e))
            error_info.connection_id = connection_id
            error_handler.handle_error(error_info)
            return {"error": error_info.to_websocket_response()["error"]}
    
    def _check_rate_limit(self, connection_id: str, message_type: MessageType, rate_limit: Optional[int]) -> bool:
        """Check if message is within rate limit"""
        if rate_limit is None:
            return True
        
        now = time.time()
        minute_ago = now - 60
        
        # Initialize tracking if needed
        if connection_id not in self.rate_limiters:
            self.rate_limiters[connection_id] = {}
        if message_type not in self.rate_limiters[connection_id]:
            self.rate_limiters[connection_id][message_type] = []
        
        # Clean old timestamps
        timestamps = self.rate_limiters[connection_id][message_type]
        timestamps[:] = [ts for ts in timestamps if ts > minute_ago]
        
        # Check limit
        if len(timestamps) >= rate_limit:
            return False
        
        # Add current timestamp
        timestamps.append(now)
        return True
    
    def _validate_message_schema(self, message_type: MessageType, data: Any) -> Union[bool, str]:
        """Validate message schema based on type"""
        # Basic validation rules for different message types
        validation_rules = {
            MessageType.JOIN_SESSION: lambda d: isinstance(d, dict) and 'session_id' in d,
            MessageType.LEAVE_SESSION: lambda d: isinstance(d, dict) and 'session_id' in d,
            MessageType.TRANSCRIBE_STREAM: lambda d: isinstance(d, dict) and 'audio_data' in d,
            MessageType.SET_CONFIG: lambda d: isinstance(d, dict) and 'config' in d,
            MessageType.HEARTBEAT: lambda d: True,  # Heartbeat can be any format
            MessageType.PING: lambda d: True,
            MessageType.PONG: lambda d: True,
        }
        
        validator = validation_rules.get(message_type)
        if validator is None:
            return True  # No validation rule defined
        
        try:
            if validator(data):
                return True
            else:
                return f"Invalid schema for {message_type.value}"
        except Exception as e:
            return f"Schema validation error: {str(e)}"
    
    def subscribe_to_events(self, connection_id: str, event_types: List[str]):
        """Subscribe connection to specific events"""
        for event_type in event_types:
            if event_type not in self.event_subscribers:
                self.event_subscribers[event_type] = []
            if connection_id not in self.event_subscribers[event_type]:
                self.event_subscribers[event_type].append(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to events: {event_types}")
    
    def unsubscribe_from_events(self, connection_id: str, event_types: Optional[List[str]] = None):
        """Unsubscribe connection from events"""
        if event_types is None:
            # Unsubscribe from all events
            for event_type in self.event_subscribers:
                if connection_id in self.event_subscribers[event_type]:
                    self.event_subscribers[event_type].remove(connection_id)
        else:
            for event_type in event_types:
                if event_type in self.event_subscribers and connection_id in self.event_subscribers[event_type]:
                    self.event_subscribers[event_type].remove(connection_id)
        
        logger.debug(f"Connection {connection_id} unsubscribed from events")
    
    def get_event_subscribers(self, event_type: str) -> List[str]:
        """Get list of connections subscribed to an event"""
        return self.event_subscribers.get(event_type, []).copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics"""
        total_messages = sum(self.message_counts.values())
        total_errors = sum(self.error_counts.values())
        
        avg_processing_times = {}
        for msg_type, total_time in self.total_processing_time.items():
            count = self.message_counts.get(msg_type, 0)
            avg_processing_times[msg_type.value] = (total_time / count * 1000) if count > 0 else 0
        
        return {
            "total_messages": total_messages,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_messages * 100) if total_messages > 0 else 0,
            "message_counts": {mt.value: count for mt, count in self.message_counts.items()},
            "error_counts": {mt.value: count for mt, count in self.error_counts.items()},
            "average_processing_times_ms": avg_processing_times,
            "registered_routes": len(self.routes),
            "global_middleware_count": len(self.global_middleware),
            "event_subscribers": {event: len(subs) for event, subs in self.event_subscribers.items()},
            "active_rate_limiters": len(self.rate_limiters)
        }
    
    def get_route_documentation(self) -> Dict[str, Any]:
        """Get documentation for all registered routes"""
        return {
            "routes": {
                route_info.message_type.value: route_info.to_dict()
                for route_info in self.routes.values()
            },
            "message_types": [mt.value for mt in MessageType],
            "permission_levels": [p.value for p in RoutePermission]
        }
    
    def cleanup_connection(self, connection_id: str):
        """Clean up connection-specific data"""
        # Remove from rate limiters
        self.rate_limiters.pop(connection_id, None)
        
        # Remove from event subscriptions
        self.unsubscribe_from_events(connection_id)
        
        logger.debug(f"Cleaned up router data for connection {connection_id}")

# Middleware functions
def logging_middleware(context: MessageContext) -> bool:
    """Log all messages"""
    logger.debug(f"Message: {context.message_type.value} from {context.connection_id}")
    return True

def authentication_middleware(context: MessageContext) -> bool:
    """Basic authentication middleware"""
    if context.route_info and context.route_info.permission != RoutePermission.PUBLIC:
        # In a real implementation, you would check authentication here
        # For now, we'll just log and allow
        logger.debug(f"Authentication check for {context.connection_id}")
    return True

def metrics_middleware(context: MessageContext) -> bool:
    """Collect metrics middleware"""
    context.middleware_data['metrics_start'] = time.time()
    return True

# Global router instance
message_router = MessageRouter()

# Add default middleware
message_router.add_global_middleware(logging_middleware)
message_router.add_global_middleware(authentication_middleware)
message_router.add_global_middleware(metrics_middleware) 