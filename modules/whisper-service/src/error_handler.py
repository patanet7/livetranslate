#!/usr/bin/env python3
"""
WebSocket Error Handling Framework

Comprehensive error handling system for real-time audio streaming WebSocket server.
Provides standardized error responses, categorization, logging, and recovery mechanisms.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Error categories for classification"""
    # Connection errors
    CONNECTION_FAILED = "connection_failed"
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_LIMIT_EXCEEDED = "connection_limit_exceeded"
    AUTHENTICATION_FAILED = "authentication_failed"
    
    # Audio processing errors
    AUDIO_FORMAT_INVALID = "audio_format_invalid"
    AUDIO_TOO_LARGE = "audio_too_large"
    AUDIO_PROCESSING_FAILED = "audio_processing_failed"
    
    # Model/AI errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    INFERENCE_FAILED = "inference_failed"
    DEVICE_ERROR = "device_error"
    OUT_OF_MEMORY = "out_of_memory"
    
    # Session errors
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_EXPIRED = "session_expired"
    SESSION_LIMIT_EXCEEDED = "session_limit_exceeded"
    
    # Validation errors
    INVALID_REQUEST = "invalid_request"
    MISSING_PARAMETER = "missing_parameter"
    INVALID_PARAMETER = "invalid_parameter"
    
    # System errors
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    
    # WebSocket specific errors
    WEBSOCKET_ERROR = "websocket_error"
    MESSAGE_TOO_LARGE = "message_too_large"
    PROTOCOL_ERROR = "protocol_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, service continues normally
    MEDIUM = "medium"     # Moderate issues, some functionality affected
    HIGH = "high"         # Serious issues, major functionality affected
    CRITICAL = "critical" # Critical issues, service may be compromised

@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    connection_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    recoverable: bool = True
    retry_after: Optional[int] = None  # seconds
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "error": {
                "category": self.category.value,
                "severity": self.severity.value,
                "message": self.message,
                "details": self.details,
                "error_code": self.error_code,
                "timestamp": self.timestamp.isoformat(),
                "session_id": self.session_id,
                "connection_id": self.connection_id,
                "user_id": self.user_id,
                "context": self.context,
                "recoverable": self.recoverable,
                "retry_after": self.retry_after,
                "suggested_action": self.suggested_action
            }
        }
    
    def to_websocket_response(self) -> Dict[str, Any]:
        """Convert to WebSocket error response format"""
        return {
            "type": "error",
            "error": {
                "category": self.category.value,
                "message": self.message,
                "details": self.details,
                "error_code": self.error_code,
                "timestamp": self.timestamp.isoformat(),
                "session_id": self.session_id,
                "recoverable": self.recoverable,
                "retry_after": self.retry_after,
                "suggested_action": self.suggested_action
            }
        }
    
    def to_http_response(self) -> tuple[Dict[str, Any], int]:
        """Convert to HTTP error response format with status code"""
        status_code = self._get_http_status_code()
        return self.to_dict(), status_code
    
    def _get_http_status_code(self) -> int:
        """Get appropriate HTTP status code based on error category"""
        status_map = {
            ErrorCategory.CONNECTION_FAILED: 503,
            ErrorCategory.CONNECTION_TIMEOUT: 408,
            ErrorCategory.CONNECTION_LIMIT_EXCEEDED: 429,
            ErrorCategory.AUTHENTICATION_FAILED: 401,
            ErrorCategory.AUDIO_FORMAT_INVALID: 400,
            ErrorCategory.AUDIO_TOO_LARGE: 413,
            ErrorCategory.AUDIO_PROCESSING_FAILED: 422,
            ErrorCategory.MODEL_NOT_FOUND: 404,
            ErrorCategory.MODEL_LOAD_FAILED: 503,
            ErrorCategory.INFERENCE_FAILED: 422,
            ErrorCategory.DEVICE_ERROR: 503,
            ErrorCategory.OUT_OF_MEMORY: 507,
            ErrorCategory.SESSION_NOT_FOUND: 404,
            ErrorCategory.SESSION_EXPIRED: 410,
            ErrorCategory.SESSION_LIMIT_EXCEEDED: 429,
            ErrorCategory.INVALID_REQUEST: 400,
            ErrorCategory.MISSING_PARAMETER: 400,
            ErrorCategory.INVALID_PARAMETER: 400,
            ErrorCategory.SERVICE_UNAVAILABLE: 503,
            ErrorCategory.INTERNAL_ERROR: 500,
            ErrorCategory.RATE_LIMIT_EXCEEDED: 429,
            ErrorCategory.WEBSOCKET_ERROR: 400,
            ErrorCategory.MESSAGE_TOO_LARGE: 413,
            ErrorCategory.PROTOCOL_ERROR: 400
        }
        return status_map.get(self.category, 500)

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.max_history_size = 1000
        
    def register_recovery_handler(self, category: ErrorCategory, handler: Callable):
        """Register a recovery handler for specific error category"""
        self.recovery_handlers[category] = handler
        logger.info(f"Registered recovery handler for {category.value}")
    
    def handle_error(self, error_info: ErrorInfo) -> ErrorInfo:
        """Process and handle an error"""
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        # Update counts
        self.error_counts[error_info.category] = self.error_counts.get(error_info.category, 0) + 1
        
        # Log the error
        self._log_error(error_info)
        
        # Attempt recovery if handler exists
        if error_info.category in self.recovery_handlers and error_info.recoverable:
            try:
                self.recovery_handlers[error_info.category](error_info)
                logger.info(f"Recovery attempted for {error_info.category.value}")
            except Exception as e:
                logger.error(f"Recovery handler failed for {error_info.category.value}: {e}")
        
        return error_info
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level based on severity"""
        log_message = f"[{error_info.category.value}] {error_info.message}"
        if error_info.details:
            log_message += f" - {error_info.details}"
        
        if error_info.session_id:
            log_message += f" (session: {error_info.session_id})"
        if error_info.connection_id:
            log_message += f" (connection: {error_info.connection_id})"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp).total_seconds() < 3600]
        
        category_stats = {}
        severity_stats = {}
        
        for error in self.error_history:
            category_stats[error.category.value] = category_stats.get(error.category.value, 0) + 1
            severity_stats[error.severity.value] = severity_stats.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "category_distribution": category_stats,
            "severity_distribution": severity_stats,
            "recovery_handlers": list(self.recovery_handlers.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors"""
        recent = self.error_history[-limit:] if self.error_history else []
        return [error.to_dict() for error in reversed(recent)]

# Error factory functions for common error types
def create_connection_error(message: str, details: str = None, connection_id: str = None) -> ErrorInfo:
    """Create a connection error"""
    return ErrorInfo(
        category=ErrorCategory.CONNECTION_FAILED,
        severity=ErrorSeverity.MEDIUM,
        message=message,
        details=details,
        connection_id=connection_id,
        suggested_action="Check network connection and try reconnecting"
    )

def create_audio_error(message: str, details: str = None, session_id: str = None) -> ErrorInfo:
    """Create an audio processing error"""
    return ErrorInfo(
        category=ErrorCategory.AUDIO_PROCESSING_FAILED,
        severity=ErrorSeverity.MEDIUM,
        message=message,
        details=details,
        session_id=session_id,
        suggested_action="Check audio format and try again"
    )

def create_model_error(message: str, details: str = None, recoverable: bool = True) -> ErrorInfo:
    """Create a model/inference error"""
    return ErrorInfo(
        category=ErrorCategory.INFERENCE_FAILED,
        severity=ErrorSeverity.HIGH,
        message=message,
        details=details,
        recoverable=recoverable,
        suggested_action="Try again with a different model or check system resources"
    )

def create_validation_error(message: str, parameter: str = None) -> ErrorInfo:
    """Create a validation error"""
    return ErrorInfo(
        category=ErrorCategory.INVALID_PARAMETER if parameter else ErrorCategory.INVALID_REQUEST,
        severity=ErrorSeverity.LOW,
        message=message,
        details=f"Invalid parameter: {parameter}" if parameter else None,
        recoverable=False,
        suggested_action="Check request format and parameters"
    )

def create_session_error(message: str, session_id: str = None, expired: bool = False) -> ErrorInfo:
    """Create a session error"""
    category = ErrorCategory.SESSION_EXPIRED if expired else ErrorCategory.SESSION_NOT_FOUND
    return ErrorInfo(
        category=category,
        severity=ErrorSeverity.MEDIUM,
        message=message,
        session_id=session_id,
        suggested_action="Create a new session or check session ID"
    )

def create_system_error(message: str, details: str = None, critical: bool = False) -> ErrorInfo:
    """Create a system error"""
    return ErrorInfo(
        category=ErrorCategory.INTERNAL_ERROR,
        severity=ErrorSeverity.CRITICAL if critical else ErrorSeverity.HIGH,
        message=message,
        details=details,
        suggested_action="Contact system administrator if problem persists"
    )

# Decorator for automatic error handling
def handle_errors(error_handler_instance: ErrorHandler):
    """Decorator to automatically handle errors in functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error info from exception
                error_info = ErrorInfo(
                    category=ErrorCategory.INTERNAL_ERROR,
                    severity=ErrorSeverity.HIGH,
                    message=str(e),
                    details=traceback.format_exc(),
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                )
                
                # Handle the error
                handled_error = error_handler_instance.handle_error(error_info)
                
                # Re-raise with additional context
                raise Exception(f"Handled error: {handled_error.message}") from e
        
        return wrapper
    return decorator

# Global error handler instance
error_handler = ErrorHandler()

# Recovery handlers for common error types
def recover_device_error(error_info: ErrorInfo):
    """Recovery handler for device errors"""
    logger.info("Attempting device error recovery...")
    # This would typically involve reloading models or switching devices
    # Implementation depends on the specific whisper service architecture

def recover_memory_error(error_info: ErrorInfo):
    """Recovery handler for memory errors"""
    logger.info("Attempting memory error recovery...")
    # This would typically involve clearing caches or reducing model size
    # Implementation depends on the specific whisper service architecture

def recover_connection_error(error_info: ErrorInfo):
    """Recovery handler for connection errors"""
    logger.info("Attempting connection error recovery...")
    # This would typically involve connection cleanup or reconnection
    # Implementation depends on the specific connection manager

# Register default recovery handlers
error_handler.register_recovery_handler(ErrorCategory.DEVICE_ERROR, recover_device_error)
error_handler.register_recovery_handler(ErrorCategory.OUT_OF_MEMORY, recover_memory_error)
error_handler.register_recovery_handler(ErrorCategory.CONNECTION_FAILED, recover_connection_error) 