"""
Audio Processing Error Types for Whisper Service

Comprehensive error handling system for whisper audio processing with specific error types,
circuit breaker patterns, retry mechanisms, and error recovery strategies.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import json


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    AUDIO_FORMAT = "audio_format"
    AUDIO_CORRUPTION = "audio_corruption"
    AUDIO_PROCESSING = "audio_processing"
    MODEL_LOADING = "model_loading"
    MODEL_INFERENCE = "model_inference"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    HARDWARE = "hardware"
    TIMEOUT = "timeout"


# Custom Exception Classes
class WhisperProcessingBaseError(Exception):
    """Base exception for all whisper processing errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None,
        category: ErrorCategory = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        correlation_id: str = None,
        details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category or ErrorCategory.AUDIO_PROCESSING
        self.severity = severity
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class AudioFormatError(WhisperProcessingBaseError):
    """Invalid or unsupported audio format"""
    
    def __init__(self, message: str, format_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.AUDIO_FORMAT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details.update(format_details or {})


class AudioCorruptionError(WhisperProcessingBaseError):
    """Corrupted audio data detected"""
    
    def __init__(self, message: str, corruption_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUDIO_CORRUPTION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(corruption_details or {})


class ModelLoadingError(WhisperProcessingBaseError):
    """Model loading failures"""
    
    def __init__(self, message: str, model_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.details.update(model_details or {})


class ModelInferenceError(WhisperProcessingBaseError):
    """Model inference failures"""
    
    def __init__(self, message: str, inference_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_INFERENCE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(inference_details or {})


class ValidationError(WhisperProcessingBaseError):
    """Input validation failures"""
    
    def __init__(self, message: str, validation_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details.update(validation_details or {})


class ConfigurationError(WhisperProcessingBaseError):
    """Configuration issues"""
    
    def __init__(self, message: str, config_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(config_details or {})


class MemoryError(WhisperProcessingBaseError):
    """Memory-related errors"""
    
    def __init__(self, message: str, memory_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.details.update(memory_details or {})


class HardwareError(WhisperProcessingBaseError):
    """Hardware-related errors"""
    
    def __init__(self, message: str, hardware_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(hardware_details or {})


class TimeoutError(WhisperProcessingBaseError):
    """Timeout errors"""
    
    def __init__(self, message: str, timeout_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(timeout_details or {})


# Circuit Breaker Implementation (Simplified for Flask)
@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    is_half_open: bool = False
    success_count: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation for whisper service failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name
        self.state = CircuitBreakerState()
        self._lock = threading.Lock() if hasattr(__builtins__, 'threading') else None
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self._should_attempt_call():
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise
        else:
            raise WhisperProcessingBaseError(
                f"Circuit breaker '{self.name}' is open",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                details={
                    "failure_count": self.state.failure_count,
                    "last_failure_time": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None
                }
            )
    
    def _should_attempt_call(self) -> bool:
        """Determine if call should be attempted"""
        if not self.state.is_open:
            return True
        
        if self.state.last_failure_time:
            time_since_failure = datetime.utcnow() - self.state.last_failure_time
            if time_since_failure.total_seconds() >= self.recovery_timeout:
                self.state.is_half_open = True
                self.state.is_open = False
                return True
        
        return False
    
    def _on_success(self):
        """Handle successful call"""
        if self.state.is_half_open:
            self.state.success_count += 1
            if self.state.success_count >= self.success_threshold:
                self.state.is_half_open = False
                self.state.failure_count = 0
                self.state.success_count = 0
        else:
            self.state.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call"""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.utcnow()
        self.state.success_count = 0
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.is_open = True
            self.state.is_half_open = False
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.state.failure_count} failures"
            )


# Error Recovery Strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def can_recover(self, error: WhisperProcessingBaseError) -> bool:
        """Check if error can be recovered"""
        raise NotImplementedError
    
    def recover(self, error: WhisperProcessingBaseError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from error"""
        raise NotImplementedError


class ModelRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for model loading errors"""
    
    def can_recover(self, error: WhisperProcessingBaseError) -> bool:
        return isinstance(error, (ModelLoadingError, ModelInferenceError))
    
    def recover(self, error: WhisperProcessingBaseError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt model fallback"""
        logger.info(f"[{error.correlation_id}] Attempting model recovery")
        
        fallback_models = context.get("fallback_models", ["whisper-tiny", "whisper-base"])
        current_model = context.get("current_model")
        
        # Find next fallback model
        if current_model in fallback_models:
            current_index = fallback_models.index(current_model)
            if current_index < len(fallback_models) - 1:
                next_model = fallback_models[current_index + 1]
                return {
                    "recovery_attempted": True,
                    "fallback_model": next_model,
                    "original_model": current_model
                }
        
        return {
            "recovery_attempted": False,
            "reason": "No suitable fallback model available"
        }


class FormatRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for audio format errors"""
    
    def can_recover(self, error: WhisperProcessingBaseError) -> bool:
        return isinstance(error, AudioFormatError)
    
    def recover(self, error: AudioFormatError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt format conversion"""
        logger.info(f"[{error.correlation_id}] Attempting audio format recovery")
        
        # Try common format conversions
        recovery_actions = [
            {"action": "convert_to_wav", "format": "wav"},
            {"action": "resample", "sample_rate": 16000},
            {"action": "convert_channels", "channels": 1}
        ]
        
        return {
            "recovery_attempted": True,
            "recovery_actions": recovery_actions,
            "fallback_format": "wav"
        }


# Error Logging with Correlation IDs
class ErrorLogger:
    """Centralized error logging with correlation IDs"""
    
    def __init__(self, logger_name: str = "whisper_errors"):
        self.logger = logging.getLogger(logger_name)
    
    def log_error(
        self,
        error: WhisperProcessingBaseError,
        context: Dict[str, Any] = None,
        request_info: Dict[str, Any] = None
    ):
        """Log error with full context"""
        log_data = {
            "correlation_id": error.correlation_id,
            "error_code": error.error_code,
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "message": error.message,
            "timestamp": error.timestamp.isoformat(),
            "details": error.details,
            "context": context or {},
            "request_info": request_info or {}
        }
        
        log_message = f"[{error.correlation_id}] {error.category.value.upper()}: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_data": log_data})
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={"error_data": log_data})
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"error_data": log_data})
        else:
            self.logger.info(log_message, extra={"error_data": log_data})
    
    def log_recovery_attempt(
        self,
        error: WhisperProcessingBaseError,
        recovery_result: Dict[str, Any]
    ):
        """Log error recovery attempt"""
        log_data = {
            "correlation_id": error.correlation_id,
            "error_code": error.error_code,
            "recovery_result": recovery_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"[{error.correlation_id}] Recovery attempted for {error.error_code}",
            extra={"recovery_data": log_data}
        )


# Error Context Manager
@contextmanager
def error_boundary(
    correlation_id: str = None,
    context: Dict[str, Any] = None,
    recovery_strategies: List[ErrorRecoveryStrategy] = None,
    circuit_breaker: CircuitBreaker = None
):
    """Error boundary context manager for comprehensive error handling"""
    correlation_id = correlation_id or str(uuid.uuid4())
    context = context or {}
    recovery_strategies = recovery_strategies or []
    error_logger = ErrorLogger()
    
    try:
        yield correlation_id
    except WhisperProcessingBaseError as e:
        e.correlation_id = e.correlation_id or correlation_id
        error_logger.log_error(e, context)
        
        # Attempt recovery
        for strategy in recovery_strategies:
            if strategy.can_recover(e):
                try:
                    recovery_result = strategy.recover(e, context)
                    error_logger.log_recovery_attempt(e, recovery_result)
                    
                    if recovery_result.get("recovery_attempted"):
                        # Recovery was attempted, may need to re-raise or return
                        if not recovery_result.get("success", False):
                            raise
                        break
                except Exception as recovery_error:
                    logger.error(
                        f"[{correlation_id}] Recovery failed: {str(recovery_error)}"
                    )
        
        raise
    except Exception as e:
        # Convert to WhisperProcessingBaseError
        whisper_error = WhisperProcessingBaseError(
            message=str(e),
            correlation_id=correlation_id,
            severity=ErrorSeverity.HIGH,
            details={"original_exception": str(type(e).__name__)}
        )
        error_logger.log_error(whisper_error, context)
        raise whisper_error


# Global instances
default_circuit_breaker = CircuitBreaker(name="whisper_service")
default_error_logger = ErrorLogger()
model_recovery = ModelRecoveryStrategy()
format_recovery = FormatRecoveryStrategy()

# Import threading for circuit breaker lock
try:
    import threading
except ImportError:
    threading = None
    logger.warning("Threading not available, circuit breaker will not be thread-safe")