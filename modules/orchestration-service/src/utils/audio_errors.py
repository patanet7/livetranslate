"""
Audio Processing Error Types and Utilities

Comprehensive error handling system for audio processing pipeline with specific error types,
circuit breaker patterns, retry mechanisms, and error recovery strategies.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager


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
    SERVICE_UNAVAILABLE = "service_unavailable"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"


# Custom Exception Classes
class AudioProcessingBaseError(Exception):
    """Base exception for all audio processing errors"""
    
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


class AudioFormatError(AudioProcessingBaseError):
    """Invalid or unsupported audio format"""
    
    def __init__(self, message: str, format_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.AUDIO_FORMAT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details.update(format_details or {})


class AudioCorruptionError(AudioProcessingBaseError):
    """Corrupted audio data detected"""
    
    def __init__(self, message: str, corruption_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUDIO_CORRUPTION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(corruption_details or {})


class AudioProcessingError(AudioProcessingBaseError):
    """Processing pipeline failures"""
    
    def __init__(self, message: str, processing_stage: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUDIO_PROCESSING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if processing_stage:
            self.details["processing_stage"] = processing_stage


class ServiceUnavailableError(AudioProcessingBaseError):
    """External service failures"""
    
    def __init__(self, message: str, service_name: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        if service_name:
            self.details["service_name"] = service_name


class ValidationError(AudioProcessingBaseError):
    """Input validation failures"""
    
    def __init__(self, message: str, validation_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details.update(validation_details or {})


class ConfigurationError(AudioProcessingBaseError):
    """Configuration issues"""
    
    def __init__(self, message: str, config_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(config_details or {})


class NetworkError(AudioProcessingBaseError):
    """Network-related errors"""
    
    def __init__(self, message: str, network_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.details.update(network_details or {})


class TimeoutError(AudioProcessingBaseError):
    """Timeout errors"""
    
    def __init__(self, message: str, timeout_details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update(timeout_details or {})


# Circuit Breaker Implementation
@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    is_half_open: bool = False
    success_count: int = 0


class CircuitBreaker:
    """Circuit breaker pattern implementation for service failures"""
    
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
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self._lock:
            if self._should_attempt_call():
                try:
                    result = await func(*args, **kwargs)
                    await self._on_success()
                    return result
                except Exception as e:
                    await self._on_failure(e)
                    raise
            else:
                raise ServiceUnavailableError(
                    f"Circuit breaker '{self.name}' is open",
                    service_name=self.name,
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
    
    async def _on_success(self):
        """Handle successful call"""
        if self.state.is_half_open:
            self.state.success_count += 1
            if self.state.success_count >= self.success_threshold:
                self.state.is_half_open = False
                self.state.failure_count = 0
                self.state.success_count = 0
        else:
            self.state.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
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


# Retry Mechanism with Exponential Backoff
@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryManager:
    """Retry mechanism with exponential backoff"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: tuple = (NetworkError, TimeoutError, ServiceUnavailableError),
        correlation_id: str = None,
        **kwargs
    ):
        """Execute function with retry mechanism"""
        correlation_id = correlation_id or str(uuid.uuid4())
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                logger.info(
                    f"[{correlation_id}] Attempt {attempt + 1}/{self.config.max_attempts} for {func.__name__}"
                )
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"[{correlation_id}] Retry successful after {attempt + 1} attempts")
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"[{correlation_id}] Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{correlation_id}] All {self.config.max_attempts} attempts failed"
                    )
            except Exception as e:
                # Non-retryable exception
                logger.error(f"[{correlation_id}] Non-retryable error: {str(e)}")
                raise
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add jitter
        
        return delay


# Error Recovery Strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    async def can_recover(self, error: AudioProcessingBaseError) -> bool:
        """Check if error can be recovered"""
        raise NotImplementedError
    
    async def recover(self, error: AudioProcessingBaseError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from error"""
        raise NotImplementedError


class FormatRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for audio format errors"""
    
    async def can_recover(self, error: AudioProcessingBaseError) -> bool:
        return isinstance(error, AudioFormatError)
    
    async def recover(self, error: AudioFormatError, context: Dict[str, Any]) -> Dict[str, Any]:
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


class ServiceRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for service unavailable errors"""
    
    async def can_recover(self, error: AudioProcessingBaseError) -> bool:
        return isinstance(error, ServiceUnavailableError)
    
    async def recover(self, error: ServiceUnavailableError, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt service fallback"""
        logger.info(f"[{error.correlation_id}] Attempting service recovery")
        
        fallback_services = context.get("fallback_services", [])
        
        return {
            "recovery_attempted": True,
            "fallback_services": fallback_services,
            "use_offline_mode": len(fallback_services) == 0
        }


# Error Logging with Correlation IDs
class ErrorLogger:
    """Centralized error logging with correlation IDs"""
    
    def __init__(self, logger_name: str = "audio_errors"):
        self.logger = logging.getLogger(logger_name)
    
    def log_error(
        self,
        error: AudioProcessingBaseError,
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
        error: AudioProcessingBaseError,
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
@asynccontextmanager
async def error_boundary(
    correlation_id: str = None,
    context: Dict[str, Any] = None,
    recovery_strategies: List[ErrorRecoveryStrategy] = None,
    circuit_breaker: CircuitBreaker = None,
    retry_manager: RetryManager = None
):
    """Error boundary context manager for comprehensive error handling"""
    correlation_id = correlation_id or str(uuid.uuid4())
    context = context or {}
    recovery_strategies = recovery_strategies or []
    error_logger = ErrorLogger()
    
    try:
        yield correlation_id
    except AudioProcessingBaseError as e:
        e.correlation_id = e.correlation_id or correlation_id
        error_logger.log_error(e, context)
        
        # Attempt recovery
        for strategy in recovery_strategies:
            if await strategy.can_recover(e):
                try:
                    recovery_result = await strategy.recover(e, context)
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
        # Convert to AudioProcessingBaseError
        audio_error = AudioProcessingBaseError(
            message=str(e),
            correlation_id=correlation_id,
            severity=ErrorSeverity.HIGH,
            details={"original_exception": str(type(e).__name__)}
        )
        error_logger.log_error(audio_error, context)
        raise audio_error


# Global instances
default_circuit_breaker = CircuitBreaker(name="audio_service")
default_retry_manager = RetryManager()
default_error_logger = ErrorLogger()