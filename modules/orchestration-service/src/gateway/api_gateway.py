#!/usr/bin/env python3
"""
API Gateway Component

Provides intelligent request routing, load balancing, and circuit breaking
for backend services. Integrates with the frontend service proxy functionality.
"""

import time
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import threading

from flask import request, jsonify, Response

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class ServiceInstance:
    """Individual service instance"""

    def __init__(self, url: str, weight: int = 1):
        self.url = url
        self.weight = weight
        self.status = ServiceStatus.UNKNOWN
        self.last_check = 0
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0


class CircuitBreaker:
    """Circuit breaker for service protection"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()

    def call_allowed(self) -> bool:
        """Check if call is allowed through circuit breaker"""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            else:  # HALF_OPEN
                return True

    def record_success(self):
        """Record successful call"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED

    def record_failure(self):
        """Record failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class APIGateway:
    """API Gateway with load balancing and circuit breaking"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize API gateway"""
        self.config = config
        self.timeout = config.get("timeout", 30)
        self.retries = config.get("retries", 3)
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 5)

        # Service registry
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancer_state: Dict[str, int] = defaultdict(int)  # Round-robin state

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "average_response_time": 0,
            "requests_per_service": defaultdict(int),
            "errors_per_service": defaultdict(int),
        }

        self._lock = threading.RLock()
        self.running = False
        self.start_time = time.time()

        # Initialize default services
        self._initialize_default_services()

        logger.info("API Gateway initialized")

    def _initialize_default_services(self):
        """Initialize default service configurations"""
        default_services = {
            "whisper": {"url": "http://localhost:5001", "health_endpoint": "/health"},
            "speaker": {"url": "http://localhost:5002", "health_endpoint": "/health"},
            "translation": {
                "url": "http://localhost:5003",
                "health_endpoint": "/api/health",
            },
        }

        for service_name, config in default_services.items():
            self.register_service(service_name, config["url"])
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold
            )

    async def start(self):
        """Start the API gateway"""
        self.running = True
        logger.info("API Gateway started")

    async def stop(self):
        """Stop the API gateway"""
        self.running = False
        logger.info("API Gateway stopped")

    def register_service(self, service_name: str, url: str, weight: int = 1):
        """Register a service instance"""
        with self._lock:
            if service_name not in self.services:
                self.services[service_name] = []

            instance = ServiceInstance(url, weight)
            self.services[service_name].append(instance)

            logger.info(f"Registered service: {service_name} at {url}")

    def route_request(
        self, service_name: str, api_path: str, flask_request
    ) -> Response:
        """Route request to appropriate service with load balancing and circuit breaking"""
        try:
            # Debug incoming request
            logger.info(
                f"Gateway routing {flask_request.method} {api_path} to {service_name}"
            )
            logger.info(f"Request content type: {flask_request.content_type}")
            logger.info(f"Request has files: {bool(flask_request.files)}")
            if flask_request.files:
                logger.info(f"Files: {list(flask_request.files.keys())}")

            # Update metrics
            with self._lock:
                self.metrics["total_requests"] += 1
                self.metrics["requests_per_service"][service_name] += 1

            # Check if service exists
            if service_name not in self.services:
                logger.warning(f"Unknown service: {service_name}")
                return jsonify({"error": f"Unknown service: {service_name}"}), 404

            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker and not circuit_breaker.call_allowed():
                logger.warning(f"Circuit breaker open for service: {service_name}")
                with self._lock:
                    self.metrics["circuit_breaker_trips"] += 1
                return (
                    jsonify(
                        {
                            "error": f"Service {service_name} temporarily unavailable",
                            "circuit_breaker": "open",
                        }
                    ),
                    503,
                )

            # Select service instance (load balancing)
            instance = self._select_service_instance(service_name)
            if not instance:
                logger.error(f"No healthy instances for service: {service_name}")
                return (
                    jsonify({"error": f"No healthy instances for {service_name}"}),
                    503,
                )

            # Build target URL
            target_url = f"{instance.url.rstrip('/')}/{api_path.lstrip('/')}"

            # Prepare request
            request_data = self._prepare_request_data(flask_request)

            # Make request with retries
            response = self._make_request_with_retries(
                flask_request.method,
                target_url,
                request_data,
                service_name,
                circuit_breaker,
            )

            return response

        except Exception as e:
            logger.error(f"Gateway routing failed: {e}")
            with self._lock:
                self.metrics["failed_requests"] += 1
                self.metrics["errors_per_service"][service_name] += 1

            return jsonify({"error": "Gateway routing error", "details": str(e)}), 500

    def _select_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Select service instance using weighted round-robin"""
        with self._lock:
            instances = self.services.get(service_name, [])
            if not instances:
                return None

            # Filter healthy instances
            healthy_instances = [
                inst
                for inst in instances
                if inst.status in [ServiceStatus.HEALTHY, ServiceStatus.UNKNOWN]
            ]
            if not healthy_instances:
                # Fall back to any instance if none are marked healthy
                healthy_instances = instances

            if len(healthy_instances) == 1:
                return healthy_instances[0]

            # Weighted round-robin selection
            current_index = self.load_balancer_state[service_name]
            selected_instance = healthy_instances[
                current_index % len(healthy_instances)
            ]
            self.load_balancer_state[service_name] = (current_index + 1) % len(
                healthy_instances
            )

            return selected_instance

    def _prepare_request_data(self, flask_request) -> Dict[str, Any]:
        """Prepare request data for forwarding"""
        request_data = {}

        # Debug incoming request details
        logger.info(f"Gateway _prepare_request_data called:")
        logger.info(f"  Method: {flask_request.method}")
        logger.info(f"  Content-Type: {flask_request.content_type}")
        logger.info(
            f"  Content-Length: {flask_request.environ.get('CONTENT_LENGTH', 'unknown')}"
        )
        logger.info(f"  Files available: {bool(flask_request.files)}")
        logger.info(f"  Form available: {bool(flask_request.form)}")
        logger.info(
            f"  Raw data length: {len(flask_request.data) if flask_request.data else 0}"
        )

        # Handle different request types
        if flask_request.method in ["POST", "PUT", "PATCH"]:
            if flask_request.is_json:
                logger.info("Processing JSON request")
                request_data["json"] = flask_request.get_json()
            elif flask_request.files:
                # Handle file uploads - preserve multipart form data
                logger.info(
                    f"Gateway processing files: {list(flask_request.files.keys())}"
                )
                files = {}
                for key, file in flask_request.files.items():
                    # Make sure to seek to beginning of file
                    file.seek(0)
                    file_content = file.read()
                    logger.info(
                        f"Gateway read file {key}: {file.filename} ({len(file_content)} bytes, {file.content_type})"
                    )

                    # Verify we actually read the content
                    if len(file_content) == 0:
                        logger.error(
                            f"Gateway read 0 bytes from file {key} - file may be empty or already consumed"
                        )
                        # Try to read from raw data as fallback
                        logger.info(f"Attempting fallback to raw data for file {key}")
                        continue

                    # Reset file position after reading for requests library
                    file.seek(0)

                    # Create file tuple for requests library
                    # Format: (filename, file_object_or_content, content_type)
                    # Use the actual file object instead of content to preserve streaming
                    files[key] = (
                        file.filename or "upload",
                        file_content,
                        file.content_type or "application/octet-stream",
                    )
                    logger.info(
                        f"Gateway created file tuple for {key}: filename={file.filename}, size={len(file_content)}, type={file.content_type}"
                    )

                if files:
                    request_data["files"] = files
                else:
                    logger.error(
                        "Gateway: No valid files found despite flask_request.files being truthy"
                    )

                # Also include form data if present alongside files
                if flask_request.form:
                    logger.info(
                        f"Gateway processing form data: {list(flask_request.form.keys())}"
                    )
                    request_data["data"] = flask_request.form.to_dict()
            elif flask_request.form:
                logger.info("Processing form data")
                request_data["data"] = flask_request.form.to_dict()
            elif flask_request.data:
                # Handle raw data for multipart requests that Flask might not parse properly
                logger.info(f"Processing raw data: {len(flask_request.data)} bytes")
                logger.info(f"Raw data preview: {flask_request.data[:100]}")
                # For multipart, we need to handle this differently
                if (
                    flask_request.content_type
                    and "multipart/form-data" in flask_request.content_type
                ):
                    logger.warning(
                        "Found multipart/form-data in raw data - Flask parsing may have failed"
                    )
                    # Try to parse manually or pass as raw data
                    request_data["data"] = flask_request.data
                    # Also try to set content-type for requests library
                    request_data["headers"] = {
                        "Content-Type": flask_request.content_type
                    }

        # Add query parameters
        if flask_request.args:
            request_data["params"] = flask_request.args.to_dict()

        # Prepare headers (filter out problematic ones)
        if "headers" not in request_data:
            headers = {}
        else:
            headers = request_data["headers"]

        skip_headers = {
            "host",
            "content-length",
            "connection",
            "upgrade",
            "sec-websocket-key",
            "content-type",
        }
        for key, value in flask_request.headers:
            if key.lower() not in skip_headers:
                headers[key] = value

        # Add forwarding headers
        headers["X-Forwarded-For"] = flask_request.environ.get("REMOTE_ADDR", "")
        headers["X-Forwarded-Proto"] = flask_request.scheme
        headers["X-Gateway-Timestamp"] = str(time.time())

        # Don't set Content-Type header when we have files - requests will set the correct multipart boundary
        if "files" not in request_data:
            # Only set content-type for non-file requests
            if flask_request.content_type:
                headers["Content-Type"] = flask_request.content_type

        request_data["headers"] = headers
        request_data["timeout"] = self.timeout

        # Debug logging for final request data
        logger.info(f"Gateway prepared request data:")
        logger.info(f"  Keys: {list(request_data.keys())}")
        if "files" in request_data:
            logger.info(f"  Files: {list(request_data['files'].keys())}")
            for key, file_info in request_data["files"].items():
                filename, content, content_type = file_info
                logger.info(
                    f"    {key}: {filename} ({len(content)} bytes, {content_type})"
                )
        if "data" in request_data:
            data_type = type(request_data["data"])
            if isinstance(request_data["data"], bytes):
                logger.info(f"  Data: {len(request_data['data'])} bytes (raw)")
            elif isinstance(request_data["data"], dict):
                logger.info(f"  Data: {len(request_data['data'])} fields (dict)")
            else:
                logger.info(f"  Data: {data_type}")

        return request_data

    def _make_request_with_retries(
        self,
        method: str,
        url: str,
        request_data: Dict,
        service_name: str,
        circuit_breaker: Optional[CircuitBreaker],
    ) -> Response:
        """Make request with retry logic"""
        last_exception = None
        start_time = time.time()

        for attempt in range(self.retries):
            try:
                # Debug the actual request being made
                if "files" in request_data:
                    logger.info(f"Making request to {url} with files:")
                    for key, file_info in request_data["files"].items():
                        if isinstance(file_info, tuple) and len(file_info) >= 2:
                            filename, content = file_info[0], file_info[1]
                            content_len = (
                                len(content)
                                if hasattr(content, "__len__")
                                else "unknown"
                            )
                            logger.info(
                                f"  File {key}: {filename} ({content_len} bytes)"
                            )
                        else:
                            logger.info(f"  File {key}: {file_info}")

                    # For requests library, we need to use BytesIO objects for file content
                    import io

                    fixed_files = {}
                    for key, file_info in request_data["files"].items():
                        if isinstance(file_info, tuple) and len(file_info) >= 2:
                            filename, content, content_type = file_info
                            # Convert content to BytesIO for requests library
                            file_obj = io.BytesIO(content)
                            file_obj.seek(0)  # Ensure we're at the beginning
                            fixed_files[key] = (filename, file_obj, content_type)
                            logger.info(
                                f"Gateway converted file {key} to BytesIO: {filename} ({len(content)} bytes, {content_type})"
                            )
                        else:
                            fixed_files[key] = file_info
                    request_data["files"] = fixed_files

                # Make the actual request with proper SSL handling
                # For HTTP URLs, disable SSL verification to avoid SSL errors
                if url.startswith("http://"):
                    request_data["verify"] = False
                    logger.debug(f"Using HTTP connection without SSL for {url}")
                
                response = requests.request(method, url, **request_data)

                # Record timing
                response_time = (time.time() - start_time) * 1000  # ms
                self._record_request_metrics(service_name, response_time, True)

                # Record circuit breaker success
                if circuit_breaker:
                    circuit_breaker.record_success()

                # Create Flask response
                flask_response = Response(
                    response.content,
                    status=response.status_code,
                    headers=dict(response.headers),
                )

                # Add gateway headers
                flask_response.headers["X-Gateway-Service"] = service_name
                flask_response.headers["X-Gateway-Response-Time"] = str(response_time)
                flask_response.headers["X-Gateway-Attempt"] = str(attempt + 1)

                return flask_response

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(
                    f"Request attempt {attempt + 1} failed for {service_name}: {e}"
                )

                # Record failure
                self._record_request_metrics(service_name, 0, False)
                if circuit_breaker:
                    circuit_breaker.record_failure()

                # Don't retry on certain errors
                if isinstance(
                    e,
                    (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
                ):
                    if attempt < self.retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue

                break

        # All retries failed
        logger.error(f"All retries failed for {service_name}: {last_exception}")
        with self._lock:
            self.metrics["failed_requests"] += 1
            self.metrics["errors_per_service"][service_name] += 1

        return (
            jsonify(
                {
                    "error": f"Service {service_name} unavailable",
                    "details": str(last_exception),
                    "attempts": self.retries,
                }
            ),
            503,
        )

    def _record_request_metrics(
        self, service_name: str, response_time: float, success: bool
    ):
        """Record request metrics"""
        with self._lock:
            if success:
                self.metrics["successful_requests"] += 1

                # Update average response time (rolling average)
                current_avg = self.metrics["average_response_time"]
                total_requests = self.metrics["successful_requests"]
                self.metrics["average_response_time"] = (
                    current_avg * (total_requests - 1) + response_time
                ) / total_requests
            else:
                self.metrics["failed_requests"] += 1

    def health_check_service(
        self, service_name: str, instance: ServiceInstance
    ) -> bool:
        """Perform health check on service instance"""
        try:
            health_url = f"{instance.url}/health"
            
            # Configure session for HTTP endpoints - disable SSL verification
            session = requests.Session()
            if health_url.startswith("http://"):
                # For HTTP URLs, ensure no SSL verification
                session.verify = False
                logger.debug(f"Using HTTP connection without SSL for {service_name} at {health_url}")
            
            response = session.get(health_url, timeout=5)

            if response.status_code == 200:
                instance.status = ServiceStatus.HEALTHY
                instance.last_check = time.time()
                logger.debug(f"Health check passed for {service_name} at {health_url}")
                return True
            else:
                instance.status = ServiceStatus.UNHEALTHY
                logger.warning(f"Health check failed for {service_name}: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            instance.status = ServiceStatus.UNHEALTHY
            return False

    def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services"""
        health_status = {}

        with self._lock:
            for service_name, instances in self.services.items():
                service_health = {
                    "instances": [],
                    "healthy_count": 0,
                    "total_count": len(instances),
                    "circuit_breaker": self.circuit_breakers.get(
                        service_name, {}
                    ).state.value
                    if service_name in self.circuit_breakers
                    else "none",
                }

                for instance in instances:
                    instance_health = {
                        "url": instance.url,
                        "status": instance.status.value,
                        "last_check": instance.last_check,
                        "response_times": instance.response_times[
                            -10:
                        ],  # Last 10 response times
                        "error_count": instance.error_count,
                        "success_count": instance.success_count,
                    }
                    service_health["instances"].append(instance_health)

                    if instance.status == ServiceStatus.HEALTHY:
                        service_health["healthy_count"] += 1

                health_status[service_name] = service_health

        return health_status

    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics"""
        with self._lock:
            return {
                **self.metrics,
                "uptime": time.time() - self.start_time,
                "requests_per_second": self.metrics["total_requests"]
                / max(time.time() - self.start_time, 1),
                "success_rate": (
                    self.metrics["successful_requests"]
                    / max(self.metrics["total_requests"], 1)
                    * 100
                ),
                "service_health": self.get_service_health(),
            }

    def get_status(self) -> Dict[str, Any]:
        """Get gateway status"""
        return {
            "component": "api_gateway",
            "status": "running" if self.running else "stopped",
            "uptime": time.time() - self.start_time,
            "configuration": {
                "timeout": self.timeout,
                "retries": self.retries,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "registered_services": list(self.services.keys()),
            },
            "metrics": self.get_metrics(),
        }
