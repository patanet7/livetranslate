"""
Health Monitor Manager

Simple health monitoring implementation that provides real system data.
"""

import asyncio
import logging
import time
import psutil
import aiohttp
import ssl
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)
# Set health monitor to only show warnings and errors to reduce console noise
logger.setLevel(logging.WARNING)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


@dataclass
class ServiceHealth:
    """Service health information"""
    name: str
    status: str  # healthy, unhealthy, degraded, unknown
    url: str
    last_check: float
    response_time: float
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass 
class SystemMetrics:
    """System performance metrics"""
    cpu: Dict[str, float]
    memory: Dict[str, float]
    disk: Dict[str, float]
    network: Dict[str, int]
    timestamp: float


class HealthMonitor:
    """Simple health monitoring manager"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.services = {}
        
        # Get service URLs from configuration if available, otherwise use defaults
        if settings:
            audio_service_url = settings.services.audio_service_url
            translation_service_url = settings.services.translation_service_url
            orchestration_url = f"http://{settings.host}:{settings.port}"
        else:
            # Fallback defaults
            audio_service_url = "http://localhost:5001"
            translation_service_url = "http://localhost:5003"
            orchestration_url = "http://localhost:3000"
        
        self.service_configs = {
            "whisper": {
                "url": audio_service_url,
                "health_endpoint": "/health"
            },
            "translation": {
                "url": translation_service_url,
                "health_endpoint": "/api/health"
            },
            "orchestration": {
                "url": orchestration_url,
                "health_endpoint": "/api/system/health"
            }
        }
        
        # Create SSL context that doesn't verify certificates for localhost
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize service trackers
        for name, config in self.service_configs.items():
            self.services[name] = ServiceHealth(
                name=name,
                status="unknown", 
                url=config["url"],
                last_check=0,
                response_time=0
            )
        
        logger.debug(f"Health monitor initialized with service URLs: {[config['url'] for config in self.service_configs.values()]}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        try:
            # Check all services
            await self._check_all_services()
            
            # Get performance metrics  
            performance = await self.get_performance_metrics()
            
            # Calculate overall status
            service_statuses = [s.status for s in self.services.values()]
            if all(s == "healthy" for s in service_statuses):
                overall_status = "healthy"
            elif any(s == "unhealthy" for s in service_statuses):
                overall_status = "degraded" 
            else:
                overall_status = "unknown"
            
            services_dict = {name: asdict(service) for name, service in self.services.items()}
            logger.debug(f"Health check returning services: {list(services_dict.keys())}")
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "services": services_dict,
                "performance": asdict(performance),
                "uptime": time.time() - psutil.boot_time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "status": "unknown",
                "timestamp": time.time(),
                "error": str(e),
                "services": {},
                "performance": {},
                "uptime": 0
            }
    
    async def get_performance_metrics(self) -> SystemMetrics:
        """Get system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics  
            disk = psutil.disk_usage('/')
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            return SystemMetrics(
                cpu={
                    "usage": cpu_percent,
                    "cores": cpu_count
                },
                memory={
                    "used": memory.used,
                    "total": memory.total,
                    "percentage": memory.percent
                },
                disk={
                    "used": disk.used,
                    "total": disk.total, 
                    "percentage": (disk.used / disk.total) * 100
                },
                network={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return SystemMetrics(
                cpu={"usage": 0, "cores": 1},
                memory={"used": 0, "total": 1, "percentage": 0}, 
                disk={"used": 0, "total": 1, "percentage": 0},
                network={"bytes_sent": 0, "bytes_recv": 0},
                timestamp=time.time()
            )
    
    async def _check_all_services(self):
        """Check health of all services"""
        tasks = []
        for name in self.services.keys():
            tasks.append(self._check_service_health(name))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service_name: str):
        """Check health of a single service"""
        service = self.services.get(service_name)
        if not service:
            return
            
        config = self.service_configs.get(service_name)
        if not config:
            return

        # Special handling for orchestration service (self-check)
        if service_name == "orchestration":
            # If we're able to execute this method, orchestration is healthy
            service.status = "healthy"
            service.error_count = 0
            service.last_error = None
            service.response_time = 1  # Minimal response time for self-check
            service.last_check = time.time()
            logger.debug(f"Orchestration service self-check: healthy")
            return
            
        health_url = f"{config['url']}{config['health_endpoint']}"
        
        try:
            start_time = time.time()
            logger.debug(f"Checking health for {service_name} at: {health_url}")
            
            # For HTTP URLs, don't use any SSL configuration
            if health_url.startswith("http://"):
                logger.debug(f"Using plain HTTP connection for {service_name}")
                # Explicitly disable SSL for HTTP connections
                connector = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(health_url) as response:
                        response_time = (time.time() - start_time) * 1000  # ms
                        
                        if response.status == 200:
                            service.status = "healthy"
                            service.error_count = 0
                            service.last_error = None
                            logger.debug(f"Service {service_name} health check successful: {health_url}")
                        else:
                            service.status = "unhealthy"
                            service.error_count += 1
                            service.last_error = f"HTTP {response.status}"
                            logger.warning(f"Service {service_name} health check failed with HTTP {response.status}: {health_url}")
                        
                        service.response_time = response_time
                        service.last_check = time.time()
            else:
                # HTTPS URLs need SSL context
                logger.debug(f"Using HTTPS connection for {service_name}")
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as session:
                    async with session.get(health_url) as response:
                        response_time = (time.time() - start_time) * 1000  # ms
                        
                        if response.status == 200:
                            service.status = "healthy"
                            service.error_count = 0
                            service.last_error = None
                            logger.debug(f"Service {service_name} health check successful: {health_url}")
                        else:
                            service.status = "unhealthy"
                            service.error_count += 1
                            service.last_error = f"HTTP {response.status}"
                            logger.warning(f"Service {service_name} health check failed with HTTP {response.status}: {health_url}")
                        
                        service.response_time = response_time
                        service.last_check = time.time()
                    
        except asyncio.TimeoutError:
            service.status = "unhealthy"
            service.error_count += 1
            service.last_error = "Connection timeout"
            service.response_time = 5000  # Timeout value
            service.last_check = time.time()
            logger.warning(f"Service {service_name} health check timeout: {health_url}")
            
        except aiohttp.ClientConnectorError as e:
            # More specific error handling for connection issues
            error_msg = str(e)
            
            # Check if this is actually just a service being down (not an SSL issue)
            if "connect call failed" in error_msg.lower() or "connection refused" in error_msg.lower():
                service.status = "unhealthy"
                service.error_count += 1
                service.last_error = f"Service unavailable: {service_name} not running"
                logger.warning(f"Service {service_name} appears to be down: {health_url}")
                logger.debug(f"  Connection error: {error_msg}")
            # Check if this is an SSL error on an HTTP endpoint
            elif "ssl" in error_msg.lower() and health_url.startswith("http://"):
                service.status = "unhealthy"
                service.error_count += 1
                service.last_error = "SSL error on HTTP endpoint - check configuration"
                logger.error(f"SSL error for {service_name} at {health_url} - this usually means HTTPS is being forced on an HTTP endpoint")
                logger.error(f"  Full error: {error_msg}")
            else:
                service.status = "unhealthy"
                service.error_count += 1
                service.last_error = f"Connection error: {error_msg}"
                logger.error(f"Connection error for {service_name} at {health_url}: {error_msg}")
            
            service.response_time = 0
            service.last_check = time.time()
            
        except Exception as e:
            service.status = "unhealthy"
            service.error_count += 1
            service.last_error = f"{e.__class__.__name__}: {str(e)}"
            service.response_time = 0
            service.last_check = time.time()
            logger.error(f"Service {service_name} health check error: {health_url}")
            logger.error(f"  Error type: {e.__class__.__name__}")
            logger.error(f"  Error details: {str(e)}")
            
            # Additional debug info for SSL errors
            if "ssl" in str(e).lower():
                logger.error(f"  This appears to be an SSL/TLS error")
                logger.error(f"  URL scheme: {health_url.split('://')[0]}")
                logger.error(f"  If using HTTP, ensure no HTTPS redirect is happening")
    
    async def get_all_services_status(self) -> List[Dict[str, Any]]:
        """Get status of all services"""
        await self._check_all_services()
        return [asdict(service) for service in self.services.values()]
    
    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific service"""
        await self._check_service_health(service_name)
        service = self.services.get(service_name)
        return asdict(service) if service else None
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        system_health = await self.get_system_health()
        return {
            **system_health,
            "detailed": True,
            "service_count": len(self.services),
            "last_updated": datetime.now().isoformat()
        }
    
    async def restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a service (placeholder implementation)"""
        return {
            "service": service_name,
            "action": "restart_initiated",
            "timestamp": time.time(),
            "note": "Service restart not implemented"
        }
    
    async def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return {
            "services": self.service_configs,
            "monitoring": {
                "health_check_interval": 30,
                "timeout": 5
            },
            "timestamp": time.time()
        }
    
    async def update_system_config(self, config_update: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration"""
        return {
            "updated_keys": list(config_update.keys()),
            "restart_required": False,
            "timestamp": time.time()
        }
    
    async def start_maintenance_mode(self) -> Dict[str, Any]:
        """Start maintenance mode"""
        return {
            "maintenance_mode": True,
            "started_at": time.time()
        }
    
    async def stop_maintenance_mode(self) -> Dict[str, Any]:
        """Stop maintenance mode"""
        return {
            "maintenance_mode": False,
            "stopped_at": time.time()
        }
    
    async def get_maintenance_status(self) -> Dict[str, Any]:
        """Get maintenance status"""
        return {
            "maintenance_mode": False,
            "timestamp": time.time()
        }
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service-specific metrics for analytics API"""
        try:
            # Get current service statuses
            await self._check_all_services()
            
            # Calculate service metrics
            total_services = len(self.services)
            healthy_services = sum(1 for s in self.services.values() if s.status == "healthy")
            unhealthy_services = sum(1 for s in self.services.values() if s.status == "unhealthy")
            degraded_services = sum(1 for s in self.services.values() if s.status == "degraded")
            
            # Calculate average response time
            response_times = [s.response_time for s in self.services.values() if s.response_time > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Calculate total error count
            total_errors = sum(s.error_count for s in self.services.values())
            
            return {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": unhealthy_services,
                "degraded_services": degraded_services,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
                "avg_response_time_ms": avg_response_time,
                "total_errors": total_errors,
                "services_detail": {
                    name: {
                        "status": service.status,
                        "response_time_ms": service.response_time,
                        "error_count": service.error_count,
                        "last_error": service.last_error,
                        "last_check": service.last_check,
                        "uptime_seconds": time.time() - service.last_check if service.last_check > 0 else 0
                    }
                    for name, service in self.services.items()
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            return {
                "total_services": 0,
                "healthy_services": 0,
                "unhealthy_services": 0,
                "degraded_services": 0,
                "health_percentage": 0,
                "avg_response_time_ms": 0,
                "total_errors": 0,
                "services_detail": {},
                "timestamp": time.time(),
                "error": str(e)
            }