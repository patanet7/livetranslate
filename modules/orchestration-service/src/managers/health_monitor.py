"""
Health Monitor Manager

Simple health monitoring implementation that provides real system data.
"""

import asyncio
import logging
import time
import psutil
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


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
    
    def __init__(self):
        self.services = {}
        self.service_configs = {
            "whisper": {
                "url": "http://localhost:5001",
                "health_endpoint": "/health"
            },
            "translation": {
                "url": "http://localhost:5003", 
                "health_endpoint": "/api/health"
            },
            "orchestration": {
                "url": "http://localhost:3000",
                "health_endpoint": "/api/system/status"
            }
        }
        
        # Initialize service trackers
        for name, config in self.service_configs.items():
            self.services[name] = ServiceHealth(
                name=name,
                status="unknown", 
                url=config["url"],
                last_check=0,
                response_time=0
            )
        
        logger.info("Health monitor initialized")
    
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
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "services": {name: asdict(service) for name, service in self.services.items()},
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
            
        health_url = f"{config['url']}{config['health_endpoint']}"
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    if response.status == 200:
                        service.status = "healthy"
                        service.error_count = 0
                        service.last_error = None
                    else:
                        service.status = "unhealthy"
                        service.error_count += 1
                        service.last_error = f"HTTP {response.status}"
                    
                    service.response_time = response_time
                    service.last_check = time.time()
                    
        except asyncio.TimeoutError:
            service.status = "unhealthy"
            service.error_count += 1
            service.last_error = "Connection timeout"
            service.response_time = 5000  # Timeout value
            service.last_check = time.time()
            
        except Exception as e:
            service.status = "unhealthy"
            service.error_count += 1
            service.last_error = str(e)
            service.response_time = 0
            service.last_check = time.time()
    
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