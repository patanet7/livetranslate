#!/usr/bin/env python3
"""
Orchestration Service - Consolidated CPU-Optimized Service Coordination

This service consolidates:
- Frontend web interface and API gateway (from frontend-service)
- WebSocket management and real-time communication (from websocket-service)
- Service health monitoring and auto-recovery (from monitoring-service)
- Session management and coordination
- Real-time dashboard and analytics

Features:
- Enterprise-grade WebSocket infrastructure
- API gateway with load balancing and circuit breaking
- Real-time service health monitoring
- Session persistence and recovery
- Comprehensive performance analytics
- Modern responsive web dashboard extracted from frontend-service
"""

import os
import sys
import json
import logging
import asyncio
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import uuid

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import requests

# Import orchestration components
from frontend.web_server import WebServer, add_web_routes
from websocket.connection_manager import EnterpriseConnectionManager
from gateway.api_gateway import APIGateway
from monitoring.health_monitor import ServiceHealthMonitor
from dashboard.real_time_dashboard import RealTimeDashboard
from utils.config_manager import ConfigManager
from utils.logger import setup_logging
from whisper_integration import add_whisper_routes

# Configure logging
logger = setup_logging(__name__)

class OrchestrationService:
    """Main orchestration service that consolidates all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize orchestration service with all components"""
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize service components
        self.web_server = WebServer(self.config.get('frontend', {}))
        self.websocket_manager = EnterpriseConnectionManager(
            **self.config.get('websocket', {})
        )
        self.api_gateway = APIGateway(self.config.get('gateway', {}))
        # Filter monitoring config to only include supported parameters
        monitoring_config = self.config.get('monitoring', {})
        health_monitor_config = {
            k: v for k, v in monitoring_config.items() 
            if k in ['health_check_interval', 'alert_threshold', 'recovery_threshold', 'auto_recovery']
        }
        self.health_monitor = ServiceHealthMonitor(**health_monitor_config)
        self.dashboard = RealTimeDashboard(self.config.get('dashboard', {}))
        
        # Service state
        self.running = False
        self.start_time = time.time()
        
        # Performance metrics
        self.metrics = {
            "start_time": self.start_time,
            "total_requests": 0,
            "active_sessions": 0,
            "websocket_connections": 0,
            "service_health_checks": 0,
            "errors": 0
        }
        
        logger.info("Orchestration service initialized with all components")
    
    async def start(self):
        """Start all service components"""
        try:
            logger.info("Starting orchestration service...")
            
            # Start components in parallel
            await asyncio.gather(
                self._start_web_server(),
                self._start_websocket_manager(),
                self._start_api_gateway(),
                self._start_health_monitor(),
                self._start_dashboard()
            )
            
            self.running = True
            logger.info(f"Orchestration service started successfully on port {self.config['frontend']['port']}")
            
        except Exception as e:
            logger.error(f"Failed to start orchestration service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop all service components"""
        logger.info("Stopping orchestration service...")
        
        self.running = False
        
        # Stop components
        if hasattr(self, 'web_server'):
            await self.web_server.stop()
        if hasattr(self, 'websocket_manager'):
            await self.websocket_manager.stop()
        if hasattr(self, 'api_gateway'):
            await self.api_gateway.stop()
        if hasattr(self, 'health_monitor'):
            await self.health_monitor.stop()
        if hasattr(self, 'dashboard'):
            await self.dashboard.stop()
        
        logger.info("Orchestration service stopped")
    
    async def _start_web_server(self):
        """Start the web server component"""
        await self.web_server.start()
    
    async def _start_websocket_manager(self):
        """Start the WebSocket manager"""
        await self.websocket_manager.start()
    
    async def _start_api_gateway(self):
        """Start the API gateway"""
        await self.api_gateway.start()
    
    async def _start_health_monitor(self):
        """Start the health monitor"""
        await self.health_monitor.start()
    
    async def _start_dashboard(self):
        """Start the real-time dashboard"""
        await self.dashboard.start()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        uptime = time.time() - self.start_time
        
        return {
            "service": "orchestration-service",
            "status": "healthy" if self.running else "stopped",
            "uptime": uptime,
            "components": {
                "web_server": self.web_server.get_status(),
                "websocket_manager": self.websocket_manager.get_statistics(),
                "api_gateway": self.api_gateway.get_status(),
                "health_monitor": self.health_monitor.get_status(),
                "dashboard": self.dashboard.get_status()
            },
            "metrics": self.metrics,
            "configuration": {
                "frontend_port": self.config['frontend']['port'],
                "websocket_max_connections": self.config['websocket']['max_connections'],
                "health_check_interval": self.config['monitoring']['health_check_interval']
            },
            "timestamp": time.time()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        return {
            **self.metrics,
            "websocket_stats": self.websocket_manager.get_statistics(),
            "gateway_stats": self.api_gateway.get_metrics(),
            "health_stats": self.health_monitor.get_metrics(),
            "dashboard_stats": self.dashboard.get_metrics()
        }


def create_flask_app(orchestration_service: OrchestrationService) -> tuple[Flask, SocketIO]:
    """Create Flask application with orchestration service integration"""
    
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    app.config['SECRET_KEY'] = orchestration_service.config.get('frontend', {}).get('secret_key', 'orchestration-secret-key')
    CORS(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Integrate WebSocket manager with SocketIO
    orchestration_service.websocket_manager.set_socketio(socketio)
    orchestration_service.dashboard.set_socketio(socketio)
    
    # Add web routes from frontend component
    add_web_routes(app, orchestration_service.web_server)
    
    # Add whisper integration routes
    whisper_integration = add_whisper_routes(
        app, 
        socketio, 
        orchestration_service.api_gateway, 
        orchestration_service.websocket_manager
    )
    
    # Orchestration-specific API endpoints
    @app.route('/api/health')
    def health_check():
        """Comprehensive orchestration service health check"""
        try:
            status = orchestration_service.get_service_status()
            # Add whisper integration stats
            status["whisper_integration"] = whisper_integration.get_statistics()
            return jsonify(status)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                "service": "orchestration-service",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }), 500
    
    # Performance metrics endpoint
    @app.route('/api/metrics')
    def get_metrics():
        """Get real-time performance metrics"""
        try:
            metrics = orchestration_service.get_performance_metrics()
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Dashboard data endpoint
    @app.route('/api/dashboard')
    def get_dashboard_data():
        """Get dashboard data"""
        try:
            data = orchestration_service.dashboard.get_dashboard_data()
            return jsonify(data)
        except Exception as e:
            logger.error(f"Dashboard data endpoint failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Service management endpoints
    @app.route('/api/services')
    def get_services():
        """Get backend service status"""
        try:
            services = orchestration_service.health_monitor.get_all_service_status()
            return jsonify(services)
        except Exception as e:
            logger.error(f"Services endpoint failed: {e}")
            return jsonify({"error": str(e)}), 500
    
    # API Gateway integration - proxy all /api/<service>/* requests  
    @app.route('/api/<service_name>/<path:api_path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
    def gateway_proxy(service_name, api_path):
        """Proxy requests through API gateway"""
        try:
            # Skip orchestration-specific endpoints
            if service_name in ['health', 'metrics', 'dashboard', 'services', 'config', 'settings']:
                return jsonify({"error": "Endpoint handled by orchestration service"}), 400
            
            return orchestration_service.api_gateway.route_request(
                service_name, api_path, request
            )
        except Exception as e:
            logger.error(f"Gateway proxy error: {e}")
            return jsonify({"error": str(e)}), 500
    
    # WebSocket events for orchestration
    @socketio.on('connect')
    def handle_websocket_connect():
        """Handle WebSocket connection"""
        result = orchestration_service.websocket_manager.handle_connect(request)
        emit('connected', result)
        return result
    
    @socketio.on('disconnect')
    def handle_websocket_disconnect():
        """Handle WebSocket disconnection"""
        return orchestration_service.websocket_manager.handle_disconnect(request)
    
    @socketio.on('service_message')
    def handle_service_message(data):
        """Handle service-specific WebSocket messages"""
        return orchestration_service.websocket_manager.handle_service_message(
            request, data
        )
    
    # WebSocket events from frontend-service functionality
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to service updates"""
        service_type = data.get('service')
        emit('subscribed', {'service': service_type, 'status': 'subscribed'})
    
    @socketio.on('transcription_stream')
    def handle_transcription_stream(data):
        """Handle real-time transcription streaming"""
        emit('transcription_received', {
            'session_id': request.sid,
            'timestamp': datetime.now().isoformat(),
            'status': 'received'
        })
    
    @socketio.on('speaker_stream')
    def handle_speaker_stream(data):
        """Handle real-time speaker diarization streaming"""
        emit('speaker_received', {
            'session_id': request.sid,
            'timestamp': datetime.now().isoformat(),
            'status': 'received'
        })
    
    @socketio.on('translation_stream')
    def handle_translation_stream(data):
        """Handle real-time translation streaming"""
        emit('translation_received', {
            'session_id': request.sid,
            'timestamp': datetime.now().isoformat(),
            'status': 'received'
        })
    
    return app, socketio


async def main():
    """Main entry point for orchestration service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Orchestration Service")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=3000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        # Create orchestration service
        orchestration_service = OrchestrationService(args.config)
        
        # Create Flask app with SocketIO
        app, socketio = create_flask_app(orchestration_service)
        
        # Start orchestration service
        await orchestration_service.start()
        
        # Start background metrics collection
        def metrics_update_loop():
            while True:
                try:
                    orchestration_service.dashboard.update_metrics(orchestration_service)
                    orchestration_service.dashboard.broadcast_metrics_update()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Metrics update failed: {e}")
                    time.sleep(5)
        
        import threading
        metrics_thread = threading.Thread(target=metrics_update_loop, daemon=True)
        metrics_thread.start()
        
        # Run Flask app with SocketIO
        logger.info(f"Starting orchestration service on {args.host}:{args.port}")
        logger.info("Dashboard available at: http://{}:{}".format(args.host, args.port))
        
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down orchestration service...")
        if 'orchestration_service' in locals():
            await orchestration_service.stop()
    except Exception as e:
        logger.error(f"Orchestration service failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())