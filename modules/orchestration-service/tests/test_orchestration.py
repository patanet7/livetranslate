#!/usr/bin/env python3
"""
Basic tests for the orchestration service to verify consolidation is working.
"""

import unittest
import asyncio
import time
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestration_service import OrchestrationService
from frontend.web_server import WebServer
from websocket.connection_manager import EnterpriseConnectionManager
from gateway.api_gateway import APIGateway
from monitoring.health_monitor import ServiceHealthMonitor
from dashboard.real_time_dashboard import RealTimeDashboard
from utils.config_manager import ConfigManager


class TestOrchestrationService(unittest.TestCase):
    """Test orchestration service consolidation"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "frontend": {"host": "localhost", "port": 3000},
            "websocket": {"max_connections": 100, "connection_timeout": 300},
            "gateway": {"timeout": 30, "retries": 3},
            "monitoring": {"health_check_interval": 10},
            "dashboard": {"refresh_interval": 5},
        }

    def test_orchestration_service_initialization(self):
        """Test that orchestration service initializes all components"""
        with patch("utils.config_manager.ConfigManager") as mock_config:
            mock_config.return_value.get_config.return_value = self.test_config

            service = OrchestrationService()

            # Verify all components are initialized
            self.assertIsInstance(service.web_server, WebServer)
            self.assertIsInstance(
                service.websocket_manager, EnterpriseConnectionManager
            )
            self.assertIsInstance(service.api_gateway, APIGateway)
            self.assertIsInstance(service.health_monitor, ServiceHealthMonitor)
            self.assertIsInstance(service.dashboard, RealTimeDashboard)
            self.assertIsInstance(service.config_manager, ConfigManager)

    def test_web_server_component(self):
        """Test web server component functionality"""
        config = self.test_config["frontend"]
        web_server = WebServer(config)

        # Test configuration
        self.assertEqual(web_server.host, "localhost")
        self.assertEqual(web_server.port, 3000)

        # Test status
        status = web_server.get_status()
        self.assertEqual(status["component"], "web_server")
        self.assertIn("configuration", status)

    def test_websocket_manager_component(self):
        """Test WebSocket manager functionality"""
        manager = EnterpriseConnectionManager(
            max_connections=100, connection_timeout=300
        )

        # Test configuration
        self.assertEqual(manager.max_connections, 100)
        self.assertEqual(manager.connection_timeout, 300)

        # Test statistics
        stats = manager.get_statistics()
        self.assertIn("active_connections", stats)
        self.assertIn("configuration", stats)

    def test_api_gateway_component(self):
        """Test API gateway functionality"""
        config = self.test_config["gateway"]
        gateway = APIGateway(config)

        # Test service registration
        gateway.register_service("test-service", "http://localhost:5000")
        self.assertIn("test-service", gateway.services)

        # Test status
        status = gateway.get_status()
        self.assertEqual(status["component"], "api_gateway")
        self.assertIn("metrics", status)

    def test_health_monitor_component(self):
        """Test health monitor functionality"""
        monitor = ServiceHealthMonitor(health_check_interval=10)

        # Test service registration
        monitor.register_service("test-service", "http://localhost:5000")
        self.assertIn("test-service", monitor.services)

        # Test status
        status = monitor.get_status()
        self.assertEqual(status["component"], "health_monitor")
        self.assertIn("configuration", status)

    def test_dashboard_component(self):
        """Test dashboard functionality"""
        config = self.test_config["dashboard"]
        dashboard = RealTimeDashboard(config)

        # Test configuration
        self.assertEqual(dashboard.refresh_interval, 5)

        # Test metrics collection
        dashboard.metrics_collector.record_metric("test_metric", 42.0)
        current_metrics = dashboard.metrics_collector.get_current_metrics()
        self.assertEqual(current_metrics["test_metric"], 42.0)

        # Test status
        status = dashboard.get_status()
        self.assertEqual(status["component"], "real_time_dashboard")

    def test_config_manager_component(self):
        """Test configuration manager functionality"""
        config_manager = ConfigManager()

        # Test default configuration
        config = config_manager.get_config()
        self.assertIn("orchestration", config)
        self.assertIn("frontend", config)
        self.assertIn("websocket", config)

        # Test configuration access
        port = config_manager.get("frontend.port", 3000)
        self.assertEqual(port, 3000)

    async def test_service_startup_sequence(self):
        """Test that all services start up correctly"""
        with patch("utils.config_manager.ConfigManager") as mock_config:
            mock_config.return_value.get_config.return_value = self.test_config

            service = OrchestrationService()

            # Mock the start methods to avoid actual network operations
            service.web_server.start = Mock(return_value=asyncio.Future())
            service.web_server.start.return_value.set_result(None)

            service.websocket_manager.start = Mock(return_value=asyncio.Future())
            service.websocket_manager.start.return_value.set_result(None)

            service.api_gateway.start = Mock(return_value=asyncio.Future())
            service.api_gateway.start.return_value.set_result(None)

            service.health_monitor.start = Mock(return_value=asyncio.Future())
            service.health_monitor.start.return_value.set_result(None)

            service.dashboard.start = Mock(return_value=asyncio.Future())
            service.dashboard.start.return_value.set_result(None)

            # Test startup
            await service.start()

            # Verify all components were started
            service.web_server.start.assert_called_once()
            service.websocket_manager.start.assert_called_once()
            service.api_gateway.start.assert_called_once()
            service.health_monitor.start.assert_called_once()
            service.dashboard.start.assert_called_once()

    def test_service_status_integration(self):
        """Test that service status includes all components"""
        with patch("utils.config_manager.ConfigManager") as mock_config:
            mock_config.return_value.get_config.return_value = self.test_config

            service = OrchestrationService()
            service.running = True
            service.start_time = time.time() - 100  # Running for 100 seconds

            status = service.get_service_status()

            # Verify overall status
            self.assertEqual(status["service"], "orchestration-service")
            self.assertEqual(status["status"], "healthy")
            self.assertAlmostEqual(status["uptime"], 100, delta=1)

            # Verify all components are included
            self.assertIn("components", status)
            components = status["components"]
            self.assertIn("web_server", components)
            self.assertIn("websocket_manager", components)
            self.assertIn("api_gateway", components)
            self.assertIn("health_monitor", components)
            self.assertIn("dashboard", components)

    def test_performance_metrics_integration(self):
        """Test that performance metrics include all components"""
        with patch("utils.config_manager.ConfigManager") as mock_config:
            mock_config.return_value.get_config.return_value = self.test_config

            service = OrchestrationService()

            metrics = service.get_performance_metrics()

            # Verify metrics structure
            self.assertIn("websocket_stats", metrics)
            self.assertIn("gateway_stats", metrics)
            self.assertIn("health_stats", metrics)
            self.assertIn("dashboard_stats", metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction"""

    def test_websocket_manager_socketio_integration(self):
        """Test WebSocket manager SocketIO integration"""
        manager = EnterpriseConnectionManager()

        # Mock SocketIO instance
        mock_socketio = Mock()
        manager.set_socketio(mock_socketio)

        self.assertEqual(manager._socketio, mock_socketio)

    def test_dashboard_socketio_integration(self):
        """Test dashboard SocketIO integration"""
        dashboard = RealTimeDashboard({"refresh_interval": 5})

        # Mock SocketIO instance
        mock_socketio = Mock()
        dashboard.set_socketio(mock_socketio)

        self.assertEqual(dashboard._socketio, mock_socketio)


if __name__ == "__main__":
    # Run async tests
    def run_async_test(test_func):
        """Helper to run async test functions"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_func())
        finally:
            loop.close()

    # Run the async test separately
    test_instance = TestOrchestrationService()
    test_instance.setUp()
    print("Running async startup test...")
    run_async_test(test_instance.test_service_startup_sequence)
    print("Async test completed successfully!")

    # Run regular unit tests
    unittest.main(verbosity=2)
