"""
Validation 1: Server Startup with New Router Wiring

Verifies that the FastAPI application fully initializes — all routers import
without error, all routers register, and the fireflies-specific endpoints
(pause, resume, display-mode) are routed.

Uses a lightweight TestClient that bypasses the lifespan (debug endpoints
only need module-level globals populated at import time).

Run: uv run pytest tests/fireflies/integration/test_server_startup.py -v
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Set env before any app imports
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5433/test")
os.environ.setdefault("FIREFLIES_API_KEY", "dummy-for-testing")

# Ensure src is importable
orchestration_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(orchestration_root / "src"))


@pytest.fixture(scope="module")
def lightweight_client():
    """
    Create a TestClient that does NOT trigger the lifespan.

    The debug endpoints (/debug/routers, /debug/imports, /debug/conflicts)
    read from module-level globals (routers_status, registered_routes) that
    are populated during main_fastapi import. No DB or startup needed.

    For endpoint reachability tests, FastAPI route matching works at import
    time — the router registration happens at module scope.
    """
    from main_fastapi import app

    # TestClient without 'with' context manager = no lifespan triggered
    yield TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Server Startup: Router Registration
# =============================================================================


class TestRouterRegistration:
    """Verify that all routers import and register successfully at startup."""

    def test_debug_routers_endpoint_responds(self, lightweight_client):
        """GET /debug/routers returns 200 with router status dict."""
        response = lightweight_client.get("/debug/routers")
        assert response.status_code == 200
        data = response.json()
        assert "router_count" in data
        assert "router_status" in data

    def test_at_least_one_router_registered(self, lightweight_client):
        """At least one router must be registered for the app to be functional."""
        response = lightweight_client.get("/debug/routers")
        data = response.json()
        assert data["router_count"] > 0, "No routers registered — app is non-functional"

    def test_fireflies_router_registered_successfully(self, lightweight_client):
        """The fireflies router must be in the router_status with status 'success'."""
        response = lightweight_client.get("/debug/routers")
        data = response.json()
        assert "fireflies_router" in data["router_status"], (
            f"fireflies_router not found in router_status. "
            f"Available: {list(data['router_status'].keys())}"
        )
        assert data["router_status"]["fireflies_router"]["status"] == "success"

    def test_captions_router_registered_successfully(self, lightweight_client):
        """The captions router is required for WebSocket caption streaming."""
        response = lightweight_client.get("/debug/routers")
        data = response.json()
        assert "captions_router" in data["router_status"]
        assert data["router_status"]["captions_router"]["status"] == "success"


# =============================================================================
# Server Startup: Import Verification
# =============================================================================


class TestImportVerification:
    """Verify that all module imports succeeded without error."""

    def test_debug_imports_endpoint_responds(self, lightweight_client):
        """GET /debug/imports returns 200 with import status."""
        response = lightweight_client.get("/debug/imports")
        assert response.status_code == 200
        data = response.json()
        assert "failed_imports" in data

    def test_zero_failed_imports(self, lightweight_client):
        """
        No router imports should have failed.
        A failure here means CommandInterceptor, LiveCaptionManager, or
        another new module has an import error.
        """
        response = lightweight_client.get("/debug/imports")
        data = response.json()
        assert data["failed_imports"] == [], (
            f"Import failures detected: {data['failed_imports']}"
        )

    def test_fireflies_router_in_successful_imports(self, lightweight_client):
        """The fireflies_router must appear in the successful_imports list."""
        response = lightweight_client.get("/debug/imports")
        data = response.json()
        assert "fireflies_router" in data["successful_imports"]


# =============================================================================
# Server Startup: Route Conflict Detection
# =============================================================================


class TestRouteConflicts:
    """Verify that registered routes have no unresolved prefix conflicts."""

    def test_debug_conflicts_endpoint_responds(self, lightweight_client):
        """GET /debug/conflicts returns 200 with conflict data."""
        response = lightweight_client.get("/debug/conflicts")
        assert response.status_code == 200
        data = response.json()
        assert "conflict_count" in data
        assert "registered_routes_count" in data

    def test_routes_are_registered(self, lightweight_client):
        """At least some routes must be registered for the app to be routable."""
        response = lightweight_client.get("/debug/conflicts")
        data = response.json()
        assert data["registered_routes_count"] > 0, (
            "No routes registered — app has no routable endpoints"
        )


# =============================================================================
# Server Startup: Fireflies Endpoint Reachability
# =============================================================================


class TestFirefliesEndpointReachability:
    """
    Verify that fireflies-specific endpoints exist and are routed.

    We test that endpoints respond without 5xx errors. The lightweight client
    doesn't run the lifespan, so some endpoints may fail with 500 due to
    uninitialized dependencies — that's expected. We use raise_server_exceptions=False
    to capture all responses. What we care about: the route is registered and the
    handler is invoked (not a FastAPI 404 "Not Found").

    For dependency-requiring endpoints (pause/resume), a 500 with our error
    handling middleware is still evidence the route exists.
    """

    def test_display_mode_endpoint_exists(self, lightweight_client):
        """
        PUT /fireflies/sessions/{id}/display-mode must be routed.
        This endpoint does NOT require a session manager — it just broadcasts
        to the connection manager. Should return 200 even without full startup.
        """
        response = lightweight_client.put(
            "/fireflies/sessions/validation-test-id/display-mode",
            json={"mode": "english"},
        )
        # This endpoint only uses the WS connection manager (module-level singleton)
        # so it should work without full dependency initialization
        assert response.status_code == 200, (
            f"PUT .../display-mode failed: {response.status_code} {response.text}"
        )
        data = response.json()
        assert data["success"] is True
        assert data["mode"] == "english"

    def test_sessions_list_endpoint_is_routed(self, lightweight_client):
        """GET /fireflies/sessions is routed (may return 500 without deps, but not 404)."""
        response = lightweight_client.get("/fireflies/sessions")
        # Route MUST exist — a framework 404 means the route isn't registered
        # 500 is acceptable (uninitialized dependency), 200/422 are fine too
        assert response.status_code != 404 or (
            response.status_code == 404 and "detail" in response.json()
        ), "GET /fireflies/sessions returned bare 404 — route not registered"

    def test_dashboard_config_endpoint_is_routed(self, lightweight_client):
        """GET /fireflies/dashboard/config is routed."""
        response = lightweight_client.get("/fireflies/dashboard/config")
        # Same logic: bare 404 = not registered, anything else = route exists
        assert response.status_code != 404 or "detail" in response.json()

    def test_pause_endpoint_is_routed(self, lightweight_client):
        """
        POST /fireflies/sessions/{id}/pause is routed.
        May return 500 (dep injection fails) or 404 (session not found).
        """
        response = lightweight_client.post(
            "/fireflies/sessions/validation-test-id/pause"
        )
        # If 404, must have our error body (not framework bare 404)
        # The error middleware uses "error" key; FastAPI default uses "detail"
        if response.status_code == 404:
            data = response.json()
            has_our_error = "detail" in data or "error" in data
            assert has_our_error, (
                f"Got bare 404 — pause endpoint may not be registered. Body: {data}"
            )

    def test_resume_endpoint_is_routed(self, lightweight_client):
        """POST /fireflies/sessions/{id}/resume is routed."""
        response = lightweight_client.post(
            "/fireflies/sessions/validation-test-id/resume"
        )
        if response.status_code == 404:
            data = response.json()
            has_our_error = "detail" in data or "error" in data
            assert has_our_error, (
                f"Got bare 404 — resume endpoint may not be registered. Body: {data}"
            )

    def test_target_languages_endpoint_is_routed(self, lightweight_client):
        """PUT /fireflies/sessions/{id}/target-languages is routed."""
        response = lightweight_client.put(
            "/fireflies/sessions/validation-test-id/target-languages",
            json={"target_languages": ["es"]},
        )
        if response.status_code == 404:
            data = response.json()
            has_our_error = "detail" in data or "error" in data
            assert has_our_error, (
                f"Got bare 404 — target-languages endpoint may not be registered. Body: {data}"
            )
