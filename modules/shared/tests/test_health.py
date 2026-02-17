"""Tests for health check endpoint factory."""

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestHealthRouter:
    def test_healthy_response(self):
        from livetranslate_common.health import create_health_router

        app = FastAPI()
        router = create_health_router(
            service_name="orchestration",
            version="1.0.0",
            checks={"db": lambda: True, "redis": lambda: True},
        )
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "orchestration"
        assert body["version"] == "1.0.0"
        assert body["checks"]["db"] == "ok"
        assert body["checks"]["redis"] == "ok"

    def test_unhealthy_response(self):
        from livetranslate_common.health import create_health_router

        app = FastAPI()
        router = create_health_router(
            service_name="orchestration",
            version="1.0.0",
            checks={"db": lambda: True, "redis": lambda: False},
        )
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["checks"]["db"] == "ok"
        assert body["checks"]["redis"] == "failing"

    def test_check_exception_is_unhealthy(self):
        from livetranslate_common.health import create_health_router

        def boom():
            raise ConnectionError("refused")

        app = FastAPI()
        router = create_health_router(
            service_name="test",
            version="0.1.0",
            checks={"broken": boom},
        )
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["checks"]["broken"] == "failing"

    def test_empty_checks_is_healthy(self):
        from livetranslate_common.health import create_health_router

        app = FastAPI()
        router = create_health_router(
            service_name="minimal",
            version="0.0.1",
            checks={},
        )
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["checks"] == {}

    def test_mixed_checks(self):
        from livetranslate_common.health import create_health_router

        app = FastAPI()
        router = create_health_router(
            service_name="mixed",
            version="2.0.0",
            checks={
                "db": lambda: True,
                "redis": lambda: False,
                "cache": lambda: True,
            },
        )
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["checks"]["db"] == "ok"
        assert body["checks"]["redis"] == "failing"
        assert body["checks"]["cache"] == "ok"
