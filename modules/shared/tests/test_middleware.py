"""Tests for request ID and logging middleware."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestRequestIDMiddleware:
    def _make_app(self):
        from livetranslate_common.logging import setup_logging
        from livetranslate_common.middleware import RequestIDMiddleware

        setup_logging(service_name="test", log_format="json")
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        return app

    def test_generates_request_id(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_propagates_incoming_request_id(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/ping", headers={"X-Request-ID": "custom-123"})
        assert resp.headers["x-request-id"] == "custom-123"

    def test_generated_id_is_valid_uuid(self):
        import uuid

        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/ping")
        request_id = resp.headers["x-request-id"]
        # Should be a valid UUID
        parsed = uuid.UUID(request_id)
        assert str(parsed) == request_id


class TestRequestLoggingMiddleware:
    def _make_app(self):
        from livetranslate_common.logging import setup_logging
        from livetranslate_common.middleware import RequestIDMiddleware, RequestLoggingMiddleware

        setup_logging(service_name="test", log_format="json")
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
        app.add_middleware(RequestIDMiddleware)

        @app.get("/hello")
        async def hello():
            return {"msg": "hi"}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/error")
        async def error():
            raise RuntimeError("unexpected")

        return app

    def test_logs_request_and_response(self, capsys):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/hello")
        assert resp.status_code == 200
        captured = capsys.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines if line.startswith("{")]
        event_names = [e["event"] for e in events]
        assert "request_started" in event_names
        assert "request_completed" in event_names

    def test_skips_health_at_info(self, capsys):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        captured = capsys.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines if line.startswith("{")]
        request_events = [e for e in events if e.get("path") == "/health"]
        assert len(request_events) == 0

    def test_completed_log_includes_duration(self, capsys):
        app = self._make_app()
        client = TestClient(app)
        client.get("/hello")
        captured = capsys.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines if line.startswith("{")]
        completed = [e for e in events if e.get("event") == "request_completed"]
        assert len(completed) == 1
        assert "duration_ms" in completed[0]
        assert completed[0]["duration_ms"] >= 0

    def test_error_request_logs_failure(self, capsys):
        app = self._make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/error")
        assert resp.status_code == 500
        captured = capsys.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines if line.startswith("{")]
        failed_events = [e for e in events if e.get("event") == "request_failed"]
        # The middleware catches exceptions and logs request_failed
        assert len(failed_events) >= 1
