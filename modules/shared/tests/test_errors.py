"""Tests for error hierarchy and FastAPI exception handlers."""

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestExceptionHierarchy:
    def test_base_error(self):
        from livetranslate_common.errors import LiveTranslateError

        err = LiveTranslateError("something broke", error_code="GENERIC_001", detail="extra")
        assert str(err) == "something broke"
        assert err.error_code == "GENERIC_001"
        assert err.context == {"detail": "extra"}

    def test_base_error_defaults(self):
        from livetranslate_common.errors import LiveTranslateError

        err = LiveTranslateError("oops")
        assert err.error_code == "INTERNAL_ERROR"
        assert err.context == {}
        assert err.status_code == 500

    def test_service_unavailable(self):
        from livetranslate_common.errors import ServiceUnavailableError

        err = ServiceUnavailableError("whisper down", service="whisper")
        assert err.error_code == "SERVICE_UNAVAILABLE"
        assert err.status_code == 503
        assert err.context["service"] == "whisper"

    def test_validation_error(self):
        from livetranslate_common.errors import ValidationError

        err = ValidationError("bad input", field="audio_format")
        assert err.error_code == "VALIDATION_ERROR"
        assert err.status_code == 422
        assert err.context["field"] == "audio_format"

    def test_audio_processing_error(self):
        from livetranslate_common.errors import AudioProcessingError

        err = AudioProcessingError("corrupt audio")
        assert err.error_code == "AUDIO_PROCESSING_ERROR"
        assert err.status_code == 500
        assert err.context == {}

    def test_all_errors_inherit_from_base(self):
        from livetranslate_common.errors import (
            AudioProcessingError,
            LiveTranslateError,
            ServiceUnavailableError,
            ValidationError,
        )

        assert issubclass(ServiceUnavailableError, LiveTranslateError)
        assert issubclass(ValidationError, LiveTranslateError)
        assert issubclass(AudioProcessingError, LiveTranslateError)

    def test_all_errors_are_exceptions(self):
        from livetranslate_common.errors import (
            AudioProcessingError,
            LiveTranslateError,
            ServiceUnavailableError,
            ValidationError,
        )

        for cls in (
            LiveTranslateError,
            ServiceUnavailableError,
            ValidationError,
            AudioProcessingError,
        ):
            assert issubclass(cls, Exception)


class TestExceptionHandler:
    def test_handler_returns_json(self):
        from livetranslate_common.errors import LiveTranslateError
        from livetranslate_common.errors.handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/fail")
        async def fail():
            raise LiveTranslateError("boom", error_code="TEST_001")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/fail")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_code"] == "TEST_001"
        assert body["message"] == "boom"

    def test_validation_error_returns_422(self):
        from livetranslate_common.errors import ValidationError
        from livetranslate_common.errors.handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/validate")
        async def validate():
            raise ValidationError("missing field", field="name")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/validate")
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["field"] == "name"

    def test_service_unavailable_returns_503(self):
        from livetranslate_common.errors import ServiceUnavailableError
        from livetranslate_common.errors.handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/unavailable")
        async def unavailable():
            raise ServiceUnavailableError("whisper is down", service="whisper")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/unavailable")
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_code"] == "SERVICE_UNAVAILABLE"
        assert body["service"] == "whisper"

    def test_audio_processing_error_returns_500(self):
        from livetranslate_common.errors import AudioProcessingError
        from livetranslate_common.errors.handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/audio-fail")
        async def audio_fail():
            raise AudioProcessingError("corrupt file", format="wav")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/audio-fail")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_code"] == "AUDIO_PROCESSING_ERROR"
        assert body["format"] == "wav"

    def test_context_fields_included_in_response(self):
        from livetranslate_common.errors import LiveTranslateError
        from livetranslate_common.errors.handlers import register_error_handlers

        app = FastAPI()
        register_error_handlers(app)

        @app.get("/ctx")
        async def ctx():
            raise LiveTranslateError("fail", error_code="CTX_TEST", foo="bar", count=42)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/ctx")
        body = resp.json()
        assert body["foo"] == "bar"
        assert body["count"] == 42
