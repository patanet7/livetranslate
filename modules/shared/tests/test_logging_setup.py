"""Tests for livetranslate_common.logging setup and processors."""

import json
import logging

import pytest


class TestSetupLogging:
    def test_setup_returns_none(self):
        from livetranslate_common.logging import setup_logging

        result = setup_logging(service_name="test-service")
        assert result is None

    def test_get_logger_returns_bound_logger(self):
        from livetranslate_common.logging import get_logger, setup_logging

        setup_logging(service_name="test-service")
        logger = get_logger()
        assert logger is not None

    def test_json_output(self, capsys):
        from livetranslate_common.logging import get_logger, setup_logging

        setup_logging(service_name="test-service", log_format="json")
        logger = get_logger()
        logger.info("test_event", key="value")
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["event"] == "test_event"
        assert data["key"] == "value"
        assert data["service"] == "test-service"
        assert "timestamp" in data
        assert "level" in data

    def test_dev_output_no_json(self, capsys):
        from livetranslate_common.logging import get_logger, setup_logging

        setup_logging(service_name="test-service", log_format="dev")
        logger = get_logger()
        logger.info("hello_dev", user="alice")
        captured = capsys.readouterr()
        output = captured.err.strip()
        with pytest.raises(json.JSONDecodeError):
            json.loads(output.split("\n")[-1])
        assert "hello_dev" in output

    def test_stdlib_logs_formatted(self, capsys):
        from livetranslate_common.logging import setup_logging

        setup_logging(service_name="test-service", log_format="json")
        stdlib_logger = logging.getLogger("some.third.party")
        stdlib_logger.warning("stdlib message")
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["event"] == "stdlib message"
        assert data["level"] == "warning"


class TestProcessors:
    def test_censor_sensitive_keys(self):
        from livetranslate_common.logging.processors import censor_sensitive_data

        event_dict = {
            "event": "login",
            "password": "secret123",  # pragma: allowlist secret
            "api_key": "sk-abc",  # pragma: allowlist secret
            "username": "alice",
        }
        result = censor_sensitive_data(None, None, event_dict)
        assert result["password"] == "***REDACTED***"
        assert result["api_key"] == "***REDACTED***"
        assert result["username"] == "alice"
