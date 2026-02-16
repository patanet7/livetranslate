"""Tests for performance logging utilities."""

import json
import time

import pytest


class TestLogPerformance:
    def test_logs_duration(self, capsys):
        from livetranslate_common.logging import get_logger, setup_logging
        from livetranslate_common.logging.performance import log_performance

        setup_logging(service_name="test", log_format="json")
        logger = get_logger()
        with log_performance(logger, "test_op"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["event"] == "operation_completed"
        assert data["operation"] == "test_op"
        assert data["duration_ms"] >= 10

    def test_extra_context(self, capsys):
        from livetranslate_common.logging import get_logger, setup_logging
        from livetranslate_common.logging.performance import log_performance

        setup_logging(service_name="test", log_format="json")
        logger = get_logger()
        with log_performance(logger, "transcribe", model="whisper-base", chunks=5):
            pass
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["model"] == "whisper-base"
        assert data["chunks"] == 5

    def test_logs_on_exception(self, capsys):
        from livetranslate_common.logging import get_logger, setup_logging
        from livetranslate_common.logging.performance import log_performance

        setup_logging(service_name="test", log_format="json")
        logger = get_logger()
        with pytest.raises(ValueError):
            with log_performance(logger, "failing_op"):
                raise ValueError("boom")
        captured = capsys.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line.strip()]
        last = json.loads(lines[-1])
        assert last["event"] == "operation_failed"
        assert last["operation"] == "failing_op"
        assert "duration_ms" in last
