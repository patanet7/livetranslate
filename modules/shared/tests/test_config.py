"""Tests for livetranslate_common.config."""


class TestServiceSettings:
    def test_defaults(self):
        from livetranslate_common.config import ServiceSettings

        settings = ServiceSettings(service_name="test")
        assert settings.service_name == "test"
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_from_env(self, monkeypatch):
        from livetranslate_common.config import ServiceSettings

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FORMAT", "dev")
        settings = ServiceSettings(service_name="test")
        assert settings.log_level == "DEBUG"
        assert settings.log_format == "dev"

    def test_log_level_case_insensitive(self):
        from livetranslate_common.config import ServiceSettings

        settings = ServiceSettings(service_name="test", log_level="debug")
        assert settings.log_level == "DEBUG"
