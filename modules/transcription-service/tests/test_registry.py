"""Tests for ModelRegistry — YAML-based language→backend routing."""
from pathlib import Path

import pytest
import yaml

from registry import ModelRegistry
from livetranslate_common.models import BackendConfig


SAMPLE_REGISTRY = {
    "version": 1,
    "backends": {
        "whisper": {"module": "backends.whisper", "class": "WhisperBackend"},
    },
    "vram_budget_mb": 10000,
    "language_routing": {
        "en": {
            "backend": "whisper",
            "model": "large-v3-turbo",
            "compute_type": "float16",
            "chunk_duration_s": 5.0,
            "stride_s": 4.5,
            "overlap_s": 0.5,
            "vad_threshold": 0.5,
            "beam_size": 1,
            "prebuffer_s": 0.3,
            "batch_profile": "realtime",
        },
        "*": {
            "backend": "whisper",
            "model": "large-v3-turbo",
            "compute_type": "float16",
            "chunk_duration_s": 5.0,
            "stride_s": 4.5,
            "overlap_s": 0.5,
            "vad_threshold": 0.5,
            "beam_size": 1,
            "prebuffer_s": 0.3,
            "batch_profile": "realtime",
        },
    },
}


class TestModelRegistry:
    @pytest.fixture
    def registry_file(self, tmp_path):
        path = tmp_path / "model_registry.yaml"
        path.write_text(yaml.dump(SAMPLE_REGISTRY))
        return path

    def test_load_registry(self, registry_file):
        reg = ModelRegistry(registry_file)
        assert reg.version == 1
        assert reg.vram_budget_mb == 10000

    def test_get_config_for_language(self, registry_file):
        reg = ModelRegistry(registry_file)
        config = reg.get_config("en")
        assert isinstance(config, BackendConfig)
        assert config.backend == "whisper"
        assert config.model == "large-v3-turbo"

    def test_fallback_to_wildcard(self, registry_file):
        reg = ModelRegistry(registry_file)
        config = reg.get_config("fr")  # not explicitly mapped
        assert config.backend == "whisper"  # falls back to "*"

    def test_missing_language_no_wildcard_raises(self, tmp_path):
        no_wildcard = {**SAMPLE_REGISTRY, "language_routing": {"en": SAMPLE_REGISTRY["language_routing"]["en"]}}
        path = tmp_path / "reg.yaml"
        path.write_text(yaml.dump(no_wildcard))
        reg = ModelRegistry(path)
        with pytest.raises(KeyError):
            reg.get_config("fr")

    def test_reload(self, registry_file):
        reg = ModelRegistry(registry_file)
        assert reg.get_config("en").beam_size == 1

        # Modify file
        data = yaml.safe_load(registry_file.read_text())
        data["language_routing"]["en"]["beam_size"] = 5
        registry_file.write_text(yaml.dump(data))

        reg.reload()
        assert reg.get_config("en").beam_size == 5
