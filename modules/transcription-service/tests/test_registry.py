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


class TestRegistryConfigValidation:
    """Task 2.3: Registry config validation at load time.

    Invalid BackendConfig values (stride/overlap mismatch, chunk > 30s)
    must raise ValidationError at load time, not at request time.
    """

    def test_invalid_stride_rejected_at_load(self, tmp_path):
        """stride_s != chunk_duration_s - overlap_s must raise ValidationError."""
        from pydantic import ValidationError

        bad_registry = {
            "version": 1,
            "backends": {"whisper": {"module": "backends.whisper", "class": "WhisperBackend"}},
            "vram_budget_mb": 10000,
            "language_routing": {
                "en": {
                    "backend": "whisper",
                    "model": "large-v3-turbo",
                    "compute_type": "float16",
                    "chunk_duration_s": 5.0,
                    "stride_s": 3.0,  # wrong: should be 4.5 (5.0 - 0.5)
                    "overlap_s": 0.5,
                    "vad_threshold": 0.5,
                    "beam_size": 1,
                    "prebuffer_s": 0.3,
                    "batch_profile": "realtime",
                },
            },
        }
        path = tmp_path / "bad_stride.yaml"
        path.write_text(yaml.dump(bad_registry))

        with pytest.raises(ValidationError):
            ModelRegistry(path)

    def test_chunk_duration_exceeds_30s_rejected(self, tmp_path):
        """chunk_duration_s > 30.0 (Whisper positional encoding limit) must raise."""
        from pydantic import ValidationError

        bad_registry = {
            "version": 1,
            "backends": {"whisper": {"module": "backends.whisper", "class": "WhisperBackend"}},
            "vram_budget_mb": 10000,
            "language_routing": {
                "en": {
                    "backend": "whisper",
                    "model": "large-v3-turbo",
                    "compute_type": "float16",
                    "chunk_duration_s": 35.0,
                    "stride_s": 34.5,
                    "overlap_s": 0.5,
                    "vad_threshold": 0.5,
                    "beam_size": 1,
                    "prebuffer_s": 0.3,
                    "batch_profile": "realtime",
                },
            },
        }
        path = tmp_path / "bad_chunk.yaml"
        path.write_text(yaml.dump(bad_registry))

        with pytest.raises(ValidationError):
            ModelRegistry(path)

    def test_vad_bounded_mode_valid(self, tmp_path):
        """VAD-bounded mode (overlap_s=0, stride_s=chunk_duration_s) is valid."""
        vad_registry = {
            "version": 1,
            "backends": {"whisper": {"module": "backends.whisper", "class": "WhisperBackend"}},
            "vram_budget_mb": 10000,
            "language_routing": {
                "en": {
                    "backend": "whisper",
                    "model": "large-v3-turbo",
                    "compute_type": "float16",
                    "chunk_duration_s": 5.0,
                    "stride_s": 5.0,  # stride == chunk_duration (no overlap)
                    "overlap_s": 0.0,
                    "vad_threshold": 0.5,
                    "beam_size": 1,
                    "prebuffer_s": 0.3,
                    "batch_profile": "realtime",
                },
            },
        }
        path = tmp_path / "vad_bounded.yaml"
        path.write_text(yaml.dump(vad_registry))

        reg = ModelRegistry(path)
        config = reg.get_config("en")
        assert config.overlap_s == 0.0
        assert config.stride_s == config.chunk_duration_s


class TestRegistryHotReloadFailureModes:
    """Phase 3 failure mode tests: registry reload with invalid YAML.

    When reload() is called with a corrupt or invalid YAML file,
    the registry must keep the LAST KNOWN GOOD configuration.
    """

    @pytest.fixture
    def registry_file(self, tmp_path):
        path = tmp_path / "model_registry.yaml"
        path.write_text(yaml.dump(SAMPLE_REGISTRY))
        return path

    def test_reload_with_invalid_yaml_keeps_old_config(self, registry_file):
        """Corrupt YAML on disk does not overwrite good in-memory config."""
        reg = ModelRegistry(registry_file)
        assert reg.get_config("en").model == "large-v3-turbo"

        # Corrupt the file
        registry_file.write_text("{{{invalid yaml: [unbalanced")

        result = reg.reload()
        assert result is False, "reload() should return False on parse error"

        # Old config should be intact
        assert reg.version == 1
        assert reg.get_config("en").model == "large-v3-turbo"
        assert reg.vram_budget_mb == 10000

    def test_reload_with_missing_required_fields_keeps_old_config(self, registry_file):
        """YAML that parses but lacks required BackendConfig fields is rejected."""
        reg = ModelRegistry(registry_file)
        original_version = reg.version

        # Write YAML missing the 'backend' field in language_routing
        bad_data = {
            "version": 99,
            "language_routing": {
                "en": {
                    "model": "large-v3-turbo",
                    # 'backend' field is missing -- BackendConfig validation fails
                },
            },
        }
        registry_file.write_text(yaml.dump(bad_data))

        result = reg.reload()
        assert result is False

        # Old version should be preserved
        assert reg.version == original_version

    def test_reload_with_empty_file_keeps_old_config(self, registry_file):
        """Empty YAML file does not wipe the registry."""
        reg = ModelRegistry(registry_file)
        assert reg.get_config("en").backend == "whisper"

        registry_file.write_text("")

        result = reg.reload()
        assert result is False

        assert reg.get_config("en").backend == "whisper"

    def test_reload_with_deleted_file_keeps_old_config(self, registry_file):
        """If the file is deleted between reloads, old config survives."""
        reg = ModelRegistry(registry_file)
        assert reg.get_config("en").backend == "whisper"

        registry_file.unlink()

        result = reg.reload()
        assert result is False

        assert reg.get_config("en").backend == "whisper"

    def test_successful_reload_after_previous_failure(self, registry_file):
        """After a failed reload, a subsequent valid reload succeeds."""
        reg = ModelRegistry(registry_file)

        # Corrupt
        registry_file.write_text("{{{bad yaml")
        assert reg.reload() is False

        # Fix -- change beam_size to verify reload actually took effect
        data = SAMPLE_REGISTRY.copy()
        data = {**SAMPLE_REGISTRY}
        fixed_routing = {}
        for lang, entry in SAMPLE_REGISTRY["language_routing"].items():
            fixed_routing[lang] = {**entry, "beam_size": 7}
        data["language_routing"] = fixed_routing
        data["version"] = 2
        registry_file.write_text(yaml.dump(data))

        assert reg.reload() is True
        assert reg.version == 2
        assert reg.get_config("en").beam_size == 7
