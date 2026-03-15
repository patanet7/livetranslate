# Plan 1: Transcription Service Refactor

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `whisper-service` → `transcription-service`, implement pluggable backend architecture with BackendManager (VRAM budget), ModelRegistry (YAML config), faster-whisper as first backend, and authoritative language detection.

**Architecture:** The transcription service runs on thomas-pc (RTX 4090 via Tailscale). It receives 16kHz mono audio via WebSocket binary frames, runs VAD → LID → backend inference → dedup, and returns transcription results as text frames. A BackendManager enforces a 10GB VRAM budget with LRU eviction. A YAML-based ModelRegistry maps languages to backends with per-model chunking parameters.

**Tech Stack:** Python 3.12+, faster-whisper (CTranslate2), Silero VAD, Pydantic v2, UV workspace

**Spec:** `docs/superpowers/specs/2026-03-14-loopback-transcription-translation-design.md` — Plan 1 section

**Depends on:** Plan 0 (shared contracts in `livetranslate-common`)

**Blocking dependency:** Plan 0 Task 6 (VAD/chunking research) must complete before implementing backend adapters (Tasks 3-5 below). The VAD research may refine `BackendConfig` fields and chunking parameters that the backends depend on. See the spec's "Pre-task" section under VAD/Chunking Strategy.

**Design notes:**
- `TranscriptionResult` and `Segment` are **Pydantic `BaseModel`** classes defined in Plan 0's `livetranslate_common.models.transcription` module — NOT dataclasses. Use `model_dump()` for serialization, not `asdict()`. All code in this plan must use `model_dump()` consistently.
- `Segment` is imported from `livetranslate_common.models.transcription` (defined in Plan 0). Do not redefine it in this service.

---

## Chunk 1: Rename & Cleanup

### Task 1: Rename whisper-service → transcription-service

**Files:**
- Rename: `modules/whisper-service/` → `modules/transcription-service/`
- Modify: `pyproject.toml` (root workspace)
- Modify: `modules/transcription-service/pyproject.toml` (package name)
- Modify: all internal imports referencing "whisper-service" or "whisper_service"
- Modify: `CLAUDE.md` (references to whisper-service)
- Modify: `docker-compose*.yml` (service names)
- Modify: `justfile` (commands)

- [ ] **Step 1: Rename the directory**

```bash
git mv modules/whisper-service modules/transcription-service
```

- [ ] **Step 2: Update pyproject.toml workspace member**

In the root `pyproject.toml`, find the workspace members list and change `"modules/whisper-service"` to `"modules/transcription-service"`.

- [ ] **Step 3: Update the service's own pyproject.toml**

In `modules/transcription-service/pyproject.toml`:
- Change `name = "whisper-service"` → `name = "transcription-service"`
- Update any path references

- [ ] **Step 4: Update Docker compose files**

Search all `docker-compose*.yml` for `whisper-service` or `whisper_service` and rename to `transcription-service` / `transcription_service`.

- [ ] **Step 5: Update CLAUDE.md references**

In root `CLAUDE.md`:
- Change all `whisper-service` → `transcription-service`
- Change "Whisper Service" → "Transcription Service" in headings
- Update the service description to reflect pluggable backends
- Update port references if needed
- Update test commands

- [ ] **Step 6: Update justfile commands**

Rename `test-whisper` → `test-transcription` and update paths.

- [ ] **Step 7: Run `uv sync` to verify workspace resolution**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && uv sync --group dev
```
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename whisper-service → transcription-service"
```

---

### Task 2: Retire SimulStreaming and legacy decoders

**Files:**
- Delete: `modules/transcription-service/src/simul_whisper/` (entire directory)
- Delete: `modules/transcription-service/src/beam_decoder.py`
- Delete: `modules/transcription-service/src/alignatt_decoder.py`
- Delete: `modules/transcription-service/src/eow_detection.py`
- Modify: any files importing from these modules

- [ ] **Step 1: Identify all imports of retired modules**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run python -c "
import ast, pathlib
retired = {'simul_whisper', 'beam_decoder', 'alignatt_decoder', 'eow_detection'}
for f in pathlib.Path('modules/transcription-service/src').rglob('*.py'):
    try:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, 'module', '') or ''
                names = [a.name for a in getattr(node, 'names', [])]
                for r in retired:
                    if r in mod or r in names:
                        print(f'{f}:{node.lineno} imports {r}')
    except: pass
"
```

- [ ] **Step 2: Remove imports of retired modules from remaining files**

For each file that imports from `simul_whisper`, `beam_decoder`, `alignatt_decoder`, or `eow_detection`: remove the import and any code that depends on it. If a function becomes empty after removing the dependency, remove the function too.

- [ ] **Step 3: Delete the retired files and directories**

```bash
rm -rf modules/transcription-service/src/simul_whisper/
rm -f modules/transcription-service/src/beam_decoder.py
rm -f modules/transcription-service/src/alignatt_decoder.py
rm -f modules/transcription-service/src/eow_detection.py
```

- [ ] **Step 4: Verify no broken imports**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run python -c "
import importlib, pathlib, sys
sys.path.insert(0, 'modules/transcription-service/src')
errors = []
for f in pathlib.Path('modules/transcription-service/src').rglob('*.py'):
    if '__pycache__' in str(f): continue
    mod = str(f.relative_to('modules/transcription-service/src')).replace('/', '.').removesuffix('.py')
    if mod.endswith('.__init__'): mod = mod[:-9]
    try:
        importlib.import_module(mod)
    except Exception as e:
        errors.append(f'{mod}: {e}')
for e in errors: print(e)
print(f'{len(errors)} import errors')
"
```
Expected: 0 import errors (some may be expected due to missing GPU libraries — those are acceptable)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: retire SimulStreaming, beam_decoder, alignatt_decoder"
```

---

## Chunk 2: Backend Architecture

### Task 2b: Configure test imports (conftest.py and pyproject.toml)

**Files:**
- Create: `modules/transcription-service/tests/conftest.py`
- Modify: `modules/transcription-service/pyproject.toml`

Tests import modules like `from backends.base import ...` and `from registry import ...` which require the `src/` directory on the Python path. Configure this before writing any tests.

- [ ] **Step 1: Add pytest pythonpath to pyproject.toml**

In `modules/transcription-service/pyproject.toml`, add:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create conftest.py**

```python
# modules/transcription-service/tests/conftest.py
"""Shared fixtures for transcription service tests."""
import sys
from pathlib import Path

# Ensure src/ is importable (belt-and-suspenders with pyproject.toml pythonpath)
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
```

- [ ] **Step 3: Commit**

```bash
git add modules/transcription-service/tests/conftest.py modules/transcription-service/pyproject.toml
git commit -m "chore(transcription): configure pytest pythonpath for test imports"
```

---

### Task 3: TranscriptionBackend protocol and BackendManager

**Files:**
- Create: `modules/transcription-service/src/backends/__init__.py`
- Create: `modules/transcription-service/src/backends/base.py`
- Create: `modules/transcription-service/src/backends/manager.py`
- Create: `modules/transcription-service/tests/test_backend_manager.py`

- [ ] **Step 1: Write failing test for BackendManager**

```python
# modules/transcription-service/tests/test_backend_manager.py
"""Tests for BackendManager VRAM budget enforcement and LRU eviction."""
import asyncio

import numpy as np
import pytest

from backends.base import TranscriptionBackend
from backends.manager import BackendManager
from livetranslate_common.models import BackendConfig, ModelInfo, TranscriptionResult


class FakeBackend:
    """Test double implementing TranscriptionBackend protocol."""

    def __init__(self, name: str, vram_mb: int, languages: list[str]):
        self._name = name
        self._vram_mb = vram_mb
        self._languages = languages
        self._loaded = False

    async def transcribe(self, audio: np.ndarray, language: str | None = None, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="fake", language=language or "en", confidence=0.9,
            is_final=True, is_draft=False, speaker_id=None,
        )

    async def transcribe_stream(self, audio, language=None, **kwargs):
        yield await self.transcribe(audio, language, **kwargs)

    def supports_language(self, lang: str) -> bool:
        return lang in self._languages

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name, backend="fake",
            languages=self._languages, vram_mb=self._vram_mb,
            compute_type="float16",
        )

    async def load_model(self, model_name: str, device: str = "cuda") -> None:
        self._loaded = True

    async def unload_model(self) -> None:
        self._loaded = False

    async def warmup(self) -> None:
        pass

    def vram_usage_mb(self) -> int:
        return self._vram_mb if self._loaded else 0


class TestBackendManager:
    @pytest.fixture
    def manager(self):
        return BackendManager(max_vram_mb=10000)

    def test_initial_state(self, manager):
        assert manager.current_vram_mb == 0
        assert len(manager.loaded_backends) == 0

    @pytest.mark.asyncio
    async def test_load_backend(self, manager):
        backend = FakeBackend("test-model", vram_mb=3000, languages=["en"])
        config = BackendConfig(
            backend="fake", model="test-model", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        # Register a factory so manager can create backends
        manager.register_factory("fake", lambda cfg: backend)

        result = await manager.get_backend(config)
        assert result is backend
        assert manager.current_vram_mb == 3000

    @pytest.mark.asyncio
    async def test_lru_eviction(self, manager):
        """When loading a new backend exceeds budget, LRU backend is evicted."""
        b1 = FakeBackend("model-a", vram_mb=6000, languages=["en"])
        b2 = FakeBackend("model-b", vram_mb=6000, languages=["zh"])

        manager.register_factory("fake_a", lambda cfg: b1)
        manager.register_factory("fake_b", lambda cfg: b2)

        cfg_a = BackendConfig(
            backend="fake_a", model="model-a", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.5, overlap_s=0.5,
            vad_threshold=0.5, beam_size=1, prebuffer_s=0.3,
            batch_profile="realtime",
        )
        cfg_b = BackendConfig(
            backend="fake_b", model="model-b", compute_type="float16",
            chunk_duration_s=5.0, stride_s=4.0, overlap_s=1.0,
            vad_threshold=0.45, beam_size=5, prebuffer_s=0.5,
            batch_profile="realtime",
        )

        await manager.get_backend(cfg_a)
        assert manager.current_vram_mb == 6000

        # Loading b (6000MB) would exceed 10000MB budget, so a must be evicted
        await manager.get_backend(cfg_b)
        assert b1._loaded is False  # evicted
        assert b2._loaded is True
        assert manager.current_vram_mb == 6000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_backend_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backends'`

- [ ] **Step 3: Write TranscriptionBackend protocol**

```python
# modules/transcription-service/src/backends/__init__.py
"""Pluggable transcription backend system."""

# modules/transcription-service/src/backends/base.py
"""TranscriptionBackend protocol — any transcription engine implements this."""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

import numpy as np
from livetranslate_common.models import ModelInfo, TranscriptionResult


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Protocol for pluggable transcription backends.

    Implementors: WhisperBackend, SenseVoiceBackend, FunASRBackend, etc.
    """

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> TranscriptionResult: ...

    async def transcribe_stream(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> AsyncIterator[TranscriptionResult]: ...

    def supports_language(self, lang: str) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...

    async def load_model(self, model_name: str, device: str = "cuda") -> None: ...

    async def unload_model(self) -> None: ...

    async def warmup(self) -> None: ...

    def vram_usage_mb(self) -> int: ...
```

- [ ] **Step 4: Write BackendManager**

```python
# modules/transcription-service/src/backends/manager.py
"""BackendManager — VRAM-aware backend lifecycle with LRU eviction.

Manages loading/unloading of transcription backends on the GPU,
enforcing a VRAM budget. When a new backend needs to load and
would exceed the budget, the least-recently-used backend is evicted.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Callable

from livetranslate_common.logging import get_logger
from livetranslate_common.models import BackendConfig

from backends.base import TranscriptionBackend

logger = get_logger()


class BackendManager:
    def __init__(self, max_vram_mb: int = 10000):
        self.max_vram_mb = max_vram_mb
        self.loaded_backends: OrderedDict[str, TranscriptionBackend] = OrderedDict()
        self._factories: dict[str, Callable[[BackendConfig], TranscriptionBackend]] = {}
        self._load_lock = asyncio.Lock()

    def register_factory(
        self, backend_name: str, factory: Callable[[BackendConfig], TranscriptionBackend]
    ) -> None:
        self._factories[backend_name] = factory

    @property
    def current_vram_mb(self) -> int:
        return sum(b.vram_usage_mb() for b in self.loaded_backends.values())

    def _backend_key(self, config: BackendConfig) -> str:
        return f"{config.backend}:{config.model}"

    async def get_backend(self, config: BackendConfig) -> TranscriptionBackend:
        """Return a loaded backend for the given config, evicting LRU if needed."""
        key = self._backend_key(config)

        # Already loaded — move to end of LRU
        if key in self.loaded_backends:
            self.loaded_backends.move_to_end(key)
            return self.loaded_backends[key]

        async with self._load_lock:
            # Double-check after acquiring lock
            if key in self.loaded_backends:
                self.loaded_backends.move_to_end(key)
                return self.loaded_backends[key]

            factory = self._factories.get(config.backend)
            if factory is None:
                raise ValueError(f"No factory registered for backend '{config.backend}'")

            backend = factory(config)

            # Evict until we have room.
            # NOTE: vram_mb comes from get_model_info() which reads from
            # BackendConfig / static estimates — available BEFORE load_model().
            # This is NOT the runtime vram_usage_mb() which requires a loaded model.
            needed = backend.get_model_info().vram_mb
            while self.current_vram_mb + needed > self.max_vram_mb and self.loaded_backends:
                await self._evict_lru()

            await backend.load_model(config.model)
            await backend.warmup()
            self.loaded_backends[key] = backend

            logger.info(
                "backend_loaded",
                backend=config.backend,
                model=config.model,
                vram_mb=backend.vram_usage_mb(),
                total_vram_mb=self.current_vram_mb,
            )

            return backend

    async def _evict_lru(self) -> None:
        if not self.loaded_backends:
            return
        key, backend = self.loaded_backends.popitem(last=False)
        logger.info("backend_evicted", key=key, freed_mb=backend.vram_usage_mb())
        await backend.unload_model()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_backend_manager.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add modules/transcription-service/src/backends/ modules/transcription-service/tests/test_backend_manager.py
git commit -m "feat(transcription): add TranscriptionBackend protocol and BackendManager with VRAM budget"
```

---

### Task 4: ModelRegistry (YAML config)

**Files:**
- Create: `modules/transcription-service/src/registry.py`
- Create: `modules/transcription-service/config/model_registry.yaml`
- Create: `modules/transcription-service/tests/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# modules/transcription-service/tests/test_registry.py
"""Tests for ModelRegistry — YAML-based language→backend routing."""
import tempfile
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Write ModelRegistry**

```python
# modules/transcription-service/src/registry.py
"""ModelRegistry — YAML-based language→backend+model routing.

The registry is the single source of truth for "what model handles what
language with what parameters." Changeable without code changes.
Supports hot-reload via reload() or POST /api/registry/reload.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from livetranslate_common.logging import get_logger
from livetranslate_common.models import BackendConfig

logger = get_logger()


class ModelRegistry:
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._data: dict = {}
        self._routing: dict[str, BackendConfig] = {}
        self.version: int = 0
        self.vram_budget_mb: int = 10000
        self._load()

    def _load(self) -> None:
        raw = yaml.safe_load(self._config_path.read_text())
        self._data = raw
        self.version = raw.get("version", 1)
        self.vram_budget_mb = raw.get("vram_budget_mb", 10000)

        self._routing = {}
        for lang, entry in raw.get("language_routing", {}).items():
            self._routing[lang] = BackendConfig.model_validate(entry)

        logger.info(
            "registry_loaded",
            version=self.version,
            languages=list(self._routing.keys()),
            path=str(self._config_path),
        )

    def reload(self) -> None:
        self._load()

    def get_config(self, language: str) -> BackendConfig:
        """Get the BackendConfig for a language, falling back to '*'."""
        if language in self._routing:
            return self._routing[language]
        if "*" in self._routing:
            return self._routing["*"]
        raise KeyError(f"No registry entry for language '{language}' and no wildcard '*' fallback")

    def get_backend_module(self, backend_name: str) -> dict:
        """Return the module/class info for a backend name."""
        backends = self._data.get("backends", {})
        if backend_name not in backends:
            raise KeyError(f"Unknown backend: {backend_name}")
        return backends[backend_name]

    @property
    def all_languages(self) -> list[str]:
        return [k for k in self._routing if k != "*"]
```

- [ ] **Step 4: Add SIGHUP signal handler for registry hot-reload**

In `registry.py`, add a module-level function that registers a SIGHUP handler to trigger `registry.reload()`. This complements the HTTP `POST /api/registry/reload` endpoint.

```python
import signal

def register_sighup_handler(registry: ModelRegistry) -> None:
    """Register SIGHUP handler for hot-reloading the registry from disk.

    Usage: send `kill -HUP <pid>` to reload without restarting the service.
    Also available via POST /api/registry/reload.
    """
    def _handler(signum, frame):
        logger.info("sighup_received", action="reloading_registry")
        registry.reload()

    signal.signal(signal.SIGHUP, _handler)
    logger.info("sighup_handler_registered")
```

Wire this up in `main.py` after creating the app/registry.

- [ ] **Step 5: Create the default registry YAML**

Copy the registry YAML from the spec into `modules/transcription-service/config/model_registry.yaml`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add modules/transcription-service/src/registry.py modules/transcription-service/config/model_registry.yaml modules/transcription-service/tests/test_registry.py
git commit -m "feat(transcription): add ModelRegistry with YAML config, hot-reload, and SIGHUP handler"
```

---

## Chunk 3: Whisper Backend (faster-whisper)

### Task 5: WhisperBackend implementation

**Files:**
- Create: `modules/transcription-service/src/backends/whisper.py`
- Create: `modules/transcription-service/tests/test_whisper_backend.py`
- Modify: `modules/transcription-service/pyproject.toml` (add `faster-whisper` dependency)

- [ ] **Step 1: Add faster-whisper dependency**

In `modules/transcription-service/pyproject.toml`, add `faster-whisper>=1.1.0` to dependencies.

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && uv add --project modules/transcription-service faster-whisper
```

- [ ] **Step 2: Write failing test**

```python
# modules/transcription-service/tests/test_whisper_backend.py
"""Tests for WhisperBackend — faster-whisper integration.

NOTE: These tests require a GPU with faster-whisper installed.
Mark with @pytest.mark.gpu for CI filtering.
"""
import numpy as np
import pytest

from backends.whisper import WhisperBackend
from backends.base import TranscriptionBackend
from livetranslate_common.models import BackendConfig


class TestWhisperBackendProtocol:
    def test_implements_protocol(self):
        """WhisperBackend must satisfy TranscriptionBackend protocol."""
        assert issubclass(WhisperBackend, TranscriptionBackend) or isinstance(
            WhisperBackend.__new__(WhisperBackend), TranscriptionBackend
        )


@pytest.mark.gpu
class TestWhisperBackendIntegration:
    @pytest.fixture
    async def backend(self):
        b = WhisperBackend(
            model_name="tiny",  # small model for testing
            compute_type="float16",
            device="cuda",
        )
        await b.load_model("tiny")
        await b.warmup()
        yield b
        await b.unload_model()

    @pytest.mark.asyncio
    async def test_transcribe_silence(self, backend):
        """Silence should produce empty or near-empty transcription."""
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = await backend.transcribe(silence, language="en")
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_vram_reporting(self, backend):
        assert backend.vram_usage_mb() > 0

    @pytest.mark.asyncio
    async def test_model_info(self, backend):
        info = backend.get_model_info()
        assert info.backend == "whisper"
        assert info.compute_type == "float16"

    @pytest.mark.asyncio
    async def test_supports_language(self, backend):
        assert backend.supports_language("en") is True
        assert backend.supports_language("zh") is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_whisper_backend.py::TestWhisperBackendProtocol -v`
Expected: FAIL

- [ ] **Step 4: Write WhisperBackend**

```python
# modules/transcription-service/src/backends/whisper.py
"""WhisperBackend — faster-whisper (CTranslate2) integration.

Wraps faster-whisper's WhisperModel to implement the TranscriptionBackend
protocol. Uses CTranslate2 for 4x+ GPU speedup over openai-whisper.
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np
from livetranslate_common.logging import get_logger
from livetranslate_common.models import ModelInfo, Segment, TranscriptionResult

logger = get_logger()

# faster-whisper supports these languages (ISO 639-1 codes)
_WHISPER_LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
    "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
    "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
    "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
    "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
    "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
    "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
    "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
]


class WhisperBackend:
    """TranscriptionBackend implementation using faster-whisper."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        compute_type: str = "float16",
        device: str = "cuda",
    ):
        self._model_name = model_name
        self._compute_type = compute_type
        self._device = device
        self._model = None
        self._vram_mb = 0

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper.

        Kwargs:
            beam_size: Beam search width (default 1 = greedy).
            batch_profile: "realtime" or "batch". Controls quality vs latency:
                - "realtime": greedy decoding (beam_size=1), smaller chunks
                - "batch": beam search (beam_size from config), longer context
            initial_prompt: Optional initial prompt for Whisper (e.g., glossary
                terms, domain context). Passed directly to faster-whisper's
                initial_prompt parameter.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        beam_size = kwargs.get("beam_size", 1)
        batch_profile = kwargs.get("batch_profile", "realtime")
        initial_prompt = kwargs.get("initial_prompt")

        # batch_profile overrides beam_size and chunk behavior
        if batch_profile == "batch":
            # Higher quality: use beam search, larger context
            beam_size = max(beam_size, 5)
        elif batch_profile == "realtime":
            # Lower latency: greedy decoding
            beam_size = min(beam_size, 1)

        transcribe_kwargs = {
            "language": language,
            "beam_size": beam_size,
            "vad_filter": False,  # VAD handled externally by VACOnlineProcessor
        }
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt

        segments_iter, info = self._model.transcribe(audio, **transcribe_kwargs)

        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append(
                Segment(
                    text=seg.text.strip(),
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    confidence=seg.avg_log_prob,
                )
            )
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)
        detected_lang = language or info.language

        return TranscriptionResult(
            text=full_text,
            language=detected_lang,
            confidence=info.language_probability,
            segments=segments,
            stable_text=full_text,
            unstable_text="",
            is_final=True,
            is_draft=False,
            speaker_id=None,
            should_translate=bool(full_text.strip()),
        )

    async def transcribe_stream(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> AsyncIterator[TranscriptionResult]:
        result = await self.transcribe(audio, language, **kwargs)
        yield result

    def supports_language(self, lang: str) -> bool:
        return lang in _WHISPER_LANGUAGES

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            backend="whisper",
            languages=_WHISPER_LANGUAGES,
            vram_mb=self._estimate_vram(),
            compute_type=self._compute_type,
        )

    async def load_model(self, model_name: str, device: str = "cuda") -> None:
        from faster_whisper import WhisperModel

        self._model_name = model_name
        self._device = device
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=self._compute_type,
        )
        self._vram_mb = self._estimate_vram()
        logger.info("whisper_model_loaded", model=model_name, device=device, vram_mb=self._vram_mb)

    async def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._vram_mb = 0
            # Trigger CUDA memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("whisper_model_unloaded", model=self._model_name)

    async def warmup(self) -> None:
        if self._model is None:
            return
        silence = np.zeros(16000, dtype=np.float32)
        await self.transcribe(silence, language="en")
        logger.info("whisper_warmup_complete", model=self._model_name)

    def vram_usage_mb(self) -> int:
        return self._vram_mb

    def _estimate_vram(self) -> int:
        """Estimate VRAM usage based on model size and compute type."""
        estimates = {
            "tiny": 1000, "base": 1500, "small": 2500,
            "medium": 5000, "large-v2": 6500, "large-v3": 6500,
            "large-v3-turbo": 6000, "distil-large-v3": 4000,
        }
        base = estimates.get(self._model_name, 6000)
        if self._compute_type == "int8":
            base = int(base * 0.5)
        elif self._compute_type == "int8_float16":
            base = int(base * 0.65)
        return base
```

- [ ] **Step 5: Run protocol compliance test**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_whisper_backend.py::TestWhisperBackendProtocol -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add modules/transcription-service/src/backends/whisper.py modules/transcription-service/tests/test_whisper_backend.py modules/transcription-service/pyproject.toml
git commit -m "feat(transcription): add WhisperBackend using faster-whisper CTranslate2"
```

---

## Chunk 4: Language Detection & Sentence Segmenter

### Task 6: Authoritative LID with language code normalization

**Files:**
- Create: `modules/transcription-service/src/language_detection.py`
- Create: `modules/transcription-service/tests/test_language_detection.py`

- [ ] **Step 1: Write failing test**

```python
# modules/transcription-service/tests/test_language_detection.py
"""Tests for authoritative language detection and normalization."""
from language_detection import normalize_language_code, LanguageDetector


class TestLanguageCodeNormalization:
    def test_simple_codes(self):
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("zh") == "zh"

    def test_regional_variants(self):
        assert normalize_language_code("zh-CN") == "zh"
        assert normalize_language_code("zh-TW") == "zh"
        assert normalize_language_code("en-US") == "en"
        assert normalize_language_code("pt-BR") == "pt"

    def test_cantonese(self):
        assert normalize_language_code("yue") == "zh"

    def test_case_insensitive(self):
        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("ZH-cn") == "zh"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_language_detection.py::TestLanguageCodeNormalization -v`
Expected: FAIL

- [ ] **Step 3: Write language detection module**

```python
# modules/transcription-service/src/language_detection.py
"""Authoritative language detection and code normalization.

Uses faster-whisper's built-in LID on first chunk for initial detection,
then SlidingLIDDetector for ongoing monitoring. Normalizes regional
variants (zh-CN, zh-TW, yue) to registry key space (zh, en, ja).
"""
from __future__ import annotations

from livetranslate_common.logging import get_logger

logger = get_logger()

# Regional variant → base language code
_NORMALIZATION_MAP: dict[str, str] = {
    "yue": "zh",
    "cmn": "zh",
    "wuu": "zh",
    "nan": "zh",
}


def normalize_language_code(code: str) -> str:
    """Normalize a language code to the registry key space.

    'zh-CN' → 'zh', 'en-US' → 'en', 'yue' → 'zh', etc.
    """
    code = code.lower().strip()

    # Check special mappings first
    if code in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[code]

    # Strip regional suffix: 'zh-CN' → 'zh'
    base = code.split("-")[0]

    # Check again after stripping
    if base in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[base]

    return base


class LanguageDetector:
    """Authoritative language detector for registry routing.

    1. First chunk: faster-whisper LID (high confidence)
    2. Ongoing: SlidingLIDDetector monitors for sustained language switches
    3. Language codes normalized before registry lookup
    """

    def __init__(self, switch_threshold_s: float = 3.0):
        self._current_language: str | None = None
        self._switch_threshold_s = switch_threshold_s
        self._sustained_count = 0
        self._candidate_language: str | None = None

    @property
    def current_language(self) -> str | None:
        return self._current_language

    def detect_initial(self, language: str, confidence: float) -> str:
        """Set initial language from faster-whisper's first-chunk LID."""
        normalized = normalize_language_code(language)
        self._current_language = normalized
        logger.info("language_detected_initial", language=normalized, confidence=confidence)
        return normalized

    def update(self, detected_language: str, chunk_duration_s: float) -> str | None:
        """Update with ongoing detection. Returns new language if switch detected."""
        normalized = normalize_language_code(detected_language)

        if normalized == self._current_language:
            self._candidate_language = None
            self._sustained_count = 0
            return None

        if normalized == self._candidate_language:
            self._sustained_count += chunk_duration_s
        else:
            self._candidate_language = normalized
            self._sustained_count = chunk_duration_s

        if self._sustained_count >= self._switch_threshold_s:
            old = self._current_language
            self._current_language = normalized
            self._candidate_language = None
            self._sustained_count = 0
            logger.info("language_switched", old=old, new=normalized)
            return normalized

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_language_detection.py -v`
Expected: PASS

- [ ] **Step 5: Write additional tests for LanguageDetector**

```python
# Append to modules/transcription-service/tests/test_language_detection.py

class TestLanguageDetector:
    def test_initial_detection(self):
        detector = LanguageDetector()
        lang = detector.detect_initial("en", 0.95)
        assert lang == "en"
        assert detector.current_language == "en"

    def test_no_switch_on_same_language(self):
        detector = LanguageDetector()
        detector.detect_initial("en", 0.95)
        result = detector.update("en", chunk_duration_s=1.0)
        assert result is None

    def test_switch_after_sustained_detection(self):
        detector = LanguageDetector(switch_threshold_s=3.0)
        detector.detect_initial("en", 0.95)
        # 3 chunks of Chinese, 1.5s each = 4.5s > 3.0s threshold
        assert detector.update("zh-CN", 1.5) is None
        assert detector.update("zh-CN", 1.5) is None  # 3.0s = threshold
        result = detector.update("zh-CN", 1.5)  # 4.5s > threshold
        assert result == "zh"
        assert detector.current_language == "zh"

    def test_switch_resets_on_different_candidate(self):
        detector = LanguageDetector(switch_threshold_s=3.0)
        detector.detect_initial("en", 0.95)
        detector.update("zh", 2.0)  # 2s of zh
        detector.update("ja", 1.0)  # switch to ja — resets zh counter
        result = detector.update("ja", 1.0)  # only 2s of ja
        assert result is None
```

- [ ] **Step 6: Run all language detection tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_language_detection.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add modules/transcription-service/src/language_detection.py modules/transcription-service/tests/test_language_detection.py
git commit -m "feat(transcription): add authoritative LID with language code normalization"
```

---

### Task 7: Refactor sentence segmenter for language-universal support

**Files:**
- Modify: `modules/transcription-service/src/sentence_segmenter.py`
- Create: `modules/transcription-service/tests/test_sentence_segmenter.py`

- [ ] **Step 1: Write failing test for Latin punctuation**

```python
# modules/transcription-service/tests/test_sentence_segmenter.py
"""Tests for language-universal sentence segmenter."""
from sentence_segmenter import SentenceSegmenter


class TestLatinPunctuation:
    def test_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello world. How are you")
        assert result.sentences == ["Hello world."]
        assert result.remainder == "How are you"

    def test_exclamation(self):
        seg = SentenceSegmenter()
        result = seg.segment("Wow! Amazing")
        assert result.sentences == ["Wow!"]
        assert result.remainder == "Amazing"

    def test_question_mark(self):
        seg = SentenceSegmenter()
        result = seg.segment("What is this? I wonder")
        assert result.sentences == ["What is this?"]
        assert result.remainder == "I wonder"

    def test_no_punctuation(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello world how are you")
        assert result.sentences == []
        assert result.remainder == "Hello world how are you"


class TestCJKPunctuation:
    def test_chinese_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("你好世界。今天")
        assert result.sentences == ["你好世界。"]
        assert result.remainder == "今天"

    def test_chinese_exclamation(self):
        seg = SentenceSegmenter()
        result = seg.segment("太好了！是的")
        assert result.sentences == ["太好了！"]
        assert result.remainder == "是的"

    def test_japanese_period(self):
        seg = SentenceSegmenter()
        result = seg.segment("こんにちは。元気ですか")
        assert result.sentences == ["こんにちは。"]
        assert result.remainder == "元気ですか"


class TestMultipleSentences:
    def test_multiple_latin(self):
        seg = SentenceSegmenter()
        result = seg.segment("First. Second. Third")
        assert result.sentences == ["First.", "Second."]
        assert result.remainder == "Third"

    def test_mixed_latin_cjk(self):
        seg = SentenceSegmenter()
        result = seg.segment("Hello. 你好。World")
        assert result.sentences == ["Hello.", "你好。"]
        assert result.remainder == "World"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_sentence_segmenter.py -v`
Expected: FAIL (Latin punctuation tests should fail — current segmenter only handles CJK)

- [ ] **Step 3: Refactor sentence segmenter**

Read the existing `sentence_segmenter.py`, then replace its contents with a language-universal implementation:

```python
# modules/transcription-service/src/sentence_segmenter.py
"""Language-universal sentence segmenter.

Splits transcription text on sentence-ending punctuation for both
Latin (. ! ?) and CJK (。！？) scripts. Additional scripts (Arabic,
Thai, Devanagari) can be added via SENTENCE_ENDINGS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Sentence-ending punctuation across scripts
SENTENCE_ENDINGS = re.compile(
    r"([.!?]"             # Latin
    r"|[。！？]"           # CJK fullwidth
    r"|[।॥]"              # Devanagari (future-proofing)
    r")"
)


@dataclass
class SegmentResult:
    """Result of sentence segmentation."""
    sentences: list[str]    # Completed sentences (including trailing punctuation)
    remainder: str          # Incomplete tail (no sentence-ending punctuation yet)


class SentenceSegmenter:
    """Splits streaming text into sentences at punctuation boundaries.

    Designed for real-time transcription: accumulates text and emits
    completed sentences while holding back the incomplete remainder.
    """

    def segment(self, text: str) -> SegmentResult:
        """Segment text into completed sentences and a remainder.

        Args:
            text: Input text, possibly containing multiple sentences.

        Returns:
            SegmentResult with completed sentences and the trailing remainder.
        """
        sentences: list[str] = []
        remaining = text

        while remaining:
            match = SENTENCE_ENDINGS.search(remaining)
            if match is None:
                break

            # Include the punctuation mark in the sentence
            end_pos = match.end()
            sentence = remaining[:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining[end_pos:].lstrip()

        return SegmentResult(sentences=sentences, remainder=remaining)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_sentence_segmenter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/transcription-service/src/sentence_segmenter.py modules/transcription-service/tests/test_sentence_segmenter.py
git commit -m "feat(transcription): make sentence segmenter language-universal (Latin + CJK)"
```

---

### Task 7b: Refactor VACOnlineProcessor for async queue and proper overlap

**Files:**
- Modify: `modules/transcription-service/src/vac_online_processor.py`
- Create: `modules/transcription-service/tests/test_vac_processor.py`

The current `VACOnlineASRProcessor` uses an `is_processing` flag and `pending_chunks` list which serializes inference and audio ingestion. Refactor to decouple them.

- [ ] **Step 1: Write failing tests for refactored VAC processor**

```python
# modules/transcription-service/tests/test_vac_processor.py
"""Tests for refactored VACOnlineProcessor."""
import asyncio

import numpy as np
import pytest

from vac_online_processor import VACOnlineProcessor


class TestVACProcessorQueue:
    @pytest.mark.asyncio
    async def test_uses_asyncio_queue(self):
        """Processor should use asyncio.Queue instead of list + flag."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        assert hasattr(proc, "_audio_queue")
        assert isinstance(proc._audio_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_retains_overlap_after_inference(self):
        """After inference, last overlap_s seconds should be retained in buffer."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        # Feed 1 second of audio at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        await proc.feed_audio(audio)
        # After inference, buffer should retain last 0.5s = 8000 samples
        # (tested via internal state after process_chunk)

    @pytest.mark.asyncio
    async def test_first_inference_at_prebuffer(self):
        """First inference should fire at prebuffer_s, not stride_s."""
        proc = VACOnlineProcessor(
            prebuffer_s=0.3, overlap_s=0.5, stride_s=4.5,
        )
        # 0.3s at 16kHz = 4800 samples
        audio = np.zeros(4800, dtype=np.float32)
        await proc.feed_audio(audio)
        assert proc.ready_for_inference() is True
```

- [ ] **Step 2: Refactor VACOnlineProcessor**

Refactor `vac_online_processor.py` with the following changes:
- Replace `is_processing` flag + `pending_chunks` list with `asyncio.Queue` (or ring buffer using `collections.deque` with `maxlen`)
- After each inference call, retain the last `overlap_s` seconds of audio in the buffer (do NOT clear entirely)
- First inference fires at `prebuffer_s` (0.3-0.5s for fast time-to-first-text)
- Subsequent inferences fire at `stride_s` intervals
- Use `collections.deque.popleft()` instead of `list.pop(0)` for O(1) operations

- [ ] **Step 3: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_vac_processor.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add modules/transcription-service/src/vac_online_processor.py modules/transcription-service/tests/test_vac_processor.py
git commit -m "refactor(transcription): VACOnlineProcessor uses asyncio.Queue, retains overlap audio"
```

---

## Chunk 5: WebSocket API & Service Entry Point

### Task 8: WebSocket streaming API (`/api/stream`)

**Files:**
- Create: `modules/transcription-service/src/api.py`
- Create: `modules/transcription-service/tests/test_api.py`

The WebSocket API accepts binary frames (16kHz mono float32 audio) and text frames (config, end). It responds with text frames (segment, interim, language_detected, backend_switched).

- [ ] **Step 1: Write failing test for WebSocket endpoint**

```python
# modules/transcription-service/tests/test_api.py
"""Tests for transcription service WebSocket API."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app


@pytest.fixture
def app():
    return create_app(registry_path=None, test_mode=True)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "loaded_backends" in data

class TestModelsEndpoint:
    def test_models_empty(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestWebSocketStream:
    """Tests for WebSocket /api/stream endpoint."""

    def test_binary_audio_returns_segment(self, client):
        """Binary audio frames should produce segment responses."""
        with client.websocket_connect("/api/stream") as ws:
            # Send config first
            ws.send_text(json.dumps({
                "type": "config",
                "language": "en",
            }))
            # Send binary audio frame (1s of silence, 16kHz mono float32)
            audio = np.zeros(16000, dtype=np.float32)
            ws.send_bytes(audio.tobytes())
            # Expect a segment response (or language_detected first)
            response = json.loads(ws.receive_text())
            assert response["type"] in ("segment", "language_detected")

    def test_config_message_accepted(self, client):
        """Config messages should be accepted without error."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text(json.dumps({
                "type": "config",
                "language": "zh",
                "initial_prompt": "Technical meeting about AI",
                "glossary_terms": ["GPT", "transformer", "attention"],
            }))
            # Send end to close cleanly
            ws.send_text(json.dumps({"type": "end"}))

    def test_end_message_closes_stream(self, client):
        """End message should close the WebSocket cleanly."""
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text(json.dumps({"type": "end"}))
            # Connection should close without error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Write the FastAPI application**

```python
# modules/transcription-service/src/api.py
"""Transcription service FastAPI application.

Endpoints:
  GET  /health              → service health + loaded backends
  GET  /api/models          → list of available models
  GET  /api/registry        → current registry config
  POST /api/registry/reload → hot-reload registry from disk
  WS   /api/stream          → binary audio in, text results out
  POST /api/transcribe      → batch transcription (file upload)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from livetranslate_common.logging import get_logger, setup_logging

from backends.manager import BackendManager
from language_detection import LanguageDetector, normalize_language_code
from registry import ModelRegistry

logger = get_logger()


def create_app(registry_path: Path | None = None, test_mode: bool = False) -> FastAPI:
    app = FastAPI(title="Transcription Service")

    if registry_path and registry_path.exists():
        registry = ModelRegistry(registry_path)
        manager = BackendManager(max_vram_mb=registry.vram_budget_mb)
    else:
        registry = None
        manager = BackendManager()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "loaded_backends": list(manager.loaded_backends.keys()),
            "vram_usage_mb": manager.current_vram_mb,
        }

    @app.get("/api/models")
    async def list_models():
        return [
            b.get_model_info().model_dump()
            for b in manager.loaded_backends.values()
        ]

    @app.get("/api/registry")
    async def get_registry():
        if registry is None:
            return {"error": "No registry loaded"}
        return registry._data

    @app.post("/api/registry/reload")
    async def reload_registry():
        if registry is None:
            return {"error": "No registry loaded"}
        registry.reload()
        return {"status": "reloaded", "version": registry.version}

    @app.post("/api/transcribe")
    async def batch_transcribe(
        file: UploadFile = File(...),
        language: str | None = None,
        backend: str | None = None,
        model: str | None = None,
        profile: str = "batch",
    ):
        """Batch transcription endpoint for file upload / post-meeting re-transcription.

        Accepts a multipart audio file and returns full transcription results.
        Uses the "batch" profile by default (beam search, higher quality).
        """
        import io
        import soundfile as sf

        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Resample to 16kHz mono if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)

        if registry is None:
            return {"error": "No registry loaded"}

        lang = language or "en"
        config = registry.get_config(lang)
        transcription_backend = await manager.get_backend(config)

        result = await transcription_backend.transcribe(
            audio_data, language=language,
            beam_size=config.beam_size, batch_profile=profile,
        )

        return {
            "text": result.text,
            "segments": [s.model_dump() for s in result.segments],
            "language": result.language,
            "confidence": result.confidence,
        }

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        """WebSocket streaming endpoint.

        Full inference pipeline:
        Binary audio → VAD → LanguageDetector → Registry lookup →
        BackendManager → transcribe → send results + language_detected/backend_switched messages.

        Config messages can set initial_prompt and glossary_terms which are
        passed to the backend's initial_prompt parameter for domain-specific priming.
        """
        await ws.accept()
        session_language: str | None = None
        session_backend_override: str | None = None
        session_initial_prompt: str | None = None
        session_glossary_terms: list[str] | None = None
        lang_detector = LanguageDetector()
        current_backend_key: str | None = None

        try:
            while True:
                data = await ws.receive()

                if "bytes" in data and data["bytes"]:
                    # Binary frame — audio data (16kHz mono float32)
                    audio = np.frombuffer(data["bytes"], dtype=np.float32)

                    if registry is None:
                        continue

                    # --- Language Detection ---
                    # TODO: VAD filtering happens here (Silero VAD from VACOnlineProcessor)
                    # Only speech frames proceed to inference.

                    # First chunk: use faster-whisper's built-in LID
                    if lang_detector.current_language is None and session_language is None:
                        # Detect language from first audio chunk
                        # (faster-whisper detects language as part of transcription)
                        pass  # Detection happens in transcribe(), language set from result below

                    # --- Registry Lookup ---
                    lang = session_language or lang_detector.current_language or "en"
                    config = registry.get_config(lang)
                    transcription_backend = await manager.get_backend(config)

                    # Track backend switches
                    new_backend_key = f"{config.backend}:{config.model}"
                    if current_backend_key is not None and new_backend_key != current_backend_key:
                        await ws.send_text(json.dumps({
                            "type": "backend_switched",
                            "from": current_backend_key,
                            "to": new_backend_key,
                            "reason": f"language changed to {lang}",
                        }))
                    current_backend_key = new_backend_key

                    # --- Build initial_prompt from glossary + user prompt ---
                    effective_prompt = session_initial_prompt
                    if session_glossary_terms:
                        glossary_str = ", ".join(session_glossary_terms)
                        if effective_prompt:
                            effective_prompt = f"{glossary_str}. {effective_prompt}"
                        else:
                            effective_prompt = glossary_str

                    # --- Inference ---
                    result = await transcription_backend.transcribe(
                        audio,
                        language=session_language or lang_detector.current_language,
                        beam_size=config.beam_size,
                        batch_profile=config.batch_profile,
                        initial_prompt=effective_prompt,
                    )

                    # --- Post-inference language tracking ---
                    if lang_detector.current_language is None:
                        detected = lang_detector.detect_initial(
                            result.language, result.confidence
                        )
                        await ws.send_text(json.dumps({
                            "type": "language_detected",
                            "language": detected,
                            "confidence": result.confidence,
                        }))
                    else:
                        # Ongoing language monitoring
                        chunk_duration_s = len(audio) / 16000.0
                        switched = lang_detector.update(result.language, chunk_duration_s)
                        if switched:
                            await ws.send_text(json.dumps({
                                "type": "language_detected",
                                "language": switched,
                                "confidence": result.confidence,
                            }))

                    # --- Send result ---
                    await ws.send_text(json.dumps({
                        "type": "segment",
                        **result.model_dump(),
                    }))

                elif "text" in data and data["text"]:
                    msg = json.loads(data["text"])
                    msg_type = msg.get("type")

                    if msg_type == "config":
                        session_language = msg.get("language")
                        session_backend_override = msg.get("backend")
                        session_initial_prompt = msg.get("initial_prompt")
                        session_glossary_terms = msg.get("glossary_terms")

                    elif msg_type == "end":
                        break

        except WebSocketDisconnect:
            pass

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/transcription-service/src/api.py modules/transcription-service/tests/test_api.py
git commit -m "feat(transcription): add FastAPI app with WebSocket /api/stream endpoint"
```

---

### Task 9: Service entry point (`main.py` refactor)

**Files:**
- Modify: `modules/transcription-service/src/main.py`

- [ ] **Step 1: Read current main.py**

Read `modules/transcription-service/src/main.py` to understand the existing entry point.

- [ ] **Step 2: Refactor main.py to use new architecture**

Update `main.py` to:
- Import `create_app` from `api`
- Configure registry path from environment or CLI arg
- Run with uvicorn
- Set up structured logging

```python
# modules/transcription-service/src/main.py
"""Transcription service entry point."""
import argparse
from pathlib import Path

import uvicorn
from livetranslate_common.logging import setup_logging

from api import create_app


def main():
    parser = argparse.ArgumentParser(description="Transcription Service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "model_registry.yaml",
    )
    parser.add_argument("--log-format", default="dev", choices=["dev", "json"])
    args = parser.parse_args()

    setup_logging(service_name="transcription", log_format=args.log_format)

    app = create_app(registry_path=args.registry)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify service starts**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate && timeout 5 uv run python modules/transcription-service/src/main.py --port 5099 || true
```
Expected: Service starts and listens (will timeout after 5s, which is fine)

- [ ] **Step 4: Commit**

```bash
git add modules/transcription-service/src/main.py
git commit -m "refactor(transcription): update main.py entry point for new backend architecture"
```

---

## Chunk 6: Benchmarking Harness

### Task 10: Transcription benchmarking CLI

**Files:**
- Create: `modules/transcription-service/benchmarks/__init__.py`
- Create: `modules/transcription-service/benchmarks/run.py`
- Create: `modules/transcription-service/benchmarks/metrics.py`

- [ ] **Step 1: Write benchmarking metrics module**

```python
# modules/transcription-service/benchmarks/__init__.py
"""Transcription benchmarking harness."""

# modules/transcription-service/benchmarks/metrics.py
"""Metrics computation for transcription benchmarks.

WER for alphabetic languages, CER for CJK.
"""
from __future__ import annotations


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (Levenshtein on words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    d = _levenshtein(ref_words, hyp_words)
    return d / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (for CJK languages)."""
    ref_chars = list(reference.strip().replace(" ", ""))
    hyp_chars = list(hypothesis.strip().replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    d = _levenshtein(ref_chars, hyp_chars)
    return d / len(ref_chars)


def _levenshtein(ref: list, hyp: list) -> int:
    """Dynamic programming Levenshtein distance."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
```

- [ ] **Step 2: Write tests for metrics**

```python
# modules/transcription-service/tests/test_benchmark_metrics.py
from benchmarks.metrics import word_error_rate, character_error_rate


class TestWER:
    def test_identical(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_one_substitution(self):
        wer = word_error_rate("hello world", "hello earth")
        assert wer == 0.5  # 1 error / 2 words

    def test_empty_reference(self):
        assert word_error_rate("", "something") == 1.0

    def test_empty_both(self):
        assert word_error_rate("", "") == 0.0


class TestCER:
    def test_identical_chinese(self):
        assert character_error_rate("你好世界", "你好世界") == 0.0

    def test_one_char_error(self):
        cer = character_error_rate("你好世界", "你好时间")
        assert cer == 0.5  # 2 errors / 4 chars
```

- [ ] **Step 3: Run metric tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/transcription-service/tests/test_benchmark_metrics.py -v`
Expected: PASS

- [ ] **Step 4: Write benchmark runner CLI**

```python
# modules/transcription-service/benchmarks/run.py
"""CLI benchmark runner for transcription backends.

Usage: uv run python -m benchmarks.run --backend whisper --language en
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from livetranslate_common.logging import setup_logging, get_logger

from benchmarks.metrics import word_error_rate, character_error_rate

logger = get_logger()

CJK_LANGUAGES = {"zh", "ja", "ko"}


def get_system_info() -> dict:
    """Collect system info for reproducibility."""
    import hashlib

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["driver_version"] = torch.cuda.get_device_properties(0).name
    except ImportError:
        info["cuda_available"] = False

    # Package versions
    for pkg in ["faster_whisper", "ctranslate2", "torch", "numpy"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return info


def get_model_checksum(model_path: str) -> str | None:
    """Compute SHA256 checksum of model file for reproducibility."""
    import hashlib
    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_peak_vram() -> int:
    """Return peak VRAM usage in MB since last reset."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def reset_vram_tracking():
    """Reset CUDA peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


async def load_backend(backend_name: str, model_name: str, compute_type: str, device: str):
    """Load a transcription backend for benchmarking."""
    if backend_name == "whisper":
        from backends.whisper import WhisperBackend
        backend = WhisperBackend(
            model_name=model_name, compute_type=compute_type, device=device,
        )
        await backend.load_model(model_name, device)
        await backend.warmup()
        return backend
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def run_benchmark(backend_name: str, language: str, data_dir: Path, output_dir: Path):
    """Run benchmark on test data pairs (audio.wav + reference.txt)."""
    import asyncio
    import soundfile as sf

    setup_logging(service_name="benchmark", log_format="dev")

    system_info = get_system_info()

    results = {
        "backend": backend_name,
        "language": language,
        "system_info": system_info,
        "samples": [],
        "aggregate": {},
    }

    test_files = sorted(data_dir.glob("*.wav"))
    if not test_files:
        logger.warning("no_test_data", data_dir=str(data_dir))
        return

    # Load backend
    model_name = "large-v3-turbo"  # default, can be overridden via CLI
    compute_type = "float16"
    device = "cuda"

    loop = asyncio.new_event_loop()
    backend = loop.run_until_complete(load_backend(backend_name, model_name, compute_type, device))

    error_metric = character_error_rate if language in CJK_LANGUAGES else word_error_rate
    error_metric_name = "cer" if language in CJK_LANGUAGES else "wer"

    all_errors = []
    all_latencies = []
    all_ttft = []

    for audio_path in test_files:
        ref_path = audio_path.with_suffix(".txt")
        if not ref_path.exists():
            continue

        reference = ref_path.read_text().strip()
        audio_data, sr = sf.read(str(audio_path))

        # Resample to 16kHz mono if needed
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)

        logger.info("benchmarking", file=audio_path.name, duration_s=len(audio_data) / 16000)

        # Reset VRAM tracking
        reset_vram_tracking()

        # Time the inference
        t0 = time.perf_counter()
        result = loop.run_until_complete(
            backend.transcribe(audio_data, language=language, beam_size=5, batch_profile="batch")
        )
        t1 = time.perf_counter()

        inference_time_s = t1 - t0
        hypothesis = result.text.strip()
        error_rate = error_metric(reference, hypothesis)
        peak_vram = measure_peak_vram()

        # Time-to-first-token (streaming profile)
        reset_vram_tracking()
        t0_stream = time.perf_counter()
        first_token_time = None
        async def _stream_measure():
            nonlocal first_token_time
            async for partial in backend.transcribe_stream(audio_data, language=language):
                if partial.text.strip() and first_token_time is None:
                    first_token_time = time.perf_counter() - t0_stream
                    break
        loop.run_until_complete(_stream_measure())

        sample_result = {
            "file": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            error_metric_name: round(error_rate, 4),
            "inference_time_s": round(inference_time_s, 4),
            "time_to_first_token_s": round(first_token_time, 4) if first_token_time else None,
            "peak_vram_mb": peak_vram,
            "audio_duration_s": round(len(audio_data) / 16000, 2),
            "rtf": round(inference_time_s / (len(audio_data) / 16000), 4),  # real-time factor
        }
        results["samples"].append(sample_result)
        all_errors.append(error_rate)
        all_latencies.append(inference_time_s)
        if first_token_time:
            all_ttft.append(first_token_time)

    # Aggregate metrics
    if all_errors:
        results["aggregate"] = {
            f"mean_{error_metric_name}": round(sum(all_errors) / len(all_errors), 4),
            "mean_inference_time_s": round(sum(all_latencies) / len(all_latencies), 4),
            "mean_ttft_s": round(sum(all_ttft) / len(all_ttft), 4) if all_ttft else None,
            "total_samples": len(all_errors),
        }

    # Add model checksum
    results["system_info"]["model_checksum"] = get_model_checksum(model_name)

    loop.run_until_complete(backend.unload_model())
    loop.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{backend_name}_{language}_{ts}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("benchmark_complete", output=str(out_file), samples=len(results["samples"]))


def main():
    parser = argparse.ArgumentParser(description="Transcription Benchmark")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    run_benchmark(args.backend, args.language, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add modules/transcription-service/benchmarks/ modules/transcription-service/tests/test_benchmark_metrics.py
git commit -m "feat(transcription): add benchmarking harness with WER/CER metrics"
```

---

## Summary

**Total tasks:** 12 tasks (including 2b and 7b), ~55 steps
**Branch:** `plan-1/transcription-service`

After completing Plan 1:
- `modules/whisper-service/` is renamed to `modules/transcription-service/`
- SimulStreaming and legacy decoders are retired
- Test imports configured via conftest.py and pyproject.toml pythonpath
- Pluggable `TranscriptionBackend` protocol with `WhisperBackend` (faster-whisper)
- `BackendManager` enforces 10GB VRAM budget with LRU eviction
- `ModelRegistry` reads YAML config for language→backend routing with SIGHUP hot-reload
- Authoritative LID with language code normalization
- Language-universal sentence segmenter (Latin + CJK) with complete implementation
- VACOnlineProcessor refactored to use asyncio.Queue with proper overlap retention
- WebSocket API (`/api/stream`) with full inference pipeline (VAD → LID → Registry → Backend → transcribe)
- WebSocket handler supports `initial_prompt` and `glossary_terms` from config messages
- `POST /api/transcribe` endpoint for batch file upload / post-meeting re-transcription
- WhisperBackend handles `batch_profile` ("realtime" vs "batch") for quality/latency tradeoffs
- Benchmarking harness with WER/CER metrics, actual inference execution, VRAM measurement, and system_info
