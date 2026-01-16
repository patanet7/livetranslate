"""
Model Registry for LiveTranslate.

Provides centralized model naming, aliases, and validation for all services.
This is the single source of truth for model names across the system.

Usage:
    from modules.shared.src.models import ModelRegistry, WhisperModel, TranslationBackend

    # Normalize model names
    model = ModelRegistry.normalize_whisper_model("base")  # Returns "whisper-base"

    # Validate model names
    if ModelRegistry.is_valid_whisper_model("whisper-large-v3"):
        ...

    # Get model info
    info = ModelRegistry.get_whisper_model_info("whisper-base")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# WHISPER MODELS
# =============================================================================

class WhisperModelSize(Enum):
    """Whisper model size categories."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


@dataclass
class WhisperModelInfo:
    """Information about a Whisper model."""
    canonical_name: str
    size: WhisperModelSize
    parameters: str  # e.g., "39M", "74M", "244M"
    multilingual: bool
    relative_speed: float  # 1.0 = base speed, higher = faster
    vram_gb: float  # Approximate VRAM requirement
    description: str
    aliases: List[str] = field(default_factory=list)
    deprecated: bool = False
    recommended_for: List[str] = field(default_factory=list)


# Canonical Whisper model definitions
WHISPER_MODELS: Dict[str, WhisperModelInfo] = {
    # Tiny models
    "whisper-tiny": WhisperModelInfo(
        canonical_name="whisper-tiny",
        size=WhisperModelSize.TINY,
        parameters="39M",
        multilingual=True,
        relative_speed=32.0,
        vram_gb=1.0,
        description="Fastest model, lower accuracy. Good for real-time on CPU.",
        aliases=["tiny", "whisper_tiny", "openai/whisper-tiny"],
        recommended_for=["real-time", "low-resource", "quick-testing"],
    ),
    "whisper-tiny.en": WhisperModelInfo(
        canonical_name="whisper-tiny.en",
        size=WhisperModelSize.TINY,
        parameters="39M",
        multilingual=False,
        relative_speed=32.0,
        vram_gb=1.0,
        description="English-only tiny model. Slightly better for English.",
        aliases=["tiny.en", "whisper_tiny_en"],
        recommended_for=["english-only", "real-time"],
    ),

    # Base models
    "whisper-base": WhisperModelInfo(
        canonical_name="whisper-base",
        size=WhisperModelSize.BASE,
        parameters="74M",
        multilingual=True,
        relative_speed=16.0,
        vram_gb=1.0,
        description="Good balance of speed and accuracy. Default for most use cases.",
        aliases=["base", "whisper_base", "openai/whisper-base"],
        recommended_for=["default", "balanced", "development"],
    ),
    "whisper-base.en": WhisperModelInfo(
        canonical_name="whisper-base.en",
        size=WhisperModelSize.BASE,
        parameters="74M",
        multilingual=False,
        relative_speed=16.0,
        vram_gb=1.0,
        description="English-only base model.",
        aliases=["base.en", "whisper_base_en"],
        recommended_for=["english-only"],
    ),

    # Small models
    "whisper-small": WhisperModelInfo(
        canonical_name="whisper-small",
        size=WhisperModelSize.SMALL,
        parameters="244M",
        multilingual=True,
        relative_speed=6.0,
        vram_gb=2.0,
        description="Better accuracy, moderate speed. Good for production.",
        aliases=["small", "whisper_small", "openai/whisper-small"],
        recommended_for=["production", "accuracy-focused"],
    ),
    "whisper-small.en": WhisperModelInfo(
        canonical_name="whisper-small.en",
        size=WhisperModelSize.SMALL,
        parameters="244M",
        multilingual=False,
        relative_speed=6.0,
        vram_gb=2.0,
        description="English-only small model.",
        aliases=["small.en", "whisper_small_en"],
        recommended_for=["english-only", "production"],
    ),

    # Medium models
    "whisper-medium": WhisperModelInfo(
        canonical_name="whisper-medium",
        size=WhisperModelSize.MEDIUM,
        parameters="769M",
        multilingual=True,
        relative_speed=2.0,
        vram_gb=5.0,
        description="High accuracy, slower. Requires GPU for real-time.",
        aliases=["medium", "whisper_medium", "openai/whisper-medium"],
        recommended_for=["high-accuracy", "gpu-available"],
    ),
    "whisper-medium.en": WhisperModelInfo(
        canonical_name="whisper-medium.en",
        size=WhisperModelSize.MEDIUM,
        parameters="769M",
        multilingual=False,
        relative_speed=2.0,
        vram_gb=5.0,
        description="English-only medium model.",
        aliases=["medium.en", "whisper_medium_en"],
        recommended_for=["english-only", "high-accuracy"],
    ),

    # Large models
    "whisper-large": WhisperModelInfo(
        canonical_name="whisper-large",
        size=WhisperModelSize.LARGE,
        parameters="1550M",
        multilingual=True,
        relative_speed=1.0,
        vram_gb=10.0,
        description="Original large model. Use large-v2 or v3 instead.",
        aliases=["large", "whisper_large", "openai/whisper-large"],
        deprecated=True,
        recommended_for=[],
    ),
    "whisper-large-v2": WhisperModelInfo(
        canonical_name="whisper-large-v2",
        size=WhisperModelSize.LARGE,
        parameters="1550M",
        multilingual=True,
        relative_speed=1.0,
        vram_gb=10.0,
        description="Improved large model. Better accuracy than v1.",
        aliases=["large-v2", "largev2", "whisper_large_v2", "openai/whisper-large-v2"],
        recommended_for=["maximum-accuracy", "batch-processing"],
    ),
    "whisper-large-v3": WhisperModelInfo(
        canonical_name="whisper-large-v3",
        size=WhisperModelSize.LARGE,
        parameters="1550M",
        multilingual=True,
        relative_speed=1.0,
        vram_gb=10.0,
        description="Latest large model. Best accuracy available.",
        aliases=["large-v3", "largev3", "whisper_large_v3", "openai/whisper-large-v3", "v3"],
        recommended_for=["maximum-accuracy", "production-critical"],
    ),

    # Turbo models (distilled)
    "whisper-large-v3-turbo": WhisperModelInfo(
        canonical_name="whisper-large-v3-turbo",
        size=WhisperModelSize.TURBO,
        parameters="809M",
        multilingual=True,
        relative_speed=8.0,
        vram_gb=6.0,
        description="Distilled v3 model. Near-v3 accuracy at 8x speed.",
        aliases=[
            "large-v3-turbo", "v3-turbo", "turbo",
            "whisper_large_v3_turbo", "openai/whisper-large-v3-turbo"
        ],
        recommended_for=["production", "real-time-gpu", "best-tradeoff"],
    ),
}

# Build alias lookup table
_WHISPER_ALIAS_MAP: Dict[str, str] = {}
for canonical, info in WHISPER_MODELS.items():
    _WHISPER_ALIAS_MAP[canonical.lower()] = canonical
    for alias in info.aliases:
        _WHISPER_ALIAS_MAP[alias.lower()] = canonical


# =============================================================================
# TRANSLATION BACKENDS & MODELS
# =============================================================================

class TranslationBackend(Enum):
    """Supported translation backends."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    GROQ = "groq"
    OPENAI = "openai"
    TRITON = "triton"


@dataclass
class TranslationModelInfo:
    """Information about a translation model."""
    canonical_name: str
    backend: TranslationBackend
    parameters: str
    context_length: int
    description: str
    aliases: List[str] = field(default_factory=list)
    requires_api_key: bool = False
    recommended_for: List[str] = field(default_factory=list)


# Common translation models per backend
TRANSLATION_MODELS: Dict[str, TranslationModelInfo] = {
    # Ollama models
    "ollama:mistral:latest": TranslationModelInfo(
        canonical_name="mistral:latest",
        backend=TranslationBackend.OLLAMA,
        parameters="7B",
        context_length=32768,
        description="Mistral 7B - Good general-purpose translation.",
        aliases=["mistral", "mistral:7b"],
        recommended_for=["general", "default"],
    ),
    "ollama:llama3.1:8b": TranslationModelInfo(
        canonical_name="llama3.1:8b",
        backend=TranslationBackend.OLLAMA,
        parameters="8B",
        context_length=128000,
        description="Llama 3.1 8B - Strong multilingual capabilities.",
        aliases=["llama3.1", "llama3"],
        recommended_for=["multilingual", "long-context"],
    ),
    "ollama:qwen3:4b": TranslationModelInfo(
        canonical_name="qwen3:4b",
        backend=TranslationBackend.OLLAMA,
        parameters="4B",
        context_length=32768,
        description="Qwen3 4B - Fast, good for Asian languages.",
        aliases=["qwen3", "qwen"],
        recommended_for=["asian-languages", "fast"],
    ),
    "ollama:gemma3:4b": TranslationModelInfo(
        canonical_name="gemma3:4b",
        backend=TranslationBackend.OLLAMA,
        parameters="4B",
        context_length=8192,
        description="Gemma3 4B - Google's efficient model.",
        aliases=["gemma3", "gemma"],
        recommended_for=["efficient", "general"],
    ),

    # Groq models
    "groq:llama-3.1-8b-instant": TranslationModelInfo(
        canonical_name="llama-3.1-8b-instant",
        backend=TranslationBackend.GROQ,
        parameters="8B",
        context_length=128000,
        description="Llama 3.1 8B on Groq - Ultra-fast inference.",
        aliases=["groq-llama", "llama-instant"],
        requires_api_key=True,
        recommended_for=["speed-critical", "cloud"],
    ),

    # OpenAI models
    "openai:gpt-4o-mini": TranslationModelInfo(
        canonical_name="gpt-4o-mini",
        backend=TranslationBackend.OPENAI,
        parameters="Unknown",
        context_length=128000,
        description="GPT-4o Mini - Cost-effective, high quality.",
        aliases=["gpt4o-mini", "gpt-mini"],
        requires_api_key=True,
        recommended_for=["quality-critical", "cloud"],
    ),
}


# =============================================================================
# MODEL REGISTRY CLASS
# =============================================================================

class ModelRegistry:
    """
    Central registry for all model names, aliases, and validation.

    This class provides the single source of truth for model naming across
    all LiveTranslate services.
    """

    # Default models for each service
    DEFAULT_WHISPER_MODEL = "whisper-base"
    DEFAULT_WHISPER_MODEL_PRODUCTION = "whisper-large-v3-turbo"
    DEFAULT_TRANSLATION_BACKEND = TranslationBackend.OLLAMA
    DEFAULT_TRANSLATION_MODEL = "mistral:latest"

    # Fallback chains
    WHISPER_FALLBACK_CHAIN = ["whisper-base", "whisper-tiny"]
    TRANSLATION_FALLBACK_CHAIN = ["mistral:latest", "llama3.1:8b"]

    # ==========================================================================
    # WHISPER MODEL METHODS
    # ==========================================================================

    @classmethod
    def normalize_whisper_model(cls, model_name: str) -> str:
        """
        Normalize a Whisper model name to its canonical form.

        Args:
            model_name: Any valid model name or alias

        Returns:
            Canonical model name (e.g., "whisper-base")

        Raises:
            ValueError: If the model name is not recognized

        Examples:
            >>> ModelRegistry.normalize_whisper_model("base")
            "whisper-base"
            >>> ModelRegistry.normalize_whisper_model("large-v3-turbo")
            "whisper-large-v3-turbo"
            >>> ModelRegistry.normalize_whisper_model("whisper-base")
            "whisper-base"
        """
        if not model_name:
            return cls.DEFAULT_WHISPER_MODEL

        normalized = model_name.lower().strip()

        # Direct lookup
        if normalized in _WHISPER_ALIAS_MAP:
            return _WHISPER_ALIAS_MAP[normalized]

        # Try with "whisper-" prefix
        with_prefix = f"whisper-{normalized}"
        if with_prefix in _WHISPER_ALIAS_MAP:
            return _WHISPER_ALIAS_MAP[with_prefix]

        # Log warning and return default
        logger.warning(
            f"Unknown Whisper model '{model_name}', using default '{cls.DEFAULT_WHISPER_MODEL}'"
        )
        return cls.DEFAULT_WHISPER_MODEL

    @classmethod
    def is_valid_whisper_model(cls, model_name: str) -> bool:
        """
        Check if a model name is valid (canonical or alias).

        Args:
            model_name: Model name to check

        Returns:
            True if valid, False otherwise
        """
        if not model_name:
            return False
        normalized = model_name.lower().strip()
        return normalized in _WHISPER_ALIAS_MAP or f"whisper-{normalized}" in _WHISPER_ALIAS_MAP

    @classmethod
    def get_whisper_model_info(cls, model_name: str) -> Optional[WhisperModelInfo]:
        """
        Get detailed information about a Whisper model.

        Args:
            model_name: Model name or alias

        Returns:
            WhisperModelInfo if found, None otherwise
        """
        try:
            canonical = cls.normalize_whisper_model(model_name)
            return WHISPER_MODELS.get(canonical)
        except ValueError:
            return None

    @classmethod
    def get_all_whisper_models(cls, include_deprecated: bool = False) -> List[str]:
        """
        Get list of all available Whisper model names.

        Args:
            include_deprecated: Include deprecated models

        Returns:
            List of canonical model names
        """
        return [
            name for name, info in WHISPER_MODELS.items()
            if include_deprecated or not info.deprecated
        ]

    @classmethod
    def get_whisper_models_by_size(cls, size: WhisperModelSize) -> List[str]:
        """Get all models of a specific size."""
        return [
            name for name, info in WHISPER_MODELS.items()
            if info.size == size and not info.deprecated
        ]

    @classmethod
    def get_recommended_whisper_model(
        cls,
        use_case: str = "default",
        gpu_available: bool = False,
        real_time: bool = False
    ) -> str:
        """
        Get recommended Whisper model for a use case.

        Args:
            use_case: Use case identifier
            gpu_available: Whether GPU is available
            real_time: Whether real-time processing is needed

        Returns:
            Recommended canonical model name
        """
        if real_time and not gpu_available:
            return "whisper-tiny"
        elif real_time and gpu_available:
            return "whisper-large-v3-turbo"
        elif gpu_available:
            return "whisper-large-v3"
        else:
            return "whisper-base"

    # ==========================================================================
    # TRANSLATION MODEL METHODS
    # ==========================================================================

    @classmethod
    def normalize_translation_model(
        cls,
        model_name: str,
        backend: Optional[TranslationBackend] = None
    ) -> Tuple[str, TranslationBackend]:
        """
        Normalize a translation model name.

        Args:
            model_name: Model name (may include backend prefix like "ollama:mistral")
            backend: Explicit backend (overrides prefix in model_name)

        Returns:
            Tuple of (canonical_model_name, backend)
        """
        if not model_name:
            return cls.DEFAULT_TRANSLATION_MODEL, cls.DEFAULT_TRANSLATION_BACKEND

        # Parse backend:model format
        if ":" in model_name and backend is None:
            parts = model_name.split(":", 1)
            try:
                backend = TranslationBackend(parts[0].lower())
                model_name = parts[1]
            except ValueError:
                # Not a backend prefix, might be model version (e.g., "mistral:latest")
                pass

        # Use default backend if not specified
        if backend is None:
            backend = cls.DEFAULT_TRANSLATION_BACKEND

        return model_name, backend

    @classmethod
    def get_translation_model_info(
        cls,
        model_name: str,
        backend: TranslationBackend
    ) -> Optional[TranslationModelInfo]:
        """Get information about a translation model."""
        key = f"{backend.value}:{model_name}"
        return TRANSLATION_MODELS.get(key)

    @classmethod
    def get_all_translation_models(
        cls,
        backend: Optional[TranslationBackend] = None
    ) -> List[str]:
        """Get all translation models, optionally filtered by backend."""
        models = []
        for key, info in TRANSLATION_MODELS.items():
            if backend is None or info.backend == backend:
                models.append(info.canonical_name)
        return models

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    @classmethod
    def validate_model_config(
        cls,
        whisper_model: Optional[str] = None,
        translation_model: Optional[str] = None,
        translation_backend: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Validate and normalize a complete model configuration.

        Args:
            whisper_model: Whisper model name
            translation_model: Translation model name
            translation_backend: Translation backend name

        Returns:
            Dict with normalized values and validation info
        """
        result = {
            "valid": True,
            "warnings": [],
            "whisper_model": None,
            "translation_model": None,
            "translation_backend": None,
        }

        # Validate Whisper model
        if whisper_model:
            if cls.is_valid_whisper_model(whisper_model):
                result["whisper_model"] = cls.normalize_whisper_model(whisper_model)
                info = cls.get_whisper_model_info(whisper_model)
                if info and info.deprecated:
                    result["warnings"].append(
                        f"Whisper model '{whisper_model}' is deprecated"
                    )
            else:
                result["valid"] = False
                result["warnings"].append(
                    f"Unknown Whisper model '{whisper_model}'"
                )

        # Validate translation backend
        if translation_backend:
            try:
                result["translation_backend"] = TranslationBackend(
                    translation_backend.lower()
                )
            except ValueError:
                result["valid"] = False
                result["warnings"].append(
                    f"Unknown translation backend '{translation_backend}'"
                )

        # Validate translation model
        if translation_model:
            model, backend = cls.normalize_translation_model(
                translation_model,
                result.get("translation_backend")
            )
            result["translation_model"] = model
            result["translation_backend"] = backend

        return result

    @classmethod
    def get_model_summary(cls) -> Dict[str, any]:
        """Get a summary of all available models."""
        return {
            "whisper": {
                "total": len(WHISPER_MODELS),
                "by_size": {
                    size.value: len(cls.get_whisper_models_by_size(size))
                    for size in WhisperModelSize
                },
                "default": cls.DEFAULT_WHISPER_MODEL,
                "production_default": cls.DEFAULT_WHISPER_MODEL_PRODUCTION,
            },
            "translation": {
                "total": len(TRANSLATION_MODELS),
                "backends": [b.value for b in TranslationBackend],
                "default_backend": cls.DEFAULT_TRANSLATION_BACKEND.value,
                "default_model": cls.DEFAULT_TRANSLATION_MODEL,
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_whisper_model(model_name: str) -> str:
    """Convenience function for ModelRegistry.normalize_whisper_model."""
    return ModelRegistry.normalize_whisper_model(model_name)


def is_valid_whisper_model(model_name: str) -> bool:
    """Convenience function for ModelRegistry.is_valid_whisper_model."""
    return ModelRegistry.is_valid_whisper_model(model_name)


def get_whisper_fallback_chain() -> List[str]:
    """Get the default Whisper model fallback chain."""
    return ModelRegistry.WHISPER_FALLBACK_CHAIN.copy()


def get_translation_fallback_chain() -> List[str]:
    """Get the default translation model fallback chain."""
    return ModelRegistry.TRANSLATION_FALLBACK_CHAIN.copy()
