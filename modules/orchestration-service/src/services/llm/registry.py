"""LLM provider registry — manages configured providers."""

from livetranslate_common.logging import get_logger

from .adapter import LLMAdapter, ModelInfo
from .providers.anthropic_provider import AnthropicAdapter
from .providers.ollama import OllamaAdapter
from .providers.openai_compat import OpenAICompatAdapter
from .providers.openai_provider import OpenAIAdapter

logger = get_logger()


# Provider factory mapping
PROVIDER_FACTORIES = {
    "ollama": lambda cfg: OllamaAdapter(
        base_url=cfg.get("base_url", "http://localhost:11434"),
        default_model=cfg.get("default_model", "qwen3.5:4b"),
    ),
    "openai": lambda cfg: OpenAIAdapter(
        api_key=cfg["api_key"],
        default_model=cfg.get("default_model", "gpt-4o"),
    ),
    "anthropic": lambda cfg: AnthropicAdapter(
        api_key=cfg["api_key"],
        default_model=cfg.get("default_model", "claude-sonnet-4-20250514"),
    ),
    "openai_compatible": lambda cfg: OpenAICompatAdapter(
        base_url=cfg["base_url"],
        api_key=cfg.get("api_key", ""),
        default_model=cfg.get("default_model", "default"),
    ),
}


class ProviderRegistry:
    """Manages configured LLM provider adapters."""

    def __init__(self):
        self._adapters: dict[str, LLMAdapter] = {}

    def configure(self, provider_name: str, config: dict) -> None:
        """Configure a provider with the given settings."""
        factory = PROVIDER_FACTORIES.get(provider_name)
        if not factory:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(PROVIDER_FACTORIES.keys())}"
            )
        try:
            self._adapters[provider_name] = factory(config)
            logger.info("llm_provider_configured", provider=provider_name)
        except Exception as e:
            logger.error(
                "llm_provider_config_failed", provider=provider_name, error=str(e)
            )
            raise

    def get_adapter(self, provider_name: str) -> LLMAdapter:
        """Get a configured adapter by provider name."""
        adapter = self._adapters.get(provider_name)
        if not adapter:
            raise ValueError(
                f"Provider '{provider_name}' is not configured. "
                f"Configured: {list(self._adapters.keys())}"
            )
        return adapter

    def list_providers(self) -> list[dict]:
        """List all available providers with their configuration status."""
        result = []
        for name in PROVIDER_FACTORIES:
            result.append({
                "name": name,
                "configured": name in self._adapters,
            })
        return result

    async def list_provider_models(self, provider_name: str) -> list[ModelInfo]:
        """List models for a specific provider."""
        adapter = self.get_adapter(provider_name)
        return await adapter.list_models()

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all configured providers."""
        results = {}
        for name, adapter in self._adapters.items():
            try:
                results[name] = await adapter.health_check()
            except Exception:
                results[name] = False
        return results

    def is_configured(self, provider_name: str) -> bool:
        """Check if a provider is configured."""
        return provider_name in self._adapters


# Global singleton
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get or create the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        # Auto-configure Ollama (local, no API key needed)
        try:
            _registry.configure("ollama", {})
            logger.info("ollama_auto_configured")
        except Exception:
            pass
    return _registry
