"""Environment-based service configuration."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServiceSettings:
    """Immutable service configuration read from environment variables."""

    service_name: str
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_level", self.log_level.upper())
