"""Structlog processors for LiveTranslate services."""

from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "authorization",
        "cookie",
        "session_id",
        "credit_card",
    }
)


def censor_sensitive_data(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact values for keys that look like secrets."""
    for key in event_dict:
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***REDACTED***"
    return event_dict


def add_service_name(service_name: str) -> Any:
    """Return a processor that binds service=<name> to every event."""

    def processor(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        event_dict.setdefault("service", service_name)
        return event_dict

    return processor
