"""Structlog configuration for LiveTranslate services."""

import logging
import sys

import structlog

from livetranslate_common.logging.processors import add_service_name, censor_sensitive_data


def setup_logging(service_name: str, log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structlog and stdlib logging for the entire process."""
    log_level = log_level.upper()
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_service_name(service_name),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        censor_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]
    if log_format == "dev":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))


def get_logger(**initial_bindings: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger."""
    return structlog.get_logger(**initial_bindings)
