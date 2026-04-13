"""Structlog configuration for LiveTranslate services."""

import logging
import os
import sys
from pathlib import Path

import structlog

from livetranslate_common.logging.buffer import buffer_processor
from livetranslate_common.logging.processors import add_service_name, censor_sensitive_data

# Persistent log directory
LOG_DIR = Path(os.environ.get("LIVETRANSLATE_LOG_DIR", "/tmp/livetranslate/logs"))


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
        buffer_processor,  # Capture to ring buffer for dashboard
    ]
    if log_format == "dev":
        # Force colors via env var (for piped-but-displayed-in-terminal scenarios like just dev),
        # otherwise auto-detect from TTY
        use_colors = os.environ.get("FORCE_COLOR", "") == "1" or sys.stderr.isatty()
        # Distinct colors per level — default structlog uses green for both info+debug
        level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors=use_colors)
        if use_colors:
            level_styles.update({
                "debug": "\x1b[36m",         # cyan
                "info": "\x1b[32m",          # green
                "warning": "\x1b[33;1m",     # bold yellow
                "error": "\x1b[31;1m",       # bold red
                "critical": "\x1b[41;37;1m", # white on red bg
            })
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=use_colors,
            level_styles=level_styles,
        )
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

    # Also write JSON logs to persistent file for dashboard viewing
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{service_name}.jsonl"
        # Use RotatingFileHandler to prevent unbounded growth
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=3,
        )
        # Always use JSON for file logs (machine-readable)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
    except OSError as e:
        # Log to stderr if file handler setup fails (don't fail startup)
        sys.stderr.write(f"[{service_name}] Failed to setup file logging: {e}\n")


def get_logger(**initial_bindings: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger."""
    return structlog.get_logger(**initial_bindings)
