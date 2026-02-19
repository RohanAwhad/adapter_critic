from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

DEFAULT_LOGGING_LEVEL = "INFO"
LOGGING_LEVEL_ENV = "LOGGING_LEVEL"
LOG_FILE_PATH = Path("logs") / "adapter_critic.log"

_VALID_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def resolve_logging_level() -> str:
    configured = os.environ.get(LOGGING_LEVEL_ENV, DEFAULT_LOGGING_LEVEL).upper()
    if configured in _VALID_LEVELS:
        return configured
    return DEFAULT_LOGGING_LEVEL


def is_debug_logging_enabled() -> bool:
    return resolve_logging_level() in {"TRACE", "DEBUG"}


def configure_logging() -> str:
    level = resolve_logging_level()
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=level, colorize=False)
    logger.add(LOG_FILE_PATH, level=level, colorize=False)

    logger.info("logging configured level={} file={}", level, LOG_FILE_PATH)
    return level
