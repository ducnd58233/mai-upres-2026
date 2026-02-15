"""Centralized logging configuration for scripts and training."""

from __future__ import annotations

import logging

DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = "%(levelname)s: %(message)s"


def configure_logging(
    level: int = DEFAULT_LEVEL,
    fmt: str = DEFAULT_FORMAT,
    force: bool = False,
) -> None:
    logging.basicConfig(level=level, format=fmt, force=force)
