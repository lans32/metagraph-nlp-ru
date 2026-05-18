"""Настройка Python logging для pipeline (CLAUDE.md §16)."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger = logging.getLogger("metagraph_nlp")
    logger.setLevel(numeric_level)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
