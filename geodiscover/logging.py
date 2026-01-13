"""Structured application and experiment logging."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    """Write one structured record per line, including request context."""

    def format(self, record: logging.LogRecord) -> str:
        """Encode application fields without losing exception details."""
        value = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id.get(),
        }
        value.update(getattr(record, "fields", {}))
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return json.dumps(value, default=str, allow_nan=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure only the application namespace, leaving the host server alone."""
    logger = logging.getLogger("geodiscover")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger within the configured application namespace."""
    return logging.getLogger(name if name.startswith("geodiscover") else f"geodiscover.{name}")
