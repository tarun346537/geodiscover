"""Runtime configuration for survey planning and service workers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_API_URL, DEFAULT_PARTICLES, DEFAULT_WORKERS


@dataclass(frozen=True)
class Settings:
    """Service endpoints and local artifact locations."""

    ray_address: str | None = None
    problem_path: Path | None = None
    checkpoint: Path | None = None
    particles: int = DEFAULT_PARTICLES
    workers: int = DEFAULT_WORKERS
    redis_url: str | None = None
    log_level: str = "INFO"
    output_dir: Path = Path("outputs")
    api_url: str = DEFAULT_API_URL
    cpu_threads: int = 1

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Read optional JSON settings, then apply GEO_* environment overrides."""
        values = json.loads(path.read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("settings must be a JSON object")
        env = {
            "ray_address": "GEO_RAY_ADDRESS",
            "problem_path": "GEO_PROBLEM_JSON",
            "checkpoint": "GEO_POLICY_CHECKPOINT",
            "particles": "GEO_PARTICLES",
            "workers": "GEO_WORKERS",
            "redis_url": "GEO_REDIS_URL",
            "log_level": "GEO_LOG_LEVEL",
            "output_dir": "GEO_OUTPUT_DIR",
            "api_url": "GEO_API_URL",
            "cpu_threads": "GEO_CPU_THREADS",
        }
        for field, variable in env.items():
            if os.environ.get(variable):
                values[field] = os.environ[variable]
        for field in ("problem_path", "checkpoint", "output_dir"):
            if values.get(field):
                values[field] = Path(values[field])
        for field in ("particles", "workers", "cpu_threads"):
            if field in values:
                values[field] = int(values[field])
        settings = cls(**values)
        if settings.particles < 2 or settings.workers < 1 or settings.cpu_threads < 1:
            raise ValueError("set GEO_PARTICLES >= 2, GEO_WORKERS >= 1 and GEO_CPU_THREADS >= 1")
        if settings.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("set GEO_LOG_LEVEL to DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return settings
