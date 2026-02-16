"""Local experiment records and durable training checkpoints."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .logging import get_logger
from .types import JsonMap

logger = get_logger(__name__)


class RunTracker:
    """Append structured phase measurements without requiring an external account."""

    def __init__(self, path: Path, run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id or uuid4().hex
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, phase: str, step: int, metrics: JsonMap) -> None:
        """Record the measured values for one completed training step."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "phase": phase,
            "step": step,
            "metrics": metrics,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        logger.info("training step complete", extra={"fields": record})


def save_checkpoint(path: Path | None, state: JsonMap) -> None:
    """Atomically replace a checkpoint after a completed optimizer step."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path, phase: str, identity: JsonMap) -> JsonMap:
    """Reject resumes that would mix different data or optimization settings."""
    state: JsonMap = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("phase") != phase or state.get("identity") != identity:
        raise ValueError(
            "checkpoint settings differ. use the original data, seed and optimizer configuration"
        )
    return state


def rng_state(rng: np.random.Generator) -> JsonMap:
    """Capture CPU, NumPy and available CUDA random states."""
    return {
        "numpy": rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: JsonMap, rng: np.random.Generator) -> None:
    """Restore random streams before continuing optimizer work."""
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
