"""Create local survey and model files for the acquisition pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geodiscover.config import Settings
from geodiscover.constants import DEFAULT_SEED
from geodiscover.core import synthetic_problem
from geodiscover.logging import configure_logging, get_logger
from geodiscover.types import JsonMap


def write_samples(output: Path, count: int, seed: int) -> JsonMap:
    """Create schema-compatible local synthetic survey and model inputs."""
    if count < 6:
        raise ValueError("use at least six models to populate train, validation and test splits")
    output.mkdir(parents=True, exist_ok=True)
    problem = synthetic_problem(seed)
    (output / "problem.json").write_text(json.dumps(problem.to_dict(), indent=2) + "\n")
    rng = np.random.default_rng(seed + 1)
    counts = {"train": 0, "validation": 0, "test": 0}
    with (output / "models.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(count):
            split = "validation" if index % 5 == 0 else "test" if index % 5 == 1 else "train"
            counts[split] += 1
            model = rng.multivariate_normal(problem.prior_mean, problem.prior_covariance)
            row = {
                "model_id": f"sample-{seed}-{index}",
                "split": split,
                "source": "synthetic",
                "model": model.tolist(),
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return {"output": str(output), "models": count, "splits": counts, "source": "synthetic"}


def main() -> None:
    """Write a small synthetic survey and explicitly split model records."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=settings.output_dir / "samples")
    parser.add_argument("--models", type=int, default=18)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    report = write_samples(args.output, args.models, args.seed)
    get_logger(__name__).info("sample inputs written", extra={"fields": report})


if __name__ == "__main__":
    main()
