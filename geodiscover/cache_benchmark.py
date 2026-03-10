"""Compare cached and uncached forward evaluation on the same model sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .cache import ForwardCache
from .config import Settings
from .core import Problem
from .logging import configure_logging, get_logger
from .simpeg_adapter import forward_for_problem
from .train_data import file_sha256, read_models
from .types import JsonMap, Vector

logger = get_logger(__name__)


def compare_forward_cache(forward: ForwardCache, models: list[Vector], repeats: int = 3) -> JsonMap:
    """Time identical representatives and assess quantization outside both timers.

    Each pass follows the same model order. The uncached pass bypasses local
    and Redis storage but retains any caching inside the numerical solver.
    Redis may already contain entries. Starting counters and hits expose that.
    """
    if not models or repeats < 1:
        raise ValueError("provide models and at least one repeat")
    started = time.perf_counter()
    exact = [np.asarray(forward.solver(model), float) for model in models]
    approximate = [forward.solve_uncached(model) for model in models]
    if any(a.shape != b.shape for a, b in zip(exact, approximate, strict=True)):
        raise ValueError("exact and quantized responses have different dimensions")
    error = np.concatenate([a - b for a, b in zip(exact, approximate, strict=True)])
    reference = np.concatenate(exact)
    setup_seconds = time.perf_counter() - started
    measurements = {}
    before = forward.stats()
    response_digests = {}
    for mode in ("uncached", "cached"):
        solve = forward.solve_uncached if mode == "uncached" else forward
        digest = hashlib.sha256()
        started = time.perf_counter()
        for _ in range(repeats):
            for model in models:
                digest.update(np.asarray(solve(model), dtype="<f8").tobytes())
        measurements[mode] = time.perf_counter() - started
        response_digests[mode] = digest.hexdigest()
    after = forward.stats()
    if response_digests["uncached"] != response_digests["cached"]:
        raise ValueError(
            "cached responses disagree with representative solves. inspect backend entries"
        )
    return {
        "schema_version": 1,
        "kind": "paired_forward_cache_measurement",
        "measured_at": datetime.now(UTC).isoformat(),
        "models_per_pass": len(models),
        "repeats": repeats,
        "calls_per_condition": len(models) * repeats,
        "order": ["uncached", "cached"],
        "diagnostic_setup_seconds": setup_seconds,
        "uncached_seconds": measurements["uncached"],
        "cached_seconds": measurements["cached"],
        "speedup": measurements["uncached"] / measurements["cached"],
        "quantization": forward.quantization,
        "quantization_prediction_rmse": float(np.sqrt(np.mean(error**2))),
        "quantization_relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(reference))
        if np.linalg.norm(reference) > 0
        else None,
        "forward_cache_start": before,
        "forward_cache_delta": {
            k: after[k] - before[k] for k in ("hits", "misses", "backend_failures")
        },
        "physics_namespace": forward.namespace,
        "response_sha256": response_digests["cached"],
        "timing_scope": "Paired forward-call wall time in fixed uncached/cached order, with diagnostics in a separate setup timer.",
    }


def main() -> None:
    """Write a measured solve-cache comparison for one configured survey."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quantization", type=float)
    parser.add_argument("--redis-url", default=settings.redis_url)
    parser.add_argument("--output", type=Path, default=settings.output_dir / "cache-benchmark.json")
    args = parser.parse_args()
    problem = Problem.from_dict(json.loads(args.problem.read_text()))
    models = read_models(args.models, problem, args.split)
    forward = forward_for_problem(problem, args.redis_url, args.quantization)
    report = compare_forward_cache(forward, models, args.repeats)
    report.update(
        problem_sha256=file_sha256(args.problem),
        models_sha256=file_sha256(args.models),
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info(
        "cache comparison complete", extra={"fields": {"output": str(args.output), **report}}
    )


if __name__ == "__main__":
    main()
