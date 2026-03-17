"""Measure reconstruction error versus acquisition budget on shared worlds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import Settings
from .constants import DEFAULT_PARTICLES, DEFAULT_SIMULATIONS, EVALUATION_SEED, EVALUATION_TRIALS
from .core import Problem
from .demo import evaluate, read_evaluation_models
from .logging import configure_logging, get_logger
from .train_data import reject_training_worlds
from .types import JsonMap

if TYPE_CHECKING:
    from .policy import AcquisitionPolicy


logger = get_logger(__name__)


def efficiency_curve(
    budgets: Sequence[int],
    trials: int = EVALUATION_TRIALS,
    seed: int = EVALUATION_SEED,
    simulations: int = DEFAULT_SIMULATIONS,
    problem: Problem | None = None,
    policy: AcquisitionPolicy | None = None,
    backend: str = "auto",
    particles: int = DEFAULT_PARTICLES,
    baseline: str = "greedy",
    baseline_budget: int | None = None,
    models: list[JsonMap] | None = None,
) -> JsonMap:
    """Compare reconstruction error across a shared acquisition-budget grid."""
    budgets = sorted(set(int(b) for b in budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("positive acquisition budgets required")
    baseline_budget = baseline_budget or budgets[-1]
    if baseline_budget not in budgets:
        raise ValueError("baseline budget must be in the evaluated budget grid")
    curves = {
        str(b): evaluate(seed, trials, b, simulations, policy, problem, backend, particles, models)
        for b in budgets
    }
    reference = curves[str(baseline_budget)]["summary"]
    if baseline not in reference:
        raise ValueError("unknown baseline method")
    target = reference[baseline]["mean_model_mse"]
    matches = {}
    for method in reference:
        matching = [
            b for b in budgets if curves[str(b)]["summary"][method]["mean_model_mse"] <= target
        ]
        minimum = min(matching) if matching else None
        matches[method] = {
            "smallest_matching_budget_on_grid": minimum,
            "measurement_reduction_fraction": 1 - minimum / baseline_budget if minimum else None,
        }
    return {
        "schema_version": 1,
        "kind": "simulated_acquisition_efficiency_curve",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "trials": trials,
        "baseline": baseline,
        "baseline_budget": baseline_budget,
        "matching_target_mean_model_mse": target,
        "matches": matches,
        "curves": curves,
        "comparison": "Mean-model-MSE threshold matching on the evaluated acquisition-budget grid.",
    }


def main() -> None:
    """Run the command-line workflow and write its result artifacts."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", default="1,2,3,4")
    parser.add_argument("--trials", type=int, default=EVALUATION_TRIALS)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backend", choices=("auto", "linear", "particle"), default="auto")
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--baseline", default="greedy")
    parser.add_argument("--baseline-budget", type=int)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--output", type=Path, default=settings.output_dir / "efficiency-curve.json"
    )
    args = parser.parse_args()
    problem = Problem.from_dict(json.loads(args.problem.read_text())) if args.problem else None
    if args.models and problem is None:
        parser.error("--models requires --problem")
    models = None
    if args.models:
        assert problem is not None
        models = read_evaluation_models(args.models, problem, args.split)
    policy = None
    if args.checkpoint:
        from .policy import load_policy

        policy, checkpoint = load_policy(args.checkpoint)
        if models is not None:
            reject_training_worlds(models, checkpoint)
    result = efficiency_curve(
        [int(x) for x in args.budgets.split(",")],
        args.trials,
        args.seed,
        args.simulations,
        problem,
        policy,
        args.backend,
        args.particles,
        args.baseline,
        args.baseline_budget,
        models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    logger.info(
        "workflow complete",
        extra={"fields": {"output": str(args.output), "matches": result["matches"]}},
    )


if __name__ == "__main__":
    main()
