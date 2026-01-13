"""Evaluate equal-budget random, greedy, MCTS, and optional learned acquisition."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import Settings
from .constants import (
    DEFAULT_BUDGET,
    DEFAULT_PARTICLES,
    DEFAULT_SIMULATIONS,
    EVALUATION_SEED,
    EVALUATION_TRIALS,
)
from .core import Problem, SurveyEnvironment, greedy_action, synthetic_problem
from .logging import configure_logging, get_logger
from .nonlinear import ParticleEnvironment, particle_prior
from .planning import plan_acquisition
from .simpeg_adapter import forward_for_problem
from .train_data import read_model_records, reject_training_worlds
from .types import JsonMap

if TYPE_CHECKING:
    from .policy import AcquisitionPolicy


logger = get_logger(__name__)


def evaluate(
    seed: int = EVALUATION_SEED,
    trials: int = EVALUATION_TRIALS,
    budget: int = DEFAULT_BUDGET,
    simulations: int = DEFAULT_SIMULATIONS,
    policy: AcquisitionPolicy | None = None,
    problem: Problem | None = None,
    backend: str = "auto",
    particles: int = DEFAULT_PARTICLES,
    models: list[JsonMap] | None = None,
) -> JsonMap:
    """Compare acquisition strategies on the same held-out worlds and noise."""
    if trials < 1:
        raise ValueError("at least one evaluation trial required")
    methods = ["random", "greedy", "mcts"] + (["policy"] if policy else [])
    if backend not in ("auto", "linear", "particle"):
        raise ValueError("unknown belief backend")
    if models is not None and (problem is None or len(models) < trials):
        raise ValueError("held-out models require a configured problem and at least trials records")
    records: list[JsonMap] = []
    timings: dict[str, list[float]] = {k: [] for k in methods}
    for trial in range(trials):
        scenario = problem if problem else synthetic_problem(seed + trial * 7919)
        rng = np.random.default_rng(seed + trial)
        truth = (
            np.asarray(models[trial]["model"], float)
            if models is not None
            else rng.multivariate_normal(scenario.prior_mean, scenario.prior_covariance)
        )
        forward = forward_for_problem(scenario)
        nonlinear = backend == "particle" or (
            backend == "auto" and scenario.metadata.get("simpeg_dc")
        )
        setup_started = time.perf_counter()
        prior = (
            particle_prior(scenario, forward, particles, seed + trial + 3) if nonlinear else None
        )
        setup_seconds = time.perf_counter() - setup_started
        # All methods see exactly the same simulated full survey and noise.
        # The cache is shared across methods, but each belief starts at prior.
        for method in methods:
            env = (
                ParticleEnvironment(scenario, truth, budget, seed + trial + 1, forward, prior)
                if prior
                else SurveyEnvironment(scenario, truth, budget, seed + trial + 1, forward)
            )
            nodes = 0
            while not env.done:
                before = time.perf_counter()
                remaining = budget - len(env.belief.selected)
                if method == "random":
                    candidates = [
                        a for a in range(scenario.n_actions) if a not in env.belief.selected
                    ]
                    action = int(rng.choice(candidates))
                elif method == "greedy":
                    action = greedy_action(scenario, env.belief)
                elif method == "policy":
                    assert policy is not None
                    action = policy.choose(scenario, env.belief, remaining)
                else:
                    action, statistics = plan_acquisition(
                        scenario, env.belief, remaining, simulations, seed + trial + remaining
                    )
                    nodes += statistics["nodes_expanded"]
                timings[method].append(time.perf_counter() - before)
                env.step(action)
            records.append(
                {
                    "trial": trial,
                    "model_id": models[trial]["model_id"]
                    if models is not None
                    else f"sample-{seed + trial}",
                    "method": method,
                    "budget": budget,
                    "selected": list(env.belief.selected),
                    "model_mse": float(np.mean((env.belief.mean - truth) ** 2)),
                    "survey_prediction_mse": float(
                        np.mean((forward(env.belief.mean) - forward(truth)) ** 2)
                    ),
                    "posterior_trace": float(np.trace(env.belief.covariance)),
                    "belief_backend": "particle" if prior else "linear",
                    "shared_simulator_setup_seconds": setup_seconds,
                    "nodes_expanded": nodes,
                    "cache": forward.stats(),
                }
            )
    summary = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        summary[method] = {
            "mean_model_mse": float(np.mean([r["model_mse"] for r in rows])),
            "mean_survey_prediction_mse": float(
                np.mean([r["survey_prediction_mse"] for r in rows])
            ),
            "decision_seconds_p50": float(np.quantile(timings[method], 0.5)),
            "decision_seconds_p99": float(np.quantile(timings[method], 0.99)),
            "total_nodes_expanded": sum(r["nodes_expanded"] for r in rows),
        }
    return {
        "schema_version": 1,
        "provenance": {
            "kind": "synthetic_demo" if not problem else "simulator_evaluation",
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "seed": seed,
        "trials": trials,
        "budget": budget,
        "simulations_per_decision": simulations,
        "evaluation_models": "held_out_input" if models is not None else "sampled_prior",
        "summary": summary,
        "records": records,
        "workload": {
            "belief_models": "Finite particle ensemble with discretized observation branches, or a linear Gaussian posterior.",
            "observations": "Simulator forward responses with configured observation noise.",
            "timing": "Sequential acquisition-decision calls.",
        },
    }


def read_evaluation_models(
    path: str | Path, problem: Problem, split: str = "test"
) -> list[JsonMap]:
    """Read an explicit validation or test split, excluding training models."""
    if split not in ("validation", "test"):
        raise ValueError("evaluation requires the validation or test split")
    rows = [row for row in read_model_records(path, problem) if row["split"] == split]
    if not rows:
        raise ValueError("no held-out models for selected split")
    return rows


def main() -> None:
    """Run the command-line workflow and write its result artifacts."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=settings.output_dir / "demo.json")
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--trials", type=int, default=EVALUATION_TRIALS)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--backend", choices=("auto", "linear", "particle"), default="auto")
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    policy = None
    policy_metadata: JsonMap = {}
    if args.checkpoint:
        from .policy import load_policy

        policy, policy_metadata = load_policy(args.checkpoint)
    problem = Problem.from_dict(json.loads(args.problem.read_text())) if args.problem else None
    if args.models and problem is None:
        parser.error("--models requires --problem")
    models = None
    if args.models:
        assert problem is not None
        models = read_evaluation_models(args.models, problem, args.split)
        reject_training_worlds(models, policy_metadata)
    result = evaluate(
        args.seed,
        args.trials,
        args.budget,
        args.simulations,
        policy,
        problem,
        args.backend,
        args.particles,
        models,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info(
        "workflow complete",
        extra={
            "fields": {
                "output": str(args.output),
                "provenance": result["provenance"],
                "summary": result["summary"],
            }
        },
    )


if __name__ == "__main__":
    main()
