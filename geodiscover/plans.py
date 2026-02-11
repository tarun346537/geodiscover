"""Stream MCTS teacher plans locally or through bounded Ray episode tasks."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .constants import (
    DEFAULT_BUDGET,
    DEFAULT_PARTICLES,
    DEFAULT_SEED,
    DEFAULT_SIMULATIONS,
    MAX_PENDING_JOBS,
    TEACHER_EPISODES,
)
from .core import Problem, SurveyEnvironment, synthetic_problem
from .logging import configure_logging, get_logger
from .nonlinear import ParticleEnvironment, particle_prior
from .plan_export import write_plans as write_plans
from .planning import plan_acquisition
from .simpeg_adapter import forward_for_problem
from .types import JsonMap

logger = get_logger(__name__)


def teacher_episode(
    episode: int,
    seed: int,
    budget: int,
    simulations: int,
    problem_data: dict | None = None,
    backend: str = "auto",
    particles: int = DEFAULT_PARTICLES,
    max_nodes: int | None = None,
) -> list[JsonMap]:
    """Collect a budgeted sequence of search-guided teacher actions."""
    scenario_seed = seed + episode * 7919
    problem = Problem.from_dict(problem_data) if problem_data else synthetic_problem(scenario_seed)
    if max_nodes is not None and max_nodes < budget:
        raise ValueError("node allowance must permit one expansion per teacher decision")
    backend = (
        ("particle" if problem.metadata.get("simpeg_dc") else "linear")
        if backend == "auto"
        else backend
    )
    if backend not in ("linear", "particle"):
        raise ValueError("unknown belief backend")
    started = time.perf_counter()
    rng = np.random.default_rng(scenario_seed + 1)
    model = rng.multivariate_normal(problem.prior_mean, problem.prior_covariance)
    forward = forward_for_problem(problem)
    prior = (
        particle_prior(problem, forward, particles, scenario_seed + 3)
        if backend == "particle"
        else None
    )
    ensemble = prior.to_ensemble() if prior else None
    env = (
        ParticleEnvironment(problem, model, budget, scenario_seed + 2, forward, prior)
        if prior
        else SurveyEnvironment(problem, model, budget, scenario_seed + 2, forward)
    )
    records, used_nodes = [], 0
    previous_cache = {"hits": 0, "misses": 0, "backend_failures": 0}
    while not env.done:
        remaining = budget - len(env.belief.selected)
        allowed_simulations = simulations
        if max_nodes is not None:
            allowed_simulations = min(simulations, max_nodes - used_nodes - (remaining - 1))
        action, statistics = plan_acquisition(
            problem, env.belief, remaining, allowed_simulations, scenario_seed + remaining
        )
        used_nodes += statistics["nodes_expanded"]
        cache = forward.stats()
        statistics["forward_cache_delta"] = {
            k: cache[k] - previous_cache[k] for k in previous_cache
        }
        previous_cache = {k: cache[k] for k in previous_cache}
        statistics["worker_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        row = {
            "schema_version": 1,
            "scenario_id": f"scenario-{scenario_seed}",
            "split": "validation" if episode % 5 == 0 else "train",
            "belief_backend": backend,
            "provenance": {
                "kind": "synthetic_demo" if not problem_data else "simulator_generated",
                "scenario_seed": scenario_seed,
            },
            "problem": problem.to_dict(),
            "selected": list(env.belief.selected),
            "observations": env.used_observations.copy(),
            "remaining_budget": remaining,
            "teacher_action": action,
            "search": statistics,
        }
        if prior is not None:
            row["particle_ensemble"] = ensemble
        records.append(row)
        env.step(action)
    return records


def episode_settings(
    episodes: int, budget: int, target_plans: int | None = None, max_nodes: int | None = None
) -> list[tuple[int, int | None]]:
    """Allocate episode lengths and a global node allowance."""
    if episodes < 2 or budget < 1 or (target_plans is not None and target_plans < 2):
        raise ValueError("at least two episodes/plans and a positive budget required")
    if target_plans is not None:
        episodes = math.ceil(target_plans / budget)
        if episodes < 2:
            raise ValueError("target must span two scenarios for train/validation splitting")
        budgets = [min(budget, target_plans - i * budget) for i in range(episodes)]
    else:
        budgets = [budget] * episodes
    allowances: list[int | None] = [None] * episodes
    if max_nodes is not None:
        if max_nodes < sum(budgets):
            raise ValueError("max_nodes must allow at least one expansion per plan")
        extra = max_nodes - sum(budgets)
        allowances = [
            b + extra // episodes + int(i < extra % episodes) for i, b in enumerate(budgets)
        ]
    return list(zip(budgets, allowances, strict=True))


def iter_episodes(
    episodes: int = TEACHER_EPISODES,
    seed: int = DEFAULT_SEED,
    budget: int = DEFAULT_BUDGET,
    simulations: int = DEFAULT_SIMULATIONS,
    problem_data: JsonMap | None = None,
    use_ray: bool = False,
    ray_address: str | None = None,
    backend: str = "auto",
    particles: int = DEFAULT_PARTICLES,
    target_plans: int | None = None,
    max_nodes: int | None = None,
    max_pending: int = MAX_PENDING_JOBS,
) -> Iterator[tuple[int, list[JsonMap]]]:
    """Yield teacher episodes locally or from bounded Ray tasks."""
    settings = episode_settings(episodes, budget, target_plans, max_nodes)
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    if use_ray:
        import ray

        owns_runtime = not ray.is_initialized()
        if owns_runtime:
            ray.init(
                address=ray_address,
                include_dashboard=False,
                runtime_env={"py_modules": [str(Path(__file__).parent)]},
            )
        pending: dict[ray.ObjectRef, int] = {}
        next_episode = 0
        try:
            task = ray.remote(num_cpus=1)(teacher_episode)
            while next_episode < len(settings) or pending:
                while next_episode < len(settings) and len(pending) < max_pending:
                    b, allowance = settings[next_episode]
                    ref = task.remote(
                        next_episode,
                        seed,
                        b,
                        simulations,
                        problem_data,
                        backend,
                        particles,
                        allowance,
                    )
                    pending[ref] = next_episode
                    next_episode += 1
                ready, _ = ray.wait(list(pending), num_returns=1)
                ref = ready[0]
                episode = pending.pop(ref)
                yield episode, ray.get(ref)
        finally:
            for ref in pending:
                ray.cancel(ref, force=True)
            if owns_runtime:
                ray.shutdown()
    else:
        for episode, (b, allowance) in enumerate(settings):
            yield (
                episode,
                teacher_episode(
                    episode, seed, b, simulations, problem_data, backend, particles, allowance
                ),
            )


def generate(
    episodes: int = TEACHER_EPISODES,
    seed: int = DEFAULT_SEED,
    budget: int = DEFAULT_BUDGET,
    simulations: int = DEFAULT_SIMULATIONS,
    problem_data: JsonMap | None = None,
    use_ray: bool = False,
    ray_address: str | None = None,
    **kwargs: Any,
) -> list[JsonMap]:
    """In-memory convenience API for small jobs. CLI streams large exports."""
    return [
        row
        for _, batch in iter_episodes(
            episodes, seed, budget, simulations, problem_data, use_ray, ray_address, **kwargs
        )
        for row in batch
    ]


def main() -> None:
    """Run the command-line workflow and write its result artifacts."""
    settings = Settings.load()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--episodes", type=int, default=TEACHER_EPISODES)
    parser.add_argument(
        "--target-plans", type=int, help="exact number of teacher action-state records"
    )
    parser.add_argument(
        "--max-nodes", type=int, help="global upper bound on actual tree-node expansions"
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--backend", choices=("auto", "linear", "particle"), default="auto")
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--ray-address")
    parser.add_argument("--max-pending", type=int, default=MAX_PENDING_JOBS)
    args = parser.parse_args()
    problem = json.loads(args.problem.read_text()) if args.problem else None
    batches = iter_episodes(
        args.episodes,
        args.seed,
        args.budget,
        args.simulations,
        problem,
        args.ray,
        args.ray_address,
        args.backend,
        args.particles,
        args.target_plans,
        args.max_nodes,
        args.max_pending,
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = write_plans(args.output, batches, config)
    logger.info("workflow complete", extra={"fields": {"output": str(args.output), **report}})


if __name__ == "__main__":
    main()
