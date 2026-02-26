"""Simulator-grounded REINFORCE with durable state and validation stopping."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .constants import (
    DEFAULT_BUDGET,
    DEFAULT_MIN_DELTA,
    DEFAULT_PARTICLES,
    DEFAULT_PATIENCE,
    DEFAULT_RL_EPISODES,
    GRADIENT_CLIP,
    REFINEMENT_SEED,
)
from .core import Problem
from .policy import AcquisitionPolicy
from .profiling import EpochProfiler, cache_delta
from .rl_env import Environment, EnvironmentFactory, validation_mse
from .tracking import RunTracker, load_checkpoint, restore_rng, rng_state, save_checkpoint
from .train_data import model_key
from .types import JsonMap, Vector


def _update(
    policy: AcquisitionPolicy,
    env: Environment,
    optimizer: torch.optim.Optimizer,
    baseline: float,
    entropy_weight: float,
) -> tuple[float, float, float]:
    rewards, log_probs, entropies = [], [], []
    while not env.done:
        distribution = policy.distribution(
            env.problem, env.belief, env.budget - len(env.belief.selected)
        )
        action = distribution.sample()
        reward, _ = env.step(int(action))
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        rewards.append(reward)
    returns = np.cumsum(rewards[::-1])[::-1].copy()
    advantages = torch.tensor(
        returns - baseline, dtype=torch.float32, device=next(policy.parameters()).device
    )
    loss = (
        -(torch.stack(log_probs) * advantages).mean()
        - entropy_weight * torch.stack(entropies).mean()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return float(sum(rewards)), float(loss.detach()), 0.9 * baseline + 0.1 * float(returns.mean())


def refine_rl(
    policy: AcquisitionPolicy,
    episodes: int = DEFAULT_RL_EPISODES,
    budget: int = DEFAULT_BUDGET,
    seed: int = REFINEMENT_SEED,
    learning_rate: float = 1e-4,
    entropy_weight: float = 0.01,
    problem: Problem | None = None,
    models: Sequence[Vector] | None = None,
    redis_url: str | None = None,
    backend: str = "auto",
    particles: int = DEFAULT_PARTICLES,
    cache_quantization: float | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    validation_models: Sequence[Vector] | None = None,
    eval_every: int = 8,
    tracker: RunTracker | None = None,
) -> list[JsonMap]:
    """Refine acquisition decisions using simulator observations and model-MSE rewards.

    Truth enters rewards only. Validation worlds are fixed independently of
    training draws, and their reconstruction MSE controls early stopping.
    """
    if (
        episodes < 0
        or budget < 1
        or learning_rate <= 0
        or entropy_weight < 0
        or patience < 1
        or eval_every < 1
        or min_delta < 0
    ):
        raise ValueError(
            "invalid RL settings. use positive budget, rate, patience and evaluation interval"
        )
    if (models is not None and not models) or (
        validation_models is not None and not validation_models
    ):
        raise ValueError("model splits must contain at least one model")
    digest = hashlib.sha256()
    if episodes == 0:
        return []
    if models is not None or validation_models is not None:
        if problem is None:
            raise ValueError("explicit model splits require a configured survey")
        known: set[bytes] = set()
        for index, split in enumerate((models, validation_models)):
            current = set()
            for model in split or ():
                value = np.asarray(model, float)
                if value.shape != problem.prior_mean.shape or not np.isfinite(value).all():
                    raise ValueError("invalid model dimensions or values")
                current.add(model_key(value))
            if index and known.intersection(current):
                raise ValueError("geological model leakage across training and validation")
            known.update(current)
    initial_weights = hashlib.sha256()
    for key, value in policy.state_dict().items():
        initial_weights.update(key.encode() + value.detach().cpu().numpy().tobytes())
    for index, split in enumerate((models, validation_models)):
        digest.update(f"split:{index}:count:{len(split) if split is not None else 0}".encode())
        for model in split or ():
            digest.update(np.asarray(model, dtype=float).tobytes())
    identity = {
        "problem": problem.to_dict() if problem else None,
        "models_sha256": digest.hexdigest(),
        "initial_policy_sha256": initial_weights.hexdigest(),
        "seed": seed,
        "budget": budget,
        "learning_rate": learning_rate,
        "entropy_weight": entropy_weight,
        "backend": backend,
        "particles": particles,
        "cache_quantization": cache_quantization,
        "hidden_size": policy.hidden_size,
        "eval_every": eval_every,
        "patience": patience,
        "min_delta": min_delta,
    }
    # JSON round-trip makes equality independent of NumPy scalar subclasses.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    factory = EnvironmentFactory(problem, backend, particles, seed, redis_url, cache_quantization)
    initialization = {
        "wall_seconds": time.perf_counter() - started,
        "forward_cache": factory.forward.stats() if factory.forward else None,
    }
    history: list[JsonMap] = []
    baseline, best_mse, stale, start_episode = 0.0, float("inf"), 0, 0
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ValueError("resume needs an existing RL checkpoint path")
        state = load_checkpoint(checkpoint_path, "refinement", identity)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], rng)
        history, baseline, best_state = state["history"], state["baseline"], state["best_state"]
        best_mse, stale, start_episode = state["best_mse"], state["stale"], state["completed"]
    policy.train()
    for episode in range(start_episode, episodes):
        if stale >= patience:
            break
        with EpochProfiler(next(policy.parameters()).device) as profile:
            scenario = factory.survey(episode)
            model = (
                models[episode % len(models)]
                if models is not None
                else rng.multivariate_normal(scenario.prior_mean, scenario.prior_covariance)
            )
            # Snapshot a fixed cache before environment construction so true-model
            # solves count in this episode. New proxy caches start with zero accesses.
            before = factory.forward.stats() if factory.forward else {}
            env, forward = factory.make(scenario, model, budget, episode)
            total_return, loss, baseline = _update(policy, env, optimizer, baseline, entropy_weight)
        row: JsonMap = {
            "episode": episode + 1,
            "return": total_return,
            "loss": loss,
            "final_model_mse": float(np.mean((env.belief.mean - model) ** 2)),
            "profile": profile.result,
            "forward_cache_delta": cache_delta(forward, before),
            "belief_backend": "particle" if factory.nonlinear else "linear",
            "initialization": initialization if episode == start_episode else None,
        }
        if (episode + 1) % eval_every == 0:
            measured = validation_mse(policy, factory, budget, validation_models)
            row["validation_model_mse"] = measured
            if measured < best_mse - min_delta:
                best_mse, stale = measured, 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
                }
            else:
                stale += 1
        history.append(row)
        save_checkpoint(
            checkpoint_path,
            {
                "phase": "refinement",
                "identity": identity,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(rng),
                "history": history,
                "baseline": baseline,
                "best_state": best_state,
                "best_mse": best_mse,
                "stale": stale,
                "completed": episode + 1,
            },
        )
        if tracker:
            tracker.log("refinement", episode + 1, row)
    if best_mse < float("inf"):
        policy.load_state_dict(best_state)
    policy.eval()
    if history:
        history[-1]["export_validation_model_mse"] = validation_mse(
            policy, factory, budget, validation_models
        )
        policy.eval()
    return history
