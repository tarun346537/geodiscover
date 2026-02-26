"""Survey environments and held-out checks used during policy refinement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .cache import ForwardCache
from .core import Problem, SurveyEnvironment, synthetic_problem
from .nonlinear import ParticleBelief, ParticleEnvironment, particle_prior
from .policy import AcquisitionPolicy
from .simpeg_adapter import forward_for_problem
from .types import Vector

type Environment = SurveyEnvironment | ParticleEnvironment


class EnvironmentFactory:
    """Reuse a fixed survey's solves without exposing evaluation truth to training."""

    def __init__(
        self,
        problem: Problem | None,
        backend: str,
        particles: int,
        seed: int,
        redis_url: str | None,
        quantization: float | None,
    ) -> None:
        if backend not in ("auto", "linear", "particle"):
            raise ValueError("choose auto, linear or particle for the belief backend")
        self.problem, self.particles, self.seed = problem, particles, seed
        self.redis_url, self.quantization = redis_url, quantization
        self.nonlinear = backend == "particle" or bool(
            backend == "auto" and problem and problem.metadata.get("simpeg_dc")
        )
        self.forward = (
            forward_for_problem(problem, redis_url, quantization) if problem is not None else None
        )
        self.prior = (
            particle_prior(problem, self.forward, particles, seed + 3)
            if self.nonlinear and problem is not None and self.forward is not None
            else None
        )

    def survey(self, episode: int, validation: bool = False) -> Problem:
        """Return the fixed survey or a split-specific local survey realization."""
        return self.problem or synthetic_problem(
            self.seed + episode * 7919 + (10000000 if validation else 0)
        )

    def make(
        self, problem: Problem, model: Vector, budget: int, episode: int, validation: bool = False
    ) -> tuple[Environment, ForwardCache]:
        """Build a fresh acquisition belief and the trusted observation provider."""
        forward = self.forward or forward_for_problem(problem, self.redis_url, self.quantization)
        prior: ParticleBelief | None = self.prior
        if prior is None and self.nonlinear:
            prior = particle_prior(problem, forward, self.particles, self.seed + episode + 3)
        seed = self.seed + episode + (10000000 if validation else 0)
        env: Environment = (
            ParticleEnvironment(problem, model, budget, seed, forward, prior)
            if prior
            else SurveyEnvironment(problem, model, budget, seed, forward)
        )
        return env, forward


def validation_mse(
    policy: AcquisitionPolicy,
    factory: EnvironmentFactory,
    budget: int,
    models: Sequence[Vector] | None = None,
) -> float:
    """Evaluate fixed, independent validation worlds without policy gradients."""
    rng = np.random.default_rng(factory.seed + 20000000)
    values = []
    policy.eval()
    with torch.no_grad():
        for episode in range(len(models) if models is not None else 3):
            problem = factory.survey(episode, validation=True)
            model = (
                models[episode]
                if models is not None
                else rng.multivariate_normal(problem.prior_mean, problem.prior_covariance)
            )
            env, _ = factory.make(problem, model, budget, episode, validation=True)
            while not env.done:
                env.step(policy.choose(problem, env.belief, budget - len(env.belief.selected)))
            values.append(float(np.mean((env.belief.mean - model) ** 2)))
    policy.train()
    return float(np.mean(values))
