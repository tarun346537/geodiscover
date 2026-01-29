"""Finite geological ensembles and Gaussian observation conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_PARTICLES, SEARCH_SEED
from .core import Problem
from .types import Forward, JsonMap, Matrix, Vector


@dataclass
class ParticleBelief:
    """Weighted geological models and their nonlinear forward responses."""

    models: np.ndarray
    predictions: np.ndarray
    weights: np.ndarray
    selected: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.models = np.asarray(self.models, float)
        self.predictions = np.asarray(self.predictions, float)
        self.weights = np.asarray(self.weights, float)
        if self.models.ndim != 2 or self.predictions.ndim != 2 or len(self.models) < 2:
            raise ValueError("at least two particle models and a prediction matrix required")
        if self.predictions.shape[0] != len(self.models) or self.weights.shape != (
            len(self.models),
        ):
            raise ValueError("particle dimensions disagree")
        if not all(np.isfinite(x).all() for x in (self.models, self.predictions, self.weights)):
            raise ValueError("particle inputs must be finite")
        if np.any(self.weights < 0) or self.weights.max() <= 0:
            raise ValueError("invalid particle weights")
        self.weights = self.weights / self.weights.max()
        self.weights = self.weights / self.weights.sum()

    def validate_for(self, problem: Problem) -> None:
        """Reject ensembles from another survey before indexing candidate arrays."""
        if (
            self.models.shape[1] != len(problem.prior_mean)
            or self.predictions.shape[1] != problem.n_actions
            or len(set(self.selected)) != len(self.selected)
            or any(a < 0 or a >= problem.n_actions for a in self.selected)
        ):
            raise ValueError(
                "particle ensemble dimensions or selected actions disagree with survey"
            )

    @property
    def effective_sample_size(self) -> float:
        """Report the effective number of particles under posterior weights."""
        return float(1 / np.sum(self.weights**2))

    @property
    def mean(self) -> Vector:
        """Return the posterior geological mean."""
        return self.weights @ self.models

    @property
    def covariance(self) -> Matrix:
        """Return the posterior geological covariance."""
        centered = self.models - self.mean
        return (centered.T * self.weights) @ centered

    def candidate_moments(self, problem: Problem | None = None) -> tuple[Vector, Vector]:
        """Return candidate response means and noiseless variances."""
        if problem is not None:
            self.validate_for(problem)
        mean = self.weights @ self.predictions
        variance = self.weights @ ((self.predictions - mean) ** 2)
        return mean, variance

    def information_gains(self, problem: Problem) -> Vector:
        # A Gaussian predictive-variance proxy for the greedy fallback/features.
        # The MCTS objective itself uses observation-conditioned particle risk.
        """Estimate candidate information gains with acquired actions masked."""
        _, variance = self.candidate_moments(problem)
        gains = 0.5 * np.log1p(variance / problem.noise_std**2)
        gains[list(self.selected)] = -np.inf
        return gains

    def update(self, problem: Problem, action: int, observation: float) -> ParticleBelief:
        """Condition the belief on one newly acquired observation."""
        self.validate_for(problem)
        if (
            action in self.selected
            or not 0 <= action < problem.n_actions
            or not np.isfinite(observation)
        ):
            raise ValueError("unavailable action or invalid observation")
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (observation - self.predictions[:, action]) / problem.noise_std[action]
        # Zero mass is outside this finite prior's support, even when an
        # observation strongly favors that particle's forward response.
        log_weights = np.full_like(self.weights, -np.inf)
        supported = self.weights > 0
        with np.errstate(over="ignore"):
            log_weights[supported] = (
                np.log(self.weights[supported]) - 0.5 * residual[supported] ** 2
            )
        if not np.isfinite(log_weights).any():
            raise ValueError(
                "observation likelihood underflowed. check observation units and noise"
            )
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        return ParticleBelief(
            self.models, self.predictions, weights, self.selected + (int(action),)
        )

    def sample_observation(self, problem: Problem, action: int, rng: np.random.Generator) -> float:
        """Sample a noisy acquisition from the posterior predictive distribution."""
        particle = int(rng.choice(len(self.models), p=self.weights))
        return float(self.predictions[particle, action] + rng.normal() * problem.noise_std[action])

    def to_ensemble(self) -> JsonMap:
        """Serialize the prior ensemble for teacher-data storage."""
        return {
            "models": self.models.tolist(),
            "predictions": self.predictions.tolist(),
            "weights": self.weights.tolist(),
        }


def particle_prior(
    problem: Problem, forward: Forward, particles: int = DEFAULT_PARTICLES, seed: int = SEARCH_SEED
) -> ParticleBelief:
    """Solve an independent geological ensemble with the configured forward model."""
    if particles < 2:
        raise ValueError("at least two prior particles required")
    rng = np.random.default_rng(seed)
    models = rng.multivariate_normal(problem.prior_mean, problem.prior_covariance, size=particles)
    predictions = np.asarray([forward(model) for model in models])
    if predictions.shape != (particles, problem.n_actions):
        raise ValueError("simulator prediction dimensions disagree with candidates")
    return ParticleBelief(models, predictions, np.ones(particles) / particles)


def particle_posterior(
    problem: Problem,
    ensemble: JsonMap,
    selected: Sequence[int] = (),
    observations: Sequence[float] = (),
) -> ParticleBelief:
    """Condition a saved ensemble on the acquired observations."""
    if len(selected) != len(observations):
        raise ValueError("one observation per selection required")
    belief = ParticleBelief(ensemble["models"], ensemble["predictions"], ensemble["weights"])
    belief.validate_for(problem)
    for action, observation in zip(selected, observations, strict=True):
        belief = belief.update(problem, action, observation)
    return belief
