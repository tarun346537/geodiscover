"""Linear-Gaussian belief updates and budget-constrained acquisition search.

The belief planner never receives the true geology. The simulator supplies
observations only after an acquisition has been chosen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .constants import DEFAULT_CANDIDATES, DEFAULT_MODEL_DIM, SEARCH_SEED
from .types import BeliefState, Features, JsonMap, Matrix, Vector


@dataclass(init=False)
class Problem:
    """Candidate responses, noise and geological prior for a survey."""

    operator: Matrix
    noise_std: Vector
    prior_mean: Vector
    prior_covariance: Matrix
    offset: Vector
    metadata: JsonMap

    def __init__(
        self,
        operator: ArrayLike,
        noise_std: ArrayLike,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        offset: ArrayLike | None = None,
        metadata: JsonMap | None = None,
    ) -> None:
        self.operator = np.asarray(operator, dtype=float)
        self.noise_std = np.asarray(noise_std, dtype=float)
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_covariance = np.asarray(prior_covariance, dtype=float)
        self.offset = (
            np.zeros(self.operator.shape[0] if self.operator.ndim == 2 else 0)
            if offset is None
            else np.asarray(offset, dtype=float)
        )
        self.metadata = {} if metadata is None else metadata
        self.__post_init__()

    def __post_init__(self) -> None:
        self.operator = np.asarray(self.operator, dtype=float)
        if self.operator.ndim != 2 or min(self.operator.shape) < 1:
            raise ValueError("operator must be a nonempty matrix")
        n, d = self.operator.shape
        self.noise_std = np.broadcast_to(np.asarray(self.noise_std, float), (n,)).copy()
        self.prior_mean = np.asarray(self.prior_mean, float)
        self.prior_covariance = np.asarray(self.prior_covariance, float)
        self.offset = np.zeros(n) if self.offset is None else np.asarray(self.offset, float)
        if self.prior_mean.shape != (d,) or self.prior_covariance.shape != (d, d):
            raise ValueError("prior dimensions do not match operator")
        if self.offset.shape != (n,) or np.any(self.noise_std <= 0):
            raise ValueError("offset shape or noise standard deviations invalid")
        for value in (
            self.operator,
            self.noise_std,
            self.prior_mean,
            self.prior_covariance,
            self.offset,
        ):
            if not np.isfinite(value).all():
                raise ValueError("all numerical inputs must be finite")
        if not np.allclose(self.prior_covariance, self.prior_covariance.T, atol=1e-10):
            raise ValueError("prior covariance must be symmetric")
        np.linalg.cholesky(self.prior_covariance)

    @property
    def n_actions(self) -> int:
        """Return the number of candidate acquisitions."""
        return self.operator.shape[0]

    def to_dict(self) -> JsonMap:
        """Serialize the survey to JSON-compatible values."""
        return {
            "operator": self.operator.tolist(),
            "noise_std": self.noise_std.tolist(),
            "prior_mean": self.prior_mean.tolist(),
            "prior_covariance": self.prior_covariance.tolist(),
            "offset": self.offset.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: JsonMap) -> Problem:
        """Load and validate a survey description."""
        return cls(
            **{k: value[k] for k in ("operator", "noise_std", "prior_mean", "prior_covariance")},
            offset=value.get("offset"),
            metadata=value.get("metadata", {}),
        )


def synthetic_problem(
    seed: int = SEARCH_SEED,
    n_actions: int = DEFAULT_CANDIDATES,
    dimensions: int = DEFAULT_MODEL_DIM,
) -> Problem:
    """Create a linear potential-field fixture for numerical checks."""
    if n_actions < 2 or dimensions < 1:
        raise ValueError("at least two actions and one model dimension required")
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, n_actions)
    cells = np.linspace(-1, 1, dimensions)
    depth = rng.uniform(0.25, 0.8, dimensions)
    operator = depth / ((x[:, None] - cells) ** 2 + depth**2) ** 1.5
    operator /= np.linalg.norm(operator, axis=1, keepdims=True)
    scales = rng.uniform(0.6, 1.6, dimensions)
    return Problem(
        operator,
        rng.uniform(0.08, 0.2, n_actions),
        np.zeros(dimensions),
        np.diag(scales**2),
        metadata={"kind": "synthetic_potential_field_proxy", "seed": seed},
    )


@dataclass
class Belief:
    """Gaussian posterior over geological parameters."""

    mean: np.ndarray
    covariance: np.ndarray
    selected: tuple[int, ...] = ()

    @classmethod
    def prior(cls, problem: Problem) -> Belief:
        """Start from the configured Gaussian prior."""
        return cls(problem.prior_mean.copy(), problem.prior_covariance.copy())

    def update(self, problem: Problem, action: int, observation: float) -> Belief:
        """Condition the belief on one newly acquired observation."""
        if action in self.selected or not 0 <= action < problem.n_actions:
            raise ValueError("action is unavailable")
        if not np.isfinite(observation):
            raise ValueError("observation must be finite")
        a = problem.operator[action]
        ca = self.covariance @ a
        variance = problem.noise_std[action] ** 2 + a @ ca
        residual = observation - problem.offset[action] - a @ self.mean
        covariance = self.covariance - np.outer(ca, ca) / variance
        covariance = (covariance + covariance.T) / 2
        return Belief(self.mean + ca * residual / variance, covariance, self.selected + (action,))

    def information_gains(self, problem: Problem) -> Vector:
        """Estimate candidate information gains with acquired actions masked."""
        variance = np.einsum("ij,jk,ik->i", problem.operator, self.covariance, problem.operator)
        gains = 0.5 * np.log1p(np.maximum(variance, 0) / problem.noise_std**2)
        gains[list(self.selected)] = -np.inf
        return gains

    def candidate_moments(self, problem: Problem | None = None) -> tuple[Vector, Vector]:
        """Return Gaussian response means and noiseless variances."""
        if problem is None:
            raise ValueError("supply the survey to compute Gaussian predictive moments")
        return problem.operator @ self.mean + problem.offset, np.einsum(
            "ij,jk,ik->i", problem.operator, self.covariance, problem.operator
        )

    def expected_update(self, problem: Problem, action: int) -> Belief:
        """Apply a covariance update at the predictive observation mean."""
        return self.update(
            problem, action, float(problem.operator[action] @ self.mean + problem.offset[action])
        )


def posterior(
    problem: Problem, selected: Sequence[int] = (), observations: Sequence[float] = ()
) -> Belief:
    """Condition the Gaussian prior on a validated acquisition history."""
    if len(selected) != len(observations):
        raise ValueError("one observation per selected action required")
    belief = Belief.prior(problem)
    for action, observation in zip(selected, observations, strict=True):
        belief = belief.update(problem, int(action), float(observation))
    return belief


def greedy_action(problem: Problem, belief: BeliefState) -> int:
    """Choose the available candidate with greatest information gain."""
    gains = belief.information_gains(problem)
    if not np.isfinite(gains).any():
        raise ValueError("all measurements already acquired")
    return int(np.argmax(gains))


def acquisition_features(problem: Problem, belief: BeliefState, remaining: int) -> Features:
    """Six per-candidate features shared across different survey sizes."""
    a = problem.operator
    prediction_mean, variance = belief.candidate_moments(problem)
    variance = np.maximum(variance, 0)
    gains = 0.5 * np.log1p(variance / problem.noise_std**2)
    selected = np.zeros(problem.n_actions)
    selected[list(belief.selected)] = 1
    return np.column_stack(
        (
            np.tanh(prediction_mean / problem.noise_std),
            np.log1p(variance / problem.noise_std**2),
            gains,
            selected,
            np.log1p(np.linalg.norm(a, axis=1)),
            np.full(problem.n_actions, remaining / problem.n_actions),
        )
    ).astype(np.float32)


class SurveyEnvironment:
    """Known truth is used only to simulate observations and training rewards."""

    def __init__(
        self,
        problem: Problem,
        model: np.ndarray,
        budget: int,
        seed: int = SEARCH_SEED,
        forward: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not 1 <= budget <= problem.n_actions:
            raise ValueError("invalid acquisition budget")
        self.problem, self.budget = problem, budget
        self.model = np.asarray(model, float)
        if self.model.shape != problem.prior_mean.shape or not np.isfinite(self.model).all():
            raise ValueError("invalid true model")
        rng = np.random.default_rng(seed)
        values = forward(self.model) if forward else problem.operator @ self.model + problem.offset
        self.observations = (
            np.asarray(values, float) + rng.normal(size=problem.n_actions) * problem.noise_std
        )
        if (
            self.observations.shape != (problem.n_actions,)
            or not np.isfinite(self.observations).all()
        ):
            raise ValueError("forward solver returned invalid observations")
        self.belief = Belief.prior(problem)
        self.used_observations: list[float] = []

    @property
    def done(self) -> bool:
        """Report whether the acquisition budget is exhausted."""
        return len(self.belief.selected) >= self.budget

    def step(self, action: int) -> tuple[float, bool]:
        """Acquire one observation and return reconstruction improvement."""
        if self.done:
            raise ValueError("acquisition budget exhausted")
        before = np.mean((self.belief.mean - self.model) ** 2)
        observation = float(self.observations[action])
        self.belief = self.belief.update(self.problem, action, observation)
        self.used_observations.append(observation)
        after = np.mean((self.belief.mean - self.model) ** 2)
        return float(before - after), self.done
