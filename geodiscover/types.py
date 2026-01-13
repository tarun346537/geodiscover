"""Numerical and service contracts shared by planning backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import Problem


type Vector = NDArray[np.float64]
type Matrix = NDArray[np.float64]
type Features = NDArray[np.float32]
type JsonMap = dict[str, Any]
type Forward = Callable[[Vector], Vector]


class BeliefState(Protocol):
    """Posterior summaries consumed by policies and acquisition planners."""

    @property
    def mean(self) -> Vector:
        """Return the geological posterior mean."""
        ...

    @property
    def covariance(self) -> Matrix:
        """Return the geological posterior covariance."""
        ...

    @property
    def selected(self) -> tuple[int, ...]:
        """Return already acquired candidate indices."""
        ...

    def candidate_moments(self, problem: Problem | None = None) -> tuple[Vector, Vector]:
        """Return each candidate's predictive mean and noiseless variance."""
        ...

    def information_gains(self, problem: Problem) -> Vector:
        """Estimate acquisition information for the greedy fallback."""
        ...


type Planner = Callable[[Problem, BeliefState, int, int, int, float], tuple[int, JsonMap]]
