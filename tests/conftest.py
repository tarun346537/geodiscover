"""Shared survey and acquisition fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from geodiscover.core import Problem, synthetic_problem
from geodiscover.plans import generate
from geodiscover.types import JsonMap

if TYPE_CHECKING:
    from geodiscover.policy import AcquisitionPolicy


@pytest.fixture
def survey() -> Problem:
    """Return a small deterministic survey for numerical checks."""
    return synthetic_problem(seed=17)


@pytest.fixture
def teacher_rows() -> list[JsonMap]:
    """Keep complete scenarios in distinct imitation splits."""
    return generate(episodes=6, budget=2, simulations=8, seed=31)


@pytest.fixture
def tiny_policy() -> AcquisitionPolicy:
    """Initialize a CPU policy small enough for exact resume comparisons."""
    torch = pytest.importorskip("torch")
    from geodiscover.policy import AcquisitionPolicy

    torch.set_num_threads(1)
    torch.manual_seed(17)
    return AcquisitionPolicy(hidden_size=8)
