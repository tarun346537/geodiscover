"""Select the appropriate acquisition search for a posterior representation."""

from __future__ import annotations

from .core import MCTS, Belief, Problem
from .nonlinear import ParticleBelief, SimulatorMCTS
from .types import BeliefState, JsonMap


def plan_acquisition(
    problem: Problem,
    belief: BeliefState,
    budget: int,
    simulations: int,
    seed: int,
    deadline: float | None = None,
) -> tuple[int, JsonMap]:
    """Dispatch validated Gaussian and particle beliefs to their search backend."""
    if isinstance(belief, ParticleBelief):
        return SimulatorMCTS(simulations, seed=seed).choose(problem, belief, budget, deadline)
    if isinstance(belief, Belief):
        return MCTS(simulations, seed=seed).choose(problem, belief, budget, deadline)
    raise TypeError("planning requires a Gaussian or particle posterior")
