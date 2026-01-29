"""Particle-belief MCTS with simulator-backed observation branches.

Forward responses are solved once per prior particle and cached by the supplied
solver. Search branches condition that ensemble on possible noisy observations.
This avoids substituting a fixed Jacobian for the nonlinear simulator in search.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    DEFAULT_EXPLORATION,
    DEFAULT_SIMULATIONS,
    OBSERVATION_BIN_STD,
    SEARCH_SEED,
)
from .core import Problem
from .particle_belief import ParticleBelief as ParticleBelief
from .particle_belief import particle_posterior as particle_posterior
from .particle_belief import particle_prior as particle_prior
from .types import Forward, JsonMap, Vector


@dataclass
class ActionNode:
    """Search statistics and observation branches for one acquisition."""

    visits: int = 0
    value_sum: float = 0.0
    observations: dict[int, BeliefNode] = field(default_factory=dict)


@dataclass
class BeliefNode:
    """A particle posterior and its available action branches."""

    belief: ParticleBelief
    visits: int = 0
    actions: dict[int, ActionNode] = field(default_factory=dict)


class SimulatorMCTS:
    """UCT over actions and discretized noisy simulator observations.

    Each simulation samples from the current posterior predictive distribution,
    updates the particle belief, then rolls out to the remaining budget. The
    utility is reduction in posterior model variance. Observation bins have
    width ``observation_bin_std * measurement_noise``. Bayes conditioning uses
    each bin center. Bin width and finite-ensemble approximation are explicit.
    """

    def __init__(
        self,
        simulations: int = DEFAULT_SIMULATIONS,
        exploration: float = DEFAULT_EXPLORATION,
        seed: int = SEARCH_SEED,
        observation_bin_std: float = OBSERVATION_BIN_STD,
    ) -> None:
        if (
            simulations < 1
            or not np.isfinite(exploration)
            or exploration < 0
            or not np.isfinite(observation_bin_std)
            or observation_bin_std <= 0
        ):
            raise ValueError("invalid nonlinear search parameters")
        self.simulations, self.exploration = int(simulations), float(exploration)
        self.bin_width = float(observation_bin_std)
        self.rng = np.random.default_rng(seed)
        self.nodes_expanded = 0

    def choose(
        self, problem: Problem, belief: ParticleBelief, budget: int, deadline: float | None = None
    ) -> tuple[int, JsonMap]:
        """Select an available acquisition under the remaining budget."""
        if not isinstance(belief, ParticleBelief):
            raise TypeError("simulator MCTS requires a ParticleBelief")
        belief.validate_for(problem)
        if not 1 <= budget <= problem.n_actions - len(belief.selected):
            raise ValueError("budget must fit remaining measurements")
        root = BeliefNode(belief)
        initial_risk = max(float(np.trace(belief.covariance)), 1e-12)
        self.nodes_expanded = 0
        for _ in range(self.simulations):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("simulator planning deadline exceeded")
            node, path = root, []
            for _depth in range(budget):
                available = [a for a in range(problem.n_actions) if a not in node.belief.selected]
                unseen = [a for a in available if a not in node.actions]
                if unseen:
                    action = int(self.rng.choice(unseen))
                    node.actions[action] = ActionNode()
                else:
                    action = max(
                        available,
                        key=lambda a: (
                            node.actions[a].value_sum / node.actions[a].visits
                            + self.exploration
                            * math.sqrt(math.log(node.visits + 1) / node.actions[a].visits)
                        ),
                    )
                action_node = node.actions[action]
                observation = node.belief.sample_observation(problem, action, self.rng)
                width = self.bin_width * problem.noise_std[action]
                observation_bin = int(np.rint(observation / width))
                new = observation_bin not in action_node.observations
                if new:
                    child_belief = node.belief.update(problem, action, observation_bin * width)
                    action_node.observations[observation_bin] = BeliefNode(child_belief)
                    self.nodes_expanded += 1
                path.append((node, action_node))
                node = action_node.observations[observation_bin]
                if new:
                    break
            rollout = node.belief
            for _ in range(len(rollout.selected) - len(belief.selected), budget):
                available = [a for a in range(problem.n_actions) if a not in rollout.selected]
                action = int(self.rng.choice(available))
                observation = rollout.sample_observation(problem, action, self.rng)
                rollout = rollout.update(problem, action, observation)
            utility = (initial_risk - float(np.trace(rollout.covariance))) / initial_risk
            node.visits += 1
            for parent, action_node in path:
                parent.visits += 1
                action_node.visits += 1
                action_node.value_sum += utility
        action = max(
            root.actions,
            key=lambda a: (
                root.actions[a].visits,
                root.actions[a].value_sum / root.actions[a].visits,
            ),
        )
        return int(action), {
            "backend": "simulator_particle_mcts",
            "simulations": self.simulations,
            "nodes_expanded": self.nodes_expanded,
            "root_visits": {str(a): n.visits for a, n in root.actions.items()},
            "particles": len(belief.models),
            "root_effective_sample_size": belief.effective_sample_size,
            "observation_bin_std": self.bin_width,
        }


class ParticleEnvironment:
    """Provide noisy simulator observations and reconstruction rewards."""

    def __init__(
        self,
        problem: Problem,
        model: Vector,
        budget: int,
        seed: int,
        forward: Forward,
        prior: ParticleBelief,
    ) -> None:
        if not 1 <= budget <= problem.n_actions:
            raise ValueError("invalid acquisition budget")
        self.problem, self.budget = problem, budget
        self.model = np.asarray(model, float)
        if self.model.shape != problem.prior_mean.shape or not np.isfinite(self.model).all():
            raise ValueError("invalid true model")
        self.observations = (
            np.asarray(forward(self.model), float)
            + np.random.default_rng(seed).normal(size=problem.n_actions) * problem.noise_std
        )
        if (
            self.observations.shape != (problem.n_actions,)
            or not np.isfinite(self.observations).all()
        ):
            raise ValueError("invalid simulator response")
        self.belief = ParticleBelief(prior.models, prior.predictions, prior.weights)
        self.used_observations: list[float] = []

    @property
    def done(self) -> bool:
        """Report whether the acquisition budget is exhausted."""
        return len(self.belief.selected) >= self.budget

    def step(self, action: int) -> tuple[float, bool]:
        """Acquire one observation and return reconstruction improvement."""
        if self.done:
            raise ValueError("acquisition budget exhausted")
        before = float(np.mean((self.belief.mean - self.model) ** 2))
        observation = float(self.observations[action])
        self.belief = self.belief.update(self.problem, action, observation)
        self.used_observations.append(observation)
        return before - float(np.mean((self.belief.mean - self.model) ** 2)), self.done
