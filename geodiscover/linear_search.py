"""Budgeted UCT search under a linear Gaussian observation model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_EXPLORATION, DEFAULT_SIMULATIONS, SEARCH_SEED
from .domain import Belief, Problem
from .types import JsonMap


@dataclass
class Node:
    """UCT statistics for a partially acquired linear survey."""

    belief: Belief
    depth: int
    action: int | None = None
    immediate_reward: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)


class MCTS:
    """UCT with random completion rollouts and a finite remaining budget.

    The reward is expected Gaussian entropy reduction. Under this particular
    linear belief model, future information gain is observation-independent.
    Each branch represents one candidate acquisition.
    """

    def __init__(
        self,
        simulations: int = DEFAULT_SIMULATIONS,
        exploration: float = DEFAULT_EXPLORATION,
        seed: int = SEARCH_SEED,
    ) -> None:
        if simulations < 1 or not np.isfinite(exploration) or exploration < 0:
            raise ValueError("invalid search parameters")
        self.simulations, self.exploration = int(simulations), float(exploration)
        self.rng = np.random.default_rng(seed)
        self.nodes_expanded = 0

    def choose(
        self, problem: Problem, belief: Belief, budget: int, deadline: float | None = None
    ) -> tuple[int, JsonMap]:
        """Select an available acquisition under the remaining budget."""
        available = problem.n_actions - len(belief.selected)
        if not 1 <= budget <= available:
            raise ValueError("budget must fit remaining candidates")
        root = Node(belief, 0)
        self.nodes_expanded = 0
        for _ in range(self.simulations):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("MCTS planning deadline exceeded")
            node, path = root, [root]
            while node.depth < budget:
                actions = [a for a in range(problem.n_actions) if a not in node.belief.selected]
                unexpanded = [a for a in actions if a not in node.children]
                if unexpanded:
                    action = int(self.rng.choice(unexpanded))
                    gain = float(node.belief.information_gains(problem)[action])
                    child = Node(
                        node.belief.expected_update(problem, action), node.depth + 1, action, gain
                    )
                    node.children[action] = child
                    self.nodes_expanded += 1
                    node = child
                    path.append(node)
                    break
                node = max(
                    node.children.values(),
                    key=lambda c: (
                        c.value_sum / c.visits
                        + self.exploration * math.sqrt(math.log(node.visits + 1) / c.visits)
                    ),
                )
                path.append(node)
            rollout_belief, value = node.belief, 0.0
            for _ in range(node.depth, budget):
                actions = [a for a in range(problem.n_actions) if a not in rollout_belief.selected]
                action = int(self.rng.choice(actions))
                value += float(rollout_belief.information_gains(problem)[action])
                rollout_belief = rollout_belief.expected_update(problem, action)
            # Each node stores return from its parent. The root stores full return.
            for item in reversed(path):
                value += item.immediate_reward
                item.visits += 1
                item.value_sum += value
        best = max(root.children.values(), key=lambda c: (c.visits, c.value_sum / c.visits))
        assert best.action is not None
        return int(best.action), {
            "simulations": root.visits,
            "nodes_expanded": self.nodes_expanded,
            "root_visits": {str(a): c.visits for a, c in root.children.items()},
        }
