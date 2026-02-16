"""Shared per-acquisition PyTorch scorer and portable checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from .constants import FEATURE_DIM, HIDDEN_SIZE
from .core import acquisition_features
from .types import BeliefState, JsonMap

if TYPE_CHECKING:
    from .core import Problem


class AcquisitionPolicy(nn.Module):
    """Score variable-size candidate sets with a shared neural network."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor, unavailable: torch.Tensor) -> torch.Tensor:
        """Score candidates with unavailable and padding positions masked."""
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(unavailable, -1e9)

    def distribution(
        self, problem: Problem, belief: BeliefState, remaining: int
    ) -> torch.distributions.Categorical:
        """Return the masked categorical acquisition distribution."""
        device = next(self.parameters()).device
        features = torch.tensor(acquisition_features(problem, belief, remaining), device=device)
        unavailable = torch.zeros(problem.n_actions, dtype=torch.bool, device=device)
        unavailable[list(belief.selected)] = True
        if bool(unavailable.all()):
            raise ValueError("no available acquisition")
        return torch.distributions.Categorical(logits=self(features, unavailable))

    @torch.no_grad()
    def choose(self, problem: Problem, belief: BeliefState, remaining: int) -> int:
        """Select an available acquisition under the remaining budget."""
        return int(self.distribution(problem, belief, remaining).logits.argmax().item())


def load_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[AcquisitionPolicy, JsonMap]:
    """Load a validated, portable policy checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != "geodiscover_acquisition_policy"
    ):
        raise ValueError("unsupported policy checkpoint")
    policy = AcquisitionPolicy(checkpoint["hidden_size"]).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, checkpoint
