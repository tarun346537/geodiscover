"""Masked minibatch imitation learning from MCTS acquisition plans."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    GRADIENT_CLIP,
)
from .core import Problem, acquisition_features
from .policy import AcquisitionPolicy
from .profiling import EpochProfiler
from .tracking import RunTracker, load_checkpoint, restore_rng, rng_state, save_checkpoint
from .train_data import belief_for_row, validate_plan_rows
from .types import Features, JsonMap


@dataclass
class TeacherState:
    """Prepared belief features and the search-selected acquisition label."""

    features: Features
    unavailable: NDArray[np.bool_]
    action: int
    split: str


def prepare_states(rows: list[JsonMap]) -> tuple[list[TeacherState], str]:
    """Validate teacher features once and fingerprint the optimization input."""
    validate_plan_rows(rows)
    states, problems = [], {}
    digest = hashlib.sha256()
    for row in rows:
        key = row["scenario_id"]
        if key not in problems:
            problems[key] = Problem.from_dict(row["problem"])
        problem = problems[key]
        belief = belief_for_row(problem, row)
        features = acquisition_features(problem, belief, row["remaining_budget"])
        mask = np.zeros(problem.n_actions, dtype=bool)
        mask[list(belief.selected)] = True
        action, split = int(row["teacher_action"]), str(row["split"])
        states.append(TeacherState(features, mask, action, split))
        digest.update(features.tobytes() + mask.tobytes() + f"{key}:{action}:{split}".encode())
    return states, digest.hexdigest()


def imitation_loss(policy: AcquisitionPolicy, row: JsonMap) -> tuple[torch.Tensor, int]:
    """Compute teacher-action loss and agreement for one state."""
    problem = Problem.from_dict(row["problem"])
    belief = belief_for_row(problem, row)
    distribution = policy.distribution(problem, belief, row["remaining_budget"])
    target = torch.tensor(row["teacher_action"], device=next(policy.parameters()).device)
    return -distribution.log_prob(target), int(
        distribution.logits.argmax().item() == row["teacher_action"]
    )


def _batch_loss(policy: AcquisitionPolicy, batch: list[TeacherState]) -> tuple[torch.Tensor, int]:
    n_actions = max(len(row.features) for row in batch)
    n_features = batch[0].features.shape[1]
    features = np.zeros((len(batch), n_actions, n_features), dtype=np.float32)
    unavailable = np.ones((len(batch), n_actions), dtype=bool)
    for i, row in enumerate(batch):
        features[i, : len(row.features)] = row.features
        unavailable[i, : len(row.features)] = row.unavailable
    device = next(policy.parameters()).device
    labels = torch.tensor([row.action for row in batch], device=device)
    logits = policy(torch.from_numpy(features).to(device), torch.from_numpy(unavailable).to(device))
    return torch.nn.functional.cross_entropy(logits, labels), int(
        (logits.argmax(-1) == labels).sum()
    )


def distill(
    policy: AcquisitionPolicy,
    rows: list[JsonMap],
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = 3e-3,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    tracker: RunTracker | None = None,
) -> list[JsonMap]:
    """Fit teacher actions with resumable optimizer state and validation stopping.

    Returns the completed epoch measurements and leaves the best validation
    weights loaded. The checkpoint retains the last optimizer state for resume.
    """
    if epochs < 1 or learning_rate <= 0 or batch_size < 1 or patience < 1 or min_delta < 0:
        raise ValueError(
            "epochs, learning rate, batch size and patience must be positive. min_delta >= 0"
        )
    started = time.perf_counter()
    prepared, fingerprint = prepare_states(rows)
    training = [row for row in prepared if row.split == "train"]
    validation = [row for row in prepared if row.split == "validation"]
    if not training or not validation:
        raise ValueError("supply separate training and validation scenarios")
    preparation_seconds = time.perf_counter() - started
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    identity = {
        "features_sha256": fingerprint,
        "learning_rate": learning_rate,
        "seed": seed,
        "batch_size": batch_size,
        "hidden_size": policy.hidden_size,
        "patience": patience,
        "min_delta": min_delta,
    }
    history: list[JsonMap] = []
    best_loss, stale, start_epoch = float("inf"), 0, 0
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    if resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise ValueError("resume needs an existing distillation checkpoint path")
        state = load_checkpoint(checkpoint_path, "distillation", identity)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], rng)
        history, best_state = state["history"], state["best_state"]
        best_loss, stale, start_epoch = state["best_loss"], state["stale"], state["completed"]
    for epoch in range(start_epoch, epochs):
        if stale >= patience:
            break
        with EpochProfiler(next(policy.parameters()).device) as profile:
            policy.train()
            total_loss = 0.0
            order = rng.permutation(len(training))
            for start in range(0, len(order), batch_size):
                batch = [training[int(i)] for i in order[start : start + batch_size]]
                loss, _ = _batch_loss(policy, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), GRADIENT_CLIP)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            policy.eval()
            validation_loss, agreements = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(validation), batch_size):
                    batch = validation[start : start + batch_size]
                    loss, correct = _batch_loss(policy, batch)
                    validation_loss += float(loss) * len(batch)
                    agreements += correct
            val_loss = validation_loss / len(validation)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(training),
            "validation_loss": val_loss,
            "validation_teacher_agreement": agreements / len(validation),
            "batch_size": batch_size,
            "preprocessing_seconds": preparation_seconds if epoch == start_epoch else 0.0,
            "profile": profile.result,
        }
        history.append(row)
        if val_loss < best_loss - min_delta:
            best_loss, stale = val_loss, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
            }
        else:
            stale += 1
        save_checkpoint(
            checkpoint_path,
            {
                "phase": "distillation",
                "identity": identity,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(rng),
                "history": history,
                "best_state": best_state,
                "best_loss": best_loss,
                "stale": stale,
                "completed": epoch + 1,
            },
        )
        if tracker:
            tracker.log("distillation", epoch + 1, row)
    policy.load_state_dict(best_state)
    policy.eval()
    return history
