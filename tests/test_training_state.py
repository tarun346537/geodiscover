"""Optimizer continuation and validation-stopping regressions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from geodiscover.types import JsonMap

if TYPE_CHECKING:
    from geodiscover.policy import AcquisitionPolicy


def clone_policy(policy: AcquisitionPolicy) -> AcquisitionPolicy:
    """Copy weights without sharing optimizer parameters between runs."""
    from geodiscover.policy import AcquisitionPolicy

    result = AcquisitionPolicy(hidden_size=policy.hidden_size)
    result.load_state_dict(deepcopy(policy.state_dict()))
    return result


def assert_same_weights(first: AcquisitionPolicy, second: AcquisitionPolicy) -> None:
    """Require exact CPU tensors rather than a loose numerical tolerance."""
    import torch

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_distillation_resume_keeps_optimizer_order_and_best_weights(
    tiny_policy: AcquisitionPolicy,
    teacher_rows: list[JsonMap],
    tmp_path: Path,
) -> None:
    import torch

    from geodiscover.distillation import distill

    whole, interrupted = clone_policy(tiny_policy), clone_policy(tiny_policy)
    torch.manual_seed(73)
    expected = distill(
        whole, teacher_rows, epochs=4, batch_size=2, seed=19, patience=20, min_delta=0
    )
    checkpoint = tmp_path / "distillation.pt"
    torch.manual_seed(73)
    distill(
        interrupted,
        teacher_rows,
        epochs=2,
        batch_size=2,
        seed=19,
        patience=20,
        min_delta=0,
        checkpoint_path=checkpoint,
    )
    resumed = clone_policy(tiny_policy)
    actual = distill(
        resumed,
        teacher_rows,
        epochs=4,
        batch_size=2,
        seed=19,
        patience=20,
        min_delta=0,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert_same_weights(whole, resumed)
    assert [row["train_loss"] for row in actual] == [row["train_loss"] for row in expected]
    assert len(actual) == 4


@pytest.mark.parametrize("cut", [1, 2])
def test_refinement_resume_is_exact_on_and_between_validation_boundaries(
    cut: int,
    tiny_policy: AcquisitionPolicy,
    tmp_path: Path,
) -> None:
    import torch

    from geodiscover.reinforcement import refine_rl

    whole, interrupted = clone_policy(tiny_policy), clone_policy(tiny_policy)
    torch.manual_seed(83)
    expected = refine_rl(
        whole, episodes=4, budget=2, seed=23, eval_every=2, patience=20, min_delta=0
    )
    checkpoint = tmp_path / "refinement.pt"
    torch.manual_seed(83)
    refine_rl(
        interrupted,
        episodes=cut,
        budget=2,
        seed=23,
        eval_every=2,
        patience=20,
        min_delta=0,
        checkpoint_path=checkpoint,
    )
    resumed = clone_policy(tiny_policy)
    actual = refine_rl(
        resumed,
        episodes=4,
        budget=2,
        seed=23,
        eval_every=2,
        patience=20,
        min_delta=0,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert_same_weights(whole, resumed)
    for metric in ("return", "loss", "final_model_mse"):
        assert [row[metric] for row in actual] == [row[metric] for row in expected]
    assert [row["episode"] for row in actual if "validation_model_mse" in row] == [2, 4]


def test_distillation_refuses_a_changed_teacher_dataset(
    tiny_policy: AcquisitionPolicy,
    teacher_rows: list[JsonMap],
    tmp_path: Path,
) -> None:
    from geodiscover.distillation import distill

    checkpoint = tmp_path / "distillation.pt"
    distill(tiny_policy, teacher_rows, epochs=1, checkpoint_path=checkpoint)
    changed = deepcopy(teacher_rows)
    row = changed[0]
    row["teacher_action"] = (row["teacher_action"] + 1) % len(row["problem"]["operator"])
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        distill(tiny_policy, changed, epochs=2, checkpoint_path=checkpoint, resume=True)


@pytest.mark.parametrize("changed", [{"seed": 6}, {"eval_every": 3}, {"min_delta": 0.02}])
def test_refinement_refuses_changed_random_or_validation_settings(
    changed: JsonMap,
    tiny_policy: AcquisitionPolicy,
    tmp_path: Path,
) -> None:
    from geodiscover.reinforcement import refine_rl

    checkpoint = tmp_path / "refinement.pt"
    options: JsonMap = {"seed": 5, "budget": 2, "eval_every": 2}
    refine_rl(tiny_policy, episodes=1, checkpoint_path=checkpoint, **options)
    options.update(changed)
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        refine_rl(tiny_policy, episodes=2, checkpoint_path=checkpoint, resume=True, **options)


def test_validation_patience_stops_imitation_and_refinement(
    tiny_policy: AcquisitionPolicy,
    teacher_rows: list[JsonMap],
) -> None:
    from geodiscover.distillation import distill
    from geodiscover.reinforcement import refine_rl

    imitation = distill(tiny_policy, teacher_rows, epochs=10, patience=1, min_delta=1e9)
    refinement = refine_rl(
        tiny_policy, episodes=10, budget=2, patience=1, min_delta=1e9, eval_every=1
    )
    assert len(imitation) == 2
    assert len(refinement) == 2
    assert not tiny_policy.training


def test_negative_refinement_minimum_improvement_is_rejected(
    tiny_policy: AcquisitionPolicy,
) -> None:
    from geodiscover.reinforcement import refine_rl

    with pytest.raises(ValueError, match="invalid RL settings"):
        refine_rl(tiny_policy, episodes=1, min_delta=-0.1)
