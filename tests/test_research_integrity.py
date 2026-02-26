"""Posterior support, dataset identity and resume lineage regressions."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from geodiscover.core import Problem
from geodiscover.demo import read_evaluation_models
from geodiscover.nonlinear import ParticleBelief, SimulatorMCTS
from geodiscover.train_data import read_models, read_plans
from geodiscover.types import JsonMap

if TYPE_CHECKING:
    from geodiscover.policy import AcquisitionPolicy


def test_zero_prior_particle_cannot_reappear_after_surprising_observation() -> None:
    problem = Problem([[1.0]], [0.1], [0.0], [[1.0]])
    prior = ParticleBelief([[0.0], [10.0]], [[0.0], [10.0]], [1.0, 0.0])
    updated = prior.update(problem, 0, 10.0)
    np.testing.assert_array_equal(updated.weights, [1.0, 0.0])


def test_impossible_numeric_likelihood_reports_units_instead_of_nan_weights() -> None:
    problem = Problem([[1.0]], [0.1], [0.0], [[1.0]])
    prior = ParticleBelief([[0.0], [1.0]], [[0.0], [1.0]], [0.5, 0.5])
    with pytest.raises(ValueError, match="check observation units"):
        prior.update(problem, 0, 1e308)


@pytest.mark.parametrize("settings", [{"exploration": np.nan}, {"observation_bin_std": np.inf}])
def test_nonfinite_tree_settings_fail_before_search(settings: JsonMap) -> None:
    with pytest.raises(ValueError, match="invalid nonlinear search"):
        SimulatorMCTS(**settings)


def test_renamed_identical_world_cannot_leak_into_the_test_split(
    survey: Problem, tmp_path: Path
) -> None:
    path = tmp_path / "worlds.jsonl"
    rows = [
        {"model_id": "original", "split": "train", "model": [0.0] * len(survey.prior_mean)},
        {"model_id": "renamed", "split": "test", "model": [-0.0] * len(survey.prior_mean)},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="model leakage"):
        read_models(path, survey)
    with pytest.raises(ValueError, match="model leakage"):
        read_evaluation_models(path, survey)


def test_evaluation_reader_cannot_select_training_split(survey: Problem, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation or test"):
        read_evaluation_models(tmp_path / "unused.jsonl", survey, split="train")


def test_same_scenario_cannot_change_surveys_mid_episode(
    teacher_rows: list[JsonMap], tmp_path: Path
) -> None:
    rows = deepcopy(teacher_rows)
    rows[1]["problem"]["operator"][0][0] += 0.1
    path = tmp_path / "changed-survey.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="scenario inputs changed"):
        read_plans(path)


def test_refinement_rejects_overlapping_explicit_worlds(
    tiny_policy: AcquisitionPolicy, survey: Problem
) -> None:
    from geodiscover.reinforcement import refine_rl

    model = np.zeros_like(survey.prior_mean)
    with pytest.raises(ValueError, match="model leakage"):
        refine_rl(
            tiny_policy, episodes=1, problem=survey, models=[model], validation_models=[model]
        )


def test_changed_distilled_weights_cannot_resume_an_old_rl_run(
    tiny_policy: AcquisitionPolicy, tmp_path: Path
) -> None:
    import torch

    from geodiscover.policy import AcquisitionPolicy
    from geodiscover.reinforcement import refine_rl

    path = tmp_path / "refinement.pt"
    refine_rl(tiny_policy, episodes=1, budget=2, checkpoint_path=path)
    replacement = AcquisitionPolicy(hidden_size=tiny_policy.hidden_size)
    with torch.no_grad():
        next(replacement.parameters()).add_(1)
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        refine_rl(replacement, episodes=2, budget=2, checkpoint_path=path, resume=True)


@pytest.mark.parametrize(
    "response",
    [
        {"action": True, "mode": "mcts"},
        {"action": 9, "mode": "mcts"},
        {"action": 0, "mode": "unknown"},
    ],
)
def test_load_measurement_rejects_invalid_success_payloads(response: JsonMap) -> None:
    httpx = pytest.importorskip("httpx")
    from geodiscover.loadtest import run_load_test

    async def measure() -> JsonMap:
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
        ) as client:
            return await run_load_test(
                client, {"problem": {"operator": [[1.0]]}}, requests=2, warmup=0
            )

    result = asyncio.run(measure())
    assert result["successful_decisions"] == 0
    assert result["errors_by_kind"] == {"invalid_response": 2}


def test_plan_record_cannot_override_its_scenario_split(
    teacher_rows: list[JsonMap], tmp_path: Path
) -> None:
    first = teacher_rows[0]
    header = {**first, "schema_version": 2, "record_type": "scenario"}
    record = {
        "schema_version": 2,
        "record_type": "plan",
        "scenario_id": first["scenario_id"],
        "split": "train",
        "selected": [],
        "observations": [],
        "remaining_budget": 2,
        "teacher_action": 0,
    }
    path = tmp_path / "override.jsonl"
    path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="override scenario context"):
        read_plans(path)


def test_in_memory_distillation_checks_scenario_split_integrity(
    tiny_policy: AcquisitionPolicy, teacher_rows: list[JsonMap]
) -> None:
    from geodiscover.distillation import distill

    rows = deepcopy(teacher_rows)
    rows[1]["split"] = "train"
    with pytest.raises(ValueError, match="scenario leakage"):
        distill(tiny_policy, rows, epochs=1)


def test_particle_normalization_does_not_overflow_large_finite_weights() -> None:
    prior = ParticleBelief([[0.0], [1.0]], [[0.0], [1.0]], [1e308, 1e308])
    np.testing.assert_array_equal(prior.weights, [0.5, 0.5])
    assert prior.effective_sample_size == 2


def test_wrong_survey_ensemble_is_rejected_before_search() -> None:
    problem = Problem([[1.0], [2.0]], [0.1, 0.1], [0.0], [[1.0]])
    prior = ParticleBelief([[0.0], [1.0]], [[0.0], [1.0]], [0.5, 0.5])
    with pytest.raises(ValueError, match="disagree with survey"):
        SimulatorMCTS(8).choose(problem, prior, 1)


def test_heldout_file_cannot_rename_a_checkpoint_training_world() -> None:
    from geodiscover.train_data import model_fingerprint, reject_training_worlds

    model = np.array([0.2, -0.3])
    checkpoint = {"training_model_sha256": [model_fingerprint(model)]}
    rows = [{"model_id": "renamed", "split": "test", "model": model.tolist()}]
    with pytest.raises(ValueError, match="RL training world"):
        reject_training_worlds(rows, checkpoint)
