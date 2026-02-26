"""Geological-model and teacher-state split regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from geodiscover.core import Belief, Problem
from geodiscover.parallel import _remote_plan
from geodiscover.train_data import read_models, read_plans
from geodiscover.types import JsonMap


def test_validation_models_can_share_a_file_with_training_and_test_models(
    survey: Problem,
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.jsonl"
    records = [
        {"model_id": split, "split": split, "model": [float(i)] * len(survey.prior_mean)}
        for i, split in enumerate(("train", "validation", "test"))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    for index, split in enumerate(("train", "validation", "test")):
        selected = read_models(path, survey, split=split)
        assert len(selected) == 1
        np.testing.assert_array_equal(selected[0], np.full(len(survey.prior_mean), index))


def test_teacher_scenario_cannot_cross_training_and_validation(
    teacher_rows: list[JsonMap],
    tmp_path: Path,
) -> None:
    changed = deepcopy(teacher_rows)
    duplicate = deepcopy(changed[0])
    duplicate["split"] = "train"
    changed.append(duplicate)
    path = tmp_path / "leaked.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in changed))
    with pytest.raises(ValueError, match="scenario leakage"):
        read_plans(path)


def test_plan_record_requires_its_scenario_header(tmp_path: Path) -> None:
    path = tmp_path / "orphan.jsonl"
    path.write_text(
        json.dumps({"schema_version": 2, "record_type": "plan", "scenario_id": "missing"})
    )
    with pytest.raises(ValueError, match="preceding scenario header"):
        read_plans(path)


def test_remote_worker_helper_preserves_budget_and_reports_actual_search(survey: Problem) -> None:
    action, statistics = _remote_plan(survey, Belief.prior(survey), 2, 8, 13, 5.0)
    assert 0 <= action < survey.n_actions
    assert statistics["simulations"] == 8
    assert 0 < statistics["nodes_expanded"] <= 8
