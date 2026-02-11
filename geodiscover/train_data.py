"""Validation and schema loading for teacher plans and geological models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import Problem, posterior
from .nonlinear import particle_posterior
from .types import BeliefState, JsonMap, Vector


def read_plans(path: str | Path) -> list[JsonMap]:
    """Load teacher states and reject split leakage or invalid actions."""
    rows: list[JsonMap] = []
    scenarios: dict[str, JsonMap] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") == 2:
                scenario_id = record["scenario_id"]
                if record["record_type"] == "scenario":
                    if scenario_id in scenarios:
                        raise ValueError("duplicate scenario header")
                    scenarios[scenario_id] = record
                    continue
                if record["record_type"] != "plan" or scenario_id not in scenarios:
                    raise ValueError("plan requires a preceding scenario header")
                allowed = {
                    "schema_version",
                    "record_type",
                    "scenario_id",
                    "selected",
                    "observations",
                    "remaining_budget",
                    "teacher_action",
                    "search",
                }
                if record.keys() - allowed:
                    raise ValueError("plan records cannot override scenario context")
                record = {**scenarios[scenario_id], **record, "schema_version": 1}
            rows.append(record)
    validate_plan_rows(rows)
    return rows


def validate_plan_rows(rows: list[JsonMap]) -> None:
    """Check in-memory records with the same split rules as JSONL inputs."""
    if not rows:
        raise ValueError("empty plans dataset")
    scenario_splits: dict[str, str] = {}
    scenario_inputs: dict[str, str] = {}
    for row in rows:
        if row.get("schema_version") != 1 or row["split"] not in ("train", "validation"):
            raise ValueError("invalid plan schema or split")
        scenario = row["scenario_id"]
        if scenario in scenario_splits and scenario_splits[scenario] != row["split"]:
            raise ValueError("scenario leakage across train and validation")
        scenario_splits[scenario] = row["split"]
        inputs = json.dumps(
            {k: row.get(k) for k in ("problem", "belief_backend", "particle_ensemble")},
            sort_keys=True,
            allow_nan=False,
        )
        if scenario in scenario_inputs and scenario_inputs[scenario] != inputs:
            raise ValueError("scenario inputs changed. use one survey and ensemble per scenario")
        scenario_inputs[scenario] = inputs
        problem = Problem.from_dict(row["problem"])
        belief = belief_for_row(problem, row)
        if (
            row["teacher_action"] in belief.selected
            or not 0 <= row["teacher_action"] < problem.n_actions
        ):
            raise ValueError("teacher action unavailable")
        if not 1 <= row["remaining_budget"] <= problem.n_actions - len(belief.selected):
            raise ValueError("invalid remaining budget")
    if set(scenario_splits.values()) != {"train", "validation"}:
        raise ValueError("separate training and validation scenarios required")


def belief_for_row(problem: Problem, row: JsonMap) -> BeliefState:
    """Restore the Gaussian or particle belief for a teacher state."""
    backend = row.get("belief_backend", "linear")
    if backend not in ("linear", "particle"):
        raise ValueError("unsupported belief backend")
    if backend == "particle":
        return particle_posterior(
            problem, row["particle_ensemble"], row["selected"], row["observations"]
        )
    return posterior(problem, row["selected"], row["observations"])


def file_sha256(path: str | Path) -> str:
    """Hash an input file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_key(model: Vector) -> bytes:
    """Canonicalize exact geological values, including signed zero."""
    values = np.asarray(model, dtype="<f8").copy()
    values[values == 0] = 0
    return values.tobytes()


def model_fingerprint(model: Vector) -> str:
    """Identify exact world values across independently named input files."""
    return hashlib.sha256(model_key(model)).hexdigest()


def reject_training_worlds(rows: list[JsonMap], checkpoint: JsonMap) -> None:
    """Block known training worlds from a policy's held-out evaluation."""
    training = set(checkpoint.get("training_model_sha256", []))
    if any(model_fingerprint(np.asarray(row["model"], float)) in training for row in rows):
        raise ValueError(
            "evaluation contains an RL training world. select independent held-out models"
        )


def read_model_records(path: str | Path, problem: Problem) -> list[JsonMap]:
    """Validate all splits before selecting any training or evaluation models."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids: set[str] = set()
    splits: dict[bytes, str] = {}
    for row in rows:
        if row["model_id"] in ids:
            raise ValueError("duplicate model IDs")
        ids.add(row["model_id"])
        model = np.asarray(row["model"], float)
        if model.shape != problem.prior_mean.shape or not np.isfinite(model).all():
            raise ValueError("invalid model dimensions or values")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("invalid model split")
        key = model_key(model)
        if key in splits and splits[key] != row["split"]:
            raise ValueError(
                "geological model leakage across splits. assign duplicate worlds together"
            )
        splits[key] = row["split"]
    return rows


def read_models(path: str | Path, problem: Problem, split: str = "train") -> list[Vector]:
    """Read one designated geological-model split without mixing other splits."""
    if split not in ("train", "validation", "test"):
        raise ValueError("invalid model split")
    training = [
        np.asarray(row["model"], float)
        for row in read_model_records(path, problem)
        if row["split"] == split
    ]
    if not training:
        raise ValueError(f"no {split} models. supply a model file with the requested split")
    return training
