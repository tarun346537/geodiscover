"""Settings, structured records and small-workload API measurements."""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pytest

from geodiscover.config import Settings
from geodiscover.logging import JsonFormatter, request_id
from geodiscover.types import JsonMap


def test_settings_file_is_overridden_by_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"particles": 8, "workers": 1, "output_dir": "local-runs"}))
    monkeypatch.setenv("GEO_PARTICLES", "16")
    monkeypatch.setenv("GEO_WORKERS", "3")
    monkeypatch.setenv("GEO_CPU_THREADS", "2")
    settings = Settings.load(path)
    assert settings.particles == 16
    assert settings.workers == 3
    assert settings.cpu_threads == 2
    assert settings.output_dir == Path("local-runs")


@pytest.mark.parametrize(
    "variable,value", [("GEO_PARTICLES", "1"), ("GEO_CPU_THREADS", "0"), ("GEO_LOG_LEVEL", "LOUD")]
)
def test_invalid_settings_fail_before_workers_start(
    variable: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="GEO_"):
        Settings.load()


def test_json_formatter_keeps_request_context_and_application_fields() -> None:
    record = logging.LogRecord(
        "geodiscover.test", logging.INFO, __file__, 1, "acquired %s", ("candidate-2",), None
    )
    record.fields = {"selected": 2}
    token = request_id.set("request-17")
    try:
        value = json.loads(JsonFormatter().format(record))
    finally:
        request_id.reset(token)
    assert value["request_id"] == "request-17"
    assert value["selected"] == 2
    assert value["event"] == "acquired candidate-2"
    assert value["level"] == "INFO"
    assert value["timestamp"].endswith("+00:00")


def test_tracker_writes_measured_training_records(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from geodiscover.tracking import RunTracker

    path = tmp_path / "steps.jsonl"
    tracker = RunTracker(path, run_id="controlled-run")
    tracker.log("distillation", 1, {"validation_loss": 0.3})
    tracker.log("distillation", 2, {"validation_loss": 0.2})
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert {row["run_id"] for row in records} == {"controlled-run"}
    assert records[-1]["metrics"]["validation_loss"] == 0.2


def small_request() -> JsonMap:
    """Use the same small, explicit linear problem for every timed request."""
    return {
        "problem": {
            "operator": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.9, 0.1]],
            "noise_std": [0.1] * 4,
            "prior_mean": [0.0, 0.0],
            "prior_covariance": [[1.0, 0.0], [0.0, 1.0]],
        },
        "remaining_budget": 1,
        "simulations": 8,
    }


def test_api_request_id_is_returned_and_logged_as_json() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from geodiscover.service import create_app

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("geodiscover")
    logger.addHandler(handler)
    try:
        with TestClient(create_app()) as client:
            response = client.post(
                "/plan", json=small_request(), headers={"x-request-id": "case-42"}
            )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "case-42"
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert any(row["request_id"] == "case-42" and row["status"] == 200 for row in records)
    finally:
        logger.removeHandler(handler)


def test_small_linear_api_fixture_stays_below_the_latency_alert_limit() -> None:
    """Check the 140 ms alert limit on a warm, sequential four-candidate fixture."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from geodiscover.service import create_app

    elapsed = []
    with TestClient(create_app()) as client:
        for _ in range(3):
            assert client.post("/plan", json=small_request()).status_code == 200
        for _ in range(20):
            started = time.perf_counter()
            response = client.post("/plan", json=small_request())
            elapsed.append(time.perf_counter() - started)
            assert response.status_code == 200
            assert response.json()["mode"] == "mcts"
    assert float(np.quantile(elapsed, 0.99)) < 0.140
