from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from geodiscover.core import Belief, synthetic_problem


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency unavailable")
class TrainingTests(unittest.TestCase):
    def test_distillation_refinement_and_portable_checkpoint(self) -> None:
        import torch

        from geodiscover.plans import generate
        from geodiscover.policy import AcquisitionPolicy, load_policy
        from geodiscover.training import distill, refine_rl

        torch.set_num_threads(1)
        torch.manual_seed(4)
        rows = generate(episodes=6, budget=2, simulations=8)
        policy = AcquisitionPolicy(hidden_size=16)
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        history = distill(policy, rows, epochs=2)
        self.assertEqual(len(history), 2)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in policy.state_dict().items()))
        rl = refine_rl(policy, episodes=2, budget=2, seed=900)
        self.assertEqual(len(rl), 2)
        self.assertTrue(all(np.isfinite(row["loss"]) for row in rl))
        problem = synthetic_problem(3)
        belief = Belief.prior(problem).expected_update(problem, 0)
        action = policy.choose(problem, belief, 2)
        self.assertNotEqual(action, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "kind": "geodiscover_acquisition_policy",
                    "hidden_size": 16,
                    "state_dict": policy.state_dict(),
                },
                path,
            )
            restored, _ = load_policy(path)
            self.assertEqual(action, restored.choose(problem, belief, 2))

    def test_particle_plans_round_trip_and_cpu_profiling(self) -> None:
        import torch

        from geodiscover.plans import iter_episodes, write_plans
        from geodiscover.policy import AcquisitionPolicy
        from geodiscover.training import distill, read_plans, refine_rl

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            write_plans(
                path,
                iter_episodes(episodes=3, budget=2, simulations=8, backend="particle", particles=8),
                {},
            )
            rows = read_plans(path)
            self.assertEqual(len(rows), 6)
            policy = AcquisitionPolicy(hidden_size=16)
            history = distill(policy, rows, epochs=1, batch_size=2)
            self.assertGreater(history[0]["profile"]["wall_seconds"], 0)
            self.assertIsNone(history[0]["profile"]["gpu_utilization_percent_mean"])
            rl = refine_rl(policy, episodes=1, budget=2, backend="particle", particles=8)
            self.assertEqual(rl[0]["belief_backend"], "particle")
            self.assertEqual(rl[0]["forward_cache_delta"]["misses"], 9)


@unittest.skipUnless(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
    "optional service test dependencies unavailable",
)
class ServiceTests(unittest.TestCase):
    def request(self) -> dict[str, Any]:
        p = synthetic_problem(4)
        data = p.to_dict()
        data.pop("metadata")
        return {"problem": data, "remaining_budget": 2, "simulations": 8}

    def test_health_planning_validation_and_metrics(self) -> None:
        from fastapi.testclient import TestClient

        from geodiscover.service import create_app

        with TestClient(create_app()) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            response = client.post("/plan", json=self.request())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "mcts")
            request = self.request()
            request["problem"]["noise_std"][0] = 0
            self.assertEqual(client.post("/plan", json=request).status_code, 422)
            self.assertIn("geodiscover_plans_total", client.get("/metrics").text)

    def test_failed_worker_and_timeout_use_greedy_fallback(self) -> None:
        from fastapi.testclient import TestClient

        from geodiscover.service import create_app

        def failed(*args: object) -> None:
            raise RuntimeError("injected solver worker failure")

        with TestClient(create_app(planner=failed)) as client:
            response = client.post("/plan", json=self.request()).json()
            self.assertEqual(response["mode"], "greedy_bayesian_fallback")
            self.assertEqual(response["fallback_reason"], "planner_failure")

        def slow(*args: object) -> tuple[object, dict[str, object]]:
            time.sleep(0.05)
            return 0, {}

        with TestClient(create_app(planner=slow)) as client:
            request = dict(self.request(), timeout_ms=10)
            response = client.post("/plan", json=request).json()
            self.assertEqual(response["fallback_reason"], "timeout")

    def test_malformed_planner_actions_fall_back_and_numpy_integer_is_accepted(self) -> None:
        from fastapi.testclient import TestClient

        from geodiscover.service import create_app

        class IndexLike:
            def __index__(self) -> int:
                return 1

        for malformed in (1.5, 1.0, True, np.bool_(True), IndexLike()):
            with self.subTest(action=type(malformed).__name__):

                def planner(
                    *_args: object, returned: object = malformed
                ) -> tuple[object, dict[str, object]]:
                    return returned, {}

                with TestClient(create_app(planner=planner)) as client:
                    response = client.post("/plan", json=self.request())
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["mode"], "greedy_bayesian_fallback")
                    self.assertEqual(response.json()["fallback_reason"], "planner_failure")
                    self.assertIs(type(response.json()["action"]), int)

        def valid(*_args: object) -> tuple[object, dict[str, object]]:
            return np.int64(1), {}

        with TestClient(create_app(planner=valid)) as client:
            response = client.post("/plan", json=self.request())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "mcts")
            self.assertEqual(response.json()["action"], 1)

    @unittest.skipUnless(
        importlib.util.find_spec("simpeg"), "optional SimPEG dependency unavailable"
    )
    def test_configured_dc_survey_serves_nonlinear_particle_search(self) -> None:
        from fastapi.testclient import TestClient

        from geodiscover.core import Problem
        from geodiscover.service import create_app

        config = {
            "hx": [10.0] * 6,
            "hz": [10.0] * 5,
            "origin": [-30.0, -50.0],
            "nky": 3,
            "candidates": [
                {"a": [-20.0, 0.0], "b": [-10.0, 0.0], "m": [10.0, 0.0], "n": [20.0, 0.0]}
            ],
        }
        problem = Problem(
            np.zeros((1, 30)),
            [0.01],
            np.full(30, np.log(0.01)),
            np.eye(30) * 0.04,
            metadata={"simpeg_dc": config},
        )
        with TestClient(create_app(problem_config=problem.to_dict(), particles=4)) as client:
            self.assertEqual(client.get("/health").json()["solver"], "simpeg_particle_belief")
            response = client.post("/plan", json={"remaining_budget": 1, "simulations": 8})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["statistics"]["backend"], "simulator_particle_mcts")


@unittest.skipUnless(importlib.util.find_spec("simpeg"), "optional SimPEG dependency unavailable")
class SimPEGTests(unittest.TestCase):
    def test_homogeneous_conductivity_scaling(self) -> None:
        from geodiscover.simpeg_adapter import SimPEGDC

        adapter = SimPEGDC(
            {
                "hx": [10.0] * 6,
                "hz": [10.0] * 5,
                "origin": [-30.0, -50.0],
                "nky": 3,
                "candidates": [
                    {"a": [-20.0, 0.0], "b": [-10.0, 0.0], "m": [10.0, 0.0], "n": [20.0, 0.0]}
                ],
            }
        )
        model = np.full(adapter.mesh.nC, np.log(0.01))
        first = adapter.forward(model)
        second = adapter.forward(model + np.log(2))
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(abs(first[0]), 0)
        np.testing.assert_allclose(second, first / 2, rtol=1e-5)


@unittest.skipUnless(importlib.util.find_spec("httpx"), "optional HTTPX dependency unavailable")
class LoadTestTests(unittest.IsolatedAsyncioTestCase):
    async def test_measured_errors_and_fallbacks_are_counted(self) -> None:
        import httpx

        from geodiscover.loadtest import run_load_test

        count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            if count % 3 == 0:
                return httpx.Response(503)
            mode = "greedy_bayesian_fallback" if count % 2 == 0 else "mcts"
            return httpx.Response(200, json={"action": 0, "mode": mode})

        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            result = await run_load_test(client, {}, requests=6, concurrency=2, warmup=0)
        self.assertEqual(result["requests"], 6)
        self.assertEqual(result["successful_decisions"], 4)
        self.assertEqual(result["error_count"], 2)
        self.assertEqual(result["fallback_count"], 2)
        self.assertEqual(result["status_codes"]["503"], 2)
        self.assertGreater(result["requests_per_second"], 0)
        self.assertGreaterEqual(result["latency_seconds_p99"], result["latency_seconds_p50"])


if __name__ == "__main__":
    unittest.main()
