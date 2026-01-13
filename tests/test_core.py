from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodiscover.core import Problem
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geodiscover.cache import ForwardCache
from geodiscover.core import (
    MCTS,
    Belief,
    Problem,
    SurveyEnvironment,
    posterior,
    synthetic_problem,
)
from geodiscover.demo import evaluate, read_evaluation_models


class BeliefTests(unittest.TestCase):
    def test_sequential_matches_closed_form_bayesian_posterior(self) -> None:
        p = synthetic_problem(7)
        actions, y = [1, 6, 9], np.array([0.3, -0.1, 0.5])
        belief = posterior(p, actions, y)
        a = p.operator[actions]
        precision = (
            np.linalg.inv(p.prior_covariance) + a.T @ np.diag(1 / p.noise_std[actions] ** 2) @ a
        )
        covariance = np.linalg.inv(precision)
        mean = covariance @ (
            np.linalg.solve(p.prior_covariance, p.prior_mean)
            + a.T @ ((y - p.offset[actions]) / p.noise_std[actions] ** 2)
        )
        np.testing.assert_allclose(belief.mean, mean, atol=1e-10)
        np.testing.assert_allclose(belief.covariance, covariance, atol=1e-10)

    def test_search_uses_budget_and_reduces_uncertainty(self) -> None:
        p = Problem(np.eye(2), [0.1, 0.1], np.zeros(2), np.diag([2.0, 1.0]))
        action, stats = MCTS(128, seed=2).choose(p, Belief.prior(p), 1)
        self.assertEqual(action, 0)
        self.assertEqual(stats["simulations"], 128)
        env = SurveyEnvironment(p, np.array([1.0, -0.3]), 2)
        before = np.trace(env.belief.covariance)
        while not env.done:
            action, _ = MCTS(32, seed=3).choose(p, env.belief, 2 - len(env.belief.selected))
            env.step(action)
        self.assertEqual(len(set(env.belief.selected)), 2)
        self.assertLess(np.trace(env.belief.covariance), before)
        with self.assertRaises(ValueError):
            env.step(0)

    def test_invalid_or_repeated_acquisition_rejected(self) -> None:
        p = synthetic_problem(2)
        with self.assertRaises(ValueError):
            posterior(p, [0, 0], [0.0, 0.0])
        with self.assertRaises(ValueError):
            posterior(p, [0], [])
        with self.assertRaises(ValueError):
            MCTS(2).choose(p, Belief.prior(p), p.n_actions + 1)

    def test_equal_budget_evaluation_provenance(self) -> None:
        result = evaluate(seed=90, trials=2, budget=2, simulations=8)
        self.assertEqual(len(result["records"]), 6)
        self.assertEqual(result["provenance"]["kind"], "synthetic_demo")
        for row in result["records"]:
            self.assertEqual(len(set(row["selected"])), 2)
            self.assertTrue(np.isfinite(row["model_mse"]))
        json.dumps(result, allow_nan=False)

    def test_explicit_held_out_model_split(self) -> None:
        problem = synthetic_problem(4)
        values = [
            dict(model_id="a", split="train", model=[0.0] * 6),
            dict(model_id="b", split="test", model=[0.2] * 6),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in values))
            selected = read_evaluation_models(path, problem)
            self.assertEqual([r["model_id"] for r in selected], ["b"])
            report = evaluate(trials=1, budget=1, simulations=4, problem=problem, models=selected)
            self.assertEqual({r["model_id"] for r in report["records"]}, {"b"})


class CacheTests(unittest.TestCase):
    def test_quantized_cache_solves_representative(self) -> None:
        calls = []
        cache = ForwardCache(lambda m: calls.append(m.copy()) or m * 2, {"operator": "twice"}, 0.1)
        first = cache([0.14])
        second = cache([0.11])
        np.testing.assert_allclose(first, [0.2])
        np.testing.assert_allclose(second, [0.2])
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.hits, 1)
        first[0] = -999
        np.testing.assert_allclose(cache([0.14]), [0.2])

    def test_operator_identity_prevents_cross_survey_alias(self) -> None:
        class MemoryRedis:
            def __init__(self) -> None:
                self.values = {}

            def get(self, key: str) -> bytes | None:
                return self.values.get(key)

            def set(self, key: str, value: bytes, ex: int) -> None:
                self.values[key] = value

        backend = MemoryRedis()
        a = ForwardCache(lambda m: m * 2, {"operator": [[2.0]]}, redis_client=backend)
        b = ForwardCache(lambda m: m * 3, {"operator": [[3.0]]}, redis_client=backend)
        np.testing.assert_allclose(a([1.0]), [2.0])
        np.testing.assert_allclose(b([1.0]), [3.0])
        self.assertEqual(len(backend.values), 2)


if __name__ == "__main__":
    unittest.main()
