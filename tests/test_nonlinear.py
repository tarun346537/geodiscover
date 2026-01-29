from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geodiscover.core import Problem
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geodiscover.benchmark import efficiency_curve
from geodiscover.core import Problem, acquisition_features
from geodiscover.nonlinear import ParticleBelief, SimulatorMCTS
from geodiscover.plans import episode_settings, iter_episodes, write_plans


class NonlinearSearchTests(unittest.TestCase):
    def problem(self) -> Problem:
        return Problem([[0.0], [1.0]], [0.05, 0.05], [0.0], [[2.5]])

    def test_observation_branches_resolve_nonlinear_sign_ambiguity(self) -> None:
        p = self.problem()
        models = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        predictions = np.column_stack((4 * models[:, 0] ** 2, models[:, 0]))
        prior = ParticleBelief(models, predictions, np.ones(4))
        action, stats = SimulatorMCTS(256, seed=3).choose(p, prior, 1)
        self.assertEqual(action, 1)
        self.assertGreater(stats["nodes_expanded"], p.n_actions)
        # Keep the surrogate Jacobian fixed and swap nonlinear forward outputs.
        # Search must follow the actual simulator table, not the Jacobian.
        swapped = ParticleBelief(models, predictions[:, ::-1], np.ones(4))
        other, _ = SimulatorMCTS(256, seed=3).choose(p, swapped, 1)
        self.assertEqual(other, 0)
        posterior = prior.update(p, 1, 1.0)
        self.assertAlmostEqual(posterior.mean[0], 1.0, places=6)
        self.assertLess(np.trace(posterior.covariance), 1e-6)
        features = acquisition_features(p, prior, 1)
        self.assertEqual(features.shape, (2, 6))
        self.assertTrue(np.isfinite(features).all())

    def test_budget_allocation_and_streamed_accounting(self) -> None:
        settings = episode_settings(20, 4, target_plans=9, max_nodes=17)
        self.assertEqual(sum(b for b, _ in settings), 9)
        self.assertEqual(sum(n for _, n in settings), 17)
        batches = iter_episodes(budget=4, target_plans=9, max_nodes=17, simulations=128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            report = write_plans(path, batches, {"test": True})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            states = [r for r in records if r["record_type"] == "plan"]
            headers = [r for r in records if r["record_type"] == "scenario"]
            self.assertEqual(len(states), 9)
            self.assertEqual(len(headers), 3)
            self.assertLessEqual(report["nodes_expanded"], 17)
            self.assertEqual(
                report["nodes_expanded"], sum(r["search"]["nodes_expanded"] for r in states)
            )
            self.assertEqual(report["forward_cache"]["misses"], 3)
            self.assertEqual(report["plans"], 9)

    def test_efficiency_curve_matches_shared_reference_budget(self) -> None:
        result = efficiency_curve([1, 2], trials=2, simulations=8, baseline_budget=2)
        self.assertEqual(result["curves"]["1"]["seed"], result["curves"]["2"]["seed"])
        self.assertEqual(
            result["matching_target_mean_model_mse"],
            result["curves"]["2"]["summary"]["greedy"]["mean_model_mse"],
        )
        self.assertIsNotNone(result["matches"]["greedy"]["smallest_matching_budget_on_grid"])


if __name__ == "__main__":
    unittest.main()
