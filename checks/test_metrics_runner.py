"""Known-answer tests for metrics and the simulation pipeline.

Run from the repo root with:
    python -m unittest checks.test_metrics_runner
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sfp import metrics, runner
from sfp.population import MEN, WOMEN


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.group = np.array([MEN, MEN, WOMEN, WOMEN])
        self.theta = np.array([0.5, 0.5, -0.5, -0.5])
        self.x = np.array([1.0, 1.0, -1.0, -1.0])

    def test_snapshot_metrics_known_values(self):
        self.assertEqual(metrics.group_means(self.x, self.group), (1.0, -1.0))
        self.assertEqual(metrics.group_gap(self.x, self.group), 2.0)
        self.assertEqual(metrics.drift(self.x, self.theta, self.group), (0.5, -0.5))
        self.assertEqual(metrics.group_spread(self.x, self.group), (0.0, 0.0))
        self.assertEqual(metrics.extremity(self.x), 1.0)
        self.assertEqual(metrics.share_positive(self.x), 0.5)
        self.assertEqual(metrics.prediction_error(np.zeros(4), self.x), 1.0)
        self.assertEqual(
            metrics.prediction_error(np.zeros(4), self.x, self.group == WOMEN),
            1.0,
        )

    def test_summarize_trajectory_known_values(self):
        x = np.stack([self.x, np.array([2.0, 0.0, 0.0, -2.0])])
        result = {
            "times": np.array([0.0, 1.0]),
            "x": x,
            "theta": self.theta,
            "group": self.group,
            "x_hat": np.zeros_like(x),
        }

        summary = metrics.summarize(result)

        np.testing.assert_allclose(summary["mean_men"], [1.0, 1.0])
        np.testing.assert_allclose(summary["mean_women"], [-1.0, -1.0])
        np.testing.assert_allclose(summary["gap"], [2.0, 2.0])
        np.testing.assert_allclose(summary["drift_men"], [0.5, 0.5])
        np.testing.assert_allclose(summary["drift_women"], [-0.5, -0.5])
        np.testing.assert_allclose(summary["spread_men"], [0.0, 1.0])
        np.testing.assert_allclose(summary["spread_women"], [0.0, 1.0])
        np.testing.assert_allclose(summary["variance"], [1.0, 2.0])
        np.testing.assert_allclose(summary["extremity"], [1.0, 1.0])
        np.testing.assert_allclose(summary["share_positive"], [0.5, 0.25])
        np.testing.assert_allclose(summary["err_vs_x"], [1.0, 1.0])
        np.testing.assert_allclose(summary["err_vs_theta"], [0.5, 0.5])
        np.testing.assert_allclose(summary["err_vs_x_women"], [1.0, 1.0])
        np.testing.assert_allclose(summary["err_vs_theta_women"], [0.5, 0.5])

    def test_summarize_without_predictions_uses_nan_errors(self):
        result = {
            "times": np.array([0.0, 1.0]),
            "x": np.stack([self.x, self.x]),
            "theta": self.theta,
            "group": self.group,
            "x_hat": None,
        }

        summary = metrics.summarize(result)

        for key in ("err_vs_x", "err_vs_theta", "err_vs_x_women",
                    "err_vs_theta_women"):
            self.assertTrue(np.isnan(summary[key]).all(), key)


class RunnerTests(unittest.TestCase):
    def test_run_wires_population_platform_and_simulator(self):
        group = np.array([MEN, WOMEN])
        theta = np.array([0.25, -0.25])
        activity = np.array([0.1, 0.05])
        population = {"group": group, "theta": theta, "a": activity}
        platform = object()
        simulation = {
            "times": np.array([0.0]),
            "x": np.array([[0.25, -0.25]]),
            "x_hat": None,
        }
        overrides = {"N": 2, "T": 0.0, "dt": 0.1, "record_every": 1}

        with (
            patch.object(runner, "make_population", return_value=population) as make_population,
            patch.object(runner, "make_platform", return_value=platform) as make_platform,
            patch.object(runner, "simulate", return_value=simulation) as simulate,
        ):
            result = runner.run(overrides, platform="true", seed=3)

        make_population.assert_called_once_with(
            N=2,
            share_women=runner.BASELINE["share_women"],
            rho=runner.BASELINE["rho"],
            Delta=runner.BASELINE["Delta"],
            sigma=runner.BASELINE["sigma"],
            eps=runner.BASELINE["eps"],
            gamma=runner.BASELINE["gamma"],
            seed=3,
        )
        make_platform.assert_called_once_with("true", group, result["params"])
        self.assertEqual(simulate.call_args.kwargs["platform"], platform)
        self.assertEqual(simulate.call_args.kwargs["seed"], 10_003)
        self.assertIs(result["group"], group)
        self.assertIs(result["theta"], theta)
        self.assertEqual(result["platform"], "true")
        self.assertEqual(result["seed"], 3)
        self.assertEqual(result["params"]["N"], 2)

    def test_full_params_rejects_unknown_override(self):
        with self.assertRaisesRegex(KeyError, "unknown parameter"):
            runner.full_params({"not_a_parameter": 1})

    def test_run_cached_saves_loads_and_reuses_result(self):
        result = {
            "times": np.array([0.0, 1.0]),
            "x": np.array([[0.0, 1.0], [0.5, -0.5]]),
            "x_hat": None,
            "params": {"N": 2},
            "platform": "none",
            "seed": 5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(runner, "RESULTS_DIR", Path(temp_dir)),
                patch.object(runner, "run", return_value=result) as run,
            ):
                first = runner.run_cached({"N": 2}, "none", 5, "test")
                second = runner.run_cached({"N": 2}, "none", 5, "test")

            run.assert_called_once_with({"N": 2}, "none", 5)
            np.testing.assert_array_equal(first["x"], result["x"])
            np.testing.assert_array_equal(second["x"], result["x"])
            self.assertIsNone(second["x_hat"])
            self.assertEqual(second["params"], result["params"])
            self.assertEqual(second["platform"], "none")
            self.assertEqual(second["seed"], 5)

    def test_run_condition_returns_seed_by_snapshot_metrics(self):
        def fake_run_cached(overrides, platform, seed, subfolder):
            x = np.array([
                [1.0, 1.0, -1.0, -1.0],
                [2.0, 0.0, 0.0, -2.0],
            ])
            return {
                "times": np.array([0.0, 1.0]),
                "x": x,
                "theta": np.array([0.5, 0.5, -0.5, -0.5]),
                "group": np.array([MEN, MEN, WOMEN, WOMEN]),
                "x_hat": None,
            }

        with patch.object(runner, "run_cached", side_effect=fake_run_cached):
            result = runner.run_condition(
                {"N": 4}, platform="none", seeds=[2, 4], subfolder="test"
            )

        self.assertEqual(result["gap"].shape, (2, 2))
        np.testing.assert_allclose(result["gap"], [[2.0, 2.0], [2.0, 2.0]])
        np.testing.assert_array_equal(result["times"], [0.0, 1.0])
        self.assertTrue(np.isnan(result["err_vs_x"]).all())


if __name__ == "__main__":
    unittest.main()
