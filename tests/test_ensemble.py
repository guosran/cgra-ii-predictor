import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "ensemble adapter requires PyTorch")
class ConvexEnsembleTest(unittest.TestCase):
    def setUp(self):
        global apply_uncertainty_gating, fit_convex_mae
        global fit_uncertainty_gating
        from adapters.ensemble_pointwise_checkpoints import (
            apply_uncertainty_gating, fit_convex_mae,
            fit_uncertainty_gating,
        )

    def test_finds_exact_interior_blend(self):
        predictions = np.asarray([[0.0, 2.0], [2.0, 0.0], [4.0, 6.0]])
        targets = np.asarray([1.0, 1.0, 5.0])
        weights = fit_convex_mae(predictions, targets)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(predictions @ weights, targets, atol=1e-12)

    def test_selects_a_dominant_component(self):
        predictions = np.asarray([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
        targets = np.asarray([1.0, 2.0, 3.0])
        weights = fit_convex_mae(predictions, targets)
        np.testing.assert_allclose(weights, [1.0, 0.0], atol=1e-12)

    def test_two_component_solution_is_no_worse_than_dense_grid(self):
        predictions = np.asarray([
            [0.2, 2.3], [1.7, 0.5], [4.1, 5.0], [8.0, 5.2], [3.3, 3.8],
        ])
        targets = np.asarray([1.0, 1.2, 4.7, 5.9, 3.7])
        weights = fit_convex_mae(predictions, targets)
        grid = np.linspace(0.0, 1.0, 10001)
        grid_losses = np.mean(np.abs(
            predictions[:, :1] * grid[None, :] +
            predictions[:, 1:] * (1.0 - grid[None, :]) -
            targets[:, None]
        ), axis=0)
        fitted_loss = np.mean(np.abs(predictions @ weights - targets))
        self.assertLessEqual(fitted_loss, float(grid_losses.min()) + 1e-12)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            fit_convex_mae(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
        with self.assertRaisesRegex(ValueError, "one value"):
            fit_convex_mae(np.ones((2, 2)), np.ones(3))
        with self.assertRaisesRegex(ValueError, "at least one"):
            fit_convex_mae(np.empty((2, 0)), np.ones(2))

    def test_uncertainty_gate_favors_relative_confidence(self):
        predictions = np.asarray([[10.0, 2.0, 8.0], [10.0, 3.0, 7.0]])
        uncertainties = np.asarray([[1.0, 0.1, 2.0], [1.0, 0.2, 2.0]])
        weights = np.asarray([0.0, 0.5, 0.5])
        scales = np.asarray([1.0, 1.0, 1.0])
        static = apply_uncertainty_gating(
            predictions, uncertainties, weights, 0.0, scales,
        )
        gated = apply_uncertainty_gating(
            predictions, uncertainties, weights, 2.0, scales,
        )
        np.testing.assert_allclose(static, [5.0, 5.0])
        self.assertTrue(np.all(np.abs(gated - predictions[:, 1]) < 0.1))

    def test_uncertainty_fit_uses_validation_relative_scale(self):
        predictions = np.asarray([
            [9.0, 1.0, 5.0], [9.0, 5.0, 2.0],
            [9.0, 3.0, 8.0], [9.0, 8.0, 4.0],
        ])
        targets = np.asarray([1.0, 2.0, 3.0, 4.0])
        uncertainties = np.asarray([
            [1.0, 0.1, 2.0], [1.0, 2.0, 0.1],
            [1.0, 0.1, 2.0], [1.0, 2.0, 0.1],
        ])
        base_weights = np.asarray([0.0, 0.5, 0.5])
        exponent, scales = fit_uncertainty_gating(
            predictions, targets, uncertainties, base_weights,
        )
        self.assertGreater(exponent, 0.0)
        np.testing.assert_allclose(scales, [1.0, 1.05, 1.05])
        static = apply_uncertainty_gating(
            predictions, uncertainties, base_weights, 0.0, scales,
        )
        gated = apply_uncertainty_gating(
            predictions, uncertainties, base_weights, exponent, scales,
        )
        self.assertLess(
            np.mean(np.abs(gated - targets)),
            np.mean(np.abs(static - targets)),
        )


@unittest.skipUnless(TORCH_AVAILABLE, "latency adapter requires PyTorch")
class LatencySummaryTest(unittest.TestCase):
    def test_percentile_interpolates_and_validates_inputs(self):
        from adapters.benchmark_pointwise_latency import percentile

        self.assertEqual(percentile([1.0, 3.0], 0.0), 1.0)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)
        self.assertEqual(percentile([1.0, 3.0], 1.0), 3.0)
        with self.assertRaisesRegex(ValueError, "empty"):
            percentile([], 0.5)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            percentile([1.0], 1.1)


@unittest.skipUnless(TORCH_AVAILABLE, "ensemble adapter requires PyTorch")
class EnsembleArtifactTest(unittest.TestCase):
    def test_checkpoint_sha256_is_content_addressed(self):
        from adapters.ensemble_pointwise_checkpoints import sha256_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint")
            self.assertEqual(
                sha256_file(path),
                "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
            )


if __name__ == "__main__":
    unittest.main()
