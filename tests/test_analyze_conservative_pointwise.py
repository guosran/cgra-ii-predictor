import unittest

import numpy as np

from adapters.analyze_conservative_pointwise import (
    fit_alpha,
    pinball_loss,
    point_metrics,
)


class ConservativePointwiseTest(unittest.TestCase):
    def test_pinball_penalizes_underprediction_more_at_high_quantile(self):
        target = np.asarray([2.0])
        under = pinball_loss(np.asarray([1.0]), target, 0.75)
        over = pinball_loss(np.asarray([3.0]), target, 0.75)
        self.assertEqual(under, 0.75)
        self.assertEqual(over, 0.25)

    def test_fit_alpha_uses_validation_targets_only(self):
        means = np.asarray([1.0, 1.0])
        targets = np.asarray([2.0, 2.0])
        expert_means = np.asarray([1.0, 1.0])
        expert_stds = np.asarray([1.0, 1.0])
        alpha, records = fit_alpha(
            means, targets, expert_means, expert_stds,
            0.75, (0.0, 0.5, 1.0, 1.5),
        )
        self.assertEqual(alpha, 1.0)
        self.assertEqual(len(records), 4)

    def test_conservative_metrics_report_underprediction_magnitude(self):
        metrics = point_metrics(
            np.asarray([1.0, 4.0]), np.asarray([3.0, 3.0]),
        )
        self.assertEqual(metrics["underprediction_rate"], 0.5)
        self.assertEqual(metrics["mean_underprediction"], 1.0)
        self.assertEqual(metrics["maximum_underprediction"], 2.0)


if __name__ == "__main__":
    unittest.main()
