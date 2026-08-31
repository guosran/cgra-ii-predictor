import unittest

from cgra_ii_predictor.dataset import Sample
from cgra_ii_predictor.model import (
    fit_ridge,
    nested_group_holdout,
    predict_ridge,
)


def sample(name, group, lower_bound, compiled_ii, pressure):
    return Sample(
        sample_id=name,
        group=group,
        lower_bound=lower_bound,
        compiled_ii=compiled_ii,
        features={"pressure": pressure, "depth": pressure + 1},
        metadata={},
    )


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            sample("a0", "a", 3, 3, 1),
            sample("a1", "a", 3, 3, 1.2),
            sample("b0", "b", 4, 6, 3),
            sample("b1", "b", 4, 6, 3.2),
            sample("c0", "c", 5, 8, 5),
            sample("c1", "c", 5, 8, 5.2),
        ]

    def test_prediction_never_falls_below_bound(self):
        model = fit_ridge(self.samples, ["pressure", "depth"], ridge=1.0)
        for row in self.samples:
            self.assertGreaterEqual(predict_ridge(model, row), row.lower_bound)

    def test_nested_holdout_keeps_groups_intact(self):
        result = nested_group_holdout(
            self.samples,
            ["pressure", "depth"],
            ridge_candidates=[0.3, 1.0, 3.0],
            dead_zone_candidates=[0.0, 0.5, 1.0],
        )
        self.assertEqual(result["groups"], ["a", "b", "c"])
        self.assertEqual(len(result["rows"]), len(self.samples))


if __name__ == "__main__":
    unittest.main()

