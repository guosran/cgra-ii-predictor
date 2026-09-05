import json
import tempfile
import unittest
from pathlib import Path

from adapters.apply_conservative_amoeba_policy import apply_policy


def catalog(name, sha, mean, std):
    return {
        "schema_version": "amoeba-task-shape-cost-v2",
        "function": "kernel",
        "namespace": name,
        "predictor_metadata": {
            "checkpoints": {name: {"sha256": sha}},
        },
        "entries": [{
            "task": "Task_0", "mapper_tile_rows": 4,
            "mapper_tile_cols": 4, "support_status": "supported",
            "predicted_ii": mean, "predicted_ii_std": std,
            "ii_mean_source": "pointwise_ensemble",
        }],
    }


class ConservativeAmoebaPolicyTest(unittest.TestCase):
    @staticmethod
    def _policy():
        return {
            "schema_version": "cgra-ii-conservative-pointwise-v1",
            "selection_split": "validation_only",
            "policy": "max_ensemble_mean_and_expert_mean_plus_alpha_std",
            "selected_alpha": 0.5,
            "quantile": 0.85,
            "checkpoints": {"point": {"sha256": "point-sha"}},
            "expert": {"sha256": "expert-sha"},
        }

    def test_attaches_upper_without_replacing_default_point_score(self):
        point = catalog("point", "point-sha", 4.0, 0.2)
        expert = catalog("expert", "expert-sha", 5.0, 2.0)
        policy = self._policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("point.json", "expert.json", "policy.json")]
            for path, value in zip(paths, (point, expert, policy)):
                path.write_text(json.dumps(value))
            output = apply_policy(*paths)
        row = output["entries"][0]
        self.assertEqual(row["point_predicted_ii"], 4.0)
        self.assertEqual(row["predicted_ii"], 4.0)
        self.assertEqual(row["predicted_ii_upper"], 6.0)
        self.assertEqual(row["ii_mean_source"], "pointwise_ensemble")
        metadata = output["predictor_metadata"]["conservative_policy"]
        self.assertEqual(metadata["role"], "mapper_replay_trigger_only")
        self.assertFalse(metadata["upper_applied_to_predicted_ii"])

    def test_can_explicitly_use_upper_for_program_scoring(self):
        point = catalog("point", "point-sha", 4.0, 0.2)
        expert = catalog("expert", "expert-sha", 5.0, 2.0)
        policy = self._policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("point.json", "expert.json", "policy.json")
            ]
            for path, value in zip(paths, (point, expert, policy)):
                path.write_text(json.dumps(value))
            output = apply_policy(*paths, use_upper_for_scoring=True)
        row = output["entries"][0]
        self.assertEqual(row["predicted_ii"], 6.0)
        self.assertEqual(row["ii_mean_source"],
                         "validation-selected-conservative-upper-v1")
        metadata = output["predictor_metadata"]["conservative_policy"]
        self.assertEqual(
            metadata["role"], "program_scoring_and_mapper_replay_trigger"
        )
        self.assertTrue(metadata["upper_applied_to_predicted_ii"])

    def test_rejects_invalid_quantile(self):
        point = catalog("point", "point-sha", 4.0, 0.2)
        expert = catalog("expert", "expert-sha", 5.0, 2.0)
        policy = self._policy()
        policy["quantile"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("point.json", "expert.json", "policy.json")
            ]
            for path, value in zip(paths, (point, expert, policy)):
                path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "quantile"):
                apply_policy(*paths)


if __name__ == "__main__":
    unittest.main()
