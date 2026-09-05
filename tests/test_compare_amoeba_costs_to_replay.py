import json
import tempfile
import unittest
from pathlib import Path

from adapters.compare_amoeba_costs_to_replay import build_report


class CompareAmoebaCostsToReplayTest(unittest.TestCase):
    def test_reports_unique_query_and_program_errors(self):
        costs = {
            "entries": [
                {"task": "Task_0", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4, "predicted_ii": 6.5},
                {"task": "Task_1", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4, "predicted_ii": 3.0},
            ]
        }
        replay = {
            "mapper_replay": {"queries": [
                {"task": "Task_0", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4, "compiled_ii": 7,
                 "status": "success"},
                {"task": "Task_0", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4, "compiled_ii": 7,
                 "status": "success"},
                {"task": "Task_1", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4, "compiled_ii": 4,
                 "status": "success"},
            ]},
            "selection": {"candidates": [{
                "candidate_id": "candidate-0", "actual_valid": True,
                "actual_compute_bottleneck": 64.0,
                "task_results": [
                    {"task": "Task_0", "mapper_tile_rows": 4,
                     "mapper_tile_cols": 4, "startup_cycles": 1,
                     "trip_count": 10},
                    {"task": "Task_1", "mapper_tile_rows": 4,
                     "mapper_tile_cols": 4, "startup_cycles": 2,
                     "trip_count": 3},
                ],
            }]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost_path = root / "costs.json"
            replay_path = root / "replay.json"
            cost_path.write_text(json.dumps(costs))
            replay_path.write_text(json.dumps(replay))
            report = build_report(cost_path, replay_path)
        self.assertEqual(report["successful_query_count"], 2)
        self.assertEqual(report["ii_metrics"]["mae"], 0.75)
        self.assertEqual(report["ii_metrics"]["mean_signed_error"], -0.75)
        self.assertEqual(report["ii_metrics"]["underprediction_rate"], 1.0)
        self.assertEqual(report["program"]["predicted_objective_cycles"], 59.5)
        self.assertEqual(report["program"]["signed_error_cycles"], -4.5)


if __name__ == "__main__":
    unittest.main()
