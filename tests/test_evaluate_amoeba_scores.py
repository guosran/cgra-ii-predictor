import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_amoeba_scores",
    ROOT / "adapters" / "evaluate_amoeba_scores.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateAmoebaScoresTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        schema = "amoeba-analytical-task-candidates-v2"
        header = {
            "schema_version": schema, "record_type": "header",
            "function": "f", "search_scope": "static-shape-only-v2",
            "shape_policy": "static-rectangles-v2",
            "architecture": {"per_cgra_tile_rows": 4, "per_cgra_tile_cols": 4},
            "cost_queries": [
                {"task": "T", "mapper_tile_rows": 4, "mapper_tile_cols": 4},
                {"task": "T", "mapper_tile_rows": 4, "mapper_tile_cols": 8},
            ],
        }
        candidates = []
        for index, cols in enumerate((4, 8)):
            candidates.append({
                "schema_version": schema, "record_type": "candidate",
                "candidate_id": f"c{index}", "task_shapes": [{
                    "task": "T", "trip_count": 4,
                    "shape": {"kind": "rect", "rows": 1,
                              "cols": cols // 4, "mapper_tile_rows": 4,
                              "mapper_tile_cols": cols},
                }],
            })
        footer = {"schema_version": schema, "record_type": "footer",
                  "candidate_count": 2}
        self.manifest = self.root / "candidates.jsonl"
        self.manifest.write_text("\n".join(json.dumps(row) for row in
            [header, *candidates, footer]) + "\n")
        score_schema = "amoeba-analytical-task-scores-v2"
        score_header = {"schema_version": score_schema, "record_type": "header",
                        "function": "f"}
        score_rows = []
        for index, cols in enumerate((4, 8)):
            score_rows.append({
                "schema_version": score_schema, "record_type": "score",
                "candidate_id": f"c{index}", "valid": True,
                "task_costs": [{"task": "T", "mapper_tile_rows": 4,
                                "mapper_tile_cols": cols, "startup_cycles": 2}],
            })
        score_footer = {
            "schema_version": score_schema, "record_type": "footer",
            "candidate_count": 2, "scored_count": 2, "valid_count": 2,
            "cache": {"entries": 2, "misses": 2, "hits": 0},
            "shortlist": [{"candidate_id": "c0", "rank": 0}],
        }
        self.scores = self.root / "scores.jsonl"
        self.scores.write_text("\n".join(json.dumps(row) for row in
            [score_header, *score_rows, score_footer]) + "\n")
        self.oracle = self.root / "oracle.json"
        self.oracle.write_text(json.dumps({
            "schema_version": "cgra-ii-amoeba-query-oracle-v1",
            "function": "f", "entries": [
                {"task": "T", "mapper_tile_rows": 4, "mapper_tile_cols": 4,
                 "status": "success", "compiled_ii": 3},
                {"task": "T", "mapper_tile_rows": 4, "mapper_tile_cols": 8,
                 "status": "success", "compiled_ii": 1},
            ],
        }))

    def tearDown(self):
        self.temporary.cleanup()

    def test_reports_oracle_regret_and_call_reduction(self):
        report = MODULE.evaluate_scores(self.manifest, self.scores, self.oracle)
        self.assertEqual(report["oracle"]["optimal_candidate_ids"], ["c1"])
        self.assertEqual(report["selection"]["top1_oracle_recall"], 0.0)
        self.assertEqual(report["selection"]["absolute_objective_regret_cycles"], 6.0)
        self.assertEqual(
            report["mapper_replay"]["projected_call_reduction_fraction"], 0.5,
        )
        self.assertEqual(
            report["mapper_replay"]["status"],
            "blocked_materialized_shape_not_preserved",
        )
        self.assertIsNone(
            report["mapper_replay"]["actual_call_reduction_fraction"]
        )

    def test_reports_actual_reduction_only_for_faithful_replay(self):
        report = MODULE.evaluate_scores(
            self.manifest, self.scores, self.oracle,
            replay_status="faithful",
        )
        self.assertEqual(report["mapper_replay"]["status"], "faithful")
        self.assertEqual(
            report["mapper_replay"]["actual_call_reduction_fraction"], 0.5,
        )

    def test_rejects_incomplete_score_coverage(self):
        rows = self.scores.read_text().splitlines()
        self.scores.write_text("\n".join([rows[0], rows[1], rows[-1]]) + "\n")
        with self.assertRaisesRegex(ValueError, "footer count"):
            MODULE.evaluate_scores(self.manifest, self.scores, self.oracle)


if __name__ == "__main__":
    unittest.main()
