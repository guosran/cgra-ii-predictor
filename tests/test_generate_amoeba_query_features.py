import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_amoeba_query_features",
    ROOT / "adapters" / "generate_amoeba_query_features.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AmoebaQueryFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.opt = self.root / "mlir-neura-opt"
        self.architecture = self.root / "architecture.yaml"
        self.opt.write_bytes(b"opt")
        self.architecture.write_text("architecture: test\n")
        self.dfg = self.root / "task.mlir"
        self.dfg.write_text("""module {
  func.func @task() {
    %0 = neura.counter -> !neura.data<index, i1>
    %1 = \"neura.data_mov\"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
    %2 = \"neura.add\"(%1) : (!neura.data<index, i1>) -> !neura.data<index, i1>
    %3 = \"neura.mul\"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
    neura.store_indexed %3 to [%0 : !neura.data<index, i1>] : !neura.data<index, i1>
    return
  }
}\n""")
        header = {
            "schema_version": "amoeba-analytical-task-candidates-v2",
            "record_type": "header", "function": "f",
            "search_scope": "static-shape-only-v2",
            "shape_policy": "static-rectangles-v2",
            "architecture": {"per_cgra_tile_rows": 4, "per_cgra_tile_cols": 4},
            "cost_queries": [
                {"task": "Task_0", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 4},
                {"task": "Task_0", "mapper_tile_rows": 4,
                 "mapper_tile_cols": 8},
            ],
        }
        candidates = []
        for index, cols in enumerate((4, 8)):
            candidates.append({
                "schema_version": "amoeba-analytical-task-candidates-v2",
                "record_type": "candidate", "candidate_id": f"c{index}",
                "task_shapes": [{
                    "task": "Task_0", "shape": {
                        "kind": "rect", "rows": 1, "cols": cols // 4,
                        "mapper_tile_rows": 4, "mapper_tile_cols": cols,
                    },
                }],
            })
        footer = {
            "schema_version": "amoeba-analytical-task-candidates-v2",
            "record_type": "footer", "candidate_count": 2,
        }
        self.manifest = self.root / "candidates.jsonl"
        self.manifest.write_text("\n".join(
            json.dumps(row) for row in [header, *candidates, footer]
        ) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_generates_every_query_and_frontend_depth(self):
        calls = []

        def runner(command):
            calls.append(command)
            output = Path(command[-1])
            cols = 8 if "y-tiles=4" in command[-3] and "x-tiles=8" in command[-3] else 4
            output.write_text(
                "module attributes {rec_res_mii_info = {"
                f"rec_mii = 1 : i32, res_mii = {cols // 4} : i32, "
                "num_nodes = 3 : i32, num_edges = 2 : i32, "
                "num_rec_edges = 0 : i32, num_mem_ops = 0 : i32, "
                "num_compute_ops = 3 : i32, num_phi_ops = 0 : i32, "
                "num_counter_ops = 1 : i32, num_regs = 8 : i32}} {}\n"
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        result, timing = MODULE.generate_query_features(
            self.manifest, {"Task_0": self.dfg}, self.opt,
            self.architecture, runner,
        )
        self.assertEqual(result["schema_version"], "cgra-ii-amoeba-query-features-v1")
        self.assertEqual(len(result["entries"]), 2)
        self.assertEqual([row["lower_bound"] for row in result["entries"]], [1, 2])
        self.assertEqual({row["startup_cycles"] for row in result["entries"]}, {4})
        self.assertEqual(timing["analysis_invocation_count"], 2)
        self.assertTrue(all("--analyze-rec-res-mii=" in call[-3] for call in calls))
        self.assertTrue(all("map-to-accelerator" not in " ".join(call) for call in calls))

    def test_analysis_failure_is_not_fabricated(self):
        def runner(command):
            return subprocess.CompletedProcess(command, 1, "", "bad analysis")

        with self.assertRaisesRegex(RuntimeError, "bad analysis"):
            MODULE.generate_query_features(
                self.manifest, {"Task_0": self.dfg}, self.opt,
                self.architecture, runner,
            )


if __name__ == "__main__":
    unittest.main()
