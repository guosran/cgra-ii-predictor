import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "enumerate_amoeba_pruned_candidates",
    ROOT / "adapters" / "enumerate_amoeba_pruned_candidates.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def dfg(operation_count):
    operations = "\n".join(
        f'      %{index} = "neura.add"() : () -> !neura.data<i32, i1>'
        for index in range(operation_count)
    )
    return f"""module {{
  func.func @f() {{
    neura.kernel attributes {{accelerator = "neura"}} {{
{operations}
      neura.data_mov %0 : !neura.data<i32, i1>
      neura.yield
    }}
    return
  }}
}}
"""


def base_manifest(tasks, rows=2, cols=2):
    return {
        "header": {
            "record_type": "header",
            "schema_version": "amoeba-analytical-task-candidates-v2",
            "search_scope": "static-shape-only-v2",
            "shape_policy": "static-rectangles-v2",
            "function": "f",
            "architecture": {
                "grid_rows": rows,
                "grid_cols": cols,
                "per_cgra_tile_rows": 4,
                "per_cgra_tile_cols": 4,
            },
            "tasks": [
                {"task": task, "trip_count": index + 1}
                for index, task in enumerate(tasks)
            ],
            "fixed_axes": {},
        }
    }


class PrunedCandidateTests(unittest.TestCase):
    def test_counts_only_materialized_kernel_operations(self):
        text = dfg(3).replace(
            "func.func @f()",
            "func.func @f(%arg: !neura.data<i32, i1>)",
        )
        self.assertEqual(MODULE.materialized_operation_count(text), 3)

    def test_op_count_caps_shape_area(self):
        records, report = MODULE.enumerate_pruned_records(
            base_manifest(["small", "large"]),
            {"small": dfg(16), "large": dfg(17)},
        )
        self.assertEqual(report["maximum_physical_cgras_by_task"], {
            "small": 1,
            "large": 2,
        })
        self.assertEqual(report["shape_counts_by_task"], {
            "small": 1,
            "large": 3,
        })
        self.assertEqual(report["unpruned_cartesian_count"], 16)
        self.assertEqual(report["op_capped_cartesian_count"], 3)
        self.assertEqual(report["allowed_shapes_by_task"], {
            "small": ["1x1"],
            "large": ["1x1", "1x2", "2x1"],
        })
        self.assertEqual(records[-1]["candidate_count"], 3)

    def test_incremental_packing_removes_overlapping_area(self):
        records, report = MODULE.enumerate_pruned_records(
            base_manifest(["a", "b"], rows=1, cols=2),
            {"a": dfg(17), "b": dfg(17)},
        )
        self.assertEqual(report["raw_unpacked_cartesian_count"], 4)
        self.assertEqual(report["grid_packable_candidate_count"], 1)
        self.assertEqual(records[-1]["candidate_count"], 1)
        self.assertEqual(
            [choice["shape"]["cgra_shape"] for choice in records[1]["task_shapes"]],
            ["1x1", "1x1"],
        )


if __name__ == "__main__":
    unittest.main()
