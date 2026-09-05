import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import torch

from adapters import amoeba_cost_catalog as adapter
from cgra_ii_predictor.graph_model import JointGraphShapeModel, Model2Config
from cgra_ii_predictor.shape_protocol import (
    AMOEBA_STATIC_PROTOCOL,
    AMOEBA_STATIC_SHAPE_PROTOCOL,
)


DFG_TEXT = """module {
  func.func @kernel() {
    %a = "neura.constant"() : () -> !neura.data<i32, i1>
    %b = "neura.constant"() : () -> !neura.data<i32, i1>
    %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    return
  }
}
"""


class AmoebaCostCatalogTest(unittest.TestCase):
    def make_inputs(self, root: Path, unsupported=False):
        tasks = ("Task_0", "Task_1")
        physical_mapper = list(AMOEBA_STATIC_PROTOCOL.physical_to_mapper)
        if unsupported:
            physical_mapper.append(((1, 5), (4, 20)))
        queries = [
            {"task": task, "mapper_tile_rows": mapper[0],
             "mapper_tile_cols": mapper[1]}
            for task in tasks for _, mapper in physical_mapper
        ]
        records = [{
            "record_type": "header",
            "schema_version": adapter.CANDIDATE_SCHEMA,
            "function": "parallel_nested_example",
            "search_scope": "static-shape-only-v2",
            "shape_policy": "static-rectangles-v2",
            "architecture": {
                "grid_rows": 5 if unsupported else 4,
                "grid_cols": 5 if unsupported else 4,
                "per_cgra_tile_rows": 4,
                "per_cgra_tile_cols": 4,
            },
            "cost_queries": queries,
        }]
        for index, choices in enumerate(itertools.product(
            physical_mapper, repeat=len(tasks),
        )):
            records.append({
                "record_type": "candidate",
                "schema_version": adapter.CANDIDATE_SCHEMA,
                "candidate_id": f"candidate-{index}",
                "task_shapes": [
                    {
                        "task": task,
                        "trip_count": 8,
                        "shape": {
                            "kind": "rect",
                            "rows": physical[0],
                            "cols": physical[1],
                            "mapper_tile_rows": mapper[0],
                            "mapper_tile_cols": mapper[1],
                        },
                    }
                    for task, (physical, mapper) in zip(tasks, choices)
                ],
            })
        records.append({
            "record_type": "footer",
            "schema_version": adapter.CANDIDATE_SCHEMA,
            "candidate_count": len(records) - 1,
        })
        manifest = root / "candidates.jsonl"
        manifest.write_text("".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        ))

        analytical_entries = []
        for query in queries:
            if (
                query["mapper_tile_rows"], query["mapper_tile_cols"]
            ) not in AMOEBA_STATIC_PROTOCOL.mapper_shapes:
                continue
            analytical_entries.append({
                **query,
                "rec_mii": 1,
                "res_mii": 2,
                "lower_bound": 2,
                "startup_cycles": 3,
            })
        analytical = root / "analytical.json"
        analytical.write_text(json.dumps({
            "schema_version": adapter.ANALYTICAL_INPUT_SCHEMA,
            "function": "parallel_nested_example",
            "entries": analytical_entries,
        }))

        task_paths = {}
        for task in tasks:
            path = root / f"{task}.mlir"
            path.write_text(DFG_TEXT.replace("@kernel", f"@{task}"))
            task_paths[task] = path
        analytical_payload = json.loads(analytical.read_text())
        analytical_payload["provenance"] = {
            "candidate_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "neura_opt_sha256": "1" * 64,
            "architecture_sha256": "2" * 64,
            "task_dfg_sha256": {
                task: hashlib.sha256(path.read_bytes()).hexdigest()
                for task, path in task_paths.items()
            },
            "rec_res_source": "test-analysis-only",
            "startup_cycles_source": "test-frontend",
        }
        analytical.write_text(json.dumps(analytical_payload))

        config = Model2Config(
            hidden_dimension=16,
            message_passing_layers=1,
            dropout=0.0,
            interaction_mode="residual_pointwise",
            shape_protocol=AMOEBA_STATIC_SHAPE_PROTOCOL,
        ).validate()
        model = JointGraphShapeModel(config)
        checkpoint = root / "model.pt"
        torch.save({
            "schema_version": "cgra-ii-joint-graph-model-v14",
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
        }, checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ensemble = root / "ensemble.json"
        ensemble.write_text(json.dumps({
            "schema_version": "cgra-ii-pointwise-convex-ensemble-v1",
            "selection_split": "validation_only",
            "manifest_sha256": "f" * 64,
            "selected_ensemble_mode": "static",
            "checkpoints": {
                "model": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "config": config.to_dict(),
                },
            },
            "weights": {"analytical_lower_bound": 0.25, "model": 0.75},
            "uncertainty_gating": {
                "exponent": 0.5,
                "scales": {
                    "analytical_lower_bound": 1.0,
                    "model": 1.0,
                },
            },
        }))
        return manifest, analytical, task_paths, {"model": checkpoint}, ensemble

    def test_applies_only_a_validation_selected_analytical_mean_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = list(self.make_inputs(root))
            analytical = json.loads(inputs[1].read_text())
            for entry in analytical["entries"]:
                entry.update({"rec_mii": 2, "res_mii": 1, "lower_bound": 2})
            inputs[1].write_text(json.dumps(analytical))
            checkpoint = inputs[3]["model"]
            hybrid = root / "hybrid.json"
            hybrid.write_text(json.dumps({
                "schema_version": "cgra-ii-pointwise-hybrid-analysis-v1",
                "selection_split": "validation_only",
                "manifest_sha256": "f" * 64,
                "checkpoints": {
                    "model": {
                        "path": str(checkpoint),
                        "sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    },
                },
                "ensemble_weights": {
                    "analytical_lower_bound": 0.25, "model": 0.75,
                },
                "uncertainty_exponent": 0.5,
                "validation": {
                    "ensemble": {"mae": 0.5},
                    "hybrid_rules": {
                        "rec_gt_res": {
                            "rule": {"kind": "rec_gt_res"},
                            "hybrid": {"mae": 0.4},
                            "mae_improvement_over_ensemble": 0.1,
                        },
                    },
                },
            }))
            catalog, _ = adapter.generate_catalog(
                *inputs, torch.device("cpu"), "selected", hybrid,
            )
        self.assertEqual(
            catalog["predictor_metadata"]["analytical_hybrid"][
                "selected_rule"
            ],
            {"kind": "rec_gt_res"},
        )
        self.assertTrue(all(
            entry["predicted_ii"] == 2 and
            entry["ii_mean_source"] == (
                "validation_selected_analytical_fallback"
            )
            for entry in catalog["entries"]
        ))

    def test_generates_scorer_compatible_catalog_once_per_unique_query(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(Path(directory))
            catalog, timing = adapter.generate_catalog(
                *inputs, torch.device("cpu"), "selected",
            )
        self.assertEqual(catalog["schema_version"], adapter.COST_SCHEMA)
        self.assertEqual(catalog["function"], "parallel_nested_example")
        self.assertTrue(catalog["namespace"].startswith("cgra-ii-v8-"))
        self.assertEqual(len(catalog["entries"]), 16)
        self.assertTrue(all(
            entry["support_status"] == "supported" and
            entry["predicted_ii"] >= entry["analytical_lower_bound"] and
            entry["startup_cycles"] == 3 and
            0.0 <= entry["mapper_success_probability"] <= 1.0
            for entry in catalog["entries"]
        ))
        self.assertEqual(timing["manifest_candidate_count"], 64)
        self.assertEqual(timing["unique_cost_query_count"], 16)
        self.assertEqual(timing["unique_cache_entry_count"], 16)
        self.assertEqual(timing["task_dfg_parse_count"], 2)
        self.assertEqual(timing["model_load_count"], 1)
        self.assertEqual(timing["model_forward_pass_count"], 8)
        self.assertEqual(timing["ensemble_mode"], "static")

    def test_tasks_may_expose_different_supported_shape_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = list(self.make_inputs(root))
            records = [
                json.loads(line) for line in inputs[0].read_text().splitlines()
            ]
            records[0]["cost_queries"] = [
                query for query in records[0]["cost_queries"]
                if not (
                    query["task"] == "Task_1" and
                    query["mapper_tile_rows"] == 16 and
                    query["mapper_tile_cols"] == 4
                )
            ]
            body = [
                record for record in records[1:-1]
                if not any(
                    choice["task"] == "Task_1" and
                    choice["shape"]["mapper_tile_rows"] == 16 and
                    choice["shape"]["mapper_tile_cols"] == 4
                    for choice in record["task_shapes"]
                )
            ]
            records = [records[0], *body, {
                **records[-1], "candidate_count": len(body),
            }]
            inputs[0].write_text("".join(
                json.dumps(record) + "\n" for record in records
            ))
            analytical = json.loads(inputs[1].read_text())
            analytical["provenance"]["candidate_manifest_sha256"] = (
                hashlib.sha256(inputs[0].read_bytes()).hexdigest()
            )
            analytical["entries"] = [
                entry for entry in analytical["entries"]
                if not (
                    entry["task"] == "Task_1" and
                    entry["mapper_tile_rows"] == 16 and
                    entry["mapper_tile_cols"] == 4
                )
            ]
            inputs[1].write_text(json.dumps(analytical))
            catalog, timing = adapter.generate_catalog(
                *inputs, torch.device("cpu"), "static",
            )
        self.assertEqual(len(catalog["entries"]), 15)
        self.assertEqual(timing["supported_query_count"], 15)
        self.assertEqual(timing["task_dfg_parse_count"], 2)
        self.assertEqual(timing["model_forward_pass_count"], 8)

    def test_out_of_domain_shape_is_explicitly_unsupported_without_numeric_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.make_inputs(Path(directory), unsupported=True)
            catalog, timing = adapter.generate_catalog(
                *inputs, torch.device("cpu"), "static",
            )
        unsupported = [
            entry for entry in catalog["entries"]
            if entry["support_status"] == "unsupported"
        ]
        self.assertEqual(len(unsupported), 2)
        self.assertTrue(all("predicted_ii" not in entry for entry in unsupported))
        self.assertTrue(all("startup_cycles" not in entry for entry in unsupported))
        self.assertEqual(timing["unsupported_query_count"], 2)

    def test_rejects_frontend_startup_or_lower_bound_fabrication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = list(self.make_inputs(root))
            raw = json.loads(inputs[1].read_text())
            raw["entries"][0]["startup_cycles"] = 0
            inputs[1].write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "startup_cycles"):
                adapter.generate_catalog(
                    *inputs, torch.device("cpu"), "static",
                )

    def test_rejects_analytical_features_for_a_different_dfg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = list(self.make_inputs(root))
            inputs[2]["Task_0"].write_text(DFG_TEXT + "// changed\n")
            with self.assertRaisesRegex(ValueError, "task DFG SHA-256"):
                adapter.generate_catalog(
                    *inputs, torch.device("cpu"), "static",
                )

    def test_rejects_non_static_candidate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = list(self.make_inputs(root))
            lines = [json.loads(line) for line in inputs[0].read_text().splitlines()]
            lines[0]["search_scope"] = "dynamic-shape-v1"
            inputs[0].write_text("".join(json.dumps(row) + "\n" for row in lines))
            with self.assertRaisesRegex(ValueError, "only static"):
                adapter.generate_catalog(
                    *inputs, torch.device("cpu"), "static",
                )


if __name__ == "__main__":
    unittest.main()
