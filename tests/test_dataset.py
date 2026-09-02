import json
import math
import tempfile
import unittest
from pathlib import Path

from cgra_ii_predictor.dataset import (
    Dataset,
    Sample,
    load_dataset,
    regroup_by_metadata,
    remap_groups,
)


def contract(lower_bound):
    return {
        "rec_mii": lower_bound,
        "res_mii": 1,
        "lower_bound_source": "rec_res_max_v1",
    }


class DatasetTest(unittest.TestCase):
    def test_loads_portable_schema(self):
        path = Path(__file__).parents[1] / "schema/example-dataset.json"
        dataset = load_dataset(path)
        self.assertEqual(len(dataset.samples), 3)
        self.assertEqual(dataset.feature_names, ("pressure", "semantic_depth"))

    def test_rejects_compiled_ii_below_bound(self):
        raw = {
            "feature_names": ["x"],
            "samples": [{
                "sample_id": "bad", "group": "bad", "lower_bound": 4,
                "compiled_ii": 3, "features": {"x": 1},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_rejects_label_hidden_in_nested_features(self):
        raw = {
            "feature_names": ["x", "compiled_ii"],
            "samples": [{
                "sample_id": "leaky", "group": "g", "lower_bound": 1,
                "compiled_ii": 2,
                "features": {"x": 1, "compiled_ii": 2},
                **contract(1),
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaky.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "reserved feature"):
                load_dataset(path)

    def test_rejects_non_finite_and_fractional_ii_values(self):
        cases = (
            (math.nan, 2),
            (1, math.inf),
            (0, 2),
            (1, 2.5),
        )
        for lower_bound, compiled_ii in cases:
            with self.subTest(lower_bound=lower_bound, compiled_ii=compiled_ii):
                raw = {
                    "feature_names": ["x"],
                    "samples": [{
                        "sample_id": "bad", "group": "g",
                        "lower_bound": lower_bound,
                        "compiled_ii": compiled_ii,
                        "features": {"x": 1},
                    }],
                }
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bad.json"
                    path.write_text(json.dumps(raw))
                    with self.assertRaises(ValueError):
                        load_dataset(path)

    def test_rejects_non_finite_feature_and_inconsistent_bound_alias(self):
        rows = (
            {"x": float("inf")},
            {"x": 1, "baseline_lb": 2},
        )
        for features in rows:
            with self.subTest(features=features):
                raw = {
                    "feature_names": list(features),
                    "samples": [{
                        "sample_id": "bad", "group": "g",
                        "lower_bound": 1, "compiled_ii": 2,
                        "features": features,
                        **contract(1),
                    }],
                }
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "bad.json"
                    path.write_text(json.dumps(raw))
                    with self.assertRaises(ValueError):
                        load_dataset(path)

    def test_rejects_duplicate_sample_ids(self):
        base = {
            "group": "g", "lower_bound": 1, "compiled_ii": 2,
            "features": {"x": 1},
            "metadata": {"candidate_id": "candidate"},
            **contract(1),
        }
        raw = {
            "feature_names": ["x"],
            "samples": [
                dict(base, sample_id="same"), dict(base, sample_id="same"),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_dataset(path)

    def test_candidate_ids_are_scoped_to_one_ranking_query(self):
        def row(sample_id, query, compiled_ii=2):
            return {
                "sample_id": sample_id, "group": "g", "lower_bound": 1,
                "compiled_ii": compiled_ii, "features": {"x": 1},
                **contract(1),
                "metadata": {
                    "ranking_query_id": query, "candidate_id": "mesh-a",
                    "architecture_id": "mesh-a",
                },
            }

        # Architecture-style candidate names may recur across DFGs, and exact
        # repeated observations within a DFG are allowed for later collapsing.
        accepted = {
            "feature_names": ["x"],
            "samples": [row("a", "dfg-a"), row("b", "dfg-b"),
                        row("a-copy", "dfg-a")],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.json"
            path.write_text(json.dumps(accepted))
            self.assertEqual(len(load_dataset(path).samples), 3)

        conflicting = {
            "feature_names": ["x"],
            "samples": [row("a", "dfg-a"), row("b", "dfg-a", 3)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflicting.json"
            path.write_text(json.dumps(conflicting))
            with self.assertRaisesRegex(ValueError, "conflicting records"):
                load_dataset(path)

    def test_flat_provenance_is_metadata_not_a_numeric_feature(self):
        raw = {
            "samples": [{
                "index": "legacy", "family": "legacy-family",
                "baseline_lb": 2, "compiled_ii": 3,
                **contract(2),
                "pressure": 1.5,
                "lineage": "source-lineage",
                "candidate_id": 123,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(raw))
            dataset = load_dataset(path)
        self.assertIn("pressure", dataset.feature_names)
        self.assertNotIn("candidate_id", dataset.feature_names)
        self.assertEqual(dataset.samples[0].metadata["lineage"], "source-lineage")
        self.assertEqual(dataset.samples[0].metadata["candidate_id"], 123)

    def test_conflicting_flat_and_nested_provenance_is_rejected(self):
        raw = {
            "feature_names": ["x"],
            "samples": [{
                "sample_id": "bad", "group": "g", "lower_bound": 1,
                "compiled_ii": 2, "features": {"x": 1},
                **contract(1),
                "lineage": "flat", "metadata": {"lineage": "nested"},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "conflicting flat"):
                load_dataset(path)

    def test_inconsistent_base_dfg_and_ranking_query_are_rejected(self):
        raw = {
            "feature_names": ["x"],
            "samples": [{
                "sample_id": "bad", "group": "g", "lower_bound": 1,
                "compiled_ii": 2, "features": {"x": 1},
                **contract(1),
                "metadata": {
                    "base_dfg_id": "dfg-a",
                    "ranking_query_id": "dfg-b",
                    "candidate_id": "mesh-a",
                },
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-query.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "inconsistent base_dfg"):
                load_dataset(path)

    def test_rec_res_contract_is_required_and_exact(self):
        base = {
            "feature_names": ["x"],
            "samples": [{
                "sample_id": "sample", "group": "g", "lower_bound": 2,
                "compiled_ii": 3, "features": {"x": 1},
            }],
        }
        cases = (
            (base, "needs rec_mii and res_mii"),
            ({
                **base,
                "samples": [{**base["samples"][0], **contract(3)}],
            }, "must equal max"),
        )
        for raw, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid-contract.json"
                path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, message):
                    load_dataset(path)

    def test_bound_and_components_cannot_be_selected_as_features(self):
        for name in ("lower_bound", "baseline_lb", "rec_mii", "res_mii"):
            raw = {
                "feature_names": [name],
                "samples": [{
                    "sample_id": "sample", "group": "g", "lower_bound": 2,
                    "compiled_ii": 3, "features": {name: 2},
                    **contract(2),
                }],
            }
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "forbidden-feature.json"
                path.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "reserved feature"):
                    load_dataset(path)

    def test_explicit_group_alias_preserves_original_group(self):
        path = Path(__file__).parents[1] / "schema/example-dataset.json"
        dataset = remap_groups(
            load_dataset(path), {"suite-a/kernel-a": "suite-a/lineage"}
        )
        self.assertEqual(dataset.samples[0].group, "suite-a/lineage")
        self.assertEqual(
            dataset.samples[0].metadata["source_group"], "suite-a/kernel-a"
        )
        self.assertEqual(
            dataset.samples[0].metadata["lineage"], "suite-a/kernel-a"
        )
        self.assertEqual(
            dataset.samples[0].metadata["leakage_lineage_id"],
            dataset.samples[0].group,
        )

    def test_declared_lineage_is_default_group_and_architecture_regroup_keeps_it(self):
        sample = Sample(
            sample_id="a", group="legacy-a", lower_bound=1, compiled_ii=2,
            features={"x": 1.0},
            metadata={"lineage": "common-source", "architecture_id": "mesh"},
        )
        dataset = Dataset((sample,), ("x",), {})
        lineage_dataset = remap_groups(dataset, {})
        self.assertEqual(lineage_dataset.samples[0].group, "common-source")
        architecture_dataset = regroup_by_metadata(
            lineage_dataset, "architecture_id"
        )
        self.assertEqual(architecture_dataset.samples[0].group, "mesh")
        self.assertEqual(
            architecture_dataset.samples[0].metadata["training_weight_group"],
            "common-source",
        )

    def test_explicit_leakage_lineage_is_authoritative_and_aliasable(self):
        sample = Sample(
            "a", "legacy-group", 1, 2, {"x": 1.0},
            {"lineage": "source", "leakage_lineage_id": "declared"},
        )
        dataset = Dataset((sample,), ("x",), {})
        remapped = remap_groups(dataset, {"declared": "merged"})
        self.assertEqual(remapped.samples[0].group, "merged")
        self.assertEqual(
            remapped.samples[0].metadata["declared_leakage_lineage_id"],
            "declared",
        )
        self.assertEqual(
            remapped.samples[0].metadata["leakage_lineage_id"], "merged"
        )


if __name__ == "__main__":
    unittest.main()
