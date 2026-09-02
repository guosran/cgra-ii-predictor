import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from cgra_ii_predictor.predict import (
    build_prediction_report,
    canonical_model_sha256,
    load_model_artifact,
    load_prediction_samples,
    parse_prediction_sample,
    predict_sample,
    validate_model,
)


def model():
    return {
        "model_type": "residual_ridge",
        "feature_names": ["x"],
        "mean": [0.0],
        "scale": [1.0],
        "weights": [0.5, 1.0],
        "ridge": 1.0,
        "residual_dead_zone": 1.0,
        "unseen_group_absolute_error_radius": 2.0,
        "unseen_group_interval_empirical_quantile": 0.9,
    }


def candidate(x=2.0):
    return {
        "sample_id": "kernel/4x4",
        "lower_bound": 5,
        "rec_mii": 5,
        "res_mii": 3,
        "features": {"x": x},
        "metadata": {
            "lower_bound_source": "rec_res_max_v1",
            "source_sha256": "source-hash",
            "architecture_id": "mesh-4x4",
            "mapper_id": "neura-heuristic",
            "mapper_revision": "revision-a",
            "mapper_config": "mapping-strategy=heuristic",
        },
    }


def frozen_v2_artifact(model_value, hashes):
    hashes = sorted(hashes)
    return {
        "schema_version": "compiled-ii-model-artifact-v2",
        "target": "compiled_ii_from_neura_heuristic_mapper",
        "artifact_status": "frozen_before_machsuite_reveal",
        "lower_bound_contract": {
            "name": "rec_res_max_v1",
            "formula": "max(rec_mii,res_mii)",
            "training_lower_bound_sources": ["rec_res_max_v1"],
            "components_are_model_features": False,
        },
        "trained_full_model": model_value,
        "trained_full_model_sha256": canonical_model_sha256(model_value),
        "provenance": {
            "training_distinct_base_dfg_count": len(hashes),
            "training_canonical_dfg_identity": {
                "scheme": "canonical_dfg_sha256_v1",
                "canonical_dfg_sha256s": hashes,
                "distinct_count": len(hashes),
                "set_sha256": hashlib.sha256(json.dumps(
                    hashes, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest(),
            },
        },
    }


class PredictionTest(unittest.TestCase):
    def _write_report(self, directory: Path, value=None) -> Path:
        artifact = model() if value is None else value
        report = {
            "schema_version": "portable-model-report-v2",
            "trained_full_model": artifact,
            "trained_full_model_sha256": canonical_model_sha256(artifact),
            "dataset_provenance": {
                "neura": {"revision": "revision-a"},
            },
        }
        path = directory / "model-report.json"
        path.write_text(json.dumps(report))
        return path

    def test_prediction_exposes_residual_steps_without_a_label(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            loaded = load_model_artifact(
                self._write_report(Path(raw_directory))
            )
            parsed = parse_prediction_sample(
                candidate(), loaded.model["feature_names"], "candidate"
            )
            result = predict_sample(loaded, parsed)
        self.assertEqual(result["raw_predicted_residual"], 2.5)
        self.assertEqual(result["nonnegative_predicted_residual"], 2.5)
        self.assertEqual(result["predicted_residual"], 2.5)
        self.assertEqual(result["predicted_compiled_ii"], 7.5)
        self.assertEqual(result["prediction_interval_lower"], 5.5)
        self.assertEqual(result["prediction_interval_upper"], 9.5)
        self.assertEqual(result["model_features"], {"x": 2.0})
        self.assertNotIn("compiled_ii", result)

    def test_negative_floor_and_dead_zone_both_return_the_lower_bound(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            loaded = load_model_artifact(
                self._write_report(Path(raw_directory))
            )
            negative = predict_sample(loaded, parse_prediction_sample(
                candidate(-2.0), loaded.model["feature_names"], "negative"
            ))
            small = predict_sample(loaded, parse_prediction_sample(
                candidate(0.0), loaded.model["feature_names"], "small"
            ))
        self.assertEqual(negative["predicted_compiled_ii"], 5.0)
        self.assertTrue(negative["nonnegative_floor_applied"])
        self.assertEqual(small["predicted_compiled_ii"], 5.0)
        self.assertTrue(small["dead_zone_applied"])

    def test_prediction_accepts_zero_rec_or_res_but_rejects_zero_floor(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            loaded = load_model_artifact(
                self._write_report(Path(raw_directory))
            )
            for rec_mii, res_mii in ((0, 3), (3, 0)):
                row = candidate()
                row.update({
                    "lower_bound": 3,
                    "rec_mii": rec_mii,
                    "res_mii": res_mii,
                })
                with self.subTest(rec_mii=rec_mii, res_mii=res_mii):
                    parsed = parse_prediction_sample(
                        row, loaded.model["feature_names"], "zero-component"
                    )
                    result = predict_sample(loaded, parsed)
                    self.assertEqual(result["lower_bound"], 3.0)
                    self.assertEqual(result["rec_mii"], float(rec_mii))
                    self.assertEqual(result["res_mii"], float(res_mii))

            zero_floor = candidate()
            zero_floor.update({
                "lower_bound": 0, "rec_mii": 0, "res_mii": 0,
            })
            with self.assertRaisesRegex(ValueError, "positive integer"):
                parse_prediction_sample(
                    zero_floor, loaded.model["feature_names"], "zero-floor"
                )

            negative_component = candidate()
            negative_component["rec_mii"] = -1
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                parse_prediction_sample(
                    negative_component,
                    loaded.model["feature_names"],
                    "negative-component",
                )

    def test_prediction_input_rejects_labels_and_inconsistent_lower_bounds(self):
        labelled = candidate()
        labelled["compiled_ii"] = 8
        with self.assertRaisesRegex(ValueError, "not allowed"):
            parse_prediction_sample(labelled, model()["feature_names"], "labelled")
        nested_label = candidate()
        nested_label["features"]["compiled_ii"] = 8
        with self.assertRaisesRegex(ValueError, "not allowed anywhere"):
            parse_prediction_sample(
                nested_label, model()["feature_names"], "nested-label"
            )
        metadata_label = candidate()
        metadata_label["metadata"]["compiled_ii"] = 8
        with self.assertRaisesRegex(ValueError, "not allowed anywhere"):
            parse_prediction_sample(
                metadata_label, model()["feature_names"], "metadata-label"
            )
        nested_bound = candidate()
        nested_bound["features"]["rec_mii"] = 5
        with self.assertRaisesRegex(ValueError, "top-level contract"):
            parse_prediction_sample(
                nested_bound, model()["feature_names"], "nested-bound"
            )
        inconsistent = candidate()
        inconsistent["baseline_lb"] = 6
        with self.assertRaisesRegex(ValueError, "lower bound disagrees"):
            parse_prediction_sample(
                inconsistent, model()["feature_names"], "inconsistent"
            )
        missing = candidate()
        del missing["features"]["x"]
        with self.assertRaisesRegex(ValueError, "missing model feature x"):
            parse_prediction_sample(missing, model()["feature_names"], "missing")
        below_component = candidate()
        below_component["rec_mii"] = 6
        with self.assertRaisesRegex(ValueError, "below proven component rec_mii"):
            parse_prediction_sample(
                below_component, model()["feature_names"], "below-component"
            )
        loose_floor = candidate()
        loose_floor["lower_bound"] = 10
        with self.assertRaisesRegex(ValueError, "must equal max"):
            parse_prediction_sample(
                loose_floor, model()["feature_names"], "loose-floor"
            )
        missing_component = candidate()
        del missing_component["res_mii"]
        with self.assertRaisesRegex(ValueError, "requires rec_mii and res_mii"):
            parse_prediction_sample(
                missing_component, model()["feature_names"], "missing-component"
            )
        for invalid_bound in (0, -1, 2.5):
            invalid = candidate()
            invalid["lower_bound"] = invalid_bound
            with self.subTest(lower_bound=invalid_bound), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                parse_prediction_sample(
                    invalid, model()["feature_names"], "invalid-bound"
                )

    def test_model_loader_validates_structure_and_report_hash(self):
        invalid_models = [
            {**model(), "model_type": "tree"},
            {**model(), "scale": [0.0]},
            {**model(), "weights": [1.0]},
            {**model(), "mean": [float("nan")]},
            {**model(), "feature_names": ["compiled_ii"]},
            {**model(), "feature_names": ["baseline_lb"]},
            {**model(), "feature_names": ["rec_mii"]},
            {**model(), "feature_names": ["res_mii"]},
        ]
        for invalid in invalid_models:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_model(invalid)
        with tempfile.TemporaryDirectory() as raw_directory:
            path = self._write_report(Path(raw_directory))
            raw = json.loads(path.read_text())
            raw["trained_full_model"]["weights"][0] += 1.0
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_model_artifact(path)
            del raw["trained_full_model_sha256"]
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "must contain"):
                load_model_artifact(path)
            raw["trained_full_model_sha256"] = canonical_model_sha256(
                raw["trained_full_model"]
            )
            raw["schema_version"] = "unrelated-regression-report-v1"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "unsupported compiled-II"):
                load_model_artifact(path)

    def test_v2_loader_validates_exact_training_canonical_identity_set(self):
        hashes = [hashlib.sha256(value.encode()).hexdigest() for value in (
            "training-a", "training-b",
        )]
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = directory / "frozen-v2.json"
            artifact = frozen_v2_artifact(model(), hashes)
            path.write_text(json.dumps(artifact))
            loaded = load_model_artifact(path)
            self.assertEqual(loaded.container, "compiled-ii-model-artifact-v2")
            self.assertEqual(
                loaded.provenance["training_canonical_dfg_identity"]
                ["canonical_dfg_sha256s"], sorted(hashes)
            )
            for mutation in ("unsorted", "duplicate", "bad-hash", "bad-count", "bad-digest", "bad-existing-count"):
                mutated = json.loads(json.dumps(artifact))
                identity = mutated["provenance"]["training_canonical_dfg_identity"]
                if mutation == "unsorted":
                    identity["canonical_dfg_sha256s"] = list(
                        reversed(sorted(hashes))
                    )
                elif mutation == "duplicate":
                    identity["canonical_dfg_sha256s"] = [hashes[0], hashes[0]]
                    identity["distinct_count"] = 2
                elif mutation == "bad-hash":
                    identity["canonical_dfg_sha256s"][0] = "BAD"
                elif mutation == "bad-count":
                    identity["distinct_count"] = 99
                elif mutation == "bad-digest":
                    identity["set_sha256"] = "0" * 64
                else:
                    mutated["provenance"]["training_distinct_base_dfg_count"] = 99
                path.write_text(json.dumps(mutated))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_model_artifact(path)

    def test_generic_loader_keeps_reading_v1_container(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            path = self._write_report(Path(raw_directory))
            raw = json.loads(path.read_text())
            raw["schema_version"] = "compiled-ii-model-artifact-v1"
            raw["artifact_status"] = "frozen_before_machsuite_reveal"
            raw["target"] = "compiled_ii_from_neura_heuristic_mapper"
            path.write_text(json.dumps(raw))
            loaded = load_model_artifact(path)
        self.assertEqual(loaded.container, "compiled-ii-model-artifact-v1")

    def test_non_finite_prediction_result_is_rejected(self):
        extreme = model()
        extreme["scale"] = [1e-308]
        extreme["weights"] = [0.0, 1e308]
        with tempfile.TemporaryDirectory() as raw_directory:
            loaded = load_model_artifact(
                self._write_report(Path(raw_directory), extreme)
            )
            parsed = parse_prediction_sample(
                candidate(1e308), loaded.model["feature_names"], "extreme"
            )
            with self.assertRaisesRegex(ValueError, "not finite"):
                predict_sample(loaded, parsed)

    def test_prediction_report_round_trip_is_order_independent(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            loaded = load_model_artifact(self._write_report(directory))
            first = candidate()
            first["features"] = {"x": 2.0}
            input_path = directory / "prediction-input.json"
            input_path.write_text(json.dumps({"samples": [first]}))
            samples, provenance = load_prediction_samples(
                input_path, loaded.model["feature_names"]
            )
            report = build_prediction_report(
                loaded, samples, input_path, provenance
            )
            serialized = json.dumps(report, allow_nan=False)
            restored = json.loads(serialized)
        self.assertEqual(restored["sample_count"], 1)
        self.assertEqual(
            restored["predictions"][0]["predicted_compiled_ii"], 7.5
        )
        self.assertFalse(
            restored["semantics"]["prediction_input_compiled_ii_used"]
        )


if __name__ == "__main__":
    unittest.main()
