import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters import machsuite_frozen, neura_experiment, neura_motifs
from cgra_ii_predictor.predict import canonical_model_sha256


def primary_model():
    names = list(neura_experiment.MODEL_FEATURE_NAMES)
    return {
        "model_type": "residual_ridge",
        "feature_names": names,
        "mean": [0.0] * len(names),
        "scale": [1.0] * len(names),
        "weights": [1.0] + [0.0] * len(names),
        "ridge": 1.0,
        "residual_dead_zone": 0.0,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cost_text(rec_mii=1, res_mii=2):
    values = {
        "analytical_ii": 3,
        "compute_mii": 1,
        "mem_mii": 1,
        "rec_mii": rec_mii,
        "reg_mii": 1,
        "res_mii": res_mii,
        "route_mii": 1,
    }
    return " ".join(f"{name} = {value} : i32" for name, value in values.items())


def generated_sample(root, motif, base_index, compiled_ii=3):
    base_seed = 1000 + base_index
    operation_count = 8 + (base_index % 7)
    base_id = f"base-{base_index:04d}-rp7"
    lineage = f"generated/motif-v1/{motif}/{base_id}"
    candidate_dir = root / f"generated-{base_index:04d}"
    candidate_dir.mkdir()
    source = candidate_dir / "input.mlir"
    architecture = candidate_dir / "architecture.yaml"
    mapped = candidate_dir / "mapped.mlir"
    source_text = neura_motifs.generate_motif_mlir(
        motif, operation_count, base_seed
    )
    source.write_text(source_text)
    neura_motifs.write_architecture(
        architecture, 4, 4, "homogeneous", 16
    )
    mapped.write_text(
        "module attributes {mapping_strategy = \"heuristic\", "
        f"compiled_ii = {compiled_ii} : i32, rec_mii = 1 : i32, "
        "res_mii = 2 : i32} {}\n"
    )
    source_sha = machsuite_frozen.raw_sha256(source)
    canonical = neura_motifs.canonical_dfg_sha256(source_text)
    architecture_sha = machsuite_frozen.raw_sha256(architecture)
    graph = neura_experiment.graph_features_from_neura(source_text, 4, 4)
    graph.update({
        "rec_mii": 1, "res_mii": 2, "baseline_lb": 2,
        "compiled_ii": compiled_ii,
    })
    graph.update({
        "index": f"{lineage}/4x4/homogeneous/r16",
        "candidate_id": f"{lineage}/4x4/homogeneous/r16",
        "training_stratum": "generated",
        "source_kind": "generated",
        "source_family": f"generated/motif-v1/{motif}",
        "family": lineage,
        "leakage_lineage_id": lineage,
        "declared_leakage_lineage_id": lineage,
        "effective_lineage": lineage,
        "base_dfg_id": canonical,
        "canonical_dfg_sha256": canonical,
        "ranking_query_id": canonical,
        "generator_family": f"generated/motif/{motif}",
        "generator_type": "generated/motif",
        "generator_version": "motif-v1",
        "motif": motif,
        "base_id": base_id,
        "base_seed": base_seed,
        "operation_count": operation_count,
        "rows": 4,
        "tiles": 16,
        "registers": 16,
        "source_path": str(source),
        "source_sha256": source_sha,
        "architecture_path": str(architecture),
        "architecture_sha256": architecture_sha,
        "architecture_variant": "homogeneous",
        "architecture_id": f"{architecture_sha}:homogeneous",
        "mapped_artifact_path": str(mapped),
        "mapped_artifact_sha256": machsuite_frozen.raw_sha256(mapped),
        "lower_bound_source": "rec_res_max_v1",
        "mapper_id": "neura-heuristic",
        "mapper_revision": machsuite_frozen.FROZEN_NEURA_REVISION,
        "mapper_config": "mapping-strategy=heuristic",
    })
    return graph


def generated_report(root, motifs, requested_per_family=200):
    training_opt = root / "mlir-neura-opt"
    training_opt.write_text("test mapper binary\n")
    samples = [
        generated_sample(root, motif, index)
        for index, motif in enumerate(motifs)
    ]
    selected_ridge, selected_dead_zone = (
        neura_experiment.select_ridge_hyperparameters(
            samples, machsuite_frozen.FROZEN_RIDGE_CANDIDATES,
            machsuite_frozen.FROZEN_DEAD_ZONE_CANDIDATES,
        )
    )
    model = neura_experiment.fit_ridge(
        samples, selected_ridge, selected_dead_zone
    )
    interval_rows = [{
        "family": samples[0]["family"],
        "prediction": float(samples[0]["compiled_ii"]),
        "compiled_ii": float(samples[0]["compiled_ii"]),
    }]
    neura_experiment.calibrate_unseen_family_interval(
        model, interval_rows, machsuite_frozen.FROZEN_INTERVAL_QUANTILE
    )
    ready = (
        len({sample["base_dfg_id"] for sample in samples}) >=
        machsuite_frozen.DEFAULT_MINIMUM_BASE_DFGS and
        {sample["generator_family"] for sample in samples} == {
            f"generated/motif/{motif}"
            for motif in neura_motifs.DEFAULT_MOTIFS
        }
    )
    revision = neura_experiment.git_provenance(
        machsuite_frozen.PROJECT_ROOT
    )["revision"]
    return {
        "schema_version": "neura-experiment-v2",
        "target": "compiled_ii_from_neura_heuristic_mapper",
        "artifact_status": "exploratory_not_frozen",
        "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
        "model_feature_names": list(neura_experiment.MODEL_FEATURE_NAMES),
        "trained_full_model": model,
        "trained_full_model_sha256": canonical_model_sha256(model),
        "provenance": {
            "adapter_sha256": machsuite_frozen.raw_sha256(
                machsuite_frozen.PROJECT_ROOT / "adapters" /
                "neura_experiment.py"
            ),
            "neura": {
                "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
                "dirty": False,
                "root": str(machsuite_frozen.DEFAULT_NEURA_ROOT.resolve()),
            },
            "mlir_neura_opt": str(training_opt),
            "mlir_neura_opt_sha256": machsuite_frozen.raw_sha256(training_opt),
            "predictor_repository": {"revision": revision, "dirty": False},
            "input_reports": [],
            "loaded_model": None,
            "mapping_strategy": "heuristic",
            "experiment_config": {
                "ridge_candidates": list(
                    machsuite_frozen.FROZEN_RIDGE_CANDIDATES
                ),
                "residual_dead_zone_candidates": list(
                    machsuite_frozen.FROZEN_DEAD_ZONE_CANDIDATES
                ),
                "interval_empirical_quantile": (
                    machsuite_frozen.FROZEN_INTERVAL_QUANTILE
                ),
                "motif_samples_per_family": requested_per_family,
                "motifs": list(machsuite_frozen.FROZEN_MOTIFS),
                "motif_shapes": list(
                    machsuite_frozen.FROZEN_MOTIF_SHAPES
                ),
                "motif_architecture_variants": list(
                    machsuite_frozen.FROZEN_ARCHITECTURE_VARIANTS
                ),
                "legacy_random_samples": 0,
                "metadata_holdout_keys": ["generator_family"],
            },
        },
        "candidate_gate": {
            "overall_ready_for_machsuite_freeze": ready,
        },
        "selected_model": "ridge",
        "nested_ridge_family_holdout": {"rows": interval_rows},
        "nested_ridge_metadata_holdouts": {
            "generator_family": {
                "status": "ok" if ready else "unavailable",
                "group_count": len(set(motifs)),
            }
        },
        "samples": samples,
    }


def frozen_preflight(root, ready_benchmark="bfs/bulk"):
    architecture = root / "architecture.yaml"
    architecture.write_text("architecture: test\n")
    architecture_sha = machsuite_frozen.raw_sha256(architecture)
    lowered_dir = root / ready_benchmark.replace("/", "__")
    lowered_dir.mkdir()
    lowered = lowered_dir / "lowered.mlir"
    cost = lowered_dir / "cost.mlir"
    lowered_text = neura_motifs.generate_motif_mlir("chain", 8, 99)
    lowered.write_text(lowered_text)
    cost.write_text(cost_text())
    canonical = neura_motifs.canonical_dfg_sha256(lowered_text)
    values = neura_experiment.parse_cost_features(cost.read_text())
    values.update(neura_experiment.graph_features_from_neura(
        lowered_text, 4, 4
    ))
    neura_experiment.add_prediction_features(values)
    all_features = {
        name: values[name]
        for name in machsuite_frozen.PREDICTION_RECORD_FEATURE_NAMES
    }
    candidate_identity = {
        "base_dfg_id": canonical,
        "architecture_sha256": architecture_sha,
        "architecture_variant": "4x4",
        "mapper_id": "neura-heuristic",
        "mapper_revision": machsuite_frozen.FROZEN_NEURA_REVISION,
        "mapper_config": "mapping-strategy=heuristic",
    }
    candidate_id = machsuite_frozen.canonical_json_sha256(candidate_identity)
    candidates = []
    samples = []
    for spec in machsuite_frozen.FROZEN_BENCHMARK_SPECS:
        record = {
            "benchmark_id": spec.benchmark_id,
            "source": spec.source,
            "top": spec.top,
            "leakage_lineage_id": spec.leakage_lineage_id,
            "source_sha256": f"source-{spec.benchmark_id}",
            "source_bundle_sha256": f"bundle-{spec.benchmark_id}",
        }
        if spec.benchmark_id == ready_benchmark:
            sample_id = f"machsuite/{spec.benchmark_id}/4x4"
            record.update({
                "status": "ready", "stage": "preflight_complete",
                "sample_id": sample_id, "candidate_id": candidate_id,
                "canonical_dfg_sha256": canonical,
                "lowered_artifact_path": str(lowered.relative_to(root)),
                "lowered_artifact_sha256": machsuite_frozen.raw_sha256(lowered),
                "cost_artifact_path": str(cost.relative_to(root)),
                "cost_artifact_sha256": machsuite_frozen.raw_sha256(cost),
                "lower_bound": 2, "rec_mii": 1, "res_mii": 2,
            })
            samples.append({
                "sample_id": sample_id,
                "lower_bound": 2, "rec_mii": 1, "res_mii": 2,
                "features": all_features,
                "metadata": {
                    "suite": "MachSuite",
                    "suite_revision": machsuite_frozen.FROZEN_SUITE_REVISION,
                    "kernel_id": spec.benchmark_id,
                    "leakage_lineage_id": spec.leakage_lineage_id,
                    "source_sha256": record["source_bundle_sha256"],
                    "benchmark_source_sha256": record["source_sha256"],
                    "dfg_source_sha256": record["lowered_artifact_sha256"],
                    "canonical_dfg_sha256": canonical,
                    "base_dfg_id": canonical,
                    "ranking_query_id": canonical,
                    "architecture_sha256": architecture_sha,
                    "architecture_variant": "4x4",
                    "architecture_id": f"{architecture_sha}:4x4",
                    "candidate_id": candidate_id,
                    "mapper_id": "neura-heuristic",
                    "mapper_revision": machsuite_frozen.FROZEN_NEURA_REVISION,
                    "mapper_config": "mapping-strategy=heuristic",
                    "lower_bound_source": "rec_res_max_v1",
                },
            })
        else:
            record.update({
                "status": "censored", "stage": "lower",
                "failure": {"status": "test-censored"},
            })
        candidates.append(record)
    inventory = machsuite_frozen.DEFAULT_INVENTORY.resolve()
    manifest = {
        "schema_version": machsuite_frozen.PREFLIGHT_SCHEMA,
        "protocol_status": "preflight_complete",
        "labels_accessed": False,
        "compiled_ii_present": False,
        "inventory": {
            "path": str(inventory),
            "sha256": machsuite_frozen.raw_sha256(inventory),
            "suite": "MachSuite",
            "repository": machsuite_frozen.FROZEN_SUITE_REPOSITORY,
            "revision": machsuite_frozen.FROZEN_SUITE_REVISION,
        },
        "architecture": {
            "path": str(architecture), "sha256": architecture_sha,
            "rows": 4, "columns": 4,
            "architecture_id": f"{architecture_sha}:4x4",
        },
        "neura_provenance": {
            "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
            "dirty": False,
        },
        "predictor_provenance": {"revision": "predictor", "dirty": False},
        "protocol_implementation": (
            machsuite_frozen.protocol_implementation_identity()
        ),
        "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
        "toolchain": {"mlir_neura_opt": {"sha256": "opt"}},
        "candidates": candidates,
        "samples": samples,
    }
    manifest["summary"] = machsuite_frozen.manifest_summary(candidates)
    return manifest, architecture


class MachSuiteFrozenTest(unittest.TestCase):
    def test_inventory_predeclares_all_official_variants(self):
        raw, specs = machsuite_frozen.load_inventory(
            machsuite_frozen.DEFAULT_INVENTORY
        )
        self.assertEqual(raw["declared_count"], 19)
        self.assertEqual(len(specs), 19)
        self.assertEqual(len({spec.leakage_lineage_id for spec in specs}), 12)
        self.assertEqual(
            raw["revision"], "6236e593012cb86b0d2f08d9fb9ba0411ff989b4"
        )
        self.assertEqual(specs[0].benchmark_id, "aes/aes")
        self.assertEqual(specs[-1].benchmark_id, "viterbi/viterbi")
        for spec in specs:
            self.assertTrue(
                (machsuite_frozen.DEFAULT_SUITE_ROOT / spec.source).is_file(),
                spec.source,
            )

    def test_preflight_command_graph_cannot_invoke_mapper(self):
        _, specs = machsuite_frozen.load_inventory(
            machsuite_frozen.DEFAULT_INVENTORY
        )
        commands = machsuite_frozen.preflight_commands(
            specs[0], Path("suite"), Path("out"), Path("clang"),
            Path("llvm-extract"), Path("mlir-translate"), Path("opt"),
            Path("architecture.yaml"), 4, 4,
        )
        joined = "\n".join(" ".join(command) for _, command, _ in commands)
        self.assertEqual([stage for stage, _, _ in commands], [
            "compile", "extract", "import", "lower", "analytical_cost"
        ])
        self.assertIn("--cost-model-analytical=", joined)
        self.assertNotIn("--map-to-accelerator", joined)

    def test_freeze_model_accepts_generated_only_primary_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            output_path = root / "frozen-model.json"
            revision = neura_experiment.git_provenance(
                machsuite_frozen.PROJECT_ROOT
            )["revision"]
            with patch.object(
                machsuite_frozen, "DEFAULT_MINIMUM_BASE_DFGS", 6
            ), patch.object(
                machsuite_frozen, "DEFAULT_MINIMUM_GENERATOR_FAMILIES", 6
            ), patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ):
                report = generated_report(
                    root, list(neura_motifs.DEFAULT_MOTIFS)
                )
                write_json(report_path, report)
                artifact = machsuite_frozen.freeze_random_training_model(
                    report_path, output_path, minimum_base_dfgs=6,
                    minimum_generator_families=6,
                )
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                artifact["artifact_status"],
                machsuite_frozen.FROZEN_MODEL_STATUS,
            )
            self.assertFalse(
                artifact["training_contract"]["bound_components_are_model_features"]
            )
            self.assertNotIn("samples", artifact)

    def test_default_freeze_gate_rejects_a_smoke_sized_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            report = generated_report(
                root, list(neura_motifs.DEFAULT_MOTIFS)
            )
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ), self.assertRaisesRegex(ValueError, "candidate gate"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json"
                    )

    def test_small_override_writes_only_a_non_predictable_smoke_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            report = generated_report(
                root, list(neura_motifs.DEFAULT_MOTIFS),
                requested_per_family=1,
            )
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ):
                artifact = machsuite_frozen.freeze_random_training_model(
                    report_path, root / "smoke.json",
                    allow_small_smoke=True,
                )
            self.assertEqual(
                artifact["artifact_status"],
                machsuite_frozen.SMOKE_MODEL_STATUS,
            )
            self.assertTrue(
                artifact["training_contract"]["scale_gate"]
                ["small_smoke_override"]
            )

    def test_freeze_model_rejects_real_or_machsuite_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            report = generated_report(
                root, list(neura_motifs.DEFAULT_MOTIFS)
            )
            report["samples"][0]["training_stratum"] = "real"
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ), self.assertRaisesRegex(ValueError, "generated-only"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json"
                    )

    def test_prediction_is_sealed_without_any_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            preflight_path = root / "preflight.json"
            prediction_path = root / "predictions.json"
            seal_path = root / "seal.json"
            model = primary_model()
            artifact = {
                "schema_version": machsuite_frozen.FROZEN_MODEL_SCHEMA,
                "target": "compiled_ii_from_neura_heuristic_mapper",
                "artifact_status": machsuite_frozen.FROZEN_MODEL_STATUS,
                "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
                "trained_full_model": model,
                "trained_full_model_sha256": canonical_model_sha256(model),
                "provenance": {
                    "neura": {
                        "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
                        "dirty": False,
                    }
                },
            }
            write_json(model_path, artifact)
            preflight, _ = frozen_preflight(root)
            write_json(preflight_path, preflight)
            predictions, seal = machsuite_frozen.freeze_predictions(
                preflight_path, model_path, prediction_path, seal_path
            )
            self.assertEqual(predictions["sample_count"], 1)
            self.assertEqual(seal["prediction_count"], 1)
            self.assertFalse(
                machsuite_frozen.recursively_contains_key(
                    predictions, "compiled_ii"
                )
            )
            self.assertEqual(
                seal["prediction_sha256"], machsuite_frozen.raw_sha256(prediction_path)
            )

    def test_fixed_inventory_cannot_be_shrunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            raw = json.loads(machsuite_frozen.DEFAULT_INVENTORY.read_text())
            raw["benchmarks"] = raw["benchmarks"][:-1]
            raw["declared_count"] = len(raw["benchmarks"])
            write_json(path, raw)
            with self.assertRaisesRegex(ValueError, "declared_count|identities"):
                machsuite_frozen.load_inventory(path)

    def test_prediction_rederives_and_rejects_modified_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "preflight.json"
            manifest, _ = frozen_preflight(root)
            manifest["samples"][0]["features"]["semantic_edges"] += 1
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "features were modified"):
                machsuite_frozen.validate_frozen_manifest(
                    manifest, manifest_path
                )

    def test_freeze_rejects_forged_generated_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            report = generated_report(
                root, list(neura_motifs.DEFAULT_MOTIFS)
            )
            report["samples"][0]["base_dfg_id"] = "arbitrary"
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ), self.assertRaisesRegex(ValueError, "DFG identity"):
                machsuite_frozen.freeze_random_training_model(
                    report_path, root / "model.json", allow_small_smoke=True
                )

    def test_forged_prediction_label_is_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            manifest_path = root / "preflight.json"
            prediction_path = root / "predictions.json"
            seal_path = root / "seal.json"
            model = primary_model()
            write_json(model_path, {
                "schema_version": machsuite_frozen.FROZEN_MODEL_SCHEMA,
                "target": "compiled_ii_from_neura_heuristic_mapper",
                "artifact_status": machsuite_frozen.FROZEN_MODEL_STATUS,
                "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
                "trained_full_model": model,
                "trained_full_model_sha256": canonical_model_sha256(model),
                "provenance": {
                    "neura": {
                        "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
                        "dirty": False,
                    }
                },
            })
            manifest, _ = frozen_preflight(root)
            write_json(manifest_path, manifest)
            predictions, seal = machsuite_frozen.freeze_predictions(
                manifest_path, model_path, prediction_path, seal_path
            )
            predictions["predictions"][0]["compiled_ii"] = 99
            write_json(prediction_path, predictions)
            seal["prediction_sha256"] = machsuite_frozen.raw_sha256(
                prediction_path
            )
            loaded = machsuite_frozen.load_model_artifact(model_path)
            with self.assertRaisesRegex(ValueError, "forbidden compiled_ii"):
                machsuite_frozen.validate_prediction_report_and_seal(
                    manifest=manifest, manifest_path=manifest_path,
                    model_path=model_path, prediction_path=prediction_path,
                    predictions=predictions, seal=seal, loaded=loaded,
                )

    def test_mapper_command_exists_only_for_reveal(self):
        command = machsuite_frozen.mapper_command(
            Path("opt"), Path("lowered.mlir"), Path("arch.yaml"),
            Path("mapped.mlir"), 4, 4,
        )
        self.assertIn("--map-to-accelerator=", " ".join(command))

    def test_reveal_rejects_a_broken_seal_before_mapper_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            model = root / "model.json"
            predictions = root / "predictions.json"
            seal = root / "seal.json"
            model_value = primary_model()
            write_json(model, {
                "schema_version": machsuite_frozen.FROZEN_MODEL_SCHEMA,
                "target": "compiled_ii_from_neura_heuristic_mapper",
                "artifact_status": machsuite_frozen.FROZEN_MODEL_STATUS,
                "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
                "trained_full_model": model_value,
                "trained_full_model_sha256": canonical_model_sha256(model_value),
                "provenance": {
                    "neura": {
                        "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
                        "dirty": False,
                    }
                },
            })
            preflight_value, architecture = frozen_preflight(root)
            write_json(preflight, preflight_value)
            machsuite_frozen.freeze_predictions(
                preflight, model, predictions, seal
            )
            broken_seal = json.loads(seal.read_text())
            broken_seal["preflight_sha256"] = "not-the-current-hash"
            write_json(seal, broken_seal)
            with self.assertRaisesRegex(ValueError, "sealed preflight_sha256"):
                machsuite_frozen.reveal_labels(
                    manifest_path=preflight,
                    model_path=model,
                    prediction_path=predictions,
                    seal_path=seal,
                    suite_root=root,
                    neura_root=root,
                    architecture=architecture,
                    opt=root / "opt",
                    output_dir=root / "reveal",
                    timeout=1,
                )

    def test_metrics_keep_declared_coverage_separate_from_accuracy(self):
        rows = [
            {
                "status": "scored", "lower_bound": 2,
                "predicted_compiled_ii": 4.0, "compiled_ii": 5,
            },
            {"status": "preflight_censored"},
        ]
        metrics = machsuite_frozen.evaluation_metrics(rows)
        self.assertEqual(metrics["scored_count"], 1)
        self.assertEqual(metrics["mae"], 1.0)
        self.assertEqual(metrics["baseline_mae"], 3.0)


if __name__ == "__main__":
    unittest.main()
