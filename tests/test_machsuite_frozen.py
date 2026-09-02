import json
import hashlib
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


def frozen_artifact(model_value, training_hashes, schema=None):
    schema = schema or machsuite_frozen.FROZEN_MODEL_SCHEMA
    provenance = {
        "neura": {
            "revision": machsuite_frozen.FROZEN_NEURA_REVISION,
            "dirty": False,
        },
    }
    if schema == machsuite_frozen.FROZEN_MODEL_SCHEMA:
        hashes = sorted(set(training_hashes))
        provenance.update({
            "training_distinct_base_dfg_count": len(hashes),
            "training_canonical_dfg_identity": {
                "scheme": "canonical_dfg_sha256_v1",
                "canonical_dfg_sha256s": hashes,
                "distinct_count": len(hashes),
                "set_sha256": machsuite_frozen.canonical_json_sha256(hashes),
            },
        })
    return {
        "schema_version": schema,
        "target": "compiled_ii_from_neura_heuristic_mapper",
        "artifact_status": machsuite_frozen.FROZEN_MODEL_STATUS,
        "lower_bound_contract": dict(machsuite_frozen.LOWER_BOUND_CONTRACT),
        "trained_full_model": model_value,
        "trained_full_model_sha256": canonical_model_sha256(model_value),
        "provenance": provenance,
    }


def cost_text(rec_mii=1, res_mii=2):
    values = {
        "rec_mii": rec_mii,
        "res_mii": res_mii,
    }
    attributes = " ".join(
        f"{name} = {value} : i32" for name, value in values.items()
    )
    return f"rec_res_mii_info = {{{attributes}}}"


def generated_sample(
    root, motif, base_index, shape="4x4", variant="homogeneous",
    compiled_ii=3,
):
    base_seed = 1000 + base_index + 100 * list(neura_motifs.DEFAULT_MOTIFS).index(motif)
    operation_count = 8 + (base_index % 7)
    base_id = f"base-{base_index:04d}-rp7"
    lineage = f"generated/{neura_motifs.GENERATOR_VERSION}/{motif}/{base_id}"
    rows, columns = neura_motifs.parse_shape(shape)
    candidate_dir = root / (
        f"generated-{motif}-{base_index:04d}-{shape}-{variant}"
    )
    candidate_dir.mkdir()
    source = candidate_dir / "input.mlir"
    architecture = candidate_dir / "architecture.yaml"
    cost = candidate_dir / "cost.mlir"
    mapped = candidate_dir / "mapped.mlir"
    source_text = neura_motifs.generate_motif_mlir(
        motif, operation_count, base_seed
    )
    source.write_text(source_text)
    neura_motifs.write_architecture(
        architecture, rows, columns, variant, 16
    )
    cost.write_text(cost_text())
    mapped.write_text(
        "module attributes {mapping_strategy = \"heuristic\", "
        f"compiled_ii = {compiled_ii} : i32, rec_mii = 1 : i32, "
        "res_mii = 2 : i32} {}\n"
    )
    source_sha = machsuite_frozen.raw_sha256(source)
    canonical = neura_motifs.canonical_dfg_sha256(source_text)
    architecture_sha = machsuite_frozen.raw_sha256(architecture)
    graph = neura_experiment.graph_features_from_neura(
        source_text, rows, columns
    )
    graph.update({
        "rec_mii": 1, "res_mii": 2, "baseline_lb": 2,
        "compiled_ii": compiled_ii,
    })
    graph.update({
        "index": f"{lineage}/{shape}/{variant}/r16",
        "candidate_id": f"{lineage}/{shape}/{variant}/r16",
        "training_stratum": "generated",
        "source_kind": "generated",
        "source_family": f"generated/{neura_motifs.GENERATOR_VERSION}/{motif}",
        "family": lineage,
        "leakage_lineage_id": lineage,
        "declared_leakage_lineage_id": lineage,
        "effective_lineage": lineage,
        "base_dfg_id": canonical,
        "canonical_dfg_sha256": canonical,
        "ranking_query_id": canonical,
        "generator_family": f"generated/motif/{motif}",
        "generator_type": "generated/motif",
        "generator_version": neura_motifs.GENERATOR_VERSION,
        "motif": motif,
        "base_id": base_id,
        "base_seed": base_seed,
        "operation_count": operation_count,
        "rows": rows,
        "tiles": rows * columns,
        "registers": 16,
        "source_path": str(source),
        "source_sha256": source_sha,
        "architecture_path": str(architecture),
        "architecture_sha256": architecture_sha,
        "architecture_variant": variant,
        "architecture_id": f"{architecture_sha}:{variant}",
        "mapped_artifact_path": str(mapped),
        "mapped_artifact_sha256": machsuite_frozen.raw_sha256(mapped),
        "lower_bound_source": "rec_res_max_v1",
        "mapper_id": "neura-heuristic",
        "mapper_revision": machsuite_frozen.FROZEN_NEURA_REVISION,
        "mapper_config": "mapping-strategy=heuristic",
        "rec_res_evidence": "neura_shared_rec_res_analysis_v1",
        "cost_artifact_path": str(cost),
        "cost_artifact_sha256": machsuite_frozen.raw_sha256(cost),
    })
    graph["split_domain"] = int(variant == "split-domain")
    return graph


def generated_report(
    root, motifs=None, requested_per_family=None, base_count=1,
    shapes=None, variants=None, compiled_ii=3,
):
    motifs = tuple(motifs or neura_motifs.DEFAULT_MOTIFS)
    if requested_per_family is None:
        requested_per_family = machsuite_frozen.FROZEN_REQUESTED_BASES_PER_FAMILY
    shapes = tuple(shapes or machsuite_frozen.FROZEN_MOTIF_SHAPES)
    variants = tuple(variants or machsuite_frozen.FROZEN_ARCHITECTURE_VARIANTS)
    training_opt = root / "mlir-neura-opt"
    training_opt.write_text("test mapper binary\n")
    samples = []
    for motif in motifs:
        for base_index in range(base_count):
            for shape in shapes:
                for variant in variants:
                    samples.append(generated_sample(
                        root, motif, base_index, shape, variant,
                        compiled_ii=compiled_ii,
                    ))
    manifest_path = root / "corpus-manifest.json"
    write_json(manifest_path, {
        "schema_version": neura_motifs.MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "candidates": [{
            "generator_family": sample["generator_family"],
            "base_dfg_id": sample["base_dfg_id"],
            "canonical_dfg_sha256": sample["canonical_dfg_sha256"],
            "rows": sample["rows"],
            "columns": sample["tiles"] // sample["rows"],
            "architecture_variant": sample["architecture_variant"],
            "status": "success",
        } for sample in samples],
    })
    required_cells = machsuite_frozen.required_shape_variant_cells()
    training_samples, training_selection = (
        neura_experiment.complete_generated_training_subset(
            samples, required_cells
        )
    )
    selected_ridge, selected_dead_zone = (
        neura_experiment.select_ridge_hyperparameters(
            training_samples, machsuite_frozen.FROZEN_RIDGE_CANDIDATES,
            machsuite_frozen.FROZEN_DEAD_ZONE_CANDIDATES,
        )
    )
    model = neura_experiment.fit_ridge(
        training_samples, selected_ridge, selected_dead_zone
    )
    nested = neura_experiment.nested_ridge_family_holdout(
        training_samples, machsuite_frozen.FROZEN_RIDGE_CANDIDATES,
        machsuite_frozen.FROZEN_DEAD_ZONE_CANDIDATES,
    )
    interval_rows = nested["rows"]
    neura_experiment.calibrate_unseen_family_interval(
        model, interval_rows, machsuite_frozen.FROZEN_INTERVAL_QUANTILE
    )
    improvement = neura_experiment.generated_nested_improvement_gate(nested)
    coverage = machsuite_frozen.generated_training_coverage(
        samples, requested_bases_per_family=requested_per_family,
        declared_candidates=json.loads(manifest_path.read_text())["candidates"],
    )
    ready = bool(
        coverage["passed"] and
        improvement["passed"] and
        requested_per_family == machsuite_frozen.FROZEN_REQUESTED_BASES_PER_FAMILY
    )
    generated_corpus = neura_experiment.motif_corpus_summary(
        samples, manifest_path
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
            "required_generator_families": list(
                machsuite_frozen.required_generator_families()
            ),
            "required_shape_variant_cells": list(
                machsuite_frozen.required_shape_variant_cells()
            ),
            "minimum_complete_bases_per_family": (
                machsuite_frozen.FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY
            ),
            "minimum_total_complete_bases": (
                machsuite_frozen.FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY *
                len(machsuite_frozen.FROZEN_MOTIFS)
            ),
            "requested_bases_per_family": requested_per_family,
            "requested_total_bases": (
                requested_per_family * len(machsuite_frozen.FROZEN_MOTIFS)
            ),
            "minimum_complete_fraction": (
                machsuite_frozen.FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY /
                requested_per_family
            ),
            "coverage": coverage,
            "generated_nested_improvement": improvement,
            "training_selection": training_selection,
            "overall_ready_for_machsuite_freeze": ready,
        },
        "selected_model": "ridge",
        "nested_ridge_family_holdout": nested,
        "nested_ridge_metadata_holdouts": {
            "generator_family": {
                "status": "ok",
                "group_count": len(set(motifs)),
            }
        },
        "motif_corpus": generated_corpus,
        "samples": training_samples,
        "labelled_samples": samples,
        "training_selection": training_selection,
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
            "compile", "extract", "import", "lower", "rec_res_analysis"
        ])
        self.assertIn("--analyze-rec-res-mii=", joined)
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
                machsuite_frozen, "FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "FROZEN_REQUESTED_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ):
                report = generated_report(
                    root, list(neura_motifs.DEFAULT_MOTIFS)
                )
                write_json(report_path, report)
                artifact = machsuite_frozen.freeze_random_training_model(
                    report_path, output_path, minimum_base_dfgs=len(
                        machsuite_frozen.FROZEN_MOTIFS
                    ),
                    minimum_generator_families=len(
                        machsuite_frozen.FROZEN_MOTIFS
                    ),
                    minimum_complete_bases_per_family=1,
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

    def test_formal_freeze_requires_ridge_to_strictly_improve_on_lb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            revision = neura_experiment.git_provenance(
                machsuite_frozen.PROJECT_ROOT
            )["revision"]
            with patch.object(
                machsuite_frozen, "FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "FROZEN_REQUESTED_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ):
                # compiled_ii equals the Rec/Res floor for every row.  Ridge
                # can tie that perfect baseline but cannot strictly improve.
                report = generated_report(root, compiled_ii=2)
                self.assertFalse(
                    report["candidate_gate"]["generated_nested_improvement"]
                    ["passed"]
                )
                write_json(report_path, report)
                with self.assertRaisesRegex(ValueError, "candidate gate"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json",
                        minimum_base_dfgs=len(machsuite_frozen.FROZEN_MOTIFS),
                        minimum_generator_families=len(
                            machsuite_frozen.FROZEN_MOTIFS
                        ),
                        minimum_complete_bases_per_family=1,
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

    def test_unbalanced_generator_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generated_report(root)
            report["samples"] = [
                sample for sample in report["samples"]
                if sample["generator_family"] != "generated/motif/chain"
            ]
            report_path = root / "training-report.json"
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ), self.assertRaisesRegex(ValueError, "candidate gate|families"):
                machsuite_frozen.freeze_random_training_model(
                    report_path, root / "frozen.json", allow_small_smoke=True
                )

    def test_missing_shape_variant_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                machsuite_frozen, "FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "FROZEN_REQUESTED_BASES_PER_FAMILY", 1
            ):
                report = generated_report(root)
                report["samples"] = [
                    sample for sample in report["samples"]
                    if not (
                        sample["generator_family"] == "generated/motif/chain" and
                        sample["architecture_variant"] == "split-domain" and
                        sample["rows"] == 3 and sample["tiles"] == 9
                    )
                ]
                report_path = root / "training-report.json"
                write_json(report_path, report)
                revision = report["provenance"]["predictor_repository"]["revision"]
                with patch.object(
                    machsuite_frozen, "require_clean_revision",
                    return_value={"revision": revision, "dirty": False},
                ), self.assertRaisesRegex(ValueError, "coverage|candidate gate"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json", allow_small_smoke=True
                    )

    def test_successful_cell_must_be_predeclared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                machsuite_frozen, "FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY", 1
            ), patch.object(
                machsuite_frozen, "FROZEN_REQUESTED_BASES_PER_FAMILY", 1
            ):
                report = generated_report(root)
                manifest_path = Path(
                    report["motif_corpus"]["manifest_path"]
                )
                manifest = json.loads(manifest_path.read_text())
                manifest["candidates"] = [
                    candidate for candidate in manifest["candidates"]
                    if not (
                        candidate["generator_family"] == "generated/motif/chain" and
                        candidate["architecture_variant"] == "homogeneous" and
                        candidate["rows"] == 3 and candidate["columns"] == 3
                    )
                ]
                write_json(manifest_path, manifest)
                report["motif_corpus"]["manifest_sha256"] = (
                    machsuite_frozen.raw_sha256(manifest_path)
                )
                report_path = root / "training-report.json"
                write_json(report_path, report)
                revision = report["provenance"]["predictor_repository"]["revision"]
                with patch.object(
                    machsuite_frozen, "require_clean_revision",
                    return_value={"revision": revision, "dirty": False},
                ), self.assertRaisesRegex(ValueError, "candidate gate|predeclared"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json", allow_small_smoke=True
                    )

    def test_declared_denominator_rejects_low_success_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                machsuite_frozen, "FROZEN_MINIMUM_COMPLETE_BASES_PER_FAMILY", 2
            ), patch.object(
                machsuite_frozen, "FROZEN_REQUESTED_BASES_PER_FAMILY", 3
            ):
                # The manifest declares three bases per family, but only one
                # complete base is successfully labelled.
                report = generated_report(
                    root, requested_per_family=3, base_count=3
                )
                report["samples"] = [
                    sample for sample in report["samples"]
                    if sample["base_id"].endswith("0000-rp7")
                ]
                report_path = root / "training-report.json"
                write_json(report_path, report)
                revision = report["provenance"]["predictor_repository"]["revision"]
                with patch.object(
                    machsuite_frozen, "require_clean_revision",
                    return_value={"revision": revision, "dirty": False},
                ), self.assertRaisesRegex(ValueError, "candidate gate|scale"):
                    machsuite_frozen.freeze_random_training_model(
                        report_path, root / "frozen.json", allow_small_smoke=True
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
            # Keep the training identity disjoint from the preflight DFG.
            artifact = frozen_artifact(
                model, [hashlib.sha256(b"training-only").hexdigest()]
            )
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

    def test_training_test_overlap_is_rejected_before_prediction_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            preflight_path = root / "preflight.json"
            prediction_path = root / "predictions.json"
            seal_path = root / "seal.json"
            preflight, _ = frozen_preflight(root)
            write_json(preflight_path, preflight)
            test_hash = preflight["samples"][0]["metadata"][
                "canonical_dfg_sha256"
            ]
            write_json(model_path, frozen_artifact(primary_model(), [test_hash]))
            with self.assertRaisesRegex(ValueError, "overlap"):
                machsuite_frozen.freeze_predictions(
                    preflight_path, model_path, prediction_path, seal_path
                )
            self.assertFalse(prediction_path.exists())
            self.assertFalse(seal_path.exists())

    def test_v1_model_is_generic_readable_but_frozen_workflow_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            preflight_path = root / "preflight.json"
            prediction_path = root / "predictions.json"
            seal_path = root / "seal.json"
            write_json(model_path, frozen_artifact(
                primary_model(), [hashlib.sha256(b"training-only").hexdigest()],
                schema=machsuite_frozen.LEGACY_FROZEN_MODEL_SCHEMA,
            ))
            preflight, _ = frozen_preflight(root)
            write_json(preflight_path, preflight)
            loaded = machsuite_frozen.load_model_artifact(model_path)
            self.assertEqual(
                loaded.container, machsuite_frozen.LEGACY_FROZEN_MODEL_SCHEMA
            )
            with self.assertRaisesRegex(ValueError, "v2|frozen artifact"):
                machsuite_frozen.freeze_predictions(
                    preflight_path, model_path, prediction_path, seal_path
                )
            self.assertFalse(prediction_path.exists())

    def test_reveal_rechecks_training_test_overlap_before_mapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight_path = root / "preflight.json"
            model_path = root / "model.json"
            prediction_path = root / "predictions.json"
            seal_path = root / "seal.json"
            preflight, architecture = frozen_preflight(root)
            write_json(preflight_path, preflight)
            disjoint = hashlib.sha256(b"training-only").hexdigest()
            write_json(model_path, frozen_artifact(primary_model(), [disjoint]))
            machsuite_frozen.freeze_predictions(
                preflight_path, model_path, prediction_path, seal_path
            )
            test_hash = preflight["samples"][0]["metadata"][
                "canonical_dfg_sha256"
            ]
            write_json(model_path, frozen_artifact(primary_model(), [test_hash]))
            with self.assertRaisesRegex(ValueError, "overlap"):
                machsuite_frozen.reveal_labels(
                    manifest_path=preflight_path,
                    model_path=model_path,
                    prediction_path=prediction_path,
                    seal_path=seal_path,
                    suite_root=root,
                    neura_root=root,
                    architecture=architecture,
                    opt=root / "opt",
                    output_dir=root / "reveal",
                    timeout=1,
                )
            self.assertFalse((root / "reveal").exists())

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

    def test_freeze_rejects_rehashed_rec_res_mapper_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training-report.json"
            report = generated_report(
                root, list(neura_motifs.DEFAULT_MOTIFS)
            )
            sample = report["samples"][0]
            cost = Path(sample["cost_artifact_path"])
            cost.write_text(cost_text(rec_mii=3, res_mii=2))
            sample["cost_artifact_sha256"] = machsuite_frozen.raw_sha256(cost)
            write_json(report_path, report)
            revision = report["provenance"]["predictor_repository"]["revision"]
            with patch.object(
                machsuite_frozen, "require_clean_revision",
                return_value={"revision": revision, "dirty": False},
            ), self.assertRaisesRegex(ValueError, "facts disagree"):
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
            write_json(model_path, frozen_artifact(
                model, [hashlib.sha256(b"training-only").hexdigest()]
            ))
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
            write_json(model, frozen_artifact(
                model_value, [hashlib.sha256(b"training-only").hexdigest()]
            ))
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
