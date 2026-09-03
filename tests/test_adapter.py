import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters import (
    machsuite_frozen, neura_experiment as adapter, neura_motifs,
    neura_motifs_v4,
)


def cost_text(**overrides):
    values = {
        "rec_mii": 4,
        "res_mii": 5,
    }
    values.update(overrides)
    attributes = " ".join(
        f"{name} = {value} : i32" for name, value in values.items()
    )
    return f"rec_res_mii_info = {{{attributes}}}"


class NeuraAdapterTest(unittest.TestCase):
    def test_portable_metadata_preserves_v4_strata(self):
        row = {
            "rows": 2, "columns": 3, "tiles": 6, "links": 14,
            "source_kind": "generated", "target_shape": "2x3",
            "target_config_id": "prefix-2x3",
            "mechanism_profile": "long_range_cutwidth",
            "operation_band": "medium", "shape_block": "transpose-2x3-3x2",
        }
        metadata = adapter.portable_sample_metadata(row)
        for name in (
            "columns", "target_shape", "target_config_id",
            "mechanism_profile", "operation_band", "shape_block",
        ):
            self.assertEqual(metadata[name], row[name])

    def test_one_by_one_features_have_no_network_division(self):
        source = neura_motifs.generate_motif_mlir("chain", 8, 11)
        features = adapter.graph_features_from_neura(source, 1, 1)
        self.assertEqual(features["tiles"], 1)
        self.assertEqual(features["links"], 0)
        self.assertEqual(features["bisection_links"], 0)
        self.assertEqual(features["memory_tiles"], 1)
        self.assertEqual(features["routing_edge_pressure"], 0.0)
        self.assertEqual(features["routing_cut_pressure"], 0.0)

    def test_one_by_two_is_canonical_for_symmetric_two_tile_features(self):
        source = neura_motifs.generate_motif_mlir("memory_stream", 8, 11)
        horizontal = adapter.graph_features_from_neura(source, 1, 2)
        vertical = adapter.graph_features_from_neura(source, 2, 1)
        self.assertEqual(horizontal["tiles"], 2)
        self.assertEqual(horizontal["links"], 2)
        self.assertEqual(horizontal["memory_tiles"], 2)
        for name in adapter.MODEL_FEATURE_NAMES:
            self.assertEqual(horizontal[name], vertical[name], name)

    def test_shape_selection_reports_area_ii_pareto_and_verification_order(self):
        predictions = [
            {"task": "k", "sample": "k-1x1", "shape": "1x1",
             "tile_count": 1, "predicted_compiled_ii": 12.0,
             "shape_training_support": "stress_only_untrained_shape",
             "lower_bound_within_mapper_search_interval": False},
            {"task": "k", "sample": "k-2x2", "shape": "2x2",
             "tile_count": 4, "predicted_compiled_ii": 8.0},
            {"task": "k", "sample": "k-3x3", "shape": "3x3",
             "tile_count": 9, "predicted_compiled_ii": 5.0},
            {"task": "k", "sample": "k-4x3", "shape": "4x3",
             "tile_count": 12, "predicted_compiled_ii": 3.0,
             "feature_support": {
                 "outside_observed_range": ["routing_cut_pressure"]
             }},
            {"task": "k", "sample": "k-4x4", "shape": "4x4",
             "tile_count": 16, "predicted_compiled_ii": 4.0},
        ]
        summary = adapter.shape_selection_summary(predictions)[0]
        self.assertEqual(
            summary["pareto_shapes"], ["2x2", "3x3", "4x4"]
        )
        self.assertEqual(summary["throughput_first_shape"], "4x4")
        self.assertEqual(summary["mapper_verification_order"][0], "k-4x4")
        self.assertEqual(
            summary["unsupported_out_of_range_candidate_ids"], ["k-4x3"]
        )
        self.assertEqual(
            summary["unsupported_untrained_shape_candidate_ids"], ["k-1x1"]
        )
        self.assertEqual(
            summary["unsupported_empty_mapper_search_candidate_ids"], ["k-1x1"]
        )

    def test_neura_root_prefers_environment_then_initialized_submodule(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"NEURA_ROOT": directory}):
                self.assertEqual(
                    adapter.resolve_configured_neura_root(), Path(directory)
                )

        with patch.dict("os.environ", {}, clear=True), patch.object(
            adapter, "SUBMODULE_NEURA_ROOT", Path("/missing/neura")
        ):
            self.assertIsNone(adapter.resolve_configured_neura_root())

    def test_cost_parser_accepts_main_branch_rec_res_fields(self):
        parsed = adapter.parse_cost_features(cost_text())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["rec_mii"], 4)
        self.assertEqual(set(parsed), set(adapter.COST_FEATURE_NAMES))

    def test_cost_parser_allows_zero_component_with_positive_derived_bound(self):
        for rec_mii, res_mii in ((0, 5), (5, 0)):
            with self.subTest(rec_mii=rec_mii, res_mii=res_mii):
                parsed = adapter.parse_cost_features(
                    cost_text(rec_mii=rec_mii, res_mii=res_mii)
                )
                self.assertEqual(
                    parsed, {"rec_mii": rec_mii, "res_mii": res_mii}
                )
                adapter.add_prediction_features(parsed)
                self.assertEqual(parsed["baseline_lb"], 5)

        zero_floor = adapter.parse_cost_features(cost_text(rec_mii=0, res_mii=0))
        self.assertEqual(zero_floor, {"rec_mii": 0, "res_mii": 0})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            adapter.add_prediction_features(zero_floor)

        self.assertIsNone(
            adapter.parse_cost_features(cost_text(rec_mii=-1, res_mii=5))
        )

    def test_cost_parser_rejects_label_contamination(self):
        for contamination in (
            "compiled_ii = 7 : i32",
            "compiled_ii = 7 : i64",
            'mapping_info = {mapping_strategy = "heuristic"}',
            "analytical_ii = 9 : i32",
        ):
            with self.subTest(contamination=contamination), \
                    self.assertRaisesRegex(ValueError, "mapping/label tokens"):
                adapter.parse_cost_features(cost_text() + " " + contamination)
        with self.assertRaisesRegex(ValueError, "mapping/label tokens"):
            adapter.parse_cost_features("compiled_ii = 7 : i32")

    def test_cost_parser_requires_analysis_pass_marker(self):
        self.assertIsNone(adapter.parse_cost_features(
            "rec_mii = 4 : i32 res_mii = 5 : i32"
        ))

    def test_mapper_label_must_match_analysis_rec_res(self):
        analysis = adapter.parse_cost_features(cost_text())
        mapped = (
            "compiled_ii = 8 : i32 rec_mii = 4 : i32 "
            "res_mii = 5 : i32"
        )
        self.assertEqual(adapter.parse_checked_mapper_label(mapped, analysis), 8)
        with self.assertRaisesRegex(ValueError, "facts disagree"):
            adapter.parse_checked_mapper_label(
                mapped.replace("rec_mii = 4", "rec_mii = 3"), analysis
            )

    def test_authoritative_bound_is_exactly_rec_res_max(self):
        row = adapter.parse_cost_features(cost_text())
        row["compiled_ii"] = 8
        adapter.add_prediction_features(row)
        self.assertEqual(row["baseline_lb"], 5)
        self.assertEqual(row["lower_bound_source"], "rec_res_max_v1")
        self.assertNotIn("proven_lower_bound", row)

    def test_explicit_alias_must_match_rec_res_contract(self):
        row = adapter.parse_cost_features(cost_text())
        row.update({"baseline_lb": 5, "compiled_ii": 8})
        adapter.add_prediction_features(row)
        self.assertEqual(row["baseline_lb"], 5)
        self.assertEqual(row["lower_bound_source"], "rec_res_max_v1")

        invalid = adapter.parse_cost_features(cost_text())
        invalid["baseline_lb"] = 7
        with self.assertRaisesRegex(ValueError, "disagrees"):
            adapter.add_prediction_features(invalid)

        with self.assertRaisesRegex(ValueError, "disagrees"):
            adapter.predict_unlabelled_candidate(
                {"feature_names": []},
                {"baseline_lb": 7, "rec_mii": 4, "res_mii": 5},
            )

    def test_portable_alias_and_implicit_bound_are_rec_res_max(self):
        portable = adapter.parse_cost_features(cost_text())
        portable.update({"lower_bound": 5, "compiled_ii": 8})
        adapter.add_prediction_features(portable)
        self.assertEqual(portable["baseline_lb"], 5)
        self.assertEqual(portable["lower_bound_source"], "rec_res_max_v1")

        implicit = adapter.parse_cost_features(cost_text())
        implicit["compiled_ii"] = 6
        adapter.add_prediction_features(implicit)
        self.assertEqual(implicit["baseline_lb"], 5)
        self.assertEqual(implicit["lower_bound_source"], "rec_res_max_v1")

        proven = adapter.parse_cost_features(cost_text())
        proven.update({"proven_lower_bound": 5, "compiled_ii": 8})
        adapter.add_prediction_features(proven)
        self.assertEqual(proven["baseline_lb"], 5)

        conflicting = adapter.parse_cost_features(cost_text())
        conflicting.update({
            "lower_bound": 5,
            "proven_lower_bound": 6,
            "compiled_ii": 8,
        })
        with self.assertRaisesRegex(ValueError, "proven_lower_bound.*disagrees"):
            adapter.add_prediction_features(conflicting)

    def test_primary_model_features_exclude_bound_and_components(self):
        forbidden = {"baseline_lb", *adapter.LOWER_BOUND_COMPONENT_NAMES}
        self.assertTrue(forbidden.isdisjoint(adapter.MODEL_FEATURE_NAMES))

    def test_generated_improvement_gate_is_strict_and_label_blind(self):
        improved = adapter.generated_nested_improvement_gate({
            "baseline_macro_family_mae": 1.0,
            "ridge_macro_family_mae": 0.75,
        })
        self.assertTrue(improved["passed"])
        self.assertEqual(improved["absolute_improvement"], 0.25)
        self.assertFalse(improved["machsuite_labels_used"])

        for ridge in (1.0, 1.25):
            with self.subTest(ridge=ridge):
                gate = adapter.generated_nested_improvement_gate({
                    "baseline_macro_family_mae": 1.0,
                    "ridge_macro_family_mae": ridge,
                })
                self.assertFalse(gate["passed"])
        self.assertEqual(
            adapter.generated_nested_improvement_gate(None)["status"],
            "unavailable",
        )

    def test_duplicate_imports_collapse_only_when_equivalent(self):
        first = {
            "index": "generated/chain/base-0001/3x3",
            "family": "generated/chain",
            "baseline_lb": 2,
            "compiled_ii": 3,
            "input_report_path": "/archive/first.json",
            "input_report_sha256": "first-digest",
        }
        repeated = dict(first)
        repeated.update({
            "input_report_path": "/archive/copy.json",
            "input_report_sha256": "copy-digest",
        })
        collapsed = adapter.deduplicate_samples_by_id([first, repeated])
        self.assertEqual(collapsed, [first])

        conflicting = dict(repeated)
        conflicting["compiled_ii"] = 4
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            adapter.deduplicate_samples_by_id([first, conflicting])

    def test_explicit_leakage_lineage_is_authoritative(self):
        row = {
            "lineage": "optimistic-family-name",
            "leakage_lineage_id": "shared-template-parent",
        }
        self.assertEqual(
            adapter.resolve_effective_lineage(row, "generated/chain", {}),
            ("shared-template-parent", "shared-template-parent"),
        )
        self.assertEqual(
            adapter.resolve_effective_lineage(
                row,
                "generated/chain",
                {"shared-template-parent": "merged-conservative-lineage"},
            ),
            ("shared-template-parent", "merged-conservative-lineage"),
        )
        self.assertEqual(
            adapter.resolve_effective_lineage(
                row,
                "generated/chain",
                {"generated/chain": "explicit-source-family-alias"},
            ),
            ("shared-template-parent", "explicit-source-family-alias"),
        )

    def test_tree_diagnostic_uses_the_same_structure_only_features(self):
        rows = []
        for index in range(4):
            row = {name: index for name in adapter.MODEL_FEATURE_NAMES}
            row.update({
                "baseline_lb": 1,
                "compiled_ii": 1 + index,
            })
            rows.append(row)
        model = adapter.fit_residual_tree(rows, max_depth=1, min_samples=1)
        self.assertEqual(
            model["feature_names"], list(adapter.MODEL_FEATURE_NAMES)
        )
        self.assertNotIn("baseline_lb", model["feature_names"])
        self.assertNotIn("rec_mii", model["feature_names"])
        self.assertNotIn("res_mii", model["feature_names"])

    def test_sibling_cost_loader_accepts_rec_res_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost_directory = root / "real-kernel-4x4"
            cost_directory.mkdir()
            (cost_directory / "cost.mlir").write_text(cost_text())
            recovered = adapter.load_sibling_cost_features(root / "report.json")
        self.assertIn("kernel-4x4", recovered)
        self.assertEqual(recovered["kernel-4x4"]["rec_mii"], 4)

    def test_portable_input_normalization_preserves_metadata(self):
        raw = {
            "sample_id": "suite/kernel/4x4",
            "group": "suite/kernel",
            "lower_bound": 6,
            "compiled_ii": 8,
            "features": {"rec_mii": 4},
            "metadata": {
                "lineage": "suite/kernel-template",
                "candidate_id": "mesh-4x4:heuristic",
                "rec_res_evidence": "neura_shared_rec_res_analysis_v1",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text("{}")
            row = adapter.normalize_input_sample(
                raw, report_path, "digest",
                {
                    "architecture": "/archive/architecture.yaml",
                    "architecture_sha256": "architecture-digest",
                    "neura": {"revision": "revision"},
                },
            )
        self.assertEqual(row["index"], raw["sample_id"])
        self.assertEqual(row["family"], raw["group"])
        self.assertEqual(row["baseline_lb"], 6)
        self.assertEqual(row["lineage"], "suite/kernel-template")
        self.assertEqual(row["input_report_sha256"], "digest")
        self.assertEqual(row["architecture_sha256"], "architecture-digest")
        self.assertEqual(row["mapper_revision"], "revision")
        self.assertEqual(row["rec_res_evidence"], "imported_report_unverified")

    def test_portable_input_normalization_accepts_proven_bound_alias(self):
        raw = {
            "sample_id": "suite/kernel/4x4",
            "group": "suite/kernel",
            "proven_lower_bound": 6,
            "compiled_ii": 8,
            "features": {"rec_mii": 4, "res_mii": 6},
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            report_path.write_text("{}")
            row = adapter.normalize_input_sample(raw, report_path, "digest")
        self.assertEqual(row["baseline_lb"], 6)

    def test_candidate_identity_includes_source_dfg_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.mlir"
            source_b = root / "b.mlir"
            architecture = root / "architecture.yaml"
            source_a.write_text("module { // a\n}\n")
            source_b.write_text("module { // b\n}\n")
            architecture.write_text("architecture: example\n")
            row_a = {}
            row_b = {}
            adapter.attach_sample_provenance(
                row_a, source_a, architecture, "3x3", "real/a"
            )
            adapter.attach_sample_provenance(
                row_b, source_b, architecture, "3x3", "real/b"
            )
        self.assertNotEqual(row_a["candidate_id"], row_b["candidate_id"])
        self.assertEqual(row_a["base_dfg_id"], row_a["source_sha256"])
        self.assertEqual(row_a["ranking_query_id"], row_a["base_dfg_id"])

    def test_interval_filters_synthetic_groups(self):
        model = {}
        rows = [
            {"family": "real-a", "prediction": 4, "compiled_ii": 3},
            {"family": "real-b", "prediction": 8, "compiled_ii": 6},
            {
                "family": "renamed-generator",
                "source_family": "synthetic-dfg",
                "prediction": 100,
                "compiled_ii": 1,
            },
        ]
        adapter.calibrate_unseen_family_interval(model, rows, quantile=1.0)
        self.assertEqual(model["unseen_group_absolute_error_radius"], 2.0)
        self.assertEqual(model["unseen_group_calibration_groups"], 2)

    def test_architecture_holdout_preserves_source_lineage_weights(self):
        rows = []
        for family_index, family in enumerate(("a", "b", "c")):
            for architecture_index, architecture in enumerate(("x", "y", "z")):
                row = {name: 0 for name in adapter.MODEL_FEATURE_NAMES}
                row.update({
                    "index": f"{family}-{architecture}",
                    "family": family,
                    "source_family": family,
                    "architecture_id": architecture,
                    "baseline_lb": 1,
                    "rec_mii": 1,
                    "res_mii": 1,
                    "lower_bound_source": "rec_res_max_v1",
                    "compiled_ii": 1 + family_index + architecture_index,
                })
                rows.append(row)
        result = adapter.nested_ridge_metadata_holdout(
            rows, "architecture_id", [1.0], [0.0]
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["training_weight_group"], "source_lineage")
        self.assertEqual(result["group_count"], 3)
        self.assertEqual(len(result["evaluation"]["rows"]), 9)
        self.assertEqual(
            set(result["evaluation"]
                ["chosen_hyperparameters_by_held_out_family"]),
            {"x", "y", "z"},
        )

        unavailable = adapter.nested_ridge_metadata_holdout(
            rows, "suite", [1.0], [0.0]
        )
        self.assertEqual(unavailable["status"], "unavailable_missing_metadata")

    def test_invoke_retains_only_bounded_diagnostics(self):
        previous = list(adapter.INVOCATION_FAILURES)
        adapter.INVOCATION_FAILURES.clear()
        try:
            ok = adapter.invoke(
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.stderr.write('x' * 10000); sys.exit(7)",
                ),
                timeout=5,
            )
            self.assertFalse(ok)
            failure = adapter.INVOCATION_FAILURES[-1]
            self.assertEqual(failure["returncode"], 7)
            self.assertLessEqual(len(failure["stderr_head"]), 1200)
            self.assertLessEqual(len(failure["stderr_tail"]), 2800)
        finally:
            adapter.INVOCATION_FAILURES[:] = previous

    def test_prediction_fixture_runs_rec_res_pass_without_mapper(self):
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mlir"
            architecture = root / "architecture.yaml"
            sample_dir = root / "prediction"
            source.write_text("module {}")
            architecture.write_text("architecture: example")
            sample_dir.mkdir()

            def fake_invoke(command, timeout):
                commands.append(tuple(command))
                Path(command[-1]).write_text(cost_text())
                return True

            with patch.object(adapter, "invoke", side_effect=fake_invoke), \
                    patch.object(
                        adapter, "graph_features_from_neura", return_value={}
                    ):
                result = adapter.collect_prediction_fixture(
                    Path("mlir-neura-opt"), sample_dir, "kernel", source,
                    architecture, 4, 4, 5,
                )
        self.assertIsNotNone(result)
        self.assertNotIn("compiled_ii", result)
        self.assertEqual(len(commands), 1)
        command = " ".join(commands[0])
        self.assertIn("--analyze-rec-res-mii=", command)
        self.assertNotIn("--map-to-accelerator", command)

    def test_motif_uses_pinned_architecture_and_target_rectangle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = neura_motifs.make_base_specs(
                1, seed=17, motifs=("chain",)
            )
            candidates = neura_motifs.make_candidates(
                base, root, ((3, 3),), ("neura-main",)
            )

            def fake_invoke(command, timeout):
                output = Path(command[-1])
                if any(
                    part.startswith("--analyze-rec-res-mii")
                    for part in command
                ):
                    output.write_text(cost_text(rec_mii=4, res_mii=5))
                else:
                    output.write_text(
                        "module attributes {"
                        'mapping_strategy = "heuristic", '
                        "x_tiles = 3 : i32, y_tiles = 3 : i32, "
                        "compiled_ii = 8 : i32, rec_mii = 4 : i32, "
                        "res_mii = 5 : i32} {}\n"
                    )
                return True

            with patch.object(adapter, "invoke", side_effect=fake_invoke):
                samples = [
                    adapter.collect_motif_sample(Path("opt"), candidate, 5)
                    for candidate in candidates
                ]

            for candidate, sample in zip(candidates, samples):
                self.assertIsNotNone(sample)
                sample = dict(sample)
                self.assertEqual(sample["architecture_variant"],
                                 candidate.architecture_variant)
                self.assertEqual(sample["split_domain"], 0)
                self.assertEqual(sample["target_config_id"], "prefix-3x3")

                # The main report assembly supplies these two provenance
                # fields before the frozen generated-sample validator runs.
                sample["declared_leakage_lineage_id"] = candidate.lineage
                sample["mapper_revision"] = (
                    machsuite_frozen.FROZEN_NEURA_REVISION
                )
                machsuite_frozen._validated_generated_sample(
                    sample, 0, machsuite_frozen.FROZEN_NEURA_REVISION
                )

    def test_motif_skips_mapper_when_lower_bound_exceeds_search_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = neura_motifs_v4.make_candidates(
                neura_motifs_v4.make_base_specs(
                    1, seed=17, motifs=("compute",)
                ),
                root, ((3, 3),), ("neura-main",),
            )[0]
            commands = []

            def fake_invocation(command, timeout):
                commands.append(tuple(command))
                Path(command[-1]).write_text(cost_text(rec_mii=21, res_mii=3))
                return True

            outcome = adapter.collect_motif_candidate(
                Path("opt"), candidate, 5, fake_invocation
            )
        self.assertEqual(outcome.status, "censored")
        self.assertEqual(outcome.stage, "mapper-search-interval")
        self.assertEqual(outcome.failure, "lower-bound-above-mapper-ceiling")
        self.assertEqual(outcome.analysis_facts["rec_mii"], 21)
        self.assertEqual(len(commands), 1)
        self.assertIn("--analyze-rec-res-mii", " ".join(commands[0]))
        self.assertNotIn("--map-to-accelerator", " ".join(commands[0]))

    def test_historical_v3_high_lower_bound_still_uses_original_mapper_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = neura_motifs.make_candidates(
                neura_motifs.make_base_specs(
                    1, seed=17, motifs=("chain",)
                ),
                root, ((3, 3),), ("neura-main",),
            )[0]
            commands = []

            def fake_invocation(command, timeout):
                commands.append(tuple(command))
                output = Path(command[-1])
                if "--analyze-rec-res-mii" in " ".join(command):
                    output.write_text(cost_text(rec_mii=21, res_mii=3))
                else:
                    output.write_text(
                        'mapping_info = {mapping_strategy = "heuristic", '
                        'x_tiles = 3 : i32, y_tiles = 3 : i32} '
                        "compiled_ii = 22 : i32 rec_mii = 21 : i32 "
                        "res_mii = 3 : i32"
                    )
                return True

            outcome = adapter.collect_motif_candidate(
                Path("opt"), candidate, 5, fake_invocation
            )
        self.assertEqual(outcome.status, "success")
        self.assertEqual(len(commands), 2)
        self.assertIn("--map-to-accelerator", " ".join(commands[1]))

    def test_feasibility_coverage_keeps_distinct_denominators(self):
        common = {
            "rec_mii": 2, "res_mii": 3, "lower_bound": 3,
            "mapper_ii_ceiling": 20,
            "cost_artifact_path": "cost.mlir",
            "cost_artifact_sha256": "0" * 64,
        }
        records = [
            {**common, "generator_family": "f", "rows": 2, "columns": 2,
             "operation_band": "low", "analysis_status": "success",
             "lower_bound_within_mapper_search_interval": True,
             "mapper_attempted": True, "status": "success"},
            {**common, "generator_family": "f", "rows": 2, "columns": 2,
             "operation_band": "low", "analysis_status": "success",
             "lower_bound_within_mapper_search_interval": True,
             "mapper_attempted": True, "status": "censored"},
            {**common, "rec_mii": 21, "lower_bound": 21,
             "generator_family": "f", "rows": 1, "columns": 1,
             "operation_band": "high", "analysis_status": "success",
             "lower_bound_within_mapper_search_interval": False,
             "mapper_attempted": False, "status": "censored"},
        ]
        result = adapter.motif_feasibility_coverage(records)
        overall = result["overall"]
        self.assertEqual(overall["declared_count"], 3)
        self.assertEqual(overall["feasible_search_interval_count"], 2)
        self.assertEqual(overall["outside_search_interval_count"], 1)
        self.assertEqual(overall["successful_label_count"], 1)
        self.assertEqual(overall["censored_feasible_count"], 1)
        self.assertEqual(overall["label_coverage_on_feasible"], 0.5)
        self.assertTrue(result["passed"])

        malformed = adapter.motif_feasibility_coverage([{
            "lower_bound_within_mapper_search_interval": True,
            "mapper_attempted": True,
            "status": "success",
        }])
        self.assertEqual(malformed["overall"]["analysis_valid_count"], 0)
        self.assertEqual(
            malformed["overall"]["feasible_search_interval_count"], 0
        )
        self.assertFalse(malformed["passed"])

    def test_v4_stratum_coverage_requires_complete_transpose_profile_band_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = neura_motifs_v4.make_candidates(
                neura_motifs_v4.make_base_specs(
                    15, seed=neura_motifs_v4.DEFAULT_SEED,
                    motifs=("compute",),
                ),
                root,
            )
            records = []
            for candidate in candidates:
                record = candidate.manifest_record()
                record["status"] = "success"
                records.append(record)
            family = "generated/motif/compute"
            result = adapter.motif_v4_stratum_coverage(records, (family,))
            self.assertTrue(result["passed"])
            self.assertEqual(result["cell_count"], result["expected_cell_count"])
            records[0]["status"] = "censored"
            strict = adapter.motif_v4_stratum_coverage(
                records, (family,), minimum_fraction=1.0
            )
            self.assertFalse(strict["passed"])

    def test_v4_acceptance_requires_strict_transfer_positive_signal_and_ranking(self):
        families = ("f0", "f1", "f2")
        family_groups = {
            family: {
                "quality": {"mae": 1.0},
                "positive_residual": {
                    "positive_residual_recall": 0.5,
                    "positive_subset_mae": 0.5,
                }
            }
            for family in families
        }
        evaluation = {
            "families": list(families),
            "outer_split_protocol": "leave_one_leakage_group_out",
            "baseline_macro_family_mae": 1.0,
            "ridge_macro_family_mae": 0.8,
            "baseline_positive_residual_metrics": {
                "positive_subset_mae": 2.0,
            },
            "ridge_positive_residual_metrics": {
                "positive_target_count": 3,
                "positive_prediction_count": 2,
                "positive_subset_mae": 0.5,
                "all_predictions_equal_lower_bound": False,
            },
            "stratified_metrics": {
                "generator_family": {
                    "lower_bound": {
                        "status": "ok",
                        "balanced_mae": 1.0,
                        "groups": family_groups,
                        "macro_positive_subset_mae": 2.0,
                    },
                    "model": {
                        "status": "ok",
                        "balanced_mae": 0.8,
                        "groups": family_groups,
                        "macro_positive_subset_mae": 0.5,
                    },
                },
                "target_shape": {
                    "lower_bound": {
                        "status": "ok", "balanced_mae": 1.0,
                        "groups": {"2x2": {"quality": {"mae": 1.0}}},
                    },
                    "model": {
                        "status": "ok", "balanced_mae": 0.75,
                        "groups": {"2x2": {"quality": {"mae": 0.75}}},
                    },
                },
                "operation_band": {
                    "lower_bound": {
                        "status": "ok", "balanced_mae": 1.0,
                        "groups": {"low": {"quality": {"mae": 1.0}}},
                    },
                    "model": {
                        "status": "ok", "balanced_mae": 0.8,
                        "groups": {"low": {"quality": {"mae": 0.8}}},
                    },
                },
            },
            "baseline_group_ranking": {
                "status": "ok",
                "candidate_identity_status": "complete",
                "missing_ranking_query_row_count": 0,
                "duplicate_candidate_rows_collapsed": 0,
                "macro_pairwise_concordance": 0.75,
                "eligible_ranking_query_count": 1,
                "groups": {"q": {"status": "eligible"}},
            },
            "ridge_group_ranking": {
                "status": "ok",
                "candidate_identity_status": "complete",
                "missing_ranking_query_row_count": 0,
                "duplicate_candidate_rows_collapsed": 0,
                "macro_pairwise_concordance": 0.8,
                "eligible_ranking_query_count": 1,
                "groups": {"q": {"status": "eligible"}},
            },
        }
        labelled = [
            {"generator_family": family, "compiled_ii": 3,
             "baseline_lb": 2, "base_dfg_id": family,
             "mechanism_profile": "profile", "operation_band": "low",
             "target_shape": "2x2"}
            for family in families
        ]
        policy = {
            "policy_version": "test-v4",
            "positive_residual_distribution": {
                "minimum_positive_base_dfgs_per_family": 1,
                "minimum_positive_mechanism_profiles_per_family": 1,
                "minimum_positive_operation_bands_per_family": 1,
                "minimum_positive_target_shapes_per_family": 1,
            },
        }
        result = adapter.generated_v4_acceptance_gates(
            {"status": "ok", "evaluation": evaluation}, labelled,
            {"passed": True}, {"passed": True, "overall": {"passed": True}},
            families, policy,
        )
        self.assertTrue(result["overall_passed"])
        evaluation["ridge_macro_family_mae"] = 1.0
        failed = adapter.generated_v4_acceptance_gates(
            {"status": "ok", "evaluation": evaluation}, labelled,
            {"passed": True}, {"passed": True, "overall": {"passed": True}},
            families, policy,
        )
        self.assertFalse(failed["overall_passed"])
        self.assertFalse(
            failed["gates"]["strict_generator_family_logo_improvement"]["passed"]
        )

        evaluation["ridge_macro_family_mae"] = 0.8
        family_groups["f0"]["positive_residual"][
            "positive_residual_recall"
        ] = "bogus"
        malformed = adapter.generated_v4_acceptance_gates(
            {"status": "ok", "evaluation": evaluation}, labelled,
            {"passed": True}, {"passed": True, "overall": {"passed": True}},
            families, policy,
        )
        self.assertFalse(malformed["overall_passed"])
        self.assertFalse(
            malformed["gates"]["positive_residual_quality"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
