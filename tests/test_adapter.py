import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters import machsuite_frozen, neura_experiment as adapter, neura_motifs


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

    def test_primary_model_features_exclude_bound_and_components(self):
        forbidden = {"baseline_lb", *adapter.LOWER_BOUND_COMPONENT_NAMES}
        self.assertTrue(forbidden.isdisjoint(adapter.MODEL_FEATURE_NAMES))

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

    def test_motif_split_domain_feature_comes_from_architecture_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = neura_motifs.make_base_specs(
                1, seed=17, motifs=("chain",)
            )
            candidates = neura_motifs.make_candidates(
                base, root, ((3, 3),), ("homogeneous", "split-domain")
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
                        "compiled_ii = 8 : i32, rec_mii = 4 : i32, "
                        "res_mii = 5 : i32} {}\n"
                    )
                return True

            with patch.object(adapter, "invoke", side_effect=fake_invoke):
                samples = [
                    adapter.collect_motif_sample(Path("opt"), candidate, 5)
                    for candidate in candidates
                ]

            for candidate, sample, expected in zip(
                candidates, samples, (0, 1)
            ):
                self.assertIsNotNone(sample)
                sample = dict(sample)
                self.assertEqual(sample["architecture_variant"],
                                 candidate.architecture_variant)
                self.assertEqual(sample["split_domain"], expected)

                # The main report assembly supplies these two provenance
                # fields before the frozen generated-sample validator runs.
                sample["declared_leakage_lineage_id"] = candidate.lineage
                sample["mapper_revision"] = (
                    machsuite_frozen.FROZEN_NEURA_REVISION
                )
                machsuite_frozen._validated_generated_sample(
                    sample, 0, machsuite_frozen.FROZEN_NEURA_REVISION
                )


if __name__ == "__main__":
    unittest.main()
