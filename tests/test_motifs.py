import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from adapters import neura_experiment, neura_motifs


class MotifCorpusTest(unittest.TestCase):
    def test_balanced_shape_design_pairs_full_array_with_one_secondary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bases = neura_motifs.make_base_specs(
                250, seed=31, motifs=("chain",)
            )
            candidates = neura_motifs.make_candidates(bases, root)
            self.assertEqual(len(candidates), 500)
            counts = {}
            for candidate in candidates:
                shape = (candidate.rows, candidate.columns)
                counts[shape] = counts.get(shape, 0) + 1
            self.assertEqual(counts[(4, 4)], 250)
            self.assertEqual(set(counts), set(neura_motifs.DEFAULT_SHAPES))
            for shape in set(neura_motifs.DEFAULT_SHAPES) - {(4, 4)}:
                self.assertIn(counts[shape], (31, 32))
            self.assertEqual(
                {candidate.architecture_sha256 for candidate in candidates},
                {neura_motifs.PINNED_ARCHITECTURE_SHA256},
            )
            self.assertEqual(
                len({candidate.architecture_path for candidate in candidates}), 1
            )

    def test_shape_parser_has_no_hidden_route_bound_tile_cap(self):
        self.assertEqual(neura_motifs.parse_shape("8x8"), (8, 8))
        with self.assertRaisesRegex(ValueError, "positive"):
            neura_motifs.parse_shape("0x8")

    def test_tiny_prediction_shapes_are_explicit_stress_candidates(self):
        self.assertIn((1, 1), neura_motifs.PREDICTION_SHAPES)
        self.assertIn((1, 2), neura_motifs.PREDICTION_SHAPES)
        self.assertNotIn((1, 1), neura_motifs.DEFAULT_SHAPES)
        self.assertNotIn((1, 2), neura_motifs.DEFAULT_SHAPES)
        base = neura_motifs.MotifBaseSpec("chain", 0, 1, 8)
        self.assertEqual(
            neura_motifs.candidate_shapes_for_base(
                base, ((1, 1), (1, 2), (4, 4))
            ),
            ((4, 4), (1, 1)),
        )
        with self.assertRaisesRegex(ValueError, "supported shape set"):
            neura_motifs.candidate_shapes_for_base(base, ((1, 3), (4, 4)))

    def test_generated_architectures_satisfy_main_branch_memory_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "neura-main.yaml"
            neura_motifs.write_architecture(
                path, 4, 4, "neura-main",
                neura_motifs.PINNED_REGISTERS_PER_TILE,
            )
            self.assertEqual(
                neura_motifs.sha256_file(path),
                neura_motifs.PINNED_ARCHITECTURE_SHA256,
            )

    def test_generation_is_deterministic_and_shape_independent(self):
        first = neura_motifs.generate_motif_mlir("mixed", 16, 123)
        second = neura_motifs.generate_motif_mlir("mixed", 16, 123)
        self.assertEqual(first, second)
        self.assertEqual(
            neura_motifs.canonical_dfg_sha256(first),
            neura_motifs.canonical_dfg_sha256(second),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = neura_motifs.MotifBaseSpec(
                "mixed", 0, 123, 16, root_seed=7,
                generator_family="generated/motif/mixed",
            )
            candidates = neura_motifs.make_candidates(
                (base,), root, ((3, 3), (3, 4), (4, 4)), ("neura-main",),
            )
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len({c.source_sha256 for c in candidates}), 1)
            self.assertEqual(len({c.canonical_dfg_sha256 for c in candidates}), 1)
            self.assertEqual(len({c.lineage for c in candidates}), 1)
            self.assertEqual(len({c.base_id for c in candidates}), 1)
        self.assertNotEqual(
            neura_motifs.canonical_dfg_sha256(first),
            neura_motifs.canonical_dfg_sha256(
                neura_motifs.generate_motif_mlir("mixed", 16, 124)
            ),
        )

    def test_family_stream_and_lineage_include_root_seed(self):
        all_motifs = neura_motifs.make_base_specs(
            2, seed=2026, motifs=("chain", "fanout", "mixed")
        )
        reordered = neura_motifs.make_base_specs(
            2, seed=2026, motifs=("mixed", "chain", "fanout")
        )
        def by_motif(specs):
            return {
                (spec.motif, spec.base_index):
                (spec.base_seed, spec.operation_count, spec.lineage)
                for spec in specs
            }
        self.assertEqual(by_motif(all_motifs), by_motif(reordered))
        changed_seed = neura_motifs.make_base_specs(
            2, seed=2027, motifs=("chain", "fanout", "mixed")
        )
        self.assertTrue(
            {spec.lineage for spec in all_motifs}.isdisjoint(
                {spec.lineage for spec in changed_seed}
            )
        )
        self.assertEqual(
            {spec.generator_family for spec in all_motifs},
            {"generated/motif/chain", "generated/motif/fanout", "generated/motif/random_dag"},
        )

    def test_motifs_have_binary_and_multi_input_graphs(self):
        legacy_motifs = {
            "chain", "fanout", "reduction", "diamond", "mixed", "random_dag",
        }
        for motif in legacy_motifs:
            text = neura_motifs.generate_motif_mlir(motif, 16, 9)
            operations = re.findall(r'"neura\.(?:add|mul)"', text)
            self.assertEqual(len(operations), 16, motif)
            self.assertIn("data_mov", text, motif)
        reduction = neura_motifs.generate_motif_mlir("reduction", 16, 9)
        self.assertGreaterEqual(reduction.count('"neura.add"'), 1)
        fanout = neura_motifs.generate_motif_mlir("fanout", 16, 9)
        # The same source constant occurs in at least two binary operands.
        self.assertGreaterEqual(fanout.count("%c0"), 3)
        diamond = neura_motifs.generate_motif_mlir("diamond", 16, 9)
        self.assertGreaterEqual(diamond.count("%c0"), 3)

    def test_v2_families_have_structural_contracts(self):
        recurrence = neura_motifs.generate_motif_mlir(
            "recurrence_chain", 16, 9
        )
        self.assertEqual(recurrence.count('"neura.add"') +
                         recurrence.count('"neura.mul"'), 16)
        self.assertEqual(recurrence.count("neura.reserve"), 1)
        self.assertEqual(recurrence.count("neura.phi_start"), 1)
        self.assertEqual(recurrence.count("neura.ctrl_mov"), 1)
        self.assertGreater(
            len(re.findall(r'"neura\.(?:add|mul)"', recurrence)), 1
        )

        control = neura_motifs.generate_motif_mlir(
            "predicated_diamond", 16, 9
        )
        self.assertGreater(control.count('"neura.icmp"'), 0)
        self.assertGreater(control.count('"neura.not"'), 0)
        self.assertGreater(control.count("neura.grant_predicate"), 0)
        self.assertEqual(
            control.count('"neura.add"') + control.count('"neura.mul"'), 16
        )
        # A complete diamond has two arm computations and one join.
        self.assertGreaterEqual(control.count("grant_predicate"), 2)

        pointer = neura_motifs.generate_motif_mlir("pointer_chase", 16, 9)
        self.assertIn("!llvm.ptr", pointer)
        self.assertGreaterEqual(pointer.count('"neura.gep"'), 2)
        self.assertGreaterEqual(pointer.count('"neura.load"'), 2)
        self.assertGreaterEqual(pointer.count('"neura.load"'),
                                pointer.count('"neura.gep"') - 1)
        self.assertEqual(pointer.count("neura.ctrl_mov"), 1)

    def test_v2_operation_count_changes_topology_not_literals(self):
        for motif in ("recurrence_chain", "predicated_diamond", "pointer_chase"):
            small = neura_motifs.generate_motif_mlir(motif, 8, 10)
            large = neura_motifs.generate_motif_mlir(motif, 16, 10)
            self.assertNotEqual(
                neura_motifs.canonical_dfg_sha256(small),
                neura_motifs.canonical_dfg_sha256(large),
                motif,
            )
            self.assertNotEqual(
                len(re.findall(r'(?m)^\s*(?:%[^=]+ = )?"?neura\.', small)),
                len(re.findall(r'(?m)^\s*(?:%[^=]+ = )?"?neura\.', large)),
                motif,
            )

    def test_v2_frozen_bands_are_mapper_feasible_with_explicit_stress_limits(self):
        self.assertIn("memory_stream", neura_motifs.DEFAULT_MOTIFS)
        self.assertIn("recurrence_chain", neura_motifs.DEFAULT_MOTIFS)
        self.assertLessEqual(
            max(high for _, high in neura_motifs.operation_bands_for_motif(
                "random_dag"
            )), 48
        )
        self.assertLessEqual(
            max(high for _, high in neura_motifs.operation_bands_for_motif(
                "recurrence_chain"
            )), 16
        )
        for motif in ("random_dag", "predicated_diamond", "pointer_chase"):
            self.assertGreaterEqual(
                neura_motifs.DIRECT_OPERATION_LIMITS[motif], 128
            )
        for motif in ("random_dag", "predicated_diamond", "pointer_chase"):
            specs = neura_motifs.make_base_specs(3, seed=17, motifs=(motif,))
            self.assertEqual(len(specs), 3)
            self.assertGreater(len(set(spec.operation_count for spec in specs)), 1)
        large = neura_motifs.generate_motif_mlir("random_dag", 128, 23)
        self.assertEqual(
            large.count('"neura.add"') + large.count('"neura.mul"'), 128
        )

    def test_random_dag_changes_edges_and_remains_connected(self):
        first = neura_motifs.generate_motif_mlir("random_dag", 24, 101)
        second = neura_motifs.generate_motif_mlir("random_dag", 24, 102)
        def dependencies(text):
            return [tuple(re.findall(
                rf'%m{index}[ab] = "neura\.data_mov"\((%[^)]+)\)', text
            )) for index in range(24)]

        first_edges = dependencies(first)
        second_edges = dependencies(second)
        self.assertNotEqual(first_edges, second_edges)
        # Apart from the first op, every random op has an earlier %v producer
        # feeding one of its data-move operands, which guarantees weak
        # connectivity by induction.
        for index in range(1, 24):
            sources = first_edges[index]
            self.assertEqual(len(sources), 2)
            self.assertTrue(
                any(source.startswith("%v") for source in sources),
                (index, sources),
            )
        # Every emitted constant is consumed, so the whole DFG rather than
        # only the binary-operation projection belongs to one weak component.
        for constant in re.findall(r'(%c\d+) = "neura\.const"', first):
            self.assertGreaterEqual(first.count(constant), 2, constant)
        self.assertNotEqual(
            neura_motifs.canonical_dfg_sha256(first),
            neura_motifs.canonical_dfg_sha256(second),
        )

    def test_canonical_dfg_hash_ignores_constant_literal_values(self):
        original = neura_motifs.generate_motif_mlir("chain", 8, 77)
        changed = re.sub(
            r"value = \d+ : i32", "value = 123456789 : i32", original,
        )
        self.assertNotEqual(original, changed)
        self.assertEqual(
            neura_motifs.canonical_dfg_sha256(original),
            neura_motifs.canonical_dfg_sha256(changed),
        )
        recurrence = neura_motifs.generate_motif_mlir("recurrence_chain", 8, 77)
        changed_recurrence = re.sub(
            r"rhs_value = \d+ : i32", "rhs_value = 987654321 : i32",
            recurrence,
        )
        self.assertEqual(
            neura_motifs.canonical_dfg_sha256(recurrence),
            neura_motifs.canonical_dfg_sha256(changed_recurrence),
        )

    def test_stratification_changes_real_operation_count(self):
        specs = neura_motifs.make_base_specs(9, seed=3, motifs=("chain",))
        counts = [spec.operation_count for spec in specs]
        self.assertEqual(len(counts), 9)
        for index, count in enumerate(counts):
            low, high = neura_motifs.operation_bands_for_motif("chain")[
                index % len(neura_motifs.OPERATION_BANDS)
            ]
            self.assertGreaterEqual(count, low)
            self.assertLessEqual(count, high)
            text = neura_motifs.generate_motif_mlir(
                "chain", count, specs[index].base_seed
            )
            self.assertEqual(
                len(re.findall(r'"neura\.(?:add|mul)"', text)), count
            )
        self.assertGreater(len(set(counts)), 1)

    def test_generator_family_holdout_has_one_identity_per_motif(self):
        rows = []
        for index, motif in enumerate(neura_motifs.DEFAULT_MOTIFS):
            row = {name: 0 for name in neura_experiment.MODEL_FEATURE_NAMES}
            row.update({
                "index": f"{motif}-base",
                "family": f"generated/motif-v1/{motif}/base-{index}",
                "baseline_lb": 1,
                "rec_mii": 1,
                "res_mii": 1,
                "lower_bound_source": "rec_res_max_v1",
                "compiled_ii": 1 + index % 2,
                "source_kind": "generated",
                "generator_family": f"generated/motif/{motif}",
                "lineage": f"generated/motif-v1/{motif}/base-{index}",
                "training_stratum": "generated",
                "leakage_lineage_id": (
                    f"generated/motif-v1/{motif}/base-{index}"
                ),
                "base_dfg_id": f"dfg-{index}",
                "ranking_query_id": f"dfg-{index}",
                "candidate_id": f"candidate-{index}",
            })
            rows.append(row)
        result = neura_experiment.nested_ridge_metadata_holdout(
            rows, "generator_family", [1.0], [0.0]
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["group_count"], len(neura_motifs.DEFAULT_MOTIFS))
        self.assertEqual(
            set(result["evaluation"]["chosen_hyperparameters_by_held_out_family"]),
            {f"generated/motif/{motif}" for motif in neura_motifs.DEFAULT_MOTIFS},
        )

    def test_manifest_is_predeclared_and_updates_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bases = neura_motifs.make_base_specs(1, seed=11, motifs=("chain",))
            candidates = neura_motifs.make_candidates(
                bases, root, ((3, 3), (3, 4)), ("neura-main",),
            )
            manifest_path = root / "corpus-manifest.json"
            manifest = neura_motifs.make_manifest(
                candidates, root, 11, ("chain",), ((3, 3), (3, 4))
            )
            neura_motifs.atomic_write_json(manifest_path, manifest)
            loaded = json.loads(manifest_path.read_text())
            self.assertEqual(loaded["status"], "predeclared")
            self.assertEqual(loaded["summary"]["declared_count"], 2)
            self.assertTrue(all(c["status"] == "declared" for c in loaded["candidates"]))
            first = loaded["candidates"][0]
            self.assertEqual(first["leakage_lineage_id"], first["lineage"])
            self.assertEqual(
                first["base_dfg_id"], first["canonical_dfg_sha256"]
            )
            self.assertEqual(
                first["ranking_query_id"], first["canonical_dfg_sha256"]
            )
            self.assertEqual(first["training_stratum"], "generated")
            # A timeout/nonzero mapper invocation is represented as censored,
            # never as a fabricated compiled-II label.
            neura_motifs.update_manifest_candidate(
                manifest_path, candidates[0].candidate_id, "censored", "mapper",
                "timeout",
            )
            loaded = json.loads(manifest_path.read_text())
            record = next(
                c for c in loaded["candidates"]
                if c["id"] == candidates[0].candidate_id
            )
            self.assertEqual(record["status"], "censored")
            self.assertEqual(record["stage"], "mapper")
            self.assertNotIn("compiled_ii", record)
            self.assertEqual(loaded["summary"]["censored_count"], 1)
            self.assertEqual(loaded["status"], "partial")
            neura_motifs.update_manifest_candidate(
                manifest_path, candidates[1].candidate_id, "success", "mapper",
                updates={"sample_id": candidates[1].candidate_id,
                         "cost_artifact_path": "cost.mlir",
                         "mapped_artifact_path": "mapped.mlir",
                         "compiled_ii": 4},
            )
            loaded = json.loads(manifest_path.read_text())
            success = next(
                c for c in loaded["candidates"]
                if c["id"] == candidates[1].candidate_id
            )
            self.assertEqual(success["sample_id"], candidates[1].candidate_id)
            self.assertEqual(success["compiled_ii"], 4)
            self.assertEqual(success["cost_artifact_path"], "cost.mlir")

    def test_predeclared_manifest_is_reproducible_across_output_roots(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifests = []
            for directory in (first, second):
                root = Path(directory)
                bases = neura_motifs.make_base_specs(
                    1, seed=19, motifs=("fanout", "chain")
                )
                candidates = neura_motifs.make_candidates(
                    bases, root, ((3, 3),), ("neura-main",)
                )
                manifests.append(neura_motifs.make_manifest(
                    candidates, root, 19, ("fanout", "chain"), ((3, 3),)
                ))
            self.assertEqual(manifests[0], manifests[1])

    def test_duplicate_base_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = neura_motifs.MotifBaseSpec(
                "chain", 0, 7, 8, root_seed=1,
                generator_family="generated/motif/chain",
            )
            second = neura_motifs.MotifBaseSpec(
                "chain", 1, 7, 8, root_seed=2,
                generator_family="generated/motif/chain",
            )
            with self.assertRaisesRegex(ValueError, "duplicate canonical DFG"):
                neura_motifs.make_candidates((first, second), root)

    def test_cross_family_duplicate_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # These bases deliberately have the same base_id but different
            # lineages.  A canonical collision must still fail closed.
            first = neura_motifs.MotifBaseSpec(
                "chain", 0, 7, 8, root_seed=1,
                generator_family="generated/motif/chain",
            )
            second = neura_motifs.MotifBaseSpec(
                "fanout", 0, 9, 8, root_seed=1,
                generator_family="generated/motif/fanout",
            )
            common = neura_motifs.generate_motif_mlir("chain", 8, 7)
            with mock.patch.object(
                neura_motifs, "generate_motif_mlir", return_value=common
            ):
                with self.assertRaisesRegex(
                    ValueError, "duplicate canonical DFG"
                ):
                    neura_motifs.make_candidates((first, second), root)


if __name__ == "__main__":
    unittest.main()
