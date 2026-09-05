import importlib.util
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Model 2 optional PyTorch dependency")
class GraphModelTest(unittest.TestCase):
    def setUp(self):
        global torch
        global CandidateRecord, QueryRecord, resolve_device, split_queries
        global load_terminal_manifest
        global add_training_only_queries, attach_placement_supervision
        global subsample_training_queries
        global shape_grouped_query_batches
        global batch_placement_targets, _validation_score
        global JointGraphShapeModel, Model2Config, candidate_context
        global censored_top1_metrics, make_cgra_graph, model2_loss
        global pad_shortest_path_distances, parse_neura_dfg
        global dfg_structural_context, DFG_STRUCTURAL_CONTEXT_NAMES
        global parse_neura_route_expanded_dfg
        global ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES
        global parse_neura_mapped_placements
        global frozen_adapter, neura_motifs_v7
        global AMOEBA_STATIC_PROTOCOL, AMOEBA_STATIC_SHAPE_PROTOCOL
        import torch
        from adapters import neura_graph_frozen as frozen_adapter
        from adapters import neura_motifs_v7
        from adapters.neura_graph_experiment import (
            CandidateRecord, QueryRecord, add_training_only_queries,
            attach_placement_supervision, batch_placement_targets,
            load_terminal_manifest, resolve_device, split_queries,
            shape_grouped_query_batches,
            subsample_training_queries,
            _validation_score,
        )
        from cgra_ii_predictor.graph_model import (
            JointGraphShapeModel, Model2Config, candidate_context,
            censored_top1_metrics, make_cgra_graph, model2_loss,
            dfg_structural_context, DFG_STRUCTURAL_CONTEXT_NAMES,
            pad_shortest_path_distances, parse_neura_dfg,
            parse_neura_route_expanded_dfg,
            parse_neura_mapped_placements,
            ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES,
        )
        from cgra_ii_predictor.shape_protocol import (
            AMOEBA_STATIC_PROTOCOL, AMOEBA_STATIC_SHAPE_PROTOCOL,
        )

    def test_device_selection_never_silently_ignores_explicit_cuda(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with self.assertRaisesRegex(ValueError, "auto, cpu, or cuda"):
            resolve_device("tpu")
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")

    def test_validation_selection_prioritizes_continuous_ii_mae(self):
        def evaluation(mae, macro_mae, decision_mae, exact):
            return {
                "successful_candidate_point_error": {
                    "mae": mae, "macro_query_mae": macro_mae,
                },
                "successful_candidate_ii_decision": {
                    "mae": decision_mae, "exact_accuracy": exact,
                    "within_one_accuracy": 0.9,
                },
                "success_classifier": {"brier_score": 0.1},
            }

        lower_mae = evaluation(0.5, 0.6, 0.7, 0.4)
        higher_exact = evaluation(0.6, 0.5, 0.5, 0.9)
        self.assertLess(
            _validation_score(lower_mae), _validation_score(higher_exact)
        )

    def test_partial_shape_training_queries_are_batched_separately(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        """)

        def candidate(identity, rows, columns):
            return CandidateRecord(
                candidate_id=identity, ranking_query_id=identity,
                rows=rows, columns=columns, rec_mii=1, res_mii=1,
                lower_bound=1, status="success", compiled_ii=1,
            )

        queries = [
            QueryRecord("full", "family", graph, (
                candidate("full-4x4", 4, 4),
                candidate("full-4x8", 4, 8),
            )),
            QueryRecord("partial-a", "family", graph, (
                candidate("partial-a", 4, 4),
            )),
            QueryRecord("partial-b", "family", graph, (
                candidate("partial-b", 4, 4),
            )),
        ]
        batches = list(shape_grouped_query_batches(
            queries, batch_size=8, seed=7,
        ))
        self.assertEqual(len(batches), 2)
        self.assertEqual(
            sorted({len(query.candidates) for query in batch}.pop()
                   for batch in batches),
            [1, 2],
        )

    def test_frozen_evaluator_loads_weights_without_constructing_optimizer(self):
        config = frozen_adapter.frozen_config()
        model = JointGraphShapeModel(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            base = neura_motifs_v7.make_base_specs(
                1, neura_motifs_v7.DEFAULT_SEED, motifs=("compute",)
            )[0]
            candidates = neura_motifs_v7.make_candidates((base,), corpus)
            manifest = neura_motifs_v7.make_manifest(
                candidates, corpus, neura_motifs_v7.DEFAULT_SEED,
                ("compute",), neura_motifs_v7.DEFAULT_SHAPES,
            )
            for index, record in enumerate(manifest["candidates"]):
                record.update({
                    "stage": "complete" if index < 2 else "mapper",
                    "status": "success" if index < 2 else "censored",
                    "rec_mii": 1,
                    "res_mii": 4,
                    "lower_bound": 4,
                })
                if index < 2:
                    record["compiled_ii"] = 5 + index
                else:
                    record["failure"] = "timeout"
            manifest["status"] = "complete"
            manifest_path = corpus / "corpus-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            model_path = root / "model.pt"
            torch.save({
                "schema_version": frozen_adapter.ARTIFACT_SCHEMA_VERSION,
                "status": "frozen_before_motif_v7_mapper_labels",
                "model_class": neura_motifs_v7.MODEL2_CONFIG["class"],
                "config": config.to_dict(),
                "candidate_context_names": list(
                    frozen_adapter.CANDIDATE_CONTEXT_NAMES
                ),
                "state_dict": model.state_dict(),
                "training": {
                    "generator_version": "motif-v6",
                    "manifest_sha256": "training-manifest",
                    "query_ids": ["different-query"],
                },
                "freeze_contract": {"motif_v7_labels_permitted": False},
            }, model_path)
            args = argparse.Namespace(
                model=model_path, manifest=manifest_path,
                output_dir=root / "evaluation",
            )
            with mock.patch.object(
                torch.optim, "AdamW",
                side_effect=AssertionError("evaluator constructed an optimizer"),
            ):
                self.assertEqual(frozen_adapter.evaluate(args), 2)
            report = json.loads((
                root / "evaluation/evaluation-report.json"
            ).read_text())
            self.assertFalse(report[
                "motif_v7_labels_used_for_training_or_model_selection"
            ])
            self.assertFalse(report["acceptance_gates"]["coverage"])

    def test_parser_removes_moves_but_preserves_dependencies(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %am = "neura.data_mov"(%a) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %bm = "neura.data_mov"(%b) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %c = "neura.add"(%am, %bm) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        graph = parse_neura_dfg(source)
        self.assertEqual(len(graph.node_types), 3)
        self.assertEqual(graph.edges, ((0, 2), (1, 2)))
        self.assertEqual(graph.node_features[0][6], 1.0)
        self.assertEqual(graph.node_features[2][7], 1.0)

    def test_route_expanded_parser_preserves_moves_return_and_feedback(self):
        source = """
        %init = "neura.constant"() : () -> !neura.data<i32, i1>
        %reserved = neura.reserve : !neura.data<i32, i1>
        %init_mov = "neura.data_mov"(%init) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %state = neura.phi_start %init_mov, %reserved : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
        %next = "neura.add"(%state, %init) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %back = "neura.data_mov"(%next) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        neura.ctrl_mov %back -> %reserved : !neura.data<i32, i1> !neura.data<i32, i1>
        func.return
        """
        graph = parse_neura_route_expanded_dfg(source)
        self.assertEqual(len(graph.node_types), 7)
        self.assertIn((5, 1), graph.edges)
        self.assertIn((4, 3), graph.semantic_edges)
        materialized = ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES.index(
            "is_materialized"
        )
        recurrence = ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES.index(
            "is_recurrence_cycle"
        )
        self.assertEqual(
            sum(row[materialized] for row in graph.node_features), 4.0
        )
        self.assertEqual(
            sum(row[recurrence] for row in graph.node_features), 4.0
        )

    def test_route_expanded_model_counts_only_materialized_pe_demand(self):
        graph = parse_neura_route_expanded_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %m = "neura.data_mov"(%a) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %b = "neura.add"(%m, %a) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        func.return
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="residual_pointwise",
            dfg_representation="route_expanded_v2",
            dfg_message_mode="dual_mean",
        )
        model = JointGraphShapeModel(config).eval()
        context = torch.tensor([[candidate_context(1, 2, 1, 2, 2)]])
        output = model([graph], [make_cgra_graph(1, 2)], context)
        expected_pressure = math.log1p(3.0 / 2.0) / math.log(129.0)
        self.assertAlmostEqual(
            float(output["cross_attention_context"][0, 0, 0]),
            expected_pressure,
        )
        self.assertEqual(
            config.to_dict()["dfg_representation"], "route_expanded_v2"
        )
        self.assertEqual(config.to_dict()["dfg_message_mode"], "dual_mean")
        self.assertEqual(Model2Config(
            interaction_mode="residual_pointwise",
            dfg_representation="route_expanded_v2",
            placement_loss_weight=0.1,
        ).validate().placement_loss_weight, 0.1)
        with self.assertRaisesRegex(ValueError, "route_expanded_v2"):
            Model2Config(dfg_message_mode="dual_mean").validate()

    def test_route_expanded_materialized_pool_excludes_movement_nodes(self):
        graph = parse_neura_route_expanded_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %m = "neura.data_mov"(%a) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %b = "neura.add"(%m, %a) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        func.return %b : !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="residual_pointwise",
            dfg_representation="route_expanded_v2",
            dfg_message_mode="dual_mean",
            dfg_pool_mode="materialized_only",
        )
        model = JointGraphShapeModel(config).eval()
        original_pool = model.dfg_encoder.pool_nodes
        with mock.patch.object(
            model.dfg_encoder, "pool_nodes", wraps=original_pool,
        ) as pool:
            model(
                [graph], [make_cgra_graph(1, 2)],
                torch.tensor([[candidate_context(1, 2, 1, 2, 2)]]),
            )
        pool_mask = pool.call_args.args[1]
        self.assertEqual(int(pool_mask.sum()), 3)
        structural = dfg_structural_context(graph)
        self.assertGreater(structural[0], structural[5])
        self.assertEqual(config.to_dict()["dfg_pool_mode"], "materialized_only")
        with self.assertRaisesRegex(ValueError, "materialized-only"):
            Model2Config(dfg_pool_mode="materialized_only").validate()

    def test_structural_context_exposes_exact_long_range_graph_facts(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %d = "neura.mul"(%c, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        values = dfg_structural_context(graph)
        self.assertEqual(len(values), len(DFG_STRUCTURAL_CONTEXT_NAMES))
        denominator = math.log1p(4096.0)
        expected = [
            math.log1p(value) / denominator
            for value in (4, 4, 3, 2, 3)
        ]
        # Four unique edges: a->c, b->c, c->d, and b->d.  Their maximum
        # live-edge cut is three after node b.
        np_values = torch.tensor(values)
        self.assertTrue(torch.allclose(
            np_values[:5], torch.tensor(expected), atol=1e-7,
        ))
        self.assertTrue(torch.allclose(
            np_values[:5], np_values[5:10], atol=1e-7,
        ))
        self.assertEqual(values[-1], 0.0)

        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="continuous_residual_pointwise",
            dfg_summary_mode="structural_v1",
        )
        model = JointGraphShapeModel(config).eval()
        output = model(
            [graph], [make_cgra_graph(2, 2)],
            torch.tensor([[candidate_context(2, 2, 1, 2, 2)]]),
        )
        self.assertEqual(tuple(output["predicted_ii"].shape), (1, 1))
        self.assertEqual(config.to_dict()["dfg_summary_mode"], "structural_v1")
        with self.assertRaisesRegex(ValueError, "dfg_summary_mode"):
            Model2Config(dfg_summary_mode="unknown").validate()

    def test_mapped_placements_align_materialized_operations_to_pes(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %m = "neura.data_mov"(%a) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %c = "neura.add"(%m, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        mapped = """
        %0 = "neura.constant"() {mapping_locs = [{resource = "tile", x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
        %1 = "neura.constant"() {mapping_locs = [{resource = "tile", x = 1 : i32, y = 0 : i32}]} : () -> !neura.data<i32, i1>
        %2 = "neura.data_mov"(%0) {mapping_locs = [{resource = "link"}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %3 = "neura.add"(%2, %1) {mapping_locs = [{resource = "tile", x = 1 : i32, y = 0 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        graph = parse_neura_dfg(source)
        self.assertEqual(
            parse_neura_mapped_placements(mapped, graph, 1, 2),
            (0, 1, 1),
        )

    def test_mapped_placements_accept_v8_wide_mapper_shape(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        """
        mapped = """
        %0 = "neura.constant"() {mapping_locs = [{resource = "tile", x = 15 : i32, y = 3 : i32}]} : () -> !neura.data<i32, i1>
        """
        self.assertEqual(
            parse_neura_mapped_placements(
                mapped, parse_neura_dfg(source), 4, 16,
                AMOEBA_STATIC_SHAPE_PROTOCOL,
            ),
            (63,),
        )

    def test_oriented_shape_context_and_graph_are_not_transpose_aliases(self):
        two_by_three = candidate_context(2, 3, 1, 4, 4)
        three_by_two = candidate_context(3, 2, 1, 4, 4)
        self.assertNotEqual(two_by_three, three_by_two)
        self.assertGreater(two_by_three[6], 0.0)
        self.assertLess(three_by_two[6], 0.0)
        self.assertNotEqual(
            make_cgra_graph(2, 3).node_features,
            make_cgra_graph(3, 2).node_features,
        )

    def test_amoeba_shape_protocol_builds_real_meshes_up_to_64_pes(self):
        for rows, columns in AMOEBA_STATIC_PROTOCOL.mapper_shapes:
            graph = make_cgra_graph(
                rows, columns, AMOEBA_STATIC_SHAPE_PROTOCOL,
            )
            context = candidate_context(
                rows, columns, 1, 2, 2, 20,
                AMOEBA_STATIC_SHAPE_PROTOCOL,
            )
            self.assertEqual(len(graph.node_types), rows * columns)
            self.assertLessEqual(context[3], 1.0)
            self.assertLessEqual(context[4], 1.0)
            self.assertLessEqual(context[5], 1.0)
            self.assertLessEqual(context[7], 1.0)
            self.assertLessEqual(context[8], 1.0)
            self.assertLessEqual(context[9], 1.0)
        self.assertEqual(
            len(make_cgra_graph(
                8, 8, AMOEBA_STATIC_SHAPE_PROTOCOL,
            ).node_types),
            64,
        )
        with self.assertRaisesRegex(ValueError, "outside finite protocol"):
            make_cgra_graph(4, 5, AMOEBA_STATIC_SHAPE_PROTOCOL)

    def test_peak_load_normalization_is_shape_protocol_versioned(self):
        def normalized_peak(config):
            model = JointGraphShapeModel(config)
            hidden = config.hidden_dimension
            _, context, _, _ = model._candidate_conditioned_interaction(
                torch.zeros((1, 1, hidden)),
                torch.ones((1, 1), dtype=torch.bool),
                torch.zeros((1, 16, hidden)),
                torch.ones((1, 16), dtype=torch.bool),
            )
            return float(context[0, 0, 1])

        legacy = Model2Config(interaction_mode="residual_pointwise")
        amoeba = Model2Config(
            interaction_mode="residual_pointwise",
            shape_protocol=AMOEBA_STATIC_SHAPE_PROTOCOL,
            routing_peak_normalization="protocol_max_tiles_v1",
        )
        self.assertAlmostEqual(normalized_peak(legacy), 1.0 / 16.0)
        self.assertAlmostEqual(normalized_peak(amoeba), 1.0 / 64.0)

        legacy_amoeba_artifact = Model2Config(
            interaction_mode="residual_pointwise",
            shape_protocol=AMOEBA_STATIC_SHAPE_PROTOCOL,
        )
        self.assertAlmostEqual(
            normalized_peak(legacy_amoeba_artifact), 1.0 / 16.0,
        )
        self.assertNotIn(
            "routing_peak_normalization", legacy_amoeba_artifact.to_dict(),
        )
        self.assertEqual(
            amoeba.to_dict()["routing_peak_normalization"],
            "protocol_max_tiles_v1",
        )

    def test_v8_manifest_candidate_order_matches_shape_graph_protocol(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "input.mlir"
            source.write_text(
                '%0 = "neura.constant"() : () -> !neura.data<i32, i1>\n'
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            mapper_to_physical = {
                mapper: physical
                for physical, mapper in AMOEBA_STATIC_PROTOCOL.physical_to_mapper
            }
            candidates = []
            for rows, columns in reversed(AMOEBA_STATIC_PROTOCOL.mapper_shapes):
                physical = mapper_to_physical[(rows, columns)]
                candidates.append({
                    "candidate_id": f"q/{rows}x{columns}",
                    "ranking_query_id": "q",
                    "generator_family": "test",
                    "source_path": source.name,
                    "source_sha256": source_sha,
                    "status": "censored",
                    "rows": rows,
                    "columns": columns,
                    "mapper_tile_rows": rows,
                    "mapper_tile_cols": columns,
                    "physical_cgra_rows": physical[0],
                    "physical_cgra_cols": physical[1],
                    "rec_mii": 1,
                    "res_mii": 1,
                    "lower_bound": 1,
                })
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "shape_protocol": AMOEBA_STATIC_PROTOCOL.to_dict(),
                "candidates": candidates,
            }))
            _, queries = load_terminal_manifest(
                manifest_path, "semantic_v1", AMOEBA_STATIC_SHAPE_PROTOCOL,
            )
            self.assertEqual(
                tuple(
                    (candidate.rows, candidate.columns)
                    for candidate in queries[0].candidates
                ),
                AMOEBA_STATIC_PROTOCOL.mapper_shapes,
            )

    def test_amoeba_pointwise_prediction_is_batch_independent(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="residual_pointwise",
            shape_protocol=AMOEBA_STATIC_SHAPE_PROTOCOL,
        )
        model = JointGraphShapeModel(config).eval()
        shapes = [
            make_cgra_graph(rows, cols, config.shape_protocol)
            for rows, cols in AMOEBA_STATIC_PROTOCOL.mapper_shapes
        ]
        contexts = [
            candidate_context(
                rows, cols, 1, 2, 2, 20, config.shape_protocol,
            )
            for rows, cols in AMOEBA_STATIC_PROTOCOL.mapper_shapes
        ]
        with torch.no_grad():
            batched = model([graph], shapes, torch.tensor([contexts]))
            for index, (shape, context) in enumerate(zip(shapes, contexts)):
                point = model([graph], [shape], torch.tensor([[context]]))
                for key in (
                    "predicted_ii", "predicted_ii_std",
                    "success_probability",
                ):
                    self.assertTrue(torch.allclose(
                        batched[key][0, index], point[key][0, 0],
                        atol=1e-6, rtol=1e-6,
                    ), key)
        self.assertEqual(
            config.to_dict()["shape_protocol"],
            AMOEBA_STATIC_SHAPE_PROTOCOL,
        )
        self.assertNotIn("shape_protocol", Model2Config().to_dict())

    def test_censored_top1_never_imputes_a_numeric_ii(self):
        rows = [[
            {"candidate_id": "q/1x1", "ranking_query_id": "q",
             "rows": 1, "columns": 1, "status": "censored",
             "compiled_ii": None, "score": 1.0},
            {"candidate_id": "q/1x2", "ranking_query_id": "q",
             "rows": 1, "columns": 2, "status": "success",
             "compiled_ii": 6, "score": 2.0},
            {"candidate_id": "q/2x2", "ranking_query_id": "q",
             "rows": 2, "columns": 2, "status": "success",
             "compiled_ii": 5, "score": 3.0},
        ]]
        result = censored_top1_metrics(rows, "score")
        self.assertEqual(result["eligible_query_count"], 1)
        self.assertEqual(result["selected_success_rate"], 0.0)
        self.assertEqual(
            result["queries"]["q"]["compiled_ii_regret"], None
        )
        self.assertGreater(result["mean_timeout_penalized_regret"], 0.0)

    def test_topk_reports_transpose_equivalent_and_optimal_ii_recall(self):
        rows = [[
            {"candidate_id": "q/1x1", "ranking_query_id": "q",
             "rows": 1, "columns": 1, "status": "censored",
             "compiled_ii": None, "score": 1.0},
            {"candidate_id": "q/3x2", "ranking_query_id": "q",
             "rows": 3, "columns": 2, "status": "success",
             "compiled_ii": 6, "score": 2.0},
            {"candidate_id": "q/2x3", "ranking_query_id": "q",
             "rows": 2, "columns": 3, "status": "success",
             "compiled_ii": 5, "score": 3.0},
            {"candidate_id": "q/2x2", "ranking_query_id": "q",
             "rows": 2, "columns": 2, "status": "success",
             "compiled_ii": 7, "score": 4.0},
        ]]
        result = censored_top1_metrics(rows, "score")
        self.assertEqual(result["shape_metric_role"], "downstream_diagnostic_only")
        self.assertEqual(result["shape_equivalence"], "transpose_equivalent")
        self.assertEqual(result["strict_top2_accuracy"], 0.0)
        self.assertEqual(result["strict_top3_accuracy"], 1.0)
        self.assertEqual(result["transpose_equivalent_top1_accuracy"], 0.0)
        self.assertEqual(result["transpose_equivalent_top2_accuracy"], 1.0)
        self.assertEqual(result["optimal_ii_top2_rate"], 0.0)
        self.assertEqual(result["optimal_ii_top3_rate"], 1.0)
        self.assertEqual(result["any_success_top1_rate"], 0.0)
        self.assertEqual(result["any_success_top2_rate"], 1.0)

    def test_joint_model_loss_backpropagates_all_three_heads(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        graph = parse_neura_dfg(source)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
        )
        model = JointGraphShapeModel(config)
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 2, 2),
        ]])
        output = model([graph], shapes, context)
        losses = model2_loss(
            output, torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 2.0]]), torch.tensor([[False, True]]),
            torch.tensor([1]), config,
        )
        losses["total"].backward()
        self.assertTrue(math.isfinite(float(losses["total"])))
        self.assertTrue(any(
            parameter.grad is not None and torch.any(parameter.grad != 0)
            for parameter in model.parameters()
        ))

    def test_cross_attention_is_candidate_conditioned_and_backpropagates(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %d = "neura.mul"(%c, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        graph = parse_neura_dfg(source)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="cross_attention",
        )
        model = JointGraphShapeModel(config)
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 2, 2),
        ]])
        output = model([graph], shapes, context)
        cross_context = output["cross_attention_context"]
        self.assertEqual(tuple(cross_context.shape), (1, 2, 3))
        self.assertGreater(
            float(cross_context[0, 0, 0]),
            float(cross_context[0, 1, 0]),
        )
        losses = model2_loss(
            output, torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 2.0]]), torch.tensor([[False, True]]),
            torch.tensor([1]), config,
        )
        losses["total"].backward()
        for name in (
            "cross_dfg_query.weight", "cross_cgra_key.weight",
            "cross_cgra_value.weight", "cross_node_fusion.0.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.any(gradient != 0), name)

    def test_pooled_config_keeps_legacy_checkpoint_contract(self):
        self.assertNotIn("interaction_mode", Model2Config().to_dict())
        self.assertEqual(
            Model2Config(interaction_mode="cross_attention").to_dict()[
                "interaction_mode"
            ],
            "cross_attention",
        )
        with self.assertRaisesRegex(ValueError, "interaction_mode"):
            Model2Config(interaction_mode="unknown").validate()

    def test_routing_set_ranker_models_paths_and_candidate_competition(self):
        source = """
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        %d = "neura.mul"(%c, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """
        graph = parse_neura_dfg(source)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="routing_set_attention",
            candidate_set_layers=1, candidate_set_heads=4,
        )
        model = JointGraphShapeModel(config)
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 2, 2),
        ]])
        output = model([graph], shapes, context)
        self.assertEqual(tuple(output["routing_context"].shape), (1, 2, 3))
        self.assertEqual(tuple(output["ranking_logits"].shape), (1, 2))
        self.assertGreater(float(output["routing_context"][0, 1, 0]), 0.0)

        changed_context = context.clone()
        changed_context[0, 1] = torch.tensor(
            candidate_context(1, 2, 1, 10, 10)
        )
        changed = model([graph], shapes, changed_context)
        self.assertFalse(torch.allclose(
            output["ranking_logits"][0, 0],
            changed["ranking_logits"][0, 0],
        ))

        losses = model2_loss(
            output, torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 2.0]]), torch.tensor([[False, True]]),
            torch.tensor([1]), config,
        )
        losses["total"].backward()
        for name in (
            "candidate_set_blocks.0.attention.in_proj_weight",
            "candidate_set_blocks.0.feed_forward.0.weight",
            "rank_head.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.any(gradient != 0), name)

    def test_cgra_shortest_paths_match_mesh_distance(self):
        distances = pad_shortest_path_distances(
            [make_cgra_graph(2, 2)], torch.device("cpu")
        )
        self.assertEqual(tuple(distances.shape), (1, 4, 4))
        self.assertEqual(float(distances[0, 0, 0]), 0.0)
        self.assertEqual(float(distances[0, 0, 1]), 1.0)
        self.assertEqual(float(distances[0, 0, 3]), 2.0)

    def test_discrete_ii_head_masks_lower_bound_and_tiebreaks_shape(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="discrete_routing_set",
            candidate_set_layers=1, candidate_set_heads=4,
        )
        model = JointGraphShapeModel(config)
        with torch.no_grad():
            model.success_head.weight.zero_()
            model.success_head.bias.fill_(10.0)
            model.discrete_ii_head.weight.zero_()
            model.discrete_ii_head.bias.zero_()
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 4, 4),
        ]])
        output = model([graph], shapes, context)
        self.assertTrue(torch.equal(
            output["predicted_ii_class"], torch.tensor([[4.0, 4.0]])
        ))
        self.assertTrue(torch.all(
            output["ii_class_logits"][..., :3] < -1e20
        ))
        self.assertLess(
            float(output["selection_cost"][0, 0]),
            float(output["selection_cost"][0, 1]),
        )

        losses = model2_loss(
            output, torch.tensor([[1.0, 1.0]]),
            torch.tensor([[1.0, 1.0]]), torch.tensor([[True, True]]),
            torch.tensor([0]), config,
        )
        self.assertGreater(float(losses["discrete_ii"]), 0.0)
        losses["total"].backward()
        gradient = model.discrete_ii_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.any(gradient != 0))

    def test_discrete_pointwise_prediction_is_candidate_independent(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="discrete_pointwise",
        )
        model = JointGraphShapeModel(config).eval()
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        first_context = candidate_context(1, 1, 1, 4, 4)
        context = torch.tensor([[
            first_context,
            candidate_context(1, 2, 1, 2, 2),
        ]])
        together = model([graph], shapes, context)
        alone = model(
            [graph], shapes[:1], torch.tensor([[first_context]]),
        )
        for name in (
            "predicted_ii_mean", "predicted_ii_mode",
            "predicted_ii_std", "success_probability",
        ):
            self.assertTrue(torch.allclose(
                together[name][:, :1], alone[name], atol=1e-6,
            ), name)
        self.assertFalse(hasattr(model, "candidate_set_blocks"))
        self.assertEqual(tuple(together["ii_class_probabilities"].shape), (1, 2, 20))

        changed_context = context.clone()
        changed_context[0, 1] = torch.tensor(
            candidate_context(1, 2, 1, 10, 10)
        )
        changed = model([graph], shapes, changed_context)
        self.assertTrue(torch.allclose(
            together["predicted_ii_mean"][0, 0],
            changed["predicted_ii_mean"][0, 0], atol=1e-6,
        ))

        losses = model2_loss(
            together, torch.tensor([[1.0, 1.0]]),
            torch.tensor([[1.0, 1.0]]), torch.tensor([[True, True]]),
            torch.tensor([0]), config,
        )
        self.assertEqual(float(losses["listwise"]), 0.0)
        losses["total"].backward()
        gradient = model.discrete_ii_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.any(gradient != 0))

        serialized = config.to_dict()
        self.assertEqual(serialized["interaction_mode"], "discrete_pointwise")
        self.assertNotIn("candidate_set_layers", serialized)

    def test_residual_pointwise_distribution_is_lower_bound_relative(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="residual_pointwise",
        )
        model = JointGraphShapeModel(config).eval()
        with torch.no_grad():
            model.discrete_ii_head.weight.zero_()
            model.discrete_ii_head.bias.zero_()
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        first_context = candidate_context(1, 1, 1, 4, 4)
        contexts = torch.tensor([[
            first_context,
            candidate_context(1, 2, 1, 2, 2),
        ]])
        together = model([graph], shapes, contexts)
        alone = model([graph], shapes[:1], torch.tensor([[first_context]]))
        self.assertTrue(torch.allclose(
            together["predicted_ii_mean"][:, :1],
            alone["predicted_ii_mean"], atol=1e-6,
        ))
        self.assertEqual(tuple(together["ii_class_probabilities"].shape), (1, 2, 21))
        self.assertAlmostEqual(float(together["lower_bound"][0, 0]), 4.0)
        self.assertAlmostEqual(float(together["predicted_ii_mode"][0, 0]), 4.0)
        self.assertGreaterEqual(
            float(together["predicted_ii_mean"][0, 0]), 4.0
        )
        self.assertTrue(torch.all(
            together["ii_class_logits"][0, 0, 17:] < -1e20
        ))

        losses = model2_loss(
            together, torch.tensor([[1.0, 1.0]]),
            torch.tensor([[3.0, 1.0]]), torch.tensor([[True, True]]),
            torch.tensor([0]), config,
        )
        self.assertGreater(float(losses["discrete_ii"]), 0.0)
        self.assertEqual(float(losses["listwise"]), 0.0)
        losses["total"].backward()
        self.assertTrue(torch.any(model.discrete_ii_head.weight.grad != 0))
        serialized = config.to_dict()
        self.assertEqual(serialized["interaction_mode"], "residual_pointwise")
        self.assertNotIn("candidate_set_layers", serialized)

    def test_continuous_residual_pointwise_trains_scalar_and_auxiliary_heads(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="continuous_residual_pointwise",
            discrete_ii_loss_weight=0.25,
            discrete_ii_decision="floor",
        )
        model = JointGraphShapeModel(config).eval()
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        first_context = candidate_context(1, 1, 1, 4, 4)
        contexts = torch.tensor([[
            first_context,
            candidate_context(1, 2, 1, 20, 20),
        ]])
        together = model([graph], shapes, contexts)
        alone = model([graph], shapes[:1], torch.tensor([[first_context]]))
        self.assertTrue(torch.allclose(
            together["predicted_ii"][:, :1],
            alone["predicted_ii"], atol=1e-6,
        ))
        self.assertEqual(
            tuple(together["ii_class_probabilities"].shape), (1, 2, 21)
        )
        self.assertGreaterEqual(float(together["predicted_ii"][0, 0]), 4.0)
        self.assertLessEqual(float(together["predicted_ii"][0, 0]), 20.0)
        self.assertEqual(float(together["predicted_ii"][0, 1]), 20.0)

        losses = model2_loss(
            together, torch.tensor([[1.0, 1.0]]),
            torch.tensor([[3.0, 0.0]]), torch.tensor([[True, True]]),
            torch.tensor([0]), config,
        )
        self.assertGreater(float(losses["residual"]), 0.0)
        self.assertGreater(float(losses["discrete_ii"]), 0.0)
        self.assertEqual(float(losses["listwise"]), 0.0)
        losses["total"].backward()
        self.assertIsNotNone(model.residual_head.weight.grad)
        self.assertTrue(torch.any(model.residual_head.weight.grad != 0))
        self.assertIsNotNone(model.discrete_ii_head.weight.grad)
        self.assertTrue(torch.any(model.discrete_ii_head.weight.grad != 0))
        self.assertFalse(hasattr(model, "candidate_set_blocks"))
        self.assertEqual(
            config.to_dict()["interaction_mode"],
            "continuous_residual_pointwise",
        )

    def test_discrete_mode_requires_integer_ii_ceiling(self):
        with self.assertRaisesRegex(ValueError, "integer mapper_ii_ceiling"):
            Model2Config(
                interaction_mode="discrete_routing_set",
                mapper_ii_ceiling=20.5,
            ).validate()
        with self.assertRaisesRegex(ValueError, "discrete_success_threshold"):
            Model2Config(
                interaction_mode="discrete_routing_set",
                discrete_success_threshold=1.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "discrete_ii_decision"):
            Model2Config(
                interaction_mode="discrete_routing_set",
                discrete_ii_decision="truncate",
            ).validate()
        calibrated = Model2Config(
            interaction_mode="discrete_routing_set",
            discrete_success_threshold=0.9,
            discrete_ii_decision="floor",
        ).to_dict()
        self.assertEqual(calibrated["discrete_success_threshold"], 0.9)
        self.assertEqual(calibrated["discrete_ii_decision"], "floor")
        with self.assertRaisesRegex(ValueError, "cross-attention mode"):
            Model2Config(placement_loss_weight=1.0).validate()

    def test_placement_loss_supervises_operation_to_pe_attention(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="discrete_routing_set",
            candidate_set_layers=1, candidate_set_heads=4,
            placement_loss_weight=0.5,
        )
        model = JointGraphShapeModel(config)
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 2, 2),
        ]])
        output = model([graph], shapes, context)
        placement = torch.tensor([[[0, 0, 0], [0, 1, 1]]])
        losses = model2_loss(
            output, torch.tensor([[1.0, 1.0]]),
            torch.tensor([[1.0, 1.0]]), torch.tensor([[False, True]]),
            torch.tensor([1]), config, placement,
        )
        self.assertGreater(float(losses["placement"]), 0.0)
        losses["total"].backward()
        gradient = model.cross_dfg_query.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.any(gradient != 0))
        self.assertEqual(config.to_dict()["placement_loss_weight"], 0.5)

    def test_strict_set_classifier_optimizes_only_oracle_shape(self):
        graph = parse_neura_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %b = "neura.constant"() : () -> !neura.data<i32, i1>
        %c = "neura.add"(%a, %b) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        """)
        config = Model2Config(
            hidden_dimension=16, message_passing_layers=1, dropout=0.0,
            interaction_mode="strict_set_classifier",
            candidate_set_layers=1, candidate_set_heads=4,
        )
        model = JointGraphShapeModel(config)
        shapes = [make_cgra_graph(1, 1), make_cgra_graph(1, 2)]
        context = torch.tensor([[
            candidate_context(1, 1, 1, 4, 4),
            candidate_context(1, 2, 1, 2, 2),
        ]])
        output = model([graph], shapes, context)
        self.assertEqual(tuple(output["ranking_logits"].shape), (1, 2))
        self.assertTrue(torch.equal(
            output["selection_cost"], -output["ranking_logits"]
        ))
        losses = model2_loss(
            output, torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 2.0]]), torch.tensor([[False, True]]),
            torch.tensor([1]), config,
        )
        self.assertTrue(torch.equal(
            losses["total"], losses["listwise_strict_tiebreak"]
        ))
        self.assertEqual(float(losses["listwise_optimal_ii"]), 0.0)
        losses["total"].backward()
        parameters = dict(model.named_parameters())
        for name in (
            "candidate_set_blocks.0.attention.in_proj_weight",
            "candidate_set_blocks.0.feed_forward.0.weight",
            "rank_head.weight",
        ):
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.any(gradient != 0), name)
        for name in ("success_head.weight", "residual_head.weight"):
            gradient = parameters[name].grad
            self.assertTrue(
                gradient is None or not torch.any(gradient != 0), name
            )
        serialized = config.to_dict()
        self.assertEqual(
            serialized["interaction_mode"], "strict_set_classifier"
        )
        self.assertNotIn("discrete_ii_loss_weight", serialized)

    def test_split_keeps_queries_intact_and_balances_families(self):
        graph = parse_neura_dfg(
            '%a = "neura.constant"() : () -> !neura.data<i32, i1>'
        )
        queries = []
        for family in ("f0", "f1"):
            for index in range(20):
                candidate = CandidateRecord(
                    candidate_id=f"{family}/{index}/1x1",
                    ranking_query_id=f"{family}/{index}", rows=1, columns=1,
                    rec_mii=1, res_mii=1, lower_bound=1,
                    status="success", compiled_ii=1,
                )
                queries.append(QueryRecord(
                    ranking_query_id=f"{family}/{index}",
                    generator_family=family, graph=graph,
                    candidates=(candidate,),
                ))
        split = split_queries(queries)
        identities = [
            {query.ranking_query_id for query in split[name]}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(identities[0] & identities[1])
        self.assertFalse(identities[0] & identities[2])
        self.assertFalse(identities[1] & identities[2])
        self.assertEqual(
            [len({q.ranking_query_id for q in split[name]})
             for name in ("train", "validation", "test")],
            [28, 6, 6],
        )

    def test_additional_queries_are_training_only_without_moving_holdouts(self):
        graph = parse_neura_dfg(
            '%a = "neura.constant"() : () -> !neura.data<i32, i1>'
        )

        def query(identity, compiled_ii=1):
            candidate = CandidateRecord(
                candidate_id=f"{identity}/1x1",
                ranking_query_id=identity, rows=1, columns=1,
                rec_mii=1, res_mii=1, lower_bound=1,
                status="success", compiled_ii=compiled_ii,
            )
            return QueryRecord(
                ranking_query_id=identity,
                generator_family="family", graph=graph,
                candidates=(candidate,),
            )

        base = [query(f"q{index}") for index in range(3)]
        split = {
            "train": [base[0]],
            "validation": [base[1]],
            "test": [base[2]],
        }
        extended, added = add_training_only_queries(
            split, base, [base[1], query("q3")],
        )
        self.assertEqual(added, ["q3"])
        self.assertEqual(
            [q.ranking_query_id for q in extended["validation"]], ["q1"]
        )
        self.assertEqual(
            [q.ranking_query_id for q in extended["test"]], ["q2"]
        )
        self.assertEqual(
            [q.ranking_query_id for q in extended["train"]], ["q0", "q3"]
        )
        with self.assertRaisesRegex(ValueError, "changes an existing query"):
            add_training_only_queries(split, base, [query("q1", 2)])

        with tempfile.TemporaryDirectory() as directory:
            supervision_path = Path(directory) / "placement.json"
            supervision_path.write_text(json.dumps({
                "schema_version": "cgra-ii-placement-supervision-v1",
                "placements": {"q0/1x1": [0]},
            }))
            attached, metadata = attach_placement_supervision(
                [base[0]], supervision_path,
            )
            self.assertEqual(metadata["attached_training_candidate_count"], 1)
            self.assertEqual(attached[0].candidates[0].placement, (0,))
            targets = batch_placement_targets(
                attached, 1, torch.device("cpu"),
            )
            self.assertTrue(torch.equal(targets, torch.tensor([[[0]]])))

    def test_training_subsamples_are_deterministic_stratified_and_nested(self):
        graph = parse_neura_dfg(
            '%a = "neura.constant"() : () -> !neura.data<i32, i1>'
        )
        queries = []
        for family in ("f0", "f1"):
            for index in range(12):
                identity = f"{family}/{index}"
                candidate = CandidateRecord(
                    candidate_id=f"{identity}/1x1",
                    ranking_query_id=identity, rows=1, columns=1,
                    rec_mii=1, res_mii=1, lower_bound=1,
                    status="success", compiled_ii=1,
                )
                queries.append(QueryRecord(
                    ranking_query_id=identity,
                    generator_family=family, graph=graph,
                    candidates=(candidate,),
                ))
        quarter = subsample_training_queries(queries, 0.25, 17)
        half = subsample_training_queries(queries, 0.5, 17)
        quarter_ids = {query.ranking_query_id for query in quarter}
        half_ids = {query.ranking_query_id for query in half}
        self.assertEqual(len(quarter), 6)
        self.assertEqual(len(half), 12)
        self.assertLessEqual(quarter_ids, half_ids)
        self.assertEqual(
            half_ids,
            {query.ranking_query_id
             for query in subsample_training_queries(queries, 0.5, 17)},
        )
        self.assertEqual(
            {query.generator_family for query in quarter}, {"f0", "f1"}
        )
        self.assertEqual(subsample_training_queries(queries, 1.0, 17), queries)
        for fraction in (0.0, 1.1):
            with self.assertRaisesRegex(ValueError, "training fraction"):
                subsample_training_queries(queries, fraction, 17)

    def test_route_expanded_placement_targets_align_to_materialized_nodes(self):
        graph = parse_neura_route_expanded_dfg("""
        %a = "neura.constant"() : () -> !neura.data<i32, i1>
        %m = "neura.data_mov"(%a) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %b = "neura.add"(%m, %a) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        func.return
        """)
        candidate = CandidateRecord(
            candidate_id="q/1x2", ranking_query_id="q", rows=1, columns=2,
            rec_mii=1, res_mii=2, lower_bound=2,
            status="success", compiled_ii=3,
        )
        query = QueryRecord(
            ranking_query_id="q", generator_family="family", graph=graph,
            candidates=(candidate,),
        )
        with tempfile.TemporaryDirectory() as directory:
            supervision_path = Path(directory) / "placement.json"
            supervision_path.write_text(json.dumps({
                "schema_version": "cgra-ii-placement-supervision-v1",
                "placements": {"q/1x2": [0, 1]},
            }))
            attached, metadata = attach_placement_supervision(
                [query], supervision_path,
            )
        self.assertEqual(attached[0].candidates[0].placement, (0, -1, 1, -1))
        self.assertEqual(metadata["attached_supervised_node_count"], 2)
        targets = batch_placement_targets(
            attached, 4, torch.device("cpu"),
        )
        self.assertTrue(torch.equal(
            targets, torch.tensor([[[0, -1, 1, -1]]]),
        ))


if __name__ == "__main__":
    unittest.main()
