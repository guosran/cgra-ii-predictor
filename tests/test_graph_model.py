import importlib.util
import argparse
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
        global JointGraphShapeModel, Model2Config, candidate_context
        global censored_top1_metrics, make_cgra_graph, model2_loss
        global parse_neura_dfg
        global frozen_adapter, neura_motifs_v7
        import torch
        from adapters import neura_graph_frozen as frozen_adapter
        from adapters import neura_motifs_v7
        from adapters.neura_graph_experiment import (
            CandidateRecord, QueryRecord, resolve_device, split_queries,
        )
        from cgra_ii_predictor.graph_model import (
            JointGraphShapeModel, Model2Config, candidate_context,
            censored_top1_metrics, make_cgra_graph, model2_loss,
            parse_neura_dfg,
        )

    def test_device_selection_never_silently_ignores_explicit_cuda(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with self.assertRaisesRegex(ValueError, "auto, cpu, or cuda"):
            resolve_device("tpu")
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")

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


if __name__ == "__main__":
    unittest.main()
