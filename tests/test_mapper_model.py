import math

import pytest
import torch

from cgra_ii_predictor.dfg import (
    ROUTE_EXPANDED_OPERATION_TYPES,
    parse_neura_route_expanded_dfg,
)
from cgra_ii_predictor.mapper_model import (
    CategoricalMapperIIModel,
    DirectMapperIIEnsemble,
    DirectMapperIIModel,
    MAPPER_ALIGNED_FEATURE_GROUPS,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_vector,
    mapper_ii_loss,
)


DFG = '''
%0 = "neura.constant"() : () -> !neura.data<i32, i1>
%1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
%2 = "neura.add"(%1, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
'''


def test_mapper_features_use_only_pre_mapper_query_facts():
    graph = parse_neura_route_expanded_dfg(DFG)
    values = mapper_feature_vector(graph, 4, 8, 2, 3, 3)
    assert len(values) == len(MAPPER_FEATURE_NAMES) == 156
    assert all(math.isfinite(value) for value in values)
    by_name = dict(zip(MAPPER_FEATURE_NAMES, values))
    assert by_name["normalized_rec_mii"] == pytest.approx(0.1)
    assert by_name["normalized_res_mii"] == pytest.approx(0.15)
    assert by_name["normalized_lower_bound"] == pytest.approx(0.15)
    assert by_name["shape_4x8"] == 1.0
    assert sum(by_name[name] for name in by_name if name.startswith("shape_")) == 1


def test_direct_model_has_one_bounded_mapper_ii_output():
    model = DirectMapperIIModel(MapperModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_161
    features = torch.zeros((2, len(MAPPER_FEATURE_NAMES)))
    lower_bound = torch.tensor([1.0, 20.0])
    prediction = model(features, lower_bound)
    assert prediction.shape == (2,)
    assert torch.all(prediction >= lower_bound)
    assert torch.all(prediction <= 20.0)


def test_direct_ensemble_is_the_arithmetic_mean_of_members():
    ensemble = DirectMapperIIEnsemble(3, MapperModelConfig())
    features = torch.zeros((2, len(MAPPER_FEATURE_NAMES)))
    lower_bound = torch.tensor([1.0, 4.0])
    with torch.inference_mode():
        expected = torch.stack([
            member(features, lower_bound) for member in ensemble.members
        ]).mean(0)
        actual = ensemble(features, lower_bound)
    assert torch.equal(actual, expected)
    assert torch.all(actual >= lower_bound)


def test_categorical_model_has_legal_integer_label_and_expected_readout():
    model = CategoricalMapperIIModel(MapperModelConfig())
    features = torch.zeros((2, len(MAPPER_FEATURE_NAMES)))
    lower_bound = torch.tensor([1.0, 20.0])
    logits = model.class_logits(features, lower_bound)
    expected = model(features, lower_bound)
    labels = model.predict_label(features, lower_bound)

    assert logits.shape == (2, 21)
    assert torch.isneginf(logits[1, 1:]).all()
    assert expected.shape == labels.shape == (2,)
    assert labels.dtype == torch.int64
    assert torch.all(expected >= lower_bound)
    assert torch.all(expected <= 20.0)
    assert torch.all(labels >= lower_bound)
    assert torch.all(labels <= 20)
    assert labels[1] == 20


def test_categorical_model_rejects_fractional_lower_bound():
    model = CategoricalMapperIIModel(MapperModelConfig())
    with pytest.raises(ValueError, match="must be integers"):
        model(
            torch.zeros((1, len(MAPPER_FEATURE_NAMES))),
            torch.tensor([1.5]),
        )


def test_feature_ablation_reduces_first_layer_and_keeps_raw_contract():
    removed = set(MAPPER_ALIGNED_FEATURE_GROUPS["alap_bucket"])
    enabled = tuple(name for name in MAPPER_FEATURE_NAMES if name not in removed)
    model = DirectMapperIIModel(
        MapperModelConfig(enabled_feature_names=enabled)
    )
    assert model.regressor[0].in_features == len(MAPPER_FEATURE_NAMES) - len(removed)
    prediction = model(
        torch.zeros((2, len(MAPPER_FEATURE_NAMES))), torch.ones(2),
    )
    assert prediction.shape == (2,)


def test_mapper_loss_rewards_correct_shape_order():
    target = torch.tensor([2.0, 5.0])
    better = torch.tensor([0])
    worse = torch.tensor([1])
    correct = mapper_ii_loss(
        torch.tensor([2.0, 5.0]), target, better, worse,
    )
    reversed_order = mapper_ii_loss(
        torch.tensor([5.0, 2.0]), target, better, worse,
    )
    assert correct["pairwise"].item() == 0.0
    assert reversed_order["pairwise"] > 0.0
    assert reversed_order["total"] > correct["total"]


def test_route_expanded_parser_keeps_mapper_visible_effect_operations():
    graph = parse_neura_route_expanded_dfg('''
      func.func @task() {
        neura.kernel inputs() attributes {dataflow_mode = "predicate"} {
          %0 = neura.counter attributes {counter_id = 0 : i32}
            -> !neura.data<index, i1>
          %1 = "neura.data_mov"(%0)
            : (!neura.data<index, i1>) -> !neura.data<index, i1>
          neura.store_indexed %1 to [%0 : !neura.data<index, i1>]
            {rhs_value = "%output"} : !neura.data<index, i1>
          neura.yield {yield_type = "void"}
        }
        return
      }
    ''')
    names = [ROUTE_EXPANDED_OPERATION_TYPES[index] for index in graph.node_types]
    assert names == ["counter", "data_mov", "store_indexed", "yield"]
    assert len(graph.edges) == 3
    assert sum(row[11] > 0.5 for row in graph.node_features) == 2
