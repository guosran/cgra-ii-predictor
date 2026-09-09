import math

import pytest
import torch

from cgra_ii_predictor.dfg import parse_neura_route_expanded_dfg
from cgra_ii_predictor.mapper_model import (
    DirectMapperIIModel,
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
    assert len(values) == len(MAPPER_FEATURE_NAMES) == 112
    assert all(math.isfinite(value) for value in values)
    by_name = dict(zip(MAPPER_FEATURE_NAMES, values))
    assert by_name["normalized_rec_mii"] == pytest.approx(0.1)
    assert by_name["normalized_res_mii"] == pytest.approx(0.15)
    assert by_name["normalized_lower_bound"] == pytest.approx(0.15)
    assert by_name["shape_4x8"] == 1.0
    assert sum(by_name[name] for name in by_name if name.startswith("shape_")) == 1


def test_direct_model_has_one_bounded_mapper_ii_output():
    model = DirectMapperIIModel(MapperModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 9_345
    features = torch.zeros((2, len(MAPPER_FEATURE_NAMES)))
    lower_bound = torch.tensor([1.0, 20.0])
    prediction = model(features, lower_bound)
    assert prediction.shape == (2,)
    assert torch.all(prediction >= lower_bound)
    assert torch.all(prediction <= 20.0)


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
