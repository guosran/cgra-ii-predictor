import hashlib
import json
from pathlib import Path

import pytest
import torch

from adapters.amoeba_cost_catalog import (
    ENSEMBLE_CHECKPOINT_SCHEMA, load_mapper_model, validate_model_architecture,
)
from cgra_ii_predictor.dfg import parse_neura_route_expanded_dfg
from cgra_ii_predictor.mapper_model import (
    CategoricalMapperIIModel, DirectMapperIIEnsemble, DirectMapperIIModel,
    MAPPER_FEATURE_NAMES, MapperModelConfig, RidgeMapperIIModel,
    mapper_feature_names, mapper_feature_vector,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID


def test_two_tile_protocol_has_its_own_feature_contract():
    graph = parse_neura_route_expanded_dfg('''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %2 = "neura.add"(%1, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    ''')
    names = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
    assert len(names) == 148
    assert mapper_feature_names() == MAPPER_FEATURE_NAMES
    for rows, cols in SHAPE_PROTOCOL_2X2.mapper_shapes:
        values = mapper_feature_vector(graph, rows, cols, 1, 1, 1,
                                      shape_protocol=SHAPE_PROTOCOL_2X2_ID)
        facts = dict(zip(names, values))
        assert len(values) == len(names)
        assert facts[f"shape_{rows}x{cols}"] == 1
        assert sum(v for k, v in facts.items() if k.startswith("shape_")) == 1
        assert facts["normalized_tiles"] == rows * cols / 16
        assert facts["normalized_bisection_links"] <= 1


@pytest.mark.parametrize("model_class", [DirectMapperIIModel, RidgeMapperIIModel,
                                         CategoricalMapperIIModel])
def test_models_accept_only_the_declared_raw_width(model_class):
    config = MapperModelConfig(shape_protocol=SHAPE_PROTOCOL_2X2_ID)
    model = model_class(config)
    lower = torch.tensor([1., 20.])
    predicted = model(torch.zeros(2, 148), lower)
    assert torch.all(predicted >= lower)
    assert torch.all(predicted <= 20)
    with pytest.raises(ValueError, match="wrong shape"):
        model(torch.zeros(2, 156), lower)


def test_two_tile_ensemble_checkpoint_roundtrip_and_protocol_binding(tmp_path):
    selected = tuple(MAPPER_FEATURE_NAMES[i] for i in
                     (*range(65, 79), *range(93, 118), *range(134, 156)))
    config = MapperModelConfig(shape_protocol=SHAPE_PROTOCOL_2X2_ID,
                               enabled_feature_names=selected)
    model = DirectMapperIIEnsemble(4, config).eval()
    artifact = dict(schema=ENSEMBLE_CHECKPOINT_SCHEMA, ensemble_member_count=4,
                    ensemble_reduction="arithmetic_mean", config=config.to_dict(),
                    feature_names=list(mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)),
                    state_dict=model.state_dict(), architecture_sha256="a" * 64,
                    training_manifest_sha256="b" * 64,
                    supported_mapper_shapes=[list(s) for s in SHAPE_PROTOCOL_2X2.mapper_shapes])
    path = tmp_path / "mapper.pt"
    torch.save(artifact, path)
    loaded, loaded_config, metadata = load_mapper_model(path, torch.device("cpu"))
    assert loaded_config.to_dict() == config.to_dict()
    assert metadata["training_architecture_sha256"] == "a" * 64
    features, lower = torch.randn(3, 148), torch.ones(3)
    with torch.inference_mode():
        assert torch.equal(model(features, lower), loaded(features, lower))
    artifact["feature_names"] = list(MAPPER_FEATURE_NAMES)
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="feature contract"):
        load_mapper_model(path, torch.device("cpu"))


def test_packaged_candidate_matches_training_evidence_and_hardware():
    package = Path(__file__).resolve().parents[1] / "models/candidates/per-cgra-2x2"
    metadata = json.loads((package / "model.json").read_text())
    report = json.loads((package / "training-report.json").read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(package / "mapper.pt") == metadata["checkpoint"]["sha256"]
    assert digest(package / "training-report.json") == metadata["training"]["report_sha256"]
    assert digest(package / "source-groups.json") == report["provenance"]["source_groups_sha256"]
    assert digest(package / "training-exclusions.json") == report["provenance"]["training_exclusions_sha256"]
    assert digest(package / "architecture.yaml") == report["architecture_sha256"]
    assert metadata["verification"]["production_replay_checkpoint_byte_identical"]
    assert metadata["verification"]["production_replay_report_byte_identical"]
    model, config, contract = load_mapper_model(package / "mapper.pt", torch.device("cpu"))
    assert config.shape_protocol == SHAPE_PROTOCOL_2X2_ID
    assert len(config.enabled_feature_names) == 61
    assert len(model.members) == 4
    assert contract["supported_mapper_shapes"] == [list(s) for s in SHAPE_PROTOCOL_2X2.mapper_shapes]
    validate_model_architecture(contract, report["architecture_sha256"])
    with pytest.raises(ValueError, match="architecture"):
        validate_model_architecture(contract, "0" * 64)
    assert report["training"]["native_status_counts"] == {
        "success": 2569, "censored": 15, "pending": 0,
    }
    assert metadata["promotion_status"] == "candidate_pending_amoeba_benchmark_overlap_audit"
