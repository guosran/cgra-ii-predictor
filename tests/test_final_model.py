import hashlib
import json
from pathlib import Path
import re
import zipfile

import torch

from adapters.amoeba_cost_catalog import load_mapper_model
from cgra_ii_predictor.dfg import parse_neura_route_expanded_dfg
from cgra_ii_predictor.mapper_model import (
    mapper_feature_vector,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final"
MODEL_SHA256 = "afc744ee15de4fc1e6bb6574dc755edc36f53b33662b6c01eefeb52ba16993fc"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        metadata = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        assert archive.testzip() is None
        return archive.read(metadata)


def test_final_model_is_bound_to_real_mapper_labels():
    metadata = json.loads((MODEL_DIR / "model.json").read_text())
    assert metadata["target"] == "compiled_ii_from_neura_heuristic_mapper"
    assert metadata["checkpoint"]["sha256"] == MODEL_SHA256
    assert metadata["checkpoint"]["parameter_count"] == 9_345
    assert metadata["checkpoint"]["feature_count"] == 112
    assert metadata["training"]["placement_supervision"] is False
    assert metadata["training"]["mapped_artifact_features"] is False
    assert sha256(MODEL_DIR / "mapper.pt") == MODEL_SHA256
    assert {path.name for path in MODEL_DIR.iterdir()} == {
        "mapper.pt", "model.json",
    }


def test_final_model_strict_load_and_all_shapes():
    artifact = torch.load(
        MODEL_DIR / "mapper.pt", map_location="cpu", weights_only=False,
    )
    assert artifact["schema"] == "cgra-ii-direct-mapper-model"
    assert len(artifact["feature_names"]) == 112
    model, config, loader_metadata = load_mapper_model(
        MODEL_DIR / "mapper.pt", torch.device("cpu"),
    )
    assert list(config.enabled_feature_names) == artifact["feature_names"]
    shapes = [tuple(shape) for shape in loader_metadata["supported_mapper_shapes"]]
    assert len(shapes) == 8
    assert all(rows * columns <= 64 for rows, columns in shapes)
    graph = parse_neura_route_expanded_dfg('''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %2 = "neura.add"(%1, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    ''')
    features = torch.tensor([
        mapper_feature_vector(graph, rows, columns, 1, 1, 1)
        for rows, columns in shapes
    ])
    lower_bound = torch.ones(len(shapes))
    with torch.inference_mode():
        prediction = model(features, lower_bound)
    assert prediction.shape == (len(shapes),)
    assert torch.isfinite(prediction).all()
    assert torch.all(prediction >= lower_bound)
    assert torch.all(prediction <= config.mapper_ii_ceiling)


def test_no_internal_iteration_tokens():
    pattern = re.compile(
        r"(?i)(?<![a-z0-9])v[1-9][0-9]*(?![a-z0-9])|model[1-9]"
    )
    roots = [ROOT / "src", MODEL_DIR]
    roots += [ROOT / name for name in ("README.md", "HANDOFF.md")]
    hits = []
    for root in roots:
        paths = [root] if root.is_file() else [
            path for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        for path in paths:
            if path.suffix == ".pt":
                content = re.sub(
                    rb"_rebuild_tensor_v[0-9]+", b"", checkpoint_metadata(path)
                ).decode("latin1")
            else:
                content = path.read_text(errors="replace")
            if pattern.search(path.name) or pattern.search(content):
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []
