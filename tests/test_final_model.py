import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import torch
import zipfile

from cgra_ii_predictor.graph_model import (
    JointGraphShapeModel,
    PointwiseConfig,
    candidate_context,
    make_cgra_graph,
    parse_neura_dfg_representation,
)
from cgra_ii_predictor.shape_protocol import (
    SHAPE_PROTOCOL,
    SHAPE_PROTOCOL_ID,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "final"
CHECKPOINTS = {
    "large-operation.pt": "eda6f5e5b2b9410859efc887c533350f96cd40e260a69fd70ffcc97d6d782fe2",
    "baseline.pt": "6e0bdb7f9ebd44d2efa13387821460609eb3b802a39edd96d876e91701856e06",
    "ranking.pt": "1d9af87fa7274303d6a5e50ae0cd05fbfb081f5d670a722c39d34863f6ce7ac0",
}
EXPECTED_SMOKE_II = {
    "large-operation.pt": 1.4982157945632935,
    "baseline.pt": 2.3798537254333496,
    "ranking.pt": 2.031731367111206,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = next(name for name in names if name.endswith("/data.pkl"))
        assert archive.testzip() is None
        return archive.read(metadata)


def load_metadata(path: Path) -> dict:
    """Read checkpoint metadata without importing the PyTorch runtime."""
    def rebuild(*arguments):
        return ("tensor", arguments[2] if len(arguments) > 2 else None)

    class MetadataUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch"):
                if name.startswith("_rebuild"):
                    return rebuild
                return type(name, (), {})
            return super().find_class(module, name)

        def persistent_load(self, identity):
            return ("storage", identity)

    return MetadataUnpickler(io.BytesIO(checkpoint_metadata(path))).load()


def test_final_checkpoint_hashes_and_metadata():
    for name, expected_hash in CHECKPOINTS.items():
        path = MODEL_DIR / name
        assert sha256(path) == expected_hash
        metadata = checkpoint_metadata(path)
        assert b"cgra-ii-pointwise-model" in metadata
        assert b"route_expanded" in metadata
        assert SHAPE_PROTOCOL_ID.encode() in metadata
        application_metadata = re.sub(
            rb"_rebuild_tensor_v[0-9]+", b"", metadata
        )
        assert re.search(
            rb"(?<![A-Za-z0-9])v[1-9][0-9]*(?![A-Za-z0-9])",
            application_metadata,
        ) is None
        artifact = load_metadata(path)
        assert artifact["schema_version"] == "cgra-ii-pointwise-model"
        assert artifact["config"]["dfg_representation"] == "route_expanded"
        assert len(artifact["state_dict"]) >= 120


def test_ensemble_is_bound_to_final_checkpoints():
    ensemble = json.loads((MODEL_DIR / "ensemble.json").read_text())
    reported = {
        Path(record["path"]).name: record["sha256"]
        for record in ensemble["checkpoints"].values()
    }
    assert reported == CHECKPOINTS
    assert abs(sum(ensemble["weights"].values()) - 1.0) < 1e-6
    assert ensemble["selection_split"] == "validation_only"


def test_shape_protocol_and_route_expanded_features():
    assert SHAPE_PROTOCOL.mapper_for_physical(2, 2) == (8, 8)
    graph = parse_neura_dfg_representation(
        '''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %2 = "neura.add"(%1, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        ''',
        "route_expanded",
    )
    assert len(graph.node_types) == 3
    cgra = make_cgra_graph(8, 8)
    assert len(cgra.node_types) == 64
    context = candidate_context(8, 8, 2, 3, 3)
    assert len(context) == 10
    config = PointwiseConfig()
    assert config.dfg_representation == "route_expanded"
    assert config.dfg_message_mode == "dual_mean"
    assert config.interaction_mode == "residual_pointwise"


def test_pytorch_strict_load_and_forward():
    dfg = parse_neura_dfg_representation(
        '''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %2 = "neura.add"(%1, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        '''
    )
    cgra = make_cgra_graph(4, 4)
    for name in CHECKPOINTS:
        artifact = torch.load(
            MODEL_DIR / name, map_location="cpu", weights_only=False
        )
        config = PointwiseConfig(**artifact["config"]).validate()
        model = JointGraphShapeModel(config)
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        context = torch.tensor([[
            candidate_context(
                4, 4, 1, 1, 1,
                config.mapper_ii_ceiling, config.shape_protocol,
            )
        ]])
        with torch.inference_mode():
            output = model([dfg], [cgra], context)
        assert torch.isfinite(output["predicted_ii"]).all()
        assert torch.isfinite(output["predicted_ii_std"]).all()
        assert output["predicted_ii"].item() >= 1.0
        assert math.isclose(
            output["predicted_ii"].item(), EXPECTED_SMOKE_II[name],
            abs_tol=1e-6,
        )


def test_no_internal_iteration_tokens():
    pattern = re.compile(r"(?i)(?<![a-z0-9])v[1-9][0-9]*(?![a-z0-9])|model[1-9]")
    roots = [ROOT / "src", ROOT / "models" / "final"]
    roots += [ROOT / name for name in ("README.md", "HANDOFF.md", "pyproject.toml")]
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
