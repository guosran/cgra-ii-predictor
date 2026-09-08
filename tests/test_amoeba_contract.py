import json
from pathlib import Path

import pytest
import torch

from adapters.amoeba_cost_catalog import (
    _packable_shape_index_tuples,
    _static_shape_alphabet,
    load_candidate_manifest,
    load_analytical_input,
    load_ensemble_report,
    source_task_body_sha256,
    validate_ensemble_architecture,
)
from adapters.amoeba_frozen_pipeline import load_cost_catalog, score_candidates
from adapters.extract_amoeba_task_dfgs import extract_task_dfg_texts
from adapters.amoeba_protocol import (
    CANDIDATE_SCHEMA,
    COST_SCHEMA,
    SEARCH_SCOPE,
    SHAPE_POLICY,
    SCORE_MODEL,
    SPATIAL_CAPACITY_POLICY,
)
from cgra_ii_predictor.graph_model import (
    CGRA_NODE_FEATURE_NAMES,
    CANDIDATE_CONTEXT_NAMES,
    JointGraphShapeModel,
    PointwiseConfig,
    candidate_context,
    make_cgra_graph,
    parse_neura_dfg_representation,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ARCHITECTURE = (
    "f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6"
)
CURRENT_AMOEBA_ARCHITECTURE = (
    "5c228166de4ceacf49b0ea6286a4a8c1ce45a718ef3da8899a21cea07aa7a174"
)


def _shape(rows, cols):
    return {
        "kind": "rect",
        "rows": rows,
        "cols": cols,
        "cgra_count": rows * cols,
        "cgra_shape": f"{rows}x{cols}",
        "mapper_tile_rows": 4 * rows,
        "mapper_tile_cols": 4 * cols,
    }


def _two_task_manifest():
    shapes = (_shape(1, 1), _shape(1, 2), _shape(2, 1))
    candidates = []
    for first in shapes:
        for second in shapes:
            # A full row and a full column must intersect on a 2x2 grid.
            if {first["cgra_shape"], second["cgra_shape"]} == {"1x2", "2x1"}:
                continue
            candidates.append({
                "candidate_id": f"candidate-{len(candidates)}",
                "task_shapes": [
                    {"task": "A", "trip_count": 10, "shape": first},
                    {"task": "B", "trip_count": 10, "shape": second},
                ],
            })
    return {
        "header": {"function": "main"},
        "candidate_count": 7,
        "candidates": candidates,
        "queries": [
            (task, rows, cols)
            for task in ("A", "B")
            for rows, cols in ((4, 4), (4, 8), (8, 4))
        ],
    }


def _write_two_task_manifest(path):
    manifest = _two_task_manifest()
    header = {
        "record_type": "header",
        "schema": CANDIDATE_SCHEMA,
        "search_scope": SEARCH_SCOPE,
        "shape_policy": SHAPE_POLICY,
        "spatial_capacity_policy": SPATIAL_CAPACITY_POLICY,
        "function": "main",
        "architecture": {
            "grid_rows": 2,
            "grid_cols": 2,
            "per_cgra_tile_rows": 4,
            "per_cgra_tile_cols": 4,
            "spec_sha256": TRAINING_ARCHITECTURE,
        },
        "max_cgras_per_task": 2,
        "tasks": [
            {"task": task, "body_sha256": "0" * 64, "trip_count": 10}
            for task in ("A", "B")
        ],
        "cost_queries": [
            {
                "task": task,
                "mapper_tile_rows": rows,
                "mapper_tile_cols": cols,
            }
            for task, rows, cols in manifest["queries"]
        ],
    }
    records = [header]
    records.extend({
        "record_type": "candidate",
        "schema": CANDIDATE_SCHEMA,
        **candidate,
    } for candidate in manifest["candidates"])
    records.append({
        "record_type": "footer",
        "schema": CANDIDATE_SCHEMA,
        "candidate_count": 7,
    })
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _catalog(success_probability):
    return {
        "namespace": "test-model",
        "by_query": {
            (task, rows, cols): {
                "support_status": "supported",
                "predicted_ii": float(1 if cols == 8 else 2),
                "startup_cycles": 1.0,
                "mapper_success_probability": success_probability,
            }
            for task in ("A", "B")
            for rows, cols in ((4, 4), (4, 8), (8, 4))
        },
    }


def test_rec_mii_zero_is_a_valid_analytical_bound(tmp_path):
    path = tmp_path / "analytical.json"
    path.write_text(json.dumps({
        "schema": "cgra-ii-amoeba-query-features",
        "function": "main",
        "provenance": {},
        "entries": [{
            "task": "A",
            "mapper_tile_rows": 4,
            "mapper_tile_cols": 4,
            "rec_mii": 0,
            "res_mii": 1,
            "lower_bound": 1,
            "startup_cycles": 2,
        }],
    }))
    entries, _ = load_analytical_input(path, "main")
    assert entries[("A", 4, 4)]["rec_mii"] == 0.0


def test_negative_rec_mii_is_rejected(tmp_path):
    path = tmp_path / "analytical.json"
    path.write_text(json.dumps({
        "schema": "cgra-ii-amoeba-query-features",
        "function": "main",
        "provenance": {},
        "entries": [{
            "task": "A",
            "mapper_tile_rows": 4,
            "mapper_tile_cols": 4,
            "rec_mii": -1,
            "res_mii": 1,
            "lower_bound": 1,
            "startup_cycles": 2,
        }],
    }))
    with pytest.raises(ValueError, match="non-negative"):
        load_analytical_input(path, "main")


def test_manifest_loader_accepts_complete_exact_packing_subset(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    manifest = load_candidate_manifest(path)
    assert manifest["candidate_count"] == 7
    assert manifest["candidates"][-1]["candidate_id"] == "candidate-6"


def test_four_by_four_exact_packing_removes_crossing_strips():
    shapes = _static_shape_alphabet(4, 4, 4)
    tuples = list(_packable_shape_index_tuples(2, shapes, 4, 4))
    horizontal = shapes.index((1, 4))
    vertical = shapes.index((4, 1))

    assert len(shapes) == 8
    assert len(tuples) == 62
    assert (horizontal, vertical) not in tuples
    assert (vertical, horizontal) not in tuples
    assert (horizontal, horizontal) in tuples
    assert (vertical, vertical) in tuples


def test_manifest_loader_rejects_fixed_orientation_overlap(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["task_shapes"][0]["shape"] = _shape(1, 2)
    records[1]["task_shapes"][1]["shape"] = _shape(2, 1)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="unpackable"):
        load_candidate_manifest(path)


def test_manifest_loader_rejects_unused_cost_query(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["cost_queries"].append({
        "task": "A", "mapper_tile_rows": 12, "mapper_tile_cols": 12,
    })
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="cost_queries"):
        load_candidate_manifest(path)


def test_cost_catalog_is_bound_to_exact_manifest_bytes(tmp_path):
    manifest_path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(manifest_path)
    manifest = load_candidate_manifest(manifest_path)
    cost_path = tmp_path / "costs.json"
    cost_path.write_text(json.dumps({
        "schema": COST_SCHEMA,
        "function": "main",
        "namespace": "test",
        "predictor_metadata": {
            "candidate_manifest_sha256": "f" * 64,
            "ranking_policy": {
                "mapper_success_probability": "diagnostic_only",
                "uses_mapper_success_probability": False,
            },
        },
        "entries": [],
    }))
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_cost_catalog(cost_path, manifest)


def test_scoring_uses_the_complete_exact_packing_subset():
    manifest = _two_task_manifest()
    scores = score_candidates(manifest, _catalog(0.01), top_k=0)

    assert scores["footer"]["candidate_count"] == 7
    assert scores["footer"]["valid_count"] == 7
    assert scores["scores"][6]["candidate_id"] == "candidate-6"
    assert scores["scores"][6]["valid"] is True
    assert scores["header"]["candidate_space"]["resource_semantics"] == (
        "all_task_rectangles_co_resident"
    )
    assert scores["header"]["score_model"] == SCORE_MODEL


def test_source_task_body_hash_is_read_from_dfg_metadata():
    body_sha = "a" * 64
    dfg = f'''module {{
      func.func @A_dfg() attributes {{
        amoeba.source_task_body_sha256 = "{body_sha}"
      }} {{ return }}
    }}'''
    assert source_task_body_sha256(dfg, "A") == body_sha


def test_source_task_body_hash_is_required():
    with pytest.raises(ValueError, match="must contain exactly one"):
        source_task_body_sha256("module {}", "A")


def test_task_dfg_extractor_preserves_amoeba_body_binding():
    body_sha = "a" * 64
    source = f"""
module {{
  taskflow.task @A {{
    amoeba.source_task_body_sha256 = "{body_sha}"
  }} {{
    neura.kernel inputs(%arg0 : memref<4xf32>) {{
      neura.yield
    }}
  }}
}}
"""
    extracted = extract_task_dfg_texts(source)["A"]
    assert source_task_body_sha256(extracted, "A") == body_sha


def test_task_dfg_extractor_rejects_unbound_task():
    source = """
module {
  taskflow.task @A {
    neura.kernel inputs(%arg0 : memref<4xf32>) {
      neura.yield
    }
  }
}
"""
    with pytest.raises(ValueError, match="run Amoeba candidate enumeration"):
        extract_task_dfg_texts(source)


def test_success_probability_is_diagnostic_only_for_ranking():
    manifest = _two_task_manifest()
    low = score_candidates(manifest, _catalog(0.01), top_k=7)
    high = score_candidates(manifest, _catalog(0.99), top_k=7)

    assert low["footer"]["shortlist"] == high["footer"]["shortlist"]
    assert low["scores"] == high["scores"]


def test_frozen_geometry_features_are_not_named_as_memory_capabilities():
    assert CGRA_NODE_FEATURE_NAMES[5] == "is_north_or_west_boundary"
    assert CANDIDATE_CONTEXT_NAMES[8] == (
        "normalized_north_or_west_boundary_tiles"
    )
    for rows, cols in SHAPE_PROTOCOL.mapper_shapes:
        graph = make_cgra_graph(rows, cols)
        boundary_count = sum(row[5] for row in graph.node_features)
        assert boundary_count == rows + cols - 1
        context = candidate_context(rows, cols, 0, 1, 1)
        assert len(context) == len(CANDIDATE_CONTEXT_NAMES)


def test_final_ensemble_rejects_the_current_amoeba_architecture():
    checkpoint_paths = {
        "large_operation": ROOT / "models/final/large-operation.pt",
        "baseline": ROOT / "models/final/baseline.pt",
        "ranking": ROOT / "models/final/ranking.pt",
    }
    ensemble = load_ensemble_report(
        ROOT / "models/final/ensemble.json", checkpoint_paths,
    )
    validate_ensemble_architecture(ensemble, TRAINING_ARCHITECTURE)
    with pytest.raises(ValueError, match="outside the deployed model contract"):
        validate_ensemble_architecture(ensemble, CURRENT_AMOEBA_ARCHITECTURE)


def test_frozen_checkpoint_runs_all_eight_static_shapes():
    artifact = torch.load(
        ROOT / "models/final/ranking.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = PointwiseConfig(**artifact["config"]).validate()
    model = JointGraphShapeModel(config)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    dfg = parse_neura_dfg_representation(
        '''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.add"(%0, %0) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        ''',
        config.dfg_representation,
    )
    with torch.inference_mode():
        for rows, cols in SHAPE_PROTOCOL.mapper_shapes:
            context = torch.tensor([[
                candidate_context(
                    rows, cols, 0, 1, 1,
                    config.mapper_ii_ceiling, config.shape_protocol,
                )
            ]])
            output = model([dfg], [make_cgra_graph(rows, cols)], context)
            assert torch.isfinite(output["predicted_ii"]).all()
