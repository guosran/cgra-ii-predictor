import json
from pathlib import Path

import pytest
import torch

from adapters.amoeba_cost_catalog import (
    CANDIDATE_SCHEMA,
    CHECKPOINT_SCHEMA,
    generate_catalog,
    load_candidate_manifest,
    sha256_file,
)
from adapters.amoeba_protocol import (
    SEARCH_SCOPE,
    SHAPE_POLICY,
    SPATIAL_CAPACITY_POLICY,
)
from adapters.amoeba_frozen_pipeline import load_cost_catalog
from cgra_ii_predictor.mapper_model import (
    DirectMapperIIModel,
    MapperModelConfig,
    mapper_feature_names,
)
from cgra_ii_predictor.shape_protocol import (
    SHAPE_PROTOCOL_2X2_ID,
    SHAPE_PROTOCOL_4X4_ID,
    get_shape_protocol,
)


ARCHITECTURE_2X2 = (
    "6f4a9a1815dcc0d97c00fd6ee20424fa9420ace90654da29e70d283cba7a611f"
)
ARCHITECTURE_4X4 = (
    "f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6"
)
TASK_BODY_SHA256 = "0" * 64


def _physical_shapes(protocol_id, grid_rows, grid_cols, max_cgras):
    protocol = get_shape_protocol(protocol_id)
    return [
        (rows, cols, mapper_rows, mapper_cols)
        for (rows, cols), (mapper_rows, mapper_cols) in sorted(
            protocol.physical_to_mapper,
            key=lambda item: (item[0][0] * item[0][1], item[0][0]),
        )
        if rows * cols <= max_cgras
        and rows <= grid_rows
        and cols <= grid_cols
    ]


def _write_manifest(
    path: Path,
    *,
    protocol_id=SHAPE_PROTOCOL_2X2_ID,
    architecture_sha256=ARCHITECTURE_2X2,
    per_cgra_rows=2,
    per_cgra_cols=2,
    max_cgras_per_task=4,
    grid_rows=4,
    grid_cols=4,
):
    shapes = _physical_shapes(
        protocol_id, grid_rows, grid_cols, max_cgras_per_task,
    )
    header = {
        "record_type": "header",
        "schema": CANDIDATE_SCHEMA,
        "search_scope": SEARCH_SCOPE,
        "shape_policy": SHAPE_POLICY,
        "spatial_capacity_policy": SPATIAL_CAPACITY_POLICY,
        "function": "main",
        "architecture": {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "per_cgra_tile_rows": per_cgra_rows,
            "per_cgra_tile_cols": per_cgra_cols,
            "spec_sha256": architecture_sha256,
        },
        "max_cgras_per_task": max_cgras_per_task,
        "tasks": [{
            "task": "A",
            "body_sha256": TASK_BODY_SHA256,
            "trip_count": 10,
        }],
        "cost_queries": [
            {
                "task": "A",
                "mapper_tile_rows": mapper_rows,
                "mapper_tile_cols": mapper_cols,
            }
            for _, _, mapper_rows, mapper_cols in shapes
        ],
    }
    records = [header]
    for rows, cols, mapper_rows, mapper_cols in shapes:
        records.append({
            "record_type": "candidate",
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": f"candidate-{len(records) - 1}",
            "task_shapes": [{
                "task": "A",
                "trip_count": 10,
                "shape": {
                    "kind": "rect",
                    "rows": rows,
                    "cols": cols,
                    "cgra_count": rows * cols,
                    "cgra_shape": f"{rows}x{cols}",
                    "mapper_tile_rows": mapper_rows,
                    "mapper_tile_cols": mapper_cols,
                },
            }],
        })
    records.append({
        "record_type": "footer",
        "schema": CANDIDATE_SCHEMA,
        "candidate_count": len(shapes),
    })
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_task_dfg(path: Path):
    path.write_text(f'''module {{
      func.func @A_dfg() attributes {{
        amoeba.source_task_body_sha256 = "{TASK_BODY_SHA256}"
      }} {{
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
        %2 = "neura.add"(%1, %1) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
        return
      }}
    }}''')


def _write_analytical_input(path, manifest, task_dfg, architecture_sha256):
    path.write_text(json.dumps({
        "schema": "cgra-ii-amoeba-query-features",
        "function": "main",
        "provenance": {
            "candidate_manifest_sha256": manifest["manifest_sha256"],
            "task_dfg_sha256": {"A": sha256_file(task_dfg)},
            "task_body_sha256": manifest["task_body_sha256"],
            "neura_opt_sha256": "1" * 64,
            "architecture_sha256": architecture_sha256,
            "rec_res_source": "analysis_only",
            "startup_cycles_source": "analysis_only",
        },
        "entries": [{
            "task": task,
            "mapper_tile_rows": rows,
            "mapper_tile_cols": cols,
            "rec_mii": 1,
            "res_mii": 1,
            "lower_bound": 1,
            "startup_cycles": 2,
        } for task, rows, cols in manifest["queries"]],
    }))


def _write_checkpoint(
    path: Path,
    *,
    protocol_id=SHAPE_PROTOCOL_2X2_ID,
    architecture_sha256=ARCHITECTURE_2X2,
    supported_architectures=None,
):
    config = MapperModelConfig(shape_protocol=protocol_id).validate()
    model = DirectMapperIIModel(config)
    protocol = get_shape_protocol(protocol_id)
    artifact = {
        "schema": CHECKPOINT_SCHEMA,
        "feature_names": list(mapper_feature_names(protocol_id)),
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "architecture_sha256": architecture_sha256,
        "supported_architecture_sha256": (
            [architecture_sha256] if supported_architectures is None
            else supported_architectures
        ),
        "training_manifest_sha256": "2" * 64,
        "supported_mapper_shapes": [list(shape) for shape in protocol.mapper_shapes],
    }
    torch.save(artifact, path)


def test_2x2_manifest_generates_catalog_with_2x2_checkpoint(tmp_path):
    manifest_path = tmp_path / "candidates.jsonl"
    _write_manifest(manifest_path)
    manifest = load_candidate_manifest(manifest_path)
    assert manifest["shape_protocol"] == SHAPE_PROTOCOL_2X2_ID
    assert len(manifest["queries"]) == 8
    assert set(manifest["queries"]) == {
        ("A", rows, cols)
        for rows, cols in get_shape_protocol(
            SHAPE_PROTOCOL_2X2_ID,
        ).mapper_shapes
    }

    task_dfg = tmp_path / "A.mlir"
    _write_task_dfg(task_dfg)
    analytical_path = tmp_path / "analytical.json"
    _write_analytical_input(
        analytical_path, manifest, task_dfg, ARCHITECTURE_2X2,
    )
    checkpoint = tmp_path / "mapper-2x2.pt"
    _write_checkpoint(checkpoint)

    catalog, _ = generate_catalog(
        manifest_path, analytical_path, {"A": task_dfg}, checkpoint,
        torch.device("cpu"),
    )
    catalog_path = tmp_path / "costs.json"
    catalog_path.write_text(json.dumps(catalog))
    loaded = load_cost_catalog(catalog_path, manifest)

    assert len(loaded["by_query"]) == 8
    assert all(
        entry["support_status"] == "supported"
        and entry["ii_mean_source"] == "direct_mapper_surrogate"
        for entry in catalog["entries"]
    )
    assert catalog["predictor_metadata"]["shape_protocol"]["protocol_id"] == (
        SHAPE_PROTOCOL_2X2_ID
    )
    assert catalog["predictor_metadata"]["model"]["supported_mapper_shapes"] == [
        list(shape) for shape in get_shape_protocol(
            SHAPE_PROTOCOL_2X2_ID,
        ).mapper_shapes
    ]


def test_manifest_rejects_unregistered_per_cgra_protocol_and_2x2_overcap(
    tmp_path,
):
    path = tmp_path / "candidates.jsonl"
    _write_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["architecture"]["per_cgra_tile_rows"] = 3
    records[0]["architecture"]["per_cgra_tile_cols"] = 3
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="unsupported per-CGRA tile shape"):
        load_candidate_manifest(path)

    _write_manifest(path, max_cgras_per_task=5)
    with pytest.raises(ValueError, match="supports at most 4"):
        load_candidate_manifest(path)


def test_4x4_candidate_domain_remains_supported(tmp_path):
    path = tmp_path / "candidates-4x4.jsonl"
    _write_manifest(
        path,
        protocol_id=SHAPE_PROTOCOL_4X4_ID,
        architecture_sha256=ARCHITECTURE_4X4,
        per_cgra_rows=4,
        per_cgra_cols=4,
        max_cgras_per_task=4,
        grid_rows=2,
        grid_cols=2,
    )
    manifest = load_candidate_manifest(path)
    assert manifest["shape_protocol"] == SHAPE_PROTOCOL_4X4_ID
    assert manifest["queries"] == [
        ("A", 4, 4), ("A", 4, 8), ("A", 8, 4), ("A", 8, 8),
    ]


@pytest.mark.parametrize(
    ("protocol_id", "architecture_sha256", "supported_architectures", "error"),
    [
        (
            SHAPE_PROTOCOL_4X4_ID,
            ARCHITECTURE_2X2,
            [ARCHITECTURE_2X2],
            "shape protocol does not match",
        ),
        (
            SHAPE_PROTOCOL_2X2_ID,
            "3" * 64,
            ["3" * 64],
            "outside the deployed model contract",
        ),
    ],
)
def test_2x2_manifest_rejects_wrong_checkpoint_domain_or_architecture(
    tmp_path, protocol_id, architecture_sha256, supported_architectures, error,
):
    manifest_path = tmp_path / "candidates.jsonl"
    _write_manifest(manifest_path)
    manifest = load_candidate_manifest(manifest_path)
    task_dfg = tmp_path / "A.mlir"
    _write_task_dfg(task_dfg)
    analytical_path = tmp_path / "analytical.json"
    _write_analytical_input(
        analytical_path, manifest, task_dfg, ARCHITECTURE_2X2,
    )
    checkpoint = tmp_path / "mapper.pt"
    _write_checkpoint(
        checkpoint,
        protocol_id=protocol_id,
        architecture_sha256=architecture_sha256,
        supported_architectures=supported_architectures,
    )

    with pytest.raises(ValueError, match=error):
        generate_catalog(
            manifest_path, analytical_path, {"A": task_dfg}, checkpoint,
            torch.device("cpu"),
        )
