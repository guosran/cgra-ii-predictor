import json
from pathlib import Path

import pytest
import torch

from adapters.amoeba_cost_catalog import (
    generate_catalog,
    load_candidate_manifest,
    load_analytical_input,
    load_mapper_model,
    sha256_file,
    source_task_body_sha256,
    validate_model_architecture,
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
from cgra_ii_predictor.dfg import (
    parse_neura_route_expanded_dfg,
)
from cgra_ii_predictor.mapper_model import (
    MAPPER_FEATURE_NAMES,
    mapper_feature_vector,
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


def _trip_count_fields(value):
    if value == "symbol_dynamic":
        return {"trip_count_kind": "symbol_dynamic"}
    return {"trip_count": value}


def _two_task_manifest(trip_counts=(10, 10)):
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
                    {"task": "A", **_trip_count_fields(trip_counts[0]),
                     "shape": first},
                    {"task": "B", **_trip_count_fields(trip_counts[1]),
                     "shape": second},
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


def _write_two_task_manifest(path, trip_counts=(10, 10)):
    manifest = _two_task_manifest(trip_counts)
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
        "tasks": [
            {"task": task, "body_sha256": "0" * 64,
             **_trip_count_fields(trip_count)}
            for task, trip_count in zip(("A", "B"), trip_counts)
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


def _catalog():
    return {
        "namespace": "test-model",
        "by_query": {
            (task, rows, cols): {
                "support_status": "supported",
                "predicted_ii": float(1 if cols == 8 else 2),
                "startup_cycles": 1.0,
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


def test_manifest_loader_accepts_corrected_frozen_manifest_without_max_cgras(
    tmp_path,
):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    manifest = load_candidate_manifest(path)
    assert manifest["candidate_count"] == 7
    assert manifest["candidates"][-1]["candidate_id"] == "candidate-6"
    assert set(manifest["queries"]) == {
        (task, rows, cols)
        for task in ("A", "B")
        for rows, cols in ((4, 4), (4, 8), (8, 4))
    }


def test_manifest_loader_accepts_symbol_dynamic_trip_counts(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path, ("symbol_dynamic", 10))

    manifest = load_candidate_manifest(path)

    assert manifest["task_trip_counts"] == {"A": None, "B": 10}
    assert manifest["task_facts"][0] == {
        "task": "A",
        "body_sha256": "0" * 64,
        "trip_count_kind": "symbol_dynamic",
    }
    assert all(
        "trip_count_kind" in choice and "trip_count" not in choice
        for candidate in manifest["candidates"]
        for choice in [candidate["task_shapes"][0]]
    )


def test_manifest_loader_rejects_mixed_trip_count_encodings(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["tasks"][0].update({
        "trip_count_kind": "symbol_dynamic",
    })
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="exactly one"):
        load_candidate_manifest(path)


def test_manifest_loader_rejects_candidate_trip_count_kind_mismatch(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["task_shapes"][0].pop("trip_count")
    records[1]["task_shapes"][0]["trip_count_kind"] = "symbol_dynamic"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="task facts"):
        load_candidate_manifest(path)


def test_manifest_loader_accepts_sparse_used_query_set(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # The frozen manifest contains only the 1x1/1x1 candidate. Therefore its
    # header must omit the otherwise legal single-task 1x2 query.
    records = [records[0], records[1], records[-1]]
    records[0]["architecture"].update({"grid_rows": 1, "grid_cols": 2})
    records[0]["cost_queries"] = [
        {"task": "A", "mapper_tile_rows": 4, "mapper_tile_cols": 4},
        {"task": "B", "mapper_tile_rows": 4, "mapper_tile_cols": 4},
    ]
    records[-1]["candidate_count"] = 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    manifest = load_candidate_manifest(path)
    assert manifest["candidate_count"] == 1
    assert manifest["queries"] == [("A", 4, 4), ("B", 4, 4)]


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


def test_manifest_loader_rejects_missing_cost_query(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["cost_queries"].pop()
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="cost_queries"):
        load_candidate_manifest(path)


def test_manifest_loader_requires_contiguous_candidate_ids(tmp_path):
    path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[2]["candidate_id"] = "candidate-99"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(ValueError, match="canonical manifest index"):
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
                "mapper_success_probability": "not_predicted",
                "uses_mapper_success_probability": False,
            },
        },
        "entries": [],
    }))
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_cost_catalog(cost_path, manifest)


def test_scoring_uses_the_frozen_manifest_candidates():
    manifest = _two_task_manifest()
    scores = score_candidates(manifest, _catalog(), top_k=0)

    assert scores["footer"]["candidate_count"] == 7
    assert scores["footer"]["valid_count"] == 7
    assert scores["scores"][6]["candidate_id"] == "candidate-6"
    assert scores["scores"][6]["valid"] is True
    assert scores["header"]["candidate_space"]["resource_semantics"] == (
        "all_task_rectangles_co_resident"
    )
    assert scores["header"]["candidate_space"]["kind"] == (
        "packing_pruned_static_shape_candidates"
    )
    assert scores["footer"]["cache"] == {
        "hits": 8,
        "misses": 6,
        "entries": 6,
    }
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


def test_task_dfg_extractor_can_select_one_function_region():
    source = """
module {
  func.func @main() {
    taskflow.task @A {
      amoeba.source_task_body_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    } {
      neura.kernel inputs(%arg0 : memref<4xf32>) {
        neura.yield
      }
    }
  }
  func.func @helper() {
    taskflow.task @B {
      amoeba.source_task_body_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    } {
      neura.kernel inputs(%arg0 : memref<4xf32>) {
        neura.yield
      }
    }
  }
}
"""

    assert set(extract_task_dfg_texts(source, "main")) == {"A"}
    assert set(extract_task_dfg_texts(source)) == {"A", "B"}


def test_task_dfg_extractor_rejects_missing_selected_function():
    with pytest.raises(ValueError, match="does not exist"):
        extract_task_dfg_texts("module { func.func @main() {} }", "missing")


def test_task_dfg_extractor_rejects_ambiguous_selected_function():
    source = """
module {
  func.func @main() {}
  func.func @main() {}
}
"""
    with pytest.raises(ValueError, match="ambiguous"):
        extract_task_dfg_texts(source, "main")


def test_direct_mapper_rejects_the_current_amoeba_architecture():
    _, _, metadata = load_mapper_model(
        ROOT / "models/final/mapper.pt", torch.device("cpu"),
    )
    validate_model_architecture(metadata, TRAINING_ARCHITECTURE)
    with pytest.raises(ValueError, match="outside the deployed model contract"):
        validate_model_architecture(metadata, CURRENT_AMOEBA_ARCHITECTURE)


def test_frozen_checkpoint_runs_all_eight_static_shapes():
    model, config, _ = load_mapper_model(
        ROOT / "models/final/mapper.pt", torch.device("cpu"),
    )
    dfg = parse_neura_route_expanded_dfg(
        '''
        %0 = "neura.constant"() : () -> !neura.data<i32, i1>
        %1 = "neura.add"(%0, %0)
        ''',
    )
    with torch.inference_mode():
        for rows, cols in SHAPE_PROTOCOL.mapper_shapes:
            features = torch.tensor([
                mapper_feature_vector(dfg, rows, cols, 0, 1, 1)
            ])
            output = model(features, torch.tensor([1.0]))
            assert len(features[0]) == len(MAPPER_FEATURE_NAMES)
            assert torch.isfinite(output).all()


def test_direct_mapper_generates_a_loadable_cost_catalog(tmp_path):
    manifest_path = tmp_path / "candidates.jsonl"
    _write_two_task_manifest(manifest_path)
    manifest = load_candidate_manifest(manifest_path)

    task_paths = {}
    for task in ("A", "B"):
        path = tmp_path / f"{task}.mlir"
        path.write_text(f'''module {{
          func.func @{task}_dfg() attributes {{
            amoeba.source_task_body_sha256 = "{'0' * 64}"
          }} {{
            %0 = "neura.constant"() : () -> !neura.data<i32, i1>
            %1 = "neura.add"(%0, %0)
            return
          }}
        }}''')
        task_paths[task] = path

    analytical_path = tmp_path / "analytical.json"
    analytical_path.write_text(json.dumps({
        "schema": "cgra-ii-amoeba-query-features",
        "function": "main",
        "provenance": {
            "candidate_manifest_sha256": manifest["manifest_sha256"],
            "task_dfg_sha256": {
                task: sha256_file(path) for task, path in task_paths.items()
            },
            "task_body_sha256": manifest["task_body_sha256"],
            "neura_opt_sha256": "1" * 64,
            "architecture_sha256": TRAINING_ARCHITECTURE,
            "analytical_lower_bound_source": "analysis_only",
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

    catalog, timing = generate_catalog(
        manifest_path, analytical_path, task_paths,
        ROOT / "models/final/mapper.pt", torch.device("cpu"),
    )
    catalog_path = tmp_path / "costs.json"
    catalog_path.write_text(json.dumps(catalog))
    loaded = load_cost_catalog(catalog_path, manifest)

    assert len(loaded["by_query"]) == 6
    assert all(
        entry["ii_mean_source"] == "direct_mapper_surrogate"
        for entry in catalog["entries"]
    )
    assert all("mapper_success_probability" not in entry for entry in catalog["entries"])
    assert timing["model_load_count"] == 1
    assert timing["model_forward_pass_count"] == 1

    mismatched = json.loads(analytical_path.read_text())
    mismatched["provenance"]["architecture_sha256"] = (
        CURRENT_AMOEBA_ARCHITECTURE
    )
    analytical_path.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="architecture does not match manifest"):
        generate_catalog(
            manifest_path, analytical_path, task_paths,
            ROOT / "models/final/mapper.pt", torch.device("cpu"),
        )
