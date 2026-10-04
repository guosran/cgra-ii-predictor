from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from adapters import train_per_cgra_2x2_model as trainer
from adapters.amoeba_cost_catalog import load_mapper_model
from cgra_ii_predictor.shape_protocol import (
    SHAPE_PROTOCOL_2X2,
    SHAPE_PROTOCOL_2X2_ID,
)
from mapping_artifact_protocol import (
    canonical_json_sha256,
    mapper_input_identity,
    parse_single_mapping,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: dict, field: str) -> dict:
    payload = dict(value)
    payload.pop(field, None)
    payload[field] = canonical_json_sha256(payload)
    return payload


def _fixture(tmp_path: Path, *, one_censored: bool = True) -> dict[str, Path]:
    collection = tmp_path / "native-collection"
    dfg_root = collection / "inputs" / "dfg"
    dfg_root.mkdir(parents=True)
    source_manifest_sha = "a" * 64
    architecture_sha = "b" * 64
    neura_opt_sha = "c" * 64
    family_by_identity = {}
    group_rows = []
    queries = []
    dfg_text_by_identity = {}

    operations = ("add", "mul", "sub", "xor", "and", "or")
    for index, operation in enumerate(operations):
        text = f'''\
%0 = "neura.constant"() : () -> !neura.data<i32, i1>
%1 = "neura.data_mov"(%0) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
%2 = "neura.{operation}"(%1, %1) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
'''
        identity = mapper_input_identity(text, normalize_static_shapes=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        visible = trainer._visible_identity(text)
        group_id = hashlib.sha256(f"group-{index}".encode()).hexdigest()
        family = f"fixture/{operation}"
        family_by_identity[identity] = family
        dfg_text_by_identity[identity] = (text, digest, visible, group_id)
        group_rows.append({
            "group_id": group_id,
            "mapper_input_identities": [identity],
            "source_program_families": [family],
        })
        dfg_path = dfg_root / f"{identity}.mlir"
        dfg_path.write_text(text)
        for rows, cols in SHAPE_PROTOCOL_2X2.mapper_shapes:
            shape_index = SHAPE_PROTOCOL_2X2.mapper_shapes.index((rows, cols))
            query = {
                "query_id": f"{identity}/{rows}x{cols}",
                "mapper_input_identity": identity,
                "model_visible_graph_identity": visible,
                "dfg_path": f"inputs/dfg/{identity}.mlir",
                "dfg_sha256": digest,
                "rows": rows,
                "cols": cols,
                "physical_cgra_rows": rows // 2,
                "physical_cgra_columns": cols // 2,
                "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
                "source_candidate_ids": [f"fixture/{index}"],
                "source_program_families": [family],
                "source_names": [f"fixture-{index}"],
                "domains": ["fixture"],
                "leakage_lineage_ids": [group_id],
            }
            queries.append(query)

    source_groups = {
        "schema": trainer.GROUPS_SCHEMA,
        "manifest_sha256": source_manifest_sha,
        "groups": group_rows,
    }
    source_groups_path = tmp_path / "groups.json"
    source_groups_path.write_text(json.dumps(source_groups, sort_keys=True))

    implementation_paths = (
        "adapters/collect_per_cgra_2x2_mappings.py",
        "adapters/collect_kernelbench_mappings.py",
        "adapters/mapping_artifact_protocol.py",
        "src/cgra_ii_predictor/shape_protocol.py",
    )
    implementation_hashes = {
        relative: _sha256_file(trainer.ROOT / relative)
        for relative in implementation_paths
    }
    provenance = _seal({
        "schema": trainer.COLLECTION_SCHEMA,
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "mapper_shapes": [list(shape) for shape in SHAPE_PROTOCOL_2X2.mapper_shapes],
        "source_manifest_schema": "cgra-ii-nine-shape-outcomes-v1",
        "source_manifest_sha256": source_manifest_sha,
        "source_dfg_count": len(group_rows),
        "source_candidate_count": len(group_rows),
        "architecture_sha256": architecture_sha,
        "architecture_facts": {
            "multi_cgra_rows": 4,
            "multi_cgra_columns": 4,
            "per_cgra_tile_rows": 2,
            "per_cgra_tile_columns": 2,
            "total_tile_count": 64,
        },
        "neura_opt_sha256": neura_opt_sha,
        "mapper_strategy": "heuristic",
        "implementation_sha256_by_path": implementation_hashes,
        "external_timeout_seconds": 60,
        "collection_worker_count": 1,
        "expected_query_count": len(queries),
        "old_native_labels_reused": False,
        "source_lineage_policy": "preserve identities and groups",
    }, "provenance_sha256")
    query_manifest = _seal({
        "schema": trainer.COLLECTION_SCHEMA,
        "provenance": provenance,
        "query_count": len(queries),
        "queries": queries,
    }, "manifest_sha256")
    query_manifest_path = collection / "query-manifest.json"
    query_manifest_path.write_text(json.dumps(query_manifest, sort_keys=True))
    (collection / "provenance.json").write_text(json.dumps(provenance, sort_keys=True))

    outcomes = []
    for query in queries:
        identity = query["mapper_input_identity"]
        group_index = next(
            index for index, row in enumerate(group_rows)
            if row["mapper_input_identities"] == [identity]
        )
        shape_index = SHAPE_PROTOCOL_2X2.mapper_shapes.index(
            (query["rows"], query["cols"]),
        )
        rec_mii, res_mii = 1 + group_index % 2, 1
        lower_bound = max(rec_mii, res_mii)
        compiled_ii = lower_bound + (group_index + shape_index) % 3
        censor = one_censored and group_index == 0 and shape_index == 0
        status = "censored" if censor else "success"
        artifact_relative = (
            f"artifacts/{identity}/{query['rows']}x{query['cols']}/mapped.mlir"
        )
        result_relative = (
            f"artifacts/{identity}/{query['rows']}x{query['cols']}/result.json"
        )
        result_path = collection / result_relative
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "schema": trainer.COLLECTION_SCHEMA,
            "query_id": query["query_id"],
            "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
            "mapper_input_identity": identity,
            "model_visible_graph_identity": query["model_visible_graph_identity"],
            "dfg_path": query["dfg_path"],
            "dfg_sha256": query["dfg_sha256"],
            "rows": query["rows"],
            "cols": query["cols"],
            "physical_cgra_rows": query["physical_cgra_rows"],
            "physical_cgra_columns": query["physical_cgra_columns"],
            "source_manifest_sha256": source_manifest_sha,
            "architecture_sha256": architecture_sha,
            "neura_opt_sha256": neura_opt_sha,
            "collection_provenance_sha256": provenance["provenance_sha256"],
            "analysis": {
                "status": "success",
                "rec_mii": rec_mii,
                "res_mii": res_mii,
                "lower_bound": lower_bound,
            },
            "status": status,
            "compiled_ii": None if censor else compiled_ii,
        }
        if censor:
            result["censor_reason"] = "mapper_native_search_failed"
        else:
            artifact_path = collection / artifact_relative
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_text = (
                "module attributes {mapping_info = {"
                f"compiled_ii = {compiled_ii} : i32, "
                f"rec_mii = {rec_mii} : i32, res_mii = {res_mii} : i32, "
                f"x_tiles = {query['cols']} : i32, "
                f"y_tiles = {query['rows']} : i32"
                "}} {\"neura.pe\"() {x = 0 : i32, y = 0 : i32}}"
            )
            artifact_path.write_text(artifact_text)
            facts = parse_single_mapping(artifact_text, query["rows"], query["cols"])
            assert facts["compiled_ii"] == compiled_ii
            result["mapped_artifact_path"] = artifact_relative
            result["mapped_artifact_sha256"] = _sha256_file(artifact_path)
        result_path.write_text(json.dumps(result, sort_keys=True))
        outcome = {
            **{field: query[field] for field in trainer._QUERY_FIELDS},
            "status": status,
            "compiled_ii": None if censor else compiled_ii,
        }
        outcome["result_path"] = result_relative
        outcome["result_sha256"] = _sha256_file(result_path)
        outcome["censor_reason"] = result.get("censor_reason")
        outcomes.append(outcome)

    success_count = sum(row["status"] == "success" for row in outcomes)
    censored_count = sum(row["status"] == "censored" for row in outcomes)
    outcome_manifest = _seal({
        "schema": trainer.OUTCOMES_SCHEMA,
        "collection_provenance_sha256": provenance["provenance_sha256"],
        "query_manifest_sha256": _sha256_file(query_manifest_path),
        "query_count": len(queries),
        "terminal_count": len(queries),
        "success_count": success_count,
        "censored_count": censored_count,
        "queries": outcomes,
    }, "manifest_sha256")
    outcomes_path = collection / "outcomes.json"
    outcomes_path.write_text(json.dumps(outcome_manifest, sort_keys=True))
    completion = {
        "schema": trainer.COLLECTION_SCHEMA,
        "query_manifest_sha256": _sha256_file(query_manifest_path),
        "outcomes_sha256": _sha256_file(outcomes_path),
        "query_count": len(queries),
        "success_count": success_count,
        "censored_count": censored_count,
    }
    (collection / "collection-complete.json").write_text(json.dumps(completion))

    exclusions = {
        "schema": trainer.EXCLUSIONS_SCHEMA,
        "source_manifest_sha256": source_manifest_sha,
        "source_groups_sha256": _sha256_file(source_groups_path),
        "excluded_source_sha256": [],
        "excluded_mapper_input_identities": [],
        "reasons_by_dfg": {},
        "policy": "fixture excludes none",
        "amoeba_benchmark_overlap_audit_complete": False,
    }
    exclusions_path = tmp_path / "training-exclusions.json"
    exclusions_path.write_text(json.dumps(exclusions, sort_keys=True))
    return {
        "collection": collection,
        "source_groups": source_groups_path,
        "exclusions": exclusions_path,
    }


def _args(paths: dict[str, Path], output_dir: Path, *, epochs: int = 1) -> Namespace:
    return Namespace(
        collection=paths["collection"],
        source_groups=paths["source_groups"],
        exclude_dfg=paths["exclusions"],
        output_dir=output_dir,
        epochs=epochs,
        seeds=(17, 41),
        prepare_only=False,
        wait_for_completion=False,
        poll_seconds=0.01,
    )


def test_fixture_trains_candidate_and_roundtrips_through_catalog_loader(tmp_path):
    paths = _fixture(tmp_path)
    output_dir = tmp_path / "candidate"
    report = trainer.run(_args(paths, output_dir))

    assert report["shape_protocol_id"] == SHAPE_PROTOCOL_2X2_ID
    assert report["training"]["feature_count"] == 61
    assert report["training"]["native_status_counts"] == {
        "pending": 0, "success": 47, "censored": 1,
    }
    assert report["training"]["eligible_successful_row_count"] == 47
    assert report["evaluation"]["test_holdout_used_for_training_or_tuning"] is False
    assert report["evaluation"]["by_split"]["test"]["row_count"] > 0
    assert report["checkpoint"]["catalog_loader_replay_max_abs_difference"] == 0.0

    checkpoint = output_dir / "mapper.pt"
    model, config, metadata = load_mapper_model(checkpoint, torch.device("cpu"))
    assert config.shape_protocol == SHAPE_PROTOCOL_2X2_ID
    assert set(config.enabled_feature_names) == set(trainer.COMPACT_FEATURE_NAMES)
    assert {tuple(shape) for shape in metadata["supported_mapper_shapes"]} == set(
        SHAPE_PROTOCOL_2X2.mapper_shapes
    )
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert artifact["schema"] == "cgra-ii-direct-mapper-ensemble"
    assert artifact["config"]["shape_protocol"] == SHAPE_PROTOCOL_2X2_ID
    assert len(artifact["feature_names"]) == 148
    assert len(artifact["compact_feature_names"]) == 61
    assert model is not None

    split = report["training"]["split"]
    assert set(split["train_group_ids"]).isdisjoint(split["validation_group_ids"])
    assert set(split["train_group_ids"]).isdisjoint(split["test_group_ids"])
    assert set(split["validation_group_ids"]).isdisjoint(split["test_group_ids"])


def test_success_result_provenance_mismatch_fails_closed(tmp_path):
    paths = _fixture(tmp_path, one_censored=False)
    collection = paths["collection"]
    outcomes_path = collection / "outcomes.json"
    outcomes = json.loads(outcomes_path.read_text())
    target = next(row for row in outcomes["queries"] if row["status"] == "success")
    result_path = collection / target["result_path"]
    result = json.loads(result_path.read_text())
    result["collection_provenance_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result, sort_keys=True))
    target["result_sha256"] = _sha256_file(result_path)
    outcomes = _seal(outcomes, "manifest_sha256")
    outcomes_path.write_text(json.dumps(outcomes, sort_keys=True))
    completion_path = collection / "collection-complete.json"
    completion = json.loads(completion_path.read_text())
    completion["outcomes_sha256"] = _sha256_file(outcomes_path)
    completion_path.write_text(json.dumps(completion))

    with pytest.raises(ValueError, match="provenance mismatch"):
        trainer.run(_args(paths, tmp_path / "must-not-exist"))


def test_censored_outcome_cannot_become_a_numeric_training_label(tmp_path):
    paths = _fixture(tmp_path, one_censored=True)
    report = trainer.run(_args(paths, tmp_path / "candidate"))
    assert report["training"]["native_status_counts"]["censored"] == 1
    assert report["training"]["native_success_count_before_exclusions"] == 47
    assert report["training"]["eligible_successful_row_count"] == 47
    assert report["training"]["censored_queries_are_missing_labels"] is True
