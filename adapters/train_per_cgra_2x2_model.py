#!/usr/bin/env python3
"""Train a grouped, candidate-only mapper-II model for 2x2-tile CGRAs.

This adapter accepts only fresh native outcomes from the independent per-CGRA
2x2 collection protocol.  It keeps the existing 4x4 model untouched and emits
an explicitly protocol-bound checkpoint under the caller's candidate path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "adapters", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from amoeba_cost_catalog import (  # noqa: E402
    ENSEMBLE_CHECKPOINT_SCHEMA,
    load_mapper_model,
    sha256_file,
)
from cgra_ii_predictor.dfg import require_neura_route_expanded_dfg  # noqa: E402
from cgra_ii_predictor.mapper_model import (  # noqa: E402
    DirectMapperIIEnsemble,
    DirectMapperIIModel,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_names,
    mapper_feature_vector,
)
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    SHAPE_PROTOCOL_2X2,
    SHAPE_PROTOCOL_2X2_ID,
)
from mapping_artifact_protocol import (  # noqa: E402
    canonical_json_sha256,
    mapper_input_identity,
    parse_single_mapping,
)


COLLECTION_SCHEMA = "cgra-ii-per-cgra-2x2-native-collection-v1"
OUTCOMES_SCHEMA = "cgra-ii-per-cgra-2x2-outcomes-v1"
GROUPS_SCHEMA = "cgra-ii-nine-shape-groups-v1"
EXCLUSIONS_SCHEMA = "cgra-ii-per-cgra-2x2-training-exclusions-v1"
REPORT_SCHEMA = "cgra-ii-per-cgra-2x2-training-report-v1"
SPLIT_SEED = 20261004
DEFAULT_SEEDS = (17, 41, 113, 239)
DEFAULT_EPOCHS = 80
HIDDEN_DIMENSIONS = (64, 32)
SHAPE_PROTOCOL = SHAPE_PROTOCOL_2X2
MAPPER_SHAPES = SHAPE_PROTOCOL.mapper_shapes
MAPPER_FEATURE_NAMES_2X2 = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
# Freeze the established compact61 mask without importing experiment drivers.
COMPACT_INDICES = tuple(range(65, 79)) + tuple(range(93, 118)) + tuple(range(134, 156))
COMPACT_FEATURE_NAMES = tuple(
    MAPPER_FEATURE_NAMES[index] for index in COMPACT_INDICES
)
if len(COMPACT_FEATURE_NAMES) != 61 or len(set(COMPACT_FEATURE_NAMES)) != 61:
    raise AssertionError("the established compact61 feature contract changed")
if not set(COMPACT_FEATURE_NAMES).issubset(MAPPER_FEATURE_NAMES_2X2):
    raise AssertionError("compact61 names do not fit the 2x2 feature contract")

_QUERY_FIELDS = (
    "query_id", "mapper_input_identity", "model_visible_graph_identity",
    "dfg_path", "dfg_sha256", "rows", "cols", "physical_cgra_rows",
    "physical_cgra_columns", "shape_protocol_id", "source_candidate_ids",
    "source_program_families", "source_names", "domains",
    "leakage_lineage_ids",
)


def _visible_identity(text: str) -> str:
    graph = require_neura_route_expanded_dfg(text)
    return canonical_json_sha256({
        "node_types": graph.node_types,
        "edges": graph.edges,
        "semantic_edges": graph.semantic_edges,
    })


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _valid_sha(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _verify_embedded_digest(value: dict[str, Any], field: str, label: str) -> str:
    digest = _valid_sha(value.get(field), f"{label} digest")
    payload = {key: item for key, item in value.items() if key != field}
    if canonical_json_sha256(payload) != digest:
        raise ValueError(f"{label} digest does not match its contents")
    return digest


def _safe_path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a nonempty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to the collection")
    base = root.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} escapes the collection root") from error
    return resolved


def _load_source_groups(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _read_json(path, "source groups")
    if document.get("schema") != GROUPS_SCHEMA:
        raise ValueError("unsupported source-groups schema")
    _valid_sha(document.get("manifest_sha256"), "source manifest SHA-256")
    groups = document.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("source groups must contain a nonempty group roster")
    by_identity: dict[str, dict[str, Any]] = {}
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("source group must be an object")
        group_id = group.get("group_id")
        identities = group.get("mapper_input_identities")
        families = group.get("source_program_families")
        if (not isinstance(group_id, str) or not group_id or
                group_id in group_ids or not isinstance(identities, list) or
                not identities or not isinstance(families, list)):
            raise ValueError("source group lacks a unique ID, identities, or families")
        group_ids.add(group_id)
        normalized = dict(group)
        normalized["group_id"] = group_id
        normalized["source_program_families"] = sorted(set(
            family for family in families
            if isinstance(family, str) and family
        ))
        for identity in identities:
            _valid_sha(identity, "source mapper-input identity")
            if identity in by_identity:
                raise ValueError("one mapper-input identity belongs to multiple groups")
            by_identity[identity] = normalized
    return document, by_identity


def _load_exclusion_groups(
    path: Path,
    document: dict[str, Any],
    groups_path: Path,
    by_identity: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    exclusions = _read_json(path, "training exclusions")
    if exclusions.get("schema") != EXCLUSIONS_SCHEMA:
        raise ValueError("unsupported training-exclusions schema")
    if exclusions.get("source_manifest_sha256") != document.get("manifest_sha256"):
        raise ValueError("training exclusions bind a different source manifest")
    if exclusions.get("source_groups_sha256") != sha256_file(groups_path):
        raise ValueError("training exclusions bind a different source-groups file")

    excluded_hashes = exclusions.get("excluded_source_sha256")
    excluded_identities = exclusions.get("excluded_mapper_input_identities")
    reasons = exclusions.get("reasons_by_dfg")
    if (not isinstance(excluded_hashes, list) or
            not isinstance(excluded_identities, list) or
            not isinstance(reasons, dict)):
        raise ValueError("training exclusions are malformed")
    excluded_hash_set = {
        _valid_sha(value, "excluded DFG SHA-256") for value in excluded_hashes
    }
    excluded_identity_set = {
        _valid_sha(value, "excluded mapper-input identity")
        for value in excluded_identities
    }
    if (len(excluded_hash_set) != len(excluded_hashes) or
            len(excluded_identity_set) != len(excluded_identities) or
            set(reasons) != excluded_hash_set):
        raise ValueError("training-exclusion identities or reasons are inconsistent")

    query_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_identities = set()
    for query in queries:
        query_by_hash[query["dfg_sha256"]].append(query)
        query_identities.add(query["mapper_input_identity"])
    if not excluded_hash_set.issubset(query_by_hash):
        raise ValueError("an excluded DFG is absent from the frozen query roster")
    if not excluded_identity_set.issubset(query_identities):
        raise ValueError("an excluded mapper identity is absent from the query roster")

    excluded_groups: set[str] = set()
    for identity in excluded_identity_set:
        excluded_groups.add(by_identity[identity]["group_id"])
    for digest, record in reasons.items():
        if not isinstance(record, dict):
            raise ValueError("training-exclusion reason must be an object")
        reason_group = record.get("group_id")
        reason_names = record.get("reasons")
        if (not isinstance(reason_group, str) or
                not isinstance(reason_names, list) or not reason_names or
                any(not isinstance(item, str) or not item for item in reason_names)):
            raise ValueError("training-exclusion reason lacks group or reason text")
        identities_for_hash = {
            query["mapper_input_identity"] for query in query_by_hash[digest]
        }
        groups_for_hash = {by_identity[item]["group_id"] for item in identities_for_hash}
        if groups_for_hash != {reason_group}:
            raise ValueError("training-exclusion DFG group identity changed")
        excluded_groups.add(reason_group)

    audit_complete = exclusions.get("amoeba_benchmark_overlap_audit_complete")
    if not isinstance(audit_complete, bool):
        raise ValueError("Amoeba overlap-audit status must be boolean")
    return excluded_groups, {
        "file_sha256": sha256_file(path),
        "excluded_source_sha256_count": len(excluded_hash_set),
        "excluded_mapper_input_identity_count": len(excluded_identity_set),
        "excluded_group_count": len(excluded_groups),
        "amoeba_benchmark_overlap_audit_complete": audit_complete,
        "policy": exclusions.get("policy"),
    }


def _validate_protocol_manifest(
    root: Path,
    source_groups_path: Path,
    exclusions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], set[str], dict[str, Any]]:
    manifest_path = root / "query-manifest.json"
    provenance_path = root / "provenance.json"
    manifest = _read_json(manifest_path, "query manifest")
    if manifest.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("unsupported per-CGRA 2x2 query-manifest schema")
    _verify_embedded_digest(manifest, "manifest_sha256", "query manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("query manifest lacks provenance")
    if provenance.get("schema") != COLLECTION_SCHEMA:
        raise ValueError("collection provenance schema changed")
    _verify_embedded_digest(provenance, "provenance_sha256", "collection provenance")
    if _read_json(provenance_path, "provenance") != provenance:
        raise ValueError("provenance.json differs from frozen query manifest")
    if (provenance.get("shape_protocol_id") != SHAPE_PROTOCOL_2X2_ID or
            provenance.get("mapper_shapes") != [list(shape) for shape in MAPPER_SHAPES]):
        raise ValueError("collection shape protocol is not the declared 2x2 protocol")
    if provenance.get("old_native_labels_reused") is not False:
        raise ValueError("2x2 protocol must use fresh native labels")
    facts = provenance.get("architecture_facts")
    expected_facts = {
        "multi_cgra_rows": 4,
        "multi_cgra_columns": 4,
        "per_cgra_tile_rows": 2,
        "per_cgra_tile_columns": 2,
        "total_tile_count": 64,
    }
    if facts != expected_facts:
        raise ValueError("collection architecture is not a 4x4 array of 2x2 CGRAs")
    for key in ("source_manifest_sha256", "architecture_sha256", "neura_opt_sha256"):
        _valid_sha(provenance.get(key), f"provenance {key}")
    implementation_hashes = provenance.get("implementation_sha256_by_path")
    expected_implementation_paths = {
        "adapters/collect_per_cgra_2x2_mappings.py",
        "adapters/collect_kernelbench_mappings.py",
        "adapters/mapping_artifact_protocol.py",
        "src/cgra_ii_predictor/shape_protocol.py",
    }
    if not isinstance(implementation_hashes, dict) or set(implementation_hashes) != expected_implementation_paths:
        raise ValueError("collection lacks the frozen mapper implementation hashes")
    for relative, expected_hash in implementation_hashes.items():
        _valid_sha(expected_hash, f"implementation hash for {relative}")
        implementation_path = ROOT / relative
        if not implementation_path.is_file() or sha256_file(implementation_path) != expected_hash:
            raise ValueError(f"collection implementation changed: {relative}")

    source_groups, group_by_identity = _load_source_groups(source_groups_path)
    if provenance.get("source_manifest_sha256") != source_groups.get("manifest_sha256"):
        raise ValueError("collection and source groups bind different manifests")
    queries = manifest.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("query manifest has no queries")
    if (manifest.get("query_count") != len(queries) or
            provenance.get("expected_query_count") != len(queries)):
        raise ValueError("query count disagrees with frozen manifest")

    by_key: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    query_by_identity: dict[str, dict[str, Any]] = {}
    graph_cache: dict[str, Any] = {}
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("query entry must be an object")
        for field in _QUERY_FIELDS:
            if field not in query:
                raise ValueError(f"query lacks required field {field}")
        identity = _valid_sha(query["mapper_input_identity"], "query mapper-input identity")
        visible = _valid_sha(query["model_visible_graph_identity"], "visible graph identity")
        dfg_sha = _valid_sha(query["dfg_sha256"], "query DFG SHA-256")
        rows, columns = query["rows"], query["cols"]
        if (isinstance(rows, bool) or not isinstance(rows, int) or
                isinstance(columns, bool) or not isinstance(columns, int)):
            raise ValueError("query shape dimensions must be integers")
        shape = SHAPE_PROTOCOL.validate_mapper_shape(rows, columns)
        if query.get("shape_protocol_id") != SHAPE_PROTOCOL_2X2_ID:
            raise ValueError("query uses a different shape protocol")
        if (query.get("physical_cgra_rows") != rows // 2 or
                query.get("physical_cgra_columns") != columns // 2):
            raise ValueError("query mapper and physical CGRA dimensions disagree")
        if query.get("query_id") != f"{identity}/{rows}x{columns}":
            raise ValueError("query ID does not bind identity and mapper shape")
        group = group_by_identity.get(identity)
        if group is None:
            raise ValueError("query mapper identity is absent from source groups")
        query_families = query.get("source_program_families")
        if (not isinstance(query_families, list) or
                not set(query_families).issubset(group["source_program_families"])):
            raise ValueError("query source families differ from canonical group metadata")
        key = (identity, shape)
        if key in by_key:
            raise ValueError("query manifest duplicates an identity/shape pair")
        by_key[key] = query
        previous = query_by_identity.get(identity)
        if previous is not None and (
                previous["dfg_sha256"] != dfg_sha or
                previous["model_visible_graph_identity"] != visible or
                previous["dfg_path"] != query["dfg_path"]):
            raise ValueError("one mapper identity points to inconsistent DFG provenance")
        query_by_identity[identity] = query
        if identity not in graph_cache:
            dfg_path = _safe_path(root, query["dfg_path"], "query DFG path")
            if not dfg_path.is_file() or sha256_file(dfg_path) != dfg_sha:
                raise ValueError(f"query DFG is missing or has changed: {dfg_path}")
            text = dfg_path.read_text()
            if mapper_input_identity(text, normalize_static_shapes=True) != identity:
                raise ValueError("DFG contents differ from mapper-input identity")
            graph = require_neura_route_expanded_dfg(text)
            if _visible_identity(text) != visible:
                raise ValueError("DFG contents differ from visible-graph identity")
            graph_cache[identity] = (text, graph)

    expected_pairs = {
        (identity, shape)
        for identity in query_by_identity for shape in MAPPER_SHAPES
    }
    if set(by_key) != expected_pairs:
        raise ValueError("frozen query roster is not complete for all eight shapes")
    if set(query_by_identity) != set(group_by_identity):
        raise ValueError("collection identities differ from canonical source groups")
    excluded_groups, exclusion_summary = _load_exclusion_groups(
        exclusions_path, source_groups, source_groups_path, group_by_identity, queries,
    )
    bundle = {
        "by_key": by_key,
        "graph_cache": graph_cache,
        "group_bindings": group_by_identity,
        "query_by_identity": query_by_identity,
        "raw_queries": queries,
        "source_groups": source_groups,
    }
    return manifest, provenance, bundle, excluded_groups, exclusion_summary


def _result_path(query: dict[str, Any]) -> str:
    return (f"artifacts/{query['mapper_input_identity']}/"
            f"{query['rows']}x{query['cols']}/result.json")


def _artifact_path(query: dict[str, Any]) -> str:
    return (f"artifacts/{query['mapper_input_identity']}/"
            f"{query['rows']}x{query['cols']}/mapped.mlir")


def _verify_result(
    root: Path,
    query: dict[str, Any],
    outcome: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if outcome.get("result_path") != _result_path(query):
        raise ValueError("outcome result path differs from the frozen query")
    path = _safe_path(root, outcome.get("result_path"), "native result path")
    if not path.is_file() or sha256_file(path) != outcome.get("result_sha256"):
        raise ValueError(f"native result is missing or its hash changed: {path}")
    result = _read_json(path, "native mapping result")
    expected = {
        "schema": COLLECTION_SCHEMA,
        "query_id": query["query_id"],
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "mapper_input_identity": query["mapper_input_identity"],
        "model_visible_graph_identity": query["model_visible_graph_identity"],
        "dfg_path": query["dfg_path"],
        "dfg_sha256": query["dfg_sha256"],
        "rows": query["rows"],
        "cols": query["cols"],
        "physical_cgra_rows": query["physical_cgra_rows"],
        "physical_cgra_columns": query["physical_cgra_columns"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "architecture_sha256": provenance["architecture_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "collection_provenance_sha256": provenance["provenance_sha256"],
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"native result {field} provenance mismatch: {path}")
    if result.get("status") != outcome.get("status"):
        raise ValueError("native result and outcome status differ")
    if outcome.get("status") == "success":
        compiled = outcome.get("compiled_ii")
        if (isinstance(compiled, bool) or not isinstance(compiled, int) or
                compiled <= 0 or compiled > 20 or result.get("compiled_ii") != compiled):
            raise ValueError("successful native result has an invalid compiled II")
        analysis = result.get("analysis")
        if not isinstance(analysis, dict) or analysis.get("status") != "success":
            raise ValueError("successful native result lacks analytical bounds")
        rec_mii, res_mii, lower_bound = (
            analysis.get("rec_mii"), analysis.get("res_mii"),
            analysis.get("lower_bound"),
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (rec_mii, res_mii, lower_bound)):
            raise ValueError("native analysis bounds must be nonnegative integers")
        if lower_bound != max(rec_mii, res_mii) or compiled < lower_bound:
            raise ValueError("compiled II is inconsistent with native analytical bounds")
        if result.get("mapped_artifact_path") != _artifact_path(query):
            raise ValueError("successful result points to an unexpected mapping artifact")
        mapped_path = _safe_path(root, result.get("mapped_artifact_path"), "mapped artifact path")
        artifact_sha = _valid_sha(result.get("mapped_artifact_sha256"), "mapped artifact SHA-256")
        if not mapped_path.is_file() or sha256_file(mapped_path) != artifact_sha:
            raise ValueError("successful native mapping artifact hash mismatch")
        facts = parse_single_mapping(mapped_path.read_text(), query["rows"], query["cols"])
        if (facts["compiled_ii"] != compiled or facts["rec_mii"] != rec_mii or
                facts["res_mii"] != res_mii):
            raise ValueError("mapping artifact facts differ from recorded native result")
        if outcome.get("compiled_ii") != result.get("compiled_ii"):
            raise ValueError("outcome label differs from its native result")
    elif outcome.get("status") == "censored":
        if (outcome.get("compiled_ii") is not None or
                result.get("compiled_ii") is not None):
            raise ValueError("censored mapping must not carry a numeric II")
    else:
        raise ValueError("pending query unexpectedly has a native result")
    return result


def _load_outcomes(
    root: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    bundle: dict[str, Any],
    *,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    outcome_path = root / "outcomes.json"
    outcomes = _read_json(outcome_path, "outcomes")
    if outcomes.get("schema") != OUTCOMES_SCHEMA:
        raise ValueError("unsupported outcomes schema")
    _verify_embedded_digest(outcomes, "manifest_sha256", "outcomes")
    if outcomes.get("query_manifest_sha256") != sha256_file(root / "query-manifest.json"):
        raise ValueError("outcomes bind a different query manifest")
    if outcomes.get("collection_provenance_sha256") != provenance["provenance_sha256"]:
        raise ValueError("outcomes bind different collection provenance")
    outcome_rows = outcomes.get("queries")
    raw_queries = bundle["raw_queries"]
    if not isinstance(outcome_rows, list) or len(outcome_rows) != len(raw_queries):
        raise ValueError("outcomes query roster length changed")
    outcome_by_id = {}
    for row in outcome_rows:
        if not isinstance(row, dict) or not isinstance(row.get("query_id"), str):
            raise ValueError("outcome query row is malformed")
        if row["query_id"] in outcome_by_id:
            raise ValueError("outcomes duplicate a query ID")
        outcome_by_id[row["query_id"]] = row
    rows: list[dict[str, Any]] = []
    counts = {"pending": 0, "success": 0, "censored": 0}
    for query in raw_queries:
        outcome = outcome_by_id.get(query["query_id"])
        if outcome is None:
            raise ValueError("outcomes omit a frozen query")
        for field in _QUERY_FIELDS:
            if outcome.get(field) != query.get(field):
                raise ValueError(f"outcome {field} differs from frozen query")
        status = outcome.get("status")
        if status not in counts:
            raise ValueError(f"unknown native outcome status: {status!r}")
        counts[status] += 1
        if status == "pending":
            if outcome.get("compiled_ii") is not None or "result_path" in outcome:
                raise ValueError("pending native query carries a result or II")
            continue
        _verify_result(root, query, outcome, provenance)
        identity = query["mapper_input_identity"]
        group = bundle["group_bindings"][identity]
        if identity in bundle["graph_cache"]:
            _text, graph = bundle["graph_cache"][identity]
        else:
            raise AssertionError("validated DFG graph cache is incomplete")
        result_path = _safe_path(root, outcome["result_path"], "native result path")
        result = _read_json(result_path, "native result")
        if status == "success" and group["group_id"] not in bundle["excluded_groups"]:
            analysis = result["analysis"]
            shape = (query["rows"], query["cols"])
            full_features = tuple(float(value) for value in mapper_feature_vector(
                graph, shape[0], shape[1], float(analysis["rec_mii"]),
                float(analysis["res_mii"]), float(analysis["lower_bound"]),
                shape_protocol=SHAPE_PROTOCOL_2X2_ID,
            ))
            if len(full_features) != len(MAPPER_FEATURE_NAMES_2X2):
                raise ValueError("2x2 mapper feature width changed")
            rows.append({
                "candidate_id": query["query_id"],
                "query": identity,
                "group": group["group_id"],
                "group_families": group["source_program_families"],
                "shape": shape,
                "full_features": full_features,
                "ii": float(result["compiled_ii"]),
                "lower_bound": float(analysis["lower_bound"]),
                "rec_mii": int(analysis["rec_mii"]),
                "res_mii": int(analysis["res_mii"]),
                "source_sha256": query["dfg_sha256"],
                "mapper_input_identity": identity,
                "status": "success",
            })
    if set(outcome_by_id) != {query["query_id"] for query in raw_queries}:
        raise ValueError("outcomes include a query absent from the frozen manifest")
    if (outcomes.get("query_count") != len(raw_queries) or
            outcomes.get("terminal_count") != counts["success"] + counts["censored"] or
            outcomes.get("success_count") != counts["success"] or
            outcomes.get("censored_count") != counts["censored"]):
        raise ValueError("outcomes terminal counts disagree with rows")

    complete_path = root / "collection-complete.json"
    completion: dict[str, Any] = {}
    if require_complete or counts["pending"] == 0:
        if counts["pending"]:
            raise ValueError(
                f"native collection has {counts['pending']} pending queries; "
                "use --wait-for-completion to wait for terminal outcomes"
            )
        completion = _read_json(complete_path, "collection completion")
        if (completion.get("schema") != COLLECTION_SCHEMA or
                completion.get("query_manifest_sha256") != sha256_file(root / "query-manifest.json") or
                completion.get("outcomes_sha256") != sha256_file(outcome_path) or
                completion.get("query_count") != len(raw_queries) or
                completion.get("success_count") != counts["success"] or
                completion.get("censored_count") != counts["censored"]):
            raise ValueError("collection completion record does not bind all outcomes")
    return rows, counts, {
        "outcomes_sha256": sha256_file(outcome_path),
        "completion_sha256": sha256_file(complete_path) if completion else None,
        "completion": completion,
    }


def _split_groups(groups: Iterable[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    ordered = sorted(set(groups))
    if len(ordered) < 3:
        raise ValueError("at least three unexcluded source groups are required")
    random.Random(seed).shuffle(ordered)
    train_count = max(1, int(len(ordered) * 0.70))
    validation_count = max(1, int(len(ordered) * 0.15))
    if train_count + validation_count >= len(ordered):
        train_count = len(ordered) - 2
        validation_count = 1
    assignment = {}
    for group in ordered[:train_count]:
        assignment[group] = "train"
    for group in ordered[train_count:train_count + validation_count]:
        assignment[group] = "validation"
    for group in ordered[train_count + validation_count:]:
        assignment[group] = "test"
    if set(assignment.values()) != {"train", "validation", "test"}:
        raise AssertionError("group split omitted one of its three folds")
    return assignment


def _balanced_group_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group"]].append(index)
    random_groups = {
        group for group, indices in by_group.items()
        if rows[indices[0]]["group_families"] and all(
            family.startswith("random-dfg/")
            for family in rows[indices[0]]["group_families"]
        )
    }
    real_groups = set(by_group) - random_groups
    if random_groups and real_groups:
        mass_by_group = {
            **{group: 0.5 / len(real_groups) for group in real_groups},
            **{group: 0.5 / len(random_groups) for group in random_groups},
        }
    else:
        mass_by_group = {group: 1.0 / len(by_group) for group in by_group}
    values = [0.0] * len(rows)
    for group, indices in by_group.items():
        each = mass_by_group[group] / len(indices)
        for index in indices:
            values[index] = each
    return torch.tensor(values, dtype=torch.float32)


def _rank_queries(
    rows: list[dict[str, Any]], weights: torch.Tensor,
) -> list[tuple[list[int], float]]:
    by_query: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_query[(row["group"], row["query"])].append(index)
    complete = []
    expected = set(MAPPER_SHAPES)
    for (group, _query), indices in by_query.items():
        shapes = {rows[index]["shape"] for index in indices}
        if shapes == expected and len(indices) == len(expected):
            if {rows[index]["group"] for index in indices} != {group}:
                raise ValueError("rank query crosses source groups")
            complete.append((indices, float(weights[indices].sum())))
    total = sum(mass for _indices, mass in complete)
    return [(indices, mass / total) for indices, mass in complete] if total else []


def _set_top1_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    logits = -predicted
    best = truth == truth.min()
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[best], dim=0)


def _fit_member(
    training: list[dict[str, Any]], seed: int, epochs: int,
) -> DirectMapperIIModel:
    raw = torch.tensor([row["full_features"] for row in training], dtype=torch.float32)
    mean = raw.mean(0)
    scale = raw.std(0, unbiased=False)
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    config = MapperModelConfig(
        hidden_dimensions=HIDDEN_DIMENSIONS,
        shape_protocol=SHAPE_PROTOCOL_2X2_ID,
        enabled_feature_names=COMPACT_FEATURE_NAMES,
    ).validate()
    torch.manual_seed(seed)
    model = DirectMapperIIModel(config, mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    lower = torch.tensor([row["lower_bound"] for row in training], dtype=torch.float32)
    truth = torch.tensor([row["ii"] for row in training], dtype=torch.float32)
    weights = _balanced_group_weights(training)
    queries = _rank_queries(training, weights)
    better: list[int] = []
    worse: list[int] = []
    pair_weights: list[float] = []
    for indices, mass in queries:
        pairs = [
            (left, right) for left in indices for right in indices
            if truth[left] < truth[right]
        ]
        if pairs:
            better.extend(left for left, _right in pairs)
            worse.extend(right for _left, right in pairs)
            pair_weights.extend([mass / len(pairs)] * len(pairs))
    pair_weight_tensor = torch.tensor(pair_weights, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(raw, lower)
        point = (
            torch.nn.functional.smooth_l1_loss(
                predicted, truth, reduction="none",
            ) * weights
        ).sum()
        pair = (
            torch.nn.functional.relu(
                0.5 - (predicted[worse] - predicted[better]),
            ) * pair_weight_tensor
        ).sum() if pair_weights else predicted.sum() * 0.0
        top1 = sum(
            _set_top1_loss(predicted[indices], truth[indices]) * mass
            for indices, mass in queries
        ) if queries else predicted.sum() * 0.0
        loss = point + 0.1 * pair + 0.3 * top1
        if not torch.isfinite(loss):
            raise ValueError("training loss became non-finite")
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model: torch.nn.Module, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    features = torch.tensor([row["full_features"] for row in rows], dtype=torch.float32)
    lower = torch.tensor([row["lower_bound"] for row in rows], dtype=torch.float32)
    with torch.inference_mode():
        return [float(value) for value in model(features, lower).tolist()]


def _score(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("evaluation prediction count changed")
    if not rows:
        return {
            "row_count": 0, "group_count": 0, "row_mae": None,
            "rounded_exact_ii": None, "rounded_within_one_ii": None,
            "complete_query_count": 0, "partial_query_count": 0,
            "shape_hit": None, "mean_shape_regret": None,
        }
    by_query: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    absolute_errors = []
    exact = []
    within_one = []
    prediction_by_candidate = {}
    shape_order = {shape: index for index, shape in enumerate(MAPPER_SHAPES)}
    for row, prediction in zip(rows, predictions):
        if not math.isfinite(prediction):
            raise ValueError("model produced a non-finite evaluation prediction")
        absolute_errors.append(abs(prediction - row["ii"]))
        rounded = int(round(prediction))
        exact.append(rounded == int(row["ii"]))
        within_one.append(abs(rounded - int(row["ii"])) <= 1)
        prediction_by_candidate[row["candidate_id"]] = prediction
        by_query[(row["group"], row["query"])].append((row, prediction))

    complete = []
    partial_count = 0
    for items in by_query.values():
        if {row["shape"] for row, _prediction in items} != set(MAPPER_SHAPES):
            partial_count += 1
            continue
        selected_row, _selected_prediction = min(
            items, key=lambda item: (
                item[1], shape_order[item[0]["shape"]],
            ),
        )
        oracle = min(row["ii"] for row, _prediction in items)
        complete.append({
            "hit": selected_row["ii"] == oracle,
            "regret": selected_row["ii"] - oracle,
        })
    return {
        "row_count": len(rows),
        "group_count": len({row["group"] for row in rows}),
        "row_mae": statistics.mean(absolute_errors),
        "rounded_exact_ii": statistics.mean(exact),
        "rounded_within_one_ii": statistics.mean(within_one),
        "complete_query_count": len(complete),
        "partial_query_count": partial_count,
        "shape_hit": statistics.mean(item["hit"] for item in complete) if complete else None,
        "mean_shape_regret": statistics.mean(item["regret"] for item in complete) if complete else None,
    }


def _group_split_summary(
    rows: list[dict[str, Any]], assignment: dict[str, str], excluded_groups: set[str],
) -> dict[str, Any]:
    all_groups = set(assignment)
    for split in ("train", "validation", "test"):
        if not {row["group"] for row in rows if assignment.get(row["group"]) == split}.issubset(all_groups):
            raise AssertionError("row group missing from split map")
    train_groups = {group for group, split in assignment.items() if split == "train"}
    validation_groups = {group for group, split in assignment.items() if split == "validation"}
    test_groups = {group for group, split in assignment.items() if split == "test"}
    if (train_groups & validation_groups or train_groups & test_groups or
            validation_groups & test_groups or excluded_groups & all_groups):
        raise ValueError("source groups cross folds or the embargo")
    return {
        "seed": SPLIT_SEED,
        "policy": "deterministic-shuffled-70-15-15-whole-source-groups",
        "source_group_count": len(assignment) + len(excluded_groups),
        "excluded_group_count": len(excluded_groups),
        "train_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "test_group_count": len(test_groups),
        "train_group_ids": sorted(train_groups),
        "validation_group_ids": sorted(validation_groups),
        "test_group_ids": sorted(test_groups),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _wait_until_terminal(collection: Path, poll_seconds: float) -> None:
    if poll_seconds <= 0 or not math.isfinite(poll_seconds):
        raise ValueError("poll seconds must be positive and finite")
    while True:
        try:
            outcomes = _read_json(collection / "outcomes.json", "outcomes")
            pending = sum(
                row.get("status") == "pending"
                for row in outcomes.get("queries", [])
                if isinstance(row, dict)
            )
            complete = (collection / "collection-complete.json").is_file()
            if pending == 0 and complete:
                return
        except ValueError:
            pass
        time.sleep(poll_seconds)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    seeds = tuple(args.seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("an ensemble requires at least two distinct seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be integers")
    if args.output_dir.exists() and not args.prepare_only:
        raise FileExistsError(args.output_dir)
    if args.wait_for_completion:
        _wait_until_terminal(args.collection, args.poll_seconds)

    manifest, provenance, bundle, excluded_groups, exclusion_summary = (
        _validate_protocol_manifest(
            args.collection, args.source_groups, args.exclude_dfg,
        )
    )
    bundle["excluded_groups"] = excluded_groups
    canonical_identity_roster = {
        identity
        for group in bundle["source_groups"]["groups"]
        for identity in group["mapper_input_identities"]
    }
    if set(bundle["query_by_identity"]) != canonical_identity_roster:
        raise ValueError("collection identities differ from source-groups roster")

    if args.prepare_only:
        _prepared_rows, prepared_counts, _prepared_source = _load_outcomes(
            args.collection, manifest, provenance, bundle, require_complete=False,
        )
        eligible = sorted({
            group["group_id"] for identity, group in bundle["group_bindings"].items()
            if group["group_id"] not in excluded_groups
        })
        assignment = _split_groups(eligible)
        return {
            "schema": REPORT_SCHEMA,
            "status": "validated_training_plan_only",
            "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
            "query_count": len(manifest["queries"]),
            "outcome_status_counts": prepared_counts,
            "excluded_source": exclusion_summary,
            "split": _group_split_summary([], assignment, excluded_groups),
            "writes_model": False,
        }

    all_success_rows, status_counts, source_artifacts = _load_outcomes(
        args.collection, manifest, provenance, bundle, require_complete=True,
    )
    if not all_success_rows:
        raise ValueError("no non-embargoed successful native rows are available")
    eligible_groups = sorted({row["group"] for row in all_success_rows})
    # Keep groups with only censored rows in their deterministic fold as well;
    # rows are simply absent from point and rank losses.
    all_nonexcluded_group_ids = {
        group["group_id"] for group in bundle["source_groups"]["groups"]
        if group["group_id"] not in excluded_groups
    }
    if not eligible_groups:
        raise ValueError("all successful native rows are embargoed")
    assignment = _split_groups(all_nonexcluded_group_ids)
    rows_by_split = {
        name: [row for row in all_success_rows if assignment[row["group"]] == name]
        for name in ("train", "validation", "test")
    }
    training = rows_by_split["train"]
    if not training:
        raise ValueError("training split has no successful native labels")
    torch.set_num_threads(1)
    members = [_fit_member(training, seed, args.epochs) for seed in seeds]
    ensemble = DirectMapperIIEnsemble(len(members), members[0].config)
    for destination, source in zip(ensemble.members, members):
        destination.load_state_dict(source.state_dict(), strict=True)
    ensemble.eval()
    predictions = {
        name: _predict(ensemble, rows) for name, rows in rows_by_split.items()
    }
    score = {
        name: _score(rows_by_split[name], predictions[name])
        for name in ("train", "validation", "test")
    }

    subset = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": row["source_sha256"],
        "mapper_input_identity": row["mapper_input_identity"],
        "shape": list(row["shape"]),
        "compiled_ii": int(row["ii"]),
        "lower_bound": int(row["lower_bound"]),
        "group": row["group"],
    } for row in training]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "mapper.pt"
    artifact = {
        "schema": ENSEMBLE_CHECKPOINT_SCHEMA,
        "output_mode": "continuous",
        "deployment_readout": "arithmetic_mean_bounded_continuous_regression",
        "ranking_readout": "arithmetic_mean_bounded_continuous_regression",
        "config": members[0].config.to_dict(),
        "feature_names": list(MAPPER_FEATURE_NAMES_2X2),
        "compact_feature_names": list(COMPACT_FEATURE_NAMES),
        "state_dict": ensemble.state_dict(),
        "ensemble_member_count": len(members),
        "ensemble_reduction": "arithmetic_mean",
        "ensemble_seeds": list(seeds),
        "architecture_sha256": provenance["architecture_sha256"],
        "supported_architecture_sha256": [provenance["architecture_sha256"]],
        "architecture_compatibility_rule": "exact_architecture_sha256",
        "supported_mapper_shapes": [list(shape) for shape in MAPPER_SHAPES],
        "training_mapper_shapes": [list(shape) for shape in MAPPER_SHAPES],
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "training_manifest_sha256": provenance["source_manifest_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "collection_provenance_sha256": provenance["provenance_sha256"],
        "query_manifest_sha256": sha256_file(args.collection / "query-manifest.json"),
        "outcomes_manifest_sha256": source_artifacts["outcomes_sha256"],
        "collection_complete_sha256": source_artifacts["completion_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "source_groups_sha256": sha256_file(args.source_groups),
        "training_exclusions_sha256": exclusion_summary["file_sha256"],
        "training_subset_sha256": canonical_json_sha256(subset),
        "training_split_seed": SPLIT_SEED,
        "training_group_ids": sorted({row["group"] for row in training}),
        "candidate_only": True,
    }
    torch.save(artifact, checkpoint_path)
    loaded, loaded_config, loader_metadata = load_mapper_model(
        checkpoint_path, torch.device("cpu"),
    )
    replay_rows = rows_by_split["test"] or training
    replay_expected = _predict(ensemble, replay_rows)
    replay_observed = _predict(loaded, replay_rows)
    max_replay_error = max(
        (abs(left - right) for left, right in zip(replay_expected, replay_observed)),
        default=0.0,
    )
    if (max_replay_error != 0.0 or
            loaded_config.to_dict() != members[0].config.to_dict() or
            loader_metadata.get("supported_mapper_shapes") != [list(shape) for shape in MAPPER_SHAPES]):
        raise ValueError("catalog loader changed 2x2 checkpoint protocol or predictions")

    report = {
        "schema": REPORT_SCHEMA,
        "status": "candidate_trained_holdout_reported_not_promoted",
        "target": "fresh_native_compiled_ii_for_given_dfg_and_2x2_cgra_shape",
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "shape_roster": [list(shape) for shape in MAPPER_SHAPES],
        "architecture_facts": provenance["architecture_facts"],
        "architecture_sha256": provenance["architecture_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "training": {
            "recipe": "compact61_reference_mlp_four_seed_arithmetic_mean",
            "feature_count": len(COMPACT_FEATURE_NAMES),
            "feature_names": list(COMPACT_FEATURE_NAMES),
            "seeds": list(seeds),
            "epochs": args.epochs,
            "hidden_dimensions": list(HIDDEN_DIMENSIONS),
            "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
            "loss": "weighted SmoothL1 + 0.1 pairwise hinge + 0.3 set top-1",
            "group_weighting": "random and program strata each receive 50% total mass when both exist; equal groups within each stratum; equal successful rows within group",
            "native_query_count": len(manifest["queries"]),
            "native_status_counts": status_counts,
            "native_success_count_before_exclusions": status_counts["success"],
            "censored_queries_are_missing_labels": True,
            "old_4x4_labels_reused": False,
            "eligible_successful_row_count": len(all_success_rows),
            "excluded_successful_row_count": (
                status_counts["success"] - len(all_success_rows)
            ),
            "successful_train_row_count": len(training),
            "successful_validation_row_count": len(rows_by_split["validation"]),
            "successful_test_row_count": len(rows_by_split["test"]),
            "training_dfg_count": len({row["source_sha256"] for row in training}),
            "split": _group_split_summary(
                all_success_rows, assignment, excluded_groups,
            ),
            "exclusion": exclusion_summary,
            "amoeba_benchmark_overlap_audit_complete": exclusion_summary[
                "amoeba_benchmark_overlap_audit_complete"
            ],
            "candidate_only_until_full_overlap_audit": not exclusion_summary[
                "amoeba_benchmark_overlap_audit_complete"
            ],
        },
        "evaluation": {
            "selection_policy": "minimum predicted II; ties follow frozen shape roster order",
            "test_holdout_used_for_training_or_tuning": False,
            "by_split": score,
        },
        "provenance": {
            "collection_provenance_sha256": provenance["provenance_sha256"],
            "query_manifest_sha256": artifact["query_manifest_sha256"],
            "outcomes_manifest_sha256": artifact["outcomes_manifest_sha256"],
            "collection_complete_sha256": artifact["collection_complete_sha256"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "source_groups_sha256": artifact["source_groups_sha256"],
            "training_exclusions_sha256": artifact["training_exclusions_sha256"],
            "training_subset_sha256": artifact["training_subset_sha256"],
        },
        "checkpoint": {
            "path": "mapper.pt",
            "schema": ENSEMBLE_CHECKPOINT_SCHEMA,
            "catalog_loader_replay_max_abs_difference": max_replay_error,
            "loader_feature_count": len(MAPPER_FEATURE_NAMES_2X2),
            "loader_supported_mapper_shapes": loader_metadata[
                "supported_mapper_shapes"
            ],
        },
    }
    _write_json(output_dir / "training-report.json", report)
    return report


def _parse_seeds(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(values) < 2 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide at least two distinct seeds")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--source-groups", type=Path, required=True)
    parser.add_argument("--exclude-dfg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
