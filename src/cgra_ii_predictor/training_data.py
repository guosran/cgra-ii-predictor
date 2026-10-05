"""Portable, integrity-checked reader for the frozen per-CGRA 2x2 dataset."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from .checkpoint import COMPACT_FEATURE_NAMES, COMPACT_INDICES
from .mapper_model import mapper_feature_names
from .shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID


DATASET_SCHEMA = "cgra-ii-per-cgra-2x2-v1-feature-dataset-v1"
_FEATURE_NAMES_2X2 = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
COMPACT_FEATURE_INDICES = tuple(
    _FEATURE_NAMES_2X2.index(name) for name in COMPACT_FEATURE_NAMES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def load_training_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Load and validate portable matrices, labels, exclusions, and splits.

    Censored labels are required to be null. They are retained in the returned
    table and never converted into a numeric training target.
    """
    dataset_dir = dataset_dir.resolve()
    package_dir = dataset_dir.parent
    manifest_path = dataset_dir / "manifest.json"
    manifest = _read_object(manifest_path, "dataset manifest")
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported per-CGRA 2x2 dataset schema")
    recorded_manifest_sha = _sha(manifest.get("manifest_sha256"), "dataset manifest digest")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_json_sha256(payload) != recorded_manifest_sha:
        raise ValueError("dataset manifest digest mismatch")
    if (manifest.get("input_contract") !=
            "original v1 raw-kernel 148-feature contract; not the current v2 feature contract"):
        raise ValueError("dataset feature contract is not the frozen original v1 contract")
    feature_names = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
    if (manifest.get("feature_count") != 148 or
            manifest.get("feature_names") != list(feature_names) or
            manifest.get("compact_feature_indices") != list(COMPACT_FEATURE_INDICES) or
            manifest.get("compact_feature_count") != 61):
        raise ValueError("dataset feature roster or compact61 mask changed")
    canonical_indices = manifest.get("canonical_feature_universe_indices")
    if canonical_indices != list(COMPACT_INDICES):
        raise ValueError("dataset canonical compact-feature selector changed")
    compact_indices = manifest["compact_feature_indices"]
    if (any(isinstance(index, bool) or not isinstance(index, int) or
            not 0 <= index < len(feature_names) for index in compact_indices) or
            tuple(feature_names[index] for index in compact_indices) != COMPACT_FEATURE_NAMES):
        raise ValueError("dataset compact-feature positions do not select the frozen C0 inputs")
    if (manifest.get("shape_protocol_id") != SHAPE_PROTOCOL_2X2_ID or
            manifest.get("shape_roster") != [list(shape) for shape in SHAPE_PROTOCOL_2X2.mapper_shapes]):
        raise ValueError("dataset shape protocol changed")
    if (manifest.get("split_seed") != 20261004 or
            manifest.get("split_policy") != "deterministic-shuffled-70-15-15-whole-source-groups"):
        raise ValueError("dataset group split contract changed")

    def asset(relative_field: str, hash_field: str) -> Path:
        relative = manifest.get(relative_field)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"dataset {relative_field} is missing")
        path = (dataset_dir / relative).resolve()
        try:
            path.relative_to(package_dir)
        except ValueError as error:
            raise ValueError("dataset provenance asset escapes its package") from error
        if not path.is_file() or sha256_file(path) != manifest.get(hash_field):
            raise ValueError(f"dataset provenance asset hash mismatch: {relative}")
        return path

    groups_path = asset("source_groups_file", "source_groups_file_sha256")
    exclusions_path = asset("training_exclusions_file", "training_exclusions_file_sha256")
    groups = _read_object(groups_path, "source groups")
    exclusions = _read_object(exclusions_path, "training exclusions")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("dataset provenance is missing")
    if (groups.get("manifest_sha256") != provenance.get("source_manifest_sha256") or
            exclusions.get("source_manifest_sha256") != provenance.get("source_manifest_sha256") or
            exclusions.get("source_groups_sha256") != sha256_file(groups_path)):
        raise ValueError("dataset, groups, and exclusions bind different source manifests")
    identity_to_group: dict[str, dict[str, Any]] = {}
    group_ids: set[str] = set()
    group_rows = groups.get("groups")
    if not isinstance(group_rows, list) or not group_rows:
        raise ValueError("source groups are missing")
    for group in group_rows:
        if not isinstance(group, dict):
            raise ValueError("source group is malformed")
        group_id, identities = group.get("group_id"), group.get("mapper_input_identities")
        if (not isinstance(group_id, str) or not group_id or group_id in group_ids or
                not isinstance(identities, list) or not identities):
            raise ValueError("source group identity roster is malformed")
        group_ids.add(group_id)
        for identity in identities:
            _sha(identity, "source mapper-input identity")
            if identity in identity_to_group:
                raise ValueError("mapper-input identity occurs in multiple source groups")
            identity_to_group[identity] = group

    excluded_identities = exclusions.get("excluded_mapper_input_identities")
    excluded_hashes = exclusions.get("excluded_source_sha256")
    reasons = exclusions.get("reasons_by_dfg")
    if not (isinstance(excluded_identities, list) and isinstance(excluded_hashes, list) and isinstance(reasons, dict)):
        raise ValueError("training exclusions are malformed")
    excluded_identity_set = set(excluded_identities)
    excluded_hash_set = set(excluded_hashes)
    if (len(excluded_identity_set) != len(excluded_identities) or
            len(excluded_hash_set) != len(excluded_hashes) or
            set(reasons) != excluded_hash_set):
        raise ValueError("training exclusion hashes or identities are inconsistent")
    if not excluded_identity_set.issubset(identity_to_group):
        raise ValueError("excluded mapper identity is absent from source groups")
    excluded_group_ids = {identity_to_group[value]["group_id"] for value in excluded_identity_set}
    if excluded_group_ids != set(manifest.get("excluded_group_ids", [])):
        raise ValueError("dataset excluded groups differ from frozen training exclusions")
    assignments = manifest.get("split_assignments")
    if (not isinstance(assignments, dict) or set(assignments) != group_ids - excluded_group_ids or
            set(assignments.values()) != {"train", "validation", "test"}):
        raise ValueError("dataset split assignment roster is incomplete")
    ordered_groups = sorted(assignments)
    random.Random(20261004).shuffle(ordered_groups)
    train_count = max(1, int(len(ordered_groups) * 0.70))
    validation_count = max(1, int(len(ordered_groups) * 0.15))
    if train_count + validation_count >= len(ordered_groups):
        train_count, validation_count = len(ordered_groups) - 2, 1
    expected_assignments = {
        **{group: "train" for group in ordered_groups[:train_count]},
        **{group: "validation" for group in ordered_groups[train_count:train_count + validation_count]},
        **{group: "test" for group in ordered_groups[train_count + validation_count:]},
    }
    if assignments != expected_assignments:
        raise ValueError("dataset source-group split differs from the frozen deterministic policy")

    data_name = manifest.get("data_file")
    if not isinstance(data_name, str) or Path(data_name).is_absolute() or ".." in Path(data_name).parts:
        raise ValueError("dataset data path must remain inside its directory")
    data_path = dataset_dir / data_name
    if not data_path.is_file() or sha256_file(data_path) != manifest.get("data_sha256"):
        raise ValueError("dataset matrix SHA-256 mismatch")
    rows = []
    with gzip.open(data_path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid dataset row {line_no}") from error
            if not isinstance(row, dict):
                raise ValueError(f"dataset row {line_no} is not an object")
            rows.append(row)
    if len(rows) != manifest.get("counts", {}).get("query_rows"):
        raise ValueError("dataset row count differs from manifest")

    by_query: dict[str, list[dict[str, Any]]] = {}
    seen_candidate_ids: set[str] = set()
    order = {shape: i for i, shape in enumerate(SHAPE_PROTOCOL_2X2.mapper_shapes)}
    for row in rows:
        identity = _sha(row.get("mapper_input_identity"), "row mapper-input identity")
        source_sha = _sha(row.get("source_sha256"), "row DFG SHA-256")
        _sha(row.get("model_visible_graph_identity"), "row visible graph identity")
        _sha(row.get("native_result_sha256"), "row native result SHA-256")
        group = identity_to_group.get(identity)
        if group is None or row.get("query") != identity or row.get("group") != group["group_id"]:
            raise ValueError("dataset row is not bound to its canonical source group")
        if (source_sha in excluded_hash_set) != (group["group_id"] in excluded_group_ids):
            raise ValueError("DFG source hash and source-group exclusion disagree")
        if not set(row.get("source_program_families", [])).issubset(set(group.get("source_program_families", []))):
            raise ValueError("dataset row source families differ from source group")
        shape_raw = row.get("shape")
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise ValueError("dataset row shape is malformed")
        shape = SHAPE_PROTOCOL_2X2.validate_mapper_shape(*shape_raw)
        if (row.get("shape_protocol_id") != SHAPE_PROTOCOL_2X2_ID or
                row.get("candidate_id") != f"{identity}/{shape[0]}x{shape[1]}"):
            raise ValueError("dataset row candidate identity changed")
        if row["candidate_id"] in seen_candidate_ids:
            raise ValueError("dataset duplicates a mapper query")
        seen_candidate_ids.add(row["candidate_id"])

        excluded = group["group_id"] in excluded_group_ids
        if (row.get("excluded") is not excluded or
                row.get("eligible_for_training") is not (not excluded) or
                row.get("split") != ("excluded" if excluded else assignments[group["group_id"]])):
            raise ValueError("dataset row eligibility or group split changed")
        status = row.get("native_status")
        label = row.get("ii")
        if status == "success":
            if isinstance(label, bool) or not isinstance(label, int) or not 1 <= label <= 20:
                raise ValueError("successful native label must be an integer in the mapper interval")
        elif status == "censored":
            if label is not None:
                raise ValueError("censored native label must remain null")
        else:
            raise ValueError("portable training data must contain terminal outcomes only")
        rec_mii, res_mii, lower = (row.get("rec_mii"), row.get("res_mii"), row.get("lower_bound"))
        if lower is not None:
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (rec_mii, res_mii, lower)):
                raise ValueError("analytical bounds must be nonnegative integers")
            if lower != max(rec_mii, res_mii):
                raise ValueError("lower bound differs from max(RecMII, ResMII)")
        features = row.get("full_features")
        if features is not None:
            if (not isinstance(features, list) or len(features) != 148 or
                    any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in features)):
                raise ValueError("row feature matrix has wrong width or non-finite values")
        if status == "success" and features is None:
            raise ValueError("successful native query has no feature vector")
        if row.get("feature_status") not in {"available", "bounds_outside_model_contract", "missing_bounds"}:
            raise ValueError("unknown feature availability status")
        if (features is None) != (row["feature_status"] != "available"):
            raise ValueError("feature availability marker disagrees with matrix")
        if (status == "censored" and lower is not None and lower <= 20 and
                features is None):
            raise ValueError("censored query with in-contract bounds lacks its input feature vector")
        if row["feature_status"] == "bounds_outside_model_contract" and (lower is None or lower <= 20):
            raise ValueError("out-of-contract feature marker lacks an out-of-range bound")
        group_families = group.get("source_program_families", [])
        row["group_families"] = group_families
        row["shape"] = shape
        if row.get("native_status") == "success":
            row["ii"] = float(label)
        if row["eligible_for_training"] and row["native_status"] == "success":
            if lower is None:
                raise ValueError("successful eligible row lacks analytical bounds")
        by_query.setdefault(identity, []).append(row)

    expected_shapes = set(SHAPE_PROTOCOL_2X2.mapper_shapes)
    for identity, roster in by_query.items():
        if len(roster) != 8 or {row["shape"] for row in roster} != expected_shapes:
            raise ValueError(f"query does not contain the complete shape roster: {identity}")

    split_summary = {}
    for split in ("train", "validation", "test", "excluded"):
        subset = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "query_rows": len(subset),
            "successful_rows": sum(row["native_status"] == "success" for row in subset),
            "censored_rows": sum(row["native_status"] == "censored" for row in subset),
            "dfg_count": len({row["query"] for row in subset}),
            "group_count": len({row["group"] for row in subset}),
        }
    if split_summary != manifest.get("counts", {}).get("split_summary"):
        raise ValueError("dataset split counts disagree with manifest")
    return {
        "manifest": manifest,
        "rows": rows,
        "rows_by_split": {
            split: [row for row in rows if row["split"] == split]
            for split in ("train", "validation", "test", "excluded")
        },
        "split_summary": split_summary,
    }
