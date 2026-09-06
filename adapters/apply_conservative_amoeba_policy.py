#!/usr/bin/env python3
"""Attach a validation-selected upper-II policy to an Amoeba cost catalogue.

The conditional mean remains available as ``point_predicted_ii``.  The
catalogue keeps its point estimate as the default scheduling ``predicted_ii``;
the conservative value selected without application labels is exported as
``predicted_ii_upper`` for mapper-replay risk gating:

``max(point_mean, expert_mean + alpha * expert_std)``.

``--use-upper-for-scoring`` retains the earlier diagnostic mode that replaces
the scheduling value, but must be requested explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

try:
    from .amoeba_protocol import COST_SCHEMA
except ImportError:  # Direct script execution.
    from amoeba_protocol import COST_SCHEMA

POLICY_SCHEMA = "cgra-ii-conservative-pointwise"
OUTPUT_POLICY = "validation-selected-conservative-upper"
Key = Tuple[str, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load_catalog(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != COST_SCHEMA:
        raise ValueError("input has an unsupported cost catalogue schema")
    if not isinstance(raw.get("entries"), list):
        raise ValueError("cost catalogue lacks entries")
    return raw


def _entry_key(row: Mapping[str, Any]) -> Key:
    return (
        str(row["task"]), int(row["mapper_tile_rows"]),
        int(row["mapper_tile_cols"]),
    )


def _checkpoint_hashes(catalog: Mapping[str, Any]) -> set[str]:
    metadata = catalog.get("predictor_metadata")
    checkpoints = metadata.get("checkpoints") if isinstance(metadata, Mapping) else None
    if not isinstance(checkpoints, Mapping):
        raise ValueError("cost catalogue lacks checkpoint provenance")
    result = set()
    for record in checkpoints.values():
        if not isinstance(record, Mapping) or not isinstance(
            record.get("sha256"), str
        ):
            raise ValueError("cost catalogue has invalid checkpoint provenance")
        result.add(str(record["sha256"]))
    return result


def apply_policy(
    point_path: Path, expert_path: Path, policy_path: Path,
    use_upper_for_scoring: bool = False,
) -> Dict[str, Any]:
    point = _load_catalog(point_path)
    expert = _load_catalog(expert_path)
    policy = json.loads(policy_path.read_text())
    if not isinstance(policy, Mapping) or policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("conservative policy schema_version mismatch")
    if policy.get("selection_split") != "validation_only":
        raise ValueError("conservative policy was not selected on validation only")
    if policy.get("policy") != (
        "max_ensemble_mean_and_expert_mean_plus_alpha_std"
    ):
        raise ValueError("unsupported conservative policy")
    alpha = float(policy.get("selected_alpha"))
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("conservative alpha must be finite and nonnegative")
    quantile = float(policy.get("quantile"))
    if not math.isfinite(quantile) or not 0.5 < quantile < 1.0:
        raise ValueError("conservative quantile must be between 0.5 and 1")
    expert_record = policy.get("expert")
    if not isinstance(expert_record, Mapping):
        raise ValueError("conservative policy lacks expert provenance")
    expert_sha = str(expert_record.get("sha256", ""))
    if _checkpoint_hashes(expert) != {expert_sha}:
        raise ValueError("expert catalogue checkpoint does not match policy")
    policy_checkpoints = policy.get("checkpoints")
    if not isinstance(policy_checkpoints, Mapping):
        raise ValueError("conservative policy lacks checkpoint provenance")
    policy_checkpoint_hashes = {
        str(record["sha256"])
        for record in policy_checkpoints.values()
        if isinstance(record, Mapping) and isinstance(record.get("sha256"), str)
    }
    point_hashes = _checkpoint_hashes(point)
    if not point_hashes or not point_hashes.issubset(policy_checkpoint_hashes):
        raise ValueError("point catalogue checkpoints do not match policy")
    if point.get("function") != expert.get("function"):
        raise ValueError("point and expert catalogues describe different functions")

    expert_entries = {}
    for row in expert["entries"]:
        if not isinstance(row, Mapping):
            raise ValueError("expert cost entry must be an object")
        key = _entry_key(row)
        if key in expert_entries:
            raise ValueError("duplicate expert cost entry")
        expert_entries[key] = row
    output_entries = []
    seen = set()
    for raw in point["entries"]:
        if not isinstance(raw, Mapping):
            raise ValueError("point cost entry must be an object")
        row = dict(raw)
        key = _entry_key(row)
        if key in seen:
            raise ValueError("duplicate point cost entry")
        seen.add(key)
        other = expert_entries.get(key)
        if other is None or other.get("support_status") != row.get("support_status"):
            raise ValueError("point and expert catalogue entries do not align")
        if row.get("support_status") == "supported":
            point_mean = float(row["predicted_ii"])
            point_std = float(row["predicted_ii_std"])
            expert_mean = float(other["predicted_ii"])
            expert_std = float(other["predicted_ii_std"])
            values = (point_mean, point_std, expert_mean, expert_std)
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("cost catalogue has invalid prediction values")
            upper = max(point_mean, expert_mean + alpha * expert_std)
            row.update({
                "point_predicted_ii": point_mean,
                "point_predicted_ii_std": point_std,
                "conservative_expert_predicted_ii": expert_mean,
                "conservative_expert_predicted_ii_std": expert_std,
                "conservative_alpha": alpha,
                "predicted_ii_upper": upper,
            })
            if use_upper_for_scoring:
                row.update({
                    "predicted_ii": upper,
                    "ii_mean_source": OUTPUT_POLICY,
                })
        output_entries.append(row)
    if set(expert_entries) != seen:
        raise ValueError("expert catalogue contains unmatched entries")

    policy_metadata = {
        "kind": OUTPUT_POLICY,
        "formula": "max(point_mean, expert_mean + alpha * expert_std)",
        "role": (
            "program_scoring_and_mapper_replay_trigger"
            if use_upper_for_scoring else "mapper_replay_trigger_only"
        ),
        "upper_applied_to_predicted_ii": use_upper_for_scoring,
        "point_estimate_preserved_as": "point_predicted_ii",
        "selection_split": "validation_only",
        "quantile": quantile,
        "alpha": alpha,
        "policy_report_sha256": sha256_file(policy_path),
        "point_catalog_sha256": sha256_file(point_path),
        "expert_catalog_sha256": sha256_file(expert_path),
        "expert_checkpoint_sha256": expert_sha,
    }
    output = dict(point)
    output["predictor_metadata"] = dict(point["predictor_metadata"])
    output["predictor_metadata"]["conservative_policy"] = policy_metadata
    output["namespace"] = "cgra-ii-upper-" + canonical_sha256(
        policy_metadata
    )[:24]
    output["entries"] = output_entries
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-catalog", required=True, type=Path)
    parser.add_argument("--expert-catalog", required=True, type=Path)
    parser.add_argument("--policy-report", required=True, type=Path)
    parser.add_argument(
        "--use-upper-for-scoring", action="store_true",
        help=("Replace predicted_ii with the conservative upper value; the "
              "default keeps point-model shape ordering."),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = apply_policy(
        args.point_catalog.resolve(), args.expert_catalog.resolve(),
        args.policy_report.resolve(), args.use_upper_for_scoring,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "namespace": output["namespace"],
        "policy": output["predictor_metadata"]["conservative_policy"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
