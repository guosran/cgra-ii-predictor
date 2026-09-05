#!/usr/bin/env python3
"""Audit a motif-v8/v9 corpus without inventing censored numeric labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Dict, Iterable, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest artifact path is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("manifest artifact path escapes corpus") from error
    return path


def numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "median": None, "histogram": {}}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "histogram": {
            str(value): count for value, count in sorted(Counter(values).items())
        },
    }


def group_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    status = Counter(str(row.get("status")) for row in rows)
    compiled = [
        float(row["compiled_ii"]) for row in rows
        if row.get("status") == "success"
    ]
    lower_bounds = [
        float(row["lower_bound"]) for row in rows
        if isinstance(row.get("lower_bound"), (int, float)) and
        not isinstance(row.get("lower_bound"), bool)
    ]
    failures = Counter(
        (str(row.get("stage")), str(row.get("failure")))
        for row in rows if row.get("status") == "censored"
    )
    total = len(rows)
    return {
        "candidate_count": total,
        "status_counts": dict(sorted(status.items())),
        "success_rate": status["success"] / total if total else None,
        "censored_rate": status["censored"] / total if total else None,
        "compiled_ii": numeric_summary(compiled),
        "lower_bound": numeric_summary(lower_bounds),
        "censor_reasons": {
            f"{stage}:{failure}": count
            for (stage, failure), count in sorted(failures.items())
        },
    }


def build_report(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    corpus_schema = manifest.get("schema_version")
    if corpus_schema not in {
        "cgra-ii-motif-corpus-v8", "cgra-ii-motif-corpus-v9",
    }:
        raise ValueError("report requires a motif-v8 or motif-v9 manifest")
    protocol = manifest.get("shape_protocol")
    if not isinstance(protocol, Mapping) or protocol.get("protocol_id") != (
        "amoeba-static-rectangles-4x4-tiles-v1"
    ):
        raise ValueError("manifest shape protocol mismatch")
    records = manifest.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no candidates")
    if manifest.get("status") not in {"complete", "partial"} or any(
        not isinstance(record, Mapping) or
        record.get("status") not in {"success", "censored"}
        for record in records
    ):
        raise ValueError("motif-v8/v9 audit requires a fully terminal manifest")

    by_shape = defaultdict(list)
    by_family = defaultdict(list)
    override_checked = 0
    override_failures = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("candidate record must be an object")
        required_shape = (
            int(record["physical_cgra_rows"]),
            int(record["physical_cgra_cols"]),
            int(record["mapper_tile_rows"]),
            int(record["mapper_tile_cols"]),
        )
        if required_shape[2:] != (int(record["rows"]), int(record["columns"])):
            raise ValueError("candidate mapper aliases disagree")
        by_shape[f"{required_shape[2]}x{required_shape[3]}"].append(record)
        by_family[str(record["generator_family"])].append(record)
        if record.get("status") != "success":
            if record.get("compiled_ii") is not None:
                raise ValueError("non-success candidate has a numeric compiled II")
            continue
        compiled_ii = record.get("compiled_ii")
        lower_bound = record.get("lower_bound")
        if (
            isinstance(compiled_ii, bool) or
            not isinstance(compiled_ii, (int, float)) or
            not math.isfinite(float(compiled_ii)) or
            float(compiled_ii) < float(lower_bound)
        ):
            raise ValueError("successful candidate has an invalid compiled II")
        mapped = safe_path(root, record.get("mapped_artifact_path"))
        if sha256_file(mapped) != record.get("mapped_artifact_sha256"):
            raise ValueError("mapped artifact SHA-256 mismatch")
        text = mapped.read_text()
        x_match = re.search(r"\bx_tiles\s*=\s*(\d+)\s*:\s*i32", text)
        y_match = re.search(r"\by_tiles\s*=\s*(\d+)\s*:\s*i32", text)
        override_checked += 1
        if (
            x_match is None or y_match is None or
            int(x_match.group(1)) != required_shape[3] or
            int(y_match.group(1)) != required_shape[2]
        ):
            override_failures.append(str(record["candidate_id"]))
            continue
        tile_locations = []
        for location in re.findall(r"\{[^{}]*\}", text):
            if 'resource = "tile"' not in location:
                continue
            tile_x = re.search(r"\bx = (\d+) : i32", location)
            tile_y = re.search(r"\by = (\d+) : i32", location)
            if tile_x is not None and tile_y is not None:
                tile_locations.append((int(tile_x.group(1)), int(tile_y.group(1))))
        if not tile_locations or any(
            not (0 <= x < required_shape[3] and 0 <= y < required_shape[2])
            for x, y in tile_locations
        ):
            override_failures.append(str(record["candidate_id"]))

    query_shapes = defaultdict(set)
    for record in records:
        query_shapes[str(record["ranking_query_id"])].add((
            int(record["mapper_tile_rows"]), int(record["mapper_tile_cols"]),
        ))
    expected_shapes = {
        (int(item["mapper_tile_rows"]), int(item["mapper_tile_cols"]))
        for item in protocol["physical_to_mapper"]
    }
    split_safe = all(shapes == expected_shapes for shapes in query_shapes.values())
    snapshot = root / "corpus-manifest.predeclared.json"
    architecture = manifest.get("architecture", {})
    collection = manifest.get("collection", {})
    return {
        "schema_version": (
            "cgra-ii-motif-v8-corpus-audit-v1"
            if corpus_schema == "cgra-ii-motif-corpus-v8" else
            "cgra-ii-motif-v9-corpus-audit-v1"
        ),
        **(
            {"corpus_schema_version": corpus_schema}
            if corpus_schema == "cgra-ii-motif-corpus-v9" else {}
        ),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "status": manifest.get("status"),
            "predeclaration_snapshot_sha256": (
                sha256_file(snapshot) if snapshot.is_file() else None
            ),
        },
        "protocol": dict(protocol),
        "provenance": {
            "generator": manifest.get("generator"),
            "architecture": architecture,
            "collection": collection,
        },
        "dfg_query_count": len(query_shapes),
        "all_queries_have_complete_shape_domain": split_safe,
        "split_unit": "ranking_query_id/canonical_dfg_sha256",
        "overall": group_summary(records),
        "by_mapper_tile_shape": {
            shape: group_summary(group)
            for shape, group in sorted(by_shape.items())
        },
        "by_generator_family": {
            family: group_summary(group)
            for family, group in sorted(by_family.items())
        },
        "mapper_override_verification": {
            "successful_artifact_count": override_checked,
            "verified_count": override_checked - len(override_failures),
            "failure_count": len(override_failures),
            "failure_candidate_ids": override_failures,
            "all_successful_artifacts_match_explicit_x_y": not override_failures,
        },
        "censored_numeric_ii_imputation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "manifest": report["manifest"],
        "overall": report["overall"],
        "mapper_override_verification": report["mapper_override_verification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
