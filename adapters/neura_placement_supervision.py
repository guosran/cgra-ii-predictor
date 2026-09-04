#!/usr/bin/env python3
"""Extract compact operation-to-PE supervision from existing mapper outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cgra_ii_predictor.graph_model import (  # noqa: E402
    GraphData, parse_neura_dfg, parse_neura_mapped_placements,
)


SCHEMA_VERSION = "cgra-ii-placement-supervision-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("candidate source_path is missing")
    root = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("candidate source_path escapes its corpus") from error
    return path


def extract_placement_supervision(
    manifest_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Build deterministic compact labels from terminal successful mappings."""
    if not manifest_paths:
        raise ValueError("at least one source manifest is required")
    placements: Dict[str, list] = {}
    source_graphs: Dict[str, GraphData] = {}
    source_manifests = []
    query_ids = set()
    censored_candidates = 0
    for manifest_path in manifest_paths:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text())
        records = manifest.get("candidates") if isinstance(manifest, Mapping) else None
        if not isinstance(records, list) or not records:
            raise ValueError("source manifest contains no candidates")
        source_manifests.append({
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "candidate_count": len(records),
        })
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("manifest candidate must be an object")
            status = record.get("status")
            if status == "censored":
                censored_candidates += 1
                continue
            if status != "success":
                raise ValueError("placement source manifest is not terminal")
            candidate_id = record.get("candidate_id", record.get("id"))
            query_id = record.get("ranking_query_id")
            source_sha = record.get("source_sha256")
            if not all(isinstance(value, str) and value for value in (
                candidate_id, query_id, source_sha,
            )):
                raise ValueError("candidate lacks placement provenance")
            if candidate_id in placements:
                raise ValueError("duplicate placement candidate identity")
            query_ids.add(query_id)
            source_path = safe_relative(
                manifest_path.parent, record.get("source_path"),
            )
            graph = source_graphs.get(source_sha)
            if graph is None:
                if sha256_file(source_path) != source_sha:
                    raise ValueError("candidate source hash mismatch")
                graph = parse_neura_dfg(source_path.read_text())
                source_graphs[source_sha] = graph
            mapped_path = source_path.parent / "mapped.mlir"
            if not mapped_path.is_file():
                raise ValueError(
                    "successful candidate lacks mapped.mlir: " + candidate_id
                )
            placement = parse_neura_mapped_placements(
                mapped_path.read_text(), graph,
                int(record["rows"]), int(record["columns"]),
            )
            placements[candidate_id] = list(placement)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifests": source_manifests,
        "query_count": len(query_ids),
        "successful_candidate_count": len(placements),
        "censored_candidate_count": censored_candidates,
        "placements": dict(sorted(placements.items())),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(payload, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = extract_placement_supervision(args.manifest)
    atomic_write_json(args.output, payload)
    print(
        f"placement_supervision={args.output.resolve()} "
        f"queries={payload['query_count']} "
        f"successful_candidates={payload['successful_candidate_count']} "
        f"sha256={sha256_file(args.output)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
