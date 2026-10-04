#!/usr/bin/env python3
"""Collect fresh native mapper outcomes for Amoeba's 2x2-tile CGRAs.

The source manifest is used only as a roster of distinct DFG inputs and for
non-label lineage metadata.  Old compiled-II and mapping-outcome fields are
never copied to this collection.  The frozen query bundle contains the DFGs
and architecture file, so its paths remain valid when the whole output root
is copied to a collection host.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable, Optional

PROJECT = Path(__file__).resolve().parents[1]
for directory in (PROJECT / "adapters", PROJECT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import collect_kernelbench_mappings as native  # noqa: E402
import mapping_artifact_protocol as artifact_protocol  # noqa: E402
from mapping_artifact_protocol import (  # noqa: E402
    canonical_json_sha256, mapper_input_identity, sha256_file,
)
from cgra_ii_predictor import shape_protocol as shape_protocol_module  # noqa: E402
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    SHAPE_PROTOCOL_2X2,
    SHAPE_PROTOCOL_2X2_ID,
)


SCHEMA = "cgra-ii-per-cgra-2x2-native-collection-v1"
SHAPE_PROTOCOL_ID = SHAPE_PROTOCOL_2X2_ID
MAPPER_SHAPES = SHAPE_PROTOCOL_2X2.mapper_shapes
EXPECTED_ARCHITECTURE = {
    "multi_cgra_rows": 4,
    "multi_cgra_columns": 4,
    "per_cgra_tile_rows": 2,
    "per_cgra_tile_columns": 2,
    "total_tile_count": 64,
}
SOURCE_SCHEMA = "cgra-ii-nine-shape-outcomes-v1"
OUTCOMES_SCHEMA = "cgra-ii-per-cgra-2x2-outcomes-v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"frozen input changed: {destination}")
        return
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"copied input hash mismatch: {source}")
    temporary.replace(destination)


def _safe_source_path(manifest_path: Path, source_path: str) -> Path:
    relative = Path(source_path)
    if relative.is_absolute():
        raise ValueError(f"source DFG path must be relative: {source_path}")
    root = manifest_path.resolve().parent
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source DFG escapes manifest directory: {source_path}") from error
    return source


def _load_source_groups(manifest_path: Path) -> tuple[dict, list[dict]]:
    """Load unique DFGs and metadata, deliberately ignoring prior labels."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source manifest schema: {manifest.get('schema')!r}")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("source manifest has no candidates")
    if manifest.get("candidate_count") != len(candidates):
        raise ValueError("source candidate_count disagrees with candidate roster")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        identity = row.get("mapper_input_identity")
        source_path = row.get("source_path")
        source_sha256 = row.get("source_sha256")
        if (not isinstance(identity, str) or len(identity) != 64 or
                not isinstance(source_path, str) or
                not isinstance(source_sha256, str) or len(source_sha256) != 64):
            raise ValueError("source candidate lacks a valid DFG identity/path/hash")
        grouped[identity].append(row)

    if manifest.get("query_count") not in (None, len(grouped)):
        raise ValueError("source query_count disagrees with distinct DFG identities")

    groups = []
    for identity, members in sorted(grouped.items()):
        candidates_by_key = {
            (row["source_path"], row["source_sha256"]) for row in members
        }
        if len(candidates_by_key) != 1:
            raise ValueError(f"source identity points to multiple DFG files: {identity}")
        source_path_text, source_sha256 = next(iter(candidates_by_key))
        source_path = _safe_source_path(manifest_path, source_path_text)
        if not source_path.is_file() or sha256_file(source_path) != source_sha256:
            raise ValueError(f"source DFG hash mismatch: {source_path}")
        text = source_path.read_text()
        if mapper_input_identity(text, normalize_static_shapes=True) != identity:
            raise ValueError(f"source DFG mapper identity mismatch: {source_path}")

        graph_ids = sorted({
            row["model_visible_graph_identity"]
            for row in members
            if isinstance(row.get("model_visible_graph_identity"), str)
        })
        if len(graph_ids) != 1:
            raise ValueError(f"source identity has ambiguous visible-graph lineage: {identity}")
        families = sorted({
            family for row in members
            for family in row.get("source_program_families", [])
            if isinstance(family, str) and family
        })
        source_names = sorted({
            name for row in members for name in row.get("source_names", [])
            if isinstance(name, str) and name
        })
        domains = sorted({
            row["domain"] for row in members
            if isinstance(row.get("domain"), str) and row["domain"]
        })
        lineage_ids = sorted({
            row.get("leakage_lineage_id", identity) for row in members
            if isinstance(row.get("leakage_lineage_id", identity), str)
        })
        candidate_ids = sorted({
            row["candidate_id"] for row in members
            if isinstance(row.get("candidate_id"), str)
        })
        # No old status, outcome_label, compiled_ii, or II feature is retained.
        groups.append({
            "mapper_input_identity": identity,
            "model_visible_graph_identity": graph_ids[0],
            "dfg_sha256": source_sha256,
            "source_path": source_path,
            "source_manifest_path": source_path_text,
            "source_candidate_ids": candidate_ids,
            "source_program_families": families,
            "source_names": source_names,
            "domains": domains,
            "leakage_lineage_ids": lineage_ids,
        })

    return manifest, groups


def _architecture_facts(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - installation diagnostic
        raise RuntimeError("PyYAML is required to validate the architecture spec") from error
    architecture = yaml.safe_load(path.read_text())
    if not isinstance(architecture, dict):
        raise ValueError("architecture spec is not a YAML mapping")
    multi = architecture.get("multi_cgra_defaults", {})
    per_cgra = architecture.get("per_cgra_defaults", {})
    facts = {
        "multi_cgra_rows": multi.get("rows"),
        "multi_cgra_columns": multi.get("columns"),
        "per_cgra_tile_rows": per_cgra.get("rows"),
        "per_cgra_tile_columns": per_cgra.get("columns"),
    }
    for name, expected in EXPECTED_ARCHITECTURE.items():
        if name == "total_tile_count":
            actual = (facts["multi_cgra_rows"] * facts["multi_cgra_columns"] *
                      facts["per_cgra_tile_rows"] * facts["per_cgra_tile_columns"])
        else:
            actual = facts[name]
        if actual != expected:
            raise ValueError(
                f"architecture {name} is {actual!r}, expected {expected!r}"
            )
    facts["total_tile_count"] = 64
    return facts


def _queries(groups: list[dict]) -> list[dict]:
    queries = []
    # Start with the smallest graphs for early throughput, using a stable
    # identity tie-break. Keep protocol orientation order within each graph.
    ordered_groups = sorted(groups, key=lambda group: (
        group["source_path"].stat().st_size,
        group["mapper_input_identity"],
    ))
    for group in ordered_groups:
        for rows, cols in MAPPER_SHAPES:
            queries.append({
                "query_id": f"{group['mapper_input_identity']}/{rows}x{cols}",
                "shape_protocol_id": SHAPE_PROTOCOL_ID,
                "mapper_input_identity": group["mapper_input_identity"],
                "model_visible_graph_identity": group[
                    "model_visible_graph_identity"],
                "dfg_path": f"inputs/dfg/{group['mapper_input_identity']}.mlir",
                "dfg_sha256": group["dfg_sha256"],
                "source_manifest_path": group["source_manifest_path"],
                "source_candidate_ids": group["source_candidate_ids"],
                "source_program_families": group["source_program_families"],
                "source_cases": group["source_program_families"],
                "source_names": group["source_names"],
                "domains": group["domains"],
                "leakage_lineage_ids": group["leakage_lineage_ids"],
                "rows": rows,
                "cols": cols,
                "physical_cgra_rows": rows // 2,
                "physical_cgra_columns": cols // 2,
            })
    # The group order, protocol shape order, and ties are all fixed.
    return queries


def _directory(root: Path, query: dict) -> Path:
    return (root / "artifacts" / query["mapper_input_identity"] /
            f"{query['rows']}x{query['cols']}")


def _decorate_result(result: dict, query: dict, provenance: dict,
                     root: Path) -> dict:
    result.update({
        "schema": SCHEMA,
        "query_id": query["query_id"],
        "shape_protocol_id": SHAPE_PROTOCOL_ID,
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "architecture_sha256": provenance["architecture_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "collection_provenance_sha256": provenance["provenance_sha256"],
        "dfg_path": query["dfg_path"],
        "mapped_artifact_path": str(
            Path("artifacts") / query["mapper_input_identity"] /
            f"{query['rows']}x{query['cols']}" / "mapped.mlir"
        ),
        "physical_cgra_rows": query["physical_cgra_rows"],
        "physical_cgra_columns": query["physical_cgra_columns"],
        "source_candidate_ids": query["source_candidate_ids"],
        "source_program_families": query["source_program_families"],
        "leakage_lineage_ids": query["leakage_lineage_ids"],
    })
    result["mapper_command"] = [
        "<neura-opt>",
        query["dfg_path"],
        "--architecture-spec=inputs/architecture.yaml",
        ("--map-to-accelerator=mapping-strategy=heuristic "
         f"x-tiles={query['cols']} y-tiles={query['rows']}"),
        "-o",
        str(Path("artifacts") / query["mapper_input_identity"] /
            f"{query['rows']}x{query['cols']}" / "mapped.mlir"),
    ]
    return result


def _map_one(query: dict, root: Path, neura_opt: Path, architecture: Path,
             timeout: Optional[int], provenance: dict) -> dict:
    execution_query = dict(query)
    execution_query["dfg_path"] = str((root / query["dfg_path"]).resolve())
    execution_query["source_cases"] = query["source_program_families"]
    # The shared native mapper helper names these physical-core dimensions.
    execution_query["physical_rows"] = query["physical_cgra_rows"]
    execution_query["physical_cols"] = query["physical_cgra_columns"]
    result = native._map(execution_query, root, neura_opt, architecture, timeout)
    result = _decorate_result(result, query, provenance, root)
    native._write_json(_directory(root, query) / "result.json", result)
    return result


def _verify_result(path: Path, root: Path, query: dict,
                   provenance: dict) -> Optional[dict]:
    if not path.is_file():
        return None
    result = native._verify_result(path, query)
    if result is None:
        return None
    expected = {
        "schema": SCHEMA,
        "query_id": query["query_id"],
        "shape_protocol_id": SHAPE_PROTOCOL_ID,
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "architecture_sha256": provenance["architecture_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "collection_provenance_sha256": provenance["provenance_sha256"],
        "physical_cgra_rows": query["physical_cgra_rows"],
        "physical_cgra_columns": query["physical_cgra_columns"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"stored result {key} mismatch: {path}")
    expected_artifact = str(Path("artifacts") /
                            query["mapper_input_identity"] /
                            f"{query['rows']}x{query['cols']}" / "mapped.mlir")
    if result.get("mapped_artifact_path") != expected_artifact:
        raise ValueError(f"stored artifact path mismatch: {path}")
    # Numeric labels exist only for parser-verified successful mappings.
    if result.get("status") == "success":
        if not isinstance(result.get("compiled_ii"), int):
            raise ValueError(f"successful mapping lacks a numeric II: {path}")
    elif result.get("compiled_ii") is not None:
        raise ValueError(f"unsuccessful mapping carries a numeric II: {path}")
    return result


def _outcomes(root: Path, queries: list[dict], provenance: dict) -> dict:
    rows = []
    for query in queries:
        result_path = _directory(root, query) / "result.json"
        result = _verify_result(result_path, root, query, provenance)
        record = {
            "query_id": query["query_id"],
            "mapper_input_identity": query["mapper_input_identity"],
            "model_visible_graph_identity": query[
                "model_visible_graph_identity"],
            "dfg_sha256": query["dfg_sha256"],
            "dfg_path": query["dfg_path"],
            "rows": query["rows"],
            "cols": query["cols"],
            "physical_cgra_rows": query["physical_cgra_rows"],
            "physical_cgra_columns": query["physical_cgra_columns"],
            "shape_protocol_id": SHAPE_PROTOCOL_ID,
            "source_candidate_ids": query["source_candidate_ids"],
            "source_program_families": query["source_program_families"],
            "source_names": query["source_names"],
            "domains": query["domains"],
            "leakage_lineage_ids": query["leakage_lineage_ids"],
            "status": result["status"] if result else "pending",
            "compiled_ii": result.get("compiled_ii") if result else None,
        }
        if result:
            record["result_path"] = str(
                Path("artifacts") / query["mapper_input_identity"] /
                f"{query['rows']}x{query['cols']}" / "result.json"
            )
            record["result_sha256"] = sha256_file(result_path)
            record["censor_reason"] = result.get("censor_reason")
        rows.append(record)
    manifest = {
        "schema": OUTCOMES_SCHEMA,
        "collection_provenance_sha256": provenance["provenance_sha256"],
        "query_manifest_sha256": sha256_file(root / "query-manifest.json"),
        "query_count": len(rows),
        "terminal_count": sum(row["status"] != "pending" for row in rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "censored_count": sum(row["status"] == "censored" for row in rows),
        "queries": rows,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _parse_shapes(values: Iterable[str]) -> set[tuple[int, int]]:
    result = set()
    for value in values:
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            raise ValueError(f"invalid shape {value!r}; expected ROWSxCOLS")
        result.add((int(parts[0]), int(parts[1])))
    unknown = result - set(MAPPER_SHAPES)
    if unknown:
        raise ValueError(f"shape(s) outside the frozen 2x2 protocol: {sorted(unknown)}")
    return result


def _implementation_hashes() -> dict[str, str]:
    files = {
        "adapters/collect_per_cgra_2x2_mappings.py": Path(__file__).resolve(),
        "adapters/collect_kernelbench_mappings.py": Path(native.__file__).resolve(),
        "adapters/mapping_artifact_protocol.py": Path(
            artifact_protocol.__file__).resolve(),
        "src/cgra_ii_predictor/shape_protocol.py": Path(
            shape_protocol_module.__file__).resolve(),
    }
    hashes = {}
    for relative_path, path in files.items():
        try:
            path.relative_to(PROJECT.resolve())
        except ValueError as error:
            raise ValueError(f"implementation module is outside project root: {path}") from error
        hashes[relative_path] = sha256_file(path)
    return hashes


def _has_unmanifested_results(root: Path) -> bool:
    return any((root / "artifacts").glob("*/*/result.json"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--neura-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--shapes", action="append", default=[],
                        help="limit this invocation to a frozen shape, e.g. 2x2; repeatable")
    parser.add_argument("--max-new-queries", type=int,
                        help="run at most this many new queries, for a bounded smoke batch")
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="per-stage timeout; 0 leaves native mapper search unbounded")
    parser.add_argument("--prepare-only", action="store_true",
                        help="freeze the manifest and portable inputs without running the mapper")
    parser.add_argument("--acknowledge-stale-running", action="store_true",
                        help="resume after checking that old running mapper processes are gone")
    args = parser.parse_args(argv)
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.max_new_queries is not None and args.max_new_queries <= 0:
        parser.error("--max-new-queries must be positive")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be nonnegative")
    if not args.neura_opt.is_file():
        parser.error(f"--neura-opt does not exist: {args.neura_opt}")
    if not args.architecture.is_file():
        parser.error(f"--architecture does not exist: {args.architecture}")

    source_path = args.source_manifest.resolve()
    architecture_source = args.architecture.resolve()
    mapper_binary = args.neura_opt.resolve()
    source_manifest, groups = _load_source_groups(source_path)
    architecture_facts = _architecture_facts(architecture_source)
    queries = _queries(groups)
    shapes = _parse_shapes(args.shapes)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    lock_stream = native._acquire_collection_lock(
        root, dry_run=False,
        acknowledge_stale_running=args.acknowledge_stale_running,
    )
    try:
        provenance = {
            "schema": SCHEMA,
            "shape_protocol_id": SHAPE_PROTOCOL_ID,
            "mapper_shapes": [list(shape) for shape in MAPPER_SHAPES],
            "source_manifest_schema": SOURCE_SCHEMA,
            "source_manifest_sha256": sha256_file(source_path),
            "source_dfg_count": len(groups),
            "source_candidate_count": source_manifest["candidate_count"],
            "architecture_sha256": sha256_file(architecture_source),
            "architecture_facts": architecture_facts,
            "neura_opt_sha256": sha256_file(mapper_binary),
            "mapper_strategy": "heuristic",
            "implementation_sha256_by_path": _implementation_hashes(),
            "external_timeout_seconds": (
                None if args.timeout_seconds == 0 else args.timeout_seconds
            ),
            "collection_worker_count": args.jobs,
            "expected_query_count": len(queries),
            "old_native_labels_reused": False,
            "source_lineage_policy": "preserve identities, families, candidate IDs, and leakage lineage only",
        }
        provenance["provenance_sha256"] = canonical_json_sha256(provenance)
        query_manifest = {
            "schema": SCHEMA,
            "provenance": provenance,
            "query_count": len(queries),
            "queries": queries,
        }
        query_manifest["manifest_sha256"] = canonical_json_sha256(query_manifest)

        provenance_path = root / "provenance.json"
        query_manifest_path = root / "query-manifest.json"
        prior_provenance = (json.loads(provenance_path.read_text())
                            if provenance_path.exists() else None)
        prior_manifest = (json.loads(query_manifest_path.read_text())
                          if query_manifest_path.exists() else None)
        if prior_provenance is None and _has_unmanifested_results(root):
            raise ValueError("output root has mapping results but no matching frozen provenance")
        if prior_provenance is not None and prior_provenance != provenance:
            raise ValueError("collection provenance changed; refusing to mix native runs")
        if prior_manifest is not None and prior_manifest != query_manifest:
            raise ValueError("frozen query manifest changed; refusing to mix native runs")

        architecture_bundle = root / "inputs" / "architecture.yaml"
        _copy_verified(architecture_source, architecture_bundle,
                       provenance["architecture_sha256"])
        for group in groups:
            destination = root / "inputs" / "dfg" / (
                group["mapper_input_identity"] + ".mlir"
            )
            _copy_verified(group["source_path"], destination, group["dfg_sha256"])
        if prior_provenance is None:
            _write_json(provenance_path, provenance)
        if prior_manifest is None:
            _write_json(query_manifest_path, query_manifest)

        if args.prepare_only:
            _write_json(root / "outcomes.json", _outcomes(root, queries, provenance))
            print(json.dumps({
                "event": "collection_prepared",
                "query_manifest": str(query_manifest_path),
                "source_dfg_count": len(groups),
                "expected_query_count": len(queries),
                "shape_protocol_id": SHAPE_PROTOCOL_ID,
                "worker_count": args.jobs,
                "external_timeout_seconds": provenance[
                    "external_timeout_seconds"],
            }, sort_keys=True))
            return 0

        if (root / "pause.requested").exists():
            raise RuntimeError(f"collection paused by {root / 'pause.requested'}")
        native._wait_for_old_mappers(root)
        native.SCHEMA = SCHEMA
        completed = {}
        pending = []
        for query in queries:
            result = _verify_result(
                _directory(root, query) / "result.json", root, query, provenance)
            key = (query["mapper_input_identity"], query["rows"], query["cols"])
            if result is None:
                if not shapes or (query["rows"], query["cols"]) in shapes:
                    pending.append(query)
            else:
                completed[key] = result
        if args.max_new_queries is not None:
            pending = pending[:args.max_new_queries]

        native._progress(root, completed, len(queries), 0)

        def map_one(query: dict, map_root: Path, opt: Path,
                    architecture: Path, timeout: Optional[int]) -> dict:
            return _map_one(query, map_root, opt, architecture, timeout, provenance)

        native._collect_pending(
            root, pending, completed, len(queries), args.jobs, mapper_binary,
            architecture_bundle,
            provenance["external_timeout_seconds"], map_one=map_one,
        )
        outcomes = _outcomes(root, queries, provenance)
        _write_json(root / "outcomes.json", outcomes)
        if outcomes["terminal_count"] == len(queries):
            complete = {
                "schema": SCHEMA,
                "query_manifest_sha256": sha256_file(query_manifest_path),
                "outcomes_sha256": sha256_file(root / "outcomes.json"),
                "query_count": len(queries),
                "success_count": outcomes["success_count"],
                "censored_count": outcomes["censored_count"],
            }
            _write_json(root / "collection-complete.json", complete)
            print(json.dumps({"event": "collection_complete", **complete},
                             sort_keys=True), flush=True)
        else:
            print(json.dumps({
                "event": "collection_partial",
                "query_count": len(queries),
                "terminal_count": outcomes["terminal_count"],
                "success_count": outcomes["success_count"],
                "censored_count": outcomes["censored_count"],
            }, sort_keys=True), flush=True)
        return 0
    finally:
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
