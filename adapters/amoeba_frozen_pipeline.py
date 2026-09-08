#!/usr/bin/env python3
"""Run a frozen-candidate ML-to-mapper DSE loop without Amoeba replay.

The pipeline scores every candidate from an existing task/shape cost catalogue,
selects a top-k shortlist, and invokes Neura's heuristic mapper exactly once for
each unique shortlisted ``(task, mapper rows, mapper cols)`` query.  Candidate
shapes are passed directly as ``x-tiles``/``y-tiles`` overrides and verified in
the resulting mapper label and physical placement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amoeba_cost_catalog import (  # noqa: E402
    COST_SCHEMA,
    load_candidate_manifest,
    sha256_file,
    source_task_body_sha256,
)
from amoeba_protocol import (  # noqa: E402
    CANDIDATE_SCHEMA,
    SCORE_MODEL,
    SCORE_SCHEMA,
)
from build_amoeba_query_oracle import parse_single_mapping  # noqa: E402
from evaluate_amoeba_scores import load_oracle  # noqa: E402


PIPELINE_SCHEMA = "cgra-ii-independent-frozen-pipeline"
QueryKey = Tuple[str, int, int]
Runner = Callable[..., subprocess.CompletedProcess]


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def load_cost_catalog(
    path: Path, manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Load a provenance-bound catalogue for the manifest's used queries."""
    root = _object(json.loads(path.read_text()), "cost catalogue")
    header = _object(manifest["header"], "candidate header")
    if root.get("schema") != COST_SCHEMA:
        raise ValueError("cost catalogue schema mismatch")
    if root.get("function") != header.get("function"):
        raise ValueError("cost catalogue function mismatch")
    namespace = root.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("cost catalogue namespace is missing")
    metadata = _object(root.get("predictor_metadata"), "predictor metadata")
    if metadata.get("candidate_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("cost catalogue candidate manifest SHA-256 mismatch")
    ranking_policy = _object(metadata.get("ranking_policy"), "ranking policy")
    if (
        ranking_policy.get("mapper_success_probability") != "diagnostic_only" or
        ranking_policy.get("uses_mapper_success_probability") is not False
    ):
        raise ValueError(
            "cost catalogue must keep mapper success probability diagnostic-only"
        )
    analytical_provenance = _object(
        metadata.get("analytical_provenance"), "analytical provenance",
    )
    task_body_hashes = _object(
        analytical_provenance.get("task_body_sha256"), "task body provenance",
    )
    if task_body_hashes != manifest["task_body_sha256"]:
        raise ValueError("cost catalogue task body provenance mismatch")
    architecture_contract = _object(
        metadata.get("architecture_contract"), "architecture contract",
    )
    architecture_sha = analytical_provenance.get("architecture_sha256")
    supported_architectures = architecture_contract.get(
        "supported_architecture_sha256"
    )
    if (
        not isinstance(supported_architectures, list) or
        architecture_sha not in supported_architectures
    ):
        raise ValueError(
            "cost catalogue architecture is outside the model contract"
        )

    entries = root.get("entries")
    if not isinstance(entries, list):
        raise ValueError("cost catalogue entries must be an array")
    by_query: Dict[QueryKey, Dict[str, Any]] = {}
    for raw in entries:
        entry = _object(raw, "cost entry")
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("cost entry task is missing")
        rows = _positive_integer(entry.get("mapper_tile_rows"), "mapper_tile_rows")
        cols = _positive_integer(entry.get("mapper_tile_cols"), "mapper_tile_cols")
        key = (task, rows, cols)
        if key in by_query:
            raise ValueError("duplicate task/mapper-shape cost entry")
        status = entry.get("support_status")
        normalized = dict(entry)
        if status == "supported":
            normalized["predicted_ii"] = _positive_number(
                entry.get("predicted_ii"), "predicted_ii",
            )
            normalized["startup_cycles"] = _positive_number(
                entry.get("startup_cycles"), "startup_cycles",
            )
            lower_bound = entry.get("analytical_lower_bound")
            if isinstance(lower_bound, bool) or not isinstance(
                lower_bound, (int, float),
            ):
                raise ValueError(
                    "analytical_lower_bound must be a positive finite number"
                )
            lower_bound = float(lower_bound)
            if not math.isfinite(lower_bound) or lower_bound <= 0.0:
                raise ValueError(
                    "analytical_lower_bound must be a positive finite number"
                )
            if normalized["predicted_ii"] < lower_bound:
                raise ValueError("predicted_ii must not be below analytical_lower_bound")
            normalized["analytical_lower_bound"] = lower_bound
        elif status != "unsupported":
            raise ValueError("support_status must be supported or unsupported")
        by_query[key] = normalized
    if set(by_query) != set(manifest["queries"]):
        raise ValueError("cost catalogue must exactly cover manifest queries")
    return {
        "root": dict(root),
        "metadata": dict(metadata),
        "namespace": namespace,
        "by_query": by_query,
        "sha256": sha256_file(path),
    }


def score_candidates(
    manifest: Mapping[str, Any], catalog: Mapping[str, Any], top_k: int,
) -> Dict[str, Any]:
    """Score every frozen candidate retained by exact concurrent packing."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    costs = _object(catalog["by_query"], "cost lookup")
    seen_queries = set()
    hits = 0
    misses = 0
    ranked: List[Tuple[float, int, str]] = []
    score_rows = []
    valid_count = 0
    manifest_header = _object(manifest["header"], "manifest header")
    function = str(manifest_header["function"])
    for manifest_index, candidate in enumerate(manifest["candidates"]):
        candidate = _object(candidate, "candidate")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate identity is missing")
        choices = candidate.get("task_shapes")
        if not isinstance(choices, list) or not choices:
            raise ValueError("candidate has no task shapes")
        # The manifest has already proved that all fixed-orientation task
        # rectangles can coexist. This layer only applies pointwise costs; it
        # neither re-runs packing nor uses temporal reuse to relax capacity.
        # TODO: a future analytical spatial-temporal scheduler may model
        # cross-time tile reuse as a separate search scope. It must not change
        # the current co-resident validity contract.
        valid = True
        reject_reason = None
        bottleneck = 0.0
        task_costs = []
        for raw_choice in choices:
            choice = _object(raw_choice, "candidate task shape")
            shape = _object(choice.get("shape"), "candidate shape")
            task = choice.get("task")
            if not isinstance(task, str) or not task:
                raise ValueError("candidate task name is missing")
            rows = _positive_integer(shape.get("mapper_tile_rows"), "mapper_tile_rows")
            cols = _positive_integer(shape.get("mapper_tile_cols"), "mapper_tile_cols")
            trip_count = _positive_integer(choice.get("trip_count"), "trip_count")
            key = (task, rows, cols)
            if key in seen_queries:
                hits += 1
            else:
                seen_queries.add(key)
                misses += 1
            if key not in costs:
                raise ValueError(f"missing cost for task/shape {key!r}")
            cost = _object(costs[key], "cost entry")
            task_cost = {
                "task": task,
                "mapper_tile_rows": rows,
                "mapper_tile_cols": cols,
                "predicted_ii": cost.get("predicted_ii", 0.0),
                "startup_cycles": cost.get("startup_cycles", 0.0),
                "trip_count": trip_count,
                "support_status": cost["support_status"],
            }
            if cost["support_status"] != "supported":
                valid = False
                reject_reason = reject_reason or "UNSUPPORTED_TASK_SHAPE"
            else:
                duration = (
                    float(cost["startup_cycles"]) +
                    float(cost["predicted_ii"]) * (trip_count - 1)
                )
                if not math.isfinite(duration):
                    raise ValueError(f"task duration overflow for {key!r}")
                task_cost["predicted_duration"] = duration
                bottleneck = max(bottleneck, duration)
            task_costs.append(task_cost)
        score_row = {
            "record_type": "score",
            "schema": SCORE_SCHEMA,
            "candidate_id": candidate_id,
            "valid": valid,
            "task_costs": task_costs,
        }
        if valid:
            score_row["predicted_compute_bottleneck"] = bottleneck
            ranked.append((bottleneck, manifest_index, candidate_id))
            valid_count += 1
        else:
            score_row["reject_reason"] = reject_reason
        score_rows.append(score_row)

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected_count = len(ranked) if top_k == 0 else min(top_k, len(ranked))
    shortlist = [
        {
            "rank": rank,
            "candidate_id": candidate_id,
            "predicted_compute_bottleneck": objective,
        }
        for rank, (objective, _, candidate_id) in enumerate(ranked[:selected_count])
    ]
    header = {
        "record_type": "header",
        "schema": SCORE_SCHEMA,
        "candidate_schema": CANDIDATE_SCHEMA,
        "function": function,
        "cost_namespace": catalog["namespace"],
        "score_model": SCORE_MODEL,
        "producer": PIPELINE_SCHEMA,
        "candidate_space": {
            "kind": "packing_pruned_static_shape_candidates",
            "resource_semantics": "all_task_rectangles_co_resident",
            "packing": "fixed_orientation_nonoverlap",
        },
    }
    footer = {
        "record_type": "footer",
        "schema": SCORE_SCHEMA,
        "candidate_count": manifest["candidate_count"],
        "scored_count": len(score_rows),
        "valid_count": valid_count,
        "top_k_requested": top_k,
        "shortlist": shortlist,
        "cache": {"hits": hits, "misses": misses, "entries": len(seen_queries)},
    }
    return {
        "header": header,
        "scores": score_rows,
        "footer": footer,
        "shortlist_ids": [row["candidate_id"] for row in shortlist],
    }


def write_scores(path: Path, scores: Mapping[str, Any]) -> None:
    records = [scores["header"], *scores["scores"], scores["footer"]]
    path.write_text("".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    ))


def parse_task_paths(values: Iterable[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        task, separator, raw_path = value.partition("=")
        if not separator or not task or not raw_path:
            raise ValueError("task DFG must be TASK=PATH")
        if task in result:
            raise ValueError(f"duplicate DFG path for task {task}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"task DFG does not exist: {path}")
        result[task] = path
    return result


def validate_mapper_provenance(
    manifest: Mapping[str, Any], catalog: Mapping[str, Any],
    task_paths: Mapping[str, Path], neura_opt: Path, architecture: Path,
    expected_neura_opt_sha256: Optional[str] = None,
    expected_architecture_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    tasks = {task for task, _, _ in manifest["queries"]}
    if set(task_paths) != tasks:
        raise ValueError("task DFG mapping must exactly cover manifest tasks")
    if not neura_opt.is_file() or not os.access(str(neura_opt), os.X_OK):
        raise ValueError(f"Neura optimizer is missing or not executable: {neura_opt}")
    if not architecture.is_file():
        raise ValueError(f"architecture does not exist: {architecture}")

    opt_sha = sha256_file(neura_opt)
    architecture_sha = sha256_file(architecture)
    if expected_neura_opt_sha256 and opt_sha != expected_neura_opt_sha256:
        raise ValueError("Neura optimizer SHA-256 differs from explicit pin")
    if expected_architecture_sha256 and architecture_sha != expected_architecture_sha256:
        raise ValueError("architecture SHA-256 differs from explicit pin")
    metadata = _object(catalog["metadata"], "predictor metadata")
    analytical = _object(
        metadata.get("analytical_provenance"), "analytical provenance",
    )
    if analytical.get("neura_opt_sha256") != opt_sha:
        raise ValueError("Neura optimizer differs from catalogue provenance")
    if analytical.get("architecture_sha256") != architecture_sha:
        raise ValueError("architecture differs from catalogue provenance")
    raw_dfg_hashes = _object(
        analytical.get("task_dfg_sha256"), "task DFG provenance",
    )
    dfg_hashes = {task: sha256_file(path) for task, path in task_paths.items()}
    if set(raw_dfg_hashes) != tasks or any(
        raw_dfg_hashes[task] != dfg_hashes[task] for task in tasks
    ):
        raise ValueError("task DFGs differ from catalogue provenance")
    for task, path in task_paths.items():
        if source_task_body_sha256(path.read_text(), task) != (
            manifest["task_body_sha256"][task]
        ):
            raise ValueError(f"task DFG source body hash mismatch for {task}")
    return {
        "neura_opt_sha256": opt_sha,
        "architecture_sha256": architecture_sha,
        "task_dfg_sha256": dfg_hashes,
    }


def _safe_task_name(task: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("._")
    if not safe:
        safe = "task"
    if safe != task:
        import hashlib
        safe += "-" + hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
    return safe


def _timeout_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def replay_queries(
    queries: Sequence[QueryKey], task_paths: Mapping[str, Path],
    neura_opt: Path, architecture: Path, output_root: Path, timeout_seconds: float,
    runner: Runner = subprocess.run,
) -> Dict[QueryKey, Dict[str, Any]]:
    """Run and verify each unique query, preserving failures as censored."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("mapper timeout must be positive and finite")
    results: Dict[QueryKey, Dict[str, Any]] = {}
    mapper_root = output_root / "mapper"
    mapper_root.mkdir(parents=True, exist_ok=True)
    for task, rows, cols in queries:
        key = (task, rows, cols)
        if key in results:
            continue
        query_root = mapper_root / _safe_task_name(task) / f"{rows}x{cols}"
        query_root.mkdir(parents=True, exist_ok=False)
        mapped_path = query_root / "mapped.mlir"
        stdout_path = query_root / "stdout.log"
        stderr_path = query_root / "stderr.log"
        command = [
            str(neura_opt), str(task_paths[task]),
            "--insert-data-mov",
            f"--architecture-spec={architecture}",
            (
                "--map-to-accelerator=mapping-strategy=heuristic "
                f"x-tiles={cols} y-tiles={rows}"
            ),
            "-o", str(mapped_path),
        ]
        started = time.perf_counter()
        row: Dict[str, Any] = {
            "task": task,
            "mapper_tile_rows": rows,
            "mapper_tile_cols": cols,
            "status": "censored",
            "compiled_ii": None,
            "command": command,
            "source_dfg_path": str(task_paths[task]),
            "source_dfg_sha256": sha256_file(task_paths[task]),
            "mapped_artifact_path": str(mapped_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        try:
            completed = runner(
                command, capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
            stdout = _timeout_output(completed.stdout)
            stderr = _timeout_output(completed.stderr)
            row["exit_code"] = completed.returncode
            if completed.returncode != 0:
                row["censor_reason"] = "mapper_nonzero_exit"
            elif not mapped_path.is_file():
                row["censor_reason"] = "mapper_output_missing"
            else:
                try:
                    facts = parse_single_mapping(mapped_path.read_text(), rows, cols)
                except (OSError, UnicodeError, ValueError) as error:
                    row["censor_reason"] = "mapper_output_invalid"
                    row["validation_error"] = str(error)
                else:
                    row.update({
                        "status": "success",
                        "compiled_ii": facts["compiled_ii"],
                        "rec_mii": facts["rec_mii"],
                        "res_mii": facts["res_mii"],
                        "lower_bound": max(facts["rec_mii"], facts["res_mii"]),
                        "verified_x_tiles": facts["x_tiles"],
                        "verified_y_tiles": facts["y_tiles"],
                        "verified_placement_coordinate_count": facts[
                            "placement_coordinate_count"
                        ],
                        "shape_faithful": True,
                        "mapped_artifact_sha256": sha256_file(mapped_path),
                    })
        except subprocess.TimeoutExpired as error:
            stdout = _timeout_output(error.stdout)
            stderr = _timeout_output(error.stderr)
            row["exit_code"] = None
            row["censor_reason"] = "mapper_timeout"
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        row["elapsed_seconds"] = time.perf_counter() - started
        row["stdout_sha256"] = sha256_file(stdout_path)
        row["stderr_sha256"] = sha256_file(stderr_path)
        results[key] = row
    return results


def shortlisted_queries(
    manifest: Mapping[str, Any], shortlist_ids: Sequence[str],
) -> Tuple[List[QueryKey], int]:
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in manifest["candidates"]
    }
    if any(candidate_id not in candidates for candidate_id in shortlist_ids):
        raise ValueError("shortlist contains an unknown candidate")
    unique = []
    seen = set()
    references = 0
    for candidate_id in shortlist_ids:
        for choice in candidates[candidate_id]["task_shapes"]:
            shape = choice["shape"]
            key = (
                choice["task"], shape["mapper_tile_rows"],
                shape["mapper_tile_cols"],
            )
            references += 1
            if key not in seen:
                seen.add(key)
                unique.append(key)
    return unique, references


def evaluate_shortlist(
    manifest: Mapping[str, Any], catalog: Mapping[str, Any],
    shortlist_ids: Sequence[str], mapping_results: Mapping[QueryKey, Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in manifest["candidates"]
    }
    costs = catalog["by_query"]
    rows = []
    for rank, candidate_id in enumerate(shortlist_ids):
        candidate = candidates[candidate_id]
        valid = True
        task_results = []
        durations = []
        predicted_durations = []
        for choice in candidate["task_shapes"]:
            shape = choice["shape"]
            key = (
                choice["task"], shape["mapper_tile_rows"],
                shape["mapper_tile_cols"],
            )
            mapping = mapping_results[key]
            predicted_duration = (
                float(costs[key]["startup_cycles"]) +
                float(costs[key]["predicted_ii"]) * (choice["trip_count"] - 1)
            )
            predicted_durations.append(predicted_duration)
            task_result = {
                "task": key[0],
                "mapper_tile_rows": key[1],
                "mapper_tile_cols": key[2],
                "trip_count": choice["trip_count"],
                "mapper_status": mapping["status"],
                "compiled_ii": mapping["compiled_ii"],
                "predicted_ii": costs[key].get("predicted_ii"),
                "startup_cycles": costs[key].get("startup_cycles"),
                "predicted_duration": predicted_duration,
            }
            if mapping["status"] == "success":
                duration = (
                    float(costs[key]["startup_cycles"]) +
                    int(mapping["compiled_ii"]) * (choice["trip_count"] - 1)
                )
                task_result["actual_duration"] = duration
                task_result["signed_prediction_error_cycles"] = (
                    predicted_duration - duration
                )
                task_result["absolute_prediction_error_cycles"] = abs(
                    predicted_duration - duration
                )
                task_result["signed_relative_prediction_error"] = (
                    (predicted_duration - duration) / duration
                )
                durations.append(duration)
            else:
                valid = False
                task_result["censor_reason"] = mapping.get("censor_reason")
            task_results.append(task_result)
        row = {
            "rank": rank,
            "candidate_id": candidate_id,
            "actual_valid": valid,
            "predicted_compute_bottleneck": max(predicted_durations),
            "task_results": task_results,
        }
        if valid:
            actual_bottleneck = max(durations)
            predicted_bottleneck = max(predicted_durations)
            row.update({
                "actual_compute_bottleneck": actual_bottleneck,
                "signed_prediction_error_cycles": (
                    predicted_bottleneck - actual_bottleneck
                ),
                "absolute_prediction_error_cycles": abs(
                    predicted_bottleneck - actual_bottleneck
                ),
                "signed_relative_prediction_error": (
                    (predicted_bottleneck - actual_bottleneck) /
                    actual_bottleneck
                ),
            })
        rows.append(row)
    valid_rows = [row for row in rows if row["actual_valid"]]
    selected = min(
        valid_rows,
        key=lambda row: (row["actual_compute_bottleneck"], row["rank"]),
    ) if valid_rows else None
    return {
        "candidates": rows,
        "evaluable_candidate_count": len(valid_rows),
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_objective_cycles": (
            selected["actual_compute_bottleneck"] if selected else None
        ),
    }


def evaluate_against_oracle(
    oracle_path: Path, manifest: Mapping[str, Any], catalog: Mapping[str, Any],
    shortlist_ids: Sequence[str], shortlist_evaluation: Mapping[str, Any],
    mapping_results: Mapping[QueryKey, Mapping[str, Any]],
) -> Dict[str, Any]:
    function = str(manifest["header"]["function"])
    oracle = load_oracle(oracle_path, function)
    if set(oracle) != set(manifest["queries"]):
        raise ValueError("oracle query set does not match frozen cost queries")
    costs = catalog["by_query"]
    manifest_header = _object(manifest["header"], "manifest header")
    objectives: Dict[str, float] = {}
    for candidate in manifest["candidates"]:
        durations = []
        for choice in candidate["task_shapes"]:
            shape = choice["shape"]
            key = (
                choice["task"], shape["mapper_tile_rows"],
                shape["mapper_tile_cols"],
            )
            oracle_row = oracle[key]
            if oracle_row["status"] != "success":
                durations = []
                break
            durations.append(
                float(costs[key]["startup_cycles"]) +
                int(oracle_row["compiled_ii"]) * (choice["trip_count"] - 1)
            )
        if durations:
            objectives[str(candidate["candidate_id"])] = max(durations)
    if not objectives:
        raise ValueError("oracle has no evaluable complete candidate")
    best = min(objectives.values())
    optimal_ids = [
        str(candidate["candidate_id"])
        for candidate in manifest["candidates"]
        if str(candidate["candidate_id"]) in objectives and math.isclose(
            objectives[str(candidate["candidate_id"])], best, abs_tol=1e-9,
        )
    ]
    shortlisted_optimal = [
        candidate_id for candidate_id in shortlist_ids
        if candidate_id in set(optimal_ids)
    ]
    selected_id = shortlist_evaluation["selected_candidate_id"]
    selected_objective = (
        objectives.get(selected_id) if selected_id is not None else None
    )
    top1 = shortlist_ids[0] if shortlist_ids else None
    top1_objective = objectives.get(top1) if top1 is not None else None
    replay_matches = []
    for key, result in mapping_results.items():
        oracle_row = oracle[key]
        replay_matches.append({
            "task": key[0], "mapper_tile_rows": key[1],
            "mapper_tile_cols": key[2],
            "matches_oracle": (
                result["status"] == oracle_row["status"] and
                result.get("compiled_ii") == oracle_row.get("compiled_ii")
            ),
        })
    return {
        "oracle_path": str(oracle_path),
        "oracle_sha256": sha256_file(oracle_path),
        "candidate_scope": "packing_pruned_static_shape_candidates",
        "evaluable_candidate_count": len(objectives),
        "best_objective_cycles": best,
        "optimal_candidate_ids": optimal_ids,
        "shortlist_oracle_optimal_ids": shortlisted_optimal,
        "top1_oracle_recall": float(top1 in set(optimal_ids)),
        "topk_any_oracle_hit": bool(shortlisted_optimal),
        "topk_oracle_recall": len(shortlisted_optimal) / len(optimal_ids),
        "top1_oracle_objective_cycles": top1_objective,
        "top1_absolute_regret_cycles": (
            top1_objective - best if top1_objective is not None else None
        ),
        "selected_absolute_regret_cycles": (
            selected_objective - best if selected_objective is not None else None
        ),
        "selected_relative_regret": (
            (selected_objective - best) / best
            if selected_objective is not None else None
        ),
        "shortlist_replay_matches": replay_matches,
        "all_shortlist_replays_match_oracle": all(
            row["matches_oracle"] for row in replay_matches
        ),
    }


def run_pipeline(
    manifest_path: Path, catalog_path: Path, task_paths: Mapping[str, Path],
    neura_opt: Path, architecture: Path, output_root: Path, top_k: int,
    timeout_seconds: float, oracle_path: Optional[Path] = None,
    expected_neura_opt_sha256: Optional[str] = None,
    expected_architecture_sha256: Optional[str] = None,
    neura_source_revision: Optional[str] = None,
    runner: Runner = subprocess.run,
) -> Dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = load_candidate_manifest(manifest_path)
    catalog = load_cost_catalog(catalog_path, manifest)
    provenance = validate_mapper_provenance(
        manifest, catalog, task_paths, neura_opt, architecture,
        expected_neura_opt_sha256, expected_architecture_sha256,
    )
    scores = score_candidates(manifest, catalog, top_k)
    if not scores["shortlist_ids"]:
        raise ValueError("scoring produced an empty shortlist")
    score_path = output_root / "scores.jsonl"
    write_scores(score_path, scores)
    queries, task_references = shortlisted_queries(
        manifest, scores["shortlist_ids"],
    )
    mapping_started = time.perf_counter()
    mapping_results = replay_queries(
        queries, task_paths, neura_opt, architecture, output_root,
        timeout_seconds, runner,
    )
    mapping_seconds = time.perf_counter() - mapping_started
    shortlist_evaluation = evaluate_shortlist(
        manifest, catalog, scores["shortlist_ids"], mapping_results,
    )
    report: Dict[str, Any] = {
        "schema": PIPELINE_SCHEMA,
        "status": (
            "complete" if shortlist_evaluation["selected_candidate_id"]
            else "complete_with_no_evaluable_shortlist_candidate"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "function": manifest["header"]["function"],
        "configuration": {
            "top_k": len(scores["shortlist_ids"]),
            "top_k_requested": top_k,
            "mapper_timeout_seconds": timeout_seconds,
            "score_model": SCORE_MODEL,
            "mapper_strategy": "heuristic",
            "mapper_preprocessing": ["insert-data-mov"],
            "shape_replay": "direct_x_tiles_y_tiles_override",
            "neura_source_revision": neura_source_revision,
        },
        "artifacts": {
            "candidate_manifest_path": str(manifest_path),
            "candidate_manifest_sha256": manifest["manifest_sha256"],
            "cost_catalog_path": str(catalog_path),
            "cost_catalog_sha256": catalog["sha256"],
            "scores_path": str(score_path),
            "scores_sha256": sha256_file(score_path),
            "neura_opt_path": str(neura_opt),
            "architecture_path": str(architecture),
            **provenance,
            "task_dfg_paths": {
                task: str(path) for task, path in task_paths.items()
            },
        },
        "scoring": {
            "candidate_count": manifest["candidate_count"],
            "scored_count": len(scores["scores"]),
            "valid_count": scores["footer"]["valid_count"],
            "cost_query_count": len(manifest["queries"]),
            "cache": scores["footer"]["cache"],
            "shortlist": scores["footer"]["shortlist"],
        },
        "mapper_replay": {
            "status": "faithful_direct_task_shape_replay",
            "exhaustive_program_candidate_calls": manifest["candidate_count"],
            "supported_program_candidates": scores["footer"]["valid_count"],
            "actual_shortlist_program_candidates": len(scores["shortlist_ids"]),
            "program_candidate_reduction_fraction": (
                1.0 - len(scores["shortlist_ids"]) / manifest["candidate_count"]
            ),
            "supported_program_candidate_reduction_fraction": (
                1.0 - len(scores["shortlist_ids"]) /
                scores["footer"]["valid_count"]
            ),
            "exhaustive_unique_task_shape_calls": len(manifest["queries"]),
            "shortlist_task_shape_references": task_references,
            "actual_unique_mapper_calls": len(mapping_results),
            "task_shape_call_reduction_fraction": (
                1.0 - len(mapping_results) / len(manifest["queries"])
            ),
            "shortlist_cache_hits": task_references - len(mapping_results),
            "success_count": sum(
                row["status"] == "success" for row in mapping_results.values()
            ),
            "censored_count": sum(
                row["status"] == "censored" for row in mapping_results.values()
            ),
            "queries": list(mapping_results.values()),
        },
        "selection": shortlist_evaluation,
        "timing": {
            "mapper_replay_seconds": mapping_seconds,
            "pipeline_seconds": time.perf_counter() - started,
        },
    }
    if oracle_path is not None:
        report["oracle_evaluation"] = evaluate_against_oracle(
            oracle_path, manifest, catalog, scores["shortlist_ids"],
            shortlist_evaluation, mapping_results,
        )
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cost-catalog", type=Path, required=True)
    parser.add_argument(
        "--task-dfg", action="append", required=True, metavar="TASK=PATH",
    )
    parser.add_argument("--neura-opt", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mapper-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--expected-neura-opt-sha256")
    parser.add_argument("--expected-architecture-sha256")
    parser.add_argument("--neura-source-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_pipeline(
        args.manifest.resolve(), args.cost_catalog.resolve(),
        parse_task_paths(args.task_dfg), args.neura_opt.resolve(),
        args.architecture.resolve(), args.output_dir.resolve(), args.top_k,
        args.mapper_timeout_seconds,
        args.oracle.resolve() if args.oracle else None,
        args.expected_neura_opt_sha256,
        args.expected_architecture_sha256,
        args.neura_source_revision,
    )
    summary = {
        "report": str((args.output_dir.resolve() / "report.json")),
        "status": report["status"],
        "predicted_shortlist": [
            row["candidate_id"] for row in report["scoring"]["shortlist"]
        ],
        "selected_candidate_id": report["selection"]["selected_candidate_id"],
        "selected_objective_cycles": report["selection"]["selected_objective_cycles"],
        "actual_unique_mapper_calls": report["mapper_replay"][
            "actual_unique_mapper_calls"
        ],
        "censored_mapper_calls": report["mapper_replay"]["censored_count"],
    }
    if "oracle_evaluation" in report:
        summary["selected_absolute_regret_cycles"] = report[
            "oracle_evaluation"
        ]["selected_absolute_regret_cycles"]
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["selection"]["selected_candidate_id"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
