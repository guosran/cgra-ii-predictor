#!/usr/bin/env python3
"""Audit an Amoeba score file against real per-task heuristic-mapper labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amoeba_cost_catalog import load_candidate_manifest, sha256_file  # noqa: E402
from amoeba_protocol import SCORE_SCHEMA  # noqa: E402


ORACLE_SCHEMA = "cgra-ii-amoeba-query-oracle"
QueryKey = Tuple[str, int, int]


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def load_scores(path: Path) -> Dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    if len(records) < 3:
        raise ValueError("score JSONL is incomplete")
    header = _object(records[0], "score header")
    footer = _object(records[-1], "score footer")
    scores = [_object(record, "score") for record in records[1:-1]]
    if any(record.get("schema_version") != SCORE_SCHEMA for record in records):
        raise ValueError("score schema_version mismatch")
    if header.get("record_type") != "header" or footer.get("record_type") != "footer":
        raise ValueError("score header/footer is invalid")
    if any(record.get("record_type") != "score" for record in scores):
        raise ValueError("score file contains a non-score body record")
    if footer.get("scored_count") != len(scores):
        raise ValueError("score footer count mismatch")
    identities = [record.get("candidate_id") for record in scores]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("score candidate identity is missing")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate score candidate identity")
    shortlist = footer.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise ValueError("score footer has no shortlist")
    shortlist_ids = []
    for rank, raw in enumerate(shortlist):
        row = _object(raw, "shortlist entry")
        if row.get("rank") != rank or not isinstance(row.get("candidate_id"), str):
            raise ValueError("score shortlist is not ordered")
        shortlist_ids.append(row["candidate_id"])
    return {
        "header": dict(header), "footer": dict(footer),
        "scores": {str(row["candidate_id"]): dict(row) for row in scores},
        "shortlist_ids": shortlist_ids, "sha256": sha256_file(path),
    }


def load_oracle(path: Path, function: str) -> Dict[QueryKey, Dict[str, Any]]:
    root = _object(json.loads(path.read_text()), "oracle")
    if root.get("schema_version") != ORACLE_SCHEMA:
        raise ValueError("oracle schema_version mismatch")
    if root.get("function") != function:
        raise ValueError("oracle function mismatch")
    entries = root.get("entries")
    if not isinstance(entries, list):
        raise ValueError("oracle entries must be an array")
    result = {}
    for raw in entries:
        row = _object(raw, "oracle entry")
        task = row.get("task")
        rows = row.get("mapper_tile_rows")
        cols = row.get("mapper_tile_cols")
        if (not isinstance(task, str) or isinstance(rows, bool) or
                isinstance(cols, bool) or not isinstance(rows, int) or
                not isinstance(cols, int) or rows <= 0 or cols <= 0):
            raise ValueError("oracle query identity is invalid")
        key = (task, rows, cols)
        if key in result:
            raise ValueError("duplicate oracle query")
        status = row.get("status")
        compiled = row.get("compiled_ii")
        if status == "success":
            if (isinstance(compiled, bool) or not isinstance(compiled, int) or
                    compiled <= 0):
                raise ValueError("successful oracle entry needs compiled_ii")
        elif status == "censored":
            if compiled is not None:
                raise ValueError("censored oracle entry must not have compiled_ii")
        else:
            raise ValueError("oracle status must be success or censored")
        result[key] = dict(row)
    return result


def evaluate_scores(
    candidate_path: Path, score_path: Path, oracle_path: Path,
    predictor_timing_path: Path = None, scorer_ms: float = None,
    replay_status: str = "blocked_materialized_shape_not_preserved",
) -> Dict[str, Any]:
    if replay_status not in {
        "blocked_materialized_shape_not_preserved", "faithful",
    }:
        raise ValueError("invalid mapper replay status")
    candidates = load_candidate_manifest(candidate_path)
    scores = load_scores(score_path)
    function = str(candidates["header"]["function"])
    if scores["header"].get("function") != function:
        raise ValueError("score function mismatch")
    oracle = load_oracle(oracle_path, function)
    if set(oracle) != set(candidates["queries"]):
        raise ValueError("oracle query set does not match frozen cost queries")
    candidate_by_id = {
        str(row["candidate_id"]): row for row in candidates["candidates"]
    }
    if set(scores["scores"]) != set(candidate_by_id):
        raise ValueError("scores do not cover every frozen candidate exactly once")
    footer = scores["footer"]
    cache = _object(footer.get("cache"), "score cache")
    task_shape_references = sum(
        len(row["task_shapes"]) for row in candidates["candidates"]
    )
    expected_queries = len(candidates["queries"])
    if (cache.get("entries") != expected_queries or
            cache.get("misses") != expected_queries or
            cache.get("hits") != task_shape_references - expected_queries):
        raise ValueError("Amoeba scorer cache accounting is inconsistent")
    if footer.get("candidate_count") != candidates["candidate_count"]:
        raise ValueError("score candidate count differs from frozen manifest")

    oracle_objectives = {}
    unevaluable = []
    for candidate_id, candidate in candidate_by_id.items():
        durations = []
        score_costs = scores["scores"][candidate_id].get("task_costs")
        if not isinstance(score_costs, list):
            raise ValueError("score lacks task costs")
        cost_by_key = {
            (row["task"], row["mapper_tile_rows"], row["mapper_tile_cols"]): row
            for row in score_costs
        }
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
            score_cost = cost_by_key.get(key)
            if score_cost is None:
                raise ValueError("score task costs differ from candidate shapes")
            startup = score_cost.get("startup_cycles")
            trip_count = choice.get("trip_count")
            if (not isinstance(startup, (int, float)) or
                    not math.isfinite(float(startup)) or float(startup) <= 0 or
                    isinstance(trip_count, bool) or
                    not isinstance(trip_count, int) or trip_count <= 0):
                raise ValueError("score startup or candidate trip count is invalid")
            durations.append(
                float(startup) + int(oracle_row["compiled_ii"]) * (trip_count - 1)
            )
        if durations:
            oracle_objectives[candidate_id] = max(durations)
        else:
            unevaluable.append(candidate_id)
    if not oracle_objectives:
        raise ValueError("oracle has no evaluable complete candidate")

    oracle_best = min(oracle_objectives.values())
    tolerance = 1e-9
    optimal_ids = sorted(
        candidate_id for candidate_id, objective in oracle_objectives.items()
        if abs(objective - oracle_best) <= tolerance
    )
    shortlist = scores["shortlist_ids"]
    top1 = shortlist[0]
    shortlist_optimal = sorted(set(shortlist).intersection(optimal_ids))
    selected_objective = oracle_objectives.get(top1)
    absolute_regret = (
        selected_objective - oracle_best
        if selected_objective is not None else None
    )
    top_k = len(shortlist)
    timing = None
    if predictor_timing_path is not None:
        timing = json.loads(predictor_timing_path.read_text())
    result = {
        "schema_version": "cgra-ii-amoeba-dse-evaluation",
        "function": function,
        "artifacts": {
            "candidate_manifest_sha256": candidates["manifest_sha256"],
            "score_sha256": scores["sha256"],
            "oracle_sha256": sha256_file(oracle_path),
        },
        "coverage": {
            "candidate_count": candidates["candidate_count"],
            "scored_count": len(scores["scores"]),
            "valid_score_count": footer.get("valid_count"),
            "unique_cost_query_count": expected_queries,
            "oracle_evaluable_candidate_count": len(oracle_objectives),
            "oracle_unevaluable_candidate_count": len(unevaluable),
            "amoeba_cache": dict(cache),
            "top_k": top_k,
            "top_k_emitted_after_scored_count": footer.get("scored_count"),
        },
        "oracle": {
            "best_objective_cycles": oracle_best,
            "optimal_candidate_ids": optimal_ids,
            "optimal_candidate_count": len(optimal_ids),
        },
        "selection": {
            "predicted_top1_candidate_id": top1,
            "predicted_shortlist_candidate_ids": shortlist,
            "top1_oracle_recall": float(top1 in optimal_ids),
            "topk_oracle_recall": len(shortlist_optimal) / len(optimal_ids),
            "topk_any_oracle_hit": bool(shortlist_optimal),
            "shortlist_oracle_optimal_ids": shortlist_optimal,
            "top1_oracle_objective_cycles": selected_objective,
            "absolute_objective_regret_cycles": absolute_regret,
            "relative_objective_regret": (
                absolute_regret / oracle_best
                if absolute_regret is not None else None
            ),
        },
        "mapper_replay": {
            "status": replay_status,
            "exhaustive_program_candidate_calls": candidates["candidate_count"],
            "projected_shortlist_program_candidate_calls": top_k,
            "projected_call_reduction_fraction": (
                1.0 - top_k / candidates["candidate_count"]
            ),
            "actual_shortlist_program_candidate_calls": (
                top_k if replay_status == "faithful" else 0
            ),
            "actual_call_reduction_fraction": (
                1.0 - top_k / candidates["candidate_count"]
                if replay_status == "faithful" else None
            ),
            "query_oracle_calls_not_counted_as_deployed_dse": len(oracle),
        },
        "timing": {
            "predictor": timing,
            "amoeba_scorer_ms": scorer_ms,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--predictor-timing", type=Path)
    parser.add_argument("--scorer-ms", type=float)
    parser.add_argument(
        "--replay-status",
        choices=("blocked_materialized_shape_not_preserved", "faithful"),
        default="blocked_materialized_shape_not_preserved",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_scores(
        args.manifest.resolve(), args.scores.resolve(), args.oracle.resolve(),
        args.predictor_timing.resolve() if args.predictor_timing else None,
        args.scorer_ms, args.replay_status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
