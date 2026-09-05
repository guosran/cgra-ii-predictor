#!/usr/bin/env python3
"""Compare a pointwise cost catalogue with an existing mapper replay report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


QueryKey = Tuple[str, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_costs(path: Path) -> Dict[QueryKey, Mapping[str, Any]]:
    raw = json.loads(path.read_text())
    entries = raw.get("entries") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("cost catalogue lacks entries")
    result: Dict[QueryKey, Mapping[str, Any]] = {}
    for row in entries:
        if not isinstance(row, Mapping):
            raise ValueError("cost entry must be an object")
        key = (
            str(row["task"]), int(row["mapper_tile_rows"]),
            int(row["mapper_tile_cols"]),
        )
        if key in result:
            raise ValueError("duplicate cost entry")
        predicted = float(row["predicted_ii"])
        if not math.isfinite(predicted) or predicted <= 0.0:
            raise ValueError("invalid predicted II")
        result[key] = row
    return result


def build_report(cost_path: Path, replay_path: Path) -> Dict[str, Any]:
    costs = _load_costs(cost_path)
    replay = json.loads(replay_path.read_text())
    mapping = replay.get("mapper_replay") if isinstance(replay, Mapping) else None
    queries = mapping.get("queries") if isinstance(mapping, Mapping) else None
    if not isinstance(queries, list):
        raise ValueError("replay report lacks mapper queries")
    actual: Dict[QueryKey, float] = {}
    for row in queries:
        if not isinstance(row, Mapping) or row.get("status") != "success":
            continue
        key = (
            str(row["task"]), int(row["mapper_tile_rows"]),
            int(row["mapper_tile_cols"]),
        )
        value = float(row["compiled_ii"])
        prior = actual.setdefault(key, value)
        if prior != value:
            raise ValueError("replay has conflicting compiled II labels")
    if not actual:
        raise ValueError("replay contains no successful mapper query")

    rows = []
    for key in sorted(actual):
        if key not in costs:
            raise ValueError(f"cost catalogue lacks replay query: {key}")
        predicted = float(costs[key]["predicted_ii"])
        error = predicted - actual[key]
        rows.append({
            "task": key[0],
            "mapper_tile_rows": key[1],
            "mapper_tile_cols": key[2],
            "predicted_ii": predicted,
            "compiled_ii": actual[key],
            "signed_error": error,
            "absolute_error": abs(error),
        })

    selection = replay.get("selection")
    selected_rows = selection.get("candidates") if isinstance(selection, Mapping) else None
    reference = next((
        row for row in selected_rows or ()
        if isinstance(row, Mapping) and row.get("actual_valid") is True
    ), None)
    program = None
    if reference is not None:
        predicted_durations = []
        task_results = reference.get("task_results")
        if not isinstance(task_results, list):
            raise ValueError("valid replay candidate lacks task results")
        for row in task_results:
            if not isinstance(row, Mapping):
                raise ValueError("task result must be an object")
            key = (
                str(row["task"]), int(row["mapper_tile_rows"]),
                int(row["mapper_tile_cols"]),
            )
            predicted_ii = float(costs[key]["predicted_ii"])
            duration = float(row["startup_cycles"]) + predicted_ii * (
                int(row["trip_count"]) - 1
            )
            predicted_durations.append(duration)
        predicted_objective = max(predicted_durations)
        actual_objective = float(reference["actual_compute_bottleneck"])
        program = {
            "reference_candidate_id": str(reference["candidate_id"]),
            "predicted_objective_cycles": predicted_objective,
            "actual_objective_cycles": actual_objective,
            "signed_error_cycles": predicted_objective - actual_objective,
            "signed_relative_error": (
                predicted_objective - actual_objective
            ) / actual_objective,
        }

    errors = [float(row["signed_error"]) for row in rows]
    return {
        "schema_version": "cgra-ii-amoeba-cost-replay-comparison-v1",
        "cost_catalog": {
            "path": str(cost_path.resolve()), "sha256": sha256_file(cost_path),
        },
        "replay_report": {
            "path": str(replay_path.resolve()), "sha256": sha256_file(replay_path),
        },
        "successful_query_count": len(rows),
        "ii_metrics": {
            "mae": sum(abs(value) for value in errors) / len(errors),
            "mean_signed_error": sum(errors) / len(errors),
            "underprediction_rate": sum(value < 0.0 for value in errors) / len(errors),
            "maximum_underprediction": max((-value for value in errors), default=0.0),
        },
        "program": program,
        "queries": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-catalog", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.cost_catalog.resolve(), args.replay_report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "ii_metrics": report["ii_metrics"],
        "program": report["program"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
