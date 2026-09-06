#!/usr/bin/env python3
"""Enumerate op-capped, exactly grid-packable static Amoeba candidates.

The op-count cap is an explicit search heuristic: a task may use at most
``ceil(materialized_ops / tiles_per_physical_cgra) + slack`` physical CGRAs.
Exact incremental rectangle packing is a hard architecture constraint.  The
tool consumes a valid base manifest for immutable function/task/trip-count and
architecture facts, then publishes a self-contained candidate manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amoeba_cost_catalog import (  # noqa: E402
    CANDIDATE_SCHEMA,
    load_candidate_manifest,
    parse_task_paths,
    sha256_file,
)
from amoeba_frozen_pipeline import _grid_packable  # noqa: E402


Shape = Tuple[int, int]

_NON_MATERIALIZED = frozenset({
    "br", "cond_br", "ctrl_mov", "data_mov", "kernel", "reserve", "yield",
})


def materialized_operation_count(text: str) -> int:
    """Count mapper-relevant Neura operations in one extracted kernel."""
    count = 0
    kernel_count = 0
    for line in text.splitlines():
        match = re.match(
            r'\s*(?:%[A-Za-z0-9_.$-]+(?::[0-9]+)?\s*=\s*)?'
            r'"?neura\.([a-z_]+)"?',
            line,
        )
        if match is None:
            continue
        kind = match.group(1)
        if kind == "kernel":
            kernel_count += 1
        elif kind not in _NON_MATERIALIZED:
            count += 1
    if kernel_count != 1:
        raise ValueError("task DFG must contain exactly one Neura kernel")
    if count <= 0:
        raise ValueError("task DFG has no materialized Neura operations")
    return count


def rectangular_shapes(
    grid_rows: int, grid_cols: int, maximum_area: int,
) -> List[Shape]:
    result = []
    for area in range(1, min(maximum_area, grid_rows * grid_cols) + 1):
        for rows in range(1, grid_rows + 1):
            if area % rows:
                continue
            cols = area // rows
            if cols <= grid_cols:
                result.append((rows, cols))
    return result


def _shape_record(
    rows: int, cols: int, per_cgra_rows: int, per_cgra_cols: int,
) -> Dict[str, Any]:
    return {
        "kind": "rect",
        "rows": rows,
        "cols": cols,
        "cgra_count": rows * cols,
        "cgra_shape": f"{rows}x{cols}",
        "mapper_tile_rows": rows * per_cgra_rows,
        "mapper_tile_cols": cols * per_cgra_cols,
    }


def enumerate_pruned_records(
    base_manifest: Mapping[str, Any], task_texts: Mapping[str, str],
    *, max_cgras_per_task: int = 4, op_cap_slack_cgras: int = 0,
    max_candidates: int = 1_000_000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_cgras_per_task <= 0 or op_cap_slack_cgras < 0:
        raise ValueError("shape cap must be positive and slack nonnegative")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    header = dict(base_manifest["header"])
    tasks = header.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("base manifest has no task facts")
    task_names = [task.get("task") for task in tasks]
    if any(not isinstance(task, str) or not task for task in task_names):
        raise ValueError("base manifest contains an invalid task name")
    if len(set(task_names)) != len(task_names):
        raise ValueError("base manifest contains duplicate task names")
    if set(task_texts) != set(task_names):
        raise ValueError("task DFG mapping must exactly cover base manifest tasks")

    architecture = header.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("base manifest architecture is missing")
    grid_rows = int(architecture["grid_rows"])
    grid_cols = int(architecture["grid_cols"])
    per_rows = int(architecture["per_cgra_tile_rows"])
    per_cols = int(architecture["per_cgra_tile_cols"])
    if min(grid_rows, grid_cols, per_rows, per_cols) <= 0:
        raise ValueError("architecture dimensions must be positive")
    tiles_per_cgra = per_rows * per_cols

    operation_counts = {
        task: materialized_operation_count(task_texts[task])
        for task in task_names
    }
    shape_caps = {
        task: min(
            max_cgras_per_task,
            max(1, math.ceil(operation_counts[task] / tiles_per_cgra)) +
            op_cap_slack_cgras,
        )
        for task in task_names
    }
    shapes_by_task = {
        task: rectangular_shapes(grid_rows, grid_cols, shape_caps[task])
        for task in task_names
    }
    if any(not shapes for shapes in shapes_by_task.values()):
        raise ValueError("op-count policy produced an empty task shape set")

    unpruned_shape_count = len(rectangular_shapes(
        grid_rows, grid_cols, max_cgras_per_task,
    ))
    unpruned_count = unpruned_shape_count ** len(task_names)
    raw_count = math.prod(len(shapes_by_task[task]) for task in task_names)
    selected: List[Shape] = []
    assignments: List[Tuple[Shape, ...]] = []

    def visit(task_index: int) -> None:
        if task_index == len(task_names):
            if len(assignments) >= max_candidates:
                raise ValueError(
                    f"grid-packable space exceeds max-candidates={max_candidates}"
                )
            assignments.append(tuple(selected))
            return
        for shape in shapes_by_task[task_names[task_index]]:
            selected.append(shape)
            if _grid_packable(grid_rows, grid_cols, tuple(selected)):
                visit(task_index + 1)
            selected.pop()

    visit(0)
    if not assignments:
        raise ValueError("no candidate satisfies exact physical-grid packing")

    header["max_cgras_per_task"] = max_cgras_per_task
    header["cost_queries"] = [
        {
            "task": task,
            "mapper_tile_rows": rows * per_rows,
            "mapper_tile_cols": cols * per_cols,
        }
        for task in task_names
        for rows, cols in shapes_by_task[task]
    ]
    fixed_axes = dict(header.get("fixed_axes", {}))
    fixed_axes["candidate_pruning"] = {
        "hardware_grid_packing": "exact-incremental-oriented-rectangles",
        "op_count_cap": "heuristic-ceil-ops-over-physical-cgra-tiles",
        "op_cap_slack_cgras": op_cap_slack_cgras,
        "materialized_op_excludes": sorted(_NON_MATERIALIZED),
        "task_caps": [
            {
                "task": task,
                "materialized_operation_count": operation_counts[task],
                "maximum_physical_cgras": shape_caps[task],
                "shape_count": len(shapes_by_task[task]),
            }
            for task in task_names
        ],
    }
    header["fixed_axes"] = fixed_axes

    records: List[Dict[str, Any]] = [header]
    for candidate_index, assignment in enumerate(assignments):
        choices = []
        for task_fact, (rows, cols) in zip(tasks, assignment):
            choices.append({
                "task": task_fact["task"],
                "trip_count": task_fact["trip_count"],
                "shape": _shape_record(rows, cols, per_rows, per_cols),
            })
        records.append({
            "record_type": "candidate",
            "schema_version": CANDIDATE_SCHEMA,
            "candidate_id": f"candidate-{candidate_index}",
            "task_shapes": choices,
        })
    records.append({
        "record_type": "footer",
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_count": len(assignments),
    })
    report = {
        "schema_version": "amoeba-pruned-candidate-enumeration",
        "function": header["function"],
        "unpruned_cartesian_count": unpruned_count,
        "unpruned_shape_count_per_task": unpruned_shape_count,
        "op_capped_cartesian_count": raw_count,
        "op_cap_pruned_count": unpruned_count - raw_count,
        "op_cap_reduction_fraction": 1.0 - raw_count / unpruned_count,
        # Kept for compatibility with the first report revision.  "Raw" here
        # means the Cartesian product after the per-task op cap, before packing.
        "raw_unpacked_cartesian_count": raw_count,
        "grid_packable_candidate_count": len(assignments),
        "grid_pack_pruned_count": raw_count - len(assignments),
        "total_reduction_fraction": 1.0 - len(assignments) / unpruned_count,
        "operation_counts": operation_counts,
        "maximum_physical_cgras_by_task": shape_caps,
        "shape_counts_by_task": {
            task: len(shapes_by_task[task]) for task in task_names
        },
        "allowed_shapes_by_task": {
            task: [f"{rows}x{cols}" for rows, cols in shapes_by_task[task]]
            for task in task_names
        },
        "task_dfg_sha256": {
            task: hashlib.sha256(task_texts[task].encode("utf-8")).hexdigest()
            for task in task_names
        },
    }
    return records, report


def _write_jsonl_atomically(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--task-dfg", action="append", default=[])
    parser.add_argument("--max-cgras-per-task", type=int, default=4)
    parser.add_argument("--op-cap-slack-cgras", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_candidate_manifest(args.base_manifest.resolve())
    task_paths = parse_task_paths(args.task_dfg)
    records, report = enumerate_pruned_records(
        base,
        {task: path.read_text() for task, path in task_paths.items()},
        max_cgras_per_task=args.max_cgras_per_task,
        op_cap_slack_cgras=args.op_cap_slack_cgras,
        max_candidates=args.max_candidates,
    )
    _write_jsonl_atomically(args.output.resolve(), records)
    report.update({
        "base_manifest_sha256": sha256_file(args.base_manifest.resolve()),
        "output_manifest_sha256": sha256_file(args.output.resolve()),
    })
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
