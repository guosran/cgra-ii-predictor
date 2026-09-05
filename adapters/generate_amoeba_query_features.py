#!/usr/bin/env python3
"""Generate analytical inputs for the Amoeba pointwise cost adapter.

RecMII and ResMII come from Neura's analysis-only pass for every frozen
task/shape query.  Startup is a frontend property: the critical-path depth of
the semantic pre-mapper DFG, with routing-only moves collapsed.  No mapper is
invoked and no compiled-II label is read or synthesized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amoeba_cost_catalog import (  # noqa: E402
    ANALYTICAL_INPUT_SCHEMA,
    load_candidate_manifest,
    parse_task_paths,
    sha256_file,
)
from cgra_ii_predictor.graph_model import parse_neura_dfg  # noqa: E402
from neura_experiment import parse_cost_features  # noqa: E402


QueryKey = Tuple[str, int, int]
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def semantic_critical_path_depth(dfg_text: str) -> int:
    """Return unit-latency semantic DAG depth used as frontend startup.

    The model graph contains result-producing operations only.  A terminal
    ``neura.store*`` has no SSA result but is a real pipeline stage, so include
    it explicitly after resolving transparent movement values.
    """
    graph = parse_neura_dfg(dfg_text)
    parents = [[] for _ in graph.node_types]
    for source, target in graph.edges:
        parents[target].append(source)
    depth = [1] * len(graph.node_types)
    for node in range(len(graph.node_types)):
        if parents[node]:
            depth[node] = 1 + max(depth[parent] for parent in parents[node])
    result = max(depth)

    value_order = []
    kinds = {}
    operands = {}
    for line in dfg_text.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if match is None:
            continue
        value, expression = match.groups()
        kind_match = re.search(r'"?neura\.([a-z_]+)', expression)
        if kind_match is None:
            continue
        kinds[value] = kind_match.group(1)
        operands[value] = re.findall(r"%[A-Za-z0-9_]+", expression)
        value_order.append(value)
    value_depth = {}
    for value in value_order:
        parent_depth = max(
            (value_depth.get(parent, 0) for parent in operands[value]),
            default=0,
        )
        if kinds[value] in {"data_mov", "ctrl_mov", "reserve", "yield"}:
            value_depth[value] = parent_depth
        else:
            value_depth[value] = parent_depth + 1
    for line in dfg_text.splitlines():
        if re.match(r"\s*%[A-Za-z0-9_]+\s*=", line):
            continue
        if re.search(r'"?neura\.store(?:_[a-z_]+)?"?', line) is None:
            continue
        store_operands = re.findall(r"%[A-Za-z0-9_]+", line)
        result = max(
            result,
            1 + max(
                (value_depth.get(value, 0) for value in store_operands),
                default=0,
            ),
        )
    return result


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command), check=False, capture_output=True, text=True,
    )


def generate_query_features(
    candidate_manifest: Path, task_paths: Mapping[str, Path], opt: Path,
    architecture: Path, runner: Runner = default_runner,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    manifest = load_candidate_manifest(candidate_manifest)
    header = manifest["header"]
    tasks = sorted({task for task, _, _ in manifest["queries"]})
    if set(task_paths) != set(tasks):
        raise ValueError("task DFG mapping must exactly cover manifest tasks")
    if not opt.is_file():
        raise ValueError(f"Neura optimizer does not exist: {opt}")
    if not architecture.is_file():
        raise ValueError(f"architecture does not exist: {architecture}")

    startup_by_task: Dict[str, int] = {}
    task_hashes: Dict[str, str] = {}
    for task in tasks:
        text = task_paths[task].read_text()
        startup_by_task[task] = semantic_critical_path_depth(text)
        task_hashes[task] = sha256_file(task_paths[task])

    entries = []
    commands = []
    with tempfile.TemporaryDirectory(prefix="amoeba-query-features-") as raw:
        temporary = Path(raw)
        for index, (task, rows, cols) in enumerate(manifest["queries"]):
            output = temporary / f"analysis-{index}.mlir"
            command = (
                str(opt.resolve()), str(task_paths[task].resolve()),
                f"--architecture-spec={architecture.resolve()}",
                f"--analyze-rec-res-mii=x-tiles={cols} y-tiles={rows}",
                "-o", str(output),
            )
            completed = runner(command)
            commands.append(list(command[:-2]) + ["-o", "<temporary>"])
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"Neura analysis failed for {(task, rows, cols)}: {detail}"
                )
            if not output.is_file():
                raise RuntimeError(
                    f"Neura analysis produced no output for {(task, rows, cols)}"
                )
            facts = parse_cost_features(output.read_text())
            if facts is None:
                raise RuntimeError(
                    f"Neura analysis output is invalid for {(task, rows, cols)}"
                )
            rec_mii = int(facts["rec_mii"])
            res_mii = int(facts["res_mii"])
            lower_bound = max(rec_mii, res_mii)
            if lower_bound < 1:
                raise RuntimeError(
                    f"Neura analysis returned a nonpositive lower bound for "
                    f"{(task, rows, cols)}"
                )
            entries.append({
                "task": task,
                "mapper_tile_rows": rows,
                "mapper_tile_cols": cols,
                "rec_mii": rec_mii,
                "res_mii": res_mii,
                "lower_bound": lower_bound,
                "startup_cycles": startup_by_task[task],
            })

    provenance = {
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "neura_opt_path": str(opt.resolve()),
        "neura_opt_sha256": sha256_file(opt),
        "architecture_path": str(architecture.resolve()),
        "architecture_sha256": sha256_file(architecture),
        "task_dfg_sha256": task_hashes,
        "rec_res_source": "neura-analysis-only-x-y-override-v1",
        "startup_cycles_source": (
            "frontend-semantic-dfg-unit-latency-critical-path-v1"
        ),
    }
    result = {
        "schema_version": ANALYTICAL_INPUT_SCHEMA,
        "function": header["function"],
        "provenance": provenance,
        "entries": entries,
    }
    timing = {
        "query_count": len(entries),
        "task_count": len(tasks),
        "analysis_invocation_count": len(commands),
        "startup_cycles_by_task": startup_by_task,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    return result, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--task-dfg", action="append", default=[], metavar="TASK=PATH",
    )
    parser.add_argument("--neura-opt", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result, timing = generate_query_features(
        args.manifest.resolve(), parse_task_paths(args.task_dfg),
        args.neura_opt.resolve(), args.architecture.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.timing_output is not None:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(
            json.dumps(timing, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps({"output": str(args.output.resolve()), **timing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
