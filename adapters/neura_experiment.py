#!/usr/bin/env python3
"""Neura adapter for reproducible compiled-II prediction experiments.

This adapter is deliberately offline.  It does not change the mapper or
the analytical lower bound: labels are the existing heuristic mapper's
``compiled_ii``, while features are available before mapping from the Neura
DFG and YAML architecture.  The script writes all generated inputs and labels
under its requested output directory (``/tmp`` by default).

The random generator is intentionally constrained to valid lowered Neura DFGs
whose operations have a clear tile-domain contract.  It varies graph shape,
fanout, depth, tile count, and homogeneous versus split FU domains.  Those
examples are a corpus supplement, not a claim that synthetic graphs replace
real frontend kernels.  Evaluation is grouped by kernel family: a model may
not train on one shape or mask of a kernel and claim to predict another shape
of that same kernel.  A random row split is retained only as a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


DATA_TYPE = "!neura.data<i32, i1>"
FEATURE_NAMES = (
    "baseline_lb",
    "rec_mii",
    "res_mii",
    "placement_lower_bound",
    "mem_mii",
    "route_lower_bound",
    "reg_mii",
    "analytical_ii",
    "rec_res_gap",
    "rec_placement_gap",
    "rec_dominant",
    "analytical_excess",
    "route_excess",
    "reg_excess",
    "issue_mii",
    "route_mii",
    "nodes",
    "moves",
    "ctrl_moves",
    "reserves",
    "phis",
    "predicates",
    "memory_ops",
    "gep_ops",
    "indirect_geps",
    "pointer_loads",
    "div_ops",
    "arithmetic_ops",
    "comparisons",
    "sources",
    "depth",
    "max_fanout",
    "semantic_edges",
    "semantic_depth",
    "semantic_width",
    "semantic_max_fanout",
    "semantic_branch_nodes",
    "semantic_cutwidth",
    "live_value_peak",
    "pointer_path",
    "memory_path",
    "control_path",
    "multi_input_nodes",
    "tiles",
    "links",
    "rows",
    "split_domain",
)

# The complete feature record above is useful for corpus analysis.  The model
# deliberately uses a smaller, physically motivated subset: with only fifteen
# independent real kernel families, fitting all correlated counters reduced
# unseen-family accuracy.  Architecture still enters through the three proven
# bounds, whose values are recomputed for each YAML/shape.
MODEL_FEATURE_NAMES = (
    "baseline_lb",
    "rec_mii",
    "res_mii",
    "placement_lower_bound",
    "semantic_edges",
    "semantic_depth",
    "semantic_width",
    "semantic_max_fanout",
    "semantic_branch_nodes",
    "semantic_cutwidth",
    "live_value_peak",
    "pointer_path",
    "memory_path",
    "control_path",
    "multi_input_nodes",
    "memory_ops",
    "gep_ops",
    "indirect_geps",
    "pointer_loads",
)

Sample = Dict[str, Any]
COST_FEATURE_NAMES = (
    "rec_mii",
    "res_mii",
    "mem_mii",
    "placement_lower_bound",
    "issue_mii",
    "route_mii",
    "route_lower_bound",
    "reg_mii",
    "analytical_ii",
)


@dataclass
class SampleSpec:
    index: int
    seed: int
    rows: int
    columns: int
    split_domain: int
    sources: int
    operations: int
    fanout_bias: float
    registers: int


@dataclass
class CSpec:
    index: int
    seed: int
    rows: int
    columns: int
    terms: int
    loop_trip_count: int


def resolve_default_opt(repo: Path) -> Path:
    return repo / "build/tools/mlir-neura-opt/mlir-neura-opt"


def resolve_default_llvm_tool(name: str) -> Path:
    resolved = shutil.which(name)
    return Path(resolved) if resolved else Path(name)


def write_architecture(path: Path, spec: SampleSpec) -> None:
    if spec.split_domain:
        tile_defaults = '["constant"]'
        first_compute_column = max(1, spec.columns // 2)
        overrides = "\n".join(
            "  - tile_x: {x}\n"
            "    tile_y: {y}\n"
            "    fu_types: [\"add\", \"mul\"]\n"
            "    num_registers: {regs}\n"
            "    existence: true".format(x=x, y=y, regs=spec.registers)
            for y in range(spec.rows)
            for x in range(first_compute_column, spec.columns)
        )
    else:
        tile_defaults = '["constant", "add", "mul"]'
        overrides = ""
    path.write_text(
        "\n".join(
            (
                "architecture:",
                '  name: "II Predictor Synthetic"',
                '  version: "1.0"',
                "",
                "multi_cgra_defaults:",
                '  base_topology: "mesh"',
                "  rows: 1",
                "  columns: 1",
                "",
                "per_cgra_defaults:",
                f"  rows: {spec.rows}",
                f"  columns: {spec.columns}",
                "  ctrl_mem_items: 20",
                '  base_topology: "mesh"',
                "",
                "tile_defaults:",
                f"  num_registers: {spec.registers}",
                f"  fu_types: {tile_defaults}",
                "",
                "link_defaults:",
                "  latency: 1",
                "  bandwidth: 32",
                "",
                "link_overrides:",
                "",
                "tile_overrides:",
                overrides,
                "",
            )
        )
    )


def pick_parent(rng: random.Random, values: Sequence[str], use_count: Dict[str, int],
                fanout_bias: float) -> str:
    # Prefer values already used by consumers at a controlled rate.  This
    # creates routing fanout pressure while retaining random graph shapes.
    weighted: List[Tuple[str, float]] = []
    for value in values:
        weight = 1.0 + fanout_bias * min(use_count[value], 3)
        weighted.append((value, weight))
    threshold = rng.random() * sum(weight for _, weight in weighted)
    for value, weight in weighted:
        threshold -= weight
        if threshold <= 0:
            return value
    return weighted[-1][0]


def write_dfg(path: Path, spec: SampleSpec) -> Dict[str, int]:
    rng = random.Random(spec.seed)
    lines = ["module {", "  func.func @synthetic() attributes {accelerator = \"neura\"} {"]
    values: List[str] = []
    use_count: Dict[str, int] = {}
    for index in range(spec.sources):
        value = f"%c{index}"
        lines.append(
            f"    {value} = \"neura.constant\"() <{{value = {index + 1} : i32}}> "
            f": () -> {DATA_TYPE}"
        )
        values.append(value)
        use_count[value] = 0

    depth: Dict[str, int] = {value: 0 for value in values}
    moves = 0
    for index in range(spec.operations):
        parent = pick_parent(rng, values, use_count, spec.fanout_bias)
        move = f"%m{index}"
        result = f"%v{index}"
        operation = "add" if rng.random() < 0.55 else "mul"
        immediate = rng.randint(1, 31)
        lines.append(
            f"    {move} = \"neura.data_mov\"({parent}) : ({DATA_TYPE}) -> {DATA_TYPE}"
        )
        lines.append(
            f"    {result} = \"neura.{operation}\"({move}) {{rhs_value = {immediate} : i32}} "
            f": ({DATA_TYPE}) -> {DATA_TYPE}"
        )
        use_count[parent] += 1
        values.append(result)
        use_count[result] = 0
        depth[result] = depth[parent] + 1
        moves += 1

    lines.extend(("    func.return", "  }", "}", ""))
    path.write_text("\n".join(lines))
    return {
        "nodes": spec.sources + spec.operations,
        "moves": moves,
        "ctrl_moves": 0,
        "reserves": 0,
        "phis": 0,
        "predicates": 0,
        "memory_ops": 0,
        "gep_ops": 0,
        "indirect_geps": 0,
        "pointer_loads": 0,
        "div_ops": 0,
        "arithmetic_ops": spec.operations,
        "comparisons": 0,
        "sources": spec.sources,
        "depth": max(depth.values()),
        "max_fanout": max(use_count.values()),
        "tiles": spec.rows * spec.columns,
        "links": 2 * (spec.rows * (spec.columns - 1) +
                      spec.columns * (spec.rows - 1)),
        "rows": spec.rows,
        "split_domain": spec.split_domain,
    }


def write_c_loop(path: Path, spec: CSpec) -> None:
    """Write a bounded memory/recurrence loop for frontend corpus coverage."""
    rng = random.Random(spec.seed)
    statements = [f"    int value_{index} = in{index % 3}[i];"
                  for index in range(spec.terms)]
    for index in range(spec.terms):
        coefficient = rng.randint(2, 17)
        # Restrict the generator to frontend operations accepted by the Neura
        # lowering pipeline (add/mul); unsupported LLVM opcodes are corpus
        # failures, not zero-II samples.
        statements.append(
            f"    state = state + value_{index} * {coefficient};")
    body = "\n".join(statements)
    path.write_text(
        "\n".join((
            'extern "C" void kernel(int *out, const int *in0,',
            '                       const int *in1, const int *in2) {',
            f"  int state = {rng.randint(0, 31)};",
            f"  for (int i = 0; i < {spec.loop_trip_count}; ++i) {{",
            body,
            "    out[i] = state;",
            "  }",
            "}",
            "",
        ))
    )


def invoke(command: Sequence[str], timeout: int) -> bool:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def git_provenance(root: Optional[Path]) -> Dict[str, object]:
    if root is None:
        return {"root": None, "revision": None, "dirty": None}
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    return {
        "root": str(root.resolve()),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_integer_attribute(text: str, name: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i32", text)
    return int(match.group(1)) if match else None


def load_sibling_cost_features(report_path: Path) -> Dict[str, Dict[str, int]]:
    """Recover full features from cost artifacts beside a legacy report.

    The initial experiment reports predated several prediction-only features.
    Their generated cost.mlir artifacts are still the source of truth, so this
    reconstruction avoids relabelling (or silently filling missing features).
    """
    result: Dict[str, Dict[str, int]] = {}
    for cost_path in report_path.parent.glob("real-*/cost.mlir"):
        directory = cost_path.parent.name
        standard = re.fullmatch(r"real-(.+)-(\d+x\d+)", directory)
        mask = re.fullmatch(r"real-(.+)-mask-(\d{3})", directory)
        if standard:
            index = f"{standard.group(1)}-{standard.group(2)}"
        elif mask:
            index = f"{mask.group(1)}-4x4-mask{mask.group(2)}"
        else:
            continue
        text = cost_path.read_text()
        values = {
            name: parse_integer_attribute(text, name)
            for name in COST_FEATURE_NAMES
        }
        if all(value is not None for value in values.values()):
            result[index] = {name: int(value) for name, value in values.items()}
    return result


def add_prediction_features(result: Sample) -> None:
    """Add only pre-mapping features; this never changes a mapping decision."""
    result["baseline_lb"] = max(
        result["rec_mii"], result["res_mii"], result["placement_lower_bound"]
    )
    result["rec_res_gap"] = result["rec_mii"] - result["res_mii"]
    result["rec_placement_gap"] = (
        result["rec_mii"] - result["placement_lower_bound"]
    )
    result["rec_dominant"] = int(
        result["rec_mii"] >= result["res_mii"] and
        result["rec_mii"] >= result["placement_lower_bound"]
    )
    # The following are deliberately *prediction* features.  RouteMII and
    # RegMII are not pruning bounds, but may still correlate with the II at
    # which this particular heuristic finds a mapping.
    if all(name in result for name in
           ("analytical_ii", "route_mii", "reg_mii")):
        result["analytical_excess"] = (
            result["analytical_ii"] - result["baseline_lb"]
        )
        result["route_excess"] = result["route_mii"] - result["baseline_lb"]
        result["reg_excess"] = result["reg_mii"] - result["baseline_lb"]


def collect_sample(opt: Path, sample_dir: Path, spec: SampleSpec,
                   timeout: int) -> Optional[Sample]:
    source = sample_dir / "input.mlir"
    arch = sample_dir / "architecture.yaml"
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    graph_features = write_dfg(source, spec)
    graph_features.update(semantic_features_from_neura(source.read_text()))
    write_architecture(arch, spec)

    if not invoke(
        (str(opt), str(source), f"--architecture-spec={arch}",
         "--cost-model-analytical", "-o", str(cost)),
        timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={arch}",
         '--map-to-accelerator=mapping-strategy=heuristic', "-o", str(mapped)),
        timeout,
    ):
        return None

    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = {
        "rec_mii": parse_integer_attribute(cost_text, "rec_mii"),
        "res_mii": parse_integer_attribute(cost_text, "res_mii"),
        "mem_mii": parse_integer_attribute(cost_text, "mem_mii"),
        "placement_lower_bound": parse_integer_attribute(
            cost_text, "placement_lower_bound"),
        "issue_mii": parse_integer_attribute(cost_text, "issue_mii"),
        "route_mii": parse_integer_attribute(cost_text, "route_mii"),
        "route_lower_bound": parse_integer_attribute(
            cost_text, "route_lower_bound"),
        "reg_mii": parse_integer_attribute(cost_text, "reg_mii"),
        "analytical_ii": parse_integer_attribute(cost_text, "analytical_ii"),
        "compiled_ii": parse_integer_attribute(mapped_text, "compiled_ii"),
    }
    if any(value is None for value in values.values()):
        return None
    result: Sample = {key: int(value) for key, value in values.items()}
    result.update(graph_features)
    add_prediction_features(result)
    result.update(asdict(spec))
    result["family"] = "synthetic"
    return result


def collect_c_sample(opt: Path, sample_dir: Path, spec: CSpec,
                     architecture: Path, cxx: str, llvm_extract: Path,
                     mlir_translate: Path, timeout: int) -> Optional[Sample]:
    """Compile a generated C loop through the normal frontend and mapper."""
    source = sample_dir / "input.cpp"
    full_ir = sample_dir / "full.ll"
    kernel_ir = sample_dir / "kernel.ll"
    imported = sample_dir / "imported.mlir"
    lowered = sample_dir / "lowered.mlir"
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    write_c_loop(source, spec)
    if not invoke(
        (cxx, "-S", "-emit-llvm", "-O3", "-fno-vectorize",
         "-fno-unroll-loops", "-std=c++17", "-o", str(full_ir), str(source)),
        timeout,
    ):
        return None
    if not invoke(
        (str(llvm_extract), "--rfunc=.*kernel.*", str(full_ir), "-o",
         str(kernel_ir)), timeout,
    ):
        return None
    if not invoke(
        (str(mlir_translate), "--import-llvm", str(kernel_ir), "-o",
         str(imported)), timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(imported), "--assign-accelerator",
         "--lower-llvm-to-neura", "--promote-input-arg-to-const",
         "--fold-constant", "--canonicalize-return",
         "--canonicalize-live-in", "--leverage-predicated-value",
         "--transform-ctrl-to-data-flow", "--fold-constant",
         "--insert-data-mov", "-o", str(lowered)), timeout,
    ):
        return None
    options = f"x-tiles={spec.columns} y-tiles={spec.rows}"
    if not invoke(
        (str(opt), str(lowered), f"--architecture-spec={architecture}",
         f"--cost-model-analytical={options}", "-o", str(cost)), timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(lowered), f"--architecture-spec={architecture}",
         f"--map-to-accelerator=mapping-strategy=heuristic {options}",
         "-o", str(mapped)), timeout,
    ):
        return None
    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = {
        "rec_mii": parse_integer_attribute(cost_text, "rec_mii"),
        "res_mii": parse_integer_attribute(cost_text, "res_mii"),
        "mem_mii": parse_integer_attribute(cost_text, "mem_mii"),
        "placement_lower_bound": parse_integer_attribute(
            cost_text, "placement_lower_bound"),
        "issue_mii": parse_integer_attribute(cost_text, "issue_mii"),
        "route_mii": parse_integer_attribute(cost_text, "route_mii"),
        "route_lower_bound": parse_integer_attribute(
            cost_text, "route_lower_bound"),
        "reg_mii": parse_integer_attribute(cost_text, "reg_mii"),
        "analytical_ii": parse_integer_attribute(cost_text, "analytical_ii"),
        "compiled_ii": parse_integer_attribute(mapped_text, "compiled_ii"),
    }
    if any(value is None for value in values.values()):
        return None
    result: Sample = {key: int(value) for key, value in values.items()}
    result.update(graph_features_from_neura(lowered.read_text(), spec.rows,
                                            spec.columns))
    add_prediction_features(result)
    result.update(asdict(spec))
    result["family"] = "synthetic-c"
    return result


def semantic_features_from_neura(text: str) -> Dict[str, int]:
    """Features invariant under a tile-shape or valid-tile-mask change."""
    ignored = {"data_mov", "reserve", "ctrl_mov", "yield"}
    op_names = re.findall(
        r'(?m)^\s*(?:%[A-Za-z0-9_]+\s*=\s*)?"?neura\.([a-z_]+)', text
    )
    materialized = [name for name in op_names if name not in ignored]
    moves = sum(name == "data_mov" for name in op_names)

    # SSA use counts give a stable fanout feature without needing to reimplement
    # MLIR parsing in the experiment harness.  Count only values defined by a
    # result-producing operation, then discount that definition occurrence.
    definitions = re.findall(r"(?m)^\s*(%[A-Za-z0-9_]+)\s*=", text)
    token_counts: Dict[str, int] = {}
    for token in re.findall(r"%[A-Za-z0-9_]+", text):
        token_counts[token] = token_counts.get(token, 0) + 1
    max_fanout = max((token_counts[value] - 1 for value in definitions), default=0)

    depths: Dict[str, int] = {}
    defining_kind: Dict[str, str] = {}
    defining_operands: Dict[str, List[str]] = {}
    definition_lines: Dict[str, int] = {}
    pointer_results: Dict[str, bool] = {}
    use_lines: Dict[str, List[int]] = {}
    indirect_geps = 0
    pointer_loads = 0
    lines = text.splitlines()
    for line_number, line in enumerate(lines):
        for token in re.findall(r"%[A-Za-z0-9_]+", line):
            use_lines.setdefault(token, []).append(line_number)
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if not match:
            continue
        value, expression = match.groups()
        kind_match = re.search(r'"?neura\.([a-z_]+)', expression)
        kind = kind_match.group(1) if kind_match else ""
        operands = re.findall(r"%[A-Za-z0-9_]+", expression)
        depths[value] = 1 + max((depths.get(operand, 0) for operand in operands),
                                default=0)
        defining_kind[value] = kind
        defining_operands[value] = operands
        definition_lines[value] = line_number
        pointer_results[value] = "-> !neura.data<!llvm.ptr" in expression

        def comes_from_pointer_load(operand: str) -> bool:
            # InsertDataMov wraps nearly every data dependency.  Follow those
            # wrappers so a load -> data_mov -> gep chain remains visible.
            seen: Set[str] = set()
            while operand not in seen:
                seen.add(operand)
                kind = defining_kind.get(operand)
                if kind == "load":
                    return True
                if kind != "data_mov":
                    return False
                move_operands = defining_operands.get(operand, [])
                if len(move_operands) != 1:
                    return False
                operand = move_operands[0]
            return False

        if kind == "gep" and any(comes_from_pointer_load(operand)
                                 for operand in operands):
            indirect_geps += 1
        if kind == "load" and "-> !neura.data<!llvm.ptr" in expression:
            pointer_loads += 1

    # Collapse inserted data_mov operations before measuring graph structure.
    # Raw operation counts otherwise make the same dependency look artificially
    # deeper merely because lowering materialized transport operations.
    transparent = {"data_mov", "reserve"}

    def meaningful_roots(value: str, seen: Optional[Set[str]] = None) -> List[str]:
        if value not in defining_kind:
            return []
        if defining_kind[value] not in transparent:
            return [value]
        visited = set() if seen is None else seen
        if value in visited:
            return []
        visited.add(value)
        roots: List[str] = []
        for operand in defining_operands.get(value, []):
            roots.extend(meaningful_roots(operand, visited))
        return roots

    semantic_depths: Dict[str, int] = {}
    memory_depths: Dict[str, int] = {}
    pointer_depths: Dict[str, int] = {}
    control_depths: Dict[str, int] = {}
    depth_widths: Dict[int, int] = {}
    semantic_fanout: Dict[str, int] = {}
    semantic_edges: List[Tuple[int, int]] = []
    multi_input_nodes = 0
    memory_kinds = {"gep", "load", "store", "memset"}
    control_kinds = {
        "grant_predicate", "phi", "phi_start", "not", "icmp", "fcmp"
    }
    for value in definition_lines:
        kind = defining_kind[value]
        parents: List[str] = []
        for operand in defining_operands.get(value, []):
            parents.extend(meaningful_roots(operand))
        parents = list(dict.fromkeys(parents))
        semantic_depths[value] = max(
            (semantic_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind not in transparent)
        memory_depths[value] = max(
            (memory_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind in memory_kinds)
        pointer_depths[value] = max(
            (pointer_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind == "gep" or (kind == "load" and pointer_results[value]))
        control_depths[value] = max(
            (control_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind in control_kinds)
        if kind in transparent:
            continue
        depth = semantic_depths[value]
        depth_widths[depth] = depth_widths.get(depth, 0) + 1
        multi_input_nodes += int(len(parents) > 1)
        for parent in parents:
            semantic_fanout[parent] = semantic_fanout.get(parent, 0) + 1
            semantic_edges.append(
                (definition_lines[parent], definition_lines[value]))

    last_definition = max(definition_lines.values(), default=0)
    semantic_cutwidth = max((
        sum(start <= position < end for start, end in semantic_edges)
        for position in range(last_definition + 1)
    ), default=0)
    live_value_peak = max((
        sum(
            start <= position < max(use_lines.get(value, [start]))
            for value, start in definition_lines.items()
        )
        for position in range(last_definition + 1)
    ), default=0)

    return {
        "nodes": len(materialized),
        "moves": moves,
        "ctrl_moves": sum(name == "ctrl_mov" for name in op_names),
        "reserves": sum(name == "reserve" for name in op_names),
        "phis": sum(name in {"phi", "phi_start"} for name in op_names),
        "predicates": sum(name == "grant_predicate" for name in op_names),
        "memory_ops": sum(name in {"load", "store", "memset"}
                          for name in op_names),
        "gep_ops": sum(name == "gep" for name in op_names),
        "indirect_geps": indirect_geps,
        "pointer_loads": pointer_loads,
        "div_ops": sum(name in {"div", "rem"} for name in op_names),
        "arithmetic_ops": sum(name in {"add", "sub", "mul", "fmul_fadd", "shl"}
                              for name in op_names),
        "comparisons": sum(name in {"icmp", "fcmp"} for name in op_names),
        "sources": sum(name in {"constant", "grant_once", "grant_always"}
                       for name in materialized),
        "depth": max(depths.values(), default=0),
        "max_fanout": max_fanout,
        "semantic_edges": len(semantic_edges),
        "semantic_depth": max(semantic_depths.values(), default=0),
        "semantic_width": max(depth_widths.values(), default=0),
        "semantic_max_fanout": max(semantic_fanout.values(), default=0),
        "semantic_branch_nodes": sum(
            fanout > 1 for fanout in semantic_fanout.values()),
        "semantic_cutwidth": semantic_cutwidth,
        "live_value_peak": live_value_peak,
        "pointer_path": max(pointer_depths.values(), default=0),
        "memory_path": max(memory_depths.values(), default=0),
        "control_path": max(control_depths.values(), default=0),
        "multi_input_nodes": multi_input_nodes,
    }


def graph_features_from_neura(
    text: str, rows: int, columns: int,
    valid_tiles: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[str, int]:
    """Extract mapper-visible structural features from already-lowered IR."""
    active_tiles = valid_tiles or {
        (x, y) for y in range(rows) for x in range(columns)
    }
    adjacent_pairs = sum(
        ((x + 1, y) in active_tiles) + ((x, y + 1) in active_tiles)
        for x, y in active_tiles
    )
    result = semantic_features_from_neura(text)
    result.update({
        "tiles": len(active_tiles),
        "links": 2 * adjacent_pairs,
        "rows": rows,
        # The production YAML has heterogeneous memory tiles, but not the
        # source/compute partition used by the synthetic split-domain mode.
        "split_domain": 0,
    })
    return result


def collect_real_fixture(opt: Path, sample_dir: Path, name: str, source: Path,
                         architecture: Path, rows: int, columns: int,
                         timeout: int,
                         valid_tiles: Optional[Set[Tuple[int, int]]] = None,
                         suffix: str = "") -> Optional[Sample]:
    """Label a pre-lowered real DFG at a selected architectural shape."""
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if valid_tiles is not None:
        valid_tile_text = ",".join(
            f"{x}_{y}" for x, y in sorted(valid_tiles, key=lambda tile: (tile[1], tile[0]))
        )
        options += f" valid-tiles={valid_tile_text}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--cost-model-analytical={options}", "-o", str(cost)),
        timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--map-to-accelerator=mapping-strategy=heuristic {options}",
         "-o", str(mapped)),
        timeout,
    ):
        return None

    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = {
        "rec_mii": parse_integer_attribute(cost_text, "rec_mii"),
        "res_mii": parse_integer_attribute(cost_text, "res_mii"),
        "mem_mii": parse_integer_attribute(cost_text, "mem_mii"),
        "placement_lower_bound": parse_integer_attribute(
            cost_text, "placement_lower_bound"),
        "issue_mii": parse_integer_attribute(cost_text, "issue_mii"),
        "route_mii": parse_integer_attribute(cost_text, "route_mii"),
        "route_lower_bound": parse_integer_attribute(
            cost_text, "route_lower_bound"),
        "reg_mii": parse_integer_attribute(cost_text, "reg_mii"),
        "analytical_ii": parse_integer_attribute(cost_text, "analytical_ii"),
        "compiled_ii": parse_integer_attribute(mapped_text, "compiled_ii"),
    }
    if any(value is None for value in values.values()):
        return None
    result: Sample = {key: int(value) for key, value in values.items()}
    result.update(graph_features_from_neura(source.read_text(), rows, columns,
                                            valid_tiles))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}{suffix}"
    result["family"] = name
    return result


def collect_completed_real_fixture(
    opt: Path, sample_dir: Path, name: str, source: Path, mapped: Path,
    architecture: Path, rows: int, columns: int, timeout: int,
) -> Optional[Sample]:
    """Reuse a completed heuristic mapping artifact as a real label.

    This is useful for expensive kernels: the mapping must have completed in a
    previous invocation of *this* heuristic mapper.  We still rerun the cheap
    analytical pass, and reject artifacts without a heuristic `compiled_ii`.
    """
    cost = sample_dir / "cost.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--cost-model-analytical={options}", "-o", str(cost)),
        timeout,
    ):
        return None
    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    if 'mapping_strategy = "heuristic"' not in mapped_text:
        return None
    values = {
        "rec_mii": parse_integer_attribute(cost_text, "rec_mii"),
        "res_mii": parse_integer_attribute(cost_text, "res_mii"),
        "mem_mii": parse_integer_attribute(cost_text, "mem_mii"),
        "placement_lower_bound": parse_integer_attribute(
            cost_text, "placement_lower_bound"),
        "issue_mii": parse_integer_attribute(cost_text, "issue_mii"),
        "route_mii": parse_integer_attribute(cost_text, "route_mii"),
        "route_lower_bound": parse_integer_attribute(
            cost_text, "route_lower_bound"),
        "reg_mii": parse_integer_attribute(cost_text, "reg_mii"),
        "analytical_ii": parse_integer_attribute(cost_text, "analytical_ii"),
        "compiled_ii": parse_integer_attribute(mapped_text, "compiled_ii"),
    }
    if any(value is None for value in values.values()):
        return None
    result: Sample = {key: int(value) for key, value in values.items()}
    result.update(graph_features_from_neura(source.read_text(), rows, columns))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}"
    result["family"] = name
    return result


def collect_prediction_fixture(
    opt: Path, sample_dir: Path, name: str, source: Path,
    architecture: Path, rows: int, columns: int, timeout: int,
) -> Optional[Sample]:
    """Extract pre-mapping prediction features without invoking the mapper."""
    cost = sample_dir / "cost.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--cost-model-analytical={options}", "-o", str(cost)), timeout,
    ):
        return None
    cost_text = cost.read_text()
    values = {
        field: parse_integer_attribute(cost_text, field)
        for field in COST_FEATURE_NAMES
    }
    if any(value is None for value in values.values()):
        return None
    result: Sample = {key: int(value) for key, value in values.items()}
    result.update(graph_features_from_neura(source.read_text(), rows, columns))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}"
    result["family"] = name
    return result


def random_connected_tile_masks(rng: random.Random, count: int,
                                min_tiles: int) -> List[Set[Tuple[int, int]]]:
    """Generate unique connected 4x4 shape masks for real-DFG labeling."""
    masks: List[Set[Tuple[int, int]]] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    attempts = 0
    while len(masks) < count and attempts < count * 50:
        attempts += 1
        target_size = rng.randint(min_tiles, 15)
        mask = {(rng.randrange(4), rng.randrange(4))}
        while len(mask) < target_size:
            frontier = []
            for x, y in mask:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < 4 and 0 <= neighbor[1] < 4 and neighbor not in mask:
                        frontier.append(neighbor)
            if not frontier:
                break
            mask.add(rng.choice(frontier))
        key = tuple(sorted(mask))
        if len(mask) == target_size and key not in seen:
            seen.add(key)
            masks.append(mask)
    return masks


def fit_ridge(train: Sequence[Sample], ridge: float,
              residual_dead_zone: float = 0.0) -> Dict[str, object]:
    """Fit a ridge model for non-negative heuristic-II residuals.

    ``residual_dead_zone`` suppresses small positive residuals after the fit.
    It is a prediction calibration parameter, not a mapping lower bound.
    """
    if not train:
        raise ValueError("cannot fit an empty training set")
    if residual_dead_zone < 0.0:
        raise ValueError("residual dead zone must be non-negative")
    x_train = np.asarray(
        [[row[name] for name in MODEL_FEATURE_NAMES] for row in train],
        dtype=float,
    )
    y_train = np.asarray(
        [row["compiled_ii"] - row["baseline_lb"] for row in train], dtype=float
    )
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (x_train - mean) / scale
    design = np.column_stack((np.ones(len(train)), normalized))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)

    return {
        "feature_names": list(MODEL_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "ridge": ridge,
        "residual_dead_zone": residual_dead_zone,
    }


def predict_ridge(model: Dict[str, object], row: Sample) -> float:
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    feature_names = model["feature_names"]
    vector = np.asarray([row[name] for name in feature_names], dtype=float)
    residual = max(0.0, weights[0] + ((vector - mean) / scale) @ weights[1:])
    if residual < float(model.get("residual_dead_zone", 0.0)):
        residual = 0.0
    return row["baseline_lb"] + float(residual)


def residual_sse(rows: Sequence[Sample]) -> float:
    if not rows:
        return 0.0
    residuals = np.asarray(
        [row["compiled_ii"] - row["baseline_lb"] for row in rows], dtype=float
    )
    return float(((residuals - residuals.mean()) ** 2).sum())


def fit_residual_tree(train: Sequence[Sample], max_depth: int,
                      min_samples: int) -> Dict[str, object]:
    """Fit a deterministic, dependency-free shallow regression tree.

    The tree predicts the residual above the proven lower bound.  It is a
    reporting experiment only; no compiler code reads this model.
    """
    if not train:
        raise ValueError("cannot fit an empty training set")

    def build(rows: Sequence[Sample], depth: int) -> Dict[str, object]:
        target = sum(row["compiled_ii"] - row["baseline_lb"] for row in rows)
        leaf = max(0.0, target / len(rows))
        parent_sse = residual_sse(rows)
        if depth >= max_depth or len(rows) < 2 * min_samples or parent_sse == 0.0:
            return {"value": leaf, "count": len(rows)}

        best: Optional[Tuple[float, int, float, List[Sample], List[Sample]]] = None
        for feature_index, name in enumerate(FEATURE_NAMES):
            values = sorted({float(row[name]) for row in rows})
            for low, high in zip(values, values[1:]):
                threshold = (low + high) / 2.0
                left = [row for row in rows if float(row[name]) <= threshold]
                right = [row for row in rows if float(row[name]) > threshold]
                if len(left) < min_samples or len(right) < min_samples:
                    continue
                child_sse = residual_sse(left) + residual_sse(right)
                candidate = (child_sse, feature_index, threshold, left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None or best[0] >= parent_sse:
            return {"value": leaf, "count": len(rows)}
        _, feature_index, threshold, left, right = best
        return {
            "feature": FEATURE_NAMES[feature_index],
            "threshold": threshold,
            "count": len(rows),
            "left": build(left, depth + 1),
            "right": build(right, depth + 1),
        }

    return {
        "feature_names": list(FEATURE_NAMES),
        "max_depth": max_depth,
        "min_samples": min_samples,
        "tree": build(list(train), 0),
    }


def predict_residual_tree(model: Dict[str, object], row: Sample) -> float:
    node = model["tree"]
    while "feature" in node:
        node = node["left"] if float(row[node["feature"]]) <= node["threshold"] else node["right"]
    return row["baseline_lb"] + max(0.0, float(node["value"]))


def prediction_rows(test: Sequence[Sample], predictor: Callable[[Sample], float]) -> List[Dict[str, object]]:
    return [
        {
            "sample": row["index"],
            "family": row["family"],
            "baseline": row["baseline_lb"],
            "prediction": predictor(row),
            "compiled_ii": row["compiled_ii"],
        }
        for row in test
    ]


def random_row_holdout(samples: Sequence[Sample], seed: int, ridge: float,
                       tree_depth: int, tree_min_samples: int) -> Dict[str, object]:
    """Diagnostic only: rows from one kernel may appear on both sides."""
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    split = max(1, int(len(ordered) * 0.8))
    train, test = ordered[:split], ordered[split:]
    ridge_model = fit_ridge(train, ridge)
    tree_model = fit_residual_tree(train, tree_depth, tree_min_samples)
    ridge_rows = prediction_rows(test, lambda row: predict_ridge(ridge_model, row))
    tree_rows = prediction_rows(
        test, lambda row: predict_residual_tree(tree_model, row))
    return {
        "train_size": len(train),
        "ridge_model": ridge_model,
        "tree_model": tree_model,
        "ridge_rows": ridge_rows,
        "tree_rows": tree_rows,
        "baseline_mae": mean_absolute_error(ridge_rows, "baseline"),
        "ridge_mae": mean_absolute_error(ridge_rows, "prediction"),
        "tree_mae": mean_absolute_error(tree_rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(ridge_rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(ridge_rows, "prediction"),
        "tree_macro_family_mae": macro_family_mae(tree_rows, "prediction"),
    }


def leave_one_family_out(samples: Sequence[Sample], ridge: float,
                         tree_depth: int, tree_min_samples: int) -> Dict[str, object]:
    """Evaluate generalization to an unseen kernel family without leakage."""
    families = sorted({str(row["family"]) for row in samples})
    ridge_rows: List[Dict[str, object]] = []
    tree_rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for family in families:
        test = [row for row in samples if row["family"] == family]
        train = [row for row in samples if row["family"] != family]
        if len(train) < max(2, 2 * tree_min_samples):
            skipped.append(family)
            continue
        ridge_model = fit_ridge(train, ridge)
        tree_model = fit_residual_tree(train, tree_depth, tree_min_samples)
        ridge_rows.extend(
            prediction_rows(test, lambda row: predict_ridge(ridge_model, row)))
        tree_rows.extend(
            prediction_rows(
                test, lambda row: predict_residual_tree(tree_model, row)))
    if not ridge_rows:
        raise ValueError("not enough independent kernel families for family holdout")
    result: Dict[str, object] = {
        "families": families,
        "skipped_families": skipped,
        "ridge_rows": ridge_rows,
        "tree_rows": tree_rows,
        "baseline_mae": mean_absolute_error(ridge_rows, "baseline"),
        "ridge_mae": mean_absolute_error(ridge_rows, "prediction"),
        "tree_mae": mean_absolute_error(tree_rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(ridge_rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(ridge_rows, "prediction"),
        "tree_macro_family_mae": macro_family_mae(tree_rows, "prediction"),
    }
    real_ridge_rows = [
        row for row in ridge_rows
        if not str(row["family"]).startswith("synthetic")
    ]
    real_tree_rows = [
        row for row in tree_rows
        if not str(row["family"]).startswith("synthetic")
    ]
    if real_ridge_rows:
        result.update({
            "real_baseline_mae": mean_absolute_error(real_ridge_rows, "baseline"),
            "real_ridge_mae": mean_absolute_error(real_ridge_rows, "prediction"),
            "real_tree_mae": mean_absolute_error(real_tree_rows, "prediction"),
            "real_baseline_macro_family_mae": macro_family_mae(
                real_ridge_rows, "baseline"),
            "real_ridge_macro_family_mae": macro_family_mae(
                real_ridge_rows, "prediction"),
            "real_tree_macro_family_mae": macro_family_mae(
                real_tree_rows, "prediction"),
        })
    return result


def add_real_holdout_metrics(result: Dict[str, object],
                             ridge_rows: Sequence[Dict[str, object]],
                             tree_rows: Optional[Sequence[Dict[str, object]]] = None) -> None:
    """Add micro/macro metrics after excluding generated synthetic families."""
    real_ridge_rows = [
        row for row in ridge_rows
        if not str(row["family"]).startswith("synthetic")
    ]
    if not real_ridge_rows:
        return
    result.update({
        "real_baseline_mae": mean_absolute_error(real_ridge_rows, "baseline"),
        "real_ridge_mae": mean_absolute_error(real_ridge_rows, "prediction"),
        "real_baseline_macro_family_mae": macro_family_mae(
            real_ridge_rows, "baseline"),
        "real_ridge_macro_family_mae": macro_family_mae(
            real_ridge_rows, "prediction"),
    })
    if tree_rows is not None:
        real_tree_rows = [
            row for row in tree_rows
            if not str(row["family"]).startswith("synthetic")
        ]
        result.update({
            "real_tree_mae": mean_absolute_error(real_tree_rows, "prediction"),
            "real_tree_macro_family_mae": macro_family_mae(
                real_tree_rows, "prediction"),
        })


def select_ridge_hyperparameters(
    train: Sequence[Sample], ridge_candidates: Sequence[float],
    dead_zone_candidates: Sequence[float],
) -> Tuple[float, float]:
    """Choose Ridge calibration only from inner family holdout predictions."""
    families = sorted({str(row["family"]) for row in train})
    if len(families) < 2:
        return (ridge_candidates[len(ridge_candidates) // 2],
                dead_zone_candidates[len(dead_zone_candidates) // 2])
    best: Optional[Tuple[float, float, float, float]] = None
    for ridge in ridge_candidates:
        raw_predictions: List[Tuple[Sample, Dict[str, object]]] = []
        for family in families:
            inner_train = [row for row in train if row["family"] != family]
            inner_test = [row for row in train if row["family"] == family]
            if not inner_train:
                continue
            model = fit_ridge(inner_train, ridge)
            raw_predictions.extend((row, model) for row in inner_test)
        for dead_zone in dead_zone_candidates:
            predictions: List[Dict[str, object]] = []
            for row, model in raw_predictions:
                calibrated = dict(model)
                calibrated["residual_dead_zone"] = dead_zone
                predictions.extend(prediction_rows(
                    [row], lambda sample: predict_ridge(calibrated, sample)))
            score = (
                macro_family_mae(predictions, "prediction"),
                mean_absolute_error(predictions, "prediction"),
                ridge,
                dead_zone,
            )
            if best is None or score < best:
                best = score
    if best is None:
        raise ValueError("not enough samples for Ridge hyperparameter selection")
    return best[2], best[3]


def nested_ridge_family_holdout(samples: Sequence[Sample],
                                ridge_candidates: Sequence[float],
                                dead_zone_candidates: Sequence[float],
                                ) -> Dict[str, object]:
    """Outer family holdout with calibration chosen only from outer training."""
    families = sorted({str(row["family"]) for row in samples})
    rows: List[Dict[str, object]] = []
    chosen: Dict[str, Dict[str, float]] = {}
    for family in families:
        train = [row for row in samples if row["family"] != family]
        test = [row for row in samples if row["family"] == family]
        if len({str(row["family"]) for row in train}) < 2:
            continue
        ridge, dead_zone = select_ridge_hyperparameters(
            train, ridge_candidates, dead_zone_candidates)
        chosen[family] = {
            "ridge": ridge,
            "residual_dead_zone": dead_zone,
        }
        model = fit_ridge(train, ridge, dead_zone)
        rows.extend(prediction_rows(test, lambda row: predict_ridge(model, row)))
    if not rows:
        raise ValueError("not enough independent kernel families for nested ridge")
    result: Dict[str, object] = {
        "rows": rows,
        "chosen_hyperparameters_by_held_out_family": chosen,
        "baseline_mae": mean_absolute_error(rows, "baseline"),
        "ridge_mae": mean_absolute_error(rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(rows, "prediction"),
    }
    add_prediction_quality_metrics(result, rows, "baseline")
    add_prediction_quality_metrics(result, rows, "ridge", "prediction")
    add_real_holdout_metrics(result, rows)
    return result


def mean_absolute_error(rows: Iterable[Dict[str, object]], prediction: str) -> float:
    rows = list(rows)
    return sum(
        abs(float(row[prediction]) - float(row["compiled_ii"])) for row in rows
    ) / len(rows)


def add_prediction_quality_metrics(
    result: Dict[str, object], rows: Sequence[Dict[str, object]], prefix: str,
    prediction: Optional[str] = None,
) -> None:
    """Record integer and tail metrics in addition to continuous MAE."""
    key = prediction or prefix
    errors = [float(row[key]) - float(row["compiled_ii"]) for row in rows]
    rounded_errors = [
        round(float(row[key])) - int(row["compiled_ii"]) for row in rows
    ]
    result.update({
        f"{prefix}_mean_signed_error": sum(errors) / len(errors),
        f"{prefix}_rounded_mae": (
            sum(abs(error) for error in rounded_errors) / len(rounded_errors)
        ),
        f"{prefix}_rounded_exact_rate": (
            sum(error == 0 for error in rounded_errors) / len(rounded_errors)
        ),
        f"{prefix}_within_one_rate": (
            sum(abs(error) <= 1.0 for error in errors) / len(errors)
        ),
        f"{prefix}_max_absolute_error": max(abs(error) for error in errors),
    })


def macro_family_mae(rows: Iterable[Dict[str, object]], prediction: str) -> float:
    """Give each kernel family equal weight despite differing mask counts."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)
    return sum(
        mean_absolute_error(group_rows, prediction)
        for group_rows in grouped.values()
    ) / len(grouped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    configured_root = os.environ.get("NEURA_ROOT")
    parser.add_argument(
        "--neura-root", type=Path,
        default=Path(configured_root) if configured_root else None,
        help="Optional Neura checkout (or set NEURA_ROOT).",
    )
    parser.add_argument(
        "--opt", type=Path,
        help="mlir-neura-opt binary; defaults under --neura-root.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/neura-ii-predictor-corpus"))
    parser.add_argument(
        "--samples", type=int, default=0,
        help=("Number of generated synthetic DFGs to label. Defaults to zero so "
              "report reuse and --predict-fixture never invoke the mapper."),
    )
    parser.add_argument(
        "--random-c-samples", type=int, default=0,
        help=("Generate bounded C loops, lower them through the normal "
              "frontend, and label them with the heuristic mapper."),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--timeout", type=int, default=15,
                        help="Per cost/map invocation timeout in seconds")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--ridge-candidates", default="0.1,0.3,1,3,10,30",
        help=("Comma-separated positive ridge values considered only by the "
              "nested family-holdout selector."),
    )
    parser.add_argument(
        "--residual-dead-zone-candidates", default="0,0.25,0.5,0.75,1,1.5,2",
        help=("Comma-separated non-negative residual thresholds considered "
              "only by the nested family-holdout selector."),
    )
    parser.add_argument("--tree-depth", type=int, default=1,
                        help="Maximum depth of the residual regression tree")
    parser.add_argument("--tree-min-samples", type=int, default=3)
    parser.add_argument(
        "--input-report", action="append", default=[], type=Path,
        help=("Existing report.json whose labels should be reused. This allows "
              "evaluation without rerunning the mapper; duplicate sample IDs "
              "are deduplicated."),
    )
    parser.add_argument(
        "--feature-source", action="append", default=[], metavar="NAME=PATH",
        help=("Source DFG used to hydrate structural features of legacy input "
              "reports. Repeat for every real family whose report predates a "
              "new feature."),
    )
    parser.add_argument(
        "--real-fixture", action="append", default=[], metavar="NAME=PATH",
        help=("Pre-lowered Neura DFG to label with the existing mapper. May be "
              "specified more than once."),
    )
    parser.add_argument("--cxx", default="clang++",
                        help="C++ compiler used by --random-c-samples")
    parser.add_argument("--llvm-extract", type=Path,
                        default=resolve_default_llvm_tool("llvm-extract"))
    parser.add_argument("--mlir-translate", type=Path,
                        default=resolve_default_llvm_tool("mlir-translate"))
    parser.add_argument(
        "--mapped-real-fixture", action="append", default=[],
        metavar="NAME=SOURCE=MAPPED=ROWSxCOLS",
        help=("Completed heuristic mapping artifact to reuse as a label. The "
              "artifact must contain mapping_strategy=heuristic and compiled_ii."),
    )
    parser.add_argument(
        "--predict-fixture", action="append", default=[], metavar="NAME=PATH",
        help=("Pre-lowered Neura DFG to estimate without invoking the mapper. "
              "Requires a labelled corpus/report to train a model."),
    )
    parser.add_argument(
        "--predict-shape", action="append", default=[], metavar="ROWSxCOLS",
        help="Shape(s) for --predict-fixture (default: 4x4).",
    )
    parser.add_argument(
        "--real-shape", action="append", default=[],
        metavar="ROWSxCOLS",
        help="Shape(s) used for each --real-fixture (default: 3x3, 3x4, 4x4).",
    )
    parser.add_argument(
        "--real-architecture", type=Path,
        help=("Base YAML architecture used for real fixtures; defaults under "
              "--neura-root."),
    )
    parser.add_argument(
        "--real-random-masks", type=int, default=0,
        help=("Also label this many connected non-rectangular 4x4 shapes per "
              "real fixture."),
    )
    parser.add_argument("--real-mask-min-tiles", type=int, default=10)
    parser.add_argument("--real-mask-seed", type=int, default=20260903)
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output directory before generation")
    args = parser.parse_args()
    if args.opt is None:
        if args.neura_root is None:
            parser.error("provide --neura-root/NEURA_ROOT or an explicit --opt")
        args.opt = resolve_default_opt(args.neura_root)
    if args.real_architecture is None:
        if args.neura_root is None:
            parser.error(
                "provide --neura-root/NEURA_ROOT or --real-architecture"
            )
        args.real_architecture = (
            args.neura_root / "test/arch_spec/architecture.yaml"
        )
    return args


def main() -> int:
    args = parse_args()
    if not args.opt.is_file():
        raise SystemExit(f"mlir-neura-opt not found: {args.opt}")
    if args.random_c_samples < 0:
        raise SystemExit("--random-c-samples must be non-negative")
    if args.random_c_samples:
        for tool in (args.llvm_extract, args.mlir_translate):
            if not tool.is_file():
                raise SystemExit(f"frontend tool not found: {tool}")
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    if args.tree_depth < 0:
        raise SystemExit("--tree-depth must be non-negative")
    if args.tree_min_samples < 1:
        raise SystemExit("--tree-min-samples must be positive")
    try:
        ridge_candidates = sorted({
            float(value) for value in args.ridge_candidates.split(",") if value
        })
    except ValueError:
        raise SystemExit("--ridge-candidates must be comma-separated numbers")
    if not ridge_candidates or any(value <= 0 for value in ridge_candidates):
        raise SystemExit("--ridge-candidates must contain positive values")
    try:
        dead_zone_candidates = sorted({
            float(value)
            for value in args.residual_dead_zone_candidates.split(",")
            if value
        })
    except ValueError:
        raise SystemExit(
            "--residual-dead-zone-candidates must be comma-separated numbers"
        )
    if (not dead_zone_candidates or
            any(value < 0 for value in dead_zone_candidates)):
        raise SystemExit(
            "--residual-dead-zone-candidates must contain non-negative values"
        )

    feature_sources: Dict[str, Path] = {}
    for value in args.feature_source:
        name, separator, raw_path = value.partition("=")
        source = Path(raw_path)
        if not separator or not name or not source.is_file():
            raise SystemExit(f"invalid --feature-source NAME=PATH: {value}")
        feature_sources[name] = source

    samples: List[Sample] = []
    sibling_cost_features: Dict[str, Dict[str, int]] = {}
    for report_path in args.input_report:
        if not report_path.is_file():
            raise SystemExit(f"input report not found: {report_path}")
        loaded = json.loads(report_path.read_text())
        sibling_cost_features.update(load_sibling_cost_features(report_path))
        for row in loaded.get("samples", []):
            if not isinstance(row, dict):
                raise SystemExit(f"invalid sample in input report: {report_path}")
            samples.append(row)
    for index in range(args.samples):
        rows, columns = rng.choice(((1, 3), (1, 4), (1, 5), (2, 3),
                                    (2, 4), (3, 3), (4, 4)))
        spec = SampleSpec(
            index=index,
            seed=rng.randrange(1 << 30),
            rows=rows,
            columns=columns,
            split_domain=rng.randint(0, 1),
            sources=rng.randint(2, min(7, rows * columns)),
            operations=rng.randint(6, 32),
            fanout_bias=rng.choice((0.0, 0.5, 1.5, 3.0)),
            # Register files are allocated in eight-register banks by the
            # Architecture implementation; smaller values describe zero
            # physical registers and are intentionally not sampled.
            registers=rng.choice((8, 16, 32, 64)),
        )
        sample_dir = args.output_dir / f"sample-{index:03d}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_sample(args.opt, sample_dir, spec, args.timeout)
        if result is not None:
            samples.append(result)
            print(
                f"sample={index:03d} lb={result['baseline_lb']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']} "
                f"moves={result['moves']} split={result['split_domain']}"
            )
        else:
            print(f"sample={index:03d} unavailable", file=sys.stderr)

    for index in range(args.random_c_samples):
        rows, columns = rng.choice(((3, 3), (3, 4), (4, 4)))
        spec = CSpec(
            index=index,
            seed=rng.randrange(1 << 30),
            rows=rows,
            columns=columns,
            terms=rng.randint(1, 4),
            loop_trip_count=rng.choice((8, 12, 16, 24, 32)),
        )
        sample_dir = args.output_dir / f"c-sample-{index:03d}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_c_sample(
            args.opt, sample_dir, spec, args.real_architecture, args.cxx,
            args.llvm_extract, args.mlir_translate, args.timeout,
        )
        if result is not None:
            samples.append(result)
            print(
                f"c-sample={index:03d} lb={result['baseline_lb']} "
                f"analytical={result['analytical_ii']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']} "
                f"terms={spec.terms} trip={spec.loop_trip_count}"
            )
        else:
            print(f"c-sample={index:03d} unavailable", file=sys.stderr)

    shapes: List[Tuple[int, int]] = []
    for value in args.real_shape or ["3x3", "3x4", "4x4"]:
        match = re.fullmatch(r"(\d+)x(\d+)", value)
        if not match:
            raise SystemExit(f"invalid --real-shape (expected ROWSxCOLS): {value}")
        rows, columns = (int(component) for component in match.groups())
        if rows * columns > 20:
            raise SystemExit("--real-shape exceeds the 20-tile all-cut RouteLB limit")
        shapes.append((rows, columns))
    for fixture in args.real_fixture:
        name, separator, raw_path = fixture.partition("=")
        source = Path(raw_path)
        if not separator or not name or not source.is_file():
            raise SystemExit(f"invalid --real-fixture NAME=PATH: {fixture}")
        for rows, columns in shapes:
            sample_dir = args.output_dir / f"real-{name}-{rows}x{columns}"
            sample_dir.mkdir(exist_ok=True)
            result = collect_real_fixture(
                args.opt, sample_dir, name, source, args.real_architecture,
                rows, columns, args.timeout,
            )
            if result is not None:
                samples.append(result)
                print(
                    f"real={result['index']} lb={result['baseline_lb']} "
                    f"compiled={result['compiled_ii']} nodes={result['nodes']}"
                )
            else:
                print(f"real={name}-{rows}x{columns} unavailable", file=sys.stderr)
        if args.real_random_masks:
            if not 1 <= args.real_mask_min_tiles <= 15:
                raise SystemExit("--real-mask-min-tiles must be in [1, 15]")
            seed_offset = sum(ord(char) for char in name)
            masks = random_connected_tile_masks(
                random.Random(args.real_mask_seed + seed_offset),
                args.real_random_masks, args.real_mask_min_tiles,
            )
            for mask_index, mask in enumerate(masks):
                sample_dir = args.output_dir / f"real-{name}-mask-{mask_index:03d}"
                sample_dir.mkdir(exist_ok=True)
                result = collect_real_fixture(
                    args.opt, sample_dir, name, source, args.real_architecture,
                    4, 4, args.timeout, mask, suffix=f"-mask{mask_index:03d}",
                )
                if result is not None:
                    samples.append(result)
                    print(
                        f"real={result['index']} lb={result['baseline_lb']} "
                        f"compiled={result['compiled_ii']} tiles={result['tiles']}"
                    )
                else:
                    print(f"real={name}-mask{mask_index:03d} unavailable",
                          file=sys.stderr)

    for fixture in args.mapped_real_fixture:
        parts = fixture.split("=", 3)
        if len(parts) != 4:
            raise SystemExit(
                "invalid --mapped-real-fixture NAME=SOURCE=MAPPED=ROWSxCOLS"
            )
        name, raw_source, raw_mapped, shape = parts
        source, mapped = Path(raw_source), Path(raw_mapped)
        match = re.fullmatch(r"(\d+)x(\d+)", shape)
        if not name or not source.is_file() or not mapped.is_file() or not match:
            raise SystemExit(
                "invalid --mapped-real-fixture NAME=SOURCE=MAPPED=ROWSxCOLS"
            )
        rows, columns = (int(component) for component in match.groups())
        if rows * columns > 20:
            raise SystemExit("--mapped-real-fixture exceeds the 20-tile RouteLB limit")
        sample_dir = args.output_dir / f"completed-{name}-{rows}x{columns}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_completed_real_fixture(
            args.opt, sample_dir, name, source, mapped, args.real_architecture,
            rows, columns, args.timeout,
        )
        if result is not None:
            samples.append(result)
            print(
                f"completed={result['index']} lb={result['baseline_lb']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']}"
            )
        else:
            print(f"completed={name}-{shape} unavailable", file=sys.stderr)

    # Labels may be present in several reports (for example an initial shape
    # sweep plus a later mask sweep).  Preserve the last occurrence, so a
    # relabelled sample can replace an earlier timeout-era result.
    unique_samples = {str(row["index"]): row for row in samples}
    samples = list(unique_samples.values())
    required_features = set(FEATURE_NAMES)
    for row in samples:
        row.update(sibling_cost_features.get(str(row["index"]), {}))
        add_prediction_features(row)
        missing = required_features.difference(row)
        if not missing:
            continue
        source = feature_sources.get(str(row["family"]))
        if source is None:
            missing_text = ", ".join(sorted(missing))
            raise SystemExit(
                f"sample {row['index']} lacks [{missing_text}]; provide "
                f"--feature-source {row['family']}=PATH or recollect it"
            )
        row.update(semantic_features_from_neura(source.read_text()))
        add_prediction_features(row)
        missing = required_features.difference(row)
        if missing:
            raise SystemExit(
                f"sample {row['index']} still lacks features: {sorted(missing)}"
            )
    if len(samples) < 12:
        print(f"only {len(samples)} labels collected; no model fitted", file=sys.stderr)
        return 1
    row_holdout = random_row_holdout(
        samples, args.seed, args.ridge, args.tree_depth, args.tree_min_samples)
    try:
        family_holdout: Optional[Dict[str, object]] = leave_one_family_out(
            samples, args.ridge, args.tree_depth, args.tree_min_samples)
    except ValueError:
        # A synthetic-only corpus has one family by design.  Its row split may
        # help debug the generator, but is not evidence of generalization.
        family_holdout = None
    try:
        nested_ridge_holdout: Optional[Dict[str, object]] = (
            nested_ridge_family_holdout(
                samples, ridge_candidates, dead_zone_candidates))
    except ValueError:
        nested_ridge_holdout = None
    real_family_count = len({
        str(row["family"]) for row in samples
        if not str(row["family"]).startswith("synthetic")
    })
    selected_model = "none"
    selected_metric_key = ""
    selected_rows: List[Dict[str, object]] = []
    trained_full_model: Optional[Dict[str, object]] = None
    if family_holdout is not None:
        # Selection is intentionally based on the real-kernel holdout only.
        # Synthetic corpus rows may augment training experiments, but may not
        # select a model for a real DSE workload.
        ridge_score = (
            nested_ridge_holdout.get("real_ridge_mae", float("inf"))
            if nested_ridge_holdout is not None else float("inf"),
            nested_ridge_holdout.get("real_ridge_macro_family_mae", float("inf"))
            if nested_ridge_holdout is not None else float("inf"),
        )
        tree_score = (
            family_holdout.get("real_tree_mae", float("inf")),
            family_holdout.get("real_tree_macro_family_mae", float("inf")),
        )
        if ridge_score <= tree_score:
            selected_model = "ridge"
            selected_metric_key = "ridge"
            selected_rows = nested_ridge_holdout["rows"]
            selected_ridge, selected_dead_zone = select_ridge_hyperparameters(
                samples, ridge_candidates, dead_zone_candidates)
            trained_full_model = fit_ridge(
                samples, selected_ridge, selected_dead_zone)
        else:
            selected_model = "residual_tree"
            selected_metric_key = "tree"
            selected_rows = family_holdout["tree_rows"]
            trained_full_model = fit_residual_tree(
                samples, args.tree_depth, args.tree_min_samples)
    numeric_gate = bool(
        family_holdout is not None and nested_ridge_holdout is not None and
        real_family_count >= 12 and
        (nested_ridge_holdout if selected_metric_key == "ridge"
         else family_holdout)[f"real_{selected_metric_key}_mae"] <
        family_holdout["real_baseline_mae"] and
        (nested_ridge_holdout if selected_metric_key == "ridge"
         else family_holdout)[f"real_{selected_metric_key}_macro_family_mae"] <
        family_holdout["real_baseline_macro_family_mae"]
    )
    predictions: List[Dict[str, object]] = []
    if args.predict_fixture:
        if trained_full_model is None:
            raise SystemExit(
                "--predict-fixture needs at least two labelled kernel families"
            )
        prediction_shapes: List[Tuple[int, int]] = []
        for value in args.predict_shape or ["4x4"]:
            match = re.fullmatch(r"(\d+)x(\d+)", value)
            if not match:
                raise SystemExit(
                    "invalid --predict-shape (expected ROWSxCOLS): " + value
                )
            rows, columns = (int(component) for component in match.groups())
            if rows * columns > 20:
                raise SystemExit("--predict-shape exceeds the 20-tile RouteLB limit")
            prediction_shapes.append((rows, columns))
        for fixture in args.predict_fixture:
            name, separator, raw_path = fixture.partition("=")
            source = Path(raw_path)
            if not separator or not name or not source.is_file():
                raise SystemExit(f"invalid --predict-fixture NAME=PATH: {fixture}")
            for rows, columns in prediction_shapes:
                sample_dir = (
                    args.output_dir / f"prediction-{name}-{rows}x{columns}"
                )
                sample_dir.mkdir(exist_ok=True)
                features = collect_prediction_fixture(
                    args.opt, sample_dir, name, source, args.real_architecture,
                    rows, columns, args.timeout,
                )
                if features is None:
                    print(f"prediction={name}-{rows}x{columns} unavailable",
                          file=sys.stderr)
                    continue
                if selected_model == "ridge":
                    predicted_ii = predict_ridge(trained_full_model, features)
                else:
                    predicted_ii = predict_residual_tree(
                        trained_full_model, features)
                prediction = {
                    "sample": features["index"],
                    "predicted_compiled_ii": predicted_ii,
                    "baseline_lb": features["baseline_lb"],
                    "analytical_ii": features["analytical_ii"],
                    "rec_mii": features["rec_mii"],
                    "res_mii": features["res_mii"],
                    "route_mii": features["route_mii"],
                }
                predictions.append(prediction)
                print(
                    f"prediction={prediction['sample']} "
                    f"compiled_ii={prediction['predicted_compiled_ii']:.2f} "
                    f"lb={prediction['baseline_lb']}"
                )
    provenance = {
        "adapter": "neura",
        "neura": git_provenance(args.neura_root),
        "mlir_neura_opt": str(args.opt.resolve()),
        "architecture": str(args.real_architecture.resolve()),
        "architecture_sha256": file_sha256(args.real_architecture),
        "mapping_strategy": "heuristic",
        "label_policy": "successful_compiled_ii_only_timeouts_are_not_labels",
    }
    portable_dataset = {
        "provenance": provenance,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "samples": [{
            "sample_id": str(row["index"]),
            "group": str(row["family"]),
            "lower_bound": row["baseline_lb"],
            "compiled_ii": row["compiled_ii"],
            "features": {name: row[name] for name in FEATURE_NAMES},
            "metadata": {
                "adapter": "neura",
                "rows": row["rows"],
                "tiles": row["tiles"],
                "links": row["links"],
                "source_kind": (
                    "generated" if str(row["family"]).startswith("synthetic")
                    else "real"
                ),
            },
        } for row in samples],
        "censored_samples": [],
    }
    report = {
        "provenance": provenance,
        "feature_names": list(FEATURE_NAMES),
        "model_feature_names": list(MODEL_FEATURE_NAMES),
        "samples": samples,
        "random_row_holdout_diagnostic": row_holdout,
        "family_holdout": family_holdout,
        "nested_ridge_family_holdout": nested_ridge_holdout,
        "selected_model": selected_model,
        "trained_full_model": trained_full_model,
        "predictions": predictions,
        "candidate_status": "offline_experiment_only_never_compiler_input",
        "candidate_gate": {
            "requires": [
                "selected model real family-holdout MAE strictly improves baseline",
                "selected model real macro-family MAE strictly improves baseline",
                "at least twelve independent real kernel families",
                "separate broader-corpus validation before compiler integration",
            ],
            "current_numeric_conditions_met": numeric_gate,
            "real_family_count": real_family_count,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    (args.output_dir / "dataset.json").write_text(
        json.dumps(portable_dataset, indent=2) + "\n")
    if family_holdout is not None:
        print(
            "family_holdout=" + str(len(family_holdout["ridge_rows"])) +
            f" baseline_mae={family_holdout['baseline_mae']:.3f}" +
            f" ridge_mae={family_holdout['ridge_mae']:.3f}" +
            f" tree_mae={family_holdout['tree_mae']:.3f}" +
            f" selected={selected_model}"
        )
    else:
        print("family_holdout=unavailable (fewer than two independent families)")
    if nested_ridge_holdout is not None:
        print(
            "nested_ridge_family_holdout=" +
            str(len(nested_ridge_holdout["rows"])) +
            f" baseline_mae={nested_ridge_holdout['baseline_mae']:.3f}" +
            f" ridge_mae={nested_ridge_holdout['ridge_mae']:.3f}" +
            f" rounded_exact="
            f"{nested_ridge_holdout['ridge_rounded_exact_rate']:.3f}"
        )
    print(
        "row_holdout_diagnostic=" + str(len(row_holdout["ridge_rows"])) +
        f" baseline_mae={row_holdout['baseline_mae']:.3f}" +
        f" ridge_mae={row_holdout['ridge_mae']:.3f}" +
        f" tree_mae={row_holdout['tree_mae']:.3f}"
    )
    if family_holdout is not None:
        for row in selected_rows:
            print(
                f"family_holdout sample={row['sample']} family={row['family']} "
                f"lb={row['baseline']:.1f} "
                f"prediction={row['prediction']:.2f} compiled={row['compiled_ii']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
