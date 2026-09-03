#!/usr/bin/env python3
"""Deterministic compute-motif corpus generation for the Neura adapter.

The legacy ``--samples`` generator in :mod:`neura_experiment` is intentionally
kept as a compatibility path.  This module is the newer, auditable corpus
stratum: a base DFG is generated once from ``(generator version, motif,
base_seed, operation count)`` and then paired with target rectangles on one
pinned architecture.  Shape variation never changes the base DFG,
which makes it possible to group all variants under one leakage-safe lineage.

The emitted IR is already in the lowered Neura dataflow dialect.  The original
compute families are joined by direct recurrence, predicated-control,
streaming-memory, and pointer-chasing DFGs.  No C
frontend or compiler lowering is involved, so each structural edge is visible
to the analysis-only Rec/Res pass and heuristic mapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


GENERATOR_VERSION = "motif-v3"
GENERATOR_TYPE = "generated/motif"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v3"
DATA_TYPE = "!neura.data<i32, i1>"
I64_DATA_TYPE = "!neura.data<i64, i1>"
PREDICATE_DATA_TYPE = "!neura.data<i1, i1>"
POINTER_DATA_TYPE = "!neura.data<!llvm.ptr, i1>"
DEFAULT_MOTIFS = (
    "chain", "fanout", "reduction", "diamond", "random_dag",
    "recurrence_chain", "predicated_diamond", "memory_stream",
    "pointer_chase",
)
MOTIF_ALIASES = {
    "broadcast": "fanout",
    "fanout_broadcast": "fanout",
    "binary_reduction": "reduction",
    "reduction_tree": "reduction",
    "split_join": "diamond",
    "mixed": "random_dag",
    "mixed_dag": "random_dag",
    "multi_input": "random_dag",
    "random": "random_dag",
    "random-dag": "random_dag",
    "recurrence": "recurrence_chain",
    "loop": "recurrence_chain",
    "predicated": "predicated_diamond",
    "control": "predicated_diamond",
    "pointer": "pointer_chase",
    "pointer-chase": "pointer_chase",
    "memory": "memory_stream",
    "memory-stream": "memory_stream",
}
DEFAULT_SHAPES = tuple(
    (rows, columns)
    for rows in range(2, 5)
    for columns in range(2, 5)
)
# Tiny targets are useful deployment/stress candidates, but they are not part
# of the frozen training population until every family passes the same
# predeclared coverage gate.  1x2 is the canonical representative of the
# orientation-symmetric two-tile strip on the pinned mesh.
PREDICTION_SHAPES = ((1, 1), (1, 2)) + DEFAULT_SHAPES
PRIMARY_SHAPE = (4, 4)
SHAPE_DESIGN = "full-4x4-plus-balanced-secondary-v1"
# Three strata make the operation-count distribution explicit and reproducible.
# The lower edge is deliberately above the tiny legacy examples: these are
# intended to exercise graph pressure while still being practical smoke tests.
OPERATION_BANDS = ((8, 15), (16, 31), (32, 48))
# v1 callers use OPERATION_BANDS directly, so retain it as a compatibility
# contract.  The v3 direct-lowered motifs
# use family-specific bands.  The frozen corpus stays in the tens-of-operations
# regime used by LISA and by mapper-feasible Neura examples; larger direct API
# limits remain available for explicitly declared stress experiments.
FAMILY_OPERATION_BANDS = {
    "chain": ((8, 23), (24, 47), (48, 96)),
    "fanout": ((8, 23), (24, 47), (48, 96)),
    "reduction": ((8, 11), (12, 19), (20, 31)),
    "diamond": ((8, 15), (16, 31), (32, 64)),
    # Dense random/predicated graphs trigger exponential backtracking in the
    # reference heuristic above roughly twenty payload ops.  Keep the frozen
    # distribution in the mapper-complete regime; the larger direct API is
    # retained for explicit stress tests that are reported as censored.
    "random_dag": ((8, 11), (12, 15), (16, 20)),
    # The pinned Neura architecture has 20 control-memory items per tile.
    # Recurrences longer than 16 fail systematically under that production
    # contract, so v3 samples the useful range instead of manufacturing
    # inevitable censored rows.
    "recurrence_chain": ((8, 10), (11, 13), (14, 16)),
    "predicated_diamond": ((8, 11), (12, 15), (16, 20)),
    "memory_stream": ((8, 15), (16, 31), (32, 64)),
    "pointer_chase": ((8, 15), (16, 31), (32, 64)),
}
DIRECT_OPERATION_LIMITS = {
    "random_dag": 160,
    "recurrence_chain": 48,
    "predicated_diamond": 128,
    "memory_stream": 128,
    "pointer_chase": 128,
}
DEFAULT_ARCHITECTURE_VARIANTS = ("neura-main",)
PINNED_ARCHITECTURE_SHA256 = (
    "f244f15be30604eb32eb96e4837a4bf1ce5c34961c3a46299b90931505cc97e6"
)
PINNED_NEURA_REVISION = "47b7e3a68c321075293e6fcb45fb3b1cabb93b88"
PINNED_ARCHITECTURE_ROWS = 4
PINNED_ARCHITECTURE_COLUMNS = 4
PINNED_REGISTERS_PER_TILE = 32
PINNED_CTRL_MEM_ITEMS = 20
PINNED_ARCHITECTURE_RELATIVE_PATH = Path(
    "third_party/neura/test/arch_spec/architecture.yaml"
)


@dataclass(frozen=True)
class MotifBaseSpec:
    """One source DFG before architecture candidates are applied."""

    motif: str
    base_index: int
    base_seed: int
    operation_count: int
    root_seed: int = 0
    generator_family: str = "generated/motif"
    generator_type: str = "generated/motif"
    generator_version: str = GENERATOR_VERSION
    # v4 adds orthogonal, pre-label structural strata.  Empty values preserve
    # the byte-for-byte v3 manifest contract and are omitted from v3 records.
    mechanism_profile: str = ""
    operation_band: str = ""

    @property
    def base_id(self) -> str:
        # Include the corpus root seed in the lineage identity.  Merely using
        # base-0000 would make two invocations with different --seed values
        # look like the same held-out source family.
        root_token = (
            f"m{abs(self.root_seed)}"
            if self.root_seed < 0 else f"p{self.root_seed}"
        )
        return f"base-{self.base_index:04d}-r{root_token}"

    @property
    def lineage(self) -> str:
        return (
            f"generated/{self.generator_version}/{self.motif}/{self.base_id}"
        )


@dataclass(frozen=True)
class MotifCandidate:
    """One mapper attempt for a generated base DFG."""

    candidate_id: str
    lineage: str
    motif: str
    generator_family: str
    generator_version: str
    generator_type: str
    base_id: str
    base_seed: int
    root_seed: int
    base_index: int
    operation_count: int
    rows: int
    columns: int
    architecture_variant: str
    registers: int
    source_path: str
    architecture_path: str
    source_sha256: str
    canonical_dfg_sha256: str
    architecture_sha256: str
    target_config_id: str
    valid_tiles: str = ""
    status: str = "declared"
    stage: str = "predeclared"
    failure: Optional[str] = None
    mechanism_profile: str = ""
    operation_band: str = ""
    shape_block: str = ""

    @property
    def architecture_id(self) -> str:
        return f"{self.architecture_sha256}:{self.target_config_id}"

    def manifest_record(self) -> Dict[str, object]:
        record = asdict(self)
        for optional_field in (
            "mechanism_profile", "operation_band", "shape_block",
        ):
            if not record[optional_field]:
                record.pop(optional_field)
        record["architecture_id"] = self.architecture_id
        # Keep all identity layers explicit before labels exist.  The
        # canonical hash denotes one exact DFG/ranking query; ``lineage`` is
        # the broader train/test leakage boundary.
        record["leakage_lineage_id"] = self.lineage
        record["base_dfg_id"] = self.canonical_dfg_sha256
        record["ranking_query_id"] = self.canonical_dfg_sha256
        record["training_stratum"] = "generated"
        # Keep this explicit: it is the key used by the adapter to update a
        # candidate after Rec/Res-analysis/mapper success or censorship.
        record["id"] = self.candidate_id
        return record


def parse_shape(value: str) -> Tuple[int, int]:
    """Parse and validate a positive ``ROWSxCOLS`` shape."""
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid shape (expected ROWSxCOLS): {value}")
    rows, columns = (int(part) for part in match.groups())
    if rows < 1 or columns < 1:
        raise ValueError("shape dimensions must be positive")
    return rows, columns


def parse_motif_names(values: Sequence[str]) -> Tuple[str, ...]:
    """Expand repeated/comma-separated motif options deterministically."""
    requested: List[str] = []
    for value in values:
        requested.extend(
            MOTIF_ALIASES.get(part.strip().lower(), part.strip().lower())
            for part in value.split(",")
        )
    if not requested:
        return DEFAULT_MOTIFS
    unknown = sorted(set(requested).difference(DEFAULT_MOTIFS))
    if unknown:
        raise ValueError(
            "unknown motif(s): " + ", ".join(unknown) +
            "; available: " + ", ".join(DEFAULT_MOTIFS)
        )
    # Preserve command-line order while avoiding duplicate families.
    return tuple(dict.fromkeys(requested))


def parse_shapes(values: Optional[Sequence[str]]) -> Tuple[Tuple[int, int], ...]:
    raw = list(values) if values else [f"{r}x{c}" for r, c in DEFAULT_SHAPES]
    result: List[Tuple[int, int]] = []
    for value in raw:
        shape = parse_shape(value)
        if shape not in result:
            result.append(shape)
    return tuple(result)


def parse_architecture_variants(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    raw_values = list(values) if values else list(DEFAULT_ARCHITECTURE_VARIANTS)
    requested: List[str] = []
    for value in raw_values:
        requested.extend(part.strip().lower() for part in value.split(","))
    result = tuple(dict.fromkeys(requested))
    unknown = sorted(set(result).difference(DEFAULT_ARCHITECTURE_VARIANTS))
    if unknown:
        raise ValueError(
            "unknown motif architecture variant(s): " + ", ".join(unknown)
        )
    if not result:
        raise ValueError("at least one motif architecture variant is required")
    return result


def operation_bands_for_motif(motif: str) -> Tuple[Tuple[int, int], ...]:
    """Return the size bands for one canonical motif family."""
    name = MOTIF_ALIASES.get(motif.strip().lower(), motif.strip().lower())
    try:
        return FAMILY_OPERATION_BANDS[name]
    except KeyError as error:
        raise ValueError(f"unknown motif: {motif}") from error


def stratified_operation_count(
    base_index: int, base_seed: int, motif: Optional[str] = None,
) -> int:
    """Choose an operation count from each family-specific size band.

    The optional ``motif`` argument preserves the v1 two-argument API.  With
    no motif, the historical 8--48 bands are used; ``make_base_specs`` passes
    the family explicitly so v3 corpora include large structural examples.
    """
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    bands = OPERATION_BANDS if motif is None else operation_bands_for_motif(motif)
    band_low, band_high = bands[base_index % len(bands)]
    rng = random.Random(base_seed)
    return rng.randint(band_low, band_high)


def make_base_specs(
    samples_per_family: int,
    seed: int = 20260902,
    motifs: Optional[Sequence[str]] = None,
) -> Tuple[MotifBaseSpec, ...]:
    """Create deterministic source DFG specifications for each motif family."""
    if samples_per_family < 0:
        raise ValueError("samples_per_family must be non-negative")
    selected = parse_motif_names(motifs or ())
    result: List[MotifBaseSpec] = []
    # A family-specific RNG stream avoids changing existing families when a
    # new motif is appended to DEFAULT_MOTIFS.
    for motif in selected:
        # Use a stable digest rather than Python's process-randomized hash and
        # do not derive the stream from list position: selecting a motif
        # subset or reordering --motif options must not change its DFGs.
        family_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{motif}".encode("utf-8")).digest()[:8],
            "big",
        )
        family_rng = random.Random(family_seed)
        seen_canonical = set()
        for base_index in range(samples_per_family):
            # Literal values are intentionally excluded from canonical graph
            # identity. Resample the deterministic family stream if two bases
            # would therefore have the same labelled dependency graph.
            for _ in range(10000):
                base_seed = family_rng.randrange(1 << 63)
                operation_count = stratified_operation_count(
                    base_index, base_seed, motif
                )
                canonical = canonical_dfg_sha256(generate_motif_mlir(
                    motif, operation_count, base_seed
                ))
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    break
            else:
                raise ValueError(
                    f"unable to generate a distinct canonical {motif} DFG"
                )
            result.append(MotifBaseSpec(
                motif=motif,
                base_index=base_index,
                base_seed=base_seed,
                operation_count=operation_count,
                root_seed=seed,
                generator_family=f"generated/motif/{motif}",
            ))
    return tuple(result)


def _constant(value: str, number: int, scalar_type: str = "i32",
              data_type: Optional[str] = None) -> str:
    result_type = data_type or f"!neura.data<{scalar_type}, i1>"
    return (
        f'    {value} = "neura.constant"() '
        f'<{{value = {number} : {scalar_type}}}> : () -> {result_type}'
    )


def _grant_once_constant(value: str, number: int, scalar_type: str = "i64",
                         data_type: Optional[str] = None) -> str:
    result_type = data_type or f"!neura.data<{scalar_type}, i1>"
    return (
        f'    {value} = "neura.grant_once"() '
        f'<{{constant_value = {number} : {scalar_type}}}> '
        f': () -> {result_type}'
    )


def _move(result: str, operand: str, data_type: str = DATA_TYPE) -> str:
    return (
        f'    {result} = "neura.data_mov"({operand}) '
        f': ({data_type}) -> {data_type}'
    )


def _binary(result: str, operation: str, lhs: str, rhs: str,
            data_type: str = DATA_TYPE) -> str:
    if operation not in ("add", "mul"):
        raise ValueError(f"unsupported generated compute operation: {operation}")
    return (
        f'    {result} = "neura.{operation}"({lhs}, {rhs}) '
        f': ({data_type}, {data_type}) -> {data_type}'
    )


def _unary_binary(result: str, operation: str, operand: str,
                  rhs_value: int, scalar_type: str = "i32",
                  data_type: str = DATA_TYPE) -> str:
    """Emit a lowered arithmetic op with an immediate second operand."""
    if operation not in ("add", "mul"):
        raise ValueError(f"unsupported generated compute operation: {operation}")
    return (
        f'    {result} = "neura.{operation}"({operand}) '
        f'{{rhs_value = {rhs_value} : {scalar_type}}} '
        f': ({data_type}) -> {data_type}'
    )


def _operation_kind(rng: random.Random, index: int) -> str:
    # The index tie-break prevents a rare all-one-op random stream from making
    # generated source hashes unhelpfully homogeneous.
    return "add" if ((rng.randrange(2) + index) % 2 == 0) else "mul"


def _emit_binary(
    lines: List[str], operation_index: int, operation: str,
    lhs: str, rhs: str, data_type: str = DATA_TYPE,
    name_prefix: str = "",
) -> str:
    prefix = f"{name_prefix}" if name_prefix else ""
    lhs_move = f"%{prefix}m{operation_index}a"
    rhs_move = f"%{prefix}m{operation_index}b"
    result = f"%{prefix}v{operation_index}"
    lines.append(_move(lhs_move, lhs, data_type))
    lines.append(_move(rhs_move, rhs, data_type))
    lines.append(_binary(result, operation, lhs_move, rhs_move, data_type))
    return result


def _emit_unary_binary(
    lines: List[str], operation_index: int, operation: str, operand: str,
    rhs_value: int, scalar_type: str = "i32", data_type: str = DATA_TYPE,
    name_prefix: str = "",
) -> str:
    prefix = f"{name_prefix}" if name_prefix else ""
    moved = f"%{prefix}m{operation_index}"
    result = f"%{prefix}v{operation_index}"
    lines.append(_move(moved, operand, data_type))
    lines.append(_unary_binary(
        result, operation, moved, rhs_value, scalar_type, data_type
    ))
    return result


def _emit_header(motif: str) -> List[str]:
    return [
        "module {",
        f'  func.func @generated_{motif}() '
        'attributes {accelerator = "neura"} {',
    ]


def _emit_footer(lines: List[str]) -> str:
    lines.extend(("    func.return", "  }", "}"))
    return "\n".join(lines) + "\n"


def _emit_constants(
    lines: List[str], count: int, start: int = 0, seed: Optional[int] = None,
) -> List[str]:
    # Include the base seed in literal values so independent bases remain
    # auditable through their source hash and provenance. Canonical identity
    # deliberately normalizes these literals; topology/op labels plus
    # deterministic collision resampling provide canonical uniqueness.
    rng = random.Random(seed) if seed is not None else None
    values = []
    for index in range(start, start + count):
        value = f"%c{index}"
        number = rng.randint(1, (1 << 30) - 1) if rng is not None else index + 1
        lines.append(_constant(value, number))
        values.append(value)
    return values


def _generate_chain(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("chain")
    current = _emit_constants(lines, 1, seed=seed)[0]
    for index in range(operation_count):
        current = _emit_unary_binary(
            lines, index, _operation_kind(rng, index), current,
            1 + rng.randrange(31),
        )
    return _emit_footer(lines)


def _generate_fanout(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("fanout")
    root = _emit_constants(lines, 1, seed=seed)[0]
    branch_count = 2 + ((abs(seed) >> 3) % min(7, operation_count - 1))
    branches: List[str] = []
    # Each branch starts from the same source.  Remaining operations extend
    # branches round-robin, preserving a large semantic fanout at the root.
    for index in range(branch_count):
        branches.append(_emit_unary_binary(
            lines, index, _operation_kind(rng, index), root,
            1 + rng.randrange(31),
        ))
    for index in range(branch_count, operation_count):
        branch = (index - branch_count) % branch_count
        branches[branch] = _emit_unary_binary(
            lines, index, _operation_kind(rng, index), branches[branch],
            1 + rng.randrange(31),
        )
    return _emit_footer(lines)


def _generate_reduction(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("reduction")
    frontier = _emit_constants(lines, operation_count + 1, seed=seed)
    operation_index = 0
    # Pairwise reduction with odd-node carry-over is a binary tree for every
    # N: N+1 leaves require exactly N internal operations.
    while len(frontier) > 1:
        next_frontier: List[str] = []
        index = 0
        while index + 1 < len(frontier):
            next_frontier.append(_emit_binary(
                lines, operation_index, _operation_kind(rng, operation_index),
                frontier[index], frontier[index + 1],
            ))
            operation_index += 1
            index += 2
        if index < len(frontier):
            next_frontier.append(frontier[index])
        frontier = next_frontier
    if operation_index != operation_count:
        raise AssertionError("binary reduction did not emit requested operation count")
    return _emit_footer(lines)


def _generate_diamond(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("diamond")
    current = _emit_constants(lines, 1, seed=seed)[0]
    index = 0
    diamond_count = 1 + ((abs(seed) >> 4) % min(6, operation_count // 3))
    for _ in range(diamond_count):
        first = _emit_unary_binary(
            lines, index, _operation_kind(rng, index), current,
            1 + rng.randrange(31),
        )
        index += 1
        second = _emit_unary_binary(
            lines, index, _operation_kind(rng, index), current,
            1 + rng.randrange(31),
        )
        index += 1
        current = _emit_binary(
            lines, index, _operation_kind(rng, index), first, second
        )
        index += 1
    while index < operation_count:
        current = _emit_unary_binary(
            lines, index, _operation_kind(rng, index), current,
            1 + rng.randrange(31),
        )
        index += 1
    return _emit_footer(lines)


def _generate_mixed(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("mixed")
    # Multiple independent sources and a rotating pool produce fanout,
    # reconvergence, and binary multi-input nodes in one DAG.
    constants = _emit_constants(lines, max(8, operation_count + 2), seed=seed)
    pool = constants[:6]
    for index in range(operation_count):
        if index == 0:
            lhs, rhs = constants[0], constants[1]
        elif index == 1:
            lhs, rhs = constants[0], constants[2]
        elif index % 5 == 0:
            lhs, rhs = pool[0], pool[-1]
        elif index % 3 == 0:
            lhs, rhs = pool[index % len(pool)], pool[(index + 2) % len(pool)]
        else:
            lhs = pool[(index * 3) % len(pool)]
            rhs = constants[(index + 3) % len(constants)]
        result = _emit_binary(
            lines, index, _operation_kind(rng, index), lhs, rhs
        )
        # Retain recent outputs but keep independent constants in the pool so
        # later nodes continue to have genuine multi-input sources.
        pool[index % len(pool)] = result
    return _emit_footer(lines)


def _generate_random_dag(operation_count: int, seed: int) -> str:
    """Generate a weakly connected random acyclic compute graph.

    Every operation after the first consumes at least one earlier operation,
    so the operation graph is connected when edge direction is ignored.  The
    second operand is sampled from constants and earlier results, which creates
    seed-dependent fanout and reconvergence rather than merely changing op
    spellings on one fixed topology.
    """
    rng = random.Random(seed)
    lines = _emit_header("random_dag")
    constant_count = max(4, min(12, operation_count // 3 + 2))
    constants = _emit_constants(lines, constant_count, seed=seed)
    # Use every emitted constant so the complete DFG—not only its binary-op
    # projection—is weakly connected.  Shuffle the introduction order to
    # preserve seed-dependent source structure.
    constant_order = list(constants)
    rng.shuffle(constant_order)
    results: List[str] = []
    for index in range(operation_count):
        if not results:
            lhs, rhs = constant_order[0], constant_order[1]
        else:
            # This mandatory predecessor makes the operation graph connected.
            lhs = rng.choice(results)
            # Introduce every remaining constant before switching to fully
            # random choices.  Since every op is attached to an earlier
            # result, each constant joins the same weak component.
            if index < constant_count - 1:
                rhs = constant_order[index + 1]
                pool = constants + results[-min(12, len(results)):]
            else:
                lookback = 4 + ((abs(seed) >> 6) % 9)
                pool = constants + results[-lookback:]
                rhs = rng.choice(pool)
            if rhs == lhs and len(pool) > 1:
                alternatives = [value for value in pool if value != lhs]
                rhs = rng.choice(alternatives)
            # Periodically reuse an older producer to increase long-range
            # fanout; otherwise random choice still permits reconvergence.
            if index >= 3 and index % 4 == 0:
                window = results[-min(len(results), 12):]
                lhs = rng.choice(window)
        results.append(_emit_binary(
            lines, index, _operation_kind(rng, index), lhs, rhs
        ))
    return _emit_footer(lines)


def _seeded_operation_kind(rng: random.Random, index: int, seed: int) -> str:
    """Choose an op kind while making the seed affect the first edge."""
    if index == 0:
        return "add" if seed % 2 == 0 else "mul"
    return _operation_kind(rng, index)


def _emit_icmp(
    lines: List[str], name: str, operand: str, cmp_type: str,
    rhs_value: int, scalar_type: str = "i32",
    data_type: str = DATA_TYPE,
) -> str:
    moved = f"%{name}_mov"
    result = f"%{name}"
    lines.append(_move(moved, operand, data_type))
    lines.append(
        f'    {result} = "neura.icmp"({moved}) '
        f'<{{cmpType = "{cmp_type}"}}> '
        f'{{rhs_value = {rhs_value} : {scalar_type}}} '
        f': ({data_type}) -> {PREDICATE_DATA_TYPE}'
    )
    return result


def _emit_not(lines: List[str], name: str, operand: str) -> str:
    moved = f"%{name}_mov"
    result = f"%{name}"
    lines.append(_move(moved, operand, PREDICATE_DATA_TYPE))
    lines.append(
        f'    {result} = "neura.not"({moved}) '
        f': ({PREDICATE_DATA_TYPE}) -> {PREDICATE_DATA_TYPE}'
    )
    return result


def _emit_sext(lines: List[str], name: str, operand: str,
               input_type: str = DATA_TYPE,
               output_type: str = I64_DATA_TYPE) -> str:
    moved = f"%{name}_mov"
    result = f"%{name}"
    lines.append(_move(moved, operand, input_type))
    lines.append(
        f"    {result} = neura.sext {moved} : "
        f"{input_type} -> {output_type}"
    )
    return result


def _emit_grant_predicate(
    lines: List[str], name: str, value: str, predicate: str,
    data_type: str = DATA_TYPE,
) -> str:
    value_mov = f"%{name}_value_mov"
    predicate_mov = f"%{name}_pred_mov"
    result = f"%{name}"
    lines.append(_move(value_mov, value, data_type))
    lines.append(_move(predicate_mov, predicate, PREDICATE_DATA_TYPE))
    lines.append(
        f"    {result} = neura.grant_predicate {value_mov}, "
        f"{predicate_mov} : {data_type}, {PREDICATE_DATA_TYPE} -> {data_type}"
    )
    return result


def _generate_recurrence_chain(operation_count: int, seed: int) -> str:
    """Generate a true reserve/phi/arithmetic/ctrl backedge recurrence.

    The requested operation count is exactly the number of arithmetic nodes in
    the recurrence cycle.  Each arithmetic input and the backedge is
    explicitly materialized with ``data_mov`` so the generated graph follows
    the same dataflow contract as hand-written Neura IR.
    """
    rng = random.Random(seed)
    lines = _emit_header("recurrence_chain")
    init = _constant("%rc_init", (abs(seed) % 17) + 3)
    lines.append(init)
    lines.append("    %rc_reserved = neura.reserve : !neura.data<i32, i1>")
    lines.append(_move("%rc_init_mov", "%rc_init"))
    lines.append(
        "    %rc_state = neura.phi_start %rc_init_mov, %rc_reserved "
        ": !neura.data<i32, i1>, !neura.data<i32, i1> "
        "-> !neura.data<i32, i1>"
    )
    current = "%rc_state"
    for index in range(operation_count):
        operation = _seeded_operation_kind(rng, index, seed)
        # Keep immediates deterministic but deliberately irrelevant to the
        # canonical topology hash.  The op sequence is the structural seed
        # variation that matters for model training.
        rhs_value = 1 + ((abs(seed) + 3 * index) % 11)
        current = _emit_unary_binary(
            lines, index, operation, current, rhs_value, name_prefix="rc"
        )
    lines.append(_move("%rc_back_mov", current))
    lines.append(
        "    neura.ctrl_mov %rc_back_mov -> %rc_reserved : "
        "!neura.data<i32, i1> !neura.data<i32, i1>"
    )
    return _emit_footer(lines)


def _generate_predicated_diamond(operation_count: int, seed: int) -> str:
    """Generate one shallow predicate diamond followed by live arithmetic.

    Multiple simultaneous predicates make the reference heuristic's search
    time highly seed-dependent even below twenty payload operations.  One
    real control diamond is enough to expose control-path pressure while
    keeping the formal corpus's mapper-censorship rate bounded.
    """
    rng = random.Random(seed)
    lines = _emit_header("predicated_diamond")
    constants = _emit_constants(lines, 2, seed=seed)
    current = constants[0]
    arithmetic_index = 0
    diamond_count = 1
    for diamond_index in range(diamond_count):
        predicate = _emit_icmp(
            lines, f"pd_cmp{diamond_index}", current,
            "sgt" if ((seed + diamond_index) & 1) else "slt",
            (abs(seed) + diamond_index) % 9, "i32", DATA_TYPE,
        )
        inverted = _emit_not(lines, f"pd_not{diamond_index}_0", predicate)
        then_current = _emit_unary_binary(
            lines, arithmetic_index,
            _seeded_operation_kind(rng, arithmetic_index, seed),
            current, 1 + ((abs(seed) + diamond_index) % 11),
            name_prefix="pd",
        )
        arithmetic_index += 1
        else_current = _emit_unary_binary(
            lines, arithmetic_index,
            _seeded_operation_kind(rng, arithmetic_index, seed),
            constants[1], 1 + ((abs(seed) + diamond_index + 1) % 11),
            name_prefix="pd",
        )
        arithmetic_index += 1
        then_value = _emit_grant_predicate(
            lines, f"pd_then{diamond_index}", then_current, predicate
        )
        else_value = _emit_grant_predicate(
            lines, f"pd_else{diamond_index}", else_current, inverted
        )
        current = _emit_binary(
            lines, arithmetic_index,
            _seeded_operation_kind(rng, arithmetic_index, seed),
            then_value, else_value, name_prefix="pd",
        )
        arithmetic_index += 1

    for tail in range(operation_count - arithmetic_index):
        current = _emit_unary_binary(
            lines, arithmetic_index,
            _seeded_operation_kind(rng, arithmetic_index, seed),
            current, 1 + ((abs(seed) + tail + 7) % 11),
            name_prefix="pd"
        )
        arithmetic_index += 1
    if arithmetic_index != operation_count:
        raise AssertionError(
            "predicated diamond did not emit requested operation count"
        )
    return _emit_footer(lines)


def _emit_pointer_root(
    lines: List[str], index_value: str, base_arg: str,
) -> str:
    """Start a serial pointer chase with an argument-backed pointer load."""
    index_mov = "%pc_root_index_mov"
    base = "%pc_root_base"
    base_mov = "%pc_root_base_mov"
    loaded_ptr = "%pc_root_loaded_ptr"
    lines.append(_move(index_mov, index_value, I64_DATA_TYPE))
    lines.append(
        f'    {base} = "neura.gep"({index_mov}) '
        '<{operandSegmentSizes = array<i32: 0, 1>}> '
        f'{{lhs_value = "{base_arg}"}} '
        f': ({I64_DATA_TYPE}) -> {POINTER_DATA_TYPE}'
    )
    lines.append(_move(base_mov, base, POINTER_DATA_TYPE))
    lines.append(
        f'    {loaded_ptr} = "neura.load"({base_mov}) '
        f': ({POINTER_DATA_TYPE}) -> {POINTER_DATA_TYPE}'
    )
    return loaded_ptr


def _emit_pointer_hop(
    lines: List[str], hop_index: int, base_pointer: str, index_value: str,
    result_type: str,
) -> str:
    """Follow one loaded pointer through an indirect GEP and load."""
    base_mov = f"%pc_hop{hop_index}_base_mov"
    index_mov = f"%pc_hop{hop_index}_index_mov"
    element_ptr = f"%pc_hop{hop_index}_element_ptr"
    element_ptr_mov = f"%pc_hop{hop_index}_element_ptr_mov"
    loaded = f"%pc_hop{hop_index}_loaded"
    lines.append(_move(base_mov, base_pointer, POINTER_DATA_TYPE))
    lines.append(_move(index_mov, index_value, I64_DATA_TYPE))
    lines.append(
        f'    {element_ptr} = "neura.gep"({base_mov}, {index_mov}) '
        '<{operandSegmentSizes = array<i32: 1, 1>}> '
        f': ({POINTER_DATA_TYPE}, {I64_DATA_TYPE}) -> {POINTER_DATA_TYPE}'
    )
    lines.append(_move(element_ptr_mov, element_ptr, POINTER_DATA_TYPE))
    lines.append(
        f'    {loaded} = "neura.load"({element_ptr_mov}) '
        f': ({POINTER_DATA_TYPE}) -> {result_type}'
    )
    return loaded


def _generate_memory_stream(operation_count: int, seed: int) -> str:
    """Generate independent scalar loads joined by a bounded arithmetic DAG."""
    rng = random.Random(seed)
    lines = [
        "module {",
        '  func.func @generated_memory_stream(%arg0: !llvm.ptr) '
        'attributes {accelerator = "neura"} {',
    ]
    lane_count = 2 + ((abs(seed) >> 6) % min(23, operation_count))
    loaded_values: List[str] = []
    for lane in range(lane_count):
        index = f"%ms_index{lane}"
        index_mov = f"%ms_index{lane}_mov"
        pointer = f"%ms_ptr{lane}"
        pointer_mov = f"%ms_ptr{lane}_mov"
        loaded = f"%ms_load{lane}"
        lines.append(_grant_once_constant(
            index, (abs(seed) + lane * 17) % 257, "i64", I64_DATA_TYPE
        ))
        lines.append(_move(index_mov, index, I64_DATA_TYPE))
        lines.append(
            f'    {pointer} = "neura.gep"({index_mov}) '
            '<{operandSegmentSizes = array<i32: 0, 1>}> '
            '{lhs_value = "%arg0"} '
            f': ({I64_DATA_TYPE}) -> {POINTER_DATA_TYPE}'
        )
        lines.append(_move(pointer_mov, pointer, POINTER_DATA_TYPE))
        lines.append(
            f'    {loaded} = "neura.load"({pointer_mov}) '
            f': ({POINTER_DATA_TYPE}) -> {DATA_TYPE}'
        )
        loaded_values.append(loaded)

    current = loaded_values[0]
    operation_index = 0
    for loaded in loaded_values[1:]:
        current = _emit_binary(
            lines, operation_index,
            _seeded_operation_kind(rng, operation_index, seed),
            current, loaded, name_prefix="ms",
        )
        operation_index += 1
    while operation_index < operation_count:
        current = _emit_unary_binary(
            lines, operation_index,
            _seeded_operation_kind(rng, operation_index, seed),
            current, 1 + rng.randrange(31), name_prefix="ms",
        )
        operation_index += 1
    return _emit_footer(lines)


def _generate_pointer_chase(operation_count: int, seed: int) -> str:
    """Generate a pointer chase with real indirect loads and a loop backedge."""
    rng = random.Random(seed)
    lines = [
        "module {",
        '  func.func @generated_pointer_chase(%arg0: !llvm.ptr, '
        '%arg1: !llvm.ptr, %arg2: !llvm.ptr) '
        'attributes {accelerator = "neura"} {',
    ]
    lines.append(_grant_once_constant("%pc_init", 0, "i64", I64_DATA_TYPE))
    lines.append("    %pc_reserved = neura.reserve : !neura.data<i64, i1>")
    lines.append(_move("%pc_init_mov", "%pc_init", I64_DATA_TYPE))
    lines.append(
        "    %pc_index = neura.phi_start %pc_init_mov, %pc_reserved "
        f": {I64_DATA_TYPE}, {I64_DATA_TYPE} -> {I64_DATA_TYPE}"
    )

    # Follow the known-good direct pointer fixture: one argument-backed GEP
    # loads a pointer, a second argument-backed GEP supplies a scalar index,
    # and the loaded pointer feeds an indirect GEP/load.  Keeping this
    # critical two-input GEP shape intact is important for the current
    # heuristic mapper; operation-count scaling is carried by the live
    # arithmetic chain below rather than by disconnected memory branches.
    current_pointer = _emit_pointer_root(lines, "%pc_index", "%arg1")
    # Derive the indirect-GEP index from a second, scalar load path.  This is
    # the same legal pattern used by Neura's checked pointer fixture and avoids
    # asking the mapper to route two independent uses of the loop phi into an
    # indirect GEP at the same scheduling level.
    lines.append(_move("%pc_index_base_index_mov", "%pc_index", I64_DATA_TYPE))
    index_base_arg = "%arg2" if (seed & 2) else "%arg0"
    lines.append(
        '    %pc_index_base = "neura.gep"(%pc_index_base_index_mov) '
        '<{operandSegmentSizes = array<i32: 0, 1>}> '
        f'{{lhs_value = "{index_base_arg}"}} '
        f': ({I64_DATA_TYPE}) -> {POINTER_DATA_TYPE}'
    )
    lines.append(_move("%pc_index_base_mov", "%pc_index_base", POINTER_DATA_TYPE))
    lines.append(
        f'    %pc_loaded_index = "neura.load"(%pc_index_base_mov) '
        f': ({POINTER_DATA_TYPE}) -> {DATA_TYPE}'
    )
    indirect_index = _emit_sext(
        lines, "pc_indirect_index", "%pc_loaded_index", DATA_TYPE, I64_DATA_TYPE
    )
    hop_count = 1 + ((abs(seed) >> 5) % 4)
    current = current_pointer
    for hop_index in range(hop_count):
        result_type = (
            DATA_TYPE if hop_index + 1 == hop_count else POINTER_DATA_TYPE
        )
        current = _emit_pointer_hop(
            lines, hop_index, current, indirect_index, result_type
        )
    for arithmetic_index in range(operation_count):
        current = _emit_unary_binary(
            lines, arithmetic_index,
            _seeded_operation_kind(rng, arithmetic_index, seed),
            current, 1 + ((abs(seed) + arithmetic_index * 5) % 13),
            name_prefix="pc"
        )

    # The terminal payload value is intentionally left as a sink, matching the
    # existing compute motifs.  Reusing the loop index for a late output GEP
    # would keep it live across the whole payload chain and makes Neura's
    # heuristic mapper repeatedly reject the candidate at the register-window
    # boundary.  The pointer/load/arithmetic path is already one weakly
    # connected component rooted at the loop index.
    # Keep the loop-index arithmetic names separate from payload arithmetic and
    # close the real recurrence with the frontend's predicate/ctrl_mov shape.
    lines.append(_move("%pc_next_index_input", "%pc_index", I64_DATA_TYPE))
    lines.append(
        f'    %pc_next_index = "neura.add"(%pc_next_index_input) '
        f'{{rhs_value = 1 : i64}} : ({I64_DATA_TYPE}) -> {I64_DATA_TYPE}'
    )
    lines.append(_move("%pc_next_index_mov", "%pc_next_index", I64_DATA_TYPE))
    _emit_icmp(
        lines, "pc_limit_cmp", "%pc_next_index", "eq",
        16 + (abs(seed) % 17), "i64", I64_DATA_TYPE
    )
    _emit_not(lines, "pc_continue", "%pc_limit_cmp")
    _emit_grant_predicate(
        lines, "pc_index_grant", "%pc_next_index", "%pc_continue",
        I64_DATA_TYPE
    )
    lines.append(
        "    neura.ctrl_mov %pc_index_grant -> %pc_reserved : "
        f"{I64_DATA_TYPE} {I64_DATA_TYPE}"
    )
    return _emit_footer(lines)


_GENERATORS = {
    "chain": _generate_chain,
    "fanout": _generate_fanout,
    "reduction": _generate_reduction,
    "diamond": _generate_diamond,
    "random_dag": _generate_random_dag,
    "recurrence_chain": _generate_recurrence_chain,
    "predicated_diamond": _generate_predicated_diamond,
    "memory_stream": _generate_memory_stream,
    "pointer_chase": _generate_pointer_chase,
}


def generate_motif_mlir(motif: str, operation_count: int, seed: int) -> str:
    """Generate one deterministic lowered Neura DFG for a motif family."""
    name = MOTIF_ALIASES.get(motif.strip().lower(), motif.strip().lower())
    if name not in _GENERATORS:
        raise ValueError(f"unknown motif: {motif}")
    bands = operation_bands_for_motif(name)
    minimum = min(low for low, _ in bands)
    maximum = max(high for _, high in bands)
    # Direct stress experiments may opt into larger graphs without silently
    # changing the frozen, mapper-feasible corpus distribution above.
    maximum = max(maximum, DIRECT_OPERATION_LIMITS.get(name, maximum))
    if operation_count < minimum:
        raise ValueError("operation_count is below the supported motif range")
    if operation_count > maximum:
        raise ValueError("operation_count exceeds the supported motif range")
    text = _GENERATORS[name](operation_count, int(seed))
    # Guard each family contract rather than assuming every op in a lowered
    # graph is a binary compute node. Pointer chasing has one
    # extra i64 index increment in its loop-control path; its payload remains
    # exactly operation_count i32 add/mul nodes.
    operation_lines = re.findall(r'"neura\.(?:add|mul)"', text)
    expected = operation_count + int(name == "pointer_chase")
    if len(operation_lines) != expected:
        raise AssertionError(
            f"{name} emitted {len(operation_lines)} operations, expected "
            f"{expected}"
        )
    if name == "recurrence_chain":
        if text.count("neura.reserve") != 1 or text.count("neura.phi_start") != 1:
            raise AssertionError("recurrence_chain must contain one reserve/phi")
        if text.count("neura.ctrl_mov") != 1:
            raise AssertionError("recurrence_chain must contain one backedge")
    elif name == "predicated_diamond":
        if not (text.count("neura.icmp") and text.count("neura.not") and
                text.count("neura.grant_predicate")):
            raise AssertionError("predicated_diamond is missing predicate arms")
        if text.count('"neura.add"') + text.count('"neura.mul"') != operation_count:
            raise AssertionError("predicated_diamond payload count mismatch")
    elif name == "pointer_chase":
        if text.count('"neura.gep"') < 2 or text.count('"neura.load"') < 2:
            raise AssertionError("pointer_chase must contain pointer/load chain")
        if text.count("neura.ctrl_mov") != 1:
            raise AssertionError("pointer_chase must contain one index backedge")
    elif name == "memory_stream":
        if text.count('"neura.gep"') < 2 or text.count('"neura.load"') < 2:
            raise AssertionError("memory_stream must contain multiple load lanes")
    return text


def canonical_dfg_text(text: str) -> str:
    """Canonicalize generated source for an auditable structural hash."""
    # Generated output is already canonical SSA.  Removing surrounding spaces
    # and blank lines makes the hash stable if a caller reformats the file,
    # while retaining operation names, operands, and types. Constant literal
    # values are normalized: changing data values alone is not a new mapping
    # graph and must not inflate the count of independent base DFGs.
    canonical_lines = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        normalized = " ".join(line.strip().split())
        normalized = re.sub(
            r"@generated_[A-Za-z0-9_]+", "@generated", normalized
        )
        normalized = re.sub(
            r"\bvalue = -?\d+ : i32\b", "value = 0 : i32", normalized
        )
        normalized = re.sub(
            r"\b(rhs_value|constant_value) = (-?\d+) : (i32|i64)\b",
            r"\1 = 0 : \3",
            normalized,
        )
        canonical_lines.append(normalized)
    return "\n".join(canonical_lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_dfg_sha256(text: str) -> str:
    return sha256_text(canonical_dfg_text(text))


def default_pinned_architecture() -> Path:
    return Path(__file__).resolve().parents[1] / PINNED_ARCHITECTURE_RELATIVE_PATH


def write_architecture(path: Path, rows: int, columns: int,
                       variant: str, registers: int = PINNED_REGISTERS_PER_TILE,
                       source: Optional[Path] = None) -> None:
    """Copy the exact pinned Neura YAML used by every target rectangle.

    Shape is supplied to Neura through the pass options, not by fabricating a
    second YAML.  Keeping this compatibility function makes that distinction
    explicit for callers that previously generated per-candidate YAML files.
    """
    if variant not in DEFAULT_ARCHITECTURE_VARIANTS:
        raise ValueError(f"unknown architecture variant: {variant}")
    if not (1 <= rows <= PINNED_ARCHITECTURE_ROWS and
            1 <= columns <= PINNED_ARCHITECTURE_COLUMNS):
        raise ValueError("target rectangle exceeds the pinned Neura 4x4 array")
    if registers != PINNED_REGISTERS_PER_TILE:
        raise ValueError("register count must match the pinned Neura architecture")
    architecture_source = (source or default_pinned_architecture()).resolve()
    if sha256_file(architecture_source) != PINNED_ARCHITECTURE_SHA256:
        raise ValueError("Neura architecture SHA-256 does not match the pinned YAML")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != architecture_source:
        shutil.copyfile(architecture_source, path)


def candidate_shapes_for_base(
    base: MotifBaseSpec, shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Return the balanced incomplete shape block for one base DFG."""
    selected = tuple(dict.fromkeys(shapes))
    if not selected:
        raise ValueError("at least one target shape is required")
    for shape in selected:
        if shape not in PREDICTION_SHAPES:
            raise ValueError(
                "target shape is not in the supported shape set: "
                f"{shape[0]}x{shape[1]}"
            )
    primary = PRIMARY_SHAPE if PRIMARY_SHAPE in selected else max(
        selected, key=lambda shape: (shape[0] * shape[1], shape[0], shape[1])
    )
    secondary = tuple(shape for shape in selected if shape != primary)
    if not secondary:
        return (primary,)
    paired = secondary[base.base_index % len(secondary)]
    return (primary, paired)


def make_candidates(
    bases: Sequence[MotifBaseSpec],
    output_dir: Path,
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
    architecture_source: Optional[Path] = None,
) -> Tuple[MotifCandidate, ...]:
    """Materialize source/architecture files and return predeclared candidates.

    This function performs no compiler or mapper invocation.  Callers can
    therefore invoke it, atomically write the manifest, and only then start
    any cost-model or mapper subprocess.
    """
    selected_shapes = tuple(shapes or DEFAULT_SHAPES)
    selected_variants = parse_architecture_variants(architecture_variants)
    if registers != PINNED_REGISTERS_PER_TILE:
        raise ValueError("register count must match the pinned Neura architecture")
    shared_architecture = output_dir / "architecture" / "neura-main-4x4.yaml"
    write_architecture(
        shared_architecture, PINNED_ARCHITECTURE_ROWS,
        PINNED_ARCHITECTURE_COLUMNS, "neura-main", registers,
        architecture_source,
    )
    architecture_sha = sha256_file(shared_architecture)
    result: List[MotifCandidate] = []
    seen_base_hashes: Dict[str, str] = {}
    for base in bases:
        source_text = generate_motif_mlir(
            base.motif, base.operation_count, base.base_seed
        )
        source_sha = sha256_text(source_text)
        canonical_sha = canonical_dfg_sha256(source_text)
        hash_key = canonical_sha
        prior_base = seen_base_hashes.get(hash_key)
        if prior_base is not None and prior_base != base.lineage:
            raise ValueError(
                "duplicate canonical DFG across motif corpus: "
                f"{base.lineage} duplicates {prior_base}"
            )
        seen_base_hashes[hash_key] = base.lineage
        for rows, columns in candidate_shapes_for_base(base, selected_shapes):
            for variant in selected_variants:
                source_dir = output_dir / "motifs" / base.motif / base.base_id
                candidate_dir = source_dir / f"{rows}x{columns}" / variant
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_path = candidate_dir / "input.mlir"
                # Every architecture candidate receives byte-identical
                # input source; the hashes are also stored in the manifest.
                source_path.write_text(source_text)
                target_config_id = f"prefix-{rows}x{columns}"
                candidate_id = (
                    f"{base.lineage}/{rows}x{columns}/{variant}/r{registers}"
                )
                result.append(MotifCandidate(
                    candidate_id=candidate_id,
                    lineage=base.lineage,
                    motif=base.motif,
                    generator_family=(
                        base.generator_family
                        if base.generator_family != "generated/motif"
                        else f"generated/motif/{base.motif}"
                    ),
                    generator_version=base.generator_version,
                    generator_type=base.generator_type,
                    base_id=base.base_id,
                    base_seed=base.base_seed,
                    root_seed=base.root_seed,
                    base_index=base.base_index,
                    operation_count=base.operation_count,
                    rows=rows,
                    columns=columns,
                    architecture_variant=variant,
                    registers=registers,
                    source_path=str(source_path),
                    architecture_path=str(shared_architecture),
                    source_sha256=source_sha,
                    canonical_dfg_sha256=canonical_sha,
                    architecture_sha256=architecture_sha,
                    target_config_id=target_config_id,
                ))
    return tuple(result)


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a JSON file in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(raw_path, path)
    finally:
        if os.path.exists(raw_path):
            os.unlink(raw_path)


def make_manifest(
    candidates: Sequence[MotifCandidate], output_dir: Path,
    seed: int, motifs: Sequence[str], shapes: Sequence[Tuple[int, int]],
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
) -> Dict[str, object]:
    """Build a pre-mapper manifest with every candidate in ``declared`` state."""
    records = []
    for candidate in candidates:
        record = candidate.manifest_record()
        record["source_path"] = str(Path(candidate.source_path).relative_to(output_dir))
        record["architecture_path"] = str(
            Path(candidate.architecture_path).relative_to(output_dir)
        )
        records.append(record)
    architecture_path = output_dir / "architecture" / "neura-main-4x4.yaml"
    if sha256_file(architecture_path) != PINNED_ARCHITECTURE_SHA256:
        raise ValueError("corpus architecture is not the pinned Neura YAML")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "predeclared",
        "generator": {
            "family": "generated/motif",
            "type": GENERATOR_TYPE,
            "version": GENERATOR_VERSION,
            "seed": seed,
            "motifs": list(motifs),
            "shapes": [f"{rows}x{columns}" for rows, columns in shapes],
            "architecture_variants": list(
                parse_architecture_variants(architecture_variants)
            ),
            "registers": int(registers),
            "candidate_design": SHAPE_DESIGN,
        },
        "architecture": {
            "path": architecture_path.relative_to(output_dir).as_posix(),
            "sha256": PINNED_ARCHITECTURE_SHA256,
            "neura_revision": PINNED_NEURA_REVISION,
            "source_path": PINNED_ARCHITECTURE_RELATIVE_PATH.as_posix(),
            "rows": PINNED_ARCHITECTURE_ROWS,
            "columns": PINNED_ARCHITECTURE_COLUMNS,
            "registers_per_tile": PINNED_REGISTERS_PER_TILE,
            "ctrl_mem_items": PINNED_CTRL_MEM_ITEMS,
            "target_shape_design": SHAPE_DESIGN,
            "valid_tiles": "",
        },
        # All paths in candidate records are relative to the manifest.  Using
        # "." here keeps a manifest reproducible when the corpus is generated
        # in two different temporary directories.
        "output_dir": ".",
        "candidate_count": len(records),
        "candidates": records,
        "summary": manifest_summary(records),
    }


def manifest_summary(candidates: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    records = list(candidates)
    return {
        "candidate_count": len(records),
        "declared_count": sum(record.get("status") == "declared" for record in records),
        "running_count": sum(record.get("status") == "running" for record in records),
        "success_count": sum(record.get("status") == "success" for record in records),
        "censored_count": sum(record.get("status") == "censored" for record in records),
    }


def update_manifest_candidate(
    manifest_path: Path, candidate_id: str, status: str, stage: str,
    failure: Optional[str] = None,
    updates: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Update one candidate and atomically persist the manifest."""
    if status not in ("declared", "running", "success", "censored"):
        raise ValueError(f"invalid manifest candidate status: {status}")
    payload = json.loads(manifest_path.read_text())
    found = False
    for record in payload.get("candidates", []):
        if record.get("id", record.get("candidate_id")) == candidate_id:
            record["status"] = status
            record["stage"] = stage
            record["failure"] = failure
            if updates:
                record.update(dict(updates))
            found = True
            break
    if not found:
        raise KeyError(f"manifest candidate not found: {candidate_id}")
    candidates = payload.get("candidates", [])
    payload["summary"] = manifest_summary(candidates)
    if candidates and all(record.get("status") == "success" for record in candidates):
        payload["status"] = "complete"
    elif any(record.get("status") == "censored" for record in candidates):
        payload["status"] = "partial"
    else:
        payload["status"] = "running"
    atomic_write_json(manifest_path, payload)
    return payload


__all__ = [
    "DEFAULT_ARCHITECTURE_VARIANTS", "DEFAULT_MOTIFS", "DEFAULT_SHAPES",
    "PREDICTION_SHAPES",
    "DIRECT_OPERATION_LIMITS", "FAMILY_OPERATION_BANDS", "GENERATOR_TYPE",
    "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MotifBaseSpec",
    "MotifCandidate", "OPERATION_BANDS", "atomic_write_json",
    "canonical_dfg_sha256", "canonical_dfg_text", "generate_motif_mlir",
    "make_base_specs", "make_candidates", "make_manifest",
    "manifest_summary", "parse_architecture_variants", "parse_motif_names",
    "operation_bands_for_motif", "parse_shape", "parse_shapes", "sha256_text",
    "MOTIF_ALIASES", "PRIMARY_SHAPE", "SHAPE_DESIGN",
    "PINNED_ARCHITECTURE_SHA256", "PINNED_NEURA_REVISION",
    "PINNED_ARCHITECTURE_ROWS", "PINNED_ARCHITECTURE_COLUMNS",
    "PINNED_REGISTERS_PER_TILE", "PINNED_CTRL_MEM_ITEMS",
    "PINNED_ARCHITECTURE_RELATIVE_PATH", "candidate_shapes_for_base",
    "default_pinned_architecture", "sha256_file", "stratified_operation_count",
    "update_manifest_candidate", "write_architecture",
]
