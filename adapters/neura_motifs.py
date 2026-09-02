#!/usr/bin/env python3
"""Deterministic compute-motif corpus generation for the Neura adapter.

The legacy ``--samples`` generator in :mod:`neura_experiment` is intentionally
kept as a compatibility path.  This module is the newer, auditable corpus
stratum: a base DFG is generated once from ``(generator version, motif,
base_seed, operation count)`` and then paired with several architectural
candidates.  Shape and architectural variation never changes the base DFG,
which makes it possible to group all variants under one leakage-safe lineage.

The emitted IR is already in the lowered Neura dataflow dialect.  The original
six families remain compute-only compatibility motifs; v2 additionally emits
recurrence, predicated-control, and pointer-chasing DFGs directly.  No C
frontend or compiler lowering is involved, so each structural edge is visible
to the analysis-only Rec/Res pass and heuristic mapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


GENERATOR_VERSION = "motif-v2"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v2"
DATA_TYPE = "!neura.data<i32, i1>"
I64_DATA_TYPE = "!neura.data<i64, i1>"
PREDICATE_DATA_TYPE = "!neura.data<i1, i1>"
POINTER_DATA_TYPE = "!neura.data<!llvm.ptr, i1>"
DEFAULT_MOTIFS = (
    "chain", "fanout", "reduction", "diamond", "mixed", "random_dag",
    "recurrence_chain", "predicated_diamond", "pointer_chase",
)
MOTIF_ALIASES = {
    "broadcast": "fanout",
    "fanout_broadcast": "fanout",
    "binary_reduction": "reduction",
    "reduction_tree": "reduction",
    "split_join": "diamond",
    "mixed_dag": "mixed",
    "multi_input": "mixed",
    "random": "random_dag",
    "random-dag": "random_dag",
    "recurrence": "recurrence_chain",
    "loop": "recurrence_chain",
    "predicated": "predicated_diamond",
    "control": "predicated_diamond",
    "pointer": "pointer_chase",
    "pointer-chase": "pointer_chase",
}
DEFAULT_SHAPES = ((3, 3), (3, 4), (4, 4))
# Three strata make the operation-count distribution explicit and reproducible.
# The lower edge is deliberately above the tiny legacy examples: these are
# intended to exercise graph pressure while still being practical smoke tests.
OPERATION_BANDS = ((8, 15), (16, 31), (32, 48))
# v1 callers use OPERATION_BANDS directly, so retain it as the compatibility
# contract for the six original compute motifs.  v2's direct-lowered motifs
# use family-specific bands.  The frozen corpus stays in the tens-of-operations
# regime used by LISA and by mapper-feasible Neura examples; larger direct API
# limits remain available for explicitly declared stress experiments.
FAMILY_OPERATION_BANDS = {
    "chain": OPERATION_BANDS,
    "fanout": OPERATION_BANDS,
    "reduction": OPERATION_BANDS,
    "diamond": OPERATION_BANDS,
    "mixed": OPERATION_BANDS,
    "random_dag": OPERATION_BANDS,
    # Corpus generation tops recurrence cycles out at 32 so the size tiers
    # cover RecMII without making every large sample dominated by one loop.
    # The direct API still accepts 33--48 for compatibility (see below).
    "recurrence_chain": ((8, 15), (16, 23), (24, 32)),
    "predicated_diamond": OPERATION_BANDS,
    "pointer_chase": OPERATION_BANDS,
}
DIRECT_OPERATION_LIMITS = {
    "random_dag": 160,
    "recurrence_chain": 48,
    "predicated_diamond": 128,
    "pointer_chase": 128,
}
DEFAULT_ARCHITECTURE_VARIANTS = ("homogeneous", "split-domain")

# These are the FU classes present in Neura's main architecture.yaml.  The
# generated architecture enables memory classes on every homogeneous tile and
# on the compute half of a split-domain mesh, so all v2 operations have a
# legal placement domain.
NEURA_FU_TYPES = (
    "add", "mul", "div", "fadd", "fmul", "fdiv", "logic", "cmp", "sel",
    "type_conv", "vfmul", "fadd_fadd", "fmul_fadd", "grant", "loop_control",
    "phi", "constant", "return", "alloca", "shift", "mem", "mem_indexed",
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
    status: str = "declared"
    stage: str = "predeclared"
    failure: Optional[str] = None

    @property
    def architecture_id(self) -> str:
        return f"{self.architecture_sha256}:{self.architecture_variant}"

    def manifest_record(self) -> Dict[str, object]:
        record = asdict(self)
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
    the family explicitly so v2 corpora include large structural examples.
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
    values = _emit_constants(lines, operation_count + 1, seed=seed)
    current = values[0]
    for index in range(operation_count):
        current = _emit_binary(
            lines, index, _operation_kind(rng, index), current, values[index + 1]
        )
    return _emit_footer(lines)


def _generate_fanout(operation_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = _emit_header("fanout")
    constants = _emit_constants(lines, operation_count + 1, seed=seed)
    branch_count = max(2, min(4, operation_count // 4))
    branches: List[str] = []
    # Each branch starts from the same source.  Remaining operations extend
    # branches round-robin, preserving a large semantic fanout at the root.
    for index in range(branch_count):
        branches.append(_emit_binary(
            lines, index, _operation_kind(rng, index), constants[0],
            constants[index + 1],
        ))
    for index in range(branch_count, operation_count):
        branch = (index - branch_count) % branch_count
        branches[branch] = _emit_binary(
            lines, index, _operation_kind(rng, index), branches[branch],
            constants[index + 1],
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
    constants = _emit_constants(lines, operation_count + 2, seed=seed)
    first = _emit_binary(lines, 0, _operation_kind(rng, 0), constants[0], constants[1])
    second = _emit_binary(lines, 1, _operation_kind(rng, 1), constants[0], constants[2])
    current = _emit_binary(lines, 2, _operation_kind(rng, 2), first, second)
    for index in range(3, operation_count):
        current = _emit_binary(
            lines, index, _operation_kind(rng, index), current, constants[index + 1]
        )
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
                pool = constants + results
            else:
                pool = constants + results
                rhs = rng.choice(pool)
            if rhs == lhs and len(pool) > 1:
                alternatives = [value for value in pool if value != lhs]
                rhs = rng.choice(alternatives)
            # Periodically reuse an older producer to increase long-range
            # fanout; otherwise random choice still permits reconvergence.
            if index >= 3 and index % 4 == 0:
                lhs = results[rng.randrange(max(1, len(results) // 2))]
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
    """Generate seeded reconvergent predicate diamonds.

    A shallow pair of predicated arms feeds one reconvergent join.  Remaining
    operations extend a serial post-join tail, so size changes graph topology
    without keeping the predicate live across a long arm or causing explosive
    control backtracking in the current heuristic mapper.
    """
    rng = random.Random(seed)
    lines = _emit_header("predicated_diamond")
    constants = _emit_constants(lines, 2, seed=seed)
    current = constants[0]
    arithmetic_index = 0
    # Keep exactly one real reconvergent control diamond.  A previous design
    # scaled the arm depths and routinely timed out at only 32 operations
    # because the predicate had to stay live across both arms.
    predicate = _emit_icmp(
        lines, "pd_cmp", current,
        "sgt" if (seed & 1) else "slt",
        (abs(seed) % 9), "i32", DATA_TYPE
    )
    inverted = _emit_not(lines, "pd_not", predicate)
    # The two arms have distinct source roots for a genuine split/join.
    then_current = _emit_unary_binary(
        lines, arithmetic_index,
        _seeded_operation_kind(rng, arithmetic_index, seed),
        current, 1 + (abs(seed) % 11), name_prefix="pd"
    )
    arithmetic_index += 1
    else_current = _emit_unary_binary(
        lines, arithmetic_index,
        _seeded_operation_kind(rng, arithmetic_index, seed),
        constants[1], 1 + ((abs(seed) + 1) % 11), name_prefix="pd"
    )
    arithmetic_index += 1
    then_value = _emit_grant_predicate(lines, "pd_then", then_current, predicate)
    else_value = _emit_grant_predicate(lines, "pd_else", else_current, inverted)
    current = _emit_binary(
        lines, arithmetic_index,
        _seeded_operation_kind(rng, arithmetic_index, seed),
        then_value, else_value, name_prefix="pd"
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
    current = _emit_pointer_hop(
        lines, 0, current_pointer, indirect_index, DATA_TYPE
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
    "mixed": _generate_mixed,
    "random_dag": _generate_random_dag,
    "recurrence_chain": _generate_recurrence_chain,
    "predicated_diamond": _generate_predicated_diamond,
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
    # graph is one of the six v1 binary compute nodes. Pointer chasing has one
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


def canonical_dfg_sha256(text: str) -> str:
    return sha256_text(canonical_dfg_text(text))


def write_architecture(path: Path, rows: int, columns: int,
                       variant: str, registers: int = 16) -> None:
    """Write a deterministic Neura architecture for a motif candidate."""
    if variant not in DEFAULT_ARCHITECTURE_VARIANTS:
        raise ValueError(f"unknown architecture variant: {variant}")
    fu_types = json.dumps(list(NEURA_FU_TYPES), separators=(", ", ": "))
    if variant == "split-domain":
        # Left tiles are memory/source-domain tiles; right tiles have the full
        # operation set.  This keeps the split architecture materially
        # different while retaining a legal placement domain for every v2 op.
        tile_defaults = json.dumps(
            ["constant", "mem", "mem_indexed"], separators=(", ", ": ")
        )
        first_compute_column = max(0, columns // 2)
        overrides = "\n".join(
            "  - tile_x: {x}\n"
            "    tile_y: {y}\n"
            "    fu_types: {fu_types}\n"
            "    num_registers: {registers}\n"
            "    existence: true".format(
                x=x, y=y, registers=registers, fu_types=fu_types
            )
            for y in range(rows)
            for x in range(first_compute_column, columns)
        )
    else:
        tile_defaults = fu_types
        overrides = ""
    path.write_text("\n".join((
        "architecture:",
        '  name: "II Predictor Generated Motif"',
        '  version: "1.0"',
        "",
        "multi_cgra_defaults:",
        '  base_topology: "mesh"',
        "  rows: 1",
        "  columns: 1",
        "",
        "per_cgra_defaults:",
        f"  rows: {rows}",
        f"  columns: {columns}",
        "  ctrl_mem_items: 64",
        '  base_topology: "mesh"',
        "",
        "tile_defaults:",
        f"  num_registers: {registers}",
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
    )))


def make_candidates(
    bases: Sequence[MotifBaseSpec],
    output_dir: Path,
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = 16,
) -> Tuple[MotifCandidate, ...]:
    """Materialize source/architecture files and return predeclared candidates.

    This function performs no compiler or mapper invocation.  Callers can
    therefore invoke it, atomically write the manifest, and only then start
    any cost-model or mapper subprocess.
    """
    selected_shapes = tuple(shapes or DEFAULT_SHAPES)
    selected_variants = parse_architecture_variants(architecture_variants)
    if registers <= 0:
        raise ValueError("registers must be positive")
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
        for rows, columns in selected_shapes:
            if rows < 1 or columns < 1:
                raise ValueError(f"invalid motif shape: {rows}x{columns}")
            for variant in selected_variants:
                source_dir = output_dir / "motifs" / base.motif / base.base_id
                candidate_dir = source_dir / f"{rows}x{columns}" / variant
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_path = candidate_dir / "input.mlir"
                architecture_path = candidate_dir / "architecture.yaml"
                # Every architecture candidate receives byte-identical
                # input source; the hashes are also stored in the manifest.
                source_path.write_text(source_text)
                write_architecture(
                    architecture_path, rows, columns, variant, registers
                )
                architecture_sha = sha256_text(architecture_path.read_text())
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
                    operation_count=base.operation_count,
                    rows=rows,
                    columns=columns,
                    architecture_variant=variant,
                    registers=registers,
                    source_path=str(source_path),
                    architecture_path=str(architecture_path),
                    source_sha256=source_sha,
                    canonical_dfg_sha256=canonical_sha,
                    architecture_sha256=architecture_sha,
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
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "predeclared",
        "generator": {
            "family": "generated/motif",
            "type": "generated/motif",
            "version": GENERATOR_VERSION,
            "seed": seed,
            "motifs": list(motifs),
            "shapes": [f"{rows}x{columns}" for rows, columns in shapes],
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
    "DIRECT_OPERATION_LIMITS", "FAMILY_OPERATION_BANDS", "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MotifBaseSpec",
    "MotifCandidate", "OPERATION_BANDS", "atomic_write_json",
    "canonical_dfg_sha256", "canonical_dfg_text", "generate_motif_mlir",
    "make_base_specs", "make_candidates", "make_manifest",
    "manifest_summary", "parse_architecture_variants", "parse_motif_names",
    "operation_bands_for_motif", "parse_shape", "parse_shapes", "sha256_text",
    "MOTIF_ALIASES", "NEURA_FU_TYPES",
    "stratified_operation_count", "update_manifest_candidate",
    "write_architecture",
]
