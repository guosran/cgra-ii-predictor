#!/usr/bin/env python3
"""Deterministic compute-motif corpus generation for the Neura adapter.

The legacy ``--samples`` generator in :mod:`neura_experiment` is intentionally
kept as a compatibility path.  This module is the newer, auditable corpus
stratum: a base DFG is generated once from ``(generator version, motif,
base_seed, operation count)`` and then paired with several architectural
candidates.  Shape and architectural variation never changes the base DFG,
which makes it possible to group all variants under one leakage-safe lineage.

Only compute motifs are generated here.  Memory and control coverage remains
owned by the existing C/frontend generator and by real benchmark fixtures.
The emitted IR is already in the lowered Neura dataflow dialect and uses only
constant, data_mov, add, and mul operations, so the normal analytical-cost and
heuristic mapping passes can consume it directly.
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


GENERATOR_VERSION = "motif-v1"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v1"
DATA_TYPE = "!neura.data<i32, i1>"
DEFAULT_MOTIFS = (
    "chain", "fanout", "reduction", "diamond", "mixed", "random_dag",
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
}
DEFAULT_SHAPES = ((3, 3), (3, 4), (4, 4))
# Three strata make the operation-count distribution explicit and reproducible.
# The lower edge is deliberately above the tiny legacy examples: these are
# intended to exercise graph pressure while still being practical smoke tests.
OPERATION_BANDS = ((8, 15), (16, 31), (32, 48))
DEFAULT_ARCHITECTURE_VARIANTS = ("homogeneous", "split-domain")


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
        # candidate after cost-model/mapper success or censorship.
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


def stratified_operation_count(base_index: int, base_seed: int) -> int:
    """Choose an operation count from each band in a cyclic stratification."""
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    band_low, band_high = OPERATION_BANDS[base_index % len(OPERATION_BANDS)]
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
                    base_index, base_seed
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


def _constant(value: str, number: int) -> str:
    return (
        f'    {value} = "neura.constant"() '
        f'<{{value = {number} : i32}}> : () -> {DATA_TYPE}'
    )


def _move(result: str, operand: str) -> str:
    return (
        f'    {result} = "neura.data_mov"({operand}) '
        f': ({DATA_TYPE}) -> {DATA_TYPE}'
    )


def _binary(result: str, operation: str, lhs: str, rhs: str) -> str:
    if operation not in ("add", "mul"):
        raise ValueError(f"unsupported generated compute operation: {operation}")
    return (
        f'    {result} = "neura.{operation}"({lhs}, {rhs}) '
        f': ({DATA_TYPE}, {DATA_TYPE}) -> {DATA_TYPE}'
    )


def _operation_kind(rng: random.Random, index: int) -> str:
    # The index tie-break prevents a rare all-one-op random stream from making
    # generated source hashes unhelpfully homogeneous.
    return "add" if ((rng.randrange(2) + index) % 2 == 0) else "mul"


def _emit_binary(
    lines: List[str], operation_index: int, operation: str,
    lhs: str, rhs: str,
) -> str:
    lhs_move = f"%m{operation_index}a"
    rhs_move = f"%m{operation_index}b"
    result = f"%v{operation_index}"
    lines.append(_move(lhs_move, lhs))
    lines.append(_move(rhs_move, rhs))
    lines.append(_binary(result, operation, lhs_move, rhs_move))
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


_GENERATORS = {
    "chain": _generate_chain,
    "fanout": _generate_fanout,
    "reduction": _generate_reduction,
    "diamond": _generate_diamond,
    "mixed": _generate_mixed,
    "random_dag": _generate_random_dag,
}


def generate_motif_mlir(motif: str, operation_count: int, seed: int) -> str:
    """Generate one deterministic lowered Neura compute DFG."""
    name = MOTIF_ALIASES.get(motif.strip().lower(), motif.strip().lower())
    if name not in _GENERATORS:
        raise ValueError(f"unknown motif: {motif}")
    if operation_count < OPERATION_BANDS[0][0]:
        raise ValueError("operation_count is below the supported motif range")
    if operation_count > OPERATION_BANDS[-1][1]:
        raise ValueError("operation_count exceeds the supported motif range")
    text = _GENERATORS[name](operation_count, int(seed))
    # Guard the key contract here instead of relying on a regex in tests.
    operation_lines = re.findall(r'"neura\.(?:add|mul)"', text)
    if len(operation_lines) != operation_count:
        raise AssertionError(
            f"{name} emitted {len(operation_lines)} operations, expected "
            f"{operation_count}"
        )
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
    if variant == "split-domain":
        # Main-branch MemMII requires both memory FU classes to exist even for
        # a compute-only DFG with zero memory operations. They remain uniform
        # across variants and are not used by the generated compute nodes.
        tile_defaults = '["constant", "mem", "mem_indexed"]'
        first_compute_column = max(0, columns // 2)
        overrides = "\n".join(
            "  - tile_x: {x}\n"
            "    tile_y: {y}\n"
            '    fu_types: ["add", "mul", "mem", "mem_indexed"]\n'
            "    num_registers: {registers}\n"
            "    existence: true".format(x=x, y=y, registers=registers)
            for y in range(rows)
            for x in range(first_compute_column, columns)
        )
    else:
        tile_defaults = '["constant", "add", "mul", "mem", "mem_indexed"]'
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
    "GENERATOR_VERSION", "MANIFEST_SCHEMA_VERSION", "MotifBaseSpec",
    "MotifCandidate", "OPERATION_BANDS", "atomic_write_json",
    "canonical_dfg_sha256", "canonical_dfg_text", "generate_motif_mlir",
    "make_base_specs", "make_candidates", "make_manifest",
    "manifest_summary", "parse_architecture_variants", "parse_motif_names",
    "parse_shape", "parse_shapes", "sha256_text", "MOTIF_ALIASES",
    "stratified_operation_count", "update_manifest_candidate",
    "write_architecture",
]
