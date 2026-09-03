#!/usr/bin/env python3
"""Predeclared motif-v4 corpus generator.

Version 4 deliberately crosses six path contexts with five topology-pressure
profiles.  The cross-product prevents a positive mapper residual from being
identified only by one generator-family name.  This module is separate from
``neura_motifs`` so the historical motif-v3 generator and MachSuite freeze
evidence remain reproducible.
"""

from __future__ import annotations

import hashlib
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from adapters import neura_motifs as v3
except ImportError:  # Running from the adapters directory.
    import neura_motifs as v3  # type: ignore


GENERATOR_VERSION = "motif-v4"
GENERATOR_TYPE = "generated/motif-v4"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v4"
DEFAULT_SEED = 20260903
DEFAULT_MOTIFS = (
    "compute", "recurrence", "predicated", "memory", "pointer", "mixed",
)
MOTIF_ALIASES = {
    "control": "predicated",
    "memory_stream": "memory",
    "pointer_chase": "pointer",
    "hybrid": "mixed",
    "mixed_paths": "mixed",
}
MECHANISM_PROFILES = (
    "layered_sparse",
    "reconvergent_fork_join",
    "long_range_cutwidth",
    "live_range_pressure",
    "mixed_path_pressure",
)
# A common operation-count grid keeps size from serving as a generator-family
# proxy.  The topology profile is crossed with all three bands before repeat.
OPERATION_BANDS = ((8, 11), (12, 15), (16, 20))
STRATIFICATION_SCHEDULE = {
    "period_bases": 75,
    "operation_band_index": "base_index % 3",
    "shape_block_index": "base_index % 5",
    "mechanism_profile_index": "((base_index // 3) + (base_index // 15)) % 5",
}
FAMILY_OPERATION_BANDS = {
    motif: OPERATION_BANDS for motif in DEFAULT_MOTIFS
}
DIRECT_OPERATION_LIMITS = {motif: 48 for motif in DEFAULT_MOTIFS}
DEFAULT_SHAPES = v3.DEFAULT_SHAPES
PREDICTION_SHAPES = v3.PREDICTION_SHAPES
PRIMARY_SHAPE = v3.PRIMARY_SHAPE
DEFAULT_ARCHITECTURE_VARIANTS = v3.DEFAULT_ARCHITECTURE_VARIANTS
PINNED_ARCHITECTURE_SHA256 = v3.PINNED_ARCHITECTURE_SHA256
PINNED_NEURA_REVISION = v3.PINNED_NEURA_REVISION
PINNED_ARCHITECTURE_ROWS = v3.PINNED_ARCHITECTURE_ROWS
PINNED_ARCHITECTURE_COLUMNS = v3.PINNED_ARCHITECTURE_COLUMNS
PINNED_REGISTERS_PER_TILE = v3.PINNED_REGISTERS_PER_TILE
PINNED_CTRL_MEM_ITEMS = v3.PINNED_CTRL_MEM_ITEMS
PINNED_ARCHITECTURE_RELATIVE_PATH = v3.PINNED_ARCHITECTURE_RELATIVE_PATH
SHAPE_DESIGN = "full-4x4-plus-balanced-transpose-block-v2"

MotifBaseSpec = v3.MotifBaseSpec
MotifCandidate = v3.MotifCandidate
atomic_write_json = v3.atomic_write_json
canonical_dfg_sha256 = v3.canonical_dfg_sha256
canonical_dfg_text = v3.canonical_dfg_text
default_pinned_architecture = v3.default_pinned_architecture
manifest_summary = v3.manifest_summary
parse_architecture_variants = v3.parse_architecture_variants
parse_shape = v3.parse_shape
parse_shapes = v3.parse_shapes
sha256_file = v3.sha256_file
sha256_text = v3.sha256_text
update_manifest_candidate = v3.update_manifest_candidate
write_architecture = v3.write_architecture


ACCEPTANCE_POLICY: Dict[str, object] = {
    "policy_version": "motif-v4-acceptance-v1",
    "labels_for_protocol_selection": "generated_only",
    "machsuite_mapper_labels_used": False,
    "coverage": {
        "requested_bases_per_family": 250,
        "minimum_complete_bases_per_family": 200,
        "minimum_complete_fraction_per_declared_marginal_cell": 0.8,
        "marginal_dimensions": [
            "target_shape", "mechanism_profile", "operation_band",
        ],
    },
    "positive_residual_distribution": {
        "minimum_positive_base_dfgs_per_family": 20,
        "minimum_positive_mechanism_profiles_per_family": 2,
        "minimum_positive_operation_bands_per_family": 2,
        "minimum_positive_target_shapes_per_family": 2,
    },
    "model": {
        "class": "residual_ridge_model_1",
        "leave_one_generator_family_out": "strictly_lower_macro_mae_than_lb",
        "reject_all_floor_predictions": True,
        "positive_residual_recall": "strictly_greater_than_zero_in_every_family",
        "positive_subset_mae": "strictly_lower_than_lb_in_macro_family_average",
        "shape_balanced_mae": "strictly_lower_than_lb",
        "tie_aware_shape_ranking": "not_lower_than_lb",
    },
    "required_reports": [
        "per_generator_family", "per_target_shape", "per_operation_band",
        "positive_residual", "censored_and_mapper_search_coverage",
    ],
}


def parse_motif_names(values: Sequence[str]) -> Tuple[str, ...]:
    requested: List[str] = []
    for value in values:
        requested.extend(
            MOTIF_ALIASES.get(part.strip().lower(), part.strip().lower())
            for part in value.split(",") if part.strip()
        )
    if not requested:
        return DEFAULT_MOTIFS
    unknown = sorted(set(requested).difference(DEFAULT_MOTIFS))
    if unknown:
        raise ValueError(
            "unknown motif-v4 family(s): " + ", ".join(unknown) +
            "; available: " + ", ".join(DEFAULT_MOTIFS)
        )
    return tuple(dict.fromkeys(requested))


def operation_bands_for_motif(motif: str) -> Tuple[Tuple[int, int], ...]:
    name = MOTIF_ALIASES.get(motif.strip().lower(), motif.strip().lower())
    try:
        return FAMILY_OPERATION_BANDS[name]
    except KeyError as error:
        raise ValueError(f"unknown motif-v4 family: {motif}") from error


def operation_band_for_index(base_index: int) -> Tuple[str, Tuple[int, int]]:
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    index = base_index % len(OPERATION_BANDS)
    return ("low", "medium", "high")[index], OPERATION_BANDS[index]


def mechanism_profile_for_index(base_index: int) -> str:
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    # Consecutive low/medium/high bases share a profile. Rotate the profile
    # assignment after each complete 5-profile cycle so profile x size x
    # shape-block combinations all occur over 75 bases, without changing the
    # exactly balanced base_index % 5 shape-block schedule.
    profile_cycle = len(OPERATION_BANDS) * len(MECHANISM_PROFILES)
    within_cycle = (base_index // len(OPERATION_BANDS)) % len(MECHANISM_PROFILES)
    cycle_rotation = base_index // profile_cycle
    return MECHANISM_PROFILES[
        (within_cycle + cycle_rotation) % len(MECHANISM_PROFILES)
    ]


def stratified_operation_count(
    base_index: int, base_seed: int, motif: Optional[str] = None,
) -> int:
    if motif is not None:
        operation_bands_for_motif(motif)
    _, (low, high) = operation_band_for_index(base_index)
    return random.Random(base_seed).randint(low, high)


def _direct_load(
    lines: List[str], prefix: str, argument: str, index_value: int,
) -> str:
    index = f"%{prefix}_index"
    index_mov = f"%{prefix}_index_mov"
    pointer = f"%{prefix}_ptr"
    pointer_mov = f"%{prefix}_ptr_mov"
    loaded = f"%{prefix}_load"
    lines.append(v3._grant_once_constant(  # type: ignore[attr-defined]
        index, index_value, "i64", v3.I64_DATA_TYPE
    ))
    lines.append(v3._move(index_mov, index, v3.I64_DATA_TYPE))  # type: ignore[attr-defined]
    lines.append(
        f'    {pointer} = "neura.gep"({index_mov}) '
        '<{operandSegmentSizes = array<i32: 0, 1>}> '
        f'{{lhs_value = "{argument}"}} '
        f': ({v3.I64_DATA_TYPE}) -> {v3.POINTER_DATA_TYPE}'
    )
    lines.append(v3._move(  # type: ignore[attr-defined]
        pointer_mov, pointer, v3.POINTER_DATA_TYPE
    ))
    lines.append(
        f'    {loaded} = "neura.load"({pointer_mov}) '
        f': ({v3.POINTER_DATA_TYPE}) -> {v3.DATA_TYPE}'
    )
    return loaded


def _pointer_scalar_roots(lines: List[str], seed: int) -> List[str]:
    lines.append(v3._grant_once_constant(  # type: ignore[attr-defined]
        "%vp_index", abs(seed) % 31, "i64", v3.I64_DATA_TYPE
    ))
    loaded_pointer = v3._emit_pointer_root(  # type: ignore[attr-defined]
        lines, "%vp_index", "%arg1"
    )
    scalar_index = _direct_load(lines, "vp_scalar_index", "%arg0", 0)
    indirect_index = v3._emit_sext(  # type: ignore[attr-defined]
        lines, "vp_indirect_index", scalar_index,
        v3.DATA_TYPE, v3.I64_DATA_TYPE,
    )
    indirect = v3._emit_pointer_hop(  # type: ignore[attr-defined]
        lines, 0, loaded_pointer, indirect_index, v3.DATA_TYPE
    )
    return [
        indirect,
        _direct_load(lines, "vp_direct0", "%arg2", 1 + abs(seed) % 17),
        _direct_load(lines, "vp_direct1", "%arg2", 19 + abs(seed) % 17),
        scalar_index,
    ]


def _source_context(motif: str, seed: int) -> Tuple[List[str], List[str], str]:
    """Return header/body lines, scalar roots, and optional recurrence reserve."""
    if motif == "compute":
        lines = v3._emit_header("v4_compute")  # type: ignore[attr-defined]
        return lines, v3._emit_constants(lines, 4, seed=seed), ""  # type: ignore[attr-defined]
    if motif == "recurrence":
        lines = v3._emit_header("v4_recurrence")  # type: ignore[attr-defined]
        lines.append(v3._constant("%vr_init", 1 + abs(seed) % 17))  # type: ignore[attr-defined]
        lines.append("    %vr_reserved = neura.reserve : !neura.data<i32, i1>")
        lines.append(v3._move("%vr_init_mov", "%vr_init"))  # type: ignore[attr-defined]
        lines.append(
            "    %vr_state = neura.phi_start %vr_init_mov, %vr_reserved "
            f": {v3.DATA_TYPE}, {v3.DATA_TYPE} -> {v3.DATA_TYPE}"
        )
        constants = v3._emit_constants(lines, 3, seed=seed)  # type: ignore[attr-defined]
        return lines, ["%vr_state"] + constants, "%vr_reserved"
    if motif == "predicated":
        lines = v3._emit_header("v4_predicated")  # type: ignore[attr-defined]
        constants = v3._emit_constants(lines, 4, seed=seed)  # type: ignore[attr-defined]
        predicate = v3._emit_icmp(  # type: ignore[attr-defined]
            lines, "vpd_cmp", constants[0], "sgt", abs(seed) % 11
        )
        inverted = v3._emit_not(lines, "vpd_not", predicate)  # type: ignore[attr-defined]
        then_value = v3._emit_grant_predicate(  # type: ignore[attr-defined]
            lines, "vpd_then", constants[1], predicate
        )
        else_value = v3._emit_grant_predicate(  # type: ignore[attr-defined]
            lines, "vpd_else", constants[2], inverted
        )
        return lines, [then_value, else_value, constants[0], constants[3]], ""
    if motif == "memory":
        lines = [
            "module {",
            '  func.func @generated_v4_memory(%arg0: !llvm.ptr) '
            'attributes {accelerator = "neura"} {',
        ]
        roots = [
            _direct_load(lines, f"vm_lane{index}", "%arg0", index * 17 + abs(seed) % 13)
            for index in range(4)
        ]
        return lines, roots, ""
    if motif == "pointer":
        lines = [
            "module {",
            '  func.func @generated_v4_pointer(%arg0: !llvm.ptr, '
            '%arg1: !llvm.ptr, %arg2: !llvm.ptr) '
            'attributes {accelerator = "neura"} {',
        ]
        return lines, _pointer_scalar_roots(lines, seed), ""
    if motif == "mixed":
        lines = [
            "module {",
            '  func.func @generated_v4_mixed(%arg0: !llvm.ptr, '
            '%arg1: !llvm.ptr, %arg2: !llvm.ptr) '
            'attributes {accelerator = "neura"} {',
        ]
        pointer_roots = _pointer_scalar_roots(lines, seed)
        predicate = v3._emit_icmp(  # type: ignore[attr-defined]
            lines, "vx_cmp", pointer_roots[1], "slt", 1 + abs(seed) % 19
        )
        inverted = v3._emit_not(lines, "vx_not", predicate)  # type: ignore[attr-defined]
        return lines, [
            v3._emit_grant_predicate(  # type: ignore[attr-defined]
                lines, "vx_then", pointer_roots[0], predicate
            ),
            v3._emit_grant_predicate(  # type: ignore[attr-defined]
                lines, "vx_else", pointer_roots[2], inverted
            ),
            pointer_roots[1], pointer_roots[3],
        ], ""
    raise ValueError(f"unknown motif-v4 family: {motif}")


def _profiled_arithmetic(
    lines: List[str], roots: Sequence[str], operation_count: int,
    seed: int, profile: str,
) -> str:
    rng = random.Random(seed)
    sources = list(roots)
    rng.shuffle(sources)
    results: List[str] = []

    def emit(lhs: str, rhs: str) -> str:
        index = len(results)
        result = v3._emit_binary(  # type: ignore[attr-defined]
            lines, index, v3._seeded_operation_kind(rng, index, seed),  # type: ignore[attr-defined]
            lhs, rhs, name_prefix="v4",
        )
        results.append(result)
        return result

    # First connect all path roots into one weak component.  Every later
    # profile builds from this connected spine while retaining earlier roots.
    current = emit(sources[0], sources[1])
    for source in sources[2:]:
        if len(results) >= operation_count:
            break
        current = emit(current, source)

    if profile == "layered_sparse":
        width = 2 + abs(seed) % 3
        frontier = list(results[-width:]) or [current]
        while len(results) < operation_count:
            lhs = frontier[len(results) % len(frontier)]
            rhs = sources[(len(results) * 3) % len(sources)]
            current = emit(lhs, rhs)
            frontier.append(current)
            frontier = frontier[-width:]
    elif profile == "reconvergent_fork_join":
        while len(results) < operation_count:
            if operation_count - len(results) >= 3:
                left = emit(current, sources[len(results) % len(sources)])
                right = emit(current, sources[(len(results) + 1) % len(sources)])
                current = emit(left, right)
            else:
                current = emit(current, sources[len(results) % len(sources)])
    elif profile == "long_range_cutwidth":
        anchors = list(results) + list(sources)
        while len(results) < operation_count:
            # Reuse old values after a growing gap so many edges cross the
            # middle of the topological order.
            anchor = anchors[(len(results) * 5 + abs(seed)) % len(anchors)]
            current = emit(current, anchor)
            if len(results) <= 7:
                anchors.append(current)
    elif profile == "live_range_pressure":
        delayed: List[str] = list(sources)
        split = max(len(results), operation_count // 2)
        while len(results) < split:
            branch = emit(current, sources[len(results) % len(sources)])
            delayed.append(branch)
            current = branch
        while len(results) < operation_count:
            current = emit(current, delayed[(len(results) + abs(seed)) % len(delayed)])
    elif profile == "mixed_path_pressure":
        while len(results) < operation_count:
            lhs = roots[len(results) % len(roots)] if len(results) % 3 == 0 else current
            pool = list(roots) + results[:-1]
            rhs = pool[(len(results) * 7 + abs(seed)) % len(pool)]
            current = emit(lhs, rhs)
    else:
        raise ValueError(f"unknown motif-v4 mechanism profile: {profile}")
    return current


def generate_motif_mlir(
    motif: str, operation_count: int, seed: int,
    mechanism_profile: Optional[str] = None,
) -> str:
    name = MOTIF_ALIASES.get(motif.strip().lower(), motif.strip().lower())
    if name not in DEFAULT_MOTIFS:
        raise ValueError(f"unknown motif-v4 family: {motif}")
    minimum = min(low for low, _ in OPERATION_BANDS)
    maximum = DIRECT_OPERATION_LIMITS[name]
    if operation_count < minimum:
        raise ValueError("operation_count is below the supported motif-v4 range")
    if operation_count > maximum:
        raise ValueError("operation_count exceeds the supported motif-v4 range")
    profile = mechanism_profile or MECHANISM_PROFILES[abs(seed) % len(MECHANISM_PROFILES)]
    if profile not in MECHANISM_PROFILES:
        raise ValueError(f"unknown motif-v4 mechanism profile: {profile}")
    lines, roots, recurrence_reserve = _source_context(name, int(seed))
    terminal = _profiled_arithmetic(lines, roots, operation_count, int(seed), profile)
    if recurrence_reserve:
        lines.append(v3._move("%vr_back_mov", terminal))  # type: ignore[attr-defined]
        lines.append(
            f"    neura.ctrl_mov %vr_back_mov -> {recurrence_reserve} : "
            f"{v3.DATA_TYPE} {v3.DATA_TYPE}"
        )
    text = v3._emit_footer(lines)  # type: ignore[attr-defined]
    arithmetic = re.findall(r'"neura\.(?:add|mul)"', text)
    if len(arithmetic) != operation_count:
        raise AssertionError(
            f"motif-v4 {name}/{profile} emitted {len(arithmetic)} arithmetic "
            f"operations, expected {operation_count}"
        )
    return text


def make_base_specs(
    samples_per_family: int, seed: int = DEFAULT_SEED,
    motifs: Optional[Sequence[str]] = None,
) -> Tuple[MotifBaseSpec, ...]:
    if samples_per_family < 0:
        raise ValueError("samples_per_family must be non-negative")
    selected = parse_motif_names(motifs or ())
    result: List[MotifBaseSpec] = []
    for motif in selected:
        family_seed = int.from_bytes(hashlib.sha256(
            f"{GENERATOR_VERSION}:{seed}:{motif}".encode("utf-8")
        ).digest()[:8], "big")
        family_rng = random.Random(family_seed)
        seen = set()
        for base_index in range(samples_per_family):
            profile = mechanism_profile_for_index(base_index)
            band_name, _ = operation_band_for_index(base_index)
            for _ in range(10000):
                base_seed = family_rng.randrange(1 << 63)
                operation_count = stratified_operation_count(
                    base_index, base_seed, motif
                )
                canonical = canonical_dfg_sha256(generate_motif_mlir(
                    motif, operation_count, base_seed, profile
                ))
                if canonical not in seen:
                    seen.add(canonical)
                    break
            else:
                raise ValueError(
                    f"unable to generate a distinct canonical motif-v4 {motif} DFG"
                )
            result.append(MotifBaseSpec(
                motif=motif,
                base_index=base_index,
                base_seed=base_seed,
                operation_count=operation_count,
                root_seed=seed,
                generator_family=f"generated/motif/{motif}",
                generator_type="generated/motif-v4",
                generator_version=GENERATOR_VERSION,
                mechanism_profile=profile,
                operation_band=band_name,
            ))
    return tuple(result)


def shape_blocks(
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
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
    remaining = set(selected) - {primary}
    blocks: List[Tuple[Tuple[int, int], ...]] = []
    for shape in sorted(remaining, key=lambda item: (item[0] * item[1], item)):
        if shape not in remaining:
            continue
        transpose = (shape[1], shape[0])
        if transpose != shape and transpose in remaining:
            blocks.append((shape, transpose))
            remaining.remove(shape)
            remaining.remove(transpose)
        else:
            blocks.append((shape,))
            remaining.remove(shape)
    return tuple((primary,) + block for block in blocks) or ((primary,),)


def candidate_shapes_for_base(
    base: MotifBaseSpec, shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    blocks = shape_blocks(shapes)
    return blocks[base.base_index % len(blocks)]


def _shape_block_name(shapes: Sequence[Tuple[int, int]]) -> str:
    secondary = list(shapes[1:])
    if not secondary:
        return "primary-only"
    return "transpose-" + "-".join(f"{r}x{c}" for r, c in secondary) if (
        len(secondary) == 2 and secondary[0] == secondary[1][::-1]
    ) else "secondary-" + "-".join(f"{r}x{c}" for r, c in secondary)


def make_candidates(
    bases: Sequence[MotifBaseSpec], output_dir: Path,
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
    architecture_source: Optional[Path] = None,
) -> Tuple[MotifCandidate, ...]:
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
        if base.generator_version != GENERATOR_VERSION:
            raise ValueError("motif-v4 candidate received a non-v4 base spec")
        source_text = generate_motif_mlir(
            base.motif, base.operation_count, base.base_seed,
            base.mechanism_profile,
        )
        source_sha = sha256_text(source_text)
        canonical_sha = canonical_dfg_sha256(source_text)
        prior = seen_base_hashes.get(canonical_sha)
        if prior is not None and prior != base.lineage:
            raise ValueError(
                "duplicate canonical DFG across motif-v4 corpus: "
                f"{base.lineage} duplicates {prior}"
            )
        seen_base_hashes[canonical_sha] = base.lineage
        selected_block = candidate_shapes_for_base(base, selected_shapes)
        block_name = _shape_block_name(selected_block)
        for rows, columns in selected_block:
            for variant in selected_variants:
                candidate_dir = (
                    output_dir / "motifs" / base.motif / base.base_id /
                    f"{rows}x{columns}" / variant
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_path = candidate_dir / "input.mlir"
                source_path.write_text(source_text)
                target_config_id = f"prefix-{rows}x{columns}"
                candidate_id = (
                    f"{base.lineage}/{rows}x{columns}/{variant}/r{registers}"
                )
                result.append(MotifCandidate(
                    candidate_id=candidate_id,
                    lineage=base.lineage,
                    motif=base.motif,
                    generator_family=base.generator_family,
                    generator_version=GENERATOR_VERSION,
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
                    mechanism_profile=base.mechanism_profile,
                    operation_band=base.operation_band,
                    shape_block=block_name,
                ))
    return tuple(result)


def make_manifest(
    candidates: Sequence[MotifCandidate], output_dir: Path,
    seed: int, motifs: Sequence[str], shapes: Sequence[Tuple[int, int]],
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
) -> Dict[str, object]:
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
            "mechanism_profiles": list(MECHANISM_PROFILES),
            "operation_bands": {
                name: list(bounds) for name, bounds in zip(
                    ("low", "medium", "high"), OPERATION_BANDS
                )
            },
            "shapes": [f"{rows}x{columns}" for rows, columns in shapes],
            "shape_blocks": [
                [f"{rows}x{columns}" for rows, columns in block]
                for block in shape_blocks(shapes)
            ],
            "stratification_schedule": dict(STRATIFICATION_SCHEDULE),
            "architecture_variants": list(
                parse_architecture_variants(architecture_variants)
            ),
            "registers": int(registers),
            "candidate_design": SHAPE_DESIGN,
        },
        "acceptance_policy": ACCEPTANCE_POLICY,
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
        "label_boundary": {
            "manifest_written_before_mapper": True,
            "candidate_generation_uses_compiled_ii": False,
            "machsuite_mapper_labels_used": False,
        },
        "output_dir": ".",
        "candidate_count": len(records),
        "candidates": records,
        "summary": manifest_summary(records),
    }


__all__ = [
    "ACCEPTANCE_POLICY", "DEFAULT_ARCHITECTURE_VARIANTS", "DEFAULT_MOTIFS",
    "DEFAULT_SEED", "DEFAULT_SHAPES", "DIRECT_OPERATION_LIMITS",
    "FAMILY_OPERATION_BANDS", "GENERATOR_TYPE", "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "MECHANISM_PROFILES", "MotifBaseSpec", "MotifCandidate", "OPERATION_BANDS",
    "PINNED_ARCHITECTURE_COLUMNS", "PINNED_ARCHITECTURE_RELATIVE_PATH",
    "PINNED_ARCHITECTURE_ROWS", "PINNED_ARCHITECTURE_SHA256",
    "PINNED_CTRL_MEM_ITEMS", "PINNED_NEURA_REVISION",
    "PINNED_REGISTERS_PER_TILE", "PREDICTION_SHAPES", "PRIMARY_SHAPE",
    "SHAPE_DESIGN", "STRATIFICATION_SCHEDULE", "atomic_write_json",
    "candidate_shapes_for_base",
    "canonical_dfg_sha256", "canonical_dfg_text", "default_pinned_architecture",
    "generate_motif_mlir", "make_base_specs", "make_candidates", "make_manifest",
    "manifest_summary", "mechanism_profile_for_index", "operation_band_for_index",
    "operation_bands_for_motif", "parse_architecture_variants",
    "parse_motif_names", "parse_shape", "parse_shapes", "sha256_file",
    "sha256_text", "shape_blocks", "stratified_operation_count",
    "update_manifest_candidate", "write_architecture",
]
