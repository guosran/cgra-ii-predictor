#!/usr/bin/env python3
"""Motif-v8: real labels for Amoeba's static multi-CGRA rectangles.

``rows``/``columns`` remain mapper-tile aliases for the existing collector.
Every manifest record additionally stores unambiguous physical-CGRA and
mapper-tile dimensions.  Only the eight finite shapes declared below are
valid, and orientation is deliberately preserved.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from adapters import neura_motifs_v5 as v5
    from adapters import neura_motifs_v6 as v6
except ImportError:  # Running from the adapters directory.
    import neura_motifs_v5 as v5  # type: ignore
    import neura_motifs_v6 as v6  # type: ignore


GENERATOR_VERSION = "motif-v8"
GENERATOR_TYPE = "generated/motif-v8"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v8"
SHAPE_PROTOCOL_ID = "amoeba-static-rectangles-4x4-tiles-v1"
DEFAULT_SEED = 20260911

PHYSICAL_TO_MAPPER = (
    ((1, 1), (4, 4)),
    ((1, 2), (4, 8)),
    ((2, 1), (8, 4)),
    ((1, 3), (4, 12)),
    ((3, 1), (12, 4)),
    ((1, 4), (4, 16)),
    ((2, 2), (8, 8)),
    ((4, 1), (16, 4)),
)
DEFAULT_SHAPES = tuple(mapper for _, mapper in PHYSICAL_TO_MAPPER)
PREDICTION_SHAPES = DEFAULT_SHAPES
PRIMARY_SHAPE = (4, 4)
SHAPE_DESIGN = "amoeba-static-eight-oriented-rectangles-v1"

DEFAULT_MOTIFS = v6.DEFAULT_MOTIFS
MECHANISM_PROFILES = v6.MECHANISM_PROFILES
OPERATION_BANDS = v6.OPERATION_BANDS
DEFAULT_ARCHITECTURE_VARIANTS = v6.DEFAULT_ARCHITECTURE_VARIANTS
PINNED_ARCHITECTURE_SHA256 = v6.PINNED_ARCHITECTURE_SHA256
PINNED_NEURA_REVISION = v6.PINNED_NEURA_REVISION
PINNED_ARCHITECTURE_ROWS = v6.PINNED_ARCHITECTURE_ROWS
PINNED_ARCHITECTURE_COLUMNS = v6.PINNED_ARCHITECTURE_COLUMNS
PINNED_REGISTERS_PER_TILE = v6.PINNED_REGISTERS_PER_TILE
PINNED_CTRL_MEM_ITEMS = v6.PINNED_CTRL_MEM_ITEMS
PINNED_ARCHITECTURE_RELATIVE_PATH = v6.PINNED_ARCHITECTURE_RELATIVE_PATH
STRATIFICATION_SCHEDULE = {
    **v6.STRATIFICATION_SCHEDULE,
    "shape_design": "every_base_crossed_with_amoeba_static_eight",
    "orientation_equivalent": False,
}
POINT_MODEL_FEATURE_NAMES = v6.POINT_MODEL_FEATURE_NAMES
HYBRID_PREDICTION_POLICY = v6.HYBRID_PREDICTION_POLICY

MotifBaseSpec = v6.MotifBaseSpec
MotifCandidate = v6.MotifCandidate
atomic_write_json = v6.atomic_write_json
canonical_dfg_sha256 = v6.canonical_dfg_sha256
canonical_dfg_text = v6.canonical_dfg_text
default_pinned_architecture = v6.default_pinned_architecture
generate_motif_mlir = v6.generate_motif_mlir
manifest_summary = v6.manifest_summary
parse_architecture_variants = v6.parse_architecture_variants
parse_motif_names = v6.parse_motif_names
parse_shape = v6.parse_shape
sha256_file = v6.sha256_file
sha256_text = v6.sha256_text
update_manifest_candidate = v6.update_manifest_candidate
write_architecture = v6.write_architecture


ACCEPTANCE_POLICY: Dict[str, object] = {
    "policy_version": "motif-v8-amoeba-pointwise-ii-v1",
    "label_source": "pinned_neura_heuristic_mapper_only",
    "prediction_unit": "independent_task_dfg_mapper_tile_shape",
    "primary_metric": "validation_successful_candidate_continuous_ii_mae",
    "numeric_ii_population": "successful_mapper_candidates_only",
    "censored_population": "all_mapper_timeout_or_failure_candidates",
    "split_unit": "canonical_dfg_sha256",
    "orientation_equivalent": False,
    "shape_scope": "static_rectangular_only",
    "dynamic_and_nonrectangular_shapes": "unsupported_todo",
    "model_outputs": [
        "continuous_predicted_ii", "predicted_ii_std",
        "mapper_success_probability",
    ],
    "startup_cycles_source": "amoeba_frontend_analytical_input",
    "selection_data_boundary": "validation_only",
}


def parse_shapes(values: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    if not values:
        return DEFAULT_SHAPES
    result: List[Tuple[int, int]] = []
    for value in values:
        shape = parse_shape(str(value))
        if shape not in DEFAULT_SHAPES:
            raise ValueError(
                f"motif-v8 mapper shape is outside the finite Amoeba domain: {value}"
            )
        if shape not in result:
            result.append(shape)
    return tuple(result)


def shape_blocks(
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    selected = tuple(dict.fromkeys(shapes))
    if selected != DEFAULT_SHAPES:
        raise ValueError(
            "motif-v8 requires the complete ordered eight mapper-shape domain"
        )
    return (selected,)


def candidate_shapes_for_base(
    base: MotifBaseSpec, shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    del base
    return shape_blocks(shapes)[0]


def make_base_specs(
    samples_per_family: int, seed: int = DEFAULT_SEED,
    motifs: Optional[Sequence[str]] = None,
) -> Tuple[MotifBaseSpec, ...]:
    return tuple(replace(
        base, generator_version=GENERATOR_VERSION,
        generator_type=GENERATOR_TYPE,
    ) for base in v5.make_base_specs(samples_per_family, seed, motifs))


def make_candidates(
    bases: Sequence[MotifBaseSpec], output_dir: Path,
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
    architecture_source: Optional[Path] = None,
) -> Tuple[MotifCandidate, ...]:
    selected_shapes = tuple(shapes or DEFAULT_SHAPES)
    shape_blocks(selected_shapes)
    selected_variants = parse_architecture_variants(architecture_variants)
    if registers != PINNED_REGISTERS_PER_TILE:
        raise ValueError("register count must match the pinned Neura architecture")
    shared_architecture = output_dir / "architecture" / "neura-main-4x4.yaml"
    # The YAML describes one physical 4x4 block.  The pass options below,
    # recorded per candidate, override the cloned mapper tile dimensions.
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
            raise ValueError("motif-v8 candidate received a non-v8 base spec")
        source_text = generate_motif_mlir(
            base.motif, base.operation_count, base.base_seed,
            base.mechanism_profile,
        )
        source_sha = sha256_text(source_text)
        canonical_sha = canonical_dfg_sha256(source_text)
        prior = seen_base_hashes.get(canonical_sha)
        if prior is not None and prior != base.lineage:
            raise ValueError(
                f"duplicate canonical DFG across motif-v8 corpus: {base.lineage}"
            )
        seen_base_hashes[canonical_sha] = base.lineage
        for rows, columns in selected_shapes:
            for variant in selected_variants:
                candidate_dir = (
                    output_dir / "motifs" / base.motif / base.base_id /
                    f"mapper-{rows}x{columns}" / variant
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_path = candidate_dir / "input.mlir"
                source_path.write_text(source_text)
                candidate_id = (
                    f"{base.lineage}/mapper-{rows}x{columns}/{variant}/r{registers}"
                )
                result.append(MotifCandidate(
                    candidate_id=candidate_id,
                    lineage=base.lineage,
                    motif=base.motif,
                    generator_family=base.generator_family,
                    generator_version=GENERATOR_VERSION,
                    generator_type=GENERATOR_TYPE,
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
                    target_config_id=f"mapper-{rows}x{columns}",
                    mechanism_profile=base.mechanism_profile,
                    operation_band=base.operation_band,
                    shape_block="amoeba-static-eight-oriented-rectangles",
                ))
    return tuple(result)


def make_manifest(
    candidates: Sequence[MotifCandidate], output_dir: Path,
    seed: int, motifs: Sequence[str], shapes: Sequence[Tuple[int, int]],
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
) -> Dict[str, object]:
    shape_blocks(shapes)
    manifest = v6.make_manifest(
        candidates, output_dir, seed, motifs, shapes,
        architecture_variants, registers,
    )
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["generator"].update({
        "type": GENERATOR_TYPE,
        "version": GENERATOR_VERSION,
        "stratification_schedule": dict(STRATIFICATION_SCHEDULE),
        "candidate_design": SHAPE_DESIGN,
        "shape_unit": "mapper_tiles",
        "physical_cgra_shapes": [
            f"{physical[0]}x{physical[1]}"
            for physical, _ in PHYSICAL_TO_MAPPER
        ],
    })
    manifest["acceptance_policy"] = ACCEPTANCE_POLICY
    manifest["architecture"]["target_shape_design"] = SHAPE_DESIGN
    manifest["architecture"].update({
        "base_physical_cgra_mapper_tile_rows": 4,
        "base_physical_cgra_mapper_tile_cols": 4,
        "mapper_shape_override": "explicit_x_tiles_y_tiles",
    })
    manifest["shape_protocol"] = {
        "protocol_id": SHAPE_PROTOCOL_ID,
        "scope": "static_rectangular_only",
        "orientation_equivalent": False,
        "physical_to_mapper": [
            {
                "physical_cgra_rows": physical[0],
                "physical_cgra_cols": physical[1],
                "mapper_tile_rows": mapper[0],
                "mapper_tile_cols": mapper[1],
            }
            for physical, mapper in PHYSICAL_TO_MAPPER
        ],
    }
    mapper_to_physical = {mapper: physical for physical, mapper in PHYSICAL_TO_MAPPER}
    for record in manifest["candidates"]:
        mapper = (int(record["rows"]), int(record["columns"]))
        physical = mapper_to_physical[mapper]
        record.update({
            "physical_cgra_rows": physical[0],
            "physical_cgra_cols": physical[1],
            "mapper_tile_rows": mapper[0],
            "mapper_tile_cols": mapper[1],
            "shape_protocol_id": SHAPE_PROTOCOL_ID,
            "cache_identity": (
                f"{record['canonical_dfg_sha256']}:{mapper[0]}x{mapper[1]}"
            ),
        })
    return manifest


__all__ = [
    "ACCEPTANCE_POLICY", "DEFAULT_ARCHITECTURE_VARIANTS", "DEFAULT_MOTIFS",
    "DEFAULT_SEED", "DEFAULT_SHAPES", "GENERATOR_TYPE", "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION", "MECHANISM_PROFILES", "MotifBaseSpec",
    "MotifCandidate", "OPERATION_BANDS", "PHYSICAL_TO_MAPPER",
    "PINNED_ARCHITECTURE_COLUMNS", "PINNED_ARCHITECTURE_RELATIVE_PATH",
    "PINNED_ARCHITECTURE_ROWS", "PINNED_ARCHITECTURE_SHA256",
    "PINNED_CTRL_MEM_ITEMS", "PINNED_NEURA_REVISION",
    "PINNED_REGISTERS_PER_TILE", "PREDICTION_SHAPES", "PRIMARY_SHAPE",
    "SHAPE_DESIGN", "SHAPE_PROTOCOL_ID", "STRATIFICATION_SCHEDULE",
    "atomic_write_json", "candidate_shapes_for_base", "canonical_dfg_sha256",
    "canonical_dfg_text", "default_pinned_architecture", "generate_motif_mlir",
    "make_base_specs", "make_candidates", "make_manifest", "manifest_summary",
    "parse_architecture_variants", "parse_motif_names", "parse_shape",
    "parse_shapes", "sha256_file", "sha256_text", "shape_blocks",
    "update_manifest_candidate", "write_architecture",
]
