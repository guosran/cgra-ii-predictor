#!/usr/bin/env python3
"""Motif-v6: full rectangular-shape Top-1 corpus declaration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from adapters import neura_motifs_v4 as v4
    from adapters import neura_motifs_v5 as v5
except ImportError:  # Running from the adapters directory.
    import neura_motifs_v4 as v4  # type: ignore
    import neura_motifs_v5 as v5  # type: ignore


GENERATOR_VERSION = "motif-v6"
GENERATOR_TYPE = "generated/motif-v6"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v6"
DEFAULT_SEED = v5.DEFAULT_SEED

DEFAULT_MOTIFS = v5.DEFAULT_MOTIFS
MECHANISM_PROFILES = v5.MECHANISM_PROFILES
OPERATION_BANDS = v5.OPERATION_BANDS
DEFAULT_SHAPES = tuple(
    (rows, columns) for rows in range(1, 5) for columns in range(1, 5)
)
PREDICTION_SHAPES = DEFAULT_SHAPES
PRIMARY_SHAPE = (4, 4)
DEFAULT_ARCHITECTURE_VARIANTS = v5.DEFAULT_ARCHITECTURE_VARIANTS
PINNED_ARCHITECTURE_SHA256 = v5.PINNED_ARCHITECTURE_SHA256
PINNED_NEURA_REVISION = v5.PINNED_NEURA_REVISION
PINNED_ARCHITECTURE_ROWS = v5.PINNED_ARCHITECTURE_ROWS
PINNED_ARCHITECTURE_COLUMNS = v5.PINNED_ARCHITECTURE_COLUMNS
PINNED_REGISTERS_PER_TILE = v5.PINNED_REGISTERS_PER_TILE
PINNED_CTRL_MEM_ITEMS = v5.PINNED_CTRL_MEM_ITEMS
PINNED_ARCHITECTURE_RELATIVE_PATH = v5.PINNED_ARCHITECTURE_RELATIVE_PATH
STRATIFICATION_SCHEDULE = {
    **v5.STRATIFICATION_SCHEDULE,
    "shape_design": "every_base_crossed_with_all_16_rectangles",
}
SHAPE_DESIGN = "full-cartesian-all-rectangles-through-4x4-v1"
POINT_MODEL_FEATURE_NAMES = v5.POINT_MODEL_FEATURE_NAMES
HYBRID_PREDICTION_POLICY = v5.HYBRID_PREDICTION_POLICY

MotifBaseSpec = v5.MotifBaseSpec
MotifCandidate = v5.MotifCandidate
atomic_write_json = v5.atomic_write_json
canonical_dfg_sha256 = v5.canonical_dfg_sha256
canonical_dfg_text = v5.canonical_dfg_text
default_pinned_architecture = v5.default_pinned_architecture
generate_motif_mlir = v5.generate_motif_mlir
manifest_summary = v5.manifest_summary
parse_architecture_variants = v5.parse_architecture_variants
parse_motif_names = v5.parse_motif_names
parse_shape = v5.parse_shape
sha256_file = v5.sha256_file
sha256_text = v5.sha256_text
update_manifest_candidate = v5.update_manifest_candidate
write_architecture = v5.write_architecture


ACCEPTANCE_POLICY: Dict[str, object] = {
    **v5.ACCEPTANCE_POLICY,
    "policy_version": "motif-v6-top1-acceptance-v1",
    "ranking_population": "complete_16_rectangle_blocks_only",
    "coverage": {
        "requested_bases_per_family": 250,
        "minimum_successful_bases_per_family": 150,
        "minimum_complete_ranking_bases_per_family": 50,
    },
    "point_model": {
        **v5.ACCEPTANCE_POLICY["point_model"],
        "hyperparameter_selection_primary_metric": (
            "strict_top1_shape_accuracy"
        ),
        "primary_shape_metric": (
            "strict_top1_accuracy_with_minimum_area_tie_break"
        ),
        "top1_transfer": "strictly_higher_than_analytical_lower_bound",
        "pairwise_concordance": "secondary_non_degradation_diagnostic",
    },
}


def parse_shapes(values: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    if not values:
        return DEFAULT_SHAPES
    result: List[Tuple[int, int]] = []
    for value in values:
        shape = parse_shape(str(value))
        if shape not in DEFAULT_SHAPES:
            raise ValueError(
                f"motif-v6 shape must be within 1x1 through 4x4: {value}"
            )
        if shape not in result:
            result.append(shape)
    return tuple(result)


def shape_blocks(
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    selected = tuple(dict.fromkeys(shapes))
    if not selected or any(shape not in DEFAULT_SHAPES for shape in selected):
        raise ValueError("motif-v6 requires rectangles within the 4x4 mesh")
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
    """Reuse the still-unlabelled v5 DFG draw while changing only shape design."""
    return tuple(replace(
        base, generator_version=GENERATOR_VERSION, generator_type=GENERATOR_TYPE,
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
            raise ValueError("motif-v6 candidate received a non-v6 base spec")
        source_text = generate_motif_mlir(
            base.motif, base.operation_count, base.base_seed,
            base.mechanism_profile,
        )
        source_sha = sha256_text(source_text)
        canonical_sha = canonical_dfg_sha256(source_text)
        prior = seen_base_hashes.get(canonical_sha)
        if prior is not None and prior != base.lineage:
            raise ValueError(
                f"duplicate canonical DFG across motif-v6 corpus: {base.lineage}"
            )
        seen_base_hashes[canonical_sha] = base.lineage
        for rows, columns in selected_shapes:
            for variant in selected_variants:
                candidate_dir = (
                    output_dir / "motifs" / base.motif / base.base_id /
                    f"{rows}x{columns}" / variant
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_path = candidate_dir / "input.mlir"
                source_path.write_text(source_text)
                candidate_id = (
                    f"{base.lineage}/{rows}x{columns}/{variant}/r{registers}"
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
                    target_config_id=f"prefix-{rows}x{columns}",
                    mechanism_profile=base.mechanism_profile,
                    operation_band=base.operation_band,
                    shape_block="all-16-rectangles",
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
        record["source_path"] = str(
            Path(candidate.source_path).relative_to(output_dir)
        )
        record["architecture_path"] = str(
            Path(candidate.architecture_path).relative_to(output_dir)
        )
        records.append(record)
    architecture_path = output_dir / "architecture" / "neura-main-4x4.yaml"
    if sha256_file(architecture_path) != PINNED_ARCHITECTURE_SHA256:
        raise ValueError("corpus architecture is not the pinned Neura YAML")
    selected_shapes = tuple(shapes)
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
            "shapes": [f"{rows}x{columns}" for rows, columns in selected_shapes],
            "shape_blocks": [[
                f"{rows}x{columns}" for rows, columns in selected_shapes
            ]],
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
    "DEFAULT_SEED", "DEFAULT_SHAPES", "GENERATOR_TYPE", "GENERATOR_VERSION",
    "HYBRID_PREDICTION_POLICY", "MANIFEST_SCHEMA_VERSION",
    "MECHANISM_PROFILES", "MotifBaseSpec", "MotifCandidate", "OPERATION_BANDS",
    "PINNED_ARCHITECTURE_COLUMNS", "PINNED_ARCHITECTURE_RELATIVE_PATH",
    "PINNED_ARCHITECTURE_ROWS", "PINNED_ARCHITECTURE_SHA256",
    "PINNED_CTRL_MEM_ITEMS", "PINNED_NEURA_REVISION",
    "PINNED_REGISTERS_PER_TILE", "POINT_MODEL_FEATURE_NAMES",
    "PREDICTION_SHAPES", "PRIMARY_SHAPE", "SHAPE_DESIGN",
    "STRATIFICATION_SCHEDULE", "atomic_write_json", "candidate_shapes_for_base",
    "canonical_dfg_sha256", "canonical_dfg_text", "default_pinned_architecture",
    "generate_motif_mlir", "make_base_specs", "make_candidates", "make_manifest",
    "manifest_summary", "parse_architecture_variants", "parse_motif_names",
    "parse_shape", "parse_shapes", "sha256_file", "sha256_text", "shape_blocks",
    "update_manifest_candidate", "write_architecture",
]
