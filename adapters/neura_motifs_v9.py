#!/usr/bin/env python3
"""Motif-v9: training-only large-operation augmentation for Model 2.

V9 keeps the frozen v8 mapper, architecture, topology grammar, and eight
oriented Amoeba rectangles.  It changes only the pre-label sampling stratum:
every family is crossed with 24--40 arithmetic operations so the point model
does not have to extrapolate from the v8 maximum of 20.  The independent root
seed and generator version make v9 lineages disjoint from the formal v8
validation/test population.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from adapters import neura_motifs_v4 as v4
    from adapters import neura_motifs_v8 as v8
except ImportError:  # Running from the adapters directory.
    import neura_motifs_v4 as v4  # type: ignore
    import neura_motifs_v8 as v8  # type: ignore


GENERATOR_VERSION = "motif-v9"
GENERATOR_TYPE = "generated/motif-v9"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v9"
DEFAULT_SEED = 20260913

DEFAULT_MOTIFS = v8.DEFAULT_MOTIFS
MECHANISM_PROFILES = v8.MECHANISM_PROFILES
# These bands are fixed before any application/taskflow labels are evaluated.
OPERATION_BANDS = ((24, 29), (30, 35), (36, 40))
FAMILY_OPERATION_BANDS = {
    motif: OPERATION_BANDS for motif in DEFAULT_MOTIFS
}
DEFAULT_SHAPES = v8.DEFAULT_SHAPES
PREDICTION_SHAPES = v8.PREDICTION_SHAPES
PRIMARY_SHAPE = v8.PRIMARY_SHAPE
PHYSICAL_TO_MAPPER = v8.PHYSICAL_TO_MAPPER
SHAPE_PROTOCOL_ID = v8.SHAPE_PROTOCOL_ID
SHAPE_DESIGN = v8.SHAPE_DESIGN
DEFAULT_ARCHITECTURE_VARIANTS = v8.DEFAULT_ARCHITECTURE_VARIANTS
PINNED_ARCHITECTURE_SHA256 = v8.PINNED_ARCHITECTURE_SHA256
PINNED_NEURA_REVISION = v8.PINNED_NEURA_REVISION
PINNED_ARCHITECTURE_ROWS = v8.PINNED_ARCHITECTURE_ROWS
PINNED_ARCHITECTURE_COLUMNS = v8.PINNED_ARCHITECTURE_COLUMNS
PINNED_REGISTERS_PER_TILE = v8.PINNED_REGISTERS_PER_TILE
PINNED_CTRL_MEM_ITEMS = v8.PINNED_CTRL_MEM_ITEMS
PINNED_ARCHITECTURE_RELATIVE_PATH = v8.PINNED_ARCHITECTURE_RELATIVE_PATH
POINT_MODEL_FEATURE_NAMES = v8.POINT_MODEL_FEATURE_NAMES
HYBRID_PREDICTION_POLICY = v8.HYBRID_PREDICTION_POLICY
STRATIFICATION_SCHEDULE = {
    **v8.STRATIFICATION_SCHEDULE,
    "operation_band_index": "base_index % 3 over fixed 24--40 bands",
    "purpose": "large-operation-training-only-augmentation",
}

MotifBaseSpec = v8.MotifBaseSpec
MotifCandidate = v8.MotifCandidate
atomic_write_json = v8.atomic_write_json
canonical_dfg_sha256 = v8.canonical_dfg_sha256
canonical_dfg_text = v8.canonical_dfg_text
default_pinned_architecture = v8.default_pinned_architecture
generate_motif_mlir = v8.generate_motif_mlir
manifest_summary = v8.manifest_summary
parse_architecture_variants = v8.parse_architecture_variants
parse_motif_names = v8.parse_motif_names
parse_shape = v8.parse_shape
sha256_file = v8.sha256_file
sha256_text = v8.sha256_text
update_manifest_candidate = v8.update_manifest_candidate
write_architecture = v8.write_architecture


ACCEPTANCE_POLICY: Dict[str, object] = {
    **v8.ACCEPTANCE_POLICY,
    "policy_version": "motif-v9-large-operation-augmentation-v1",
    "training_role": "training_only_augmentation",
    "selection_and_test_population": "frozen_motif_v8_only",
    "application_labels_used_for_generation_or_selection": False,
    "split_unit": "canonical_dfg_sha256",
}


def parse_shapes(values: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    return v8.parse_shapes(values)


def shape_blocks(
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return v8.shape_blocks(shapes)


def candidate_shapes_for_base(
    base: MotifBaseSpec, shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    return v8.candidate_shapes_for_base(base, shapes)


def operation_band_for_index(base_index: int) -> Tuple[str, Tuple[int, int]]:
    if base_index < 0:
        raise ValueError("base_index must be non-negative")
    index = base_index % len(OPERATION_BANDS)
    return ("low", "medium", "high")[index], OPERATION_BANDS[index]


def stratified_operation_count(
    base_index: int, base_seed: int, motif: Optional[str] = None,
) -> int:
    if motif is not None:
        parse_motif_names((motif,))
    _, (low, high) = operation_band_for_index(base_index)
    return random.Random(base_seed).randint(low, high)


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
            profile = v4.mechanism_profile_for_index(base_index)
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
                    f"unable to generate a distinct motif-v9 {motif} DFG"
                )
            result.append(MotifBaseSpec(
                motif=motif,
                base_index=base_index,
                base_seed=base_seed,
                operation_count=operation_count,
                root_seed=seed,
                generator_family=f"generated/motif/{motif}",
                generator_type=GENERATOR_TYPE,
                generator_version=GENERATOR_VERSION,
                mechanism_profile=profile,
                operation_band=band_name,
            ))
    return tuple(result)


def make_candidates(
    bases: Sequence[MotifBaseSpec], output_dir: Path,
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
    architecture_source: Optional[Path] = None,
) -> Tuple[MotifCandidate, ...]:
    """Materialize through the unchanged v8 shape protocol and relabel lineage."""
    for base in bases:
        if base.generator_version != GENERATOR_VERSION:
            raise ValueError("motif-v9 candidate received a non-v9 base spec")
    v8_bases = tuple(replace(
        base,
        generator_version=v8.GENERATOR_VERSION,
        generator_type=v8.GENERATOR_TYPE,
    ) for base in bases)
    v8_candidates = v8.make_candidates(
        v8_bases, output_dir, shapes, architecture_variants, registers,
        architecture_source,
    )
    base_by_id = {(base.motif, base.base_id): base for base in bases}
    result = []
    for candidate in v8_candidates:
        base = base_by_id[(candidate.motif, candidate.base_id)]
        suffix = candidate.candidate_id[len(candidate.lineage):]
        result.append(replace(
            candidate,
            candidate_id=base.lineage + suffix,
            lineage=base.lineage,
            generator_version=GENERATOR_VERSION,
            generator_type=GENERATOR_TYPE,
        ))
    return tuple(result)


def make_manifest(
    candidates: Sequence[MotifCandidate], output_dir: Path,
    seed: int, motifs: Sequence[str], shapes: Sequence[Tuple[int, int]],
    architecture_variants: Optional[Sequence[str]] = None,
    registers: int = PINNED_REGISTERS_PER_TILE,
) -> Dict[str, object]:
    manifest = v8.make_manifest(
        candidates, output_dir, seed, motifs, shapes,
        architecture_variants, registers,
    )
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["generator"].update({
        "type": GENERATOR_TYPE,
        "version": GENERATOR_VERSION,
        "operation_bands": {
            name: list(bounds) for name, bounds in zip(
                ("low", "medium", "high"), OPERATION_BANDS
            )
        },
        "stratification_schedule": dict(STRATIFICATION_SCHEDULE),
        "training_role": "training_only_augmentation",
    })
    manifest["acceptance_policy"] = ACCEPTANCE_POLICY
    return manifest


__all__ = [
    "ACCEPTANCE_POLICY", "DEFAULT_ARCHITECTURE_VARIANTS", "DEFAULT_MOTIFS",
    "DEFAULT_SEED", "DEFAULT_SHAPES", "FAMILY_OPERATION_BANDS",
    "GENERATOR_TYPE", "GENERATOR_VERSION", "MANIFEST_SCHEMA_VERSION",
    "MECHANISM_PROFILES", "MotifBaseSpec", "MotifCandidate",
    "OPERATION_BANDS", "PHYSICAL_TO_MAPPER", "PINNED_ARCHITECTURE_COLUMNS",
    "PINNED_ARCHITECTURE_RELATIVE_PATH", "PINNED_ARCHITECTURE_ROWS",
    "PINNED_ARCHITECTURE_SHA256", "PINNED_CTRL_MEM_ITEMS",
    "PINNED_NEURA_REVISION", "PINNED_REGISTERS_PER_TILE",
    "PREDICTION_SHAPES", "PRIMARY_SHAPE", "SHAPE_DESIGN",
    "SHAPE_PROTOCOL_ID", "STRATIFICATION_SCHEDULE", "atomic_write_json",
    "candidate_shapes_for_base", "canonical_dfg_sha256",
    "canonical_dfg_text", "default_pinned_architecture",
    "generate_motif_mlir", "make_base_specs", "make_candidates",
    "make_manifest", "manifest_summary", "operation_band_for_index",
    "parse_architecture_variants", "parse_motif_names", "parse_shape",
    "parse_shapes", "sha256_file", "sha256_text", "shape_blocks",
    "stratified_operation_count", "update_manifest_candidate",
    "write_architecture",
]
