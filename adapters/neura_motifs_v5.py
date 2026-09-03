#!/usr/bin/env python3
"""Label-free motif-v5 corpus declaration.

V5 deliberately reuses the frozen v4 graph grammar and shape-block design but
draws a disjoint deterministic corpus.  The version bump is for the changed
training/evaluation contract: every successful mapper result is a point-model
label, while only complete shape blocks are eligible for ranking metrics.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from adapters import neura_motifs as v3
    from adapters import neura_motifs_v4 as v4
except ImportError:  # Running from the adapters directory.
    import neura_motifs as v3  # type: ignore
    import neura_motifs_v4 as v4  # type: ignore


GENERATOR_VERSION = "motif-v5"
GENERATOR_TYPE = "generated/motif-v5"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v5"
DEFAULT_SEED = 20260904

DEFAULT_MOTIFS = v4.DEFAULT_MOTIFS
MECHANISM_PROFILES = v4.MECHANISM_PROFILES
OPERATION_BANDS = v4.OPERATION_BANDS
FAMILY_OPERATION_BANDS = v4.FAMILY_OPERATION_BANDS
DIRECT_OPERATION_LIMITS = v4.DIRECT_OPERATION_LIMITS
DEFAULT_SHAPES = v4.DEFAULT_SHAPES
PREDICTION_SHAPES = v4.PREDICTION_SHAPES
PRIMARY_SHAPE = v4.PRIMARY_SHAPE
DEFAULT_ARCHITECTURE_VARIANTS = v4.DEFAULT_ARCHITECTURE_VARIANTS
PINNED_ARCHITECTURE_SHA256 = v4.PINNED_ARCHITECTURE_SHA256
PINNED_NEURA_REVISION = v4.PINNED_NEURA_REVISION
PINNED_ARCHITECTURE_ROWS = v4.PINNED_ARCHITECTURE_ROWS
PINNED_ARCHITECTURE_COLUMNS = v4.PINNED_ARCHITECTURE_COLUMNS
PINNED_REGISTERS_PER_TILE = v4.PINNED_REGISTERS_PER_TILE
PINNED_CTRL_MEM_ITEMS = v4.PINNED_CTRL_MEM_ITEMS
PINNED_ARCHITECTURE_RELATIVE_PATH = v4.PINNED_ARCHITECTURE_RELATIVE_PATH
SHAPE_DESIGN = v4.SHAPE_DESIGN
STRATIFICATION_SCHEDULE = v4.STRATIFICATION_SCHEDULE

MotifBaseSpec = v4.MotifBaseSpec
MotifCandidate = v4.MotifCandidate
atomic_write_json = v4.atomic_write_json
canonical_dfg_sha256 = v4.canonical_dfg_sha256
canonical_dfg_text = v4.canonical_dfg_text
default_pinned_architecture = v4.default_pinned_architecture
generate_motif_mlir = v4.generate_motif_mlir
manifest_summary = v4.manifest_summary
parse_architecture_variants = v4.parse_architecture_variants
parse_motif_names = v4.parse_motif_names
parse_shape = v4.parse_shape
parse_shapes = v4.parse_shapes
sha256_file = v4.sha256_file
sha256_text = v4.sha256_text
shape_blocks = v4.shape_blocks
candidate_shapes_for_base = v4.candidate_shapes_for_base
update_manifest_candidate = v4.update_manifest_candidate
write_architecture = v4.write_architecture


POINT_MODEL_FEATURE_NAMES = (
    "semantic_depth",
    "semantic_width",
    "semantic_branch_density",
    "semantic_cut_fraction",
    "multi_input_density",
    "memory_op_density",
    "pointer_path_fraction",
    "memory_path_fraction",
    "compute_fu_peak_pressure",
    "memory_fu_pressure",
    "routing_cut_pressure",
    "register_pressure",
)

HYBRID_PREDICTION_POLICY: Dict[str, object] = {
    "type": "analytical_safe_ml_risk_v1",
    "learned_residual_when": {
        "maximum_tile_count": 9,
        "res_mii_at_least_rec_mii": True,
    },
    "otherwise": "analytical_lower_bound",
}

ACCEPTANCE_POLICY: Dict[str, object] = {
    "policy_version": "motif-v5-hybrid-acceptance-v1",
    "labels_for_protocol_selection": "generated_only",
    "machsuite_mapper_labels_used": False,
    "point_training_population": "all_successful_mapper_results",
    "ranking_population": "complete_declared_shape_blocks_only",
    "coverage": {
        "requested_bases_per_family": 250,
        "minimum_successful_bases_per_family": 150,
        "minimum_complete_ranking_bases_per_family": 75,
    },
    "positive_residual_distribution": {
        "minimum_positive_base_dfgs_per_family": 20,
        "minimum_positive_mechanism_profiles_per_family": 2,
        "minimum_positive_operation_bands_per_family": 2,
        "minimum_positive_target_shapes_per_family": 2,
    },
    "point_model": {
        "class": "gated_residual_ridge_v1",
        "prediction_policy": HYBRID_PREDICTION_POLICY,
        "leave_one_generator_family_out": (
            "strictly_lower_macro_mae_than_analytical_lower_bound"
        ),
        "shape_balanced_mae": "strictly_lower_than_analytical_lower_bound",
        "tie_aware_shape_ranking": "not_lower_than_analytical_lower_bound",
    },
    "timeout_risk": {
        "target": "mapper_timeout_or_nonzero_exit",
        "numeric_ii_imputation": "forbidden",
        "analysis_failure": "excluded_and_reported",
        "lower_bound_above_mapper_ceiling": "excluded_and_reported",
        "estimator": "laplace_smoothed_pre_mapping_stratum_rate_v1",
        "stratum_fields": [
            "generator_family", "mechanism_profile", "operation_band",
            "target_shape",
        ],
    },
}


def make_base_specs(
    samples_per_family: int, seed: int = DEFAULT_SEED,
    motifs: Optional[Sequence[str]] = None,
) -> Tuple[MotifBaseSpec, ...]:
    """Draw v5 identities from the unchanged v4 graph grammar."""
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
            band_name, _ = v4.operation_band_for_index(base_index)
            for _ in range(10000):
                base_seed = family_rng.randrange(1 << 63)
                operation_count = v4.stratified_operation_count(
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
                    f"unable to generate a distinct motif-v5 {motif} DFG"
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
    """Materialize v5 candidates through the frozen v4 emitter."""
    if any(base.generator_version != GENERATOR_VERSION for base in bases):
        raise ValueError("motif-v5 candidate received a non-v5 base spec")
    v4_bases = tuple(replace(
        base, generator_version=v4.GENERATOR_VERSION,
        generator_type=v4.GENERATOR_TYPE,
    ) for base in bases)
    emitted = v4.make_candidates(
        v4_bases, output_dir, shapes, architecture_variants, registers,
        architecture_source,
    )
    by_key = {(base.motif, base.base_index): base for base in bases}
    result: List[MotifCandidate] = []
    for candidate in emitted:
        base = by_key[(candidate.motif, candidate.base_index)]
        candidate_id = (
            f"{base.lineage}/{candidate.rows}x{candidate.columns}/"
            f"{candidate.architecture_variant}/r{candidate.registers}"
        )
        result.append(replace(
            candidate,
            candidate_id=candidate_id,
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
    "GENERATOR_TYPE", "GENERATOR_VERSION", "HYBRID_PREDICTION_POLICY",
    "MANIFEST_SCHEMA_VERSION", "MECHANISM_PROFILES", "MotifBaseSpec",
    "MotifCandidate", "OPERATION_BANDS", "PINNED_ARCHITECTURE_COLUMNS",
    "PINNED_ARCHITECTURE_RELATIVE_PATH", "PINNED_ARCHITECTURE_ROWS",
    "PINNED_ARCHITECTURE_SHA256", "PINNED_CTRL_MEM_ITEMS",
    "PINNED_NEURA_REVISION", "PINNED_REGISTERS_PER_TILE",
    "POINT_MODEL_FEATURE_NAMES", "PREDICTION_SHAPES", "PRIMARY_SHAPE",
    "SHAPE_DESIGN", "STRATIFICATION_SCHEDULE", "atomic_write_json",
    "candidate_shapes_for_base", "canonical_dfg_sha256",
    "canonical_dfg_text", "default_pinned_architecture",
    "generate_motif_mlir", "make_base_specs", "make_candidates",
    "make_manifest", "manifest_summary", "parse_architecture_variants",
    "parse_motif_names", "parse_shape", "parse_shapes", "sha256_file",
    "sha256_text", "shape_blocks", "update_manifest_candidate",
    "write_architecture",
]
