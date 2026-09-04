#!/usr/bin/env python3
"""Motif-v7: held-out censored-aware Model-2 evaluation declaration."""

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


GENERATOR_VERSION = "motif-v7"
GENERATOR_TYPE = "generated/motif-v7"
MANIFEST_SCHEMA_VERSION = "cgra-ii-motif-corpus-v7"
# 20260905 produced one canonical collision with the complete motif-v6 draw.
# The next deterministic seed was checked against all 1,500 v6 DFGs and is
# disjoint; the development split seed remains independently frozen below.
DEFAULT_SEED = 20260906

DEFAULT_MOTIFS = v6.DEFAULT_MOTIFS
MECHANISM_PROFILES = v6.MECHANISM_PROFILES
OPERATION_BANDS = v6.OPERATION_BANDS
DEFAULT_SHAPES = v6.DEFAULT_SHAPES
PREDICTION_SHAPES = DEFAULT_SHAPES
PRIMARY_SHAPE = v6.PRIMARY_SHAPE
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
    "evaluation_role": "held_out_model2_censored_aware",
}
SHAPE_DESIGN = "full-cartesian-16-rectangles-censored-aware-v1"
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


MODEL2_CONFIG: Dict[str, object] = {
    "class": "joint_directed_gnn_dual_head_listwise_v1",
    "hidden_dimension": 64,
    "message_passing_layers": 3,
    "dropout": 0.10,
    "mapper_ii_ceiling": 20.0,
    "listwise_temperature": 1.0,
    "success_loss_weight": 1.0,
    "residual_loss_weight": 1.0,
    "listwise_loss_weight": 1.0,
    "strict_tiebreak_loss_weight": 1.0,
    "optimizer": "adamw",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size_queries": 32,
    "development_split_seed": 20260905,
    "development_best_epoch": 28,
    "frozen_refit_epochs_on_all_v6_queries": 28,
}

ACCEPTANCE_POLICY: Dict[str, object] = {
    "policy_version": "motif-v7-model2-censored-top1-v1",
    "training_labels": "terminal_motif_v6_development_corpus_only",
    "motif_v7_labels_may_refit_or_select_model": False,
    "machsuite_mapper_labels_used": False,
    "candidate_population": "all_16_declared_rectangles",
    "numeric_ii_population": "successful_mapper_candidates_only",
    "censorship_population": "all_declared_candidates",
    "ranking_population": (
        "declared_16_shape_queries_with_at_least_two_successful_candidates"
    ),
    "coverage": {
        "requested_bases_per_family": 250,
        "minimum_analyzed_bases_per_family": 240,
        "minimum_ranking_eligible_bases_per_family": 200,
        "minimum_successful_candidates_per_ranking_query": 2,
    },
    "model2": MODEL2_CONFIG,
    "selection_policy": {
        "cost": (
            "p_success*predicted_ii+(1-p_success)*(mapper_ii_ceiling+1)"
        ),
        "oracle_order": [
            "successful_mapper_status", "compiled_ii", "tile_count", "rows",
            "columns", "candidate_id",
        ],
        "prediction_order": [
            "expected_timeout_aware_cost", "tile_count", "rows", "columns",
            "candidate_id",
        ],
    },
    "loss": {
        "success": "binary_cross_entropy_all_declared_candidates",
        "ii": "smooth_l1_successful_candidates_only",
        "listwise": (
            "optimal_ii_set_log_mass_plus_strict_tiebreak_cross_entropy"
        ),
        "numeric_ii_imputation_for_censored_candidates": False,
    },
    "primary_acceptance": {
        "strict_top1": "strictly_higher_than_analytical_lower_bound",
        "optimal_ii_rate": "not_lower_than_analytical_lower_bound",
        "selected_success_rate": "not_lower_than_analytical_lower_bound",
        "timeout_penalized_regret": "strictly_lower_than_analytical_lower_bound",
        "per_family_optimal_ii_rate": "no_generator_family_regression",
        "per_family_selected_success_rate": "no_generator_family_regression",
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
                f"motif-v7 shape must be within 1x1 through 4x4: {value}"
            )
        if shape not in result:
            result.append(shape)
    return tuple(result)


def shape_blocks(
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    selected = tuple(dict.fromkeys(shapes))
    if selected != DEFAULT_SHAPES:
        raise ValueError("motif-v7 requires the complete ordered 16-shape block")
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
    """Draw a new deterministic DFG population from the frozen v5 grammar."""
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
    if any(base.generator_version != GENERATOR_VERSION for base in bases):
        raise ValueError("motif-v7 candidate received a non-v7 base spec")
    v6_bases = tuple(replace(
        base, generator_version=v6.GENERATOR_VERSION,
        generator_type=v6.GENERATOR_TYPE,
    ) for base in bases)
    emitted = v6.make_candidates(
        v6_bases, output_dir, selected_shapes, architecture_variants,
        registers, architecture_source,
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
            shape_block="all-16-rectangles-censored-aware",
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
    })
    manifest["acceptance_policy"] = ACCEPTANCE_POLICY
    manifest["architecture"]["target_shape_design"] = SHAPE_DESIGN
    return manifest


__all__ = [
    "ACCEPTANCE_POLICY", "DEFAULT_ARCHITECTURE_VARIANTS", "DEFAULT_MOTIFS",
    "DEFAULT_SEED", "DEFAULT_SHAPES", "GENERATOR_TYPE", "GENERATOR_VERSION",
    "HYBRID_PREDICTION_POLICY", "MANIFEST_SCHEMA_VERSION",
    "MECHANISM_PROFILES", "MODEL2_CONFIG", "MotifBaseSpec", "MotifCandidate",
    "OPERATION_BANDS", "PINNED_ARCHITECTURE_COLUMNS",
    "PINNED_ARCHITECTURE_RELATIVE_PATH", "PINNED_ARCHITECTURE_ROWS",
    "PINNED_ARCHITECTURE_SHA256", "PINNED_CTRL_MEM_ITEMS",
    "PINNED_NEURA_REVISION", "PINNED_REGISTERS_PER_TILE",
    "POINT_MODEL_FEATURE_NAMES", "PREDICTION_SHAPES", "PRIMARY_SHAPE",
    "SHAPE_DESIGN", "STRATIFICATION_SCHEDULE", "atomic_write_json",
    "candidate_shapes_for_base", "canonical_dfg_sha256", "canonical_dfg_text",
    "default_pinned_architecture", "generate_motif_mlir", "make_base_specs",
    "make_candidates", "make_manifest", "manifest_summary",
    "parse_architecture_variants", "parse_motif_names", "parse_shape",
    "parse_shapes", "sha256_file", "sha256_text", "shape_blocks",
    "update_manifest_candidate", "write_architecture",
]
