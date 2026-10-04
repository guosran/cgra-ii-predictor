"""Safety tests for the label-free real-source admission protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))

import prepare_real_per_cgra_2x2_sources as preparation


def test_machsuite_required_sources_fit_frozen_query_budget():
    assert preparation.REQUIRED_MACH_CASES == {
        "gemm/blocked", "gemm/ncubed", "spmv/crs", "spmv/ellpack",
        "stencil/stencil2d", "stencil/stencil3d",
    }
    assert len(preparation.SHAPES_2X2) == 8
    assert 32 * len(preparation.SHAPES_2X2) == 256


def test_max_new_query_argument_is_explicit_and_hard_capped():
    assert preparation._parser().parse_args([]).max_new_queries == 256
    with pytest.raises(SystemExit, match="between 1 and 256"):
        preparation.main(["--lower-only", "--max-new-queries", "257"])


def test_related_source_variants_are_test_only_in_one_connected_group():
    candidates = [
        {
            "candidate_id": "gemm-blocked",
            "mapper_input_identity": "a" * 64,
            "model_visible_graph_identity": "1" * 64,
            "leakage_lineage_id": "l1",
            "source_program_families": ["machsuite/gemm"],
        },
        {
            "candidate_id": "gemm-ncubed",
            "mapper_input_identity": "b" * 64,
            "model_visible_graph_identity": "2" * 64,
            "leakage_lineage_id": "l2",
            "source_program_families": ["machsuite/gemm"],
        },
        {
            "candidate_id": "spmv-crs",
            "mapper_input_identity": "c" * 64,
            "model_visible_graph_identity": "3" * 64,
            "leakage_lineage_id": "l3",
            "source_program_families": ["machsuite/spmv"],
        },
    ]

    groups = preparation._group_admitted(candidates)

    assert len(groups) == 2
    gemm_group = next(group for group in groups
                      if group["source_program_families"] == ["machsuite/gemm"])
    assert gemm_group["candidate_ids"] == ["gemm-blocked", "gemm-ncubed"]
    assert all(group["split"] == "test_only" for group in groups)


def test_canonical_exclusion_loader_rejects_pre_route_identity_map(tmp_path):
    old_map = tmp_path / "pre-route.json"
    old_map.write_text(json.dumps({
        "schema": "orbit-final-benchmark-exclusion-audit-v1",
        "evaluation_only": True,
        "identities": {"a" * 64: ["task"]},
    }))

    with pytest.raises(ValueError, match="route-expanded audit"):
        preparation._canonical_identity_sets(old_map)


def test_canonical_exclusion_loader_requires_both_identity_domains(tmp_path):
    audit = tmp_path / "expanded.json"
    audit.write_text(json.dumps({
        "schema": preparation.CANONICAL_ID_MAP_SCHEMA,
        "comparison": {
            "final_task_count": 118,
            "final_route_expanded_unique_mapper_input_identity_count": 109,
            "final_route_expanded_unique_model_visible_graph_identity_count": 84,
            "overlap_unique_mapper_input_identity_count": 0,
            "overlap_unique_visible_graph_identity_count": 0,
            "mapper_input_overlaps": [],
            "visible_graph_overlaps": [],
            "training_unique_mapper_input_identity_count": 323,
            "training_architecture_sha256": preparation.EXPECTED_ARCHITECTURE_SHA256,
        },
        "source": {
            "neura_opt_sha256": preparation.EXPECTED_NEURA_OPT_SHA256,
            "neura_opt_hash_matches_expected": True,
        },
        "tasks": [
            {
                "mapper_input_identity": format(index % 109, "064x"),
                "model_visible_graph_identity": format(index % 84, "064x"),
                "route_expanded_validation": "passed",
                "insert_data_mov_exit_code": 0,
                "program": "program",
                "task": "task-{}".format(index),
            }
            for index in range(118)
        ],
    }))

    mapper_ids, visible_ids, _ = preparation._canonical_identity_sets(audit)

    assert len(mapper_ids) == 109
    assert len(visible_ids) == 84
