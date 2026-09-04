#!/usr/bin/env python3
"""Audit information collisions in the Model 2 graph representation.

This is a label-reading diagnostic, not a training or model-selection tool.  It
groups candidates by the exact tensors available to the pointwise predictor
before learned transformations.  Conflicting outcomes inside one group are
therefore impossible for the current model to distinguish.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgra_ii_predictor.graph_model import (  # noqa: E402
    GraphData,
    OPERATION_TO_ID,
)
from neura_graph_experiment import (  # noqa: E402
    QueryRecord,
    load_terminal_manifest,
)


def _graph_signature(graph: GraphData) -> str:
    payload = json.dumps(
        {
            "node_types": graph.node_types,
            "node_features": graph.node_features,
            "edges": graph.edges,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_input_key(
    query: QueryRecord, candidate_index: int,
) -> Tuple[Any, ...]:
    candidate = query.candidates[candidate_index]
    return (
        _graph_signature(query.graph),
        candidate.rows,
        candidate.columns,
        candidate.rec_mii,
        candidate.res_mii,
        candidate.lower_bound,
    )


def _source_inventory(text: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for line in text.splitlines():
        if "neura." not in line:
            continue
        kind_match = re.search(r'"?neura\.([a-z_]+)', line)
        if kind_match is None:
            continue
        kind = kind_match.group(1)
        result["neura_operations"] += 1
        if re.match(r"\s*%[A-Za-z0-9_]+\s*=", line):
            result["result_operations"] += 1
        else:
            result["non_result_operations"] += 1
        if kind in {"data_mov", "ctrl_mov", "reserve", "yield"}:
            result["transparent_operations"] += 1
        if kind == "ctrl_mov":
            result["ctrl_mov_operations"] += 1
        if kind not in OPERATION_TO_ID and kind not in {
            "data_mov", "ctrl_mov", "reserve", "yield",
        }:
            result["unknown_materialized_operations"] += 1
    return result


def _safe_source_path(manifest_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("candidate lacks source_path")
    root = manifest_root.resolve()
    path = (root / raw_path).resolve()
    path.relative_to(root)
    return path


def audit_manifest(path: Path, example_limit: int = 8) -> Dict[str, Any]:
    path = path.resolve()
    raw = json.loads(path.read_text())
    _, queries = load_terminal_manifest(path)
    raw_candidates = raw["candidates"]
    raw_by_query: Dict[str, Mapping[str, Any]] = {}
    for row in raw_candidates:
        raw_by_query.setdefault(str(row["ranking_query_id"]), row)

    graph_groups: Dict[str, list[QueryRecord]] = defaultdict(list)
    input_groups: Dict[Tuple[Any, ...], list[Tuple[QueryRecord, int]]] = (
        defaultdict(list)
    )
    inventory: Counter[str] = Counter()
    for query in queries:
        signature = _graph_signature(query.graph)
        graph_groups[signature].append(query)
        for index in range(len(query.candidates)):
            input_groups[_candidate_input_key(query, index)].append((query, index))
        source_path = _safe_source_path(
            path.parent, raw_by_query[query.ranking_query_id]["source_path"],
        )
        source_inventory = _source_inventory(source_path.read_text())
        inventory.update(source_inventory)
        inventory["parsed_materialized_nodes"] += len(query.graph.node_types)
        inventory["parsed_semantic_edges"] += len(query.graph.edges)
        if any(node_type == OPERATION_TO_ID["<unknown>"] for node_type in query.graph.node_types):
            inventory["queries_with_unknown_materialized_operations"] += 1
        if source_inventory["ctrl_mov_operations"]:
            inventory["queries_with_ctrl_mov"] += 1

    aliased_graph_groups = [
        group for group in graph_groups.values()
        if len({query.ranking_query_id for query in group}) > 1
    ]
    duplicate_input_groups = [
        group for group in input_groups.values()
        if len({query.ranking_query_id for query, _ in group}) > 1
    ]
    mixed_outcome_groups = []
    conflicting_label_groups = []
    empirical_absolute_error = 0.0
    successful_candidates = 0
    for key, group in input_groups.items():
        statuses = {query.candidates[index].status for query, index in group}
        labels = [
            float(query.candidates[index].compiled_ii)
            for query, index in group
            if query.candidates[index].compiled_ii is not None
        ]
        successful_candidates += len(labels)
        if len(statuses) > 1:
            mixed_outcome_groups.append((key, group))
        if len(set(labels)) > 1:
            conflicting_label_groups.append((key, group))
        if labels:
            median = float(statistics.median(labels))
            empirical_absolute_error += sum(abs(label - median) for label in labels)

    def examples(
        groups: Iterable[Tuple[Tuple[Any, ...], Sequence[Tuple[QueryRecord, int]]]],
    ) -> list[Dict[str, Any]]:
        result = []
        for key, group in list(groups)[:example_limit]:
            result.append({
                "input": {
                    "graph_sha256": key[0],
                    "rows": key[1],
                    "columns": key[2],
                    "rec_mii": key[3],
                    "res_mii": key[4],
                    "lower_bound": key[5],
                },
                "records": [
                    {
                        "ranking_query_id": query.ranking_query_id,
                        "generator_family": query.generator_family,
                        "candidate_id": query.candidates[index].candidate_id,
                        "status": query.candidates[index].status,
                        "compiled_ii": query.candidates[index].compiled_ii,
                    }
                    for query, index in group[:8]
                ],
            })
        return result

    graph_alias_examples = []
    for group in aliased_graph_groups[:example_limit]:
        graph_alias_examples.append({
            "graph_sha256": _graph_signature(group[0].graph),
            "query_count": len({query.ranking_query_id for query in group}),
            "queries": [
                {
                    "ranking_query_id": query.ranking_query_id,
                    "generator_family": query.generator_family,
                }
                for query in group[:8]
            ],
        })

    return {
        "schema_version": "cgra-ii-graph-representation-audit-v1",
        "manifest": str(path),
        "counts": {
            "queries": len(queries),
            "candidates": sum(len(query.candidates) for query in queries),
            "successful_candidates": successful_candidates,
            "unique_graph_representations": len(graph_groups),
            "graph_alias_groups_across_queries": len(aliased_graph_groups),
            "queries_in_graph_alias_groups": sum(
                len({query.ranking_query_id for query in group})
                for group in aliased_graph_groups
            ),
            "unique_pointwise_input_keys": len(input_groups),
            "duplicate_input_groups_across_queries": len(duplicate_input_groups),
            "mixed_success_censored_input_groups": len(mixed_outcome_groups),
            "conflicting_success_label_input_groups": len(conflicting_label_groups),
        },
        "empirical_exact_input_collision_mae_floor": (
            empirical_absolute_error / successful_candidates
            if successful_candidates else None
        ),
        "source_representation_inventory": dict(sorted(inventory.items())),
        "examples": {
            "graph_aliases": graph_alias_examples,
            "mixed_success_censored": examples(mixed_outcome_groups),
            "conflicting_success_labels": examples(conflicting_label_groups),
        },
        "interpretation": {
            "pointwise_input_key": (
                "exact parsed DFG tensors plus rows, columns, RecMII, ResMII, "
                "and their maximum lower bound"
            ),
            "collision_floor_scope": (
                "empirical contradiction floor only; it is not a generalization "
                "error bound and singleton inputs contribute zero"
            ),
            "parser_omissions": [
                "transparent data_mov/reserve/yield nodes are collapsed",
                "non-result ctrl_mov recurrence/control edges are absent",
                "dependency edges have no type, operand index, width, latency, or loop distance",
                "mapper time slots and link/register occupancy are unavailable at inference",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.example_limit < 0:
        raise ValueError("--example-limit must be nonnegative")
    report = audit_manifest(args.manifest, args.example_limit)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
