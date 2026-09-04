#!/usr/bin/env python3
"""Diagnose where a trained pointwise graph checkpoint makes II errors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from cgra_ii_predictor.graph_model import (  # noqa: E402
    DFG_NODE_FEATURE_NAMES,
    GraphData,
    JointGraphShapeModel,
    Model2Config,
    make_cgra_graph,
)
from neura_graph_experiment import (  # noqa: E402
    QueryRecord,
    batch_targets,
    load_terminal_manifest,
    query_batches,
    split_queries,
)


def _graph_facts(graph: GraphData) -> Dict[str, float]:
    node_count = len(graph.node_types)
    parents = [[] for _ in range(node_count)]
    children = [[] for _ in range(node_count)]
    for source, target in graph.edges:
        if source >= target:
            continue
        parents[target].append(source)
        children[source].append(target)
    depth = [1] * node_count
    for node in range(node_count):
        if parents[node]:
            depth[node] = 1 + max(depth[parent] for parent in parents[node])
    layer_counts: Dict[int, int] = defaultdict(int)
    for value in depth:
        layer_counts[value] += 1
    cutwidth = 0
    for cut in range(max(0, node_count - 1)):
        cutwidth = max(cutwidth, sum(
            source <= cut < target for source, target in graph.edges
        ))
    feature_index = {
        name: index for index, name in enumerate(DFG_NODE_FEATURE_NAMES)
    }
    return {
        "nodes": float(node_count),
        "edges": float(len(graph.edges)),
        "depth": float(max(depth)),
        "width": float(max(layer_counts.values())),
        "cutwidth": float(cutwidth),
        "sources": float(sum(not values for values in parents)),
        "sinks": float(sum(not values for values in children)),
        "max_fanout": float(max(map(len, children))),
        "multi_input_nodes": float(sum(len(values) > 1 for values in parents)),
        "memory_nodes": float(sum(
            row[feature_index["is_memory"]] > 0.5
            for row in graph.node_features
        )),
        "pointer_nodes": float(sum(
            row[feature_index["is_pointer"]] > 0.5
            for row in graph.node_features
        )),
        "control_nodes": float(sum(
            row[feature_index["is_control"]] > 0.5
            for row in graph.node_features
        )),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x) *
        sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(
        left * right for left, right in zip(centered_x, centered_y)
    ) / denominator


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    errors = [float(row["signed_error"]) for row in rows]
    baseline = [float(row["analytical_signed_error"]) for row in rows]
    return {
        "count": len(rows),
        "mae": statistics.fmean(map(abs, errors)),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "bias": statistics.fmean(errors),
        "underprediction_rate": sum(error < 0.0 for error in errors) / len(errors),
        "analytical_mae": statistics.fmean(map(abs, baseline)),
        "exact_after_floor_rate": sum(
            math.floor(float(row["predicted_ii"])) == int(row["compiled_ii"])
            for row in rows
        ) / len(rows),
    }


def _grouped(
    rows: Sequence[Mapping[str, Any]], key: str,
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: _summary(group) for name, group in sorted(groups.items())
    }


def _residual_band(residual: float) -> str:
    if residual == 0.0:
        return "0"
    if residual == 1.0:
        return "1"
    if residual <= 3.0:
        return "2-3"
    return "4+"


def analyze(
    manifest_path: Path, checkpoint_path: Path, split_name: str,
    batch_size: int, seed: int, device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Model2Config(**checkpoint["config"]).validate()
    if config.interaction_mode not in {
        "discrete_pointwise", "residual_pointwise",
        "continuous_residual_pointwise",
    }:
        raise ValueError("checkpoint is not an independent pointwise II model")
    _, queries = load_terminal_manifest(
        manifest_path, config.dfg_representation,
    )
    splits = split_queries(queries, seed)
    selected = splits[split_name]
    model = JointGraphShapeModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    shape_graphs = [
        make_cgra_graph(rows, columns)
        for rows in range(1, 5) for columns in range(1, 5)
    ]
    rows: list[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in query_batches(selected, batch_size):
            context, _, _, _, _ = batch_targets(batch, config, device)
            output = model(
                [query.graph for query in batch], shape_graphs, context,
            )
            predictions = output["predicted_ii"].cpu().tolist()
            deviations = output["predicted_ii_std"].cpu().tolist()
            probabilities = output["success_probability"].cpu().tolist()
            for query_index, query in enumerate(batch):
                facts = _graph_facts(query.graph)
                for candidate_index, candidate in enumerate(query.candidates):
                    if candidate.compiled_ii is None:
                        continue
                    tiles = candidate.rows * candidate.columns
                    links = 2 * (
                        candidate.rows * max(0, candidate.columns - 1) +
                        candidate.columns * max(0, candidate.rows - 1)
                    )
                    prediction = float(predictions[query_index][candidate_index])
                    residual = float(candidate.compiled_ii - candidate.lower_bound)
                    row: Dict[str, Any] = {
                        "candidate_id": candidate.candidate_id,
                        "ranking_query_id": query.ranking_query_id,
                        "family": query.generator_family,
                        "shape": f"{candidate.rows}x{candidate.columns}",
                        "tiles": float(tiles),
                        "links": float(links),
                        "rec_mii": candidate.rec_mii,
                        "res_mii": candidate.res_mii,
                        "lower_bound": candidate.lower_bound,
                        "bound_kind": (
                            "rec" if candidate.rec_mii > candidate.res_mii else
                            "res" if candidate.res_mii > candidate.rec_mii else
                            "tie"
                        ),
                        "compiled_ii": candidate.compiled_ii,
                        "target_residual": residual,
                        "residual_band": _residual_band(residual),
                        "predicted_ii": prediction,
                        "predicted_std": float(
                            deviations[query_index][candidate_index]
                        ),
                        "success_probability": float(
                            probabilities[query_index][candidate_index]
                        ),
                        "signed_error": prediction - candidate.compiled_ii,
                        "analytical_signed_error": (
                            candidate.lower_bound - candidate.compiled_ii
                        ),
                        **facts,
                    }
                    row["nodes_per_tile"] = row["nodes"] / tiles
                    row["edges_per_link"] = (
                        row["edges"] / links if links else 0.0
                    )
                    rows.append(row)

    numeric = (
        "nodes", "edges", "depth", "width", "cutwidth", "sources",
        "sinks", "max_fanout", "multi_input_nodes", "memory_nodes",
        "pointer_nodes", "control_nodes", "tiles", "links",
        "nodes_per_tile", "edges_per_link", "rec_mii", "res_mii",
        "lower_bound", "target_residual", "predicted_std",
        "success_probability",
    )
    absolute_errors = [abs(float(row["signed_error"])) for row in rows]
    correlations = {
        name: _pearson(
            [float(row[name]) for row in rows], absolute_errors,
        )
        for name in numeric
    }
    correlations = dict(sorted(
        correlations.items(),
        key=lambda item: abs(item[1]) if item[1] is not None else -1.0,
        reverse=True,
    ))
    worst = sorted(rows, key=lambda row: abs(row["signed_error"]), reverse=True)[:25]
    return {
        "schema_version": "cgra-ii-checkpoint-error-audit-v1",
        "manifest": str(manifest_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "split": split_name,
        "config": config.to_dict(),
        "overall": _summary(rows),
        "by_family": _grouped(rows, "family"),
        "by_shape": _grouped(rows, "shape"),
        "by_target_residual": _grouped(rows, "residual_band"),
        "by_bound_kind": _grouped(rows, "bound_kind"),
        "absolute_error_pearson_correlations": correlations,
        "worst_candidates": worst,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(
        args.manifest, args.checkpoint, args.split, args.batch_size,
        args.seed, torch.device(args.device),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
