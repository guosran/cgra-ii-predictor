#!/usr/bin/env python3
"""Train a compact surrogate directly on heuristic-mapper compiled-II labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch  # noqa: E402

from cgra_ii_predictor.dfg import (  # noqa: E402
    parse_neura_route_expanded_dfg,
)
from cgra_ii_predictor.mapper_model import (  # noqa: E402
    DirectMapperIIModel,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_vector,
    mapper_ii_loss,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL  # noqa: E402


CHECKPOINT_SCHEMA = "cgra-ii-direct-mapper-model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _mapper_shape(record: Mapping[str, Any]) -> Tuple[int, int]:
    """Accept current mapper dimensions or convert older physical dimensions."""
    physical_rows = record.get("physical_cgra_rows")
    physical_columns = record.get("physical_cgra_cols")
    if physical_rows is not None or physical_columns is not None:
        if physical_rows is None or physical_columns is None:
            raise ValueError("candidate has an incomplete physical shape")
        rows, columns = int(record["rows"]), int(record["columns"])
        expected = SHAPE_PROTOCOL.mapper_for_physical(
            int(physical_rows), int(physical_columns),
        )
        if (rows, columns) != expected:
            raise ValueError("physical and mapper candidate shapes disagree")
        return SHAPE_PROTOCOL.validate_mapper_shape(rows, columns)
    return SHAPE_PROTOCOL.mapper_for_physical(
        int(record["rows"]), int(record["columns"]),
    )


def load_examples(
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, str]]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("candidates") if isinstance(manifest, Mapping) else None
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no candidates")
    nonterminal = sum(
        isinstance(record, Mapping) and record.get("status") not in {
            "success", "censored",
        }
        for record in records
    )
    if nonterminal:
        raise ValueError("training manifest is not terminal")

    graphs = {}
    examples: List[Dict[str, Any]] = []
    query_population: Dict[str, Dict[str, str]] = {}
    skipped_shapes = 0
    architecture_hashes = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("manifest candidate must be an object")
        try:
            rows, columns = _mapper_shape(record)
        except ValueError:
            skipped_shapes += 1
            continue
        architecture = record.get("architecture_sha256")
        if not isinstance(architecture, str) or len(architecture) != 64:
            raise ValueError("candidate lacks architecture identity")
        architecture_hashes.add(architecture)
        source_hash = record.get("source_sha256")
        source_path_value = record.get("source_path")
        if not isinstance(source_hash, str) or not isinstance(source_path_value, str):
            raise ValueError("candidate lacks source identity")
        if source_hash not in graphs:
            source_path = (manifest_path.parent / source_path_value).resolve()
            try:
                source_path.relative_to(manifest_path.parent)
            except ValueError as error:
                raise ValueError("candidate source escapes its corpus") from error
            if sha256_file(source_path) != source_hash:
                raise ValueError("candidate source hash mismatch")
            graphs[source_hash] = parse_neura_route_expanded_dfg(
                source_path.read_text()
            )
        lineage = record.get("leakage_lineage_id")
        query = record.get("ranking_query_id")
        family = record.get("generator_family")
        if not all(isinstance(value, str) and value for value in (
            lineage, query, family,
        )):
            raise ValueError("candidate lacks split/query identity")
        identity = {
            "query": str(query),
            "lineage": str(lineage),
            "family": str(family),
        }
        previous_identity = query_population.setdefault(str(query), identity)
        if previous_identity != identity:
            raise ValueError("ranking query identity is inconsistent")
        status = record.get("status")
        if status == "censored":
            if record.get("compiled_ii") is not None:
                raise ValueError("censored candidate has a numeric II")
            continue
        compiled_ii = record.get("compiled_ii")
        if (
            isinstance(compiled_ii, bool) or
            not isinstance(compiled_ii, (int, float))
        ):
            raise ValueError("successful candidate lacks compiled_ii")
        rec_mii = float(record["rec_mii"])
        res_mii = float(record["res_mii"])
        lower_bound = float(record["lower_bound"])
        if float(compiled_ii) < lower_bound:
            raise ValueError("compiled_ii is below its analytical lower bound")
        examples.append({
            "features": mapper_feature_vector(
                graphs[source_hash], rows, columns,
                rec_mii, res_mii, lower_bound,
            ),
            "lower_bound": lower_bound,
            "compiled_ii": float(compiled_ii),
            "lineage": lineage,
            "query": query,
            "family": family,
        })
    if len(architecture_hashes) != 1:
        raise ValueError("training examples must use one fixed architecture")
    return examples, {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "architecture_sha256": next(iter(architecture_hashes)),
        "declared_candidate_count": len(records),
        "successful_supported_candidate_count": len(examples),
        "skipped_unsupported_shape_count": skipped_shapes,
        "unique_dfg_count": len(graphs),
        "label_source": "heuristic_mapper_compiled_ii",
    }, list(query_population.values())


def split_examples(
    examples: Sequence[Mapping[str, Any]], seed: int,
    query_population: Sequence[Mapping[str, str]],
) -> Dict[str, List[int]]:
    by_family: Dict[str, set] = defaultdict(set)
    query_lineages: Dict[str, set] = defaultdict(set)
    for identity in query_population:
        query = str(identity["query"])
        by_family[str(identity["family"])].add(query)
        query_lineages[query].add(str(identity["lineage"]))
    if any(len(lineages) != 1 for lineages in query_lineages.values()):
        raise ValueError("a ranking query spans multiple leakage lineages")
    lineage_queries: Dict[str, set] = defaultdict(set)
    for identity in query_population:
        lineage_queries[str(identity["lineage"])].add(str(identity["query"]))
    if any(len(queries) != 1 for queries in lineage_queries.values()):
        raise ValueError("a leakage lineage spans multiple ranking queries")
    query_split = {}
    for family, queries in by_family.items():
        ordered = sorted(
            queries,
            key=lambda value: (_stable_key(seed, f"{family}:{value}"), value),
        )
        train_end = int(len(ordered) * 0.70)
        validation_end = train_end + int(len(ordered) * 0.15)
        for index, query in enumerate(ordered):
            query_split[query] = (
                "train" if index < train_end else
                "validation" if index < validation_end else "test"
            )
    splits: Dict[str, List[int]] = {
        "train": [], "validation": [], "test": [],
    }
    for index, example in enumerate(examples):
        splits[query_split[str(example["query"])]].append(index)
    return splits


def pair_indices(
    indices: Sequence[int], examples: Sequence[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    by_query: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        by_query[str(examples[index]["query"])].append(index)
    better, worse = [], []
    for query_indices in by_query.values():
        for left_position, left in enumerate(query_indices):
            for right in query_indices[left_position + 1:]:
                left_ii = float(examples[left]["compiled_ii"])
                right_ii = float(examples[right]["compiled_ii"])
                if left_ii < right_ii:
                    better.append(left)
                    worse.append(right)
                elif right_ii < left_ii:
                    better.append(right)
                    worse.append(left)
    return torch.tensor(better, dtype=torch.long), torch.tensor(
        worse, dtype=torch.long,
    )


def top1_groups(
    indices: Sequence[int], examples: Sequence[Mapping[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each multi-shape query and all of its tied optimal shapes."""
    by_query: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        by_query[str(examples[index]["query"])].append(index)
    result = []
    for query_indices in by_query.values():
        if len(query_indices) < 2:
            continue
        best = min(float(examples[index]["compiled_ii"]) for index in query_indices)
        result.append((query_indices, [
            float(examples[index]["compiled_ii"]) == best
            for index in query_indices
        ]))
    width = max((len(group) for group, _ in result), default=0)
    group_indices = torch.zeros((len(result), width), dtype=torch.long)
    valid_mask = torch.zeros((len(result), width), dtype=torch.bool)
    best_mask = torch.zeros((len(result), width), dtype=torch.bool)
    for row, (group, best) in enumerate(result):
        group_indices[row, :len(group)] = torch.tensor(group)
        valid_mask[row, :len(group)] = True
        best_mask[row, :len(group)] = torch.tensor(best)
    return group_indices, valid_mask, best_mask


def evaluate(
    predicted: torch.Tensor,
    examples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> Dict[str, float]:
    targets = torch.tensor([
        float(examples[index]["compiled_ii"]) for index in indices
    ])
    values = predicted[torch.tensor(indices, dtype=torch.long)].cpu()
    errors = values - targets
    by_query: Dict[str, List[int]] = defaultdict(list)
    for index in indices:
        by_query[str(examples[index]["query"])].append(index)
    pair_correct = pair_count = 0
    top1_hits, regrets = [], []
    for query_indices in by_query.values():
        if len(query_indices) < 2:
            continue
        selected = min(query_indices, key=lambda index: (
            float(predicted[index]), index,
        ))
        best = min(float(examples[index]["compiled_ii"]) for index in query_indices)
        selected_ii = float(examples[selected]["compiled_ii"])
        top1_hits.append(selected_ii == best)
        regrets.append(selected_ii - best)
        for position, left in enumerate(query_indices):
            for right in query_indices[position + 1:]:
                left_target = float(examples[left]["compiled_ii"])
                right_target = float(examples[right]["compiled_ii"])
                if left_target == right_target:
                    continue
                pair_count += 1
                pair_correct += (
                    (float(predicted[left]) < float(predicted[right])) ==
                    (left_target < right_target)
                )
    rounded = torch.floor(values + 0.5)
    return {
        "candidate_count": float(len(indices)),
        "mae": float(errors.abs().mean()),
        "rmse": float(errors.square().mean().sqrt()),
        "round_exact_ii": float((rounded == targets).float().mean()),
        "round_within_one_ii": float(
            ((rounded - targets).abs() <= 1.0).float().mean()
        ),
        "pairwise_accuracy": pair_correct / pair_count if pair_count else 0.0,
        "pair_count": float(pair_count),
        "top1_true_ii_hit_rate": (
            sum(top1_hits) / len(top1_hits) if top1_hits else 0.0
        ),
        "top1_mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    examples, manifest, query_population = load_examples(args.manifest)
    splits = split_examples(examples, args.seed, query_population)
    features = torch.tensor([example["features"] for example in examples])
    lower_bounds = torch.tensor([example["lower_bound"] for example in examples])
    targets = torch.tensor([example["compiled_ii"] for example in examples])
    train_indices = torch.tensor(splits["train"], dtype=torch.long)
    mean = features[train_indices].mean(dim=0)
    scale = features[train_indices].std(dim=0).clamp_min(1e-5)
    config = MapperModelConfig(hidden_dimensions=tuple(args.hidden_dimensions))
    torch.manual_seed(args.training_seed)
    model = DirectMapperIIModel(config, mean, scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    better, worse = pair_indices(splits["train"], examples)
    ranking_indices, ranking_mask, ranking_best = top1_groups(
        splits["train"], examples,
    )
    best_selection_key = None
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    validation_indices = torch.tensor(splits["validation"], dtype=torch.long)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, lower_bounds)
        losses = mapper_ii_loss(
            prediction[train_indices], targets[train_indices],
            # Pair indices address the full tensor, so compute their term here.
            pairwise_weight=0.0,
        )
        pairwise = torch.nn.functional.relu(
            args.pairwise_margin - (prediction[worse] - prediction[better])
        ).mean() if better.numel() else prediction.sum() * 0.0
        if ranking_indices.numel():
            ranking_values = -prediction[ranking_indices] / args.top1_temperature
            top1 = (
                torch.logsumexp(ranking_values.masked_fill(~ranking_mask, -torch.inf), dim=1)
                - torch.logsumexp(ranking_values.masked_fill(~ranking_best, -torch.inf), dim=1)
            ).mean()
        else:
            top1 = prediction.sum() * 0.0
        total = (
            losses["point"] + args.pairwise_weight * pairwise
            + args.top1_weight * top1
        )
        total.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(features, lower_bounds)
            validation_metrics = evaluate(
                prediction, examples, splits["validation"],
            )
            validation_mae = validation_metrics["mae"]
        history.append({
            "epoch": epoch,
            "training_point_loss": float(losses["point"].detach()),
            "training_pairwise_loss": float(pairwise.detach()),
            "training_top1_loss": float(top1.detach()),
            "validation_mae": validation_mae,
            "validation_pairwise_accuracy": validation_metrics[
                "pairwise_accuracy"
            ],
            "validation_top1_true_ii_hit_rate": validation_metrics[
                "top1_true_ii_hit_rate"
            ],
            "validation_top1_mean_regret": validation_metrics[
                "top1_mean_regret"
            ],
        })
        selection_key = (
            validation_metrics["top1_mean_regret"],
            -validation_metrics["top1_true_ii_hit_rate"],
            -validation_metrics["pairwise_accuracy"],
            validation_mae,
        ) if args.selection_metric == "ranking" else (validation_mae,)
        if best_selection_key is None or selection_key < best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(features, lower_bounds)
    metrics = {
        name: evaluate(prediction, examples, indices)
        for name, indices in splits.items()
    }
    lower_bound_test = lower_bounds[torch.tensor(splits["test"])]
    target_test = targets[torch.tensor(splits["test"])]
    metrics["test"]["analytical_lower_bound_mae"] = float(
        (lower_bound_test - target_test).abs().mean()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "config": config.to_dict(),
        "feature_names": list(MAPPER_FEATURE_NAMES),
        "state_dict": model.state_dict(),
        "architecture_sha256": manifest["architecture_sha256"],
        "training_manifest_sha256": manifest["sha256"],
        "split_seed": args.seed,
        "training_seed": args.training_seed,
    }, model_path)
    report = {
        "schema": "cgra-ii-direct-mapper-training",
        "status": "trained",
        "target": "heuristic_mapper_compiled_ii",
        "manifest": manifest,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "feature_count": len(MAPPER_FEATURE_NAMES),
            "config": config.to_dict(),
        },
        "split": {
            "unit": "ranking_query_id (one-to-one with leakage_lineage_id)",
            "stratification": "generator_family",
            "seed": args.seed,
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "candidate_counts": {
                name: len(indices) for name, indices in splits.items()
            },
        },
        "objective": {
            "point": "smooth_l1_final_compiled_ii",
            "pairwise_weight": args.pairwise_weight,
            "pairwise_margin": args.pairwise_margin,
            "top1_weight": args.top1_weight,
            "top1_temperature": args.top1_temperature,
            "placement_supervision": False,
            "success_prediction": False,
            "checkpoint_selection": (
                "validation_top1_regret_then_hit_rate_then_pairwise_then_mae"
                if args.selection_metric == "ranking" else "validation_mae"
            ),
        },
        "training": {
            "seed": args.training_seed,
            "best_epoch": best_epoch,
            "executed_epochs": len(history),
            "history": history,
        },
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--training-seed", type=int, default=20260911)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--hidden-dimensions", type=int, nargs=2, default=(64, 32),
        metavar=("FIRST", "SECOND"),
    )
    parser.add_argument("--pairwise-weight", type=float, default=0.1)
    parser.add_argument("--pairwise-margin", type=float, default=0.5)
    parser.add_argument("--top1-weight", type=float, default=0.3)
    parser.add_argument("--top1-temperature", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric", choices=("ranking", "mae"), default="ranking",
    )
    return parser.parse_args()


def main() -> int:
    report = train(parse_args())
    print(json.dumps({
        "model": report["model"],
        "test": report["metrics"]["test"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
