#!/usr/bin/env python3
"""Train and evaluate Model 2 on a terminal Neura motif manifest.

The script treats mapper censorship as a categorical outcome.  Numeric II
loss is applied only to successful candidates, while listwise shape selection
considers every declared candidate through an expected timeout-aware cost.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from cgra_ii_predictor.graph_model import (
    CANDIDATE_CONTEXT_NAMES,
    CROSS_ATTENTION_CONTEXT_NAMES,
    ROUTING_CONTEXT_NAMES,
    GraphData,
    JointGraphShapeModel,
    Model2Config,
    candidate_context,
    censored_top1_metrics,
    make_cgra_graph,
    model2_loss,
    parse_neura_dfg,
)


SCHEMA_VERSION = "cgra-ii-model2-experiment-v1"
DEFAULT_SPLIT_SEED = 20260905
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    ranking_query_id: str
    rows: int
    columns: int
    rec_mii: float
    res_mii: float
    lower_bound: float
    status: str
    compiled_ii: Optional[float]

    def identity(self) -> Tuple[int, int, int, str]:
        return (
            self.rows * self.columns, self.rows, self.columns,
            self.candidate_id,
        )


@dataclass(frozen=True)
class QueryRecord:
    ranking_query_id: str
    generator_family: str
    graph: GraphData
    candidates: Tuple[CandidateRecord, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest source_path is missing")
    root = root.resolve()
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("manifest source_path escapes the corpus") from error
    return path


def load_terminal_manifest(path: Path) -> Tuple[Dict[str, Any], List[QueryRecord]]:
    """Load all declared candidates without converting censorship into II."""
    path = path.resolve()
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    records = manifest.get("candidates")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest contains no candidates")
    if any(record.get("status") == "declared" for record in records):
        raise ValueError("manifest collection is not terminal")
    root = path.parent
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("manifest candidate must be an object")
        query = record.get("ranking_query_id")
        if not isinstance(query, str) or not query:
            raise ValueError("candidate lacks ranking_query_id")
        grouped[query].append(record)
    queries: List[QueryRecord] = []
    source_cache: Dict[str, GraphData] = {}
    for query_id in sorted(grouped):
        raw_candidates = grouped[query_id]
        families = {str(row.get("generator_family", "")) for row in raw_candidates}
        if len(families) != 1 or "" in families:
            raise ValueError("ranking query spans generator families")
        sources = {str(row.get("source_sha256", "")) for row in raw_candidates}
        if len(sources) != 1 or "" in sources:
            raise ValueError("ranking query spans source identities")
        source_sha = next(iter(sources))
        source_path = _safe_relative(root, raw_candidates[0].get("source_path"))
        if sha256_file(source_path) != source_sha:
            raise ValueError(f"source hash mismatch for query {query_id}")
        graph = source_cache.get(source_sha)
        if graph is None:
            graph = parse_neura_dfg(source_path.read_text())
            source_cache[source_sha] = graph
        candidates: List[CandidateRecord] = []
        seen = set()
        for row in raw_candidates:
            candidate_id = str(row.get("candidate_id", row.get("id", "")))
            if not candidate_id or candidate_id in seen:
                raise ValueError("duplicate or missing candidate identity")
            seen.add(candidate_id)
            status = str(row.get("status"))
            if status not in {"success", "censored"}:
                raise ValueError("terminal candidate has an invalid status")
            try:
                rows = int(row["rows"])
                columns = int(row["columns"])
                rec_mii = float(row["rec_mii"])
                res_mii = float(row["res_mii"])
                lower_bound = float(row["lower_bound"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("candidate lacks valid analytical facts") from error
            if not all(math.isfinite(value) for value in (
                rec_mii, res_mii, lower_bound,
            )) or max(rec_mii, res_mii) != lower_bound:
                raise ValueError("candidate analytical facts are inconsistent")
            compiled = row.get("compiled_ii") if status == "success" else None
            if status == "success" and (
                isinstance(compiled, bool) or
                not isinstance(compiled, (int, float)) or
                not math.isfinite(float(compiled)) or
                float(compiled) < lower_bound
            ):
                raise ValueError("successful candidate lacks a valid compiled II")
            if status == "censored" and row.get("compiled_ii") is not None:
                raise ValueError("censored candidate must not have a numeric II")
            candidates.append(CandidateRecord(
                candidate_id=candidate_id,
                ranking_query_id=query_id,
                rows=rows,
                columns=columns,
                rec_mii=rec_mii,
                res_mii=res_mii,
                lower_bound=lower_bound,
                status=status,
                compiled_ii=float(compiled) if compiled is not None else None,
            ))
        candidates.sort(key=lambda candidate: (candidate.rows, candidate.columns))
        shapes = [(candidate.rows, candidate.columns) for candidate in candidates]
        expected = [(rows, columns) for rows in range(1, 5) for columns in range(1, 5)]
        if shapes != expected:
            raise ValueError("Model 2 requires all 16 declared rectangular shapes")
        queries.append(QueryRecord(
            ranking_query_id=query_id,
            generator_family=next(iter(families)),
            graph=graph,
            candidates=tuple(candidates),
        ))
    return manifest, queries


def split_queries(
    queries: Sequence[QueryRecord], seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> Dict[str, List[QueryRecord]]:
    """Create an exact family-stratified deterministic query split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction leaves no test population")
    by_family: Dict[str, List[QueryRecord]] = defaultdict(list)
    for query in queries:
        by_family[query.generator_family].append(query)
    result: Dict[str, List[QueryRecord]] = {
        "train": [], "validation": [], "test": [],
    }
    for family in sorted(by_family):
        ordered = sorted(by_family[family], key=lambda query: (
            hashlib.sha256(
                f"{seed}:{family}:{query.ranking_query_id}".encode("utf-8")
            ).hexdigest(),
            query.ranking_query_id,
        ))
        count = len(ordered)
        train_end = int(count * train_fraction)
        validation_end = train_end + int(count * validation_fraction)
        if train_end == 0 or validation_end == train_end or validation_end == count:
            raise ValueError("each family needs train, validation, and test queries")
        result["train"].extend(ordered[:train_end])
        result["validation"].extend(ordered[train_end:validation_end])
        result["test"].extend(ordered[validation_end:])
    return result


def _oracle_index(query: QueryRecord) -> int:
    successful = [
        (index, candidate) for index, candidate in enumerate(query.candidates)
        if candidate.status == "success"
    ]
    if not successful:
        return -1
    return min(successful, key=lambda item: (
        float(item[1].compiled_ii), *item[1].identity(),
    ))[0]


def batch_targets(
    queries: Sequence[QueryRecord], config: Model2Config, device: torch.device,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    contexts = []
    successes = []
    residuals = []
    optimal_ii_targets = []
    oracles = []
    for query in queries:
        contexts.append([
            candidate_context(
                candidate.rows, candidate.columns, candidate.rec_mii,
                candidate.res_mii, candidate.lower_bound,
                config.mapper_ii_ceiling,
            )
            for candidate in query.candidates
        ])
        successes.append([
            float(candidate.status == "success") for candidate in query.candidates
        ])
        residuals.append([
            max(0.0, float(candidate.compiled_ii) - candidate.lower_bound)
            if candidate.compiled_ii is not None else 0.0
            for candidate in query.candidates
        ])
        successful_iis = [
            float(candidate.compiled_ii) for candidate in query.candidates
            if candidate.compiled_ii is not None
        ]
        minimum_ii = min(successful_iis) if successful_iis else None
        optimal_ii_targets.append([
            float(
                minimum_ii is not None and candidate.compiled_ii is not None and
                float(candidate.compiled_ii) == minimum_ii
            )
            for candidate in query.candidates
        ])
        oracles.append(_oracle_index(query))
    return (
        torch.tensor(contexts, dtype=torch.float32, device=device),
        torch.tensor(successes, dtype=torch.float32, device=device),
        torch.tensor(residuals, dtype=torch.float32, device=device),
        torch.tensor(optimal_ii_targets, dtype=torch.bool, device=device),
        torch.tensor(oracles, dtype=torch.long, device=device),
    )


def query_batches(
    queries: Sequence[QueryRecord], batch_size: int, *, seed: Optional[int] = None,
) -> Iterable[List[QueryRecord]]:
    indices = list(range(len(queries)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [queries[index] for index in indices[start:start + batch_size]]


def _without_query_details(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "queries"}


def evaluate_model(
    model: JointGraphShapeModel, queries: Sequence[QueryRecord],
    shape_graphs: Sequence[GraphData], config: Model2Config,
    batch_size: int, device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    graph_rows: List[List[Dict[str, Any]]] = []
    baseline_rows: List[List[Dict[str, Any]]] = []
    point_errors: List[float] = []
    baseline_errors: List[float] = []
    probabilities: List[float] = []
    targets: List[float] = []
    loss_sums: Dict[str, float] = defaultdict(float)
    loss_query_count = 0
    with torch.no_grad():
        for batch in query_batches(queries, batch_size):
            context, success, residual, optimal, oracle = batch_targets(
                batch, config, device,
            )
            output = model([query.graph for query in batch], shape_graphs, context)
            losses = model2_loss(
                output, success, residual, optimal, oracle, config,
            )
            for name, value in losses.items():
                loss_sums[name] += float(value) * len(batch)
            loss_query_count += len(batch)
            selection_cost = output["selection_cost"].cpu().tolist()
            predicted_ii = output["predicted_ii"].cpu().tolist()
            probability = output["success_probability"].cpu().tolist()
            for query_index, query in enumerate(batch):
                query_graph_rows = []
                query_baseline_rows = []
                for candidate_index, candidate in enumerate(query.candidates):
                    common = {
                        "candidate_id": candidate.candidate_id,
                        "ranking_query_id": candidate.ranking_query_id,
                        "rows": candidate.rows,
                        "columns": candidate.columns,
                        "status": candidate.status,
                        "compiled_ii": candidate.compiled_ii,
                    }
                    query_graph_rows.append({
                        **common,
                        "selection_cost": (
                            selection_cost[query_index][candidate_index]
                        ),
                    })
                    query_baseline_rows.append({
                        **common,
                        "selection_cost": candidate.lower_bound,
                    })
                    probabilities.append(probability[query_index][candidate_index])
                    target = float(candidate.status == "success")
                    targets.append(target)
                    if candidate.compiled_ii is not None:
                        point_errors.append(abs(
                            predicted_ii[query_index][candidate_index] -
                            candidate.compiled_ii
                        ))
                        baseline_errors.append(abs(
                            candidate.lower_bound - candidate.compiled_ii
                        ))
                graph_rows.append(query_graph_rows)
                baseline_rows.append(query_baseline_rows)
    graph_top1 = censored_top1_metrics(
        graph_rows, "selection_cost",
        mapper_ii_ceiling=config.mapper_ii_ceiling,
    )
    baseline_top1 = censored_top1_metrics(
        baseline_rows, "selection_cost",
        mapper_ii_ceiling=config.mapper_ii_ceiling,
    )
    binary = [float(probability >= 0.5) for probability in probabilities]
    true_positive = sum(
        prediction == target == 1.0 for prediction, target in zip(binary, targets)
    )
    predicted_positive = sum(binary)
    actual_positive = sum(targets)
    return {
        "query_count": len(queries),
        "candidate_count": sum(len(query.candidates) for query in queries),
        "loss": {
            name: value / max(1, loss_query_count)
            for name, value in sorted(loss_sums.items())
        },
        "analytical_top1": _without_query_details(baseline_top1),
        "model2_top1": _without_query_details(graph_top1),
        "successful_candidate_point_mae": (
            sum(point_errors) / len(point_errors) if point_errors else None
        ),
        "analytical_successful_candidate_mae": (
            sum(baseline_errors) / len(baseline_errors)
            if baseline_errors else None
        ),
        "success_classifier": {
            "accuracy_at_0_5": sum(
                prediction == target
                for prediction, target in zip(binary, targets)
            ) / len(targets),
            "precision_at_0_5": (
                true_positive / predicted_positive if predicted_positive else None
            ),
            "recall_at_0_5": (
                true_positive / actual_positive if actual_positive else None
            ),
            "brier_score": sum(
                (probability - target) ** 2
                for probability, target in zip(probabilities, targets)
            ) / len(targets),
        },
    }


def evaluate_by_family(
    model: JointGraphShapeModel, queries: Sequence[QueryRecord],
    shape_graphs: Sequence[GraphData], config: Model2Config,
    batch_size: int, device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    by_family: Dict[str, List[QueryRecord]] = defaultdict(list)
    for query in queries:
        by_family[query.generator_family].append(query)
    return {
        family: evaluate_model(
            model, family_queries, shape_graphs, config, batch_size, device,
        )
        for family, family_queries in sorted(by_family.items())
    }


def _validation_score(evaluation: Mapping[str, Any]) -> Tuple[float, ...]:
    top1 = evaluation["model2_top1"]
    return (
        -float(top1["strict_top1_accuracy"]),
        -float(top1["optimal_ii_rate"]),
        -float(top1["selected_success_rate"]),
        float(top1["mean_timeout_penalized_regret"]),
        float(evaluation["loss"]["total"]),
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit training device without silently ignoring CUDA."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but torch.cuda.is_available() is false; "
            "check the NVIDIA driver and /dev/nvidia* device nodes"
        )
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(normalized)


def train_model(
    splits: Mapping[str, Sequence[QueryRecord]], config: Model2Config,
    *, epochs: int, batch_size: int, learning_rate: float,
    weight_decay: float, patience: int, seed: int, threads: int,
    device: Optional[torch.device] = None,
) -> Tuple[JointGraphShapeModel, Dict[str, Any]]:
    config.validate()
    if epochs < 1 or batch_size < 1 or patience < 1 or threads < 1:
        raise ValueError("epochs, batch size, patience, and threads must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    device = device or torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = JointGraphShapeModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    shapes = [(rows, columns) for rows in range(1, 5) for columns in range(1, 5)]
    shape_graphs = [make_cgra_graph(rows, columns) for rows, columns in shapes]
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score: Optional[Tuple[float, ...]] = None
    best_epoch = 0
    history = []
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss: Dict[str, float] = defaultdict(float)
        training_queries = 0
        for batch in query_batches(
            splits["train"], batch_size, seed=seed + epoch,
        ):
            context, success, residual, optimal, oracle = batch_targets(
                batch, config, device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model([query.graph for query in batch], shape_graphs, context)
            losses = model2_loss(
                output, success, residual, optimal, oracle, config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in losses.items():
                training_loss[name] += float(value.detach()) * len(batch)
            training_queries += len(batch)
        validation = evaluate_model(
            model, splits["validation"], shape_graphs, config,
            batch_size, device,
        )
        score = _validation_score(validation)
        improved = best_score is None or score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "training_loss": {
                name: value / max(1, training_queries)
                for name, value in sorted(training_loss.items())
            },
            "validation": validation,
            "selected_as_best": improved,
        }
        history.append(record)
        top1 = validation["model2_top1"]
        print(
            f"epoch={epoch} loss={record['training_loss']['total']:.4f} "
            f"validation_top1={top1['strict_top1_accuracy']:.4f} "
            f"validation_success={top1['selected_success_rate']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "executed_epochs": len(history),
        "selection_order": [
            "maximum_validation_strict_top1_accuracy",
            "maximum_validation_optimal_ii_rate",
            "maximum_validation_selected_success_rate",
            "minimum_validation_timeout_penalized_regret",
            "minimum_validation_total_loss",
        ],
        "history": history,
    }


def train_fixed_epochs(
    queries: Sequence[QueryRecord], config: Model2Config, *, epochs: int,
    batch_size: int, learning_rate: float, weight_decay: float,
    seed: int, threads: int, device: Optional[torch.device] = None,
) -> Tuple[JointGraphShapeModel, Dict[str, Any]]:
    """Refit a frozen configuration on every development query.

    This deliberately has no validation path or early stopping: both the
    hyperparameters and epoch count must already have been selected before the
    held-out manifest is labelled.
    """
    config.validate()
    if not queries:
        raise ValueError("fixed-epoch training requires at least one query")
    if epochs < 1 or batch_size < 1 or threads < 1:
        raise ValueError("epochs, batch size, and threads must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    device = device or torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = JointGraphShapeModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    shape_graphs = [
        make_cgra_graph(rows, columns)
        for rows in range(1, 5) for columns in range(1, 5)
    ]
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sums: Dict[str, float] = defaultdict(float)
        trained_queries = 0
        for batch in query_batches(queries, batch_size, seed=seed + epoch):
            context, success, residual, optimal, oracle = batch_targets(
                batch, config, device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model([query.graph for query in batch], shape_graphs, context)
            losses = model2_loss(
                output, success, residual, optimal, oracle, config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in losses.items():
                loss_sums[name] += float(value.detach()) * len(batch)
            trained_queries += len(batch)
        record = {
            "epoch": epoch,
            "training_loss": {
                name: value / trained_queries
                for name, value in sorted(loss_sums.items())
            },
        }
        history.append(record)
        print(
            f"refit_epoch={epoch}/{epochs} "
            f"loss={record['training_loss']['total']:.4f}",
            flush=True,
        )
    return model, {
        "mode": "fixed_epoch_all_development_queries",
        "epochs": epochs,
        "history": history,
    }


def split_summary(splits: Mapping[str, Sequence[QueryRecord]]) -> Dict[str, Any]:
    return {
        name: {
            "query_count": len(queries),
            "candidate_count": sum(len(query.candidates) for query in queries),
            "generator_family_query_counts": dict(sorted(Counter(
                query.generator_family for query in queries
            ).items())),
            "queries_with_no_success": sum(
                not any(candidate.status == "success" for candidate in query.candidates)
                for query in queries
            ),
            "queries_with_at_least_two_successes": sum(
                sum(candidate.status == "success" for candidate in query.candidates) >= 2
                for query in queries
            ),
        }
        for name, queries in splits.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help=("Training device. Explicit --device cuda fails instead of "
              "silently falling back when the NVIDIA driver is unavailable."),
    )
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--message-passing-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--strict-tiebreak-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--interaction-mode", choices=(
            "pooled", "cross_attention", "routing_set_attention",
            "discrete_routing_set",
        ),
        default="pooled",
        help=("pooled reproduces Model 2; cross_attention performs "
              "candidate-conditioned operation-to-PE interaction; "
              "routing_set_attention adds route pressure and joint "
              "candidate ranking; discrete_routing_set predicts integer II "
              "classes and applies deterministic shape tie-breaking."),
    )
    parser.add_argument("--candidate-set-layers", type=int, default=2)
    parser.add_argument("--candidate-set-heads", type=int, default=4)
    parser.add_argument("--discrete-ii-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--discrete-success-threshold", type=float, default=0.5,
    )
    parser.add_argument(
        "--discrete-ii-decision",
        choices=("map", "round", "floor", "ceil"), default="map",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        raise SystemExit(str(error))
    config = Model2Config(
        hidden_dimension=args.hidden_dimension,
        message_passing_layers=args.message_passing_layers,
        dropout=args.dropout,
        strict_tiebreak_loss_weight=args.strict_tiebreak_loss_weight,
        interaction_mode=args.interaction_mode,
        candidate_set_layers=args.candidate_set_layers,
        candidate_set_heads=args.candidate_set_heads,
        discrete_ii_loss_weight=args.discrete_ii_loss_weight,
        discrete_success_threshold=args.discrete_success_threshold,
        discrete_ii_decision=args.discrete_ii_decision,
    ).validate()
    manifest, queries = load_terminal_manifest(args.manifest)
    generator_versions = sorted({
        str(candidate.get("generator_version", "unknown"))
        for candidate in manifest["candidates"]
    })
    splits = split_queries(queries, args.seed)
    model, training = train_model(
        splits, config, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, seed=args.seed, threads=args.threads,
        device=device,
    )
    shape_graphs = [
        make_cgra_graph(rows, columns)
        for rows in range(1, 5) for columns in range(1, 5)
    ]
    evaluations = {
        name: evaluate_model(
            model, split, shape_graphs, config, args.batch_size, device,
        )
        for name, split in splits.items()
    }
    family_evaluations = {
        name: evaluate_by_family(
            model, split, shape_graphs, config, args.batch_size, device,
        )
        for name, split in splits.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.pt"
    torch.save({
        "schema_version": (
            "cgra-ii-joint-graph-model-v4"
            if config.interaction_mode == "discrete_routing_set" else
            "cgra-ii-joint-graph-model-v3"
            if config.interaction_mode == "routing_set_attention" else
            "cgra-ii-joint-graph-model-v2"
            if config.interaction_mode == "cross_attention" else
            "cgra-ii-joint-graph-model-v1"
        ),
        "config": config.to_dict(),
        "candidate_context_names": list(CANDIDATE_CONTEXT_NAMES),
        "cross_attention_context_names": (
            list(CROSS_ATTENTION_CONTEXT_NAMES)
            if config.interaction_mode in {
                "cross_attention", "routing_set_attention",
                "discrete_routing_set",
            } else []
        ),
        "routing_context_names": (
            list(ROUTING_CONTEXT_NAMES)
            if config.interaction_mode in {
                "routing_set_attention", "discrete_routing_set",
            } else []
        ),
        "state_dict": model.state_dict(),
        "training_manifest_sha256": sha256_file(args.manifest.resolve()),
    }, model_path)
    test_baseline = evaluations["test"]["analytical_top1"]
    test_model = evaluations["test"]["model2_top1"]
    gates = {
        "strict_top1_improvement": (
            float(test_model["strict_top1_accuracy"]) >
            float(test_baseline["strict_top1_accuracy"])
        ),
        "selected_success_non_degradation": (
            float(test_model["selected_success_rate"]) >=
            float(test_baseline["selected_success_rate"])
        ),
        "timeout_penalized_regret_improvement": (
            float(test_model["mean_timeout_penalized_regret"]) <
            float(test_baseline["mean_timeout_penalized_regret"])
        ),
        "optimal_ii_rate_non_degradation": (
            float(test_model["optimal_ii_rate"]) >=
            float(test_baseline["optimal_ii_rate"])
        ),
        "every_family_optimal_ii_rate_non_degradation": all(
            float(record["model2_top1"]["optimal_ii_rate"]) >=
            float(record["analytical_top1"]["optimal_ii_rate"])
            for record in family_evaluations["test"].values()
        ),
        "every_family_selected_success_non_degradation": all(
            float(record["model2_top1"]["selected_success_rate"]) >=
            float(record["analytical_top1"]["selected_success_rate"])
            for record in family_evaluations["test"].values()
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "exploratory_v6_labels_already_disclosed"
            if generator_versions == ["motif-v6"] else
            "exploratory_combined_labels_already_disclosed"
        ),
        "model_class": (
            "discrete_ii_routing_candidate_set_ranker_v4"
            if config.interaction_mode == "discrete_routing_set" else
            "routing_aware_candidate_set_ranker_v3"
            if config.interaction_mode == "routing_set_attention" else
            "joint_directed_gnn_candidate_conditioned_dual_head_listwise_v2"
            if config.interaction_mode == "cross_attention" else
            "joint_directed_gnn_dual_head_listwise_v1"
        ),
        "training_device": str(device),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
            "schema_version": manifest.get("schema_version"),
            "candidate_count": len(manifest["candidates"]),
            "generator_versions": generator_versions,
        },
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "config": config.to_dict(),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "split_protocol": {
            "unit": "ranking_query_id",
            "stratification": "generator_family",
            "seed": args.seed,
            "fractions": {
                "train": DEFAULT_TRAIN_FRACTION,
                "validation": DEFAULT_VALIDATION_FRACTION,
                "test": 1.0 - DEFAULT_TRAIN_FRACTION - DEFAULT_VALIDATION_FRACTION,
            },
            "summary": split_summary(splits),
        },
        "loss_contract": {
            "success": "binary_cross_entropy_all_declared_candidates",
            "ii": "smooth_l1_successful_candidates_only",
            "discrete_ii": (
                "cross_entropy_integer_ii_successful_candidates_only"
                if config.interaction_mode == "discrete_routing_set" else None
            ),
            "listwise": (
                "optimal_ii_set_log_mass_plus_weighted_strict_tiebreak_cross_entropy"
            ),
            "selection_cost": (
                "predicted_success_then_integer_ii_then_area_rows_columns"
                if config.interaction_mode == "discrete_routing_set" else
                "negative_candidate_set_ranking_logit"
                if config.interaction_mode == "routing_set_attention" else
                "p_success*predicted_ii+(1-p_success)*(mapper_ii_ceiling+1)"
            ),
            "ranking_prior": (
                "negative_timeout_aware_expected_discrete_ii"
                if config.interaction_mode == "discrete_routing_set" else
                "negative_timeout_aware_expected_cost_plus_learned_set_adjustment"
                if config.interaction_mode == "routing_set_attention" else None
            ),
            "numeric_ii_imputation_for_censored_candidates": False,
        },
        "training": training,
        "evaluation": evaluations,
        "evaluation_by_generator_family": family_evaluations,
        "development_gates": gates,
        "development_gates_passed": all(gates.values()),
        "frozen_blind_claim": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"model2_report={report_path.resolve()} "
        f"test_top1={test_model['strict_top1_accuracy']:.4f} "
        f"baseline_top1={test_baseline['strict_top1_accuracy']:.4f} "
        f"gates_passed={all(gates.values())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
