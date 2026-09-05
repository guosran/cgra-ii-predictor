#!/usr/bin/env python3
"""Fit a validation-only convex ensemble of independent II checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cgra_ii_predictor.graph_model import (  # noqa: E402
    JointGraphShapeModel,
    Model2Config,
    make_cgra_graph,
)
from cgra_ii_predictor.shape_protocol import get_shape_protocol  # noqa: E402
from neura_graph_experiment import (  # noqa: E402
    batch_targets,
    load_terminal_manifest,
    query_batches,
    split_queries,
)


Prediction = Dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checkpoint(value: str) -> Tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def predict_checkpoint(
    manifest_path: Path, checkpoint_path: Path, split_names: Sequence[str],
    batch_size: int, seed: int, device: torch.device,
) -> Tuple[Model2Config, Dict[str, Dict[str, Prediction]]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False,
    )
    config = Model2Config(**checkpoint["config"]).validate()
    if config.interaction_mode not in {
        "discrete_pointwise", "residual_pointwise",
        "continuous_residual_pointwise",
    }:
        raise ValueError(f"{checkpoint_path} is not a pointwise checkpoint")
    _, queries = load_terminal_manifest(
        manifest_path, config.dfg_representation, config.shape_protocol,
    )
    splits = split_queries(queries, seed)
    model = JointGraphShapeModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    shape_graphs = [
        make_cgra_graph(rows, columns, config.shape_protocol)
        for rows, columns in get_shape_protocol(config.shape_protocol).mapper_shapes
    ]
    result: Dict[str, Dict[str, Prediction]] = {}
    with torch.no_grad():
        for split_name in split_names:
            records: Dict[str, Prediction] = {}
            for batch in query_batches(splits[split_name], batch_size):
                context, _, _, _, _ = batch_targets(batch, config, device)
                output = model(
                    [query.graph for query in batch], shape_graphs, context,
                )
                predictions = output["predicted_ii"].cpu().tolist()
                deviations = output["predicted_ii_std"].cpu().tolist()
                success = output["success_probability"].cpu().tolist()
                for query_index, query in enumerate(batch):
                    for candidate_index, candidate in enumerate(query.candidates):
                        records[candidate.candidate_id] = {
                            "candidate_id": candidate.candidate_id,
                            "ranking_query_id": query.ranking_query_id,
                            "family": query.generator_family,
                            "rows": candidate.rows,
                            "columns": candidate.columns,
                            "status": candidate.status,
                            "compiled_ii": candidate.compiled_ii,
                            "lower_bound": candidate.lower_bound,
                            "predicted_ii": float(
                                predictions[query_index][candidate_index]
                            ),
                            "predicted_std": float(
                                deviations[query_index][candidate_index]
                            ),
                            "success_probability": float(
                                success[query_index][candidate_index]
                            ),
                        }
            result[split_name] = records
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return config, result


def _successful_matrix(
    predictions: Mapping[str, Mapping[str, Prediction]],
) -> Tuple[list[str], np.ndarray, np.ndarray, list[Prediction]]:
    names = list(predictions)
    identities = set(predictions[names[0]])
    if any(set(predictions[name]) != identities for name in names[1:]):
        raise ValueError("checkpoint candidate populations differ")
    ordered = sorted(identities)
    reference = predictions[names[0]]
    rows = [reference[identity] for identity in ordered]
    successful_indices = [
        index for index, row in enumerate(rows)
        if row["compiled_ii"] is not None
    ]
    successful_rows = [rows[index] for index in successful_indices]
    targets = np.asarray([
        float(row["compiled_ii"]) for row in successful_rows
    ], dtype=np.float64)
    columns = [np.asarray([
        float(row["lower_bound"]) for row in successful_rows
    ], dtype=np.float64)]
    output_names = ["analytical_lower_bound"]
    for name in names:
        records = predictions[name]
        columns.append(np.asarray([
            float(records[ordered[index]]["predicted_ii"])
            for index in successful_indices
        ], dtype=np.float64))
        output_names.append(name)
    return output_names, np.stack(columns, axis=1), targets, successful_rows


def _successful_uncertainties(
    predictions: Mapping[str, Mapping[str, Prediction]],
) -> Tuple[list[str], np.ndarray]:
    """Return model uncertainty columns aligned with ``_successful_matrix``.

    The analytical lower bound has no learned uncertainty, so its modulation
    stays fixed at one.  Learned standard deviations are normalized only from
    the validation population before they are used on any held-out split.
    """
    names = list(predictions)
    identities = set(predictions[names[0]])
    if any(set(predictions[name]) != identities for name in names[1:]):
        raise ValueError("checkpoint candidate populations differ")
    ordered = sorted(identities)
    reference = predictions[names[0]]
    successful = [
        identity for identity in ordered
        if reference[identity]["compiled_ii"] is not None
    ]
    columns = [np.ones(len(successful), dtype=np.float64)]
    for name in names:
        columns.append(np.asarray([
            float(predictions[name][identity]["predicted_std"])
            for identity in successful
        ], dtype=np.float64))
    return ["analytical_lower_bound", *names], np.stack(columns, axis=1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = ordered_weights.sum() / 2.0
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def fit_convex_mae(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Coordinate-minimize validation MAE over the probability simplex."""
    if predictions.ndim != 2:
        raise ValueError("predictions must be a two-dimensional matrix")
    if targets.shape != (predictions.shape[0],):
        raise ValueError("targets must have one value per prediction row")
    predictor_count = predictions.shape[1]
    if predictor_count == 0:
        raise ValueError("at least one predictor is required")
    starting_points = [
        np.full(predictor_count, 1.0 / predictor_count, dtype=np.float64)
    ] + [np.eye(predictor_count, dtype=np.float64)[index]
         for index in range(predictor_count)]

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.abs(predictions @ weights - targets)))

    best_weights = starting_points[0]
    best_loss = objective(best_weights)
    for start in starting_points:
        weights = start.copy()
        for _ in range(100):
            before = objective(weights)
            for left in range(predictor_count):
                for right in range(left + 1, predictor_count):
                    total = weights[left] + weights[right]
                    if total <= 0.0:
                        continue
                    fixed = predictions @ weights
                    fixed -= (
                        predictions[:, left] * weights[left] +
                        predictions[:, right] * weights[right]
                    )
                    direction = predictions[:, left] - predictions[:, right]
                    residual = targets - fixed - total * predictions[:, right]
                    active = np.abs(direction) > 1e-12
                    if not np.any(active):
                        continue
                    optimum = _weighted_median(
                        residual[active] / direction[active],
                        np.abs(direction[active]),
                    )
                    optimum = min(total, max(0.0, optimum))
                    trial = weights.copy()
                    trial[left] = optimum
                    trial[right] = total - optimum
                    if objective(trial) <= objective(weights) + 1e-14:
                        weights = trial
            if before - objective(weights) < 1e-12:
                break
        loss = objective(weights)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights
    best_weights[best_weights < 1e-12] = 0.0
    return best_weights / best_weights.sum()


def apply_uncertainty_gating(
    predictions: np.ndarray, uncertainties: np.ndarray,
    base_weights: np.ndarray, exponent: float,
    uncertainty_scales: np.ndarray,
) -> np.ndarray:
    """Blend predictors while trusting relatively confident models more."""
    if predictions.shape != uncertainties.shape:
        raise ValueError("predictions and uncertainties must have equal shape")
    predictor_count = predictions.shape[1]
    if base_weights.shape != (predictor_count,):
        raise ValueError("base_weights has the wrong shape")
    if uncertainty_scales.shape != (predictor_count,):
        raise ValueError("uncertainty_scales has the wrong shape")
    if exponent < 0.0 or not math.isfinite(exponent):
        raise ValueError("uncertainty exponent must be finite and nonnegative")
    if np.any(~np.isfinite(uncertainties)) or np.any(uncertainties < 0.0):
        raise ValueError("uncertainties must be finite and nonnegative")
    if np.any(~np.isfinite(uncertainty_scales)) or np.any(
        uncertainty_scales <= 0.0
    ):
        raise ValueError("uncertainty scales must be finite and positive")
    relative = np.maximum(
        uncertainties / uncertainty_scales[None, :], 1e-4,
    )
    modulation = np.power(relative, -exponent)
    # Column zero is the deterministic analytical lower bound.  It has no
    # uncertainty estimate and therefore keeps its validation-fitted weight.
    modulation[:, 0] = 1.0
    sample_weights = modulation * base_weights[None, :]
    denominators = sample_weights.sum(axis=1, keepdims=True)
    if np.any(denominators <= 0.0):
        raise ValueError("uncertainty gating produced zero total weight")
    return np.sum(predictions * sample_weights / denominators, axis=1)


def fit_uncertainty_gating(
    predictions: np.ndarray, targets: np.ndarray,
    uncertainties: np.ndarray, base_weights: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Select one confidence exponent using validation MAE only."""
    if predictions.shape != uncertainties.shape:
        raise ValueError("predictions and uncertainties must have equal shape")
    if targets.shape != (predictions.shape[0],):
        raise ValueError("targets must have one value per prediction row")
    scales = np.ones(predictions.shape[1], dtype=np.float64)
    for column in range(1, predictions.shape[1]):
        positive = uncertainties[:, column][uncertainties[:, column] > 0.0]
        if len(positive):
            scales[column] = float(np.median(positive))
    best_exponent = 0.0
    best_loss = math.inf
    # A single scalar gate is intentionally low capacity.  A 0.05 grid is
    # precise enough relative to checkpoint noise and avoids a SciPy runtime
    # dependency on inference hosts.
    for exponent in np.linspace(0.0, 6.0, 121):
        blended = apply_uncertainty_gating(
            predictions, uncertainties, base_weights,
            float(exponent), scales,
        )
        loss = float(np.mean(np.abs(blended - targets)))
        if loss < best_loss:
            best_loss = loss
            best_exponent = float(exponent)
    return best_exponent, scales


def _metrics(
    predictions: np.ndarray, targets: np.ndarray, rows: Sequence[Prediction],
) -> Dict[str, Any]:
    errors = predictions - targets
    by_query: Dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors):
        by_query[str(row["ranking_query_id"])].append(abs(float(error)))
    return {
        "candidate_count": len(rows),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "macro_query_mae": statistics.fmean(
            statistics.fmean(values) for values in by_query.values()
        ),
        "mean_signed_error": float(np.mean(errors)),
        "underprediction_rate": float(np.mean(errors < 0.0)),
        "floor_exact_ii": float(np.mean(np.floor(predictions) == targets)),
        "floor_within_one_ii": float(np.mean(
            np.abs(np.floor(predictions) - targets) <= 1.0
        )),
        "floor_decision_mae": float(np.mean(
            np.abs(np.floor(predictions) - targets)
        )),
        "round_exact_ii": float(np.mean(
            np.floor(predictions + 0.5) == targets
        )),
        "round_within_one_ii": float(np.mean(
            np.abs(np.floor(predictions + 0.5) - targets) <= 1.0
        )),
        "round_decision_mae": float(np.mean(
            np.abs(np.floor(predictions + 0.5) - targets)
        )),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _success_metrics(
    probabilities: np.ndarray, targets: np.ndarray,
) -> Dict[str, Any]:
    decisions = probabilities >= 0.5
    positives = targets > 0.5
    true_positive = int(np.sum(decisions & positives))
    predicted_positive = int(np.sum(decisions))
    actual_positive = int(np.sum(positives))
    calibration_bins = []
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        selected = ((probabilities >= lower) & (
            probabilities <= upper if bin_index == 9 else probabilities < upper
        ))
        count = int(np.sum(selected))
        if not count:
            continue
        confidence = float(np.mean(probabilities[selected]))
        observed = float(np.mean(targets[selected]))
        ece += count / len(targets) * abs(confidence - observed)
        calibration_bins.append({
            "lower": lower, "upper": upper, "count": count,
            "mean_probability": confidence,
            "observed_success_rate": observed,
        })
    return {
        "candidate_count": int(len(targets)),
        "accuracy_at_0_5": float(np.mean(decisions == positives)),
        "precision_at_0_5": (
            true_positive / predicted_positive if predicted_positive else None
        ),
        "recall_at_0_5": (
            true_positive / actual_positive if actual_positive else None
        ),
        "brier_score": float(np.mean(np.square(probabilities - targets))),
        "expected_calibration_error_10_bins": ece,
        "calibration_bins": calibration_bins,
    }


def _ensemble_success_probabilities(
    predictions: Mapping[str, Mapping[str, Prediction]],
    weights: np.ndarray, uncertainty_exponent: float,
    uncertainty_scales: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, list[Prediction]]:
    names = list(predictions)
    identities = sorted(predictions[names[0]])
    if any(set(predictions[name]) != set(identities) for name in names[1:]):
        raise ValueError("checkpoint candidate populations differ")
    rows = [predictions[names[0]][identity] for identity in identities]
    targets = np.asarray([
        float(row["status"] == "success") for row in rows
    ], dtype=np.float64)
    probabilities = np.stack([
        np.asarray([
            float(predictions[name][identity]["success_probability"])
            for identity in identities
        ], dtype=np.float64)
        for name in names
    ], axis=1)
    deviations = np.stack([
        np.asarray([
            float(predictions[name][identity]["predicted_std"])
            for identity in identities
        ], dtype=np.float64)
        for name in names
    ], axis=1)
    model_weights = weights[1:].copy()
    relative = np.maximum(
        deviations / uncertainty_scales[None, 1:], 1e-4,
    )
    sample_weights = model_weights[None, :] * np.power(
        relative, -uncertainty_exponent,
    )
    totals = sample_weights.sum(axis=1, keepdims=True)
    zero_rows = totals[:, 0] <= 0.0
    if np.any(zero_rows):
        sample_weights[zero_rows, :] = 1.0
        totals = sample_weights.sum(axis=1, keepdims=True)
    blended = np.sum(probabilities * sample_weights / totals, axis=1)
    return blended, targets, rows


def evaluate_ensemble(
    predictions: Mapping[str, Mapping[str, Prediction]],
    weights: np.ndarray, uncertainty_exponent: float = 0.0,
    uncertainty_scales: np.ndarray | None = None,
) -> Dict[str, Any]:
    names, matrix, targets, rows = _successful_matrix(predictions)
    if len(weights) != len(names):
        raise ValueError("ensemble weight count differs from predictors")
    uncertainty_names, uncertainties = _successful_uncertainties(predictions)
    if uncertainty_names != names:
        raise ValueError("uncertainty predictors differ from prediction matrix")
    if uncertainty_scales is None:
        uncertainty_scales = np.ones(len(names), dtype=np.float64)
    static_ensemble = matrix @ weights
    ensemble = apply_uncertainty_gating(
        matrix, uncertainties, weights, uncertainty_exponent,
        uncertainty_scales,
    )
    individual = {
        name: _metrics(matrix[:, index], targets, rows)
        for index, name in enumerate(names)
    }
    gated_success, success_targets, success_rows = (
        _ensemble_success_probabilities(
            predictions, weights, uncertainty_exponent, uncertainty_scales,
        )
    )
    static_success, _, _ = _ensemble_success_probabilities(
        predictions, weights, 0.0, uncertainty_scales,
    )
    by_family: Dict[str, list[int]] = defaultdict(list)
    by_shape: Dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_family[str(row["family"])].append(index)
        by_shape[f"{row['rows']}x{row['columns']}"].append(index)
    all_by_family: Dict[str, list[int]] = defaultdict(list)
    all_by_shape: Dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(success_rows):
        all_by_family[str(row["family"])].append(index)
        all_by_shape[f"{row['rows']}x{row['columns']}"].append(index)
    signed_errors = matrix - targets[:, None]
    correlations = {
        left: {
            right: _correlation(
                signed_errors[:, left_index], signed_errors[:, right_index],
            )
            for right_index, right in enumerate(names)
        }
        for left_index, left in enumerate(names)
    }
    return {
        "ensemble": _metrics(ensemble, targets, rows),
        "static_ensemble": _metrics(static_ensemble, targets, rows),
        "success_classifier": _success_metrics(
            gated_success, success_targets,
        ),
        "static_success_classifier": _success_metrics(
            static_success, success_targets,
        ),
        "individual": individual,
        "by_family": {
            family: _metrics(
                ensemble[indices], targets[indices],
                [rows[index] for index in indices],
            )
            for family, indices in sorted(by_family.items())
        },
        "by_mapper_tile_shape": {
            shape: _metrics(
                ensemble[indices], targets[indices],
                [rows[index] for index in indices],
            )
            for shape, indices in sorted(by_shape.items())
        },
        "static_by_family": {
            family: _metrics(
                static_ensemble[indices], targets[indices],
                [rows[index] for index in indices],
            )
            for family, indices in sorted(by_family.items())
        },
        "static_by_mapper_tile_shape": {
            shape: _metrics(
                static_ensemble[indices], targets[indices],
                [rows[index] for index in indices],
            )
            for shape, indices in sorted(by_shape.items())
        },
        "success_classifier_by_family": {
            family: _success_metrics(
                gated_success[indices], success_targets[indices],
            )
            for family, indices in sorted(all_by_family.items())
        },
        "success_classifier_by_mapper_tile_shape": {
            shape: _success_metrics(
                gated_success[indices], success_targets[indices],
            )
            for shape, indices in sorted(all_by_shape.items())
        },
        "static_success_classifier_by_family": {
            family: _success_metrics(
                static_success[indices], success_targets[indices],
            )
            for family, indices in sorted(all_by_family.items())
        },
        "static_success_classifier_by_mapper_tile_shape": {
            shape: _success_metrics(
                static_success[indices], success_targets[indices],
            )
            for shape, indices in sorted(all_by_shape.items())
        },
        "signed_error_correlations": correlations,
        "oracle_best_component_mae": float(np.mean(
            np.min(np.abs(signed_errors), axis=1)
        )),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    all_predictions: Dict[str, Dict[str, Dict[str, Prediction]]] = {}
    configs = {}
    for name, path in args.checkpoint:
        if name in all_predictions:
            raise ValueError("duplicate checkpoint name: " + name)
        config, split_predictions = predict_checkpoint(
            args.manifest, path, ("validation", "test"),
            args.batch_size, args.seed, device,
        )
        configs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "config": config.to_dict(),
        }
        all_predictions[name] = split_predictions
    protocol_ids = {
        config.shape_protocol for config in (
            Model2Config(**record["config"]).validate()
            for record in configs.values()
        )
    }
    if len(protocol_ids) != 1:
        raise ValueError("ensemble checkpoints use different shape protocols")
    validation = {
        name: split_predictions["validation"]
        for name, split_predictions in all_predictions.items()
    }
    names, matrix, targets, _ = _successful_matrix(validation)
    weights = fit_convex_mae(matrix, targets)
    uncertainty_names, uncertainties = _successful_uncertainties(validation)
    if uncertainty_names != names:
        raise ValueError("uncertainty predictors differ from prediction matrix")
    uncertainty_exponent, uncertainty_scales = fit_uncertainty_gating(
        matrix, targets, uncertainties, weights,
    )
    validation_evaluation = evaluate_ensemble(
        validation, weights, uncertainty_exponent, uncertainty_scales,
    )
    selected_ensemble_mode = (
        "uncertainty_gated"
        if validation_evaluation["ensemble"]["mae"] <
        validation_evaluation["static_ensemble"]["mae"] else "static"
    )
    best_single_model = min(
        all_predictions,
        key=lambda name: (
            float(validation_evaluation["individual"][name]["mae"]), name,
        ),
    )
    test_evaluation = evaluate_ensemble({
        name: split_predictions["test"]
        for name, split_predictions in all_predictions.items()
    }, weights, uncertainty_exponent, uncertainty_scales)
    selected_metric_key = (
        "ensemble" if selected_ensemble_mode == "uncertainty_gated" else
        "static_ensemble"
    )
    selected_success_key = (
        "success_classifier"
        if selected_ensemble_mode == "uncertainty_gated" else
        "static_success_classifier"
    )
    selected_family_key = (
        "by_family" if selected_ensemble_mode == "uncertainty_gated" else
        "static_by_family"
    )
    selected_shape_key = (
        "by_mapper_tile_shape"
        if selected_ensemble_mode == "uncertainty_gated" else
        "static_by_mapper_tile_shape"
    )
    selected_success_family_key = (
        "success_classifier_by_family"
        if selected_ensemble_mode == "uncertainty_gated" else
        "static_success_classifier_by_family"
    )
    selected_success_shape_key = (
        "success_classifier_by_mapper_tile_shape"
        if selected_ensemble_mode == "uncertainty_gated" else
        "static_success_classifier_by_mapper_tile_shape"
    )
    report = {
        "schema_version": "cgra-ii-pointwise-convex-ensemble-v1",
        "selection_split": "validation_only",
        "primary_metric": "successful_candidate_continuous_ii_mae",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "seed": args.seed,
        "checkpoints": configs,
        "shape_protocol": get_shape_protocol(next(iter(protocol_ids))).to_dict(),
        "weights": {
            name: float(weight) for name, weight in zip(names, weights)
        },
        "uncertainty_gating": {
            "exponent": uncertainty_exponent,
            "normalization": "validation_positive_median",
            "scales": {
                name: float(scale)
                for name, scale in zip(names, uncertainty_scales)
            },
        },
        "selected_ensemble_mode": selected_ensemble_mode,
        "selected_ensemble_metric": "validation_continuous_ii_mae",
        "selected_ensemble": {
            "mode": selected_ensemble_mode,
            "validation": validation_evaluation[selected_metric_key],
            "test": test_evaluation[selected_metric_key],
            "validation_success_classifier": (
                validation_evaluation[selected_success_key]
            ),
            "test_success_classifier": test_evaluation[selected_success_key],
            "test_by_family": test_evaluation[selected_family_key],
            "test_by_mapper_tile_shape": test_evaluation[selected_shape_key],
            "test_success_classifier_by_family": (
                test_evaluation[selected_success_family_key]
            ),
            "test_success_classifier_by_mapper_tile_shape": (
                test_evaluation[selected_success_shape_key]
            ),
        },
        "best_single_model": {
            "name": best_single_model,
            "selection_split": "validation_only",
            "selection_metric": "successful_candidate_continuous_ii_mae",
            "validation": validation_evaluation["individual"][best_single_model],
            "test": test_evaluation["individual"][best_single_model],
        },
        "validation": validation_evaluation,
        "test": test_evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "weights": report["weights"],
        "uncertainty_gating": report["uncertainty_gating"],
        "selected_ensemble_mode": report["selected_ensemble_mode"],
        "validation": report["selected_ensemble"]["validation"],
        "test": report["selected_ensemble"]["test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
