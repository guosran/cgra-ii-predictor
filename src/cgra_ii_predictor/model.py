"""Dependency-light residual Ridge model with leakage-safe group validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .dataset import Sample


Model = Dict[str, Any]
Prediction = Dict[str, Any]


def fit_ridge(samples: Sequence[Sample], feature_names: Sequence[str],
              ridge: float, residual_dead_zone: float = 0.0) -> Model:
    if not samples:
        raise ValueError("cannot fit an empty sample set")
    if ridge <= 0.0 or residual_dead_zone < 0.0:
        raise ValueError("ridge must be positive and dead zone non-negative")
    x = np.asarray([
        [sample.features[name] for name in feature_names]
        for sample in samples
    ], dtype=float)
    y = np.asarray([
        sample.compiled_ii - sample.lower_bound for sample in samples
    ], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(samples)), (x - mean) / scale))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    return {
        "model_type": "residual_ridge",
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "ridge": ridge,
        "residual_dead_zone": residual_dead_zone,
    }


def predict_ridge(model: Mapping[str, Any], sample: Sample) -> float:
    names = model["feature_names"]
    vector = np.asarray([sample.features[name] for name in names], dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    residual = max(0.0, weights[0] + ((vector - mean) / scale) @ weights[1:])
    if residual < float(model.get("residual_dead_zone", 0.0)):
        residual = 0.0
    return sample.lower_bound + float(residual)


def prediction_rows(model: Mapping[str, Any], samples: Sequence[Sample]) -> List[Prediction]:
    return [{
        "sample_id": sample.sample_id,
        "group": sample.group,
        "lower_bound": sample.lower_bound,
        "prediction": predict_ridge(model, sample),
        "compiled_ii": sample.compiled_ii,
    } for sample in samples]


def mae(rows: Sequence[Prediction], key: str) -> float:
    return sum(abs(float(row[key]) - float(row["compiled_ii"]))
               for row in rows) / len(rows)


def macro_group_mae(rows: Sequence[Prediction], key: str) -> float:
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return sum(mae(group_rows, key) for group_rows in grouped.values()) / len(grouped)


def quality_metrics(rows: Sequence[Prediction], key: str) -> Dict[str, float]:
    errors = [float(row[key]) - float(row["compiled_ii"]) for row in rows]
    rounded = [round(float(row[key])) - round(float(row["compiled_ii"]))
               for row in rows]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "macro_group_mae": macro_group_mae(rows, key),
        "mean_signed_error": sum(errors) / len(errors),
        "rounded_mae": sum(abs(error) for error in rounded) / len(rounded),
        "rounded_exact_rate": sum(error == 0 for error in rounded) / len(rounded),
        "within_one_rate": sum(abs(error) <= 1.0 for error in errors) / len(errors),
        "max_absolute_error": max(abs(error) for error in errors),
    }


def _select_hyperparameters(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Tuple[float, float]:
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        raise ValueError("hyperparameter selection needs at least two groups")
    best: Tuple[float, float, float, float] = None  # type: ignore[assignment]
    for ridge in ridge_candidates:
        raw: List[Tuple[Sample, Model]] = []
        for group in groups:
            train = [sample for sample in samples if sample.group != group]
            test = [sample for sample in samples if sample.group == group]
            model = fit_ridge(train, feature_names, ridge)
            raw.extend((sample, model) for sample in test)
        for dead_zone in dead_zone_candidates:
            rows: List[Prediction] = []
            for sample, uncalibrated in raw:
                model = dict(uncalibrated)
                model["residual_dead_zone"] = dead_zone
                rows.extend(prediction_rows(model, [sample]))
            score = (
                macro_group_mae(rows, "prediction"),
                mae(rows, "prediction"),
                ridge,
                dead_zone,
            )
            if best is None or score < best:
                best = score
    return best[2], best[3]


def nested_group_holdout(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Dict[str, Any]:
    """Leave one source/template group out; tune only inside outer training."""
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 3:
        raise ValueError("nested group holdout needs at least three groups")
    rows: List[Prediction] = []
    chosen: Dict[str, Dict[str, float]] = {}
    for group in groups:
        train = [sample for sample in samples if sample.group != group]
        test = [sample for sample in samples if sample.group == group]
        ridge, dead_zone = _select_hyperparameters(
            train, feature_names, ridge_candidates, dead_zone_candidates
        )
        model = fit_ridge(train, feature_names, ridge, dead_zone)
        rows.extend(prediction_rows(model, test))
        chosen[group] = {"ridge": ridge, "residual_dead_zone": dead_zone}
    return {
        "groups": groups,
        "rows": rows,
        "chosen_hyperparameters_by_held_out_group": chosen,
        "lower_bound_metrics": quality_metrics(rows, "lower_bound"),
        "model_metrics": quality_metrics(rows, "prediction"),
    }


def fit_calibrated_model(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Model:
    ridge, dead_zone = _select_hyperparameters(
        samples, feature_names, ridge_candidates, dead_zone_candidates
    )
    return fit_ridge(samples, feature_names, ridge, dead_zone)

