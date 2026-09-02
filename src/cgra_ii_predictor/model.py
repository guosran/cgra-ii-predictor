"""Dependency-light residual Ridge model with leakage-safe group validation."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset import (
    FORBIDDEN_MODEL_FEATURE_NAMES,
    LOWER_BOUND_COMPONENT_NAMES,
    Sample,
)


Model = Dict[str, Any]
Prediction = Dict[str, Any]


def _validate_samples(
    samples: Sequence[Sample], feature_names: Sequence[str],
) -> None:
    """Validate direct ``Sample`` callers as strictly as the JSON loader."""
    if len(set(feature_names)) != len(feature_names) or any(
        not isinstance(name, str) or not name for name in feature_names
    ):
        raise ValueError("feature names must be unique non-empty strings")
    forbidden_features = sorted(
        set(feature_names).intersection(FORBIDDEN_MODEL_FEATURE_NAMES)
    )
    if forbidden_features:
        raise ValueError(
            "labels and lower-bound fields cannot be model features: "
            f"{forbidden_features}"
        )
    seen_sample_ids = set()
    scoped_candidates: Dict[Tuple[str, str], Tuple[Any, ...]] = {}
    query_groups: Dict[str, str] = {}
    provenance_fields = (
        "architecture_id", "architecture_variant", "mapper_id",
        "mapper_revision", "mapper_config", "source_sha256",
        "dfg_source_sha256", "canonical_dfg_sha256",
    )
    for sample in samples:
        if not isinstance(sample.sample_id, str) or not sample.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)
        if not isinstance(sample.group, str) or not sample.group:
            raise ValueError(
                f"sample {sample.sample_id} group must be a non-empty string"
            )
        for field, value in (
            ("lower_bound", sample.lower_bound),
            ("compiled_ii", sample.compiled_ii),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"sample {sample.sample_id} {field} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
                raise ValueError(
                    f"sample {sample.sample_id} {field} must be a positive integer"
                )
        if sample.compiled_ii < sample.lower_bound:
            raise ValueError(
                f"sample {sample.sample_id} compiled_ii is below lower_bound"
            )
        leaked_bound_features = sorted(
            set(sample.features).intersection(FORBIDDEN_MODEL_FEATURE_NAMES)
        )
        if leaked_bound_features:
            raise ValueError(
                f"sample {sample.sample_id} stores labels or lower-bound "
                f"fields as model features: {leaked_bound_features}"
            )
        components: Dict[str, float] = {}
        for name in LOWER_BOUND_COMPONENT_NAMES:
            value = sample.metadata.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"sample {sample.sample_id} metadata.{name} must be a "
                    "non-negative integer"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise ValueError(
                    f"sample {sample.sample_id} metadata.{name} must be a "
                    "non-negative integer"
                )
            components[name] = numeric
        expected_bound = max(components.values())
        if float(sample.lower_bound) != expected_bound:
            raise ValueError(
                f"sample {sample.sample_id} lower_bound must equal "
                f"max(rec_mii,res_mii)={expected_bound}"
            )
        lower_bound_source = sample.metadata.get("lower_bound_source")
        if lower_bound_source not in (None, "rec_res_max_v1"):
            raise ValueError(
                f"sample {sample.sample_id} lower_bound_source must be "
                "rec_res_max_v1"
            )
        for name in feature_names:
            if name not in sample.features:
                raise ValueError(f"sample {sample.sample_id} lacks feature {name}")
            value = sample.features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"sample {sample.sample_id} feature {name} must be numeric"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"sample {sample.sample_id} feature {name} must be finite"
                )
        base_dfg_id = sample.metadata.get("base_dfg_id")
        ranking_query_id = sample.metadata.get("ranking_query_id")
        if (
            base_dfg_id not in (None, "")
            and ranking_query_id not in (None, "")
            and base_dfg_id != ranking_query_id
        ):
            raise ValueError(
                f"sample {sample.sample_id} has inconsistent base_dfg_id "
                "and ranking_query_id"
            )
        # ``ranking_query_id`` and ``base_dfg_id`` are aliases for the same
        # workload identity.  Do not let the presence of an empty primary
        # alias hide a non-empty fallback alias, and apply the group-ownership
        # check even when candidate_id is absent (or differs between rows).
        query_aliases = [
            (name, sample.metadata.get(name))
            for name in ("ranking_query_id", "base_dfg_id")
            if sample.metadata.get(name) not in (None, "")
        ]
        for name, value in query_aliases:
            if not isinstance(value, str):
                raise ValueError(
                    f"sample {sample.sample_id} metadata.{name} must be a "
                    "string"
                )
        query = query_aliases[0][1] if query_aliases else None
        if len(query_aliases) == 2 and query_aliases[0][1] != query_aliases[1][1]:
            # Keep the historical error wording for callers that depend on
            # this consistency check, while using the non-empty aliases above
            # so an empty alias cannot bypass it.
            raise ValueError(
                f"sample {sample.sample_id} has inconsistent base_dfg_id "
                "and ranking_query_id"
            )
        if query is not None:
            prior_group = query_groups.setdefault(query, sample.group)
            if prior_group != sample.group:
                raise ValueError(
                    f"ranking query {query} maps to multiple sample.group "
                    f"values: {prior_group!r} and {sample.group!r}"
                )
        candidate = sample.metadata.get("candidate_id")
        if query in (None, "") or candidate in (None, ""):
            continue
        if not isinstance(query, str) or not isinstance(candidate, str):
            raise ValueError(
                "ranking_query_id/base_dfg_id and candidate_id must be strings"
            )
        key = (query, candidate)
        fingerprint = (
            sample.group, float(sample.lower_bound), float(sample.compiled_ii),
            tuple((name, float(sample.features[name])) for name in feature_names),
            tuple(
                (name, json.dumps(sample.metadata.get(name), sort_keys=True))
                for name in provenance_fields
            ),
        )
        prior = scoped_candidates.setdefault(key, fingerprint)
        if prior != fingerprint:
            raise ValueError(
                "conflicting records for ranking query/candidate "
                f"{query}/{candidate}"
            )


def observation_identity(
    sample: Sample, feature_names: Sequence[str],
) -> Tuple[Any, ...]:
    """Identify one measured candidate under the active model projection.

    Unselected auxiliary counters do not create extra statistical weight.
    A workload identity is deliberately separate from the candidate identity:
    the same architecture candidate may recur for many DFGs.  Selected model
    features, bound, and label provide a conservative fallback for legacy rows.
    """
    workload = next((
        (name, str(sample.metadata[name]))
        for name in (
            "ranking_query_id", "base_dfg_id", "canonical_dfg_sha256",
            "dfg_source_sha256", "source_sha256", "source_group",
            "original_lineage", "lineage",
        )
        if sample.metadata.get(name) not in (None, "")
    ), None)
    candidate = tuple(
        (name, str(sample.metadata[name]))
        for name in (
            "candidate_id", "architecture_id", "architecture_variant",
            "mapper_id", "mapper_revision", "mapper_config",
        )
        if sample.metadata.get(name) is not None
    )
    return (
        workload,
        candidate or None,
        tuple((str(name), float(sample.features[name])) for name in feature_names),
        float(sample.lower_bound),
        float(sample.compiled_ii),
    )


def group_balanced_sample_weights(
    samples: Sequence[Sample], feature_names: Sequence[str],
) -> np.ndarray:
    """Give each group and each distinct observation equal total influence.

    Rows with one observation identity under the active feature projection
    share the weight of one distinct observation. The weights sum to the number
    of groups, so adding model-indistinguishable architecture rows cannot
    silently weaken the fixed Ridge penalty.
    """
    training_groups = [
        str(sample.metadata.get("training_weight_group", sample.group))
        for sample in samples
    ]
    declared_strata = [sample.metadata.get("training_stratum") for sample in samples]
    has_strata = any(value not in (None, "") for value in declared_strata)
    if has_strata and any(value in (None, "") for value in declared_strata):
        raise ValueError(
            "training_stratum must be present on every sample when enabled"
        )
    training_strata = [str(value) for value in declared_strata]
    group_to_stratum: Dict[str, str] = {}
    if has_strata:
        for group, stratum in zip(training_groups, training_strata):
            prior = group_to_stratum.setdefault(group, stratum)
            if prior != stratum:
                raise ValueError(
                    f"training group {group} appears in multiple strata"
                )
    signatures = [
        (
            training_group,
            observation_identity(sample, feature_names),
        )
        for sample, training_group in zip(samples, training_groups)
    ]
    signature_counts = Counter(signatures)
    unique_per_group = Counter(signature[0] for signature in set(signatures))
    groups_per_stratum = Counter(group_to_stratum.values())
    stratum_scale = (
        len(group_to_stratum) / len(groups_per_stratum)
        if has_strata else 1.0
    )
    weights = []
    for signature in signatures:
        group = signature[0]
        denominator = unique_per_group[group] * signature_counts[signature]
        if has_strata:
            denominator *= groups_per_stratum[group_to_stratum[group]]
        weights.append(stratum_scale / denominator)
    return np.asarray(weights, dtype=float)


def distinct_observations(
    samples: Sequence[Sample], feature_names: Sequence[str],
) -> List[Sample]:
    """Keep one representative per active-projection observation identity."""
    result: List[Sample] = []
    seen = set()
    for sample in samples:
        signature = (
            sample.group,
            str(sample.metadata.get("training_weight_group", sample.group)),
            observation_identity(sample, feature_names),
        )
        if signature not in seen:
            seen.add(signature)
            result.append(sample)
    return result


def _validated_control(value: Any, name: str, *, strictly_positive: bool) -> float:
    """Return a finite Ridge control value with a stable public error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validated_control_grid(
    values: Sequence[float], name: str, *, strictly_positive: bool,
) -> Tuple[float, ...]:
    """Validate one hyperparameter grid before any cross-validation work."""
    if isinstance(values, (str, bytes)) or values is None:
        raise ValueError(f"{name} candidate grid must be a non-empty sequence")
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise ValueError(
            f"{name} candidate grid must be a non-empty sequence"
        ) from error
    if not candidates:
        raise ValueError(f"{name} candidate grid must be non-empty")
    return tuple(
        _validated_control(value, f"{name} candidate", strictly_positive=strictly_positive)
        for value in candidates
    )


def _require_finite_array(array: np.ndarray, name: str) -> np.ndarray:
    """Raise a useful contract error instead of returning NaN model fields."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"fitted {name} must be finite")
    return array


def fit_ridge(samples: Sequence[Sample], feature_names: Sequence[str],
              ridge: float, residual_dead_zone: float = 0.0, *,
              include_training_diagnostics: bool = True) -> Model:
    if not samples:
        raise ValueError("cannot fit an empty sample set")
    ridge = _validated_control(ridge, "ridge", strictly_positive=True)
    residual_dead_zone = _validated_control(
        residual_dead_zone, "residual dead zone", strictly_positive=False,
    )
    _validate_samples(samples, feature_names)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            x = np.asarray([
                [sample.features[name] for name in feature_names]
                for sample in samples
            ], dtype=float)
            y = np.asarray([
                sample.compiled_ii - sample.lower_bound for sample in samples
            ], dtype=float)
            sample_weights = group_balanced_sample_weights(
                samples, feature_names
            )
            _require_finite_array(x, "feature statistics")
            _require_finite_array(y, "training labels")
            _require_finite_array(sample_weights, "sample weights")
            mean = np.average(x, axis=0, weights=sample_weights)
            _require_finite_array(mean, "feature means")
            centered = x - mean
            _require_finite_array(centered, "centered features")
            scale = np.sqrt(np.average(
                centered ** 2, axis=0, weights=sample_weights
            ))
            _require_finite_array(scale, "feature scales")
            scale[scale == 0.0] = 1.0
            _require_finite_array(scale, "feature scales")
            design = np.column_stack((
                np.ones(len(samples)), centered / scale,
            ))
            _require_finite_array(design, "design matrix")
            penalty = np.eye(design.shape[1]) * ridge
            penalty[0, 0] = 0.0
            weighted_design = sample_weights[:, None] * design
            _require_finite_array(weighted_design, "weighted design matrix")
            normal_matrix = design.T @ weighted_design + penalty
            rhs = design.T @ (sample_weights * y)
            _require_finite_array(normal_matrix, "normal equation")
            _require_finite_array(rhs, "normal-equation right-hand side")
            weights = np.linalg.solve(normal_matrix, rhs)
            _require_finite_array(weights, "weights")
            diagnostics: Model = {}
            if include_training_diagnostics:
                singular_values = np.linalg.svd(design, compute_uv=False)
                _require_finite_array(singular_values, "design singular values")
                tolerance = (
                    singular_values[0] * max(design.shape) *
                    np.finfo(singular_values.dtype).eps
                )
                design_rank = int(np.sum(singular_values > tolerance))
                condition_number = (
                    float(singular_values[0] / singular_values[-1])
                    if singular_values[-1] > 0.0 else None
                )
                diagnostics = {
                    "training_feature_support": {
                        name: {
                            "minimum": float(np.min(x[:, index])),
                            "p01": float(np.quantile(x[:, index], 0.01)),
                            "p99": float(np.quantile(x[:, index], 0.99)),
                            "maximum": float(np.max(x[:, index])),
                        }
                        for index, name in enumerate(feature_names)
                    },
                    "training_design_rank": design_rank,
                    "training_design_column_count": int(design.shape[1]),
                    "training_design_condition_number": condition_number,
                }
    except FloatingPointError as error:
        raise ValueError(
            "Ridge fit overflowed; fitted statistics must be finite"
        ) from error
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Ridge fit failed while solving finite normal equations"
        ) from error
    training_strata = {
        str(sample.metadata["training_stratum"])
        for sample in samples
        if sample.metadata.get("training_stratum") not in (None, "")
    }
    _require_finite_array(sample_weights, "sample weights")
    _require_finite_array(mean, "feature means")
    _require_finite_array(scale, "feature scales")
    _require_finite_array(weights, "weights")
    if not math.isfinite(float(sample_weights.sum())):
        raise ValueError("fitted training weight sum must be finite")
    model: Model = {
        "model_type": "residual_ridge",
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "ridge": ridge,
        "residual_dead_zone": residual_dead_zone,
        "training_weighting": (
            "equal_total_weight_per_stratum_group_and_distinct_observation"
            if training_strata else
            "equal_total_weight_per_group_and_distinct_observation"
        ),
        "training_weight_sum": float(sample_weights.sum()),
        "training_group_count": len({
            str(sample.metadata.get("training_weight_group", sample.group))
            for sample in samples
        }),
        "training_stratum_count": len(training_strata),
        "training_strata": sorted(training_strata),
        "training_weight_group_source": (
            "metadata.training_weight_group_or_sample.group"
        ),
        "training_distinct_observation_count": len({
            (
                str(sample.metadata.get("training_weight_group", sample.group)),
                observation_identity(sample, feature_names),
            )
            for sample in samples
        }),
    }
    model.update(diagnostics)
    return model


def raw_ridge_residual_from_features(
    model: Mapping[str, Any], features: Mapping[str, float],
) -> float:
    """Evaluate the unconstrained residual from an unlabeled feature mapping."""
    names = model["feature_names"]
    vector = np.asarray([features[name] for name in names], dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = float(weights[0] + ((vector - mean) / scale) @ weights[1:])
    if not math.isfinite(residual):
        raise ValueError("raw predicted residual is not finite")
    return residual


def constrained_predicted_residual(
    model: Mapping[str, Any], raw_residual: float,
) -> float:
    """Apply the non-negative floor and the selected small-residual dead zone."""
    residual = max(0.0, float(raw_residual))
    if residual < float(model.get("residual_dead_zone", 0.0)):
        residual = 0.0
    return float(residual)


def predict_compiled_ii(
    model: Mapping[str, Any], lower_bound: float,
    features: Mapping[str, float], *, rec_mii: float, res_mii: float,
) -> float:
    """Predict compiled II after independently checking the Rec/Res floor."""
    if isinstance(lower_bound, bool) or not isinstance(lower_bound, (int, float)):
        raise ValueError("prediction lower bound must be a positive integer")
    lower_bound = float(lower_bound)
    components = {"rec_mii": rec_mii, "res_mii": res_mii}
    for name, raw_value in components.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"prediction {name} must be a non-negative integer")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            raise ValueError(f"prediction {name} must be a non-negative integer")
        components[name] = value
    expected_bound = max(components.values())
    if (
        not math.isfinite(lower_bound) or lower_bound < 1.0 or
        not lower_bound.is_integer()
    ):
        raise ValueError("prediction lower bound must be a positive integer")
    if lower_bound != expected_bound:
        raise ValueError(
            "prediction lower bound must equal "
            f"max(rec_mii,res_mii)={expected_bound}"
        )
    raw_residual = raw_ridge_residual_from_features(model, features)
    residual = constrained_predicted_residual(model, raw_residual)
    prediction = lower_bound + residual
    if not math.isfinite(prediction):
        raise ValueError("predicted compiled II is not finite")
    return prediction


def raw_ridge_residual(model: Mapping[str, Any], sample: Sample) -> float:
    """Compatibility wrapper for labelled training/evaluation samples."""
    return raw_ridge_residual_from_features(model, sample.features)


def predict_ridge(model: Mapping[str, Any], sample: Sample) -> float:
    """Compatibility wrapper for labelled training/evaluation samples."""
    _validate_samples([sample], model["feature_names"])
    return predict_compiled_ii(
        model, sample.lower_bound, sample.features,
        rec_mii=sample.metadata["rec_mii"],
        res_mii=sample.metadata["res_mii"],
    )


def prediction_rows(model: Mapping[str, Any], samples: Sequence[Sample]) -> List[Prediction]:
    rows: List[Prediction] = []
    radius = model.get("unseen_group_absolute_error_radius")
    for sample in samples:
        prediction = predict_ridge(model, sample)
        row: Prediction = {
            "sample_id": sample.sample_id,
            "group": sample.group,
            "lower_bound": sample.lower_bound,
            "prediction": prediction,
            "compiled_ii": sample.compiled_ii,
            "raw_predicted_residual": raw_ridge_residual(model, sample),
        }
        for name in (
            "candidate_id", "architecture_id", "architecture_variant",
            "source_family", "source_kind", "leakage_lineage_id",
            "base_dfg_id", "ranking_query_id", "training_stratum",
            "generator_family", "generator_version", "motif",
        ):
            if name in sample.metadata:
                row[name] = sample.metadata[name]
        if radius is not None:
            row["prediction_interval_lower"] = max(
                sample.lower_bound, prediction - float(radius)
            )
            row["prediction_interval_upper"] = prediction + float(radius)
        rows.append(row)
    return rows


def calibrate_unseen_group_interval(
    model: Mapping[str, Any], holdout_rows: Sequence[Prediction],
    quantile: float = 0.9, *, coverage: Optional[float] = None,
) -> Model:
    """Attach an interval radius calibrated on held-out source groups.

    Each group contributes its maximum absolute error, so shape and mask
    multiplicity cannot make the interval artificially narrow.  This is an
    empirical cross-group diagnostic, not a mapping feasibility guarantee.
    """
    if coverage is not None:
        if quantile != 0.9:
            raise ValueError("provide quantile or legacy coverage, not both")
        quantile = coverage
    if not 0.0 < quantile <= 1.0:
        raise ValueError("empirical quantile must be in (0, 1]")
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for row in holdout_rows:
        grouped[str(row["group"])].append(row)
    if not grouped:
        raise ValueError("interval calibration needs held-out rows")
    maxima = sorted(max(
        abs(float(row["prediction"]) - float(row["compiled_ii"]))
        for row in rows
    ) for rows in grouped.values())
    quantile_index = min(len(maxima) - 1, math.ceil(quantile * len(maxima)) - 1)
    calibrated = dict(model)
    calibrated.update({
        "unseen_group_interval_empirical_quantile": quantile,
        "unseen_group_absolute_error_radius": maxima[quantile_index],
        "unseen_group_calibration_groups": len(maxima),
        "unseen_group_interval_method": "held_out_group_max_error_quantile",
    })
    return calibrated


def mae(rows: Sequence[Prediction], key: str) -> float:
    return sum(abs(float(row[key]) - float(row["compiled_ii"]))
               for row in rows) / len(rows)


def macro_group_mae(rows: Sequence[Prediction], key: str) -> float:
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    return sum(mae(group_rows, key) for group_rows in grouped.values()) / len(grouped)


def stratified_macro_group_mae(
    rows: Sequence[Prediction], key: str,
) -> float:
    """Give each declared training stratum equal validation influence."""
    present = [row.get("training_stratum") for row in rows]
    if not any(value not in (None, "") for value in present):
        return macro_group_mae(rows, key)
    if any(value in (None, "") for value in present):
        raise ValueError(
            "training_stratum must be present on every validation row"
        )
    strata: Dict[str, List[Prediction]] = defaultdict(list)
    for row in rows:
        strata[str(row["training_stratum"])].append(row)
    return sum(
        macro_group_mae(stratum_rows, key)
        for stratum_rows in strata.values()
    ) / len(strata)


def quality_metrics(rows: Sequence[Prediction], key: str) -> Dict[str, float]:
    errors = [float(row[key]) - float(row["compiled_ii"]) for row in rows]
    rounded = [round(float(row[key])) - round(float(row["compiled_ii"]))
               for row in rows]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "macro_group_mae": macro_group_mae(rows, key),
        "stratified_macro_group_mae": stratified_macro_group_mae(rows, key),
        "mean_signed_error": sum(errors) / len(errors),
        "rounded_mae": sum(abs(error) for error in rounded) / len(rounded),
        "rounded_exact_rate": sum(error == 0 for error in rounded) / len(rounded),
        "within_one_rate": sum(abs(error) <= 1.0 for error in errors) / len(errors),
        "max_absolute_error": max(abs(error) for error in errors),
    }


def group_ranking_metrics(
    rows: Sequence[Prediction], key: str,
) -> Dict[str, Any]:
    """Measure DSE ordering only within an explicit base-DFG query.

    The leakage group used for train/test isolation is intentionally *not* a
    ranking query: one lineage can contain op-count, unroll, or compiler
    descendants that must stay in the same split but are different workloads.
    Only rows carrying ``ranking_query_id`` and ``candidate_id`` participate.
    Prediction ties receive half credit; target ties create no comparable pair.
    """
    identified = [row for row in rows if row.get("ranking_query_id") not in (None, "")]
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for row in identified:
        grouped[str(row["ranking_query_id"])].append(row)
    comparable = correct = tied = 0
    group_scores: List[float] = []
    group_results: Dict[str, Dict[str, Any]] = {}
    excluded = {
        "ranking_query_spans_multiple_leakage_groups": 0,
        "missing_candidate_identity": 0,
        "single_sample": 0,
        "constant_target": 0,
        "no_comparable_pair": 0,
        "conflicting_candidate_identity": 0,
    }
    duplicate_candidate_rows = 0
    for group, group_rows in grouped.items():
        group_result: Dict[str, Any] = {"sample_count": len(group_rows)}
        leakage_groups = sorted({str(row["group"]) for row in group_rows})
        group_result["leakage_groups"] = leakage_groups
        if len(leakage_groups) != 1:
            group_result["status"] = (
                "ranking_query_spans_multiple_leakage_groups"
            )
            excluded["ranking_query_spans_multiple_leakage_groups"] += 1
            group_results[group] = group_result
            continue
        if not group_rows or not all(
            row.get("candidate_id") not in (None, "") for row in group_rows
        ):
            group_result["status"] = "missing_candidate_identity"
            excluded["missing_candidate_identity"] += 1
            group_results[group] = group_result
            continue
        candidates: Dict[str, Prediction] = {}
        conflict = False
        for row in group_rows:
            candidate_id = str(row["candidate_id"])
            prior = candidates.get(candidate_id)
            if prior is not None:
                duplicate_candidate_rows += 1
                if (
                    float(prior["compiled_ii"]) !=
                    float(row["compiled_ii"]) or
                    float(prior[key]) != float(row[key])
                ):
                    conflict = True
            else:
                candidates[candidate_id] = row
        group_result["candidate_count"] = len(candidates)
        group_result["duplicate_candidate_rows_collapsed"] = (
            len(group_rows) - len(candidates)
        )
        if conflict:
            group_result["status"] = "conflicting_candidate_identity"
            excluded["conflicting_candidate_identity"] += 1
            group_results[group] = group_result
            continue
        group_rows = list(candidates.values())
        if len(group_rows) < 2:
            group_result["status"] = "single_sample"
            excluded["single_sample"] += 1
            group_results[group] = group_result
            continue
        if len({float(row["compiled_ii"]) for row in group_rows}) < 2:
            group_result["status"] = "constant_target"
            excluded["constant_target"] += 1
            group_results[group] = group_result
            continue
        group_comparable = group_correct = group_tied = 0
        for left_index, left in enumerate(group_rows):
            for right in group_rows[left_index + 1:]:
                target_delta = float(left["compiled_ii"]) - float(right["compiled_ii"])
                if target_delta == 0.0:
                    continue
                group_comparable += 1
                prediction_delta = float(left[key]) - float(right[key])
                product = prediction_delta * target_delta
                if product > 0.0:
                    group_correct += 1
                elif product == 0.0:
                    group_tied += 1
        if not group_comparable:
            group_result["status"] = "no_comparable_pair"
            excluded["no_comparable_pair"] += 1
            group_results[group] = group_result
            continue
        score = (group_correct + 0.5 * group_tied) / group_comparable
        group_scores.append(score)
        comparable += group_comparable
        correct += group_correct
        tied += group_tied
        group_result.update({
            "status": "eligible",
            "comparable_pairs": group_comparable,
            "strictly_concordant_pairs": group_correct,
            "predicted_ties": group_tied,
            "concordant_score": group_correct + 0.5 * group_tied,
            "pairwise_concordance": score,
            "predicted_tie_rate": group_tied / group_comparable,
        })
        group_results[group] = group_result
    candidate_identity_count = sum(
        row.get("candidate_id") not in (None, "") for row in identified
    )
    micro = (correct + 0.5 * tied) / comparable if comparable else None
    macro = sum(group_scores) / len(group_scores) if group_scores else None
    cross_group_queries = excluded[
        "ranking_query_spans_multiple_leakage_groups"
    ]
    if cross_group_queries and group_scores:
        status = "partial_invalid_cross_leakage_query"
    elif cross_group_queries:
        status = "invalid_cross_leakage_query"
    elif not identified:
        status = "unavailable_missing_ranking_query_id"
    elif len(identified) != len(rows):
        status = "ok_partial_identity" if group_scores else "partial_identity_no_eligible_queries"
    else:
        status = "ok" if group_scores else "no_eligible_queries"
    return {
        "status": status,
        "direction": "minimize",
        "group_key": "ranking_query_id",
        "ranking_query_key": "metadata.ranking_query_id",
        "prediction_key": key,
        "input_row_count": len(rows),
        "identified_row_count": len(identified),
        "missing_ranking_query_row_count": len(rows) - len(identified),
        "total_group_count": len(grouped),
        "total_ranking_query_count": len(grouped),
        "eligible_group_count": len(group_scores),
        "eligible_ranking_query_count": len(group_scores),
        "excluded_group_counts": excluded,
        "comparable_pairs": comparable,
        "groups_with_ranking_signal": len(group_scores),
        "strict_pairwise_accuracy": (
            correct / comparable if comparable else None
        ),
        "tie_aware_pairwise_accuracy": micro,
        "micro_pairwise_concordance": micro,
        "macro_pairwise_concordance": macro,
        "predicted_tie_rate": tied / comparable if comparable else None,
        "candidate_identity_status": (
            "complete"
            if identified and len(identified) == len(rows)
            and candidate_identity_count == len(identified)
            else "partial"
            if identified and candidate_identity_count
            else "unavailable"
        ),
        "duplicate_candidate_rows_collapsed": duplicate_candidate_rows,
        "groups": group_results,
    }


def raw_residual_metrics(rows: Sequence[Prediction]) -> Dict[str, float]:
    raw = [float(row["raw_predicted_residual"]) for row in rows]
    negative = [-value for value in raw if value < 0.0]
    return {
        "negative_rate": len(negative) / len(raw) if raw else 0.0,
        "mean_negative_magnitude": (
            sum(negative) / len(negative) if negative else 0.0
        ),
        "max_negative_magnitude": max(negative, default=0.0),
    }


def _validation_group_folds(
    samples: Sequence[Sample], *, loog_threshold: int = 20,
    maximum_folds: int = 10,
) -> List[Tuple[str, ...]]:
    """Build deterministic, leakage-safe validation folds.

    Small datasets retain leave-one-group-out validation.  Paper-scale corpora
    use at most ten folds, assigned by a stable hash and balanced independently
    inside each declared training stratum.
    """
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        raise ValueError("group validation needs at least two groups")
    fold_count = len(groups) if len(groups) <= loog_threshold else min(
        maximum_folds, len(groups)
    )
    group_to_stratum: Dict[str, str] = {}
    has_strata = any(
        sample.metadata.get("training_stratum") not in (None, "")
        for sample in samples
    )
    for sample in samples:
        value = sample.metadata.get("training_stratum")
        if has_strata and value in (None, ""):
            raise ValueError(
                "training_stratum must be present on every sample when enabled"
            )
        stratum = str(value) if has_strata else "__all__"
        prior = group_to_stratum.setdefault(sample.group, stratum)
        if prior != stratum:
            raise ValueError(
                f"leakage group {sample.group} appears in multiple strata"
            )
    by_stratum: Dict[str, List[str]] = defaultdict(list)
    for group in groups:
        by_stratum[group_to_stratum[group]].append(group)
    folds: List[List[str]] = [[] for _ in range(fold_count)]
    offset = 0
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda group: (
                hashlib.sha256(group.encode("utf-8")).hexdigest(), group
            ),
        )
        for index, group in enumerate(ordered):
            folds[(offset + index) % fold_count].append(group)
        offset = (offset + len(ordered)) % fold_count
    return [tuple(sorted(fold)) for fold in folds if fold]


def select_ridge_hyperparameters(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Tuple[float, float]:
    _validate_samples(samples, feature_names)
    ridge_candidates = _validated_control_grid(
        ridge_candidates, "ridge", strictly_positive=True,
    )
    dead_zone_candidates = _validated_control_grid(
        dead_zone_candidates, "residual dead zone", strictly_positive=False,
    )
    selection_samples = distinct_observations(samples, feature_names)
    folds = _validation_group_folds(selection_samples)
    best: Tuple[float, float, float, float, float] = None  # type: ignore[assignment]
    for ridge in ridge_candidates:
        raw: List[Tuple[Sample, float]] = []
        for held_out_groups in folds:
            held_out = set(held_out_groups)
            train = [
                sample for sample in selection_samples
                if sample.group not in held_out
            ]
            test = [
                sample for sample in selection_samples
                if sample.group in held_out
            ]
            model = fit_ridge(
                train, feature_names, ridge,
                include_training_diagnostics=False,
            )
            raw.extend(
                (sample, raw_ridge_residual(model, sample))
                for sample in test
            )
        for dead_zone in dead_zone_candidates:
            group_errors: Dict[str, List[float]] = defaultdict(list)
            stratum_groups: Dict[str, set[str]] = defaultdict(set)
            all_errors: List[float] = []
            for sample, raw_residual in raw:
                residual = max(0.0, raw_residual)
                if residual < dead_zone:
                    residual = 0.0
                error = abs(
                    sample.lower_bound + residual - sample.compiled_ii
                )
                all_errors.append(error)
                group_errors[sample.group].append(error)
                stratum = str(sample.metadata.get(
                    "training_stratum", "__all__"
                ))
                stratum_groups[stratum].add(sample.group)
            group_maes = {
                group: sum(errors) / len(errors)
                for group, errors in group_errors.items()
            }
            macro = sum(group_maes.values()) / len(group_maes)
            stratified_macro = sum(
                sum(group_maes[group] for group in groups) / len(groups)
                for groups in stratum_groups.values()
            ) / len(stratum_groups)
            score = (
                stratified_macro,
                macro,
                sum(all_errors) / len(all_errors),
                ridge,
                dead_zone,
            )
            if best is None or score < best:
                best = score
    return best[3], best[4]


def nested_group_holdout(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate unseen lineages; tune only inside each outer training split."""
    _validate_samples(samples, feature_names)
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 3:
        raise ValueError("nested group holdout needs at least three groups")
    outer_folds = _validation_group_folds(samples)
    rows: List[Prediction] = []
    chosen: Dict[str, Dict[str, float]] = {}
    evaluation_input_count = 0
    for held_out_groups in outer_folds:
        held_out = set(held_out_groups)
        train = [sample for sample in samples if sample.group not in held_out]
        test = [sample for sample in samples if sample.group in held_out]
        evaluation_input_count += len(test)
        test = distinct_observations(test, feature_names)
        ridge, dead_zone = select_ridge_hyperparameters(
            train, feature_names, ridge_candidates, dead_zone_candidates
        )
        model = fit_ridge(
            train, feature_names, ridge, dead_zone,
            include_training_diagnostics=False,
        )
        rows.extend(prediction_rows(model, test))
        for group in held_out_groups:
            chosen[group] = {
                "ridge": ridge, "residual_dead_zone": dead_zone,
            }
    return {
        "groups": groups,
        "outer_split_protocol": (
            "leave_one_leakage_group_out"
            if len(outer_folds) == len(groups)
            else "deterministic_stratified_group_k_fold"
        ),
        "outer_fold_count": len(outer_folds),
        "outer_held_out_groups_by_fold": [list(fold) for fold in outer_folds],
        "rows": rows,
        "evaluation_input_row_count": evaluation_input_count,
        "evaluation_distinct_observation_count": len(rows),
        "evaluation_duplicate_rows_collapsed": evaluation_input_count - len(rows),
        "chosen_hyperparameters_by_held_out_group": chosen,
        "lower_bound_metrics": quality_metrics(rows, "lower_bound"),
        "model_metrics": quality_metrics(rows, "prediction"),
        "lower_bound_group_ranking": group_ranking_metrics(rows, "lower_bound"),
        "model_group_ranking": group_ranking_metrics(rows, "prediction"),
        "raw_residual_metrics": raw_residual_metrics(rows),
    }


def fit_calibrated_model(
    samples: Sequence[Sample], feature_names: Sequence[str],
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Model:
    ridge, dead_zone = select_ridge_hyperparameters(
        samples, feature_names, ridge_candidates, dead_zone_candidates
    )
    return fit_ridge(samples, feature_names, ridge, dead_zone)
