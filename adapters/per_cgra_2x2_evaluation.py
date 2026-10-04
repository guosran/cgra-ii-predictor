"""Evaluation helpers for the frozen per-CGRA 2x2 shape roster.

The module works on already joined prediction and native-outcome rows. It has
no model or torch dependency, and it treats an incomplete native query as
unknown evidence rather than imputing an II value.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


SHAPE_ROSTER = (
    (2, 2), (2, 4), (4, 2), (2, 6),
    (6, 2), (2, 8), (8, 2), (4, 4),
)
EVALUATION_SCHEMA = "cgra-ii-per-cgra-2x2-evaluation-v1"
_REQUIRED_FIELDS = (
    "identity", "source_group", "source_families", "source_kind", "motif",
    "split", "shape", "rec_mii", "res_mii", "lower_bound", "true_ii",
    "native_status", "scores",
)
_UNKNOWN_STATUSES = {
    "pending", "unknown", "not_run", "not-run", "not_attempted",
    "not-attempted", "unattempted", "unavailable", "missing", "skipped",
}


def _finite_number(value: object, label: str, *, nullable: bool = True) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number" + (" or null" if nullable else ""))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    try:
        raw_rows = list(rows)
    except TypeError as error:
        raise ValueError("rows must be an iterable of objects") from error
    if not raw_rows:
        raise ValueError("rows must not be empty")

    by_identity: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    metadata_by_identity: dict[str, tuple[Any, ...]] = {}
    roster = set(SHAPE_ROSTER)
    for row_number, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"row {row_number} must be an object")
        missing_fields = [field for field in _REQUIRED_FIELDS if field not in raw]
        if missing_fields:
            raise ValueError(f"row {row_number} is missing required fields: {missing_fields}")

        identity = raw["identity"]
        source_group = raw["source_group"]
        source_kind = raw["source_kind"]
        motif = raw["motif"]
        split = raw["split"]
        source_families = raw["source_families"]
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"row {row_number} identity must be a non-empty string")
        if not isinstance(source_group, str) or not source_group:
            raise ValueError(f"row {row_number} source_group must be a non-empty string")
        if not isinstance(source_kind, str) or source_kind not in {"random", "program"}:
            raise ValueError(f"row {row_number} source_kind must be 'random' or 'program'")
        if not isinstance(motif, str) or not isinstance(split, str) or not split:
            raise ValueError(f"row {row_number} motif and split must be strings")
        if (not isinstance(source_families, list) or
                any(not isinstance(family, str) or not family for family in source_families) or
                len(set(source_families)) != len(source_families)):
            raise ValueError(f"row {row_number} source_families must be unique non-empty strings")

        shape_value = raw["shape"]
        if (not isinstance(shape_value, (list, tuple)) or len(shape_value) != 2 or
                any(isinstance(dimension, bool) or not isinstance(dimension, int)
                    for dimension in shape_value)):
            raise ValueError(f"row {row_number} shape must be a pair of integers")
        shape = (int(shape_value[0]), int(shape_value[1]))
        if shape not in roster:
            raise ValueError(f"row {row_number} has shape outside the fixed roster: {shape}")

        status = raw["native_status"]
        if not isinstance(status, str) or not status:
            raise ValueError(f"row {row_number} native_status must be a non-empty string")
        scores = raw["scores"]
        if not isinstance(scores, Mapping):
            raise ValueError(f"row {row_number} scores must be an object")
        normalized_scores: dict[str, float | None] = {}
        for method, value in scores.items():
            if not isinstance(method, str) or not method:
                raise ValueError(f"row {row_number} score names must be non-empty strings")
            normalized_scores[method] = _finite_number(
                value, f"row {row_number} score {method!r}")

        rec_mii = _finite_number(raw["rec_mii"], f"row {row_number} rec_mii")
        res_mii = _finite_number(raw["res_mii"], f"row {row_number} res_mii")
        lower_bound = _finite_number(raw["lower_bound"], f"row {row_number} lower_bound")
        true_ii = _finite_number(raw["true_ii"], f"row {row_number} true_ii")
        if true_ii is not None and true_ii <= 0:
            raise ValueError(f"row {row_number} true_ii must be positive")

        metadata = (source_group, tuple(source_families), source_kind, motif, split)
        previous = metadata_by_identity.setdefault(identity, metadata)
        if previous != metadata:
            raise ValueError(f"DFG {identity!r} has inconsistent source metadata across shapes")
        if shape in by_identity[identity]:
            raise ValueError(f"duplicate row for DFG {identity!r} and shape {shape}")

        by_identity[identity][shape] = {
            "identity": identity,
            "source_group": source_group,
            "source_families": list(source_families),
            "source_kind": source_kind,
            "motif": motif,
            "split": split,
            "shape": shape,
            "rec_mii": rec_mii,
            "res_mii": res_mii,
            "lower_bound": lower_bound,
            "true_ii": true_ii,
            "native_status": status,
            "scores": normalized_scores,
        }

    normalized: dict[str, list[dict[str, Any]]] = {}
    for identity, shape_rows in by_identity.items():
        actual = set(shape_rows)
        if actual != roster:
            missing = [list(shape) for shape in SHAPE_ROSTER if shape not in actual]
            extra = [list(shape) for shape in actual if shape not in roster]
            raise ValueError(
                f"DFG {identity!r} must have exactly the fixed shape roster; "
                f"missing={missing}, extra={extra}")
        normalized[identity] = [shape_rows[shape] for shape in SHAPE_ROSTER]
    return normalized


def _row_score(row: Mapping[str, Any], score_name: str) -> float | None:
    return row["scores"].get(score_name)


def _is_success(row: Mapping[str, Any]) -> bool:
    return row["native_status"].strip().lower() == "success"


def _outcome_bucket(row: Mapping[str, Any]) -> str:
    status = row["native_status"].strip().lower()
    if status == "success" and row["true_ii"] is not None:
        return "successful_label"
    if status == "success":
        return "unknown"
    if status in _UNKNOWN_STATUSES or "pending" in status or "unknown" in status:
        return "unknown"
    return "failed"


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _quantile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    left = math.floor(position)
    right = math.ceil(position)
    fraction = position - left
    return float(ordered[left] * (1.0 - fraction) + ordered[right] * fraction)


def _point_metrics(records: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "count": 0, "mae": None, "mean_signed_error": None,
            "rounded_exact_accuracy": None, "within_one_accuracy": None,
            "rounded_within_one_accuracy": None,
        }
    errors = [prediction - truth for truth, prediction in records]
    rounded_matches = [
        _round_half_away_from_zero(prediction) == _round_half_away_from_zero(truth)
        for truth, prediction in records
    ]
    within_one = [abs(prediction - truth) <= 1.0 for truth, prediction in records]
    rounded_within_one = [
        abs(_round_half_away_from_zero(prediction) -
            _round_half_away_from_zero(truth)) <= 1
        for truth, prediction in records
    ]
    return {
        "count": len(records),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mean_signed_error": sum(errors) / len(errors),
        "rounded_exact_accuracy": sum(rounded_matches) / len(records),
        "within_one_accuracy": sum(within_one) / len(records),
        "rounded_within_one_accuracy": sum(rounded_within_one) / len(records),
    }


def _make_rank_decision(
    identity: str, rows: list[dict[str, Any]], score_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    shape_index = {shape: index for index, shape in enumerate(SHAPE_ROSTER)}
    ranked_rows = sorted(
        (row for row in rows if _row_score(row, score_name) is not None),
        key=lambda row: (_row_score(row, score_name), shape_index[row["shape"]]),
    )
    ranked_shapes = [row["shape"] for row in ranked_rows]
    missing_outcomes = []
    for row in rows:
        reasons = []
        if not _is_success(row):
            reasons.append("native_not_success")
        if row["true_ii"] is None:
            reasons.append("missing_true_ii")
        if _row_score(row, score_name) is None:
            reasons.append("missing_score")
        if reasons:
            missing_outcomes.append({"shape": list(row["shape"]), "reasons": reasons})

    complete = not missing_outcomes
    public: dict[str, Any] = {
        "identity": identity,
        "source_group": rows[0]["source_group"],
        "split": rows[0]["split"],
        "rank_metrics_evaluable": complete,
        "ranked_shapes": [
            {"shape": list(row["shape"]), "score": _row_score(row, score_name)}
            for row in ranked_rows
        ],
        "predicted_shape": list(ranked_shapes[0]) if ranked_shapes else None,
        "oracle_shape": None,
        "oracle_optimal_shapes": None,
        "oracle_true_ii": None,
        "absolute_regret": None,
        "relative_regret": None,
        "best_set_hit": None,
        "top2_oracle_hit": None,
        "top3_oracle_hit": None,
        "pairwise_comparable_count": 0,
        "pairwise_correct_count": None,
        "pairwise_predicted_tie_count": 0,
        "pairwise_accuracy": None,
        "missing_outcomes": missing_outcomes,
    }
    if not complete:
        return public, None

    truth_by_shape = {row["shape"]: row["true_ii"] for row in rows}
    best_truth = min(truth_by_shape.values())
    optimal_shapes = [shape for shape in SHAPE_ROSTER if truth_by_shape[shape] == best_truth]
    predicted_shape = ranked_shapes[0]
    selected_truth = truth_by_shape[predicted_shape]
    absolute_regret = selected_truth - best_truth
    relative_regret = absolute_regret / best_truth

    pairwise_comparable = 0
    pairwise_correct = 0.0
    pairwise_predicted_ties = 0
    for left_index, left_shape in enumerate(SHAPE_ROSTER):
        for right_shape in SHAPE_ROSTER[left_index + 1:]:
            left_truth = truth_by_shape[left_shape]
            right_truth = truth_by_shape[right_shape]
            if left_truth == right_truth:
                continue
            pairwise_comparable += 1
            left_score = _row_score(rows[shape_index[left_shape]], score_name)
            right_score = _row_score(rows[shape_index[right_shape]], score_name)
            if left_score == right_score:
                pairwise_correct += 0.5
                pairwise_predicted_ties += 1
            elif (left_score < right_score) == (left_truth < right_truth):
                pairwise_correct += 1.0

    pairwise_accuracy = (
        pairwise_correct / pairwise_comparable if pairwise_comparable else None)
    public.update({
        "oracle_shape": list(optimal_shapes[0]),
        "oracle_optimal_shapes": [list(shape) for shape in optimal_shapes],
        "oracle_true_ii": best_truth,
        "absolute_regret": absolute_regret,
        "relative_regret": relative_regret,
        "best_set_hit": predicted_shape in optimal_shapes,
        "top2_oracle_hit": any(shape in optimal_shapes for shape in ranked_shapes[:2]),
        "top3_oracle_hit": any(shape in optimal_shapes for shape in ranked_shapes[:3]),
        "pairwise_comparable_count": pairwise_comparable,
        "pairwise_correct_count": pairwise_correct,
        "pairwise_predicted_tie_count": pairwise_predicted_ties,
        "pairwise_accuracy": pairwise_accuracy,
        "optimal_set_size": len(optimal_shapes),
        "all8_equal": len(optimal_shapes) == len(SHAPE_ROSTER),
    })
    internal = {
        "best_set_hit": float(predicted_shape in optimal_shapes),
        "top2_oracle_hit": float(any(shape in optimal_shapes for shape in ranked_shapes[:2])),
        "top3_oracle_hit": float(any(shape in optimal_shapes for shape in ranked_shapes[:3])),
        "absolute_regret": absolute_regret,
        "relative_regret": relative_regret,
        "pairwise_comparable_count": pairwise_comparable,
        "pairwise_correct_count": pairwise_correct,
        "optimal_set_size": len(optimal_shapes),
        "all8_equal": float(len(optimal_shapes) == len(SHAPE_ROSTER)),
    }
    return public, internal


def rank_decisions(rows: Iterable[Mapping[str, Any]], score_name: str) -> list[dict[str, Any]]:
    """Return deterministic per-DFG rankings, leaving incomplete oracles null."""
    if not isinstance(score_name, str) or not score_name:
        raise ValueError("score_name must be a non-empty string")
    grouped = _normalize_rows(rows)
    return [
        _make_rank_decision(identity, grouped[identity], score_name)[0]
        for identity in sorted(grouped)
    ]


def _rank_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "complete_dfg_count": 0,
            "best_set_hit_rate": None,
            "top2_oracle_hit_rate": None,
            "top3_oracle_hit_rate": None,
            "mean_absolute_regret": None,
            "mean_relative_regret": None,
            "absolute_regret_p50": None,
            "absolute_regret_p95": None,
            "absolute_regret_max": None,
            "relative_regret_p50": None,
            "relative_regret_p95": None,
            "relative_regret_max": None,
            "pairwise_accuracy": None,
            "pairwise_comparable_count": 0,
            "pairwise_correct_count": None,
            "mean_optimal_set_size": None,
            "all8_equal_fraction": None,
        }
    absolute = [record["absolute_regret"] for record in records]
    relative = [record["relative_regret"] for record in records]
    pairwise_count = sum(record["pairwise_comparable_count"] for record in records)
    pairwise_correct = sum(record["pairwise_correct_count"] for record in records)
    count = len(records)
    return {
        "complete_dfg_count": count,
        "best_set_hit_rate": _mean(record["best_set_hit"] for record in records),
        "top2_oracle_hit_rate": _mean(record["top2_oracle_hit"] for record in records),
        "top3_oracle_hit_rate": _mean(record["top3_oracle_hit"] for record in records),
        "mean_absolute_regret": _mean(absolute),
        "mean_relative_regret": _mean(relative),
        "absolute_regret_p50": _quantile(absolute, 0.50),
        "absolute_regret_p95": _quantile(absolute, 0.95),
        "absolute_regret_max": max(absolute),
        "relative_regret_p50": _quantile(relative, 0.50),
        "relative_regret_p95": _quantile(relative, 0.95),
        "relative_regret_max": max(relative),
        "pairwise_accuracy": (
            pairwise_correct / pairwise_count if pairwise_count else None),
        "pairwise_comparable_count": pairwise_count,
        "pairwise_correct_count": pairwise_correct if pairwise_count else None,
        "mean_optimal_set_size": _mean(record["optimal_set_size"] for record in records),
        "all8_equal_fraction": _mean(record["all8_equal"] for record in records),
    }


def _shortlist_metrics(
    grouped: Mapping[str, list[dict[str, Any]]], score_name: str,
) -> list[dict[str, Any]]:
    shape_index = {shape: index for index, shape in enumerate(SHAPE_ROSTER)}
    dfg_count = len(grouped)
    result = []
    for budget in range(1, len(SHAPE_ROSTER) + 1):
        query_rows = []
        unranked_missing_score = 0
        dfgs_queried = 0
        dfgs_with_success = set()
        for identity in sorted(grouped):
            ranked = sorted(
                (row for row in grouped[identity] if _row_score(row, score_name) is not None),
                key=lambda row: (_row_score(row, score_name), shape_index[row["shape"]]),
            )
            selected = ranked[:budget]
            query_rows.extend(selected)
            unranked_missing_score += budget - len(selected)
            if selected:
                dfgs_queried += 1
            if any(_outcome_bucket(row) == "successful_label" for row in selected):
                dfgs_with_success.add(identity)

        counts = Counter(_outcome_bucket(row) for row in query_rows)
        successful_values = [
            row["true_ii"] for row in query_rows
            if _outcome_bucket(row) == "successful_label"
        ]
        requested = dfg_count * budget
        successful_count = counts["successful_label"]
        failed_count = counts["failed"]
        unknown_count = counts["unknown"]
        attempted_with_known_result = successful_count + failed_count
        result.append({
            "budget_per_dfg": budget,
            "requested_query_count": requested,
            "queried_count": len(query_rows),
            "unranked_missing_score_count": unranked_missing_score,
            "query_coverage": len(query_rows) / requested if requested else None,
            "successful_label_count": successful_count,
            "failed_count": failed_count,
            "unknown_count": unknown_count,
            "successful_label_coverage": successful_count / requested if requested else None,
            "known_outcome_coverage": (
                attempted_with_known_result / requested if requested else None),
            "successful_rate_among_known_outcomes": (
                successful_count / attempted_with_known_result
                if attempted_with_known_result else None),
            "dfgs_queried": dfgs_queried,
            "dfgs_with_successful_label": len(dfgs_with_success),
            "dfg_query_coverage": dfgs_queried / dfg_count if dfg_count else None,
            "dfg_successful_label_coverage": (
                len(dfgs_with_success) / dfg_count if dfg_count else None),
            "observed_successful_true_ii_mean": _mean(successful_values),
            "observed_successful_true_ii_best": min(successful_values) if successful_values else None,
        })
    return result


def _oracle_confirmation_observations(
    rows: list[dict[str, Any]], score_name: str,
) -> dict[int, dict[str, Any]] | None:
    """Summarize top-k confirmation only when all eight outcomes are known."""
    if any(not _is_success(row) or row["true_ii"] is None or
           _row_score(row, score_name) is None for row in rows):
        return None

    shape_index = {shape: index for index, shape in enumerate(SHAPE_ROSTER)}
    ranked = sorted(
        rows,
        key=lambda row: (_row_score(row, score_name), shape_index[row["shape"]]),
    )
    oracle_ii = min(row["true_ii"] for row in rows)
    result = {}
    for budget in range(1, len(SHAPE_ROSTER) + 1):
        confirmed = ranked[:budget]
        best_measured_ii = min(row["true_ii"] for row in confirmed)
        best_rows = [row for row in confirmed if row["true_ii"] == best_measured_ii]
        selected = min(
            best_rows,
            key=lambda row: (row["shape"][0] * row["shape"][1],
                             shape_index[row["shape"]]),
        )
        absolute_regret = best_measured_ii - oracle_ii
        result[budget] = {
            "best_measured_ii": best_measured_ii,
            "absolute_regret": absolute_regret,
            "relative_regret": absolute_regret / oracle_ii,
            "optimal_hit": float(best_measured_ii == oracle_ii),
            "within_one_ii": float(absolute_regret <= 1),
            "selected_shape": selected["shape"],
            "selected_tile_count": selected["shape"][0] * selected["shape"][1],
        }
    return result


def _oracle_confirmation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected_shape_counts = Counter(
        f"{record['selected_shape'][0]}x{record['selected_shape'][1]}"
        for record in records
    )
    return {
        "complete_dfg_count": len(records),
        "mean_best_measured_ii": _mean(record["best_measured_ii"] for record in records),
        "mean_absolute_regret": _mean(record["absolute_regret"] for record in records),
        "mean_relative_regret": _mean(record["relative_regret"] for record in records),
        "optimal_hit_rate": _mean(record["optimal_hit"] for record in records),
        "within_one_ii_fraction": _mean(record["within_one_ii"] for record in records),
        "mean_selected_tile_count": _mean(record["selected_tile_count"] for record in records),
        "selected_shape_counts": dict(sorted(selected_shape_counts.items())),
    }


def _oracle_confirmation_curve(
    grouped_observations: Mapping[int, Mapping[str, list[dict[str, Any]]]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    result = []
    ci_metrics = (
        "mean_absolute_regret", "mean_relative_regret", "optimal_hit_rate",
        "within_one_ii_fraction",
    )
    rng = random.Random(seed)
    for budget in range(1, len(SHAPE_ROSTER) + 1):
        observations_by_group = grouped_observations.get(budget, {})
        records = [
            record
            for group in sorted(observations_by_group)
            for record in observations_by_group[group]
        ]
        summary = _oracle_confirmation_summary(records)
        source_group_count = len(observations_by_group)
        summary.update({
            "budget_per_dfg": budget,
            "source_group_count": source_group_count,
            "confirmed_query_count": len(records) * budget,
        })
        if bootstrap_samples > 0 and observations_by_group:
            bootstrap = _bootstrap_group_metrics(
                observations_by_group,
                _oracle_confirmation_summary,
                ci_metrics,
                bootstrap_samples,
                rng,
            )
            summary["bootstrap_ci"] = bootstrap["metrics"]
        else:
            summary["bootstrap_ci"] = None
        result.append(summary)
    return result


def _bootstrap_group_metrics(
    grouped_observations: Mapping[str, list[Any]],
    summarize: Any,
    metric_names: tuple[str, ...],
    samples: int,
    rng: random.Random,
) -> dict[str, Any]:
    group_names = sorted(grouped_observations)
    intervals: dict[str, Any] = {}
    for metric in metric_names:
        values = []
        for _ in range(samples):
            sampled_groups = [rng.choice(group_names) for _ in group_names]
            observations = [
                observation
                for group in sampled_groups
                for observation in grouped_observations[group]
            ]
            value = summarize(observations).get(metric)
            if value is not None:
                values.append(float(value))
        intervals[metric] = {
            "lower_95": _quantile(values, 0.025),
            "upper_95": _quantile(values, 0.975),
            "valid_replicates": len(values),
        }
    return {"source_group_count": len(group_names), "metrics": intervals}


def _bootstrap_metrics(
    point_groups: Mapping[str, list[tuple[float, float]]],
    rank_groups: Mapping[str, list[dict[str, Any]]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    point_names = (
        "mae", "mean_signed_error", "rounded_exact_accuracy", "within_one_accuracy",
        "rounded_within_one_accuracy",
    )
    rank_names = (
        "best_set_hit_rate", "top2_oracle_hit_rate", "top3_oracle_hit_rate",
        "mean_absolute_regret", "mean_relative_regret", "absolute_regret_p50",
        "absolute_regret_p95", "absolute_regret_max", "relative_regret_p50",
        "relative_regret_p95", "relative_regret_max", "pairwise_accuracy",
        "mean_optimal_set_size", "all8_equal_fraction",
    )
    result: dict[str, Any] = {
        "method": "source_group_percentile_bootstrap",
        "confidence_level": 0.95,
        "requested_samples": samples,
        "seed": seed,
        "resampling_unit": "source_group",
        "point": {"source_group_count": len(point_groups), "metrics": {}},
        "rank": {"source_group_count": len(rank_groups), "metrics": {}},
    }
    if samples <= 0:
        return result
    rng = random.Random(seed)
    if point_groups:
        result["point"] = _bootstrap_group_metrics(
            point_groups, _point_metrics, point_names, samples, rng)
    if rank_groups:
        result["rank"] = _bootstrap_group_metrics(
            rank_groups, _rank_metrics, rank_names, samples, rng)
    return result


def evaluate_predictions(
    rows: Iterable[Mapping[str, Any]],
    score_name: str,
    *,
    bootstrap_samples: int = 0,
    seed: int = 20261004,
) -> dict[str, Any]:
    """Evaluate one named prediction score against complete native outcomes.

    Point metrics use every successful, numerically labeled, scored query.
    Shape selection metrics require all eight roster queries to be successful,
    labeled, and scored for the DFG. Bootstrap intervals resample source groups
    and retain all observations from each sampled group.
    """
    if not isinstance(score_name, str) or not score_name:
        raise ValueError("score_name must be a non-empty string")
    if (isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or
            bootstrap_samples < 0):
        raise ValueError("bootstrap_samples must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    grouped = _normalize_rows(rows)
    point_records: list[tuple[float, float]] = []
    point_by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    point_strata: dict[float, list[tuple[float, float]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    scored_count = 0
    successful_label_count = 0
    successful_label_dfg_count = 0
    rank_decision_rows: list[dict[str, Any]] = []
    complete_rank_records: list[dict[str, Any]] = []
    rank_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    oracle_observations_by_budget: dict[int, dict[str, list[dict[str, Any]]]] = {
        budget: defaultdict(list)
        for budget in range(1, len(SHAPE_ROSTER) + 1)
    }
    excluded_dfgs = []

    for identity in sorted(grouped):
        identity_rows = grouped[identity]
        for row in identity_rows:
            status_counts[row["native_status"]] += 1
            if _row_score(row, score_name) is not None:
                scored_count += 1
            if _is_success(row) and row["true_ii"] is not None:
                successful_label_count += 1
            prediction = _row_score(row, score_name)
            if _is_success(row) and row["true_ii"] is not None and prediction is not None:
                record = (row["true_ii"], prediction)
                point_records.append(record)
                point_by_group[row["source_group"]].append(record)
                point_strata[row["true_ii"]].append(record)
        if any(_is_success(row) and row["true_ii"] is not None for row in identity_rows):
            successful_label_dfg_count += 1

        public, internal = _make_rank_decision(identity, identity_rows, score_name)
        rank_decision_rows.append(public)
        if internal is None:
            reasons = Counter(
                reason
                for outcome in public["missing_outcomes"]
                for reason in outcome["reasons"]
            )
            excluded_dfgs.append({
                "identity": identity,
                "source_group": identity_rows[0]["source_group"],
                "reasons": dict(sorted(reasons.items())),
                "incomplete_shapes": [
                    outcome["shape"] for outcome in public["missing_outcomes"]
                ],
            })
        else:
            complete_rank_records.append(internal)
            source_group = identity_rows[0]["source_group"]
            rank_by_group[source_group].append(internal)
            confirmation = _oracle_confirmation_observations(identity_rows, score_name)
            if confirmation is None:  # Keep the rank and confirmation filters identical.
                raise AssertionError("complete rank DFG is not complete for confirmation")
            for budget, record in confirmation.items():
                oracle_observations_by_budget[budget][source_group].append(record)

    point_summary = _point_metrics(point_records)
    rank_summary = _rank_metrics(complete_rank_records)
    point_stratum_summary = [
        {
            "true_ii": truth,
            **_point_metrics(point_strata[truth]),
        }
        for truth in sorted(point_strata)
    ]
    optimal_set_sizes = Counter(
        decision["optimal_set_size"] for decision in complete_rank_records)
    rank_eligible_count = len(complete_rank_records)
    total_dfg_count = len(grouped)
    row_count = total_dfg_count * len(SHAPE_ROSTER)
    bootstrap = _bootstrap_metrics(
        point_by_group, rank_by_group, bootstrap_samples, seed)
    oracle_confirmation = _oracle_confirmation_curve(
        oracle_observations_by_budget, bootstrap_samples, seed + 1)
    roster_source_group_count = len({
        identity_rows[0]["source_group"] for identity_rows in grouped.values()
    })

    return {
        "schema": EVALUATION_SCHEMA,
        "score_name": score_name,
        "shape_roster": [list(shape) for shape in SHAPE_ROSTER],
        "metric_definitions": {
            "error": "prediction minus true_ii",
            "rounded_exact": "nearest integer, halves away from zero, matches rounded true_ii",
            "within_one": "absolute unrounded prediction error is at most one II",
            "rounded_within_one": (
                "absolute difference between rounded prediction and rounded true_ii is at most one II"),
            "pairwise_accuracy": (
                "lower score predicts lower true_ii; equal true_ii pairs are excluded "
                "and tied scores receive half credit"),
            "bootstrap": "percentile intervals resample source groups with replacement",
            "shortlist": (
                "observed successful labels and native outcome coverage only; no "
                "oracle hit or regret is inferred from unqueried shapes"),
        },
        "coverage": {
            "dfg_count": total_dfg_count,
            "roster_query_count": row_count,
            "scored_query_count": scored_count,
            "successful_numeric_label_query_count": successful_label_count,
            "successful_numeric_label_dfg_count": successful_label_dfg_count,
            "point_evaluable_query_count": point_summary["count"],
            "point_excluded_query_count": row_count - point_summary["count"],
            "native_status_counts": dict(sorted(status_counts.items())),
            "point_source_group_count": len(point_by_group),
            "roster_source_group_count": roster_source_group_count,
            "complete_rank_dfg_count": rank_eligible_count,
            "incomplete_rank_dfg_count": len(excluded_dfgs),
            "rank_dfg_coverage": rank_eligible_count / total_dfg_count,
            "rank_source_group_count": len(rank_by_group),
            "excluded_dfgs": excluded_dfgs,
        },
        "point": {
            **point_summary,
            "true_ii_strata": point_stratum_summary,
        },
        "rank": {
            "metrics": rank_summary,
            "optimal_set_size_counts": {
                str(size): optimal_set_sizes[size]
                for size in sorted(optimal_set_sizes)
            },
        },
        "shortlist_native_confirmation": _shortlist_metrics(grouped, score_name),
        "oracle_confirmation": oracle_confirmation,
        "bootstrap": bootstrap,
        "rank_decisions": rank_decision_rows,
    }


def choose_fixed_shape(validation_rows: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """Choose one shape by mean true II over complete validation DFGs only.

    Requiring the validation split in the input prevents accidental use of test
    labels for this baseline. Every supplied validation DFG must have all eight
    successful numeric labels so each shape is compared on the same DFG set.
    """
    grouped = _normalize_rows(validation_rows)
    for identity, rows in grouped.items():
        if rows[0]["split"].strip().lower() not in {"validation", "val"}:
            raise ValueError(
                f"fixed-shape selection accepts validation rows only; "
                f"DFG {identity!r} has split {rows[0]['split']!r}")
        if any(not _is_success(row) or row["true_ii"] is None for row in rows):
            raise ValueError(
                f"validation DFG {identity!r} does not have eight successful numeric labels")

    means = {
        shape: sum(
            next(row["true_ii"] for row in grouped[identity] if row["shape"] == shape)
            for identity in grouped
        ) / len(grouped)
        for shape in SHAPE_ROSTER
    }
    # A per-DFG oracle regret subtracts the same oracle II from every shape,
    # so minimizing mean regret is equivalent to minimizing mean true II.
    return min(SHAPE_ROSTER, key=lambda shape: (means[shape], SHAPE_ROSTER.index(shape)))
