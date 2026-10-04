from __future__ import annotations

import json

import pytest

from adapters.per_cgra_2x2_evaluation import (
    SHAPE_ROSTER,
    choose_fixed_shape,
    evaluate_predictions,
    rank_decisions,
)


def _rows(
    identity: str,
    *,
    truths: list[int | None] | None = None,
    scores: list[float | None] | None = None,
    statuses: list[str] | None = None,
    source_group: str = "group-a",
    split: str = "test",
) -> list[dict]:
    truths = truths if truths is not None else [1] * len(SHAPE_ROSTER)
    scores = scores if scores is not None else [1.0] * len(SHAPE_ROSTER)
    statuses = statuses if statuses is not None else ["success"] * len(SHAPE_ROSTER)
    assert len(truths) == len(scores) == len(statuses) == len(SHAPE_ROSTER)
    result = []
    for shape, truth, score, status in zip(SHAPE_ROSTER, truths, scores, statuses):
        result.append({
            "identity": identity,
            "source_group": source_group,
            "source_families": ["family-a"],
            "source_kind": "program",
            "motif": "reduction",
            "split": split,
            "shape": list(shape),
            "rec_mii": 1,
            "res_mii": 1,
            "lower_bound": 1,
            "true_ii": truth,
            "native_status": status,
            "scores": {"model": score},
        })
    return result


def test_point_strata_tied_oracle_set_and_pairwise_metrics_are_deterministic():
    truths = [1, 1, 2, 3, 4, 5, 5, 5]
    predictions = [1.4, 1.6, 1.49, 4.2, 3.5, 5, 5, 5]
    rows = _rows("dfg-a", truths=truths, scores=predictions)

    report = evaluate_predictions(rows, "model")
    assert report["coverage"]["complete_rank_dfg_count"] == 1
    assert report["point"]["count"] == 8
    assert report["point"]["mae"] == pytest.approx(0.40125)
    assert report["point"]["mean_signed_error"] == pytest.approx(0.14875)
    assert report["point"]["rounded_exact_accuracy"] == pytest.approx(5 / 8)
    assert report["point"]["within_one_accuracy"] == pytest.approx(7 / 8)
    assert report["point"]["rounded_within_one_accuracy"] == 1
    assert [row["count"] for row in report["point"]["true_ii_strata"]] == [2, 1, 1, 1, 3]

    rank = report["rank"]["metrics"]
    assert rank["best_set_hit_rate"] == 1
    assert rank["top2_oracle_hit_rate"] == 1
    assert rank["top3_oracle_hit_rate"] == 1
    assert rank["mean_optimal_set_size"] == 2
    assert rank["pairwise_comparable_count"] == 24
    assert rank["all8_equal_fraction"] == 0
    decision = rank_decisions(rows, "model")[0]
    assert decision["predicted_shape"] == [2, 2]
    assert decision["oracle_optimal_shapes"] == [[2, 2], [2, 4]]
    assert decision["oracle_shape"] == [2, 2]
    assert decision["pairwise_comparable_count"] == 24
    json.dumps(report, allow_nan=False)


def test_failed_and_unknown_outcomes_stay_unlabeled_and_consume_shortlist_budget():
    truths = [None, None, 3, 4, 5, 6, 7, 8]
    statuses = ["censored", "pending", "success", "success", "success",
                "success", "success", "success"]
    rows = _rows("partial", truths=truths, statuses=statuses)

    report = evaluate_predictions(rows, "model")
    assert report["coverage"]["point_evaluable_query_count"] == 6
    assert report["coverage"]["successful_numeric_label_dfg_count"] == 1
    assert report["coverage"]["roster_source_group_count"] == 1
    assert report["coverage"]["complete_rank_dfg_count"] == 0
    assert report["coverage"]["incomplete_rank_dfg_count"] == 1
    assert report["rank"]["metrics"]["best_set_hit_rate"] is None
    excluded = report["coverage"]["excluded_dfgs"][0]
    assert excluded["reasons"] == {"missing_true_ii": 2, "native_not_success": 2}

    shortlist = report["shortlist_native_confirmation"]
    assert shortlist[0]["failed_count"] == 1
    assert shortlist[0]["successful_label_count"] == 0
    assert shortlist[1]["unknown_count"] == 1
    assert shortlist[2]["successful_label_count"] == 1
    assert shortlist[2]["observed_successful_true_ii_best"] == 3
    assert not any("oracle" in key for key in shortlist[0])
    assert all(row["complete_dfg_count"] == 0 for row in report["oracle_confirmation"])
    assert all(row["mean_absolute_regret"] is None for row in report["oracle_confirmation"])


def test_rounded_within_one_has_rounded_boundary_and_baseline_accuracy():
    baseline = _rows(
        "round-baseline", truths=[1] * 8,
        scores=[1, 1, 1, 1, 1, 1, 1, 2.6])
    baseline_report = evaluate_predictions(baseline, "model")
    assert baseline_report["point"]["rounded_within_one_accuracy"] == pytest.approx(0.875)

    boundary = _rows(
        "round-boundary", truths=[1] * 8,
        scores=[1, 1, 1, 1, 1, 1, 1, 2.4])
    boundary_report = evaluate_predictions(boundary, "model")
    assert boundary_report["point"]["within_one_accuracy"] == pytest.approx(0.875)
    assert boundary_report["point"]["rounded_within_one_accuracy"] == 1


def test_complete_dfg_confirmation_curve_tracks_budget_regret_hits_and_ties():
    tie_dfg = _rows(
        "tie", truths=[1, 1, 1, 2, 2, 2, 2, 2],
        scores=[3, 1, 2, 4, 5, 6, 7, 8], source_group="group-tie")
    regret_dfg = _rows(
        "regret", truths=[2, 4, 4, 4, 4, 4, 4, 4],
        scores=[2, 3, 4, 1, 5, 6, 7, 8], source_group="group-regret")

    report = evaluate_predictions(
        tie_dfg + regret_dfg, "model", bootstrap_samples=40, seed=17)
    curve = report["oracle_confirmation"]
    assert [entry["budget_per_dfg"] for entry in curve] == list(range(1, 9))
    assert curve[0]["complete_dfg_count"] == 2
    assert curve[0]["source_group_count"] == 2
    assert curve[0]["mean_absolute_regret"] == pytest.approx(1)
    assert curve[0]["mean_relative_regret"] == pytest.approx(0.5)
    assert curve[0]["optimal_hit_rate"] == pytest.approx(0.5)
    assert curve[0]["within_one_ii_fraction"] == pytest.approx(0.5)
    assert curve[0]["bootstrap_ci"]["mean_absolute_regret"]["valid_replicates"] == 40
    assert curve[0]["bootstrap_ci"]["optimal_hit_rate"]["valid_replicates"] == 40

    # Native-equal II candidates use fewer mapper tiles, then frozen roster order.
    assert curve[1]["selected_shape_counts"] == {"2x2": 1, "2x4": 1}
    assert curve[2]["selected_shape_counts"] == {"2x2": 2}
    assert curve[2]["mean_selected_tile_count"] == 4
    assert curve[1]["mean_absolute_regret"] == 0
    assert curve[1]["optimal_hit_rate"] == 1


def test_bootstrap_counts_and_resamples_source_groups_not_individual_dfgs():
    # Two DFGs share one source group. The report must preserve the distinction
    # between the three complete DFGs and the two independent resampling units.
    first = _rows("a", source_group="shared", scores=[1.0] * 8)
    second = _rows("b", source_group="shared", scores=[2.0] * 8)
    third = _rows("c", source_group="independent", scores=[4.0] * 8)
    rows = first + second + third

    one = evaluate_predictions(rows, "model", bootstrap_samples=60, seed=13)
    two = evaluate_predictions(rows, "model", bootstrap_samples=60, seed=13)
    assert one["coverage"]["complete_rank_dfg_count"] == 3
    assert one["coverage"]["rank_source_group_count"] == 2
    assert one["bootstrap"]["resampling_unit"] == "source_group"
    assert one["bootstrap"]["point"]["source_group_count"] == 2
    assert one["bootstrap"]["rank"]["source_group_count"] == 2
    assert one["bootstrap"]["point"]["metrics"]["mae"] == two["bootstrap"]["point"]["metrics"]["mae"]
    assert one["bootstrap"]["point"]["metrics"]["mae"]["valid_replicates"] == 60
    assert one["bootstrap"]["rank"]["metrics"]["best_set_hit_rate"]["valid_replicates"] == 60


def test_fixed_shape_uses_only_complete_validation_labels_and_roster_ties():
    first = _rows(
        "validation-a", truths=[5, 1, 2, 3, 4, 5, 6, 7],
        split="validation")
    second = _rows(
        "validation-b", truths=[5, 1, 4, 2, 5, 6, 7, 8],
        split="validation")
    assert choose_fixed_shape(first + second) == (2, 4)

    all_tied = _rows("validation-tied", split="validation")
    assert choose_fixed_shape(all_tied) == SHAPE_ROSTER[0]
    with pytest.raises(ValueError, match="validation rows only"):
        choose_fixed_shape(_rows("test-dfg", split="test"))

    incomplete = _rows(
        "validation-incomplete", truths=[None, 1, 1, 1, 1, 1, 1, 1],
        statuses=["censored", "success", "success", "success", "success",
                  "success", "success", "success"],
        split="validation")
    with pytest.raises(ValueError, match="eight successful numeric labels"):
        choose_fixed_shape(incomplete)


def test_duplicate_or_missing_roster_rows_fail_closed():
    complete = _rows("dfg-a")
    with pytest.raises(ValueError, match="duplicate row"):
        evaluate_predictions(complete + [complete[0]], "model")
    with pytest.raises(ValueError, match="exactly the fixed shape roster"):
        evaluate_predictions(complete[:-1], "model")
