import unittest
import math

from cgra_ii_predictor.dataset import Sample
from cgra_ii_predictor.model import (
    calibrate_unseen_group_interval,
    fit_ridge,
    group_balanced_sample_weights,
    group_ranking_metrics,
    nested_group_holdout,
    predict_compiled_ii,
    prediction_rows,
    predict_ridge,
    raw_residual_metrics,
    select_ridge_hyperparameters,
)


def contract(lower_bound, **metadata):
    return {
        "rec_mii": lower_bound,
        "res_mii": 1,
        "lower_bound_source": "rec_res_max_v1",
        **metadata,
    }


def sample(name, group, lower_bound, compiled_ii, pressure):
    return Sample(
        sample_id=name,
        group=group,
        lower_bound=lower_bound,
        compiled_ii=compiled_ii,
        features={"pressure": pressure, "depth": pressure + 1},
        metadata=contract(lower_bound),
    )


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            sample("a0", "a", 3, 3, 1),
            sample("a1", "a", 3, 3, 1.2),
            sample("b0", "b", 4, 6, 3),
            sample("b1", "b", 4, 6, 3.2),
            sample("c0", "c", 5, 8, 5),
            sample("c1", "c", 5, 8, 5.2),
        ]

    def test_prediction_never_falls_below_bound(self):
        model = fit_ridge(self.samples, ["pressure", "depth"], ridge=1.0)
        self.assertEqual(
            model["training_weighting"],
            "equal_total_weight_per_group_and_distinct_observation",
        )
        self.assertEqual(model["training_weight_sum"], 3.0)
        self.assertEqual(set(model["training_feature_support"]), {
            "pressure", "depth",
        })
        self.assertEqual(model["training_feature_support"]["pressure"]["minimum"], 1.0)
        self.assertEqual(model["training_feature_support"]["pressure"]["maximum"], 5.2)
        self.assertEqual(model["training_design_rank"], 2)
        self.assertEqual(model["training_design_column_count"], 3)
        for row in self.samples:
            self.assertGreaterEqual(predict_ridge(model, row), row.lower_bound)
        with self.assertRaisesRegex(ValueError, "must equal max"):
            predict_compiled_ii(
                model, 5, self.samples[0].features,
                rec_mii=3, res_mii=4,
            )

    def test_direct_sample_api_rejects_invalid_values_and_scoped_conflicts(self):
        invalid = Sample(
            "bad", "g", 1, 2, {"pressure": math.inf}, contract(1),
        )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            fit_ridge([invalid], ["pressure"], ridge=1.0)
        label_feature = Sample(
            "label-feature", "g", 1, 2, {"compiled_ii": 2}, contract(1),
        )
        with self.assertRaisesRegex(ValueError, "label"):
            fit_ridge([label_feature], ["compiled_ii"], ridge=1.0)
        bound_feature = Sample(
            "bound-feature", "g", 1, 2,
            {"pressure": 1, "rec_mii": 1}, contract(1),
        )
        with self.assertRaisesRegex(ValueError, "lower-bound"):
            fit_ridge([bound_feature], ["pressure"], ridge=1.0)
        missing_contract = Sample(
            "missing-contract", "g", 1, 2, {"pressure": 1}, {},
        )
        with self.assertRaisesRegex(ValueError, "metadata.rec_mii"):
            fit_ridge([missing_contract], ["pressure"], ridge=1.0)
        loose_bound = Sample(
            "loose-bound", "g", 3, 4, {"pressure": 1}, contract(2),
        )
        with self.assertRaisesRegex(ValueError, "must equal max"):
            fit_ridge([loose_bound], ["pressure"], ridge=1.0)
        first = Sample(
            "a", "g", 1, 2, {"pressure": 1},
            contract(1, ranking_query_id="dfg", candidate_id="mesh"),
        )
        second = Sample(
            "b", "g", 1, 3, {"pressure": 1},
            contract(1, ranking_query_id="dfg", candidate_id="mesh"),
        )
        with self.assertRaisesRegex(ValueError, "conflicting records"):
            fit_ridge([first, second], ["pressure"], ridge=1.0)
        inconsistent = Sample(
            "bad-query", "g", 1, 2, {"pressure": 1}, contract(
                1, base_dfg_id="dfg-a", ranking_query_id="dfg-b",
                candidate_id="mesh",
            ),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent base_dfg"):
            fit_ridge([inconsistent], ["pressure"], ridge=1.0)

    def test_nested_holdout_rejects_query_identity_spanning_groups(self):
        # Candidate names are intentionally different: query ownership is a
        # separate leakage boundary and cannot be inferred from candidate IDs.
        rows = [
            Sample(
                "query-a", "lineage-a", 1, 1, {"pressure": 1},
                contract(
                    1, ranking_query_id="shared-dfg", candidate_id="mesh-a",
                ),
            ),
            Sample(
                "query-b", "lineage-b", 1, 1, {"pressure": 2},
                contract(
                    1, base_dfg_id="shared-dfg", candidate_id="mesh-b",
                ),
            ),
            sample("query-c", "lineage-c", 1, 1, 3),
        ]
        with self.assertRaisesRegex(ValueError, "multiple sample.group"):
            nested_group_holdout(
                rows, ["pressure"],
                ridge_candidates=[1.0], dead_zone_candidates=[0.0],
            )

    def test_ridge_controls_reject_non_finite_and_invalid_values(self):
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(ridge=value), self.assertRaises(ValueError):
                fit_ridge(self.samples, ["pressure"], ridge=value)
        for value in (-1.0, math.nan, math.inf, -math.inf, True, "0"):
            with self.subTest(dead_zone=value), self.assertRaises(ValueError):
                fit_ridge(
                    self.samples, ["pressure"], ridge=1.0,
                    residual_dead_zone=value,
                )

    def test_ridge_selection_rejects_empty_or_invalid_grids(self):
        with self.assertRaisesRegex(ValueError, "ridge candidate grid must be"):
            select_ridge_hyperparameters(
                self.samples, ["pressure"], [], [0.0],
            )
        with self.assertRaisesRegex(ValueError, "residual dead zone candidate grid"):
            select_ridge_hyperparameters(
                self.samples, ["pressure"], [1.0], [],
            )
        with self.assertRaisesRegex(ValueError, "ridge candidate must be finite"):
            select_ridge_hyperparameters(
                self.samples, ["pressure"], [math.nan], [0.0],
            )
        with self.assertRaisesRegex(ValueError, "residual dead zone candidate must be finite"):
            select_ridge_hyperparameters(
                self.samples, ["pressure"], [1.0], [math.inf],
            )

    def test_ridge_fit_reports_extreme_statistics_overflow(self):
        rows = [
            Sample(
                "extreme-a", "extreme-a", 1, 1, {"pressure": -1e308},
                contract(1),
            ),
            Sample(
                "extreme-b", "extreme-b", 1, 1, {"pressure": 1e308},
                contract(1),
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overflowed|finite"):
            fit_ridge(rows, ["pressure"], ridge=1.0)

    def test_direct_sample_api_allows_zero_rec_or_res_but_not_zero_floor(self):
        for rec_mii, res_mii in ((0, 3), (3, 0)):
            row = Sample(
                f"zero-{rec_mii}-{res_mii}", "g", 3, 3,
                {"pressure": 1.0}, {
                    "rec_mii": rec_mii, "res_mii": res_mii,
                    "lower_bound_source": "rec_res_max_v1",
                },
            )
            with self.subTest(rec_mii=rec_mii, res_mii=res_mii):
                fitted = fit_ridge([row], ["pressure"], ridge=1.0)
                self.assertGreaterEqual(predict_ridge(fitted, row), 3.0)

        zero_floor = Sample(
            "zero-floor", "g", 0, 1, {"pressure": 1.0}, {
                "rec_mii": 0, "res_mii": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fit_ridge([zero_floor], ["pressure"], ridge=1.0)

        negative_component = Sample(
            "negative-component", "g", 3, 3, {"pressure": 1.0}, {
                "rec_mii": -1, "res_mii": 3,
            },
        )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            fit_ridge([negative_component], ["pressure"], ridge=1.0)

    def test_predict_compiled_ii_allows_zero_rec_or_res(self):
        fitted = fit_ridge([
            Sample(
                "fit", "g", 3, 3, {"pressure": 1.0},
                {"rec_mii": 0, "res_mii": 3},
            ),
        ], ["pressure"], ridge=1.0)
        for rec_mii, res_mii in ((0, 3), (3, 0)):
            with self.subTest(rec_mii=rec_mii, res_mii=res_mii):
                prediction = predict_compiled_ii(
                    fitted, 3, {"pressure": 1.0},
                    rec_mii=rec_mii, res_mii=res_mii,
                )
                self.assertGreaterEqual(prediction, 3.0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            predict_compiled_ii(
                fitted, 0, {"pressure": 1.0}, rec_mii=0, res_mii=0,
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            predict_compiled_ii(
                fitted, 3, {"pressure": 1.0}, rec_mii=-1, res_mii=3,
            )

    def test_each_group_gets_equal_total_training_weight(self):
        # This is deliberately nontrivial: a feature has predictive power, so
        # changing the effective Ridge penalty would move the prediction.
        original = [
            sample("a0", "a", 1, 1, 0),
            sample("b0", "b", 1, 5, 4),
        ]
        duplicated = original + [sample(f"a{i}", "a", 1, 1, 0)
                                 for i in range(1, 10)]
        first = fit_ridge(original, ["pressure"], ridge=1.0)
        second = fit_ridge(duplicated, ["pressure"], ridge=1.0)
        self.assertEqual(first["training_distinct_observation_count"], 2)
        self.assertEqual(second["training_distinct_observation_count"], 2)
        probe = sample("p", "probe", 1, 1, 2)
        self.assertAlmostEqual(
            predict_ridge(first, probe), predict_ridge(second, probe)
        )
        for first_weight, second_weight in zip(
            first["weights"], second["weights"]
        ):
            self.assertAlmostEqual(first_weight, second_weight)

    def test_unselected_features_do_not_create_extra_training_weight(self):
        rows = [
            Sample("a0", "a", 1, 2,
                   {"pressure": 1, "architecture_detail": 0}, contract(1)),
            Sample("a1", "a", 1, 2,
                   {"pressure": 1, "architecture_detail": 1}, contract(1)),
            Sample("b0", "b", 1, 4,
                   {"pressure": 4, "architecture_detail": 0}, contract(1)),
        ]
        model = fit_ridge(rows, ["pressure"], ridge=1.0)
        self.assertEqual(model["training_distinct_observation_count"], 2)

    def test_training_strata_prevent_generated_lineages_from_swamping_real(self):
        real = Sample(
            "real", "real-lineage", 1, 5, {"pressure": 0},
            contract(1, training_stratum="real"),
        )
        generated = [
            Sample(
                f"generated-{index}", f"generated-{index}", 1, 1,
                {"pressure": 0}, contract(1, training_stratum="generated"),
            )
            for index in range(20)
        ]
        model = fit_ridge([real] + generated, ["pressure"], ridge=1.0)
        self.assertEqual(model["training_stratum_count"], 2)
        self.assertAlmostEqual(model["training_weight_sum"], 21.0)
        # Equal real/generated stratum mass makes the intercept residual 2,
        # instead of letting twenty generated lineages pull it near zero.
        self.assertAlmostEqual(model["weights"][0], 2.0)

    def test_single_training_stratum_preserves_legacy_ridge_scale(self):
        rows = [
            sample("a", "a", 1, 1, 0),
            sample("b", "b", 1, 3, 2),
            sample("c", "c", 1, 6, 4),
        ]
        stratified = [Sample(
            row.sample_id, row.group, row.lower_bound, row.compiled_ii,
            row.features, contract(row.lower_bound, training_stratum="real"),
        ) for row in rows]
        legacy_model = fit_ridge(rows, ["pressure"], ridge=1.0)
        stratum_model = fit_ridge(stratified, ["pressure"], ridge=1.0)
        self.assertAlmostEqual(stratum_model["training_weight_sum"], 3.0)
        for legacy, declared in zip(
            legacy_model["weights"], stratum_model["weights"]
        ):
            self.assertAlmostEqual(legacy, declared)

    def test_stratum_weights_keep_total_group_mass(self):
        rows = [
            Sample("real", "real", 1, 2, {"pressure": 0},
                   contract(1, training_stratum="real")),
            Sample("g0", "g0", 1, 2, {"pressure": 0},
                   contract(1, training_stratum="generated")),
            Sample("g1", "g1", 1, 2, {"pressure": 0},
                   contract(1, training_stratum="generated")),
        ]
        weights = group_balanced_sample_weights(rows, ["pressure"])
        self.assertAlmostEqual(float(weights.sum()), 3.0)
        self.assertAlmostEqual(float(weights[0]), 1.5)
        self.assertAlmostEqual(float(weights[1] + weights[2]), 1.5)

    def test_different_ranking_queries_are_distinct_observations(self):
        rows = [
            Sample("a", "lineage", 1, 2, {"pressure": 1}, {
                "ranking_query_id": "dfg-a", "candidate_id": "mesh",
                **contract(1),
            }),
            Sample("b", "lineage", 1, 2, {"pressure": 1}, {
                "ranking_query_id": "dfg-b", "candidate_id": "mesh",
                **contract(1),
            }),
        ]
        model = fit_ridge(rows, ["pressure"], ridge=1.0)
        self.assertEqual(model["training_distinct_observation_count"], 2)

    def test_duplicate_rows_do_not_change_hyperparameter_selection(self):
        original = [
            sample("a0", "a", 1, 1, 0),
            sample("b0", "b", 1, 2, 2),
            sample("c0", "c", 1, 5, 4),
        ]
        duplicated = original + [sample(f"a{i}", "a", 1, 1, 0)
                                 for i in range(1, 10)]
        arguments = (["pressure"], [0.1, 1.0, 10.0], [0.0, 1.0])
        self.assertEqual(
            select_ridge_hyperparameters(original, *arguments),
            select_ridge_hyperparameters(duplicated, *arguments),
        )

    def test_nested_holdout_keeps_groups_intact(self):
        result = nested_group_holdout(
            self.samples,
            ["pressure", "depth"],
            ridge_candidates=[0.3, 1.0, 3.0],
            dead_zone_candidates=[0.0, 0.5, 1.0],
        )
        self.assertEqual(result["groups"], ["a", "b", "c"])
        self.assertEqual(len(result["rows"]), len(self.samples))

    def test_interval_uses_one_max_error_per_group_and_respects_bound(self):
        model = fit_ridge(self.samples, ["pressure", "depth"], ridge=1.0)
        held_out = [
            {"group": "a", "prediction": 4.0, "compiled_ii": 3.0},
            {"group": "a", "prediction": 8.0, "compiled_ii": 5.0},
            {"group": "b", "prediction": 5.0, "compiled_ii": 7.0},
        ]
        calibrated = calibrate_unseen_group_interval(
            model, held_out, quantile=0.5
        )
        self.assertEqual(calibrated["unseen_group_absolute_error_radius"], 2.0)
        self.assertEqual(
            calibrated["unseen_group_interval_empirical_quantile"], 0.5
        )
        row = prediction_rows(calibrated, [self.samples[0]])[0]
        self.assertGreaterEqual(
            row["prediction_interval_lower"], row["lower_bound"]
        )

    def test_group_ranking_excludes_groups_without_target_variation(self):
        rows = [
            {"group": "lineage", "ranking_query_id": "a",
             "candidate_id": "a0", "prediction": 1.0, "compiled_ii": 1.0},
            {"group": "lineage", "ranking_query_id": "a",
             "candidate_id": "a1", "prediction": 3.0, "compiled_ii": 3.0},
            {"group": "lineage", "ranking_query_id": "b",
             "candidate_id": "b0", "prediction": 7.0, "compiled_ii": 5.0},
            {"group": "lineage", "ranking_query_id": "b",
             "candidate_id": "b1", "prediction": 8.0, "compiled_ii": 5.0},
        ]
        metrics = group_ranking_metrics(rows, "prediction")
        self.assertEqual(metrics["comparable_pairs"], 1)
        self.assertEqual(metrics["groups_with_ranking_signal"], 1)
        self.assertEqual(metrics["strict_pairwise_accuracy"], 1.0)
        self.assertEqual(metrics["eligible_group_count"], 1)
        self.assertEqual(metrics["excluded_group_counts"]["constant_target"], 1)

    def test_group_ranking_reports_macro_micro_and_prediction_ties(self):
        rows = [
            {"group": "lineage", "ranking_query_id": "a",
             "candidate_id": "a0", "prediction": 1.0, "compiled_ii": 1.0},
            {"group": "lineage", "ranking_query_id": "a",
             "candidate_id": "a1", "prediction": 2.0, "compiled_ii": 2.0},
            {"group": "lineage", "ranking_query_id": "b",
             "candidate_id": "b0", "prediction": 4.0, "compiled_ii": 1.0},
            {"group": "lineage", "ranking_query_id": "b",
             "candidate_id": "b1", "prediction": 4.0, "compiled_ii": 2.0},
            {"group": "lineage", "ranking_query_id": "b",
             "candidate_id": "b2", "prediction": 4.0, "compiled_ii": 3.0},
            {"group": "lineage", "ranking_query_id": "c",
             "candidate_id": "c0", "prediction": 4.0, "compiled_ii": 4.0},
        ]
        metrics = group_ranking_metrics(rows, "prediction")
        self.assertEqual(metrics["comparable_pairs"], 4)
        self.assertEqual(metrics["eligible_group_count"], 2)
        self.assertAlmostEqual(metrics["micro_pairwise_concordance"], 0.625)
        self.assertAlmostEqual(metrics["macro_pairwise_concordance"], 0.75)
        self.assertAlmostEqual(metrics["predicted_tie_rate"], 0.75)
        self.assertEqual(metrics["groups"]["c"]["status"], "single_sample")

    def test_group_ranking_collapses_identical_candidate_ids(self):
        rows = [
            {
                "group": "lineage", "ranking_query_id": "a",
                "candidate_id": "x",
                "prediction": 1.0, "compiled_ii": 1.0,
            },
            {
                "group": "lineage", "ranking_query_id": "a",
                "candidate_id": "x",
                "prediction": 1.0, "compiled_ii": 1.0,
            },
            {
                "group": "lineage", "ranking_query_id": "a",
                "candidate_id": "y",
                "prediction": 2.0, "compiled_ii": 2.0,
            },
        ]
        metrics = group_ranking_metrics(rows, "prediction")
        self.assertEqual(metrics["comparable_pairs"], 1)
        self.assertEqual(metrics["duplicate_candidate_rows_collapsed"], 1)
        self.assertEqual(metrics["candidate_identity_status"], "complete")

    def test_ranking_never_compares_different_dfgs_in_one_leakage_lineage(self):
        rows = [
            {"group": "merged-lineage", "ranking_query_id": "dfg-a",
             "candidate_id": "a", "prediction": 1.0, "compiled_ii": 1.0},
            {"group": "merged-lineage", "ranking_query_id": "dfg-b",
             "candidate_id": "b", "prediction": 3.0, "compiled_ii": 3.0},
        ]
        metrics = group_ranking_metrics(rows, "prediction")
        self.assertEqual(metrics["status"], "no_eligible_queries")
        self.assertEqual(metrics["comparable_pairs"], 0)
        self.assertEqual(metrics["total_ranking_query_count"], 2)

    def test_ranking_query_spanning_leakage_groups_is_excluded(self):
        rows = [
            {"group": "lineage-a", "ranking_query_id": "same-dfg",
             "candidate_id": "a", "prediction": 1.0, "compiled_ii": 1.0},
            {"group": "lineage-b", "ranking_query_id": "same-dfg",
             "candidate_id": "b", "prediction": 2.0, "compiled_ii": 2.0},
        ]
        metrics = group_ranking_metrics(rows, "prediction")
        self.assertEqual(metrics["status"], "invalid_cross_leakage_query")
        self.assertEqual(metrics["comparable_pairs"], 0)
        self.assertEqual(
            metrics["excluded_group_counts"][
                "ranking_query_spans_multiple_leakage_groups"
            ],
            1,
        )

    def test_large_corpus_uses_deterministic_group_k_fold(self):
        rows = [
            sample(str(index), f"g{index:02d}", 1, 1 + index % 4, index)
            for index in range(21)
        ]
        result = nested_group_holdout(
            rows, ["pressure"], ridge_candidates=[1.0],
            dead_zone_candidates=[0.0],
        )
        self.assertEqual(
            result["outer_split_protocol"],
            "deterministic_stratified_group_k_fold",
        )
        self.assertEqual(result["outer_fold_count"], 10)
        held_out = [
            group for fold in result["outer_held_out_groups_by_fold"]
            for group in fold
        ]
        self.assertEqual(sorted(held_out), result["groups"])
        self.assertEqual(len(result["rows"]), len(rows))

    def test_nested_metrics_are_invariant_to_duplicate_held_out_rows(self):
        original = [
            sample("a0", "a", 1, 1, 0),
            sample("b0", "b", 1, 2, 2),
            sample("c0", "c", 1, 5, 4),
        ]
        duplicated = original + [sample("a-copy", "a", 1, 1, 0)]
        arguments = (["pressure"], [0.1, 1.0], [0.0, 1.0])
        first = nested_group_holdout(original, *arguments)
        second = nested_group_holdout(duplicated, *arguments)
        self.assertEqual(first["model_metrics"], second["model_metrics"])
        self.assertEqual(second["evaluation_duplicate_rows_collapsed"], 1)

    def test_nested_holdout_reports_raw_residual_diagnostics(self):
        result = nested_group_holdout(
            self.samples,
            ["pressure", "depth"],
            ridge_candidates=[1.0],
            dead_zone_candidates=[0.0],
        )
        self.assertIn("raw_residual_metrics", result)
        self.assertEqual(
            result["raw_residual_metrics"],
            raw_residual_metrics(result["rows"]),
        )


if __name__ == "__main__":
    unittest.main()
