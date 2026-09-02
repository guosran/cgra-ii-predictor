"""Command-line evaluation for portable compiled-II datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

from .dataset import load_dataset, regroup_by_metadata, remap_groups
from .model import (
    calibrate_unseen_group_interval,
    fit_calibrated_model,
    nested_group_holdout,
)


def _numbers(value: str, allow_zero: bool) -> List[float]:
    result = sorted({float(item) for item in value.split(",") if item})
    if not result or any(number < 0.0 or (number == 0.0 and not allow_zero)
                         for number in result):
        raise argparse.ArgumentTypeError("invalid numeric candidate list")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a compiled-II predictor with nested group holdout"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--features", help="Comma-separated feature override")
    parser.add_argument(
        "--group-alias", action="append", default=[], metavar="OLD=LINEAGE",
        help=("Conservatively merge related source groups for leakage-safe "
              "evaluation; repeat as needed"),
    )
    parser.add_argument(
        "--holdout-metadata-key", action="append", default=[],
        metavar="KEY",
        help=("Also run nested holdout by a metadata identity such as "
              "architecture_id; source-lineage training weights are retained. "
              "This can be expensive for many distinct identities"),
    )
    parser.add_argument("--ridge-candidates", default="0.1,0.3,1,3,10,30")
    parser.add_argument("--dead-zone-candidates", default="0,0.25,0.5,0.75,1,1.5,2")
    interval = parser.add_mutually_exclusive_group()
    interval.add_argument(
        "--interval-quantile", type=float, default=0.9,
        help=("Empirical quantile of held-out group maxima; this is not a "
              "formal coverage guarantee"),
    )
    interval.add_argument(
        "--interval-coverage", type=float, dest="legacy_interval_coverage",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    interval_quantile = (
        args.legacy_interval_coverage
        if args.legacy_interval_coverage is not None
        else args.interval_quantile
    )
    if not 0.0 < interval_quantile <= 1.0:
        parser.error("--interval-quantile must be in (0, 1]")

    aliases = {}
    for value in args.group_alias:
        source, separator, lineage = value.partition("=")
        if not separator or not source or not lineage:
            parser.error(f"invalid --group-alias OLD=LINEAGE: {value}")
        if source in aliases and aliases[source] != lineage:
            parser.error(f"conflicting --group-alias entries for {source}")
        aliases[source] = lineage

    names = args.features.split(",") if args.features else None
    dataset = remap_groups(load_dataset(args.dataset, names), aliases)
    ridge_candidates = _numbers(args.ridge_candidates, allow_zero=False)
    dead_zone_candidates = _numbers(args.dead_zone_candidates, allow_zero=True)
    holdout = nested_group_holdout(
        dataset.samples, dataset.feature_names,
        ridge_candidates, dead_zone_candidates,
    )
    model = fit_calibrated_model(
        dataset.samples, dataset.feature_names,
        ridge_candidates, dead_zone_candidates,
    )
    model = calibrate_unseen_group_interval(
        model, holdout["rows"], interval_quantile
    )
    metadata_holdouts = {}
    for metadata_key in dict.fromkeys(args.holdout_metadata_key):
        try:
            regrouped = regroup_by_metadata(dataset, metadata_key)
            metadata_holdouts[metadata_key] = nested_group_holdout(
                regrouped.samples, regrouped.feature_names,
                ridge_candidates, dead_zone_candidates,
            )
        except ValueError as error:
            parser.error(f"--holdout-metadata-key {metadata_key}: {error}")
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    implementation_sha256 = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("cli.py", "dataset.py", "model.py")
    }
    model_sha256 = hashlib.sha256(json.dumps(
        model, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    report = {
        "schema_version": "portable-model-report-v2",
        "target": "compiled_ii_from_the_recorded_mapper",
        "lower_bound_contract": {
            "name": "rec_res_max_v1",
            "formula": "max(rec_mii,res_mii)",
            "training_lower_bound_sources": ["rec_res_max_v1"],
            "components_are_model_features": False,
        },
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "dataset_provenance": dataset.provenance,
        "implementation_sha256": implementation_sha256,
        "sample_count": len(dataset.samples),
        "group_count": len({sample.group for sample in dataset.samples}),
        "group_aliases": aliases,
        "grouping": {
            "primary_group_key": "group",
            "effective_leakage_lineage_key": "metadata.leakage_lineage_id",
            "declared_lineage_key": "metadata.lineage",
            "aliases": aliases,
            "additional_holdout_metadata_keys": list(metadata_holdouts),
        },
        "feature_names": list(dataset.feature_names),
        "nested_group_holdout": holdout,
        "nested_metadata_holdouts": metadata_holdouts,
        "trained_full_model": model,
        "trained_full_model_sha256": model_sha256,
        "status": "offline_experiment_only",
        "report_metadata": {
            "evaluation_status": "exploratory",
            "dataset_role": "labeled_train_validation",
            "holdout_protocol": "nested_group_holdout",
            "labels_available_at_evaluation": True,
            "feature_set_status": "provided_by_dataset_or_cli",
            "model_class_status": "predeclared_residual_ridge",
            "model_selection_status": "selected_on_this_labeled_dataset",
            "frozen_test": False,
            "blind": False,
        },
        "evidence_scope": (
            "exploratory_model_development_not_a_frozen_evaluation"
        ),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
