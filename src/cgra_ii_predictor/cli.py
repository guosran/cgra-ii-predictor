"""Command-line evaluation for portable compiled-II datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .dataset import load_dataset
from .model import fit_calibrated_model, nested_group_holdout


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
    parser.add_argument("--ridge-candidates", default="0.1,0.3,1,3,10,30")
    parser.add_argument("--dead-zone-candidates", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    names = args.features.split(",") if args.features else None
    dataset = load_dataset(args.dataset, names)
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
    report = {
        "dataset": str(args.dataset),
        "sample_count": len(dataset.samples),
        "group_count": len({sample.group for sample in dataset.samples}),
        "feature_names": list(dataset.feature_names),
        "nested_group_holdout": holdout,
        "trained_full_model": model,
        "status": "offline_experiment_only",
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

