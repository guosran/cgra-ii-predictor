#!/usr/bin/env python3
"""Fit a validation-only conservative II estimate from a second checkpoint.

The deployed ensemble remains the conditional-mean predictor.  This tool fits
an optional upper estimate for latency-sensitive program scoring:

``max(ensemble_mean, expert_mean + alpha * expert_std)``

``alpha`` is selected on validation only with pinball loss.  The test split is
evaluated once after selection and never participates in fitting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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

from ensemble_pointwise_checkpoints import (  # noqa: E402
    _successful_matrix,
    _successful_uncertainties,
    apply_uncertainty_gating,
    predict_checkpoint,
    sha256_file,
)


Prediction = Dict[str, Any]


def pinball_loss(
    predictions: np.ndarray, targets: np.ndarray, quantile: float,
) -> float:
    """Return quantile loss, with underprediction weighted by ``quantile``."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0.5 and 1")
    residual = targets - predictions
    return float(np.mean(np.maximum(
        quantile * residual, (quantile - 1.0) * residual,
    )))


def point_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    errors = predictions - targets
    under = np.maximum(targets - predictions, 0.0)
    return {
        "candidate_count": int(len(targets)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_signed_error": float(np.mean(errors)),
        "underprediction_rate": float(np.mean(errors < 0.0)),
        "mean_underprediction": float(np.mean(under)),
        "maximum_underprediction": float(np.max(under)),
    }


def ensemble_arrays(
    predictions: Mapping[str, Mapping[str, Prediction]],
    ensemble_report: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, list[Prediction]]:
    names, matrix, targets, rows = _successful_matrix(predictions)
    expected_names = list(ensemble_report["weights"])
    if names != expected_names:
        raise ValueError("prediction and ensemble predictor order differ")
    weights = np.asarray([
        float(ensemble_report["weights"][name]) for name in names
    ], dtype=np.float64)
    uncertainty_names, uncertainties = _successful_uncertainties(predictions)
    if uncertainty_names != names:
        raise ValueError("prediction and uncertainty order differ")
    gating = ensemble_report["uncertainty_gating"]
    scales = np.asarray([
        float(gating["scales"][name]) for name in names
    ], dtype=np.float64)
    mode = str(ensemble_report["selected_ensemble_mode"])
    if mode == "static":
        result = matrix @ weights
    elif mode == "uncertainty_gated":
        result = apply_uncertainty_gating(
            matrix, uncertainties, weights, float(gating["exponent"]), scales,
        )
    else:
        raise ValueError("unknown selected ensemble mode")
    return result, targets, rows


def aligned_expert_arrays(
    expert: Mapping[str, Prediction], rows: Sequence[Prediction],
) -> Tuple[np.ndarray, np.ndarray]:
    ordered = []
    for row in rows:
        identity = str(row["candidate_id"])
        if identity not in expert:
            raise ValueError("expert candidate population differs")
        record = expert[identity]
        if record.get("compiled_ii") is None:
            raise ValueError("expert success population differs")
        ordered.append(record)
    return (
        np.asarray([float(row["predicted_ii"]) for row in ordered]),
        np.asarray([float(row["predicted_std"]) for row in ordered]),
    )


def fit_alpha(
    means: np.ndarray, targets: np.ndarray,
    expert_means: np.ndarray, expert_stds: np.ndarray,
    quantile: float, alpha_grid: Sequence[float],
) -> Tuple[float, list[Dict[str, Any]]]:
    if not alpha_grid:
        raise ValueError("alpha grid must not be empty")
    records = []
    for alpha in alpha_grid:
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha values must be finite and nonnegative")
        conservative = np.maximum(means, expert_means + alpha * expert_stds)
        records.append({
            "alpha": float(alpha),
            "validation_pinball_loss": pinball_loss(
                conservative, targets, quantile,
            ),
            "validation": point_metrics(conservative, targets),
        })
    selected = min(records, key=lambda row: (
        float(row["validation_pinball_loss"]),
        float(row["validation"]["mae"]),
        float(row["alpha"]),
    ))
    return float(selected["alpha"]), records


def parse_alpha_grid(value: str) -> Tuple[float, ...]:
    result = tuple(float(part) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("alpha grid must not be empty")
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            "alpha grid values must be finite and nonnegative"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ensemble-report", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Ensemble checkpoint as NAME=PATH, in report order.",
    )
    parser.add_argument("--expert-checkpoint", required=True, type=Path)
    parser.add_argument("--expert-name", default="structural_expert")
    parser.add_argument("--quantile", default=0.75, type=float)
    parser.add_argument(
        "--alpha-grid", type=parse_alpha_grid,
        default=parse_alpha_grid("0,0.25,0.5,0.75,1,1.25,1.5,2"),
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument(
        "--seed", type=int,
        help="Split seed; defaults to and must match the ensemble report.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ensemble_report = json.loads(args.ensemble_report.read_text())
    if ensemble_report.get("schema_version") != (
        "cgra-ii-pointwise-convex-ensemble-v1"
    ):
        raise ValueError("ensemble report schema_version mismatch")
    if ensemble_report.get("selection_split") != "validation_only":
        raise ValueError("ensemble must be selected on validation only")
    if ensemble_report.get("manifest_sha256") != sha256_file(
        args.manifest.resolve()
    ):
        raise ValueError("ensemble report manifest SHA-256 mismatch")
    report_seed = int(ensemble_report["seed"])
    seed = report_seed if args.seed is None else args.seed
    if seed != report_seed:
        raise ValueError("split seed must match the ensemble report")
    parsed = []
    for value in args.checkpoint:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("checkpoint must be NAME=PATH")
        parsed.append((name, Path(raw_path)))
    report_names = [
        name for name in ensemble_report["weights"]
        if name != "analytical_lower_bound"
    ]
    if [name for name, _ in parsed] != report_names:
        raise ValueError("checkpoints must follow ensemble report order")
    report_checkpoints = ensemble_report.get("checkpoints")
    if not isinstance(report_checkpoints, Mapping):
        raise ValueError("ensemble report lacks checkpoint provenance")
    for name, path in parsed:
        record = report_checkpoints.get(name)
        if not isinstance(record, Mapping) or record.get("sha256") != (
            sha256_file(path.resolve())
        ):
            raise ValueError(f"ensemble checkpoint SHA-256 mismatch for {name}")

    device = torch.device(args.device)
    by_model: Dict[str, Dict[str, Dict[str, Prediction]]] = {}
    checkpoint_metadata = {}
    for name, path in parsed:
        _, splits = predict_checkpoint(
            args.manifest, path, ("validation", "test"),
            args.batch_size, seed, device,
        )
        by_model[name] = splits
        checkpoint_metadata[name] = {
            "path": str(path.resolve()), "sha256": sha256_file(path.resolve()),
        }
    _, expert = predict_checkpoint(
        args.manifest, args.expert_checkpoint,
        ("validation", "test"), args.batch_size, seed, device,
    )

    split_arrays = {}
    for split in ("validation", "test"):
        base, targets, rows = ensemble_arrays({
            name: by_model[name][split] for name in report_names
        }, ensemble_report)
        expert_means, expert_stds = aligned_expert_arrays(expert[split], rows)
        split_arrays[split] = (
            base, targets, rows, expert_means, expert_stds,
        )
    validation = split_arrays["validation"]
    selected_alpha, grid = fit_alpha(
        validation[0], validation[1], validation[3], validation[4],
        args.quantile, args.alpha_grid,
    )

    evaluations = {}
    for split, (base, targets, _, expert_means, expert_stds) in (
        split_arrays.items()
    ):
        conservative = np.maximum(
            base, expert_means + selected_alpha * expert_stds,
        )
        evaluations[split] = {
            "ensemble_mean": point_metrics(base, targets),
            "conservative": point_metrics(conservative, targets),
            "ensemble_mean_pinball_loss": pinball_loss(
                base, targets, args.quantile,
            ),
            "conservative_pinball_loss": pinball_loss(
                conservative, targets, args.quantile,
            ),
        }

    output = {
        "schema_version": "cgra-ii-conservative-pointwise-v1",
        "selection_split": "validation_only",
        "selection_metric": f"pinball_loss_q{args.quantile:g}",
        "seed": seed,
        "policy": "max_ensemble_mean_and_expert_mean_plus_alpha_std",
        "quantile": args.quantile,
        "selected_alpha": selected_alpha,
        "alpha_grid": grid,
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "ensemble_report_sha256": sha256_file(args.ensemble_report.resolve()),
        "checkpoints": checkpoint_metadata,
        "expert": {
            "name": args.expert_name,
            "path": str(args.expert_checkpoint.resolve()),
            "sha256": sha256_file(args.expert_checkpoint.resolve()),
        },
        "evaluation": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        output, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    print(json.dumps({
        "selected_alpha": selected_alpha,
        "validation": evaluations["validation"],
        "test": evaluations["test"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
