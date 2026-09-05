#!/usr/bin/env python3
"""Compare legacy and v8 ensembles on the same held-out mapper 4x4 labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence


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
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    AMOEBA_STATIC_SHAPE_PROTOCOL,
)
from ensemble_pointwise_checkpoints import _metrics, _success_metrics  # noqa: E402
from neura_graph_experiment import (  # noqa: E402
    batch_targets,
    load_terminal_manifest,
    query_batches,
    split_queries,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def load_ensemble(path: Path) -> Dict[str, Any]:
    report = _object(json.loads(path.read_text()), "ensemble report")
    if report.get("selection_split") != "validation_only":
        raise ValueError("ensemble was not selected using validation only")
    source_manifest = Path(str(report.get("manifest"))).resolve()
    if not source_manifest.is_file():
        raise ValueError(f"ensemble source manifest does not exist: {source_manifest}")
    if sha256_file(source_manifest) != report.get("manifest_sha256"):
        raise ValueError("ensemble source manifest SHA-256 mismatch")
    checkpoints = _object(report.get("checkpoints"), "checkpoints")
    resolved = {}
    for name, raw_record in checkpoints.items():
        record = _object(raw_record, f"checkpoint {name}")
        checkpoint_path = Path(str(record.get("path"))).resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
        if sha256_file(checkpoint_path) != record.get("sha256"):
            raise ValueError(f"checkpoint SHA-256 mismatch for {name}")
        resolved[str(name)] = checkpoint_path
    weights = {
        str(name): float(value)
        for name, value in _object(report.get("weights"), "weights").items()
    }
    if set(weights) != {"analytical_lower_bound", *resolved}:
        raise ValueError("ensemble weights do not match checkpoints")
    if any(value < 0.0 or not math.isfinite(value) for value in weights.values()):
        raise ValueError("ensemble weights are invalid")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("ensemble weights do not sum to one")
    gating = _object(report.get("uncertainty_gating"), "uncertainty gating")
    scales = {
        str(name): float(value)
        for name, value in _object(gating.get("scales"), "scales").items()
    }
    if set(scales) != set(weights) or any(
        value <= 0.0 or not math.isfinite(value) for value in scales.values()
    ):
        raise ValueError("uncertainty scales are invalid")
    exponent = float(gating.get("exponent"))
    if exponent < 0.0 or not math.isfinite(exponent):
        raise ValueError("uncertainty exponent is invalid")
    selected_mode = report.get("selected_ensemble_mode")
    if selected_mode not in {"static", "uncertainty_gated"}:
        validation = _object(report.get("validation"), "validation")
        gated = _object(validation.get("ensemble"), "gated validation")
        static = _object(validation.get("static_ensemble"), "static validation")
        selected_mode = (
            "uncertainty_gated"
            if float(gated["mae"]) < float(static["mae"]) else "static"
        )
    return {
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "source_manifest": source_manifest,
        "checkpoints": resolved,
        "weights": weights,
        "scales": scales,
        "exponent": exponent,
        "selected_mode": selected_mode,
    }


def predict_mapper4x4(
    manifest: Path, checkpoint_path: Path, seed: int,
    batch_size: int, device: torch.device,
    excluded_query_ids: Sequence[str] = (),
    split_name: str = "test",
) -> Dict[str, Dict[str, Any]]:
    artifact = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Model2Config(**artifact["config"]).validate()
    _, queries = load_terminal_manifest(
        manifest, config.dfg_representation, AMOEBA_STATIC_SHAPE_PROTOCOL,
    )
    selected = []
    excluded = set(excluded_query_ids)
    if split_name not in {"validation", "test"}:
        raise ValueError("comparison split must be validation or test")
    for query in split_queries(queries, seed)[split_name]:
        if query.ranking_query_id in excluded:
            continue
        matches = tuple(
            candidate for candidate in query.candidates
            if (candidate.rows, candidate.columns) == (4, 4)
        )
        if len(matches) != 1:
            raise ValueError("each query must contain exactly one mapper 4x4 label")
        selected.append(replace(query, candidates=matches))
    model = JointGraphShapeModel(config).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    shape_graph = [make_cgra_graph(4, 4, config.shape_protocol)]
    result: Dict[str, Dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in query_batches(selected, batch_size):
            context, _, _, _, _ = batch_targets(batch, config, device)
            output = model(
                [query.graph for query in batch], shape_graph, context,
            )
            means = output["predicted_ii"].cpu().numpy()[:, 0]
            deviations = output["predicted_ii_std"].cpu().numpy()[:, 0]
            success = output["success_probability"].cpu().numpy()[:, 0]
            for query, mean, deviation, probability in zip(
                batch, means, deviations, success,
            ):
                candidate = query.candidates[0]
                result[candidate.candidate_id] = {
                    "candidate_id": candidate.candidate_id,
                    "ranking_query_id": query.ranking_query_id,
                    "family": query.generator_family,
                    "rows": 4,
                    "columns": 4,
                    "status": candidate.status,
                    "compiled_ii": candidate.compiled_ii,
                    "lower_bound": candidate.lower_bound,
                    "mean": float(mean),
                    "std": float(deviation),
                    "success_probability": float(probability),
                }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate(
    manifest: Path, ensemble: Mapping[str, Any], seed: int,
    batch_size: int, device: torch.device,
    excluded_query_ids: Sequence[str] = (),
    split_name: str = "test",
) -> Dict[str, Any]:
    model_predictions = {
        name: predict_mapper4x4(
            manifest, checkpoint_path, seed, batch_size, device,
            excluded_query_ids, split_name,
        )
        for name, checkpoint_path in ensemble["checkpoints"].items()
    }
    names = list(model_predictions)
    identities = set(model_predictions[names[0]])
    if any(set(model_predictions[name]) != identities for name in names[1:]):
        raise ValueError("checkpoint candidate populations differ")
    successful = sorted(
        identity for identity in identities
        if model_predictions[names[0]][identity]["compiled_ii"] is not None
    )
    rows = [model_predictions[names[0]][identity] for identity in successful]
    targets = np.asarray([row["compiled_ii"] for row in rows], dtype=np.float64)

    def weights_for(identity: str) -> Dict[str, float]:
        weights = dict(ensemble["weights"])
        if ensemble["selected_mode"] == "uncertainty_gated":
            for name in names:
                relative = max(
                    model_predictions[name][identity]["std"] /
                    ensemble["scales"][name],
                    1e-4,
                )
                weights[name] *= relative ** (-ensemble["exponent"])
            total = sum(weights.values())
            weights = {name: value / total for name, value in weights.items()}
        return weights

    predictions = []
    for identity in successful:
        reference = model_predictions[names[0]][identity]
        sample_weights = weights_for(identity)
        value = sample_weights["analytical_lower_bound"] * reference["lower_bound"]
        value += sum(
            sample_weights[name] * model_predictions[name][identity]["mean"]
            for name in names
        )
        predictions.append(max(float(reference["lower_bound"]), value))
    values = np.asarray(predictions, dtype=np.float64)
    ordered = sorted(identities)
    success_predictions = []
    for identity in ordered:
        sample_weights = weights_for(identity)
        model_weight = sum(sample_weights[name] for name in names)
        success_predictions.append(
            sum(
                sample_weights[name] *
                model_predictions[name][identity]["success_probability"]
                for name in names
            ) / model_weight
            if model_weight > 0.0 else
            sum(
                model_predictions[name][identity]["success_probability"]
                for name in names
            ) / len(names)
        )
    success_targets = np.asarray([
        float(model_predictions[names[0]][identity]["status"] == "success")
        for identity in ordered
    ], dtype=np.float64)
    return {
        "selected_mode": ensemble["selected_mode"],
        "successful_candidate_count": len(successful),
        "metrics": _metrics(values, targets, rows),
        "success_classifier": _success_metrics(
            np.asarray(success_predictions, dtype=np.float64), success_targets,
        ),
        "by_family": {
            family: _metrics(
                values[indices], targets[indices], [rows[index] for index in indices],
            )
            for family in sorted({str(row["family"]) for row in rows})
            for indices in [[
                index for index, row in enumerate(rows)
                if str(row["family"]) == family
            ]]
        },
    }


def query_identities(manifest: Path) -> set[str]:
    root = _object(json.loads(manifest.read_text()), "training manifest")
    candidates = root.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("training manifest contains no candidates")
    identities = set()
    for raw in candidates:
        row = _object(raw, "training candidate")
        identity = row.get("ranking_query_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("training candidate lacks ranking_query_id")
        identities.add(identity)
    return identities


def formal_split_query_identities(
    manifest: Path, seed: int, split_name: str,
) -> set[str]:
    _, queries = load_terminal_manifest(
        manifest, "semantic_v1", AMOEBA_STATIC_SHAPE_PROTOCOL,
    )
    return {
        query.ranking_query_id
        for query in split_queries(queries, seed)[split_name]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-ensemble", type=Path, required=True)
    parser.add_argument("--v8-ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    legacy = load_ensemble(args.legacy_ensemble.resolve())
    v8 = load_ensemble(args.v8_ensemble.resolve())
    legacy_training_ids = query_identities(legacy["source_manifest"])
    comparisons = {}
    leakage = {}
    for split_name in ("validation", "test"):
        formal_ids = formal_split_query_identities(
            args.manifest.resolve(), args.seed, split_name,
        )
        overlap = sorted(formal_ids.intersection(legacy_training_ids))
        leakage[split_name] = overlap
        comparisons[split_name] = {
            "legacy_ensemble": evaluate(
                args.manifest.resolve(), legacy, args.seed,
                args.batch_size, device, overlap, split_name,
            ),
            "v8_ensemble": evaluate(
                args.manifest.resolve(), v8, args.seed,
                args.batch_size, device, overlap, split_name,
            ),
        }
    selected_source = min(
        ("legacy_ensemble", "v8_ensemble"),
        key=lambda name: (
            comparisons["validation"][name]["metrics"]["mae"], name,
        ),
    )
    report = {
        "schema_version": "cgra-ii-legacy-v8-mapper4x4-comparison-v1",
        "comparison_contract": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest.resolve()),
            "splits": ["validation", "test"],
            "split_seed": args.seed,
            "mapper_tile_shape": {"rows": 4, "columns": 4},
            "selection": "precommitted_validation_selected_ensembles",
            "legacy_training_overlap_excluded": {
                split_name: {
                    "query_count": len(overlap),
                    "query_ids_sha256": hashlib.sha256(
                        "\n".join(overlap).encode()
                    ).hexdigest(),
                }
                for split_name, overlap in leakage.items()
            },
        },
        "ensemble_artifacts": {
            "legacy_ensemble": {
            "report": str(legacy["path"]),
            "report_sha256": legacy["sha256"],
            },
            "v8_ensemble": {
            "report": str(v8["path"]),
            "report_sha256": v8["sha256"],
            },
        },
        "comparison": comparisons,
        "validation_selected_mapper4x4_source": selected_source,
        "validation_selected_mapper4x4_test_metrics": (
            comparisons["test"][selected_source]
        ),
        "v8_minus_legacy_mae": {
            split_name: (
                split_result["v8_ensemble"]["metrics"]["mae"] -
                split_result["legacy_ensemble"]["metrics"]["mae"]
            )
            for split_name, split_result in comparisons.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
