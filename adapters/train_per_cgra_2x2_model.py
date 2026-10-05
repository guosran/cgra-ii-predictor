#!/usr/bin/env python3
"""Train or evaluate the portable, frozen-contract per-CGRA 2x2 model.

Training consumes only the packaged v1 feature matrix. This command never
extracts sources or invokes a mapper. Training defaults are frozen to the
original four seeds, 80 epochs, compact61 mask, group split, and loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "adapters", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import torch  # noqa: E402

from cgra_ii_predictor.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    COMPACT_FEATURE_NAMES,
    EXPECTED_ARCHITECTURE_SHA256,
    load_checkpoint,
    load_published_candidate,
    sha256_file,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID  # noqa: E402
from cgra_ii_predictor.training_data import (  # noqa: E402
    canonical_json_sha256,
    load_training_dataset,
)
from per_cgra_2x2_training_core import (  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    FEATURE_NAMES_2X2,
    _fit_member,
    _predict,
    _score,
    ensemble_from_members,
)


DEFAULT_PACKAGE = ROOT / "models/candidates/per-cgra-2x2"


def _rows_by_split(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [
            row for row in data["rows_by_split"][split]
            if row["eligible_for_training"] and row["native_status"] == "success"
        ]
        for split in ("train", "validation", "test")
    }


def evaluate(dataset_dir: Path, checkpoint: Path | None) -> dict[str, Any]:
    data = load_training_dataset(dataset_dir)
    if checkpoint is None:
        model, _config, metadata = load_published_candidate(DEFAULT_PACKAGE)
        checkpoint_path = DEFAULT_PACKAGE / metadata["checkpoint"]["path"]
        checkpoint_sha = metadata["checkpoint"]["sha256"]
    elif checkpoint.resolve() == (DEFAULT_PACKAGE / "mapper.pt").resolve():
        model, _config, metadata = load_published_candidate(DEFAULT_PACKAGE)
        checkpoint_path = checkpoint.resolve()
        checkpoint_sha = metadata["checkpoint"]["sha256"]
    else:
        model, _config, artifact = load_checkpoint(checkpoint)
        checkpoint_path = checkpoint.resolve()
        checkpoint_sha = sha256_file(checkpoint_path)
        metadata = {"architecture": {"sha256": artifact["architecture_sha256"]}}
    rows = _rows_by_split(data)
    scores = {
        split: _score(rows[split], _predict(model, rows[split]))
        for split in ("train", "validation", "test")
    }
    return {
        "schema": "cgra-ii-per-cgra-2x2-portable-evaluation-v1",
        "checkpoint_sha256": checkpoint_sha,
        "architecture_sha256": metadata["architecture"]["sha256"],
        "dataset_manifest_sha256": data["manifest"]["manifest_sha256"],
        "input_contract": data["manifest"]["input_contract"],
        "eligible_labeled_rows": {name: len(rows[name]) for name in rows},
        "by_split": scores,
        "checkpoint_path": str(checkpoint_path),
        "test_holdout_used_for_training_or_tuning": False,
    }


def train(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    data = load_training_dataset(dataset_dir)
    rows = _rows_by_split(data)
    if (len(rows["train"]) != 1469 or len(rows["validation"]) != 312 or
            len(rows["test"]) != 304):
        raise ValueError("packaged split counts differ from the frozen C0 recipe")
    if not rows["train"]:
        raise ValueError("training fold contains no labeled rows")
    torch.set_num_threads(1)
    members = [
        _fit_member(rows["train"], seed=seed, epochs=DEFAULT_EPOCHS)
        for seed in DEFAULT_SEEDS
    ]
    ensemble = ensemble_from_members(members)
    scores = {
        split: _score(rows[split], _predict(ensemble, rows[split]))
        for split in ("train", "validation", "test")
    }
    manifest = data["manifest"]
    provenance = manifest["provenance"]
    subset = [{
        "candidate_id": row["candidate_id"],
        "source_sha256": row["source_sha256"],
        "mapper_input_identity": row["mapper_input_identity"],
        "shape": list(row["shape"]),
        "compiled_ii": int(row["ii"]),
        "lower_bound": int(row["lower_bound"]),
        "group": row["group"],
    } for row in rows["train"]]
    output_dir.mkdir(parents=True)
    checkpoint_path = output_dir / "mapper.pt"
    artifact = {
        "schema": CHECKPOINT_SCHEMA,
        "output_mode": "continuous",
        "deployment_readout": "arithmetic_mean_bounded_continuous_regression",
        "ranking_readout": "arithmetic_mean_bounded_continuous_regression",
        "config": members[0].config.to_dict(),
        "feature_names": list(FEATURE_NAMES_2X2),
        "compact_feature_names": list(COMPACT_FEATURE_NAMES),
        "state_dict": ensemble.state_dict(),
        "ensemble_member_count": len(members),
        "ensemble_reduction": "arithmetic_mean",
        "ensemble_seeds": list(DEFAULT_SEEDS),
        "architecture_sha256": provenance["architecture_sha256"],
        "supported_architecture_sha256": [provenance["architecture_sha256"]],
        "architecture_compatibility_rule": "exact_architecture_sha256",
        "supported_mapper_shapes": [list(shape) for shape in SHAPE_PROTOCOL_2X2.mapper_shapes],
        "training_mapper_shapes": [list(shape) for shape in SHAPE_PROTOCOL_2X2.mapper_shapes],
        "shape_protocol_id": SHAPE_PROTOCOL_2X2_ID,
        "training_manifest_sha256": provenance["source_manifest_sha256"],
        "neura_opt_sha256": provenance["neura_opt_sha256"],
        "collection_provenance_sha256": provenance["collection_provenance_sha256"],
        "query_manifest_sha256": provenance["query_manifest_sha256"],
        "outcomes_manifest_sha256": provenance["outcomes_manifest_sha256"],
        "collection_complete_sha256": provenance["collection_complete_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "source_groups_sha256": provenance["source_groups_sha256"],
        "training_exclusions_sha256": provenance["training_exclusions_sha256"],
        "training_subset_sha256": canonical_json_sha256(subset),
        "training_split_seed": 20261004,
        "training_group_ids": sorted({row["group"] for row in rows["train"]}),
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "candidate_only": True,
    }
    torch.save(artifact, checkpoint_path)
    loaded, _config, _artifact = load_checkpoint(
        checkpoint_path,
        expected_architecture_sha256=provenance["architecture_sha256"],
        expected_checkpoint_sha256=sha256_file(checkpoint_path),
    )
    replay_error = max(
        (abs(a - b) for a, b in zip(
            _predict(ensemble, rows["test"]), _predict(loaded, rows["test"]),
        )),
        default=0.0,
    )
    if replay_error != 0.0:
        raise ValueError("packaged loader changed trained-model predictions")
    report = {
        "schema": "cgra-ii-per-cgra-2x2-portable-training-report-v1",
        "status": "trained_from_frozen_v1_feature_matrix_candidate_only",
        "input_contract": manifest["input_contract"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "architecture_sha256": provenance["architecture_sha256"],
        "feature_count": 61,
        "hidden_dimensions": [64, 32],
        "epochs": DEFAULT_EPOCHS,
        "seeds": list(DEFAULT_SEEDS),
        "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
        "loss": "weighted SmoothL1 + 0.1 pairwise hinge + 0.3 set top-1",
        "split_seed": 20261004,
        "eligible_successful_rows": {name: len(rows[name]) for name in rows},
        "censored_native_labels_remain_null": True,
        "scores_by_split": scores,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_loader_replay_max_abs_difference": replay_error,
        "test_holdout_used_for_training_or_tuning": False,
    }
    (output_dir / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument(
        "--dataset", type=Path,
        default=DEFAULT_PACKAGE / "dataset",
        help="portable 2x2 v1 feature-matrix dataset directory",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        if args.output_dir is None:
            parser.error("--output-dir is required in train mode")
        report = train(args.dataset, args.output_dir)
    else:
        if args.output_dir is not None:
            parser.error("--output-dir is not used in evaluate mode")
        report = evaluate(args.dataset, args.checkpoint)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
