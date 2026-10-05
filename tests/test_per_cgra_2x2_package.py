import json
from pathlib import Path

from adapters.train_per_cgra_2x2_model import evaluate
from cgra_ii_predictor.checkpoint import COMPACT_FEATURE_NAMES, sha256_file
from cgra_ii_predictor.mapper_model import mapper_feature_names
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2_ID
from cgra_ii_predictor.training_data import load_training_dataset


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "models/candidates/per-cgra-2x2"


def test_portable_dataset_preserves_original_rows_splits_exclusions_and_nulls():
    dataset = load_training_dataset(PACKAGE / "dataset")
    manifest = dataset["manifest"]
    feature_names = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
    compact_indices = manifest["compact_feature_indices"]
    assert all(0 <= index < len(feature_names) for index in compact_indices)
    assert tuple(feature_names[index] for index in compact_indices) == COMPACT_FEATURE_NAMES
    assert manifest["counts"]["query_rows"] == 2584
    assert manifest["counts"]["dfg_count"] == 323
    assert manifest["counts"]["eligible_successful_rows"] == 2085
    assert manifest["counts"]["eligible_censored_rows"] == 11
    assert dataset["split_summary"] == {
        "train": {"query_rows": 1480, "successful_rows": 1469, "censored_rows": 11, "dfg_count": 185, "group_count": 164},
        "validation": {"query_rows": 312, "successful_rows": 312, "censored_rows": 0, "dfg_count": 39, "group_count": 35},
        "test": {"query_rows": 304, "successful_rows": 304, "censored_rows": 0, "dfg_count": 38, "group_count": 36},
        "excluded": {"query_rows": 488, "successful_rows": 484, "censored_rows": 4, "dfg_count": 61, "group_count": 3},
    }
    null_rows = [row for row in dataset["rows"] if row["native_status"] == "censored"]
    assert len(null_rows) == 15
    assert all(row["ii"] is None for row in null_rows)
    assert sum(row["eligible_for_training"] for row in null_rows) == 11
    assert all(row["full_features"] is not None for row in dataset["rows"] if row["native_status"] == "success")


def test_original_c0_weights_and_forward_evaluation_are_preserved():
    model_metadata = json.loads((PACKAGE / "model.json").read_text())
    original_report = json.loads((PACKAGE / "training-report.json").read_text())
    checkpoint = PACKAGE / "mapper.pt"
    assert sha256_file(checkpoint) == "bb7196d5b37d54b1e5245a5e47cb6323cebfb10deea6d902e2ac6425f74c155e"
    result = evaluate(PACKAGE / "dataset", None)
    assert result["checkpoint_sha256"] == model_metadata["checkpoint"]["sha256"]
    for split in ("train", "validation", "test"):
        observed = result["by_split"][split]
        expected = original_report["evaluation"]["by_split"][split]
        for key in ("row_count", "group_count", "complete_query_count", "partial_query_count"):
            assert observed[key] == expected[key]
        for key in ("row_mae", "rounded_exact_ii", "rounded_within_one_ii", "shape_hit", "mean_shape_regret"):
            assert abs(observed[key] - expected[key]) < 1e-7
