import json
from pathlib import Path

from adapters.apply_conservative_amoeba_policy import apply_policy
from adapters.amoeba_protocol import COST_SCHEMA


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_policy_preserves_point_ordering_by_default(tmp_path):
    point_hash = "a" * 64
    expert_hash = "b" * 64
    base = {
        "schema_version": COST_SCHEMA,
        "function": "attention",
        "namespace": "point",
        "predictor_metadata": {"checkpoints": {"point": {"sha256": point_hash}}},
        "entries": [{
            "task": "task", "mapper_tile_rows": 4, "mapper_tile_cols": 4,
            "support_status": "supported", "predicted_ii": 2.0,
            "predicted_ii_std": 0.2,
        }],
    }
    expert = dict(base)
    expert["predictor_metadata"] = {"checkpoints": {"expert": {"sha256": expert_hash}}}
    expert["entries"] = [dict(base["entries"][0], predicted_ii=4.0, predicted_ii_std=1.0)]
    policy = {
        "schema_version": "cgra-ii-conservative-pointwise",
        "selection_split": "validation_only",
        "policy": "max_ensemble_mean_and_expert_mean_plus_alpha_std",
        "selected_alpha": 0.5,
        "quantile": 0.85,
        "checkpoints": {"point": {"sha256": point_hash}},
        "expert": {"sha256": expert_hash},
    }
    point_path = tmp_path / "point.json"
    expert_path = tmp_path / "expert.json"
    policy_path = tmp_path / "policy.json"
    write_json(point_path, base)
    write_json(expert_path, expert)
    write_json(policy_path, policy)
    output = apply_policy(point_path, expert_path, policy_path)
    row = output["entries"][0]
    assert row["predicted_ii"] == 2.0
    assert row["predicted_ii_upper"] == 4.5
    assert row["point_predicted_ii"] == 2.0
