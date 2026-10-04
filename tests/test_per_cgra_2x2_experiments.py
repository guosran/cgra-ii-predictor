from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters import run_per_cgra_2x2_experiments as experiments
from adapters import train_per_cgra_2x2_model as reference
from cgra_ii_predictor.mapper_model import (
    DirectMapperIIModel,
    MapperModelConfig,
    SHAPE_PROTOCOL_ID,
    mapper_feature_names,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2_ID


SHAPES = tuple(reference.MAPPER_SHAPES)
FEATURE_WIDTH = len(reference.MAPPER_FEATURE_NAMES_2X2)


def _families(group: str, kind: str) -> list[str]:
    return [f"random-dfg/{group}"] if kind == "random" else [f"kernelbench/{group}"]


def _dfg_rows(group: str, kind: str, query_index: int, group_index: int) -> list[dict]:
    families = _families(group, kind)
    query = f"{group}/dfg-{query_index}"
    rows = []
    for shape_index, shape in enumerate(SHAPES):
        features = [
            ((group_index * 17 + query_index * 11 + shape_index * 7 + index * 3 +
              shape_index * index) % 101) / 101.0
            for index in range(FEATURE_WIDTH)
        ]
        rows.append({
            "group": group,
            "group_families": families,
            "query": query,
            "shape": tuple(shape),
            "ii": float(2 + shape_index % 4),
            "lower_bound": 1.0,
            "full_features": features,
        })
    return rows


def _fit_data(tmp_path: Path) -> tuple[Path, list[dict], list[dict], dict]:
    train_specs = [
        ("random-a", "random", 2, 0),
        ("random-b", "random", 1, 1),
        ("program-a", "program", 1, 2),
        ("program-b", "program", 2, 3),
    ]
    validation_specs = [
        ("random-validation", "random", 1, 4),
        ("program-validation", "program", 1, 5),
    ]
    training_rows = [
        row
        for group, kind, query_count, group_index in train_specs
        for query_index in range(query_count)
        for row in _dfg_rows(group, kind, query_index, group_index)
    ]
    validation_rows = [
        row
        for group, kind, query_count, group_index in validation_specs
        for query_index in range(query_count)
        for row in _dfg_rows(group, kind, query_index, group_index)
    ]
    train_groups = [group for group, _kind, _count, _index in train_specs]
    validation_groups = [group for group, _kind, _count, _index in validation_specs]
    development_rows = (
        [{"split": "train"} for _row in training_rows] +
        [{"split": "validation"} for _row in validation_rows]
    )
    development = tmp_path / "development.pt"
    torch.save({
        "rows": development_rows,
        "training_rows": training_rows + validation_rows,
        "assignment": {
            **{group: "train" for group in train_groups},
            **{group: "validation" for group in validation_groups},
        },
    }, development)
    partition = {
        "name": "synthetic",
        "train": train_groups,
        "validation": validation_groups,
        "evaluation": [],
    }
    return development, training_rows, validation_rows, partition


def _fit_spec(
    development: Path,
    partition: dict,
    output: Path,
    *,
    name: str,
    normalization: str = "row",
    select_validation: bool = False,
) -> dict:
    return {
        "development": str(development),
        "partition": partition,
        "config": {
            "name": name,
            "feature_names": list(reference.COMPACT_FEATURE_NAMES),
            "normalization": normalization,
            "output_parameterization": "residual_softplus",
            "updates": 2,
            "select_validation": select_validation,
        },
        "seed": 41,
        "output": str(output),
    }


def test_development_loader_rejects_test_excluded_and_frozen_test_inputs(tmp_path):
    path = tmp_path / "development.pt"
    for split in ("test", "excluded"):
        torch.save({"rows": [{"split": split}]}, path)
        with pytest.raises(ValueError, match="holdout or excluded"):
            experiments.load_development(path)

    frozen_test = tmp_path / "frozen-test.pt"
    torch.save({"rows": [{"split": "test"}]}, frozen_test)
    with pytest.raises(ValueError, match="only the sealed development.pt"):
        experiments.load_development(frozen_test)


def test_nested_partitions_are_group_disjoint_stratified_and_omit_fixed_test_groups():
    kinds = {
        **{f"random-{index}": "random" for index in range(9)},
        **{f"program-{index}": "program" for index in range(9)},
    }
    fixed_test_groups = {"frozen-test-random", "frozen-test-program"}
    assignment = {
        group: ("train" if index < 5 else "validation")
        for kind in ("random", "program")
        for index, group in enumerate(sorted(g for g in kinds if kinds[g] == kind))
    }
    assignment.update({group: "test" for group in fixed_test_groups})
    data = {
        "rows": [
            {"source_group": group, "source_kind": kind}
            for group, kind in kinds.items()
        ],
        "assignment": assignment,
    }

    partitions = experiments.group_partitions(data)
    development_groups = set(kinds)
    for partition in partitions[:3]:
        train = set(partition["train"])
        validation = set(partition["validation"])
        holdout = set(partition["evaluation"])
        assert not train & validation
        assert not train & holdout
        assert not validation & holdout
        assert train | validation | holdout == development_groups
        assert fixed_test_groups.isdisjoint(train | validation | holdout)
        for role in (train, validation, holdout):
            assert {kinds[group] for group in role} == {"random", "program"}

    original = partitions[3]
    assert original["evaluation"] == []
    assert fixed_test_groups.isdisjoint(
        set(original["train"]) | set(original["validation"]))
    assert not set(original["train"]) & set(original["validation"])


def test_weighted_fit_balances_strata_groups_and_model_normalization(tmp_path):
    torch.set_num_threads(1)
    assert FEATURE_WIDTH == 148
    development, training_rows, _validation_rows, partition = _fit_data(tmp_path)
    weights = reference._balanced_group_weights(training_rows)
    mass_by_kind: dict[str, float] = defaultdict(float)
    mass_by_group: dict[str, float] = defaultdict(float)
    for row, weight in zip(training_rows, weights.tolist()):
        kind = "random" if all(
            family.startswith("random-dfg/") for family in row["group_families"]
        ) else "program"
        mass_by_kind[kind] += weight
        mass_by_group[row["group"]] += weight
    assert mass_by_kind == pytest.approx({"random": 0.5, "program": 0.5})
    assert mass_by_group == pytest.approx({
        "random-a": 0.25, "random-b": 0.25,
        "program-a": 0.25, "program-b": 0.25,
    })

    raw = torch.tensor([row["full_features"] for row in training_rows])
    expected_mean = (raw * weights[:, None]).sum(0)
    expected_scale = ((raw - expected_mean).square() * weights[:, None]).sum(0).sqrt()
    expected_scale = torch.where(expected_scale < 1e-5, torch.ones_like(expected_scale), expected_scale)

    output = tmp_path / "weighted-fit"
    experiments.fit_job(_fit_spec(
        development, partition, output, name="weighted-small",
        normalization="loss", select_validation=True))
    checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    assert torch.equal(state["feature_mean"], expected_mean)
    assert torch.equal(state["feature_scale"], expected_scale)


def test_reference80_fit_matches_legacy_member_and_archive_replay(tmp_path):
    torch.set_num_threads(1)
    development, training_rows, validation_rows, partition = _fit_data(tmp_path)
    output = tmp_path / "reference-fit"
    experiments.fit_job(_fit_spec(
        development, partition, output, name="reference80",
        normalization="row", select_validation=False))

    checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
    reference_model = reference._fit_member(training_rows, seed=41, epochs=2)
    for name, value in reference_model.state_dict().items():
        assert torch.equal(checkpoint["state_dict"][name], value), name

    selected_epoch = checkpoint["selected_epoch"]
    assert selected_epoch == 2
    archive = np.load(output / "epoch-checkpoints.npz")
    replay = DirectMapperIIModel(
        MapperModelConfig(**checkpoint["config"]),
        feature_mean=archive["feature_mean"],
        feature_scale=archive["feature_scale"],
    )
    replay_state = replay.state_dict()
    for name, _parameter in replay.named_parameters():
        replay_state[name] = torch.as_tensor(archive[name][selected_epoch - 1].copy())
    replay.load_state_dict(replay_state)
    actual = reference._predict(replay, validation_rows)
    predictions = json.loads((output / "predictions.json").read_text())["validation"]
    recorded = [row["prediction"] for row in predictions]
    assert actual == recorded


def test_direct_softplus_loss_gradient_floor_ceiling_and_legacy_config_serialization():
    torch.set_num_threads(1)
    feature_name = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)[0]
    config = MapperModelConfig(
        hidden_dimensions=(8, 8),
        shape_protocol=SHAPE_PROTOCOL_2X2_ID,
        enabled_feature_names=(feature_name,),
        output_parameterization="direct_softplus",
    )
    model = DirectMapperIIModel(config)
    features = torch.zeros((2, FEATURE_WIDTH))
    lower = torch.tensor([3.0, 4.0])
    with torch.no_grad():
        model.regressor[-1].weight.zero_()
        model.regressor[-1].bias.fill_(-2.0)
    training_prediction = model.prediction_for_loss(features, lower)
    assert torch.all(training_prediction < lower)
    training_prediction.sum().backward()
    assert model.regressor[-1].bias.grad is not None
    assert torch.all(model.regressor[-1].bias.grad.abs() > 0)
    assert torch.equal(model(features, lower), lower)

    with torch.no_grad():
        model.regressor[-1].bias.fill_(100.0)
    assert torch.equal(model.prediction_for_loss(features, lower), torch.full((2,), 20.0))
    assert torch.equal(model(features, lower), torch.full((2,), 20.0))

    assert MapperModelConfig().to_dict() == {
        "hidden_dimensions": [64, 32],
        "mapper_ii_ceiling": 20.0,
        "shape_protocol": SHAPE_PROTOCOL_ID,
        "enabled_feature_names": None,
    }


def test_validation_selection_criterion_orders_regret_then_hit_then_mae():
    def metrics(regret: float, hit: float, mae: float) -> dict:
        return {
            "balanced_group_regret": regret,
            "balanced_group_hit": hit,
            "mae": mae,
        }

    assert experiments.criterion(metrics(1.0, 0.0, 10.0)) < experiments.criterion(
        metrics(1.1, 1.0, 0.0))
    assert experiments.criterion(metrics(1.0, 0.8, 10.0)) < experiments.criterion(
        metrics(1.0, 0.7, 0.0))
    assert experiments.criterion(metrics(1.0, 0.8, 1.0)) < experiments.criterion(
        metrics(1.0, 0.8, 1.1))


def _integrity_fixture(root: Path, *, mapper_hash: str | None = None) -> tuple[Path, Path, dict]:
    cache = root / "cache"
    output = root / "run"
    cache.mkdir(parents=True)
    output.mkdir(parents=True)
    for name, content in (
        ("development.pt", b"development-cache"),
        ("frozen-test.pt", b"frozen-holdout"),
        ("evaluation-only.pt", b"excluded-rows"),
    ):
        (cache / name).write_bytes(content)
    files = {
        name: experiments.sha256_file(cache / name)
        for name in experiments.CACHE_FILES
    }
    manifest = {
        "schema": experiments.CACHE_MANIFEST_SCHEMA,
        "files": files,
    }
    (cache / "cache-manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    plan = {
        "schema": "cgra-ii-2x2-bounded-experiment-v1",
        "development_sha256": files["development.pt"],
        "cache_manifest_sha256": experiments.sha256_file(cache / "cache-manifest.json"),
        "cache_files": files,
        "frozen_test_sha256": files["frozen-test.pt"],
        "mapper_model_sha256": mapper_hash or experiments.sha256_file(
            experiments.ROOT / "src/cgra_ii_predictor/mapper_model.py"
        ),
        "code_sha256": experiments.sha256_file(Path(experiments.__file__)),
        "configs": [{"name": "reference80"}],
        "seeds": [41],
        "partitions": [],
    }
    experiments._write_sealed_json(output / "experiment-plan.json", plan)
    return cache / "development.pt", output, plan


def _plan_cache(root: Path) -> tuple[Path, Path]:
    cache = root / "cache"
    output = root / "run"
    cache.mkdir(parents=True)
    torch.save({"rows": [], "training_rows": [], "assignment": {}}, cache / "development.pt")
    (cache / "frozen-test.pt").write_bytes(b"fixed-holdout")
    (cache / "evaluation-only.pt").write_bytes(b"excluded")
    files = {
        name: experiments.sha256_file(cache / name)
        for name in experiments.CACHE_FILES
    }
    (cache / "cache-manifest.json").write_text(json.dumps({
        "schema": experiments.CACHE_MANIFEST_SCHEMA,
        "files": files,
    }, sort_keys=True))
    return cache / "development.pt", output


def _write_frozen_selection(output: Path, development: Path, plan: dict) -> dict:
    checkpoint = output / "fits/original/reference80/41/selected.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selected-model-checkpoint")
    summary = output / "development-summary.json"
    summary.write_text("{}\n")
    plan_path = output / "experiment-plan.json"
    selection = {
        "schema": "cgra-ii-2x2-frozen-selection-v1",
        "plan_sha256": experiments.sha256_file(plan_path),
        "summary_sha256": experiments.sha256_file(summary),
        "test_opened": False,
        "selected_configs": ["reference80"],
        "cache_manifest_sha256": plan["cache_manifest_sha256"],
        "frozen_test_sha256": plan["frozen_test_sha256"],
        "selected_checkpoint_sha256": {
            "fits/original/reference80/41/selected.pt": experiments.sha256_file(checkpoint),
        },
    }
    experiments._write_sealed_json(output / "frozen-selection.json", selection)
    return selection


def test_plan_validates_and_records_cache_manifest_and_expected_test_hash(tmp_path):
    development, output = _plan_cache(tmp_path)
    experiments.plan(SimpleNamespace(development=development, output=output))

    plan = json.loads((output / "experiment-plan.json").read_text())
    manifest_path = development.parent / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert plan["cache_manifest_sha256"] == experiments.sha256_file(manifest_path)
    assert plan["cache_files"] == manifest["files"]
    assert plan["development_sha256"] == manifest["files"]["development.pt"]
    assert plan["frozen_test_sha256"] == manifest["files"]["frozen-test.pt"]
    assert experiments._sidecar(output / "experiment-plan.json").is_file()


def test_plan_rejects_cache_member_mutation_before_loading_development(tmp_path, monkeypatch):
    development, output = _plan_cache(tmp_path)
    (development.parent / "frozen-test.pt").write_bytes(b"changed-after-audit")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("cache validation must precede development deserialization")

    monkeypatch.setattr(experiments, "load_development", unexpected_load)
    with pytest.raises(ValueError, match="cache seal does not match frozen-test.pt"):
        experiments.plan(SimpleNamespace(development=development, output=output))


def test_fit_and_summarize_verify_mapper_model_hash_before_work(tmp_path, monkeypatch):
    development, output, _plan = _integrity_fixture(tmp_path, mapper_hash="0" * 64)
    args = SimpleNamespace(development=development, output=output, jobs=1)

    def unexpected_work(*_args, **_kwargs):
        pytest.fail("integrity failure must precede training or development loading")

    monkeypatch.setattr(experiments, "ProcessPoolExecutor", unexpected_work)
    with pytest.raises(ValueError, match="mapper model changed"):
        experiments.fit(args)

    monkeypatch.setattr(experiments, "load_development", unexpected_work)
    with pytest.raises(ValueError, match="mapper model changed"):
        experiments.summarize(args)


@pytest.mark.parametrize("mutation", ["frozen-test", "selected-checkpoint"])
def test_test_rejects_changed_holdout_or_selected_checkpoint_before_torch_load(
    tmp_path, monkeypatch, mutation
):
    development, output, plan = _integrity_fixture(tmp_path)
    _write_frozen_selection(output, development, plan)
    if mutation == "frozen-test":
        (development.parent / "frozen-test.pt").write_bytes(b"mutated-holdout")
    else:
        (output / "fits/original/reference80/41/selected.pt").write_bytes(b"mutated-checkpoint")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("integrity validation must finish before any tensor is loaded")

    monkeypatch.setattr(experiments.torch, "load", unexpected_load)
    args = SimpleNamespace(development=development, output=output)
    message = "cache seal does not match frozen-test.pt" if mutation == "frozen-test" else "selected checkpoints changed"
    with pytest.raises(ValueError, match=message):
        experiments.test(args)


def test_summarize_refuses_to_reset_an_existing_frozen_selection(tmp_path, monkeypatch):
    development, output, _plan = _integrity_fixture(tmp_path)
    selection_path = output / "frozen-selection.json"
    selection_path.write_text('{"test_opened":true,"owner":"frozen-run"}\n')
    original = selection_path.read_bytes()

    def unexpected_work(*_args, **_kwargs):
        pytest.fail("existing selection must be detected before summary recomputation")

    monkeypatch.setattr(experiments, "load_development", unexpected_work)
    with pytest.raises(ValueError, match="refusing to replace an existing summary or frozen selection"):
        experiments.summarize(SimpleNamespace(development=development, output=output))
    assert selection_path.read_bytes() == original


def test_plan_refuses_existing_plan_or_frozen_selection_before_loading_cache(tmp_path, monkeypatch):
    development, output, _plan = _integrity_fixture(tmp_path)
    (output / "experiment-plan.json").unlink()
    experiments._sidecar(output / "experiment-plan.json").unlink()
    selection_path = output / "frozen-selection.json"
    selection_path.write_text("{}\n")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("plan overwrite guard must run before reading development data")

    monkeypatch.setattr(experiments, "load_development", unexpected_load)
    with pytest.raises(ValueError, match="refusing to overwrite a plan or frozen selection"):
        experiments.plan(SimpleNamespace(development=development, output=output))
