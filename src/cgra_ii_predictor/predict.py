"""Load a trained residual-Ridge artifact and predict compiled II without labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .dataset import (
    FORBIDDEN_MODEL_FEATURE_NAMES,
    LOWER_BOUND_COMPONENT_NAMES,
    LOWER_BOUND_KEYS,
)
from .model import (
    constrained_predicted_residual,
    predict_compiled_ii,
    raw_ridge_residual_from_features,
)

PROVEN_COMPONENT_KEYS = LOWER_BOUND_COMPONENT_NAMES


@dataclass(frozen=True)
class PredictionSample:
    """An unlabeled candidate: only prediction-time facts are represented."""

    sample_id: str
    lower_bound: float
    rec_mii: float
    res_mii: float
    lower_bound_source: str
    features: Mapping[str, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedModel:
    """A validated model plus the identities needed to audit its use."""

    model: Mapping[str, Any]
    model_sha256: str
    source_path: Path
    source_sha256: str
    container: str
    target: str
    artifact_status: Optional[str]
    lower_bound_contract: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be a finite number")
    return number


def _numeric_vector(
    model: Mapping[str, Any], name: str, expected_length: int,
    *, strictly_positive: bool = False,
) -> List[float]:
    raw = model.get(name)
    if not isinstance(raw, list) or len(raw) != expected_length:
        raise ValueError(
            f"model.{name} must contain exactly {expected_length} numbers"
        )
    values = [
        _finite_number(value, f"model.{name}[{index}]")
        for index, value in enumerate(raw)
    ]
    if strictly_positive and any(value <= 0.0 for value in values):
        raise ValueError(f"every model.{name} value must be positive")
    return values


def validate_model(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-compatible validated residual-Ridge model."""
    if not isinstance(model, Mapping):
        raise ValueError("model must be a JSON object")
    if model.get("model_type") != "residual_ridge":
        raise ValueError("model.model_type must be residual_ridge")
    raw_names = model.get("feature_names")
    if not isinstance(raw_names, list) or any(
        not isinstance(name, str) or not name for name in raw_names
    ):
        raise ValueError("model.feature_names must be a list of non-empty strings")
    if len(set(raw_names)) != len(raw_names):
        raise ValueError("model.feature_names must not contain duplicates")
    forbidden_features = sorted(
        set(raw_names).intersection(FORBIDDEN_MODEL_FEATURE_NAMES)
    )
    if forbidden_features:
        raise ValueError(
            "model.feature_names must not contain labels or lower-bound "
            f"fields: {forbidden_features}"
        )
    feature_count = len(raw_names)
    _numeric_vector(model, "mean", feature_count)
    _numeric_vector(model, "scale", feature_count, strictly_positive=True)
    _numeric_vector(model, "weights", feature_count + 1)
    ridge = _finite_number(model.get("ridge"), "model.ridge")
    if ridge <= 0.0:
        raise ValueError("model.ridge must be positive")
    dead_zone = _finite_number(
        model.get("residual_dead_zone", 0.0), "model.residual_dead_zone"
    )
    if dead_zone < 0.0:
        raise ValueError("model.residual_dead_zone must be non-negative")
    if "unseen_group_absolute_error_radius" in model:
        radius = _finite_number(
            model["unseen_group_absolute_error_radius"],
            "model.unseen_group_absolute_error_radius",
        )
        if radius < 0.0:
            raise ValueError(
                "model.unseen_group_absolute_error_radius must be non-negative"
            )
    if "unseen_group_interval_empirical_quantile" in model:
        quantile = _finite_number(
            model["unseen_group_interval_empirical_quantile"],
            "model.unseen_group_interval_empirical_quantile",
        )
        if not 0.0 < quantile <= 1.0:
            raise ValueError(
                "model.unseen_group_interval_empirical_quantile must be in (0, 1]"
            )
    # Round-trip through JSON to detach the artifact from custom mapping types
    # and reject non-JSON values. allow_nan=False preserves the finite contract.
    return json.loads(json.dumps(dict(model), allow_nan=False))


def canonical_model_sha256(model: Mapping[str, Any]) -> str:
    """Hash a model independently of JSON whitespace and key order."""
    payload = json.dumps(
        model, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_training_canonical_dfg_identity(
    provenance: Mapping[str, Any], source: Path,
) -> None:
    """Validate the exact training-base identity set carried by v2."""
    identity = provenance.get("training_canonical_dfg_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"{source}: v2 artifact is missing training canonical DFG identity"
        )
    if identity.get("scheme") != "canonical_dfg_sha256_v1":
        raise ValueError(f"{source}: unsupported training DFG identity scheme")
    hashes = identity.get("canonical_dfg_sha256s")
    if not isinstance(hashes, list) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes
    ):
        raise ValueError(
            f"{source}: training canonical DFG hashes must be lowercase SHA-256"
        )
    if hashes != sorted(hashes) or len(set(hashes)) != len(hashes):
        raise ValueError(
            f"{source}: training canonical DFG hashes must be sorted and unique"
        )
    count = identity.get("distinct_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(hashes):
        raise ValueError(
            f"{source}: training canonical DFG count does not match hash list"
        )
    expected_digest = _canonical_json_sha256(hashes)
    if identity.get("set_sha256") != expected_digest:
        raise ValueError(f"{source}: training canonical DFG set digest changed")
    existing_count = provenance.get("training_distinct_base_dfg_count")
    if (
        isinstance(existing_count, bool) or
        not isinstance(existing_count, int) or existing_count != count
    ):
        raise ValueError(
            f"{source}: training canonical DFG count disagrees with provenance"
        )


def training_canonical_dfg_hashes(loaded: LoadedModel) -> frozenset[str]:
    """Return the validated v2 training canonical DFG identity set."""
    identity = loaded.provenance.get("training_canonical_dfg_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("frozen model training canonical DFG identity is missing")
    hashes = identity.get("canonical_dfg_sha256s")
    if not isinstance(hashes, list):
        raise ValueError("frozen model training canonical DFG identity is invalid")
    return frozenset(str(value) for value in hashes)


def load_model_artifact(path: Path) -> LoadedModel:
    """Load either a direct model object or a training report containing one."""
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: model artifact must be a JSON object")
    if raw.get("model_type") == "residual_ridge":
        candidate = raw
        expected_sha256 = None
        container = "direct_model"
        target = "compiled_ii_unspecified_direct_model"
        artifact_status = None
        lower_bound_contract: Mapping[str, Any] = {}
        provenance: Mapping[str, Any] = {}
    else:
        schema_version = raw.get("schema_version")
        accepted_containers = {
            "portable-model-report-v2",
            "neura-experiment-v2",
            "compiled-ii-model-artifact-v1",
            "compiled-ii-model-artifact-v2",
        }
        if schema_version not in accepted_containers:
            raise ValueError(
                f"{path}: unsupported compiled-II model container "
                f"{schema_version!r}"
            )
        candidate = raw.get("trained_full_model")
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"{path}: expected a residual-Ridge model or trained_full_model"
            )
        expected_sha256 = raw.get("trained_full_model_sha256")
        if not isinstance(expected_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ) is None:
            raise ValueError(
                f"{path}: report must contain a 64-character lowercase "
                "trained_full_model_sha256"
            )
        container = str(schema_version)
        raw_target = raw.get("target", "compiled_ii_from_the_recorded_mapper")
        accepted_targets = {
            "compiled_ii_from_the_recorded_mapper",
            "compiled_ii_from_neura_heuristic_mapper",
        }
        if raw_target not in accepted_targets:
            raise ValueError(f"{path}: unsupported prediction target {raw_target!r}")
        target = str(raw_target)
        raw_status = raw.get("artifact_status")
        artifact_status = str(raw_status) if raw_status is not None else None
        if artifact_status is None:
            report_metadata = raw.get("report_metadata")
            if (
                isinstance(report_metadata, Mapping) and
                report_metadata.get("frozen_test") is False
            ):
                artifact_status = "exploratory_not_frozen"
        raw_bound_contract = raw.get("lower_bound_contract", {})
        if not isinstance(raw_bound_contract, Mapping):
            raise ValueError(f"{path}: lower_bound_contract must be an object")
        lower_bound_contract = dict(raw_bound_contract)
        report_provenance = raw.get("provenance", raw.get("dataset_provenance", {}))
        provenance = (
            dict(report_provenance)
            if isinstance(report_provenance, Mapping) else {}
        )
        if schema_version == "compiled-ii-model-artifact-v2":
            _validate_training_canonical_dfg_identity(provenance, path)
    model = validate_model(candidate)
    actual_sha256 = canonical_model_sha256(model)
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise ValueError(
            f"{path}: trained_full_model_sha256 does not match model contents"
        )
    return LoadedModel(
        model=model,
        model_sha256=actual_sha256,
        source_path=path.resolve(),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        container=container,
        target=target,
        artifact_status=artifact_status,
        lower_bound_contract=lower_bound_contract,
        provenance=provenance,
    )


def _agree(values: Sequence[Tuple[str, float]], description: str) -> float:
    first_name, first_value = values[0]
    for name, value in values[1:]:
        if value != first_value:
            raise ValueError(
                f"{description} disagrees between {first_name}={first_value} "
                f"and {name}={value}"
            )
    return first_value


def parse_prediction_sample(
    row: Mapping[str, Any], feature_names: Sequence[str], source: str,
) -> PredictionSample:
    """Parse one unlabeled input and enforce lower-bound consistency."""
    forbidden_requested = sorted(
        set(feature_names).intersection(FORBIDDEN_MODEL_FEATURE_NAMES)
    )
    if forbidden_requested:
        raise ValueError(
            f"{source}: model features contain labels or lower-bound fields: "
            f"{forbidden_requested}"
        )
    if not isinstance(row, Mapping):
        raise ValueError(f"{source}: prediction sample must be an object")
    if "compiled_ii" in row:
        raise ValueError(
            f"{source}: compiled_ii is a label and is not allowed in prediction input"
        )
    sample_id = row.get("sample_id", row.get("index"))
    if sample_id in (None, ""):
        raise ValueError(f"{source}: sample_id is required")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: metadata must be an object")
    nested = row.get("features", {})
    if not isinstance(nested, Mapping):
        raise ValueError(f"{source}: features must be an object")
    if "compiled_ii" in nested or "compiled_ii" in metadata:
        raise ValueError(
            f"{source}: compiled_ii is a label and is not allowed anywhere "
            "in prediction input"
        )
    forbidden_nested = sorted(
        set(nested).intersection(FORBIDDEN_MODEL_FEATURE_NAMES)
    )
    if forbidden_nested:
        raise ValueError(
            f"{source}: lower-bound fields must be top-level contract values, "
            f"not model features: {forbidden_nested}"
        )

    bound_values: List[Tuple[str, float]] = []
    for name in LOWER_BOUND_KEYS:
        if name in row:
            bound_values.append((name, _finite_number(row[name], f"{source}.{name}")))
    if not bound_values:
        raise ValueError(
            f"{source}: provide one authoritative lower_bound/proven_lower_bound"
        )
    lower_bound = _agree(bound_values, f"{source} lower bound")
    if lower_bound < 1.0 or not lower_bound.is_integer():
        raise ValueError(f"{source}: lower bound must be a positive integer")

    components: Dict[str, float] = {}
    for component_name in PROVEN_COMPONENT_KEYS:
        component_values: List[Tuple[str, float]] = []
        if component_name in row:
            component_values.append((
                component_name,
                _finite_number(row[component_name], f"{source}.{component_name}"),
            ))
        if component_name in metadata:
            component_values.append((
                f"metadata.{component_name}",
                _finite_number(
                    metadata[component_name],
                    f"{source}.metadata.{component_name}",
                ),
            ))
        if not component_values:
            raise ValueError(
                f"{source}: rec_res_max_v1 requires rec_mii and res_mii"
            )
        component = _agree(
            component_values, f"{source} proven component {component_name}"
        )
        if component < 0.0 or not component.is_integer():
            raise ValueError(
                f"{source}: {component_name} must be a non-negative integer"
            )
        components[component_name] = component
        if component > lower_bound:
            raise ValueError(
                f"{source}: lower bound {lower_bound} is below proven "
                f"component {component_name}={component}"
            )

    lower_bound_source = row.get(
        "lower_bound_source", metadata.get("lower_bound_source", bound_values[0][0])
    )
    if str(lower_bound_source) not in {
        "rec_res_max_v1", "lower_bound", "proven_lower_bound", "baseline_lb"
    }:
        raise ValueError(
            f"{source}: lower_bound_source must be rec_res_max_v1"
        )
    lower_bound_source = "rec_res_max_v1"
    expected_bound = max(components[name] for name in PROVEN_COMPONENT_KEYS)
    if lower_bound != expected_bound:
        raise ValueError(
            f"{source}: lower bound {lower_bound} must equal "
            f"max(rec_mii,res_mii)={expected_bound}"
        )

    features: Dict[str, float] = {}
    for name in feature_names:
        occurrences: List[Tuple[str, float]] = []
        if name in nested:
            occurrences.append((
                f"features.{name}",
                _finite_number(nested[name], f"{source}.features.{name}"),
            ))
        if name in row:
            occurrences.append((
                name, _finite_number(row[name], f"{source}.{name}"),
            ))
        if not occurrences:
            raise ValueError(f"{source}: missing model feature {name}")
        value = _agree(occurrences, f"{source} feature {name}")
        features[name] = value

    return PredictionSample(
        sample_id=str(sample_id),
        lower_bound=lower_bound,
        rec_mii=components["rec_mii"],
        res_mii=components["res_mii"],
        lower_bound_source=str(lower_bound_source),
        features=features,
        metadata=dict(metadata),
    )


def load_prediction_samples(
    path: Path, feature_names: Sequence[str],
) -> Tuple[List[PredictionSample], Mapping[str, Any]]:
    """Load the prediction-only JSON schema; no compiled-II labels are accepted."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("samples"), list):
        raise ValueError(f"{path}: expected an object containing a samples array")
    samples = [
        parse_prediction_sample(row, feature_names, f"{path}: samples[{index}]")
        for index, row in enumerate(raw["samples"])
    ]
    if not samples:
        raise ValueError(f"{path}: prediction sample array is empty")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError(f"{path}: prediction sample_id values must be unique")
    provenance = raw.get("provenance", {})
    return samples, dict(provenance) if isinstance(provenance, Mapping) else {}


def _contract_warnings(
    loaded: LoadedModel, sample: PredictionSample,
) -> List[str]:
    warnings: List[str] = []
    missing = [
        name for name in (
            "source_sha256", "architecture_id", "mapper_id", "mapper_revision",
            "mapper_config",
        )
        if sample.metadata.get(name) in (None, "")
    ]
    if missing:
        warnings.append("unrecorded_prediction_identity:" + ",".join(missing))
    if (
        loaded.artifact_status is not None and
        ("exploratory" in loaded.artifact_status or
         "not_frozen" in loaded.artifact_status)
    ):
        warnings.append("model_artifact_status:" + loaded.artifact_status)
    if loaded.target == "compiled_ii_unspecified_direct_model":
        warnings.append("model_target_unspecified_direct_model")
    training_bound_sources = loaded.lower_bound_contract.get(
        "training_lower_bound_sources"
    )
    if isinstance(training_bound_sources, list) and training_bound_sources:
        expected_sources = {str(value) for value in training_bound_sources}
        if sample.lower_bound_source not in expected_sources:
            warnings.append(
                "lower_bound_source_mismatch:model=" +
                ",".join(sorted(expected_sources)) +
                f",input={sample.lower_bound_source}"
            )
    elif not loaded.lower_bound_contract:
        warnings.append("model_lower_bound_contract_unrecorded")
    neura = loaded.provenance.get("neura")
    expected_revision = neura.get("revision") if isinstance(neura, Mapping) else None
    if isinstance(neura, Mapping) and neura.get("dirty") is True:
        warnings.append("model_training_producer_dirty")
    actual_revision = sample.metadata.get("mapper_revision")
    if expected_revision and actual_revision and str(expected_revision) != str(actual_revision):
        warnings.append(
            f"mapper_revision_mismatch:model={expected_revision},input={actual_revision}"
        )
    return warnings


def predict_sample(
    loaded: LoadedModel, sample: PredictionSample,
) -> Dict[str, Any]:
    """Predict one candidate while exposing every residual post-processing step."""
    try:
        raw_residual = raw_ridge_residual_from_features(
            loaded.model, sample.features
        )
        nonnegative_residual = max(0.0, raw_residual)
        predicted_residual = constrained_predicted_residual(
            loaded.model, raw_residual
        )
        prediction = predict_compiled_ii(
            loaded.model, sample.lower_bound, sample.features,
            rec_mii=sample.rec_mii, res_mii=sample.res_mii,
        )
    except (KeyError, ValueError, OverflowError) as error:
        raise ValueError(f"sample {sample.sample_id}: {error}") from error
    result: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "lower_bound": sample.lower_bound,
        "rec_mii": sample.rec_mii,
        "res_mii": sample.res_mii,
        "lower_bound_source": sample.lower_bound_source,
        "raw_predicted_residual": raw_residual,
        "nonnegative_predicted_residual": nonnegative_residual,
        "predicted_residual": predicted_residual,
        "nonnegative_floor_applied": raw_residual < 0.0,
        "dead_zone": float(loaded.model.get("residual_dead_zone", 0.0)),
        "dead_zone_applied": (
            nonnegative_residual > 0.0 and predicted_residual == 0.0
        ),
        "predicted_compiled_ii": prediction,
        "model_features": dict(sample.features),
        "model_sha256": loaded.model_sha256,
        "warnings": _contract_warnings(loaded, sample),
    }
    if sample.metadata:
        result["metadata"] = dict(sample.metadata)
    radius = loaded.model.get("unseen_group_absolute_error_radius")
    if radius is not None:
        interval_lower = max(
            sample.lower_bound, prediction - float(radius)
        )
        interval_upper = prediction + float(radius)
        _finite_number(
            interval_lower, f"sample {sample.sample_id} interval lower"
        )
        _finite_number(
            interval_upper, f"sample {sample.sample_id} interval upper"
        )
        result.update({
            "prediction_interval_lower": interval_lower,
            "prediction_interval_upper": interval_upper,
            "prediction_interval_kind": (
                "empirical_held_out_group_max_error_quantile"
            ),
            "prediction_interval_empirical_quantile": loaded.model.get(
                "unseen_group_interval_empirical_quantile"
            ),
            "formal_interval_coverage_guarantee": False,
        })
    return result


def build_prediction_report(
    loaded: LoadedModel, samples: Sequence[PredictionSample], input_path: Path,
    input_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    predictions = [predict_sample(loaded, sample) for sample in samples]
    return {
        "schema_version": "compiled-ii-point-predictions-v1",
        "prediction_kind": "continuous_point_estimate",
        "target": loaded.target,
        "model": {
            "path": str(loaded.source_path),
            "source_sha256": loaded.source_sha256,
            "model_sha256": loaded.model_sha256,
            "container": loaded.container,
            "target": loaded.target,
            "artifact_status": loaded.artifact_status,
            "lower_bound_contract": dict(loaded.lower_bound_contract),
            "feature_names": list(loaded.model["feature_names"]),
        },
        "input": {
            "path": str(input_path.resolve()),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "provenance": dict(input_provenance),
        },
        "sample_count": len(predictions),
        "predictions": predictions,
        "semantics": {
            "formula": "lower_bound + constrained_predicted_residual",
            "integer_rounding": "not_applied",
            "mapper_invocation": "not_part_of_this_generic_inference_command",
            "prediction_is_a_proven_lower_bound": False,
            "prediction_is_a_mapping_feasibility_guarantee": False,
            "prediction_input_compiled_ii_present": False,
            "prediction_input_compiled_ii_used": False,
            "model_container_may_include_historical_training_labels": (
                loaded.container in {
                    "portable-model-report-v2", "neura-experiment-v2"
                }
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict compiled II from a trained model without labels"
    )
    parser.add_argument(
        "--model-report", "--model", dest="model_report", required=True,
        type=Path, help="Direct residual-Ridge JSON or report containing it",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Prediction-only JSON containing lower bounds and numeric features",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        loaded = load_model_artifact(args.model_report)
        samples, provenance = load_prediction_samples(
            args.input, loaded.model["feature_names"]
        )
        report = build_prediction_report(
            loaded, samples, args.input, provenance
        )
        text = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
