"""Portable dataset schema for mapper-result prediction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


LOWER_BOUND_KEYS = ("lower_bound", "proven_lower_bound", "baseline_lb")
LOWER_BOUND_COMPONENT_NAMES = ("rec_mii", "res_mii")
FORBIDDEN_MODEL_FEATURE_NAMES = frozenset({
    "compiled_ii", *LOWER_BOUND_KEYS, *LOWER_BOUND_COMPONENT_NAMES,
})


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group: str
    lower_bound: float
    compiled_ii: float
    features: Mapping[str, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Dataset:
    samples: Sequence[Sample]
    feature_names: Sequence[str]
    provenance: Mapping[str, Any]


_RESERVED = {
    "index",
    "sample_id",
    "family",
    "group",
    "baseline_lb",
    "lower_bound",
    "proven_lower_bound",
    "rec_mii",
    "res_mii",
    "compiled_ii",
    "features",
    "metadata",
    "lineage",
    "leakage_lineage_id",
    "declared_leakage_lineage_id",
    "base_dfg_id",
    "ranking_query_id",
    "suite",
    "source_family",
    "source_kind",
    "original_lineage",
    "effective_lineage",
    "source_path",
    "source_sha256",
    "dfg_source_path",
    "dfg_source_sha256",
    "architecture_id",
    "architecture_path",
    "architecture_sha256",
    "architecture_variant",
    "candidate_id",
    "mapper_id",
    "mapper_revision",
    "mapper_config",
    "lower_bound_source",
    "training_weight_group",
    "mapped_artifact_path",
    "mapped_artifact_sha256",
    "input_report_path",
    "input_report_sha256",
    "training_stratum",
    "generator_type",
    "generator_family",
    "generator_version",
    "motif",
    "base_id",
    "base_seed",
    "operation_count",
    "canonical_dfg_sha256",
    "registers",
}

_FLAT_METADATA_FIELDS = _RESERVED.difference({
    "index", "sample_id", "family", "group", "baseline_lb", "lower_bound",
    "proven_lower_bound", "compiled_ii", "features", "metadata",
})


_ALLOWED_RESERVED_FEATURES = {
    "operation_count", "registers",
}


def _finite_number(value: Any, field: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} must be finite")
    return result


def _agree(values: Sequence[tuple], field: str, source: Path) -> float:
    first_name, first_value = values[0]
    for name, value in values[1:]:
        if value != first_value:
            raise ValueError(
                f"{source}: {field} disagrees between "
                f"{first_name}={first_value} and {name}={value}"
            )
    return first_value


def _numeric_features(
    row: Mapping[str, Any], source: Path, sample_id: str,
) -> Dict[str, float]:
    nested = row.get("features")
    if "features" in row:
        if not isinstance(nested, Mapping):
            raise ValueError(
                f"{source}: sample {sample_id} features must be an object"
            )
        forbidden = sorted(
            str(name) for name in nested
            if str(name) in _RESERVED
            and str(name) not in _ALLOWED_RESERVED_FEATURES
        )
        if forbidden:
            raise ValueError(
                f"{source}: sample {sample_id} has reserved feature(s) "
                f"{forbidden}"
            )
        return {
            str(name): _finite_number(
                value, f"sample {sample_id} feature {name}", source
            )
            for name, value in nested.items()
        }
    result: Dict[str, float] = {}
    for name, value in row.items():
        if name in _RESERVED or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[str(name)] = _finite_number(
                value, f"sample {sample_id} feature {name}", source
            )
    return result


def _parse_sample(row: Mapping[str, Any], source: Path) -> Sample:
    sample_id = row.get("sample_id", row.get("index"))
    group = row.get("group", row.get("family"))
    compiled_ii = row.get("compiled_ii")
    if sample_id is None or group is None or not str(sample_id) or not str(group):
        raise ValueError(f"{source}: every sample needs sample_id/index and group/family")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: sample {sample_id} metadata must be an object")

    bound_values = [
        (name, _finite_number(
            row[name], f"sample {sample_id} {name}", source
        ))
        for name in LOWER_BOUND_KEYS if name in row
    ]
    if not bound_values or compiled_ii is None:
        raise ValueError(f"{source}: every sample needs lower_bound and compiled_ii")
    lower_bound = _agree(
        bound_values, f"sample {sample_id} lower-bound aliases", source
    )
    compiled_ii = _finite_number(
        compiled_ii, f"sample {sample_id} compiled_ii", source
    )
    if lower_bound < 1.0 or not lower_bound.is_integer():
        raise ValueError(
            f"{source}: sample {sample_id} lower_bound must be a positive integer"
        )
    if compiled_ii < 1.0 or not compiled_ii.is_integer():
        raise ValueError(
            f"{source}: sample {sample_id} compiled_ii must be a positive integer"
        )
    if compiled_ii < lower_bound:
        raise ValueError(
            f"{source}: sample {sample_id} has compiled_ii below its lower bound"
        )

    components: Dict[str, float] = {}
    for name in LOWER_BOUND_COMPONENT_NAMES:
        values = []
        if name in row:
            values.append((name, _finite_number(
                row[name], f"sample {sample_id} {name}", source
            )))
        if name in metadata:
            values.append((f"metadata.{name}", _finite_number(
                metadata[name], f"sample {sample_id} metadata.{name}", source
            )))
        if not values:
            raise ValueError(
                f"{source}: sample {sample_id} needs rec_mii and res_mii"
            )
        value = _agree(
            values, f"sample {sample_id} {name}", source
        )
        if value < 0.0 or not value.is_integer():
            raise ValueError(
                f"{source}: sample {sample_id} {name} must be a non-negative integer"
            )
        components[name] = value
    expected_bound = max(components.values())
    if lower_bound != expected_bound:
        raise ValueError(
            f"{source}: sample {sample_id} lower_bound {lower_bound} must equal "
            f"max(rec_mii,res_mii)={expected_bound}"
        )
    lower_bound_source = row.get(
        "lower_bound_source", metadata.get("lower_bound_source")
    )
    if lower_bound_source not in (None, "rec_res_max_v1"):
        raise ValueError(
            f"{source}: sample {sample_id} lower_bound_source must be "
            "rec_res_max_v1"
        )

    features = _numeric_features(row, source, str(sample_id))
    merged_metadata = dict(metadata)
    for name in _FLAT_METADATA_FIELDS:
        if name not in row:
            continue
        if name in merged_metadata and merged_metadata[name] != row[name]:
            raise ValueError(
                f"{source}: sample {sample_id} has conflicting flat and "
                f"metadata values for {name}"
            )
        merged_metadata.setdefault(name, row[name])
    merged_metadata.setdefault("rec_mii", int(components["rec_mii"]))
    merged_metadata.setdefault("res_mii", int(components["res_mii"]))
    merged_metadata.setdefault("lower_bound_source", "rec_res_max_v1")
    base_dfg_id = merged_metadata.get("base_dfg_id")
    ranking_query_id = merged_metadata.get("ranking_query_id")
    if (
        base_dfg_id not in (None, "")
        and ranking_query_id not in (None, "")
        and base_dfg_id != ranking_query_id
    ):
        raise ValueError(
            f"{source}: sample {sample_id} has inconsistent base_dfg_id "
            "and ranking_query_id"
        )
    return Sample(
        sample_id=str(sample_id),
        group=str(group),
        lower_bound=lower_bound,
        compiled_ii=compiled_ii,
        features=features,
        metadata=merged_metadata,
    )


def load_dataset(path: Path, feature_names: Optional[Sequence[str]] = None) -> Dataset:
    """Load the portable schema or a flat Neura experiment report."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("samples"), list):
        raise ValueError(f"{path}: expected an object containing a samples array")
    samples = [_parse_sample(row, path) for row in raw["samples"]]
    if not samples:
        raise ValueError(f"{path}: dataset is empty")
    seen_sample_ids = set()
    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"{path}: duplicate sample_id {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)
    selected = feature_names
    if selected is None:
        declared = raw.get("model_feature_names", raw.get("feature_names"))
        if isinstance(declared, list):
            selected = [str(name) for name in declared]
        else:
            selected = sorted(set.intersection(
                *(set(sample.features) for sample in samples)
            ))
    assert selected is not None
    if len(set(selected)) != len(selected) or any(not name for name in selected):
        raise ValueError(f"{path}: feature names must be unique and non-empty")
    forbidden_selected = sorted(
        name for name in selected if name in FORBIDDEN_MODEL_FEATURE_NAMES
    )
    if forbidden_selected:
        raise ValueError(
            f"{path}: lower-bound fields and labels cannot be model features: "
            f"{forbidden_selected}"
        )
    missing = {
        sample.sample_id: sorted(set(selected).difference(sample.features))
        for sample in samples
        if set(selected).difference(sample.features)
    }
    if missing:
        first_id = next(iter(missing))
        raise ValueError(
            f"{path}: sample {first_id} lacks features {missing[first_id]}"
        )
    # Candidate labels are scoped by one exact base-DFG/ranking query.  An
    # architecture-style candidate name may legitimately recur for another
    # DFG, while exact repeated observations inside one query are harmless and
    # are collapsed by the model.  Only contradictory scoped records fail.
    scoped_candidates: Dict[tuple, tuple] = {}
    provenance_fields = (
        "architecture_id", "architecture_variant", "mapper_id",
        "mapper_revision", "mapper_config", "source_sha256",
        "dfg_source_sha256", "canonical_dfg_sha256",
    )
    for sample in samples:
        query = sample.metadata.get(
            "ranking_query_id", sample.metadata.get("base_dfg_id")
        )
        candidate = sample.metadata.get("candidate_id")
        if query in (None, "") or candidate in (None, ""):
            continue
        key = (str(query), str(candidate))
        fingerprint = (
            sample.group,
            sample.lower_bound,
            sample.compiled_ii,
            tuple((name, float(sample.features[name])) for name in selected),
            tuple(
                (name, json.dumps(sample.metadata.get(name), sort_keys=True))
                for name in provenance_fields
            ),
        )
        prior = scoped_candidates.setdefault(key, fingerprint)
        if prior != fingerprint:
            raise ValueError(
                f"{path}: conflicting records for ranking query/candidate "
                f"{key[0]}/{key[1]}"
            )
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    return Dataset(samples=samples, feature_names=tuple(selected),
                   provenance=dict(provenance))


def remap_groups(dataset: Dataset, aliases: Mapping[str, str]) -> Dataset:
    """Return a dataset whose leakage groups follow an explicit lineage map."""
    if any(not source or not target for source, target in aliases.items()):
        raise ValueError("group aliases must have non-empty source and target")
    samples: List[Sample] = []
    for sample in dataset.samples:
        metadata = dict(sample.metadata)
        declared_lineage = str(metadata.get(
            "leakage_lineage_id", metadata.get("lineage", sample.group)
        ))
        metadata.setdefault("lineage", declared_lineage)
        metadata.setdefault("declared_leakage_lineage_id", declared_lineage)
        group = aliases.get(
            sample.group, aliases.get(declared_lineage, declared_lineage)
        )
        metadata["effective_lineage"] = group
        metadata["leakage_lineage_id"] = group
        if group != sample.group:
            metadata.setdefault("source_group", sample.group)
        samples.append(Sample(
            sample_id=sample.sample_id,
            group=group,
            lower_bound=sample.lower_bound,
            compiled_ii=sample.compiled_ii,
            features=sample.features,
            metadata=metadata,
        ))
    return Dataset(
        samples=tuple(samples),
        feature_names=dataset.feature_names,
        provenance=dataset.provenance,
    )


def regroup_by_metadata(dataset: Dataset, metadata_key: str) -> Dataset:
    """Use one metadata identity for holdout while retaining lineage weights."""
    if not metadata_key:
        raise ValueError("holdout metadata key must be non-empty")
    samples: List[Sample] = []
    for sample in dataset.samples:
        if sample.metadata.get(metadata_key) in (None, ""):
            raise ValueError(
                f"sample {sample.sample_id} lacks metadata.{metadata_key}"
            )
        metadata = dict(sample.metadata)
        metadata.setdefault("training_weight_group", sample.group)
        samples.append(Sample(
            sample_id=sample.sample_id,
            group=str(metadata[metadata_key]),
            lower_bound=sample.lower_bound,
            compiled_ii=sample.compiled_ii,
            features=sample.features,
            metadata=metadata,
        ))
    return Dataset(
        samples=tuple(samples),
        feature_names=dataset.feature_names,
        provenance=dataset.provenance,
    )
