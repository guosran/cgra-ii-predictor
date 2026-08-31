"""Portable dataset schema for mapper-result prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


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
    "compiled_ii",
    "features",
    "metadata",
}


def _numeric_features(row: Mapping[str, Any]) -> Dict[str, float]:
    nested = row.get("features")
    if isinstance(nested, Mapping):
        return {
            str(name): float(value)
            for name, value in nested.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    return {
        str(name): float(value)
        for name, value in row.items()
        if name not in _RESERVED
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _parse_sample(row: Mapping[str, Any], source: Path) -> Sample:
    sample_id = row.get("sample_id", row.get("index"))
    group = row.get("group", row.get("family"))
    lower_bound = row.get("lower_bound", row.get("baseline_lb"))
    compiled_ii = row.get("compiled_ii")
    if sample_id is None or group is None:
        raise ValueError(f"{source}: every sample needs sample_id/index and group/family")
    if lower_bound is None or compiled_ii is None:
        raise ValueError(f"{source}: every sample needs lower_bound and compiled_ii")
    lower_bound = float(lower_bound)
    compiled_ii = float(compiled_ii)
    if compiled_ii < lower_bound:
        raise ValueError(
            f"{source}: sample {sample_id} has compiled_ii below its lower bound"
        )
    features = _numeric_features(row)
    # Lower-bound values are also available as optional model features.  Keep
    # both names so flat adapter reports remain portable without teaching the
    # generic model about a compiler-specific spelling.
    features.setdefault("lower_bound", lower_bound)
    features.setdefault("baseline_lb", lower_bound)
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: sample {sample_id} metadata must be an object")
    return Sample(
        sample_id=str(sample_id),
        group=str(group),
        lower_bound=lower_bound,
        compiled_ii=compiled_ii,
        features=features,
        metadata=dict(metadata),
    )


def load_dataset(path: Path, feature_names: Optional[Sequence[str]] = None) -> Dataset:
    """Load the portable schema or a flat Neura experiment report."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("samples"), list):
        raise ValueError(f"{path}: expected an object containing a samples array")
    samples = [_parse_sample(row, path) for row in raw["samples"]]
    if not samples:
        raise ValueError(f"{path}: dataset is empty")
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
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    return Dataset(samples=samples, feature_names=tuple(selected),
                   provenance=dict(provenance))
