"""Strict loader for the frozen per-CGRA 2x2 predictor package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .mapper_model import (
    DirectMapperIIEnsemble,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_names,
)
from .shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID


CHECKPOINT_SCHEMA = "cgra-ii-direct-mapper-ensemble"
EXPECTED_ARCHITECTURE_SHA256 = "6f4a9a1815dcc0d97c00fd6ee20424fa9420ace90654da29e70d283cba7a611f"
COMPACT_INDICES = tuple(range(65, 79)) + tuple(range(93, 118)) + tuple(range(134, 156))
COMPACT_FEATURE_NAMES = tuple(MAPPER_FEATURE_NAMES[index] for index in COMPACT_INDICES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
    checkpoint_path: Path,
    *,
    expected_architecture_sha256: str = EXPECTED_ARCHITECTURE_SHA256,
    expected_checkpoint_sha256: str | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, MapperModelConfig, dict[str, Any]]:
    """Load only the 4-member, 2x2 compact61 model contract."""
    checkpoint_path = checkpoint_path.resolve()
    if expected_checkpoint_sha256 and sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs from frozen metadata")
    artifact = torch.load(
        checkpoint_path,
        map_location=device or torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(artifact, dict) or artifact.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is not the frozen 2x2 ensemble")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is missing")
    config = MapperModelConfig(**raw_config).validate()
    if config.shape_protocol != SHAPE_PROTOCOL_2X2_ID:
        raise ValueError("checkpoint uses a different shape protocol")
    if tuple(config.enabled_feature_names or ()) != COMPACT_FEATURE_NAMES:
        raise ValueError("checkpoint compact61 input mask changed")
    if artifact.get("feature_names") != list(mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)):
        raise ValueError("checkpoint full 148-feature roster changed")
    if artifact.get("compact_feature_names") != list(COMPACT_FEATURE_NAMES):
        raise ValueError("checkpoint compact feature names changed")
    if artifact.get("ensemble_member_count") != 4 or artifact.get("ensemble_reduction") != "arithmetic_mean":
        raise ValueError("checkpoint ensemble contract changed")
    if artifact.get("ensemble_seeds") != [17, 41, 113, 239]:
        raise ValueError("checkpoint ensemble seed roster changed")
    if artifact.get("architecture_sha256") != expected_architecture_sha256:
        raise ValueError("checkpoint architecture identity changed")
    if artifact.get("supported_mapper_shapes") != [list(shape) for shape in SHAPE_PROTOCOL_2X2.mapper_shapes]:
        raise ValueError("checkpoint shape roster changed")
    model = DirectMapperIIEnsemble(4, config).to(device or torch.device("cpu"))
    model.load_state_dict(artifact.get("state_dict", {}), strict=True)
    model.eval()
    return model, config, artifact


def load_published_candidate(
    package_dir: Path,
    *,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, MapperModelConfig, dict[str, Any]]:
    """Verify package sidecars and load the byte-pinned original C0 weights."""
    package_dir = package_dir.resolve()
    metadata = json.loads((package_dir / "model.json").read_text())
    report_path = package_dir / metadata["training"]["report_path"]
    architecture_path = package_dir / metadata["architecture"]["path"]
    checkpoint_path = package_dir / metadata["checkpoint"]["path"]
    checked_paths = (
        (report_path, "report_sha256", metadata["training"]["report_sha256"]),
        (architecture_path, "architecture_sha256", metadata["architecture"]["sha256"]),
        (package_dir / metadata["training"]["source_groups_path"], "source_groups_sha256", metadata["training"]["source_groups_sha256"]),
        (package_dir / metadata["training"]["exclusions_path"], "exclusions_sha256", metadata["training"]["exclusions_sha256"]),
    )
    for path, key, expected in checked_paths:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"candidate {key} does not match its sidecar")
    model, config, artifact = load_checkpoint(
        checkpoint_path,
        expected_architecture_sha256=metadata["architecture"]["sha256"],
        expected_checkpoint_sha256=metadata["checkpoint"]["sha256"],
        device=device,
    )
    report = json.loads(report_path.read_text())
    if (metadata["architecture"]["sha256"] != EXPECTED_ARCHITECTURE_SHA256 or
            artifact.get("training_manifest_sha256") != report["provenance"]["source_manifest_sha256"] or
            artifact.get("source_groups_sha256") != metadata["training"]["source_groups_sha256"] or
            artifact.get("training_exclusions_sha256") != metadata["training"]["exclusions_sha256"]):
        raise ValueError("candidate architecture or original training binding changed")
    return model, config, metadata
