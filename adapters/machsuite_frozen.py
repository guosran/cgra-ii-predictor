#!/usr/bin/env python3
"""Leakage-resistant random-train / frozen-MachSuite evaluation workflow.

The three evaluation phases are deliberately separate:

* ``preflight`` lowers every predeclared MachSuite variant and runs only the
  RecMII/ResMII analysis pass.  It never invokes the mapper and emits no label.
* ``predict`` loads an already frozen random-DFG model, writes predictions,
  and seals the model/input/prediction hashes.
* ``reveal`` verifies that seal before invoking the unchanged heuristic mapper
  on the exact lowered artifacts from preflight.

Unsupported frontend cases remain in the denominator as censored records.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
DEFAULT_INVENTORY = PROJECT_ROOT / "benchmarks" / "machsuite-v1.json"
DEFAULT_SUITE_ROOT = PROJECT_ROOT / "third_party" / "machsuite"
DEFAULT_NEURA_ROOT = PROJECT_ROOT / "third_party" / "neura"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from adapters import neura_experiment, neura_motifs  # noqa: E402
from cgra_ii_predictor.predict import (  # noqa: E402
    build_prediction_report,
    canonical_model_sha256,
    load_model_artifact,
    load_prediction_samples,
)


PREFLIGHT_SCHEMA = "machsuite-frozen-preflight-v1"
FROZEN_MODEL_SCHEMA = "compiled-ii-model-artifact-v1"
PREDICTION_SEAL_SCHEMA = "machsuite-prediction-seal-v1"
REVEALED_LABEL_SCHEMA = "machsuite-revealed-labels-v1"
EVALUATION_SCHEMA = "machsuite-frozen-evaluation-v1"
FROZEN_MODEL_STATUS = "frozen_before_machsuite_reveal"
SMOKE_MODEL_STATUS = "smoke_only_not_for_frozen_evaluation"
DEFAULT_MINIMUM_BASE_DFGS = 1000
DEFAULT_MINIMUM_GENERATOR_FAMILIES = 6
FROZEN_RIDGE_CANDIDATES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
FROZEN_DEAD_ZONE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
FROZEN_INTERVAL_QUANTILE = 0.9
FROZEN_MOTIFS = tuple(neura_motifs.DEFAULT_MOTIFS)
FROZEN_MOTIF_SHAPES = ("3x3", "3x4", "4x4")
FROZEN_ARCHITECTURE_VARIANTS = ("homogeneous", "split-domain")
FROZEN_MINIMUM_SAMPLES_PER_FAMILY = 200
FROZEN_SUITE_NAME = "MachSuite"
FROZEN_SUITE_REPOSITORY = "https://github.com/breagen/MachSuite.git"
FROZEN_SUITE_REVISION = "6236e593012cb86b0d2f08d9fb9ba0411ff989b4"
FROZEN_NEURA_REVISION = "47b7e3a68c321075293e6fcb45fb3b1cabb93b88"
FROZEN_ARCHITECTURE_ROWS = 4
FROZEN_ARCHITECTURE_COLUMNS = 4
LOWER_BOUND_CONTRACT = {
    "name": "rec_res_max_v1",
    "formula": "max(rec_mii,res_mii)",
    "training_lower_bound_sources": ["rec_res_max_v1"],
    "components_are_model_features": False,
}
PREDICTION_RECORD_FEATURE_NAMES = tuple(
    name for name in neura_experiment.FEATURE_NAMES
    if name not in {"baseline_lb", "rec_mii", "res_mii"}
)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    source: str
    top: str
    leakage_lineage_id: str


# This tuple, rather than a user-supplied JSON length, defines the frozen test
# population.  ``--inventory`` may point at a relocated byte-for-byte/equivalent
# inventory, but it cannot silently shrink or otherwise redefine the suite.
FROZEN_BENCHMARK_SPECS = tuple(BenchmarkSpec(*values) for values in (
    ("aes/aes", "aes/aes/aes.c", "aes256_encrypt_ecb", "machsuite/aes"),
    ("backprop/backprop", "backprop/backprop/backprop.c", "backprop", "machsuite/backprop"),
    ("bfs/bulk", "bfs/bulk/bfs.c", "bfs", "machsuite/bfs"),
    ("bfs/queue", "bfs/queue/bfs.c", "bfs", "machsuite/bfs"),
    ("fft/strided", "fft/strided/fft.c", "fft", "machsuite/fft"),
    ("fft/transpose", "fft/transpose/fft.c", "fft1D_512", "machsuite/fft"),
    ("gemm/blocked", "gemm/blocked/gemm.c", "bbgemm", "machsuite/gemm"),
    ("gemm/ncubed", "gemm/ncubed/gemm.c", "gemm", "machsuite/gemm"),
    ("kmp/kmp", "kmp/kmp/kmp.c", "kmp", "machsuite/kmp"),
    ("md/grid", "md/grid/md.c", "md", "machsuite/md"),
    ("md/knn", "md/knn/md.c", "md_kernel", "machsuite/md"),
    ("nw/nw", "nw/nw/nw.c", "needwun", "machsuite/nw"),
    ("sort/merge", "sort/merge/sort.c", "ms_mergesort", "machsuite/sort"),
    ("sort/radix", "sort/radix/sort.c", "ss_sort", "machsuite/sort"),
    ("spmv/crs", "spmv/crs/spmv.c", "spmv", "machsuite/spmv"),
    ("spmv/ellpack", "spmv/ellpack/spmv.c", "ellpack", "machsuite/spmv"),
    ("stencil/stencil2d", "stencil/stencil2d/stencil.c", "stencil", "machsuite/stencil"),
    ("stencil/stencil3d", "stencil/stencil3d/stencil.c", "stencil3d", "machsuite/stencil"),
    ("viterbi/viterbi", "viterbi/viterbi/viterbi.c", "viterbi", "machsuite/viterbi"),
))

PROTOCOL_IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    (PROJECT_ROOT / "adapters" / "neura_experiment.py").resolve(),
    (PROJECT_ROOT / "adapters" / "neura_motifs.py").resolve(),
    (SOURCE_ROOT / "cgra_ii_predictor" / "predict.py").resolve(),
    (SOURCE_ROOT / "cgra_ii_predictor" / "model.py").resolve(),
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = neura_experiment.file_sha256(path)
    if digest is None:
        raise ValueError(f"missing required file: {path}")
    return digest


def protocol_implementation_identity() -> Mapping[str, Any]:
    """Hash every local source file that can change the frozen prediction."""
    files = {
        str(path.relative_to(PROJECT_ROOT)): raw_sha256(path)
        for path in PROTOCOL_IMPLEMENTATION_FILES
    }
    return {
        "files": files,
        "combined_sha256": canonical_json_sha256(files),
    }


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def recursively_contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(
            recursively_contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(recursively_contains_key(item, forbidden) for item in value)
    return False


def load_inventory(path: Path) -> Tuple[Mapping[str, Any], Tuple[BenchmarkSpec, ...]]:
    raw = read_json(path)
    if raw.get("schema_version") != "machsuite-inventory-v1":
        raise ValueError(f"{path}: unsupported inventory schema")
    expected_header = {
        "suite": FROZEN_SUITE_NAME,
        "repository": FROZEN_SUITE_REPOSITORY,
        "revision": FROZEN_SUITE_REVISION,
        "declared_count": len(FROZEN_BENCHMARK_SPECS),
    }
    for name, expected in expected_header.items():
        if raw.get(name) != expected:
            raise ValueError(
                f"{path}: frozen {name} must be {expected!r}, "
                f"found {raw.get(name)!r}"
            )
    architecture = raw.get("primary_architecture")
    expected_architecture = {
        "rows": FROZEN_ARCHITECTURE_ROWS,
        "columns": FROZEN_ARCHITECTURE_COLUMNS,
        "mapper_id": "neura-heuristic",
        "mapper_config": "mapping-strategy=heuristic",
    }
    if architecture != expected_architecture:
        raise ValueError(f"{path}: frozen primary_architecture changed")
    entries = raw.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: benchmarks must be a non-empty array")
    specs: List[BenchmarkSpec] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: benchmarks[{index}] must be an object")
        values = [
            entry.get("id"), entry.get("source"), entry.get("top"),
            entry.get("leakage_lineage_id"),
        ]
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{path}: incomplete benchmark at index {index}")
        source = Path(str(values[1]))
        if source.is_absolute() or ".." in source.parts:
            raise ValueError(f"{path}: unsafe source path {source}")
        specs.append(BenchmarkSpec(*[str(value) for value in values]))
    if len({spec.benchmark_id for spec in specs}) != len(specs):
        raise ValueError(f"{path}: duplicate benchmark id")
    if int(raw.get("declared_count", -1)) != len(specs):
        raise ValueError(f"{path}: declared_count does not match benchmarks")
    if tuple(specs) != FROZEN_BENCHMARK_SPECS:
        raise ValueError(
            f"{path}: benchmark identities/order differ from frozen MachSuite v1"
        )
    return raw, tuple(specs)


def resolve_executable(value: Path) -> Path:
    candidate = value.expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(str(value))
    if resolved is None:
        raise ValueError(f"executable not found: {value}")
    return Path(resolved).resolve()


def tool_identity(path: Path) -> Mapping[str, Any]:
    completed = subprocess.run(
        (str(path), "--version"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    first_line = completed.stdout.splitlines()[0] if completed.stdout else None
    return {
        "path": str(path),
        "sha256": raw_sha256(path),
        "version_first_line": first_line,
        "version_returncode": completed.returncode,
    }


def require_clean_revision(root: Path, expected: Optional[str], name: str) -> Mapping[str, Any]:
    state = neura_experiment.git_provenance(root)
    revision = state.get("revision")
    if expected is not None and revision != expected:
        raise ValueError(
            f"{name} revision mismatch: expected {expected}, found {revision}"
        )
    if state.get("dirty") is not False:
        raise ValueError(f"{name} checkout must be clean for a frozen run")
    return state


def dependency_hashes(suite_root: Path, source: Path) -> Mapping[str, str]:
    dependencies = [source]
    dependencies.extend(sorted(source.parent.glob("*.h")))
    common_header = suite_root / "common" / "support.h"
    if common_header.is_file():
        dependencies.append(common_header)
    result: Dict[str, str] = {}
    for dependency in dependencies:
        relative = str(dependency.resolve().relative_to(suite_root.resolve()))
        result[relative] = raw_sha256(dependency)
    return dict(sorted(result.items()))


def bounded_stderr(stream: Any) -> str:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size <= 4000:
        stream.seek(0)
        payload = stream.read()
    else:
        stream.seek(0)
        head = stream.read(1200)
        stream.seek(max(0, size - 2800))
        payload = head + b"\n... stderr middle omitted ...\n" + stream.read(2800)
    return payload.decode(errors="replace")


def run_stage(stage: str, command: Sequence[str], timeout: int) -> Mapping[str, Any]:
    with tempfile.TemporaryFile() as stderr_stream:
        try:
            completed = subprocess.run(
                tuple(command), stdout=subprocess.DEVNULL, stderr=stderr_stream,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False, "stage": stage, "status": "timeout",
                "timeout_seconds": timeout, "command": list(command),
                "stderr": bounded_stderr(stderr_stream),
            }
        stderr = bounded_stderr(stderr_stream)
    return {
        "ok": completed.returncode == 0,
        "stage": stage,
        "status": "success" if completed.returncode == 0 else "nonzero_exit",
        "returncode": completed.returncode,
        "timeout_seconds": timeout,
        "command": list(command),
        "stderr": stderr,
    }


def preflight_commands(
    spec: BenchmarkSpec, suite_root: Path, candidate_dir: Path,
    clang: Path, llvm_extract: Path, mlir_translate: Path, opt: Path,
    architecture: Path, rows: int, columns: int,
) -> Tuple[Tuple[str, Tuple[str, ...], Path], ...]:
    source = suite_root / spec.source
    full_ir = candidate_dir / "full.ll"
    kernel_ir = candidate_dir / "kernel.ll"
    imported = candidate_dir / "imported.mlir"
    lowered = candidate_dir / "lowered.mlir"
    cost = candidate_dir / "cost.mlir"
    commands = (
        (
            "compile",
            (
                str(clang), "-S", "-emit-llvm", "-O3", "-fno-vectorize",
                "-fno-slp-vectorize", "-fno-unroll-loops", "-Xclang",
                "-disable-lifetime-markers", "-std=c11", "-I",
                str(source.parent), "-I", str(suite_root / "common"), "-o",
                str(full_ir), str(source),
            ),
            full_ir,
        ),
        (
            "extract",
            (
                str(llvm_extract), "--recursive", f"--func={spec.top}",
                str(full_ir), "-S", "-o", str(kernel_ir),
            ),
            kernel_ir,
        ),
        (
            "import",
            (
                str(mlir_translate), "--import-llvm", str(kernel_ir), "-o",
                str(imported),
            ),
            imported,
        ),
        (
            "lower",
            (
                str(opt), str(imported), "--assign-accelerator",
                "--lower-llvm-to-neura", "--promote-input-arg-to-const",
                "--fold-constant", "--canonicalize-return",
                "--canonicalize-live-in", "--leverage-predicated-value",
                "--transform-ctrl-to-data-flow", "--fold-constant",
                "--insert-data-mov", "-o", str(lowered),
            ),
            lowered,
        ),
        (
            "rec_res_analysis",
            (
                str(opt), str(lowered), f"--architecture-spec={architecture}",
                f"--analyze-rec-res-mii=x-tiles={columns} y-tiles={rows}",
                "-o", str(cost),
            ),
            cost,
        ),
    )
    if any("--map-to-accelerator" in token for _, command, _ in commands for token in command):
        raise AssertionError("preflight command construction included the mapper")
    return commands


def manifest_summary(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    return {
        "declared_count": len(candidates),
        "ready_count": sum(item.get("status") == "ready" for item in candidates),
        "censored_count": sum(item.get("status") == "censored" for item in candidates),
        "pending_count": sum(item.get("status") == "declared" for item in candidates),
    }


def _manifest_artifact(manifest_path: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe {label} path: {relative}")
    root = manifest_path.parent.resolve()
    resolved = (root / candidate).resolve()
    if root not in resolved.parents:
        raise ValueError(f"{label} escapes the preflight directory")
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    return resolved


def _validate_ready_candidate(
    candidate: Mapping[str, Any], sample: Mapping[str, Any],
    spec: BenchmarkSpec, manifest: Mapping[str, Any], manifest_path: Path,
) -> None:
    architecture = manifest["architecture"]
    rows = int(architecture["rows"])
    columns = int(architecture["columns"])
    shape = f"{rows}x{columns}"
    expected_sample_id = f"machsuite/{spec.benchmark_id}/{shape}"
    if candidate.get("sample_id") != expected_sample_id:
        raise ValueError(f"{spec.benchmark_id}: unexpected sample_id")
    if sample.get("sample_id") != expected_sample_id:
        raise ValueError(f"{spec.benchmark_id}: sample/candidate ID mismatch")

    lowered = _manifest_artifact(
        manifest_path, candidate.get("lowered_artifact_path"),
        f"{spec.benchmark_id} lowered artifact",
    )
    cost = _manifest_artifact(
        manifest_path, candidate.get("cost_artifact_path"),
        f"{spec.benchmark_id} cost artifact",
    )
    if raw_sha256(lowered) != candidate.get("lowered_artifact_sha256"):
        raise ValueError(f"{spec.benchmark_id}: lowered artifact hash changed")
    if raw_sha256(cost) != candidate.get("cost_artifact_sha256"):
        raise ValueError(f"{spec.benchmark_id}: cost artifact hash changed")

    lowered_text = lowered.read_text()
    canonical_dfg = neura_motifs.canonical_dfg_sha256(lowered_text)
    if canonical_dfg != candidate.get("canonical_dfg_sha256"):
        raise ValueError(f"{spec.benchmark_id}: canonical DFG identity changed")
    values = neura_experiment.parse_cost_features(cost.read_text())
    if values is None:
        raise ValueError(f"{spec.benchmark_id}: Rec/Res facts are unavailable")
    recomputed: Dict[str, Any] = dict(values)
    recomputed.update(neura_experiment.graph_features_from_neura(
        lowered_text, rows, columns
    ))
    neura_experiment.add_prediction_features(recomputed)
    expected_features = {
        name: recomputed[name] for name in PREDICTION_RECORD_FEATURE_NAMES
    }
    if sample.get("features") != expected_features:
        raise ValueError(f"{spec.benchmark_id}: prediction features were modified")

    bound = max(int(recomputed["rec_mii"]), int(recomputed["res_mii"]))
    for container_name, container in (("candidate", candidate), ("sample", sample)):
        if int(container.get("lower_bound", -1)) != bound:
            raise ValueError(
                f"{spec.benchmark_id}: {container_name} lower bound changed"
            )
        if int(container.get("rec_mii", -1)) != int(recomputed["rec_mii"]):
            raise ValueError(f"{spec.benchmark_id}: RecMII changed")
        if int(container.get("res_mii", -1)) != int(recomputed["res_mii"]):
            raise ValueError(f"{spec.benchmark_id}: ResMII changed")

    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{spec.benchmark_id}: prediction metadata is missing")
    neura_revision = manifest["neura_provenance"]["revision"]
    architecture_sha256 = architecture["sha256"]
    candidate_identity = {
        "base_dfg_id": canonical_dfg,
        "architecture_sha256": architecture_sha256,
        "architecture_variant": shape,
        "mapper_id": "neura-heuristic",
        "mapper_revision": neura_revision,
        "mapper_config": "mapping-strategy=heuristic",
    }
    candidate_id = canonical_json_sha256(candidate_identity)
    expected_metadata = {
        "suite": FROZEN_SUITE_NAME,
        "suite_revision": FROZEN_SUITE_REVISION,
        "kernel_id": spec.benchmark_id,
        "leakage_lineage_id": spec.leakage_lineage_id,
        "source_sha256": candidate.get("source_bundle_sha256"),
        "benchmark_source_sha256": candidate.get("source_sha256"),
        "dfg_source_sha256": candidate.get("lowered_artifact_sha256"),
        "canonical_dfg_sha256": canonical_dfg,
        "base_dfg_id": canonical_dfg,
        "ranking_query_id": canonical_dfg,
        "architecture_sha256": architecture_sha256,
        "architecture_variant": shape,
        "architecture_id": f"{architecture_sha256}:{shape}",
        "candidate_id": candidate_id,
        "mapper_id": "neura-heuristic",
        "mapper_revision": neura_revision,
        "mapper_config": "mapping-strategy=heuristic",
        "lower_bound_source": "rec_res_max_v1",
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"{spec.benchmark_id}: metadata {name} changed")
    if candidate.get("candidate_id") != candidate_id:
        raise ValueError(f"{spec.benchmark_id}: candidate identity changed")


def validate_frozen_manifest(
    manifest: Mapping[str, Any], manifest_path: Path,
) -> None:
    """Re-derive every prediction input from the sealed preflight artifacts."""
    if manifest.get("schema_version") != PREFLIGHT_SCHEMA:
        raise ValueError("not a frozen MachSuite preflight manifest")
    if manifest.get("protocol_status") != "preflight_complete":
        raise ValueError("preflight is not complete")
    if manifest.get("labels_accessed") is not False:
        raise ValueError("preflight does not attest labels_accessed=false")
    if manifest.get("compiled_ii_present") is not False:
        raise ValueError("preflight does not attest compiled_ii_present=false")
    if recursively_contains_key(manifest, "compiled_ii"):
        raise ValueError("preflight contains a forbidden compiled_ii label")
    if manifest.get("lower_bound_contract") != LOWER_BOUND_CONTRACT:
        raise ValueError("preflight lower-bound contract changed")

    inventory_record = manifest.get("inventory")
    if not isinstance(inventory_record, Mapping):
        raise ValueError("preflight inventory identity is missing")
    inventory_path = Path(str(inventory_record.get("path", ""))).resolve()
    inventory, specs = load_inventory(inventory_path)
    if raw_sha256(inventory_path) != inventory_record.get("sha256"):
        raise ValueError("frozen inventory file changed after preflight")
    expected_inventory = {
        "suite": inventory["suite"],
        "repository": inventory["repository"],
        "revision": inventory["revision"],
    }
    for name, expected in expected_inventory.items():
        if inventory_record.get(name) != expected:
            raise ValueError(f"preflight inventory {name} changed")

    architecture = manifest.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("preflight architecture identity is missing")
    if (
        int(architecture.get("rows", -1)) != FROZEN_ARCHITECTURE_ROWS or
        int(architecture.get("columns", -1)) != FROZEN_ARCHITECTURE_COLUMNS
    ):
        raise ValueError("frozen MachSuite protocol requires the 4x4 architecture")
    architecture_path = Path(str(architecture.get("path", ""))).resolve()
    if raw_sha256(architecture_path) != architecture.get("sha256"):
        raise ValueError("architecture changed after preflight")
    expected_architecture_id = (
        f"{architecture['sha256']}:"
        f"{FROZEN_ARCHITECTURE_ROWS}x{FROZEN_ARCHITECTURE_COLUMNS}"
    )
    if architecture.get("architecture_id") != expected_architecture_id:
        raise ValueError("architecture identity changed")

    implementation = manifest.get("protocol_implementation")
    if implementation != protocol_implementation_identity():
        raise ValueError("frozen protocol implementation changed after preflight")
    predictor = manifest.get("predictor_provenance")
    if not isinstance(predictor, Mapping) or predictor.get("dirty") is not False:
        raise ValueError("preflight was not produced by a clean predictor checkout")
    neura = manifest.get("neura_provenance")
    if (
        not isinstance(neura, Mapping) or
        neura.get("revision") != FROZEN_NEURA_REVISION
    ):
        raise ValueError("preflight Neura revision differs from the frozen protocol")
    if neura.get("dirty") is not False:
        raise ValueError("preflight Neura checkout was dirty")

    candidates = manifest.get("candidates")
    samples = manifest.get("samples")
    if not isinstance(candidates, list) or len(candidates) != len(specs):
        raise ValueError("preflight must contain all 19 frozen candidates")
    if not isinstance(samples, list):
        raise ValueError("preflight samples array is missing")
    ready_samples: Dict[str, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"preflight samples[{index}] is not an object")
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id or sample_id in ready_samples:
            raise ValueError("preflight sample IDs must be nonempty and unique")
        ready_samples[sample_id] = sample

    expected_ready_ids: List[str] = []
    for candidate, spec in zip(candidates, specs):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{spec.benchmark_id}: candidate is not an object")
        identity = (
            candidate.get("benchmark_id"), candidate.get("source"),
            candidate.get("top"), candidate.get("leakage_lineage_id"),
        )
        expected_identity = (
            spec.benchmark_id, spec.source, spec.top, spec.leakage_lineage_id,
        )
        if identity != expected_identity:
            raise ValueError(f"{spec.benchmark_id}: frozen candidate identity changed")
        status = candidate.get("status")
        if status == "ready":
            sample_id = str(candidate.get("sample_id", ""))
            if sample_id not in ready_samples:
                raise ValueError(f"{spec.benchmark_id}: ready sample is missing")
            _validate_ready_candidate(
                candidate, ready_samples[sample_id], spec, manifest, manifest_path
            )
            expected_ready_ids.append(sample_id)
        elif status != "censored":
            raise ValueError(f"{spec.benchmark_id}: incomplete preflight status")
    if list(ready_samples) != expected_ready_ids:
        raise ValueError("preflight samples do not match ready candidate order")
    if manifest.get("summary") != manifest_summary(candidates):
        raise ValueError("preflight summary does not match candidate states")


def _validated_generated_sample(
    sample: Mapping[str, Any], index: int, neura_revision: str,
) -> Tuple[str, str]:
    label = f"training sample {index}"
    required_strings = (
        "base_dfg_id", "canonical_dfg_sha256", "generator_family",
        "generator_type", "generator_version", "motif", "base_id",
        "candidate_id", "source_path", "source_sha256", "architecture_path",
        "architecture_sha256", "architecture_variant", "mapped_artifact_path",
        "mapped_artifact_sha256", "leakage_lineage_id", "lower_bound_source",
        "mapper_id", "mapper_revision", "mapper_config", "cost_artifact_path",
        "cost_artifact_sha256", "rec_res_evidence",
    )
    for name in required_strings:
        if not isinstance(sample.get(name), str) or not sample[name]:
            raise ValueError(f"{label}: missing {name}")
    if sample.get("training_stratum") != "generated":
        raise ValueError("frozen primary model requires generated-only training")
    if sample.get("source_kind") != "generated":
        raise ValueError(f"{label}: source_kind must be generated")
    if sample.get("rec_res_evidence") != "neura_shared_rec_res_analysis_v1":
        raise ValueError(f"{label}: Rec/Res evidence is not compiler-derived")
    motif = str(sample["motif"])
    if motif not in neura_motifs.DEFAULT_MOTIFS:
        raise ValueError(f"{label}: generator motif is not predeclared")
    expected_family = f"generated/motif/{motif}"
    if sample["generator_family"] != expected_family:
        raise ValueError(f"{label}: generator_family does not match motif")
    if sample["generator_type"] != "generated/motif":
        raise ValueError(f"{label}: unsupported generator_type")
    if sample["generator_version"] != neura_motifs.GENERATOR_VERSION:
        raise ValueError(f"{label}: unsupported generator_version")

    try:
        base_seed = int(sample["base_seed"])
        operation_count = int(sample["operation_count"])
        rows = int(sample["rows"])
        columns = max(1, int(sample["tiles"]) // rows)
        registers = int(sample["registers"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label}: invalid generator parameters") from error
    source_text = neura_motifs.generate_motif_mlir(
        motif, operation_count, base_seed
    )
    expected_source_sha = neura_motifs.sha256_text(source_text)
    expected_canonical = neura_motifs.canonical_dfg_sha256(source_text)
    if sample["source_sha256"] != expected_source_sha:
        raise ValueError(f"{label}: source hash is not generator-derived")
    if any(sample[name] != expected_canonical for name in (
        "base_dfg_id", "canonical_dfg_sha256", "ranking_query_id",
    )):
        raise ValueError(f"{label}: canonical/base DFG identity changed")
    source_path = Path(str(sample["source_path"])).resolve()
    if source_path.read_text() != source_text:
        raise ValueError(f"{label}: source artifact is not generator-derived")

    lineage = (
        f"generated/{neura_motifs.GENERATOR_VERSION}/{motif}/"
        f"{sample['base_id']}"
    )
    for name in ("leakage_lineage_id", "declared_leakage_lineage_id",
                 "effective_lineage", "family"):
        if sample.get(name) != lineage:
            raise ValueError(f"{label}: {name} does not match generated base")
    expected_candidate = (
        f"{lineage}/{rows}x{columns}/{sample['architecture_variant']}/"
        f"r{registers}"
    )
    if sample["candidate_id"] != expected_candidate:
        raise ValueError(f"{label}: generated candidate identity changed")

    architecture_path = Path(str(sample["architecture_path"])).resolve()
    if raw_sha256(architecture_path) != sample["architecture_sha256"]:
        raise ValueError(f"{label}: architecture artifact hash changed")
    if sample.get("architecture_id") != (
        f"{sample['architecture_sha256']}:{sample['architecture_variant']}"
    ):
        raise ValueError(f"{label}: architecture identity changed")
    structural = neura_experiment.graph_features_from_neura(
        source_text, rows, columns
    )
    structural["split_domain"] = int(
        sample["architecture_variant"] == "split-domain"
    )
    for name in neura_experiment.MODEL_FEATURE_NAMES:
        if name not in structural or sample.get(name) != structural[name]:
            raise ValueError(f"{label}: structural feature {name} changed")

    mapped_path = Path(str(sample["mapped_artifact_path"])).resolve()
    if raw_sha256(mapped_path) != sample["mapped_artifact_sha256"]:
        raise ValueError(f"{label}: mapped artifact hash changed")
    cost_path = Path(str(sample["cost_artifact_path"])).resolve()
    if raw_sha256(cost_path) != sample["cost_artifact_sha256"]:
        raise ValueError(f"{label}: Rec/Res analysis artifact hash changed")
    analysis = neura_experiment.parse_cost_features(cost_path.read_text())
    if analysis is None:
        raise ValueError(f"{label}: Rec/Res analysis artifact is unavailable")
    mapped_text = mapped_path.read_text()
    if 'mapping_strategy = "heuristic"' not in mapped_text:
        raise ValueError(f"{label}: mapped artifact is not heuristic")
    compiled_ii = neura_experiment.parse_checked_mapper_label(
        mapped_text, analysis
    )
    if compiled_ii is None:
        raise ValueError(f"{label}: mapped label/RecMII/ResMII is unavailable")
    rec_mii = int(analysis["rec_mii"])
    res_mii = int(analysis["res_mii"])
    bound = max(int(rec_mii), int(res_mii))
    expected_numbers = {
        "compiled_ii": int(compiled_ii), "rec_mii": int(rec_mii),
        "res_mii": int(res_mii), "baseline_lb": bound,
    }
    for name, expected in expected_numbers.items():
        if int(sample.get(name, -1)) != expected:
            raise ValueError(f"{label}: {name} differs from mapped artifact")
    if int(compiled_ii) < bound:
        raise ValueError(f"{label}: compiled_ii is below Rec/Res floor")
    if sample["lower_bound_source"] != "rec_res_max_v1":
        raise ValueError(f"{label}: lower-bound source changed")
    if (
        sample["mapper_id"] != "neura-heuristic" or
        sample["mapper_config"] != "mapping-strategy=heuristic" or
        sample["mapper_revision"] != neura_revision
    ):
        raise ValueError(f"{label}: mapper identity changed")
    return expected_canonical, expected_family


def validate_generated_training_report(
    report: Mapping[str, Any], report_path: Path, *,
    allow_small_smoke: bool = False,
) -> Tuple[set, set, bool]:
    """Validate generator, labels, holdouts, and the fitted model independently."""
    if report.get("schema_version") != "neura-experiment-v2":
        raise ValueError("unsupported training report schema")
    if report.get("target") != "compiled_ii_from_neura_heuristic_mapper":
        raise ValueError("training report target changed")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training report provenance is missing")
    neura = provenance.get("neura")
    predictor = provenance.get("predictor_repository")
    if (
        not isinstance(neura, Mapping) or
        neura.get("revision") != FROZEN_NEURA_REVISION
    ):
        raise ValueError("training Neura revision differs from the frozen protocol")
    if neura.get("dirty") is not False:
        raise ValueError("training Neura checkout was dirty")
    neura_root = neura.get("root")
    if not isinstance(neura_root, str) or not neura_root:
        raise ValueError("training Neura checkout path is missing")
    require_clean_revision(
        Path(neura_root).resolve(), FROZEN_NEURA_REVISION, "training Neura"
    )
    opt_path = provenance.get("mlir_neura_opt")
    if not isinstance(opt_path, str) or not opt_path:
        raise ValueError("training mapper binary path is missing")
    if raw_sha256(Path(opt_path).resolve()) != provenance.get(
        "mlir_neura_opt_sha256"
    ):
        raise ValueError("training mapper binary changed after label collection")
    if not isinstance(predictor, Mapping) or predictor.get("dirty") is not False:
        raise ValueError("training predictor checkout was dirty")
    current_predictor = require_clean_revision(PROJECT_ROOT, None, "predictor")
    if current_predictor.get("revision") != predictor.get("revision"):
        raise ValueError("predictor revision changed after training")
    if provenance.get("adapter_sha256") != raw_sha256(
        PROJECT_ROOT / "adapters" / "neura_experiment.py"
    ):
        raise ValueError("training adapter changed after report generation")
    config = provenance.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("training experiment configuration is missing")
    if tuple(config.get("ridge_candidates", ())) != FROZEN_RIDGE_CANDIDATES:
        raise ValueError("training Ridge grid differs from the frozen protocol")
    if tuple(config.get("residual_dead_zone_candidates", ())) != (
        FROZEN_DEAD_ZONE_CANDIDATES
    ):
        raise ValueError("training dead-zone grid differs from the frozen protocol")
    if float(config.get("interval_empirical_quantile", -1.0)) != (
        FROZEN_INTERVAL_QUANTILE
    ):
        raise ValueError("training interval quantile differs from the frozen protocol")
    if tuple(config.get("motifs", ())) != FROZEN_MOTIFS:
        raise ValueError("training motifs differ from the frozen protocol")
    if tuple(config.get("motif_shapes", ())) != FROZEN_MOTIF_SHAPES:
        raise ValueError("training shapes differ from the frozen protocol")
    if tuple(config.get("motif_architecture_variants", ())) != (
        FROZEN_ARCHITECTURE_VARIANTS
    ):
        raise ValueError("training architecture variants differ from the frozen protocol")
    requested_per_family = int(config.get("motif_samples_per_family", -1))
    if requested_per_family <= 0:
        raise ValueError("training samples per family must be positive")
    if (
        requested_per_family < FROZEN_MINIMUM_SAMPLES_PER_FAMILY and
        not allow_small_smoke
    ):
        raise ValueError("training samples per family are below the frozen protocol")
    if int(config.get("legacy_random_samples", -1)) != 0:
        raise ValueError("legacy random samples are outside the frozen protocol")
    if "generator_family" not in config.get("metadata_holdout_keys", ()):
        raise ValueError("generator-family holdout was not requested")
    if provenance.get("input_reports") != []:
        raise ValueError("frozen training must be produced in one direct run")
    if provenance.get("loaded_model") is not None:
        raise ValueError("frozen training report must fit a new model")
    if provenance.get("mapping_strategy") != "heuristic":
        raise ValueError("training mapper strategy changed")

    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("training report has no samples")
    base_dfg_ids = set()
    generator_families = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"training sample {index} is not an object")
        identity_text = json.dumps(sample, sort_keys=True).lower()
        if "machsuite" in identity_text:
            raise ValueError("MachSuite data is forbidden in the training report")
        base_dfg, generator_family = _validated_generated_sample(
            sample, index, str(neura["revision"])
        )
        base_dfg_ids.add(base_dfg)
        generator_families.add(generator_family)

    nested = report.get("nested_ridge_family_holdout")
    metadata_holdouts = report.get("nested_ridge_metadata_holdouts")
    generator_holdout = (
        metadata_holdouts.get("generator_family")
        if isinstance(metadata_holdouts, Mapping) else None
    )
    independently_ready = bool(
        isinstance(nested, Mapping) and isinstance(nested.get("rows"), list) and
        bool(nested.get("rows")) and
        isinstance(generator_holdout, Mapping) and
        generator_holdout.get("status") == "ok" and
        int(generator_holdout.get("group_count", -1)) == len(
            neura_motifs.DEFAULT_MOTIFS
        ) and
        report.get("selected_model") == "ridge" and
        requested_per_family >= FROZEN_MINIMUM_SAMPLES_PER_FAMILY and
        len(base_dfg_ids) >= DEFAULT_MINIMUM_BASE_DFGS and
        generator_families == {
            f"generated/motif/{motif}"
            for motif in neura_motifs.DEFAULT_MOTIFS
        }
    )
    candidate_gate = report.get("candidate_gate")
    declared_ready = bool(
        isinstance(candidate_gate, Mapping) and
        candidate_gate.get("overall_ready_for_machsuite_freeze") is True
    )
    if declared_ready != independently_ready:
        raise ValueError("training candidate gate disagrees with verified corpus")

    loaded = load_model_artifact(report_path)
    model = loaded.model
    selected_ridge, selected_dead_zone = (
        neura_experiment.select_ridge_hyperparameters(
            samples, FROZEN_RIDGE_CANDIDATES,
            FROZEN_DEAD_ZONE_CANDIDATES,
        )
    )
    if (
        float(model["ridge"]) != selected_ridge or
        float(model.get("residual_dead_zone", 0.0)) != selected_dead_zone
    ):
        raise ValueError("frozen hyperparameters do not reproduce nested selection")
    retrained = neura_experiment.fit_ridge(
        samples, selected_ridge, selected_dead_zone,
    )
    if isinstance(nested, Mapping) and nested.get("rows"):
        neura_experiment.calibrate_unseen_family_interval(
            retrained, nested["rows"], FROZEN_INTERVAL_QUANTILE
        )
    if canonical_model_sha256(retrained) != canonical_model_sha256(model):
        raise ValueError("frozen model cannot be reproduced from training rows")
    return base_dfg_ids, generator_families, independently_ready


def preflight_machsuite(
    *, inventory_path: Path, suite_root: Path, neura_root: Path,
    architecture: Path, clang: Path, llvm_extract: Path,
    mlir_translate: Path, opt: Path, output_dir: Path, timeout: int,
) -> Mapping[str, Any]:
    inventory, specs = load_inventory(inventory_path)
    expected_revision = str(inventory["revision"])
    suite_root = suite_root.resolve()
    neura_root = neura_root.resolve()
    suite_state = require_clean_revision(
        suite_root, expected_revision, "MachSuite"
    )
    neura_state = require_clean_revision(
        neura_root, FROZEN_NEURA_REVISION, "Neura"
    )
    predictor_state = require_clean_revision(PROJECT_ROOT, None, "predictor")
    architecture = architecture.resolve()
    if not architecture.is_file():
        raise ValueError(f"architecture not found: {architecture}")
    clang = resolve_executable(clang)
    llvm_extract = resolve_executable(llvm_extract)
    mlir_translate = resolve_executable(mlir_translate)
    opt = resolve_executable(opt)
    neura_experiment.require_opt_argument(opt, "--analyze-rec-res-mii")
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite preflight directory: {output_dir}")
    output_dir.mkdir(parents=True)
    rows = int(inventory["primary_architecture"]["rows"])
    columns = int(inventory["primary_architecture"]["columns"])
    architecture_variant = f"{rows}x{columns}"
    architecture_sha256 = raw_sha256(architecture)
    records: List[Dict[str, Any]] = []
    for spec in specs:
        source = suite_root / spec.source
        if not source.is_file():
            raise ValueError(f"missing MachSuite source: {source}")
        dependencies = dependency_hashes(suite_root, source)
        records.append({
            "benchmark_id": spec.benchmark_id,
            "source": spec.source,
            "top": spec.top,
            "leakage_lineage_id": spec.leakage_lineage_id,
            "source_sha256": raw_sha256(source),
            "source_dependency_sha256": dependencies,
            "source_bundle_sha256": canonical_json_sha256(dependencies),
            "status": "declared",
            "stage": "predeclared",
        })
    manifest_path = output_dir / "preflight.json"
    manifest: Dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "protocol_status": "preflight_in_progress",
        "labels_accessed": False,
        "compiled_ii_present": False,
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": raw_sha256(inventory_path),
            "suite": inventory["suite"],
            "repository": inventory["repository"],
            "revision": expected_revision,
        },
        "suite_provenance": suite_state,
        "neura_provenance": neura_state,
        "predictor_provenance": predictor_state,
        "protocol_implementation": protocol_implementation_identity(),
        "architecture": {
            "path": str(architecture), "sha256": architecture_sha256,
            "rows": rows, "columns": columns,
            "architecture_id": f"{architecture_sha256}:{architecture_variant}",
        },
        "mapper_contract": {
            "mapper_id": "neura-heuristic",
            "mapper_revision": neura_state["revision"],
            "mapper_config": "mapping-strategy=heuristic",
            "labels_hidden_until_prediction_seal": True,
        },
        "lower_bound_contract": dict(LOWER_BOUND_CONTRACT),
        "toolchain": {
            "clang": tool_identity(clang),
            "llvm_extract": tool_identity(llvm_extract),
            "mlir_translate": tool_identity(mlir_translate),
            "mlir_neura_opt": tool_identity(opt),
        },
        "compile_flags": [
            "-O3", "-fno-vectorize", "-fno-slp-vectorize",
            "-fno-unroll-loops", "-Xclang", "-disable-lifetime-markers",
            "-std=c11",
        ],
        "candidates": records,
        "samples": [],
        "summary": manifest_summary(records),
    }
    neura_motifs.atomic_write_json(manifest_path, manifest)

    samples: List[Mapping[str, Any]] = []
    for spec, record in zip(specs, records):
        candidate_dir = output_dir / spec.benchmark_id.replace("/", "__")
        candidate_dir.mkdir()
        failed: Optional[Mapping[str, Any]] = None
        for stage, command, expected_output in preflight_commands(
            spec, suite_root, candidate_dir, clang, llvm_extract,
            mlir_translate, opt, architecture, rows, columns,
        ):
            outcome = run_stage(stage, command, timeout)
            if not outcome["ok"]:
                failed = outcome
                break
            if not expected_output.is_file():
                failed = {
                    "ok": False, "stage": stage,
                    "status": "missing_output", "command": list(command),
                    "expected_output": str(expected_output),
                }
                break
        if failed is not None:
            record.update({
                "status": "censored", "stage": failed["stage"],
                "failure": {key: value for key, value in failed.items() if key != "ok"},
            })
            manifest["summary"] = manifest_summary(records)
            neura_motifs.atomic_write_json(manifest_path, manifest)
            continue

        lowered = candidate_dir / "lowered.mlir"
        cost = candidate_dir / "cost.mlir"
        values = neura_experiment.parse_cost_features(cost.read_text())
        if values is None:
            record.update({
                "status": "censored", "stage": "rec_res_analysis_parse",
                "failure": {"status": "required_rec_res_facts_unavailable"},
            })
            manifest["summary"] = manifest_summary(records)
            neura_motifs.atomic_write_json(manifest_path, manifest)
            continue
        features: Dict[str, Any] = dict(values)
        features.update(neura_experiment.graph_features_from_neura(
            lowered.read_text(), rows, columns
        ))
        neura_experiment.add_prediction_features(features)
        missing = [
            name for name in neura_experiment.FEATURE_NAMES if name not in features
        ]
        if missing:
            record.update({
                "status": "censored", "stage": "feature_extraction",
                "failure": {"status": "missing_features", "features": missing},
            })
            manifest["summary"] = manifest_summary(records)
            neura_motifs.atomic_write_json(manifest_path, manifest)
            continue

        lowered_sha256 = raw_sha256(lowered)
        canonical_dfg_sha256 = neura_motifs.canonical_dfg_sha256(
            lowered.read_text()
        )
        candidate_identity = {
            "base_dfg_id": canonical_dfg_sha256,
            "architecture_sha256": architecture_sha256,
            "architecture_variant": architecture_variant,
            "mapper_id": "neura-heuristic",
            "mapper_revision": neura_state["revision"],
            "mapper_config": "mapping-strategy=heuristic",
        }
        candidate_id = canonical_json_sha256(candidate_identity)
        sample_id = f"machsuite/{spec.benchmark_id}/{architecture_variant}"
        sample = {
            "sample_id": sample_id,
            "lower_bound": int(features["baseline_lb"]),
            "rec_mii": int(features["rec_mii"]),
            "res_mii": int(features["res_mii"]),
            "features": {
                name: features[name] for name in PREDICTION_RECORD_FEATURE_NAMES
            },
            "metadata": {
                "suite": "MachSuite",
                "suite_revision": expected_revision,
                "kernel_id": spec.benchmark_id,
                "leakage_lineage_id": spec.leakage_lineage_id,
                "source_sha256": record["source_bundle_sha256"],
                "benchmark_source_sha256": record["source_sha256"],
                "dfg_source_sha256": lowered_sha256,
                "canonical_dfg_sha256": canonical_dfg_sha256,
                "base_dfg_id": canonical_dfg_sha256,
                "ranking_query_id": canonical_dfg_sha256,
                "architecture_sha256": architecture_sha256,
                "architecture_variant": architecture_variant,
                "architecture_id": (
                    f"{architecture_sha256}:{architecture_variant}"
                ),
                "candidate_id": candidate_id,
                "mapper_id": "neura-heuristic",
                "mapper_revision": neura_state["revision"],
                "mapper_config": "mapping-strategy=heuristic",
                "lower_bound_source": "rec_res_max_v1",
            },
        }
        record.update({
            "status": "ready", "stage": "preflight_complete",
            "sample_id": sample_id, "candidate_id": candidate_id,
            "canonical_dfg_sha256": canonical_dfg_sha256,
            "lowered_artifact_path": str(lowered.relative_to(output_dir)),
            "lowered_artifact_sha256": lowered_sha256,
            "cost_artifact_path": str(cost.relative_to(output_dir)),
            "cost_artifact_sha256": raw_sha256(cost),
            "lower_bound": int(features["baseline_lb"]),
            "rec_mii": int(features["rec_mii"]),
            "res_mii": int(features["res_mii"]),
        })
        samples.append(sample)
        manifest["samples"] = samples
        manifest["summary"] = manifest_summary(records)
        neura_motifs.atomic_write_json(manifest_path, manifest)

    manifest["protocol_status"] = "preflight_complete"
    manifest["summary"] = manifest_summary(records)
    if recursively_contains_key(manifest, "compiled_ii"):
        raise AssertionError("preflight manifest unexpectedly contains a label")
    neura_motifs.atomic_write_json(manifest_path, manifest)
    validate_frozen_manifest(manifest, manifest_path)
    return manifest


def freeze_random_training_model(
    report_path: Path, output_path: Path, *,
    minimum_base_dfgs: int = DEFAULT_MINIMUM_BASE_DFGS,
    minimum_generator_families: int = DEFAULT_MINIMUM_GENERATOR_FAMILIES,
    allow_small_smoke: bool = False,
) -> Mapping[str, Any]:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite frozen model: {output_path}")
    report = read_json(report_path)
    samples = report.get("samples")
    base_dfg_ids, generator_families, report_gate_ready = (
        validate_generated_training_report(
            report, report_path, allow_small_smoke=allow_small_smoke
        )
    )
    if minimum_base_dfgs <= 0 or minimum_generator_families <= 0:
        raise ValueError("minimum training-scale gates must be positive")
    if (
        minimum_base_dfgs < DEFAULT_MINIMUM_BASE_DFGS or
        minimum_generator_families < DEFAULT_MINIMUM_GENERATOR_FAMILIES
    ) and not allow_small_smoke:
        raise ValueError(
            "frozen scale thresholds may only be raised; use "
            "--allow-small-smoke for a non-frozen artifact"
        )
    scale_ready = (
        len(base_dfg_ids) >= minimum_base_dfgs and
        len(generator_families) >= minimum_generator_families
    )
    if not report_gate_ready and not allow_small_smoke:
        raise ValueError(
            "training report did not pass its generated-corpus candidate gate"
        )
    if not scale_ready and not allow_small_smoke:
        raise ValueError(
            "generated corpus is below the frozen-model scale gate: "
            f"{len(base_dfg_ids)}/{minimum_base_dfgs} base DFGs, "
            f"{len(generator_families)}/{minimum_generator_families} families"
        )
    final_ready = scale_ready and report_gate_ready
    loaded = load_model_artifact(report_path)
    expected_features = list(neura_experiment.MODEL_FEATURE_NAMES)
    if list(loaded.model["feature_names"]) != expected_features:
        raise ValueError(
            "model feature contract is not the predeclared structure-only set"
        )
    bound_contract = report.get("lower_bound_contract")
    if not isinstance(bound_contract, Mapping) or bound_contract.get("name") != "rec_res_max_v1":
        raise ValueError("training report does not use rec_res_max_v1")
    provenance = report.get("provenance", {})
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    frozen = {
        "schema_version": FROZEN_MODEL_SCHEMA,
        "target": "compiled_ii_from_neura_heuristic_mapper",
        "artifact_status": (
            FROZEN_MODEL_STATUS if final_ready else SMOKE_MODEL_STATUS
        ),
        "lower_bound_contract": dict(LOWER_BOUND_CONTRACT),
        "trained_full_model": dict(loaded.model),
        "trained_full_model_sha256": canonical_model_sha256(loaded.model),
        "provenance": {
            "neura": provenance.get("neura"),
            "predictor_repository": provenance.get("predictor_repository"),
            "training_report_path": str(report_path.resolve()),
            "training_report_sha256": raw_sha256(report_path),
            "training_sample_count": len(samples),
            "training_distinct_base_dfg_count": len(base_dfg_ids),
            "training_generator_family_count": len(generator_families),
            "training_sample_identity_sha256": canonical_json_sha256([
                {
                    "sample_id": sample.get("index", sample.get("sample_id")),
                    "lineage": sample.get("leakage_lineage_id", sample.get("lineage")),
                    "candidate_id": sample.get("candidate_id"),
                    "source_sha256": sample.get("source_sha256"),
                }
                for sample in samples
            ]),
        },
        "training_contract": {
            "dataset_role": "generated_random_dfg_training_only",
            "machsuite_labels_accessed": False,
            "feature_names": expected_features,
            "bound_components_are_model_features": False,
            "scale_gate": {
                "minimum_base_dfgs": minimum_base_dfgs,
                "minimum_generator_families": minimum_generator_families,
                "actual_base_dfgs": len(base_dfg_ids),
                "actual_generator_families": len(generator_families),
                "passed": final_ready,
                "training_report_candidate_gate_passed": report_gate_ready,
                "small_smoke_override": allow_small_smoke and not final_ready,
            },
        },
    }
    neura_motifs.atomic_write_json(output_path, frozen)
    return frozen


def ready_sample_ids(manifest: Mapping[str, Any]) -> List[str]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("preflight candidates are missing")
    return [
        str(candidate["sample_id"])
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("status") == "ready"
    ]


def validate_prediction_report_and_seal(
    *, manifest: Mapping[str, Any], manifest_path: Path,
    model_path: Path, prediction_path: Path,
    predictions: Mapping[str, Any], seal: Mapping[str, Any], loaded: Any,
) -> None:
    if loaded.container != FROZEN_MODEL_SCHEMA:
        raise ValueError("model is not the frozen artifact container")
    if loaded.target != "compiled_ii_from_neura_heuristic_mapper":
        raise ValueError("frozen model target changed")
    if loaded.artifact_status != FROZEN_MODEL_STATUS:
        raise ValueError("model is not frozen before MachSuite reveal")
    if list(loaded.model["feature_names"]) != list(
        neura_experiment.MODEL_FEATURE_NAMES
    ):
        raise ValueError("frozen model feature contract changed")
    if loaded.lower_bound_contract != LOWER_BOUND_CONTRACT:
        raise ValueError("frozen model lower-bound contract changed")
    if seal.get("schema_version") != PREDICTION_SEAL_SCHEMA:
        raise ValueError("invalid prediction seal")
    if seal.get("labels_accessed") is not False:
        raise ValueError("prediction seal does not attest labels_accessed=false")
    expected_paths = {
        "preflight_path": manifest_path.resolve(),
        "model_path": model_path.resolve(),
        "prediction_path": prediction_path.resolve(),
    }
    for name, expected in expected_paths.items():
        try:
            actual = Path(str(seal.get(name, ""))).resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"sealed {name} is invalid") from error
        if actual != expected:
            raise ValueError(f"sealed {name} does not match command input")
    expected_hashes = {
        "preflight_sha256": raw_sha256(manifest_path),
        "model_source_sha256": raw_sha256(model_path),
        "prediction_sha256": raw_sha256(prediction_path),
    }
    for name, expected in expected_hashes.items():
        if seal.get(name) != expected:
            raise ValueError(f"sealed {name} does not match current file")
    implementation = protocol_implementation_identity()
    if seal.get("protocol_implementation_sha256") != implementation["combined_sha256"]:
        raise ValueError("prediction protocol implementation changed")
    if seal.get("predictor_source_sha256") != raw_sha256(
        SOURCE_ROOT / "cgra_ii_predictor" / "predict.py"
    ):
        raise ValueError("prediction source hash changed")
    if manifest.get("protocol_implementation") != implementation:
        raise ValueError("preflight and prediction implementations differ")
    if seal.get("canonical_model_sha256") != loaded.model_sha256:
        raise ValueError("canonical model hash changed after prediction")

    if predictions.get("schema_version") != "compiled-ii-point-predictions-v1":
        raise ValueError("invalid frozen prediction report schema")
    if recursively_contains_key(predictions, "compiled_ii"):
        raise ValueError("prediction report contains a forbidden compiled_ii label")
    expected_ids = ready_sample_ids(manifest)
    rows = predictions.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("prediction report has no predictions array")
    row_ids = [
        str(row.get("sample_id", "")) if isinstance(row, Mapping) else ""
        for row in rows
    ]
    if row_ids != expected_ids or len(set(row_ids)) != len(row_ids):
        raise ValueError("prediction rows differ from the frozen candidate order")
    if predictions.get("sample_count") != len(expected_ids):
        raise ValueError("prediction report sample_count changed")
    if seal.get("prediction_count") != len(expected_ids):
        raise ValueError("sealed prediction_count changed")
    candidate_set_hash = canonical_json_sha256(expected_ids)
    if seal.get("candidate_identity_set_sha256") != candidate_set_hash:
        raise ValueError("sealed candidate identity set changed")
    frozen_protocol = predictions.get("frozen_protocol")
    expected_frozen_protocol = {
        "preflight_sha256": expected_hashes["preflight_sha256"],
        "labels_accessed": False,
        "predictions_frozen_before_reveal": True,
        "candidate_identity_set_sha256": candidate_set_hash,
        "protocol_implementation_sha256": implementation["combined_sha256"],
        "chronology_proof": "requires_external_timestamp_or_publication",
    }
    if frozen_protocol != expected_frozen_protocol:
        raise ValueError("prediction frozen_protocol attestation changed")
    input_record = predictions.get("input")
    if not isinstance(input_record, Mapping) or (
        Path(str(input_record.get("path", ""))).resolve() !=
        manifest_path.resolve()
    ) or input_record.get("sha256") != expected_hashes["preflight_sha256"]:
        raise ValueError("prediction input identity changed")
    model_record = predictions.get("model")
    if not isinstance(model_record, Mapping) or (
        Path(str(model_record.get("path", ""))).resolve() != model_path.resolve() or
        model_record.get("source_sha256") != expected_hashes["model_source_sha256"] or
        model_record.get("model_sha256") != loaded.model_sha256 or
        model_record.get("artifact_status") != FROZEN_MODEL_STATUS
    ):
        raise ValueError("prediction model identity changed")

    sample_by_id = {
        str(sample["sample_id"]): sample for sample in manifest["samples"]
    }
    for row in rows:
        sample = sample_by_id[str(row["sample_id"])]
        if row.get("lower_bound") != sample.get("lower_bound"):
            raise ValueError(f"{row['sample_id']}: prediction lower bound changed")
        if row.get("rec_mii") != sample.get("rec_mii"):
            raise ValueError(f"{row['sample_id']}: prediction RecMII changed")
        if row.get("res_mii") != sample.get("res_mii"):
            raise ValueError(f"{row['sample_id']}: prediction ResMII changed")
        if row.get("lower_bound_source") != "rec_res_max_v1":
            raise ValueError(f"{row['sample_id']}: lower-bound source changed")
        expected_features = {
            name: sample["features"][name]
            for name in neura_experiment.MODEL_FEATURE_NAMES
        }
        if row.get("model_features") != expected_features:
            raise ValueError(f"{row['sample_id']}: prediction features changed")
        if row.get("metadata") != sample.get("metadata"):
            raise ValueError(f"{row['sample_id']}: prediction metadata changed")
        if row.get("model_sha256") != loaded.model_sha256:
            raise ValueError(f"{row['sample_id']}: prediction model hash changed")


def freeze_predictions(
    manifest_path: Path, model_path: Path, output_path: Path, seal_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output_path.exists() or seal_path.exists():
        raise ValueError("refusing to overwrite predictions or seal")
    manifest = read_json(manifest_path)
    validate_frozen_manifest(manifest, manifest_path)
    loaded = load_model_artifact(model_path)
    if loaded.artifact_status != FROZEN_MODEL_STATUS:
        raise ValueError("model was not frozen before MachSuite label reveal")
    if list(loaded.model["feature_names"]) != list(neura_experiment.MODEL_FEATURE_NAMES):
        raise ValueError("model does not use the primary structure-only feature set")
    if loaded.lower_bound_contract.get("name") != "rec_res_max_v1":
        raise ValueError("model lower-bound contract is not rec_res_max_v1")
    samples, provenance = load_prediction_samples(
        manifest_path, loaded.model["feature_names"]
    )
    expected_ids = ready_sample_ids(manifest)
    if [sample.sample_id for sample in samples] != expected_ids:
        raise ValueError("prediction samples do not match ready candidate order")
    report = build_prediction_report(loaded, samples, manifest_path, provenance)
    warnings = [
        warning
        for prediction in report["predictions"]
        for warning in prediction.get("warnings", [])
    ]
    if warnings:
        raise ValueError("frozen prediction contract warnings: " + "; ".join(warnings))
    manifest_sha256 = raw_sha256(manifest_path)
    report["frozen_protocol"] = {
        "preflight_sha256": manifest_sha256,
        "labels_accessed": False,
        "predictions_frozen_before_reveal": True,
        "candidate_identity_set_sha256": canonical_json_sha256(expected_ids),
        "protocol_implementation_sha256": (
            protocol_implementation_identity()["combined_sha256"]
        ),
        "chronology_proof": "requires_external_timestamp_or_publication",
    }
    if recursively_contains_key(report, "compiled_ii"):
        raise AssertionError("prediction report unexpectedly contains a label")
    neura_motifs.atomic_write_json(output_path, report)
    seal = {
        "schema_version": PREDICTION_SEAL_SCHEMA,
        "preflight_path": str(manifest_path.resolve()),
        "preflight_sha256": manifest_sha256,
        "model_path": str(model_path.resolve()),
        "model_source_sha256": raw_sha256(model_path),
        "canonical_model_sha256": loaded.model_sha256,
        "prediction_path": str(output_path.resolve()),
        "prediction_sha256": raw_sha256(output_path),
        "candidate_identity_set_sha256": canonical_json_sha256(expected_ids),
        "prediction_count": len(expected_ids),
        "predictor_source_sha256": raw_sha256(SOURCE_ROOT / "cgra_ii_predictor" / "predict.py"),
        "protocol_implementation_sha256": (
            protocol_implementation_identity()["combined_sha256"]
        ),
        "labels_accessed": False,
    }
    validate_prediction_report_and_seal(
        manifest=manifest, manifest_path=manifest_path,
        model_path=model_path, prediction_path=output_path,
        predictions=report, seal=seal, loaded=loaded,
    )
    neura_motifs.atomic_write_json(seal_path, seal)
    return report, seal


def mapper_command(
    opt: Path, lowered: Path, architecture: Path, output: Path,
    rows: int, columns: int,
) -> Tuple[str, ...]:
    return (
        str(opt), str(lowered), f"--architecture-spec={architecture}",
        f"--map-to-accelerator=mapping-strategy=heuristic "
        f"x-tiles={columns} y-tiles={rows}",
        "-o", str(output),
    )


def evaluation_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    scored = [row for row in rows if row.get("status") == "scored"]
    if not scored:
        return {"status": "unavailable", "scored_count": 0}
    errors = [
        float(row["predicted_compiled_ii"]) - float(row["compiled_ii"])
        for row in scored
    ]
    baseline_errors = [
        float(row["lower_bound"]) - float(row["compiled_ii"])
        for row in scored
    ]
    return {
        "status": "ok",
        "scored_count": len(scored),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "baseline_mae": (
            sum(abs(value) for value in baseline_errors) / len(baseline_errors)
        ),
        "underprediction_rate": sum(value < 0.0 for value in errors) / len(errors),
    }


def reveal_labels(
    *, manifest_path: Path, model_path: Path, prediction_path: Path,
    seal_path: Path, suite_root: Path, neura_root: Path, architecture: Path,
    opt: Path, output_dir: Path, timeout: int,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite reveal directory: {output_dir}")
    manifest = read_json(manifest_path)
    predictions = read_json(prediction_path)
    seal = read_json(seal_path)
    validate_frozen_manifest(manifest, manifest_path)
    loaded = load_model_artifact(model_path)
    if loaded.artifact_status != FROZEN_MODEL_STATUS:
        raise ValueError("model is not a frozen random-DFG artifact")
    validate_prediction_report_and_seal(
        manifest=manifest, manifest_path=manifest_path,
        model_path=model_path, prediction_path=prediction_path,
        predictions=predictions, seal=seal, loaded=loaded,
    )
    expected_revision = str(manifest["inventory"]["revision"])
    require_clean_revision(suite_root.resolve(), expected_revision, "MachSuite")
    neura_state = require_clean_revision(
        neura_root.resolve(), FROZEN_NEURA_REVISION, "Neura"
    )
    if neura_state.get("revision") != manifest["neura_provenance"].get("revision"):
        raise ValueError("Neura revision changed after preflight")
    architecture = architecture.resolve()
    if raw_sha256(architecture) != manifest["architecture"].get("sha256"):
        raise ValueError("architecture changed after preflight")
    opt = resolve_executable(opt)
    if raw_sha256(opt) != manifest["toolchain"]["mlir_neura_opt"].get("sha256"):
        raise ValueError("mapper binary changed after preflight")
    prediction_rows = predictions.get("predictions")
    by_sample = {
        str(row["sample_id"]): row for row in prediction_rows
    }
    ready_ids = ready_sample_ids(manifest)

    output_dir.mkdir(parents=True)
    rows = int(manifest["architecture"]["rows"])
    columns = int(manifest["architecture"]["columns"])
    label_records: List[Dict[str, Any]] = []
    joined: List[Dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        benchmark_id = str(candidate["benchmark_id"])
        if candidate.get("status") != "ready":
            label_records.append({
                "benchmark_id": benchmark_id,
                "status": "preflight_censored",
                "preflight_stage": candidate.get("stage"),
                "failure": candidate.get("failure"),
            })
            joined.append({
                "benchmark_id": benchmark_id,
                "status": "preflight_censored",
            })
            continue
        sample_id = str(candidate["sample_id"])
        lowered = manifest_path.parent / str(candidate["lowered_artifact_path"])
        if raw_sha256(lowered) != candidate.get("lowered_artifact_sha256"):
            raise ValueError(f"lowered artifact changed: {benchmark_id}")
        candidate_dir = output_dir / benchmark_id.replace("/", "__")
        candidate_dir.mkdir()
        mapped = candidate_dir / "mapped.mlir"
        command = mapper_command(
            opt, lowered, architecture, mapped, rows, columns
        )
        outcome = run_stage("mapper", command, timeout)
        if not outcome["ok"]:
            label_records.append({
                "benchmark_id": benchmark_id, "sample_id": sample_id,
                "candidate_id": candidate["candidate_id"],
                "status": "mapper_censored",
                "failure": {key: value for key, value in outcome.items() if key != "ok"},
            })
            joined.append({
                "benchmark_id": benchmark_id, "sample_id": sample_id,
                "status": "mapper_censored",
            })
            continue
        compiled_ii = neura_experiment.parse_checked_mapper_label(
            mapped.read_text(), {
                "rec_mii": candidate["rec_mii"],
                "res_mii": candidate["res_mii"],
            }
        )
        if compiled_ii is None or compiled_ii < int(candidate["lower_bound"]):
            failure = {
                "status": "invalid_compiled_ii",
                "value": compiled_ii,
            }
            label_records.append({
                "benchmark_id": benchmark_id, "sample_id": sample_id,
                "candidate_id": candidate["candidate_id"],
                "status": "mapper_censored", "failure": failure,
            })
            joined.append({
                "benchmark_id": benchmark_id, "sample_id": sample_id,
                "status": "mapper_censored",
            })
            continue
        label_records.append({
            "benchmark_id": benchmark_id, "sample_id": sample_id,
            "candidate_id": candidate["candidate_id"], "status": "success",
            "compiled_ii": int(compiled_ii),
            "mapped_artifact_path": str(mapped.relative_to(output_dir)),
            "mapped_artifact_sha256": raw_sha256(mapped),
        })
        prediction = by_sample[sample_id]
        joined.append({
            "benchmark_id": benchmark_id, "sample_id": sample_id,
            "candidate_id": candidate["candidate_id"], "status": "scored",
            "lower_bound": int(candidate["lower_bound"]),
            "predicted_compiled_ii": float(prediction["predicted_compiled_ii"]),
            "compiled_ii": int(compiled_ii),
        })

    labels = {
        "schema_version": REVEALED_LABEL_SCHEMA,
        "prediction_seal_sha256": raw_sha256(seal_path),
        "predictions_were_frozen_before_labels": True,
        "declared_count": len(manifest["candidates"]),
        "records": label_records,
    }
    neura_motifs.atomic_write_json(output_dir / "labels.json", labels)
    evaluation = {
        "schema_version": EVALUATION_SCHEMA,
        "preflight_sha256": raw_sha256(manifest_path),
        "model_source_sha256": raw_sha256(model_path),
        "prediction_sha256": raw_sha256(prediction_path),
        "prediction_seal_sha256": raw_sha256(seal_path),
        "label_sha256": raw_sha256(output_dir / "labels.json"),
        "declared_count": len(manifest["candidates"]),
        "preflight_ready_count": len(ready_ids),
        "scored_count": sum(row.get("status") == "scored" for row in joined),
        "coverage_over_declared": (
            sum(row.get("status") == "scored" for row in joined) /
            len(manifest["candidates"])
        ),
        "metrics": evaluation_metrics(joined),
        "records": joined,
        "interpretation": (
            "Results cover the fixed compatible subset; censored variants remain "
            "in the declared-suite denominator."
        ),
    }
    neura_motifs.atomic_write_json(output_dir / "evaluation.json", evaluation)
    return labels, evaluation


def add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--neura-root", type=Path, default=DEFAULT_NEURA_ROOT)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--clang", type=Path, default=Path("clang"))
    parser.add_argument("--llvm-extract", type=Path, default=Path("llvm-extract"))
    parser.add_argument("--mlir-translate", type=Path, default=Path("mlir-translate"))
    parser.add_argument("--opt", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen random-DFG train / MachSuite test protocol"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    add_preflight_arguments(preflight_parser)
    model_parser = subparsers.add_parser("freeze-model")
    model_parser.add_argument("--training-report", type=Path, required=True)
    model_parser.add_argument("--output", type=Path, required=True)
    model_parser.add_argument(
        "--minimum-base-dfgs", type=int, default=DEFAULT_MINIMUM_BASE_DFGS
    )
    model_parser.add_argument(
        "--minimum-generator-families", type=int,
        default=DEFAULT_MINIMUM_GENERATOR_FAMILIES,
    )
    model_parser.add_argument(
        "--allow-small-smoke", action="store_true",
        help=("Write a smoke-only artifact when the scale gate fails; such an "
              "artifact is rejected by frozen MachSuite prediction."),
    )
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--preflight", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--seal", type=Path, required=True)
    reveal_parser = subparsers.add_parser("reveal")
    reveal_parser.add_argument("--preflight", type=Path, required=True)
    reveal_parser.add_argument("--model", type=Path, required=True)
    reveal_parser.add_argument("--predictions", type=Path, required=True)
    reveal_parser.add_argument("--seal", type=Path, required=True)
    reveal_parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    reveal_parser.add_argument("--neura-root", type=Path, default=DEFAULT_NEURA_ROOT)
    reveal_parser.add_argument("--architecture", type=Path)
    reveal_parser.add_argument("--opt", type=Path)
    reveal_parser.add_argument("--timeout", type=int, default=120)
    reveal_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "timeout", 1) <= 0:
        parser.error("--timeout must be positive")
    try:
        if args.command == "preflight":
            architecture = args.architecture or (
                args.neura_root / "test" / "arch_spec" / "architecture.yaml"
            )
            opt = args.opt or (
                args.neura_root / "build" / "tools" / "mlir-neura-opt" /
                "mlir-neura-opt"
            )
            result = preflight_machsuite(
                inventory_path=args.inventory, suite_root=args.suite_root,
                neura_root=args.neura_root, architecture=architecture,
                clang=args.clang, llvm_extract=args.llvm_extract,
                mlir_translate=args.mlir_translate, opt=opt,
                output_dir=args.output_dir, timeout=args.timeout,
            )
            print(json.dumps(result["summary"], sort_keys=True))
        elif args.command == "freeze-model":
            result = freeze_random_training_model(
                args.training_report, args.output,
                minimum_base_dfgs=args.minimum_base_dfgs,
                minimum_generator_families=args.minimum_generator_families,
                allow_small_smoke=args.allow_small_smoke,
            )
            print(result["trained_full_model_sha256"])
        elif args.command == "predict":
            _, seal = freeze_predictions(
                args.preflight, args.model, args.output, args.seal
            )
            print(seal["prediction_sha256"])
        else:
            manifest = read_json(args.preflight)
            architecture = args.architecture or Path(
                str(manifest["architecture"]["path"])
            )
            opt = args.opt or Path(
                str(manifest["toolchain"]["mlir_neura_opt"]["path"])
            )
            _, evaluation = reveal_labels(
                manifest_path=args.preflight, model_path=args.model,
                prediction_path=args.predictions, seal_path=args.seal,
                suite_root=args.suite_root, neura_root=args.neura_root,
                architecture=architecture, opt=opt,
                output_dir=args.output_dir, timeout=args.timeout,
            )
            print(json.dumps({
                "declared_count": evaluation["declared_count"],
                "scored_count": evaluation["scored_count"],
                "coverage_over_declared": evaluation["coverage_over_declared"],
            }, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
