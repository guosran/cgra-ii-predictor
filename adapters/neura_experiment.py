#!/usr/bin/env python3
"""Neura adapter for reproducible compiled-II prediction experiments.

This adapter is deliberately offline.  It does not change the mapper or
the RecMII/ResMII lower bound: labels are the existing heuristic mapper's
``compiled_ii``, while features are available before mapping from the Neura
DFG and YAML architecture.  The script writes all generated inputs and labels
under its requested output directory (``/tmp`` by default).

The random generator is intentionally constrained to valid lowered Neura DFGs
whose operations have a clear tile-domain contract.  It varies graph shape,
fanout, depth, tile count, and homogeneous versus split FU domains.  Those
examples are a corpus supplement, not a claim that synthetic graphs replace
real frontend kernels.  Evaluation is grouped by kernel family: a model may
not train on one shape or mask of a kernel and claim to predict another shape
of that same kernel.  A random row split is retained only as a diagnostic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np

try:
    from adapters import neura_motifs
except ImportError:  # Running the file directly from its adapters directory.
    import neura_motifs  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SUBMODULE_NEURA_ROOT = PROJECT_ROOT / "third_party" / "neura"
DEFAULT_SEED = 20260829
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cgra_ii_predictor.dataset import Sample as CoreSample  # noqa: E402
from cgra_ii_predictor.model import (  # noqa: E402
    calibrate_unseen_group_interval as core_calibrate_interval,
    constrained_predicted_residual as core_constrained_predicted_residual,
    fit_ridge as core_fit_ridge,
    nested_group_holdout as core_nested_group_holdout,
    predict_compiled_ii as core_predict_compiled_ii,
    predict_ridge as core_predict_ridge,
    raw_ridge_residual_from_features as core_raw_residual_from_features,
    select_ridge_hyperparameters as core_select_ridge_hyperparameters,
)
from cgra_ii_predictor.predict import (  # noqa: E402
    LoadedModel,
    canonical_model_sha256,
    load_model_artifact,
)


DATA_TYPE = "!neura.data<i32, i1>"
FEATURE_NAMES = (
    "baseline_lb",
    "rec_mii",
    "res_mii",
    "rec_res_gap",
    "rec_dominant",
    "nodes",
    "moves",
    "ctrl_moves",
    "reserves",
    "phis",
    "predicates",
    "memory_ops",
    "gep_ops",
    "indirect_geps",
    "pointer_loads",
    "div_ops",
    "arithmetic_ops",
    "comparisons",
    "sources",
    "depth",
    "max_fanout",
    "semantic_edges",
    "semantic_depth",
    "semantic_width",
    "semantic_max_fanout",
    "semantic_branch_nodes",
    "semantic_cutwidth",
    "live_value_peak",
    "pointer_path",
    "memory_path",
    "control_path",
    "multi_input_nodes",
    "tiles",
    "links",
    "rows",
    "columns",
    "memory_tiles",
    "bisection_links",
    "total_registers",
    "fu_class_peak_ops",
    "compute_fu_peak_pressure",
    "memory_fu_pressure",
    "routing_edge_pressure",
    "routing_cut_pressure",
    "register_pressure",
    "split_domain",
)

# The complete feature record above is useful for corpus analysis.  The model
# deliberately uses a smaller, physically motivated subset.  The hard floor is
# max(RecMII, ResMII), so neither that floor nor its two components are learned
# features.  The Ridge model sees only pre-mapping graph structure and raw
# architecture topology and predicts the residual above that floor.
MODEL_FEATURE_NAMES = (
    "semantic_edges",
    "semantic_depth",
    "semantic_width",
    "semantic_max_fanout",
    "semantic_branch_nodes",
    "semantic_cutwidth",
    "live_value_peak",
    "multi_input_nodes",
    "memory_ops",
    "phis",
    "predicates",
    "pointer_path",
    "memory_path",
    "control_path",
    "compute_fu_peak_pressure",
    "memory_fu_pressure",
    "routing_edge_pressure",
    "routing_cut_pressure",
    "register_pressure",
)

LOWER_BOUND_COMPONENT_NAMES = ("rec_mii", "res_mii")

Sample = Dict[str, Any]
INVOCATION_FAILURES: List[Dict[str, object]] = []
COST_FEATURE_NAMES = (
    "rec_mii",
    "res_mii",
)


@dataclass(frozen=True)
class InvocationResult:
    """Result of one isolated compiler/mapper subprocess.

    A worker returns this value to the coordinator instead of mutating the
    process-wide failure list.  Keeping the command diagnostics in the value
    makes candidate-level parallelism deterministic and leaves manifest writes
    exclusively to the main thread.
    """

    ok: bool
    status: str
    stage: str
    timeout_seconds: int
    command: Tuple[str, ...]
    returncode: Optional[int] = None
    output: Optional[str] = None
    stderr_head: str = ""
    stderr_tail: str = ""

    def failure_record(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "stage": self.stage,
            "timeout_seconds": self.timeout_seconds,
            "returncode": self.returncode,
            "output": self.output,
            "command": list(self.command),
            "stderr_head": self.stderr_head,
            "stderr_tail": self.stderr_tail,
        }


@dataclass
class MotifCollectionResult:
    """Structured result returned by one motif worker."""

    candidate_id: str
    status: str
    stage: str
    failure: Optional[str] = None
    sample: Optional[Sample] = None
    invocations: Tuple[InvocationResult, ...] = ()


@dataclass
class MotifCoordinatorResult:
    """Harvested motif results in manifest/ordinal order."""

    samples: List[Sample]
    results: List[MotifCollectionResult]
    interrupted: bool = False


@dataclass
class SampleSpec:
    index: int
    seed: int
    rows: int
    columns: int
    split_domain: int
    sources: int
    operations: int
    fanout_bias: float
    registers: int


@dataclass
class CSpec:
    index: int
    seed: int
    rows: int
    columns: int
    terms: int
    loop_trip_count: int


def resolve_default_opt(repo: Path) -> Path:
    return repo / "build/tools/mlir-neura-opt/mlir-neura-opt"


def resolve_configured_neura_root() -> Optional[Path]:
    """Prefer an explicit environment override, then the pinned submodule."""
    configured_root = os.environ.get("NEURA_ROOT")
    if configured_root:
        return Path(configured_root)
    if (SUBMODULE_NEURA_ROOT / "CMakeLists.txt").is_file():
        return SUBMODULE_NEURA_ROOT
    return None


def resolve_default_llvm_tool(name: str) -> Path:
    resolved = shutil.which(name)
    return Path(resolved) if resolved else Path(name)


def write_architecture(path: Path, spec: SampleSpec) -> None:
    if spec.split_domain:
        tile_defaults = '["constant"]'
        first_compute_column = max(1, spec.columns // 2)
        overrides = "\n".join(
            "  - tile_x: {x}\n"
            "    tile_y: {y}\n"
            "    fu_types: [\"add\", \"mul\"]\n"
            "    num_registers: {regs}\n"
            "    existence: true".format(x=x, y=y, regs=spec.registers)
            for y in range(spec.rows)
            for x in range(first_compute_column, spec.columns)
        )
    else:
        tile_defaults = '["constant", "add", "mul"]'
        overrides = ""
    path.write_text(
        "\n".join(
            (
                "architecture:",
                '  name: "II Predictor Synthetic"',
                '  version: "1.0"',
                "",
                "multi_cgra_defaults:",
                '  base_topology: "mesh"',
                "  rows: 1",
                "  columns: 1",
                "",
                "per_cgra_defaults:",
                f"  rows: {spec.rows}",
                f"  columns: {spec.columns}",
                "  ctrl_mem_items: 20",
                '  base_topology: "mesh"',
                "",
                "tile_defaults:",
                f"  num_registers: {spec.registers}",
                f"  fu_types: {tile_defaults}",
                "",
                "link_defaults:",
                "  latency: 1",
                "  bandwidth: 32",
                "",
                "link_overrides:",
                "",
                "tile_overrides:",
                overrides,
                "",
            )
        )
    )


def pick_parent(rng: random.Random, values: Sequence[str], use_count: Dict[str, int],
                fanout_bias: float) -> str:
    # Prefer values already used by consumers at a controlled rate.  This
    # creates routing fanout pressure while retaining random graph shapes.
    weighted: List[Tuple[str, float]] = []
    for value in values:
        weight = 1.0 + fanout_bias * min(use_count[value], 3)
        weighted.append((value, weight))
    threshold = rng.random() * sum(weight for _, weight in weighted)
    for value, weight in weighted:
        threshold -= weight
        if threshold <= 0:
            return value
    return weighted[-1][0]


def write_dfg(path: Path, spec: SampleSpec) -> Dict[str, int]:
    rng = random.Random(spec.seed)
    lines = ["module {", "  func.func @synthetic() attributes {accelerator = \"neura\"} {"]
    values: List[str] = []
    use_count: Dict[str, int] = {}
    for index in range(spec.sources):
        value = f"%c{index}"
        lines.append(
            f"    {value} = \"neura.constant\"() <{{value = {index + 1} : i32}}> "
            f": () -> {DATA_TYPE}"
        )
        values.append(value)
        use_count[value] = 0

    depth: Dict[str, int] = {value: 0 for value in values}
    moves = 0
    for index in range(spec.operations):
        parent = pick_parent(rng, values, use_count, spec.fanout_bias)
        move = f"%m{index}"
        result = f"%v{index}"
        operation = "add" if rng.random() < 0.55 else "mul"
        immediate = rng.randint(1, 31)
        lines.append(
            f"    {move} = \"neura.data_mov\"({parent}) : ({DATA_TYPE}) -> {DATA_TYPE}"
        )
        lines.append(
            f"    {result} = \"neura.{operation}\"({move}) {{rhs_value = {immediate} : i32}} "
            f": ({DATA_TYPE}) -> {DATA_TYPE}"
        )
        use_count[parent] += 1
        values.append(result)
        use_count[result] = 0
        depth[result] = depth[parent] + 1
        moves += 1

    lines.extend(("    func.return", "  }", "}", ""))
    path.write_text("\n".join(lines))
    return {
        "nodes": spec.sources + spec.operations,
        "moves": moves,
        "ctrl_moves": 0,
        "reserves": 0,
        "phis": 0,
        "predicates": 0,
        "memory_ops": 0,
        "gep_ops": 0,
        "indirect_geps": 0,
        "pointer_loads": 0,
        "div_ops": 0,
        "arithmetic_ops": spec.operations,
        "comparisons": 0,
        "sources": spec.sources,
        "depth": max(depth.values()),
        "max_fanout": max(use_count.values()),
        "tiles": spec.rows * spec.columns,
        "links": 2 * (spec.rows * (spec.columns - 1) +
                      spec.columns * (spec.rows - 1)),
        "rows": spec.rows,
        "split_domain": spec.split_domain,
    }


def write_c_loop(path: Path, spec: CSpec) -> None:
    """Write a bounded memory/recurrence loop for frontend corpus coverage."""
    rng = random.Random(spec.seed)
    statements = [f"    int value_{index} = in{index % 3}[i];"
                  for index in range(spec.terms)]
    for index in range(spec.terms):
        coefficient = rng.randint(2, 17)
        # Restrict the generator to frontend operations accepted by the Neura
        # lowering pipeline (add/mul); unsupported LLVM opcodes are corpus
        # failures, not zero-II samples.
        statements.append(
            f"    state = state + value_{index} * {coefficient};")
    body = "\n".join(statements)
    path.write_text(
        "\n".join((
            'extern "C" void kernel(int *out, const int *in0,',
            '                       const int *in1, const int *in2) {',
            f"  int state = {rng.randint(0, 31)};",
            f"  for (int i = 0; i < {spec.loop_trip_count}; ++i) {{",
            body,
            "    out[i] = state;",
            "  }",
            "}",
            "",
        ))
    )


def invocation_stage(command: Sequence[str]) -> str:
    joined = " ".join(command)
    if "--map-to-accelerator" in joined:
        return "mapper"
    if "--analyze-rec-res-mii" in joined:
        return "rec-res-analysis"
    if any(token in joined for token in
           ("--lower-", "--import-llvm", "--assign-accelerator")):
        return "lowering"
    return "frontend"


def invocation_output(command: Sequence[str]) -> Optional[str]:
    try:
        output_index = command.index("-o") + 1
    except ValueError:
        return None
    return command[output_index] if output_index < len(command) else None


def record_invocation_failure(
    command: Sequence[str], status: str, timeout: int,
    returncode: Optional[int] = None, stderr: str = "",
) -> None:
    INVOCATION_FAILURES.append({
        "status": status,
        "stage": invocation_stage(command),
        "timeout_seconds": timeout,
        "returncode": returncode,
        "output": invocation_output(command),
        "command": list(command),
        "stderr_head": stderr[:1200],
        "stderr_tail": stderr[-2800:],
    })


def bounded_stderr_excerpt(stream: Any) -> str:
    """Read at most the retained head/tail from a seekable binary stream."""
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
        tail = stream.read(2800)
        payload = head + b"\n... stderr middle omitted ...\n" + tail
    return payload.decode(errors="replace")


def run_invocation(command: Sequence[str], timeout: int) -> InvocationResult:
    """Run one isolated command without mutating adapter-global state."""
    # A disk-backed temporary stream avoids buffering arbitrarily verbose
    # mapper diagnostics in memory.  Only a bounded excerpt enters the report.
    with tempfile.TemporaryFile() as stderr_stream:
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_stream,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            excerpt = bounded_stderr_excerpt(stderr_stream)
            return InvocationResult(
                ok=False,
                status="timeout",
                stage=invocation_stage(command),
                timeout_seconds=timeout,
                command=tuple(str(part) for part in command),
                output=invocation_output(command),
                stderr_head=excerpt[:1200],
                stderr_tail=excerpt[-2800:],
            )
        if completed.returncode != 0:
            excerpt = bounded_stderr_excerpt(stderr_stream)
            return InvocationResult(
                ok=False,
                status="nonzero-exit",
                stage=invocation_stage(command),
                timeout_seconds=timeout,
                command=tuple(str(part) for part in command),
                returncode=completed.returncode,
                output=invocation_output(command),
                stderr_head=excerpt[:1200],
                stderr_tail=excerpt[-2800:],
            )
    return InvocationResult(
        ok=True,
        status="success",
        stage=invocation_stage(command),
        timeout_seconds=timeout,
        command=tuple(str(part) for part in command),
        output=invocation_output(command),
    )


def invoke(command: Sequence[str], timeout: int) -> bool:
    """Compatibility wrapper for the legacy sequential collection paths."""
    result = run_invocation(command, timeout)
    if not result.ok:
        record_invocation_failure(
            command, result.status, timeout,
            returncode=result.returncode,
            stderr=(result.stderr_head + result.stderr_tail),
        )
    return result.ok


def command_stdout_sha256(command: Sequence[str]) -> Optional[str]:
    """Hash command output without retaining the entire stream in memory."""
    with tempfile.TemporaryFile() as output_stream:
        completed = subprocess.run(
            command, stdout=output_stream, stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            return None
        output_stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: output_stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def require_opt_argument(opt: Path, argument: str) -> None:
    """Fail before label collection when the selected compiler lacks a pass."""
    try:
        completed = subprocess.run(
            (str(opt), "--help"), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"cannot inspect {opt}: {error}") from error
    if completed.returncode != 0 or argument not in completed.stdout:
        raise ValueError(
            f"{opt} does not provide required pass argument {argument}"
        )


def git_provenance(root: Optional[Path]) -> Dict[str, object]:
    if root is None:
        return {"root": None, "revision": None, "dirty": None}
    revision = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    status = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    status_text = status.stdout if status.returncode == 0 else ""
    return {
        "root": str(root.resolve()),
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status_text.strip()) if status.returncode == 0 else None,
        "status_sha256": (
            hashlib.sha256(status_text.encode()).hexdigest()
            if status.returncode == 0 else None
        ),
        "tracked_diff_sha256": command_stdout_sha256(
            ("git", "-C", str(root), "diff", "--binary", "HEAD")
        ),
    }


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_sample_provenance(
    result: Sample, source: Path, architecture: Path,
    architecture_variant: str, lineage: str,
    dfg_source: Optional[Path] = None,
) -> None:
    """Attach auditable input identities without guessing suite membership."""
    source_hash = file_sha256(source)
    dfg_hash = file_sha256(dfg_source) if dfg_source is not None else source_hash
    architecture_hash = file_sha256(architecture)
    result.update({
        "lineage": lineage,
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "architecture_path": str(architecture.resolve()),
        "architecture_sha256": architecture_hash,
        "architecture_variant": architecture_variant,
        "architecture_id": f"{architecture_hash}:{architecture_variant}",
        # Include the DFG/source identity: the same architecture candidate can
        # label many different kernels and must never share a candidate ID.
        "candidate_id": (
            f"{dfg_hash}:{architecture_hash}:{architecture_variant}:heuristic"
        ),
        "base_dfg_id": dfg_hash,
        "ranking_query_id": dfg_hash,
        "mapper_id": "neura-heuristic",
        "mapper_config": "mapping-strategy=heuristic",
    })
    if dfg_source is not None:
        result["dfg_source_path"] = str(dfg_source.resolve())
        result["dfg_source_sha256"] = dfg_hash


def parse_integer_attribute(text: str, name: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i32", text)
    return int(match.group(1)) if match else None


def parse_cost_features(text: str) -> Optional[Dict[str, object]]:
    """Parse the analysis-only facts emitted by Neura main.

    The compiler pass calls the same C++ RecMII/ResMII implementation as the
    mapper.  No placement or routing is attempted, and ``compiled_ii`` is not
    part of this artifact.
    """
    forbidden = tuple(
        name for name in (
            "compiled_ii", "mapping_info", "mapping_strategy",
            "analytical_ii", "compute_mii", "mem_mii", "reg_mii",
            "route_mii", "infeasible", "exceeds_max_ii",
        )
        if re.search(rf"\b{re.escape(name)}\b", text)
    )
    if forbidden:
        raise ValueError(
            "analysis-only Rec/Res artifact contains mapping/label tokens: "
            + ", ".join(forbidden)
        )
    if re.search(r"\brec_res_mii_info\b", text) is None:
        return None
    required = {
        name: parse_integer_attribute(text, name) for name in COST_FEATURE_NAMES
    }
    if any(value is None for value in required.values()):
        return None
    result = {name: int(value) for name, value in required.items()}
    if any(value < 0 for value in result.values()):
        return None
    return result


def parse_checked_mapper_label(
    mapped_text: str, analysis: Mapping[str, object],
) -> Optional[int]:
    """Parse the label and require mapper/analysis Rec/Res identity.

    A mismatch is a compiler/protocol error, not a censored sample: training
    must never pair a label with lower-bound facts computed under a different
    contract.
    """
    compiled_ii = parse_integer_attribute(mapped_text, "compiled_ii")
    mapper_rec_mii = parse_integer_attribute(mapped_text, "rec_mii")
    mapper_res_mii = parse_integer_attribute(mapped_text, "res_mii")
    if compiled_ii is None or mapper_rec_mii is None or mapper_res_mii is None:
        return None
    expected = (int(analysis["rec_mii"]), int(analysis["res_mii"]))
    observed = (mapper_rec_mii, mapper_res_mii)
    if observed != expected:
        raise ValueError(
            "analysis-only and mapper Rec/Res facts disagree: "
            f"analysis={expected}, mapper={observed}"
        )
    if compiled_ii < max(expected):
        raise ValueError("compiled_ii is below max(RecMII, ResMII)")
    return compiled_ii


def resolve_rec_res_lower_bound(result: Sample) -> Tuple[int, str]:
    """Return the single authoritative v1 floor: max(RecMII, ResMII)."""
    components: Dict[str, int] = {}
    for name in LOWER_BOUND_COMPONENT_NAMES:
        raw_value = result.get(name)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{name} must be a non-negative integer")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0 or not value.is_integer():
            raise ValueError(f"{name} must be a non-negative integer")
        components[name] = int(value)
    bound = max(components.values())
    if bound < 1:
        raise ValueError("lower bound must be a positive integer")
    for alias in ("lower_bound", "baseline_lb", "proven_lower_bound"):
        if result.get(alias) is None:
            continue
        raw_alias = result[alias]
        if isinstance(raw_alias, bool) or not isinstance(raw_alias, (int, float)):
            raise ValueError(f"{alias} must be a positive integer")
        alias_value = float(raw_alias)
        if (
            not math.isfinite(alias_value) or alias_value < 1.0 or
            not alias_value.is_integer()
        ):
            raise ValueError(f"{alias} must be a positive integer")
        if int(alias_value) != bound:
            raise ValueError(
                f"{alias}={raw_alias} disagrees with "
                f"max(rec_mii,res_mii)={bound}"
            )
    return bound, "rec_res_max_v1"


SAMPLE_PROVENANCE_FIELDS = (
    "lineage",
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
    "rec_res_evidence",
    "mapped_artifact_path",
    "mapped_artifact_sha256",
    "cost_artifact_path",
    "cost_artifact_sha256",
    "generator_family",
    "generator_type",
    "generator_version",
    "motif",
    "base_id",
    "base_seed",
    "root_seed",
    "base_index",
    "operation_count",
    "canonical_dfg_sha256",
    "registers",
    "leakage_lineage_id",
    "declared_leakage_lineage_id",
    "base_dfg_id",
    "ranking_query_id",
    "training_stratum",
    "target_config_id",
    "valid_tiles",
)


def normalize_input_sample(
    raw: Dict[str, object], report_path: Path,
    report_sha256: Optional[str],
    report_provenance: Optional[Mapping[str, object]] = None,
) -> Sample:
    """Accept both legacy flat rows and portable nested samples."""
    row: Sample = dict(raw)
    nested_features = row.pop("features", None)
    if isinstance(nested_features, dict):
        row.update(nested_features)
    metadata = row.pop("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("sample metadata must be an object")
    row.setdefault("index", row.get("sample_id"))
    row.setdefault("family", row.get("group"))
    if row.get("baseline_lb") is None:
        authoritative_alias = row.get("lower_bound")
        if authoritative_alias is None:
            authoritative_alias = row.get("proven_lower_bound")
        if authoritative_alias is not None:
            row["baseline_lb"] = authoritative_alias
    for field in SAMPLE_PROVENANCE_FIELDS:
        if field not in row and field in metadata:
            row[field] = metadata[field]
    if report_provenance:
        if report_provenance.get("architecture") is not None:
            row.setdefault("architecture_path", report_provenance["architecture"])
        if report_provenance.get("architecture_sha256") is not None:
            row.setdefault(
                "architecture_sha256", report_provenance["architecture_sha256"]
            )
        if report_provenance.get("mapping_strategy") is not None:
            row.setdefault("mapper_id", report_provenance["mapping_strategy"])
        neura = report_provenance.get("neura")
        if isinstance(neura, Mapping) and neura.get("revision") is not None:
            row.setdefault("mapper_revision", neura["revision"])
    if row.get("index") is None or row.get("family") is None:
        raise ValueError("input sample needs index/sample_id and family/group")
    source_kind = str(row.get("source_kind", "")).lower()
    source_identity = str(row.get("source_family", row.get("family", "")))
    generated = source_kind in {"generated", "synthetic"} or source_identity.startswith(
        ("generated", "synthetic")
    )
    row.setdefault("training_stratum", "generated" if generated else "real")
    row.setdefault(
        "leakage_lineage_id",
        row.get("lineage", row.get("family")),
    )
    # Keep ranking unavailable for legacy rows that have no explicit source
    # identity.  A merged lineage is a fit/holdout unit, not a DSE query.
    base_dfg_id = row.get("base_dfg_id")
    if base_dfg_id in (None, ""):
        base_dfg_id = row.get(
            "canonical_dfg_sha256",
            row.get("dfg_source_sha256", row.get("source_sha256")),
        )
    if base_dfg_id not in (None, ""):
        row.setdefault("base_dfg_id", base_dfg_id)
        row.setdefault("ranking_query_id", base_dfg_id)
    row["input_report_path"] = str(report_path.resolve())
    row["input_report_sha256"] = report_sha256
    row["rec_res_evidence"] = "imported_report_unverified"
    return row


def deduplicate_samples_by_id(samples: Sequence[Sample]) -> List[Sample]:
    """Collapse identical imports and reject any conflicting repeated ID."""
    unique_samples: Dict[str, Sample] = {}
    unique_fingerprints: Dict[str, str] = {}
    for row in samples:
        sample_id = str(row["index"])
        fingerprint = json.dumps({
            name: value for name, value in row.items()
            if name not in {"input_report_path", "input_report_sha256"}
        }, sort_keys=True, allow_nan=False)
        prior = unique_fingerprints.get(sample_id)
        if prior is not None and prior != fingerprint:
            raise ValueError(
                f"conflicting duplicate sample ID across reports: {sample_id}"
            )
        unique_fingerprints[sample_id] = fingerprint
        unique_samples.setdefault(sample_id, row)
    return list(unique_samples.values())


def resolve_effective_lineage(
    row: Mapping[str, object], source_family: str,
    family_lineages: Mapping[str, str],
) -> Tuple[str, str]:
    """Honor an explicit conservative lineage unless an alias merges it."""
    declared = str(row.get(
        "leakage_lineage_id", row.get("lineage", source_family)
    ))
    effective = family_lineages.get(
        source_family, family_lineages.get(declared, declared)
    )
    return declared, effective


def load_sibling_cost_features(report_path: Path) -> Dict[str, Dict[str, int]]:
    """Recover Rec/Res facts from current analysis artifacts beside a report.

    Solver-branch artifacts are deliberately rejected by
    :func:`parse_cost_features`; their extra MII fields identify a different
    producer contract and must be recollected on main.
    """
    result: Dict[str, Dict[str, int]] = {}
    for cost_path in report_path.parent.glob("real-*/cost.mlir"):
        directory = cost_path.parent.name
        standard = re.fullmatch(r"real-(.+)-(\d+x\d+)", directory)
        mask = re.fullmatch(r"real-(.+)-mask-(\d{3})", directory)
        if standard:
            index = f"{standard.group(1)}-{standard.group(2)}"
        elif mask:
            index = f"{mask.group(1)}-4x4-mask{mask.group(2)}"
        else:
            continue
        text = cost_path.read_text()
        values = parse_cost_features(text)
        if values is not None:
            result[index] = values
    return result


def add_prediction_features(result: Sample) -> None:
    """Add only pre-mapping features; this never changes a mapping decision."""
    bound, source = resolve_rec_res_lower_bound(result)
    result["baseline_lb"] = bound
    result["lower_bound_source"] = source
    if "compiled_ii" in result and int(result["compiled_ii"]) < bound:
        raise ValueError("compiled_ii is below max(RecMII, ResMII)")
    result["rec_res_gap"] = result["rec_mii"] - result["res_mii"]
    result["rec_dominant"] = int(result["rec_mii"] >= result["res_mii"])


def attach_rec_res_artifact(result: Sample, artifact: Path) -> None:
    """Record the compiler-produced analysis evidence used by this row."""
    result.update({
        "rec_res_evidence": "neura_shared_rec_res_analysis_v1",
        "cost_artifact_path": str(artifact.resolve()),
        "cost_artifact_sha256": file_sha256(artifact),
    })


def collect_sample(opt: Path, sample_dir: Path, spec: SampleSpec,
                   timeout: int) -> Optional[Sample]:
    source = sample_dir / "input.mlir"
    arch = sample_dir / "architecture.yaml"
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    graph_features = write_dfg(source, spec)
    graph_features.update(semantic_features_from_neura(source.read_text()))
    write_architecture(arch, spec)

    if not invoke(
        (str(opt), str(source), f"--architecture-spec={arch}",
         "--analyze-rec-res-mii", "-o", str(cost)),
        timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={arch}",
         '--map-to-accelerator=mapping-strategy=heuristic', "-o", str(mapped)),
        timeout,
    ):
        return None

    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = parse_cost_features(cost_text)
    if values is None:
        return None
    compiled_ii = parse_checked_mapper_label(mapped_text, values)
    if compiled_ii is None:
        return None
    result: Sample = dict(values)
    result["compiled_ii"] = int(compiled_ii)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features)
    add_prediction_features(result)
    result.update(asdict(spec))
    result["family"] = "synthetic"
    attach_sample_provenance(
        result, source, arch,
        f"{spec.rows}x{spec.columns}:split-domain={spec.split_domain}",
        "synthetic-dfg-template",
    )
    return result


def _motif_sample_from_artifacts(
    candidate: neura_motifs.MotifCandidate,
    values: Mapping[str, object],
    compiled_ii: int,
    cost: Path,
    mapped: Path,
) -> Sample:
    """Build a sample from already validated artifacts, without subprocesses."""
    source = Path(candidate.source_path)
    architecture = Path(candidate.architecture_path)
    result: Sample = dict(values)
    result["compiled_ii"] = int(compiled_ii)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features_from_neura(
        source.read_text(), candidate.rows, candidate.columns
    ))
    add_prediction_features(result)
    result.update({
        "index": candidate.candidate_id,
        "family": candidate.lineage,
        "lineage": candidate.lineage,
        "effective_lineage": candidate.lineage,
        "leakage_lineage_id": candidate.lineage,
        "base_dfg_id": candidate.canonical_dfg_sha256,
        "ranking_query_id": candidate.canonical_dfg_sha256,
        "training_stratum": "generated",
        "source_family": (
            f"generated/{candidate.generator_version}/{candidate.motif}"
        ),
        "source_kind": "generated",
        "generator_family": candidate.generator_family,
        "generator_type": candidate.generator_type,
        "generator_version": candidate.generator_version,
        "motif": candidate.motif,
        "base_id": candidate.base_id,
        "base_seed": candidate.base_seed,
        "root_seed": candidate.root_seed,
        "base_index": candidate.base_index,
        "operation_count": candidate.operation_count,
        "canonical_dfg_sha256": candidate.canonical_dfg_sha256,
        "source_sha256": candidate.source_sha256,
        "source_path": str(source.resolve()),
        "architecture_path": str(architecture.resolve()),
        "architecture_sha256": candidate.architecture_sha256,
        "architecture_variant": candidate.architecture_variant,
        "architecture_id": candidate.architecture_id,
        "target_config_id": candidate.target_config_id,
        "valid_tiles": candidate.valid_tiles,
        "candidate_id": candidate.candidate_id,
        "mapper_id": "neura-heuristic",
        "mapper_config": (
            "mapping-strategy=heuristic "
            f"x-tiles={candidate.columns} y-tiles={candidate.rows}"
        ),
        "mapped_artifact_path": str(mapped.resolve()),
        "mapped_artifact_sha256": file_sha256(mapped),
        "registers": candidate.registers,
    })
    return result


def _coerce_invocation_result(
    value: object, command: Sequence[str], timeout: int,
) -> InvocationResult:
    """Accept bool-returning test doubles while keeping workers structured."""
    if isinstance(value, InvocationResult):
        return value
    if isinstance(value, bool):
        return InvocationResult(
            ok=value,
            status="success" if value else "invocation-failed",
            stage=invocation_stage(command),
            timeout_seconds=timeout,
            command=tuple(str(part) for part in command),
        )
    raise TypeError("isolated invocation must return InvocationResult or bool")


def collect_motif_candidate(
    opt: Path, candidate: neura_motifs.MotifCandidate, timeout: int,
    invocation: Callable[[Sequence[str], int], object] = run_invocation,
) -> MotifCollectionResult:
    """Collect one candidate with no global, manifest, or stdout mutation.

    The analysis and mapper invocations are intentionally serial within this
    function.  A coordinator may run independent candidates concurrently.
    """
    source = Path(candidate.source_path)
    architecture = Path(candidate.architecture_path)
    sample_dir = source.parent
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    calls: List[InvocationResult] = []
    if (
        file_sha256(source) != candidate.source_sha256 or
        file_sha256(architecture) != candidate.architecture_sha256
    ):
        raise ValueError(
            f"candidate {candidate.candidate_id} predeclared input hash changed"
        )

    if candidate.valid_tiles:
        raise ValueError("motif-v3 does not permit valid-tiles masks")
    target_options = f"x-tiles={candidate.columns} y-tiles={candidate.rows}"
    analysis_command = (
        str(opt), str(source), f"--architecture-spec={architecture}",
        f"--analyze-rec-res-mii={target_options}", "-o", str(cost),
    )
    analysis = _coerce_invocation_result(
        invocation(analysis_command, timeout), analysis_command, timeout,
    )
    calls.append(analysis)
    if not analysis.ok:
        return MotifCollectionResult(
            candidate.candidate_id, "censored", "rec-res-analysis",
            analysis.status, invocations=tuple(calls),
        )
    try:
        values = parse_cost_features(cost.read_text())
    except OSError:
        values = None
    if values is None:
        raise ValueError(
            f"candidate {candidate.candidate_id} has invalid Rec/Res facts"
        )

    mapper_command = (
        str(opt), str(source), f"--architecture-spec={architecture}",
        "--map-to-accelerator=mapping-strategy=heuristic " + target_options,
        "-o", str(mapped),
    )
    mapper = _coerce_invocation_result(
        invocation(mapper_command, timeout), mapper_command, timeout,
    )
    calls.append(mapper)
    if not mapper.ok:
        return MotifCollectionResult(
            candidate.candidate_id, "censored", "mapper", mapper.status,
            invocations=tuple(calls),
        )
    try:
        mapped_text = mapped.read_text()
        compiled_ii = parse_checked_mapper_label(mapped_text, values)
        expected_dimensions = {
            "x_tiles": candidate.columns,
            "y_tiles": candidate.rows,
        }
        for name, expected in expected_dimensions.items():
            match = re.search(rf"\b{name}\s*=\s*(\d+)\s*:\s*i32", mapped_text)
            if match is None or int(match.group(1)) != expected:
                compiled_ii = None
    except OSError:
        compiled_ii = None
    if compiled_ii is None:
        return MotifCollectionResult(
            candidate.candidate_id, "censored", "label-parse",
            "compiled_ii-unavailable", invocations=tuple(calls),
        )
    sample = _motif_sample_from_artifacts(
        candidate, values, int(compiled_ii), cost, mapped
    )
    return MotifCollectionResult(
        candidate.candidate_id, "success", "mapper", sample=sample,
        invocations=tuple(calls),
    )


def collect_motif_sample(
    opt: Path, candidate: neura_motifs.MotifCandidate, timeout: int,
    manifest_path: Optional[Path] = None,
) -> Optional[Sample]:
    """Legacy sequential wrapper retaining the old bool/manifest API."""
    def legacy_invocation(command: Sequence[str], limit: int) -> object:
        return invoke(command, limit)

    outcome = collect_motif_candidate(opt, candidate, timeout, legacy_invocation)
    if manifest_path is not None:
        updates: Optional[Mapping[str, object]] = None
        if outcome.sample is not None:
            cost = Path(candidate.source_path).parent / "cost.mlir"
            mapped = Path(candidate.source_path).parent / "mapped.mlir"
            updates = {
                "sample_id": candidate.candidate_id,
                "compiled_ii": int(outcome.sample["compiled_ii"]),
                "lower_bound": int(outcome.sample["baseline_lb"]),
                "cost_artifact_path": str(cost.relative_to(manifest_path.parent)),
                "cost_artifact_sha256": file_sha256(cost),
                "mapped_artifact_path": str(mapped.relative_to(manifest_path.parent)),
                "mapped_artifact_sha256": file_sha256(mapped),
            }
        neura_motifs.update_manifest_candidate(
            manifest_path, candidate.candidate_id, outcome.status,
            outcome.stage, outcome.failure, updates,
        )
    return outcome.sample


MOTIF_REGISTERS = neura_motifs.PINNED_REGISTERS_PER_TILE
MOTIF_IMMUTABLE_FIELDS = (
    "id", "candidate_id", "lineage", "motif", "generator_family",
    "generator_version", "generator_type", "base_id", "base_seed",
    "root_seed", "base_index", "operation_count", "rows", "columns",
    "architecture_variant", "registers", "source_path",
    "architecture_path", "source_sha256", "canonical_dfg_sha256",
    "architecture_sha256", "target_config_id", "valid_tiles",
    "architecture_id", "leakage_lineage_id",
    "base_dfg_id", "ranking_query_id", "training_stratum",
)
MOTIF_SUCCESS_ARTIFACTS = (
    "cost_artifact_path", "cost_artifact_sha256",
    "mapped_artifact_path", "mapped_artifact_sha256",
)
MOTIF_TRANSIENT_FIELDS = (
    "sample_id", "compiled_ii", "lower_bound",
    *MOTIF_SUCCESS_ARTIFACTS,
)


def _path_inside(root: Path, raw_path: object, field: str) -> Path:
    """Resolve a manifest path and reject absolute/escaping/symlink paths."""
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise ValueError(f"manifest {field} must be a relative path")
    root_resolved = root.resolve()
    resolved = (root / raw_path).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"manifest {field} escapes output directory")
    return resolved


def _relative_to_manifest(root: Path, path: Path) -> str:
    """Return the canonical relative representation used in manifests."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"path escapes output directory: {path}")
    return resolved_path.relative_to(resolved_root).as_posix()


def _expected_manifest_record(
    candidate: neura_motifs.MotifCandidate, temporary_root: Path,
) -> Dict[str, object]:
    record = candidate.manifest_record()
    record["source_path"] = _relative_to_manifest(
        temporary_root, Path(candidate.source_path)
    )
    record["architecture_path"] = _relative_to_manifest(
        temporary_root, Path(candidate.architecture_path)
    )
    return record


def _motif_collection_config(
    timeout: int, jobs: int, checkpoint_every: int, *,
    opt_path: str, opt_sha256: Optional[str],
) -> Dict[str, object]:
    return {
        "timeout_seconds": int(timeout),
        "motif_jobs": int(jobs),
        "motif_checkpoint_every": int(checkpoint_every),
        "candidate_execution": "thread-pool-candidate-serial-stages",
        "manifest_updates": "main-thread-ordinal-batch-atomic",
        "mlir_neura_opt": opt_path,
        "mlir_neura_opt_sha256": opt_sha256,
        "analysis_argument": "--analyze-rec-res-mii",
        "mapping_strategy": "heuristic",
    }


def _candidate_from_manifest(
    record: Mapping[str, object], output_dir: Path,
) -> neura_motifs.MotifCandidate:
    """Convert a validated manifest record to a worker candidate."""
    source = _path_inside(output_dir, record.get("source_path"), "source_path")
    architecture = _path_inside(
        output_dir, record.get("architecture_path"), "architecture_path"
    )
    required = (
        "id", "lineage", "motif", "generator_family", "generator_version",
        "generator_type", "base_id", "base_seed", "root_seed", "base_index",
        "operation_count", "rows", "columns", "architecture_variant",
        "registers", "source_sha256", "canonical_dfg_sha256",
        "architecture_sha256", "target_config_id", "valid_tiles",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError("manifest candidate missing fields: " + ", ".join(missing))
    candidate_id = str(record["id"])
    return neura_motifs.MotifCandidate(
        candidate_id=candidate_id,
        lineage=str(record["lineage"]),
        motif=str(record["motif"]),
        generator_family=str(record["generator_family"]),
        generator_version=str(record["generator_version"]),
        generator_type=str(record["generator_type"]),
        base_id=str(record["base_id"]),
        base_seed=int(record["base_seed"]),
        root_seed=int(record["root_seed"]),
        base_index=int(record["base_index"]),
        operation_count=int(record["operation_count"]),
        rows=int(record["rows"]),
        columns=int(record["columns"]),
        architecture_variant=str(record["architecture_variant"]),
        registers=int(record["registers"]),
        source_path=str(source),
        architecture_path=str(architecture),
        source_sha256=str(record["source_sha256"]),
        canonical_dfg_sha256=str(record["canonical_dfg_sha256"]),
        architecture_sha256=str(record["architecture_sha256"]),
        target_config_id=str(record["target_config_id"]),
        valid_tiles=str(record["valid_tiles"]),
    )


def _validate_manifest_record_identity(
    actual: Mapping[str, object], expected: Mapping[str, object],
) -> None:
    for field in MOTIF_IMMUTABLE_FIELDS:
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"manifest immutable field mismatch for {field}: "
                f"{actual.get(field)!r} != {expected.get(field)!r}"
            )


def _clear_transient_manifest_fields(record: Dict[str, object]) -> None:
    for field in MOTIF_TRANSIENT_FIELDS:
        record.pop(field, None)


def _validate_cached_motif_success(
    candidate: neura_motifs.MotifCandidate,
    record: Mapping[str, object], output_dir: Path,
) -> Sample:
    """Validate all four files of a cached success and rebuild its sample."""
    source = _path_inside(output_dir, record.get("source_path"), "source_path")
    architecture = _path_inside(
        output_dir, record.get("architecture_path"), "architecture_path"
    )
    candidate_dir = source.parent
    expected_cost = candidate_dir / "cost.mlir"
    expected_mapped = candidate_dir / "mapped.mlir"
    for field, expected in (
        ("cost_artifact_path", expected_cost),
        ("mapped_artifact_path", expected_mapped),
    ):
        actual = _path_inside(output_dir, record.get(field), field)
        if actual != expected.resolve():
            raise ValueError(
                f"cached success {field} does not match candidate artifact path"
            )
    if file_sha256(source) != candidate.source_sha256:
        raise ValueError("cached success source hash mismatch")
    if file_sha256(architecture) != candidate.architecture_sha256:
        raise ValueError("cached success architecture hash mismatch")
    if file_sha256(expected_cost) != record.get("cost_artifact_sha256"):
        raise ValueError("cached success cost artifact hash mismatch")
    if file_sha256(expected_mapped) != record.get("mapped_artifact_sha256"):
        raise ValueError("cached success mapped artifact hash mismatch")
    try:
        values = parse_cost_features(expected_cost.read_text())
        if values is None:
            raise ValueError("cached success has invalid Rec/Res artifact")
        mapped_text = expected_mapped.read_text()
        compiled_ii = parse_checked_mapper_label(mapped_text, values)
        for name, expected in (
            ("x_tiles", candidate.columns), ("y_tiles", candidate.rows),
        ):
            match = re.search(rf"\b{name}\s*=\s*(\d+)\s*:\s*i32", mapped_text)
            if match is None or int(match.group(1)) != expected:
                raise ValueError(f"cached success {name} mismatch")
        if compiled_ii is None:
            raise ValueError("cached success has no compiled_ii")
    except (OSError, ValueError) as error:
        raise ValueError(f"cached success artifact validation failed: {error}") from error
    if record.get("sample_id") != candidate.candidate_id:
        raise ValueError("cached success sample_id mismatch")
    if int(record.get("compiled_ii", -1)) != int(compiled_ii):
        raise ValueError("cached success compiled_ii mismatch")
    bound = max(int(values["rec_mii"]), int(values["res_mii"]))
    if int(record.get("lower_bound", -1)) != bound:
        raise ValueError("cached success lower_bound mismatch")
    return _motif_sample_from_artifacts(
        candidate, values, int(compiled_ii), expected_cost, expected_mapped
    )


def _load_or_create_motif_manifest(
    output_dir: Path, manifest_path: Path, *, resume: bool, clean: bool,
    count: int, seed: int, motifs: Sequence[str],
    shapes: Sequence[Tuple[int, int]], variants: Sequence[str],
    timeout: int, jobs: int, checkpoint_every: int,
    opt: Path,
    registers: int = MOTIF_REGISTERS,
    architecture_source: Optional[Path] = None,
) -> Tuple[
    Dict[str, object], Tuple[neura_motifs.MotifCandidate, ...],
    Dict[str, Sample], List[str], Dict[str, List[Dict[str, object]]],
]:
    """Prepare a fresh or resumed corpus before any tool invocation."""
    if resume and clean:
        raise ValueError("--clean and --motif-resume are mutually exclusive")
    output_dir = output_dir.resolve()
    manifest_path = manifest_path.resolve()
    if resume:
        if not manifest_path.is_file():
            raise ValueError(f"motif manifest not found for resume: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid motif manifest: {error}") from error
        if not isinstance(manifest, dict):
            raise ValueError("motif manifest must be an object")
        if manifest.get("schema_version") != neura_motifs.MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported motif manifest schema_version")
        if manifest.get("output_dir") != ".":
            raise ValueError("motif manifest output_dir must be relative '.'")
        generator = manifest.get("generator")
        if not isinstance(generator, dict):
            raise ValueError("motif manifest lacks generator configuration")
        if generator.get("family") != "generated/motif":
            raise ValueError("motif manifest generator family mismatch")
        if generator.get("type") != "generated/motif":
            raise ValueError("motif manifest generator type mismatch")
        if generator.get("version") != neura_motifs.GENERATOR_VERSION:
            raise ValueError("motif manifest generator version mismatch")
        if generator.get("candidate_design") != neura_motifs.SHAPE_DESIGN:
            raise ValueError("motif manifest candidate design mismatch")
        if "count_per_family" not in generator:
            raise ValueError("motif manifest lacks count_per_family")
        manifest_count = int(generator.get("count_per_family", 0))
        manifest_motifs = tuple(str(value) for value in generator.get("motifs", ()))
        manifest_shapes = tuple(
            neura_motifs.parse_shape(str(value))
            for value in generator.get("shapes", ())
        )
        manifest_variants = tuple(
            str(value) for value in generator.get("architecture_variants", ())
        )
        manifest_seed = int(generator.get("seed"))
        manifest_registers = int(generator.get("registers", registers))
        if count and count != manifest_count:
            raise ValueError("resume generator count does not match manifest")
        if tuple(motifs) != manifest_motifs:
            raise ValueError("resume motif family configuration does not match manifest")
        if tuple(shapes) != manifest_shapes:
            raise ValueError("resume shape configuration does not match manifest")
        if tuple(variants) != manifest_variants:
            raise ValueError(
                "resume architecture variant configuration does not match manifest"
            )
        if int(seed) != manifest_seed:
            raise ValueError("resume seed does not match manifest")
        if manifest_registers != int(registers):
            raise ValueError("resume register configuration does not match manifest")
        architecture_record = manifest.get("architecture")
        expected_architecture_record = {
            "sha256": neura_motifs.PINNED_ARCHITECTURE_SHA256,
            "neura_revision": neura_motifs.PINNED_NEURA_REVISION,
            "source_path": (
                neura_motifs.PINNED_ARCHITECTURE_RELATIVE_PATH.as_posix()
            ),
            "rows": neura_motifs.PINNED_ARCHITECTURE_ROWS,
            "columns": neura_motifs.PINNED_ARCHITECTURE_COLUMNS,
            "registers_per_tile": neura_motifs.PINNED_REGISTERS_PER_TILE,
            "ctrl_mem_items": neura_motifs.PINNED_CTRL_MEM_ITEMS,
            "target_shape_design": neura_motifs.SHAPE_DESIGN,
            "valid_tiles": "",
        }
        if not isinstance(architecture_record, dict):
            raise ValueError("motif manifest lacks the pinned architecture contract")
        for name, expected in expected_architecture_record.items():
            if architecture_record.get(name) != expected:
                raise ValueError(f"motif manifest architecture {name} mismatch")
        architecture_path = _path_inside(
            output_dir, architecture_record.get("path"), "architecture.path"
        )
        if file_sha256(architecture_path) != neura_motifs.PINNED_ARCHITECTURE_SHA256:
            raise ValueError("motif corpus pinned architecture hash mismatch")
        collection = manifest.get("collection")
        if not isinstance(collection, dict):
            raise ValueError("motif manifest lacks collection configuration")
        required_collection = {
            "timeout_seconds",
            "candidate_execution",
            "manifest_updates",
            "mlir_neura_opt",
            "mlir_neura_opt_sha256",
            "analysis_argument",
            "mapping_strategy",
        }
        missing_collection = required_collection.difference(collection)
        if missing_collection:
            raise ValueError(
                "motif manifest collection configuration is incomplete: "
                + ", ".join(sorted(missing_collection))
            )
        if int(collection["timeout_seconds"]) != int(timeout):
            raise ValueError("resume timeout does not match manifest")
        if collection["candidate_execution"] != "thread-pool-candidate-serial-stages":
            raise ValueError("motif manifest candidate execution contract mismatch")
        if collection["manifest_updates"] != "main-thread-ordinal-batch-atomic":
            raise ValueError("motif manifest update contract mismatch")
        if collection["analysis_argument"] != "--analyze-rec-res-mii":
            raise ValueError("motif manifest analysis contract mismatch")
        if collection["mapping_strategy"] != "heuristic":
            raise ValueError("motif manifest mapper contract mismatch")
        stored_opt_path = collection["mlir_neura_opt"]
        stored_opt_sha256 = collection["mlir_neura_opt_sha256"]
        if not isinstance(stored_opt_path, str) or not stored_opt_path:
            raise ValueError("motif manifest has invalid compiler path")
        if (
            not isinstance(stored_opt_sha256, str) or
            re.fullmatch(r"[0-9a-f]{64}", stored_opt_sha256) is None
        ):
            raise ValueError("motif manifest has invalid compiler SHA-256")
        records = manifest.get("candidates")
        if not isinstance(records, list):
            raise ValueError("resume manifest candidate declaration is incomplete")
        cached_samples: Dict[str, Sample] = {}
        prior_failure_events: Dict[str, List[Dict[str, object]]] = {}
        candidates: List[neura_motifs.MotifCandidate] = []
        declared_ids: List[str] = []
        with tempfile.TemporaryDirectory(prefix="ii-motif-expected-") as raw_root:
            temporary_root = Path(raw_root)
            # Re-materialize only in the temporary tree to derive the expected
            # relative identities; never repair the user's output tree.
            expected_materialized = neura_motifs.make_candidates(
                neura_motifs.make_base_specs(
                    manifest_count, manifest_seed, manifest_motifs
                ), temporary_root, manifest_shapes, manifest_variants,
                manifest_registers, architecture_path,
            )
            expected_count = len(expected_materialized)
            if manifest.get("candidate_count") != expected_count:
                raise ValueError("motif manifest candidate_count mismatch")
            if len(records) != expected_count:
                raise ValueError("resume manifest candidate declaration is incomplete")
            for ordinal, (raw_record, expected_candidate) in enumerate(
                zip(records, expected_materialized)
            ):
                if not isinstance(raw_record, dict):
                    raise ValueError(f"manifest candidate {ordinal} is not an object")
                expected_record = _expected_manifest_record(
                    expected_candidate, temporary_root
                )
                _validate_manifest_record_identity(raw_record, expected_record)
                candidate = _candidate_from_manifest(raw_record, output_dir)
                if candidate.candidate_id != expected_candidate.candidate_id:
                    raise ValueError("resume candidate order/identity mismatch")
                # Check both immutable source files before any compiler command.
                if file_sha256(Path(candidate.source_path)) != candidate.source_sha256:
                    raise ValueError(
                        f"resume source hash mismatch before invocation: {candidate.candidate_id}"
                    )
                if file_sha256(Path(candidate.architecture_path)) != candidate.architecture_sha256:
                    raise ValueError(
                        f"resume architecture hash mismatch before invocation: {candidate.candidate_id}"
                    )
                status = str(raw_record.get("status", "declared"))
                if status == "running":
                    # Older interrupted manifests are normalized without ever
                    # retrying a supposed success or persisting running again.
                    raw_record["status"] = "declared"
                    raw_record["stage"] = "predeclared"
                    raw_record["failure"] = None
                    _clear_transient_manifest_fields(raw_record)
                    raw_record.pop("invocation_failures", None)
                    status = "declared"
                if status not in {"declared", "censored", "success"}:
                    raise ValueError(f"invalid resume candidate status: {status}")
                if status == "success":
                    cached_samples[candidate.candidate_id] = _validate_cached_motif_success(
                        candidate, raw_record, output_dir
                    )
                elif status == "censored":
                    events = raw_record.get("invocation_failures")
                    if not isinstance(events, list):
                        raise ValueError(
                            "censored resume record lacks structured invocation_failures"
                        )
                    normalized_events: List[Dict[str, object]] = []
                    for event in events:
                        if not isinstance(event, dict):
                            raise ValueError(
                                "censored resume invocation failure is not an object"
                            )
                        restored = dict(event)
                        restored["candidate_id"] = candidate.candidate_id
                        normalized_events.append(restored)
                    prior_failure_events[candidate.candidate_id] = normalized_events
                elif status == "declared":
                    declared_ids.append(candidate.candidate_id)
                candidates.append(candidate)
        if declared_ids:
            current_opt_sha256 = file_sha256(opt)
            if current_opt_sha256 != stored_opt_sha256:
                raise ValueError(
                    "resume compiler SHA-256 does not match the original corpus"
                )
            effective_opt_path = str(opt.resolve())
            effective_opt_sha256 = current_opt_sha256
        else:
            # A terminal resume needs no compiler and may run after the
            # originally recorded executable has moved or disappeared.
            effective_opt_path = stored_opt_path
            effective_opt_sha256 = stored_opt_sha256
        manifest["collection"] = _motif_collection_config(
            timeout, jobs, checkpoint_every,
            opt_path=effective_opt_path, opt_sha256=effective_opt_sha256,
        )
        manifest["summary"] = neura_motifs.manifest_summary(records)
        manifest["status"] = (
            "incomplete" if any(
                record.get("status") == "declared" for record in records
            ) else "partial" if any(
                record.get("status") == "censored" for record in records
            ) else "complete"
        )
        # Persist running->declared recovery before the first resumed command.
        neura_motifs.atomic_write_json(manifest_path, manifest)
        return (
            manifest, tuple(candidates), cached_samples, declared_ids,
            prior_failure_events,
        )

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise ValueError(
            f"motif manifest already exists; use --motif-resume or --clean: {manifest_path}"
        )
    candidates = neura_motifs.make_candidates(
        neura_motifs.make_base_specs(count, seed, motifs), output_dir,
        shapes, variants, registers, architecture_source,
    )
    manifest = neura_motifs.make_manifest(
        candidates, output_dir, seed, motifs, shapes, variants, registers
    )
    manifest["generator"].update({"count_per_family": int(count)})
    manifest["collection"] = _motif_collection_config(
        timeout, jobs, checkpoint_every,
        opt_path=str(opt.resolve()), opt_sha256=file_sha256(opt),
    )
    neura_motifs.atomic_write_json(manifest_path, manifest)
    return (
        manifest, candidates, {}, [candidate.candidate_id for candidate in candidates],
        {},
    )


class MotifCollectionCoordinator:
    """Bounded candidate-parallel collector with main-thread checkpoints."""

    def __init__(
        self, opt: Path, candidates: Sequence[neura_motifs.MotifCandidate],
        manifest_path: Path, manifest: Dict[str, object], timeout: int,
        jobs: int = 1, checkpoint_every: int = 32,
        cached_samples: Optional[Mapping[str, Sample]] = None,
        prior_failure_events: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None,
    ) -> None:
        if jobs < 1:
            raise ValueError("motif jobs must be positive")
        if checkpoint_every < 1:
            raise ValueError("motif checkpoint interval must be positive")
        self.opt = opt
        self.candidates = tuple(candidates)
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.timeout = timeout
        self.jobs = jobs
        self.checkpoint_every = checkpoint_every
        self.cached_samples = dict(cached_samples or {})
        self._failure_events: Dict[str, List[Dict[str, object]]] = {
            str(candidate_id): [dict(event) for event in events]
            for candidate_id, events in (prior_failure_events or {}).items()
        }
        self._failure_events_finalized = False
        self._records = list(manifest.get("candidates", []))
        self._ordinal = {
            candidate.candidate_id: index
            for index, candidate in enumerate(self.candidates)
        }
        self._stop_requested = False
        self._interrupted = False
        self._completion_count = 0
        self._results: Dict[str, MotifCollectionResult] = {}
        self._errors: List[BaseException] = []

    def request_stop(self) -> None:
        """Request a cooperative stop (also useful for deterministic tests)."""
        self._interrupted = True
        self._stop_requested = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def _signal_handler(self, _signum: int, _frame: object) -> None:
        self._interrupted = True
        self.request_stop()

    def _record_for(self, candidate_id: str) -> Dict[str, object]:
        ordinal = self._ordinal[candidate_id]
        record = self._records[ordinal]
        if record.get("id", record.get("candidate_id")) != candidate_id:
            raise RuntimeError("manifest candidate order changed during collection")
        return record

    def _apply_result(self, outcome: MotifCollectionResult) -> None:
        record = self._record_for(outcome.candidate_id)
        if outcome.status not in {"success", "censored"}:
            raise RuntimeError(f"worker returned invalid status: {outcome.status}")
        success_updates: Dict[str, object] = {}
        if outcome.status == "success":
            if outcome.sample is None:
                raise RuntimeError("successful worker result lacks sample")
            candidate = self.candidates[self._ordinal[outcome.candidate_id]]
            cost = Path(candidate.source_path).parent / "cost.mlir"
            mapped = Path(candidate.source_path).parent / "mapped.mlir"
            cost_sha256 = file_sha256(cost)
            mapped_sha256 = file_sha256(mapped)
            if cost_sha256 is None or mapped_sha256 is None:
                raise RuntimeError(
                    "successful worker result lacks hashed cost/mapped artifacts"
                )
            success_updates = {
                "sample_id": outcome.candidate_id,
                "compiled_ii": int(outcome.sample["compiled_ii"]),
                "lower_bound": int(outcome.sample["baseline_lb"]),
                "cost_artifact_path": _relative_to_manifest(
                    self.manifest_path.parent, cost
                ),
                "cost_artifact_sha256": cost_sha256,
                "mapped_artifact_path": _relative_to_manifest(
                    self.manifest_path.parent, mapped
                ),
                "mapped_artifact_sha256": mapped_sha256,
            }
        self._failure_events[outcome.candidate_id] = [
            {
                **invocation_result.failure_record(),
                "candidate_id": outcome.candidate_id,
            }
            for invocation_result in outcome.invocations
            if not invocation_result.ok
        ]
        record["status"] = outcome.status
        record["stage"] = outcome.stage
        record["failure"] = outcome.failure
        if outcome.status == "success":
            record.update(success_updates)
            self.cached_samples[outcome.candidate_id] = outcome.sample
        else:
            _clear_transient_manifest_fields(record)
        record["invocation_failures"] = list(
            self._failure_events[outcome.candidate_id]
        )
        self._results[outcome.candidate_id] = outcome
        self._completion_count += 1

    def _finalize_failure_events(self) -> None:
        if self._failure_events_finalized:
            return
        for candidate in self.candidates:
            INVOCATION_FAILURES.extend(
                self._failure_events.get(candidate.candidate_id, ())
            )
        self._failure_events_finalized = True

    def flush(self) -> None:
        """Persist only stable declared/success/censored states atomically."""
        self.manifest["candidates"] = self._records
        self.manifest["summary"] = neura_motifs.manifest_summary(self._records)
        statuses = {str(record.get("status")) for record in self._records}
        if "declared" in statuses:
            self.manifest["status"] = (
                "interrupted" if self._interrupted else "incomplete"
            )
        elif "censored" in statuses:
            self.manifest["status"] = "partial"
        else:
            self.manifest["status"] = "complete"
        neura_motifs.atomic_write_json(self.manifest_path, self.manifest)

    def _harvest_done(
        self, pending: Dict[concurrent.futures.Future, str],
        done: Iterable[concurrent.futures.Future],
    ) -> None:
        ordered = sorted(
            done, key=lambda future: self._ordinal[pending[future]]
        )
        for future in ordered:
            candidate_id = pending.pop(future)
            if future.cancelled():
                continue
            try:
                outcome = future.result()
            except BaseException as error:  # preserve declared status on error
                self._errors.append(error)
                self.request_stop()
                continue
            if not isinstance(outcome, MotifCollectionResult):
                self._errors.append(TypeError("motif worker returned invalid result"))
                self.request_stop()
                continue
            self._apply_result(outcome)
            if self._completion_count % self.checkpoint_every == 0:
                self.flush()

    def _flush_cached(self) -> None:
        for candidate_id, sample in self.cached_samples.items():
            self._results.setdefault(
                candidate_id,
                MotifCollectionResult(
                    candidate_id, "success", "cached", sample=sample
                ),
            )

    def run(self) -> MotifCoordinatorResult:
        """Run declared candidates, or stop safely on signal/worker failure."""
        old_handler: object = None
        handler_installed = False
        try:
            try:
                old_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._signal_handler)
                handler_installed = True
            except (AttributeError, ValueError):
                # A coordinator can be tested from a non-main thread; the
                # explicit request_stop() API remains available there.
                handler_installed = False
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.jobs, thread_name_prefix="motif-worker"
            )
            pending: Dict[concurrent.futures.Future, str] = {}
            next_ordinal = 0
            try:
                while True:
                    # Observe every already-completed future before filling an
                    # available slot.  In particular, do not submit more work
                    # after a protocol exception has already occurred but was
                    # not part of the preceding FIRST_COMPLETED snapshot.
                    eager_done = tuple(
                        future for future in pending if future.done()
                    )
                    if eager_done:
                        self._harvest_done(pending, eager_done)
                    if self.stop_requested:
                        for future in tuple(pending):
                            future.cancel()
                        done, _ = concurrent.futures.wait(tuple(pending))
                        self._harvest_done(pending, done)
                        break
                    while not self.stop_requested and len(pending) < self.jobs:
                        while (
                            next_ordinal < len(self.candidates) and
                            self.candidates[next_ordinal].candidate_id in self.cached_samples
                        ):
                            next_ordinal += 1
                        if next_ordinal >= len(self.candidates):
                            break
                        candidate = self.candidates[next_ordinal]
                        record = self._record_for(candidate.candidate_id)
                        next_ordinal += 1
                        if record.get("status") != "declared":
                            continue  # censored/success are never retried
                        future = executor.submit(
                            collect_motif_candidate,
                            self.opt, candidate, self.timeout, run_invocation,
                        )
                        pending[future] = candidate.candidate_id
                    if not pending:
                        break
                    done, _ = concurrent.futures.wait(
                        tuple(pending),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    self._harvest_done(pending, done)
                    if self.stop_requested:
                        for future in tuple(pending):
                            future.cancel()
                        # At most ``jobs`` futures are in flight.  Drain all
                        # running ones, harvesting only normal outcomes.
                        done, _ = concurrent.futures.wait(tuple(pending))
                        self._harvest_done(pending, done)
                        break
                if self._errors:
                    # Unresolved declarations intentionally remain declared;
                    # flush before surfacing the worker exception.
                    self.flush()
                    self._finalize_failure_events()
                    raise RuntimeError("motif worker failed") from self._errors[0]
                self._flush_cached()
                self.flush()
            except KeyboardInterrupt:
                self._interrupted = True
                self.request_stop()
                for future in tuple(pending):
                    future.cancel()
                done, _ = concurrent.futures.wait(tuple(pending))
                self._harvest_done(pending, done)
                self.flush()
                self._finalize_failure_events()
            except BaseException:
                # The stable manifest is useful for resume even when caller
                # supplied an invalid worker or an unexpected exception.
                self.flush()
                self._finalize_failure_events()
                raise
            finally:
                # ``cancel_futures`` is only available in newer Python
                # versions; all not-yet-started futures are cancelled above.
                executor.shutdown(wait=True)
        finally:
            if handler_installed:
                signal.signal(signal.SIGINT, old_handler)
        self._finalize_failure_events()
        self._flush_cached()
        samples = [
            self.cached_samples[candidate.candidate_id]
            for candidate in self.candidates
            if candidate.candidate_id in self.cached_samples
        ]
        results = [
            self._results[candidate.candidate_id]
            for candidate in self.candidates
            if candidate.candidate_id in self._results
        ]
        return MotifCoordinatorResult(samples, results, self._interrupted)


def collect_c_sample(opt: Path, sample_dir: Path, spec: CSpec,
                     architecture: Path, cxx: str, llvm_extract: Path,
                     mlir_translate: Path, timeout: int) -> Optional[Sample]:
    """Compile a generated C loop through the normal frontend and mapper."""
    source = sample_dir / "input.cpp"
    full_ir = sample_dir / "full.ll"
    kernel_ir = sample_dir / "kernel.ll"
    imported = sample_dir / "imported.mlir"
    lowered = sample_dir / "lowered.mlir"
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    write_c_loop(source, spec)
    if not invoke(
        (cxx, "-S", "-emit-llvm", "-O3", "-fno-vectorize",
         "-fno-unroll-loops", "-std=c++17", "-o", str(full_ir), str(source)),
        timeout,
    ):
        return None
    if not invoke(
        (str(llvm_extract), "--rfunc=.*kernel.*", str(full_ir), "-o",
         str(kernel_ir)), timeout,
    ):
        return None
    if not invoke(
        (str(mlir_translate), "--import-llvm", str(kernel_ir), "-o",
         str(imported)), timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(imported), "--assign-accelerator",
         "--lower-llvm-to-neura", "--promote-input-arg-to-const",
         "--fold-constant", "--canonicalize-return",
         "--canonicalize-live-in", "--leverage-predicated-value",
         "--transform-ctrl-to-data-flow", "--fold-constant",
         "--insert-data-mov", "-o", str(lowered)), timeout,
    ):
        return None
    options = f"x-tiles={spec.columns} y-tiles={spec.rows}"
    if not invoke(
        (str(opt), str(lowered), f"--architecture-spec={architecture}",
         f"--analyze-rec-res-mii={options}", "-o", str(cost)), timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(lowered), f"--architecture-spec={architecture}",
         f"--map-to-accelerator=mapping-strategy=heuristic {options}",
         "-o", str(mapped)), timeout,
    ):
        return None
    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = parse_cost_features(cost_text)
    if values is None:
        return None
    compiled_ii = parse_checked_mapper_label(mapped_text, values)
    if compiled_ii is None:
        return None
    result: Sample = dict(values)
    result["compiled_ii"] = int(compiled_ii)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features_from_neura(lowered.read_text(), spec.rows,
                                            spec.columns))
    add_prediction_features(result)
    result.update(asdict(spec))
    result["family"] = "synthetic-c"
    attach_sample_provenance(
        result, source, architecture, f"{spec.rows}x{spec.columns}",
        "synthetic-c-loop-template", dfg_source=lowered,
    )
    return result


def semantic_features_from_neura(text: str) -> Dict[str, float]:
    """Features invariant under a tile-shape or valid-tile-mask change."""
    ignored = {"data_mov", "reserve", "ctrl_mov", "yield"}
    op_names = re.findall(
        r'(?m)^\s*(?:%[A-Za-z0-9_]+\s*=\s*)?"?neura\.([a-z_]+)', text
    )
    materialized = [name for name in op_names if name not in ignored]
    moves = sum(name == "data_mov" for name in op_names)
    fu_classes = {
        "add": {"add", "sub"},
        "mul": {"mul"},
        "div": {"div", "rem"},
        "fadd": {"fadd", "fsub"},
        "fmul": {"fmul"},
        "fdiv": {"fdiv"},
        "logic": {"or", "and", "xor", "not"},
        "cmp": {"icmp", "fcmp"},
        "sel": {"sel"},
        "type_conv": {"cast", "sext", "zext"},
        "vfmul": {"vfmul"},
        "fadd_fadd": {"fadd_fadd"},
        "fmul_fadd": {"fmul_fadd"},
        "grant": {"grant_predicate", "grant_once", "grant_always"},
        "loop_control": {"loop_control"},
        "phi": {"phi", "phi_start"},
        "constant": {"constant"},
        "alloca": {"alloca"},
        "shift": {"shl"},
    }
    fu_class_peak_ops = max((
        sum(name in kinds for name in materialized)
        for kinds in fu_classes.values()
    ), default=0)

    # SSA use counts give a stable fanout feature without needing to reimplement
    # MLIR parsing in the experiment harness.  Count only values defined by a
    # result-producing operation, then discount that definition occurrence.
    definitions = re.findall(r"(?m)^\s*(%[A-Za-z0-9_]+)\s*=", text)
    token_counts: Dict[str, int] = {}
    for token in re.findall(r"%[A-Za-z0-9_]+", text):
        token_counts[token] = token_counts.get(token, 0) + 1
    max_fanout = max((token_counts[value] - 1 for value in definitions), default=0)

    depths: Dict[str, int] = {}
    defining_kind: Dict[str, str] = {}
    defining_operands: Dict[str, List[str]] = {}
    definition_lines: Dict[str, int] = {}
    pointer_results: Dict[str, bool] = {}
    use_lines: Dict[str, List[int]] = {}
    indirect_geps = 0
    pointer_loads = 0
    lines = text.splitlines()
    for line_number, line in enumerate(lines):
        for token in re.findall(r"%[A-Za-z0-9_]+", line):
            use_lines.setdefault(token, []).append(line_number)
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if not match:
            continue
        value, expression = match.groups()
        kind_match = re.search(r'"?neura\.([a-z_]+)', expression)
        kind = kind_match.group(1) if kind_match else ""
        operands = re.findall(r"%[A-Za-z0-9_]+", expression)
        depths[value] = 1 + max((depths.get(operand, 0) for operand in operands),
                                default=0)
        defining_kind[value] = kind
        defining_operands[value] = operands
        definition_lines[value] = line_number
        pointer_results[value] = "-> !neura.data<!llvm.ptr" in expression

        def comes_from_pointer_load(operand: str) -> bool:
            # InsertDataMov wraps nearly every data dependency.  Follow those
            # wrappers so a load -> data_mov -> gep chain remains visible.
            seen: Set[str] = set()
            while operand not in seen:
                seen.add(operand)
                kind = defining_kind.get(operand)
                if kind == "load":
                    return True
                if kind != "data_mov":
                    return False
                move_operands = defining_operands.get(operand, [])
                if len(move_operands) != 1:
                    return False
                operand = move_operands[0]
            return False

        if kind == "gep" and any(comes_from_pointer_load(operand)
                                 for operand in operands):
            indirect_geps += 1
        if kind == "load" and "-> !neura.data<!llvm.ptr" in expression:
            pointer_loads += 1

    # Collapse inserted data_mov operations before measuring graph structure.
    # Raw operation counts otherwise make the same dependency look artificially
    # deeper merely because lowering materialized transport operations.
    transparent = {"data_mov", "reserve"}

    def meaningful_roots(value: str, seen: Optional[Set[str]] = None) -> List[str]:
        if value not in defining_kind:
            return []
        if defining_kind[value] not in transparent:
            return [value]
        visited = set() if seen is None else seen
        if value in visited:
            return []
        visited.add(value)
        roots: List[str] = []
        for operand in defining_operands.get(value, []):
            roots.extend(meaningful_roots(operand, visited))
        return roots

    semantic_depths: Dict[str, int] = {}
    memory_depths: Dict[str, int] = {}
    pointer_depths: Dict[str, int] = {}
    control_depths: Dict[str, int] = {}
    depth_widths: Dict[int, int] = {}
    semantic_fanout: Dict[str, int] = {}
    semantic_edges: List[Tuple[int, int]] = []
    multi_input_nodes = 0
    memory_kinds = {
        "gep", "load", "store", "memset", "load_indexed", "store_indexed"
    }
    control_kinds = {
        "grant_predicate", "phi", "phi_start", "not", "icmp", "fcmp"
    }
    for value in definition_lines:
        kind = defining_kind[value]
        parents: List[str] = []
        for operand in defining_operands.get(value, []):
            parents.extend(meaningful_roots(operand))
        parents = list(dict.fromkeys(parents))
        semantic_depths[value] = max(
            (semantic_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind not in transparent)
        memory_depths[value] = max(
            (memory_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind in memory_kinds)
        pointer_depths[value] = max(
            (pointer_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind == "gep" or (kind == "load" and pointer_results[value]))
        control_depths[value] = max(
            (control_depths.get(parent, 0) for parent in parents), default=0
        ) + int(kind in control_kinds)
        if kind in transparent:
            continue
        depth = semantic_depths[value]
        depth_widths[depth] = depth_widths.get(depth, 0) + 1
        multi_input_nodes += int(len(parents) > 1)
        for parent in parents:
            semantic_fanout[parent] = semantic_fanout.get(parent, 0) + 1
            semantic_edges.append(
                (definition_lines[parent], definition_lines[value]))

    last_definition = max(definition_lines.values(), default=0)
    semantic_cutwidth = max((
        sum(start <= position < end for start, end in semantic_edges)
        for position in range(last_definition + 1)
    ), default=0)
    live_value_peak = max((
        sum(
            start <= position < max(use_lines.get(value, [start]))
            for value, start in definition_lines.items()
        )
        for position in range(last_definition + 1)
    ), default=0)

    return {
        "nodes": len(materialized),
        "moves": moves,
        "ctrl_moves": sum(name == "ctrl_mov" for name in op_names),
        "reserves": sum(name == "reserve" for name in op_names),
        "phis": sum(name in {"phi", "phi_start"} for name in op_names),
        "predicates": sum(name == "grant_predicate" for name in op_names),
        "memory_ops": sum(name in {
                              "load", "store", "memset",
                              "load_indexed", "store_indexed",
                          }
                          for name in op_names),
        "gep_ops": sum(name == "gep" for name in op_names),
        "indirect_geps": indirect_geps,
        "pointer_loads": pointer_loads,
        "div_ops": sum(name in {"div", "rem"} for name in op_names),
        "arithmetic_ops": sum(name in {"add", "sub", "mul", "fmul_fadd", "shl"}
                              for name in op_names),
        "comparisons": sum(name in {"icmp", "fcmp"} for name in op_names),
        "sources": sum(name in {"constant", "grant_once", "grant_always"}
                       for name in materialized),
        "depth": max(depths.values(), default=0),
        "max_fanout": max_fanout,
        "semantic_edges": len(semantic_edges),
        "semantic_depth": max(semantic_depths.values(), default=0),
        "semantic_width": max(depth_widths.values(), default=0),
        "semantic_max_fanout": max(semantic_fanout.values(), default=0),
        "semantic_branch_nodes": sum(
            fanout > 1 for fanout in semantic_fanout.values()),
        "semantic_cutwidth": semantic_cutwidth,
        "live_value_peak": live_value_peak,
        "pointer_path": max(pointer_depths.values(), default=0),
        "memory_path": max(memory_depths.values(), default=0),
        "control_path": max(control_depths.values(), default=0),
        "multi_input_nodes": multi_input_nodes,
        "fu_class_peak_ops": fu_class_peak_ops,
    }


def graph_features_from_neura(
    text: str, rows: int, columns: int,
    valid_tiles: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Extract mapper-visible structural features from already-lowered IR."""
    active_tiles = valid_tiles or {
        (x, y) for y in range(rows) for x in range(columns)
    }
    adjacent_pairs = sum(
        ((x + 1, y) in active_tiles) + ((x, y + 1) in active_tiles)
        for x, y in active_tiles
    )
    result = semantic_features_from_neura(text)
    tile_count = len(active_tiles)
    memory_tiles = sum(x == 0 or y == 0 for x, y in active_tiles)
    # For the supported rectangular prefix targets, the minimum directed
    # bisection contains two directed links per tile on the shorter boundary.
    bisection_links = 2 * min(rows, columns)
    total_registers = tile_count * neura_motifs.PINNED_REGISTERS_PER_TILE
    result.update({
        "tiles": tile_count,
        "links": 2 * adjacent_pairs,
        "rows": rows,
        "columns": columns,
        "memory_tiles": memory_tiles,
        "bisection_links": bisection_links,
        "total_registers": total_registers,
        # The production YAML has heterogeneous memory tiles, but not the
        # source/compute partition used by the synthetic split-domain mode.
        "split_domain": 0,
    })
    result.update({
        "compute_fu_peak_pressure": result["fu_class_peak_ops"] / tile_count,
        "memory_fu_pressure": result["memory_ops"] / memory_tiles,
        "routing_edge_pressure": result["semantic_edges"] / result["links"],
        "routing_cut_pressure": (
            result["semantic_cutwidth"] / bisection_links
        ),
        "register_pressure": result["live_value_peak"] / total_registers,
    })
    return result


def collect_real_fixture(opt: Path, sample_dir: Path, name: str, source: Path,
                         architecture: Path, rows: int, columns: int,
                         timeout: int,
                         valid_tiles: Optional[Set[Tuple[int, int]]] = None,
                         suffix: str = "") -> Optional[Sample]:
    """Label a pre-lowered real DFG at a selected architectural shape."""
    cost = sample_dir / "cost.mlir"
    mapped = sample_dir / "mapped.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if valid_tiles is not None:
        valid_tile_text = ",".join(
            f"{x}_{y}" for x, y in sorted(valid_tiles, key=lambda tile: (tile[1], tile[0]))
        )
        options += f" valid-tiles={valid_tile_text}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--analyze-rec-res-mii={options}", "-o", str(cost)),
        timeout,
    ):
        return None
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--map-to-accelerator=mapping-strategy=heuristic {options}",
         "-o", str(mapped)),
        timeout,
    ):
        return None

    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    values = parse_cost_features(cost_text)
    if values is None:
        return None
    compiled_ii = parse_checked_mapper_label(mapped_text, values)
    if compiled_ii is None:
        return None
    result: Sample = dict(values)
    result["compiled_ii"] = int(compiled_ii)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features_from_neura(source.read_text(), rows, columns,
                                            valid_tiles))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}{suffix}"
    result["family"] = name
    mask = "all" if valid_tiles is None else ",".join(
        f"{x}_{y}" for x, y in sorted(valid_tiles)
    )
    attach_sample_provenance(
        result, source, architecture, f"{rows}x{columns}:tiles={mask}", name
    )
    return result


def collect_completed_real_fixture(
    opt: Path, sample_dir: Path, name: str, source: Path, mapped: Path,
    architecture: Path, rows: int, columns: int, timeout: int,
) -> Optional[Sample]:
    """Reuse a completed heuristic mapping artifact as a real label.

    This is useful for expensive kernels: the mapping must have completed in a
    previous invocation of *this* heuristic mapper.  We still rerun the cheap
    Rec/Res analysis pass, and reject artifacts without a heuristic `compiled_ii`.
    """
    cost = sample_dir / "cost.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--analyze-rec-res-mii={options}", "-o", str(cost)),
        timeout,
    ):
        return None
    cost_text = cost.read_text()
    mapped_text = mapped.read_text()
    if 'mapping_strategy = "heuristic"' not in mapped_text:
        return None
    values = parse_cost_features(cost_text)
    if values is None:
        return None
    compiled_ii = parse_checked_mapper_label(mapped_text, values)
    if compiled_ii is None:
        return None
    result: Sample = dict(values)
    result["compiled_ii"] = int(compiled_ii)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features_from_neura(source.read_text(), rows, columns))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}"
    result["family"] = name
    result["training_stratum"] = "real"
    result["leakage_lineage_id"] = name
    attach_sample_provenance(
        result, source, architecture, f"{rows}x{columns}:tiles=all", name
    )
    result["mapped_artifact_path"] = str(mapped.resolve())
    result["mapped_artifact_sha256"] = file_sha256(mapped)
    return result


def collect_prediction_fixture(
    opt: Path, sample_dir: Path, name: str, source: Path,
    architecture: Path, rows: int, columns: int, timeout: int,
) -> Optional[Sample]:
    """Extract pre-mapping prediction features without invoking the mapper."""
    cost = sample_dir / "cost.mlir"
    options = f"x-tiles={columns} y-tiles={rows}"
    if not invoke(
        (str(opt), str(source), f"--architecture-spec={architecture}",
         f"--analyze-rec-res-mii={options}", "-o", str(cost)), timeout,
    ):
        return None
    cost_text = cost.read_text()
    values = parse_cost_features(cost_text)
    if values is None:
        return None
    result: Sample = dict(values)
    attach_rec_res_artifact(result, cost)
    result.update(graph_features_from_neura(source.read_text(), rows, columns))
    add_prediction_features(result)
    result["index"] = f"{name}-{rows}x{columns}"
    result["family"] = name
    result["training_stratum"] = "real"
    result["leakage_lineage_id"] = name
    attach_sample_provenance(
        result, source, architecture, f"{rows}x{columns}:tiles=all", name
    )
    return result


def random_connected_tile_masks(rng: random.Random, count: int,
                                min_tiles: int) -> List[Set[Tuple[int, int]]]:
    """Generate unique connected 4x4 shape masks for real-DFG labeling."""
    masks: List[Set[Tuple[int, int]]] = []
    seen: Set[Tuple[Tuple[int, int], ...]] = set()
    attempts = 0
    while len(masks) < count and attempts < count * 50:
        attempts += 1
        target_size = rng.randint(min_tiles, 15)
        mask = {(rng.randrange(4), rng.randrange(4))}
        while len(mask) < target_size:
            frontier = []
            for x, y in mask:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < 4 and 0 <= neighbor[1] < 4 and neighbor not in mask:
                        frontier.append(neighbor)
            if not frontier:
                break
            mask.add(rng.choice(frontier))
        key = tuple(sorted(mask))
        if len(mask) == target_size and key not in seen:
            seen.add(key)
            masks.append(mask)
    return masks


def to_core_sample(row: Sample) -> CoreSample:
    """Translate a Neura flat row into the compiler-agnostic core schema."""
    lower_bound = float(row["baseline_lb"])
    return CoreSample(
        sample_id=str(row.get("index", "prediction")),
        group=str(row.get("family", "prediction")),
        lower_bound=lower_bound,
        compiled_ii=float(row.get("compiled_ii", lower_bound)),
        features={
            name: float(row[name]) for name in MODEL_FEATURE_NAMES if name in row
        },
        metadata={
            name: row[name]
            for name in (
                "candidate_id", "architecture_id", "architecture_variant",
                "mapper_id", "mapper_revision", "mapper_config",
                "source_family", "source_kind", "training_weight_group",
                "source_sha256", "dfg_source_sha256",
                "lineage", "declared_leakage_lineage_id",
                "generator_family", "generator_version", "motif",
                "generator_type", "base_id", "base_seed", "root_seed",
                "base_index", "operation_count", "target_config_id",
                "valid_tiles",
                "canonical_dfg_sha256",
                "leakage_lineage_id", "base_dfg_id", "ranking_query_id",
                "training_stratum", "rec_mii", "res_mii",
                "lower_bound_source",
            )
            if name in row
        },
    )


def fit_ridge(train: Sequence[Sample], ridge: float,
              residual_dead_zone: float = 0.0) -> Dict[str, object]:
    """Compatibility wrapper around the single core Ridge implementation."""
    return core_fit_ridge(
        [to_core_sample(row) for row in train], MODEL_FEATURE_NAMES,
        ridge, residual_dead_zone,
    )


def predict_ridge(model: Dict[str, object], row: Sample) -> float:
    return core_predict_ridge(model, to_core_sample(row))


def predict_unlabelled_candidate(
    model: Mapping[str, object], row: Sample,
) -> Dict[str, object]:
    """Evaluate an unlabeled flat Neura feature record without a fake label."""
    lower_bound, _ = resolve_rec_res_lower_bound(row)
    model_features = {
        feature_name: float(row[feature_name])
        for feature_name in model["feature_names"]
    }
    raw_residual = core_raw_residual_from_features(model, model_features)
    predicted_residual = core_constrained_predicted_residual(
        model, raw_residual
    )
    prediction = core_predict_compiled_ii(
        model, float(lower_bound), model_features,
        rec_mii=float(row["rec_mii"]), res_mii=float(row["res_mii"]),
    )
    support = model.get("training_feature_support")
    outside_central: List[str] = []
    outside_range: List[str] = []
    if isinstance(support, Mapping):
        for name, value in model_features.items():
            record = support.get(name)
            if not isinstance(record, Mapping):
                continue
            if value < float(record["minimum"]) or value > float(record["maximum"]):
                outside_range.append(name)
            if value < float(record["p01"]) or value > float(record["p99"]):
                outside_central.append(name)
    return {
        "model_features": model_features,
        "raw_predicted_residual": raw_residual,
        "nonnegative_predicted_residual": max(0.0, raw_residual),
        "predicted_residual": predicted_residual,
        "predicted_compiled_ii": prediction,
        "feature_support": {
            "outside_central_98_percent": outside_central,
            "outside_observed_range": outside_range,
        },
    }


def shape_selection_summary(
    predictions: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Build label-free area/II Pareto frontiers for each input DFG.

    The model ranks candidates; it does not claim that a predicted shape is a
    legal final mapping.  Callers should try the returned verification order
    with Neura's mapper and stop according to their area/throughput objective.
    """
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for prediction in predictions:
        task = prediction.get("task")
        if isinstance(task, str) and task:
            grouped.setdefault(task, []).append(prediction)
    summaries: List[Dict[str, object]] = []
    for task, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["predicted_compiled_ii"]),
                int(row["tile_count"]),
                str(row["shape"]),
            ),
        )
        frontier = []
        for candidate in candidates:
            area = int(candidate["tile_count"])
            ii = float(candidate["predicted_compiled_ii"])
            dominated = any(
                int(other["tile_count"]) <= area and
                float(other["predicted_compiled_ii"]) <= ii and
                (
                    int(other["tile_count"]) < area or
                    float(other["predicted_compiled_ii"]) < ii
                )
                for other in candidates if other is not candidate
            )
            if not dominated:
                frontier.append(candidate)
        frontier.sort(key=lambda row: (
            int(row["tile_count"]), float(row["predicted_compiled_ii"]),
            str(row["shape"]),
        ))
        summaries.append({
            "task": task,
            "objective": "minimize_predicted_ii_and_active_tile_count",
            "pareto_candidate_ids": [row["sample"] for row in frontier],
            "pareto_shapes": [row["shape"] for row in frontier],
            "throughput_first_candidate_id": ordered[0]["sample"],
            "throughput_first_shape": ordered[0]["shape"],
            "mapper_verification_order": [row["sample"] for row in ordered],
            "selection_status": "prediction_ranking_requires_mapper_verification",
        })
    return summaries


def residual_sse(rows: Sequence[Sample]) -> float:
    if not rows:
        return 0.0
    residuals = np.asarray(
        [row["compiled_ii"] - row["baseline_lb"] for row in rows], dtype=float
    )
    return float(((residuals - residuals.mean()) ** 2).sum())


def fit_residual_tree(train: Sequence[Sample], max_depth: int,
                      min_samples: int) -> Dict[str, object]:
    """Fit a deterministic, dependency-free shallow regression tree.

    The tree predicts the residual above max(RecMII, ResMII). It is a
    reporting experiment only; no compiler code reads this model.
    """
    if not train:
        raise ValueError("cannot fit an empty training set")

    def build(rows: Sequence[Sample], depth: int) -> Dict[str, object]:
        target = sum(row["compiled_ii"] - row["baseline_lb"] for row in rows)
        leaf = max(0.0, target / len(rows))
        parent_sse = residual_sse(rows)
        if depth >= max_depth or len(rows) < 2 * min_samples or parent_sse == 0.0:
            return {"value": leaf, "count": len(rows)}

        best: Optional[Tuple[float, int, float, List[Sample], List[Sample]]] = None
        for feature_index, name in enumerate(MODEL_FEATURE_NAMES):
            values = sorted({float(row[name]) for row in rows})
            for low, high in zip(values, values[1:]):
                threshold = (low + high) / 2.0
                left = [row for row in rows if float(row[name]) <= threshold]
                right = [row for row in rows if float(row[name]) > threshold]
                if len(left) < min_samples or len(right) < min_samples:
                    continue
                child_sse = residual_sse(left) + residual_sse(right)
                candidate = (child_sse, feature_index, threshold, left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None or best[0] >= parent_sse:
            return {"value": leaf, "count": len(rows)}
        _, feature_index, threshold, left, right = best
        return {
            "feature": MODEL_FEATURE_NAMES[feature_index],
            "threshold": threshold,
            "count": len(rows),
            "left": build(left, depth + 1),
            "right": build(right, depth + 1),
        }

    return {
        "feature_names": list(MODEL_FEATURE_NAMES),
        "max_depth": max_depth,
        "min_samples": min_samples,
        "tree": build(list(train), 0),
    }


def predict_residual_tree(model: Dict[str, object], row: Sample) -> float:
    node = model["tree"]
    while "feature" in node:
        node = node["left"] if float(row[node["feature"]]) <= node["threshold"] else node["right"]
    return row["baseline_lb"] + max(0.0, float(node["value"]))


def prediction_rows(test: Sequence[Sample], predictor: Callable[[Sample], float]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for row in test:
        prediction: Dict[str, object] = {
            "sample": row["index"],
            "family": row["family"],
            "baseline": row["baseline_lb"],
            "prediction": predictor(row),
            "compiled_ii": row["compiled_ii"],
        }
        for name in (
            "candidate_id", "architecture_id", "architecture_variant",
            "source_family", "source_kind", "leakage_lineage_id",
            "base_dfg_id", "ranking_query_id", "training_stratum",
            "generator_family", "generator_version", "motif",
        ):
            if name in row:
                prediction[name] = row[name]
        result.append(prediction)
    return result


def is_synthetic_row(row: Dict[str, object]) -> bool:
    if str(row.get("source_kind", "")).lower() in {"generated", "synthetic"}:
        return True
    identity = row.get(
        "generator_family",
        row.get("source_family", row.get("family", row.get("group", ""))),
    )
    return str(identity).startswith(("synthetic", "generated"))


def random_row_holdout(samples: Sequence[Sample], seed: int, ridge: float,
                       tree_depth: int, tree_min_samples: int) -> Dict[str, object]:
    """Diagnostic only: rows from one kernel may appear on both sides."""
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    split = max(1, int(len(ordered) * 0.8))
    train, test = ordered[:split], ordered[split:]
    ridge_model = fit_ridge(train, ridge)
    tree_model = fit_residual_tree(train, tree_depth, tree_min_samples)
    ridge_rows = prediction_rows(test, lambda row: predict_ridge(ridge_model, row))
    tree_rows = prediction_rows(
        test, lambda row: predict_residual_tree(tree_model, row))
    return {
        "train_size": len(train),
        "ridge_model": ridge_model,
        "tree_model": tree_model,
        "ridge_rows": ridge_rows,
        "tree_rows": tree_rows,
        "baseline_mae": mean_absolute_error(ridge_rows, "baseline"),
        "ridge_mae": mean_absolute_error(ridge_rows, "prediction"),
        "tree_mae": mean_absolute_error(tree_rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(ridge_rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(ridge_rows, "prediction"),
        "tree_macro_family_mae": macro_family_mae(tree_rows, "prediction"),
    }


def leave_one_family_out(samples: Sequence[Sample], ridge: float,
                         tree_depth: int, tree_min_samples: int) -> Dict[str, object]:
    """Evaluate generalization to an unseen kernel family without leakage."""
    families = sorted({str(row["family"]) for row in samples})
    ridge_rows: List[Dict[str, object]] = []
    tree_rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for family in families:
        test = [row for row in samples if row["family"] == family]
        train = [row for row in samples if row["family"] != family]
        if len(train) < max(2, 2 * tree_min_samples):
            skipped.append(family)
            continue
        ridge_model = fit_ridge(train, ridge)
        tree_model = fit_residual_tree(train, tree_depth, tree_min_samples)
        ridge_rows.extend(
            prediction_rows(test, lambda row: predict_ridge(ridge_model, row)))
        tree_rows.extend(
            prediction_rows(
                test, lambda row: predict_residual_tree(tree_model, row)))
    if not ridge_rows:
        raise ValueError("not enough independent kernel families for family holdout")
    result: Dict[str, object] = {
        "families": families,
        "skipped_families": skipped,
        "ridge_rows": ridge_rows,
        "tree_rows": tree_rows,
        "baseline_mae": mean_absolute_error(ridge_rows, "baseline"),
        "ridge_mae": mean_absolute_error(ridge_rows, "prediction"),
        "tree_mae": mean_absolute_error(tree_rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(ridge_rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(ridge_rows, "prediction"),
        "tree_macro_family_mae": macro_family_mae(tree_rows, "prediction"),
    }
    real_ridge_rows = [
        row for row in ridge_rows
        if not is_synthetic_row(row)
    ]
    real_tree_rows = [
        row for row in tree_rows
        if not is_synthetic_row(row)
    ]
    if real_ridge_rows:
        result.update({
            "real_baseline_mae": mean_absolute_error(real_ridge_rows, "baseline"),
            "real_ridge_mae": mean_absolute_error(real_ridge_rows, "prediction"),
            "real_tree_mae": mean_absolute_error(real_tree_rows, "prediction"),
            "real_baseline_macro_family_mae": macro_family_mae(
                real_ridge_rows, "baseline"),
            "real_ridge_macro_family_mae": macro_family_mae(
                real_ridge_rows, "prediction"),
            "real_tree_macro_family_mae": macro_family_mae(
                real_tree_rows, "prediction"),
        })
    return result


def add_real_holdout_metrics(result: Dict[str, object],
                             ridge_rows: Sequence[Dict[str, object]],
                             tree_rows: Optional[Sequence[Dict[str, object]]] = None) -> None:
    """Add micro/macro metrics after excluding generated synthetic families."""
    real_ridge_rows = [
        row for row in ridge_rows
        if not is_synthetic_row(row)
    ]
    if not real_ridge_rows:
        return
    result.update({
        "real_baseline_mae": mean_absolute_error(real_ridge_rows, "baseline"),
        "real_ridge_mae": mean_absolute_error(real_ridge_rows, "prediction"),
        "real_baseline_macro_family_mae": macro_family_mae(
            real_ridge_rows, "baseline"),
        "real_ridge_macro_family_mae": macro_family_mae(
            real_ridge_rows, "prediction"),
    })
    if tree_rows is not None:
        real_tree_rows = [
            row for row in tree_rows
            if not is_synthetic_row(row)
        ]
        result.update({
            "real_tree_mae": mean_absolute_error(real_tree_rows, "prediction"),
            "real_tree_macro_family_mae": macro_family_mae(
                real_tree_rows, "prediction"),
        })


def select_ridge_hyperparameters(
    train: Sequence[Sample], ridge_candidates: Sequence[float],
    dead_zone_candidates: Sequence[float],
) -> Tuple[float, float]:
    """Choose Ridge calibration through the compiler-agnostic core."""
    families = sorted({str(row["family"]) for row in train})
    if len(families) < 2:
        return (ridge_candidates[len(ridge_candidates) // 2],
                dead_zone_candidates[len(dead_zone_candidates) // 2])
    return core_select_ridge_hyperparameters(
        [to_core_sample(row) for row in train], MODEL_FEATURE_NAMES,
        ridge_candidates, dead_zone_candidates,
    )


def nested_ridge_family_holdout(samples: Sequence[Sample],
                                ridge_candidates: Sequence[float],
                                dead_zone_candidates: Sequence[float],
                                ) -> Dict[str, object]:
    """Outer family holdout with calibration chosen only from outer training."""
    core_result = core_nested_group_holdout(
        [to_core_sample(row) for row in samples], MODEL_FEATURE_NAMES,
        ridge_candidates, dead_zone_candidates,
    )
    rows: List[Dict[str, object]] = []
    for row in core_result["rows"]:
        translated: Dict[str, object] = {
            "sample": row["sample_id"],
            "family": row["group"],
            "baseline": row["lower_bound"],
            "prediction": row["prediction"],
            "compiled_ii": row["compiled_ii"],
            "raw_predicted_residual": row["raw_predicted_residual"],
        }
        for name in (
            "candidate_id", "architecture_id", "architecture_variant",
            "source_family", "source_kind", "leakage_lineage_id",
            "base_dfg_id", "ranking_query_id", "training_stratum",
            "generator_family", "generator_version", "motif",
        ):
            if name in row:
                translated[name] = row[name]
        rows.append(translated)
    result: Dict[str, object] = {
        "rows": rows,
        "outer_split_protocol": core_result["outer_split_protocol"],
        "outer_fold_count": core_result["outer_fold_count"],
        "outer_held_out_groups_by_fold": (
            core_result["outer_held_out_groups_by_fold"]
        ),
        "evaluation_input_row_count": core_result["evaluation_input_row_count"],
        "evaluation_distinct_observation_count": (
            core_result["evaluation_distinct_observation_count"]
        ),
        "evaluation_duplicate_rows_collapsed": (
            core_result["evaluation_duplicate_rows_collapsed"]
        ),
        "chosen_hyperparameters_by_held_out_family": (
            core_result["chosen_hyperparameters_by_held_out_group"]
        ),
        "baseline_mae": mean_absolute_error(rows, "baseline"),
        "ridge_mae": mean_absolute_error(rows, "prediction"),
        "baseline_macro_family_mae": macro_family_mae(rows, "baseline"),
        "ridge_macro_family_mae": macro_family_mae(rows, "prediction"),
        "baseline_group_ranking": core_result["lower_bound_group_ranking"],
        "ridge_group_ranking": core_result["model_group_ranking"],
        "raw_residual_metrics": core_result["raw_residual_metrics"],
    }
    add_prediction_quality_metrics(result, rows, "baseline")
    add_prediction_quality_metrics(result, rows, "ridge", "prediction")
    add_real_holdout_metrics(result, rows)
    return result


def generated_nested_improvement_gate(
    nested_holdout: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    """Require held-out Ridge MAE to strictly improve on the Rec/Res floor.

    This gate consumes only generated-lineage validation results.  It is
    deliberately based on macro MAE so every base DFG has equal influence and
    never inspects MachSuite labels or the diagnostic random-row split.
    """
    baseline = None
    ridge = None
    if isinstance(nested_holdout, Mapping):
        baseline = nested_holdout.get("baseline_macro_family_mae")
        ridge = nested_holdout.get("ridge_macro_family_mae")
    valid = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (baseline, ridge)
    )
    baseline_value = float(baseline) if valid else None
    ridge_value = float(ridge) if valid else None
    improvement = (
        baseline_value - ridge_value
        if baseline_value is not None and ridge_value is not None else None
    )
    relative = (
        improvement / baseline_value
        if improvement is not None and baseline_value is not None and
        baseline_value > 0.0 else None
    )
    return {
        "status": "ok" if valid else "unavailable",
        "source": "generated_nested_base_lineage_holdout",
        "metric": "macro_mean_absolute_error",
        "rule": "ridge_macro_mae < rec_res_lower_bound_macro_mae",
        "baseline_macro_mae": baseline_value,
        "ridge_macro_mae": ridge_value,
        "absolute_improvement": improvement,
        "relative_improvement": relative,
        "passed": bool(valid and ridge_value < baseline_value),
        "machsuite_labels_used": False,
    }


def nested_ridge_metadata_holdout(
    samples: Sequence[Sample], metadata_key: str,
    ridge_candidates: Sequence[float], dead_zone_candidates: Sequence[float],
) -> Dict[str, object]:
    """Hold out a metadata-defined domain while weighting source lineages."""
    missing = [str(row["index"]) for row in samples if not row.get(metadata_key)]
    if missing:
        return {
            "status": "unavailable_missing_metadata",
            "holdout_key": metadata_key,
            "missing_sample_count": len(missing),
            "sample_count": len(samples),
        }
    groups = sorted({str(row[metadata_key]) for row in samples})
    if len(groups) < 3:
        return {
            "status": "unavailable_too_few_groups",
            "holdout_key": metadata_key,
            "group_count": len(groups),
            "minimum_group_count": 3,
        }
    regrouped: List[Sample] = []
    for row in samples:
        copy = dict(row)
        copy["training_weight_group"] = str(row["family"])
        copy["family"] = str(row[metadata_key])
        regrouped.append(copy)
    return {
        "status": "ok",
        "holdout_key": metadata_key,
        "group_count": len(groups),
        "training_weight_group": "source_lineage",
        "evaluation": nested_ridge_family_holdout(
            regrouped, ridge_candidates, dead_zone_candidates
        ),
    }


def mean_absolute_error(rows: Iterable[Dict[str, object]], prediction: str) -> float:
    rows = list(rows)
    return sum(
        abs(float(row[prediction]) - float(row["compiled_ii"])) for row in rows
    ) / len(rows)


def add_prediction_quality_metrics(
    result: Dict[str, object], rows: Sequence[Dict[str, object]], prefix: str,
    prediction: Optional[str] = None,
) -> None:
    """Record integer and tail metrics in addition to continuous MAE."""
    key = prediction or prefix
    errors = [float(row[key]) - float(row["compiled_ii"]) for row in rows]
    rounded_errors = [
        round(float(row[key])) - int(row["compiled_ii"]) for row in rows
    ]
    result.update({
        f"{prefix}_mean_signed_error": sum(errors) / len(errors),
        f"{prefix}_rounded_mae": (
            sum(abs(error) for error in rounded_errors) / len(rounded_errors)
        ),
        f"{prefix}_rounded_exact_rate": (
            sum(error == 0 for error in rounded_errors) / len(rounded_errors)
        ),
        f"{prefix}_within_one_rate": (
            sum(abs(error) <= 1.0 for error in errors) / len(errors)
        ),
        f"{prefix}_max_absolute_error": max(abs(error) for error in errors),
    })


def macro_family_mae(rows: Iterable[Dict[str, object]], prediction: str) -> float:
    """Give each kernel family equal weight despite differing mask counts."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)
    return sum(
        mean_absolute_error(group_rows, prediction)
        for group_rows in grouped.values()
    ) / len(grouped)


def calibrate_unseen_family_interval(
    model: Dict[str, object], rows: Sequence[Dict[str, object]], quantile: float,
) -> None:
    """Adapter-compatible wrapper around the core empirical interval."""
    real_rows = [
        row for row in rows
        if not is_synthetic_row(row)
    ] or list(rows)
    if not real_rows:
        return
    portable_rows = [{
        "group": row["family"],
        "prediction": row["prediction"],
        "compiled_ii": row["compiled_ii"],
    } for row in real_rows]
    model.update(core_calibrate_interval(
        model, portable_rows, quantile=quantile
    ))


def parse_name_mapping(values: Sequence[str], option: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise SystemExit(f"invalid {option} NAME=VALUE: {value}")
        if source in mapping and mapping[source] != target:
            raise SystemExit(f"conflicting {option} entries for {source}")
        mapping[source] = target
    return mapping


def portable_sample_metadata(row: Sample) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "adapter": "neura",
        "rows": row["rows"],
        "tiles": row["tiles"],
        "links": row["links"],
        "source_kind": (
            "generated"
            if is_synthetic_row(row)
            else "real"
        ),
    }
    for field in SAMPLE_PROVENANCE_FIELDS + (
        "input_report_path", "input_report_sha256",
    ):
        if field in row:
            metadata[field] = row[field]
    return metadata


def motif_corpus_summary(
    samples: Sequence[Sample], manifest_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Report generated-corpus units without treating censored rows as labels."""
    generated_rows = [row for row in samples if is_synthetic_row(row)]
    manifest: Optional[Mapping[str, object]] = None
    if manifest_path is not None and manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text())
        if isinstance(loaded, Mapping):
            manifest = loaded
    records = list(manifest.get("candidates", [])) if manifest else []
    source_rows = generated_rows
    base_ids = {
        (str(row.get("motif")), str(row.get("base_id")))
        for row in source_rows
        if row.get("base_id")
    }
    canonical_dfgs = {
        str(row.get("canonical_dfg_sha256"))
        for row in source_rows
        if row.get("canonical_dfg_sha256")
    }
    lineages = {
        str(row.get("lineage", row.get("family")))
        for row in source_rows
        if row.get("lineage", row.get("family"))
    }
    candidates = {
        str(row.get("candidate_id", row.get("index")))
        for row in source_rows
        if row.get("candidate_id", row.get("index"))
    }
    summary: Dict[str, object] = {
        "base_dfg_count": len(base_ids),
        "distinct_canonical_dfg_count": len(canonical_dfgs),
        "lineage_count": len(lineages),
        "successful_candidate_count": len(candidates),
        "label_count": len(generated_rows),
        "censored_candidate_count": 0,
        "manifest_path": (
            str(manifest_path.resolve()) if manifest_path is not None else None
        ),
        "manifest_sha256": (
            file_sha256(manifest_path)
            if manifest_path is not None and manifest_path.is_file() else None
        ),
    }
    if manifest is not None:
        summary.update({
            "manifest_schema_version": manifest.get("schema_version"),
            "manifest_status": manifest.get("status"),
            "declared_candidate_count": len(records),
            "successful_candidate_count": sum(
                record.get("status") == "success" for record in records
            ),
            "censored_candidate_count": sum(
                record.get("status") == "censored" for record in records
            ),
            "running_candidate_count": sum(
                record.get("status") == "running" for record in records
            ),
            "distinct_manifest_base_dfg_count": len({
                (str(record.get("motif")), str(record.get("base_id")))
                for record in records
                if record.get("base_id")
            }),
            "distinct_manifest_canonical_dfg_count": len({
                str(record.get("canonical_dfg_sha256")) for record in records
                if record.get("canonical_dfg_sha256")
            }),
            "generator_family_count": len({
                str(record.get("generator_family"))
                for record in records
                if record.get("generator_family")
            }),
            "distinct_manifest_lineage_count": len({
                str(record.get("lineage")) for record in records
                if record.get("lineage")
            }),
        })
    return summary


def motif_coverage_summary(
    samples: Sequence[Sample], required_families: Sequence[str],
    required_shapes: Sequence[str], required_variants: Sequence[str],
    minimum_complete_bases_per_family: int,
    requested_bases_per_family: Optional[int] = None,
    declared_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, object]:
    """Compute the auditable generated-motif coverage contract.

    A base is identified only by its canonical DFG hash.  Under the v3
    balanced-incomplete design, each base declares the full 4x4 target plus
    one secondary rectangle.  A base is complete when every *declared* cell
    succeeds; global cell coverage is balanced across base indices.  This
    avoids treating an unattempted shape as a mapper failure while retaining
    paired per-DFG evidence for the shape effect.
    """
    families = tuple(dict.fromkeys(str(value) for value in required_families))
    shapes = tuple(dict.fromkeys(str(value) for value in required_shapes))
    variants = tuple(dict.fromkeys(str(value) for value in required_variants))
    cells = tuple(f"{shape}/{variant}" for shape in shapes for variant in variants)
    cell_set = set(cells)
    requested = (
        None if requested_bases_per_family is None
        else int(requested_bases_per_family)
    )
    family_bases: Dict[str, Dict[str, Set[str]]] = {
        family: {} for family in families
    }
    cell_bases: Dict[Tuple[str, str], Set[str]] = {
        (family, cell): set() for family in families for cell in cells
    }
    cell_candidate_counts: Dict[Tuple[str, str], int] = {
        (family, cell): 0 for family in families for cell in cells
    }
    declared_family_bases: Dict[str, Set[str]] = {
        family: set() for family in families
    }
    declared_family_cells: Dict[Tuple[str, str], Set[str]] = {
        (family, cell): set() for family in families for cell in cells
    }
    declared_invalid_rows = 0
    declared_duplicate_rows = 0
    declared_cross_family_duplicate_rows = 0
    declared_count = 0
    declared_seen_pairs: Set[Tuple[str, str, str]] = set()
    declared_base_cells: Dict[Tuple[str, str], Set[str]] = {}
    declared_base_indices: Dict[Tuple[str, str], int] = {}
    declared_family_indices: Dict[str, Set[int]] = {
        family: set() for family in families
    }
    declared_design_mismatch_count = 0
    declared_canonical_family: Dict[str, str] = {}
    observed_families: Set[str] = set()
    canonical_family: Dict[str, str] = {}
    invalid_rows = 0
    duplicate_rows = 0
    cross_family_duplicate_rows = 0
    unpredeclared_successful_rows = 0
    successful_rows = 0
    hash_pattern = re.compile(r"[0-9a-f]{64}")

    for row in declared_candidates or ():
        declared_count += 1
        if not isinstance(row, Mapping):
            declared_invalid_rows += 1
            continue
        family = str(row.get("generator_family", ""))
        canonical = row.get("canonical_dfg_sha256", row.get("base_dfg_id"))
        raw_rows = row.get("rows")
        raw_columns = row.get("columns")
        raw_tiles = row.get("tiles")
        variant = row.get("architecture_variant")
        raw_base_index = row.get("base_index")
        if (
            raw_tiles is None and isinstance(raw_rows, int) and
            isinstance(raw_columns, int)
        ):
            raw_tiles = raw_rows * raw_columns
        if (
            family not in family_bases or
            not isinstance(canonical, str) or
            hash_pattern.fullmatch(canonical) is None or
            isinstance(raw_rows, bool) or not isinstance(raw_rows, int) or
            isinstance(raw_tiles, bool) or not isinstance(raw_tiles, int) or
            raw_rows <= 0 or raw_tiles <= 0 or raw_tiles % raw_rows != 0 or
            not isinstance(variant, str) or not variant or
            isinstance(raw_base_index, bool) or
            not isinstance(raw_base_index, int) or raw_base_index < 0
        ):
            declared_invalid_rows += 1
            continue
        cell = f"{raw_rows}x{raw_tiles // raw_rows}/{variant}"
        if cell not in cell_set:
            declared_invalid_rows += 1
            continue
        owner = declared_canonical_family.get(canonical)
        if owner is not None and owner != family:
            declared_invalid_rows += 1
            declared_cross_family_duplicate_rows += 1
            continue
        declared_canonical_family[canonical] = family
        pair = (family, canonical, cell)
        if pair in declared_seen_pairs:
            declared_duplicate_rows += 1
            continue
        declared_seen_pairs.add(pair)
        declared_family_bases[family].add(canonical)
        declared_family_cells[(family, cell)].add(canonical)
        base_key = (family, canonical)
        previous_index = declared_base_indices.get(base_key)
        if previous_index is not None and previous_index != raw_base_index:
            declared_invalid_rows += 1
            continue
        declared_base_indices[base_key] = raw_base_index
        declared_base_cells.setdefault(base_key, set()).add(cell)

    parsed_shapes = tuple(neura_motifs.parse_shape(shape) for shape in shapes)
    primary_shape = (
        neura_motifs.PRIMARY_SHAPE
        if neura_motifs.PRIMARY_SHAPE in parsed_shapes else
        max(parsed_shapes, key=lambda item: (item[0] * item[1], item[0], item[1]))
    ) if parsed_shapes else None
    secondary_shapes = tuple(
        shape for shape in parsed_shapes if shape != primary_shape
    )

    def expected_cells_for_index(base_index: int) -> Set[str]:
        if primary_shape is None:
            return set()
        selected = [primary_shape]
        if secondary_shapes:
            selected.append(secondary_shapes[base_index % len(secondary_shapes)])
        return {
            f"{rows}x{columns}/{variant}"
            for rows, columns in selected for variant in variants
        }

    for base_key, declared_cells in declared_base_cells.items():
        family, _ = base_key
        base_index = declared_base_indices[base_key]
        if base_index in declared_family_indices[family]:
            declared_design_mismatch_count += 1
        declared_family_indices[family].add(base_index)
        if declared_cells != expected_cells_for_index(base_index):
            declared_design_mismatch_count += 1

    for row in samples:
        if not is_synthetic_row(row):
            continue
        family = str(row.get("generator_family", ""))
        observed_families.add(family)
        canonical = row.get("canonical_dfg_sha256", row.get("base_dfg_id"))
        if not isinstance(canonical, str) or hash_pattern.fullmatch(canonical) is None:
            invalid_rows += 1
            continue
        owner = canonical_family.get(canonical)
        if owner is not None and owner != family:
            # The global minimum is a distinct-canonical-D FG gate.  Reject
            # cross-family reuse explicitly instead of allowing one DFG to
            # satisfy two per-family thresholds.
            invalid_rows += 1
            cross_family_duplicate_rows += 1
            continue
        canonical_family[canonical] = family
        raw_rows = row.get("rows")
        raw_tiles = row.get("tiles")
        variant = row.get("architecture_variant")
        if (
            isinstance(raw_rows, bool) or not isinstance(raw_rows, (int, float)) or
            not float(raw_rows).is_integer() or int(raw_rows) <= 0 or
            isinstance(raw_tiles, bool) or not isinstance(raw_tiles, (int, float)) or
            not float(raw_tiles).is_integer() or int(raw_tiles) <= 0 or
            int(raw_tiles) % int(raw_rows) != 0 or
            not isinstance(variant, str) or not variant
        ):
            invalid_rows += 1
            continue
        shape = f"{int(raw_rows)}x{int(raw_tiles) // int(raw_rows)}"
        cell = f"{shape}/{variant}"
        if family not in family_bases or cell not in cell_set:
            invalid_rows += 1
            continue
        family_map = family_bases[family]
        base_cells = family_map.setdefault(canonical, set())
        if cell in base_cells:
            duplicate_rows += 1
            continue
        base_cells.add(cell)
        cell_bases[(family, cell)].add(canonical)
        cell_candidate_counts[(family, cell)] += 1
        if (family, canonical, cell) not in declared_seen_pairs:
            unpredeclared_successful_rows += 1
        successful_rows += 1

    family_records: Dict[str, Dict[str, object]] = {}
    complete_by_family: Dict[str, Set[str]] = {}
    declared_complete_by_family: Dict[str, Set[str]] = {}
    all_successful_bases: Set[str] = set()
    for family in families:
        base_map = family_bases[family]
        complete = {
            canonical for canonical, seen_cells in base_map.items()
            if set(seen_cells) == declared_base_cells.get(
                (family, canonical), cell_set
            ) and len(seen_cells) == len(declared_base_cells.get(
                (family, canonical), cell_set
            ))
        }
        complete_by_family[family] = complete
        declared_complete_by_family[family] = {
            canonical for canonical in declared_family_bases[family]
            if declared_base_cells.get((family, canonical), set()) ==
            expected_cells_for_index(declared_base_indices[(family, canonical)])
        }
        all_successful_bases.update(base_map)
        family_records[family] = {
            "successful_distinct_base_dfg_count": len(base_map),
            "complete_distinct_base_dfg_count": len(complete),
            "requested_base_dfg_count": requested,
            "complete_fraction": (
                len(complete) / requested
                if requested is not None and requested > 0 else None
            ),
            "shape_variant_distinct_base_counts": {
                cell: len(cell_bases[(family, cell)]) for cell in cells
            },
            "shape_variant_successful_candidate_counts": {
                cell: cell_candidate_counts[(family, cell)] for cell in cells
            },
            "missing_shape_variant_cells": {
                canonical: [
                    cell for cell in sorted(declared_base_cells.get(
                        (family, canonical), cell_set
                    ))
                    if cell not in seen_cells
                ]
                for canonical, seen_cells in sorted(base_map.items())
                if set(seen_cells) != declared_base_cells.get(
                    (family, canonical), cell_set
                )
            },
            "minimum_complete_bases_per_family": (
                minimum_complete_bases_per_family
            ),
            "complete_fraction_threshold": (
                minimum_complete_bases_per_family / requested
                if requested is not None and requested > 0 else None
            ),
            "passed": (
                len(complete) >= minimum_complete_bases_per_family and
                (requested is None or (
                    requested > 0 and len(complete) <= requested and
                    len(complete) / requested >= (
                        minimum_complete_bases_per_family / requested
                    )
                ))
            ),
            "declared_distinct_base_dfg_count": len(
                declared_family_bases[family]
            ),
            "declared_complete_distinct_base_dfg_count": len(
                declared_complete_by_family[family]
            ),
        }

    missing_families = [family for family in families if family not in observed_families]
    unexpected_families = sorted(set(observed_families).difference(families))
    complete_ids = {
        canonical
        for complete in complete_by_family.values()
        for canonical in complete
    }
    complete_count = len(complete_ids)
    minimum_total = minimum_complete_bases_per_family * len(families)
    cells_per_base = len(variants) * (2 if secondary_shapes else 1)
    expected_declared_count = (
        requested * len(families) * cells_per_base
        if requested is not None else None
    )
    expected_declared_by_family = {
        family: requested for family in families
    }
    expected_declared_by_cell = {
        f"{family}/{cell}": sum(
            cell in expected_cells_for_index(base_index)
            for base_index in range(requested or 0)
        )
        for family in families for cell in cells
    }
    declaration_contract_passed = bool(
        requested is not None and requested > 0 and
        declared_candidates is not None and
        declared_count == expected_declared_count and
        declared_invalid_rows == 0 and declared_duplicate_rows == 0 and
        declared_cross_family_duplicate_rows == 0 and
        declared_design_mismatch_count == 0 and
        {
            family: len(declared_family_bases[family]) for family in families
        } == expected_declared_by_family and
        {family: declared_family_indices[family] for family in families} == {
            family: set(range(requested)) for family in families
        } and
        {
            f"{family}/{cell}": len(declared_family_cells[(family, cell)])
            for family in families for cell in cells
        } == expected_declared_by_cell
    )
    passed = bool(
        minimum_complete_bases_per_family > 0 and
        not missing_families and not unexpected_families and
        invalid_rows == 0 and duplicate_rows == 0 and
        unpredeclared_successful_rows == 0 and declaration_contract_passed and
        all(record["passed"] for record in family_records.values()) and
        complete_count >= minimum_total
    )
    return {
        "required_generator_families": list(families),
        "required_shape_variant_cells": list(cells),
        "minimum_complete_bases_per_family": minimum_complete_bases_per_family,
        "minimum_total_complete_bases": minimum_total,
        "requested_bases_per_family": requested,
        "requested_total_bases": (
            requested * len(families) if requested is not None else None
        ),
        "declared_candidate_count": declared_count,
        "declared_invalid_candidate_count": declared_invalid_rows,
        "declared_duplicate_candidate_count": declared_duplicate_rows,
        "declared_cross_family_duplicate_candidate_count": (
            declared_cross_family_duplicate_rows
        ),
        "declared_design_mismatch_count": declared_design_mismatch_count,
        "candidate_design": neura_motifs.SHAPE_DESIGN,
        "declared_cells_per_base": cells_per_base,
        "declared_distinct_base_dfg_count": len({
            canonical for values in declared_family_bases.values()
            for canonical in values
        }),
        "declared_distinct_base_dfg_counts_by_family": {
            family: len(declared_family_bases[family]) for family in families
        },
        "declared_complete_base_dfg_counts_by_family": {
            family: len(declared_complete_by_family[family])
            for family in families
        },
        "declared_shape_variant_distinct_base_counts": {
            f"{family}/{cell}": len(declared_family_cells[(family, cell)])
            for family in families for cell in cells
        },
        "declared_complete_base_dfg_count": len({
            canonical
            for values in declared_complete_by_family.values()
            for canonical in values
        }),
        "expected_declared_candidate_count": expected_declared_count,
        "declaration_contract_passed": declaration_contract_passed,
        "observed_generator_families": sorted(observed_families),
        "missing_generator_families": missing_families,
        "unexpected_generator_families": unexpected_families,
        "successful_candidate_count": successful_rows,
        "invalid_candidate_count": invalid_rows,
        "duplicate_candidate_count": duplicate_rows,
        "cross_family_duplicate_candidate_count": cross_family_duplicate_rows,
        "unpredeclared_successful_candidate_count": (
            unpredeclared_successful_rows
        ),
        "successful_distinct_base_dfg_count": len(all_successful_bases),
        "complete_base_dfg_count": complete_count,
        "complete_distinct_base_dfg_count": len(complete_ids),
        "families": family_records,
        "passed": passed,
    }


def complete_generated_training_subset(
    samples: Sequence[Sample], required_cells: Sequence[str],
    declared_candidates: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[Sample], Dict[str, object]]:
    """Keep generated bases with one successful row per declared cell.

    The full labelled report remains available for auditing and denominator
    accounting.  Only this returned subset may be used for model fitting or
    holdout evaluation when generated motif rows are present.
    """
    cell_set = set(str(value) for value in required_cells)
    declared_cells: Dict[Tuple[str, str], Set[str]] = {}
    declared_lineages: Dict[Tuple[str, str], str] = {}
    for candidate in declared_candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        family = str(candidate.get("generator_family", ""))
        canonical = str(candidate.get(
            "canonical_dfg_sha256", candidate.get("base_dfg_id", "")
        ))
        raw_rows = candidate.get("rows")
        raw_columns = candidate.get("columns")
        raw_tiles = candidate.get("tiles")
        variant = candidate.get("architecture_variant")
        if raw_tiles is None and isinstance(raw_rows, int) and isinstance(
            raw_columns, int
        ):
            raw_tiles = raw_rows * raw_columns
        if (
            not family.startswith("generated/motif/") or not canonical or
            isinstance(raw_rows, bool) or not isinstance(raw_rows, int) or
            isinstance(raw_tiles, bool) or not isinstance(raw_tiles, int) or
            raw_rows <= 0 or raw_tiles <= 0 or raw_tiles % raw_rows != 0 or
            not isinstance(variant, str)
        ):
            continue
        cell = f"{raw_rows}x{raw_tiles // raw_rows}/{variant}"
        key = (family, canonical)
        declared_cells.setdefault(key, set()).add(cell)
        lineage = candidate.get("leakage_lineage_id", candidate.get("lineage"))
        if isinstance(lineage, str) and lineage:
            declared_lineages[key] = lineage
    generated_rows: List[Sample] = []
    other_rows: List[Sample] = []
    groups: Dict[Tuple[str, str], List[Sample]] = {}
    group_cells: Dict[Tuple[str, str], Set[str]] = {}
    for row in samples:
        family = str(row.get("generator_family", ""))
        # ``samples`` in an imported experiment report is already the exact
        # fit/evaluation population selected by that report.  Reapplying the
        # current corpus design without its original pre-mapper declarations
        # would incorrectly interpret a balanced incomplete design as a
        # partial full Cartesian design.  Newly collected rows have no
        # input_report_path and remain subject to the declaration check below.
        if (not family.startswith("generated/motif/") or
                "input_report_path" in row):
            other_rows.append(row)
            continue
        generated_rows.append(row)
        canonical = str(row.get(
            "canonical_dfg_sha256", row.get("base_dfg_id", "")
        ))
        raw_rows = row.get("rows")
        raw_tiles = row.get("tiles")
        variant = row.get("architecture_variant")
        if (
            isinstance(raw_rows, bool) or not isinstance(raw_rows, (int, float)) or
            not float(raw_rows).is_integer() or
            isinstance(raw_tiles, bool) or not isinstance(raw_tiles, (int, float)) or
            not float(raw_tiles).is_integer() or int(raw_rows) <= 0 or
            int(raw_tiles) <= 0 or int(raw_tiles) % int(raw_rows) != 0 or
            not isinstance(variant, str)
        ):
            cell = "<invalid>"
        else:
            cell = f"{int(raw_rows)}x{int(raw_tiles) // int(raw_rows)}/{variant}"
        key = (family, canonical)
        groups.setdefault(key, []).append(row)
        group_cells.setdefault(key, set()).add(cell)

    if not generated_rows:
        return list(samples), {
            "required_shape_variant_cells": list(required_cells),
            "included_sample_count": len(samples),
            "excluded_sample_count": 0,
            "excluded_lineage_count": 0,
            "excluded_lineage_ids": [],
            "excluded_sample_ids": [],
            "partial_generated_base_count": 0,
            "all_generated_lineages_complete": True,
            "passed": True,
        }

    excluded_rows: List[Sample] = []
    excluded_keys: List[Tuple[str, str]] = []
    for key in sorted(set(groups).union(declared_cells)):
        # A complete lineage has exactly one successful sample for each
        # required cell.  Set equality alone would admit an extra duplicate
        # row for a cell and make the training denominator ambiguous.
        expected_cells = declared_cells.get(key, cell_set)
        if not (
            group_cells.get(key, set()) == expected_cells and
            len(groups.get(key, ())) == len(expected_cells)
        ):
            excluded_rows.extend(groups.get(key, ()))
            excluded_keys.append(key)
    # Keep the original report order.  Besides making the fit deterministic,
    # this lets the frozen validator compare the producer's selected rows
    # with its independently recomputed subset byte-for-byte.
    excluded_key_set = set(excluded_keys)
    included_keys = set(groups).difference(excluded_key_set)
    included = [
        row for row in samples
        if (not str(row.get("generator_family", "")).startswith(
            "generated/motif/"
        ) or "input_report_path" in row) or (
            str(row.get("generator_family", "")), str(row.get(
                "canonical_dfg_sha256", row.get("base_dfg_id", "")
            ))) in included_keys
    ]
    excluded_lineages = sorted({
        declared_lineages.get(key, key[1]) for key in excluded_keys
    })
    excluded_sample_ids = sorted(
        str(row.get("index", row.get("sample_id", "")))
        for row in excluded_rows
    )
    return included, {
        "required_shape_variant_cells": list(required_cells),
        "candidate_design": (
            neura_motifs.SHAPE_DESIGN
            if declared_candidates is not None else "full-cartesian-legacy"
        ),
        "included_sample_count": len(included),
        "excluded_sample_count": len(excluded_rows),
        "excluded_lineage_count": len(excluded_lineages),
        "excluded_lineage_ids": excluded_lineages,
        "excluded_sample_ids": excluded_sample_ids,
        "partial_generated_base_count": len(excluded_keys),
        "all_generated_lineages_complete": not excluded_rows,
        # ``passed`` describes that filtering was successfully applied, not
        # that there happened to be no incomplete labels to exclude.
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neura-root", type=Path,
        default=resolve_configured_neura_root(),
        help=("Neura checkout; defaults to NEURA_ROOT, then the initialized "
              "third_party/neura submodule."),
    )
    parser.add_argument(
        "--opt", type=Path,
        help="mlir-neura-opt binary; defaults under the selected Neura root.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/neura-ii-predictor-corpus"))
    parser.add_argument(
        "--samples", type=int, default=0,
        help=("Number of generated synthetic DFGs to label. Defaults to zero so "
              "report reuse and --predict-fixture never invoke the mapper."),
    )
    parser.add_argument(
        "--motif-samples-per-family", type=int, default=0,
        help=("Number of deterministic base DFGs per compute motif family. "
              "Defaults to zero; unlike legacy --samples this emits a "
              "multi-motif corpus and a pre-mapper manifest."),
    )
    parser.add_argument(
        "--motif", dest="motif", action="append", default=[],
        metavar="NAME[,NAME...]",
        help=("Compute motif family to generate; repeat or use commas. "
              "Defaults to all nine motif-v3 families."),
    )
    parser.add_argument(
        "--motifs", dest="motifs_alias", action="append", default=[],
        metavar="NAME[,NAME...]",
        help="Alias for --motif (kept for experiment scripts).",
    )
    parser.add_argument(
        "--motif-shape", action="append", default=[], metavar="ROWSxCOLS",
        help=("Target rectangles covered by the balanced design (default: "
              "all 2x2 through 4x4 shapes)."),
    )
    parser.add_argument(
        "--motif-architecture-variant", action="append", default=[],
        metavar="NAME[,NAME...]",
        help="Pinned architecture identity (default and only value: neura-main).",
    )
    parser.add_argument(
        "--motif-jobs", type=int, default=1,
        help="Maximum number of independent motif candidates collected concurrently.",
    )
    parser.add_argument(
        "--motif-resume", action="store_true",
        help="Resume a predeclared motif manifest; cached successes are revalidated.",
    )
    parser.add_argument(
        "--motif-checkpoint-every", type=int, default=32,
        help="Atomically checkpoint the motif manifest after this many completions.",
    )
    parser.add_argument(
        "--random-c-samples", type=int, default=0,
        help=("Generate bounded C loops, lower them through the normal "
              "frontend, and label them with the heuristic mapper."),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=(f"Generator seed (fresh default: {DEFAULT_SEED}; on motif "
              "resume, an omitted seed is inherited from the manifest)"),
    )
    parser.add_argument("--timeout", type=int, default=15,
                        help="Per cost/map invocation timeout in seconds")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--ridge-candidates", default="0.1,0.3,1,3,10,30",
        help=("Comma-separated positive ridge values considered only by the "
              "nested family-holdout selector."),
    )
    parser.add_argument(
        "--residual-dead-zone-candidates", default="0,0.25,0.5,0.75,1,1.5,2",
        help=("Comma-separated non-negative residual thresholds considered "
              "only by the nested family-holdout selector."),
    )
    interval = parser.add_mutually_exclusive_group()
    interval.add_argument(
        "--interval-quantile", type=float, default=0.9,
        help=("Empirical quantile of held-out-family maximum errors; not a "
              "formal coverage guarantee"),
    )
    interval.add_argument(
        "--interval-coverage", type=float, dest="legacy_interval_coverage",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tree-depth", type=int, default=1,
                        help="Maximum depth of the residual regression tree")
    parser.add_argument("--tree-min-samples", type=int, default=3)
    parser.add_argument(
        "--input-report", action="append", default=[], type=Path,
        help=("Existing report.json whose labels should be reused. This allows "
              "evaluation without rerunning the mapper; duplicate sample IDs "
              "are deduplicated."),
    )
    parser.add_argument(
        "--model-report", type=Path,
        help=("Load an already trained residual-Ridge artifact for "
              "--predict-fixture. In this prediction-only mode no candidate "
              "label is used, no model is refitted, and no mapper is run."),
    )
    parser.add_argument(
        "--feature-source", action="append", default=[], metavar="NAME=PATH",
        help=("Source DFG used to hydrate structural features of legacy input "
              "reports. Repeat for every real family whose report predates a "
              "new feature."),
    )
    parser.add_argument(
        "--family-lineage", action="append", default=[],
        metavar="FAMILY=LINEAGE",
        help=("Merge related source variants into one leakage-safe evaluation "
              "lineage; applied before every holdout and fit"),
    )
    parser.add_argument(
        "--family-suite", action="append", default=[], metavar="FAMILY=SUITE",
        help="Record an explicitly known benchmark suite without guessing it",
    )
    parser.add_argument(
        "--metadata-holdout-key", action="append", default=[],
        choices=("architecture_id", "suite", "generator_family"),
        help=("Also run nested holdout by an auditable metadata domain. This "
              "can be expensive when the domain has many distinct values."),
    )
    parser.add_argument(
        "--real-fixture", action="append", default=[], metavar="NAME=PATH",
        help=("Pre-lowered Neura DFG to label with the existing mapper. May be "
              "specified more than once."),
    )
    parser.add_argument("--cxx", default="clang++",
                        help="C++ compiler used by --random-c-samples")
    parser.add_argument("--llvm-extract", type=Path,
                        default=resolve_default_llvm_tool("llvm-extract"))
    parser.add_argument("--mlir-translate", type=Path,
                        default=resolve_default_llvm_tool("mlir-translate"))
    parser.add_argument(
        "--mapped-real-fixture", action="append", default=[],
        metavar="NAME=SOURCE=MAPPED=ROWSxCOLS",
        help=("Completed heuristic mapping artifact to reuse as a label. The "
              "artifact must contain mapping_strategy=heuristic and compiled_ii."),
    )
    parser.add_argument(
        "--predict-fixture", action="append", default=[], metavar="NAME=PATH",
        help=("Pre-lowered Neura DFG to estimate without invoking the mapper. "
              "Requires either --model-report or a labelled corpus to train "
              "a model."),
    )
    parser.add_argument(
        "--predict-shape", action="append", default=[], metavar="ROWSxCOLS",
        help=("Shape(s) for --predict-fixture (default: scan every rectangle "
              "from 2x2 through 4x4 and report the area/II Pareto frontier)."),
    )
    parser.add_argument(
        "--real-shape", action="append", default=[],
        metavar="ROWSxCOLS",
        help="Shape(s) used for each --real-fixture (default: 3x3, 3x4, 4x4).",
    )
    parser.add_argument(
        "--real-architecture", type=Path,
        help=("Base YAML architecture used for real fixtures; defaults under "
              "--neura-root."),
    )
    parser.add_argument(
        "--real-random-masks", type=int, default=0,
        help=("Also label this many connected non-rectangular 4x4 shapes per "
              "real fixture."),
    )
    parser.add_argument("--real-mask-min-tiles", type=int, default=10)
    parser.add_argument("--real-mask-seed", type=int, default=20260903)
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output directory before generation")
    args = parser.parse_args()
    if args.motif_jobs < 1:
        parser.error("--motif-jobs must be a positive integer")
    if args.motif_checkpoint_every < 1:
        parser.error("--motif-checkpoint-every must be a positive integer")
    if args.clean and args.motif_resume:
        parser.error("--clean and --motif-resume are mutually exclusive")
    if args.opt is None:
        if args.neura_root is None:
            parser.error(
                "initialize third_party/neura, provide --neura-root/NEURA_ROOT, "
                "or provide an explicit --opt"
            )
        args.opt = resolve_default_opt(args.neura_root)
    if args.real_architecture is None:
        if args.neura_root is None:
            parser.error(
                "initialize third_party/neura, provide --neura-root/NEURA_ROOT, "
                "or provide --real-architecture"
            )
        args.real_architecture = (
            args.neura_root / "test/arch_spec/architecture.yaml"
        )
    return args


def main() -> int:
    args = parse_args()
    interval_quantile = (
        args.legacy_interval_coverage
        if args.legacy_interval_coverage is not None
        else args.interval_quantile
    )
    if not 0.0 < interval_quantile <= 1.0:
        raise SystemExit("--interval-quantile must be in (0, 1]")
    if args.random_c_samples < 0:
        raise SystemExit("--random-c-samples must be non-negative")
    if args.random_c_samples:
        for tool in (args.llvm_extract, args.mlir_translate):
            if not tool.is_file():
                raise SystemExit(f"frontend tool not found: {tool}")
    if args.samples < 0:
        raise SystemExit("--samples must be non-negative")
    if args.motif_samples_per_family < 0:
        raise SystemExit("--motif-samples-per-family must be non-negative")
    if args.model_report is not None and (
        args.motif_resume or args.motif_samples_per_family or args.samples or
        args.random_c_samples or args.input_report or args.real_fixture or
        args.mapped_real_fixture or args.real_random_masks or
        args.metadata_holdout_key
    ):
        raise SystemExit(
            "--model-report is prediction-only and cannot be combined with "
            "motif resume or label collection"
        )
    try:
        selected_motifs = neura_motifs.parse_motif_names(
            list(args.motif) + list(args.motifs_alias)
        )
        selected_motif_shapes = neura_motifs.parse_shapes(args.motif_shape)
        selected_motif_architectures = neura_motifs.parse_architecture_variants(
            args.motif_architecture_variant
        )
    except ValueError as error:
        raise SystemExit(str(error))

    # Motif inputs and the complete manifest are prepared before inspecting
    # the compiler executable or running any other subprocess.  On resume,
    # omitted generator options inherit the immutable values in the manifest;
    # explicitly supplied options are checked by the loader below.
    motif_manifest_path = (args.output_dir / "corpus-manifest.json").resolve()
    motif_count = args.motif_samples_per_family
    if args.motif_resume and motif_manifest_path.is_file():
        try:
            existing_manifest = json.loads(motif_manifest_path.read_text())
            existing_generator = existing_manifest.get("generator", {})
            if args.seed is None:
                args.seed = int(existing_generator["seed"])
            if not args.motif and not args.motifs_alias:
                selected_motifs = tuple(
                    str(value) for value in existing_generator.get(
                        "motifs", selected_motifs
                    )
                )
            if not args.motif_shape:
                selected_motif_shapes = neura_motifs.parse_shapes(
                    existing_generator.get("shapes", ())
                )
            if not args.motif_architecture_variant:
                selected_motif_architectures = neura_motifs.parse_architecture_variants(
                    existing_generator.get(
                        "architecture_variants", selected_motif_architectures
                    )
                )
            if not motif_count:
                motif_count = int(existing_generator.get("count_per_family", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid motif resume manifest: {error}")
    if args.seed is None:
        args.seed = DEFAULT_SEED

    # Materialize every new corpus candidate and atomically predeclare it
    # before *any* mapper invocation (including legacy --samples below).
    # With the default zero count no files/manifest are created, preserving
    # the legacy generator's behavior and cost.
    motif_candidates: Tuple[neura_motifs.MotifCandidate, ...] = ()
    motif_manifest: Optional[Dict[str, object]] = None
    cached_motif_samples: Dict[str, Sample] = {}
    declared_motif_ids: List[str] = []
    prior_motif_failure_events: Dict[str, List[Dict[str, object]]] = {}
    if motif_count or args.motif_resume:
        try:
            (
                motif_manifest, motif_candidates, cached_motif_samples,
                declared_motif_ids, prior_motif_failure_events,
            ) = _load_or_create_motif_manifest(
                args.output_dir, motif_manifest_path,
                resume=args.motif_resume, clean=args.clean,
                count=motif_count, seed=args.seed, motifs=selected_motifs,
                shapes=selected_motif_shapes,
                variants=selected_motif_architectures,
                timeout=args.timeout, jobs=args.motif_jobs,
                checkpoint_every=args.motif_checkpoint_every,
                opt=args.opt,
                architecture_source=args.real_architecture,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid motif corpus configuration: {error}")
        print(
            f"motif_manifest={'resumed' if args.motif_resume else 'predeclared'} "
            f"candidates={len(motif_candidates)} path={motif_manifest_path}"
        )
    else:
        if args.clean and args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    active_motif_manifest_path = (
        motif_manifest_path if motif_manifest is not None else None
    )

    # This check is deliberately after motif predeclaration/resume validation:
    # a corrupt cached success must fail before any compiler invocation.  A
    # terminal motif-only resume rebuilds samples from hashed artifacts and
    # therefore neither requires nor probes a compiler executable.
    compiler_required = bool(
        declared_motif_ids or args.samples or args.random_c_samples or
        args.real_fixture or args.mapped_real_fixture or args.predict_fixture
    )
    if compiler_required:
        if not args.opt.is_file():
            raise SystemExit(f"mlir-neura-opt not found: {args.opt}")
        if declared_motif_ids and motif_manifest is not None:
            collection = motif_manifest.get("collection", {})
            if file_sha256(args.opt) != collection.get("mlir_neura_opt_sha256"):
                raise SystemExit(
                    "mlir-neura-opt changed after motif corpus predeclaration"
                )
        try:
            require_opt_argument(args.opt, "--analyze-rec-res-mii")
        except ValueError as error:
            raise SystemExit(str(error))
    predictor_repository_provenance = git_provenance(PROJECT_ROOT)
    neura_repository_provenance = git_provenance(args.neura_root)
    external_model: Optional[LoadedModel] = None
    if args.model_report is not None:
        if not args.predict_fixture:
            raise SystemExit("--model-report requires at least one --predict-fixture")
        if (
            args.samples or args.random_c_samples or args.input_report or
            args.real_fixture or args.mapped_real_fixture or
            args.real_random_masks or args.metadata_holdout_key or
            args.motif_samples_per_family or args.motif_resume
        ):
            raise SystemExit(
                "--model-report is prediction-only and cannot be combined with "
                "label collection or --input-report"
            )
        try:
            external_model = load_model_artifact(args.model_report)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid --model-report: {error}")
        available_features = set(MODEL_FEATURE_NAMES)
        missing_model_features = set(
            external_model.model["feature_names"]
        ).difference(available_features)
        if missing_model_features:
            raise SystemExit(
                "--model-report requires features unavailable from the Neura "
                f"prediction pass: {sorted(missing_model_features)}"
            )
        if tuple(external_model.model["feature_names"]) != MODEL_FEATURE_NAMES:
            raise SystemExit(
                "--model-report must use the predeclared structure-only "
                "Model-1 feature contract"
            )
        expected_bound_contract = {
            "name": "rec_res_max_v1",
            "formula": "max(rec_mii,res_mii)",
            "training_lower_bound_sources": ["rec_res_max_v1"],
            "components_are_model_features": False,
        }
        if external_model.lower_bound_contract != expected_bound_contract:
            raise SystemExit(
                "--model-report must use the exact RecMII/ResMII lower-bound "
                "contract"
            )

    rng = random.Random(args.seed)
    if args.tree_depth < 0:
        raise SystemExit("--tree-depth must be non-negative")
    if args.tree_min_samples < 1:
        raise SystemExit("--tree-min-samples must be positive")
    try:
        ridge_candidates = sorted({
            float(value) for value in args.ridge_candidates.split(",") if value
        })
    except ValueError:
        raise SystemExit("--ridge-candidates must be comma-separated numbers")
    if not ridge_candidates or any(value <= 0 for value in ridge_candidates):
        raise SystemExit("--ridge-candidates must contain positive values")
    try:
        dead_zone_candidates = sorted({
            float(value)
            for value in args.residual_dead_zone_candidates.split(",")
            if value
        })
    except ValueError:
        raise SystemExit(
            "--residual-dead-zone-candidates must be comma-separated numbers"
        )
    if (not dead_zone_candidates or
            any(value < 0 for value in dead_zone_candidates)):
        raise SystemExit(
            "--residual-dead-zone-candidates must contain non-negative values"
        )
    family_lineages = parse_name_mapping(
        args.family_lineage, "--family-lineage"
    )
    family_suites = parse_name_mapping(args.family_suite, "--family-suite")

    feature_sources: Dict[str, Path] = {}
    for value in args.feature_source:
        name, separator, raw_path = value.partition("=")
        source = Path(raw_path)
        if not separator or not name or not source.is_file():
            raise SystemExit(f"invalid --feature-source NAME=PATH: {value}")
        feature_sources[name] = source

    samples: List[Sample] = []
    sibling_cost_features: Dict[str, Dict[str, int]] = {}
    input_report_provenance: List[Dict[str, object]] = []
    for report_path in args.input_report:
        if not report_path.is_file():
            raise SystemExit(f"input report not found: {report_path}")
        report_hash = file_sha256(report_path)
        loaded = json.loads(report_path.read_text())
        input_report_provenance.append({
            "path": str(report_path.resolve()),
            "sha256": report_hash,
            "provenance": loaded.get("provenance", {}),
        })
        sibling_cost_features.update(load_sibling_cost_features(report_path))
        for row in loaded.get("samples", []):
            if not isinstance(row, dict):
                raise SystemExit(f"invalid sample in input report: {report_path}")
            try:
                samples.append(normalize_input_sample(
                    row, report_path, report_hash,
                    loaded.get("provenance", {}),
                ))
            except ValueError as error:
                raise SystemExit(
                    f"invalid sample in input report {report_path}: {error}"
                )
    for index in range(args.samples):
        rows, columns = rng.choice(((1, 3), (1, 4), (1, 5), (2, 3),
                                    (2, 4), (3, 3), (4, 4)))
        spec = SampleSpec(
            index=index,
            seed=rng.randrange(1 << 30),
            rows=rows,
            columns=columns,
            split_domain=rng.randint(0, 1),
            sources=rng.randint(2, min(7, rows * columns)),
            operations=rng.randint(6, 32),
            fanout_bias=rng.choice((0.0, 0.5, 1.5, 3.0)),
            # Register files are allocated in eight-register banks by the
            # Architecture implementation; smaller values describe zero
            # physical registers and are intentionally not sampled.
            registers=rng.choice((8, 16, 32, 64)),
        )
        sample_dir = args.output_dir / f"sample-{index:03d}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_sample(args.opt, sample_dir, spec, args.timeout)
        if result is not None:
            samples.append(result)
            print(
                f"sample={index:03d} lb={result['baseline_lb']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']} "
                f"moves={result['moves']} split={result['split_domain']}"
            )
        else:
            print(f"sample={index:03d} unavailable", file=sys.stderr)

    if motif_candidates:
        if motif_manifest is None:
            raise SystemExit("motif candidates lack a predeclared manifest")
        coordinator = MotifCollectionCoordinator(
            args.opt, motif_candidates, motif_manifest_path, motif_manifest,
            args.timeout, args.motif_jobs, args.motif_checkpoint_every,
            cached_motif_samples, prior_motif_failure_events,
        )
        try:
            motif_result = coordinator.run()
        except RuntimeError as error:
            print(f"motif collection failed: {error}", file=sys.stderr)
            return 1
        samples.extend(motif_result.samples)
        for result in motif_result.results:
            if result.status == "success" and result.sample is not None:
                print(
                    f"motif={result.candidate_id} "
                    f"lb={result.sample['baseline_lb']} "
                    f"compiled={result.sample['compiled_ii']}"
                )
            elif result.status == "censored":
                print(
                    f"motif={result.candidate_id} unavailable",
                    file=sys.stderr,
                )
        if motif_result.interrupted:
            # Do not fit or emit a training report from an interrupted corpus;
            # resume can continue from the atomically checkpointed manifest.
            return 130

    for index in range(args.random_c_samples):
        rows, columns = rng.choice(((3, 3), (3, 4), (4, 4)))
        spec = CSpec(
            index=index,
            seed=rng.randrange(1 << 30),
            rows=rows,
            columns=columns,
            terms=rng.randint(1, 4),
            loop_trip_count=rng.choice((8, 12, 16, 24, 32)),
        )
        sample_dir = args.output_dir / f"c-sample-{index:03d}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_c_sample(
            args.opt, sample_dir, spec, args.real_architecture, args.cxx,
            args.llvm_extract, args.mlir_translate, args.timeout,
        )
        if result is not None:
            samples.append(result)
            print(
                f"c-sample={index:03d} lb={result['baseline_lb']} "
                f"rec={result['rec_mii']} res={result['res_mii']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']} "
                f"terms={spec.terms} trip={spec.loop_trip_count}"
            )
        else:
            print(f"c-sample={index:03d} unavailable", file=sys.stderr)

    shapes: List[Tuple[int, int]] = []
    for value in args.real_shape or ["3x3", "3x4", "4x4"]:
        match = re.fullmatch(r"(\d+)x(\d+)", value)
        if not match:
            raise SystemExit(f"invalid --real-shape (expected ROWSxCOLS): {value}")
        rows, columns = (int(component) for component in match.groups())
        if rows < 1 or columns < 1:
            raise SystemExit("--real-shape dimensions must be positive")
        shapes.append((rows, columns))
    for fixture in args.real_fixture:
        name, separator, raw_path = fixture.partition("=")
        source = Path(raw_path)
        if not separator or not name or not source.is_file():
            raise SystemExit(f"invalid --real-fixture NAME=PATH: {fixture}")
        for rows, columns in shapes:
            sample_dir = args.output_dir / f"real-{name}-{rows}x{columns}"
            sample_dir.mkdir(exist_ok=True)
            result = collect_real_fixture(
                args.opt, sample_dir, name, source, args.real_architecture,
                rows, columns, args.timeout,
            )
            if result is not None:
                samples.append(result)
                print(
                    f"real={result['index']} lb={result['baseline_lb']} "
                    f"compiled={result['compiled_ii']} nodes={result['nodes']}"
                )
            else:
                print(f"real={name}-{rows}x{columns} unavailable", file=sys.stderr)
        if args.real_random_masks:
            if not 1 <= args.real_mask_min_tiles <= 15:
                raise SystemExit("--real-mask-min-tiles must be in [1, 15]")
            seed_offset = sum(ord(char) for char in name)
            masks = random_connected_tile_masks(
                random.Random(args.real_mask_seed + seed_offset),
                args.real_random_masks, args.real_mask_min_tiles,
            )
            for mask_index, mask in enumerate(masks):
                sample_dir = args.output_dir / f"real-{name}-mask-{mask_index:03d}"
                sample_dir.mkdir(exist_ok=True)
                result = collect_real_fixture(
                    args.opt, sample_dir, name, source, args.real_architecture,
                    4, 4, args.timeout, mask, suffix=f"-mask{mask_index:03d}",
                )
                if result is not None:
                    samples.append(result)
                    print(
                        f"real={result['index']} lb={result['baseline_lb']} "
                        f"compiled={result['compiled_ii']} tiles={result['tiles']}"
                    )
                else:
                    print(f"real={name}-mask{mask_index:03d} unavailable",
                          file=sys.stderr)

    for fixture in args.mapped_real_fixture:
        parts = fixture.split("=", 3)
        if len(parts) != 4:
            raise SystemExit(
                "invalid --mapped-real-fixture NAME=SOURCE=MAPPED=ROWSxCOLS"
            )
        name, raw_source, raw_mapped, shape = parts
        source, mapped = Path(raw_source), Path(raw_mapped)
        match = re.fullmatch(r"(\d+)x(\d+)", shape)
        if not name or not source.is_file() or not mapped.is_file() or not match:
            raise SystemExit(
                "invalid --mapped-real-fixture NAME=SOURCE=MAPPED=ROWSxCOLS"
            )
        rows, columns = (int(component) for component in match.groups())
        if rows < 1 or columns < 1:
            raise SystemExit("--mapped-real-fixture dimensions must be positive")
        sample_dir = args.output_dir / f"completed-{name}-{rows}x{columns}"
        sample_dir.mkdir(exist_ok=True)
        result = collect_completed_real_fixture(
            args.opt, sample_dir, name, source, mapped, args.real_architecture,
            rows, columns, args.timeout,
        )
        if result is not None:
            samples.append(result)
            print(
                f"completed={result['index']} lb={result['baseline_lb']} "
                f"compiled={result['compiled_ii']} nodes={result['nodes']}"
            )
        else:
            print(f"completed={name}-{shape} unavailable", file=sys.stderr)

    # Multiple reports may repeat a byte-equivalent logical row, but an ID may
    # never silently replace different features, provenance, or a new label.
    try:
        samples = deduplicate_samples_by_id(samples)
    except ValueError as error:
        raise SystemExit(str(error))
    required_features = set(FEATURE_NAMES)
    for row in samples:
        row.update(sibling_cost_features.get(str(row["index"]), {}))
        add_prediction_features(row)
        row.setdefault(
            "rec_res_evidence",
            "imported_report_unverified"
            if "input_report_path" in row else
            "rec_res_analysis_evidence_missing",
        )
        missing = required_features.difference(row)
        source_family = str(row["family"])
        source = feature_sources.get(source_family)
        if missing:
            if source is None:
                missing_text = ", ".join(sorted(missing))
                raise SystemExit(
                    f"sample {row['index']} lacks [{missing_text}]; provide "
                    f"--feature-source {source_family}=PATH or recollect it"
                )
            row.update(semantic_features_from_neura(source.read_text()))
            add_prediction_features(row)
            missing = required_features.difference(row)
            if missing:
                raise SystemExit(
                    f"sample {row['index']} still lacks features: {sorted(missing)}"
                )
        row.setdefault("source_family", source_family)
        if neura_repository_provenance.get("revision") is not None:
            row.setdefault(
                "mapper_revision", neura_repository_provenance["revision"]
            )
        if source is not None:
            row.setdefault("source_path", str(source.resolve()))
            row.setdefault("source_sha256", file_sha256(source))
        declared_lineage, effective_lineage = resolve_effective_lineage(
            row, source_family, family_lineages
        )
        row.setdefault("lineage", declared_lineage)
        row.setdefault(
            "declared_leakage_lineage_id",
            row.get("leakage_lineage_id", declared_lineage),
        )
        row["effective_lineage"] = effective_lineage
        row["family"] = effective_lineage
        generated = is_synthetic_row(row)
        row.setdefault("training_stratum", "generated" if generated else "real")
        # This is the authoritative identity actually used by the outer split.
        # Preserve any earlier declaration above, but never report a stale ID
        # after --family-lineage merges related source variants.
        row["leakage_lineage_id"] = effective_lineage
        # Ranking queries require an explicit source/DFG identity.  In
        # particular, never fall back to the leakage lineage: aliases may
        # intentionally merge related source variants for fitting while those
        # variants are not the same DSE query.
        base_dfg_id = row.get("base_dfg_id")
        if base_dfg_id in (None, ""):
            base_dfg_id = row.get(
                "canonical_dfg_sha256",
                row.get("dfg_source_sha256", row.get("source_sha256")),
            )
        if base_dfg_id not in (None, ""):
            row.setdefault("base_dfg_id", base_dfg_id)
            row.setdefault("ranking_query_id", base_dfg_id)
        suite = family_suites.get(source_family, family_suites.get(effective_lineage))
        if suite is not None:
            row["suite"] = suite
    # A generated base is eligible for formal fitting only when every frozen
    # shape/architecture cell produced one successful label.  Keep the full
    # labelled list in ``samples`` for coverage/denominator accounting, but
    # route only this independently derived subset to every fit and holdout.
    required_training_shapes = tuple(
        f"{rows}x{columns}" for rows, columns in neura_motifs.DEFAULT_SHAPES
    )
    required_training_cells = tuple(
        f"{shape}/{variant}"
        for shape in required_training_shapes
        for variant in neura_motifs.DEFAULT_ARCHITECTURE_VARIANTS
    )
    selection_candidates = (
        motif_manifest.get("candidates", ())
        if isinstance(motif_manifest, Mapping) else ()
    )
    training_samples, training_selection = complete_generated_training_subset(
        samples, required_training_cells,
        selection_candidates if isinstance(selection_candidates, list) else (),
    )
    row_holdout: Optional[Dict[str, object]] = None
    family_holdout: Optional[Dict[str, object]] = None
    nested_ridge_holdout: Optional[Dict[str, object]] = None
    metadata_holdouts: Dict[str, Dict[str, object]] = {
        key: {"status": "not_requested", "holdout_key": key}
        for key in ("architecture_id", "suite", "generator_family")
    }
    selected_model = "none"
    selected_rows: List[Dict[str, object]] = []
    trained_full_model: Optional[Dict[str, object]] = None
    if external_model is not None:
        selected_model = "ridge"
        trained_full_model = dict(external_model.model)
    else:
        if len(training_samples) < 12:
            print(
                f"only {len(training_samples)} complete labels collected; "
                "no model fitted",
                file=sys.stderr,
            )
            return 1
        row_holdout = random_row_holdout(
            training_samples, args.seed, args.ridge,
            args.tree_depth, args.tree_min_samples,
        )
        try:
            family_holdout = leave_one_family_out(
                training_samples, args.ridge, args.tree_depth,
                args.tree_min_samples
            )
        except ValueError:
            # A synthetic-only corpus has one family by design.  Its row split
            # may debug the generator, but is not evidence of generalization.
            family_holdout = None
        try:
            nested_ridge_holdout = nested_ridge_family_holdout(
                training_samples, ridge_candidates, dead_zone_candidates
            )
        except ValueError:
            nested_ridge_holdout = None
        for key in dict.fromkeys(args.metadata_holdout_key):
            metadata_holdouts[key] = nested_ridge_metadata_holdout(
                training_samples, key, ridge_candidates, dead_zone_candidates
            )
        if nested_ridge_holdout is not None:
            # Model 1 is predeclared as residual Ridge. Tree/row-split results
            # remain diagnostics and never select the reported point model.
            selected_model = "ridge"
            selected_rows = nested_ridge_holdout["rows"]
            selected_ridge, selected_dead_zone = select_ridge_hyperparameters(
                training_samples, ridge_candidates, dead_zone_candidates
            )
            trained_full_model = fit_ridge(
                training_samples, selected_ridge, selected_dead_zone
            )
            if selected_rows:
                calibrate_unseen_family_interval(
                    trained_full_model, selected_rows, interval_quantile
                )
    active_model_sha256 = (
        canonical_model_sha256(trained_full_model)
        if trained_full_model is not None else None
    )
    prediction_request_count = 0
    prediction_failures: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    if args.predict_fixture:
        if trained_full_model is None:
            raise SystemExit(
                "--predict-fixture needs at least two labelled kernel families"
            )
        prediction_shapes: List[Tuple[int, int]] = []
        default_prediction_shapes = [
            f"{rows}x{columns}"
            for rows, columns in neura_motifs.DEFAULT_SHAPES
        ]
        for value in args.predict_shape or default_prediction_shapes:
            match = re.fullmatch(r"(\d+)x(\d+)", value)
            if not match:
                raise SystemExit(
                    "invalid --predict-shape (expected ROWSxCOLS): " + value
                )
            rows, columns = (int(component) for component in match.groups())
            if not (2 <= rows <= 4 and 2 <= columns <= 4):
                raise SystemExit(
                    "--predict-shape must be a rectangle from 2x2 through 4x4"
                )
            if (rows, columns) not in prediction_shapes:
                prediction_shapes.append((rows, columns))
        for fixture in args.predict_fixture:
            name, separator, raw_path = fixture.partition("=")
            source = Path(raw_path)
            if not separator or not name or not source.is_file():
                raise SystemExit(f"invalid --predict-fixture NAME=PATH: {fixture}")
            for rows, columns in prediction_shapes:
                prediction_request_count += 1
                sample_dir = (
                    args.output_dir / f"prediction-{name}-{rows}x{columns}"
                )
                sample_dir.mkdir(exist_ok=True)
                features = collect_prediction_fixture(
                    args.opt, sample_dir, name, source, args.real_architecture,
                    rows, columns, args.timeout,
                )
                if features is None:
                    prediction_failures.append({
                        "sample": f"{name}-{rows}x{columns}",
                        "stage": "feature_collection",
                        "error": "Rec/Res analysis or feature extraction unavailable",
                    })
                    print(f"prediction={name}-{rows}x{columns} unavailable",
                          file=sys.stderr)
                    continue
                if neura_repository_provenance.get("revision") is not None:
                    features.setdefault(
                        "mapper_revision",
                        neura_repository_provenance["revision"],
                    )
                try:
                    point = predict_unlabelled_candidate(
                        trained_full_model, features
                    )
                except (KeyError, ValueError, OverflowError) as error:
                    prediction_failures.append({
                        "sample": features.get("index"),
                        "stage": "point_prediction",
                        "error": str(error),
                    })
                    print(
                        f"prediction={name}-{rows}x{columns} failed: {error}",
                        file=sys.stderr,
                    )
                    continue
                model_features = point["model_features"]
                raw_residual = float(point["raw_predicted_residual"])
                predicted_residual = float(point["predicted_residual"])
                predicted_ii = float(point["predicted_compiled_ii"])
                prediction_warnings: List[str] = []
                feature_support = point["feature_support"]
                outside_range = feature_support["outside_observed_range"]
                if outside_range:
                    prediction_warnings.append(
                        "feature_outside_training_range:" +
                        ",".join(str(value) for value in outside_range)
                    )
                if external_model is not None:
                    training_neura = external_model.provenance.get("neura")
                    expected_revision = (
                        training_neura.get("revision")
                        if isinstance(training_neura, Mapping) else None
                    )
                    current_revision = neura_repository_provenance.get("revision")
                    if current_revision is None:
                        prediction_warnings.append(
                            "mapper_revision_unrecorded_for_prediction"
                        )
                    if (
                        expected_revision and current_revision and
                        str(expected_revision) != str(current_revision)
                    ):
                        prediction_warnings.append(
                            "mapper_revision_mismatch:"
                            f"model={expected_revision},input={current_revision}"
                        )
                    if (
                        isinstance(training_neura, Mapping) and
                        training_neura.get("dirty") is True
                    ):
                        prediction_warnings.append(
                            "model_training_producer_dirty"
                        )
                    if neura_repository_provenance.get("dirty") is True:
                        prediction_warnings.append(
                            "prediction_feature_producer_dirty"
                        )
                    if (
                        external_model.artifact_status is not None and
                        ("exploratory" in external_model.artifact_status or
                         "not_frozen" in external_model.artifact_status or
                         "unverified" in external_model.artifact_status)
                    ):
                        prediction_warnings.append(
                            "model_artifact_status:"
                            f"{external_model.artifact_status}"
                        )
                    training_bound_sources = (
                        external_model.lower_bound_contract.get(
                            "training_lower_bound_sources"
                        )
                    )
                    if not external_model.lower_bound_contract:
                        prediction_warnings.append(
                            "model_lower_bound_contract_unrecorded"
                        )
                    if (
                        isinstance(training_bound_sources, list) and
                        training_bound_sources and
                        str(features["lower_bound_source"]) not in {
                            str(value) for value in training_bound_sources
                        }
                    ):
                        prediction_warnings.append(
                            "lower_bound_source_mismatch:model=" +
                            ",".join(sorted(
                                str(value) for value in training_bound_sources
                            )) + ",input=" +
                            str(features["lower_bound_source"])
                        )
                prediction = {
                    "sample": features["index"],
                    "task": name,
                    "shape": f"{rows}x{columns}",
                    "rows": rows,
                    "columns": columns,
                    "tile_count": rows * columns,
                    "predicted_compiled_ii": predicted_ii,
                    "prediction_kind": "continuous_point_estimate",
                    "raw_predicted_residual": raw_residual,
                    "nonnegative_predicted_residual": max(0.0, raw_residual),
                    "predicted_residual": predicted_residual,
                    "nonnegative_floor_applied": raw_residual < 0.0,
                    "dead_zone": float(trained_full_model.get(
                        "residual_dead_zone", 0.0
                    )),
                    "dead_zone_applied": (
                        raw_residual > 0.0 and predicted_residual == 0.0
                    ),
                    "baseline_lb": features["baseline_lb"],
                    "lower_bound": features["baseline_lb"],
                    "lower_bound_source": features["lower_bound_source"],
                    "rec_mii": features["rec_mii"],
                    "res_mii": features["res_mii"],
                    "lineage": features.get("lineage"),
                    "leakage_lineage_id": features.get("leakage_lineage_id"),
                    "base_dfg_id": features.get("base_dfg_id"),
                    "ranking_query_id": features.get("ranking_query_id"),
                    "training_stratum": features.get("training_stratum", "real"),
                    "generator_family": features.get("generator_family"),
                    "generator_version": features.get("generator_version"),
                    "motif": features.get("motif"),
                    "source_path": features.get("source_path"),
                    "source_sha256": features.get("source_sha256"),
                    "architecture_id": features.get("architecture_id"),
                    "candidate_id": features.get("candidate_id"),
                    "mapper_id": features.get("mapper_id"),
                    "mapper_revision": features.get("mapper_revision"),
                    "mapper_config": features.get("mapper_config"),
                    "model_features": model_features,
                    "feature_support": feature_support,
                    "model_sha256": active_model_sha256,
                    "warnings": prediction_warnings,
                    "evaluation": {
                        "status": "prediction_only_unlabelled_in_this_invocation",
                        "prediction_input_compiled_ii_present": False,
                        "prediction_input_compiled_ii_used": False,
                        "model_container_may_include_historical_training_labels": (
                            external_model is not None and
                            external_model.container in {
                                "portable-model-report-v2",
                                "neura-experiment-v2",
                            }
                        ),
                        "heuristic_mapper_invoked_by_this_invocation": False,
                        "model_refit_by_this_invocation": external_model is None,
                        "frozen_blind_claim": False,
                    },
                }
                radius = trained_full_model.get(
                    "unseen_group_absolute_error_radius"
                )
                if radius is not None:
                    interval_lower = max(
                        float(features["baseline_lb"]),
                        predicted_ii - float(radius),
                    )
                    interval_upper = predicted_ii + float(radius)
                    if not (
                        np.isfinite(interval_lower) and
                        np.isfinite(interval_upper)
                    ):
                        prediction_failures.append({
                            "sample": features.get("index"),
                            "stage": "prediction_interval",
                            "error": "prediction interval is not finite",
                        })
                        print(
                            f"prediction={name}-{rows}x{columns} failed: "
                            "prediction interval is not finite",
                            file=sys.stderr,
                        )
                        continue
                    prediction["interval_lower"] = interval_lower
                    prediction["interval_upper"] = interval_upper
                    prediction["interval_kind"] = (
                        "empirical_held_out_group_max_error_quantile"
                    )
                    prediction["interval_empirical_quantile"] = (
                        trained_full_model.get(
                            "unseen_group_interval_empirical_quantile"
                        )
                    )
                    prediction["formal_interval_coverage_guarantee"] = False
                predictions.append(prediction)
                print(
                    f"prediction={prediction['sample']} "
                    f"compiled_ii={prediction['predicted_compiled_ii']:.2f} "
                    f"lb={prediction['baseline_lb']}"
                )
    shape_selections = shape_selection_summary(predictions)
    prediction_complete = (
        prediction_request_count == len(predictions)
    )
    motif_collection_identity = (
        motif_manifest.get("collection", {})
        if motif_manifest is not None else {}
    )
    reported_opt_path = (
        motif_collection_identity.get("mlir_neura_opt")
        if motif_candidates else str(args.opt.resolve())
    )
    reported_opt_sha256 = (
        motif_collection_identity.get("mlir_neura_opt_sha256")
        if motif_candidates else file_sha256(args.opt)
    )
    provenance = {
        "adapter": "neura",
        "predictor_repository": predictor_repository_provenance,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": file_sha256(Path(__file__)),
        "neura": neura_repository_provenance,
        "mlir_neura_opt": reported_opt_path,
        "mlir_neura_opt_sha256": reported_opt_sha256,
        "motif_collection": motif_collection_identity or None,
        "architecture": str(args.real_architecture.resolve()),
        "architecture_sha256": file_sha256(args.real_architecture),
        "mapping_strategy": "heuristic",
        "label_policy": (
            "no_prediction_input_labels_used"
            if external_model is not None else
            "successful_compiled_ii_only_timeouts_are_not_labels"
        ),
        "loaded_model": ({
            "path": str(external_model.source_path),
            "source_sha256": external_model.source_sha256,
            "model_sha256": external_model.model_sha256,
            "container": external_model.container,
            "target": external_model.target,
            "artifact_status": external_model.artifact_status,
            "lower_bound_contract": dict(external_model.lower_bound_contract),
        } if external_model is not None else None),
        "input_reports": input_report_provenance,
        "experiment_config": {
            "mode": (
                "prediction_only_loaded_model"
                if external_model is not None else "train_evaluate_optional_predict"
            ),
            "seed": args.seed,
            "timeout_seconds": args.timeout,
            "ridge_candidates": ridge_candidates,
            "residual_dead_zone_candidates": dead_zone_candidates,
            "interval_empirical_quantile": interval_quantile,
            "motif_samples_per_family": motif_count,
            "motif_jobs": args.motif_jobs,
            "motif_resume": args.motif_resume,
            "motif_checkpoint_every": args.motif_checkpoint_every,
            "motifs": list(selected_motifs),
            "motif_shapes": [
                f"{rows}x{columns}"
                for rows, columns in selected_motif_shapes
            ],
            "motif_architecture_variants": list(selected_motif_architectures),
            "legacy_random_samples": args.samples,
            "metadata_holdout_keys": list(dict.fromkeys(
                args.metadata_holdout_key
            )),
        },
    }
    portable_dataset = {
        "schema_version": "portable-v1",
        "provenance": provenance,
        "feature_names": list(MODEL_FEATURE_NAMES),
        "samples": [{
            "sample_id": str(row["index"]),
            "group": str(row["family"]),
            "lower_bound": row["baseline_lb"],
            "rec_mii": row["rec_mii"],
            "res_mii": row["res_mii"],
            "compiled_ii": row["compiled_ii"],
            "features": {
                name: row[name] for name in FEATURE_NAMES
                if name not in {
                    "baseline_lb", "rec_mii", "res_mii"
                }
            },
            "metadata": portable_sample_metadata(row),
        } for row in training_samples],
        "censored_samples": list(INVOCATION_FAILURES),
    }
    generated_corpus = motif_corpus_summary(
        samples, active_motif_manifest_path
    )
    portable_dataset["motif_corpus"] = generated_corpus
    generated_rows = [row for row in samples if is_synthetic_row(row)]
    generated_base_dfg_count = len({
        str(row.get("base_dfg_id", row.get("canonical_dfg_sha256")))
        for row in generated_rows
        if row.get("base_dfg_id", row.get("canonical_dfg_sha256"))
    })
    generated_family_count = len({
        str(row["generator_family"]) for row in generated_rows
        if row.get("generator_family")
    })
    generated_only_training = bool(samples) and len(generated_rows) == len(samples)
    generator_family_holdout_available = (
        metadata_holdouts["generator_family"].get("status") == "ok"
    )
    required_generator_families = tuple(
        f"generated/motif/{motif}" for motif in neura_motifs.DEFAULT_MOTIFS
    )
    required_shapes = tuple(
        f"{rows}x{columns}" for rows, columns in neura_motifs.DEFAULT_SHAPES
    )
    required_shape_variant_cells = tuple(
        f"{shape}/{variant}"
        for shape in required_shapes
        for variant in neura_motifs.DEFAULT_ARCHITECTURE_VARIANTS
    )
    declared_motif_candidates: Sequence[Mapping[str, Any]] = ()
    if (active_motif_manifest_path is not None and
            active_motif_manifest_path.is_file()):
        declared_manifest = json.loads(active_motif_manifest_path.read_text())
        if isinstance(declared_manifest, Mapping) and isinstance(
            declared_manifest.get("candidates"), list
        ):
            declared_motif_candidates = declared_manifest["candidates"]
    generated_coverage = motif_coverage_summary(
        generated_rows,
        required_generator_families,
        required_shapes,
        neura_motifs.DEFAULT_ARCHITECTURE_VARIANTS,
        200,
        motif_count,
        declared_motif_candidates,
    )
    generated_improvement = generated_nested_improvement_gate(
        nested_ridge_holdout
    )
    model_design_full_rank = bool(
        isinstance(trained_full_model, Mapping) and
        trained_full_model.get("training_design_rank") ==
        trained_full_model.get("training_design_column_count") ==
        len(MODEL_FEATURE_NAMES) + 1
    )
    frozen_model_scale_ready = bool(
        generated_only_training and trained_full_model is not None and
        model_design_full_rank and
        nested_ridge_holdout is not None and
        generator_family_holdout_available and
        generated_coverage["passed"] is True and
        generated_improvement["passed"] is True
    )
    trained_full_model_sha256 = active_model_sha256
    report_metadata = (
        {
            "evaluation_status": "prediction_only",
            "dataset_role": "unlabelled_prediction_inputs",
            "holdout_protocol": "not_run",
            "labels_available_at_evaluation": False,
            "feature_set_status": "loaded_from_model_artifact",
            "model_class_status": "loaded_residual_ridge",
            "model_selection_status": "not_run_loaded_existing_model",
            "frozen_test": False,
            "blind": False,
        }
        if external_model is not None else
        {
            "evaluation_status": "exploratory",
            "dataset_role": "labeled_train_validation",
            "holdout_protocol": "nested_lineage_holdout",
            "labels_available_at_evaluation": True,
            "feature_set_status": "predeclared_model_1_feature_set",
            "model_class_status": "predeclared_residual_ridge",
            "model_selection_status": "selected_on_this_labeled_dataset",
            "frozen_test": False,
            "blind": False,
        }
    )
    report = {
        "schema_version": "neura-experiment-v2",
        "target": "compiled_ii_from_neura_heuristic_mapper",
        "artifact_status": (
            external_model.artifact_status
            if external_model is not None else
            ("exploratory_imported_rec_res_unverified"
             if input_report_provenance else "exploratory_not_frozen")
        ),
        "lower_bound_contract": {
            "name": "rec_res_max_v1",
            "formula": "max(rec_mii,res_mii)",
            "training_lower_bound_sources": ["rec_res_max_v1"],
            "components_are_model_features": False,
        },
        "provenance": provenance,
        "feature_names": list(FEATURE_NAMES),
        "model_feature_names": (
            list(trained_full_model["feature_names"])
            if trained_full_model is not None else list(MODEL_FEATURE_NAMES)
        ),
        # ``samples`` is the exact fit/evaluation population.  Preserve all
        # successful labels separately so a partial generated lineage remains
        # visible in the report and coverage accounting without entering the
        # model or holdout splits.
        "samples": training_samples,
        "labelled_samples": samples,
        "training_selection": training_selection,
        "random_row_holdout_diagnostic": row_holdout,
        "family_holdout": family_holdout,
        "nested_ridge_family_holdout": nested_ridge_holdout,
        "nested_ridge_metadata_holdouts": metadata_holdouts,
        "selected_model": selected_model,
        "trained_full_model": trained_full_model,
        "trained_full_model_sha256": trained_full_model_sha256,
        "prediction_request_count": prediction_request_count,
        "prediction_completed_count": len(predictions),
        "prediction_completion_status": (
            "complete" if prediction_complete else "incomplete"
        ),
        "prediction_failures": prediction_failures,
        "predictions": predictions,
        "shape_selections": shape_selections,
        "invocation_failures": list(INVOCATION_FAILURES),
        "candidate_status": (
            (
                "offline_point_prediction_complete"
                if prediction_complete else "offline_point_prediction_incomplete"
            ) if external_model is not None else
            "offline_experiment_only_never_compiler_input"
        ),
        "report_metadata": report_metadata,
        "grouping": {
            "effective_group_key": "family",
            "declared_lineage_key": "lineage",
            "effective_lineage_key": "effective_lineage",
            "family_lineages": family_lineages,
            "family_suites": family_suites,
            "effective_group_count": len({
                str(row["family"]) for row in training_samples
            }),
        },
        "motif_corpus": generated_corpus,
        "candidate_gate": {
            "requires": [
                "generated-only training rows with source/canonical identities",
                "at least 200 complete distinct base DFGs per generator family",
                "both declared shape cells for every complete base",
                "balanced global coverage of all 2x2-through-4x4 rectangles",
                "nested generated-base lineage model selection",
                "whole-generator-family holdout requested and available",
                "nested generated-lineage Ridge macro MAE strictly below LB",
                "full-rank intercept-plus-feature training design",
                "structure-only model features disjoint from Rec/Res floor",
            ],
            "required_generator_families": list(required_generator_families),
            "required_shape_variant_cells": list(required_shape_variant_cells),
            "requested_bases_per_family": motif_count,
            "requested_total_bases": (
                motif_count * len(required_generator_families)
            ),
            "minimum_complete_bases_per_family": 200,
            "minimum_total_complete_bases": 200 * len(required_generator_families),
            "minimum_complete_fraction": (
                200 / motif_count if motif_count > 0 else None
            ),
            "coverage": generated_coverage,
            "generated_nested_improvement": generated_improvement,
            "model_design_full_rank": model_design_full_rank,
            "model_design_rank": (
                trained_full_model.get("training_design_rank")
                if isinstance(trained_full_model, Mapping) else None
            ),
            "model_design_column_count": (
                trained_full_model.get("training_design_column_count")
                if isinstance(trained_full_model, Mapping) else None
            ),
            "training_selection": training_selection,
            "generated_only_training": generated_only_training,
            "generated_distinct_base_dfg_count": generated_base_dfg_count,
            "generated_generator_family_count": generated_family_count,
            "nested_lineage_holdout_available": nested_ridge_holdout is not None,
            "generator_family_holdout_available": generator_family_holdout_available,
            "architecture_holdout_available": (
                metadata_holdouts["architecture_id"]["status"] == "ok"
            ),
            "dse_ranking_eligible_group_count": (
                nested_ridge_holdout["ridge_group_ranking"]
                ["eligible_group_count"]
                if nested_ridge_holdout is not None else 0
            ),
            "overall_ready_for_machsuite_freeze": frozen_model_scale_ready,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    (args.output_dir / "dataset.json").write_text(
        json.dumps(portable_dataset, indent=2, allow_nan=False) + "\n")
    if external_model is not None:
        print(
            "mode=prediction_only "
            f"model_sha256={external_model.model_sha256} "
            f"predictions={len(predictions)}"
        )
    elif family_holdout is not None:
        print(
            "family_holdout=" + str(len(family_holdout["ridge_rows"])) +
            f" baseline_mae={family_holdout['baseline_mae']:.3f}" +
            f" ridge_mae={family_holdout['ridge_mae']:.3f}" +
            f" tree_mae={family_holdout['tree_mae']:.3f}" +
            f" selected={selected_model}"
        )
    else:
        print("family_holdout=unavailable (fewer than two independent families)")
    if nested_ridge_holdout is not None:
        print(
            "nested_ridge_family_holdout=" +
            str(len(nested_ridge_holdout["rows"])) +
            f" baseline_mae={nested_ridge_holdout['baseline_mae']:.3f}" +
            f" ridge_mae={nested_ridge_holdout['ridge_mae']:.3f}" +
            f" rounded_exact="
            f"{nested_ridge_holdout['ridge_rounded_exact_rate']:.3f}"
        )
    if row_holdout is not None:
        print(
            "row_holdout_diagnostic=" + str(len(row_holdout["ridge_rows"])) +
            f" baseline_mae={row_holdout['baseline_mae']:.3f}" +
            f" ridge_mae={row_holdout['ridge_mae']:.3f}" +
            f" tree_mae={row_holdout['tree_mae']:.3f}"
        )
    if family_holdout is not None:
        for row in selected_rows:
            print(
                f"family_holdout sample={row['sample']} family={row['family']} "
                f"lb={row['baseline']:.1f} "
                f"prediction={row['prediction']:.2f} compiled={row['compiled_ii']}"
            )
    if not prediction_complete:
        print(
            f"only {len(predictions)} of {prediction_request_count} requested "
            "predictions completed",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
