#!/usr/bin/env python3
"""Generate an Amoeba task/shape cost catalogue from pointwise models.

The adapter consumes a frozen candidate JSONL manifest, one pre-mapper Neura
DFG per task, analytical RecMII/ResMII facts, and frontend-provided startup
cycles.  It predicts only unique ``(task DFG, mapper rows, mapper cols)``
queries.  Program-level scoring and top-k selection remain Amoeba's job.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from amoeba_protocol import (  # noqa: E402
    CANDIDATE_SCHEMA,
    COST_SCHEMA,
    SEARCH_SCOPE,
    SHAPE_POLICY,
    SOURCE_TASK_BODY_SHA_ATTR,
    SPATIAL_CAPACITY_POLICY,
)
from cgra_ii_predictor.graph_model import (  # noqa: E402
    JointGraphShapeModel,
    PointwiseConfig,
    candidate_context,
    make_cgra_graph,
    parse_neura_dfg_representation,
)
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    SHAPE_PROTOCOL,
    SHAPE_PROTOCOL_ID,
    get_shape_protocol,
)


ANALYTICAL_INPUT_SCHEMA = "cgra-ii-amoeba-query-features"
ADAPTER_FEATURE_SCHEMA = "cgra-ii-amoeba-pointwise-features"
CHECKPOINT_SCHEMA = "cgra-ii-pointwise-model"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final"
DEFAULT_CHECKPOINTS = (
    f"large_operation={FINAL_MODEL_DIR / 'large-operation.pt'}",
    f"baseline={FINAL_MODEL_DIR / 'baseline.pt'}",
    f"ranking={FINAL_MODEL_DIR / 'ranking.pt'}",
)

QueryKey = Tuple[str, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    """Read an analytical bound, for which zero is a valid value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return result


def _sha256_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64 or
        any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def source_task_body_sha256(dfg_text: str, task: str) -> str:
    """Read the source Taskflow body identity embedded by the DFG exporter."""
    import re

    pattern = re.compile(
        rf'"?{re.escape(SOURCE_TASK_BODY_SHA_ATTR)}"?\s*=\s*"([^"]+)"'
    )
    values = pattern.findall(dfg_text)
    if len(values) != 1:
        raise ValueError(
            f"task DFG for {task} must contain exactly one "
            f"{SOURCE_TASK_BODY_SHA_ATTR} attribute"
        )
    return _sha256_string(values[0], f"source body hash for task {task}")


def _static_shape_alphabet(
    grid_rows: int, grid_cols: int, max_cgras_per_task: int,
) -> List[Tuple[int, int]]:
    """Mirror Amoeba's deterministic fixed-orientation rectangle order."""
    shapes = []
    for count in range(1, min(max_cgras_per_task, grid_rows * grid_cols) + 1):
        for rows in range(1, grid_rows + 1):
            if count % rows == 0 and count // rows <= grid_cols:
                shapes.append((rows, count // rows))
    return shapes


@lru_cache(maxsize=None)
def _pack_normalized_rectangles(
    rectangles: Tuple[Tuple[int, int], ...], grid_rows: int, grid_cols: int,
) -> bool:
    """Exactly place fixed-orientation rectangles using a physical-cell mask."""
    placements = []
    for rows, cols in rectangles:
        masks = []
        for origin_row in range(grid_rows - rows + 1):
            for origin_col in range(grid_cols - cols + 1):
                mask = 0
                for row in range(origin_row, origin_row + rows):
                    for col in range(origin_col, origin_col + cols):
                        mask |= 1 << (row * grid_cols + col)
                masks.append(mask)
        placements.append(tuple(masks))

    @lru_cache(maxsize=None)
    def place(rectangle_index: int, occupied: int) -> bool:
        if rectangle_index == len(placements):
            return True
        for mask in placements[rectangle_index]:
            if mask & occupied == 0 and place(
                rectangle_index + 1, occupied | mask,
            ):
                return True
        return False

    return place(0, 0)


def _can_pack_fixed_rectangles(
    rectangles: Sequence[Tuple[int, int]], grid_rows: int, grid_cols: int,
) -> bool:
    """Check simultaneous fit; no rectangle is rotated implicitly."""
    if sum(rows * cols for rows, cols in rectangles) > grid_rows * grid_cols:
        return False
    if any(
        rows <= 0 or cols <= 0 or rows > grid_rows or cols > grid_cols
        for rows, cols in rectangles
    ):
        return False
    # Packing feasibility ignores task identity. Normalizing improves cache reuse
    # without changing the direction of any rectangle.
    normalized = tuple(sorted(
        rectangles,
        key=lambda shape: (shape[0] * shape[1], max(shape), shape),
        reverse=True,
    ))
    return _pack_normalized_rectangles(normalized, grid_rows, grid_cols)


def _packable_shape_index_tuples(
    task_count: int, shape_alphabet: Sequence[Tuple[int, int]],
    grid_rows: int, grid_cols: int,
) -> Iterable[Tuple[int, ...]]:
    """Yield the exact packed subset in Amoeba's task-major shape order."""
    grid_area = grid_rows * grid_cols
    selected_indices: List[int] = []
    selected_shapes: List[Tuple[int, int]] = []

    def visit(task_index: int, selected_area: int) -> Iterable[Tuple[int, ...]]:
        if task_index == task_count:
            if _can_pack_fixed_rectangles(
                selected_shapes, grid_rows, grid_cols,
            ):
                yield tuple(selected_indices)
            return
        for shape_index, shape in enumerate(shape_alphabet):
            area = shape[0] * shape[1]
            if selected_area + area > grid_area:
                continue
            selected_indices.append(shape_index)
            selected_shapes.append(shape)
            yield from visit(task_index + 1, selected_area + area)
            selected_shapes.pop()
            selected_indices.pop()

    yield from visit(0, 0)


def load_candidate_manifest(path: Path) -> Dict[str, Any]:
    """Validate Amoeba's packing-pruned static space and derive its queries.

    Amoeba enumerates the full single-task shape alphabet, keeps only tuples
    whose fixed-orientation rectangles can coexist on the physical grid, and
    assigns contiguous IDs to those survivors.  This adapter validates that
    frozen feasible set; it does not use temporal reuse to admit an
    over-capacity tuple.  ``cost_queries`` is the exact set of task/shape
    pairs referenced by at least one retained candidate.
    """
    header: Optional[Mapping[str, Any]] = None
    footer: Optional[Mapping[str, Any]] = None
    candidates: List[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = _object(json.loads(line), f"line {line_number}")
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid candidate JSONL at line {line_number}: {error}"
            ) from error
        if record.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("candidate manifest schema mismatch")
        kind = record.get("record_type")
        if kind == "header":
            if header is not None or candidates or footer is not None:
                raise ValueError("candidate manifest header is misplaced")
            header = record
        elif kind == "candidate":
            if header is None or footer is not None:
                raise ValueError("candidate record is outside header/footer")
            candidates.append(record)
        elif kind == "footer":
            if header is None or footer is not None:
                raise ValueError("candidate manifest footer is misplaced")
            footer = record
        else:
            raise ValueError(f"unknown candidate record_type: {kind!r}")
    if header is None or footer is None:
        raise ValueError("candidate manifest is incomplete")
    if not isinstance(header.get("function"), str) or not header["function"]:
        raise ValueError("candidate manifest function is missing")
    if header.get("search_scope") != SEARCH_SCOPE:
        raise ValueError("only static Amoeba candidate manifests are supported")
    if header.get("shape_policy") != SHAPE_POLICY:
        raise ValueError("only rectangular Amoeba candidate manifests are supported")
    if header.get("spatial_capacity_policy") != SPATIAL_CAPACITY_POLICY:
        raise ValueError("candidate manifest spatial capacity policy mismatch")
    if footer.get("candidate_count") != len(candidates):
        raise ValueError("candidate manifest footer count mismatch")
    architecture = _object(header.get("architecture"), "header architecture")
    grid_rows = _positive_integer(architecture.get("grid_rows"), "grid_rows")
    grid_cols = _positive_integer(architecture.get("grid_cols"), "grid_cols")
    per_rows = _positive_integer(
        architecture.get("per_cgra_tile_rows"), "per_cgra_tile_rows",
    )
    per_cols = _positive_integer(
        architecture.get("per_cgra_tile_cols"), "per_cgra_tile_cols",
    )
    if (per_rows, per_cols) != (4, 4):
        raise ValueError("model protocol requires 4x4 mapper tiles per physical CGRA")
    architecture_sha256 = _sha256_string(
        architecture.get("spec_sha256"), "architecture spec_sha256",
    )
    max_cgras_per_task = _positive_integer(
        header.get("max_cgras_per_task"), "max_cgras_per_task",
    )
    shape_alphabet = _static_shape_alphabet(
        grid_rows, grid_cols, max_cgras_per_task,
    )

    raw_tasks = header.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("candidate manifest has no task facts")
    task_facts = []
    task_names = set()
    for raw_task in raw_tasks:
        task = _object(raw_task, "task fact")
        name = task.get("task")
        if not isinstance(name, str) or not name or name in task_names:
            raise ValueError("candidate manifest task names must be unique")
        task_names.add(name)
        task_facts.append((
            name,
            _sha256_string(task.get("body_sha256"), "task body_sha256"),
            _positive_integer(task.get("trip_count"), "task trip_count"),
        ))
    task_body_sha256 = {name: body_sha for name, body_sha, _ in task_facts}

    raw_queries = header.get("cost_queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("candidate manifest has no cost_queries")
    queries: List[QueryKey] = []
    seen = set()
    for raw in raw_queries:
        query = _object(raw, "cost query")
        task = query.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("cost query task is missing")
        rows = _positive_integer(query.get("mapper_tile_rows"), "mapper_tile_rows")
        cols = _positive_integer(query.get("mapper_tile_cols"), "mapper_tile_cols")
        key = (task, rows, cols)
        if key in seen:
            raise ValueError("duplicate cost query")
        seen.add(key)
        queries.append(key)

    task_order = [fact[0] for fact in task_facts]
    for task, _, _ in queries:
        if task not in task_names:
            raise ValueError("cost query names a task outside header task facts")

    candidate_queries = set()
    expected_tuples = iter(_packable_shape_index_tuples(
        len(task_order), shape_alphabet, grid_rows, grid_cols,
    ))
    for candidate_index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id")
        if candidate_id != f"candidate-{candidate_index}":
            raise ValueError(
                "candidate identity does not match its canonical manifest index"
            )
        task_shapes = candidate.get("task_shapes")
        if not isinstance(task_shapes, list) or not task_shapes:
            raise ValueError("candidate has no task_shapes")
        if len(task_shapes) != len(task_facts):
            raise ValueError("candidate task count does not match header")
        actual_shapes = []
        for task_index, raw_choice in enumerate(task_shapes):
            choice = _object(raw_choice, "candidate task shape")
            task = choice.get("task")
            expected_task, _, expected_trip_count = task_facts[task_index]
            if task != expected_task or choice.get("trip_count") != expected_trip_count:
                raise ValueError("candidate task facts do not match header")
            shape = _object(choice.get("shape"), "candidate shape")
            if shape.get("kind") != "rect":
                raise ValueError("non-rectangular candidate shape is unsupported")
            physical_rows = _positive_integer(shape.get("rows"), "shape rows")
            physical_cols = _positive_integer(shape.get("cols"), "shape cols")
            if shape.get("cgra_count") != physical_rows * physical_cols:
                raise ValueError("candidate physical CGRA count is invalid")
            if shape.get("cgra_shape") != f"{physical_rows}x{physical_cols}":
                raise ValueError("candidate physical shape label is invalid")
            mapper_rows = _positive_integer(
                shape.get("mapper_tile_rows"), "shape mapper_tile_rows",
            )
            mapper_cols = _positive_integer(
                shape.get("mapper_tile_cols"), "shape mapper_tile_cols",
            )
            if (mapper_rows, mapper_cols) != (
                physical_rows * per_rows, physical_cols * per_cols,
            ):
                raise ValueError("candidate physical-to-mapper conversion is invalid")
            actual_shapes.append((physical_rows, physical_cols))
            candidate_queries.add((task, mapper_rows, mapper_cols))
        try:
            expected_indices = next(expected_tuples)
        except StopIteration as error:
            raise ValueError(
                "candidate manifest contains more than the packable shape space"
            ) from error
        expected_shapes = [shape_alphabet[index] for index in expected_indices]
        if actual_shapes != expected_shapes:
            raise ValueError(
                "candidate manifest is incomplete, unpackable, duplicated, "
                "or out of order"
            )
    try:
        next(expected_tuples)
    except StopIteration:
        pass
    else:
        raise ValueError("candidate manifest omits packable shape tuples")
    if candidate_queries != seen:
        raise ValueError("header cost_queries do not match candidate task shapes")
    return {
        "header": dict(header),
        "candidates": [dict(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
        "queries": queries,
        "task_body_sha256": task_body_sha256,
        "architecture_sha256": architecture_sha256,
        "manifest_sha256": sha256_file(path),
    }


def load_analytical_input(
    path: Path, expected_function: str,
) -> Tuple[Dict[QueryKey, Dict[str, float]], Dict[str, Any]]:
    root = _object(json.loads(path.read_text()), "analytical input")
    if root.get("schema") != ANALYTICAL_INPUT_SCHEMA:
        raise ValueError("analytical input schema mismatch")
    if root.get("function") != expected_function:
        raise ValueError("analytical input function mismatch")
    provenance = dict(_object(
        root.get("provenance"), "analytical input provenance",
    ))
    entries = root.get("entries")
    if not isinstance(entries, list):
        raise ValueError("analytical input entries must be an array")
    result: Dict[QueryKey, Dict[str, float]] = {}
    for raw in entries:
        entry = _object(raw, "analytical entry")
        task = entry.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("analytical entry task is missing")
        rows = _positive_integer(entry.get("mapper_tile_rows"), "mapper_tile_rows")
        cols = _positive_integer(entry.get("mapper_tile_cols"), "mapper_tile_cols")
        # Neura uses RecMII=1 when no recurrence exists, but preserves
        # RecMII=0 for a present zero-length recurrence cycle made only from
        # non-materialized reserve/movement operations. A scoreable task still
        # needs positive ResMII, lower bound, and startup cycles.
        rec_mii = _nonnegative_number(entry.get("rec_mii"), "rec_mii")
        res_mii = _positive_number(entry.get("res_mii"), "res_mii")
        lower_bound = _positive_number(entry.get("lower_bound"), "lower_bound")
        startup = _positive_number(entry.get("startup_cycles"), "startup_cycles")
        if lower_bound != max(rec_mii, res_mii):
            raise ValueError("analytical lower_bound must equal max(RecMII, ResMII)")
        key = (task, rows, cols)
        if key in result:
            raise ValueError("duplicate analytical query")
        result[key] = {
            "rec_mii": rec_mii,
            "res_mii": res_mii,
            "lower_bound": lower_bound,
            "startup_cycles": startup,
        }
    return result, provenance


def parse_task_paths(values: Iterable[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        task, separator, raw_path = value.partition("=")
        if not separator or not task or not raw_path:
            raise ValueError("task DFG must be TASK=PATH")
        if task in result:
            raise ValueError(f"duplicate DFG path for task {task}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"task DFG does not exist: {path}")
        result[task] = path
    return result


def parse_checkpoint_paths(values: Iterable[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("checkpoint must be NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate checkpoint name: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint does not exist: {path}")
        result[name] = path
    if not result:
        raise ValueError("at least one checkpoint is required")
    return result


def load_ensemble_report(
    path: Path, checkpoint_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    report = _object(json.loads(path.read_text()), "ensemble report")
    if report.get("selection_split") != "validation_only":
        raise ValueError("ensemble must be selected on validation only")
    raw_checkpoints = _object(report.get("checkpoints"), "ensemble checkpoints")
    for name, checkpoint_path in checkpoint_paths.items():
        record = _object(raw_checkpoints.get(name), f"checkpoint {name}")
        if record.get("sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"checkpoint SHA-256 mismatch for {name}")
    weights = _object(report.get("weights"), "ensemble weights")
    expected_names = {"analytical_lower_bound", *checkpoint_paths}
    if set(weights) != expected_names:
        raise ValueError("ensemble weights do not match checkpoints")
    numeric_weights = {
        name: float(value) for name, value in weights.items()
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in numeric_weights.values()
    ) or not math.isclose(sum(numeric_weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("ensemble weights must form a probability simplex")
    gating = _object(report.get("uncertainty_gating"), "uncertainty gating")
    exponent = float(gating.get("exponent"))
    scales = _object(gating.get("scales"), "uncertainty scales")
    if set(scales) != expected_names:
        raise ValueError("uncertainty scales do not match ensemble weights")
    numeric_scales = {name: float(value) for name, value in scales.items()}
    if not math.isfinite(exponent) or exponent < 0.0 or any(
        not math.isfinite(value) or value <= 0.0
        for value in numeric_scales.values()
    ):
        raise ValueError("uncertainty gating parameters are invalid")
    architecture_contract = _object(
        report.get("architecture_contract"), "architecture contract",
    )
    contract_id = architecture_contract.get("contract_id")
    training_architecture = architecture_contract.get(
        "training_architecture_sha256"
    )
    supported_architectures = architecture_contract.get(
        "supported_architecture_sha256"
    )
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("ensemble architecture contract_id is missing")
    if (
        not isinstance(training_architecture, str) or
        len(training_architecture) != 64 or
        any(character not in "0123456789abcdef" for character in training_architecture)
    ):
        raise ValueError("ensemble training architecture SHA-256 is invalid")
    if (
        not isinstance(supported_architectures, list) or
        not supported_architectures or
        any(
            not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)
            for value in supported_architectures
        ) or training_architecture not in supported_architectures
    ):
        raise ValueError("ensemble supported architecture set is invalid")

    selected_mode = report.get("selected_ensemble_mode")
    if selected_mode not in {"static", "uncertainty_gated"}:
        validation = _object(report.get("validation"), "validation metrics")
        gated = _object(validation.get("ensemble"), "gated validation metrics")
        static = _object(
            validation.get("static_ensemble"), "static validation metrics",
        )
        selected_mode = (
            "uncertainty_gated"
            if float(gated["mae"]) < float(static["mae"]) else "static"
        )
    return {
        "weights": numeric_weights,
        "uncertainty_exponent": exponent,
        "uncertainty_scales": numeric_scales,
        "selected_mode": selected_mode,
        "manifest_sha256": report.get("manifest_sha256"),
        "architecture_contract": dict(architecture_contract),
        "supported_architecture_sha256": tuple(supported_architectures),
        "sha256": sha256_file(path),
    }


def _gated_weights(
    names: Sequence[str], base_weights: Mapping[str, float],
    deviations: Mapping[str, float], exponent: float,
    scales: Mapping[str, float],
) -> Dict[str, float]:
    result = {}
    for name in names:
        modulation = 1.0
        if name != "analytical_lower_bound":
            relative = max(deviations[name] / scales[name], 1e-4)
            modulation = relative ** (-exponent)
        result[name] = base_weights[name] * modulation
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError("uncertainty gating produced zero total weight")
    return {name: value / total for name, value in result.items()}


def _combine_prediction(
    facts: Mapping[str, float], model_values: Mapping[str, Mapping[str, float]],
    ensemble: Mapping[str, Any], mode: str,
) -> Dict[str, float]:
    names = ["analytical_lower_bound", *model_values]
    means = {"analytical_lower_bound": facts["lower_bound"]}
    means.update({name: value["mean"] for name, value in model_values.items()})
    deviations = {"analytical_lower_bound": 0.0}
    deviations.update({name: value["std"] for name, value in model_values.items()})
    weights = dict(ensemble["weights"])
    if mode == "uncertainty_gated":
        weights = _gated_weights(
            names, weights, deviations,
            float(ensemble["uncertainty_exponent"]),
            ensemble["uncertainty_scales"],
        )
    mean = sum(weights[name] * means[name] for name in names)
    second = sum(
        weights[name] * (deviations[name] ** 2 + means[name] ** 2)
        for name in names
    )
    model_weight = sum(weights[name] for name in model_values)
    success_probability = (
        sum(
            weights[name] * model_values[name]["success_probability"]
            for name in model_values
        ) / model_weight
        if model_weight > 0.0 else
        sum(value["success_probability"] for value in model_values.values()) /
        len(model_values)
    )
    return {
        "predicted_ii": max(facts["lower_bound"], mean),
        "predicted_ii_std": math.sqrt(max(0.0, second - mean * mean)),
        "mapper_success_probability": success_probability,
    }


def validate_ensemble_architecture(
    ensemble: Mapping[str, Any], architecture_sha256: object,
) -> None:
    """Reject inference on hardware for which these labels were not trained."""
    supported = ensemble["supported_architecture_sha256"]
    if architecture_sha256 not in supported:
        supported_text = ", ".join(supported)
        raise ValueError(
            "analytical architecture is outside the deployed model contract: "
            f"got {architecture_sha256!r}; supported SHA-256: {supported_text}. "
            "Collect labels and retrain before using a different architecture."
        )


def generate_catalog(
    candidate_manifest: Path, analytical_input: Path,
    task_paths: Mapping[str, Path], checkpoint_paths: Mapping[str, Path],
    ensemble_report: Path, device: torch.device,
    ensemble_mode: str = "selected",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total_started = time.perf_counter()
    manifest = load_candidate_manifest(candidate_manifest)
    header = manifest["header"]
    function = str(header["function"])
    analytical, analytical_provenance = load_analytical_input(
        analytical_input, function,
    )
    if analytical_provenance.get("candidate_manifest_sha256") != (
        manifest["manifest_sha256"]
    ):
        raise ValueError("analytical input candidate manifest SHA-256 mismatch")
    tasks = sorted({task for task, _, _ in manifest["queries"]})
    if set(task_paths) != set(tasks):
        raise ValueError("task DFG mapping must exactly cover manifest tasks")

    load_started = time.perf_counter()
    models: Dict[str, Tuple[JointGraphShapeModel, PointwiseConfig]] = {}
    checkpoint_metadata = {}
    representations = set()
    for name, path in checkpoint_paths.items():
        artifact = torch.load(path, map_location=device, weights_only=False)
        if artifact.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError(f"checkpoint {name} has an unsupported schema")
        config = PointwiseConfig(**artifact["config"]).validate()
        if config.interaction_mode != "residual_pointwise":
            raise ValueError(f"checkpoint {name} is not an ensemble component")
        if config.shape_protocol != SHAPE_PROTOCOL_ID:
            raise ValueError(f"checkpoint {name} does not support Amoeba shapes")
        model = JointGraphShapeModel(config).to(device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        models[name] = (model, config)
        representations.add(config.dfg_representation)
        checkpoint_metadata[name] = {
            "sha256": sha256_file(path),
            "config_sha256": canonical_json_sha256(config.to_dict()),
        }
    if len(representations) != 1:
        raise ValueError("ensemble checkpoints must share one DFG representation")
    representation = next(iter(representations))
    ensemble = load_ensemble_report(ensemble_report, checkpoint_paths)
    validate_ensemble_architecture(
        ensemble, analytical_provenance.get("architecture_sha256"),
    )
    selected_mode = (
        ensemble["selected_mode"] if ensemble_mode == "selected" else
        ensemble_mode
    )
    if selected_mode not in {"static", "uncertainty_gated"}:
        raise ValueError("ensemble mode must be static or uncertainty_gated")
    model_load_ms = (time.perf_counter() - load_started) * 1000.0

    parse_started = time.perf_counter()
    graphs = {}
    task_identities = {}
    for task in tasks:
        path = task_paths[task]
        task_identities[task] = sha256_file(path)
        dfg_text = path.read_text()
        source_body_sha = source_task_body_sha256(dfg_text, task)
        if source_body_sha != manifest["task_body_sha256"][task]:
            raise ValueError(
                f"task DFG source body hash mismatch for {task}"
            )
        graphs[task] = parse_neura_dfg_representation(
            dfg_text, representation,
        )
    provenance_task_hashes = _object(
        analytical_provenance.get("task_dfg_sha256"),
        "analytical provenance task DFG hashes",
    )
    if provenance_task_hashes != task_identities:
        raise ValueError("analytical input task DFG SHA-256 mismatch")
    provenance_body_hashes = _object(
        analytical_provenance.get("task_body_sha256"),
        "analytical provenance task body hashes",
    )
    if provenance_body_hashes != manifest["task_body_sha256"]:
        raise ValueError("analytical input task body SHA-256 mismatch")
    for field in (
        "neura_opt_sha256", "architecture_sha256",
        "rec_res_source", "startup_cycles_source",
    ):
        if not isinstance(analytical_provenance.get(field), str) or not (
            analytical_provenance[field]
        ):
            raise ValueError(f"analytical input provenance lacks {field}")
    dfg_parse_ms = (time.perf_counter() - parse_started) * 1000.0

    supported_queries = [
        key for key in manifest["queries"]
        if (key[1], key[2]) in SHAPE_PROTOCOL.mapper_shapes
    ]
    unsupported_queries = {
        key for key in manifest["queries"] if key not in supported_queries
    }
    if any(key not in analytical for key in supported_queries):
        missing = next(key for key in supported_queries if key not in analytical)
        raise ValueError(f"supported query lacks analytical input: {missing}")
    extra_analytical = set(analytical).difference(manifest["queries"])
    if extra_analytical:
        raise ValueError("analytical input contains queries outside the manifest")

    queries_by_shape: Dict[Tuple[int, int], List[QueryKey]] = {}
    for key in supported_queries:
        queries_by_shape.setdefault((key[1], key[2]), []).append(key)
    cgra_graphs = {
        shape: make_cgra_graph(*shape, SHAPE_PROTOCOL_ID)
        for shape in queries_by_shape
    }

    inference_started = time.perf_counter()
    predictions: Dict[QueryKey, Dict[str, Dict[str, float]]] = {
        key: {} for key in supported_queries
    }
    with torch.inference_mode():
        for name, (model, config) in models.items():
            for shape, shape_queries in queries_by_shape.items():
                rows, cols = shape
                contexts = [
                    [
                        candidate_context(
                            rows, cols,
                            analytical[(task, rows, cols)]["rec_mii"],
                            analytical[(task, rows, cols)]["res_mii"],
                            analytical[(task, rows, cols)]["lower_bound"],
                            config.mapper_ii_ceiling, config.shape_protocol,
                        )
                    ]
                    for task, _, _ in shape_queries
                ]
                output = model(
                    [graphs[task] for task, _, _ in shape_queries],
                    [cgra_graphs[shape]],
                    torch.tensor(contexts, dtype=torch.float32, device=device),
                )
                means = output["predicted_ii"].detach().cpu().tolist()
                stds = output["predicted_ii_std"].detach().cpu().tolist()
                success = output["success_probability"].detach().cpu().tolist()
                modes = output.get("predicted_ii_mode")
                mode_values = (
                    modes.detach().cpu().tolist() if modes is not None else means
                )
                for query_index, key in enumerate(shape_queries):
                    predictions[key][name] = {
                        "mean": float(means[query_index][0]),
                        "std": float(stds[query_index][0]),
                        "success_probability": float(
                            success[query_index][0]
                        ),
                        "integer_mode": float(
                            mode_values[query_index][0]
                        ),
                    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    entries = []
    cache_keys = set()
    for key in manifest["queries"]:
        task, rows, cols = key
        cache_key = (task_identities[task], rows, cols)
        cache_keys.add(cache_key)
        entry: Dict[str, Any] = {
            "task": task,
            "mapper_tile_rows": rows,
            "mapper_tile_cols": cols,
        }
        if key in unsupported_queries:
            entry["support_status"] = "unsupported"
        else:
            combined = _combine_prediction(
                analytical[key], predictions[key], ensemble, selected_mode,
            )
            entry.update({
                "support_status": "supported",
                "predicted_ii": combined["predicted_ii"],
                "startup_cycles": analytical[key]["startup_cycles"],
                "predicted_ii_std": combined["predicted_ii_std"],
                "mapper_success_probability": (
                    combined["mapper_success_probability"]
                ),
                "analytical_lower_bound": analytical[key]["lower_bound"],
                "ii_mean_source": "pointwise_ensemble",
            })
        entries.append(entry)

    namespace_contract = {
        "feature_schema": ADAPTER_FEATURE_SCHEMA,
        "shape_protocol": get_shape_protocol(
            SHAPE_PROTOCOL_ID
        ).to_dict(),
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "analytical_input_sha256": sha256_file(analytical_input),
        "analytical_provenance": {
            "neura_opt_sha256": analytical_provenance["neura_opt_sha256"],
            "architecture_sha256": analytical_provenance[
                "architecture_sha256"
            ],
            "task_body_sha256": manifest["task_body_sha256"],
            "task_dfg_sha256": task_identities,
            "rec_res_source": analytical_provenance["rec_res_source"],
            "startup_cycles_source": analytical_provenance[
                "startup_cycles_source"
            ],
        },
        "checkpoints": checkpoint_metadata,
        "ensemble_report_sha256": ensemble["sha256"],
        "ensemble_mode": selected_mode,
        "architecture_contract": ensemble["architecture_contract"],
        # The classifier output is retained for later diagnostics. Per the
        # current DSE policy it never changes support_status, predicted_ii,
        # candidate score, or top-k order.
        "ranking_policy": {
            "objective": "predicted_compute_bottleneck",
            "mapper_success_probability": "diagnostic_only",
            "uses_mapper_success_probability": False,
        },
        "ensemble_weights": ensemble["weights"],
        "uncertainty_exponent": ensemble["uncertainty_exponent"],
        "uncertainty_scales": ensemble["uncertainty_scales"],
    }
    namespace_hash = canonical_json_sha256(namespace_contract)
    catalog = {
        "schema": COST_SCHEMA,
        "function": function,
        "namespace": f"cgra-ii-pointwise-{namespace_hash[:24]}",
        "predictor_metadata": namespace_contract,
        "entries": entries,
    }
    timing = {
        "model_load_ms": model_load_ms,
        "dfg_parse_ms": dfg_parse_ms,
        "steady_state_inference_ms": inference_ms,
        "steady_state_ms_per_supported_query": (
            inference_ms / len(supported_queries) if supported_queries else None
        ),
        "total_ms": (time.perf_counter() - total_started) * 1000.0,
        "manifest_candidate_count": manifest["candidate_count"],
        "unique_cost_query_count": len(manifest["queries"]),
        "unique_cache_entry_count": len(cache_keys),
        "supported_query_count": len(supported_queries),
        "unsupported_query_count": len(unsupported_queries),
        "task_dfg_parse_count": len(graphs),
        "model_load_count": len(models),
        "model_forward_pass_count": len(models) * len(queries_by_shape),
        "ensemble_mode": selected_mode,
    }
    return catalog, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analytical-input", type=Path, required=True)
    parser.add_argument(
        "--task-dfg", action="append", default=[], metavar="TASK=PATH",
        help="Pre-mapper Neura DFG; repeat exactly once for each manifest task.",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
    )
    parser.add_argument(
        "--ensemble-report", type=Path,
        default=FINAL_MODEL_DIR / "ensemble.json",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=("selected", "static", "uncertainty_gated"),
        default="selected",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_paths = parse_task_paths(args.task_dfg)
    checkpoint_paths = parse_checkpoint_paths(
        args.checkpoint if args.checkpoint else DEFAULT_CHECKPOINTS
    )
    catalog, timing = generate_catalog(
        args.manifest.resolve(), args.analytical_input.resolve(),
        task_paths, checkpoint_paths, args.ensemble_report.resolve(),
        torch.device(args.device), args.ensemble_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.timing_output is not None:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(
            json.dumps(timing, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "namespace": catalog["namespace"],
        **timing,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
