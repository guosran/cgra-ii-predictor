#!/usr/bin/env python3
"""Generate an Amoeba v2 task/shape cost catalogue from pointwise models.

The adapter consumes a frozen candidate JSONL manifest, one pre-mapper Neura
DFG per task, analytical RecMII/ResMII facts, and frontend-provided startup
cycles.  It predicts only unique ``(task DFG, mapper rows, mapper cols)``
queries.  Program-level scoring and top-k selection remain Amoeba's job.
"""

from __future__ import annotations

import argparse
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

from cgra_ii_predictor.graph_model import (  # noqa: E402
    JointGraphShapeModel,
    Model2Config,
    candidate_context,
    make_cgra_graph,
    parse_neura_dfg_representation,
)
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    AMOEBA_STATIC_PROTOCOL,
    AMOEBA_STATIC_SHAPE_PROTOCOL,
    get_shape_protocol,
)


CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates-v2"
COST_SCHEMA = "amoeba-task-shape-cost-v2"
ANALYTICAL_INPUT_SCHEMA = "cgra-ii-amoeba-query-features-v1"
ADAPTER_FEATURE_SCHEMA = "cgra-ii-amoeba-pointwise-features-v1"

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


def load_candidate_manifest(path: Path) -> Dict[str, Any]:
    """Validate the frozen Amoeba JSONL space and return unique cost queries."""
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
        if record.get("schema_version") != CANDIDATE_SCHEMA:
            raise ValueError("candidate manifest schema_version mismatch")
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
    if header.get("search_scope") != "static-shape-only-v2":
        raise ValueError("only static Amoeba candidate manifests are supported")
    if header.get("shape_policy") != "static-rectangles-v2":
        raise ValueError("only rectangular Amoeba candidate manifests are supported")
    if footer.get("candidate_count") != len(candidates):
        raise ValueError("candidate manifest footer count mismatch")
    architecture = _object(header.get("architecture"), "header architecture")
    per_rows = _positive_integer(
        architecture.get("per_cgra_tile_rows"), "per_cgra_tile_rows",
    )
    per_cols = _positive_integer(
        architecture.get("per_cgra_tile_cols"), "per_cgra_tile_cols",
    )
    if (per_rows, per_cols) != (4, 4):
        raise ValueError("model protocol requires 4x4 mapper tiles per physical CGRA")

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

    candidate_queries = set()
    candidate_ids = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate identity is missing")
        if candidate_id in candidate_ids:
            raise ValueError("duplicate candidate identity")
        candidate_ids.add(candidate_id)
        task_shapes = candidate.get("task_shapes")
        if not isinstance(task_shapes, list) or not task_shapes:
            raise ValueError("candidate has no task_shapes")
        for raw_choice in task_shapes:
            choice = _object(raw_choice, "candidate task shape")
            task = choice.get("task")
            shape = _object(choice.get("shape"), "candidate shape")
            if shape.get("kind") != "rect":
                raise ValueError("non-rectangular candidate shape is unsupported")
            physical_rows = _positive_integer(shape.get("rows"), "shape rows")
            physical_cols = _positive_integer(shape.get("cols"), "shape cols")
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
            candidate_queries.add((task, mapper_rows, mapper_cols))
    if candidate_queries != seen:
        raise ValueError("header cost_queries do not match candidate task shapes")
    return {
        "header": dict(header),
        "candidates": [dict(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
        "queries": queries,
        "manifest_sha256": sha256_file(path),
    }


def load_analytical_input(
    path: Path, expected_function: str,
) -> Tuple[Dict[QueryKey, Dict[str, float]], Dict[str, Any]]:
    root = _object(json.loads(path.read_text()), "analytical input")
    if root.get("schema_version") != ANALYTICAL_INPUT_SCHEMA:
        raise ValueError("analytical input schema_version mismatch")
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
        rec_mii = _positive_number(entry.get("rec_mii"), "rec_mii")
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
        "sha256": sha256_file(path),
    }


def load_analytical_hybrid_report(
    path: Path, checkpoint_paths: Mapping[str, Path],
    ensemble: Mapping[str, Any],
) -> Dict[str, Any]:
    """Load one deployment-safe fallback chosen entirely on validation."""
    report = _object(json.loads(path.read_text()), "analytical hybrid report")
    if report.get("schema_version") != "cgra-ii-pointwise-hybrid-analysis-v1":
        raise ValueError("analytical hybrid report schema_version mismatch")
    if report.get("selection_split") != "validation_only":
        raise ValueError("analytical hybrid must be selected on validation only")
    if report.get("manifest_sha256") != ensemble.get("manifest_sha256"):
        raise ValueError("analytical hybrid training manifest mismatch")
    records = _object(report.get("checkpoints"), "hybrid checkpoints")
    if set(records) != set(checkpoint_paths):
        raise ValueError("analytical hybrid checkpoints do not match ensemble")
    for name, checkpoint_path in checkpoint_paths.items():
        record = _object(records[name], f"hybrid checkpoint {name}")
        if record.get("sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"analytical hybrid checkpoint mismatch for {name}")
    hybrid_weights = {
        str(name): float(value) for name, value in _object(
            report.get("ensemble_weights"), "hybrid ensemble weights",
        ).items()
    }
    if set(hybrid_weights) != set(ensemble["weights"]) or any(
        not math.isclose(
            hybrid_weights[name], float(ensemble["weights"][name]),
            abs_tol=1e-9,
        )
        for name in hybrid_weights
    ):
        raise ValueError("analytical hybrid ensemble weights mismatch")
    if not math.isclose(
        float(report.get("uncertainty_exponent")),
        float(ensemble["uncertainty_exponent"]), abs_tol=1e-9,
    ):
        raise ValueError("analytical hybrid uncertainty gate mismatch")

    validation = _object(report.get("validation"), "hybrid validation")
    baseline = float(_object(
        validation.get("ensemble"), "hybrid validation ensemble",
    )["mae"])
    raw_rules = _object(
        validation.get("hybrid_rules"), "hybrid validation rules",
    )
    deployable = []
    allowed_kinds = {
        "rec_gt_res", "lb_le", "prediction_le", "rec_gt_res_or_lb_le",
    }
    for name, raw_record in raw_rules.items():
        record = _object(raw_record, f"hybrid rule {name}")
        rule = dict(_object(record.get("rule"), f"hybrid rule {name} body"))
        if rule.get("kind") not in allowed_kinds:
            raise ValueError("analytical hybrid contains a non-deployable rule")
        hybrid_mae = float(_object(
            record.get("hybrid"), f"hybrid rule {name} metrics",
        )["mae"])
        improvement = float(record.get("mae_improvement_over_ensemble"))
        if not all(math.isfinite(value) for value in (
            baseline, hybrid_mae, improvement,
        )):
            raise ValueError("analytical hybrid metrics must be finite")
        if improvement > 0.0 and hybrid_mae < baseline:
            deployable.append((hybrid_mae, len(str(rule["kind"])), str(name), rule))
    selected_rule = min(deployable)[3] if deployable else None
    return {
        "sha256": sha256_file(path),
        "selected_rule": selected_rule,
        "validation_ensemble_mae": baseline,
        "validation_selected_hybrid_mae": (
            min(row[0] for row in deployable) if deployable else baseline
        ),
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


def _use_analytical_mean(
    rule: Optional[Mapping[str, Any]], facts: Mapping[str, float],
    predicted_ii: float,
) -> bool:
    if rule is None:
        return False
    kind = str(rule["kind"])
    if kind == "rec_gt_res":
        return facts["rec_mii"] > facts["res_mii"]
    if kind == "lb_le":
        return facts["lower_bound"] <= float(rule["threshold"])
    if kind == "prediction_le":
        return predicted_ii <= float(rule["threshold"])
    if kind == "rec_gt_res_or_lb_le":
        return (
            facts["rec_mii"] > facts["res_mii"] or
            facts["lower_bound"] <= float(rule["threshold"])
        )
    raise ValueError("unknown analytical hybrid rule")


def generate_catalog(
    candidate_manifest: Path, analytical_input: Path,
    task_paths: Mapping[str, Path], checkpoint_paths: Mapping[str, Path],
    ensemble_report: Path, device: torch.device,
    ensemble_mode: str = "selected",
    analytical_hybrid_report: Optional[Path] = None,
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
    models: Dict[str, Tuple[JointGraphShapeModel, Model2Config]] = {}
    checkpoint_metadata = {}
    representations = set()
    for name, path in checkpoint_paths.items():
        artifact = torch.load(path, map_location=device, weights_only=False)
        config = Model2Config(**artifact["config"]).validate()
        if config.interaction_mode not in {
            "discrete_pointwise", "residual_pointwise",
            "continuous_residual_pointwise",
        }:
            raise ValueError(f"checkpoint {name} is not pointwise")
        if config.shape_protocol != AMOEBA_STATIC_SHAPE_PROTOCOL:
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
    analytical_hybrid = (
        load_analytical_hybrid_report(
            analytical_hybrid_report, checkpoint_paths, ensemble,
        )
        if analytical_hybrid_report is not None else None
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
        graphs[task] = parse_neura_dfg_representation(
            path.read_text(), representation,
        )
    provenance_task_hashes = _object(
        analytical_provenance.get("task_dfg_sha256"),
        "analytical provenance task DFG hashes",
    )
    if provenance_task_hashes != task_identities:
        raise ValueError("analytical input task DFG SHA-256 mismatch")
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
        if (key[1], key[2]) in AMOEBA_STATIC_PROTOCOL.mapper_shapes
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
        shape: make_cgra_graph(*shape, AMOEBA_STATIC_SHAPE_PROTOCOL)
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
            use_analytical = _use_analytical_mean(
                analytical_hybrid["selected_rule"]
                if analytical_hybrid is not None else None,
                analytical[key], combined["predicted_ii"],
            )
            if use_analytical:
                combined["predicted_ii"] = analytical[key]["lower_bound"]
            entry.update({
                "support_status": "supported",
                "predicted_ii": combined["predicted_ii"],
                "startup_cycles": analytical[key]["startup_cycles"],
                "predicted_ii_std": combined["predicted_ii_std"],
                "mapper_success_probability": (
                    combined["mapper_success_probability"]
                ),
                "analytical_lower_bound": analytical[key]["lower_bound"],
                "ii_mean_source": (
                    "validation_selected_analytical_fallback"
                    if use_analytical else "pointwise_ensemble"
                ),
            })
        entries.append(entry)

    namespace_contract = {
        "feature_schema": ADAPTER_FEATURE_SCHEMA,
        "shape_protocol": get_shape_protocol(
            AMOEBA_STATIC_SHAPE_PROTOCOL
        ).to_dict(),
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "analytical_input_sha256": sha256_file(analytical_input),
        "analytical_provenance": {
            "neura_opt_sha256": analytical_provenance["neura_opt_sha256"],
            "architecture_sha256": analytical_provenance[
                "architecture_sha256"
            ],
            "task_dfg_sha256": task_identities,
            "rec_res_source": analytical_provenance["rec_res_source"],
            "startup_cycles_source": analytical_provenance[
                "startup_cycles_source"
            ],
        },
        "checkpoints": checkpoint_metadata,
        "ensemble_report_sha256": ensemble["sha256"],
        "ensemble_mode": selected_mode,
        "ensemble_weights": ensemble["weights"],
        "uncertainty_exponent": ensemble["uncertainty_exponent"],
        "uncertainty_scales": ensemble["uncertainty_scales"],
        "analytical_hybrid": analytical_hybrid,
    }
    namespace_hash = canonical_json_sha256(namespace_contract)
    catalog = {
        "schema_version": COST_SCHEMA,
        "function": function,
        "namespace": f"cgra-ii-v8-{namespace_hash[:24]}",
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
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--analytical-hybrid-report", type=Path)
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
    checkpoint_paths = parse_checkpoint_paths(args.checkpoint)
    catalog, timing = generate_catalog(
        args.manifest.resolve(), args.analytical_input.resolve(),
        task_paths, checkpoint_paths, args.ensemble_report.resolve(),
        torch.device(args.device), args.ensemble_mode,
        (args.analytical_hybrid_report.resolve()
         if args.analytical_hybrid_report else None),
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
