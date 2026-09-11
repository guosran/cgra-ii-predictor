#!/usr/bin/env python3
"""Generate an Amoeba task/shape cost catalogue from a mapper surrogate.

The adapter consumes a frozen candidate JSONL manifest, one pre-mapper Neura
DFG per task, analytical lower-bound facts, and frontend-provided startup
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


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
from cgra_ii_predictor.dfg import (  # noqa: E402
    parse_neura_route_expanded_dfg,
)
from cgra_ii_predictor.mapper_model import (  # noqa: E402
    DirectMapperIIModel,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_vector,
)
from cgra_ii_predictor.shape_protocol import (  # noqa: E402
    SHAPE_PROTOCOL,
    SHAPE_PROTOCOL_ID,
    get_shape_protocol,
)


ANALYTICAL_INPUT_SCHEMA = "cgra-ii-amoeba-query-features"
ADAPTER_FEATURE_SCHEMA = "cgra-ii-amoeba-direct-mapper-features"
CHECKPOINT_SCHEMA = "cgra-ii-direct-mapper-model"
FINAL_MODEL_DIR = PROJECT_ROOT / "models" / "final"
DEFAULT_MODEL = FINAL_MODEL_DIR / "mapper.pt"

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


def _parse_trip_count(
    record: Mapping[str, Any], label: str,
) -> Optional[int]:
    """Parse the mutually exclusive static and symbol-dynamic encodings.

    A symbolic trip count is intentionally represented as ``None``.  The
    predictor can still build a shape cost catalogue because inference only
    needs the task DFG and analytical shape facts; a later duration scorer must
    bind the runtime value or reject ranking rather than treating it as one.
    """
    has_numeric = "trip_count" in record
    has_kind = "trip_count_kind" in record
    if has_numeric == has_kind:
        raise ValueError(
            f"{label} must contain exactly one of trip_count or "
            "trip_count_kind"
        )
    if has_numeric:
        return _positive_integer(record.get("trip_count"), f"{label} trip_count")

    kind = record.get("trip_count_kind")
    if not isinstance(kind, str):
        raise ValueError(f"{label} trip_count_kind must be a string")
    if kind != "symbol_dynamic":
        raise ValueError(f"unsupported {label} trip_count_kind {kind!r}")
    return None


def load_candidate_manifest(path: Path) -> Dict[str, Any]:
    """Validate a frozen static-shape manifest and derive its used queries.

    Candidate enumeration, canonical ordering, and exact concurrent packing
    belong to Amoeba's C++ manifest reader.  This adapter deliberately does not
    reconstruct that space.  It validates only the JSONL record order/schema,
    task facts, candidate IDs, redundant shape fields, and that ``cost_queries``
    is exactly the set referenced by the frozen candidates.
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
    footer_count = _positive_integer(
        footer.get("candidate_count"), "candidate manifest candidate_count",
    )
    if footer_count != len(candidates):
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
        trip_count = _parse_trip_count(task, "task fact")
        task_facts.append((
            name,
            _sha256_string(task.get("body_sha256"), "task body_sha256"),
            trip_count,
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

    for task, _, _ in queries:
        if task not in task_names:
            raise ValueError("cost query names a task outside header task facts")

    candidate_queries = set()
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
        for task_index, raw_choice in enumerate(task_shapes):
            choice = _object(raw_choice, "candidate task shape")
            task = choice.get("task")
            expected_task, _, expected_trip_count = task_facts[task_index]
            trip_count = _parse_trip_count(choice, "candidate task shape")
            if task != expected_task or trip_count != expected_trip_count:
                raise ValueError("candidate task facts do not match header")
            shape = _object(choice.get("shape"), "candidate shape")
            if shape.get("kind") != "rect":
                raise ValueError("non-rectangular candidate shape is unsupported")
            physical_rows = _positive_integer(shape.get("rows"), "shape rows")
            physical_cols = _positive_integer(shape.get("cols"), "shape cols")
            if physical_rows > grid_rows or physical_cols > grid_cols:
                raise ValueError(
                    "candidate shape exceeds the manifest architecture grid"
                )
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
            candidate_queries.add((task, mapper_rows, mapper_cols))
    if candidate_queries != seen:
        raise ValueError("header cost_queries do not match candidate task shapes")
    return {
        "header": dict(header),
        "candidates": [dict(candidate) for candidate in candidates],
        "candidate_count": len(candidates),
        "queries": queries,
        "task_facts": [
            {
                "task": task,
                "body_sha256": body_sha,
                **(
                    {"trip_count": trip_count}
                    if trip_count is not None
                    else {"trip_count_kind": "symbol_dynamic"}
                ),
            }
            for task, body_sha, trip_count in task_facts
        ],
        "task_trip_counts": {
            task: trip_count for task, _, trip_count in task_facts
        },
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


def load_mapper_model(
    path: Path, device: torch.device,
) -> Tuple[DirectMapperIIModel, MapperModelConfig, Dict[str, Any]]:
    """Strictly load the one fixed-heuristic mapper surrogate."""
    artifact = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(artifact, Mapping) or artifact.get("schema") != (
        CHECKPOINT_SCHEMA
    ):
        raise ValueError("mapper checkpoint has an unsupported schema")
    if artifact.get("feature_names") != list(MAPPER_FEATURE_NAMES):
        raise ValueError("mapper checkpoint feature contract mismatch")
    raw_config = _object(artifact.get("config"), "mapper model config")
    config = MapperModelConfig(**raw_config).validate()
    model = DirectMapperIIModel(config).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    architecture = _sha256_string(
        artifact.get("architecture_sha256"), "training architecture SHA-256",
    )
    return model, config, {
        "sha256": sha256_file(path),
        "training_manifest_sha256": _sha256_string(
            artifact.get("training_manifest_sha256"),
            "training manifest SHA-256",
        ),
        "training_architecture_sha256": architecture,
        "supported_architecture_sha256": [architecture],
        "compatibility_rule": "exact_architecture_sha256",
        "config_sha256": canonical_json_sha256(config.to_dict()),
    }


def validate_model_architecture(
    model_metadata: Mapping[str, Any], architecture_sha256: object,
) -> None:
    supported = model_metadata["supported_architecture_sha256"]
    if architecture_sha256 not in supported:
        raise ValueError(
            "analytical architecture is outside the deployed model contract"
        )


def generate_catalog(
    candidate_manifest: Path, analytical_input: Path,
    task_paths: Mapping[str, Path], model_path: Path, device: torch.device,
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
    if analytical_provenance.get("architecture_sha256") != (
        manifest["architecture_sha256"]
    ):
        raise ValueError("analytical input architecture does not match manifest")
    tasks = sorted({task for task, _, _ in manifest["queries"]})
    if set(task_paths) != set(tasks):
        raise ValueError("task DFG mapping must exactly cover manifest tasks")

    load_started = time.perf_counter()
    model, config, model_metadata = load_mapper_model(model_path, device)
    architecture_sha256 = analytical_provenance.get("architecture_sha256")
    validate_model_architecture(model_metadata, architecture_sha256)
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
        graphs[task] = parse_neura_route_expanded_dfg(dfg_text)
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
        "analytical_lower_bound_source", "startup_cycles_source",
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

    inference_started = time.perf_counter()
    predictions: Dict[QueryKey, float] = {}
    with torch.inference_mode():
        for offset in range(0, len(supported_queries), 4096):
            keys = supported_queries[offset:offset + 4096]
            features = torch.tensor([
                mapper_feature_vector(
                    graphs[task], rows, cols,
                    analytical[key]["rec_mii"],
                    analytical[key]["res_mii"],
                    analytical[key]["lower_bound"],
                    mapper_ii_ceiling=config.mapper_ii_ceiling,
                    shape_protocol=config.shape_protocol,
                )
                for key in keys for task, rows, cols in (key,)
            ], dtype=torch.float32, device=device)
            lower_bounds = torch.tensor([
                analytical[key]["lower_bound"] for key in keys
            ], dtype=torch.float32, device=device)
            values = model(features, lower_bounds).detach().cpu().tolist()
            predictions.update(zip(keys, map(float, values)))
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
            entry.update({
                "support_status": "supported",
                "predicted_ii": predictions[key],
                "startup_cycles": analytical[key]["startup_cycles"],
                "analytical_lower_bound": analytical[key]["lower_bound"],
                "ii_mean_source": "direct_mapper_surrogate",
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
            "analytical_lower_bound_source": analytical_provenance[
                "analytical_lower_bound_source"
            ],
            "startup_cycles_source": analytical_provenance[
                "startup_cycles_source"
            ],
        },
        "model": model_metadata,
        "architecture_contract": model_metadata,
        "ranking_policy": {
            "objective": "predicted_compute_bottleneck",
            "mapper_success_probability": "not_predicted",
            "uses_mapper_success_probability": False,
        },
    }
    namespace_hash = canonical_json_sha256(namespace_contract)
    catalog = {
        "schema": COST_SCHEMA,
        "function": function,
        "namespace": f"cgra-ii-direct-mapper-{namespace_hash[:24]}",
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
        "model_load_count": 1,
        "model_forward_pass_count": math.ceil(len(supported_queries) / 4096),
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
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_paths = parse_task_paths(args.task_dfg)
    catalog, timing = generate_catalog(
        args.manifest.resolve(), args.analytical_input.resolve(),
        task_paths, args.model.resolve(), torch.device(args.device),
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
