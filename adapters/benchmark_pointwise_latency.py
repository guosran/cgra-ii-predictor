#!/usr/bin/env python3
"""Benchmark single-candidate pointwise checkpoint latency.

Model loading, manifest parsing, and validation-only ensemble fitting are
excluded.  Each timed call includes graph padding, message passing, soft
operation-to-PE placement, routing-context construction, and output heads for
one independent ``(DFG, CGRA candidate)`` pair.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple


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
    make_cgra_graph,
)
from cgra_ii_predictor.shape_protocol import get_shape_protocol  # noqa: E402
from neura_graph_experiment import (  # noqa: E402
    QueryRecord,
    batch_targets,
    load_terminal_manifest,
    split_queries,
)


CheckpointSpec = Tuple[str, Path]
PreparedCall = Tuple[QueryRecord, List[Any], torch.Tensor]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> CheckpointSpec:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty latency values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(milliseconds: Sequence[float]) -> Dict[str, float]:
    return {
        "call_count": len(milliseconds),
        "mean_ms": statistics.mean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "p90_ms": percentile(milliseconds, 0.90),
        "p95_ms": percentile(milliseconds, 0.95),
        "max_ms": max(milliseconds),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def selected_calls(
    queries: Sequence[QueryRecord], sample_count: int,
) -> List[Tuple[str, int, int]]:
    """Choose deterministic DFG-size quantiles and rotate through shapes."""
    if sample_count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(queries, key=lambda query: (
        len(query.graph.node_types), query.ranking_query_id,
    ))
    count = min(sample_count, len(ordered))
    positions = [
        round(index * (len(ordered) - 1) / max(1, count - 1))
        for index in range(count)
    ]
    result = []
    for sample_index, position in enumerate(positions):
        query = ordered[position]
        candidates = sorted(query.candidates, key=lambda candidate: (
            candidate.rows, candidate.columns, candidate.candidate_id,
        ))
        candidate = candidates[sample_index % len(candidates)]
        result.append((
            query.ranking_query_id, candidate.rows, candidate.columns,
        ))
    return result


def prepare_calls(
    manifest_path: Path, config: Model2Config,
    identities: Sequence[Tuple[str, int, int]], seed: int,
    device: torch.device,
) -> List[PreparedCall]:
    _, queries = load_terminal_manifest(
        manifest_path, config.dfg_representation, config.shape_protocol,
    )
    test_by_id = {
        query.ranking_query_id: query
        for query in split_queries(queries, seed)["test"]
    }
    prepared = []
    for query_id, rows, columns in identities:
        query = test_by_id[query_id]
        candidate = next(
            candidate for candidate in query.candidates
            if candidate.rows == rows and candidate.columns == columns
        )
        point_query = replace(query, candidates=(candidate,))
        context, _, _, _, _ = batch_targets(
            [point_query], config, device,
        )
        prepared.append((
            point_query, [make_cgra_graph(
                rows, columns, config.shape_protocol,
            )], context,
        ))
    return prepared


def forward_call(
    model: JointGraphShapeModel, prepared: PreparedCall,
) -> None:
    query, cgra_graphs, context = prepared
    output = model([query.graph], cgra_graphs, context)
    # Materialize the scalar result without transferring it to the CPU.
    output["predicted_ii"].sum()


def time_calls(
    models: Sequence[JointGraphShapeModel],
    prepared_by_model: Sequence[Sequence[PreparedCall]],
    warmups: int, repetitions: int, device: torch.device,
) -> List[float]:
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    call_count = len(prepared_by_model[0])
    if any(len(prepared) != call_count for prepared in prepared_by_model):
        raise ValueError("checkpoint benchmark populations differ")
    with torch.inference_mode():
        for index in range(warmups):
            position = index % call_count
            for model, prepared in zip(models, prepared_by_model):
                forward_call(model, prepared[position])
        synchronize(device)
        timings = []
        for _ in range(repetitions):
            for position in range(call_count):
                synchronize(device)
                started = time.perf_counter()
                for model, prepared in zip(models, prepared_by_model):
                    forward_call(model, prepared[position])
                synchronize(device)
                timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Select one common candidate population using the first checkpoint's
    # representation, then reconstruct the same IDs for every model.
    first_checkpoint = torch.load(
        args.checkpoint[0][1], map_location="cpu", weights_only=False,
    )
    first_config = Model2Config(**first_checkpoint["config"]).validate()
    _, first_queries = load_terminal_manifest(
        args.manifest, first_config.dfg_representation,
        first_config.shape_protocol,
    )
    identities = selected_calls(
        split_queries(first_queries, args.seed)["test"], args.samples,
    )

    models = []
    prepared_by_model = []
    checkpoint_metadata = []
    for name, path in args.checkpoint:
        artifact = (
            first_checkpoint if path == args.checkpoint[0][1]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        config = Model2Config(**artifact["config"]).validate()
        model = JointGraphShapeModel(config).to(device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        models.append(model)
        prepared_by_model.append(prepare_calls(
            args.manifest, config, identities, args.seed, device,
        ))
        checkpoint_metadata.append({
            "name": name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "dfg_representation": config.dfg_representation,
            "message_passing_layers": config.message_passing_layers,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        })

    individual = {}
    for index, (name, _) in enumerate(args.checkpoint):
        individual[name] = latency_summary(time_calls(
            [models[index]], [prepared_by_model[index]],
            args.warmups, args.repetitions, device,
        ))
    ensemble = latency_summary(time_calls(
        models, prepared_by_model, args.warmups, args.repetitions, device,
    ))
    report = {
        "schema_version": "cgra-ii-pointwise-latency-v1",
        "contract": "one_independent_dfg_cgra_candidate_per_timed_call",
        "timing_scope": (
            "model_forward_including_graph_tensorization_and_routing_context;"
            "excluding_model_load_manifest_parse_and_input_compilation"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
        },
        "threads": args.threads,
        "sample_count": len(identities),
        "repetitions": args.repetitions,
        "checkpoints": checkpoint_metadata,
        "individual": individual,
        "sequential_ensemble": ensemble,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
