#!/usr/bin/env python3
"""Freeze Model 2 on motif-v6, or evaluate it once on held-out motif-v7.

``freeze`` consumes only the already-disclosed motif-v6 development labels and
uses the configuration/epoch selected by the recorded development experiment.
``evaluate`` loads that frozen artifact read-only and never constructs an
optimizer, so motif-v7 labels cannot alter the model.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

try:
    from adapters import neura_motifs_v6, neura_motifs_v7
    from adapters.neura_graph_experiment import (
        QueryRecord,
        evaluate_by_family,
        evaluate_model,
        load_terminal_manifest,
        sha256_file,
        train_fixed_epochs,
    )
except ImportError:  # Running from the adapters directory.
    import neura_motifs_v6  # type: ignore
    import neura_motifs_v7  # type: ignore
    from neura_graph_experiment import (  # type: ignore
        QueryRecord,
        evaluate_by_family,
        evaluate_model,
        load_terminal_manifest,
        sha256_file,
        train_fixed_epochs,
    )

from cgra_ii_predictor.graph_model import (
    CANDIDATE_CONTEXT_NAMES,
    JointGraphShapeModel,
    Model2Config,
    make_cgra_graph,
)


ARTIFACT_SCHEMA_VERSION = "cgra-ii-frozen-joint-graph-model-v1"
FREEZE_REPORT_SCHEMA_VERSION = "cgra-ii-model2-freeze-report-v1"
EVALUATION_REPORT_SCHEMA_VERSION = "cgra-ii-model2-v7-evaluation-v1"


def _json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_strings(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_config() -> Model2Config:
    declaration = neura_motifs_v7.MODEL2_CONFIG
    return Model2Config(
        hidden_dimension=int(declaration["hidden_dimension"]),
        message_passing_layers=int(declaration["message_passing_layers"]),
        dropout=float(declaration["dropout"]),
        mapper_ii_ceiling=float(declaration["mapper_ii_ceiling"]),
        listwise_temperature=float(declaration["listwise_temperature"]),
        success_loss_weight=float(declaration["success_loss_weight"]),
        residual_loss_weight=float(declaration["residual_loss_weight"]),
        listwise_loss_weight=float(declaration["listwise_loss_weight"]),
        strict_tiebreak_loss_weight=float(
            declaration["strict_tiebreak_loss_weight"]
        ),
    ).validate()


def _shape_graphs():
    return [
        make_cgra_graph(rows, columns)
        for rows in range(1, 5) for columns in range(1, 5)
    ]


def _family_counts(queries: Sequence[QueryRecord]) -> Dict[str, int]:
    return dict(sorted(Counter(
        query.generator_family for query in queries
    ).items()))


def _validate_development_selection(
    report: Mapping[str, Any], manifest_sha256: str, config: Model2Config,
) -> None:
    if report.get("status") != "exploratory_v6_labels_already_disclosed":
        raise ValueError("development report is not the disclosed v6 experiment")
    if report.get("frozen_blind_claim") is not False:
        raise ValueError("development report must not claim a blind result")
    if report.get("manifest", {}).get("sha256") != manifest_sha256:
        raise ValueError("development report used a different v6 manifest")
    if report.get("model", {}).get("config") != config.to_dict():
        raise ValueError("development report does not match the frozen config")
    selected_epoch = int(neura_motifs_v7.MODEL2_CONFIG["development_best_epoch"])
    if report.get("training", {}).get("best_epoch") != selected_epoch:
        raise ValueError("development report does not select the frozen epoch")
    if report.get("development_gates_passed") is not True:
        raise ValueError("development experiment did not pass its declared gates")


def freeze(args: argparse.Namespace) -> int:
    model_path = args.output_dir / "model.pt"
    report_path = args.output_dir / "freeze-report.json"
    if model_path.exists() or report_path.exists():
        raise ValueError("refusing to overwrite an existing frozen artifact")
    manifest, queries = load_terminal_manifest(args.training_manifest)
    generator = manifest.get("generator", {})
    if generator.get("version") != neura_motifs_v6.GENERATOR_VERSION:
        raise ValueError("frozen training manifest must be motif-v6")
    manifest_sha = sha256_file(args.training_manifest.resolve())
    development = _json(args.development_report.resolve())
    config = frozen_config()
    _validate_development_selection(development, manifest_sha, config)
    declared = neura_motifs_v7.MODEL2_CONFIG
    epochs = int(declared["frozen_refit_epochs_on_all_v6_queries"])
    batch_size = int(declared["batch_size_queries"])
    seed = int(declared["development_split_seed"])
    model, training = train_fixed_epochs(
        queries, config, epochs=epochs, batch_size=batch_size,
        learning_rate=float(declared["learning_rate"]),
        weight_decay=float(declared["weight_decay"]), seed=seed,
        threads=args.threads,
    )
    query_ids = [query.ranking_query_id for query in queries]
    protocol_path = (PROJECT_ROOT / "protocols/motif-v7.json").resolve()
    if not protocol_path.is_file():
        raise ValueError("motif-v7 protocol declaration is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "frozen_before_motif_v7_mapper_labels",
        "model_class": neura_motifs_v7.MODEL2_CONFIG["class"],
        "config": config.to_dict(),
        "candidate_context_names": list(CANDIDATE_CONTEXT_NAMES),
        "state_dict": model.state_dict(),
        "training": {
            "generator_version": neura_motifs_v6.GENERATOR_VERSION,
            "manifest_sha256": manifest_sha,
            "manifest_schema_version": manifest.get("schema_version"),
            "query_count": len(queries),
            "candidate_count": sum(len(query.candidates) for query in queries),
            "query_ids": sorted(query_ids),
            "query_ids_sha256": _sha256_strings(query_ids),
            "family_query_counts": _family_counts(queries),
            "labels": "terminal_motif_v6_only",
        },
        "freeze_contract": {
            "protocol_sha256": sha256_file(protocol_path),
            "development_report_sha256": sha256_file(
                args.development_report.resolve()
            ),
            "development_best_epoch": int(
                declared["development_best_epoch"]
            ),
            "refit_epochs": epochs,
            "motif_v7_labels_permitted": False,
        },
    }
    torch.save(artifact, model_path)
    device = torch.device("cpu")
    refit_diagnostic = evaluate_model(
        model, queries, _shape_graphs(), config, batch_size, device,
    )
    report = {
        "schema_version": FREEZE_REPORT_SCHEMA_VERSION,
        "status": "frozen_before_motif_v7_mapper_labels",
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "config": config.to_dict(),
        },
        "training_manifest": {
            "path": str(args.training_manifest.resolve()),
            "sha256": manifest_sha,
            "query_count": len(queries),
            "candidate_count": sum(len(query.candidates) for query in queries),
            "family_query_counts": _family_counts(queries),
        },
        "development_selection": {
            "path": str(args.development_report.resolve()),
            "sha256": sha256_file(args.development_report.resolve()),
            "selected_epoch": int(declared["development_best_epoch"]),
        },
        "training": training,
        "all_v6_refit_diagnostic_not_a_holdout": refit_diagnostic,
        "held_out_motif_v7_labels_accessed": False,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"frozen_model={model_path.resolve()} sha256={sha256_file(model_path)} "
        f"training_queries={len(queries)} held_out_labels_accessed=false",
        flush=True,
    )
    return 0


def _load_frozen_model(
    path: Path,
) -> Tuple[Dict[str, Any], JointGraphShapeModel]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("frozen model artifact must be a dictionary")
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("frozen model artifact schema mismatch")
    if artifact.get("status") != "frozen_before_motif_v7_mapper_labels":
        raise ValueError("model was not frozen for motif-v7")
    config = frozen_config()
    if artifact.get("config") != config.to_dict():
        raise ValueError("frozen model configuration changed")
    if artifact.get("candidate_context_names") != list(CANDIDATE_CONTEXT_NAMES):
        raise ValueError("candidate context contract changed")
    if artifact.get("freeze_contract", {}).get(
        "motif_v7_labels_permitted"
    ) is not False:
        raise ValueError("artifact does not forbid motif-v7 training labels")
    model = JointGraphShapeModel(config)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return artifact, model


def _comparison_gates(evaluation: Mapping[str, Any]) -> Dict[str, bool]:
    baseline = evaluation["analytical_top1"]
    model = evaluation["model2_top1"]
    if not all(
        record.get(key) is not None
        for record in (baseline, model)
        for key in (
            "strict_top1_accuracy", "optimal_ii_rate",
            "selected_success_rate", "mean_timeout_penalized_regret",
        )
    ):
        return {
            "strict_top1_improvement": False,
            "optimal_ii_rate_non_degradation": False,
            "selected_success_non_degradation": False,
            "timeout_penalized_regret_improvement": False,
        }
    return {
        "strict_top1_improvement": (
            float(model["strict_top1_accuracy"]) >
            float(baseline["strict_top1_accuracy"])
        ),
        "optimal_ii_rate_non_degradation": (
            float(model["optimal_ii_rate"]) >=
            float(baseline["optimal_ii_rate"])
        ),
        "selected_success_non_degradation": (
            float(model["selected_success_rate"]) >=
            float(baseline["selected_success_rate"])
        ),
        "timeout_penalized_regret_improvement": (
            float(model["mean_timeout_penalized_regret"]) <
            float(baseline["mean_timeout_penalized_regret"])
        ),
    }


def evaluate(args: argparse.Namespace) -> int:
    manifest, queries = load_terminal_manifest(args.manifest)
    generator = manifest.get("generator", {})
    if generator.get("version") != neura_motifs_v7.GENERATOR_VERSION:
        raise ValueError("held-out evaluation manifest must be motif-v7")
    artifact, model = _load_frozen_model(args.model.resolve())
    training_ids = set(artifact.get("training", {}).get("query_ids", ()))
    evaluation_ids = {query.ranking_query_id for query in queries}
    overlap = training_ids & evaluation_ids
    if overlap:
        raise ValueError("motif-v7 evaluation overlaps frozen v6 training DFGs")
    config = frozen_config()
    batch_size = int(neura_motifs_v7.MODEL2_CONFIG["batch_size_queries"])
    device = torch.device("cpu")
    shape_graphs = _shape_graphs()
    overall = evaluate_model(
        model, queries, shape_graphs, config, batch_size, device,
    )
    by_family = evaluate_by_family(
        model, queries, shape_graphs, config, batch_size, device,
    )
    coverage_policy = neura_motifs_v7.ACCEPTANCE_POLICY["coverage"]
    minimum_analyzed = int(coverage_policy["minimum_analyzed_bases_per_family"])
    minimum_ranking = int(
        coverage_policy["minimum_ranking_eligible_bases_per_family"]
    )
    family_coverage: Dict[str, Dict[str, Any]] = {}
    for family in sorted(by_family):
        family_queries = [
            query for query in queries if query.generator_family == family
        ]
        eligible = sum(
            sum(candidate.status == "success" for candidate in query.candidates)
            >= int(coverage_policy[
                "minimum_successful_candidates_per_ranking_query"
            ])
            for query in family_queries
        )
        family_coverage[family] = {
            "analyzed_query_count": len(family_queries),
            "ranking_eligible_query_count": eligible,
            "analyzed_gate_passed": len(family_queries) >= minimum_analyzed,
            "ranking_gate_passed": eligible >= minimum_ranking,
        }
    comparison = _comparison_gates(overall)
    per_family_optimal = all(
        _comparison_gates(record)["optimal_ii_rate_non_degradation"]
        for record in by_family.values()
    )
    per_family_success = all(
        _comparison_gates(record)["selected_success_non_degradation"]
        for record in by_family.values()
    )
    gates = {
        **comparison,
        "every_family_optimal_ii_rate_non_degradation": per_family_optimal,
        "every_family_selected_success_non_degradation": per_family_success,
        "coverage": all(
            record["analyzed_gate_passed"] and record["ranking_gate_passed"]
            for record in family_coverage.values()
        ) and len(family_coverage) == len(neura_motifs_v7.DEFAULT_MOTIFS),
        "training_evaluation_dfg_disjoint": not overlap,
    }
    accepted = all(gates.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation-report.json"
    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "status": "accepted" if accepted else "rejected",
        "primary_metric": "strict_top1_accuracy",
        "frozen_model": {
            "path": str(args.model.resolve()),
            "sha256": sha256_file(args.model.resolve()),
            "training_manifest_sha256": artifact["training"]["manifest_sha256"],
            "training_generator_version": artifact["training"][
                "generator_version"
            ],
        },
        "held_out_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
            "generator_version": generator.get("version"),
            "query_count": len(queries),
            "candidate_count": sum(len(query.candidates) for query in queries),
        },
        "evaluation": overall,
        "evaluation_by_generator_family": by_family,
        "coverage_by_generator_family": family_coverage,
        "acceptance_gates": gates,
        "acceptance_gates_passed": accepted,
        "motif_v7_labels_used_for_training_or_model_selection": False,
        "trusted_external_timestamp": False,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"model2_v7_report={report_path.resolve()} status={report['status']} "
        f"top1={overall['model2_top1']['strict_top1_accuracy']:.4f} "
        f"baseline_top1={overall['analytical_top1']['strict_top1_accuracy']:.4f}",
        flush=True,
    )
    return 0 if accepted else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--training-manifest", required=True, type=Path)
    freeze_parser.add_argument("--development-report", required=True, type=Path)
    freeze_parser.add_argument("--output-dir", required=True, type=Path)
    freeze_parser.add_argument("--threads", type=int, default=6)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--model", required=True, type=Path)
    evaluate_parser.add_argument("--manifest", required=True, type=Path)
    evaluate_parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        return freeze(args)
    if args.command == "evaluate":
        return evaluate(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
