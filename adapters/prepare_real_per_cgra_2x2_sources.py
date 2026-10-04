#!/usr/bin/env python3
"""Prepare a label-free, test-only MachSuite roster for fresh 2x2 queries.

The source lowering phase reuses the pinned auxiliary importer command path
but compiles only its bounded MachSuite roster.  Admission is a separate
phase and requires the completed route-expanded ORBIT identity audit.
Neither phase invokes the mapper.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _directory in (PROJECT_ROOT / "adapters", PROJECT_ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from mapping_artifact_protocol import (  # noqa: E402
    canonical_json_sha256,
    mapper_input_identity as current_mapper_input_identity,
    sha256_file,
)


ORIGINAL_PROJECT = Path("/home/x/shiran/project/cgra-ii-predictor")
DEFAULT_SOURCE_ROOT = ORIGINAL_PROJECT / "third_party" / "machsuite"
DEFAULT_IMPORTER = ORIGINAL_PROJECT / "adapters" / "import_aux_c_kernels.py"
DEFAULT_DATA_ROOT = Path(
    "/home/x/shiran/data/cgra-per-cgra-2x2-improvement-20261004/"
    "real-source-preparation"
)
DEFAULT_BASE_ROOT = Path(
    "/home/x/shiran/data/cgra-per-cgra-2x2-20261004"
)
DEFAULT_CANONICAL_MAP = Path(
    "/home/x/shiran/data/cgra-per-cgra-2x2-improvement-20261004/"
    "orbit-benchmark-identity-audit/identity-map.json"
)
DEFAULT_ARCHITECTURE = (
    DEFAULT_BASE_ROOT / "native-collection-2x2-v1" / "inputs" / "architecture.yaml"
)
DEFAULT_NEURA_OPT = Path(
    "/home/x/shiran/project/neura/build/tools/mlir-neura-opt/mlir-neura-opt"
)
EXPECTED_MACH_REVISION = "6236e593012cb86b0d2f08d9fb9ba0411ff989b4"
EXPECTED_ARCHITECTURE_SHA256 = (
    "6f4a9a1815dcc0d97c00fd6ee20424fa9420ace90654da29e70d283cba7a611f"
)
EXPECTED_NEURA_OPT_SHA256 = (
    "b837fa5922f099c0aec1776178e2704e20772af3a9fd849ccbf62df6196e5ce7"
)
CANONICAL_ID_MAP_SCHEMA = "orbit-final-benchmark-route-expanded-identity-audit-v1"
SOURCE_SCHEMA = "cgra-ii-nine-shape-outcomes-v1"
SOURCE_GROUP_SCHEMA = "cgra-ii-per-cgra-2x2-real-test-source-groups-v1"
ADMISSION_SCHEMA = "cgra-ii-per-cgra-2x2-real-test-admission-v1"
PREPARATION_SCHEMA = "cgra-ii-per-cgra-2x2-real-source-preparation-v1"
SHAPES_2X2 = (
    (2, 2), (2, 4), (4, 2), (2, 6),
    (6, 2), (2, 8), (8, 2), (4, 4),
)
MAX_NEW_QUERY_BUDGET = 256
MAX_NEW_IDENTITY_BUDGET = 32
REQUIRED_MACH_CASES = {
    "gemm/blocked", "gemm/ncubed", "spmv/crs", "spmv/ellpack",
    "stencil/stencil2d", "stencil/stencil3d",
}


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read {}: {}".format(label, path)) from error
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object: {}".format(label, path))
    return value


def _load_importer(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise ValueError("pinned source importer does not exist: {}".format(path))
    adapters = str(path.parent)
    sources = str(path.parent.parent / "src")
    for directory in (adapters, sources):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(
        "pinned_import_aux_c_kernels", path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned source importer: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lower_case(
    importer: Any,
    source_root: Path,
    row: Mapping[str, Any],
    case_directory: Path,
    clang: Path,
    clangxx: Path,
    llvm_extract: Path,
    mlir_translate: Path,
    neura_opt: Path,
    timeout: int,
) -> Dict[str, Any]:
    source = Path(str(row["source_path"])).resolve()
    result: Dict[str, Any] = {
        "case_id": row["case_id"],
        "top": row["top"],
        "source_relative_path": row["source_relative_path"],
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_program_families": list(row["source_program_families"]),
        "status": "rejected",
        "semantic_equivalence_verified": False,
        "native_mapping_started": False,
    }
    case_directory.mkdir(parents=True)
    full_ir = case_directory / "full.ll"
    kernel_ir = case_directory / "kernel.ll"
    imported = case_directory / "imported.mlir"
    lowered = case_directory / "route-expanded.mlir"
    stage = "compile"
    try:
        compiler = clangxx if source.suffix == ".cpp" else clang
        standard = "-std=c++17" if source.suffix == ".cpp" else "-std=c11"
        command = [
            str(compiler), "-S", "-emit-llvm", "-O3", "-fno-vectorize",
            "-fno-slp-vectorize", "-fno-unroll-loops", "-Xclang",
            "-disable-lifetime-markers", standard, "-I", str(source.parent),
            "-I", str(source_root / "common"), "-o", str(full_ir), str(source),
        ]
        importer._run(command, stage, full_ir, timeout)
        stage = "extract"
        importer._run([
            str(llvm_extract), "--recursive", "--func={}".format(row["top"]),
            str(full_ir), "-S", "-o", str(kernel_ir),
        ], stage, kernel_ir, timeout)
        stage = "llvm_mlir_import"
        importer._run([
            str(mlir_translate), "--import-llvm", str(kernel_ir),
            "-o", str(imported),
        ], stage, imported, timeout)
        stage = "neura_lower"
        importer._run([
            str(neura_opt), str(imported), "--assign-accelerator",
            "--lower-llvm-to-neura", "--promote-input-arg-to-const",
            "--fold-constant", "--canonicalize-return", "--canonicalize-live-in",
            "--leverage-predicated-value", "--transform-ctrl-to-data-flow",
            "--fold-constant", "--insert-data-mov", "-o", str(lowered),
        ], stage, lowered, timeout)
        stage = "route_dfg_validation"
        text = lowered.read_text()
        importer.require_neura_route_expanded_dfg(text)
        legacy_identity = importer.mapper_input_identity(
            text, normalize_static_shapes=True,
        )
        identity = current_mapper_input_identity(
            text, normalize_static_shapes=True,
        )
        if identity != legacy_identity:
            raise ValueError(
                "pinned importer and 2x2 collector mapper identities disagree"
            )
        result.update({
            "status": "lowered",
            "top": row["top"],
            "dfg_relative_path": str(lowered.relative_to(case_directory.parent.parent)),
            "dfg_sha256": sha256_file(lowered),
            "mapper_input_identity": identity,
            "model_visible_graph_identity": importer._visible_identity(text),
            "leakage_lineage_id": canonical_json_sha256({
                "suite": "machsuite",
                "upstream_commit": EXPECTED_MACH_REVISION,
                "source_relative_path": row["source_relative_path"],
            }),
            "native_mapping_invoked": False,
        })
    except Exception as error:
        result.update({
            "failure_stage": stage,
            "reject_reason": "{}: {}".format(type(error).__name__, str(error)[-1800:]),
        })
    audit_path = case_directory / "audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def lower_sources(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise ValueError("source preparation output already exists: {}".format(output_root))
    source_root = args.source_root.resolve()
    importer = _load_importer(args.importer)
    revision = importer._revision(source_root, "machsuite")
    if revision != EXPECTED_MACH_REVISION:
        raise ValueError("MachSuite source revision is not the frozen revision")
    source_rows = importer._inventory("machsuite", source_root)
    if len(source_rows) > 32 or not REQUIRED_MACH_CASES.issubset({
        row["case_id"] for row in source_rows
    }):
        raise ValueError("pinned MachSuite roster is outside the requested source scope")
    output_root.mkdir(parents=True)
    work_root = output_root / "lowering"
    work_root.mkdir()
    audits = []
    for ordinal, row in enumerate(source_rows):
        result = _lower_case(
            importer, source_root, row,
            work_root / "case_{:03d}".format(ordinal),
            args.clang, args.clangxx, args.llvm_extract, args.mlir_translate,
            args.neura_opt, args.stage_timeout,
        )
        audits.append(result)
        print(json.dumps({
            "event": "source_lowering_finished",
            "case_id": row["case_id"],
            "status": result["status"],
            "failure_stage": result.get("failure_stage"),
            "mapper_input_identity": result.get("mapper_input_identity"),
        }), flush=True)
    lowered = [row for row in audits if row["status"] == "lowered"]
    identity_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in lowered:
        identity_groups[row["mapper_input_identity"]].append(row)
    visible_groups: Dict[str, List[str]] = defaultdict(list)
    for identity, rows in identity_groups.items():
        visible = {row["model_visible_graph_identity"] for row in rows}
        if len(visible) != 1:
            raise ValueError("same mapper identity has conflicting visible graphs")
        visible_groups[next(iter(visible))].append(identity)
    manifest = {
        "schema": PREPARATION_SCHEMA,
        "phase": "lowering_only_admission_not_frozen",
        "source_root": str(source_root),
        "upstream_commit": revision,
        "pinned_importer_path": str(args.importer.resolve()),
        "pinned_importer_sha256": sha256_file(args.importer),
        "source_roster_count": len(source_rows),
        "lowered_source_case_count": len(lowered),
        "rejected_source_case_count": len(audits) - len(lowered),
        "unique_mapper_input_identity_count": len(identity_groups),
        "unique_visible_graph_identity_count": len(visible_groups),
        "projected_2x2_query_count_before_admission": (
            len(identity_groups) * len(SHAPES_2X2)
        ),
        "shape_roster": [list(shape) for shape in SHAPES_2X2],
        "max_concurrent_tool_processes": 1,
        "native_mapping_invoked": False,
        "mapping_labels_created": False,
        "reduction_case_available_in_pinned_machsuite_roster": False,
        "required_cases": sorted(REQUIRED_MACH_CASES),
        "case_audit": audits,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_root / "lowering-audit.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_root / "lowering-command.json").write_text(json.dumps({
        "command": [
            "python3", str(Path(__file__).resolve()), "--lower-only",
            "--source-root", str(source_root),
            "--importer", str(args.importer.resolve()),
            "--output-dir", str(output_root),
            "--architecture", str(args.architecture.resolve()),
            "--neura-opt", str(args.neura_opt.resolve()),
            "--stage-timeout", str(args.stage_timeout),
        ],
        "max_concurrent_tool_processes": 1,
        "native_mapping_invoked": False,
    }, indent=2, sort_keys=True) + "\n")
    return manifest


def _canonical_identity_sets(path: Path) -> Tuple[set[str], set[str], Dict[str, List[str]]]:
    document = _read_json(path, "completed ORBIT benchmark identity audit")
    if document.get("schema") != CANONICAL_ID_MAP_SCHEMA:
        raise ValueError(
            "canonical benchmark exclusions require the completed route-expanded audit"
        )
    comparison = document.get("comparison")
    source = document.get("source")
    tasks = document.get("tasks")
    if not isinstance(comparison, dict) or not isinstance(source, dict) or not isinstance(tasks, list):
        raise ValueError("completed canonical audit lacks comparison/source/task records")
    if (comparison.get("final_task_count") != 118 or
            comparison.get("final_route_expanded_unique_mapper_input_identity_count") != 109 or
            comparison.get("final_route_expanded_unique_model_visible_graph_identity_count") != 84 or
            comparison.get("overlap_unique_mapper_input_identity_count") != 0 or
            comparison.get("overlap_unique_visible_graph_identity_count") != 0 or
            comparison.get("mapper_input_overlaps") != [] or
            comparison.get("visible_graph_overlaps") != [] or
            comparison.get("training_unique_mapper_input_identity_count") != 323 or
            comparison.get("training_architecture_sha256") != EXPECTED_ARCHITECTURE_SHA256):
        raise ValueError("canonical audit summary does not match the frozen 2x2 comparison")
    if (source.get("neura_opt_sha256") != EXPECTED_NEURA_OPT_SHA256 or
            source.get("neura_opt_hash_matches_expected") is not True):
        raise ValueError("canonical audit uses an unexpected Neura binary")
    if len(tasks) != 118:
        raise ValueError("canonical audit does not contain all 118 final tasks")
    identity_set: set[str] = set()
    visible_set: set[str] = set()
    names: Dict[str, List[str]] = defaultdict(list)
    for task in tasks:
        identity = task.get("mapper_input_identity")
        visible = task.get("model_visible_graph_identity")
        if (not isinstance(identity, str) or len(identity) != 64 or
                not isinstance(visible, str) or len(visible) != 64 or
                task.get("route_expanded_validation") != "passed" or
                task.get("insert_data_mov_exit_code") != 0):
            raise ValueError("canonical route-expanded task record is incomplete")
        program = task.get("program")
        task_name = task.get("task")
        if not isinstance(program, str) or not isinstance(task_name, str):
            raise ValueError("canonical task has no stable source name")
        if (not isinstance(identity, str) or len(identity) != 64 or
                not isinstance(visible, str) or len(visible) != 64):
            raise ValueError("canonical route-expanded identities are malformed")
        identity_set.add(identity)
        visible_set.add(visible)
        names[identity].append("{}/{}".format(program, task_name))
    if len(identity_set) != 109 or len(visible_set) != 84:
        raise ValueError("canonical audit identity counts do not match the frozen roster")
    return identity_set, visible_set, names


def _load_existing_training_metadata(
    base_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, set[str]]]:
    manifest_path = base_root / "source-corpus" / "manifest.json"
    groups_path = base_root / "source-corpus" / "groups.json"
    exclusions_path = base_root / "training-exclusions.json"
    manifest = _read_json(manifest_path, "existing 2x2 source corpus")
    groups = _read_json(groups_path, "existing 2x2 source groups")
    exclusions = _read_json(exclusions_path, "existing source exclusions")
    manifest_sha = sha256_file(manifest_path)
    groups_sha = sha256_file(groups_path)
    if (manifest.get("schema") != "cgra-ii-nine-shape-outcomes-v1" or
            exclusions.get("source_manifest_sha256") != manifest_sha or
            exclusions.get("source_groups_sha256") != groups_sha):
        raise ValueError("existing corpus/exclusion provenance hashes do not agree")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != manifest.get("candidate_count"):
        raise ValueError("existing source corpus candidate roster is malformed")
    sets: Dict[str, set[str]] = {
        name: set() for name in (
            "mapper_input_identity", "model_visible_graph_identity",
            "source_sha256", "leakage_lineage_id", "source_name",
            "source_program_family", "candidate_id",
        )
    }
    for row in candidates:
        for field in (
            "mapper_input_identity", "model_visible_graph_identity",
            "source_sha256", "leakage_lineage_id", "candidate_id",
        ):
            value = row.get(field)
            if isinstance(value, str) and value:
                sets[field].add(value)
        for value in row.get("source_names", []):
            if isinstance(value, str) and value:
                sets["source_name"].add(value)
        for value in row.get("source_program_families", []):
            if isinstance(value, str) and value:
                sets["source_program_family"].add(value)
    excluded = exclusions.get("excluded_mapper_input_identities")
    reasons = exclusions.get("reasons_by_dfg")
    if (not isinstance(excluded, list) or not isinstance(reasons, dict) or
            not set(excluded).issubset(sets["mapper_input_identity"])):
        raise ValueError("existing candidate exclusions do not match the source roster")
    groups_list = groups.get("groups")
    if not isinstance(groups_list, list) or not groups_list:
        raise ValueError("existing 2x2 source group roster is missing")
    return manifest, groups, exclusions, sets


def _group_admitted(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parent = list(range(len(candidates)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    first_by_key: Dict[Tuple[str, str], int] = {}
    for index, candidate in enumerate(candidates):
        keys = [
            ("identity", candidate["mapper_input_identity"]),
            ("visible", candidate["model_visible_graph_identity"]),
            ("lineage", candidate["leakage_lineage_id"]),
        ]
        keys.extend(("family", family)
                    for family in candidate["source_program_families"])
        for key in keys:
            if key in first_by_key:
                union(index, first_by_key[key])
            else:
                first_by_key[key] = index
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[find(index)].append(candidate)
    result = []
    for members in grouped.values():
        identities = sorted({item["mapper_input_identity"] for item in members})
        families = sorted({family for item in members
                           for family in item["source_program_families"]})
        group_id = canonical_json_sha256({
            "mapper_input_identities": identities,
            "source_program_families": families,
        })
        result.append({
            "group_id": group_id,
            "mapper_input_identities": identities,
            "source_program_families": families,
            "candidate_ids": sorted({item["candidate_id"] for item in members}),
            "split": "test_only",
        })
    return sorted(result, key=lambda item: item["group_id"])


def finalize_admission(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.output_dir.resolve()
    lowering_path = root / "lowering-audit.json"
    lowering = _read_json(lowering_path, "source-lowering audit")
    if lowering.get("native_mapping_invoked") is not False:
        raise ValueError("source-lowering audit reports native mapping")
    mach_ids, mach_visible, canonical_names = _canonical_identity_sets(
        args.canonical_map.resolve(),
    )
    current_manifest, current_groups, current_exclusions, current_sets = (
        _load_existing_training_metadata(args.base_root.resolve())
    )
    mach_name_set = {
        name for members in canonical_names.values() for name in members
    }
    excluded_by_id = set(current_exclusions["excluded_mapper_input_identities"])
    audit_by_case = lowering.get("case_audit")
    if not isinstance(audit_by_case, list):
        raise ValueError("source-lowering audit case roster is malformed")
    accepted_by_identity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    decision_rows = []
    for case in audit_by_case:
        if case.get("status") != "lowered":
            decision_rows.append({
                "case_id": case.get("case_id"),
                "admission": "rejected_source_lowering",
                "reasons": [case.get("reject_reason", "source lowering failed")],
            })
            continue
        identity = case["mapper_input_identity"]
        visible = case["model_visible_graph_identity"]
        source_families = list(case["source_program_families"])
        source_name = "machsuite/{}".format(case["case_id"])
        # Existing-corpus source_sha256 denotes the lowered DFG bytes.  The
        # lowering audit's source_sha256 denotes the upstream C/C++ file.
        upstream_source_sha = case["source_sha256"]
        dfg_sha = case["dfg_sha256"]
        lineage = case["leakage_lineage_id"]
        canonical_source_names = set(mach_name_set)
        canonical_source_names.update(
            "machsuite/{}".format(name) for name in mach_name_set
        )
        reasons = []
        if identity in mach_ids:
            reasons.append("canonical_orbit_mapper_identity")
        if visible in mach_visible:
            reasons.append("canonical_orbit_model_visible_identity")
        if case["case_id"] in canonical_source_names or source_name in canonical_source_names:
            reasons.append("canonical_orbit_source_name")
        if identity in current_sets["mapper_input_identity"]:
            reasons.append("current_323_mapper_input_identity")
        if visible in current_sets["model_visible_graph_identity"]:
            reasons.append("current_323_model_visible_graph_identity")
        if dfg_sha in current_sets["source_sha256"]:
            reasons.append("current_323_lowered_dfg_hash")
        if lineage in current_sets["leakage_lineage_id"]:
            reasons.append("current_323_leakage_lineage")
        if set(source_families) & current_sets["source_program_family"]:
            reasons.append("shared_current_source_program_family")
        if source_name in current_sets["source_name"]:
            reasons.append("current_323_source_name")
        if identity in excluded_by_id:
            reasons.append("preserved_existing_candidate_exclusion")
        if reasons:
            decision_rows.append({
                "case_id": case["case_id"],
                "mapper_input_identity": identity,
                "model_visible_graph_identity": visible,
                "source_sha256": dfg_sha,
                "upstream_source_sha256": upstream_source_sha,
                "source_program_families": source_families,
                "admission": "excluded_overlap",
                "reasons": sorted(set(reasons)),
            })
            continue
        case = dict(case)
        case.update({"source_name": source_name})
        accepted_by_identity[identity].append(case)

    accepted = []
    input_root = root / "inputs" / "dfg"
    input_root.mkdir(parents=True, exist_ok=True)
    for identity, members in sorted(accepted_by_identity.items()):
        if len({member["model_visible_graph_identity"] for member in members}) != 1:
            raise ValueError("one admitted mapper identity has conflicting visible graphs")
        representative = min(members, key=lambda item: item["case_id"])
        dfg_path = (root / representative["dfg_relative_path"]).resolve()
        if sha256_file(dfg_path) != representative["dfg_sha256"]:
            raise ValueError("lowered DFG changed before admission: {}".format(dfg_path))
        destination = input_root / (identity + ".mlir")
        shutil.copyfile(dfg_path, destination)
        if sha256_file(destination) != representative["dfg_sha256"]:
            raise ValueError("admitted DFG copy hash mismatch")
        members_families = sorted({family for member in members
                                   for family in member["source_program_families"]})
        lineage_ids = sorted({member["leakage_lineage_id"] for member in members})
        source_names = sorted({member["source_name"] for member in members})
        candidate_ids = sorted({
            "machsuite/{}/{}".format(member["case_id"], identity)
            for member in members
        })
        accepted.append({
            "candidate_id": candidate_ids[0],
            "candidate_ids": candidate_ids,
            "mapper_input_identity": identity,
            "model_visible_graph_identity": representative[
                "model_visible_graph_identity"],
            "source_path": str(destination.relative_to(root)),
            "source_sha256": representative["dfg_sha256"],
            "source_names": source_names,
            "source_program_families": members_families,
            "domain": "real_program",
            "domains": ["real_program"],
            "leakage_lineage_id": lineage_ids[0],
            "leakage_lineage_ids": lineage_ids,
            "source_cases": sorted({member["case_id"] for member in members}),
            "source_relative_paths": sorted({member["source_relative_path"]
                                              for member in members}),
            "native_mapping_invoked": False,
            "native_labels_present": False,
        })
        for member in members:
            decision_rows.append({
                "case_id": member["case_id"],
                "mapper_input_identity": identity,
                "model_visible_graph_identity": member[
                    "model_visible_graph_identity"],
                "source_sha256": member["dfg_sha256"],
                "upstream_source_sha256": member["source_sha256"],
                "source_program_families": member["source_program_families"],
                "admission": "test_only",
                "reasons": [],
            })

    if len(accepted) > MAX_NEW_IDENTITY_BUDGET:
        raise ValueError("admitted independent DFG identities exceed the cap of 32")
    projected_queries = len(accepted) * len(SHAPES_2X2)
    if projected_queries > args.max_new_queries:
        raise ValueError("admitted 2x2 queries exceed --max-new-queries")
    missing_required = sorted(
        REQUIRED_MACH_CASES - {
            case_id for candidate in accepted for case_id in candidate["source_cases"]
        }
    )
    if missing_required:
        raise ValueError("required source categories did not pass admission: {}".format(
            ", ".join(missing_required),
        ))

    source_manifest = {
        "schema": SOURCE_SCHEMA,
        "evaluation_only": True,
        "training_eligible": False,
        "source_protocol": "pinned-machsuite-source-only-no-native-labels",
        "architecture_sha256": sha256_file(args.architecture),
        "shape_set": [list(shape) for shape in SHAPES_2X2],
        "candidate_count": len(accepted),
        "query_count": len(accepted),
        "native_query_count_projected": projected_queries,
        "candidates": accepted,
    }
    source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
    source_manifest_path = root / "source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
    )

    source_groups = {
        "schema": SOURCE_GROUP_SCHEMA,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "split_policy": "all_admitted_groups_test_only_before_native_mapping",
        "groups": _group_admitted(accepted),
    }
    source_groups["manifest_sha256"] = canonical_json_sha256(source_groups)
    source_groups_path = root / "source-groups.json"
    source_groups_path.write_text(
        json.dumps(source_groups, indent=2, sort_keys=True) + "\n",
    )

    admission = {
        "schema": ADMISSION_SCHEMA,
        "split": "test_only",
        "admission_frozen_before_mapping": True,
        "native_mapping_invoked": False,
        "native_labels_present": False,
        "admitted_unique_mapper_input_identity_count": len(accepted),
        "admitted_connected_source_group_count": len(source_groups["groups"]),
        "shape_count_per_identity": len(SHAPES_2X2),
        "projected_native_query_count": projected_queries,
        "caps": {
            "unique_identities": MAX_NEW_IDENTITY_BUDGET,
            "native_queries": args.max_new_queries,
            "absolute_native_query_cap": MAX_NEW_QUERY_BUDGET,
        },
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_groups_path": str(source_groups_path),
        "source_groups_sha256": sha256_file(source_groups_path),
        "source_corpus_path": str(args.base_root.resolve() / "source-corpus" / "manifest.json"),
        "source_corpus_sha256": sha256_file(
            args.base_root.resolve() / "source-corpus" / "manifest.json",
        ),
        "source_corpus_groups_path": str(
            args.base_root.resolve() / "source-corpus" / "groups.json",
        ),
        "source_corpus_groups_sha256": sha256_file(
            args.base_root.resolve() / "source-corpus" / "groups.json",
        ),
        "training_exclusions_sha256": sha256_file(
            args.base_root.resolve() / "training-exclusions.json",
        ),
        "canonical_benchmark_identity_audit_path": str(
            args.canonical_map.resolve(),
        ),
        "canonical_benchmark_identity_audit_sha256": sha256_file(
            args.canonical_map.resolve(),
        ),
        "decisions": decision_rows,
    }
    admission["manifest_sha256"] = canonical_json_sha256(admission)
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")

    command = [
        "python3", str(PROJECT_ROOT / "adapters" / "collect_per_cgra_2x2_mappings.py"),
        "--source-manifest", str(source_manifest_path),
        "--architecture", str(args.architecture.resolve()),
        "--neura-opt", str(args.neura_opt.resolve()),
        "--output-dir", str(root / "fresh-native-collection"),
        "--jobs", "8", "--timeout-seconds", "120",
        "--max-new-queries", str(args.max_new_queries),
    ]
    protocol = {
        "schema": PREPARATION_SCHEMA,
        "phase": "admission_frozen_collection_not_started",
        "mapping_authorized": False,
        "source_manifest_schema": SOURCE_SCHEMA,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_groups_path": str(source_groups_path),
        "source_groups_sha256": sha256_file(source_groups_path),
        "admission_path": str(admission_path),
        "admission_sha256": sha256_file(admission_path),
        "architecture_path": str(args.architecture.resolve()),
        "architecture_sha256": sha256_file(args.architecture),
        "neura_opt_path": str(args.neura_opt.resolve()),
        "neura_opt_sha256": sha256_file(args.neura_opt),
        "mapper_strategy": "heuristic",
        "external_timeout_seconds": 120,
        "jobs": 8,
        "max_new_queries": args.max_new_queries,
        "projected_query_count": projected_queries,
        "native_collection_command": command,
        "native_mapping_invoked": False,
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
    )
    return admission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-only", action="store_true",
                        help="compile/lower the pinned bounded source roster only")
    parser.add_argument("--finalize-admission", action="store_true",
                        help="freeze a test-only source manifest after identity audit")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--importer", type=Path, default=DEFAULT_IMPORTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--canonical-map", type=Path, default=DEFAULT_CANONICAL_MAP)
    parser.add_argument("--max-new-queries", type=int,
                        default=MAX_NEW_QUERY_BUDGET)
    parser.add_argument("--architecture", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--neura-opt", type=Path, default=DEFAULT_NEURA_OPT)
    parser.add_argument("--clang", type=Path, default=Path("/usr/local/bin/clang"))
    parser.add_argument("--clangxx", type=Path, default=Path("/usr/local/bin/clang++"))
    parser.add_argument("--llvm-extract", type=Path,
                        default=Path("/usr/local/bin/llvm-extract"))
    parser.add_argument("--mlir-translate", type=Path,
                        default=Path("/home/x/shiran/llvm-project/build/bin/mlir-translate"))
    parser.add_argument("--stage-timeout", type=int, default=90)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.lower_only == args.finalize_admission:
        raise SystemExit("choose exactly one of --lower-only or --finalize-admission")
    if args.stage_timeout <= 0:
        raise SystemExit("--stage-timeout must be positive")
    if not 0 < args.max_new_queries <= MAX_NEW_QUERY_BUDGET:
        raise SystemExit("--max-new-queries must be between 1 and 256")
    for path in (args.architecture, args.neura_opt):
        if not path.is_file():
            raise SystemExit("required tool input does not exist: {}".format(path))
    if sha256_file(args.architecture) != EXPECTED_ARCHITECTURE_SHA256:
        raise SystemExit("2x2 architecture spec hash does not match frozen protocol")
    if sha256_file(args.neura_opt) != EXPECTED_NEURA_OPT_SHA256:
        raise SystemExit("Neura binary hash does not match frozen collection protocol")
    if args.lower_only:
        result = lower_sources(args)
        print(json.dumps({
            "event": "lowering_complete",
            "lowered_source_cases": result["lowered_source_case_count"],
            "rejected_source_cases": result["rejected_source_case_count"],
            "unique_identities": result["unique_mapper_input_identity_count"],
            "projected_queries_before_admission": (
                result["projected_2x2_query_count_before_admission"]
            ),
            "admission_frozen": False,
        }, sort_keys=True))
    else:
        result = finalize_admission(args)
        print(json.dumps({
            "event": "admission_complete",
            "admitted_unique_identities": (
                result["admitted_unique_mapper_input_identity_count"]
            ),
            "source_groups": result["admitted_connected_source_group_count"],
            "projected_queries": result["projected_native_query_count"],
            "native_mapping_invoked": False,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
