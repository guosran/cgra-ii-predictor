#!/usr/bin/env python3
"""Collect all 16 heuristic-mapper labels per eligible KernelBench L1/L2 DFG.

The controller is resumable and has no external mapper timeout. Every query
gets a terminal result only after Neura exits and its mapping is parsed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "adapters"))

from mapping_artifact_protocol import (  # noqa: E402
    canonical_json_sha256, mapper_input_identity, parse_single_mapping,
    sha256_file,
)
from neura_cost_features import parse_cost_features  # noqa: E402
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL  # noqa: E402

SCHEMA = "cgra-ii-kernelbench-level1-mapper-collection-v1"
STDERR_TAIL_BYTES = 65536


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _queries(import_manifest: dict) -> list[dict]:
    generic = import_manifest.get("schema") == "cgra-ii-aux-kernel-import-v1"
    groups = import_manifest["identity_groups"]
    if len(groups) != import_manifest["distinct_mapper_input_dfg_count"]:
        raise ValueError("distinct DFG count disagrees with identity groups")
    if import_manifest.get("mapper_input_identity_mode") != (
        "ignore-static-memref-tensor-extents-and-counter-bounds"
    ):
        raise ValueError("KernelBench mapper identity normalization is missing")
    if not generic and not import_manifest.get("heldout_manifests"):
        raise ValueError("held-out lineage exclusion provenance is missing")
    shapes = SHAPE_PROTOCOL.mapper_shapes
    if len(shapes) != 16:
        raise ValueError("expected exactly 16 oriented rectangles")
    queries = []
    for identity, members in sorted(groups.items()):
        if not members:
            raise ValueError(f"empty mapper identity group: {identity}")
        representative = sorted(members, key=lambda x: (
            str(x["case_id"]) if generic else x["case_id"], x["task"
        ]))[0]
        dfg = Path(representative["dfg_path"])
        if sha256_file(dfg) != representative["dfg_sha256"]:
            raise ValueError(f"DFG hash mismatch: {dfg}")
        if mapper_input_identity(dfg.read_text(), normalize_static_shapes=True) != identity:
            raise ValueError(f"DFG identity mismatch: {dfg}")
        for rows, cols in shapes:
            query = {
                "mapper_input_identity": identity,
                "model_visible_graph_identity": representative["model_visible_graph_identity"],
                "dfg_path": str(dfg.resolve()),
                "dfg_sha256": representative["dfg_sha256"],
                "source_cases": (
                    sorted({str(member["case_id"]) for member in members})
                    if generic else
                    sorted({int(member["case_id"]) for member in members})
                ),
                "source_tasks": [{"case_id": member["case_id"], "task": member["task"]}
                                 for member in members],
                "rows": rows, "cols": cols,
                "physical_rows": rows // 4, "physical_cols": cols // 4,
            }
            if generic:
                query["source_program_families"] = sorted({
                    family for member in members
                    for family in member.get("source_program_families", [])
                })
            queries.append(query)
    if len(queries) != import_manifest["projected_16_shape_query_count"]:
        raise ValueError("projected query count disagrees with 16-shape expansion")
    # Vary graph and shape across workers, while starting with smaller graphs.
    queries.sort(key=lambda q: (
        len(Path(q["dfg_path"]).read_text()), q["rows"] * q["cols"],
        q["mapper_input_identity"], q["rows"], q["cols"],
    ))
    return queries


def _directory(root: Path, query: dict) -> Path:
    return (root / "artifacts" / query["mapper_input_identity"] /
            f"{query['rows']}x{query['cols']}")


def _verify_result(path: Path, query: dict) -> Optional[dict]:
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    if (row.get("mapper_input_identity") != query["mapper_input_identity"] or
            row.get("dfg_sha256") != query["dfg_sha256"] or
            row.get("rows") != query["rows"] or row.get("cols") != query["cols"]):
        raise ValueError(f"stored mapper result provenance mismatch: {path}")
    if row.get("status") == "success":
        artifact = _directory(path.parents[3], query) / "mapped.mlir"
        # path.parents[3] is the collection root for artifacts/identity/shape/result.
        if (not artifact.is_file() or sha256_file(artifact) != row.get("mapped_artifact_sha256")):
            raise ValueError(f"stored successful mapping artifact changed: {artifact}")
        facts = parse_single_mapping(artifact.read_text(), query["rows"], query["cols"])
        if facts["compiled_ii"] != row["compiled_ii"]:
            raise ValueError(f"stored compiled II changed: {artifact}")
    elif row.get("status") != "censored" or row.get("compiled_ii") is not None:
        raise ValueError(f"stored query result is not valid censored data: {path}")
    return row


def _native_mappers_for_root(root: Path) -> list[int]:
    marker = str(root / "artifacts").encode()
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes()
        except OSError:
            continue
        if marker in command and b"--map-to-accelerator" in command:
            found.append(int(cmdline.parent.name))
    return found


def _wait_for_old_mappers(root: Path) -> None:
    last_report = 0.0
    while True:
        found = _native_mappers_for_root(root)
        if not found:
            return
        if time.monotonic() - last_report >= 60:
            print(json.dumps({"event": "waiting_for_previous_mapper_processes",
                              "pids": found}), flush=True)
            last_report = time.monotonic()
        time.sleep(15)


def _recent_unfinished_native_logs(root: Path, max_age_seconds: int = 120) -> list[str]:
    """Detect a collector launched before filesystem locking was introduced.

    A separate sandbox can hide its process from /proc, while its mapper log
    still advances on the shared filesystem.  This transition guard is
    conservative; future collectors also hold the advisory root lock.
    """
    now = time.time()
    active = []
    for log in (root / "artifacts").glob("*/*/mapper.stderr.log"):
        if (log.parent / "result.json").exists():
            continue
        try:
            age = now - log.stat().st_mtime
        except FileNotFoundError:
            continue
        if age <= max_age_seconds:
            active.append(str(log))
    return sorted(active)


def _acquire_collection_lock(root: Path, *, dry_run: bool,
                             acknowledge_stale_running: bool = False):
    """Keep the returned stream alive for the entire collection invocation."""
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = (root / ".collector.lock").open("a+")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dry_run:
            recent_logs = _recent_unfinished_native_logs(root)
            if recent_logs:
                raise RuntimeError(
                    "recent unfinished native mapper logs indicate an older "
                    f"collector may still be active: {recent_logs[:4]}")
            progress_path = root / "progress.json"
            if progress_path.exists():
                prior_progress = json.loads(progress_path.read_text())
                if (prior_progress.get("running_count", 0) > 0 and
                        not acknowledge_stale_running):
                    raise RuntimeError(
                        "previous progress reports running native queries; "
                        "verify the old collector is gone, then use "
                        "--acknowledge-stale-running")
    except BaseException:
        lock_stream.close()
        raise
    return lock_stream


def _analyze(query: dict, opt: Path, architecture: Path, directory: Path,
             timeout: Optional[int] = None) -> dict:
    output = directory / "analysis.mlir"
    command = [
        str(opt), query["dfg_path"], f"--architecture-spec={architecture}",
        f"--analyze-rec-res-mii=x-tiles={query['cols']} y-tiles={query['rows']}",
        "-o", str(output),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, check=False,
                                   timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "analysis_resource_timeout",
                "external_timeout_seconds": timeout}
    if completed.returncode != 0 or not output.is_file():
        return {"status": "failed", "exit_code": completed.returncode,
                "stderr_tail": completed.stderr[-4000:]}
    facts = parse_cost_features(output.read_text())
    if facts is None:
        return {"status": "failed", "reason": "analysis output has no RecMII/ResMII"}
    rec, res = int(facts["rec_mii"]), int(facts["res_mii"])
    return {"status": "success", "rec_mii": rec, "res_mii": res,
            "lower_bound": max(rec, res), "artifact_sha256": sha256_file(output)}


def _map(query: dict, root: Path, opt: Path, architecture: Path,
         timeout: Optional[int] = None) -> dict:
    directory = _directory(root, query)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    analysis = _analyze(query, opt, architecture, directory, timeout)
    artifact = directory / "mapped.mlir"
    command = [
        str(opt), query["dfg_path"], f"--architecture-spec={architecture}",
        ("--map-to-accelerator=mapping-strategy=heuristic "
         f"x-tiles={query['cols']} y-tiles={query['rows']}"),
        "-o", str(artifact),
    ]
    tail = b""
    tail_path = directory / "mapper.stderr.log"
    last_tail_write = 0.0
    timed_out = False
    if analysis["status"] == "success":
        with subprocess.Popen(command, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE) as process:
            assert process.stderr is not None
            deadline = time.monotonic() + timeout if timeout is not None else None
            while True:
                if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                ready, _, _ = select.select([process.stderr], [], [], 0.5)
                if ready:
                    chunk = os.read(process.stderr.fileno(), 65536)
                    if not chunk:
                        break
                    tail = (tail + chunk)[-STDERR_TAIL_BYTES:]
                    if time.monotonic() - last_tail_write >= 5:
                        tail_path.write_bytes(tail)
                        last_tail_write = time.monotonic()
            return_code = process.wait()
    else:
        return_code = None
    tail_path.write_bytes(tail)
    result: dict[str, Any] = {
        "schema": SCHEMA, "mapper_input_identity": query["mapper_input_identity"],
        "model_visible_graph_identity": query["model_visible_graph_identity"],
        "dfg_path": query["dfg_path"], "dfg_sha256": query["dfg_sha256"],
        "source_cases": query["source_cases"],
        "rows": query["rows"], "cols": query["cols"],
        "physical_rows": query["physical_rows"],
        "physical_cols": query["physical_cols"],
        "analysis": analysis, "native_exit_code": return_code,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": _now(), "external_timeout_seconds": timeout,
        "mapper_command": command, "mapped_artifact_path": str(artifact),
        "status": "censored", "compiled_ii": None,
    }
    if analysis["status"] != "success":
        result["censor_reason"] = "analysis_failed"
    elif timed_out:
        result["censor_reason"] = "mapper_resource_timeout"
    elif return_code != 0:
        result["censor_reason"] = "mapper_native_search_failed"
    elif not artifact.is_file():
        result["censor_reason"] = "mapper_output_missing"
    else:
        try:
            facts = parse_single_mapping(artifact.read_text(), query["rows"], query["cols"])
        except (OSError, UnicodeError, ValueError) as error:
            result["censor_reason"] = "mapper_output_invalid"
            result["validation_error"] = str(error)
        else:
            result.update(status="success", compiled_ii=int(facts["compiled_ii"]),
                          mapped_artifact_sha256=sha256_file(artifact),
                          verified_placement_coordinate_count=facts[
                              "placement_coordinate_count"
                          ])
    if result["status"] != "success":
        result["stderr_tail"] = tail.decode(errors="replace")[-4000:]
    _write_json(directory / "result.json", result)
    return result


def _progress(root: Path, completed: dict, total: int, running: int) -> dict:
    success = sum(row["status"] == "success" for row in completed.values())
    censored = sum(row["status"] == "censored" for row in completed.values())
    value = {
        "schema": SCHEMA, "updated_at_utc": _now(),
        "expected_query_count": total, "terminal_count": len(completed),
        "success_count": success, "censored_count": censored,
        "running_count": running, "pending_count": total - len(completed) - running,
    }
    _write_json(root / "progress.json", value)
    return value


def _collect_pending(root: Path, pending: list[dict], completed: dict,
                     expected: int, jobs: int, neura_opt: Path,
                     architecture: Path, timeout: Optional[int],
                     map_one=None) -> dict:
    """Dispatch at most ``jobs`` queries and drain active work on pause."""
    mapper = _map if map_one is None else map_one
    pause_marker = root / "pause.requested"
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        remaining = iter(pending)
        active = {}

        def start_one() -> bool:
            if pause_marker.exists():
                return False
            query = next(remaining, None)
            if query is None:
                return False
            future = pool.submit(mapper, query, root, neura_opt.resolve(),
                                 architecture.resolve(), timeout)
            active[future] = query
            return True

        for _ in range(min(jobs, len(pending))):
            start_one()
        _progress(root, completed, expected, len(active))
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                query = active.pop(future)
                row = future.result()
                key = (query["mapper_input_identity"], query["rows"], query["cols"])
                completed[key] = row
                start_one()
                status = _progress(root, completed, expected, len(active))
                print(json.dumps({"event": "query_finished",
                                  "case_ids": query["source_cases"],
                                  "shape": f"{query['rows']}x{query['cols']}",
                                  "status": row["status"],
                                  "compiled_ii": row["compiled_ii"],
                                  "terminal_count": status["terminal_count"],
                                  "expected_query_count": expected,
                                  "elapsed_seconds": row["elapsed_seconds"]}),
                      flush=True)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-manifest", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--neura-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--max-new-queries", type=int,
                        help="run a bounded smoke batch; omit for the entire corpus")
    parser.add_argument("--include-source-case", action="append", default=[],
                        help="dispatch only queries from this source case; repeatable")
    parser.add_argument("--include-mapper-shape", action="append", default=[],
                        help="dispatch only this mapper rectangle, e.g. 8x8; repeatable")
    parser.add_argument("--acknowledge-stale-running", action="store_true",
                        help="resume after independently verifying an old "
                             "nonzero running_count has no live mapper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query-timeout-seconds", type=int,
                        help="censor queries whose analysis or native search exceeds this cap")
    parser.add_argument("--reuse-success-root", type=Path,
                        help="copy parser-verified successes from a capped audit; rerun every non-success natively")
    parser.add_argument("--prediction-seal", type=Path,
                        help="required before revealing sealed TorchBench mapper labels")
    args = parser.parse_args()
    if args.jobs <= 0 or (args.max_new_queries is not None and args.max_new_queries <= 0):
        parser.error("--jobs and --max-new-queries must be positive")
    if args.query_timeout_seconds is not None and args.query_timeout_seconds <= 0:
        parser.error("--query-timeout-seconds must be positive")
    imported = json.loads(args.import_manifest.read_text())
    generic = imported.get("schema") == "cgra-ii-aux-kernel-import-v1"
    level = imported.get("level", 1)
    if not generic and (level not in (1, 2) or imported.get("schema") != f"cgra-ii-kernelbench-level{level}-import-v1"):
        raise ValueError("mapper import manifest schema mismatch")
    global SCHEMA
    SCHEMA = ("cgra-ii-aux-kernel-mapper-collection-v1" if generic else
              f"cgra-ii-kernelbench-level{level}-mapper-collection-v1")
    prediction_seal_sha256 = None
    if generic and imported.get("suite") == "torchbench-sealed":
        if imported.get("status") != "ready" or args.prediction_seal is None:
            raise ValueError("sealed TorchBench mapping requires ready test and predictions")
        prediction_seal = json.loads(args.prediction_seal.read_text())
        if (prediction_seal.get("schema") != "cgra-ii-torchbench-prediction-seal-v1" or
                prediction_seal.get("test_import_sha256") != sha256_file(args.import_manifest) or
                prediction_seal.get("compiled_ii_labels_accessed") is not False or
                sha256_file(args.prediction_seal.parent / "predictions.json") !=
                prediction_seal.get("predictions_sha256")):
            raise ValueError("sealed TorchBench prediction provenance mismatch")
        prediction_seal_sha256 = sha256_file(args.prediction_seal)
    queries = _queries(imported)
    selected_cases = set(args.include_source_case)
    if generic and imported.get("suite") in (
            "polybench-c", "polybench-o0-repair", "taclebench-v1.9",
            "taclebench-v1.9-inline", "hls-eval-c2hlsc-small",
            "hls-eval-gnn-small"):
        if not selected_cases:
            parser.error("public C benchmark mapping requires an explicit --include-source-case")
    selected_shapes = set()
    for value in args.include_mapper_shape:
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(part.isdecimal() for part in parts):
            parser.error(f"invalid --include-mapper-shape: {value}")
        selected_shapes.add((int(parts[0]), int(parts[1])))
    if not selected_shapes <= set(SHAPE_PROTOCOL.mapper_shapes):
        parser.error("--include-mapper-shape is outside the 16 oriented rectangles")
    available_cases = {str(case) for query in queries
                       for case in query["source_cases"]}
    if not selected_cases <= available_cases:
        raise ValueError("selected source case is absent from the query manifest: "
                         f"{sorted(selected_cases - available_cases)}")
    reuse_root = args.reuse_success_root.resolve() if args.reuse_success_root else None
    reuse_provenance = None
    if reuse_root is not None:
        if args.query_timeout_seconds is not None:
            raise ValueError("success migration requires an unbounded destination")
        old_query_path = reuse_root / "query-manifest.json"
        old_query_manifest = json.loads(old_query_path.read_text())
        reuse_provenance = old_query_manifest["provenance"]
        if (old_query_manifest.get("queries") != queries or
                reuse_provenance.get("external_timeout_seconds") is None or
                reuse_provenance.get("import_manifest_sha256") !=
                sha256_file(args.import_manifest) or
                reuse_provenance.get("architecture_sha256") !=
                sha256_file(args.architecture) or
                reuse_provenance.get("neura_opt_sha256") !=
                sha256_file(args.neura_opt)):
            raise ValueError("capped success source does not match this mapper protocol")
    root = args.output_dir.resolve()
    if reuse_root == root:
        raise ValueError("success source and unbounded destination must differ")
    # Keep this descriptor live through main(); flock spans tool-session PID
    # namespaces and prevents a second new collector from sharing the queue.
    lock_stream = _acquire_collection_lock(
        root, dry_run=args.dry_run,
        acknowledge_stale_running=args.acknowledge_stale_running)
    provenance_path = root / "provenance.json"
    prior_provenance = (json.loads(provenance_path.read_text())
                        if provenance_path.exists() else None)
    provenance = {
        "schema": SCHEMA, "import_manifest_path": str(args.import_manifest.resolve()),
        "import_manifest_sha256": sha256_file(args.import_manifest),
        "import_manifest_identity": imported["manifest_sha256"],
        "upstream_commit": imported["upstream_commit"],
        "architecture_path": str(args.architecture.resolve()),
        "architecture_sha256": sha256_file(args.architecture),
        "neura_opt_path": str(args.neura_opt.resolve()),
        "neura_opt_sha256": sha256_file(args.neura_opt),
        "mapper_strategy": "heuristic",
        "external_timeout_seconds": args.query_timeout_seconds,
        "oriented_rectangle_count": 16, "expected_query_count": len(queries),
    }
    # Preserve legacy roots exactly; bind worker count on every new bounded
    # root so a later resume cannot silently change the resource target.
    if (args.query_timeout_seconds is not None and
            (prior_provenance is None or
             "collection_worker_count" in prior_provenance)):
        provenance["collection_worker_count"] = args.jobs
    if prediction_seal_sha256 is not None:
        provenance["prediction_seal_sha256"] = prediction_seal_sha256
    if reuse_root is not None:
        provenance["reused_successes_from"] = {
            "root": str(reuse_root),
            "query_manifest_sha256": sha256_file(reuse_root / "query-manifest.json"),
            "source_external_timeout_seconds": reuse_provenance[
                "external_timeout_seconds"],
            "rule": "copy_only_parser_verified_successes; rerun_censored_and_pending",
        }
    if provenance_path.exists():
        if prior_provenance != provenance:
            raise ValueError("collection provenance changed; refusing to mix mapping runs")
    else:
        _write_json(provenance_path, provenance)
    query_manifest_path = root / "query-manifest.json"
    query_manifest = {"schema": SCHEMA, "provenance": provenance,
                      "query_count": len(queries), "queries": queries}
    query_manifest["manifest_sha256"] = canonical_json_sha256(query_manifest)
    if query_manifest_path.exists():
        if json.loads(query_manifest_path.read_text()) != query_manifest:
            raise ValueError("query manifest changed; refusing to mix mapping runs")
    else:
        _write_json(query_manifest_path, query_manifest)
    if args.dry_run:
        print(json.dumps({"query_manifest": str(query_manifest_path),
                          "expected_query_count": len(queries), "jobs": args.jobs,
                          "selected_source_cases": sorted(selected_cases),
                          "selected_mapper_shapes": sorted(selected_shapes),
                          "external_timeout_seconds": args.query_timeout_seconds}))
        return 0
    if (root / "pause.requested").exists():
        raise RuntimeError(f"mapping collection paused by {root / 'pause.requested'}")
    _wait_for_old_mappers(root)
    if reuse_root is not None:
        reused = []
        for query in queries:
            old_directory = _directory(reuse_root, query)
            old_result_path = old_directory / "result.json"
            old_result = _verify_result(old_result_path, query)
            if old_result is None or old_result["status"] != "success":
                continue
            destination = _directory(root, query)
            if (destination / "result.json").exists():
                destination_result = _verify_result(destination / "result.json", query)
                if destination_result["status"] != "success":
                    continue
            else:
                shutil.copytree(old_directory, destination, dirs_exist_ok=True)
                _verify_result(destination / "result.json", query)
            reused.append({
                "mapper_input_identity": query["mapper_input_identity"],
                "rows": query["rows"], "cols": query["cols"],
                "source_result_sha256": sha256_file(old_result_path),
                "destination_result_sha256": sha256_file(
                    destination / "result.json"),
            })
        _write_json(root / "reused-successes.json", {
            "schema": "cgra-ii-mapper-success-migration-v1",
            "source_root": str(reuse_root),
            "source_query_manifest_sha256": sha256_file(
                reuse_root / "query-manifest.json"),
            "destination_query_manifest_sha256": sha256_file(query_manifest_path),
            "reused_success_count": len(reused),
            "reused_successes": reused,
        })
    completed = {}
    pending = []
    for query in queries:
        key = (query["mapper_input_identity"], query["rows"], query["cols"])
        row = _verify_result(_directory(root, query) / "result.json", query)
        if row is None:
            if ((not selected_cases or selected_cases.intersection(
                    str(case) for case in query["source_cases"])) and
                    (not selected_shapes or
                     (query["rows"], query["cols"]) in selected_shapes)):
                pending.append(query)
        else:
            completed[key] = row
    if args.max_new_queries is not None:
        pending = pending[:args.max_new_queries]
    print(json.dumps({"event": "queue_ready", "expected_query_count": len(queries),
                      "already_terminal": len(completed), "new_queries": len(pending),
                      "selected_source_cases": sorted(selected_cases),
                      "selected_mapper_shapes": sorted(selected_shapes),
                      "jobs": args.jobs,
                      "external_timeout_seconds": args.query_timeout_seconds}), flush=True)
    _progress(root, completed, len(queries), 0)
    _collect_pending(root, pending, completed, len(queries), args.jobs,
                     args.neura_opt, args.architecture,
                     args.query_timeout_seconds)
    if (root / "pause.requested").exists():
        print(json.dumps({"event": "pause_requested_drained_active_queries",
                          "terminal_count": len(completed),
                          "expected_query_count": len(queries)}), flush=True)
    if len(completed) == len(queries):
        final = _progress(root, completed, len(queries), 0)
        final["query_manifest_sha256"] = sha256_file(query_manifest_path)
        _write_json(root / "collection-complete.json", final)
        print(json.dumps({"event": "collection_complete", **final}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
