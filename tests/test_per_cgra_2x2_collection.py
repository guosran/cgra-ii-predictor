"""Safety and smoke coverage for the frozen per-CGRA 2x2 collector."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))

import collect_per_cgra_2x2_mappings as collection
from mapping_artifact_protocol import mapper_input_identity, sha256_file


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    dfg_dir = source_root / "dfg"
    dfg_dir.mkdir(parents=True)
    dfg_text = "module {\n  func.func @tiny_dfg() {\n    return\n  }\n}\n"
    dfg_path = dfg_dir / "tiny.mlir"
    dfg_path.write_text(dfg_text)
    dfg_sha = sha256_file(dfg_path)
    identity = mapper_input_identity(dfg_text, normalize_static_shapes=True)

    old_shapes = ((4, 4), (4, 8), (8, 4), (4, 12), (12, 4),
                  (8, 8), (8, 12), (12, 8), (12, 12))
    candidates = []
    for rows, cols in old_shapes:
        candidates.append({
            "candidate_id": f"{identity}/{rows}x{cols}",
            "mapper_input_identity": identity,
            "model_visible_graph_identity": "b" * 64,
            "source_path": "dfg/tiny.mlir",
            "source_sha256": dfg_sha,
            "source_program_families": ["kernelbench/case-001"],
            "source_names": ["kernelbench-level1"],
            "domain": "dl",
            "leakage_lineage_id": identity,
            # These intentionally contradictory source labels must be ignored.
            "compiled_ii": 999,
            "outcome_label": "native_failure",
            "status": "success",
            "rows": rows,
            "columns": cols,
        })
    source_manifest = source_root / "manifest.json"
    source_manifest.write_text(json.dumps({
        "schema": collection.SOURCE_SCHEMA,
        "query_count": 1,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }, sort_keys=True))

    architecture = tmp_path / "architecture.yaml"
    architecture.write_text(
        "multi_cgra_defaults:\n"
        "  rows: 4\n"
        "  columns: 4\n"
        "per_cgra_defaults:\n"
        "  rows: 2\n"
        "  columns: 2\n"
    )
    fake_opt = tmp_path / "fake-neura-opt"
    fake_opt.write_text(f"""#!{sys.executable}
import pathlib
import re
import sys

args = sys.argv[1:]
output = pathlib.Path(args[args.index('-o') + 1])
analysis = next((arg for arg in args if arg.startswith('--analyze-rec-res-mii=')), None)
if analysis:
    output.write_text('rec_res_mii_info = {{ rec_mii = 1 : i32, res_mii = 1 : i32 }}\\n')
    raise SystemExit(0)
map_arg = next(arg for arg in args if arg.startswith('--map-to-accelerator='))
cols = int(re.search(r'x-tiles=(\\d+)', map_arg).group(1))
rows = int(re.search(r'y-tiles=(\\d+)', map_arg).group(1))
if (rows, cols) == (2, 4):
    print('[MapToAcceleratorPass] Mapping failed for all target II values.', file=sys.stderr)
    raise SystemExit(1)
output.write_text(
    'mapping_info = {{ compiled_ii = 3 : i32, rec_mii = 1 : i32, '
    'res_mii = 1 : i32, x_tiles = ' + str(cols) + ' : i32, y_tiles = '
    + str(rows) + ' : i32 }}\\n'
    'x = 0 : i32, y = 0 : i32\\n'
)
""")
    fake_opt.chmod(0o755)
    return source_manifest, architecture, fake_opt


def _prepare(source: Path, architecture: Path, fake_opt: Path,
             output: Path, jobs: int = 1) -> None:
    assert collection.main([
        "--source-manifest", str(source),
        "--architecture", str(architecture),
        "--neura-opt", str(fake_opt),
        "--output-dir", str(output),
        "--jobs", str(jobs),
        "--prepare-only",
    ]) == 0


def test_frozen_roster_is_deterministic_and_uses_only_fresh_metadata(tmp_path):
    source, architecture, fake_opt = _fixture_files(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _prepare(source, architecture, fake_opt, first)
    _prepare(source, architecture, fake_opt, second)

    first_manifest = (first / "query-manifest.json").read_bytes()
    second_manifest = (second / "query-manifest.json").read_bytes()
    assert first_manifest == second_manifest
    manifest = json.loads(first_manifest)
    assert manifest["query_count"] == 8
    assert manifest["provenance"]["old_native_labels_reused"] is False
    assert "compiled_ii" not in first_manifest.decode()
    assert "outcome_label" not in first_manifest.decode()
    assert sha256_file(first / "inputs" / "dfg" /
                       f"{manifest['queries'][0]['mapper_input_identity']}.mlir") == \
        manifest["queries"][0]["dfg_sha256"]

    relocated = tmp_path / "relocated"
    shutil.copytree(first, relocated)
    _prepare(source, architecture, fake_opt, relocated)
    assert (relocated / "query-manifest.json").read_bytes() == first_manifest
    outcomes = json.loads((relocated / "outcomes.json").read_text())
    assert outcomes["queries"][0]["dfg_path"].startswith("inputs/")
    assert outcomes["queries"][0]["status"] == "pending"
    assert not Path(manifest["queries"][0]["dfg_path"]).is_absolute()


def test_shapes_are_the_oriented_eight_rectangles_on_two_by_two_tiles():
    assert collection.SHAPE_PROTOCOL_ID == "amoeba-static-rectangles-2x2-per-cgra-max4"
    assert collection.MAPPER_SHAPES == (
        (2, 2), (2, 4), (4, 2), (2, 6),
        (6, 2), (2, 8), (8, 2), (4, 4),
    )
    assert [(rows // 2, cols // 2) for rows, cols in collection.MAPPER_SHAPES] == [
        (1, 1), (1, 2), (2, 1), (1, 3),
        (3, 1), (1, 4), (4, 1), (2, 2),
    ]


def test_resume_rejects_worker_count_or_architecture_protocol_changes(tmp_path):
    source, architecture, fake_opt = _fixture_files(tmp_path)
    output = tmp_path / "prepared"
    _prepare(source, architecture, fake_opt, output, jobs=1)
    with pytest.raises(ValueError, match="collection provenance changed"):
        _prepare(source, architecture, fake_opt, output, jobs=2)


def test_smoke_cli_maps_tiny_dfg_and_keeps_native_failure_unlabeled(tmp_path):
    source, architecture, fake_opt = _fixture_files(tmp_path)
    output = tmp_path / "collection"
    assert collection.main([
        "--source-manifest", str(source),
        "--architecture", str(architecture),
        "--neura-opt", str(fake_opt),
        "--output-dir", str(output),
        "--jobs", "2",
        "--shapes", "2x2",
        "--shapes", "2x4",
        "--max-new-queries", "2",
    ]) == 0

    outcomes = json.loads((output / "outcomes.json").read_text())
    assert outcomes["terminal_count"] == 2
    assert outcomes["success_count"] == 1
    assert outcomes["censored_count"] == 1
    records = {f"{row['rows']}x{row['cols']}": row
               for row in outcomes["queries"]}
    assert records["2x2"]["status"] == "success"
    assert records["2x2"]["compiled_ii"] == 3
    assert records["2x4"]["status"] == "censored"
    assert records["2x4"]["compiled_ii"] is None
    assert "outcome_label" not in records["2x4"]
    assert not Path(records["2x2"]["result_path"]).is_absolute()
    result_path = output / records["2x4"]["result_path"]
    failed_result = json.loads(result_path.read_text())
    assert failed_result["status"] == "censored"
    assert failed_result["censor_reason"] == "mapper_native_search_failed"
    assert failed_result["compiled_ii"] is None


def test_result_resume_fails_closed_if_a_censored_record_has_an_ii(tmp_path):
    source, architecture, fake_opt = _fixture_files(tmp_path)
    output = tmp_path / "collection"
    assert collection.main([
        "--source-manifest", str(source),
        "--architecture", str(architecture),
        "--neura-opt", str(fake_opt),
        "--output-dir", str(output),
        "--shapes", "2x4",
        "--max-new-queries", "1",
    ]) == 0
    outcomes = json.loads((output / "outcomes.json").read_text())
    censored = next(row for row in outcomes["queries"]
                    if row["status"] == "censored")
    result_path = output / censored["result_path"]
    result = json.loads(result_path.read_text())
    result["compiled_ii"] = 1
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        collection.main([
            "--source-manifest", str(source),
            "--architecture", str(architecture),
            "--neura-opt", str(fake_opt),
            "--output-dir", str(output),
            "--prepare-only",
        ])
