#!/usr/bin/env python3
"""Build an audited per-query oracle from real Neura mapper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amoeba_cost_catalog import load_candidate_manifest, sha256_file  # noqa: E402


QueryKey = Tuple[str, int, int]


def parse_mapped_argument(value: str) -> Tuple[QueryKey, Path]:
    identity, separator, raw_path = value.partition("=")
    parts = identity.split(",")
    if not separator or len(parts) != 2 or "x" not in parts[1]:
        raise argparse.ArgumentTypeError(
            "mapped artifact must be TASK,ROWSxCOLS=PATH"
        )
    task = parts[0]
    raw_rows, raw_cols = parts[1].split("x", 1)
    try:
        rows, cols = int(raw_rows), int(raw_cols)
    except ValueError as error:
        raise argparse.ArgumentTypeError("mapped shape must be integer ROWSxCOLS") from error
    path = Path(raw_path).resolve()
    if not task or rows <= 0 or cols <= 0 or not path.is_file():
        raise argparse.ArgumentTypeError("mapped artifact identity or path is invalid")
    return (task, rows, cols), path


def parse_single_mapping(text: str, rows: int, cols: int) -> Dict[str, int]:
    mappings = re.findall(r"mapping_info\s*=\s*\{([^{}]+)\}", text)
    if len(mappings) != 1:
        raise ValueError("oracle artifact must contain exactly one mapped kernel")
    body = mappings[0]

    def integer(name: str) -> int:
        match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i32", body)
        if match is None:
            raise ValueError(f"mapping_info lacks {name}")
        return int(match.group(1))

    result = {
        "compiled_ii": integer("compiled_ii"),
        "rec_mii": integer("rec_mii"),
        "res_mii": integer("res_mii"),
        "x_tiles": integer("x_tiles"),
        "y_tiles": integer("y_tiles"),
    }
    if result["compiled_ii"] < max(result["rec_mii"], result["res_mii"], 1):
        raise ValueError("compiled II is below the analytical lower bound")
    if (result["y_tiles"], result["x_tiles"]) != (rows, cols):
        raise ValueError("mapped artifact x/y override differs from query shape")
    coordinates = [
        (int(x), int(y)) for x, y in re.findall(
            r"\bx = (-?\d+) : i32, y = (-?\d+) : i32", text,
        )
    ]
    if not coordinates:
        raise ValueError("mapped artifact has no physical placement coordinates")
    if any(x < 0 or x >= cols or y < 0 or y >= rows for x, y in coordinates):
        raise ValueError("mapped placement is outside the requested tile array")
    result["placement_coordinate_count"] = len(coordinates)
    return result


def build_oracle(
    candidate_path: Path, mapped_paths: Mapping[QueryKey, Path],
) -> Dict[str, Any]:
    manifest = load_candidate_manifest(candidate_path)
    if set(mapped_paths) != set(manifest["queries"]):
        raise ValueError("mapped artifact set must exactly cover cost queries")
    entries = []
    for task, rows, cols in manifest["queries"]:
        path = mapped_paths[(task, rows, cols)]
        facts = parse_single_mapping(path.read_text(), rows, cols)
        entries.append({
            "task": task,
            "mapper_tile_rows": rows,
            "mapper_tile_cols": cols,
            "status": "success",
            "compiled_ii": facts["compiled_ii"],
            "rec_mii": facts["rec_mii"],
            "res_mii": facts["res_mii"],
            "lower_bound": max(facts["rec_mii"], facts["res_mii"]),
            "mapped_artifact_path": str(path),
            "mapped_artifact_sha256": sha256_file(path),
            "verified_x_tiles": facts["x_tiles"],
            "verified_y_tiles": facts["y_tiles"],
            "verified_placement_coordinate_count": (
                facts["placement_coordinate_count"]
            ),
        })
    return {
        "schema": "cgra-ii-amoeba-query-oracle",
        "function": manifest["header"]["function"],
        "candidate_manifest_sha256": manifest["manifest_sha256"],
        "query_count": len(entries),
        "label_source": "real_neura_heuristic_mapper_artifacts",
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mapped", action="append", type=parse_mapped_argument, required=True,
        metavar="TASK,ROWSxCOLS=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapped = dict(args.mapped)
    if len(mapped) != len(args.mapped):
        raise ValueError("duplicate mapped query identity")
    oracle = build_oracle(args.manifest.resolve(), mapped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(oracle, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "query_count": oracle["query_count"],
        "sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
