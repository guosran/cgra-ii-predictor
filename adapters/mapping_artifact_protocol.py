"""Dependency-free helpers shared by mapper collection and model tooling.

Keep native collection runnable on a fresh worker host.  In particular, this
module must not import PyTorch, NumPy, or the predictor model package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Dict


FAILURE_MARKERS = (
    b"[MapToAcceleratorPass] Mapping failed for all target II values.",
    b"[MapToAcceleratorPass] Empty target II range [",
)


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


def mapper_input_identity(
    text: str, *, normalize_static_shapes: bool = False,
) -> str:
    """Hash mapper input, optionally merging static-size-only variants."""
    normalized = re.sub(
        r'(amoeba\.source_task_body_sha256\s*=\s*")[0-9a-f]{64}("\s*)',
        r"\1<source-body>\2", text,
    )
    normalized = re.sub(
        r"\bfunc\.func\s+@[A-Za-z0-9_.$-]+_dfg\b",
        "func.func @task_dfg", normalized,
    )
    if normalize_static_shapes:
        normalized = re.sub(
            r"\b(memref|tensor)<([^>]+)>",
            lambda match: (
                f"{match.group(1)}<" +
                re.sub(r"\d+x", "Dx", match.group(2)) + ">"
            ), normalized,
        )
        normalized = re.sub(
            r"\b(lower_bound_value|upper_bound_value)\s*=\s*-?\d+\s*:\s*index",
            r"\1 = <bound> : index", normalized,
        )
    return hashlib.sha256(normalized.encode()).hexdigest()


def parse_single_mapping(text: str, rows: int, cols: int) -> Dict[str, int]:
    """Parse and validate one native Neura mapping artifact."""
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


def outcome_label(result: dict | None, stderr_tail: bytes = b"") -> str:
    """Return success/native_failure/unknown for one native mapper query."""
    if result is None:
        return "unknown"
    if result.get("status") == "success":
        if result.get("compiled_ii") is None:
            raise ValueError("successful mapping lacks compiled II")
        return "success"
    if (result.get("status") == "censored" and
            result.get("compiled_ii") is None and
            result.get("censor_reason") == "mapper_native_search_failed" and
            result.get("analysis", {}).get("status") == "success" and
            isinstance(result.get("native_exit_code"), int) and
            result["native_exit_code"] != 0 and
            any(marker in stderr_tail for marker in FAILURE_MARKERS)):
        return "native_failure"
    return "unknown"


__all__ = [
    "FAILURE_MARKERS",
    "canonical_json_sha256",
    "mapper_input_identity",
    "outcome_label",
    "parse_single_mapping",
    "sha256_file",
]
