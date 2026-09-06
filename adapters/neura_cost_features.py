"""Parse analysis-only recurrence and resource lower bounds from Neura IR."""

from __future__ import annotations

import re
from typing import Dict, Optional


COST_FEATURE_NAMES = ("rec_mii", "res_mii")
FORBIDDEN_MAPPING_FIELDS = (
    "compiled_ii", "mapping_info", "mapping_strategy", "analytical_ii",
    "compute_mii", "mem_mii", "reg_mii", "route_mii", "infeasible",
    "exceeds_max_ii",
)


def parse_integer_attribute(text: str, name: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i32", text)
    return int(match.group(1)) if match else None


def parse_cost_features(text: str) -> Optional[Dict[str, int]]:
    """Return RecMII/ResMII only when the artifact contains no mapper label."""
    forbidden = tuple(
        name for name in FORBIDDEN_MAPPING_FIELDS
        if re.search(rf"\b{re.escape(name)}\b", text)
    )
    if forbidden:
        raise ValueError(
            "analysis-only Rec/Res artifact contains mapping/label tokens: "
            + ", ".join(forbidden)
        )
    if re.search(r"\brec_res_mii_info\b", text) is None:
        return None
    values = {
        name: parse_integer_attribute(text, name) for name in COST_FEATURE_NAMES
    }
    if any(value is None or value < 0 for value in values.values()):
        return None
    return {name: int(value) for name, value in values.items()}


__all__ = ["parse_cost_features", "parse_integer_attribute"]
