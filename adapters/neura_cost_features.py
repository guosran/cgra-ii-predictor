"""Parse an analysis-only initiation-interval lower bound from Neura IR."""

from __future__ import annotations

import re
from typing import Dict, Optional


COST_FEATURE_NAMES = ("rec_mii", "res_mii", "analytical_lower_bound")
FORBIDDEN_MAPPING_FIELDS = (
    "compiled_ii", "mapping_info", "mapping_strategy", "analytical_ii",
    "compute_mii", "mem_mii", "reg_mii", "route_mii", "infeasible",
    "exceeds_max_ii",
)


def parse_integer_attribute(text: str, name: str) -> Optional[int]:
    match = re.search(rf"\b{re.escape(name)} = (-?\d+) : i32", text)
    return int(match.group(1)) if match else None


def parse_cost_features(text: str) -> Optional[Dict[str, int]]:
    """Return the analytical bound only when no mapper label is present."""
    forbidden = tuple(
        name for name in FORBIDDEN_MAPPING_FIELDS
        if re.search(rf"\b{re.escape(name)}\b", text)
    )
    if forbidden:
        raise ValueError(
            "analytical lower-bound artifact contains mapping/label tokens: "
            + ", ".join(forbidden)
        )
    if re.search(r"\banalytical_lower_bound_info\b", text) is None:
        return None
    values = {
        name: parse_integer_attribute(text, name) for name in COST_FEATURE_NAMES
    }
    if any(value is None or value < 0 for value in values.values()):
        return None
    result = {name: int(value) for name, value in values.items()}
    if result["analytical_lower_bound"] != max(
        result["rec_mii"], result["res_mii"]
    ):
        return None
    return result


__all__ = ["parse_cost_features", "parse_integer_attribute"]
