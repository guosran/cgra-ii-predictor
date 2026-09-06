"""External wire identifiers required by the current Amoeba executable.

These values belong to Amoeba's JSON interface, not to predictor model
iterations. Change them only when both repositories are migrated together.
"""

CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates-v2"
COST_SCHEMA = "amoeba-task-shape-cost-v2"
SCORE_SCHEMA = "amoeba-analytical-task-scores-v2"
SEARCH_SCOPE = "static-shape-only-v2"
SHAPE_POLICY = "static-rectangles-v2"


__all__ = [
    "CANDIDATE_SCHEMA", "COST_SCHEMA", "SCORE_SCHEMA", "SEARCH_SCOPE",
    "SHAPE_POLICY",
]
