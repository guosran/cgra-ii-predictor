"""External wire identifiers required by the current Amoeba executable.

These values belong to Amoeba's JSON interface, not to predictor model
iterations. Change them only when both repositories are migrated together.
"""

CANDIDATE_SCHEMA = "amoeba-analytical-task-candidates"
COST_SCHEMA = "amoeba-task-shape-cost"
SCORE_SCHEMA = "amoeba-analytical-task-scores"
SEARCH_SCOPE = "static-shape-concurrent-fit"
SHAPE_POLICY = "static-oriented-rectangles"
SPATIAL_CAPACITY_POLICY = "all-tasks-simultaneous-exact-pack"
SCORE_MODEL = "static-shape-compute-bottleneck"
SOURCE_TASK_BODY_SHA_ATTR = "amoeba.source_task_body_sha256"


__all__ = [
    "CANDIDATE_SCHEMA", "COST_SCHEMA", "SCORE_SCHEMA", "SEARCH_SCOPE",
    "SHAPE_POLICY", "SPATIAL_CAPACITY_POLICY", "SCORE_MODEL",
    "SOURCE_TASK_BODY_SHA_ATTR",
]
