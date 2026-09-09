"""Compact black-box surrogate for one fixed heuristic mapper.

The model deliberately does not approximate placement or routing.  It consumes
only pre-mapper DFG summaries, the requested mapper shape, and analytical
RecMII/ResMII facts, then regresses the mapper's final compiled II directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .dfg import (
    GraphData,
    ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES,
    ROUTE_EXPANDED_OPERATION_TYPES,
)
from .shape_protocol import SHAPE_PROTOCOL_ID, get_shape_protocol


GRAPH_SUMMARY_NAMES = tuple(
    f"log_count_{name}" for name in ROUTE_EXPANDED_OPERATION_TYPES
) + tuple(
    f"mean_{name}" for name in ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES
) + tuple(
    f"max_{name}" for name in ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES
) + (
    "log_node_count",
    "log_raw_edge_count",
    "log_semantic_edge_count",
)

TOPOLOGY_SUMMARY_NAMES = (
    "log_materialized_node_count",
    "log_movement_node_count",
    "log_forward_depth",
    "log_semantic_forward_depth",
    "log_max_forward_width",
    "normalized_max_forward_width",
    "log_source_count",
    "log_sink_count",
    "log_max_indegree",
    "log_max_outdegree",
    "raw_edge_density",
    "semantic_edge_density",
    "log_recurrence_node_count",
)

MAPPER_CONTEXT_NAMES = (
    "normalized_rows",
    "normalized_columns",
    "normalized_tiles",
    "log_aspect_ratio",
    "normalized_links",
    "normalized_bisection_links",
    "normalized_rec_mii",
    "normalized_res_mii",
    "normalized_lower_bound",
) + tuple(
    f"shape_{rows}x{columns}"
    for rows, columns in get_shape_protocol(SHAPE_PROTOCOL_ID).mapper_shapes
) + (
    "log_nodes_per_tile",
    "log_materialized_nodes_per_tile",
    "log_raw_edges_per_link",
    "log_semantic_edges_per_link",
    "forward_depth_per_max_dimension",
    "forward_width_per_tile",
)

MAPPER_FEATURE_NAMES = (
    GRAPH_SUMMARY_NAMES + TOPOLOGY_SUMMARY_NAMES + MAPPER_CONTEXT_NAMES
)


def _forward_topology(
    node_count: int, edges: Sequence[Tuple[int, int]],
) -> Tuple[int, int, int, int, int]:
    """Summarize the source-ordered DAG, excluding loop feedback edges."""
    parents = [[] for _ in range(node_count)]
    children = [[] for _ in range(node_count)]
    for source, target in edges:
        if source < target:
            parents[target].append(source)
            children[source].append(target)
    levels = [1] * node_count
    for node in range(node_count):
        if parents[node]:
            levels[node] = 1 + max(levels[parent] for parent in parents[node])
    widths = [0] * max(levels)
    for level in levels:
        widths[level - 1] += 1
    return (
        max(levels), max(widths),
        sum(not values for values in parents),
        sum(not values for values in children),
        max((len(values) for values in parents), default=0),
    )


@dataclass(frozen=True)
class MapperModelConfig:
    """Architecture and fixed-mapper output contract."""

    hidden_dimensions: Tuple[int, int] = (64, 32)
    mapper_ii_ceiling: float = 20.0
    shape_protocol: str = SHAPE_PROTOCOL_ID

    def validate(self) -> "MapperModelConfig":
        if len(self.hidden_dimensions) != 2 or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 8
            for width in self.hidden_dimensions
        ):
            raise ValueError("hidden_dimensions must contain two integers >= 8")
        if (
            not math.isfinite(float(self.mapper_ii_ceiling)) or
            self.mapper_ii_ceiling <= 0.0
        ):
            raise ValueError("mapper_ii_ceiling must be finite and positive")
        get_shape_protocol(self.shape_protocol)
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self.validate())
        result["hidden_dimensions"] = list(self.hidden_dimensions)
        return result


def mapper_feature_vector(
    graph: GraphData,
    rows: int,
    columns: int,
    rec_mii: float,
    res_mii: float,
    lower_bound: float,
    *,
    mapper_ii_ceiling: float = 20.0,
    shape_protocol: str = SHAPE_PROTOCOL_ID,
) -> Tuple[float, ...]:
    """Summarize one pre-mapper DFG/shape query without mapper-derived input."""
    protocol = get_shape_protocol(shape_protocol)
    protocol.validate_mapper_shape(rows, columns)
    if max(rec_mii, res_mii) != lower_bound:
        raise ValueError("lower_bound must equal max(rec_mii, res_mii)")
    if lower_bound < 0.0 or lower_bound > mapper_ii_ceiling:
        raise ValueError("lower_bound is outside the mapper search interval")
    graph.validate(len(ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES))

    type_counts = [0] * len(ROUTE_EXPANDED_OPERATION_TYPES)
    for node_type in graph.node_types:
        if node_type < 0 or node_type >= len(type_counts):
            raise ValueError("DFG node type is outside the route-expanded vocabulary")
        type_counts[node_type] += 1
    columns_of_features = tuple(zip(*graph.node_features))
    node_count = float(len(graph.node_types))
    means = tuple(sum(values) / node_count for values in columns_of_features)
    maxima = tuple(max(values) for values in columns_of_features)

    raw_depth, forward_width, source_count, sink_count, max_indegree = (
        _forward_topology(len(graph.node_types), graph.edges)
    )
    semantic_edges = graph.semantic_edges or ()
    semantic_depth, _, _, _, _ = _forward_topology(
        len(graph.node_types), semantic_edges,
    )
    raw_outdegree = [0] * len(graph.node_types)
    for source, target in graph.edges:
        if source < target:
            raw_outdegree[source] += 1
    max_outdegree = max(raw_outdegree, default=0)
    materialized_count = sum(row[11] > 0.5 for row in graph.node_features)
    movement_count = sum(row[12] > 0.5 for row in graph.node_features)
    recurrence_count = sum(row[13] > 0.5 for row in graph.node_features)
    possible_edges = max(1, len(graph.node_types) * (len(graph.node_types) - 1))

    tiles = rows * columns
    links = 2 * (
        rows * max(0, columns - 1) +
        columns * max(0, rows - 1)
    )
    bisection_links = 0 if tiles == 1 else 2 * min(rows, columns)
    values = (
        *(math.log1p(count) for count in type_counts),
        *means,
        *maxima,
        math.log1p(len(graph.node_types)),
        math.log1p(len(graph.edges)),
        math.log1p(len(semantic_edges)),
        math.log1p(materialized_count),
        math.log1p(movement_count),
        math.log1p(raw_depth),
        math.log1p(semantic_depth),
        math.log1p(forward_width),
        forward_width / node_count,
        math.log1p(source_count),
        math.log1p(sink_count),
        math.log1p(max_indegree),
        math.log1p(max_outdegree),
        len(graph.edges) / float(possible_edges),
        len(semantic_edges) / float(possible_edges),
        math.log1p(recurrence_count),
        rows / float(protocol.max_mapper_rows),
        columns / float(protocol.max_mapper_cols),
        tiles / float(protocol.max_mapper_tiles),
        math.log(float(columns) / float(rows)),
        links / float(protocol.max_directed_links),
        bisection_links / float(protocol.max_bisection_links),
        rec_mii / mapper_ii_ceiling,
        res_mii / mapper_ii_ceiling,
        lower_bound / mapper_ii_ceiling,
        *(float((rows, columns) == shape) for shape in protocol.mapper_shapes),
        math.log1p(node_count / tiles),
        math.log1p(materialized_count / tiles),
        math.log1p(len(graph.edges) / max(1, links)),
        math.log1p(len(semantic_edges) / max(1, links)),
        raw_depth / float(max(rows, columns)),
        forward_width / float(tiles),
    )
    if len(values) != len(MAPPER_FEATURE_NAMES):
        raise AssertionError("mapper feature contract width changed")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("mapper features must be finite")
    return tuple(float(value) for value in values)


class DirectMapperIIModel(nn.Module):
    """Predict only the final compiled II returned by the fixed mapper."""

    def __init__(
        self,
        config: MapperModelConfig = MapperModelConfig(),
        feature_mean: Optional[Sequence[float]] = None,
        feature_scale: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.config = config.validate()
        width = len(MAPPER_FEATURE_NAMES)
        mean = torch.zeros(width) if feature_mean is None else torch.as_tensor(
            feature_mean, dtype=torch.float32,
        ).detach().clone()
        scale = torch.ones(width) if feature_scale is None else torch.as_tensor(
            feature_scale, dtype=torch.float32,
        ).detach().clone()
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("feature normalization has the wrong width")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("feature normalization must be finite")
        if torch.any(scale <= 0.0):
            raise ValueError("feature scales must be positive")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        first, second = self.config.hidden_dimensions
        self.regressor = nn.Sequential(
            nn.Linear(width, first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, 1),
        )

    def forward(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(MAPPER_FEATURE_NAMES):
            raise ValueError("mapper feature tensor has the wrong shape")
        if lower_bound.shape != features.shape[:1]:
            raise ValueError("lower_bound tensor has the wrong shape")
        standardized = (
            features.to(dtype=torch.float32) - self.feature_mean
        ) / self.feature_scale
        residual = functional.softplus(self.regressor(standardized).squeeze(-1))
        prediction = lower_bound.to(
            device=features.device, dtype=torch.float32,
        ) + residual
        return torch.minimum(
            prediction,
            torch.full_like(prediction, self.config.mapper_ii_ceiling),
        )


def mapper_ii_loss(
    predicted_ii: Tensor,
    target_ii: Tensor,
    better_indices: Optional[Tensor] = None,
    worse_indices: Optional[Tensor] = None,
    *,
    pairwise_weight: float = 0.1,
    pairwise_margin: float = 0.5,
) -> Dict[str, Tensor]:
    """Direct mapper-output regression plus optional within-DFG ordering."""
    if predicted_ii.shape != target_ii.shape or predicted_ii.ndim != 1:
        raise ValueError("predicted and target II tensors must be equal vectors")
    point = functional.smooth_l1_loss(predicted_ii, target_ii)
    pairwise = predicted_ii.sum() * 0.0
    if better_indices is not None or worse_indices is not None:
        if better_indices is None or worse_indices is None:
            raise ValueError("both pairwise index tensors are required")
        if better_indices.shape != worse_indices.shape:
            raise ValueError("pairwise index tensors must have equal shape")
        if better_indices.numel():
            gaps = predicted_ii[worse_indices] - predicted_ii[better_indices]
            pairwise = functional.relu(pairwise_margin - gaps).mean()
    total = point + pairwise_weight * pairwise
    return {"total": total, "point": point, "pairwise": pairwise}


__all__ = [
    "DirectMapperIIModel",
    "MAPPER_CONTEXT_NAMES",
    "MAPPER_FEATURE_NAMES",
    "MapperModelConfig",
    "mapper_feature_vector",
    "mapper_ii_loss",
]
