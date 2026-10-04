"""Compact black-box surrogate for one fixed heuristic mapper.

The model deliberately does not approximate placement or routing.  It consumes
only pre-mapper DFG summaries, the requested mapper shape, and analytical
RecMII/ResMII facts, then regresses the mapper's final compiled II directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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

MAPPER_PRESSURE_NAMES = (
    "mapper_bucket_width_per_tile",
    "mapper_bucket_width_per_tile_ii",
    "recurrence_nodes_per_tile",
    "fanin_collision_pressure",
    "fanout_collision_pressure",
    "route_edge_pressure",
)

MAPPER_TRAVERSAL_NAMES = (
    "forward_depth_per_rows",
    "forward_depth_per_columns",
    "semantic_depth_per_rows",
    "semantic_depth_per_columns",
    "mapper_bucket_width_per_rows",
    "mapper_bucket_width_per_columns",
    "mapper_bucket_degree_pressure",
    "mapper_bucket_fanin_collision_pressure",
    "mapper_bucket_fanout_collision_pressure",
    "mapper_bucket_quadratic_occupancy",
)

MAPPER_FEATURE_NAMES = (
    GRAPH_SUMMARY_NAMES + TOPOLOGY_SUMMARY_NAMES + MAPPER_CONTEXT_NAMES
    + MAPPER_PRESSURE_NAMES + MAPPER_TRAVERSAL_NAMES
)


def mapper_feature_names(shape_protocol: str = SHAPE_PROTOCOL_ID) -> Tuple[str, ...]:
    """Return the ordered feature contract for a declared hardware domain."""
    shapes = get_shape_protocol(shape_protocol).mapper_shapes
    context = (
        MAPPER_CONTEXT_NAMES[:9]
        + tuple(f"shape_{rows}x{columns}" for rows, columns in shapes)
        + MAPPER_CONTEXT_NAMES[-6:]
    )
    return (GRAPH_SUMMARY_NAMES + TOPOLOGY_SUMMARY_NAMES + context
            + MAPPER_PRESSURE_NAMES + MAPPER_TRAVERSAL_NAMES)

# Disjoint, mapper-aligned families used for validation-only ablation. Features
# outside these groups form the generic baseline. Every value is available
# before mapping; no family observes placement or mapped artifacts.
MAPPER_ALIGNED_FEATURE_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "alap_bucket": (
        "mapper_bucket_width_per_tile",
        "mapper_bucket_width_per_tile_ii",
        "mapper_bucket_width_per_rows",
        "mapper_bucket_width_per_columns",
        "mapper_bucket_quadratic_occupancy",
    ),
    "recurrence_critical": (
        "mean_is_recurrence_cycle",
        "max_is_recurrence_cycle",
        "log_recurrence_node_count",
        "recurrence_nodes_per_tile",
    ),
    "mapper_traversal_order": (
        "mean_normalized_asap",
        "mean_normalized_reverse_depth",
        "mean_normalized_slack",
        "mean_normalized_position",
        "max_normalized_asap",
        "max_normalized_reverse_depth",
        "max_normalized_slack",
        "max_normalized_position",
        "log_forward_depth",
        "log_semantic_forward_depth",
        "forward_depth_per_rows",
        "forward_depth_per_columns",
        "semantic_depth_per_rows",
        "semantic_depth_per_columns",
    ),
    "producer_user_pressure": (
        "mean_normalized_indegree",
        "mean_normalized_outdegree",
        "max_normalized_indegree",
        "max_normalized_outdegree",
        "log_max_indegree",
        "log_max_outdegree",
        "fanin_collision_pressure",
        "fanout_collision_pressure",
        "mapper_bucket_degree_pressure",
        "mapper_bucket_fanin_collision_pressure",
        "mapper_bucket_fanout_collision_pressure",
    ),
    "shape_capacity_pressure": (
        "normalized_links",
        "normalized_bisection_links",
        "log_nodes_per_tile",
        "log_materialized_nodes_per_tile",
        "log_raw_edges_per_link",
        "log_semantic_edges_per_link",
        "forward_depth_per_max_dimension",
        "forward_width_per_tile",
        "route_edge_pressure",
    ),
}

_grouped_feature_names = tuple(
    name for names in MAPPER_ALIGNED_FEATURE_GROUPS.values() for name in names
)
if len(set(_grouped_feature_names)) != len(_grouped_feature_names):
    raise AssertionError("mapper-aligned feature groups must be disjoint")
if not set(_grouped_feature_names).issubset(MAPPER_FEATURE_NAMES):
    raise AssertionError("mapper-aligned feature group names are invalid")


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
    enabled_feature_names: Optional[Tuple[str, ...]] = None

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
        if self.enabled_feature_names is not None:
            names = tuple(self.enabled_feature_names)
            if not names:
                raise ValueError("enabled_feature_names must not be empty")
            if len(set(names)) != len(names):
                raise ValueError("enabled_feature_names must be unique")
            if not set(names).issubset(mapper_feature_names(self.shape_protocol)):
                raise ValueError("enabled_feature_names contains unknown features")
        return self

    @property
    def model_input_feature_names(self) -> Tuple[str, ...]:
        self.validate()
        names = mapper_feature_names(self.shape_protocol)
        if self.enabled_feature_names is None:
            return names
        enabled = set(self.enabled_feature_names)
        return tuple(name for name in names if name in enabled)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self.validate())
        result["hidden_dimensions"] = list(self.hidden_dimensions)
        if self.enabled_feature_names is not None:
            result["enabled_feature_names"] = list(
                self.model_input_feature_names
            )
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

    # Approximate the pressure seen by the heuristic mapper's ALAP traversal.
    # These are aggregate pre-mapper facts, not placement labels: concurrent
    # bucket width competes for tiles, while shared producers/users compete for
    # routing links.  The analytical II scales the available time slots.
    materialized = [row[11] > 0.5 for row in graph.node_features]
    forward_children = [[] for _ in graph.node_types]
    forward_parents = [[] for _ in graph.node_types]
    for source, target in graph.edges:
        if source < target:
            forward_children[source].append(target)
            forward_parents[target].append(source)
    reverse_materialized_depth = [0] * len(graph.node_types)
    for node in reversed(range(len(graph.node_types))):
        reverse_materialized_depth[node] = max(
            (
                reverse_materialized_depth[user] + int(materialized[user])
                for user in forward_children[node]
            ),
            default=0,
        )
    maximum_reverse_depth = max(reverse_materialized_depth, default=0)
    bucket_widths = [0] * (maximum_reverse_depth + 1)
    bucket_nodes = [[] for _ in bucket_widths]
    for node, reverse_value in enumerate(reverse_materialized_depth):
        if materialized[node]:
            bucket = maximum_reverse_depth - reverse_value
            bucket_widths[bucket] += 1
            bucket_nodes[bucket].append(node)
    maximum_bucket_width = max(bucket_widths, default=0)
    fanin_collisions = sum(
        len(values) * (len(values) - 1) / 2.0
        for values in forward_parents
    )
    fanout_collisions = sum(
        len(values) * (len(values) - 1) / 2.0
        for values in forward_children
    )
    bucket_degree = max((
        sum(len(forward_parents[node]) + len(forward_children[node])
            for node in nodes)
        for nodes in bucket_nodes
    ), default=0)
    bucket_fanin_collisions = max((
        sum(
            len(forward_parents[node]) * (len(forward_parents[node]) - 1)
            / 2.0
            for node in nodes
        )
        for nodes in bucket_nodes
    ), default=0.0)
    bucket_fanout_collisions = max((
        sum(
            len(forward_children[node]) * (len(forward_children[node]) - 1)
            / 2.0
            for node in nodes
        )
        for nodes in bucket_nodes
    ), default=0.0)

    tiles = rows * columns
    links = 2 * (
        rows * max(0, columns - 1) +
        columns * max(0, rows - 1)
    )
    bisection_links = 0 if tiles == 1 else 2 * min(rows, columns)
    routed_capacity = float(max(1.0, links * lower_bound))
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
        maximum_bucket_width / float(tiles),
        maximum_bucket_width / float(max(1.0, tiles * lower_bound)),
        recurrence_count / float(tiles),
        fanin_collisions / routed_capacity,
        fanout_collisions / routed_capacity,
        len(graph.edges) / routed_capacity,
        raw_depth / float(rows),
        raw_depth / float(columns),
        semantic_depth / float(rows),
        semantic_depth / float(columns),
        maximum_bucket_width / float(rows),
        maximum_bucket_width / float(columns),
        bucket_degree / routed_capacity,
        bucket_fanin_collisions / routed_capacity,
        bucket_fanout_collisions / routed_capacity,
        sum(width * width for width in bucket_widths)
        / float(max(1.0, tiles * tiles * lower_bound)),
    )
    if len(values) != len(mapper_feature_names(shape_protocol)):
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
        self.feature_names = mapper_feature_names(self.config.shape_protocol)
        raw_width = len(self.feature_names)
        mean = torch.zeros(raw_width) if feature_mean is None else torch.as_tensor(
            feature_mean, dtype=torch.float32,
        ).detach().clone()
        scale = torch.ones(raw_width) if feature_scale is None else torch.as_tensor(
            feature_scale, dtype=torch.float32,
        ).detach().clone()
        if mean.shape != (raw_width,) or scale.shape != (raw_width,):
            raise ValueError("feature normalization has the wrong width")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("feature normalization must be finite")
        if torch.any(scale <= 0.0):
            raise ValueError("feature scales must be positive")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        selected = set(self.config.model_input_feature_names)
        self._feature_indices = tuple(
            index for index, name in enumerate(self.feature_names)
            if name in selected
        )
        first, second = self.config.hidden_dimensions
        self.regressor = nn.Sequential(
            nn.Linear(len(self._feature_indices), first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, 1),
        )

    def forward(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("mapper feature tensor has the wrong shape")
        if lower_bound.shape != features.shape[:1]:
            raise ValueError("lower_bound tensor has the wrong shape")
        standardized = (
            features.to(dtype=torch.float32) - self.feature_mean
        ) / self.feature_scale
        standardized = standardized[:, self._feature_indices]
        residual = functional.softplus(
            self.regressor(standardized).squeeze(-1)
        )
        prediction = lower_bound.to(
            device=features.device, dtype=torch.float32,
        ) + residual
        return torch.minimum(
            prediction,
            torch.full_like(prediction, self.config.mapper_ii_ceiling),
        )


class DirectMapperIIEnsemble(nn.Module):
    """Average independently trained direct mapper-II regressors."""

    def __init__(
        self,
        member_count: int,
        config: MapperModelConfig = MapperModelConfig(),
    ) -> None:
        super().__init__()
        if isinstance(member_count, bool) or not isinstance(member_count, int):
            raise ValueError("ensemble member_count must be an integer")
        if member_count < 2:
            raise ValueError("ensemble requires at least two members")
        self.config = config.validate()
        self.members = nn.ModuleList(
            DirectMapperIIModel(self.config) for _ in range(member_count)
        )

    def forward(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        predictions = [member(features, lower_bound) for member in self.members]
        return torch.stack(predictions, dim=0).mean(dim=0)


class RidgeMapperIIModel(nn.Module):
    """Deploy a frozen linear residual-II fit with its original float64 math."""

    def __init__(self, config: MapperModelConfig = MapperModelConfig()) -> None:
        super().__init__()
        self.config = config.validate()
        self.feature_names = mapper_feature_names(self.config.shape_protocol)
        width = len(self.feature_names)
        if self.config.enabled_feature_names is not None:
            raise ValueError("ridge checkpoint requires the full feature contract")
        self.register_buffer("feature_mean", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(width, dtype=torch.float64))
        self.register_buffer("coefficients", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("residual_mean", torch.zeros((), dtype=torch.float64))

    def forward(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("mapper feature tensor has the wrong shape")
        if lower_bound.shape != features.shape[:1]:
            raise ValueError("lower_bound tensor has the wrong shape")
        normalized = (
            features.to(dtype=torch.float64) - self.feature_mean
        ) / self.feature_scale
        residual = self.residual_mean + normalized @ self.coefficients
        bound = lower_bound.to(device=features.device, dtype=torch.float64)
        return torch.maximum(
            bound,
            torch.minimum(bound + residual,
                          torch.full_like(bound, self.config.mapper_ii_ceiling)),
        )


class CategoricalMapperIIModel(nn.Module):
    """Predict an integer residual-II label while retaining a ranking score.

    Classes represent ``compiled_ii - lower_bound``.  ``forward`` returns the
    probability-weighted expected II so pairwise and top-1 ranking objectives
    remain differentiable; :meth:`predict_label` returns the discrete II used
    by a categorical deployment.
    """

    def __init__(
        self,
        config: MapperModelConfig = MapperModelConfig(),
        feature_mean: Optional[Sequence[float]] = None,
        feature_scale: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.config = config.validate()
        if not float(self.config.mapper_ii_ceiling).is_integer():
            raise ValueError(
                "categorical mapper II ceiling must be an integer"
            )
        self.feature_names = mapper_feature_names(self.config.shape_protocol)
        raw_width = len(self.feature_names)
        mean = torch.zeros(raw_width) if feature_mean is None else torch.as_tensor(
            feature_mean, dtype=torch.float32,
        ).detach().clone()
        scale = torch.ones(raw_width) if feature_scale is None else torch.as_tensor(
            feature_scale, dtype=torch.float32,
        ).detach().clone()
        if mean.shape != (raw_width,) or scale.shape != (raw_width,):
            raise ValueError("feature normalization has the wrong width")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("feature normalization must be finite")
        if torch.any(scale <= 0.0):
            raise ValueError("feature scales must be positive")
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_scale", scale)
        selected = set(self.config.model_input_feature_names)
        self._feature_indices = tuple(
            index for index, name in enumerate(self.feature_names)
            if name in selected
        )
        # The largest possible residual is ``ceiling - 0``.  Query-specific
        # masks below remove classes that exceed ``ceiling - lower_bound``.
        self.class_count = int(self.config.mapper_ii_ceiling) + 1
        first, second = self.config.hidden_dimensions
        self.classifier = nn.Sequential(
            nn.Linear(len(self._feature_indices), first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, self.class_count),
        )

    def _validated_lower_bound(
        self, features: Tensor, lower_bound: Tensor,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("mapper feature tensor has the wrong shape")
        if lower_bound.shape != features.shape[:1]:
            raise ValueError("lower_bound tensor has the wrong shape")
        values = lower_bound.to(device=features.device, dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("categorical lower bounds must be finite")
        if torch.any(values < 0.0) or torch.any(
            values > self.config.mapper_ii_ceiling
        ):
            raise ValueError("categorical lower bound is outside the II range")
        if not torch.allclose(values, torch.round(values)):
            raise ValueError("categorical lower bounds must be integers")
        return values

    def class_logits(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        """Return residual-class logits masked by each query's II bounds."""
        bounds = self._validated_lower_bound(features, lower_bound)
        standardized = (
            features.to(dtype=torch.float32) - self.feature_mean
        ) / self.feature_scale
        standardized = standardized[:, self._feature_indices]
        logits = self.classifier(standardized)
        residuals = torch.arange(
            self.class_count, device=features.device, dtype=torch.float32,
        )
        valid = residuals.unsqueeze(0) <= (
            self.config.mapper_ii_ceiling - bounds
        ).unsqueeze(1)
        return logits.masked_fill(~valid, -torch.inf)

    def expected_ii_from_logits(
        self, logits: Tensor, lower_bound: Tensor,
    ) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.class_count:
            raise ValueError("categorical mapper logits have the wrong shape")
        if lower_bound.shape != logits.shape[:1]:
            raise ValueError("lower_bound tensor has the wrong shape")
        residuals = torch.arange(
            self.class_count, device=logits.device, dtype=torch.float32,
        )
        expected_residual = (
            torch.softmax(logits, dim=1) * residuals.unsqueeze(0)
        ).sum(dim=1)
        return lower_bound.to(
            device=logits.device, dtype=torch.float32,
        ) + expected_residual

    def forward(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        logits = self.class_logits(features, lower_bound)
        return self.expected_ii_from_logits(logits, lower_bound)

    def predict_label(self, features: Tensor, lower_bound: Tensor) -> Tensor:
        """Return the most likely legal integer compiled-II label."""
        logits = self.class_logits(features, lower_bound)
        bounds = torch.round(lower_bound).to(
            device=logits.device, dtype=torch.int64,
        )
        return bounds + torch.argmax(logits, dim=1)


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
    "CategoricalMapperIIModel",
    "DirectMapperIIModel",
    "DirectMapperIIEnsemble",
    "RidgeMapperIIModel",
    "MAPPER_ALIGNED_FEATURE_GROUPS",
    "MAPPER_CONTEXT_NAMES",
    "MAPPER_FEATURE_NAMES",
    "mapper_feature_names",
    "MAPPER_PRESSURE_NAMES",
    "MAPPER_TRAVERSAL_NAMES",
    "MapperModelConfig",
    "mapper_feature_vector",
    "mapper_ii_loss",
]
