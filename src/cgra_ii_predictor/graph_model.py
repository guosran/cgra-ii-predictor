"""Small joint DFG/CGRA graph model for censored-aware shape selection.

This module intentionally depends only on PyTorch rather than PyTorch
Geometric.  Graphs in the current study are small, so dense padded adjacency
matrices keep the implementation auditable and deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError:  # Keep the base NumPy package importable without Model 2.
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    functional = None  # type: ignore


OPERATION_TYPES = (
    "<pad>", "<unknown>", "<pe>", "constant", "grant_once",
    "grant_always", "grant_predicate", "loop_control", "phi", "phi_start",
    "add", "sub", "mul", "div", "rem", "fadd", "fsub", "fmul", "fdiv",
    "fmul_fadd", "fadd_fadd", "vfmul", "or", "and", "xor", "not", "shl",
    "icmp", "fcmp", "sel", "cast", "sext", "zext", "alloca", "gep",
    "load", "store", "memset", "load_indexed", "store_indexed",
)
OPERATION_TO_ID = {name: index for index, name in enumerate(OPERATION_TYPES)}
TRANSPARENT_OPERATIONS = frozenset({"data_mov", "ctrl_mov", "reserve", "yield"})
MEMORY_OPERATIONS = frozenset({
    "load", "store", "memset", "load_indexed", "store_indexed",
})
POINTER_OPERATIONS = frozenset({"alloca", "gep"})
CONTROL_OPERATIONS = frozenset({
    "grant_predicate", "loop_control", "phi", "phi_start", "not", "icmp",
    "fcmp", "sel",
})
DFG_NODE_FEATURE_NAMES = (
    "normalized_indegree", "normalized_outdegree", "normalized_asap",
    "normalized_reverse_depth", "normalized_slack", "normalized_position",
    "is_source", "is_sink", "is_memory", "is_pointer", "is_control",
)
CGRA_NODE_FEATURE_NAMES = (
    "normalized_x", "normalized_y", "normalized_rows", "normalized_columns",
    "normalized_degree", "is_memory_tile", "is_left_boundary",
    "is_top_boundary", "is_right_boundary", "is_bottom_boundary", "bias",
)
CANDIDATE_CONTEXT_NAMES = (
    "normalized_rec_mii", "normalized_res_mii", "normalized_lower_bound",
    "normalized_rows", "normalized_columns", "normalized_tiles",
    "log_aspect_ratio", "normalized_links", "normalized_memory_tiles",
    "normalized_bisection_links",
)
CROSS_ATTENTION_CONTEXT_NAMES = (
    "normalized_operation_pressure", "normalized_peak_pe_load",
    "normalized_pe_load_entropy",
)
ROUTING_CONTEXT_NAMES = (
    "normalized_mean_edge_distance", "normalized_max_edge_distance",
    "normalized_routing_demand_per_link",
)


@dataclass(frozen=True)
class GraphData:
    """A compact directed graph with categorical and scalar node features."""

    node_types: Tuple[int, ...]
    node_features: Tuple[Tuple[float, ...], ...]
    edges: Tuple[Tuple[int, int], ...]

    def validate(self, scalar_feature_count: int) -> "GraphData":
        if not self.node_types:
            raise ValueError("graph must contain at least one node")
        if len(self.node_types) != len(self.node_features):
            raise ValueError("graph node type/feature counts differ")
        if any(len(row) != scalar_feature_count for row in self.node_features):
            raise ValueError("graph scalar feature width mismatch")
        node_count = len(self.node_types)
        if any(
            source < 0 or target < 0 or source >= node_count or
            target >= node_count or source == target
            for source, target in self.edges
        ):
            raise ValueError("graph contains an invalid edge")
        return self


@dataclass(frozen=True)
class Model2Config:
    hidden_dimension: int = 64
    message_passing_layers: int = 3
    dropout: float = 0.10
    mapper_ii_ceiling: float = 20.0
    listwise_temperature: float = 1.0
    success_loss_weight: float = 1.0
    residual_loss_weight: float = 1.0
    listwise_loss_weight: float = 1.0
    strict_tiebreak_loss_weight: float = 1.0
    interaction_mode: str = "pooled"
    candidate_set_layers: int = 2
    candidate_set_heads: int = 4
    discrete_ii_loss_weight: float = 1.0
    discrete_success_threshold: float = 0.5
    discrete_ii_decision: str = "map"
    placement_loss_weight: float = 0.0

    def validate(self) -> "Model2Config":
        if self.hidden_dimension < 8:
            raise ValueError("hidden_dimension must be at least eight")
        if self.message_passing_layers < 1:
            raise ValueError("message_passing_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.interaction_mode not in {
            "pooled", "cross_attention", "routing_set_attention",
            "discrete_routing_set", "discrete_pointwise",
            "strict_set_classifier",
        }:
            raise ValueError(
                "interaction_mode must be pooled, cross_attention, "
                "routing_set_attention, discrete_routing_set, "
                "discrete_pointwise, or strict_set_classifier"
            )
        if self.interaction_mode in {
            "routing_set_attention", "discrete_routing_set",
            "strict_set_classifier",
        }:
            if self.candidate_set_layers < 1:
                raise ValueError("candidate_set_layers must be positive")
            if self.candidate_set_heads < 1:
                raise ValueError("candidate_set_heads must be positive")
            if self.hidden_dimension % self.candidate_set_heads != 0:
                raise ValueError(
                    "hidden_dimension must be divisible by candidate_set_heads"
                )
        if self.interaction_mode in {
            "discrete_routing_set", "discrete_pointwise",
        } and (
            not float(self.mapper_ii_ceiling).is_integer()
        ):
            raise ValueError(
                "discrete modes require an integer mapper_ii_ceiling"
            )
        if self.interaction_mode in {
            "discrete_routing_set", "discrete_pointwise",
        }:
            if not 0.0 <= self.discrete_success_threshold <= 1.0:
                raise ValueError(
                    "discrete_success_threshold must be in [0, 1]"
                )
            if self.discrete_ii_decision not in {
                "map", "round", "floor", "ceil",
            }:
                raise ValueError(
                    "discrete_ii_decision must be map, round, floor, or ceil"
                )
        if (
            not math.isfinite(float(self.placement_loss_weight)) or
            self.placement_loss_weight < 0.0
        ):
            raise ValueError("placement_loss_weight must be finite and nonnegative")
        if self.placement_loss_weight > 0.0 and self.interaction_mode not in {
            "cross_attention", "routing_set_attention",
            "discrete_routing_set", "discrete_pointwise",
            "strict_set_classifier",
        }:
            raise ValueError(
                "placement supervision requires a cross-attention mode"
            )
        for name in (
            "mapper_ii_ceiling", "listwise_temperature",
            "success_loss_weight", "residual_loss_weight",
            "listwise_loss_weight", "strict_tiebreak_loss_weight",
            "discrete_ii_loss_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self.validate())
        # Preserve the exact configuration contract of existing pooled-model
        # checkpoints.  New architectures identify themselves explicitly.
        if self.interaction_mode == "pooled":
            values.pop("interaction_mode")
        if self.interaction_mode not in {
            "routing_set_attention", "discrete_routing_set",
            "strict_set_classifier",
        }:
            values.pop("candidate_set_layers")
            values.pop("candidate_set_heads")
        if self.interaction_mode not in {
            "discrete_routing_set", "discrete_pointwise",
        }:
            values.pop("discrete_ii_loss_weight")
            values.pop("discrete_success_threshold")
            values.pop("discrete_ii_decision")
        if self.placement_loss_weight == 0.0:
            values.pop("placement_loss_weight")
        return values


def _meaningful_roots(
    value: str, kinds: Mapping[str, str], operands: Mapping[str, Sequence[str]],
    seen: Optional[set] = None,
) -> List[str]:
    if value not in kinds:
        return []
    if kinds[value] not in TRANSPARENT_OPERATIONS:
        return [value]
    visited = set() if seen is None else set(seen)
    if value in visited:
        return []
    visited.add(value)
    roots: List[str] = []
    for parent in operands.get(value, ()):
        roots.extend(_meaningful_roots(parent, kinds, operands, visited))
    return roots


def parse_neura_dfg(text: str) -> GraphData:
    """Parse result-producing Neura operations into a semantic dependency DAG."""
    order: List[str] = []
    kinds: Dict[str, str] = {}
    operands: Dict[str, List[str]] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if not match:
            continue
        value, expression = match.groups()
        kind_match = re.search(r'"?neura\.([a-z_]+)', expression)
        if not kind_match:
            continue
        kinds[value] = kind_match.group(1)
        operands[value] = re.findall(r"%[A-Za-z0-9_]+", expression)
        order.append(value)
    values = [
        value for value in order
        if kinds[value] not in TRANSPARENT_OPERATIONS
    ]
    if not values:
        raise ValueError("Neura DFG contains no materialized operations")
    index = {value: position for position, value in enumerate(values)}
    edge_set = set()
    for target in values:
        for operand in operands.get(target, ()):
            for source in _meaningful_roots(operand, kinds, operands):
                if source in index and source != target:
                    edge_set.add((index[source], index[target]))
    edges = tuple(sorted(edge_set))
    node_count = len(values)
    parents: List[List[int]] = [[] for _ in values]
    children: List[List[int]] = [[] for _ in values]
    for source, target in edges:
        parents[target].append(source)
        children[source].append(target)
    asap = [1] * node_count
    for node in range(node_count):
        if parents[node]:
            asap[node] = 1 + max(asap[parent] for parent in parents[node])
    reverse_depth = [1] * node_count
    for node in reversed(range(node_count)):
        if children[node]:
            reverse_depth[node] = 1 + max(
                reverse_depth[child] for child in children[node]
            )
    makespan = max(asap)
    alap = [makespan - reverse_depth[node] + 1 for node in range(node_count)]
    node_features: List[Tuple[float, ...]] = []
    denominator = float(max(1, node_count - 1))
    for node, value in enumerate(values):
        kind = kinds[value]
        node_features.append((
            len(parents[node]) / float(max(1, node_count)),
            len(children[node]) / float(max(1, node_count)),
            asap[node] / float(max(1, makespan)),
            reverse_depth[node] / float(max(1, makespan)),
            max(0, alap[node] - asap[node]) / float(max(1, makespan)),
            node / denominator,
            float(not parents[node]),
            float(not children[node]),
            float(kind in MEMORY_OPERATIONS),
            float(kind in POINTER_OPERATIONS or kind == "load"),
            float(kind in CONTROL_OPERATIONS),
        ))
    graph = GraphData(
        node_types=tuple(
            OPERATION_TO_ID.get(kinds[value], OPERATION_TO_ID["<unknown>"])
            for value in values
        ),
        node_features=tuple(node_features),
        edges=edges,
    )
    return graph.validate(len(DFG_NODE_FEATURE_NAMES))


def parse_neura_mapped_placements(
    text: str, source_graph: GraphData, rows: int, columns: int,
) -> Tuple[int, ...]:
    """Extract one mapper-assigned PE index for every materialized DFG node."""
    if not 1 <= rows <= 4 or not 1 <= columns <= 4:
        raise ValueError("mapped placement shape is outside the pinned mesh")
    mapped_types: List[int] = []
    placements: List[int] = []
    for line in text.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if not match:
            continue
        expression = match.group(2)
        kind_match = re.search(r'"?neura\.([a-z_]+)', expression)
        if not kind_match or kind_match.group(1) in TRANSPARENT_OPERATIONS:
            continue
        kind = kind_match.group(1)
        mapped_types.append(
            OPERATION_TO_ID.get(kind, OPERATION_TO_ID["<unknown>"])
        )
        tile_locations = []
        for location in re.findall(r"\{[^{}]*\}", expression):
            if 'resource = "tile"' not in location:
                continue
            x_match = re.search(r"x = (\d+) : i32", location)
            y_match = re.search(r"y = (\d+) : i32", location)
            if x_match is not None and y_match is not None:
                tile_locations.append((
                    int(x_match.group(1)), int(y_match.group(1)),
                ))
        unique_locations = set(tile_locations)
        if len(unique_locations) != 1:
            raise ValueError(
                "mapped materialized operation lacks one stable tile location"
            )
        x, y = next(iter(unique_locations))
        if not 0 <= x < columns or not 0 <= y < rows:
            raise ValueError("mapped tile location is outside the candidate shape")
        placements.append(y * columns + x)
    if tuple(mapped_types) != source_graph.node_types:
        raise ValueError("mapped operation sequence differs from the source DFG")
    return tuple(placements)


def make_cgra_graph(rows: int, columns: int) -> GraphData:
    """Create an oriented rectangular prefix of the pinned 4x4 mesh."""
    if isinstance(rows, bool) or isinstance(columns, bool):
        raise ValueError("CGRA dimensions must be integers")
    if not isinstance(rows, int) or not isinstance(columns, int):
        raise ValueError("CGRA dimensions must be integers")
    if not 1 <= rows <= 4 or not 1 <= columns <= 4:
        raise ValueError("CGRA dimensions must be within the pinned 4x4 mesh")
    coordinates = [
        (x, y) for y in range(rows) for x in range(columns)
    ]
    index = {coordinate: position for position, coordinate in enumerate(coordinates)}
    edges = set()
    degrees = [0] * len(coordinates)
    for position, (x, y) in enumerate(coordinates):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (x + dx, y + dy)
            if neighbor in index:
                edges.add((position, index[neighbor]))
                degrees[position] += 1
    features = []
    for position, (x, y) in enumerate(coordinates):
        features.append((
            x / float(max(1, columns - 1)),
            y / float(max(1, rows - 1)),
            rows / 4.0,
            columns / 4.0,
            degrees[position] / 4.0,
            float(x == 0 or y == 0),
            float(x == 0),
            float(y == 0),
            float(x == columns - 1),
            float(y == rows - 1),
            1.0,
        ))
    graph = GraphData(
        node_types=tuple(OPERATION_TO_ID["<pe>"] for _ in coordinates),
        node_features=tuple(features),
        edges=tuple(sorted(edges)),
    )
    return graph.validate(len(CGRA_NODE_FEATURE_NAMES))


def candidate_context(
    rows: int, columns: int, rec_mii: float, res_mii: float,
    lower_bound: float, mapper_ii_ceiling: float = 20.0,
) -> Tuple[float, ...]:
    """Return explicit oriented geometry and analytical context."""
    if max(rec_mii, res_mii) != lower_bound:
        raise ValueError("lower_bound must equal max(rec_mii, res_mii)")
    tiles = rows * columns
    links = 2 * (
        rows * max(0, columns - 1) + columns * max(0, rows - 1)
    )
    memory_tiles = rows + columns - 1
    bisection = 0 if tiles == 1 else 2 * min(rows, columns)
    return (
        rec_mii / mapper_ii_ceiling,
        res_mii / mapper_ii_ceiling,
        lower_bound / mapper_ii_ceiling,
        rows / 4.0,
        columns / 4.0,
        tiles / 16.0,
        math.log(float(columns) / float(rows)),
        links / 48.0,
        memory_tiles / 7.0,
        bisection / 8.0,
    )


def _require_torch() -> None:
    if torch is None or nn is None or functional is None:
        raise RuntimeError(
            "Model 2 requires PyTorch; install the optional graph dependency"
        )


def pad_graphs(
    graphs: Sequence[GraphData], scalar_feature_count: int, device: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad small graphs into dense deterministic tensors."""
    _require_torch()
    if not graphs:
        raise ValueError("cannot batch an empty graph sequence")
    for graph in graphs:
        graph.validate(scalar_feature_count)
    maximum_nodes = max(len(graph.node_types) for graph in graphs)
    batch = len(graphs)
    node_types = torch.zeros((batch, maximum_nodes), dtype=torch.long, device=device)
    scalars = torch.zeros(
        (batch, maximum_nodes, scalar_feature_count),
        dtype=torch.float32, device=device,
    )
    adjacency = torch.zeros(
        (batch, maximum_nodes, maximum_nodes),
        dtype=torch.float32, device=device,
    )
    mask = torch.zeros((batch, maximum_nodes), dtype=torch.bool, device=device)
    for graph_index, graph in enumerate(graphs):
        count = len(graph.node_types)
        node_types[graph_index, :count] = torch.tensor(
            graph.node_types, dtype=torch.long, device=device,
        )
        scalars[graph_index, :count] = torch.tensor(
            graph.node_features, dtype=torch.float32, device=device,
        )
        mask[graph_index, :count] = True
        for source, target in graph.edges:
            adjacency[graph_index, source, target] = 1.0
    return node_types, scalars, adjacency, mask


def pad_shortest_path_distances(
    graphs: Sequence[GraphData], device: Any,
) -> Tensor:
    """Return padded directed shortest-path distances for small graphs."""
    _require_torch()
    if not graphs:
        raise ValueError("cannot batch an empty graph sequence")
    maximum_nodes = max(len(graph.node_types) for graph in graphs)
    matrices = torch.zeros(
        (len(graphs), maximum_nodes, maximum_nodes),
        dtype=torch.float32, device=device,
    )
    for graph_index, graph in enumerate(graphs):
        node_count = len(graph.node_types)
        distances = [[math.inf] * node_count for _ in range(node_count)]
        for node in range(node_count):
            distances[node][node] = 0.0
        for source, target in graph.edges:
            distances[source][target] = 1.0
        for intermediate in range(node_count):
            for source in range(node_count):
                prefix = distances[source][intermediate]
                if not math.isfinite(prefix):
                    continue
                for target in range(node_count):
                    candidate = prefix + distances[intermediate][target]
                    if candidate < distances[source][target]:
                        distances[source][target] = candidate
        if any(
            not math.isfinite(distances[source][target])
            for source in range(node_count)
            for target in range(node_count)
        ):
            raise ValueError("routing graph must be strongly connected")
        matrices[graph_index, :node_count, :node_count] = torch.tensor(
            distances, dtype=torch.float32, device=device,
        )
    return matrices


if nn is not None:
    class DirectedGraphEncoder(nn.Module):
        """Directed message passing followed by masked mean/max pooling."""

        def __init__(
            self, scalar_feature_count: int, hidden_dimension: int,
            layers: int, dropout: float,
        ) -> None:
            super().__init__()
            self.scalar_feature_count = scalar_feature_count
            self.type_embedding = nn.Embedding(
                len(OPERATION_TYPES), hidden_dimension, padding_idx=0,
            )
            self.input_projection = nn.Linear(
                hidden_dimension + scalar_feature_count, hidden_dimension,
            )
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dimension * 3, hidden_dimension * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dimension * 2, hidden_dimension),
                )
                for _ in range(layers)
            ])
            self.normalizations = nn.ModuleList([
                nn.LayerNorm(hidden_dimension) for _ in range(layers)
            ])
            self.readout = nn.Sequential(
                nn.Linear(hidden_dimension * 2, hidden_dimension),
                nn.GELU(),
                nn.LayerNorm(hidden_dimension),
            )

        def encode_nodes(
            self, graphs: Sequence[GraphData],
        ) -> Tuple[Tensor, Tensor, Tensor]:
            """Return node states, padding mask, and directed adjacency."""
            device = self.type_embedding.weight.device
            node_types, scalars, adjacency, mask = pad_graphs(
                graphs, self.scalar_feature_count, device,
            )
            hidden = self.input_projection(torch.cat((
                self.type_embedding(node_types), scalars,
            ), dim=-1))
            float_mask = mask.unsqueeze(-1).to(hidden.dtype)
            hidden = hidden * float_mask
            for layer, normalization in zip(self.layers, self.normalizations):
                incoming = torch.bmm(adjacency.transpose(1, 2), hidden)
                outgoing = torch.bmm(adjacency, hidden)
                update = layer(torch.cat((hidden, incoming, outgoing), dim=-1))
                hidden = normalization(hidden + update) * float_mask
            return hidden, mask, adjacency

        def pool_nodes(self, hidden: Tensor, mask: Tensor) -> Tensor:
            """Pool encoded nodes without discarding padding semantics."""
            float_mask = mask.unsqueeze(-1).to(hidden.dtype)
            counts = float_mask.sum(dim=1).clamp_min(1.0)
            mean = hidden.sum(dim=1) / counts
            negative = torch.finfo(hidden.dtype).min
            maximum = hidden.masked_fill(~mask.unsqueeze(-1), negative).max(dim=1).values
            return self.readout(torch.cat((mean, maximum), dim=-1))

        def forward(self, graphs: Sequence[GraphData]) -> Tensor:
            hidden, mask, _ = self.encode_nodes(graphs)
            return self.pool_nodes(hidden, mask)


    class CandidateSetBlock(nn.Module):
        """Let candidate shapes compare themselves before final ranking."""

        def __init__(self, hidden: int, heads: int, dropout: float) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True,
            )
            self.attention_dropout = nn.Dropout(dropout)
            self.attention_normalization = nn.LayerNorm(hidden)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden, hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            )
            self.feed_forward_dropout = nn.Dropout(dropout)
            self.feed_forward_normalization = nn.LayerNorm(hidden)

        def forward(self, candidates: Tensor) -> Tensor:
            attended, _ = self.attention(
                candidates, candidates, candidates, need_weights=False,
            )
            candidates = self.attention_normalization(
                candidates + self.attention_dropout(attended)
            )
            update = self.feed_forward(candidates)
            return self.feed_forward_normalization(
                candidates + self.feed_forward_dropout(update)
            )


    class JointGraphShapeModel(nn.Module):
        """Encode DFG and oriented CGRA graphs, then predict risk and II."""

        def __init__(self, config: Model2Config = Model2Config()) -> None:
            super().__init__()
            self.config = config.validate()
            hidden = self.config.hidden_dimension
            self.dfg_encoder = DirectedGraphEncoder(
                len(DFG_NODE_FEATURE_NAMES), hidden,
                self.config.message_passing_layers, self.config.dropout,
            )
            self.cgra_encoder = DirectedGraphEncoder(
                len(CGRA_NODE_FEATURE_NAMES), hidden,
                self.config.message_passing_layers, self.config.dropout,
            )
            interaction_width = hidden * 4 + len(CANDIDATE_CONTEXT_NAMES)
            if self.config.interaction_mode in {
                "cross_attention", "routing_set_attention",
                "discrete_routing_set", "discrete_pointwise",
                "strict_set_classifier",
            }:
                self.cross_dfg_query = nn.Linear(hidden, hidden)
                self.cross_cgra_key = nn.Linear(hidden, hidden)
                self.cross_cgra_value = nn.Linear(hidden, hidden)
                self.cross_node_fusion = nn.Sequential(
                    nn.Linear(hidden * 4, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                )
                self.cross_readout = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                )
                interaction_width += hidden + len(CROSS_ATTENTION_CONTEXT_NAMES)
                if self.config.interaction_mode in {
                    "routing_set_attention", "discrete_routing_set",
                    "discrete_pointwise", "strict_set_classifier",
                }:
                    interaction_width += len(ROUTING_CONTEXT_NAMES)
            self.interaction = nn.Sequential(
                nn.Linear(interaction_width, hidden * 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.success_head = nn.Linear(hidden, 1)
            self.residual_head = nn.Linear(hidden, 1)
            if self.config.interaction_mode in {
                "routing_set_attention", "discrete_routing_set",
                "strict_set_classifier",
            }:
                self.candidate_set_blocks = nn.ModuleList([
                    CandidateSetBlock(
                        hidden, self.config.candidate_set_heads,
                        self.config.dropout,
                    )
                    for _ in range(self.config.candidate_set_layers)
                ])
                self.candidate_set_normalization = nn.LayerNorm(hidden)
            if self.config.interaction_mode in {
                "routing_set_attention", "strict_set_classifier",
            }:
                self.rank_head = nn.Linear(hidden, 1)
            elif self.config.interaction_mode in {
                "discrete_routing_set", "discrete_pointwise",
            }:
                self.discrete_ii_head = nn.Linear(
                    hidden, int(self.config.mapper_ii_ceiling),
                )

        def _candidate_conditioned_interaction(
            self, dfg_nodes: Tensor, dfg_mask: Tensor,
            cgra_nodes: Tensor, cgra_mask: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            """Attend every operation to the PEs of each candidate shape.

            Besides a learned cross-graph summary, return differentiable load
            statistics.  The latter exposes operation/PE pressure that global
            mean/max pooling intentionally removes.
            """
            hidden_width = dfg_nodes.shape[-1]
            query = self.cross_dfg_query(dfg_nodes)
            key = self.cross_cgra_key(cgra_nodes)
            value = self.cross_cgra_value(cgra_nodes)
            logits = torch.einsum(
                "bnh,cmh->bcnm", query, key,
            ) / math.sqrt(float(hidden_width))
            logits = logits.masked_fill(
                ~cgra_mask[None, :, None, :],
                torch.finfo(logits.dtype).min,
            )
            attention = torch.softmax(logits, dim=-1)
            valid_operations = dfg_mask[:, None, :, None].to(attention.dtype)
            attention = attention * valid_operations
            attended = torch.einsum(
                "bcnm,cmh->bcnh", attention, value,
            )
            batch, candidates, operation_count = attended.shape[:3]
            operations = dfg_nodes[:, None, :, :].expand(
                batch, candidates, operation_count, hidden_width,
            )
            local = self.cross_node_fusion(torch.cat((
                operations, attended, operations * attended,
                torch.abs(operations - attended),
            ), dim=-1))
            operation_mask = dfg_mask[:, None, :, None]
            float_operation_mask = operation_mask.to(local.dtype)
            counts = float_operation_mask.sum(dim=2).clamp_min(1.0)
            mean = (local * float_operation_mask).sum(dim=2) / counts
            maximum = local.masked_fill(
                ~operation_mask, torch.finfo(local.dtype).min,
            ).max(dim=2).values
            cross_summary = self.cross_readout(torch.cat((mean, maximum), dim=-1))

            pe_load = attention.sum(dim=2)
            pe_mask = cgra_mask[None, :, :]
            float_pe_mask = pe_mask.to(pe_load.dtype)
            operation_counts = dfg_mask.sum(dim=1).to(pe_load.dtype)
            pe_counts = cgra_mask.sum(dim=1).to(pe_load.dtype)
            mean_load = (
                operation_counts[:, None] / pe_counts[None, :].clamp_min(1.0)
            )
            pressure = (
                torch.log1p(mean_load) / math.log(129.0)
            ).clamp(max=1.0)
            peak_load = pe_load.masked_fill(~pe_mask, 0.0).max(dim=-1).values
            normalized_peak = (
                peak_load / mean_load.clamp_min(1e-6) / 16.0
            ).clamp(max=1.0)
            load_probability = (
                pe_load / operation_counts[:, None, None].clamp_min(1.0)
            ) * float_pe_mask
            entropy = -(
                load_probability * load_probability.clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropy_denominator = pe_counts.log().clamp_min(1.0)
            normalized_entropy = entropy / entropy_denominator[None, :]
            cross_context = torch.stack((
                pressure, normalized_peak, normalized_entropy,
            ), dim=-1)
            return cross_summary, cross_context, attention, logits

        def _routing_context(
            self, attention: Tensor, dfg_adjacency: Tensor,
            cgra_graphs: Sequence[GraphData],
        ) -> Tensor:
            """Estimate route length and link pressure under soft placement."""
            distances = pad_shortest_path_distances(
                cgra_graphs, attention.device,
            )
            projected = torch.einsum(
                "bcnm,cmp->bcnp", attention, distances,
            )
            pair_distances = torch.einsum(
                "bcnp,bcjp->bcnj", projected, attention,
            )
            edge_mask = dfg_adjacency[:, None, :, :] > 0.0
            edge_distances = pair_distances * edge_mask.to(
                pair_distances.dtype
            )
            edge_counts = edge_mask.sum(dim=(-1, -2)).clamp_min(1).to(
                pair_distances.dtype
            )
            mean_distance = edge_distances.sum(dim=(-1, -2)) / edge_counts
            max_distance = edge_distances.masked_fill(
                ~edge_mask, 0.0,
            ).flatten(start_dim=2).max(dim=-1).values
            link_counts = torch.tensor([
                max(1.0, len(graph.edges) / 2.0) for graph in cgra_graphs
            ], dtype=pair_distances.dtype, device=pair_distances.device)
            demand_per_link = (
                edge_distances.sum(dim=(-1, -2)) / link_counts[None, :]
            )
            normalized_demand = (
                torch.log1p(demand_per_link) / math.log(129.0)
            ).clamp(max=1.0)
            return torch.stack((
                (mean_distance / 6.0).clamp(max=1.0),
                (max_distance / 6.0).clamp(max=1.0),
                normalized_demand,
            ), dim=-1)

        def forward(
            self, dfg_graphs: Sequence[GraphData],
            cgra_graphs: Sequence[GraphData], context: Tensor,
        ) -> Dict[str, Tensor]:
            if context.ndim != 3 or context.shape[-1] != len(
                CANDIDATE_CONTEXT_NAMES
            ):
                raise ValueError("candidate context has the wrong shape")
            if context.shape[0] != len(dfg_graphs):
                raise ValueError("DFG batch and candidate context differ")
            if context.shape[1] != len(cgra_graphs):
                raise ValueError("CGRA candidate count and context differ")
            device = next(self.parameters()).device
            context = context.to(device=device, dtype=torch.float32)
            if self.config.interaction_mode in {
                "cross_attention", "routing_set_attention",
                "discrete_routing_set", "discrete_pointwise",
                "strict_set_classifier",
            }:
                dfg_nodes, dfg_mask, dfg_adjacency = (
                    self.dfg_encoder.encode_nodes(dfg_graphs)
                )
                cgra_nodes, cgra_mask, _ = self.cgra_encoder.encode_nodes(
                    cgra_graphs
                )
                dfg = self.dfg_encoder.pool_nodes(dfg_nodes, dfg_mask)
                cgra = self.cgra_encoder.pool_nodes(cgra_nodes, cgra_mask)
                (
                    cross_summary, cross_context, cross_attention,
                    cross_attention_logits,
                ) = (
                    self._candidate_conditioned_interaction(
                        dfg_nodes, dfg_mask, cgra_nodes, cgra_mask,
                    )
                )
                routing_context = (
                    self._routing_context(
                        cross_attention, dfg_adjacency, cgra_graphs,
                    )
                    if self.config.interaction_mode in {
                        "routing_set_attention", "discrete_routing_set",
                        "discrete_pointwise", "strict_set_classifier",
                    }
                    else None
                )
            else:
                dfg = self.dfg_encoder(dfg_graphs)
                cgra = self.cgra_encoder(cgra_graphs)
                cross_summary = None
                cross_context = None
                routing_context = None
                cross_attention_logits = None
            batch, candidates = context.shape[:2]
            dfg = dfg[:, None, :].expand(batch, candidates, -1)
            cgra = cgra[None, :, :].expand(batch, candidates, -1)
            joint_parts = [
                dfg, cgra, dfg * cgra, torch.abs(dfg - cgra), context,
            ]
            if cross_summary is not None and cross_context is not None:
                joint_parts.extend((cross_summary, cross_context))
            if routing_context is not None:
                joint_parts.append(routing_context)
            joint = torch.cat(joint_parts, dim=-1)
            hidden = self.interaction(joint)
            success_logits = self.success_head(hidden).squeeze(-1)
            residual = functional.softplus(self.residual_head(hidden).squeeze(-1))
            lower_bound = context[..., CANDIDATE_CONTEXT_NAMES.index(
                "normalized_lower_bound"
            )] * self.config.mapper_ii_ceiling
            predicted_ii = lower_bound + residual
            success_probability = torch.sigmoid(success_logits)
            timeout_cost = self.config.mapper_ii_ceiling + 1.0
            expected_cost = (
                success_probability * predicted_ii.clamp(max=timeout_cost) +
                (1.0 - success_probability) * timeout_cost
            )
            ranking_logits = None
            ii_class_logits = None
            predicted_ii_class = None
            selection_cost = expected_cost
            ranked = None
            if self.config.interaction_mode in {
                "routing_set_attention", "discrete_routing_set",
                "strict_set_classifier",
            }:
                ranked = hidden
                for block in self.candidate_set_blocks:
                    ranked = block(ranked)
                ranked = self.candidate_set_normalization(ranked)
            if self.config.interaction_mode == "routing_set_attention":
                rank_adjustment = self.rank_head(ranked).squeeze(-1)
                ranking_logits = -expected_cost + rank_adjustment
                selection_cost = -ranking_logits
            elif self.config.interaction_mode == "strict_set_classifier":
                ranking_logits = self.rank_head(ranked).squeeze(-1)
                selection_cost = -ranking_logits
            elif self.config.interaction_mode in {
                "discrete_routing_set", "discrete_pointwise",
            }:
                discrete_features = (
                    ranked
                    if self.config.interaction_mode == "discrete_routing_set"
                    else hidden
                )
                ii_class_logits = self.discrete_ii_head(discrete_features)
                class_values = torch.arange(
                    1, int(self.config.mapper_ii_ceiling) + 1,
                    dtype=ii_class_logits.dtype, device=ii_class_logits.device,
                )
                valid_classes = (
                    class_values[None, None, :] >=
                    torch.ceil(lower_bound - 1e-6)[..., None]
                )
                ii_class_logits = ii_class_logits.masked_fill(
                    ~valid_classes, torch.finfo(ii_class_logits.dtype).min,
                )
                ii_probabilities = torch.softmax(ii_class_logits, dim=-1)
                predicted_ii = (
                    ii_probabilities * class_values[None, None, :]
                ).sum(dim=-1)
                predicted_ii_variance = (
                    ii_probabilities * (
                        class_values[None, None, :] - predicted_ii[..., None]
                    ).square()
                ).sum(dim=-1)
                predicted_ii_std = predicted_ii_variance.clamp_min(0.0).sqrt()
                map_ii_class = (
                    ii_class_logits.argmax(dim=-1) + 1
                ).to(predicted_ii.dtype)
                if self.config.discrete_ii_decision == "map":
                    predicted_ii_class = map_ii_class
                elif self.config.discrete_ii_decision == "round":
                    predicted_ii_class = torch.floor(predicted_ii + 0.5)
                elif self.config.discrete_ii_decision == "floor":
                    predicted_ii_class = torch.floor(predicted_ii)
                else:
                    predicted_ii_class = torch.ceil(predicted_ii)
                predicted_ii_class = predicted_ii_class.clamp(
                    min=1.0, max=self.config.mapper_ii_ceiling,
                )
                expected_cost = (
                    success_probability * predicted_ii +
                    (1.0 - success_probability) * timeout_cost
                )
                ranking_logits = -expected_cost
                selection_cost = expected_cost
                if self.config.interaction_mode == "discrete_routing_set":
                    rows = torch.round(
                        context[..., CANDIDATE_CONTEXT_NAMES.index(
                            "normalized_rows"
                        )] * 4.0
                    )
                    columns = torch.round(
                        context[..., CANDIDATE_CONTEXT_NAMES.index(
                            "normalized_columns"
                        )] * 4.0
                    )
                    identity_tiebreak = (
                        rows * columns * 100.0 + rows * 10.0 + columns
                    )
                    predicted_safe = (
                        success_probability >=
                        self.config.discrete_success_threshold
                    )
                    fallback = success_probability == success_probability.max(
                        dim=1, keepdim=True
                    ).values
                    eligible = torch.where(
                        predicted_safe.any(dim=1, keepdim=True),
                        predicted_safe, fallback,
                    )
                    discrete_cost = (
                        predicted_ii_class * 10000.0 + identity_tiebreak
                    )
                    selection_cost = discrete_cost.masked_fill(
                        ~eligible, 1_000_000.0,
                    )
            result = {
                "success_logits": success_logits,
                "success_probability": success_probability,
                "predicted_residual": residual,
                "predicted_ii": predicted_ii,
                "lower_bound": lower_bound,
                "expected_cost": expected_cost,
                "selection_cost": selection_cost,
            }
            if cross_context is not None:
                result["cross_attention_context"] = cross_context
            if cross_attention_logits is not None:
                result["placement_logits"] = cross_attention_logits
            if routing_context is not None and ranking_logits is not None:
                result["routing_context"] = routing_context
                result["ranking_logits"] = ranking_logits
            if ii_class_logits is not None and predicted_ii_class is not None:
                result["ii_class_logits"] = ii_class_logits
                result["ii_class_probabilities"] = ii_probabilities
                result["predicted_ii_mean"] = predicted_ii
                result["predicted_ii_mode"] = map_ii_class
                result["predicted_ii_class"] = predicted_ii_class
                result["predicted_ii_std"] = predicted_ii_std
                if self.config.interaction_mode == "discrete_pointwise":
                    result["predicted_residual"] = (
                        predicted_ii - lower_bound
                    ).clamp_min(0.0)
            return result


else:  # pragma: no cover - exercised only in NumPy-only installations.
    class JointGraphShapeModel:  # type: ignore
        def __init__(self, config: Model2Config = Model2Config()) -> None:
            del config
            _require_torch()


def model2_loss(
    output: Mapping[str, Tensor], success_target: Tensor,
    residual_target: Tensor, optimal_ii_target: Tensor, oracle_index: Tensor,
    config: Model2Config, placement_target: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Combine the configured candidate objectives.

    ``strict_set_classifier`` deliberately optimizes only the deterministic
    oracle shape cross-entropy.  The otherwise shared success and residual
    heads stay in the forward contract so this diagnostic architecture can be
    evaluated by the same pipeline without pretending its auxiliary outputs
    were trained.
    """
    _require_torch()
    config.validate()
    logits = output["success_logits"]
    if logits.shape != success_target.shape or logits.shape != residual_target.shape:
        raise ValueError("Model 2 target tensors have inconsistent shapes")
    success_target = success_target.to(logits.device, dtype=logits.dtype)
    residual_target = residual_target.to(logits.device, dtype=logits.dtype)
    optimal_ii_target = optimal_ii_target.to(logits.device, dtype=torch.bool)
    oracle_index = oracle_index.to(logits.device, dtype=torch.long)
    if optimal_ii_target.shape != logits.shape:
        raise ValueError("optimal-II target tensor has an inconsistent shape")
    success_loss = functional.binary_cross_entropy_with_logits(
        logits, success_target,
    )
    successful = success_target > 0.5
    if successful.any():
        residual_loss = functional.smooth_l1_loss(
            output["predicted_residual"][successful],
            residual_target[successful],
        )
    else:
        residual_loss = logits.sum() * 0.0
    discrete_ii_loss = logits.sum() * 0.0
    placement_loss = logits.sum() * 0.0
    if config.placement_loss_weight > 0.0:
        if placement_target is None:
            raise ValueError(
                "positive placement_loss_weight requires placement targets"
            )
        placement_logits = output.get("placement_logits")
        if placement_logits is None:
            raise ValueError("model output lacks placement logits")
        placement_target = placement_target.to(
            placement_logits.device, dtype=torch.long,
        )
        if placement_target.shape != placement_logits.shape[:-1]:
            raise ValueError("placement target tensor has an inconsistent shape")
        supervised_placements = placement_target >= 0
        if supervised_placements.any():
            placement_loss = functional.cross_entropy(
                placement_logits[supervised_placements],
                placement_target[supervised_placements],
            )
    if "ii_class_logits" in output and successful.any():
        target_ii = (
            output["lower_bound"] + residual_target
        )[successful]
        rounded_target_ii = target_ii.round()
        class_count = output["ii_class_logits"].shape[-1]
        if torch.any(torch.abs(target_ii - rounded_target_ii) > 1e-5):
            raise ValueError("discrete-II targets must be integers")
        if torch.any(
            (rounded_target_ii < 1) | (rounded_target_ii > class_count)
        ):
            raise ValueError("discrete-II target is outside the class range")
        discrete_ii_loss = functional.cross_entropy(
            output["ii_class_logits"][successful],
            rounded_target_ii.to(dtype=torch.long) - 1,
        )
    eligible = optimal_ii_target.any(dim=1)
    if config.interaction_mode == "discrete_pointwise":
        optimal_ii_loss = logits.sum() * 0.0
        strict_tiebreak_loss = logits.sum() * 0.0
        listwise_loss = logits.sum() * 0.0
    elif eligible.any():
        ranking_logits = output.get(
            "ranking_logits", -output["expected_cost"],
        )
        listwise_logits = (
            ranking_logits[eligible] / config.listwise_temperature
        )
        log_probabilities = functional.log_softmax(listwise_logits, dim=1)
        optimal_log_mass = torch.logsumexp(
            log_probabilities.masked_fill(
                ~optimal_ii_target[eligible],
                torch.finfo(log_probabilities.dtype).min,
            ),
            dim=1,
        )
        optimal_ii_loss = -optimal_log_mass.mean()
        strict_tiebreak_loss = functional.cross_entropy(
            listwise_logits, oracle_index[eligible],
        )
        listwise_loss = (
            optimal_ii_loss +
            config.strict_tiebreak_loss_weight * strict_tiebreak_loss
        )
    else:
        optimal_ii_loss = logits.sum() * 0.0
        strict_tiebreak_loss = logits.sum() * 0.0
        listwise_loss = logits.sum() * 0.0
    if config.interaction_mode == "strict_set_classifier":
        optimal_ii_loss = logits.sum() * 0.0
        listwise_loss = strict_tiebreak_loss
        total = strict_tiebreak_loss
    elif config.interaction_mode == "discrete_pointwise":
        total = (
            config.success_loss_weight * success_loss +
            config.residual_loss_weight * residual_loss +
            config.discrete_ii_loss_weight * discrete_ii_loss
        )
    else:
        total = (
            config.success_loss_weight * success_loss +
            config.residual_loss_weight * residual_loss +
            config.listwise_loss_weight * listwise_loss +
            config.discrete_ii_loss_weight * discrete_ii_loss
        )
    if config.placement_loss_weight > 0.0:
        total = total + config.placement_loss_weight * placement_loss
    return {
        "total": total,
        "success": success_loss,
        "residual": residual_loss,
        "discrete_ii": discrete_ii_loss,
        "placement": placement_loss,
        "listwise": listwise_loss,
        "listwise_optimal_ii": optimal_ii_loss,
        "listwise_strict_tiebreak": strict_tiebreak_loss,
    }


def censored_top1_metrics(
    queries: Sequence[Sequence[Mapping[str, Any]]], prediction_key: str,
    *, minimum_successful_candidates: int = 2,
    mapper_ii_ceiling: float = 20.0,
) -> Dict[str, Any]:
    """Evaluate shape choice without assigning a numeric II to censorship."""
    if minimum_successful_candidates < 1:
        raise ValueError("minimum_successful_candidates must be positive")
    if not math.isfinite(mapper_ii_ceiling) or mapper_ii_ceiling <= 0.0:
        raise ValueError("mapper_ii_ceiling must be finite and positive")
    evaluated_topk = (1, 2, 3)
    strict_correct = {k: 0 for k in evaluated_topk}
    transpose_correct = {k: 0 for k in evaluated_topk}
    optimal_ii_correct = {k: 0 for k in evaluated_topk}
    any_success = {k: 0 for k in evaluated_topk}
    selected_success = 0
    eligible = 0
    total_regret = total_penalized_regret = 0.0
    exclusions: Dict[str, int] = {}
    selections: Dict[str, Dict[str, Any]] = {}

    def identity(row: Mapping[str, Any]) -> Tuple[int, int, int, str]:
        rows = int(row["rows"])
        columns = int(row["columns"])
        return rows * columns, rows, columns, str(row["candidate_id"])

    def transpose_identity(row: Mapping[str, Any]) -> Tuple[int, int]:
        return tuple(sorted((int(row["rows"]), int(row["columns"]))))

    for query_rows in queries:
        if not query_rows:
            exclusions["empty_query"] = exclusions.get("empty_query", 0) + 1
            continue
        query_id = str(query_rows[0]["ranking_query_id"])
        if len({str(row["ranking_query_id"]) for row in query_rows}) != 1:
            raise ValueError("candidate sequence spans multiple ranking queries")
        successful = [row for row in query_rows if row.get("status") == "success"]
        if len(successful) < minimum_successful_candidates:
            exclusions["too_few_successful_candidates"] = (
                exclusions.get("too_few_successful_candidates", 0) + 1
            )
            continue
        if any(prediction_key not in row for row in query_rows):
            exclusions["missing_prediction"] = exclusions.get(
                "missing_prediction", 0
            ) + 1
            continue
        oracle = min(successful, key=lambda row: (
            float(row["compiled_ii"]), *identity(row),
        ))
        ranked = sorted(query_rows, key=lambda row: (
            float(row[prediction_key]), *identity(row),
        ))
        selected = ranked[0]
        eligible += 1
        is_success = selected.get("status") == "success"
        selected_success += int(is_success)
        exact = str(selected["candidate_id"]) == str(oracle["candidate_id"])
        oracle_ii = float(oracle["compiled_ii"])
        oracle_transpose = transpose_identity(oracle)
        for k in evaluated_topk:
            topk = ranked[:k]
            strict_correct[k] += int(any(
                str(row["candidate_id"]) == str(oracle["candidate_id"])
                for row in topk
            ))
            transpose_correct[k] += int(any(
                transpose_identity(row) == oracle_transpose for row in topk
            ))
            optimal_ii_correct[k] += int(any(
                row.get("status") == "success" and
                float(row["compiled_ii"]) == oracle_ii
                for row in topk
            ))
            any_success[k] += int(any(
                row.get("status") == "success" for row in topk
            ))
        regret: Optional[float] = None
        if is_success:
            regret = float(selected["compiled_ii"]) - oracle_ii
            total_regret += regret
            total_penalized_regret += regret
        else:
            total_penalized_regret += (
                mapper_ii_ceiling + 1.0 - float(oracle["compiled_ii"])
            )
        selections[query_id] = {
            "oracle_candidate_id": oracle["candidate_id"],
            "oracle_shape": f"{oracle['rows']}x{oracle['columns']}",
            "oracle_compiled_ii": oracle["compiled_ii"],
            "selected_candidate_id": selected["candidate_id"],
            "selected_shape": f"{selected['rows']}x{selected['columns']}",
            "selected_status": selected.get("status"),
            "strict_top1_correct": exact,
            "transpose_equivalent_top1_correct": (
                transpose_identity(selected) == oracle_transpose
            ),
            "top3_candidate_ids": [
                row["candidate_id"] for row in ranked[:3]
            ],
            "top3_shapes": [
                f"{row['rows']}x{row['columns']}" for row in ranked[:3]
            ],
            "compiled_ii_regret": regret,
        }
    topk_metrics = {
        f"strict_top{k}_accuracy": (
            strict_correct[k] / eligible if eligible else None
        )
        for k in evaluated_topk
    }
    topk_metrics.update({
        f"transpose_equivalent_top{k}_accuracy": (
            transpose_correct[k] / eligible if eligible else None
        )
        for k in evaluated_topk
    })
    topk_metrics.update({
        f"optimal_ii_top{k}_rate": (
            optimal_ii_correct[k] / eligible if eligible else None
        )
        for k in evaluated_topk
    })
    topk_metrics.update({
        f"any_success_top{k}_rate": (
            any_success[k] / eligible if eligible else None
        )
        for k in evaluated_topk
    })
    return {
        "status": "ok" if eligible else "unavailable_no_eligible_queries",
        "population": "all_declared_shapes_with_censorship_categorical",
        "minimum_successful_candidates": minimum_successful_candidates,
        "censored_selection_penalty_ii": mapper_ii_ceiling + 1.0,
        "eligible_query_count": eligible,
        "shape_metric_role": "downstream_diagnostic_only",
        "shape_equivalence": "transpose_equivalent",
        "strict_correct_query_count": strict_correct[1],
        "transpose_equivalent_correct_query_count": transpose_correct[1],
        "optimal_ii_query_count": optimal_ii_correct[1],
        "optimal_ii_rate": (
            optimal_ii_correct[1] / eligible if eligible else None
        ),
        **topk_metrics,
        "selected_success_query_count": selected_success,
        "selected_success_rate": selected_success / eligible if eligible else None,
        "mean_regret_over_successful_selections": (
            total_regret / selected_success if selected_success else None
        ),
        "mean_timeout_penalized_regret": (
            total_penalized_regret / eligible if eligible else None
        ),
        "excluded_queries": dict(sorted(exclusions.items())),
        "queries": selections,
    }


__all__ = [
    "CANDIDATE_CONTEXT_NAMES", "CGRA_NODE_FEATURE_NAMES",
    "CROSS_ATTENTION_CONTEXT_NAMES",
    "DFG_NODE_FEATURE_NAMES", "ROUTING_CONTEXT_NAMES", "GraphData",
    "JointGraphShapeModel",
    "Model2Config", "OPERATION_TYPES", "candidate_context",
    "censored_top1_metrics", "make_cgra_graph", "model2_loss",
    "pad_graphs", "pad_shortest_path_distances", "parse_neura_dfg",
    "parse_neura_mapped_placements",
]
