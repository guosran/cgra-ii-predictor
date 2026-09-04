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

    def validate(self) -> "Model2Config":
        if self.hidden_dimension < 8:
            raise ValueError("hidden_dimension must be at least eight")
        if self.message_passing_layers < 1:
            raise ValueError("message_passing_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.interaction_mode not in {"pooled", "cross_attention"}:
            raise ValueError(
                "interaction_mode must be pooled or cross_attention"
            )
        for name in (
            "mapper_ii_ceiling", "listwise_temperature",
            "success_loss_weight", "residual_loss_weight",
            "listwise_loss_weight", "strict_tiebreak_loss_weight",
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
        ) -> Tuple[Tensor, Tensor]:
            """Return message-passed node states and their padding mask."""
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
            return hidden, mask

        def pool_nodes(self, hidden: Tensor, mask: Tensor) -> Tensor:
            """Pool encoded nodes without discarding padding semantics."""
            float_mask = mask.unsqueeze(-1).to(hidden.dtype)
            counts = float_mask.sum(dim=1).clamp_min(1.0)
            mean = hidden.sum(dim=1) / counts
            negative = torch.finfo(hidden.dtype).min
            maximum = hidden.masked_fill(~mask.unsqueeze(-1), negative).max(dim=1).values
            return self.readout(torch.cat((mean, maximum), dim=-1))

        def forward(self, graphs: Sequence[GraphData]) -> Tensor:
            return self.pool_nodes(*self.encode_nodes(graphs))


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
            if self.config.interaction_mode == "cross_attention":
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
                interaction_width += (
                    hidden + len(CROSS_ATTENTION_CONTEXT_NAMES)
                )
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

        def _candidate_conditioned_interaction(
            self, dfg_nodes: Tensor, dfg_mask: Tensor,
            cgra_nodes: Tensor, cgra_mask: Tensor,
        ) -> Tuple[Tensor, Tensor]:
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
            return cross_summary, cross_context

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
            if self.config.interaction_mode == "cross_attention":
                dfg_nodes, dfg_mask = self.dfg_encoder.encode_nodes(dfg_graphs)
                cgra_nodes, cgra_mask = self.cgra_encoder.encode_nodes(cgra_graphs)
                dfg = self.dfg_encoder.pool_nodes(dfg_nodes, dfg_mask)
                cgra = self.cgra_encoder.pool_nodes(cgra_nodes, cgra_mask)
                cross_summary, cross_context = (
                    self._candidate_conditioned_interaction(
                        dfg_nodes, dfg_mask, cgra_nodes, cgra_mask,
                    )
                )
            else:
                dfg = self.dfg_encoder(dfg_graphs)
                cgra = self.cgra_encoder(cgra_graphs)
                cross_summary = None
                cross_context = None
            batch, candidates = context.shape[:2]
            dfg = dfg[:, None, :].expand(batch, candidates, -1)
            cgra = cgra[None, :, :].expand(batch, candidates, -1)
            joint_parts = [
                dfg, cgra, dfg * cgra, torch.abs(dfg - cgra), context,
            ]
            if cross_summary is not None and cross_context is not None:
                joint_parts.extend((cross_summary, cross_context))
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
            result = {
                "success_logits": success_logits,
                "success_probability": success_probability,
                "predicted_residual": residual,
                "predicted_ii": predicted_ii,
                "expected_cost": expected_cost,
            }
            if cross_context is not None:
                result["cross_attention_context"] = cross_context
            return result


else:  # pragma: no cover - exercised only in NumPy-only installations.
    class JointGraphShapeModel:  # type: ignore
        def __init__(self, config: Model2Config = Model2Config()) -> None:
            del config
            _require_torch()


def model2_loss(
    output: Mapping[str, Tensor], success_target: Tensor,
    residual_target: Tensor, optimal_ii_target: Tensor, oracle_index: Tensor,
    config: Model2Config,
) -> Dict[str, Tensor]:
    """Combine binary risk, successful-II, and candidate-set Top-1 losses."""
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
    eligible = optimal_ii_target.any(dim=1)
    if eligible.any():
        listwise_logits = (
            -output["expected_cost"][eligible] / config.listwise_temperature
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
    total = (
        config.success_loss_weight * success_loss +
        config.residual_loss_weight * residual_loss +
        config.listwise_loss_weight * listwise_loss
    )
    return {
        "total": total,
        "success": success_loss,
        "residual": residual_loss,
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
    strict_correct = optimal_ii_correct = selected_success = 0
    eligible = 0
    total_regret = total_penalized_regret = 0.0
    exclusions: Dict[str, int] = {}
    selections: Dict[str, Dict[str, Any]] = {}

    def identity(row: Mapping[str, Any]) -> Tuple[int, int, int, str]:
        rows = int(row["rows"])
        columns = int(row["columns"])
        return rows * columns, rows, columns, str(row["candidate_id"])

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
        selected = min(query_rows, key=lambda row: (
            float(row[prediction_key]), *identity(row),
        ))
        eligible += 1
        is_success = selected.get("status") == "success"
        selected_success += int(is_success)
        exact = str(selected["candidate_id"]) == str(oracle["candidate_id"])
        strict_correct += int(exact)
        regret: Optional[float] = None
        if is_success:
            regret = float(selected["compiled_ii"]) - float(oracle["compiled_ii"])
            total_regret += regret
            optimal_ii_correct += int(regret == 0.0)
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
            "compiled_ii_regret": regret,
        }
    return {
        "status": "ok" if eligible else "unavailable_no_eligible_queries",
        "population": "all_declared_shapes_with_censorship_categorical",
        "minimum_successful_candidates": minimum_successful_candidates,
        "censored_selection_penalty_ii": mapper_ii_ceiling + 1.0,
        "eligible_query_count": eligible,
        "strict_correct_query_count": strict_correct,
        "strict_top1_accuracy": strict_correct / eligible if eligible else None,
        "optimal_ii_query_count": optimal_ii_correct,
        "optimal_ii_rate": optimal_ii_correct / eligible if eligible else None,
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
    "DFG_NODE_FEATURE_NAMES", "GraphData", "JointGraphShapeModel",
    "Model2Config", "OPERATION_TYPES", "candidate_context",
    "censored_top1_metrics", "make_cgra_graph", "model2_loss",
    "pad_graphs", "parse_neura_dfg",
]
