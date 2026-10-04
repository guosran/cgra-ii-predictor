"""Parse Neura DFGs into the compact representation used by the predictor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


OPERATION_TYPES = (
    "<pad>", "<unknown>", "<pe>", "constant", "grant_once",
    "grant_always", "grant_predicate", "loop_control", "phi", "phi_start",
    "add", "sub", "mul", "div", "rem", "fadd", "fsub", "fmul", "fdiv",
    "fmul_fadd", "fadd_fadd", "vfmul", "or", "and", "xor", "not", "shl",
    "icmp", "fcmp", "sel", "cast", "sext", "zext", "alloca", "gep",
    "load", "store", "memset", "load_indexed", "store_indexed",
)
OPERATION_TO_ID = {name: index for index, name in enumerate(OPERATION_TYPES)}
ROUTE_EXPANDED_OPERATION_TYPES = OPERATION_TYPES + (
    "data_mov", "ctrl_mov", "reserve", "yield", "return",
    "counter", "fneg", "fmax", "fmin", "gather", "br", "return_void",
    "return_value", "vmul", "vadd", "vfadd", "vector_reduce_add",
    "mul_add", "extract_predicate", "true_steer", "false_steer", "carry",
    "merge", "invariant", "fused_op",
)
ROUTE_EXPANDED_OPERATION_TO_ID = {
    name: index for index, name in enumerate(ROUTE_EXPANDED_OPERATION_TYPES)
}
TRANSPARENT_OPERATIONS = frozenset({"data_mov", "ctrl_mov", "reserve", "yield"})
MEMORY_OPERATIONS = frozenset({
    "load", "store", "memset", "load_indexed", "store_indexed", "gather",
})
POINTER_OPERATIONS = frozenset({"alloca", "gep"})
CONTROL_OPERATIONS = frozenset({
    "grant_predicate", "loop_control", "phi", "phi_start", "not", "icmp",
    "fcmp", "sel", "counter", "extract_predicate", "true_steer",
    "false_steer", "carry", "merge", "invariant",
})
DFG_NODE_FEATURE_NAMES = (
    "normalized_indegree", "normalized_outdegree", "normalized_asap",
    "normalized_reverse_depth", "normalized_slack", "normalized_position",
    "is_source", "is_sink", "is_memory", "is_pointer", "is_control",
)
ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES = DFG_NODE_FEATURE_NAMES + (
    "is_materialized", "is_data_movement", "is_recurrence_cycle",
)


@dataclass(frozen=True)
class GraphData:
    """A compact directed graph with categorical and scalar node features."""

    node_types: Tuple[int, ...]
    node_features: Tuple[Tuple[float, ...], ...]
    edges: Tuple[Tuple[int, int], ...]
    semantic_edges: Optional[Tuple[Tuple[int, int], ...]] = None

    def validate(self, scalar_feature_count: int) -> "GraphData":
        if not self.node_types:
            raise ValueError("graph must contain at least one node")
        if len(self.node_types) != len(self.node_features):
            raise ValueError("graph node type/feature counts differ")
        if any(len(row) != scalar_feature_count for row in self.node_features):
            raise ValueError("graph scalar feature width mismatch")
        node_count = len(self.node_types)
        all_edges = self.edges + (self.semantic_edges or ())
        if any(
            source < 0 or target < 0 or source >= node_count or
            target >= node_count or source == target
            for source, target in all_edges
        ):
            raise ValueError("graph contains an invalid edge")
        return self
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


def parse_neura_route_expanded_dfg(text: str) -> GraphData:
    """Preserve value-movement structure and loop-carried control feedback.

    The mapper places materialized operations on PEs but also routes every
    ``data_mov`` through links/registers. The semantic projection collapses
    those operations and cannot represent the ``ctrl_mov`` edge back
    into a reserve/phi.  This representation retains both while providing a
    materialized-only shortcut graph for the placement/routing interaction.
    """
    order: List[str] = []
    kinds: Dict[str, str] = {}
    operands: Dict[str, List[str]] = {}
    control_moves: List[Tuple[str, str]] = []
    effect_index = 0
    has_kernel_region = bool(re.search(r'\bneura\.kernel\b', text))
    for line in text.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)", line)
        if match:
            value, expression = match.groups()
            kind_match = re.search(
                r'"?neura\.([a-z_][a-z0-9_.]*)', expression,
            )
            if kind_match is None:
                continue
            kind = kind_match.group(1).replace(".", "_")
            kinds[value] = kind
            operands[value] = re.findall(r"%[A-Za-z0-9_]+", expression)
            order.append(value)
            continue
        control_match = re.search(
            r'"?neura\.ctrl_mov"?\s+(%[A-Za-z0-9_]+)\s*->\s*'
            r'(%[A-Za-z0-9_]+)',
            line,
        )
        if control_match is not None:
            control_moves.append(control_match.groups())
        # MLIR operations such as store_indexed, ctrl_mov, and yield have no
        # SSA result, but they still participate in the mapper's topological
        # ordering.  Stores are materialized and their high fan-in is often a
        # decisive heuristic-mapper constraint, so dropping them changes the
        # prediction problem rather than merely simplifying the graph.
        effect_match = re.match(
            r'\s*"?neura\.([a-z_][a-z0-9_.]*)"?(?:\s|\()(.*)', line,
        )
        if effect_match is not None and effect_match.group(1) != "kernel":
            value = f"%__graph_effect_{effect_index}"
            effect_index += 1
            kinds[value] = effect_match.group(1).replace(".", "_")
            operands[value] = re.findall(r"%[A-Za-z0-9_]+", line)
            order.append(value)
            continue
        # Legacy generated mapper corpora map the function body directly and
        # therefore include func.return.  Static-DL DFGs map only the nested
        # neura.kernel region; the outer function return must not be invented
        # as a mapper node in that representation.
        return_match = re.match(r"\s*(?:func\.)?return\b(.*)", line)
        if return_match is not None and not has_kernel_region:
            value = f"%__graph_effect_{effect_index}"
            effect_index += 1
            kinds[value] = "return"
            operands[value] = re.findall(
                r"%[A-Za-z0-9_]+", return_match.group(1),
            )
            order.append(value)
    if not order:
        raise ValueError("Neura DFG contains no operations")
    index = {value: position for position, value in enumerate(order)}
    raw_edge_set = {
        (index[source], index[target])
        for target in order
        for source in operands.get(target, ())
        if source in index and source != target
    }
    control_edge_set = {
        (index[source], index[target])
        for source, target in control_moves
        if source in index and target in index and source != target
    }
    raw_edge_set.update(control_edge_set)

    materialized = {
        value for value in order
        if kinds[value] not in TRANSPARENT_OPERATIONS
    }
    semantic_edge_set = set()
    for target in materialized:
        for operand in operands.get(target, ()):
            for source in _meaningful_roots(operand, kinds, operands):
                if source in materialized and source != target:
                    semantic_edge_set.add((index[source], index[target]))
    # A ctrl_mov writes a late value into a reserve consumed near the loop
    # header.  Expose the corresponding distance-one semantic feedback edge.
    for source, reserve in control_moves:
        roots = _meaningful_roots(source, kinds, operands)
        reserve_users = [
            target for target in materialized
            if reserve in operands.get(target, ())
        ]
        for root in roots:
            for target in reserve_users:
                if root in materialized and root != target:
                    semantic_edge_set.add((index[root], index[target]))

    node_count = len(order)
    forward_edges = {
        edge for edge in raw_edge_set
        if edge not in control_edge_set and edge[0] < edge[1]
    }
    parents: List[List[int]] = [[] for _ in order]
    children: List[List[int]] = [[] for _ in order]
    for source, target in forward_edges:
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

    cycle_nodes = set()
    for feedback_source, feedback_target in control_edge_set:
        descendants = {feedback_target}
        work = [feedback_target]
        while work:
            node = work.pop()
            for child in children[node]:
                if child not in descendants:
                    descendants.add(child)
                    work.append(child)
        ancestors = {feedback_source}
        work = [feedback_source]
        while work:
            node = work.pop()
            for parent in parents[node]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    work.append(parent)
        cycle_nodes.update(descendants.intersection(ancestors))
        cycle_nodes.update((feedback_source, feedback_target))

    node_features: List[Tuple[float, ...]] = []
    denominator = float(max(1, node_count - 1))
    for node, value in enumerate(order):
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
            float(kind not in TRANSPARENT_OPERATIONS),
            float(kind == "data_mov"),
            float(node in cycle_nodes),
        ))
    graph = GraphData(
        node_types=tuple(
            ROUTE_EXPANDED_OPERATION_TO_ID.get(
                kinds[value], ROUTE_EXPANDED_OPERATION_TO_ID["<unknown>"],
            )
            for value in order
        ),
        node_features=tuple(node_features),
        edges=tuple(sorted(raw_edge_set)),
        semantic_edges=tuple(sorted(semantic_edge_set)),
    )
    return graph.validate(len(ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES))


def require_neura_route_expanded_dfg(text: str) -> GraphData:
    """Parse a DFG and reject semantic edges that bypass ``data_mov``.

    Neura's mapper consumes the result of ``--insert-data-mov``.  Merely
    parsing an unexpanded DFG with the richer graph parser is legal, but it
    silently changes the feature distribution seen by the mapper surrogate.
    This stricter entry point is used at training and deployment boundaries so
    both sides observe the exact same deterministic pre-mapper representation.
    """
    graph = parse_neura_route_expanded_dfg(text)
    semantic_edges = set(graph.semantic_edges or ())
    if not semantic_edges:
        return graph
    movement_count = sum(row[12] > 0.5 for row in graph.node_features)
    direct_semantic_edges = semantic_edges.intersection(graph.edges)
    if movement_count == 0 or direct_semantic_edges:
        raise ValueError(
            "Neura DFG is not route-expanded; run --insert-data-mov exactly "
            "once before mapper feature extraction"
        )
    return graph


def parse_neura_dfg_representation(
    text: str, representation: str = "route_expanded",
) -> GraphData:
    if representation == "semantic":
        return parse_neura_dfg(text)
    if representation == "route_expanded":
        return parse_neura_route_expanded_dfg(text)
    raise ValueError("unknown DFG representation: " + representation)


__all__ = [
    "DFG_NODE_FEATURE_NAMES",
    "GraphData",
    "OPERATION_TYPES",
    "ROUTE_EXPANDED_DFG_NODE_FEATURE_NAMES",
    "ROUTE_EXPANDED_OPERATION_TYPES",
    "parse_neura_dfg",
    "parse_neura_dfg_representation",
    "parse_neura_route_expanded_dfg",
    "require_neura_route_expanded_dfg",
]
