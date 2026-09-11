"""Placing a diagram so it can be read.

flodiedi handed this to Graphviz `dot` with ``rankdir=LR``
(``flowdiagramscene.cpp:499``), which is the right tool. It also extracted the
edge splines Graphviz computed -- ``GVGraph::edges()`` builds a complete
``QPainterPath`` with ``cubicTo`` over the control points
(``gvgraph.cpp:216-265``) -- and then **never used them**: ``layout()`` reads
only node positions and resets every connection to a straight line
(``flowdiagramscene.cpp:501-514``). All that routing work was computed and
thrown away, which is exactly why edges ran through nodes.

Rather than take on libgraphviz -- whose ``libgraph`` API flodiedi used was
removed from Graphviz in 2012, and is one reason its build no longer links --
this is a layered (Sugiyama) layout in pure Python:

1. **Layering.** Longest path from the sources, so every edge points forward.
2. **Dummy vertices.** An edge spanning several layers gets a placeholder in
   each layer it passes over. They take part in the next two steps, which is
   what reserves a lane for the edge instead of letting it cut across whatever
   happens to sit in between.
3. **Ordering.** Repeated barycentre sweeps to reduce crossings.
4. **Coordinates.** Each node is pulled towards the average of its neighbours,
   then the column is pushed apart so nothing overlaps.

Steps 2 and 3 are what the first version lacked: it sorted each column
alphabetically and drew long edges as a single curve, so they crossed for no
reason and passed straight through unrelated nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Edge, Graph

__all__ = [
    "NodeBox",
    "LayoutResult",
    "layered_layout",
    "count_crossings",
    "DEFAULT_BOX",
]


@dataclass(frozen=True, slots=True)
class NodeBox:
    """How much room a node needs. The renderer knows; the layout does not."""

    width: float
    height: float


@dataclass(slots=True)
class LayoutResult:
    positions: dict[str, tuple[float, float]]
    """Top-left corner per node id."""
    layers: list[list[str]] = field(default_factory=list)
    """Node ids per layer, real nodes only, in the order the layout settled on."""
    waypoints: dict[tuple[str, str, str, str], list[tuple[float, float]]] = field(
        default_factory=dict
    )
    """Intermediate points for edges that span more than one layer.

    Keyed by ``(src, src_port, dst, dst_port)``. Empty for a short edge, which
    the renderer can draw as a simple curve.
    """

    def route_for(self, edge: Edge) -> list[tuple[float, float]]:
        return self.waypoints.get((edge.src, edge.src_port, edge.dst, edge.dst_port), [])

    @property
    def layer_of(self) -> dict[str, int]:
        return {n: i for i, layer in enumerate(self.layers) for n in layer}


DEFAULT_BOX = NodeBox(150.0, 80.0)
LAYER_GAP = 90.0
NODE_GAP = 30.0
DUMMY_HEIGHT = 16.0
"""How much vertical room a passing edge reserves in a layer it crosses."""
SWEEPS = 10


def layered_layout(
    graph: Graph,
    boxes: dict[str, NodeBox] | None = None,
    *,
    layer_gap: float = LAYER_GAP,
    node_gap: float = NODE_GAP,
    sweeps: int = SWEEPS,
) -> LayoutResult:
    """Place every node left to right, in dependency order."""
    boxes = dict(boxes or {})
    layers = _assign_layers(graph)
    layer_of = {n: i for i, layer in enumerate(layers) for n in layer}

    expanded, chains, adjacency = _insert_dummies(graph, layers, layer_of)
    expanded = _reduce_crossings(expanded, adjacency, sweeps)

    for chain in chains.values():
        for dummy in chain:
            boxes[dummy] = NodeBox(0.0, DUMMY_HEIGHT)

    positions = _assign_coordinates(expanded, adjacency, boxes, layer_gap, node_gap)

    waypoints = {
        key: [
            (positions[d][0], positions[d][1] + DUMMY_HEIGHT / 2.0) for d in chain
        ]
        for key, chain in chains.items()
    }
    real_layers = [[n for n in layer if n in graph.nodes] for layer in expanded]
    real_positions = {n: p for n, p in positions.items() if n in graph.nodes}
    return LayoutResult(
        positions=real_positions, layers=real_layers, waypoints=waypoints
    )


# -- 1. layering ----------------------------------------------------------


def _assign_layers(graph: Graph) -> list[list[str]]:
    """Longest path from the sources, so every edge spans at least one layer.

    Cycles cannot be layered; a node in one keeps the deepest layer it reached
    before the walk gave up. A cyclic diagram cannot run, but it must still be
    viewable -- that is when you most need to look at it.
    """
    predecessors: dict[str, set[str]] = {n: set() for n in graph.nodes}
    successors: dict[str, set[str]] = {n: set() for n in graph.nodes}
    for edge in graph.edges:
        if edge.src in graph.nodes and edge.dst in graph.nodes and edge.src != edge.dst:
            predecessors[edge.dst].add(edge.src)
            successors[edge.src].add(edge.dst)

    depth = dict.fromkeys(graph.nodes, 0)
    ready = [n for n, preds in predecessors.items() if not preds]
    remaining = {n: set(preds) for n, preds in predecessors.items()}
    seen: set[str] = set()

    while ready:
        node = ready.pop()
        seen.add(node)
        for successor in sorted(successors[node]):
            depth[successor] = max(depth[successor], depth[node] + 1)
            remaining[successor].discard(node)
            if not remaining[successor] and successor not in seen:
                ready.append(successor)

    for node in graph.nodes:
        if node not in seen:
            resolved = [depth[p] for p in predecessors[node] if p in seen]
            depth[node] = max(resolved, default=0) + 1

    layers: list[list[str]] = [[] for _ in range(max(depth.values(), default=0) + 1)]
    for node in graph.nodes:  # insertion order, so the result is deterministic
        layers[depth[node]].append(node)
    return layers


# -- 2. dummy vertices ----------------------------------------------------


def _insert_dummies(
    graph: Graph, layers: list[list[str]], layer_of: dict[str, int]
) -> tuple[
    list[list[str]],
    dict[tuple[str, str, str, str], list[str]],
    dict[str, list[str]],
]:
    """Give every long edge a placeholder in each layer it passes over.

    Without this an edge from layer 0 to layer 4 is drawn as one curve across
    four columns, straight over whatever sits in them. With it, the edge
    occupies a slot in each intermediate layer, takes part in crossing
    reduction, and ends up with a lane of its own.
    """
    expanded = [list(layer) for layer in layers]
    chains: dict[tuple[str, str, str, str], list[str]] = {}
    adjacency: dict[str, list[str]] = {n: [] for n in graph.nodes}

    for index, edge in enumerate(graph.edges):
        if edge.src not in layer_of or edge.dst not in layer_of:
            continue
        start, end = layer_of[edge.src], layer_of[edge.dst]
        if end - start <= 1:
            adjacency[edge.src].append(edge.dst)
            continue

        chain: list[str] = []
        previous = edge.src
        for layer_index in range(start + 1, end):
            dummy = f"\x00dummy:{index}:{layer_index}"
            expanded[layer_index].append(dummy)
            adjacency.setdefault(previous, []).append(dummy)
            adjacency.setdefault(dummy, [])
            chain.append(dummy)
            previous = dummy
        adjacency[previous].append(edge.dst)
        chains[(edge.src, edge.src_port, edge.dst, edge.dst_port)] = chain

    return expanded, chains, adjacency


# -- 3. crossing reduction ------------------------------------------------


def _reduce_crossings(
    layers: list[list[str]], adjacency: dict[str, list[str]], sweeps: int
) -> list[list[str]]:
    """Barycentre ordering, alternating forwards and backwards.

    Each node is pulled towards the average position of its neighbours in the
    adjacent layer; repeating in both directions settles quickly. The best
    arrangement seen is kept, because a barycentre sweep is not monotonic.
    """
    if len(layers) < 2:
        return layers

    incoming: dict[str, list[str]] = {}
    for source, targets in adjacency.items():
        for target in targets:
            incoming.setdefault(target, []).append(source)

    best = [list(layer) for layer in layers]
    best_score = count_crossings(best, adjacency)
    current = [list(layer) for layer in layers]

    for sweep in range(sweeps):
        forward = sweep % 2 == 0
        indices = range(1, len(current)) if forward else range(len(current) - 2, -1, -1)
        for i in indices:
            neighbours = incoming if forward else adjacency
            reference = {
                n: pos for pos, n in enumerate(current[i - 1 if forward else i + 1])
            }
            current[i] = _order_by_barycentre(current[i], neighbours, reference)

        score = count_crossings(current, adjacency)
        if score < best_score:
            best_score, best = score, [list(layer) for layer in current]
        if best_score == 0:
            break
    return best


def _order_by_barycentre(
    layer: list[str], neighbours: dict[str, list[str]], reference: dict[str, int]
) -> list[str]:
    original = {n: i for i, n in enumerate(layer)}

    def key(node: str) -> tuple[float, int]:
        positions = [reference[n] for n in neighbours.get(node, ()) if n in reference]
        # A node with no neighbour in the reference layer keeps its place, so
        # unconnected blocks do not drift to one end.
        barycentre = (
            sum(positions) / len(positions) if positions else float(original[node])
        )
        return (barycentre, original[node])

    return sorted(layer, key=key)


def count_crossings(layers: list[list[str]], adjacency: dict[str, list[str]]) -> int:
    """How many edge pairs cross between consecutive layers.

    Straightforward O(e^2) per layer pair -- diagrams here have tens of nodes,
    and a clearer implementation is worth more than the asymptotics.
    """
    total = 0
    for left, right in zip(layers, layers[1:], strict=False):
        left_pos = {n: i for i, n in enumerate(left)}
        right_pos = {n: i for i, n in enumerate(right)}
        pairs = [
            (left_pos[src], right_pos[dst])
            for src in left
            for dst in adjacency.get(src, ())
            if dst in right_pos
        ]
        for i, (a1, b1) in enumerate(pairs):
            for a2, b2 in pairs[i + 1 :]:
                if (a1 - a2) * (b1 - b2) < 0:
                    total += 1
    return total


# -- 4. coordinates -------------------------------------------------------


def _assign_coordinates(
    layers: list[list[str]],
    adjacency: dict[str, list[str]],
    boxes: dict[str, NodeBox],
    layer_gap: float,
    node_gap: float,
) -> dict[str, tuple[float, float]]:
    """Columns by layer; within a column, pull towards connected neighbours."""

    def box(node: str) -> NodeBox:
        return boxes.get(node, DEFAULT_BOX)

    xs: list[float] = []
    x = 0.0
    for layer in layers:
        xs.append(x)
        widest = max((box(n).width for n in layer), default=DEFAULT_BOX.width)
        x += widest + layer_gap

    ys: dict[str, float] = {}
    for layer in layers:
        total = sum(box(n).height for n in layer) + node_gap * max(0, len(layer) - 1)
        y = -total / 2.0
        for node in layer:
            ys[node] = y
            y += box(node).height + node_gap

    incoming: dict[str, list[str]] = {}
    for source, targets in adjacency.items():
        for target in targets:
            incoming.setdefault(target, []).append(source)

    # Pull each node towards the centre of what feeds it, then push the column
    # apart again so nothing overlaps. This is what turns a correct-but-jagged
    # layout into one where an edge mostly runs straight.
    for layer in layers[1:]:
        desired: list[tuple[str, float]] = []
        for node in layer:
            sources = [
                ys[s] + box(s).height / 2.0 for s in incoming.get(node, ()) if s in ys
            ]
            centre = (
                sum(sources) / len(sources)
                if sources
                else ys[node] + box(node).height / 2.0
            )
            desired.append((node, centre - box(node).height / 2.0))
        desired.sort(key=lambda pair: pair[1])

        placed: list[tuple[str, float]] = []
        for node, wanted in desired:
            if placed:
                previous, previous_y = placed[-1]
                wanted = max(wanted, previous_y + box(previous).height + node_gap)
            placed.append((node, wanted))

        # Re-centre on the span the column wanted, so repeated pulls do not
        # drift the whole diagram downwards.
        shift = sum(y for _, y in desired) / len(desired) - sum(
            y for _, y in placed
        ) / len(placed)
        for node, y in placed:
            ys[node] = y + shift

    return {node: (xs[i], ys[node]) for i, layer in enumerate(layers) for node in layer}
