"""Scheduling and execution.

flodiedi's scheduler (``FlowDiagram::findExecutionOrder``) was a greedy
O(n^2) scan that seeded from blocks without inputs and then repeatedly looked
for the first block whose predecessors were all placed. Anything left over --
which is what a cycle produces -- was logged as
``"WARNING: UNCONNECTED BLOCKS"`` and then **silently never executed**. A
cyclic diagram therefore appeared to run while doing nothing.

Here the sort is Kahn's algorithm in O(V+E), and a cycle is a hard error that
names the nodes involved.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from . import registry
from .block import BlockSpec
from .graph import Graph, Node
from .types import CoercionError, Preview, coerce

__all__ = [
    "topological_order",
    "coerce_params",
    "Executor",
    "RunResult",
    "CycleError",
    "NodeExecutionError",
]


def coerce_params(
    node: Node, spec: BlockSpec, base_dir: Path | None = None
) -> dict[str, Any]:
    """Map a node's YAML literals onto the types its ports declare.

    Relative ``Path`` values are resolved against ``base_dir`` -- the directory
    of the diagram file -- so that a diagram and its data can be moved or
    checked out anywhere.
    """
    coerced: dict[str, Any] = {}
    for key, value in node.params.items():
        port = spec.input(key)
        if port is None:
            continue  # reported by Graph.validate()
        try:
            resolved = coerce(value, port.type)
        except CoercionError as exc:
            raise CoercionError(f"{node.where()}: param {key!r}: {exc}") from None
        if base_dir is not None and isinstance(resolved, Path) and not resolved.is_absolute():
            resolved = base_dir / resolved
        coerced[key] = resolved
    return coerced


class CycleError(ValueError):
    """The diagram contains a cycle, so no execution order exists."""


class NodeExecutionError(RuntimeError):
    """A block raised while running. Names the node, block and original cause."""

    def __init__(self, node_id: str, block: str, cause: BaseException) -> None:
        self.node_id = node_id
        self.block = block
        self.cause = cause
        super().__init__(
            f"node {node_id!r} (block {block!r}) failed: "
            f"{type(cause).__name__}: {cause}"
        )


def topological_order(graph: Graph) -> list[str]:
    """Node ids in a runnable order. Raises :class:`CycleError` if none exists."""
    indegree: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    successors: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}

    # Parallel edges between the same pair of nodes each count, so a node is
    # only released once every distinct incoming edge has been satisfied.
    for edge in graph.edges:
        if edge.src not in graph.nodes or edge.dst not in graph.nodes:
            continue  # reported by Graph.validate()
        successors[edge.src].append(edge.dst)
        indegree[edge.dst] += 1

    # Sorted seeds and sorted release order make the result deterministic,
    # which keeps test fixtures and error messages stable.
    ready = deque(sorted(n for n, d in indegree.items() if d == 0))
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        newly_ready = []
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                newly_ready.append(successor)
        ready.extend(sorted(newly_ready))

    if len(order) != len(graph.nodes):
        remaining = {n for n in graph.nodes if n not in set(order)}
        raise CycleError(
            "diagram contains a cycle, so it has no execution order: "
            + _describe_cycle(graph, remaining)
        )
    return order


def _describe_cycle(graph: Graph, remaining: set[str]) -> str:
    """Find and render one concrete cycle, e.g. ``a -> b -> c -> a``."""
    adjacency: dict[str, list[str]] = {n: [] for n in remaining}
    for edge in graph.edges:
        if edge.src in remaining and edge.dst in remaining:
            adjacency[edge.src].append(edge.dst)

    for start in sorted(remaining):
        path: list[str] = []
        on_path: set[str] = set()

        def walk(node: str) -> list[str] | None:
            if node in on_path:
                return path[path.index(node) :] + [node]
            if node in seen:
                return None
            seen.add(node)
            path.append(node)
            on_path.add(node)
            for nxt in sorted(adjacency.get(node, ())):
                found = walk(nxt)
                if found:
                    return found
            path.pop()
            on_path.discard(node)
            return None

        seen: set[str] = set()
        cycle = walk(start)
        if cycle:
            return " -> ".join(cycle)
    return ", ".join(sorted(remaining))  # should not happen


@dataclass(slots=True)
class RunResult:
    """What one sweep produced."""

    order: list[str] = field(default_factory=list)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per node id, its output ports and their values."""
    previews: list[Preview] = field(default_factory=list)
    iteration: int = 0

    def value(self, node_id: str, port: str) -> Any:
        return self.outputs[node_id][port]


class Executor:
    """Runs a validated graph.

    Execution is a serial sweep in topological order, as in flodiedi. Running
    independent branches in parallel is possible -- ``cv2`` releases the GIL --
    but that is a phase 1 concern, and correctness comes first.
    """

    def __init__(self, graph: Graph, *, validate: bool = True) -> None:
        self.graph = graph
        if validate:
            graph.validate()
        self.order = topological_order(graph)
        self._specs = {
            node.id: registry.get(node.block) for node in graph.nodes.values()
        }
        # Literals from the file are coerced once, not on every sweep: a
        # diagram running at 30 fps should not re-parse its enums 30 times a
        # second.
        self._params = {
            node.id: coerce_params(node, self._specs[node.id], graph.base_dir)
            for node in graph.nodes.values()
        }

    def step(self, iteration: int = 0) -> RunResult:
        """Execute every node once, in order."""
        result = RunResult(order=list(self.order), iteration=iteration)
        # (node_id, port_name) -> value, filled in as producers run
        wire: dict[tuple[str, str], Any] = {}

        for node_id in self.order:
            node = self.graph.nodes[node_id]
            spec = self._specs[node_id]
            values = dict(self._params[node_id])
            for edge in self.graph.incoming(node_id):
                values[edge.dst_port] = wire[(edge.dst, edge.dst_port)]

            try:
                produced = spec.call(values)
            except Exception as exc:
                raise NodeExecutionError(node_id, node.block, exc) from exc

            result.outputs[node_id] = produced
            for port_name, value in produced.items():
                if isinstance(value, Preview):
                    result.previews.append(value)
                for edge in self.graph.outgoing(node_id):
                    if edge.src_port == port_name:
                        wire[(edge.dst, edge.dst_port)] = value

        return result

    def run(self, iterations: int = 1, interval: float = 0.0) -> RunResult:
        """Sweep ``iterations`` times, returning the last result.

        ``iterations <= 0`` runs until interrupted, which is what the GUI's
        run button will use.
        """
        result = RunResult()
        i = 0
        while iterations <= 0 or i < iterations:
            if interval and i:
                time.sleep(interval)
            result = self.step(i)
            i += 1
        return result
