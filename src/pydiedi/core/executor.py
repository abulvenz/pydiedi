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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from . import registry
from .block import BlockSpec
from .graph import Graph, Node
from .types import CoercionError, Preview, coerce

__all__ = [
    "topological_order",
    "coerce_params",
    "Executor",
    "OnError",
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


class StopExecution(Exception):
    """A block asks for the run to end. Not an error.

    A video source that reached its last frame raises this, and ``run()``
    returns normally instead of propagating a failure. flodiedi expressed the
    same idea as a ``terminateExecution`` block.
    """


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
    errors: dict[str, NodeExecutionError] = field(default_factory=dict)
    """Nodes that raised, when running with ``on_error='skip'``."""
    skipped: list[str] = field(default_factory=list)
    """Nodes not run because something they depend on failed."""
    stopped_by: str | None = None
    """Why the run ended early, if a block raised :class:`StopExecution`."""

    def value(self, node_id: str, port: str) -> Any:
        return self.outputs[node_id][port]

    @property
    def ok(self) -> bool:
        return not self.errors


class OnError(str, Enum):
    """What to do when a block raises."""

    raise_ = "raise"
    """Abort the sweep. The default, and what a script or a test wants."""
    skip = "skip"
    """Record the failure, skip whatever depended on it, keep other branches
    running. This is what the editor wants: one broken node should colour
    itself red rather than stop the whole diagram."""


class Executor:
    """Runs a validated graph.

    Execution is a serial sweep in topological order, as in flodiedi. Running
    independent branches in parallel is possible -- ``cv2`` releases the GIL --
    but correctness comes first.

    Stateful blocks get one instance per node, created here and released by
    :meth:`close`. Use it as a context manager to be sure that happens::

        with Executor(graph) as executor:
            executor.run(iterations=0)
    """

    def __init__(
        self,
        graph: Graph,
        *,
        validate: bool = True,
        on_error: OnError | str = OnError.raise_,
    ) -> None:
        self.graph = graph
        if validate:
            graph.validate()
        self.on_error = OnError(on_error)
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
        # One instance per node, so two 'camera' nodes are two devices rather
        # than one shared and mutually corrupting object.
        self._instances: dict[str, Any] = {
            node_id: spec.instantiate() for node_id, spec in self._specs.items()
        }
        self._dependents = self._build_dependents()

    def _build_dependents(self) -> dict[str, set[str]]:
        """For each node, everything downstream of it (transitively)."""
        direct: dict[str, set[str]] = {n: set() for n in self.graph.nodes}
        for edge in self.graph.edges:
            if edge.src in direct:
                direct[edge.src].add(edge.dst)
        # Walking in reverse topological order lets each node simply union what
        # its successors already computed.
        dependents: dict[str, set[str]] = {n: set() for n in self.graph.nodes}
        for node_id in reversed(self.order):
            for successor in direct[node_id]:
                dependents[node_id].add(successor)
                dependents[node_id] |= dependents.get(successor, set())
        return dependents

    @property
    def stateful_nodes(self) -> list[str]:
        return [n for n, spec in self._specs.items() if spec.is_stateful]

    def step(self, iteration: int = 0) -> RunResult:
        """Execute every node once, in order."""
        result = RunResult(order=list(self.order), iteration=iteration)
        wire: dict[tuple[str, str], Any] = {}
        blocked: set[str] = set()

        for node_id in self.order:
            if node_id in blocked:
                result.skipped.append(node_id)
                continue

            node = self.graph.nodes[node_id]
            spec = self._specs[node_id]
            values = dict(self._params[node_id])
            for edge in self.graph.incoming(node_id):
                values[edge.dst_port] = wire[(edge.dst, edge.dst_port)]

            try:
                produced = spec.call(values, self._instances[node_id])
            except StopExecution:
                # A deliberate end, not a failure: let it pass through run().
                raise
            except Exception as exc:
                error = NodeExecutionError(node_id, node.block, exc)
                if self.on_error is OnError.raise_:
                    raise error from exc
                result.errors[node_id] = error
                # Downstream nodes would run on stale or absent inputs, so they
                # are skipped rather than fed garbage.
                blocked |= self._dependents[node_id]
                continue

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

        ``iterations <= 0`` runs until interrupted, which is what a live camera
        and the GUI's run button need.
        """
        result = RunResult()
        i = 0
        while iterations <= 0 or i < iterations:
            if interval and i:
                time.sleep(interval)
            try:
                result = self.step(i)
            except StopExecution as stop:
                result.stopped_by = str(stop)
                break
            i += 1
        return result

    def close(self) -> None:
        """Release every stateful block's resources.

        Errors from one ``close()`` must not prevent the others, so they are
        collected and re-raised together.
        """
        failures: list[BaseException] = []
        for node_id, instance in self._instances.items():
            try:
                self._specs[node_id].close(instance)
            except Exception as exc:  # noqa: PERF203
                failures.append(exc)
        self._instances = dict.fromkeys(self._instances)
        if failures:
            raise ExceptionGroup("errors while closing blocks", failures)

    def __enter__(self) -> Executor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
