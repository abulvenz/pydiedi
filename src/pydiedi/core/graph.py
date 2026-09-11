"""The diagram: nodes, edges, layout -- and validation.

Deliberate departures from flodiedi's model:

* **Nodes have readable ids.** flodiedi identified blocks by ``QUuid``, so a
  diagram with four ``cvtColor`` blocks (``SampleDiagrams/wallpaper.xml`` has
  exactly that) could only be understood by cross-referencing 36-character
  identifiers. Here the id is a name the author chooses -- ``gray_live``,
  ``gray_ref``, ``gray_warp`` -- and it carries the intent.
* **Layout is separate.** flodiedi stored ``__position_x``/``__position_y`` as
  dynamic properties on the model objects, mixed in with the semantic ones.
  Here geometry lives in its own section; delete it and the diagram still runs.
* **No derived data is stored.** flodiedi persisted ``executionorder`` into the
  file, where it went stale on every edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry
from .block import PortKind
from .types import CoercionError, coerce, is_compatible

__all__ = ["Node", "Edge", "Graph", "ValidationError"]


class ValidationError(ValueError):
    """A diagram is not runnable. Carries every problem found, not just the first."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        if len(problems) == 1:
            super().__init__(problems[0])
        else:
            body = "\n".join(f"  - {p}" for p in problems)
            super().__init__(f"{len(problems)} problems in diagram:\n{body}")


@dataclass(slots=True)
class Node:
    id: str
    block: str
    params: dict[str, Any] = field(default_factory=dict)
    line: int | None = None
    """Source line in the YAML file, for error messages. ``None`` if built in code."""

    def where(self) -> str:
        return f"node {self.id!r}" + (f" (line {self.line})" if self.line else "")


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str
    line: int | None = field(default=None, compare=False)
    """Source line in the YAML file, for error messages.

    Excluded from equality and hashing on purpose: it is metadata about where
    an edge was written down, not part of what the edge *is*. Two edges between
    the same ports are the same edge whether one came from a file and the other
    from the editor.
    """

    def __str__(self) -> str:
        return f"{self.src}.{self.src_port} -> {self.dst}.{self.dst_port}"

    def where(self) -> str:
        return f"edge {self}" + (f" (line {self.line})" if self.line else "")


@dataclass(slots=True)
class Graph:
    name: str = "untitled"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    layout: dict[str, tuple[float, float]] = field(default_factory=dict)
    base_dir: Path | None = None
    """Directory of the file this graph was loaded from.

    Relative path parameters resolve against it, which makes a diagram plus its
    data portable. flodiedi had no such notion, and the consequence is visible
    in its own repository: ``loadPointCloudplugin`` shipped with
    ``/home/kochas/schnecke.pcd`` compiled into it.
    """

    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> Edge:
        edge = Edge(src, src_port, dst, dst_port)
        self.edges.append(edge)
        return edge

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.src == node_id]

    def validate(self) -> None:
        """Raise :class:`ValidationError` listing every problem, or return quietly."""
        problems = self.problems()
        if problems:
            raise ValidationError(problems)

    def problems(self) -> list[str]:
        problems: list[str] = []
        specs = self._collect_specs(problems)
        self._check_params(specs, problems)
        self._check_edges(specs, problems)
        self._check_required_inputs(specs, problems)
        self._check_layout(problems)
        return problems

    # -- individual checks ------------------------------------------------

    def _collect_specs(self, problems: list[str]) -> dict[str, Any]:
        specs: dict[str, Any] = {}
        for node in self.nodes.values():
            try:
                specs[node.id] = registry.get(node.block)
            except registry.UnknownBlockError as exc:
                problems.append(f"{node.where()}: {exc}")
        return specs

    def _check_params(self, specs: dict[str, Any], problems: list[str]) -> None:
        for node in self.nodes.values():
            spec = specs.get(node.id)
            if spec is None:
                continue
            for key, value in node.params.items():
                port = spec.input(key)
                if port is None:
                    known = ", ".join(p.name for p in spec.inputs) or "none"
                    problems.append(
                        f"{node.where()}: block {node.block!r} has no input {key!r}. "
                        f"Known inputs: {known}"
                    )
                    continue
                try:
                    coerce(value, port.type)
                except CoercionError as exc:
                    problems.append(f"{node.where()}: param {key!r}: {exc}")

    def _check_edges(self, specs: dict[str, Any], problems: list[str]) -> None:
        seen_targets: dict[tuple[str, str], Edge] = {}
        for edge in self.edges:
            for role, node_id in (("source", edge.src), ("target", edge.dst)):
                if node_id not in self.nodes:
                    problems.append(
                        f"{edge.where()}: {role} node {node_id!r} does not exist"
                    )
            src_spec, dst_spec = specs.get(edge.src), specs.get(edge.dst)
            out_port = in_port = None

            if src_spec is not None:
                out_port = src_spec.output(edge.src_port)
                if out_port is None:
                    known = ", ".join(p.name for p in src_spec.outputs) or "none"
                    problems.append(
                        f"{edge.where()}: block {src_spec.name!r} has no output "
                        f"{edge.src_port!r}. Known outputs: {known}"
                    )
            if dst_spec is not None:
                in_port = dst_spec.input(edge.dst_port)
                if in_port is None:
                    known = ", ".join(p.name for p in dst_spec.inputs) or "none"
                    problems.append(
                        f"{edge.where()}: block {dst_spec.name!r} has no input "
                        f"{edge.dst_port!r}. Known inputs: {known}"
                    )

            if out_port is not None and in_port is not None:
                if out_port.kind is not PortKind.OUT or in_port.kind is not PortKind.IN:
                    problems.append(f"{edge.where()}: wrong port direction")
                elif not is_compatible(out_port.type, in_port.type):
                    problems.append(
                        f"{edge.where()}: type mismatch, "
                        f"{out_port.type_name} cannot feed {in_port.type_name}"
                    )

            # One source per input port. flodiedi never checked this; a second
            # connection silently overwrote the first during transmit().
            key = (edge.dst, edge.dst_port)
            if key in seen_targets:
                problems.append(
                    f"{edge.where()}: input {edge.dst}.{edge.dst_port} is already fed "
                    f"by {seen_targets[key]}. An input takes at most one connection."
                )
            else:
                seen_targets[key] = edge

            # A connected input must not also carry a literal.
            node = self.nodes.get(edge.dst)
            if node is not None and edge.dst_port in node.params:
                problems.append(
                    f"{edge.where()}: input {edge.dst}.{edge.dst_port} is both connected "
                    f"and given as a param. Remove one."
                )
        return None

    def _check_required_inputs(self, specs: dict[str, Any], problems: list[str]) -> None:
        connected = {(e.dst, e.dst_port) for e in self.edges}
        for node in self.nodes.values():
            spec = specs.get(node.id)
            if spec is None:
                continue
            for port in spec.required_inputs:
                if (node.id, port.name) in connected or port.name in node.params:
                    continue
                problems.append(
                    f"{node.where()}: required input {port.name!r} "
                    f"({port.type_name}) is neither connected nor given as a param"
                )

    def _check_layout(self, problems: list[str]) -> None:
        for node_id in self.layout:
            if node_id not in self.nodes:
                problems.append(f"layout references unknown node {node_id!r}")
