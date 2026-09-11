"""Editing a diagram: commands, validation and undo.

This lives in ``core`` rather than in the editor, and imports no GUI toolkit.
Two reasons. Undo is the part of an editor most worth testing and least
pleasant to test through a GUI; and a second renderer -- a web front end --
needs exactly the same operations, so putting them behind Qt would mean
writing them twice.

Every change goes through a :class:`Command`. Nothing mutates a
:class:`~pydiedi.core.graph.Graph` directly, which is what makes undo a
property of the design rather than a feature bolted on afterwards.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from . import registry
from .block import PortKind
from .graph import Edge, Graph, Node
from .types import is_compatible

__all__ = [
    "Command",
    "AddNode",
    "RemoveNode",
    "Connect",
    "Disconnect",
    "MoveNode",
    "SetLayout",
    "SetParam",
    "RenameNode",
    "EditSession",
    "can_connect",
    "unique_node_id",
    "EditError",
]


class EditError(ValueError):
    """An edit was refused. The message is meant for the user."""


# -- validation -----------------------------------------------------------


def can_connect(
    graph: Graph, src: str, src_port: str, dst: str, dst_port: str
) -> str | None:
    """Why this connection is not allowed, or ``None`` if it is.

    Returning the reason rather than a bool is what lets the editor explain
    itself while a connection is being dragged. flodiedi checked only type
    compatibility and port direction, so it happily accepted a second edge into
    an occupied input -- where the later one silently overwrote the earlier
    during ``transmit()`` -- and a cycle, which its scheduler then dropped from
    the execution order with a warning nobody read.
    """
    if src == dst:
        return "a block cannot be connected to itself"

    source_node, target_node = graph.nodes.get(src), graph.nodes.get(dst)
    if source_node is None:
        return f"no such node: {src}"
    if target_node is None:
        return f"no such node: {dst}"

    try:
        source_spec = registry.get(source_node.block)
        target_spec = registry.get(target_node.block)
    except registry.UnknownBlockError as exc:
        return str(exc)

    out_port = source_spec.output(src_port)
    if out_port is None:
        return f"{source_spec.name} has no output {src_port!r}"
    in_port = target_spec.input(dst_port)
    if in_port is None:
        return f"{target_spec.name} has no input {dst_port!r}"
    if out_port.kind is not PortKind.OUT or in_port.kind is not PortKind.IN:
        return "connections run from an output to an input"

    if not is_compatible(out_port.type, in_port.type):
        return f"{out_port.type_name} cannot feed {in_port.type_name}"

    for edge in graph.edges:
        if edge.dst == dst and edge.dst_port == dst_port:
            return f"{dst}.{dst_port} is already connected to {edge.src}.{edge.src_port}"

    if _reaches(graph, dst, src):
        return f"this would create a cycle: {src} is already downstream of {dst}"

    return None


def _reaches(graph: Graph, start: str, target: str) -> bool:
    """Whether ``target`` is reachable from ``start`` by following edges."""
    seen: set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id == target:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(e.dst for e in graph.edges if e.src == node_id)
    return False


_ID_SUFFIX = re.compile(r"^(?P<stem>.*?)_(?P<number>\d+)$")


def unique_node_id(graph: Graph, preferred: str) -> str:
    """A node id not yet taken, derived from ``preferred``.

    ``cvt_color`` if free, else ``cvt_color_2``, ``cvt_color_3``. The editor
    names nodes after their block; the user renames them to something that
    carries intent, which is the whole reason node ids are not UUIDs.
    """
    if preferred not in graph.nodes:
        return preferred
    match = _ID_SUFFIX.match(preferred)
    stem = match["stem"] if match else preferred
    number = 2
    while f"{stem}_{number}" in graph.nodes:
        number += 1
    return f"{stem}_{number}"


# -- commands -------------------------------------------------------------


class Command(ABC):
    """One undoable change.

    ``apply`` must be callable again after ``revert`` and produce the same
    result, so a command records everything it needs at construction time or
    on its first application.
    """

    @abstractmethod
    def apply(self, graph: Graph) -> None: ...

    @abstractmethod
    def revert(self, graph: Graph) -> None: ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Shown in the undo menu, e.g. 'add threshold'."""

    def merge_with(self, later: Command) -> Command | None:
        """Combine with a command that immediately follows, or return ``None``.

        Used so that dragging a node across the canvas is one undo step rather
        than one per mouse-move event.
        """
        return None


@dataclass
class AddNode(Command):
    node_id: str
    block: str
    params: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] | None = None

    def apply(self, graph: Graph) -> None:
        if self.node_id in graph.nodes:
            raise EditError(f"a node named {self.node_id!r} already exists")
        graph.nodes[self.node_id] = Node(
            id=self.node_id, block=self.block, params=dict(self.params)
        )
        if self.position is not None:
            graph.layout[self.node_id] = self.position

    def revert(self, graph: Graph) -> None:
        graph.nodes.pop(self.node_id, None)
        graph.layout.pop(self.node_id, None)

    @property
    def description(self) -> str:
        return f"add {self.block}"


@dataclass
class RemoveNode(Command):
    node_id: str
    _node: Node | None = field(default=None, init=False, repr=False)
    _edges: list[Edge] = field(default_factory=list, init=False, repr=False)
    _position: tuple[float, float] | None = field(default=None, init=False, repr=False)

    def apply(self, graph: Graph) -> None:
        node = graph.nodes.get(self.node_id)
        if node is None:
            raise EditError(f"no such node: {self.node_id}")
        # Remembered so undo can restore the node *and* everything that was
        # attached to it. Deleting a block and undoing it must not silently
        # lose its wiring.
        self._node = node
        self._position = graph.layout.get(self.node_id)
        self._edges = [
            e for e in graph.edges if e.src == self.node_id or e.dst == self.node_id
        ]
        graph.edges = [e for e in graph.edges if e not in self._edges]
        del graph.nodes[self.node_id]
        graph.layout.pop(self.node_id, None)

    def revert(self, graph: Graph) -> None:
        if self._node is None:
            return
        graph.nodes[self.node_id] = self._node
        if self._position is not None:
            graph.layout[self.node_id] = self._position
        graph.edges.extend(self._edges)

    @property
    def description(self) -> str:
        return f"remove {self.node_id}"


@dataclass
class Connect(Command):
    src: str
    src_port: str
    dst: str
    dst_port: str

    def apply(self, graph: Graph) -> None:
        reason = can_connect(graph, self.src, self.src_port, self.dst, self.dst_port)
        if reason is not None:
            raise EditError(reason)
        graph.edges.append(Edge(self.src, self.src_port, self.dst, self.dst_port))

    def revert(self, graph: Graph) -> None:
        target = Edge(self.src, self.src_port, self.dst, self.dst_port)
        graph.edges = [e for e in graph.edges if e != target]

    @property
    def description(self) -> str:
        return f"connect {self.src}.{self.src_port} to {self.dst}.{self.dst_port}"


@dataclass
class Disconnect(Command):
    src: str
    src_port: str
    dst: str
    dst_port: str

    def apply(self, graph: Graph) -> None:
        target = Edge(self.src, self.src_port, self.dst, self.dst_port)
        if target not in graph.edges:
            raise EditError(f"no such connection: {target}")
        graph.edges = [e for e in graph.edges if e != target]

    def revert(self, graph: Graph) -> None:
        graph.edges.append(Edge(self.src, self.src_port, self.dst, self.dst_port))

    @property
    def description(self) -> str:
        return f"disconnect {self.dst}.{self.dst_port}"


@dataclass
class MoveNode(Command):
    node_id: str
    to: tuple[float, float]
    _from: tuple[float, float] | None = field(default=None, init=False, repr=False)

    def apply(self, graph: Graph) -> None:
        if self._from is None:
            self._from = graph.layout.get(self.node_id)
        graph.layout[self.node_id] = self.to

    def revert(self, graph: Graph) -> None:
        if self._from is None:
            graph.layout.pop(self.node_id, None)
        else:
            graph.layout[self.node_id] = self._from

    def merge_with(self, later: Command) -> Command | None:
        # A drag arrives as a stream of positions; collapsing them keeps one
        # undo step per gesture rather than per mouse-move event.
        if isinstance(later, MoveNode) and later.node_id == self.node_id:
            merged = MoveNode(self.node_id, later.to)
            merged._from = self._from
            return merged
        return None

    @property
    def description(self) -> str:
        return f"move {self.node_id}"


@dataclass
class SetLayout(Command):
    """Replace every position at once.

    Auto-layout moves everything; recording it as one command rather than one
    per node keeps a single, obvious undo step.
    """

    layout: dict[str, tuple[float, float]]
    _previous: dict[str, tuple[float, float]] | None = field(
        default=None, init=False, repr=False
    )

    def apply(self, graph: Graph) -> None:
        if self._previous is None:
            self._previous = dict(graph.layout)
        graph.layout = dict(self.layout)

    def revert(self, graph: Graph) -> None:
        graph.layout = dict(self._previous or {})

    @property
    def description(self) -> str:
        return "auto layout"


_UNSET = object()


@dataclass
class SetParam(Command):
    node_id: str
    name: str
    value: Any
    _previous: Any = field(default=_UNSET, init=False, repr=False)

    def apply(self, graph: Graph) -> None:
        node = graph.nodes.get(self.node_id)
        if node is None:
            raise EditError(f"no such node: {self.node_id}")
        if self._previous is _UNSET:
            self._previous = node.params.get(self.name, _UNSET)
        node.params[self.name] = self.value

    def revert(self, graph: Graph) -> None:
        node = graph.nodes.get(self.node_id)
        if node is None:
            return
        if self._previous is _UNSET:
            node.params.pop(self.name, None)
        else:
            node.params[self.name] = self._previous

    def merge_with(self, later: Command) -> Command | None:
        # Typing in a spin box emits a value per keystroke.
        if (
            isinstance(later, SetParam)
            and later.node_id == self.node_id
            and later.name == self.name
        ):
            merged = SetParam(self.node_id, self.name, later.value)
            merged._previous = self._previous
            return merged
        return None

    @property
    def description(self) -> str:
        return f"set {self.node_id}.{self.name}"


@dataclass
class RenameNode(Command):
    old_id: str
    new_id: str

    def apply(self, graph: Graph) -> None:
        self._rename(graph, self.old_id, self.new_id)

    def revert(self, graph: Graph) -> None:
        self._rename(graph, self.new_id, self.old_id)

    @staticmethod
    def _rename(graph: Graph, old: str, new: str) -> None:
        if old not in graph.nodes:
            raise EditError(f"no such node: {old}")
        if new in graph.nodes:
            raise EditError(f"a node named {new!r} already exists")
        if not new or not re.match(r"^[A-Za-z_][\w-]*$", new):
            raise EditError(
                f"{new!r} is not a valid node id (letters, digits, underscore "
                f"and hyphen; must not start with a digit)"
            )

        # Insertion order is preserved: a rename should not reshuffle the file
        # and produce a diff that touches every node.
        graph.nodes = {
            (new if key == old else key): (
                Node(id=new, block=node.block, params=node.params, line=node.line)
                if key == old
                else node
            )
            for key, node in graph.nodes.items()
        }
        graph.edges = [
            Edge(
                new if e.src == old else e.src,
                e.src_port,
                new if e.dst == old else e.dst,
                e.dst_port,
                e.line,
            )
            for e in graph.edges
        ]
        if old in graph.layout:
            graph.layout = {
                (new if key == old else key): value
                for key, value in graph.layout.items()
            }

    @property
    def description(self) -> str:
        return f"rename {self.old_id} to {self.new_id}"


# -- the session ----------------------------------------------------------


class EditSession:
    """A graph plus its undo history.

    ``modified`` tracks whether the graph differs from what was last saved,
    and is undo-aware: undoing back to the saved state clears it rather than
    leaving the document permanently dirty.
    """

    def __init__(self, graph: Graph | None = None, merge_window: bool = True) -> None:
        self.graph = graph if graph is not None else Graph()
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._saved_depth = 0
        self._merge_window = merge_window
        self._merge_armed = False

    # -- state ------------------------------------------------------------

    @property
    def modified(self) -> bool:
        return len(self._undo) != self._saved_depth

    def mark_saved(self) -> None:
        self._saved_depth = len(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_description(self) -> str:
        return self._undo[-1].description if self._undo else ""

    @property
    def redo_description(self) -> str:
        return self._redo[-1].description if self._redo else ""

    def history(self) -> list[str]:
        return [c.description for c in self._undo]

    # -- doing ------------------------------------------------------------

    def do(self, command: Command, *, allow_merge: bool = False) -> None:
        """Apply a command and push it onto the undo stack.

        If it raises, nothing is pushed: a refused edit leaves no trace in the
        history.
        """
        command.apply(self.graph)
        self._redo.clear()

        if allow_merge and self._merge_window and self._merge_armed and self._undo:
            merged = self._undo[-1].merge_with(command)
            if merged is not None:
                self._undo[-1] = merged
                return

        self._undo.append(command)
        self._merge_armed = allow_merge

    def undo(self) -> str | None:
        if not self._undo:
            return None
        command = self._undo.pop()
        command.revert(self.graph)
        self._redo.append(command)
        self._merge_armed = False
        return command.description

    def redo(self) -> str | None:
        if not self._redo:
            return None
        command = self._redo.pop()
        command.apply(self.graph)
        self._undo.append(command)
        self._merge_armed = False
        return command.description

    def end_gesture(self) -> None:
        """Close the merge window, so the next edit starts a new undo step."""
        self._merge_armed = False

    def reset(self, graph: Graph) -> None:
        """Replace the document. Clears the history, as opening a file should."""
        self.graph = graph
        self._undo.clear()
        self._redo.clear()
        self._saved_depth = 0
        self._merge_armed = False

    # -- convenience used by the editor -----------------------------------

    def add_node(
        self,
        block: str,
        position: tuple[float, float] | None = None,
        node_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Add a block, returning the id it was given."""
        spec = registry.get(block)  # raises with a suggestion if unknown
        chosen = unique_node_id(self.graph, node_id or spec.name)
        self.do(AddNode(chosen, block, params or {}, position))
        return chosen

    def remove_node(self, node_id: str) -> None:
        self.do(RemoveNode(node_id))

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> None:
        self.do(Connect(src, src_port, dst, dst_port))

    def disconnect(self, src: str, src_port: str, dst: str, dst_port: str) -> None:
        self.do(Disconnect(src, src_port, dst, dst_port))

    def move_node(self, node_id: str, position: tuple[float, float]) -> None:
        self.do(MoveNode(node_id, position), allow_merge=True)

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        self.do(SetParam(node_id, name, value), allow_merge=True)

    def rename_node(self, old_id: str, new_id: str) -> None:
        self.do(RenameNode(old_id, new_id))

    def set_layout(self, layout: dict[str, tuple[float, float]]) -> None:
        self.do(SetLayout(layout))
