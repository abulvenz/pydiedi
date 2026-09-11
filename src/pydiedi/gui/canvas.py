"""The diagram canvas: a QGraphicsScene of nodes, ports and edges.

Structurally this follows flodiedi's ``FlowRenderer``, which got the essentials
right -- a ``QGraphicsScene`` with one item per block, ports as child items,
edges as line items below the nodes. Three things are done differently:

* **A graphics item never owns a model object.** flodiedi's
  ``BlockGraphicsItem`` destructor did ``delete m_pBlock``, while
  ``FlowDiagram`` held the same pointer in its own list; deleting a scene tore
  the model apart underneath the diagram. Here items hold a node *id* and read
  the graph, which owns everything.
* **Geometry is not smuggled into the model.** flodiedi wrote
  ``__position_x``/``__position_y`` as dynamic properties onto the blocks;
  positions live in ``graph.layout``.
* **Execution order is not drawn from a stored property.** It is recomputed.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QPixmap,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QStyle,
)

from ..core import registry
from ..core.block import BlockSpec, Port, PortKind
from ..core.edit import can_connect
from ..core.graph import Edge, Graph

__all__ = ["DiagramScene", "DiagramView", "NodeItem", "PortItem", "EdgeItem"]

# Layout constants, in scene units.
NODE_WIDTH = 150.0
NODE_HEADER = 26.0
PORT_SPACING = 18.0
PORT_RADIUS = 5.0
NODE_PADDING = 10.0

COLOUR_NODE = QColor("#2f333c")
COLOUR_NODE_SELECTED = QColor("#3c4250")
COLOUR_NODE_ERROR = QColor("#5a2c2c")
COLOUR_BORDER = QColor("#4a5060")
COLOUR_BORDER_SELECTED = QColor("#7aa2f7")
COLOUR_TEXT = QColor("#d5d8e0")
COLOUR_SUBTEXT = QColor("#8b93a7")
COLOUR_EDGE = QColor("#6d7488")
COLOUR_BACKGROUND = QColor("#22242a")
COLOUR_GRID = QColor("#282b32")
COLOUR_INVALID = QColor("#f7768e")
COLOUR_VALUE = QColor("#9ece6a")

THUMBNAIL_MARGIN = 8.0
THUMBNAIL_MAX_HEIGHT = 90.0
VALUE_LINE_HEIGHT = 13.0

CLICK_SLOP = 4
"""Movement below this is a click, not a drag. Ports are small, and a few
pixels of wobble between press and release is normal."""

SNAP_RADIUS = 45.0
"""How far a dragged connection reaches for a compatible port, in scene units.

flodiedi did the same thing and it was one of the better parts of its editor:
ports are small, and having the line reach for a valid target makes wiring
much less fiddly than requiring a precise drop.
"""

MIME_BLOCK = "application/x-pydiedi-block"

# Port colours by type name. flodiedi kept the same idea in a plain-text file
# (Plugins/portcolors.txt) loaded from a Qt resource.
PORT_COLOURS = {
    "Mat": QColor("#7aa2f7"),
    "Preview": QColor("#bb9af7"),
    "int": QColor("#9ece6a"),
    "float": QColor("#e0af68"),
    "bool": QColor("#f7768e"),
    "str": QColor("#7dcfff"),
    "Path": QColor("#73daca"),
}
PORT_COLOUR_DEFAULT = QColor("#9aa0b0")


def port_colour(type_name: str) -> QColor:
    return PORT_COLOURS.get(type_name, PORT_COLOUR_DEFAULT)


class PortItem(QGraphicsItem):
    """A single connection point on a node."""

    def __init__(self, port: Port, node: NodeItem) -> None:
        super().__init__(node)
        self.port = port
        self.node = node
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{port.name}: {port.type_name}")
        self._hovered = False
        self._highlighted = False
        self._watched = False

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        r = PORT_RADIUS + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option: object, widget: object = None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        colour = port_colour(self.port.type_name)
        radius = PORT_RADIUS + (1.5 if self._hovered else 0.0)
        if self._highlighted:
            # A halo on every port the dragged connection may land on.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(colour.lighter(140), 1.5))
            painter.drawEllipse(QPointF(0, 0), radius + 4.0, radius + 4.0)
        if self._watched:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(COLOUR_VALUE, 1.5))
            painter.drawEllipse(QPointF(0, 0), radius + 3.0, radius + 3.0)
        painter.setBrush(QBrush(colour))
        painter.setPen(QPen(colour.darker(160), 1.0))
        painter.drawEllipse(QPointF(0, 0), radius, radius)
        # An unsatisfied required input is hollow, so a diagram that cannot run
        # says so before it is run.
        if self.node.port_is_unsatisfied(self):
            painter.setBrush(QBrush(COLOUR_NODE))
            painter.drawEllipse(QPointF(0, 0), radius - 2.0, radius - 2.0)

    def set_watched(self, on: bool) -> None:
        """Mark this port as one whose value is being reported."""
        if on != self._watched:
            self._watched = on
            self.update()

    def set_highlighted(self, on: bool) -> None:
        """Mark this port as a legal target while a connection is dragged."""
        self._highlighted = on
        self.update()

    def hoverEnterEvent(self, event: object) -> None:  # noqa: N802
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event: object) -> None:  # noqa: N802
        self._hovered = False
        self.update()


class NodeItem(QGraphicsItem):
    """One block instance. Holds a node id, never a model object."""

    def __init__(self, node_id: str, spec: BlockSpec, scene: DiagramScene) -> None:
        super().__init__()
        self.node_id = node_id
        self.spec = spec
        self._scene = scene
        self.has_error = False
        self._error_text = ""

        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(1.0)

        self._thumbnail: QPixmap | None = None
        self._value_text: str = ""
        self._inputs: list[PortItem] = []
        self._outputs: list[PortItem] = []
        self._build_ports()
        self._height = self._compute_height()
        self._place_ports()

    # -- geometry ---------------------------------------------------------

    def _build_ports(self) -> None:
        for port in self.spec.inputs:
            self._inputs.append(PortItem(port, self))
        for port in self.spec.outputs:
            self._outputs.append(PortItem(port, self))

    def _compute_height(self) -> float:
        rows = max(len(self._inputs), len(self._outputs), 1)
        height = NODE_HEADER + rows * PORT_SPACING + NODE_PADDING
        if self._thumbnail is not None:
            height += self._thumbnail_rect().height() + 6.0
        if self._value_text:
            height += VALUE_LINE_HEIGHT
        return height

    def _thumbnail_rect(self) -> QRectF:
        """Where the inline preview goes, scaled to fit the node's width."""
        if self._thumbnail is None:
            return QRectF()
        available = NODE_WIDTH - 2 * THUMBNAIL_MARGIN
        scale = min(
            available / max(1, self._thumbnail.width()),
            THUMBNAIL_MAX_HEIGHT / max(1, self._thumbnail.height()),
        )
        width = self._thumbnail.width() * scale
        height = self._thumbnail.height() * scale
        rows = max(len(self._inputs), len(self._outputs), 1)
        top = NODE_HEADER + rows * PORT_SPACING + 3.0
        return QRectF((NODE_WIDTH - width) / 2.0, top, width, height)

    def set_thumbnail(self, pixmap: QPixmap | None) -> None:
        """Show an image inside the node.

        flodiedi did this too, but by hosting a real ``QWidget`` in a
        ``QGraphicsProxyWidget`` that the *block* created -- which is how
        painting ended up on the worker thread. Here the block returns a
        Preview and the item draws it; nothing crosses a thread boundary that
        is not already a queued signal.
        """
        if pixmap is None and self._thumbnail is None:
            return
        self.prepareGeometryChange()
        self._thumbnail = pixmap
        self._height = self._compute_height()
        self._place_ports()
        self.update()

    def set_value_text(self, text: str) -> None:
        """A one-line value readout under the ports, for a watched port."""
        if text == self._value_text:
            return
        self.prepareGeometryChange()
        self._value_text = text
        self._height = self._compute_height()
        self.update()

    def _place_ports(self) -> None:
        for index, item in enumerate(self._inputs):
            item.setPos(0.0, NODE_HEADER + (index + 0.5) * PORT_SPACING)
        for index, item in enumerate(self._outputs):
            item.setPos(NODE_WIDTH, NODE_HEADER + (index + 0.5) * PORT_SPACING)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-2, -2, NODE_WIDTH + 4, self._height + 4)

    def port_item(self, name: str, kind: PortKind) -> PortItem | None:
        items = self._inputs if kind is PortKind.IN else self._outputs
        return next((i for i in items if i.port.name == name), None)

    def port_is_unsatisfied(self, port_item: PortItem) -> bool:
        """Whether this input would stop the diagram from running.

        A required input is satisfied by *either* a connection or a parameter
        value -- ``imread``'s ``path`` has no default but is normally typed in,
        not wired. Marking it unsatisfied merely because nothing is connected
        would flag a perfectly valid diagram.
        """
        port = port_item.port
        if port.kind is not PortKind.IN or not port.required:
            return False
        return not self._scene.is_satisfied(self.node_id, port.name)

    # -- painting ---------------------------------------------------------

    def paint(self, painter: QPainter, option: object, widget: object = None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        selected = bool(option.state & QStyle.State_Selected)  # type: ignore[attr-defined]

        body = QRectF(0, 0, NODE_WIDTH, self._height)
        path = QPainterPath()
        path.addRoundedRect(body, 5, 5)

        if self.has_error:
            fill = COLOUR_NODE_ERROR
        else:
            fill = COLOUR_NODE_SELECTED if selected else COLOUR_NODE
        painter.setBrush(QBrush(fill))
        painter.setPen(
            QPen(COLOUR_BORDER_SELECTED if selected else COLOUR_BORDER, 1.6 if selected else 1.0)
        )
        painter.drawPath(path)

        # Header separator
        painter.setPen(QPen(COLOUR_BORDER, 1.0))
        painter.drawLine(QPointF(1, NODE_HEADER), QPointF(NODE_WIDTH - 1, NODE_HEADER))

        title_font = QFont()
        title_font.setPointSizeF(9.0)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(COLOUR_TEXT))
        painter.drawText(
            QRectF(8, 0, NODE_WIDTH - 16, NODE_HEADER),
            Qt.AlignVCenter | Qt.AlignLeft,
            _elide(self.node_id, title_font, NODE_WIDTH - 16),
        )

        label_font = QFont()
        label_font.setPointSizeF(7.5)
        painter.setFont(label_font)
        painter.setPen(QPen(COLOUR_SUBTEXT))
        for index, item in enumerate(self._inputs):
            y = NODE_HEADER + index * PORT_SPACING
            painter.drawText(
                QRectF(10, y, NODE_WIDTH / 2 - 12, PORT_SPACING),
                Qt.AlignVCenter | Qt.AlignLeft,
                item.port.name,
            )
        for index, item in enumerate(self._outputs):
            y = NODE_HEADER + index * PORT_SPACING
            painter.drawText(
                QRectF(NODE_WIDTH / 2, y, NODE_WIDTH / 2 - 10, PORT_SPACING),
                Qt.AlignVCenter | Qt.AlignRight,
                item.port.name,
            )

        if self._thumbnail is not None:
            target = self._thumbnail_rect()
            painter.setPen(QPen(COLOUR_BORDER, 1.0))
            painter.setBrush(QBrush(QColor("#15171b")))
            painter.drawRect(target.adjusted(-1, -1, 1, 1))
            painter.drawPixmap(target, self._thumbnail, QRectF(self._thumbnail.rect()))

        if self._value_text:
            painter.setFont(label_font)
            painter.setPen(QPen(COLOUR_VALUE))
            painter.drawText(
                QRectF(
                    6,
                    self._height - NODE_PADDING - VALUE_LINE_HEIGHT,
                    NODE_WIDTH - 12,
                    VALUE_LINE_HEIGHT,
                ),
                Qt.AlignVCenter | Qt.AlignLeft,
                _elide(self._value_text, label_font, NODE_WIDTH - 12),
            )

        # The block name, small, under the node id -- two 'cvt_color' nodes
        # named gray_live and gray_ref still show what they are.
        painter.setFont(label_font)
        painter.setPen(QPen(COLOUR_SUBTEXT))
        painter.drawText(
            QRectF(8, self._height - NODE_PADDING - 2, NODE_WIDTH - 16, NODE_PADDING + 2),
            Qt.AlignVCenter | Qt.AlignLeft,
            _elide(self.spec.name, label_font, NODE_WIDTH - 16),
        )

    def set_error(self, message: str | None) -> None:
        self.has_error = bool(message)
        self._error_text = message or ""
        self.setToolTip(message or self.spec.signature())
        self.update()

    def itemChange(self, change: object, value: object) -> object:  # noqa: N802
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._scene.node_moved(self.node_id, self.pos())
        return super().itemChange(change, value)  # type: ignore[arg-type]


def _elide(text: str, font: QFont, width: float) -> str:
    metrics = QFontMetrics(font)
    return metrics.elidedText(text, Qt.ElideRight, int(width))


class EdgeItem(QGraphicsPathItem):
    """A connection, drawn as a horizontal bezier between two ports."""

    def __init__(self, edge: Edge, source: PortItem, target: PortItem) -> None:
        super().__init__()
        self.edge = edge
        self.source = source
        self.target = target
        self.setZValue(-1.0)
        self.setPen(QPen(COLOUR_EDGE, 1.8, Qt.SolidLine, Qt.RoundCap))
        self.setToolTip(str(edge))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._hovered = False
        self.refresh()

    def shape(self) -> QPainterPath:
        """A fat invisible outline, so a 2px curve can actually be clicked."""
        stroker = QPainterPathStroker()
        stroker.setWidth(12.0)
        return stroker.createStroke(self.path())

    def hoverEnterEvent(self, event: object) -> None:  # noqa: N802
        self._hovered = True
        self.refresh()

    def hoverLeaveEvent(self, event: object) -> None:  # noqa: N802
        self._hovered = False
        self.refresh()

    def refresh(self) -> None:
        start = self.source.scenePos()
        end = self.target.scenePos()
        # A horizontal tangent proportional to the gap reads as a cable and
        # keeps edges distinguishable when nodes overlap vertically.
        stretch = max(40.0, abs(end.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + stretch, start.y()),
            QPointF(end.x() - stretch, end.y()),
            end,
        )
        self.setPath(path)
        colour = port_colour(self.source.port.type_name)
        if self.isSelected():
            self.setPen(QPen(COLOUR_BORDER_SELECTED, 2.8, Qt.SolidLine, Qt.RoundCap))
        elif self._hovered:
            self.setPen(QPen(colour.lighter(140), 2.6, Qt.SolidLine, Qt.RoundCap))
        else:
            self.setPen(QPen(colour.darker(130), 1.8, Qt.SolidLine, Qt.RoundCap))

    def itemChange(self, change: object, value: object) -> object:  # noqa: N802
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.refresh()
        return super().itemChange(change, value)  # type: ignore[arg-type]


class PendingConnection(QGraphicsPathItem):
    """The rubber band drawn while a connection is being dragged."""

    def __init__(self, origin: PortItem) -> None:
        super().__init__()
        self.origin = origin
        self.setZValue(100.0)
        self._valid = False
        self.setPen(QPen(COLOUR_INVALID, 2.0, Qt.DashLine, Qt.RoundCap))

    def set_endpoint(self, point: QPointF, valid: bool) -> None:
        self._valid = valid
        start = self.origin.scenePos()
        stretch = max(30.0, abs(point.x() - start.x()) * 0.5)
        if self.origin.port.kind is PortKind.OUT:
            c1, c2 = QPointF(start.x() + stretch, start.y()), QPointF(point.x() - stretch, point.y())
        else:
            c1, c2 = QPointF(start.x() - stretch, start.y()), QPointF(point.x() + stretch, point.y())
        path = QPainterPath(start)
        path.cubicTo(c1, c2, point)
        self.setPath(path)
        colour = port_colour(self.origin.port.type_name) if valid else COLOUR_INVALID
        self.setPen(QPen(colour, 2.0, Qt.SolidLine if valid else Qt.DashLine, Qt.RoundCap))


class DiagramScene(QGraphicsScene):
    """Renders a :class:`~pydiedi.core.graph.Graph` and reports edits back.

    The scene never mutates the graph. It reports what the user did, and the
    window turns that into a command on the
    :class:`~pydiedi.core.edit.EditSession`, so every change is undoable.
    """

    selection_changed_to = Signal(object)
    """The selected node id, or ``None``."""
    node_moved_to = Signal(str, float, float)
    """A node was dragged. The scene never writes to the graph itself."""
    layout_computed = Signal(dict)
    """Positions the scene worked out for nodes that had none."""
    node_move_finished = Signal()
    """A drag gesture ended, so the next move starts a new undo step."""
    connect_requested = Signal(str, str, str, str)
    """src, src_port, dst, dst_port -- the user dropped a connection."""
    connection_refused = Signal(str)
    """Why a dropped connection was not made."""
    disconnect_requested = Signal(object)
    """An :class:`~pydiedi.core.graph.Edge` the user wants removed."""
    delete_requested = Signal(list, list)
    """node ids and Edges the user wants removed."""
    block_dropped = Signal(str, float, float)
    """A block name dragged in from the palette, and where it was dropped."""
    rename_requested = Signal(str)
    """The node id the user wants to rename."""
    watches_changed = Signal(set)
    """The set of (node id, port) the GUI wants values for."""
    port_clicked = Signal(str, str, bool)
    """node id, port, now watched -- a port was clicked without dragging."""

    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(COLOUR_BACKGROUND))
        self.graph: Graph | None = None
        self._nodes: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._connected: set[tuple[str, str]] = set()
        # Placing items calls setPos(), which Qt reports as ItemPositionHasChanged
        # exactly as it does for a user drag. Without this guard, merely opening
        # a file would mark the document modified.
        self._building = False
        self._pending: PendingConnection | None = None
        self._candidates: list[PortItem] = []
        self._watched: set[tuple[str, str]] = set()
        self._press_point: QPointF | None = None
        self.selectionChanged.connect(self._on_selection_changed)

    # -- building ---------------------------------------------------------

    def set_graph(self, graph: Graph, specs: dict[str, BlockSpec]) -> None:
        """Rebuild the scene for ``graph``.

        ``specs`` maps node id to its block spec, resolved by the caller so
        that an unknown block is reported once rather than raising here.
        """
        self._building = True
        try:
            self.clear()
            self._nodes.clear()
            self._edges.clear()
            self.graph = graph
            self._connected = {(e.dst, e.dst_port) for e in graph.edges}

            for node_id, node in graph.nodes.items():
                spec = specs.get(node_id)
                if spec is None:
                    continue
                item = NodeItem(node_id, spec, self)
                position = graph.layout.get(node_id)
                if position is not None:
                    item.setPos(QPointF(*position))
                self.addItem(item)
                self._nodes[node_id] = item

            computed: dict[str, tuple[float, float]] = {}
            if any(n not in graph.layout for n in self._nodes):
                computed = self.auto_layout()

            for edge in graph.edges:
                self._add_edge_item(edge)
        finally:
            self._building = False

        self.tighten_scene_rect()

        # A diagram that arrived without positions has just been given some,
        # which is a change worth saving -- unlike merely opening one.
        if computed:
            self.layout_computed.emit(computed)

    def tighten_scene_rect(self, margin: float = 400.0) -> None:
        """Keep the scrollable area related to the content.

        A generous margin, so dragging a node past the edge does not hit an
        invisible wall, but not the unbounded rect the scene would keep.
        """
        rect = self.itemsBoundingRect()
        if rect.isEmpty():
            self.setSceneRect(-500, -400, 1000, 800)
        else:
            self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

    def _add_edge_item(self, edge: Edge) -> None:
        source_node = self._nodes.get(edge.src)
        target_node = self._nodes.get(edge.dst)
        if source_node is None or target_node is None:
            return
        source = source_node.port_item(edge.src_port, PortKind.OUT)
        target = target_node.port_item(edge.dst_port, PortKind.IN)
        if source is None or target is None:
            return
        item = EdgeItem(edge, source, target)
        self.addItem(item)
        self._edges.append(item)

    def auto_layout(self) -> dict[str, tuple[float, float]]:
        """Place nodes in dependency columns, returning the new positions.

        The scene applies them to its items but never writes them into the
        graph: the caller turns the result into a command, so auto-layout is
        one undo step like any other edit.

        flodiedi shelled out to Graphviz for this, through a wrapper around
        ``libgraph`` -- a library Graphviz removed in 2012, which is one of the
        reasons the C++ build no longer links. A depth-based column layout
        needs no dependency and is adequate for a dataflow graph.
        """
        if self.graph is None:
            return {}
        depth: dict[str, int] = {}
        for node_id in self._topological_ids():
            incoming = self.graph.incoming(node_id)
            depth[node_id] = (
                0
                if not incoming
                else 1 + max(depth.get(e.src, 0) for e in incoming)
            )

        columns: dict[int, list[str]] = {}
        for node_id, column in sorted(depth.items(), key=lambda kv: (kv[1], kv[0])):
            columns.setdefault(column, []).append(node_id)

        computed: dict[str, tuple[float, float]] = {}
        was_building, self._building = self._building, True
        try:
            for column, ids in columns.items():
                for row, node_id in enumerate(ids):
                    item = self._nodes.get(node_id)
                    if item is None:
                        continue
                    x = column * (NODE_WIDTH + 90.0)
                    y = row * 140.0 - (len(ids) - 1) * 70.0
                    item.setPos(QPointF(x, y))
                    computed[node_id] = (x, y)
        finally:
            self._building = was_building
        self.refresh_edges()
        self.tighten_scene_rect()
        return computed

    def _topological_ids(self) -> list[str]:
        from ..core.executor import CycleError, topological_order

        if self.graph is None:
            return []
        try:
            return topological_order(self.graph)
        except CycleError:
            # A cyclic diagram is still worth looking at; it just cannot run.
            return list(self.graph.nodes)

    # -- queries used by items -------------------------------------------

    def is_connected(self, node_id: str, port_name: str) -> bool:
        return (node_id, port_name) in self._connected

    def is_satisfied(self, node_id: str, port_name: str) -> bool:
        """Whether an input has a value from either a connection or a param."""
        if (node_id, port_name) in self._connected:
            return True
        if self.graph is None:
            return False
        node = self.graph.nodes.get(node_id)
        return node is not None and port_name in node.params

    # -- updates ----------------------------------------------------------

    def refresh_edges(self) -> None:
        for edge in self._edges:
            edge.refresh()

    def node_moved(self, node_id: str, position: QPointF) -> None:
        self.refresh_edges()
        if not self._building:
            self.node_moved_to.emit(node_id, position.x(), position.y())

    def set_errors(self, errors: dict[str, str]) -> None:
        """Colour failing nodes. The editor's equivalent of flodiedi's red block."""
        for node_id, item in self._nodes.items():
            item.set_error(errors.get(node_id))

    def show_previews(self, previews_by_node: dict[str, list]) -> None:
        """Draw each preview inside the node that produced it."""
        from .preview import to_qimage

        for node_id, item in self._nodes.items():
            previews = previews_by_node.get(node_id)
            image = previews[0].image if previews else None
            if image is None:
                item.set_thumbnail(None)
                continue
            try:
                item.set_thumbnail(QPixmap.fromImage(to_qimage(image)))
            except ValueError:
                item.set_thumbnail(None)
        self.refresh_edges()

    def clear_previews(self) -> None:
        for item in self._nodes.values():
            item.set_thumbnail(None)
            item.set_value_text("")
        self.refresh_edges()

    # -- watching port values ---------------------------------------------

    def toggle_watch(self, node_id: str, port: str) -> bool:
        """Start or stop watching a port. Returns whether it is now watched."""
        key = (node_id, port)
        if key in self._watched:
            self._watched.discard(key)
            watched_now = False
        else:
            self._watched.add(key)
            watched_now = True
        self._apply_watch_marks()
        self.watches_changed.emit(set(self._watched))
        return watched_now

    def watched_ports(self) -> set[tuple[str, str]]:
        return set(self._watched)

    def clear_watches(self) -> None:
        if not self._watched:
            return
        self._watched.clear()
        self._apply_watch_marks()
        for item in self._nodes.values():
            item.set_value_text("")
        self.watches_changed.emit(set())

    def _apply_watch_marks(self) -> None:
        for node_id, node_item in self._nodes.items():
            for child in node_item.childItems():
                if isinstance(child, PortItem):
                    child.set_watched((node_id, child.port.name) in self._watched)

    def show_port_values(self, values: list) -> None:
        """Display the latest value of every watched port on its node."""
        by_node: dict[str, list[str]] = {}
        for value in values:
            by_node.setdefault(value.node_id, []).append(
                f"{value.port} = {value.summary}"
            )
        for node_id, item in self._nodes.items():
            item.set_value_text(" | ".join(by_node.get(node_id, ())))
        self.refresh_edges()

    def clear_errors(self) -> None:
        for item in self._nodes.values():
            item.set_error(None)

    def _on_selection_changed(self) -> None:
        items = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        self.selection_changed_to.emit(items[0].node_id if len(items) == 1 else None)

    def select_node(self, node_id: str | None) -> None:
        self.clearSelection()
        if node_id and node_id in self._nodes:
            self._nodes[node_id].setSelected(True)

    def selected_node_ids(self) -> list[str]:
        return [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]

    # -- incremental updates ----------------------------------------------
    #
    # Rebuilding the whole scene after every edit would be correct but would
    # drop the selection and flicker. These keep the view in step with single
    # changes; rebuild() handles the rest.

    def add_node_item(self, node_id: str) -> NodeItem | None:
        if self.graph is None or node_id in self._nodes:
            return None
        node = self.graph.nodes.get(node_id)
        if node is None:
            return None
        try:
            spec = registry.get(node.block)
        except registry.UnknownBlockError:
            return None
        # Guarded, because setPos() is reported as ItemPositionHasChanged
        # exactly as a user drag is. Without this, adding a block would also
        # record a move, and one palette click would cost two undo steps.
        was_building, self._building = self._building, True
        try:
            item = NodeItem(node_id, spec, self)
            position = self.graph.layout.get(node_id)
            if position is not None:
                item.setPos(QPointF(*position))
            self.addItem(item)
        finally:
            self._building = was_building
        self._nodes[node_id] = item
        self.tighten_scene_rect()
        return item

    def remove_node_item(self, node_id: str) -> None:
        for edge_item in [
            e for e in self._edges if e.edge.src == node_id or e.edge.dst == node_id
        ]:
            self._remove_edge_item(edge_item)
        item = self._nodes.pop(node_id, None)
        if item is not None:
            self.removeItem(item)

    def add_edge_item(self, edge: Edge) -> None:
        self._connected.add((edge.dst, edge.dst_port))
        self._add_edge_item(edge)
        self._refresh_ports()

    def remove_edge_item(self, edge: Edge) -> None:
        for item in [e for e in self._edges if e.edge == edge]:
            self._remove_edge_item(item)
        self._connected.discard((edge.dst, edge.dst_port))
        self._refresh_ports()

    def _remove_edge_item(self, item: EdgeItem) -> None:
        self.removeItem(item)
        self._edges.remove(item)

    def _refresh_ports(self) -> None:
        """Redraw ports, whose hollow/filled state depends on connections."""
        for node in self._nodes.values():
            for child in node.childItems():
                child.update()

    def rebuild(self, specs: dict[str, BlockSpec]) -> None:
        """Re-render the current graph, keeping the selection where possible."""
        if self.graph is None:
            return
        selected = [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]
        self.set_graph(self.graph, specs)
        for node_id in selected:
            if node_id in self._nodes:
                self._nodes[node_id].setSelected(True)

    # -- interaction ------------------------------------------------------

    def _port_at(self, point: QPointF) -> PortItem | None:
        for item in self.items(point):
            if isinstance(item, PortItem):
                return item
        return None

    def _snap_target(self, point: QPointF) -> PortItem | None:
        """The nearest compatible port within :data:`SNAP_RADIUS`."""
        best: PortItem | None = None
        best_distance = SNAP_RADIUS
        for candidate in self._candidates:
            delta = candidate.scenePos() - point
            distance = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
            if distance < best_distance:
                best, best_distance = candidate, distance
        return best

    def _collect_candidates(self, origin: PortItem) -> list[PortItem]:
        """Every port the dragged connection could legally land on."""
        if self.graph is None:
            return []
        origin_node = origin.node.node_id
        candidates: list[PortItem] = []
        for node_id, node_item in self._nodes.items():
            if node_id == origin_node:
                continue
            for child in node_item.childItems():
                if not isinstance(child, PortItem) or child.port.kind is origin.port.kind:
                    continue
                src, dst = (origin, child) if origin.port.kind is PortKind.OUT else (child, origin)
                if can_connect(
                    self.graph,
                    src.node.node_id,
                    src.port.name,
                    dst.node.node_id,
                    dst.port.name,
                ) is None:
                    candidates.append(child)
        return candidates

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:  # type: ignore[attr-defined]
            port = self._port_at(event.scenePos())  # type: ignore[attr-defined]
            if port is not None:
                self._press_point = event.scenePos()  # type: ignore[attr-defined]
                self._pending = PendingConnection(port)
                self._candidates = self._collect_candidates(port)
                for candidate in self._candidates:
                    candidate.set_highlighted(True)
                self.addItem(self._pending)
                self._pending.set_endpoint(event.scenePos(), False)  # type: ignore[attr-defined]
                event.accept()  # type: ignore[attr-defined]
                return
        super().mousePressEvent(event)  # type: ignore[arg-type]

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802
        if self._pending is not None:
            point = event.scenePos()  # type: ignore[attr-defined]
            target = self._snap_target(point)
            self._pending.set_endpoint(
                target.scenePos() if target is not None else point, target is not None
            )
            event.accept()  # type: ignore[attr-defined]
            return
        super().mouseMoveEvent(event)  # type: ignore[arg-type]

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802
        if self._pending is not None:
            origin = self._pending.origin
            point = event.scenePos()  # type: ignore[attr-defined]
            target = self._snap_target(point) or self._port_at(point)
            moved = self._press_point is not None and (
                (point - self._press_point).manhattanLength() > CLICK_SLOP
            )
            self._end_pending()
            self._press_point = None

            if target is not None and target is not origin:
                self._request_connection(origin, target)
            elif not moved:
                # Pressed and released on the same port without dragging: the
                # user wants to see what is on it, not to wire it up.
                node_id = origin.node.node_id
                watched = self.toggle_watch(node_id, origin.port.name)
                self.port_clicked.emit(node_id, origin.port.name, watched)
            event.accept()  # type: ignore[attr-defined]
            return
        super().mouseReleaseEvent(event)  # type: ignore[arg-type]
        # A drag of one or more nodes has ended; close the undo merge window so
        # the next drag is a separate step.
        if event.button() == Qt.LeftButton:  # type: ignore[attr-defined]
            self.node_move_finished.emit()

    def _end_pending(self) -> None:
        for candidate in self._candidates:
            candidate.set_highlighted(False)
        self._candidates = []
        if self._pending is not None:
            self.removeItem(self._pending)
            self._pending = None

    def _request_connection(self, origin: PortItem, target: PortItem) -> None:
        if origin.port.kind is target.port.kind:
            self.connection_refused.emit("connections run from an output to an input")
            return
        source, sink = (
            (origin, target) if origin.port.kind is PortKind.OUT else (target, origin)
        )
        if self.graph is None:
            return
        reason = can_connect(
            self.graph,
            source.node.node_id,
            source.port.name,
            sink.node.node_id,
            sink.port.name,
        )
        if reason is not None:
            self.connection_refused.emit(reason)
            return
        self.connect_requested.emit(
            source.node.node_id, source.port.name, sink.node.node_id, sink.port.name
        )

    def keyPressEvent(self, event: object) -> None:  # noqa: N802
        key = event.key()  # type: ignore[attr-defined]
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._request_delete()
            event.accept()  # type: ignore[attr-defined]
            return
        if key == Qt.Key_F2:
            nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
            if len(nodes) == 1:
                self.rename_requested.emit(nodes[0].node_id)
                event.accept()  # type: ignore[attr-defined]
                return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def _request_delete(self) -> None:
        node_ids = [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]
        edges = [i.edge for i in self.selectedItems() if isinstance(i, EdgeItem)]
        if node_ids or edges:
            self.delete_requested.emit(node_ids, edges)

    # -- drag and drop from the palette -----------------------------------

    def dragEnterEvent(self, event: object) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_BLOCK):  # type: ignore[attr-defined]
            event.acceptProposedAction()  # type: ignore[attr-defined]
            return
        super().dragEnterEvent(event)  # type: ignore[arg-type]

    def dragMoveEvent(self, event: object) -> None:  # noqa: N802
        if event.mimeData().hasFormat(MIME_BLOCK):  # type: ignore[attr-defined]
            event.acceptProposedAction()  # type: ignore[attr-defined]
            return
        super().dragMoveEvent(event)  # type: ignore[arg-type]

    def dropEvent(self, event: object) -> None:  # noqa: N802
        data = event.mimeData()  # type: ignore[attr-defined]
        if data.hasFormat(MIME_BLOCK):
            name = bytes(data.data(MIME_BLOCK)).decode("utf-8")
            point = event.scenePos()  # type: ignore[attr-defined]
            self.block_dropped.emit(name, point.x(), point.y())
            event.acceptProposedAction()  # type: ignore[attr-defined]
            return
        super().dropEvent(event)  # type: ignore[arg-type]


class DiagramView(QGraphicsView):
    """The canvas widget: pans with the middle button, zooms with Ctrl+wheel."""

    def __init__(self, scene: DiagramScene, parent: object | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self._zoom = 1.0
        # While true, every resize refits.
        #
        # Fitting once -- at any chosen moment -- is not enough, because the
        # requested window size is only a request. Under a tiling window
        # manager it is ignored outright: the window is mapped and then resized
        # to its tile, so a fit computed at show() time is scaled for a
        # viewport that never existed. Tiling also resizes the window later,
        # whenever the layout changes.
        #
        # So the initial view keeps fitting until the user takes control by
        # zooming or panning.
        self._auto_fit = False

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        # A grid gives the eye something to judge position against while
        # dragging; it is cheap because only the exposed rect is drawn.
        step = 25.0
        if self._zoom < 0.5:
            return
        left = rect.left() - (rect.left() % step)
        top = rect.top() - (rect.top() % step)
        painter.setPen(QPen(COLOUR_GRID, 1.0))
        x = left
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = top
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        # Ctrl+wheel zooms, plain wheel scrolls -- as flodiedi's MyGraphicsView
        # did, and using angleDelta() rather than the delta()/orientation()
        # pair that Qt6 removed.
        if event.modifiers() & Qt.ControlModifier:  # type: ignore[attr-defined]
            steps = event.angleDelta().y() / 120.0  # type: ignore[attr-defined]
            self.zoom_by(1.15**steps)
            event.accept()  # type: ignore[attr-defined]
        else:
            self._user_took_control()  # scrolling is panning
            super().wheelEvent(event)  # type: ignore[arg-type]

    def zoom_by(self, factor: float) -> None:
        target = self._zoom * factor
        if not 0.15 <= target <= 6.0:
            return
        self._user_took_control()
        self._zoom = target
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        self._user_took_control()
        self.resetTransform()
        self._zoom = 1.0

    def fit_contents(self) -> None:
        scene = self.scene()
        items_rect = scene.itemsBoundingRect()
        if items_rect.isEmpty():
            return
        rect = items_rect.adjusted(-40, -40, 40, 40)
        # Without this, fitInView sets the right transform but the scrollable
        # area still comes from sceneRect -- which QGraphicsScene grows to
        # cover everything ever added and never shrinks -- so the diagram ends
        # up correctly scaled and scrolled off to one side.
        scene.setSceneRect(rect)
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def enable_auto_fit(self) -> None:
        """Keep the whole diagram in view until the user zooms or pans."""
        self._auto_fit = True
        if self._viewport_is_usable():
            self.fit_contents()

    def _viewport_is_usable(self) -> bool:
        size = self.viewport().size()
        return size.width() > 50 and size.height() > 50

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self._auto_fit and self._viewport_is_usable():
            self.fit_contents()

    def _user_took_control(self) -> None:
        self._auto_fit = False
