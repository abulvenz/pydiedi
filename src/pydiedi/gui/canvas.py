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
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
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

from ..core.block import BlockSpec, Port, PortKind
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

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt naming
        r = PORT_RADIUS + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option: object, widget: object = None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        colour = port_colour(self.port.type_name)
        radius = PORT_RADIUS + (1.5 if self._hovered else 0.0)
        painter.setBrush(QBrush(colour))
        painter.setPen(QPen(colour.darker(160), 1.0))
        painter.drawEllipse(QPointF(0, 0), radius, radius)
        # An unsatisfied required input is hollow, so a diagram that cannot run
        # says so before it is run.
        if self.node.port_is_unsatisfied(self):
            painter.setBrush(QBrush(COLOUR_NODE))
            painter.drawEllipse(QPointF(0, 0), radius - 2.0, radius - 2.0)

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
        return NODE_HEADER + rows * PORT_SPACING + NODE_PADDING

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

        # The block name, small, under the node id -- two 'cvt_color' nodes
        # named gray_live and gray_ref still show what they are.
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
        self.setPen(QPen(colour.darker(130), 1.8, Qt.SolidLine, Qt.RoundCap))


class DiagramScene(QGraphicsScene):
    """Renders a :class:`~pydiedi.core.graph.Graph` and reports edits back."""

    selection_changed_to = Signal(object)
    """The selected node id, or ``None``."""
    layout_changed = Signal()
    """A node was moved; the graph's layout section is now out of date on disk."""

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

            missing_layout = [n for n in self._nodes if n not in graph.layout]
            if missing_layout:
                self.auto_layout()

            for edge in graph.edges:
                self._add_edge_item(edge)
        finally:
            self._building = False

        self.tighten_scene_rect()

        # A diagram that arrived without positions has just been given some,
        # which is a change worth saving -- unlike merely opening one.
        if missing_layout:
            self.layout_changed.emit()

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

    def auto_layout(self) -> None:
        """Place nodes in dependency columns.

        flodiedi shelled out to Graphviz for this, through a wrapper around
        ``libgraph`` -- a library Graphviz removed in 2012, which is one of the
        reasons the C++ build no longer links. A depth-based column layout
        needs no dependency and is adequate for a dataflow graph.
        """
        if self.graph is None:
            return
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

        for column, ids in columns.items():
            for row, node_id in enumerate(ids):
                item = self._nodes.get(node_id)
                if item is None:
                    continue
                x = column * (NODE_WIDTH + 90.0)
                y = row * 140.0 - (len(ids) - 1) * 70.0
                item.setPos(QPointF(x, y))
                if self.graph is not None:
                    self.graph.layout[node_id] = (x, y)
        self.refresh_edges()
        if not self._building:
            self.layout_changed.emit()

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
        if self.graph is not None:
            self.graph.layout[node_id] = (position.x(), position.y())
        self.refresh_edges()
        if not self._building:
            self.layout_changed.emit()

    def set_errors(self, errors: dict[str, str]) -> None:
        """Colour failing nodes. The editor's equivalent of flodiedi's red block."""
        for node_id, item in self._nodes.items():
            item.set_error(errors.get(node_id))

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
