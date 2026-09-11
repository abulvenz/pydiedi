"""The block palette and the parameter editor.

Both are generated entirely from :class:`~pydiedi.core.block.BlockSpec`, which
is itself derived from a function signature. flodiedi needed a whole imported
library for the right-hand side of this -- ``Utils/QPropertyEditor``, a
GPLv3 third-party ``QAbstractItemModel`` stack driven by ``Q_PROPERTY``
declarations and ``Q_CLASSINFO`` hint strings, plus a second home-grown
mechanism next to it. Type hints carry the same information.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import registry
from ..core.block import BlockSpec, Port
from ..core.graph import Graph
from .canvas import MIME_BLOCK

__all__ = ["BlockPalette", "ParameterEditor"]


class BlockPalette(QTreeWidget):
    """Every registered block, grouped by category.

    A block reaches the canvas either by being dragged onto it or by being
    double-clicked, which drops it in the middle of the view. Both exist
    because dragging is discoverable and double-clicking is faster.
    """

    block_activated = Signal(str)
    """A block name the user chose to add, without saying where."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setAlternatingRowColors(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.itemDoubleClicked.connect(self._on_activated)
        self.reload()

    def block_name_of(self, item: QTreeWidgetItem | None) -> str | None:
        return item.data(0, Qt.UserRole) if item is not None else None

    def startDrag(self, actions: object) -> None:  # noqa: N802 - Qt naming
        name = self.block_name_of(self.currentItem())
        if not name:
            return
        payload = QMimeData()
        payload.setData(MIME_BLOCK, name.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(payload)
        drag.setPixmap(_drag_pixmap(name))
        drag.exec(Qt.CopyAction)

    def reload(self) -> None:
        self.clear()
        for category, specs in registry.by_category().items():
            group = QTreeWidgetItem([category])
            group.setFlags(Qt.ItemIsEnabled)
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            for spec in specs:
                child = QTreeWidgetItem([spec.name])
                child.setData(0, Qt.UserRole, spec.name)
                child.setToolTip(0, _spec_tooltip(spec))
                group.addChild(child)
            self.addTopLevelItem(group)
            group.setExpanded(True)

    def _on_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        name = item.data(0, Qt.UserRole)
        if name:
            self.block_activated.emit(name)


def _drag_pixmap(name: str) -> QPixmap:
    """A small label that follows the cursor, so the drag is visible."""
    font = QFont()
    font.setPointSizeF(9.0)
    width = QFontMetrics(font).horizontalAdvance(name) + 16
    pixmap = QPixmap(width, 24)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(QColor("#3c4250"))
    painter.setPen(QColor("#7aa2f7"))
    painter.drawRoundedRect(0, 0, width - 1, 23, 4, 4)
    painter.setFont(font)
    painter.setPen(QColor("#d5d8e0"))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, name)
    painter.end()
    return pixmap


def _spec_tooltip(spec: BlockSpec) -> str:
    lines = [f"<b>{spec.signature()}</b>"]
    if spec.doc:
        lines.append(spec.doc.splitlines()[0])
    if spec.is_stateful:
        lines.append("<i>keeps state between sweeps</i>")
    return "<br>".join(lines)


class ParameterEditor(QWidget):
    """Edits the parameters of one node.

    Each editor widget is chosen from the port's declared type: a spin box for
    a number, a combo box for an enum, a file chooser for a ``Path``. A port
    that is fed by an edge is shown disabled, because a value there would
    conflict with the connection -- which the validator rejects.
    """

    parameter_changed = Signal(str, str, object)
    """node id, parameter name, new value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._graph: Graph | None = None
        self._node_id: str | None = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._placeholder = QLabel("Select a node to edit its parameters.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: #667;")
        self._layout.addWidget(self._placeholder)
        self._form_host: QWidget | None = None

    def set_graph(self, graph: Graph | None) -> None:
        self._graph = graph
        self.show_node(None)

    def show_node(self, node_id: str | None) -> None:
        self._node_id = node_id
        if self._form_host is not None:
            self._form_host.deleteLater()
            self._form_host = None

        if self._graph is None or node_id is None or node_id not in self._graph.nodes:
            self._placeholder.setVisible(True)
            return

        node = self._graph.nodes[node_id]
        try:
            spec = registry.get(node.block)
        except registry.UnknownBlockError as exc:
            self._placeholder.setText(str(exc))
            self._placeholder.setVisible(True)
            return

        self._placeholder.setVisible(False)
        host = QWidget(self)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)

        heading = QLabel(f"<b>{node_id}</b> &mdash; {spec.name}")
        heading.setWordWrap(True)
        outer.addWidget(heading)
        if spec.doc:
            doc = QLabel(spec.doc.splitlines()[0])
            doc.setWordWrap(True)
            doc.setStyleSheet("color: #8b93a7; font-size: 11px;")
            outer.addWidget(doc)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        connected = {e.dst_port for e in self._graph.incoming(node_id)}

        for port in spec.inputs:
            widget = self._editor_for(port, node.params.get(port.name))
            if port.name in connected:
                widget.setEnabled(False)
                widget.setToolTip("fed by a connection")
            label = port.name if not port.required else f"{port.name} *"
            form.addRow(label, widget)

        if not spec.inputs:
            form.addRow(QLabel("no parameters"))

        outer.addLayout(form)
        outer.addStretch(1)
        self._layout.addWidget(host)
        self._form_host = host

    def _emit(self, port_name: str, value: Any) -> None:
        if self._node_id is not None:
            self.parameter_changed.emit(self._node_id, port_name, value)

    def _editor_for(self, port: Port, current: Any) -> QWidget:
        target = port.type
        value = current if current is not None else _default_value(port)

        if isinstance(target, type) and issubclass(target, Enum):
            combo = QComboBox()
            for member in target:
                combo.addItem(member.name, member.name)
            name = value.name if isinstance(value, Enum) else str(value)
            index = combo.findData(name)
            combo.setCurrentIndex(max(0, index))
            combo.currentTextChanged.connect(lambda text: self._emit(port.name, text))
            return combo

        if target is bool:
            check = QCheckBox()
            check.setChecked(bool(value))
            check.toggled.connect(lambda state: self._emit(port.name, state))
            return check

        if target is int:
            spin = QSpinBox()
            spin.setRange(-(2**31), 2**31 - 1)
            spin.setValue(int(value or 0))
            spin.valueChanged.connect(lambda number: self._emit(port.name, number))
            return spin

        if target is float:
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-1e9, 1e9)
            spin.setValue(float(value or 0.0))
            spin.valueChanged.connect(lambda number: self._emit(port.name, number))
            return spin

        if target is Path:
            return self._path_editor(port, value)

        line = QLineEdit("" if value is None else str(value))
        line.editingFinished.connect(lambda: self._emit(port.name, line.text()))
        return line

    def _path_editor(self, port: Port, value: Any) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        line = QLineEdit("" if value is None else str(value))
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        row.addWidget(line, 1)
        row.addWidget(browse)

        def emit_text() -> None:
            self._emit(port.name, line.text())

        def choose() -> None:
            start = str(self._graph.base_dir) if self._graph and self._graph.base_dir else ""
            chosen, _ = QFileDialog.getOpenFileName(self, f"Choose {port.name}", start)
            if not chosen:
                return
            path = Path(chosen)
            # Store it relative to the diagram when possible, so the file stays
            # portable -- the mistake that put /home/kochas/schnecke.pcd into
            # flodiedi's shipped plugin.
            if self._graph is not None and self._graph.base_dir is not None:
                try:
                    path = path.relative_to(self._graph.base_dir)
                except ValueError:
                    pass
            line.setText(str(path))
            emit_text()

        line.editingFinished.connect(emit_text)
        browse.clicked.connect(choose)
        return host


def _default_value(port: Port) -> Any:
    if not port.required:
        return port.default
    return None
