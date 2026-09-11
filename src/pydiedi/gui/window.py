"""The editor window.

Kept deliberately small, because flodiedi's ``EditorMainWindow`` is a cautionary
tale: 34 KB of source, and four parallel ``QMap``s keyed by the *tab tooltip
string*, which ``getCurrentDiagramName()`` read back as the primary key of the
whole application. Worse, the sentinels it compared those keys against were
passed through ``tr()``, so the editor broke in any non-English locale.

Here the window holds one document at a time -- a ``Graph``, a path and a dirty
flag -- and everything else is derived.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QToolBar,
    QWidget,
)

from .. import __version__
from ..core import io, registry
from ..core.block import BlockSpec
from ..core.edit import EditError, EditSession
from ..core.executor import CycleError, OnError, topological_order
from ..core.graph import Edge, Graph, ValidationError
from .canvas import DiagramScene, DiagramView
from .palette import BlockPalette, ParameterEditor
from .preview import PreviewPanel
from .worker import ExecutionWorker, SweepReport

__all__ = ["EditorWindow"]

FILE_FILTER = "pydiedi diagrams (*.yaml *.yml);;All files (*)"


class EditorWindow(QMainWindow):
    """One window, one diagram."""

    document_changed = Signal()

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        registry.discover()

        # Every change goes through the session, so everything is undoable
        # and 'modified' is undo-aware rather than a flag that latches on.
        self.session = EditSession(Graph(name="untitled"))
        self._path: Path | None = None
        self._worker: ExecutionWorker | None = None

        self.scene = DiagramScene(self)
        self.view = DiagramView(self.scene, self)
        self.setCentralWidget(self.view)

        self.palette_widget = BlockPalette(self)
        self.parameters = ParameterEditor(self)
        self.previews = PreviewPanel(self)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)

        self._build_docks()
        self._build_actions()
        self._build_statusbar()

        self.scene.selection_changed_to.connect(self.parameters.show_node)
        self.scene.node_moved_to.connect(self._on_node_moved)
        self.scene.layout_computed.connect(self._on_layout_computed)
        self.scene.node_move_finished.connect(self.session.end_gesture)
        self.scene.connect_requested.connect(self._on_connect_requested)
        self.scene.connection_refused.connect(self._on_connection_refused)
        self.scene.delete_requested.connect(self._on_delete_requested)
        self.scene.rename_requested.connect(self._on_rename_requested)
        self.scene.block_dropped.connect(self._on_block_dropped)
        self.parameters.parameter_changed.connect(self._on_parameter_changed)
        self.palette_widget.block_activated.connect(self._on_palette_activated)

        self.resize(1280, 820)
        self._fitted_once = False
        if path is not None:
            self.open_path(path)
        else:
            self._apply_graph(self.session.graph)

    def showEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        """Fit the diagram the first time the window is actually on screen.

        Fitting during __init__ does nothing useful: the viewport still has its
        placeholder size, so the transform is computed against the wrong
        rectangle and the diagram ends up clipped.
        """
        super().showEvent(event)  # type: ignore[arg-type]
        if not self._fitted_once:
            self._fitted_once = True
            self.view.enable_auto_fit()

    # -- construction -----------------------------------------------------

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.lower().replace(' ', '_')}")
        dock.setWidget(widget)
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        blocks = self._dock("Blocks", self.palette_widget, Qt.LeftDockWidgetArea)
        parameters = self._dock("Parameters", self.parameters, Qt.RightDockWidgetArea)
        previews = self._dock("Previews", self.previews, Qt.RightDockWidgetArea)
        log = self._dock("Log", self.log, Qt.BottomDockWidgetArea)

        # Blocks and Parameters share one column as tabs. Three separate dock
        # columns cost three times their width, which a tiling window manager
        # will not grant: in a 745 px tile they left the canvas 261 px. Tabbing
        # the two reference panels together buys that width back, and Previews
        # -- the panel worth watching while a diagram runs -- keeps its own
        # column opposite them.
        self.tabifyDockWidget(blocks, parameters)
        blocks.raise_()

        self.palette_widget.setMinimumWidth(130)
        for dock in (blocks, parameters, previews):
            dock.widget().setMaximumWidth(460)
        self.log.setMaximumHeight(240)

        # The canvas is the point of the window, so it gets a floor that the
        # docks must yield to rather than the other way round.
        self.view.setMinimumWidth(280)
        self.view.setMinimumHeight(200)

        self.resizeDocks([blocks, parameters], [200, 200], Qt.Horizontal)
        self.resizeDocks([previews], [240], Qt.Horizontal)
        self.resizeDocks([log], [120], Qt.Vertical)

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("toolbar_main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        file_menu = self.menuBar().addMenu("&File")
        edit_menu = self.menuBar().addMenu("&Edit")
        run_menu = self.menuBar().addMenu("&Run")
        view_menu = self.menuBar().addMenu("&View")

        def add(
            menu: object,
            text: str,
            slot: object,
            shortcut: str | None = None,
            to_toolbar: bool = False,
        ) -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)  # type: ignore[arg-type]
            menu.addAction(action)  # type: ignore[attr-defined]
            if to_toolbar:
                toolbar.addAction(action)
            return action

        add(file_menu, "&New", self.new_diagram, "Ctrl+N")
        add(file_menu, "&Open…", self.open_dialog, "Ctrl+O")
        self.action_save = add(file_menu, "&Save", self.save, "Ctrl+S", True)
        add(file_menu, "Save &As…", self.save_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        add(file_menu, "&Quit", self.close, "Ctrl+Q")

        toolbar.addSeparator()
        self.action_undo = add(edit_menu, "&Undo", self.undo, "Ctrl+Z", True)
        self.action_redo = add(edit_menu, "&Redo", self.redo, "Ctrl+Shift+Z", True)
        edit_menu.addSeparator()
        add(edit_menu, "&Delete", self.delete_selection, "Del")
        add(edit_menu, "Re&name…", self.rename_selection, "F2")
        self.action_undo.setEnabled(False)
        self.action_redo.setEnabled(False)

        toolbar.addSeparator()
        self.action_run = add(run_menu, "&Run", self.run_continuous, "F5", True)
        self.action_step = add(run_menu, "&Step", self.run_once, "F6", True)
        self.action_stop = add(run_menu, "S&top", self.stop, "Shift+F5", True)
        run_menu.addSeparator()
        add(run_menu, "&Check", self.check, "F8")

        toolbar.addSeparator()
        add(view_menu, "&Fit", self.view.enable_auto_fit, "Ctrl+0", True)
        add(view_menu, "Zoom &in", lambda: self.view.zoom_by(1.2), "Ctrl++")
        add(view_menu, "Zoom &out", lambda: self.view.zoom_by(1 / 1.2), "Ctrl+-")
        view_menu.addSeparator()
        add(view_menu, "&Auto layout", self.auto_layout, "Ctrl+L", True)

        self.action_stop.setEnabled(False)

    def _build_statusbar(self) -> None:
        self._status = QLabel("")
        self.statusBar().addPermanentWidget(self._status)
        self.statusBar().showMessage(f"pydiedi {__version__}", 4000)

    # -- document ---------------------------------------------------------

    def _specs_for(self, graph: Graph) -> tuple[dict[str, BlockSpec], list[str]]:
        specs: dict[str, BlockSpec] = {}
        problems: list[str] = []
        for node_id, node in graph.nodes.items():
            try:
                specs[node_id] = registry.get(node.block)
            except registry.UnknownBlockError as exc:
                problems.append(f"{node.where()}: {exc}")
        return specs, problems

    def _apply_graph(self, graph: Graph) -> None:
        """Load a new document: rebuild the scene and drop the undo history."""
        self.session.reset(graph)
        # A different document, so nothing is worth carrying over.
        self._refresh_scene(keep_selection=False)
        self.previews.clear()
        self._update_title()
        self._update_status()

    def _refresh_scene(self, keep_selection: bool = True) -> None:
        """Re-render the current document.

        The selection is carried across, because a rebuild happens after an
        undo -- and losing the node you were editing, along with its parameter
        panel, every time you press Ctrl+Z makes the undo useless for the very
        thing it is most used on.
        """
        graph = self.session.graph
        selected = self.scene.selected_node_ids() if keep_selection else []
        specs, problems = self._specs_for(graph)
        self.scene.set_graph(graph, specs)
        self.parameters.set_graph(graph, keep_selection=keep_selection)
        for node_id in selected:
            if node_id in graph.nodes:
                self.scene.select_node(node_id)
                break
        for problem in problems:
            self._log(f"warning: {problem}")

    def new_diagram(self) -> None:
        if not self._confirm_discard():
            return
        self._path = None
        self._apply_graph(Graph(name="untitled"))
        self._log("new diagram")

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        chosen, _ = QFileDialog.getOpenFileName(self, "Open diagram", "", FILE_FILTER)
        if chosen:
            self.open_path(Path(chosen))

    def open_path(self, path: Path) -> None:
        try:
            graph = io.load(path)
        except io.DiagramSyntaxError as exc:
            QMessageBox.critical(self, "Cannot open diagram", str(exc))
            self._log(f"error: {exc}")
            return
        self._path = path
        self._apply_graph(graph)
        self._log(f"opened {path}")
        self.view.enable_auto_fit()

    def save(self) -> bool:
        if self._path is None:
            return self.save_as()
        try:
            io.dump(self.session.graph, self._path)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save", str(exc))
            return False
        self.session.mark_saved()
        self._update_title()
        self._log(f"saved {self._path}")
        return True

    def save_as(self) -> bool:
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save diagram", str(self._path or "diagram.yaml"), FILE_FILTER
        )
        if not chosen:
            return False
        path = Path(chosen)
        if not path.suffix:
            path = path.with_suffix(".yaml")
        self._path = path
        # base_dir decides how relative paths resolve, so it must follow the file.
        self.session.graph.base_dir = path.parent.resolve()
        return self.save()

    def _confirm_discard(self) -> bool:
        if not self.session.modified:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "This diagram has unsaved changes.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Save:
            return self.save()
        return answer == QMessageBox.Discard

    def _mark_dirty(self) -> None:
        self._update_title()

    def _update_title(self) -> None:
        name = str(self._path) if self._path else "untitled"
        marker = "*" if self.session.modified else ""
        self.setWindowTitle(f"{marker}{name} — pydiedi")
        self._update_edit_actions()

    def _update_status(self) -> None:
        nodes, edges = len(self.session.graph.nodes), len(self.session.graph.edges)
        try:
            topological_order(self.session.graph)
            order = "acyclic"
        except CycleError:
            order = "cyclic!"
        self._status.setText(f"{nodes} nodes, {edges} edges, {order}")

    # -- editing ----------------------------------------------------------
    #
    # Nothing here touches the graph directly. Each handler turns a gesture
    # into a command on the session, which is what makes undo a property of
    # the design rather than a feature bolted on.

    def _on_parameter_changed(self, node_id: str, name: str, value: object) -> None:
        try:
            self.session.set_param(node_id, name, value)
        except EditError as exc:
            self._log(f"refused: {exc}")
            return
        # Parameters are live: a running diagram picks the new value up on its
        # next sweep, which is what makes turning a threshold while watching
        # the preview worth doing at all.
        if self._worker is not None:
            self._worker.set_param(node_id, name, value)
        self._mark_dirty()

    def _note_structural_change(self) -> None:
        """Tell the user that this kind of edit needs a restart.

        Parameters reach a running diagram; structure does not, because adding
        a node or an edge mid-sweep would change the execution order underneath
        the loop walking it. Saying so is better than appearing to ignore the
        edit.
        """
        if self._worker is not None:
            self.statusBar().showMessage(
                "Structure changed — restart the run for it to take effect", 5000
            )

    def _on_node_moved(self, node_id: str, x: float, y: float) -> None:
        self.session.move_node(node_id, (x, y))
        self._mark_dirty()

    def _on_palette_activated(self, block_name: str) -> None:
        """Double-clicking the palette drops a block in the middle of the view."""
        centre = self.view.mapToScene(self.view.viewport().rect().center())
        self._add_block(block_name, centre.x() - 75.0, centre.y() - 30.0)

    def _on_block_dropped(self, block_name: str, x: float, y: float) -> None:
        self._add_block(block_name, x - 75.0, y - 20.0)

    def _add_block(self, block_name: str, x: float, y: float) -> None:
        try:
            node_id = self.session.add_node(block_name, position=(x, y))
        except (EditError, registry.UnknownBlockError) as exc:
            self._log(f"refused: {exc}")
            return
        self.scene.add_node_item(node_id)
        self.scene.select_node(node_id)
        self._after_edit(f"added {node_id}")

    def _on_connect_requested(
        self, src: str, src_port: str, dst: str, dst_port: str
    ) -> None:
        try:
            self.session.connect(src, src_port, dst, dst_port)
        except EditError as exc:
            self._on_connection_refused(str(exc))
            return
        self.scene.add_edge_item(self.session.graph.edges[-1])
        self._after_edit(f"connected {src}.{src_port} to {dst}.{dst_port}")

    def _on_connection_refused(self, reason: str) -> None:
        self._log(f"refused: {reason}")
        self.statusBar().showMessage(reason, 4000)

    def _on_delete_requested(self, node_ids: list[str], edges: list[Edge]) -> None:
        if not node_ids and not edges:
            return
        removed: list[str] = []
        # Edges first: deleting a node takes its edges with it, and doing it
        # the other way round would try to remove them twice.
        for edge in edges:
            if edge.src in node_ids or edge.dst in node_ids:
                continue
            try:
                self.session.disconnect(edge.src, edge.src_port, edge.dst, edge.dst_port)
            except EditError as exc:
                self._log(f"refused: {exc}")
                continue
            self.scene.remove_edge_item(edge)
            removed.append(str(edge))
        for node_id in node_ids:
            try:
                self.session.remove_node(node_id)
            except EditError as exc:
                self._log(f"refused: {exc}")
                continue
            self.scene.remove_node_item(node_id)
            removed.append(node_id)
        if removed:
            self.parameters.show_node(None)
            self._after_edit(f"removed {', '.join(removed)}")

    def _on_rename_requested(self, node_id: str) -> None:
        new_id, accepted = QInputDialog.getText(
            self, "Rename node", "New name:", text=node_id
        )
        if not accepted or new_id == node_id:
            return
        try:
            self.session.rename_node(node_id, new_id)
        except EditError as exc:
            QMessageBox.warning(self, "Cannot rename", str(exc))
            return
        self._refresh_scene()
        self.scene.select_node(new_id)
        self._after_edit(f"renamed {node_id} to {new_id}")

    def _after_edit(self, message: str) -> None:
        self._log(message)
        self._mark_dirty()
        self._update_status()
        self._note_structural_change()

    def undo(self) -> None:
        description = self.session.undo()
        if description is None:
            return
        self._refresh_scene()
        if self._worker is not None:
            self._worker.sync_params(self.session.graph)
        self._after_edit(f"undo: {description}")

    def redo(self) -> None:
        description = self.session.redo()
        if description is None:
            return
        self._refresh_scene()
        if self._worker is not None:
            self._worker.sync_params(self.session.graph)
        self._after_edit(f"redo: {description}")

    def _update_edit_actions(self) -> None:
        undo, redo = getattr(self, "action_undo", None), getattr(self, "action_redo", None)
        if undo is None or redo is None:
            return
        undo.setEnabled(self.session.can_undo)
        redo.setEnabled(self.session.can_redo)
        undo.setText(
            f"&Undo {self.session.undo_description}".rstrip()
            if self.session.can_undo
            else "&Undo"
        )
        redo.setText(
            f"&Redo {self.session.redo_description}".rstrip()
            if self.session.can_redo
            else "&Redo"
        )

    def delete_selection(self) -> None:
        self.scene._request_delete()

    def rename_selection(self) -> None:
        selected = self.scene.selected_node_ids()
        if len(selected) == 1:
            self._on_rename_requested(selected[0])

    def auto_layout(self) -> None:
        computed = self.scene.auto_layout()
        if computed:
            self.session.set_layout(computed)
            self._after_edit("auto layout")
        self.view.enable_auto_fit()

    def _on_layout_computed(self, layout: dict) -> None:
        """The scene placed nodes that arrived without positions."""
        self.session.set_layout({**self.session.graph.layout, **layout})
        self._mark_dirty()

    # -- running ----------------------------------------------------------

    def check(self) -> None:
        self.scene.clear_errors()
        try:
            self.session.graph.validate()
        except ValidationError as exc:
            for problem in exc.problems:
                self._log(f"invalid: {problem}")
            QMessageBox.warning(self, "Diagram is not runnable", str(exc))
            return
        try:
            order = topological_order(self.session.graph)
        except CycleError as exc:
            self._log(f"invalid: {exc}")
            QMessageBox.warning(self, "Diagram is not runnable", str(exc))
            return
        self._log(f"ok: {' -> '.join(order)}")
        self.statusBar().showMessage("Diagram is valid", 3000)

    def run_once(self) -> None:
        self._start(iterations=1, interval=0.0)

    def run_continuous(self) -> None:
        self._start(iterations=0, interval=0.02)

    def _start(self, iterations: int, interval: float) -> None:
        if self._worker is not None:
            return
        self.scene.clear_errors()
        try:
            self.session.graph.validate()
        except ValidationError as exc:
            for problem in exc.problems:
                self._log(f"invalid: {problem}")
            QMessageBox.warning(self, "Diagram is not runnable", str(exc))
            return

        worker = ExecutionWorker(
            self.session.graph,
            iterations=iterations,
            interval=interval,
            on_error=OnError.skip,
        )
        # A queued connection by default, because the worker lives on another
        # thread: this is the boundary flodiedi did not have.
        worker.swept.connect(self._on_swept)
        worker.failed.connect(self._on_failed)
        worker.stopped.connect(self._on_stopped)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker

        self.action_run.setEnabled(False)
        self.action_step.setEnabled(False)
        self.action_stop.setEnabled(True)
        self._log("running" if iterations == 0 else "single step")
        worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.statusBar().showMessage("Stopping…", 1500)

    def _on_swept(self, report: SweepReport) -> None:
        self.previews.show_previews(report.previews)
        self.scene.set_errors(report.errors)
        for node_id, message in report.errors.items():
            self._log(f"error in {node_id}: {message}")
        self.statusBar().showMessage(
            f"sweep {report.iteration} — {report.duration_ms:.1f} ms"
            + (f", skipped {len(report.skipped)}" if report.skipped else ""),
            2000,
        )
        # Releases the worker to send the next report. Without this the view
        # freezes after one frame.
        if self._worker is not None:
            self._worker.preview_consumed()

    def _on_failed(self, message: str) -> None:
        self._log(f"failed: {message}")
        self.statusBar().showMessage("Run failed", 4000)

    def _on_stopped(self, message: str) -> None:
        self._log(f"stopped: {message}")
        self.statusBar().showMessage("Run finished", 3000)

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.action_run.setEnabled(True)
        self.action_step.setEnabled(True)
        self.action_stop.setEnabled(False)

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)

    # -- shutdown ---------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        """Stop the worker before the window goes away.

        flodiedi's ``deleteTab()`` did ``setRunning(false)`` and then
        ``delete di;`` with the ``wait()`` commented out -- deleting a running
        QThread, which is undefined behaviour.
        """
        if self._worker is not None:
            self._worker.request_stop()
            if not self._worker.wait(3000):
                self._worker.terminate()
                self._worker.wait()
            self._worker = None
        if not self._confirm_discard():
            event.ignore()  # type: ignore[attr-defined]
            return
        event.accept()  # type: ignore[attr-defined]
