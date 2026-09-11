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
from ..core.executor import CycleError, OnError, topological_order
from ..core.graph import Graph, ValidationError
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

        self._graph: Graph = Graph(name="untitled")
        self._path: Path | None = None
        self._dirty = False
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
        self.scene.layout_changed.connect(self._mark_dirty)
        self.parameters.parameter_changed.connect(self._on_parameter_changed)
        self.palette_widget.block_activated.connect(self._on_palette_activated)

        self.resize(1280, 820)
        self._fitted_once = False
        if path is not None:
            self.open_path(path)
        else:
            self._apply_graph(self._graph)

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
        self._graph = graph
        specs, problems = self._specs_for(graph)
        self.scene.set_graph(graph, specs)
        self.parameters.set_graph(graph)
        self.previews.clear()
        for problem in problems:
            self._log(f"warning: {problem}")
        self._update_title()
        self._update_status()

    def new_diagram(self) -> None:
        if not self._confirm_discard():
            return
        self._path = None
        self._dirty = False
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
        self._dirty = False
        self._apply_graph(graph)
        self._log(f"opened {path}")
        self.view.enable_auto_fit()

    def save(self) -> bool:
        if self._path is None:
            return self.save_as()
        try:
            io.dump(self._graph, self._path)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot save", str(exc))
            return False
        self._dirty = False
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
        self._graph.base_dir = path.parent.resolve()
        return self.save()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
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
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = str(self._path) if self._path else "untitled"
        self.setWindowTitle(f"{'*' if self._dirty else ''}{name} — pydiedi")

    def _update_status(self) -> None:
        nodes, edges = len(self._graph.nodes), len(self._graph.edges)
        try:
            topological_order(self._graph)
            order = "acyclic"
        except CycleError:
            order = "cyclic!"
        self._status.setText(f"{nodes} nodes, {edges} edges, {order}")

    def _on_parameter_changed(self, node_id: str, name: str, value: object) -> None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return
        node.params[name] = value
        self._mark_dirty()

    def _on_palette_activated(self, block_name: str) -> None:
        QMessageBox.information(
            self,
            "Not yet",
            f"Adding blocks from the palette is the next step.\n\n"
            f"{registry.get(block_name).signature()}",
        )

    def auto_layout(self) -> None:
        self.scene.auto_layout()
        self.scene.tighten_scene_rect()
        self.view.enable_auto_fit()

    # -- running ----------------------------------------------------------

    def check(self) -> None:
        self.scene.clear_errors()
        try:
            self._graph.validate()
        except ValidationError as exc:
            for problem in exc.problems:
                self._log(f"invalid: {problem}")
            QMessageBox.warning(self, "Diagram is not runnable", str(exc))
            return
        try:
            order = topological_order(self._graph)
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
            self._graph.validate()
        except ValidationError as exc:
            for problem in exc.problems:
                self._log(f"invalid: {problem}")
            QMessageBox.warning(self, "Diagram is not runnable", str(exc))
            return

        worker = ExecutionWorker(
            self._graph,
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
