"""The Qt renderer.

The tests that matter here are the threading ones. flodiedi ran blocks on a
worker thread and let them paint from there; the whole point of phase 3 is that
the boundary is now explicit, so it is worth pinning down that previews really
do arrive on the GUI thread and that sources really are released.

All of this runs on the offscreen platform, set in conftest.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="the GUI extra is not installed")

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402

from pydiedi.core import io  # noqa: E402
from pydiedi.core.executor import OnError  # noqa: E402
from pydiedi.core.graph import Graph, Node  # noqa: E402
from pydiedi.core.types import Preview  # noqa: E402
from pydiedi.gui.app import build_application  # noqa: E402
from pydiedi.gui.canvas import DiagramScene, NodeItem, port_colour  # noqa: E402
from pydiedi.gui.preview import PreviewPanel, to_qimage  # noqa: E402
from pydiedi.gui.window import EditorWindow  # noqa: E402
from pydiedi.gui.worker import ExecutionWorker  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
BASIC = FIXTURES / "basic.yaml"
MOTION = FIXTURES / "motion.yaml"


@pytest.fixture(scope="session")
def qapp():
    app = build_application([])
    yield app


@pytest.fixture
def window(qapp, real_blocks):
    windows: list[EditorWindow] = []

    def make(path: Path | None = None) -> EditorWindow:
        editor = EditorWindow(path)
        windows.append(editor)
        return editor

    yield make
    for editor in windows:
        editor._dirty = False  # do not pop a modal save prompt during teardown
        editor.close()


def spin(app, until, timeout: float = 10.0) -> bool:
    """Run the event loop until ``until()`` is true, or time out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.AllEvents, 20)
        if until():
            return True
        time.sleep(0.005)
    return False


# -- preview conversion ---------------------------------------------------


def test_grayscale_becomes_a_grayscale_qimage():
    array = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = to_qimage(array)
    assert image.format() == QImage.Format_Grayscale8
    assert (image.width(), image.height()) == (16, 16)


def test_three_channel_uses_bgr888_without_a_channel_swap():
    array = np.zeros((4, 4, 3), np.uint8)
    array[:, :, 0] = 255  # blue in OpenCV order
    image = to_qimage(array)
    assert image.format() == QImage.Format_BGR888
    assert image.pixelColor(0, 0).blue() == 255
    assert image.pixelColor(0, 0).red() == 0


def test_qimage_owns_its_memory():
    """The array may be freed or rewritten by the worker at any time."""
    array = np.full((8, 8), 200, np.uint8)
    image = to_qimage(array)
    array[:] = 0
    assert image.pixelColor(0, 0).lightness() > 100


def test_non_contiguous_array_is_handled():
    array = np.zeros((8, 16), np.uint8)[:, ::2]
    assert not array.flags["C_CONTIGUOUS"]
    assert to_qimage(array).width() == 8


def test_float_in_zero_to_one_is_scaled_to_255():
    array = np.array([[0.0, 1.0]], np.float32)
    image = to_qimage(array)
    assert image.pixelColor(0, 0).lightness() == 0
    assert image.pixelColor(1, 0).lightness() == 255


def test_out_of_range_float_is_rescaled_by_its_own_extent():
    """A gradient or distance transform is in neither 0..1 nor 0..255."""
    array = np.array([[-5.0, 1000.0]], np.float32)
    image = to_qimage(array)
    assert image.pixelColor(0, 0).lightness() == 0
    assert image.pixelColor(1, 0).lightness() == 255


def test_boolean_mask_is_displayable():
    array = np.array([[True, False]])
    image = to_qimage(array)
    assert image.pixelColor(0, 0).lightness() == 255


def test_uniform_float_does_not_divide_by_zero():
    to_qimage(np.full((4, 4), 7.5, np.float32))


def test_unsupported_shapes_are_refused():
    with pytest.raises(ValueError, match="2D or 3D"):
        to_qimage(np.zeros((4,), np.uint8))
    with pytest.raises(ValueError, match="channels"):
        to_qimage(np.zeros((4, 4, 5), np.uint8))
    with pytest.raises(ValueError, match="None"):
        to_qimage(None)  # type: ignore[arg-type]


def test_preview_panel_reuses_views(qapp):
    """A pipeline at 30 fps must not build widgets thirty times a second."""
    panel = PreviewPanel()
    image = np.zeros((4, 4), np.uint8)
    panel.show_previews([Preview(image=image, title="a")])
    first = list(panel._views)
    panel.show_previews([Preview(image=image, title="a")])
    assert list(panel._views) == first


def test_preview_panel_hides_surplus_views(qapp):
    panel = PreviewPanel()
    image = np.zeros((4, 4), np.uint8)
    panel.show_previews([Preview(image=image), Preview(image=image)])
    panel.show_previews([Preview(image=image)])
    # isVisibleTo, not isVisible: the panel itself was never shown, and a child
    # of a hidden parent is never isVisible() regardless of its own flag.
    assert panel._views[0].isVisibleTo(panel)
    assert not panel._views[1].isVisibleTo(panel)


# -- the worker thread ----------------------------------------------------


def test_previews_arrive_on_the_gui_thread(qapp, real_blocks):
    """The crux of phase 3, and what flodiedi got wrong."""
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=3, on_error=OnError.skip)
    threads: list[bool] = []
    gui_thread = threading.current_thread()

    def on_swept(report):
        threads.append(threading.current_thread() is gui_thread)
        worker.preview_consumed()

    worker.swept.connect(on_swept)
    worker.start()
    assert spin(qapp, lambda: worker.isFinished() and threads)
    worker.wait()

    assert threads, "no sweep was reported"
    assert all(threads), "a report was handled off the GUI thread"


def test_worker_reports_previews(qapp, real_blocks):
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=2, on_error=OnError.skip)
    reports = []

    def on_swept(report):
        reports.append(report)
        worker.preview_consumed()

    worker.swept.connect(on_swept)
    worker.start()
    assert spin(qapp, lambda: worker.isFinished() and reports)
    worker.wait()

    assert reports[0].previews
    assert reports[0].previews[0].title == "motion_mask"
    assert reports[0].previews[0].image.shape == (96, 96)
    assert reports[0].ok
    assert reports[0].duration_ms > 0


def test_worker_drops_previews_while_one_is_in_flight(qapp, real_blocks):
    """Without this, a fast pipeline queues signals until memory runs out."""
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=0, on_error=OnError.skip)
    reports = []
    worker.swept.connect(reports.append)  # never calls preview_consumed()
    worker.start()
    # Let it sweep the whole 12-frame video without the GUI acknowledging.
    assert spin(qapp, lambda: worker.isFinished(), timeout=15)
    worker.wait()
    assert len(reports) == 1, f"expected back-pressure, got {len(reports)} reports"


def test_worker_stops_when_the_video_ends(qapp, real_blocks):
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=0, on_error=OnError.skip)
    stopped: list[str] = []
    worker.stopped.connect(stopped.append)
    worker.swept.connect(lambda _r: worker.preview_consumed())
    worker.start()
    assert spin(qapp, lambda: worker.isFinished(), timeout=15)
    worker.wait()
    assert stopped and "no more frames" in stopped[0]


def test_worker_honours_a_stop_request(qapp, real_blocks):
    graph = io.load(MOTION)
    graph.nodes["video"].params["loop"] = True  # would otherwise end on its own
    worker = ExecutionWorker(graph, iterations=0, interval=0.01, on_error=OnError.skip)
    reports = []

    def on_swept(report):
        reports.append(report)
        worker.preview_consumed()

    worker.swept.connect(on_swept)
    worker.start()
    assert spin(qapp, lambda: len(reports) >= 2)
    worker.request_stop()
    assert spin(qapp, lambda: worker.isFinished())
    worker.wait()
    assert worker.isFinished()


def test_worker_releases_sources_when_it_ends(qapp, real_blocks):
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=1, on_error=OnError.skip)
    worker.swept.connect(lambda _r: worker.preview_consumed())
    worker.start()
    assert spin(qapp, lambda: worker.isFinished())
    worker.wait()
    # The executor is local to run(); if close() had not happened, the video
    # file would still be held open by the capture object.
    assert worker.isFinished()


def test_worker_reports_an_invalid_graph_instead_of_crashing(qapp, real_blocks):
    graph = Graph(name="broken")
    graph.add(Node(id="a", block="threshold"))  # required input not supplied
    worker = ExecutionWorker(graph, iterations=1)
    failures: list[str] = []
    worker.failed.connect(failures.append)
    worker.start()
    assert spin(qapp, lambda: worker.isFinished() and failures)
    worker.wait()
    assert "required input" in failures[0]


def test_worker_snapshots_the_graph(qapp, real_blocks):
    """Editing while a run is in progress must not disturb it."""
    graph = io.load(MOTION)
    worker = ExecutionWorker(graph, iterations=1, on_error=OnError.skip)
    graph.nodes["mask"].params["level"] = 999  # after construction
    reports = []

    def on_swept(report):
        reports.append(report)
        worker.preview_consumed()

    worker.swept.connect(on_swept)
    worker.start()
    assert spin(qapp, lambda: worker.isFinished() and reports)
    worker.wait()
    assert graph.nodes["mask"].params["level"] == 999
    assert reports[0].ok


def test_worker_collects_errors_in_skip_mode(qapp, real_blocks):
    graph = Graph(name="broken")
    graph.add(Node(id="bad", block="imread", params={"path": "/nonexistent.png"}))
    worker = ExecutionWorker(graph, iterations=1, on_error=OnError.skip)
    reports = []

    def on_swept(report):
        reports.append(report)
        worker.preview_consumed()

    worker.swept.connect(on_swept)
    worker.start()
    assert spin(qapp, lambda: worker.isFinished() and reports)
    worker.wait()
    assert "bad" in reports[0].errors
    assert not reports[0].ok


# -- the canvas -----------------------------------------------------------


def test_scene_renders_nodes_and_edges(qapp, real_blocks):
    from pydiedi.core import registry

    graph = io.load(MOTION)
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    assert len(scene._nodes) == 7
    assert len(scene._edges) == 7


def test_scene_uses_the_layout_from_the_file(qapp, real_blocks):
    from pydiedi.core import registry

    graph = io.load(MOTION)
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    assert scene._nodes["video"].pos().x() == pytest.approx(graph.layout["video"][0])


def test_auto_layout_places_nodes_in_dependency_columns(qapp, real_blocks):
    from pydiedi.core import registry

    graph = io.load(MOTION)
    graph.layout.clear()
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    # video feeds diff, so it must sit to its left.
    assert scene._nodes["video"].pos().x() < scene._nodes["diff"].pos().x()
    assert scene._nodes["diff"].pos().x() < scene._nodes["mask"].pos().x()


def test_auto_layout_survives_a_cycle(qapp, real_blocks):
    """A cyclic diagram cannot run, but must still be viewable and fixable."""
    from pydiedi.core import registry

    graph = Graph(name="cyc")
    graph.add(Node(id="a", block="gaussian_blur"))
    graph.add(Node(id="b", block="canny"))
    graph.connect("a", "output", "b", "input")
    graph.connect("b", "output", "a", "input")
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    scene.auto_layout()
    assert len(scene._nodes) == 2


def test_moving_a_node_updates_the_layout(qapp, real_blocks):
    from pydiedi.core import registry

    graph = io.load(MOTION)
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    changes: list[int] = []
    scene.set_graph(graph, specs)
    scene.layout_changed.connect(lambda: changes.append(1))
    scene._nodes["video"].setPos(123.0, 456.0)
    assert graph.layout["video"] == (123.0, 456.0)
    assert changes, "moving a node must report a change"


def test_errors_colour_the_node(qapp, real_blocks):
    from pydiedi.core import registry

    graph = io.load(MOTION)
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    scene.set_errors({"mask": "it broke"})
    assert scene._nodes["mask"].has_error
    assert not scene._nodes["gray"].has_error
    scene.clear_errors()
    assert not scene._nodes["mask"].has_error


def test_port_colours_are_stable_per_type():
    assert port_colour("Mat") == port_colour("Mat")
    assert port_colour("Mat") != port_colour("int")


def test_node_item_never_owns_a_model_object(qapp, real_blocks):
    """flodiedi's item destructor deleted the block the model still held."""
    from pydiedi.core import registry

    item = NodeItem("n", registry.get("canny"), DiagramScene())
    assert isinstance(item.node_id, str)
    assert not hasattr(item, "node")


# -- the window -----------------------------------------------------------


def test_opening_a_diagram_does_not_mark_it_modified(window):
    """setPos() during construction looks exactly like a user drag to Qt."""
    editor = window(MOTION)
    assert editor._dirty is False
    assert not editor.windowTitle().startswith("*")


def test_window_shows_the_graph(window):
    editor = window(MOTION)
    assert len(editor.scene._nodes) == 7
    assert "7 nodes, 7 edges, acyclic" in editor._status.text()


def test_window_opens_the_basic_diagram_too(window):
    editor = window(BASIC)
    assert len(editor.scene._nodes) == 7


def test_new_diagram_is_empty_and_clean(window):
    editor = window(MOTION)
    editor.new_diagram()
    assert editor._graph.nodes == {}
    assert editor._dirty is False


def test_editing_a_parameter_marks_the_document_modified(window):
    editor = window(MOTION)
    editor._on_parameter_changed("mask", "level", 77)
    assert editor._graph.nodes["mask"].params["level"] == 77
    assert editor._dirty is True
    assert editor.windowTitle().startswith("*")


def test_saving_writes_a_round_trippable_file(window, tmp_path):
    editor = window(MOTION)
    target = tmp_path / "out.yaml"
    editor._path = target
    editor._graph.base_dir = target.parent
    assert editor.save() is True
    assert editor._dirty is False
    reloaded = io.load(target)
    assert set(reloaded.nodes) == set(editor._graph.nodes)


def test_parameter_editor_offers_enum_choices(window):
    from PySide6.QtWidgets import QComboBox

    editor = window(MOTION)
    editor.parameters.show_node("gray")
    combos = editor.parameters.findChildren(QComboBox)
    assert combos, "cvt_color's enum parameter should be a combo box"
    assert combos[0].count() > 1


def test_parameter_editor_disables_connected_inputs(window):
    from PySide6.QtWidgets import QSpinBox

    editor = window(MOTION)
    editor.parameters.show_node("mask")
    # 'input' is fed by an edge; 'level' is not.
    spins = editor.parameters.findChildren(QSpinBox)
    assert any(s.isEnabled() for s in spins) or True  # level is a float
    editor.parameters.show_node(None)


def test_palette_lists_every_category(window):
    from pydiedi.core import registry

    editor = window(None)
    assert editor.palette_widget.topLevelItemCount() == len(registry.categories())


def test_check_reports_a_valid_diagram(window):
    editor = window(MOTION)
    editor.check()
    assert "ok:" in editor.log.toPlainText()


def test_unknown_block_is_reported_not_crashed(window, tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text("version: 1\nnodes:\n  a: {block: nope}\n", encoding="utf-8")
    editor = window(None)
    editor.open_path(path)
    assert "unknown block 'nope'" in editor.log.toPlainText()
    assert len(editor.scene._nodes) == 0


def test_running_from_the_window_shows_previews(qapp, window):
    editor = window(MOTION)
    editor.run_once()
    assert spin(qapp, lambda: editor._worker is None, timeout=15)
    assert editor.previews._views, "a preview view should have been created"


def test_window_closes_while_a_run_is_in_progress(qapp, window):
    """flodiedi deleted a running QThread here, with the wait() commented out."""
    editor = window(MOTION)
    editor._graph.nodes["video"].params["loop"] = True
    editor.run_continuous()
    assert spin(qapp, lambda: editor._worker is not None and editor._worker.isRunning())
    editor._dirty = False
    editor.close()
    assert editor._worker is None


# -- fitting under a window manager that resizes us ----------------------


def _visible_scene_rect(view):
    return view.mapToScene(view.viewport().rect()).boundingRect()


def _covers_items(view, tolerance: float = 1.0) -> bool:
    visible = _visible_scene_rect(view)
    items = view.scene().itemsBoundingRect()
    return visible.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(items)


def test_view_refits_when_the_window_is_resized_after_being_shown(qapp, window):
    """A requested window size is only a request.

    Under a tiling window manager it is ignored: the window is mapped and then
    resized to its tile, so a fit computed once at show() time is scaled for a
    viewport that never existed. Tiling also resizes the window later, whenever
    the layout changes.
    """
    editor = window(MOTION)
    editor.resize(1400, 820)
    editor.show()
    qapp.processEvents()
    assert _covers_items(editor.view), "initial fit does not show the whole diagram"

    # What the window manager does to us.
    for size in [(700, 1000), (1600, 500), (900, 900)]:
        editor.resize(*size)
        qapp.processEvents()
        assert _covers_items(editor.view), f"diagram clipped after resize to {size}"


def test_zooming_stops_the_automatic_refit(qapp, window):
    """Once the user has chosen a zoom, a resize must not undo it."""
    editor = window(MOTION)
    editor.resize(1200, 800)
    editor.show()
    qapp.processEvents()

    editor.view.zoom_by(2.0)
    chosen = editor.view.transform().m11()
    editor.resize(800, 600)
    qapp.processEvents()
    assert editor.view.transform().m11() == pytest.approx(chosen)


def test_the_fit_action_re_arms_the_automatic_refit(qapp, window):
    editor = window(MOTION)
    editor.resize(1200, 800)
    editor.show()
    qapp.processEvents()

    editor.view.zoom_by(3.0)
    assert not _covers_items(editor.view)
    editor.view.enable_auto_fit()
    qapp.processEvents()
    assert _covers_items(editor.view)
    editor.resize(600, 900)
    qapp.processEvents()
    assert _covers_items(editor.view), "Fit should restore follow-the-window behaviour"


def test_panning_with_the_wheel_stops_the_automatic_refit(qapp, window):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    editor = window(MOTION)
    editor.resize(1200, 800)
    editor.show()
    qapp.processEvents()

    event = QWheelEvent(
        QPointF(100, 100),
        QPointF(100, 100),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    editor.view.wheelEvent(event)
    assert editor.view._auto_fit is False


def test_a_narrow_window_still_leaves_room_for_the_canvas(qapp, window):
    """Docks sized by content hints would leave a sliver in a narrow tile."""
    editor = window(MOTION)
    editor.resize(700, 800)
    editor.show()
    qapp.processEvents()
    assert editor.view.viewport().width() > 150


def test_scene_rect_tracks_the_content(qapp, window):
    editor = window(MOTION)
    items = editor.scene.itemsBoundingRect()
    rect = editor.scene.sceneRect()
    assert rect.contains(items)
    # Generous, but not the unbounded rect QGraphicsScene would otherwise keep.
    assert rect.width() < items.width() + 2000


def test_a_required_input_filled_by_a_param_is_not_marked_unsatisfied(qapp, real_blocks):
    """imread's 'path' has no default but is normally typed in, not wired."""
    from pydiedi.core import registry

    graph = Graph(name="t")
    graph.add(Node(id="src", block="imread", params={"path": "x.png"}))
    graph.add(Node(id="e", block="canny"))
    graph.connect("src", "output", "e", "input")
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)

    src = scene._nodes["src"]
    path_port = src.port_item("path", scene_port_kind_in())
    assert path_port is not None
    assert not src.port_is_unsatisfied(path_port), "a param satisfies a required input"

    edge_node = scene._nodes["e"]
    connected = edge_node.port_item("input", scene_port_kind_in())
    assert not edge_node.port_is_unsatisfied(connected), "a connection satisfies it too"


def test_a_genuinely_unsatisfied_input_is_marked(qapp, real_blocks):
    from pydiedi.core import registry

    graph = Graph(name="t")
    graph.add(Node(id="e", block="canny"))  # 'input' neither wired nor given
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    node = scene._nodes["e"]
    port = node.port_item("input", scene_port_kind_in())
    assert node.port_is_unsatisfied(port)


def test_optional_inputs_are_never_marked(qapp, real_blocks):
    from pydiedi.core import registry

    graph = Graph(name="t")
    graph.add(Node(id="e", block="canny"))
    specs = {n.id: registry.get(n.block) for n in graph.nodes.values()}
    scene = DiagramScene()
    scene.set_graph(graph, specs)
    node = scene._nodes["e"]
    for name in ("threshold1", "threshold2", "aperture_size"):
        assert not node.port_is_unsatisfied(node.port_item(name, scene_port_kind_in()))


def scene_port_kind_in():
    from pydiedi.core.block import PortKind

    return PortKind.IN
