"""Running a diagram off the GUI thread.

This module is the reason phase 3 is possible at all, and it is where flodiedi
failed. There, ``FlowDiagram`` *was* a ``QThread``, and blocks painted directly
from ``run()``: ``displayplugin`` and ``showimageplugin`` called
``setPixmap()`` on a ``QGraphicsPixmapItem`` that lived in the GUI thread's
scene. Qt4 tolerated it; Qt5 and Qt6 abort on cross-thread painting.

Here the split is strict:

* the worker owns the :class:`~pydiedi.core.executor.Executor` and creates it
  *inside* ``run()``, so a camera or video file is opened on the thread that
  will read from it;
* blocks return :class:`~pydiedi.core.types.Preview` values -- data, not
  widgets -- which cross to the GUI thread as a queued signal;
* nothing in ``core`` or ``blocks`` knows this module exists.

Two hazards are handled explicitly, both of which a naive implementation hits
as soon as a live camera is attached:

**Unbounded queueing.** A worker sweeping faster than the GUI can repaint would
pile up queued signals until memory ran out. The worker therefore drops
previews while one is still in flight -- for a live view, showing the newest
frame and discarding the rest is the correct behaviour.

**Shared arrays.** A ``Preview`` carries a reference to a numpy array, not a
copy, so the GUI reads the worker's buffer. This is safe only because blocks
treat their inputs as read-only and return freshly allocated arrays, which is
the contract for a pydiedi block. A block calling an OpenCV function with
``dst=`` pointing at its input would break it.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from ..core import registry
from ..core.executor import Executor, NodeExecutionError, OnError, StopExecution
from ..core.graph import Graph, ValidationError
from ..core.inspect import PortValue, inspect_value
from ..core.types import CoercionError, Preview

__all__ = ["ExecutionWorker", "SweepReport"]


@dataclass(slots=True)
class SweepReport:
    """What the GUI needs to know about one sweep.

    Deliberately not the :class:`~pydiedi.core.executor.RunResult` itself: that
    holds every intermediate value of the whole graph, and keeping those alive
    across a thread boundary would pin every frame of a running pipeline.
    Values the user asked to watch are summarised into
    :class:`~pydiedi.core.inspect.PortValue` first, so what travels is a line of
    text and at most a thumbnail.
    """

    iteration: int
    previews_by_node: dict[str, list[Preview]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    watched: list[PortValue] = field(default_factory=list)
    """Values for the ports the GUI asked to watch."""

    @property
    def previews(self) -> list[Preview]:
        return [p for previews in self.previews_by_node.values() for p in previews]

    @property
    def ok(self) -> bool:
        return not self.errors


class ExecutionWorker(QThread):
    """Sweeps a diagram until stopped, reporting each sweep to the GUI."""

    swept = Signal(SweepReport)
    """One sweep completed. Emitted at most once per GUI repaint."""
    stopped = Signal(str)
    """The run ended by itself, e.g. a video reached its last frame."""
    failed = Signal(str)
    """The run could not start or aborted. Carries a message for the user."""
    started_running = Signal()
    """The executor is built and the first sweep is about to happen."""

    def __init__(
        self,
        graph: Graph,
        *,
        iterations: int = 0,
        interval: float = 0.0,
        on_error: OnError | str = OnError.skip,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        # A snapshot: the user may keep editing while this runs.
        self._graph = copy.deepcopy(graph)
        self._graph.base_dir = graph.base_dir
        self._iterations = iterations
        self._interval = interval
        self._on_error = OnError(on_error)
        self._mutex = QMutex()
        self._in_flight = 0
        self._pending_params: dict[tuple[str, str], Any] = {}
        self._watched: set[tuple[str, str]] = set()

    # -- called from the GUI thread ---------------------------------------

    def request_stop(self) -> None:
        """Ask the run to end after the current sweep."""
        self.requestInterruption()

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        """Change a parameter of the running diagram.

        Queued rather than written straight into the executor, so the value
        lands between sweeps on the thread that runs them instead of racing a
        sweep in progress. The newest value for a given parameter wins: a slider
        dragged across its range should not make the worker replay every
        intermediate position.
        """
        with QMutexLocker(self._mutex):
            self._pending_params[(node_id, name)] = value

    def sync_params(self, graph: Graph) -> None:
        """Push every parameter of ``graph`` to the running diagram.

        Used after an undo, where any number of values may have changed at once.
        """
        with QMutexLocker(self._mutex):
            for node_id, node in graph.nodes.items():
                for name, value in node.params.items():
                    self._pending_params[(node_id, name)] = value

    def _take_pending_params(self) -> dict[tuple[str, str], Any]:
        with QMutexLocker(self._mutex):
            pending, self._pending_params = self._pending_params, {}
            return pending

    def set_watched(self, ports: set[tuple[str, str]]) -> None:
        """Choose which ports report their value with each sweep.

        Only what is asked for is summarised, so watching one port costs one
        thumbnail per sweep rather than shipping the whole graph's intermediate
        values across the thread boundary.
        """
        with QMutexLocker(self._mutex):
            self._watched = set(ports)

    def _current_watches(self) -> set[tuple[str, str]]:
        with QMutexLocker(self._mutex):
            return set(self._watched)

    def _collect_watched(self, result: object, iteration: int) -> list[PortValue]:
        watches = self._current_watches()
        if not watches:
            return []
        values: list[PortValue] = []
        for node_id, port in sorted(watches):
            source = self._resolve(node_id, port)
            if source is None:
                continue
            source_node, source_port = source
            outputs = result.outputs.get(source_node)  # type: ignore[attr-defined]
            if outputs is None or source_port not in outputs:
                # The node was skipped after an upstream failure.
                continue
            values.append(
                inspect_value(node_id, port, outputs[source_port], iteration)
            )
        return values

    def _resolve(self, node_id: str, port: str) -> tuple[str, str] | None:
        """Find where a watched port gets its value.

        An output carries its own. An input carries whatever is wired into
        it, which is the more interesting thing to see: clicking the input of
        a failing block shows what actually arrived there.
        """
        node = self._graph.nodes.get(node_id)
        if node is None:
            return None
        try:
            spec = registry.get(node.block)
        except registry.UnknownBlockError:
            return None
        if spec.output(port) is not None:
            return (node_id, port)
        for edge in self._graph.incoming(node_id):
            if edge.dst_port == port:
                return (edge.src, edge.src_port)
        return None

    def preview_consumed(self) -> None:
        """The GUI has finished displaying a report; allow the next one.

        Connect this to whatever handles :attr:`swept`, or previews will be
        dropped forever after the first one.
        """
        with QMutexLocker(self._mutex):
            self._in_flight = max(0, self._in_flight - 1)

    # -- worker thread ----------------------------------------------------

    def _may_emit(self) -> bool:
        with QMutexLocker(self._mutex):
            if self._in_flight > 0:
                return False
            self._in_flight += 1
            return True

    def run(self) -> None:  # noqa: C901 - the shape follows the failure modes
        # Blocks are registered per-process, but a worker may be the first
        # thing to need them in a fresh interpreter.
        registry.discover()

        try:
            # Built here, not in __init__: a VideoCapture must be opened on the
            # thread that reads from it.
            executor = Executor(self._graph, on_error=self._on_error)
        except (ValidationError, ValueError) as exc:
            self.failed.emit(str(exc))
            return

        self.started_running.emit()
        iteration = 0
        try:
            while not self.isInterruptionRequested():
                if self._iterations > 0 and iteration >= self._iterations:
                    break
                if self._interval and iteration:
                    # Sleeping in small slices keeps stop responsive even with
                    # a long interval.
                    deadline = time.monotonic() + self._interval
                    while (
                        time.monotonic() < deadline
                        and not self.isInterruptionRequested()
                    ):
                        self.msleep(5)
                    if self.isInterruptionRequested():
                        break

                for (node_id, name), value in self._take_pending_params().items():
                    try:
                        executor.set_param(node_id, name, value)
                    except (KeyError, CoercionError) as exc:
                        # A half-typed value in a text field is normal while
                        # editing; report it and keep the previous one rather
                        # than stopping the run.
                        self.failed.emit(f"{node_id}.{name}: {exc}")

                started = time.perf_counter()
                try:
                    result = executor.step(iteration)
                except StopExecution as stop:
                    self.stopped.emit(str(stop) or "a block ended the run")
                    return
                except NodeExecutionError as exc:
                    # on_error=raise: report and stop rather than dying silently.
                    self.failed.emit(str(exc))
                    return
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                if self._may_emit():
                    self.swept.emit(
                        SweepReport(
                            iteration=iteration,
                            previews_by_node={
                                node: list(previews)
                                for node, previews in result.previews_by_node.items()
                            },
                            errors={n: str(e) for n, e in result.errors.items()},
                            skipped=list(result.skipped),
                            duration_ms=elapsed_ms,
                            watched=self._collect_watched(result, iteration),
                        )
                    )
                iteration += 1
        except Exception as exc:  # a defect in pydiedi itself
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            # Releases every camera and file, on the thread that opened them.
            try:
                executor.close()
            except Exception as exc:
                self.failed.emit(f"while releasing sources: {exc}")
