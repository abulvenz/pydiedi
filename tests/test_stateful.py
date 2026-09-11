"""Stateful blocks: state across sweeps, and resources that get released.

flodiedi's ``VideoFile`` held a ``VideoCapture`` and a ``QMutex`` as members
and never released either -- the device stayed claimed until the process
ended. The lifecycle tests here exist so that cannot come back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pydiedi.core import block, registry
from pydiedi.core.executor import Executor, NodeExecutionError, OnError, StopExecution
from pydiedi.core.graph import Edge, Graph, Node

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO = FIXTURES / "moving_square.avi"


def build(nodes: dict[str, tuple[str, dict]], edges: list[str]) -> Graph:
    graph = Graph(name="t")
    for node_id, (blk, params) in nodes.items():
        graph.add(Node(id=node_id, block=blk, params=params))
    for spec in edges:
        left, right = spec.split("->")
        src, src_port = left.strip().split(".")
        dst, dst_port = right.strip().split(".")
        graph.edges.append(Edge(src, src_port, dst, dst_port))
    return graph


# -- state across sweeps ---------------------------------------------------


def test_instance_state_survives_between_sweeps(isolated_registry, register):
    @block(category="t", name="t_counter", register_globally=False)
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, step: int = 1) -> int:
            self.n += step
            return self.n

    register(Counter)
    graph = build({"c": ("t_counter", {"step": 2})}, [])
    with Executor(graph) as executor:
        assert executor.step(0).value("c", "output") == 2
        assert executor.step(1).value("c", "output") == 4
        assert executor.step(2).value("c", "output") == 6


def test_two_nodes_of_one_stateful_block_have_separate_state(
    isolated_registry, register
):
    """Two 'camera' nodes must be two devices, not one shared object."""

    @block(category="t", name="t_counter", register_globally=False)
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, step: int = 1) -> int:
            self.n += step
            return self.n

    register(Counter)
    graph = build({"a": ("t_counter", {"step": 1}), "b": ("t_counter", {"step": 10})}, [])
    with Executor(graph) as executor:
        executor.step(0)
        result = executor.step(1)
    assert result.value("a", "output") == 2
    assert result.value("b", "output") == 20


def test_a_fresh_executor_starts_from_fresh_state(isolated_registry, register):
    @block(category="t", name="t_counter", register_globally=False)
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self) -> int:
            self.n += 1
            return self.n

    register(Counter)
    graph = build({"c": ("t_counter", {})}, [])
    with Executor(graph) as first:
        first.run(iterations=3)
    with Executor(graph) as second:
        assert second.step().value("c", "output") == 1


def test_stateful_nodes_are_reported(isolated_registry, register, real_blocks):
    graph = build(
        {"v": ("video_file", {"path": str(VIDEO)}), "b": ("frame_buffer", {})},
        ["v.image -> b.input"],
    )
    with Executor(graph) as executor:
        assert sorted(executor.stateful_nodes) == ["b", "v"]


# -- lifecycle -------------------------------------------------------------


def test_close_is_called_on_context_exit(isolated_registry, register):
    closed: list[str] = []

    @block(category="t", name="t_res", register_globally=False)
    class Resource:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append("closed")

    register(Resource)
    graph = build({"r": ("t_res", {})}, [])
    with Executor(graph) as executor:
        executor.step()
    assert closed == ["closed"]


def test_close_runs_once_per_node(isolated_registry, register):
    closed: list[int] = []

    @block(category="t", name="t_res", register_globally=False)
    class Resource:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append(1)

    register(Resource)
    graph = build({"a": ("t_res", {}), "b": ("t_res", {})}, [])
    with Executor(graph) as executor:
        executor.step()
    assert len(closed) == 2


def test_close_happens_even_when_the_body_raises(isolated_registry, register):
    closed: list[str] = []

    @block(category="t", name="t_res", register_globally=False)
    class Resource:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append("closed")

    register(Resource)
    graph = build({"r": ("t_res", {})}, [])
    with pytest.raises(ZeroDivisionError):
        with Executor(graph):
            raise ZeroDivisionError
    assert closed == ["closed"]


def test_one_failing_close_does_not_prevent_the_others(isolated_registry, register):
    closed: list[str] = []

    @block(category="t", name="t_bad_close", register_globally=False)
    class BadClose:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            raise OSError("device busy")

    @block(category="t", name="t_good_close", register_globally=False)
    class GoodClose:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append("good")

    register(BadClose, GoodClose)
    graph = build({"a": ("t_bad_close", {}), "b": ("t_good_close", {})}, [])
    executor = Executor(graph)
    with pytest.raises(ExceptionGroup):
        executor.close()
    assert closed == ["good"]


# -- the real video source -------------------------------------------------


def test_video_file_yields_successive_frames(real_blocks):
    graph = build({"v": ("video_file", {"path": str(VIDEO)})}, [])
    with Executor(graph) as executor:
        first = executor.step(0)
        second = executor.step(1)

    assert first.value("v", "index") == 0
    assert second.value("v", "index") == 1
    assert first.value("v", "fps") == pytest.approx(10.0)
    assert first.value("v", "image").shape == (96, 96, 3)
    assert not np.array_equal(first.value("v", "image"), second.value("v", "image"))


def test_video_file_ends_the_run_rather_than_failing(real_blocks):
    graph = build({"v": ("video_file", {"path": str(VIDEO)})}, [])
    with Executor(graph) as executor:
        result = executor.run(iterations=0)  # until the source stops
    assert result.stopped_by is not None
    assert "no more frames" in result.stopped_by


def test_video_file_loops_when_asked(real_blocks):
    graph = build({"v": ("video_file", {"path": str(VIDEO), "loop": True})}, [])
    with Executor(graph) as executor:
        result = executor.run(iterations=15)  # more sweeps than the 12 frames
    assert result.stopped_by is None
    assert result.value("v", "index") < 12  # wrapped around


def test_missing_video_is_reported_with_the_node(real_blocks):
    graph = build({"v": ("video_file", {"path": "/nonexistent/x.avi"})}, [])
    with Executor(graph) as executor:
        with pytest.raises(NodeExecutionError, match="node 'v'"):
            executor.step()


def test_video_capture_is_released(real_blocks):
    graph = build({"v": ("video_file", {"path": str(VIDEO)})}, [])
    executor = Executor(graph)
    executor.step()
    instance = executor._instances["v"]
    assert instance._capture is not None
    executor.close()
    assert instance._capture is None


def test_an_unchanged_path_keeps_reading_the_same_capture(real_blocks):
    graph = build({"v": ("video_file", {"path": str(VIDEO)})}, [])
    with Executor(graph) as executor:
        executor.step(0)
        executor.step(1)
        instance = executor._instances["v"]
        capture = instance._capture
        executor.step(2)
        assert instance._capture is capture, "must not reopen on every sweep"
        assert instance._index == 3


def test_changing_the_path_reopens_the_source(real_blocks, tmp_path):
    """Ports live on __call__ rather than __init__, so a changed path takes effect.

    flodiedi needed an explicit ``setFilename()`` slot that reloaded the
    capture; here it falls out of where the ports are declared.
    """
    import shutil

    other = tmp_path / "copy.avi"
    shutil.copy(VIDEO, other)

    graph = build({"v": ("video_file", {"path": str(VIDEO)})}, [])
    with Executor(graph) as executor:
        executor.step(0)
        executor.step(1)
        instance = executor._instances["v"]
        assert instance._index == 2

        frame = instance(other)
        assert instance._opened == other
        assert instance._index == 1, "a new file must restart the frame count"
        assert frame.index == 0


def test_frame_buffer_delays_by_one_sweep(real_blocks):
    graph = build(
        {"v": ("video_file", {"path": str(VIDEO)}), "b": ("frame_buffer", {})},
        ["v.image -> b.input"],
    )
    with Executor(graph) as executor:
        first = executor.step(0)
        second = executor.step(1)

    # First sweep has nothing to delay, so it passes the current frame through.
    assert np.array_equal(first.value("b", "output"), first.value("v", "image"))
    # Second sweep returns the frame from the first.
    assert np.array_equal(second.value("b", "output"), first.value("v", "image"))


def test_frame_differencing_pipeline_detects_the_moving_square(real_blocks):
    """The motivating use case for FrameBuffer, end to end."""
    graph = build(
        {
            "v": ("video_file", {"path": str(VIDEO)}),
            "prev": ("frame_buffer", {}),
            "diff": ("abs_diff", {}),
            "gray": ("cvt_color", {"code": "bgr2gray"}),
            "mask": ("threshold", {"level": 40}),
            "motion": ("count_non_zero", {}),
        },
        [
            "v.image -> prev.input",
            "v.image -> diff.a",
            "prev.output -> diff.b",
            "diff.output -> gray.input",
            "gray.output -> mask.input",
            "mask.output -> motion.input",
        ],
    )
    with Executor(graph) as executor:
        first = executor.step(0)
        second = executor.step(1)

    # Sweep 0 diffs a frame against itself: no motion.
    assert first.value("motion", "output") == 0
    # Sweep 1 diffs against the previous frame: the square moved.
    assert second.value("motion", "output") > 0


# -- tolerant error handling ----------------------------------------------


@pytest.fixture
def failing_graph(register):
    @block(category="t", name="t_src", register_globally=False)
    def src(value: int = 1) -> int:
        return value

    @block(category="t", name="t_boom", register_globally=False)
    def boom(input: int) -> int:
        raise RuntimeError("nope")

    @block(category="t", name="t_pass", register_globally=False)
    def passthrough(input: int) -> int:
        return input

    register(src, boom, passthrough)
    #   src -+-> boom -> after      (fails, so 'after' is skipped)
    #        +-> other              (independent, must still run)
    return build(
        {
            "src": ("t_src", {"value": 7}),
            "boom": ("t_boom", {}),
            "after": ("t_pass", {}),
            "other": ("t_pass", {}),
        },
        [
            "src.output -> boom.input",
            "boom.output -> after.input",
            "src.output -> other.input",
        ],
    )


def test_raise_mode_aborts_the_sweep(isolated_registry, failing_graph):
    with pytest.raises(NodeExecutionError, match="node 'boom'"):
        Executor(failing_graph, on_error=OnError.raise_).step()


def test_skip_mode_records_the_error(isolated_registry, failing_graph):
    result = Executor(failing_graph, on_error="skip").step()
    assert set(result.errors) == {"boom"}
    assert result.ok is False


def test_skip_mode_skips_only_what_depended_on_the_failure(
    isolated_registry, failing_graph
):
    result = Executor(failing_graph, on_error="skip").step()
    assert result.skipped == ["after"]
    assert result.value("other", "output") == 7


def test_skip_mode_blocks_transitive_dependents(isolated_registry, register):
    @block(category="t", name="t_boom", register_globally=False)
    def boom(input: int = 0) -> int:
        raise RuntimeError("nope")

    @block(category="t", name="t_pass", register_globally=False)
    def passthrough(input: int) -> int:
        return input

    register(boom, passthrough)
    graph = build(
        {"a": ("t_boom", {}), "b": ("t_pass", {}), "c": ("t_pass", {})},
        ["a.output -> b.input", "b.output -> c.input"],
    )
    result = Executor(graph, on_error="skip").step()
    assert sorted(result.skipped) == ["b", "c"]


def test_skip_mode_keeps_running_over_iterations(isolated_registry, failing_graph):
    result = Executor(failing_graph, on_error="skip").run(iterations=3)
    assert result.iteration == 2
    assert set(result.errors) == {"boom"}


def test_stop_execution_is_not_treated_as_an_error(isolated_registry, register):
    @block(category="t", name="t_stop", register_globally=False)
    def stop() -> int:
        raise StopExecution("enough")

    register(stop)
    graph = build({"s": ("t_stop", {})}, [])
    result = Executor(graph, on_error="skip").run(iterations=0)
    assert result.errors == {}
    assert result.stopped_by == "enough"
