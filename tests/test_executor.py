"""Scheduling, data flow and failure reporting.

The cycle tests are the important ones. flodiedi's scheduler put whatever it
could not order into a ``"WARNING: UNCONNECTED BLOCKS"`` log line and then
never executed those blocks, so a cyclic diagram looked like it was running
while doing nothing at all.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from pydiedi.core import block, registry
from pydiedi.core.executor import (
    CycleError,
    Executor,
    NodeExecutionError,
    topological_order,
)
from pydiedi.core.graph import Edge, Graph, Node, ValidationError
from pydiedi.core.types import CoercionError, Preview


# -- test blocks, defined once and registered per test --------------------


@block(category="t", name="t_const", register_globally=False)
def const(value: int = 1) -> int:
    return value


@block(category="t", name="t_inc", register_globally=False)
def inc(input: int, by: int = 1) -> int:
    return input + by


@block(category="t", name="t_add", register_globally=False)
def add(a: int, b: int) -> int:
    return a + b


class Split(NamedTuple):
    lo: int
    hi: int


@block(category="t", name="t_split", register_globally=False)
def split(input: int) -> Split:
    return Split(lo=input - 1, hi=input + 1)


@block(category="t", name="t_boom", register_globally=False)
def boom(input: int) -> int:
    raise RuntimeError("deliberate failure")


@block(category="t", name="t_show", register_globally=False)
def show(input: int) -> Preview:
    return Preview(title=f"value={input}")


@block(category="t", name="t_sink", register_globally=False)
def sink(input: int) -> None:
    return None


ALL = (const, inc, add, split, boom, show, sink)


@pytest.fixture(autouse=True)
def _blocks(isolated_registry, register):
    register(*ALL)


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


# -- ordering --------------------------------------------------------------


def test_linear_chain_runs_in_order():
    graph = build(
        {"a": ("t_const", {"value": 1}), "b": ("t_inc", {}), "c": ("t_inc", {})},
        ["a.output -> b.input", "b.output -> c.input"],
    )
    assert topological_order(graph) == ["a", "b", "c"]
    result = Executor(graph).step()
    assert result.value("c", "output") == 3


def test_fan_out_feeds_both_branches():
    """One output into two inputs -- the case a linear chain would not catch."""
    graph = build(
        {
            "src": ("t_const", {"value": 10}),
            "left": ("t_inc", {"by": 1}),
            "right": ("t_inc", {"by": 100}),
            "join": ("t_add", {}),
        },
        [
            "src.output -> left.input",
            "src.output -> right.input",
            "left.output -> join.a",
            "right.output -> join.b",
        ],
    )
    result = Executor(graph).step()
    assert result.value("left", "output") == 11
    assert result.value("right", "output") == 110
    assert result.value("join", "output") == 121


def test_fan_in_from_multiple_outputs_of_one_block():
    graph = build(
        {"src": ("t_const", {"value": 5}), "s": ("t_split", {}), "join": ("t_add", {})},
        ["src.output -> s.input", "s.lo -> join.a", "s.hi -> join.b"],
    )
    result = Executor(graph).step()
    assert result.outputs["s"] == {"lo": 4, "hi": 6}
    assert result.value("join", "output") == 10


def test_order_is_deterministic_across_runs():
    graph = build(
        {
            "z": ("t_const", {}),
            "y": ("t_const", {}),
            "x": ("t_const", {}),
            "j": ("t_add", {}),
        },
        ["z.output -> j.a", "y.output -> j.b"],
    )
    assert topological_order(graph) == topological_order(graph)
    assert topological_order(graph)[:3] == ["x", "y", "z"]


def test_independent_nodes_all_execute():
    graph = build({"a": ("t_const", {}), "b": ("t_const", {})}, [])
    result = Executor(graph).step()
    assert set(result.outputs) == {"a", "b"}


# -- cycles ----------------------------------------------------------------


def test_self_loop_is_rejected():
    graph = build({"a": ("t_inc", {})}, ["a.output -> a.input"])
    with pytest.raises(CycleError, match="a -> a"):
        topological_order(graph)


def test_two_node_cycle_is_rejected_and_named():
    graph = build(
        {"a": ("t_inc", {}), "b": ("t_inc", {})},
        ["a.output -> b.input", "b.output -> a.input"],
    )
    with pytest.raises(CycleError) as excinfo:
        topological_order(graph)
    assert "cycle" in str(excinfo.value)
    assert "a -> b -> a" in str(excinfo.value) or "b -> a -> b" in str(excinfo.value)


def test_longer_cycle_is_named():
    graph = build(
        {"a": ("t_inc", {}), "b": ("t_inc", {}), "c": ("t_inc", {})},
        ["a.output -> b.input", "b.output -> c.input", "c.output -> a.input"],
    )
    with pytest.raises(CycleError, match=r"a -> b -> c -> a"):
        topological_order(graph)


def test_cycle_is_reported_even_with_a_valid_prefix():
    graph = build(
        {"src": ("t_const", {}), "a": ("t_add", {}), "b": ("t_inc", {})},
        ["src.output -> a.a", "a.output -> b.input", "b.output -> a.b"],
    )
    with pytest.raises(CycleError):
        topological_order(graph)


# -- values and previews ---------------------------------------------------


def test_params_supply_values_for_unconnected_inputs():
    graph = build({"a": ("t_inc", {"input": 5, "by": 3})}, [])
    assert Executor(graph).step().value("a", "output") == 8


def test_connection_overrides_nothing_but_supplies_the_port():
    graph = build(
        {"src": ("t_const", {"value": 7}), "a": ("t_inc", {"by": 3})},
        ["src.output -> a.input"],
    )
    assert Executor(graph).step().value("a", "output") == 10


def test_previews_are_collected():
    graph = build(
        {"src": ("t_const", {"value": 4}), "p": ("t_show", {})},
        ["src.output -> p.input"],
    )
    result = Executor(graph).step()
    assert [p.title for p in result.previews] == ["value=4"]


def test_block_without_outputs_runs():
    graph = build(
        {"src": ("t_const", {}), "s": ("t_sink", {})}, ["src.output -> s.input"]
    )
    assert Executor(graph).step().outputs["s"] == {}


def test_run_repeats_and_reports_iteration():
    graph = build({"a": ("t_const", {"value": 2})}, [])
    result = Executor(graph).run(iterations=3)
    assert result.iteration == 2  # zero-based, three sweeps


# -- failures --------------------------------------------------------------


def test_failing_block_names_node_block_and_cause():
    graph = build(
        {"src": ("t_const", {}), "bad": ("t_boom", {})}, ["src.output -> bad.input"]
    )
    with pytest.raises(NodeExecutionError) as excinfo:
        Executor(graph).step()
    error = excinfo.value
    assert error.node_id == "bad"
    assert error.block == "t_boom"
    assert isinstance(error.cause, RuntimeError)
    assert "deliberate failure" in str(error)


def test_unknown_block_is_reported_with_node_id():
    graph = build({"a": ("nonexistent", {})}, [])
    with pytest.raises(ValidationError, match="node 'a'.*unknown block 'nonexistent'"):
        graph.validate()


def test_missing_required_input_is_reported():
    graph = build({"a": ("t_add", {"a": 1})}, [])
    with pytest.raises(ValidationError, match="required input 'b'"):
        graph.validate()


def test_unknown_port_on_edge_lists_known_ports():
    graph = build(
        {"src": ("t_const", {}), "a": ("t_inc", {})}, ["src.output -> a.nope"]
    )
    with pytest.raises(ValidationError, match="has no input 'nope'.*Known inputs"):
        graph.validate()


def test_unknown_param_lists_known_inputs():
    graph = build({"a": ("t_const", {"vlaue": 1})}, [])
    with pytest.raises(ValidationError, match="no input 'vlaue'.*Known inputs: value"):
        graph.validate()


def test_two_edges_into_one_input_is_rejected():
    graph = build(
        {"x": ("t_const", {}), "y": ("t_const", {}), "a": ("t_inc", {})},
        ["x.output -> a.input", "y.output -> a.input"],
    )
    with pytest.raises(ValidationError, match="already fed by"):
        graph.validate()


def test_connected_input_with_a_param_is_rejected():
    graph = build(
        {"src": ("t_const", {}), "a": ("t_inc", {"input": 3})},
        ["src.output -> a.input"],
    )
    with pytest.raises(ValidationError, match="both connected and given as a param"):
        graph.validate()


def test_edge_to_missing_node_is_reported():
    graph = build({"a": ("t_const", {})}, ["a.output -> ghost.input"])
    with pytest.raises(ValidationError, match="target node 'ghost' does not exist"):
        graph.validate()


def test_type_mismatch_is_reported():
    @block(category="t", name="t_stringy", register_globally=False)
    def stringy(input: str = "x") -> str:
        return input

    registry.register(stringy.spec)
    graph = build(
        {"s": ("t_stringy", {}), "a": ("t_inc", {})}, ["s.output -> a.input"]
    )
    with pytest.raises(ValidationError, match="type mismatch, str cannot feed int"):
        graph.validate()


def test_all_problems_are_reported_at_once():
    """A hand-edited file should not need six runs to find six mistakes."""
    graph = build({"a": ("t_add", {"wrong": 1})}, ["a.output -> ghost.input"])
    problems = graph.problems()
    assert len(problems) >= 3
    joined = "\n".join(problems)
    assert "no input 'wrong'" in joined
    assert "does not exist" in joined
    assert "required input" in joined


def test_bad_enum_param_is_reported_before_running(real_blocks):
    graph = Graph(name="t")
    graph.add(Node(id="c", block="cvt_color", params={"code": "nonsense", "input": None}))
    problems = "\n".join(graph.problems())
    assert "not a valid ColorCode" in problems
    assert "bgr2gray" in problems  # lists the valid options


def test_layout_referencing_unknown_node_is_reported():
    graph = build({"a": ("t_const", {})}, [])
    graph.layout["ghost"] = (0.0, 0.0)
    with pytest.raises(ValidationError, match="layout references unknown node 'ghost'"):
        graph.validate()


def test_coercion_error_names_the_node(real_blocks):
    graph = Graph(name="t")
    graph.add(Node(id="blur", block="gaussian_blur", params={"kernel_size": "big"}))
    with pytest.raises((ValidationError, CoercionError), match="blur"):
        graph.validate()


# -- live structural edits ------------------------------------------------
#
# flodiedi's defining quality: you never stopped a diagram to change it. Its
# execute() re-ran findExecutionOrder() at the head of any sweep that followed
# an edit (flowdiagram.cpp:183), so adding a block, dragging a wire or deleting
# an arrow took effect within one SleepTime.


def test_sync_picks_up_a_node_added_between_sweeps():
    graph = build({"a": ("t_const", {"value": 1})}, [])
    with Executor(graph) as executor:
        executor.step(0)
        graph.add(Node(id="b", block="t_inc", params={"input": 5}))
        executor.sync(graph)
        result = executor.step(1)
    assert result.value("b", "output") == 6


def test_sync_picks_up_a_new_connection():
    graph = build(
        {"a": ("t_const", {"value": 7}), "b": ("t_inc", {"input": 0})}, []
    )
    with Executor(graph) as executor:
        assert executor.step(0).value("b", "output") == 1
        graph.nodes["b"].params.pop("input")
        graph.connect("a", "output", "b", "input")
        executor.sync(graph)
        assert executor.step(1).value("b", "output") == 8


def test_sync_reorders_when_the_graph_changes():
    graph = build({"a": ("t_const", {}), "b": ("t_inc", {"input": 0})}, [])
    with Executor(graph) as executor:
        graph.nodes["b"].params.pop("input")
        graph.connect("a", "output", "b", "input")
        executor.sync(graph)
        assert executor.order.index("a") < executor.order.index("b")


def test_sync_drops_a_removed_node():
    graph = build({"a": ("t_const", {}), "b": ("t_const", {})}, [])
    with Executor(graph) as executor:
        del graph.nodes["b"]
        executor.sync(graph)
        result = executor.step()
    assert set(result.outputs) == {"a"}


def test_sync_keeps_the_instance_of_a_surviving_stateful_node(
    isolated_registry, register
):
    """Adding a filter downstream of a camera must not reopen the camera."""

    @block(category="t", name="t_counter", register_globally=False)
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self) -> int:
            self.n += 1
            return self.n

    register(Counter)
    graph = build({"c": ("t_counter", {})}, [])
    with Executor(graph) as executor:
        executor.step(0)
        executor.step(1)
        instance = executor._instances["c"]

        graph.add(Node(id="other", block="t_const"))
        executor.sync(graph)

        assert executor._instances["c"] is instance, "the instance must survive"
        assert executor.step(2).value("c", "output") == 3, "state must survive"


def test_sync_closes_the_instance_of_a_removed_node(isolated_registry, register):
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
        del graph.nodes["r"]
        executor.sync(graph)
        assert closed == ["closed"]


def test_sync_replaces_the_instance_when_the_block_behind_an_id_changes():
    graph = build({"a": ("t_const", {})}, [])
    with Executor(graph) as executor:
        graph.nodes["a"] = Node(id="a", block="t_inc", params={"input": 1})
        executor.sync(graph)
        assert executor.step().value("a", "output") == 2


# -- an unplugged input keeps its last value ------------------------------
#
# flodiedi's removeConnection touched only the edge list (flowdiagram.cpp:112);
# the receiving block kept whatever was last written into its property. Pulling
# a wire froze the downstream image instead of blanking it, which was a useful
# debugging move.


def test_an_input_keeps_its_value_when_its_edge_is_removed():
    graph = build(
        {"src": ("t_const", {"value": 5}), "b": ("t_inc", {})},
        ["src.output -> b.input"],
    )
    with Executor(graph) as executor:
        assert executor.step(0).value("b", "output") == 6
        graph.edges.clear()
        executor.sync(graph)
        # Still 6: the input held the 5 it last received, rather than falling
        # back to a default or blanking.
        assert executor.step(1).value("b", "output") == 6


def test_the_held_value_stops_updating_once_unplugged(isolated_registry, register):
    @block(category="t", name="t_ramp", register_globally=False)
    class Ramp:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self) -> int:
            self.n += 1
            return self.n

    register(Ramp)
    graph = build({"src": ("t_ramp", {}), "b": ("t_inc", {})}, ["src.output -> b.input"])
    with Executor(graph) as executor:
        executor.step(0)
        assert executor.step(1).value("b", "output") == 3  # ramp 2 + 1

        graph.edges.clear()
        executor.sync(graph)
        frozen = executor.step(2).value("b", "output")
        assert executor.step(3).value("b", "output") == frozen, "must stay frozen"
        assert executor.step(4).value("b", "output") == frozen


def test_setting_a_parameter_takes_an_unplugged_input_back():
    """Disconnect, then hand-drive the input -- as flodiedi allowed, because
    there a port and a parameter were the same property."""
    graph = build(
        {"src": ("t_const", {"value": 5}), "b": ("t_inc", {})},
        ["src.output -> b.input"],
    )
    with Executor(graph) as executor:
        executor.step(0)
        graph.edges.clear()
        executor.sync(graph)
        executor.set_param("b", "input", 100)
        assert executor.step(1).value("b", "output") == 101


def test_a_connected_input_is_not_overridden_by_a_stale_held_value():
    graph = build(
        {"src": ("t_const", {"value": 5}), "b": ("t_inc", {})},
        ["src.output -> b.input"],
    )
    with Executor(graph) as executor:
        executor.step(0)
        graph.nodes["src"].params["value"] = 50
        executor.sync(graph)
        executor.set_param("src", "value", 50)
        assert executor.step(1).value("b", "output") == 51


def test_a_never_connected_input_uses_its_parameter():
    graph = build({"b": ("t_inc", {"input": 3})}, [])
    with Executor(graph) as executor:
        assert executor.step().value("b", "output") == 4
