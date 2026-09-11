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


@block(category="t", name="const", register_globally=False)
def const(value: int = 1) -> int:
    return value


@block(category="t", name="inc", register_globally=False)
def inc(input: int, by: int = 1) -> int:
    return input + by


@block(category="t", name="add", register_globally=False)
def add(a: int, b: int) -> int:
    return a + b


class Split(NamedTuple):
    lo: int
    hi: int


@block(category="t", name="split", register_globally=False)
def split(input: int) -> Split:
    return Split(lo=input - 1, hi=input + 1)


@block(category="t", name="boom", register_globally=False)
def boom(input: int) -> int:
    raise RuntimeError("deliberate failure")


@block(category="t", name="show", register_globally=False)
def show(input: int) -> Preview:
    return Preview(title=f"value={input}")


@block(category="t", name="sink", register_globally=False)
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
        {"a": ("const", {"value": 1}), "b": ("inc", {}), "c": ("inc", {})},
        ["a.output -> b.input", "b.output -> c.input"],
    )
    assert topological_order(graph) == ["a", "b", "c"]
    result = Executor(graph).step()
    assert result.value("c", "output") == 3


def test_fan_out_feeds_both_branches():
    """One output into two inputs -- the case a linear chain would not catch."""
    graph = build(
        {
            "src": ("const", {"value": 10}),
            "left": ("inc", {"by": 1}),
            "right": ("inc", {"by": 100}),
            "join": ("add", {}),
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
        {"src": ("const", {"value": 5}), "s": ("split", {}), "join": ("add", {})},
        ["src.output -> s.input", "s.lo -> join.a", "s.hi -> join.b"],
    )
    result = Executor(graph).step()
    assert result.outputs["s"] == {"lo": 4, "hi": 6}
    assert result.value("join", "output") == 10


def test_order_is_deterministic_across_runs():
    graph = build(
        {
            "z": ("const", {}),
            "y": ("const", {}),
            "x": ("const", {}),
            "j": ("add", {}),
        },
        ["z.output -> j.a", "y.output -> j.b"],
    )
    assert topological_order(graph) == topological_order(graph)
    assert topological_order(graph)[:3] == ["x", "y", "z"]


def test_independent_nodes_all_execute():
    graph = build({"a": ("const", {}), "b": ("const", {})}, [])
    result = Executor(graph).step()
    assert set(result.outputs) == {"a", "b"}


# -- cycles ----------------------------------------------------------------


def test_self_loop_is_rejected():
    graph = build({"a": ("inc", {})}, ["a.output -> a.input"])
    with pytest.raises(CycleError, match="a -> a"):
        topological_order(graph)


def test_two_node_cycle_is_rejected_and_named():
    graph = build(
        {"a": ("inc", {}), "b": ("inc", {})},
        ["a.output -> b.input", "b.output -> a.input"],
    )
    with pytest.raises(CycleError) as excinfo:
        topological_order(graph)
    assert "cycle" in str(excinfo.value)
    assert "a -> b -> a" in str(excinfo.value) or "b -> a -> b" in str(excinfo.value)


def test_longer_cycle_is_named():
    graph = build(
        {"a": ("inc", {}), "b": ("inc", {}), "c": ("inc", {})},
        ["a.output -> b.input", "b.output -> c.input", "c.output -> a.input"],
    )
    with pytest.raises(CycleError, match=r"a -> b -> c -> a"):
        topological_order(graph)


def test_cycle_is_reported_even_with_a_valid_prefix():
    graph = build(
        {"src": ("const", {}), "a": ("add", {}), "b": ("inc", {})},
        ["src.output -> a.a", "a.output -> b.input", "b.output -> a.b"],
    )
    with pytest.raises(CycleError):
        topological_order(graph)


# -- values and previews ---------------------------------------------------


def test_params_supply_values_for_unconnected_inputs():
    graph = build({"a": ("inc", {"input": 5, "by": 3})}, [])
    assert Executor(graph).step().value("a", "output") == 8


def test_connection_overrides_nothing_but_supplies_the_port():
    graph = build(
        {"src": ("const", {"value": 7}), "a": ("inc", {"by": 3})},
        ["src.output -> a.input"],
    )
    assert Executor(graph).step().value("a", "output") == 10


def test_previews_are_collected():
    graph = build(
        {"src": ("const", {"value": 4}), "p": ("show", {})},
        ["src.output -> p.input"],
    )
    result = Executor(graph).step()
    assert [p.title for p in result.previews] == ["value=4"]


def test_block_without_outputs_runs():
    graph = build(
        {"src": ("const", {}), "s": ("sink", {})}, ["src.output -> s.input"]
    )
    assert Executor(graph).step().outputs["s"] == {}


def test_run_repeats_and_reports_iteration():
    graph = build({"a": ("const", {"value": 2})}, [])
    result = Executor(graph).run(iterations=3)
    assert result.iteration == 2  # zero-based, three sweeps


# -- failures --------------------------------------------------------------


def test_failing_block_names_node_block_and_cause():
    graph = build(
        {"src": ("const", {}), "bad": ("boom", {})}, ["src.output -> bad.input"]
    )
    with pytest.raises(NodeExecutionError) as excinfo:
        Executor(graph).step()
    error = excinfo.value
    assert error.node_id == "bad"
    assert error.block == "boom"
    assert isinstance(error.cause, RuntimeError)
    assert "deliberate failure" in str(error)


def test_unknown_block_is_reported_with_node_id():
    graph = build({"a": ("nonexistent", {})}, [])
    with pytest.raises(ValidationError, match="node 'a'.*unknown block 'nonexistent'"):
        graph.validate()


def test_missing_required_input_is_reported():
    graph = build({"a": ("add", {"a": 1})}, [])
    with pytest.raises(ValidationError, match="required input 'b'"):
        graph.validate()


def test_unknown_port_on_edge_lists_known_ports():
    graph = build(
        {"src": ("const", {}), "a": ("inc", {})}, ["src.output -> a.nope"]
    )
    with pytest.raises(ValidationError, match="has no input 'nope'.*Known inputs"):
        graph.validate()


def test_unknown_param_lists_known_inputs():
    graph = build({"a": ("const", {"vlaue": 1})}, [])
    with pytest.raises(ValidationError, match="no input 'vlaue'.*Known inputs: value"):
        graph.validate()


def test_two_edges_into_one_input_is_rejected():
    graph = build(
        {"x": ("const", {}), "y": ("const", {}), "a": ("inc", {})},
        ["x.output -> a.input", "y.output -> a.input"],
    )
    with pytest.raises(ValidationError, match="already fed by"):
        graph.validate()


def test_connected_input_with_a_param_is_rejected():
    graph = build(
        {"src": ("const", {}), "a": ("inc", {"input": 3})},
        ["src.output -> a.input"],
    )
    with pytest.raises(ValidationError, match="both connected and given as a param"):
        graph.validate()


def test_edge_to_missing_node_is_reported():
    graph = build({"a": ("const", {})}, ["a.output -> ghost.input"])
    with pytest.raises(ValidationError, match="target node 'ghost' does not exist"):
        graph.validate()


def test_type_mismatch_is_reported():
    @block(category="t", name="stringy", register_globally=False)
    def stringy(input: str = "x") -> str:
        return input

    registry.register(stringy.spec)
    graph = build(
        {"s": ("stringy", {}), "a": ("inc", {})}, ["s.output -> a.input"]
    )
    with pytest.raises(ValidationError, match="type mismatch, str cannot feed int"):
        graph.validate()


def test_all_problems_are_reported_at_once():
    """A hand-edited file should not need six runs to find six mistakes."""
    graph = build({"a": ("add", {"wrong": 1})}, ["a.output -> ghost.input"])
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
    graph = build({"a": ("const", {})}, [])
    graph.layout["ghost"] = (0.0, 0.0)
    with pytest.raises(ValidationError, match="layout references unknown node 'ghost'"):
        graph.validate()


def test_coercion_error_names_the_node(real_blocks):
    graph = Graph(name="t")
    graph.add(Node(id="blur", block="gaussian_blur", params={"kernel_size": "big"}))
    with pytest.raises((ValidationError, CoercionError), match="blur"):
        graph.validate()
