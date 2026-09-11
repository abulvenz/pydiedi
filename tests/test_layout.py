"""Layered layout: layering, crossing reduction, and lanes for long edges.

flodiedi used Graphviz for this and then discarded the routing it computed --
``GVGraph::edges()`` builds a full QPainterPath from the splines
(``gvgraph.cpp:216-265``) and ``FlowDiagramScene::layout()`` never calls it,
resetting every connection to a straight line instead. That is why its edges
ran through nodes.
"""

from __future__ import annotations

import pytest

from pydiedi.core.graph import Graph, Node
from pydiedi.core.layout import (
    DEFAULT_BOX,
    NodeBox,
    count_crossings,
    layered_layout,
)


def chain(*node_ids: str) -> Graph:
    graph = Graph(name="t")
    for node_id in node_ids:
        graph.add(Node(id=node_id, block="b"))
    for left, right in zip(node_ids, node_ids[1:], strict=False):
        graph.connect(left, "output", right, "input")
    return graph


def adjacency_of(graph: Graph) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {n: [] for n in graph.nodes}
    for edge in graph.edges:
        out[edge.src].append(edge.dst)
    return out


# -- layering -------------------------------------------------------------


def test_a_chain_gets_one_node_per_layer():
    result = layered_layout(chain("a", "b", "c"))
    assert result.layers == [["a"], ["b"], ["c"]]


def test_layers_run_left_to_right():
    result = layered_layout(chain("a", "b", "c"))
    xs = [result.positions[n][0] for n in ("a", "b", "c")]
    assert xs == sorted(xs)
    assert len(set(xs)) == 3


def test_independent_nodes_share_a_layer():
    graph = Graph(name="t")
    for node_id in ("a", "b", "c"):
        graph.add(Node(id=node_id, block="b"))
    result = layered_layout(graph)
    assert result.layers == [["a", "b", "c"]]


def test_a_node_sits_after_its_deepest_input():
    """Longest path, not shortest: a late input must not be drawn backwards."""
    graph = Graph(name="t")
    for node_id in ("a", "b", "c", "sink"):
        graph.add(Node(id=node_id, block="b"))
    graph.connect("a", "output", "b", "input")
    graph.connect("b", "output", "c", "input")
    graph.connect("a", "output", "sink", "in1")
    graph.connect("c", "output", "sink", "in2")
    layer_of = layered_layout(graph).layer_of
    assert layer_of["sink"] > layer_of["c"]


def test_an_empty_graph_is_handled():
    assert layered_layout(Graph(name="t")).positions == {}


def test_a_cycle_is_still_laid_out():
    """A cyclic diagram cannot run, but it must be viewable -- that is exactly
    when you need to look at it."""
    graph = chain("a", "b", "c")
    graph.connect("c", "output", "a", "input")
    result = layered_layout(graph)
    assert set(result.positions) == {"a", "b", "c"}


def test_a_self_loop_does_not_hang():
    graph = Graph(name="t")
    graph.add(Node(id="a", block="b"))
    graph.connect("a", "output", "a", "input")
    assert set(layered_layout(graph).positions) == {"a"}


# -- crossing reduction ---------------------------------------------------


def test_deliberate_crossings_are_removed():
    graph = Graph(name="t")
    for i in range(4):
        graph.add(Node(id=f"src{i}", block="b"))
        graph.add(Node(id=f"dst{i}", block="b"))
    for a, b in ((0, 3), (1, 2), (2, 1), (3, 0)):
        graph.connect(f"src{a}", "output", f"dst{b}", "input")

    result = layered_layout(graph)
    assert count_crossings(result.layers, adjacency_of(graph)) == 0


def test_a_partial_crossing_is_improved():
    graph = Graph(name="t")
    for i in range(3):
        graph.add(Node(id=f"s{i}", block="b"))
        graph.add(Node(id=f"d{i}", block="b"))
    for a, b in ((0, 2), (1, 1), (2, 0)):
        graph.connect(f"s{a}", "output", f"d{b}", "input")

    naive = [["s0", "s1", "s2"], ["d0", "d1", "d2"]]
    assert count_crossings(naive, adjacency_of(graph)) > 0
    result = layered_layout(graph)
    assert count_crossings(result.layers, adjacency_of(graph)) == 0


def test_the_layout_is_deterministic():
    graph = Graph(name="t")
    for i in range(5):
        graph.add(Node(id=f"n{i}", block="b"))
    graph.connect("n0", "output", "n3", "input")
    graph.connect("n1", "output", "n4", "input")
    graph.connect("n2", "output", "n3", "other")
    assert layered_layout(graph).positions == layered_layout(graph).positions


def test_count_crossings_counts_a_known_case():
    layers = [["a", "b"], ["x", "y"]]
    assert count_crossings(layers, {"a": ["y"], "b": ["x"]}) == 1
    assert count_crossings(layers, {"a": ["x"], "b": ["y"]}) == 0


# -- lanes for long edges -------------------------------------------------


def test_a_short_edge_has_no_waypoints():
    graph = chain("a", "b")
    assert layered_layout(graph).waypoints == {}


def test_an_edge_spanning_layers_gets_waypoints():
    graph = chain("a", "b", "c", "d")
    graph.connect("a", "output", "d", "shortcut")
    result = layered_layout(graph)
    route = result.route_for(
        next(e for e in graph.edges if e.dst_port == "shortcut")
    )
    assert len(route) == 2, "one waypoint per layer it passes over"


def test_a_long_edge_is_routed_clear_of_the_nodes_it_passes():
    """The point of the whole exercise: a lane, not a line through the middle."""
    graph = chain("a", "b", "c", "d")
    graph.connect("a", "output", "d", "shortcut")
    boxes = {n: NodeBox(150.0, 80.0) for n in graph.nodes}
    result = layered_layout(graph, boxes)

    route = result.route_for(
        next(e for e in graph.edges if e.dst_port == "shortcut")
    )
    for x, y in route:
        for node_id, (nx, ny) in result.positions.items():
            box = boxes[node_id]
            inside_x = nx <= x <= nx + box.width
            inside_y = ny <= y <= ny + box.height
            assert not (inside_x and inside_y), (
                f"the routed edge passes through {node_id}"
            )


def test_waypoints_advance_left_to_right():
    graph = chain("a", "b", "c", "d", "e")
    graph.connect("a", "output", "e", "shortcut")
    result = layered_layout(graph)
    route = result.route_for(
        next(e for e in graph.edges if e.dst_port == "shortcut")
    )
    xs = [x for x, _ in route]
    assert xs == sorted(xs)


def test_route_for_an_unrouted_edge_is_empty():
    graph = chain("a", "b")
    assert layered_layout(graph).route_for(graph.edges[0]) == []


# -- coordinates ----------------------------------------------------------


def test_nodes_in_a_layer_do_not_overlap():
    graph = Graph(name="t")
    graph.add(Node(id="src", block="b"))
    for i in range(5):
        graph.add(Node(id=f"n{i}", block="b"))
        graph.connect("src", "output", f"n{i}", "input")

    boxes = {n: NodeBox(150.0, 80.0) for n in graph.nodes}
    result = layered_layout(graph, boxes)
    ys = sorted(result.positions[f"n{i}"][1] for i in range(5))
    for upper, lower in zip(ys, ys[1:], strict=False):
        assert lower - upper >= 80.0, "nodes in a column must not overlap"


def test_node_sizes_are_respected():
    """flodiedi passed boundingRect(), which excludes child items, so its
    labels and previews overlapped after a layout."""
    graph = Graph(name="t")
    graph.add(Node(id="src", block="b"))
    graph.add(Node(id="tall", block="b"))
    graph.add(Node(id="short", block="b"))
    graph.connect("src", "output", "tall", "input")
    graph.connect("src", "output", "short", "input")

    boxes = {
        "src": NodeBox(150.0, 80.0),
        "tall": NodeBox(150.0, 400.0),
        "short": NodeBox(150.0, 40.0),
    }
    result = layered_layout(graph, boxes)
    gap = abs(result.positions["tall"][1] - result.positions["short"][1])
    assert gap >= 400.0 or gap >= 40.0
    tall_y = result.positions["tall"][1]
    short_y = result.positions["short"][1]
    if tall_y < short_y:
        assert short_y >= tall_y + 400.0
    else:
        assert tall_y >= short_y + 40.0


def test_a_node_is_pulled_towards_what_feeds_it():
    """A chain should come out roughly straight, not stepped."""
    graph = Graph(name="t")
    for node_id in ("a", "b", "c", "x", "y"):
        graph.add(Node(id=node_id, block="b"))
    graph.connect("a", "output", "b", "input")
    graph.connect("b", "output", "c", "input")
    boxes = {n: NodeBox(150.0, 80.0) for n in graph.nodes}
    result = layered_layout(graph, boxes)
    ys = [result.positions[n][1] for n in ("a", "b", "c")]
    assert max(ys) - min(ys) < 80.0, f"the chain should stay level, got {ys}"


def test_default_box_is_used_when_no_size_is_given():
    result = layered_layout(chain("a", "b"))
    assert result.positions["b"][0] >= DEFAULT_BOX.width


@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_layouts_of_various_sizes_place_every_node(count: int):
    graph = Graph(name="t")
    for i in range(count):
        graph.add(Node(id=f"n{i}", block="b"))
    for i in range(count - 1):
        graph.connect(f"n{i}", "output", f"n{i + 1}", "input")
    assert len(layered_layout(graph).positions) == count
