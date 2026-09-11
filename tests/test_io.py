"""Reading and writing diagram files.

Two things are being pinned down here:

* **Round-trip stability.** Load, save, load again must give the same graph,
  and the second save must be byte-identical to the first. Without that,
  opening a diagram in the editor and closing it again produces a spurious
  diff -- which is exactly what made flodiedi's files unpleasant to version.
* **Errors name the line.** flodiedi's loader dereferenced the result of
  ``createBlockByKey()`` without a NULL check, so a diagram naming an unbuilt
  plugin crashed the editor with no indication of where the problem was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pydiedi.core import io
from pydiedi.core.graph import Edge, Graph, Node
from pydiedi.core.io import FORMAT_VERSION, DiagramSyntaxError

MINIMAL = """
version: 1
name: demo
nodes:
  a: {block: imread, params: {path: x.png}}
  b: {block: cvt_color, params: {code: bgr2gray}}
edges:
  - a.output -> b.input
layout:
  a: [0, 0]
  b: [160, 40]
"""


def test_loads_nodes_edges_and_layout():
    graph = io.loads(MINIMAL)
    assert graph.name == "demo"
    assert list(graph.nodes) == ["a", "b"]
    assert graph.nodes["a"].block == "imread"
    assert graph.nodes["a"].params == {"path": "x.png"}
    assert [str(e) for e in graph.edges] == ["a.output -> b.input"]
    assert graph.layout == {"a": (0.0, 0.0), "b": (160.0, 40.0)}


def test_node_order_is_preserved():
    text = """
    version: 1
    nodes:
      zulu:  {block: imread, params: {path: z}}
      alpha: {block: imread, params: {path: a}}
      mike:  {block: imread, params: {path: m}}
    """
    assert list(io.loads(text).nodes) == ["zulu", "alpha", "mike"]


def test_layout_is_optional_and_diagram_still_runs_without_it():
    graph = io.loads("version: 1\nnodes:\n  a: {block: imread, params: {path: x}}\n")
    assert graph.layout == {}
    assert list(graph.nodes) == ["a"]


def test_params_are_optional():
    graph = io.loads("version: 1\nnodes:\n  a: {block: canny}\n")
    assert graph.nodes["a"].params == {}


def test_line_numbers_are_recorded_on_nodes():
    graph = io.loads(MINIMAL)
    assert graph.nodes["a"].line == 5
    assert graph.nodes["b"].line == 6


def test_internal_line_key_does_not_leak_into_params():
    graph = io.loads(MINIMAL)
    for node in graph.nodes.values():
        assert "__line__" not in node.params


# -- round trip ------------------------------------------------------------


def test_round_trip_preserves_the_graph():
    original = io.loads(MINIMAL)
    reloaded = io.loads(io.dumps(original))
    assert reloaded.name == original.name
    assert list(reloaded.nodes) == list(original.nodes)
    assert {k: v.params for k, v in reloaded.nodes.items()} == {
        k: v.params for k, v in original.nodes.items()
    }
    assert [str(e) for e in reloaded.edges] == [str(e) for e in original.edges]
    assert reloaded.layout == original.layout


def test_save_is_idempotent():
    """A second save must be byte-identical, or every open/close makes a diff."""
    once = io.dumps(io.loads(MINIMAL))
    twice = io.dumps(io.loads(once))
    assert once == twice


def test_written_file_is_readable_and_compact():
    text = io.dumps(io.loads(MINIMAL))
    assert f"version: {FORMAT_VERSION}" in text
    # Node definitions stay on one line each, edges are single strings.
    assert "a: {block: imread, params: {path: x.png}}" in text
    assert "- a.output -> b.input" in text
    assert "a: [0, 0]" in text


def test_whole_number_positions_are_written_without_a_decimal_point():
    graph = Graph(name="t")
    graph.add(Node(id="a", block="canny"))
    graph.layout["a"] = (12.0, -34.5)
    assert "a: [12, -34.5]" in io.dumps(graph)


def test_dump_and_load_a_file(tmp_path: Path):
    graph = io.loads(MINIMAL)
    target = tmp_path / "d.yaml"
    io.dump(graph, target)
    assert io.load(target).name == "demo"


def test_enum_params_survive_a_round_trip_as_names():
    from enum import Enum

    class Mode(Enum):
        fast = 0

    graph = Graph(name="t")
    graph.add(Node(id="a", block="x", params={"mode": Mode.fast}))
    assert "mode: fast" in io.dumps(graph)
    assert io.loads(io.dumps(graph)).nodes["a"].params == {"mode": "fast"}


def test_path_params_are_written_as_strings():
    graph = Graph(name="t")
    graph.add(Node(id="a", block="imread", params={"path": Path("a/b.png")}))
    assert "path: a/b.png" in io.dumps(graph)


def test_fixture_round_trips():
    """The shipped acceptance diagram must survive a round trip unchanged."""
    fixture = Path(__file__).parent / "fixtures" / "basic.yaml"
    graph = io.load(fixture)
    assert io.dumps(io.loads(io.dumps(graph))) == io.dumps(graph)


# -- malformed input -------------------------------------------------------


def test_empty_file_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="file is empty"):
        io.loads("")


def test_missing_version_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="missing 'version'"):
        io.loads("nodes: {}\n")


def test_unsupported_version_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="format version 99"):
        io.loads("version: 99\nnodes: {}\n")


def test_top_level_list_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="expected a mapping"):
        io.loads("- a\n- b\n")


def test_invalid_yaml_names_the_source():
    with pytest.raises(DiagramSyntaxError, match="mydiagram.yaml"):
        io.loads("version: 1\nnodes: {a: [\n", source="mydiagram.yaml")


def test_node_without_block_is_rejected_with_its_line():
    text = "version: 1\nnodes:\n  a: {params: {x: 1}}\n"
    with pytest.raises(DiagramSyntaxError, match=r":3: node 'a' has no 'block' key"):
        io.loads(text, source="d.yaml")


def test_node_that_is_not_a_mapping_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="must be a mapping with a 'block' key"):
        io.loads("version: 1\nnodes:\n  a: imread\n")


def test_unknown_node_key_is_rejected():
    text = "version: 1\nnodes:\n  a: {block: canny, colour: red}\n"
    with pytest.raises(DiagramSyntaxError, match="unknown key\\(s\\) 'colour'"):
        io.loads(text)


def test_invalid_node_id_is_rejected():
    with pytest.raises(DiagramSyntaxError, match="not a valid node id"):
        io.loads("version: 1\nnodes:\n  '9lives': {block: canny}\n")


def test_params_that_are_not_a_mapping_are_rejected():
    with pytest.raises(DiagramSyntaxError, match="'params' of node 'a' must be a mapping"):
        io.loads("version: 1\nnodes:\n  a: {block: canny, params: [1, 2]}\n")


@pytest.mark.parametrize(
    "edge",
    [
        "a.output",
        "a -> b",
        "a.output -> b",
        "a.output - b.input",
        "a.output -> b.input -> c.input",
        ".output -> b.input",
    ],
)
def test_malformed_edge_is_rejected(edge: str):
    text = f"version: 1\nnodes:\n  a: {{block: canny}}\nedges:\n  - {edge}\n"
    with pytest.raises(DiagramSyntaxError, match="cannot parse edge"):
        io.loads(text)


def test_edge_that_is_not_a_string_is_rejected():
    text = "version: 1\nnodes:\n  a: {block: canny}\nedges:\n  - {from: a, to: b}\n"
    with pytest.raises(DiagramSyntaxError, match="edge #1 must be a string"):
        io.loads(text)


def test_edges_that_are_not_a_list_are_rejected():
    with pytest.raises(DiagramSyntaxError, match="'edges' must be a list"):
        io.loads("version: 1\nnodes: {}\nedges: {a: b}\n")


def test_whitespace_around_the_arrow_is_tolerated():
    text = (
        "version: 1\nnodes:\n  a: {block: canny}\n  b: {block: canny}\n"
        "edges:\n  - 'a.output->b.input'\n"
    )
    assert [str(e) for e in io.loads(text).edges] == ["a.output -> b.input"]


def test_bad_layout_entry_is_rejected():
    text = "version: 1\nnodes:\n  a: {block: canny}\nlayout:\n  a: [1, 2, 3]\n"
    with pytest.raises(DiagramSyntaxError, match=r"layout of 'a' must be \[x, y\]"):
        io.loads(text)


def test_non_numeric_layout_is_rejected():
    text = "version: 1\nnodes:\n  a: {block: canny}\nlayout:\n  a: [left, top]\n"
    with pytest.raises(DiagramSyntaxError, match="must be two numbers"):
        io.loads(text)


def test_missing_file_is_reported_clearly():
    with pytest.raises(DiagramSyntaxError, match="cannot read"):
        io.load("/nonexistent/diagram.yaml")


def test_yaml_tags_are_not_executed():
    """SafeLoader only: a diagram file must never be able to run code."""
    with pytest.raises(DiagramSyntaxError):
        io.loads("version: 1\nnodes: !!python/object/apply:os.system ['echo hi']\n")
