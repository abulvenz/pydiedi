"""Editing commands, validation and undo -- without a GUI in sight.

That is the point of putting them in ``core``: the part of an editor most
worth testing is the hardest to test through a window.
"""

from __future__ import annotations

import pytest

from pydiedi.core import io, registry
from pydiedi.core.edit import (
    AddNode,
    Connect,
    Disconnect,
    EditError,
    EditSession,
    MoveNode,
    RemoveNode,
    RenameNode,
    SetParam,
    can_connect,
    unique_node_id,
)
from pydiedi.core.graph import Graph, Node


@pytest.fixture
def session(real_blocks) -> EditSession:
    return EditSession(Graph(name="t"))


@pytest.fixture
def chain(session: EditSession) -> EditSession:
    """imread -> cvt_color -> canny, wired up."""
    session.add_node("imread", node_id="src", params={"path": "x.png"})
    session.add_node("cvt_color", node_id="gray")
    session.add_node("canny", node_id="edges")
    session.connect("src", "output", "gray", "input")
    session.connect("gray", "output", "edges", "input")
    return session


# -- node ids -------------------------------------------------------------


def test_unique_id_uses_the_preferred_name_when_free(session):
    assert unique_node_id(session.graph, "canny") == "canny"


def test_unique_id_appends_a_number(session):
    session.add_node("canny")
    session.add_node("canny")
    assert set(session.graph.nodes) == {"canny", "canny_2"}


def test_unique_id_continues_past_existing_numbers(session):
    for _ in range(3):
        session.add_node("canny")
    assert set(session.graph.nodes) == {"canny", "canny_2", "canny_3"}


def test_add_node_returns_the_id_it_chose(session):
    assert session.add_node("canny") == "canny"
    assert session.add_node("canny") == "canny_2"


def test_adding_an_unknown_block_suggests_a_close_match(session):
    with pytest.raises(registry.UnknownBlockError, match="Did you mean 'canny'"):
        session.add_node("cany")


# -- connection validation ------------------------------------------------


def test_a_valid_connection_is_allowed(chain):
    chain.add_node("gaussian_blur", node_id="blur")
    assert can_connect(chain.graph, "gray", "output", "blur", "input") is None


def test_self_connection_is_refused(chain):
    assert "cannot be connected to itself" in can_connect(
        chain.graph, "gray", "output", "gray", "input"
    )


def test_type_mismatch_is_refused_with_both_type_names(chain):
    chain.add_node("count_non_zero", node_id="count")
    chain.add_node("gaussian_blur", node_id="blur")
    chain.connect("edges", "output", "count", "input")
    reason = can_connect(chain.graph, "count", "output", "blur", "input")
    assert reason == "int cannot feed Mat"


def test_occupied_input_is_refused_and_names_the_existing_source(chain):
    chain.add_node("imread", node_id="other", params={"path": "y.png"})
    reason = can_connect(chain.graph, "other", "output", "gray", "input")
    assert "already connected to src.output" in reason


def test_cycle_is_refused(session):
    # The target input must be free, or the occupied-input rule fires first --
    # both refusals are correct, this one isolates the cycle check.
    session.add_node("gaussian_blur", node_id="a")
    session.add_node("canny", node_id="b")
    session.connect("a", "output", "b", "input")
    # Feeding a from b is refused because b is already fed by a.
    reason = can_connect(session.graph, "b", "output", "a", "input")
    assert reason == "this would create a cycle: b is already downstream of a"


def test_indirect_cycle_is_refused(session):
    for node_id in ("a", "b", "c"):
        session.add_node("gaussian_blur", node_id=node_id)
    session.connect("a", "output", "b", "input")
    session.connect("b", "output", "c", "input")
    assert "cycle" in can_connect(session.graph, "c", "output", "a", "input")


def test_unknown_port_is_refused(chain):
    assert "no output 'nope'" in can_connect(chain.graph, "src", "nope", "gray", "input")
    assert "no input 'nope'" in can_connect(chain.graph, "src", "output", "gray", "nope")


def test_wrong_direction_is_refused(chain):
    # 'input' is an input on both sides, so neither is an output.
    assert can_connect(chain.graph, "gray", "input", "edges", "input") is not None


def test_missing_node_is_refused(chain):
    assert "no such node: ghost" in can_connect(
        chain.graph, "ghost", "output", "gray", "input"
    )


def test_connect_command_refuses_the_same_cases(session):
    session.add_node("gaussian_blur", node_id="a")
    session.add_node("canny", node_id="b")
    session.connect("a", "output", "b", "input")
    with pytest.raises(EditError, match="cycle"):
        session.connect("b", "output", "a", "input")


def test_an_occupied_input_is_refused_by_the_command_too(chain):
    chain.add_node("imread", node_id="other", params={"path": "y.png"})
    with pytest.raises(EditError, match="already connected"):
        chain.connect("other", "output", "gray", "input")


def test_a_refused_edit_leaves_no_history(chain):
    before = chain.history()
    with pytest.raises(EditError):
        chain.connect("src", "output", "gray", "input")  # already connected
    assert chain.history() == before


# -- undo and redo --------------------------------------------------------


def test_undo_add_node(session):
    session.add_node("canny")
    session.undo()
    assert session.graph.nodes == {}


def test_redo_add_node(session):
    session.add_node("canny")
    session.undo()
    session.redo()
    assert set(session.graph.nodes) == {"canny"}


def test_undo_remove_node_restores_its_edges(chain):
    """Deleting a block and undoing must not silently lose its wiring."""
    assert len(chain.graph.edges) == 2
    chain.remove_node("gray")
    assert len(chain.graph.edges) == 0
    chain.undo()
    assert set(chain.graph.nodes) == {"src", "gray", "edges"}
    assert {str(e) for e in chain.graph.edges} == {
        "src.output -> gray.input",
        "gray.output -> edges.input",
    }


def test_undo_remove_node_restores_its_position(chain):
    chain.move_node("gray", (10.0, 20.0))
    chain.remove_node("gray")
    chain.undo()
    assert chain.graph.layout["gray"] == (10.0, 20.0)


def test_undo_connect(chain):
    chain.add_node("gaussian_blur", node_id="blur")
    chain.connect("gray", "output", "blur", "input")
    chain.undo()
    assert "gray.output -> blur.input" not in {str(e) for e in chain.graph.edges}


def test_undo_disconnect(chain):
    chain.disconnect("src", "output", "gray", "input")
    assert len(chain.graph.edges) == 1
    chain.undo()
    assert len(chain.graph.edges) == 2


def test_disconnecting_something_absent_is_refused(chain):
    with pytest.raises(EditError, match="no such connection"):
        chain.disconnect("src", "output", "edges", "input")


def test_undo_set_param(chain):
    chain.set_param("edges", "threshold1", 99)
    chain.end_gesture()
    chain.set_param("edges", "threshold2", 111)
    chain.undo()
    assert chain.graph.nodes["edges"].params == {"threshold1": 99}


def test_undo_set_param_removes_a_key_that_did_not_exist(chain):
    chain.set_param("edges", "threshold1", 5)
    chain.undo()
    assert "threshold1" not in chain.graph.nodes["edges"].params


def test_undo_stack_order_is_respected(session):
    session.add_node("canny", node_id="a")
    session.add_node("canny", node_id="b")
    session.add_node("canny", node_id="c")
    session.undo()
    assert set(session.graph.nodes) == {"a", "b"}
    session.undo()
    assert set(session.graph.nodes) == {"a"}
    session.redo()
    session.redo()
    assert set(session.graph.nodes) == {"a", "b", "c"}


def test_a_new_edit_clears_the_redo_stack(session):
    session.add_node("canny", node_id="a")
    session.undo()
    assert session.can_redo
    session.add_node("gaussian_blur", node_id="b")
    assert not session.can_redo


def test_undo_on_an_empty_history_is_harmless(session):
    assert session.undo() is None
    assert session.redo() is None


def test_descriptions_are_human_readable(chain):
    chain.add_node("gaussian_blur", node_id="blur")
    assert chain.undo_description == "add gaussian_blur"
    chain.undo()
    assert chain.redo_description == "add gaussian_blur"


# -- gesture merging ------------------------------------------------------


def test_a_drag_collapses_into_one_undo_step(session):
    """A drag arrives as a stream of positions, not one move."""
    session.add_node("canny", node_id="a", position=(0.0, 0.0))
    for x in range(1, 20):
        session.move_node("a", (float(x), 0.0))
    assert session.graph.layout["a"] == (19.0, 0.0)
    session.undo()
    assert session.graph.layout["a"] == (0.0, 0.0)


def test_ending_a_gesture_starts_a_new_undo_step(session):
    session.add_node("canny", node_id="a", position=(0.0, 0.0))
    session.move_node("a", (10.0, 0.0))
    session.end_gesture()
    session.move_node("a", (20.0, 0.0))
    session.undo()
    assert session.graph.layout["a"] == (10.0, 0.0)


def test_moves_of_different_nodes_do_not_merge(session):
    session.add_node("canny", node_id="a", position=(0.0, 0.0))
    session.add_node("canny", node_id="b", position=(0.0, 0.0))
    session.move_node("a", (5.0, 0.0))
    session.move_node("b", (5.0, 0.0))
    session.undo()
    assert session.graph.layout["b"] == (0.0, 0.0)
    assert session.graph.layout["a"] == (5.0, 0.0)


def test_typing_in_a_parameter_collapses_into_one_step(chain):
    for value in (1, 12, 123):
        chain.set_param("edges", "threshold1", value)
    chain.undo()
    assert "threshold1" not in chain.graph.nodes["edges"].params


def test_params_of_different_names_do_not_merge(chain):
    chain.set_param("edges", "threshold1", 5)
    chain.set_param("edges", "threshold2", 6)
    chain.undo()
    assert chain.graph.nodes["edges"].params == {"threshold1": 5}


# -- renaming -------------------------------------------------------------


def test_rename_updates_edges(chain):
    chain.rename_node("gray", "luminance")
    assert "luminance" in chain.graph.nodes
    assert "gray" not in chain.graph.nodes
    assert {str(e) for e in chain.graph.edges} == {
        "src.output -> luminance.input",
        "luminance.output -> edges.input",
    }


def test_rename_updates_the_layout(chain):
    chain.move_node("gray", (7.0, 8.0))
    chain.end_gesture()
    chain.rename_node("gray", "luminance")
    assert chain.graph.layout["luminance"] == (7.0, 8.0)
    assert "gray" not in chain.graph.layout


def test_rename_preserves_insertion_order(chain):
    """A rename should not reshuffle the file and touch every line in the diff."""
    before = list(chain.graph.nodes)
    chain.rename_node("gray", "luminance")
    after = list(chain.graph.nodes)
    assert after == ["luminance" if n == "gray" else n for n in before]


def test_rename_keeps_the_node_id_field_consistent(chain):
    chain.rename_node("gray", "luminance")
    assert chain.graph.nodes["luminance"].id == "luminance"


def test_rename_to_an_existing_name_is_refused(chain):
    with pytest.raises(EditError, match="already exists"):
        chain.rename_node("gray", "edges")


def test_rename_to_an_invalid_id_is_refused(chain):
    with pytest.raises(EditError, match="not a valid node id"):
        chain.rename_node("gray", "9lives")


def test_undo_rename(chain):
    chain.rename_node("gray", "luminance")
    chain.undo()
    assert "gray" in chain.graph.nodes
    assert {str(e) for e in chain.graph.edges} == {
        "src.output -> gray.input",
        "gray.output -> edges.input",
    }


# -- modified tracking ----------------------------------------------------


def test_a_fresh_session_is_unmodified(session):
    assert not session.modified


def test_an_edit_marks_it_modified(session):
    session.add_node("canny")
    assert session.modified


def test_undoing_back_to_the_saved_state_clears_modified(session):
    """Otherwise a document stays dirty forever once touched."""
    session.add_node("canny")
    session.mark_saved()
    session.add_node("gaussian_blur")
    assert session.modified
    session.undo()
    assert not session.modified


def test_redoing_past_the_saved_state_marks_it_modified_again(session):
    session.add_node("canny")
    session.mark_saved()
    session.add_node("gaussian_blur")
    session.undo()
    session.redo()
    assert session.modified


def test_reset_clears_the_history(session):
    session.add_node("canny")
    session.reset(Graph(name="other"))
    assert not session.can_undo
    assert not session.modified
    assert session.graph.name == "other"


# -- the result stays a valid diagram -------------------------------------


def test_an_edited_graph_validates_and_round_trips(chain, tmp_path):
    chain.add_node("preview", node_id="show")
    chain.connect("edges", "output", "show", "input")
    chain.graph.validate()

    target = tmp_path / "d.yaml"
    io.dump(chain.graph, target)
    reloaded = io.load(target)
    assert set(reloaded.nodes) == set(chain.graph.nodes)
    assert [str(e) for e in reloaded.edges] == [str(e) for e in chain.graph.edges]


def test_undo_restores_a_graph_that_still_validates(chain):
    chain.remove_node("gray")
    chain.undo()
    chain.graph.validate()


def test_a_built_graph_is_executable(chain):
    from pydiedi.core.executor import Executor

    chain.graph.nodes["src"].params["path"] = str(
        __import__("pathlib").Path(__file__).parent / "fixtures" / "checkerboard.png"
    )
    with Executor(chain.graph) as executor:
        result = executor.step()
    assert result.value("edges", "output") is not None


# -- edge identity --------------------------------------------------------


def test_edges_are_equal_regardless_of_which_line_they_came_from():
    """The line number is metadata about where an edge was written down.

    Including it in equality made Disconnect unable to find an edge that had
    been loaded from a file, because the command constructs its target without
    a line number.
    """
    from pydiedi.core.graph import Edge

    from_file = Edge("a", "output", "b", "input", line=7)
    from_editor = Edge("a", "output", "b", "input")
    assert from_file == from_editor
    assert hash(from_file) == hash(from_editor)
    assert from_editor in [from_file]


def test_disconnecting_an_edge_that_came_from_a_file(real_blocks, tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "version: 1\n"
        "nodes:\n"
        "  a: {block: gaussian_blur}\n"
        "  b: {block: canny}\n"
        "edges:\n"
        "  - a.output -> b.input\n",
        encoding="utf-8",
    )
    session = EditSession(io.load(path))
    assert session.graph.edges[0].line is not None
    session.disconnect("a", "output", "b", "input")
    assert session.graph.edges == []
