"""The phase 0 acceptance criteria, as tests.

These exercise the real block library and the real CLI, so they are the ones
that would notice if the premise stopped holding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pydiedi.cli import main
from pydiedi.core import io, registry
from pydiedi.core.executor import Executor

FIXTURES = Path(__file__).parent / "fixtures"
BASIC = FIXTURES / "basic.yaml"
REPO = Path(__file__).resolve().parent.parent


# -- criterion 1: ports come from reflection, with no declaration file ----


def test_every_block_is_discovered_by_import_alone(real_blocks):
    blocks = registry.all_blocks()
    assert len(blocks) >= 8
    for name, spec in blocks.items():
        assert spec.name == name
        assert spec.category
        assert spec.doc, f"block {name!r} has no docstring"


def test_no_declaration_files_exist():
    """The registry is derived, not maintained.

    flodiedi needed functions.fdf plus one CMakeLists.txt per plugin -- 172 of
    them -- to describe what a signature already says.
    """
    src = REPO / "src"
    for pattern in ("*.fdf", "CMakeLists.txt", "*.pro", "*.pri"):
        assert not list(src.rglob(pattern)), f"unexpected {pattern} under src/"


def test_blocks_command_lists_ports_types_and_defaults(capsys, real_blocks):
    assert main(["blocks"]) == 0
    out = capsys.readouterr().out
    assert "threshold(input: Mat, level: float = 128" in out
    assert "-> Mat" in out
    assert "imwrite(input: Mat, path: Path) -> none" in out
    assert "categories" in out


def test_blocks_verbose_shows_enum_choices(capsys, real_blocks):
    assert main(["-v", "blocks"]) == 0
    out = capsys.readouterr().out
    assert "one of:" in out
    assert "bgr2gray" in out


# -- criterion 2: a diagram with a branch runs and produces an image ------


def test_fixture_diagram_runs_and_writes_a_verifiable_image(real_blocks, tmp_path):
    graph = io.load(BASIC)
    graph.nodes["save"].params["path"] = str(tmp_path / "result.png")
    result = Executor(graph).step()

    written = tmp_path / "result.png"
    assert written.is_file()

    import cv2

    image = cv2.imread(str(written), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.shape == (256, 256)
    # A binary_inv threshold of a checkerboard: both values present, nothing else.
    assert set(np.unique(image)) <= {0, 255}
    assert len(set(np.unique(image))) == 2

    assert [p.title for p in result.previews] == ["mask_vs_edges"]
    preview = result.previews[0].image
    assert preview is not None
    assert preview.shape == (256, 512, 3), "side_by_side should double the width"


def test_branch_really_diverges(real_blocks):
    """The two branches off 'gray' must produce different images.

    If fan-out were broken -- one branch overwriting the other's input -- this
    is what would catch it.
    """
    graph = io.load(BASIC)
    result = Executor(graph).step()
    mask = result.value("mask", "output")
    edges = result.value("edges", "output")
    assert mask.shape == edges.shape
    assert not np.array_equal(mask, edges)


def test_execution_order_respects_dependencies(real_blocks):
    graph = io.load(BASIC)
    order = Executor(graph).order
    assert order.index("source") < order.index("gray")
    assert order.index("gray") < order.index("blur")
    assert order.index("gray") < order.index("edges")
    assert order.index("blur") < order.index("mask")
    assert order.index("mask") < order.index("compare")
    assert order.index("edges") < order.index("compare")


def test_run_command_succeeds(capsys, real_blocks):
    assert main(["run", str(BASIC)]) == 0
    assert "mask_vs_edges" in capsys.readouterr().out


def test_check_command_reports_the_order(capsys, real_blocks):
    assert main(["check", str(BASIC)]) == 0
    out = capsys.readouterr().out
    assert "ok -- 7 nodes, 7 edges" in out
    assert "execution order: source -> gray" in out


def test_run_saves_previews_on_request(tmp_path, capsys, real_blocks):
    assert main(["run", str(BASIC), "--save-previews", str(tmp_path)]) == 0
    assert (tmp_path / "mask_vs_edges.png").is_file()


def test_multiple_iterations_run(capsys, real_blocks):
    assert main(["run", str(BASIC), "-n", "3"]) == 0


def test_paths_resolve_relative_to_the_diagram(real_blocks, tmp_path, monkeypatch):
    """Running from an unrelated working directory must still work."""
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(BASIC)]) == 0


# -- criteria 4 and 5: failures are named, not swallowed ------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "d.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_block_exits_with_a_message_naming_it(tmp_path, capsys, real_blocks):
    path = _write(tmp_path, "version: 1\nnodes:\n  a: {block: gaussian_blurr}\n")
    assert main(["check", str(path)]) == 2
    err = capsys.readouterr().err
    assert "unknown block 'gaussian_blurr'" in err
    assert "Did you mean 'gaussian_blur'" in err


def test_cycle_exits_with_a_message_naming_the_cycle(tmp_path, capsys, real_blocks):
    path = _write(
        tmp_path,
        "version: 1\n"
        "nodes:\n"
        "  a: {block: gaussian_blur}\n"
        "  b: {block: canny}\n"
        "edges:\n"
        "  - a.output -> b.input\n"
        "  - b.output -> a.input\n",
    )
    assert main(["check", str(path)]) == 2
    err = capsys.readouterr().err
    assert "cycle" in err
    assert "a -> b -> a" in err or "b -> a -> b" in err


def test_syntax_error_names_the_line(tmp_path, capsys, real_blocks):
    path = _write(tmp_path, "version: 1\nnodes:\n  a: {params: {x: 1}}\n")
    assert main(["check", str(path)]) == 2
    assert ":3:" in capsys.readouterr().err


def test_failing_block_exits_three(tmp_path, capsys, real_blocks):
    path = _write(
        tmp_path,
        "version: 1\nnodes:\n  a: {block: imread, params: {path: nope.png}}\n",
    )
    assert main(["run", str(path)]) == 3
    err = capsys.readouterr().err
    assert "node 'a'" in err
    assert "block 'imread'" in err


def test_bad_kernel_size_is_reported_with_the_node(tmp_path, capsys, real_blocks):
    path = _write(
        tmp_path,
        "version: 1\n"
        "nodes:\n"
        "  src: {block: imread, params: {path: "
        + str(FIXTURES / "checkerboard.png")
        + "}}\n"
        "  b: {block: gaussian_blur, params: {kernel_size: 4}}\n"
        "edges:\n"
        "  - src.output -> b.input\n",
    )
    assert main(["run", str(path)]) == 3
    err = capsys.readouterr().err
    assert "node 'b'" in err
    assert "odd number" in err


def test_otsu_on_a_colour_image_is_refused_with_a_hint(tmp_path, capsys, real_blocks):
    path = _write(
        tmp_path,
        "version: 1\n"
        "nodes:\n"
        "  src: {block: imread, params: {path: "
        + str(FIXTURES / "checkerboard.png")
        + "}}\n"
        "  t: {block: threshold, params: {type: otsu}}\n"
        "edges:\n"
        "  - src.output -> t.input\n",
    )
    assert main(["run", str(path)]) == 3
    assert "cvt_color" in capsys.readouterr().err  # the hint names the fix


# -- the installed entry point actually works ----------------------------


@pytest.mark.parametrize("args", [["blocks"], ["check", str(BASIC)], ["run", str(BASIC)]])
def test_console_script_runs_in_a_subprocess(args: list[str]):
    proc = subprocess.run(
        [sys.executable, "-m", "pydiedi.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# -- phase 1: stateful sources and tolerant execution --------------------

MOTION = FIXTURES / "motion.yaml"


def test_motion_fixture_validates(capsys, real_blocks):
    assert main(["check", str(MOTION)]) == 0
    out = capsys.readouterr().out
    assert "ok -- 7 nodes, 7 edges" in out
    assert "video -> prev -> diff" in out


def test_motion_fixture_runs_to_the_end_of_the_video(capsys, real_blocks):
    """-n 0 must terminate on its own: the source stops the run."""
    assert main(["run", str(MOTION), "-n", "0"]) == 0
    captured = capsys.readouterr()
    assert "motion_mask" in captured.out
    assert "no more frames" in captured.err


def test_motion_detects_movement(real_blocks):
    graph = io.load(MOTION)
    with Executor(graph) as executor:
        first = executor.step(0)
        second = executor.step(1)
    assert first.value("motion", "output") == 0, "a frame against itself is still"
    assert second.value("motion", "output") > 0, "the square moved"


def test_video_paths_resolve_relative_to_the_diagram(real_blocks, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(MOTION), "-n", "2"]) == 0


def test_stateful_nodes_are_listed_in_verbose_mode(capsys, real_blocks):
    assert main(["-v", "run", str(MOTION), "-n", "1"]) == 0
    assert "stateful: video, prev" in capsys.readouterr().err


def test_on_error_skip_keeps_independent_branches_running(
    tmp_path, capsys, real_blocks
):
    """One broken branch must not stop the other."""
    diagram = tmp_path / "d.yaml"
    diagram.write_text(
        "version: 1\n"
        "nodes:\n"
        f"  good: {{block: imread, params: {{path: {FIXTURES / 'checkerboard.png'}}}}}\n"
        "  bad:  {block: imread, params: {path: /nonexistent/x.png}}\n"
        "  edge: {block: canny}\n"
        "  show: {block: preview, params: {title: from_bad}}\n"
        "edges:\n"
        "  - good.output -> edge.input\n"
        "  - bad.output -> show.input\n",
        encoding="utf-8",
    )
    assert main(["run", str(diagram), "--on-error", "skip"]) == 3
    err = capsys.readouterr().err
    assert "error in bad" in err
    assert "skipped: show" in err
    assert "error in good" not in err


def test_on_error_raise_is_the_default(tmp_path, capsys, real_blocks):
    diagram = tmp_path / "d.yaml"
    diagram.write_text(
        "version: 1\nnodes:\n  bad: {block: imread, params: {path: /nope.png}}\n",
        encoding="utf-8",
    )
    assert main(["run", str(diagram)]) == 3
    assert "node 'bad'" in capsys.readouterr().err


def test_executor_releases_sources_after_a_cli_run(real_blocks):
    """A camera or file must not stay claimed once the run is over."""
    graph = io.load(MOTION)
    executor = Executor(graph)
    executor.step()
    instance = executor._instances["video"]
    assert instance._capture is not None
    executor.close()
    assert instance._capture is None


def test_stateful_blocks_appear_in_the_block_list(capsys, real_blocks):
    assert main(["blocks"]) == 0
    out = capsys.readouterr().out
    assert "video_file(path: Path, loop: bool = False)" in out
    assert "camera(index: int = 0" in out
    assert "(image: Mat, index: int, fps: float)" in out
