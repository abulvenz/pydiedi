"""Summarising the value on a port.

flodiedi showed this when you clicked a connector, and it was one of its better
ideas. The difference here is where the work happens: the value is summarised
on the worker thread and only the summary crosses to the GUI, so a 4K frame
stays where it was produced.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from pydiedi.core.inspect import (
    THUMBNAIL_MAX,
    PortValue,
    inspect_value,
    summarise,
    thumbnail,
)
from pydiedi.core.types import Preview


class Mode(Enum):
    fast = 0


# -- summaries ------------------------------------------------------------


def test_none_is_distinguishable_from_an_empty_image():
    assert summarise(None) == "none"
    assert "empty" in summarise(np.zeros((0, 0), np.uint8))


def test_an_image_reports_shape_dtype_and_range():
    """The range is the point: a float image in 0..1 read as 0..255 looks
    black, and the summary says so at a glance."""
    image = np.full((96, 128), 200, np.uint8)
    assert summarise(image) == "Mat 96x128 uint8, 200..200"


def test_a_colour_image_reports_all_three_dimensions():
    assert summarise(np.zeros((4, 5, 3), np.uint8)).startswith("Mat 4x5x3 uint8")


def test_a_float_image_reports_its_actual_range():
    image = np.array([[0.0, 0.5]], np.float32)
    assert summarise(image) == "Mat 1x2 float32, 0..0.5"


def test_an_all_nan_image_says_so_rather_than_crashing():
    image = np.full((4, 4), np.nan, np.float32)
    assert "no finite values" in summarise(image)


def test_nan_does_not_poison_the_range():
    image = np.array([[0.0, np.nan, 10.0]], np.float32)
    assert summarise(image) == "Mat 1x3 float32, 0..10"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "bool True"),
        (7, "int 7"),
        (2.5, "float 2.5"),
        (Mode.fast, "Mode.fast"),
        (Path("a/b.png"), "Path a/b.png"),
        ("hello", "str 'hello'"),
        ([1, 2, 3], "list of 3"),
        ((1, 2), "tuple of 2"),
    ],
)
def test_scalars_and_containers(value, expected):
    assert summarise(value) == expected


def test_numpy_scalars_read_like_python_ones():
    assert summarise(np.int32(7)) == "int 7"
    assert summarise(np.float64(2.5)) == "float 2.5"


def test_a_long_string_is_shortened():
    summary = summarise("x" * 200)
    assert len(summary) < 80
    assert "..." in summary  # truncated inside the quotes: str 'xxx...'


def test_a_preview_describes_the_image_inside_it():
    assert summarise(Preview(image=np.zeros((8, 8), np.uint8))) == (
        "Preview(Mat 8x8 uint8, 0..0)"
    )
    assert summarise(Preview()) == "Preview(empty)"


def test_an_unknown_object_still_gets_a_summary():
    class Thing:
        def __repr__(self):
            return "<a thing>"

    assert summarise(Thing()) == "Thing <a thing>"


# -- thumbnails -----------------------------------------------------------


def test_a_large_image_is_reduced():
    small = thumbnail(np.zeros((1080, 1920), np.uint8))
    assert small is not None
    assert max(small.shape[:2]) <= THUMBNAIL_MAX


def test_a_small_image_is_left_alone():
    image = np.zeros((20, 30), np.uint8)
    assert thumbnail(image).shape == image.shape


def test_a_colour_image_keeps_its_channels():
    small = thumbnail(np.zeros((900, 900, 3), np.uint8))
    assert small.shape[2] == 3


def test_the_thumbnail_is_a_copy():
    """The caller holds it while the worker moves on to the next frame."""
    image = np.full((400, 400), 200, np.uint8)
    small = thumbnail(image)
    image[:] = 0
    assert small.max() == 200


def test_the_thumbnail_is_contiguous():
    """QImage needs tightly packed rows; striding alone does not give them."""
    assert thumbnail(np.zeros((900, 900), np.uint8)).flags["C_CONTIGUOUS"]


def test_a_preview_is_unwrapped():
    assert thumbnail(Preview(image=np.zeros((500, 500), np.uint8))) is not None


@pytest.mark.parametrize(
    "value",
    [None, 7, "text", np.zeros((4,), np.uint8), np.zeros((4, 4, 5), np.uint8)],
)
def test_things_that_are_not_images_have_no_thumbnail(value):
    assert thumbnail(value) is None


def test_an_empty_image_has_no_thumbnail():
    assert thumbnail(np.zeros((0, 10), np.uint8)) is None


# -- the packaged value ---------------------------------------------------


def test_inspect_value_packages_everything():
    value = inspect_value("gray", "output", np.zeros((8, 8), np.uint8), iteration=3)
    assert value.node_id == "gray"
    assert value.port == "output"
    assert value.key == ("gray", "output")
    assert value.iteration == 3
    assert value.thumbnail is not None
    assert "Mat 8x8" in value.summary


def test_a_thumbnail_can_be_skipped():
    value = inspect_value("a", "b", np.zeros((8, 8), np.uint8), with_thumbnail=False)
    assert value.thumbnail is None


def test_port_value_reads_well():
    assert str(PortValue("gray", "output", "int 5")) == "gray.output = int 5"
