"""Describing a value that sits on a port.

flodiedi had a genuinely good feature here: clicking a connector showed the
value currently on it. Its ``CustomToolTip`` polled the running diagram every
30 ms, reading the live value under ``prLock.tryLock(1)`` -- one of the few
places in that codebase where cross-thread access was handled correctly.

The mechanism is different here, because the value is summarised on the worker
thread and only the summary crosses to the GUI. A 4K frame stays where it was
produced; what travels is a line of text and, for an image, a thumbnail a few
kilobytes in size.

Summarising lives in ``core`` and uses numpy only, so it stays free of both Qt
and OpenCV: a web renderer gets the same :class:`PortValue` and decides for
itself how to present it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .types import Mat, Preview

__all__ = ["PortValue", "summarise", "thumbnail", "inspect_value"]

THUMBNAIL_MAX = 160
"""Longest edge of a generated thumbnail, in pixels."""


@dataclass(slots=True)
class PortValue:
    """What is currently on one port, in a form safe to send anywhere."""

    node_id: str
    port: str
    summary: str
    """One line, e.g. ``Mat 96x96 uint8, 0..255`` or ``float 128.0``."""
    thumbnail: Mat | None = None
    """A small copy for image-like values, or ``None``."""
    iteration: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.node_id, self.port)

    def __str__(self) -> str:
        return f"{self.node_id}.{self.port} = {self.summary}"


def summarise(value: Any) -> str:
    """A short, honest description of a value.

    Honest meaning: it says what is actually there. ``None`` and an empty array
    are distinguishable, and an image reports its dtype and range rather than
    just its shape -- the usual reason a pipeline looks black is a float image
    in 0..1 being read as 0..255, and the range says so at a glance.
    """
    if value is None:
        return "none"

    if isinstance(value, Preview):
        inner = summarise(value.image)
        return f"Preview({inner})" if value.image is not None else "Preview(empty)"

    if isinstance(value, np.ndarray):
        return _summarise_array(value)

    if isinstance(value, bool):
        return f"bool {value}"
    if isinstance(value, (int, np.integer)):
        return f"int {value}"
    if isinstance(value, (float, np.floating)):
        return f"float {value:g}"
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, Path):
        return f"Path {value}"
    if isinstance(value, str):
        shown = value if len(value) <= 60 else value[:57] + "..."
        return f"str {shown!r}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} of {len(value)}"

    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{type(value).__name__} {text}"


def _summarise_array(array: np.ndarray) -> str:
    shape = "x".join(str(n) for n in array.shape)
    if array.size == 0:
        return f"Mat {shape} {array.dtype} (empty)"
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    if finite.size == 0:
        return f"Mat {shape} {array.dtype} (no finite values)"
    low, high = finite.min(), finite.max()
    fmt = "{:g}" if array.dtype.kind == "f" else "{}"
    return f"Mat {shape} {array.dtype}, {fmt.format(low)}..{fmt.format(high)}"


def thumbnail(value: Any, max_edge: int = THUMBNAIL_MAX) -> Mat | None:
    """A small copy of an image-like value, or ``None``.

    Subsampled by striding rather than resampled: this module must not import
    OpenCV, and for a thumbnail the difference is not worth a dependency. The
    result is always a copy, so the caller may hold it while the worker moves
    on to the next frame.
    """
    image = value.image if isinstance(value, Preview) else value
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        return None
    if image.size == 0:
        return None
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return None

    height, width = image.shape[:2]
    step = max(1, (max(height, width) + max_edge - 1) // max_edge)
    return np.ascontiguousarray(image[::step, ::step])


def inspect_value(
    node_id: str, port: str, value: Any, iteration: int = 0, with_thumbnail: bool = True
) -> PortValue:
    """Package one port's value for display."""
    return PortValue(
        node_id=node_id,
        port=port,
        summary=summarise(value),
        thumbnail=thumbnail(value) if with_thumbnail else None,
        iteration=iteration,
    )
