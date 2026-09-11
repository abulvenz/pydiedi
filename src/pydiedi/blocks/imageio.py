"""Reading images from disk.

Compare with flodiedi's ``imreadplugin``: 67 lines of hand-written logic inside
a generated 220-line class plus a 102-line CMakeLists, all to call one OpenCV
function.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2

from ..core import Mat, block


class ColorMode(Enum):
    """How to decode the file. Mirrors ``cv2.IMREAD_*``."""

    color = cv2.IMREAD_COLOR
    grayscale = cv2.IMREAD_GRAYSCALE
    unchanged = cv2.IMREAD_UNCHANGED


@block(category="imageio")
def imread(path: Path, mode: ColorMode = ColorMode.color) -> Mat:
    """Load an image from a file.

    Raises rather than returning an empty matrix: flodiedi propagated empty
    ``Mat``s downstream, where they surfaced as confusing failures in whichever
    block happened to touch them first.
    """
    image = cv2.imread(str(path), mode.value)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


@block(category="imageio")
def imwrite(input: Mat, path: Path) -> None:
    """Write an image to a file, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), input):
        raise OSError(f"cannot write image: {path} (unsupported extension?)")
