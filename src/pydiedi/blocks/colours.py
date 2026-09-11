"""Colour space conversion."""

from __future__ import annotations

from enum import Enum

import cv2

from ..core import Mat, block


class ColorCode(Enum):
    """A subset of ``cv2.COLOR_*``, named as in flodiedi's cvtColor block."""

    bgr2gray = cv2.COLOR_BGR2GRAY
    gray2bgr = cv2.COLOR_GRAY2BGR
    bgr2rgb = cv2.COLOR_BGR2RGB
    rgb2bgr = cv2.COLOR_RGB2BGR
    bgr2hsv = cv2.COLOR_BGR2HSV
    hsv2bgr = cv2.COLOR_HSV2BGR
    bgr2lab = cv2.COLOR_BGR2Lab
    lab2bgr = cv2.COLOR_Lab2BGR


@block(category="colours")
def cvt_color(input: Mat, code: ColorCode = ColorCode.bgr2gray) -> Mat:
    """Convert an image between colour spaces."""
    return cv2.cvtColor(input, code.value)
