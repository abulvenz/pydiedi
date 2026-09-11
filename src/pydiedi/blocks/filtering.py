"""Smoothing, thresholding, edge detection.

The parameter validation here is the part worth carrying over from flodiedi:
its generated plugins passed values straight into OpenCV, so an even kernel
size or a 3-channel Otsu threshold aborted the process from inside the C++
library with no indication of which block was at fault.
"""

from __future__ import annotations

from enum import Enum

import cv2

from ..core import Mat, block


class ThresholdType(Enum):
    binary = cv2.THRESH_BINARY
    binary_inv = cv2.THRESH_BINARY_INV
    trunc = cv2.THRESH_TRUNC
    to_zero = cv2.THRESH_TOZERO
    to_zero_inv = cv2.THRESH_TOZERO_INV
    otsu = cv2.THRESH_OTSU | cv2.THRESH_BINARY
    triangle = cv2.THRESH_TRIANGLE | cv2.THRESH_BINARY


_AUTO_THRESHOLD = {ThresholdType.otsu, ThresholdType.triangle}


@block(category="filtering")
def gaussian_blur(input: Mat, kernel_size: int = 5, sigma: float = 0.0) -> Mat:
    """Blur an image with a Gaussian kernel.

    ``sigma = 0`` lets OpenCV derive the deviation from the kernel size.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}")
    return cv2.GaussianBlur(input, (kernel_size, kernel_size), sigma)


@block(category="filtering")
def threshold(
    input: Mat,
    level: float = 128,
    max_value: float = 255,
    type: ThresholdType = ThresholdType.binary,
) -> Mat:
    """Binarise an image at a fixed level.

    ``otsu`` and ``triangle`` derive the level from the histogram and ignore
    ``level``; both require a single-channel image.
    """
    if type in _AUTO_THRESHOLD and input.ndim > 2:
        raise ValueError(
            f"{type.name} threshold needs a single-channel image, "
            f"got {input.shape[2]} channels. Insert a cvt_color block first."
        )
    return cv2.threshold(input, level, max_value, type.value)[1]


@block(category="filtering")
def canny(
    input: Mat,
    threshold1: float = 50,
    threshold2: float = 150,
    aperture_size: int = 3,
) -> Mat:
    """Detect edges with the Canny operator."""
    if aperture_size not in (3, 5, 7):
        raise ValueError(f"aperture_size must be 3, 5 or 7, got {aperture_size}")
    return cv2.Canny(input, threshold1, threshold2, apertureSize=aperture_size)
