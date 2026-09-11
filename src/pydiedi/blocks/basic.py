"""Element-wise arithmetic.

flodiedi's ``Basic`` category held 18 generated plugins of this kind, each
about 320 lines of scaffolding around one OpenCV call.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core import Mat, block


def _require_same_shape(a: Mat, b: Mat, what: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{what} needs images of the same shape, got {a.shape} and {b.shape}. "
            f"Insert a resize or a cvt_color block."
        )


@block(category="basic")
def abs_diff(a: Mat, b: Mat) -> Mat:
    """Absolute per-element difference of two images."""
    _require_same_shape(a, b, "abs_diff")
    return cv2.absdiff(a, b)


@block(category="basic")
def add(a: Mat, b: Mat) -> Mat:
    """Saturating per-element sum."""
    _require_same_shape(a, b, "add")
    return cv2.add(a, b)


@block(category="basic")
def subtract(a: Mat, b: Mat) -> Mat:
    """Saturating per-element difference."""
    _require_same_shape(a, b, "subtract")
    return cv2.subtract(a, b)


@block(category="basic")
def multiply(a: Mat, b: Mat, scale: float = 1.0) -> Mat:
    """Per-element product, optionally scaled."""
    _require_same_shape(a, b, "multiply")
    return cv2.multiply(a, b, scale=scale)


@block(category="basic")
def add_weighted(a: Mat, b: Mat, alpha: float = 0.5, gamma: float = 0.0) -> Mat:
    """Blend two images: ``a * alpha + b * (1 - alpha) + gamma``."""
    _require_same_shape(a, b, "add_weighted")
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, gamma)


@block(category="basic")
def bitwise_and(input: Mat, mask: Mat) -> Mat:
    """Keep ``input`` where ``mask`` is non-zero."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be single-channel, got shape {mask.shape}")
    return cv2.bitwise_and(input, input, mask=mask)


@block(category="basic")
def convert_scale_abs(input: Mat, alpha: float = 1.0, beta: float = 0.0) -> Mat:
    """Scale, take the absolute value and convert to 8-bit."""
    return cv2.convertScaleAbs(input, alpha=alpha, beta=beta)


@block(category="basic")
def count_non_zero(input: Mat) -> int:
    """Number of non-zero elements. Useful as a cheap motion measure."""
    if input.ndim != 2:
        raise ValueError(
            f"count_non_zero needs a single-channel image, got shape {input.shape}"
        )
    return int(cv2.countNonZero(input))


@block(category="basic")
def mean_value(input: Mat) -> float:
    """Mean over all channels and pixels."""
    return float(np.mean(input))
