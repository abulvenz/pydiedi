"""Display blocks.

These return a :class:`~pydiedi.core.types.Preview` -- a description of what
to show -- and never touch a GUI toolkit. That is the whole reason a headless
run, a Qt window and a future web renderer can share one block library.

flodiedi took the opposite route: ``displayplugin`` and ``showimageplugin``
held a ``QGraphicsPixmapItem`` and called ``setPixmap()`` from inside
``run()``, which executes on the worker thread. Qt4 tolerated it; Qt5 and Qt6
abort on cross-thread painting.
"""

from __future__ import annotations

import cv2

from ..core import Mat, Preview, block


@block(category="display")
def preview(input: Mat, title: str = "") -> Preview:
    """Show an image in the renderer's preview area."""
    return Preview(image=input, title=title)


@block(category="display")
def side_by_side(left: Mat, right: Mat, title: str = "") -> Preview:
    """Show two images next to each other, scaled to a common height."""
    lh, rh = left.shape[0], right.shape[0]
    height = max(lh, rh)
    scaled = [
        img
        if img.shape[0] == height
        else cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height))
        for img in (left, right)
    ]
    scaled = [_as_bgr(img) for img in scaled]
    import numpy as np

    return Preview(image=np.hstack(scaled), title=title)


def _as_bgr(image: Mat) -> Mat:
    """Promote a single-channel image so it can be stacked with a colour one."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
