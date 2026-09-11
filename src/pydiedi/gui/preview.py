"""Turning a :class:`~pydiedi.core.types.Preview` into something on screen.

The conversion lives here, in the GUI package, and not in the block library --
that separation is what lets a web renderer send a PNG over a socket from the
same ``Preview`` object.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..core.types import Preview

__all__ = ["to_qimage", "PreviewPanel", "PreviewView"]


def to_qimage(array: np.ndarray) -> QImage:
    """Convert an OpenCV image to a QImage.

    The result always owns its memory. ``QImage`` does not copy the buffer it
    is constructed from, so returning one that points into a numpy array owned
    by the worker thread would show torn frames at best and crash once the
    array was freed. The ``copy()`` here is the price of that safety, and at
    preview sizes it does not matter.
    """
    if array is None:
        raise ValueError("cannot convert None to an image")
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")

    data = _as_uint8(array)
    # QImage needs tightly packed rows; a slice or a transpose is not.
    data = np.ascontiguousarray(data)
    height, width = data.shape[:2]

    if data.ndim == 2:
        image = QImage(data.data, width, height, data.strides[0], QImage.Format_Grayscale8)
    elif data.shape[2] == 3:
        # OpenCV is BGR and Qt has a matching format, so no channel swap.
        image = QImage(data.data, width, height, data.strides[0], QImage.Format_BGR888)
    elif data.shape[2] == 4:
        image = QImage(data.data, width, height, data.strides[0], QImage.Format_ARGB32)
    else:
        raise ValueError(
            f"cannot display an image with {data.shape[2]} channels; "
            f"expected 1, 3 or 4"
        )
    return image.copy()


def _as_uint8(array: np.ndarray) -> np.ndarray:
    """Map any numeric image onto 8 bits, so it can be displayed.

    Float images from OpenCV are conventionally in 0..1, integer ones in
    0..255, but a gradient or a distance transform is in neither. Anything
    outside the expected range is scaled by its own extent rather than clipped,
    because a black rectangle is a worse answer than a rescaled one.
    """
    if array.dtype == np.uint8:
        return array
    if array.dtype == bool:
        return array.astype(np.uint8) * 255

    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    if finite.size == 0:
        return np.zeros(array.shape, np.uint8)

    low, high = float(finite.min()), float(finite.max())
    if array.dtype.kind == "f" and 0.0 <= low and high <= 1.0:
        return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if 0 <= low and high <= 255:
        return np.clip(array, 0, 255).astype(np.uint8)
    if high == low:
        return np.zeros(array.shape, np.uint8)
    return ((np.nan_to_num(array) - low) * (255.0 / (high - low))).astype(np.uint8)


class PreviewView(QWidget):
    """One titled image, scaled to fit while keeping its aspect ratio."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None

        self._title = QLabel(title)
        self._title.setStyleSheet("color: #aab; font-size: 11px;")
        self._image = QLabel("no image")
        self._image.setAlignment(Qt.AlignCenter)
        self._image.setMinimumSize(120, 90)
        self._image.setStyleSheet("background: #1b1d22; color: #666;")
        self._image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._image, 1)

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def set_image(self, array: np.ndarray | None) -> None:
        if array is None:
            self._pixmap = None
            self._image.setText("no image")
            return
        try:
            self._pixmap = QPixmap.fromImage(to_qimage(array))
        except ValueError as exc:
            self._pixmap = None
            self._image.setText(str(exc))
            return
        self._rescale()

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self._image.setPixmap(
            self._pixmap.scaled(
                self._image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class PreviewPanel(QWidget):
    """Shows whatever the last sweep produced, one view per preview block.

    Views are created on demand and reused, so a pipeline running at 30 fps
    does not build and destroy widgets thirty times a second.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._views: list[PreviewView] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(8)
        self._placeholder = QLabel("Run the diagram to see previews.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #667;")
        self._layout.addWidget(self._placeholder)

    def show_previews(self, previews: list[Preview]) -> None:
        self._placeholder.setVisible(not previews)
        while len(self._views) < len(previews):
            view = PreviewView()
            self._views.append(view)
            self._layout.addWidget(view, 1)
        for index, view in enumerate(self._views):
            if index < len(previews):
                preview = previews[index]
                view.set_title(preview.title or f"preview {index + 1}")
                view.set_image(preview.image)
                view.setVisible(True)
            else:
                view.setVisible(False)

    def clear(self) -> None:
        for view in self._views:
            view.setVisible(False)
        self._placeholder.setVisible(True)
