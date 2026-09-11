"""Live sources: camera and video file.

These are the first stateful blocks -- they hold a ``cv2.VideoCapture`` across
sweeps. Two things are handled deliberately, because flodiedi got both wrong:

* **The capture is released.** flodiedi's ``VideoFile`` opened a
  ``VideoCapture`` and never closed it, so the device stayed claimed until the
  process ended.
* **A changed input reopens the device.** Because ports are declared on
  ``__call__`` rather than ``__init__``, ``path`` can be edited or even driven
  by an edge; the block notices and reopens rather than being stuck with what
  it was constructed with.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2

from ..core import Mat, StopExecution, block


class Frame(NamedTuple):
    """One frame plus the metadata a pipeline usually wants next to it."""

    image: Mat
    index: int
    fps: float


class EndOfVideo(StopExecution):
    """A video file ran out of frames and looping is off.

    A StopExecution, not an error: reaching the end of a file is the normal
    way a ``pydiedi run -n 0`` over a video terminates."""


@block(category="sources")
class VideoFile:
    """Read a video file one frame per sweep.

    With ``loop`` off, running past the last frame raises, which ends a
    ``pydiedi run -n 0``. With ``loop`` on, it starts over.
    """

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._opened: Path | None = None
        self._index = 0

    def __call__(self, path: Path, loop: bool = False) -> Frame:
        capture = self._ensure_open(path)
        ok, image = capture.read()
        if not ok:
            if not loop:
                raise EndOfVideo(
                    f"{path} has no more frames after {self._index}; "
                    f"set loop: true to start over"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._index = 0
            ok, image = capture.read()
            if not ok:
                raise EndOfVideo(f"{path} yielded no frames at all")

        self._index += 1
        return Frame(
            image=image,
            index=self._index - 1,
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
        )

    def _ensure_open(self, path: Path) -> cv2.VideoCapture:
        if self._capture is not None and self._opened == path:
            return self._capture
        self.close()
        if not path.is_file():
            raise FileNotFoundError(f"no such video file: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"cannot open video file: {path}")
        self._capture, self._opened, self._index = capture, path, 0
        return capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened = None


@block(category="sources")
class Camera:
    """Grab a frame from an attached camera.

    ``index`` selects the device, as in ``cv2.VideoCapture(0)``.
    """

    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._opened: int | None = None
        self._count = 0

    def __call__(self, index: int = 0, width: int = 0, height: int = 0) -> Frame:
        capture = self._ensure_open(index, width, height)
        ok, image = capture.read()
        if not ok:
            raise OSError(f"camera {index} returned no frame")
        self._count += 1
        return Frame(
            image=image,
            index=self._count - 1,
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
        )

    def _ensure_open(self, index: int, width: int, height: int) -> cv2.VideoCapture:
        if self._capture is not None and self._opened == index:
            return self._capture
        self.close()
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise OSError(
                f"cannot open camera {index}. Is it connected and not in use?"
            )
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture, self._opened, self._count = capture, index, 0
        return capture

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._opened = None


@block(category="sources")
class FrameBuffer:
    """Delay a stream by one sweep, yielding the previous frame.

    Useful for frame differencing: feed the same image into this and into an
    ``abs_diff``, and the second input is the frame before. On the first sweep
    it returns the current frame, so downstream blocks see a valid image
    rather than ``None``.
    """

    def __init__(self) -> None:
        self._previous: Mat | None = None

    def __call__(self, input: Mat) -> Mat:
        previous = self._previous
        self._previous = input
        return input if previous is None else previous
