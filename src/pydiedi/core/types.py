"""Data types that flow through a diagram.

This module is intentionally free of any GUI or OpenCV import. ``Mat`` is a
plain ``numpy.ndarray`` alias, so the core never needs ``cv2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

__all__ = ["Mat", "Preview", "type_name", "is_compatible", "coerce", "CoercionError"]


# An image or matrix. In flodiedi this was ``cv::Mat`` plus a hand-maintained
# table of subtypes (Mat1f, Mat3b, ...) in ``Plugins/datatypes``. Under numpy
# those are all one type with a dtype and a shape; distinguishing them is a
# runtime concern, deferred to phase 1.
Mat: TypeAlias = np.ndarray


@dataclass(slots=True)
class Preview:
    """A renderer-agnostic display request.

    Blocks that want to show something return this instead of building a
    widget. The Qt renderer turns it into a ``QGraphicsPixmapItem``, a web
    renderer would send a PNG, a headless run ignores it.

    This is the single most important type in pydiedi: it is what keeps
    ``core`` and ``blocks`` free of any GUI dependency. In flodiedi the block
    interface itself carried ``createExtraWidget()`` and
    ``renderAdditionalStuff()``, which made headless execution a special case
    and caused cross-thread painting crashes.
    """

    image: Mat | None = None
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def type_name(tp: Any) -> str:
    """A short, readable name for a type annotation.

    Used by the CLI, by error messages and by the GUI. Handles the
    ``Mat``/``ndarray`` alias and unparameterised generics without dragging in
    ``typing`` internals at call sites.
    """
    if tp is None or tp is type(None):
        return "none"
    if tp is Any:
        return "any"
    if tp is np.ndarray:
        return "Mat"
    name = getattr(tp, "__name__", None)
    if name:
        return name
    return str(tp).replace("typing.", "")


def is_compatible(source: Any, target: Any) -> bool:
    """Whether a value of type ``source`` may be fed into a ``target`` port.

    Phase 0 keeps this deliberately strict: identical types, or ``Any`` on
    either side. The conversion table that flodiedi kept in the plain-text
    file ``Plugins/datatypes`` (``double->int``, ``Mat->Mat1f``, ...) comes in
    phase 1, once there is a second Mat-like type to convert between.
    """
    if source is Any or target is Any:
        return True
    if source is target:
        return True
    # bool is a subclass of int in Python, but feeding a bool into an int port
    # is virtually always a mistake in a dataflow graph, so it is not allowed.
    if source is bool or target is bool:
        return source is target
    try:
        return issubclass(source, target)
    except TypeError:
        return False


class CoercionError(TypeError):
    """A literal from a diagram file does not fit the port it is assigned to."""


def coerce(value: Any, target: Any) -> Any:
    """Turn a YAML literal into the type a port expects.

    A diagram file holds only what YAML can express, so an enum arrives as the
    string ``"binary"`` and a path as a plain string. This maps those onto the
    declared types. Anything unrecognised is passed through untouched rather
    than guessed at.
    """
    if target is Any or target is None:
        return value

    if isinstance(target, type) and issubclass(target, Enum):
        if isinstance(value, target):
            return value
        try:
            return target[value] if isinstance(value, str) else target(value)
        except (KeyError, ValueError):
            options = ", ".join(m.name for m in target)
            raise CoercionError(
                f"{value!r} is not a valid {target.__name__}; expected one of: {options}"
            ) from None

    if target is Path:
        return value if isinstance(value, Path) else Path(value)

    if target is bool:
        if isinstance(value, bool):
            return value
        raise CoercionError(f"expected true or false, got {value!r}")

    if target in (int, float):
        if isinstance(value, bool):  # YAML 'true' is not a number
            raise CoercionError(f"expected {target.__name__}, got boolean {value!r}")
        if isinstance(value, (int, float)):
            return target(value)
        raise CoercionError(f"expected {target.__name__}, got {type(value).__name__}")

    if target is str:
        if isinstance(value, str):
            return value
        raise CoercionError(f"expected str, got {type(value).__name__}")

    return value
