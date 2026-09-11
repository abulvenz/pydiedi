"""The Qt renderer. Optional, and the only package allowed to import Qt.

``pydiedi.core`` and ``pydiedi.blocks`` never depend on this, which is what
keeps a headless run possible and leaves room for a second renderer -- the
block library needs no change, because a block returns a
:class:`~pydiedi.core.types.Preview` rather than drawing anything.

Install with ``pip install pydiedi[gui]`` or ``uv sync --extra gui``.
"""

from __future__ import annotations

__all__ = ["main", "EditorWindow", "require_qt"]


def require_qt() -> None:
    """Fail with an instruction rather than a bare ImportError."""
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "The editor needs PySide6, which is an optional dependency.\n"
            "Install it with:  uv sync --extra gui   (or: pip install 'pydiedi[gui]')"
        ) from exc


def __getattr__(name: str) -> object:
    # Imported lazily so that `import pydiedi.gui` does not pull in Qt until
    # something is actually used from it.
    if name in ("main", "EditorWindow"):
        require_qt()
        from .app import main
        from .window import EditorWindow

        return {"main": main, "EditorWindow": EditorWindow}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
