"""The architectural constraint, enforced.

``pydiedi.core`` and ``pydiedi.blocks`` must not depend on a GUI toolkit. This
is not a style preference: it is what allows a headless run, a Qt editor and a
future web renderer to share one block library.

flodiedi lost this property. ``FlowDiagramBlock`` -- the model's base class --
declared ``createExtraWidget()``, ``createDisplayWidget()`` and
``renderAdditionalStuff(QGraphicsItem*)``, so the model knew the view. The
consequences were a global ``usesGui_`` flag to special-case headless
execution, and ``setPixmap()`` calls from the worker thread that Qt5 and Qt6
abort on.

Two independent checks, because each catches what the other misses:
the runtime check sees transitive imports, the static check sees code that
merely is not reached yet.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

GUI_MODULES = (
    "PySide6",
    "PySide2",
    "PyQt5",
    "PyQt6",
    "qtpy",
    "Qt",
    "tkinter",
    "wx",
    "gi",
)

SRC = Path(__file__).resolve().parent.parent / "src"
GUI_FREE_PACKAGES = ("pydiedi.core", "pydiedi.blocks")


@pytest.mark.parametrize("package", GUI_FREE_PACKAGES)
def test_importing_package_loads_no_gui_toolkit(package: str) -> None:
    """Import the package in a clean interpreter and inspect sys.modules."""
    script = (
        "import importlib, sys, json\n"
        f"importlib.import_module({package!r})\n"
        "from pydiedi.core import registry\n"
        + ("registry.discover()\n" if package == "pydiedi.blocks" else "")
        + f"found = sorted(m for m in sys.modules if m.split('.')[0] in {GUI_MODULES!r})\n"
        "print(json.dumps(found))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"importing {package} failed:\n{proc.stderr}"
    leaked = proc.stdout.strip()
    assert leaked == "[]", (
        f"{package} pulled in a GUI toolkit: {leaked}. "
        f"Rendering belongs in pydiedi.gui; blocks describe what to show "
        f"(return a Preview) rather than drawing it."
    )


def _source_files(package: str) -> list[Path]:
    root = SRC / Path(package.replace(".", "/"))
    return sorted(root.rglob("*.py"))


@pytest.mark.parametrize("package", GUI_FREE_PACKAGES)
def test_no_gui_import_statements_in_source(package: str) -> None:
    """Scan the AST, so unreachable or lazily-imported Qt is caught too."""
    files = _source_files(package)
    assert files, f"no source files found for {package}"

    offenders: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in GUI_MODULES:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {name}")

    assert not offenders, "GUI imports found in a GUI-free package:\n" + "\n".join(
        offenders
    )


def test_core_does_not_import_cv2() -> None:
    """The core is also OpenCV-free; only the block library needs cv2.

    Keeping ``core`` free of cv2 means the graph, the executor and the file
    format can be tested and reasoned about without an image processing
    library, and a future block library for a different domain reuses them
    unchanged.
    """
    offenders: list[str] = []
    for path in _source_files("pydiedi.core"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            if any(n.split(".")[0] == "cv2" for n in names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")

    assert not offenders, "pydiedi.core imports cv2 at:\n" + "\n".join(offenders)
