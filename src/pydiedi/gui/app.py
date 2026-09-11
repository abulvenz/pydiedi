"""Application bootstrap for the editor."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .. import __version__

__all__ = ["main", "build_application"]

# A dark palette by stylesheet rather than QPalette: it applies to the docks
# and the tree as well, and keeps the canvas colours from fighting the chrome.
STYLESHEET = """
QMainWindow, QDockWidget, QWidget { background: #262a31; color: #d5d8e0; }
QDockWidget::title { background: #2f333c; padding: 5px 8px; }
QMenuBar, QToolBar { background: #2f333c; border: 0; }
QMenuBar::item:selected, QMenu::item:selected { background: #3c4250; }
QMenu { background: #2f333c; border: 1px solid #454b59; }
QToolButton { padding: 4px 9px; }
QToolButton:hover { background: #3c4250; border-radius: 3px; }
QToolButton:disabled { color: #6b7280; }
QTreeWidget, QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #1e2127; border: 1px solid #3a4050; border-radius: 3px;
    selection-background-color: #3c5a8c;
}
QPlainTextEdit { font-family: monospace; font-size: 11px; }
QTreeWidget::item { padding: 2px; }
QTreeWidget::item:selected { background: #3c5a8c; }
QStatusBar { background: #2f333c; }
QScrollBar:vertical, QScrollBar:horizontal { background: #262a31; border: 0; }
QScrollBar::handle { background: #454b59; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
"""


def build_application(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv or sys.argv[:1])
    app.setApplicationName("pydiedi")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(STYLESHEET)
    return app  # type: ignore[return-value]


def main(path: Path | None = None, argv: list[str] | None = None) -> int:
    from .window import EditorWindow

    app = build_application(argv)

    # Qt swallows SIGINT by default, so Ctrl-C in the launching terminal would
    # do nothing; the timer gives Python a chance to see the signal.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    ticker = QTimer()
    ticker.start(200)
    ticker.timeout.connect(lambda: None)

    window = EditorWindow(path)
    window.show()
    return app.exec()
