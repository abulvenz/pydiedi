"""The GUI-free core: blocks, graph, executor, I/O.

Nothing in this package may import a GUI toolkit. ``tests/test_core_is_gui_free.py``
enforces that.
"""

from __future__ import annotations

from .block import BlockDefinitionError, BlockSpec, Port, PortKind, block
from .executor import StopExecution
from .types import Mat, Preview, is_compatible, type_name

__all__ = [
    "Mat",
    "Preview",
    "Port",
    "PortKind",
    "BlockSpec",
    "BlockDefinitionError",
    "block",
    "StopExecution",
    "is_compatible",
    "type_name",
]
