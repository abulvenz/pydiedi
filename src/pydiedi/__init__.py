"""pydiedi -- visual dataflow for image processing.

A Python port of flodiedi (2010-2013, C++/Qt4). The architectural rule that
shapes this package: :mod:`pydiedi.core` and :mod:`pydiedi.blocks` never import
a GUI toolkit. Rendering lives in :mod:`pydiedi.gui` and is optional, which is
what makes a headless run and a future web renderer possible.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
