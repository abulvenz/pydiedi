"""Shared fixtures.

The registry is process-global, so tests that define their own blocks must not
leak them into tests that expect the real library. ``isolated_registry`` gives
each test a clean slate and restores what was there before.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Must happen before anything imports Qt, so the GUI tests never try to reach a
# display. conftest is imported ahead of the test modules.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pydiedi.core import registry  # noqa: E402
from pydiedi.core.block import BlockSpec  # noqa: E402


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    saved = dict(registry.all_blocks())
    registry.clear()
    yield
    registry.clear()
    for spec in saved.values():
        registry.register(spec)


@pytest.fixture
def register():
    """Register decorated functions for the duration of one test."""

    def _register(*functions) -> None:
        for fn in functions:
            spec: BlockSpec = fn.spec
            registry.register(spec)

    return _register


@pytest.fixture
def real_blocks() -> Iterator[None]:
    """Ensure the shipped block library is discovered."""
    registry.discover()
    yield
