"""The block registry: discovery by import, lookup by name.

flodiedi scanned a directory for ``.so`` files, loaded each with
``QPluginLoader``, cast it to the ``FlowDiagramBlock`` interface and -- when
that failed -- shelled out to ``system("ldd ... | grep 'not found'")`` to guess
why. Here, discovery is an import: the ``@block`` decorator registers itself.
"""

from __future__ import annotations

import difflib
import importlib
import pkgutil
from types import ModuleType

from .block import BlockSpec

__all__ = [
    "register",
    "get",
    "all_blocks",
    "categories",
    "by_category",
    "discover",
    "clear",
    "UnknownBlockError",
    "DuplicateBlockError",
]

_REGISTRY: dict[str, BlockSpec] = {}


class UnknownBlockError(KeyError):
    """A diagram referenced a block name that is not registered."""

    def __str__(self) -> str:  # KeyError would quote the message
        return self.args[0] if self.args else ""


class DuplicateBlockError(ValueError):
    """Two blocks claim the same name."""


def register(spec: BlockSpec) -> None:
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing.fn is not spec.fn:
        raise DuplicateBlockError(
            f"block name {spec.name!r} is already registered by "
            f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
            f"{spec.fn.__module__}.{spec.fn.__qualname__} would shadow it. "
            f"Pass @block(name=...) to disambiguate."
        )
    _REGISTRY[spec.name] = spec


def get(name: str) -> BlockSpec:
    """Look up a block, or raise with a suggestion.

    The suggestion matters: flodiedi's loader called ``createBlockByKey()`` and
    then dereferenced the result without a NULL check, so a diagram naming an
    unbuilt plugin crashed the editor with no indication which block was at
    fault.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownBlockError(_unknown_message(name)) from None


def _unknown_message(name: str) -> str:
    if not _REGISTRY:
        return (
            f"unknown block {name!r}: no blocks are registered at all. "
            f"Call discover() first."
        )
    close = difflib.get_close_matches(name, sorted(_REGISTRY), n=3, cutoff=0.6)
    msg = f"unknown block {name!r}"
    if close:
        msg += f". Did you mean {', '.join(repr(c) for c in close)}?"
    else:
        msg += f". {len(_REGISTRY)} blocks are registered; run 'pydiedi blocks' to list them."
    return msg


def all_blocks() -> dict[str, BlockSpec]:
    """All registered blocks, sorted by name."""
    return dict(sorted(_REGISTRY.items()))


def categories() -> list[str]:
    return sorted({spec.category for spec in _REGISTRY.values()})


def by_category() -> dict[str, list[BlockSpec]]:
    grouped: dict[str, list[BlockSpec]] = {}
    for spec in _REGISTRY.values():
        grouped.setdefault(spec.category, []).append(spec)
    return {cat: sorted(specs, key=lambda s: s.name) for cat, specs in sorted(grouped.items())}


def clear() -> None:
    """Drop all registrations. For tests."""
    _REGISTRY.clear()


def discover(package: str | ModuleType = "pydiedi.blocks") -> int:
    """Find every block in ``package`` and register it.

    Returns the number of registered blocks afterwards. Import errors are not
    swallowed: a block library that cannot be imported is a defect worth
    seeing immediately, not a silently missing entry in a toolbox.

    Registration reads each module's members rather than relying on the
    ``@block`` decorator's side effect, because a module is only executed on
    its first import. Depending on the side effect would make a second
    ``discover()`` call a no-op -- which is fine until something clears the
    registry, and then it silently returns nothing.
    """
    root = importlib.import_module(package) if isinstance(package, str) else package
    modules = [root]
    if hasattr(root, "__path__"):
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{root.__name__}."):
            modules.append(importlib.import_module(info.name))

    for module in modules:
        for name in dir(module):
            if name.startswith("_"):
                continue
            spec = getattr(getattr(module, name), "spec", None)
            # Only claim objects defined in this module, so that a re-exported
            # block is not registered twice under a different owner.
            if isinstance(spec, BlockSpec) and spec.fn.__module__ == module.__name__:
                register(spec)
    return len(_REGISTRY)
