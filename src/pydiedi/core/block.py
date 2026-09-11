"""The block definition, derived from a function signature.

This module is the whole point of pydiedi. In flodiedi, a block needed a
``.fdf`` entry, a generated ``.h`` plus ``.cpp`` (~220 lines), a 102-line
``CMakeLists.txt``, ``Q_PROPERTY`` declarations, ``qRegisterMetaType`` calls,
``Q_INVOKABLE`` constructors and a ``Q_EXPORT_PLUGIN2`` macro -- roughly 320
lines of scaffolding to wrap a single call to OpenCV. All of it existed to
synthesise metadata that C++ cannot introspect: which ports does this block
have, of which type, with which default.

Python has that metadata for free. A block is a function; its type hints are
its ports.

    @block(category="filtering")
    def threshold(input: Mat, threshold: float = 128) -> Mat:
        '''Fixed-level binarisation.'''
        return cv2.threshold(input, threshold, 255, cv2.THRESH_BINARY)[1]


Ports and parameters are the same thing
---------------------------------------
Every function parameter becomes an input port. A parameter with a default is
optional: the GUI shows it as an editable field, and connecting an edge to it
overrides the default. A parameter without a default must be connected.

flodiedi split these into ``INPORT_`` and ``PARAM_`` in the ``.fdf`` file, but
then undermined the split itself: ``setInPort()``/``setOutPort()`` let the user
right-click any property and promote it to a port at runtime. Unifying the two
is simpler and strictly more capable.
"""

from __future__ import annotations

import inspect
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import type_name

__all__ = ["Port", "PortKind", "BlockSpec", "block", "BlockDefinitionError"]

DEFAULT_OUTPUT_NAME = "output"


class BlockDefinitionError(TypeError):
    """A block was declared in a way pydiedi cannot derive ports from.

    Always raised at import time, never at execution time, so a broken block
    definition cannot reach a running diagram.
    """


class PortKind(str, Enum):
    IN = "in"
    OUT = "out"


_MISSING: Any = inspect.Parameter.empty


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    type: Any
    kind: PortKind
    default: Any = _MISSING

    @property
    def required(self) -> bool:
        """An input with no default must be connected for the graph to run."""
        return self.default is _MISSING

    @property
    def type_name(self) -> str:
        return type_name(self.type)

    @property
    def choices(self) -> tuple[str, ...] | None:
        """Enum member names, for a GUI dropdown. ``None`` for other types."""
        if isinstance(self.type, type) and issubclass(self.type, Enum):
            return tuple(m.name for m in self.type)
        return None

    def describe(self) -> str:
        if self.kind is PortKind.OUT or self.required:
            return f"{self.name}: {self.type_name}"
        return f"{self.name}: {self.type_name} = {_format_default(self.default)}"


def _format_default(value: Any) -> str:
    if isinstance(value, Enum):
        return value.name
    return repr(value)


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """Everything the executor, the CLI and the GUI need to know about a block."""

    name: str
    category: str
    doc: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    fn: Callable[..., Any]

    def input(self, name: str) -> Port | None:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> Port | None:
        return next((p for p in self.outputs if p.name == name), None)

    @property
    def required_inputs(self) -> tuple[Port, ...]:
        return tuple(p for p in self.inputs if p.required)

    def signature(self) -> str:
        """A one-line rendering, e.g. ``threshold(input: Mat, level: float = 128) -> Mat``."""
        args = ", ".join(p.describe() for p in self.inputs)
        if not self.outputs:
            out = "none"
        elif len(self.outputs) == 1 and self.outputs[0].name == DEFAULT_OUTPUT_NAME:
            out = self.outputs[0].type_name
        else:
            out = "(" + ", ".join(p.describe() for p in self.outputs) + ")"
        return f"{self.name}({args}) -> {out}"

    def call(self, values: dict[str, Any]) -> dict[str, Any]:
        """Invoke the block and return its outputs keyed by port name."""
        result = self.fn(**values)
        return self._map_outputs(result)

    def _map_outputs(self, result: Any) -> dict[str, Any]:
        if not self.outputs:
            return {}
        if len(self.outputs) == 1:
            return {self.outputs[0].name: result}
        names = [p.name for p in self.outputs]
        # A NamedTuple return: read fields by name rather than by position, so
        # that reordering the class does not silently rewire the graph.
        if all(hasattr(result, n) for n in names):
            return {n: getattr(result, n) for n in names}
        if not isinstance(result, tuple):
            raise TypeError(
                f"block {self.name!r} declares {len(names)} outputs "
                f"({', '.join(names)}) but returned a single {type(result).__name__}"
            )
        if len(result) != len(names):
            raise TypeError(
                f"block {self.name!r} declares {len(names)} outputs "
                f"({', '.join(names)}) but returned {len(result)} values"
            )
        return dict(zip(names, result, strict=True))


def _is_named_tuple(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, tuple) and hasattr(tp, "_fields")


def _resolve_hints(
    fn: Callable[..., Any], localns: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a function's annotations to real types.

    ``localns`` is the scope the block was declared in. It is needed because
    ``from __future__ import annotations`` turns annotations into strings, and
    ``get_type_hints`` then only searches module globals -- so a NamedTuple
    defined inside a function or a notebook cell would not be found.
    """
    try:
        return typing.get_type_hints(fn, localns=localns)
    except Exception as exc:  # unresolvable forward reference, bad import, ...
        raise BlockDefinitionError(
            f"cannot resolve type hints of block {fn.__name__!r}: {exc}. "
            f"If the type is declared in a local scope, make sure it is visible "
            f"where the block is defined."
        ) from exc


def _build_inputs(fn: Callable[..., Any], hints: dict[str, Any]) -> tuple[Port, ...]:
    sig = inspect.signature(fn)
    ports: list[Port] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise BlockDefinitionError(
                f"block {fn.__name__!r}: *args/**kwargs cannot be turned into ports "
                f"(parameter {name!r}). Declare the ports explicitly."
            )
        if name not in hints:
            raise BlockDefinitionError(
                f"block {fn.__name__!r}: parameter {name!r} has no type annotation. "
                f"pydiedi derives ports from type hints, so every parameter needs one."
            )
        ports.append(
            Port(
                name=name,
                type=hints[name],
                kind=PortKind.IN,
                default=param.default if param.default is not param.empty else _MISSING,
            )
        )
    return tuple(ports)


def _build_outputs(
    fn: Callable[..., Any], hints: dict[str, Any], localns: dict[str, Any] | None = None
) -> tuple[Port, ...]:
    if "return" not in hints:
        raise BlockDefinitionError(
            f"block {fn.__name__!r} has no return annotation. Use '-> None' for a "
            f"block that produces nothing."
        )
    ret = hints["return"]
    if ret is None or ret is type(None):
        return ()
    if _is_named_tuple(ret):
        field_hints = typing.get_type_hints(ret, localns=localns)
        return tuple(
            Port(name=f, type=field_hints.get(f, Any), kind=PortKind.OUT)
            for f in ret._fields
        )
    # A bare tuple[...] would give unnamed outputs; ports need names, and
    # positional wiring is exactly the kind of fragility worth refusing.
    if typing.get_origin(ret) is tuple:
        raise BlockDefinitionError(
            f"block {fn.__name__!r} returns a plain tuple, so its output ports would "
            f"have no names. Declare a NamedTuple instead:\n"
            f"    class {fn.__name__.title().replace('_', '')}Out(NamedTuple):\n"
            f"        image: Mat\n"
            f"        count: int"
        )
    return (Port(name=DEFAULT_OUTPUT_NAME, type=ret, kind=PortKind.OUT),)


def _spec_from_function(
    fn: Callable[..., Any],
    *,
    name: str | None,
    category: str,
    localns: dict[str, Any] | None = None,
) -> BlockSpec:
    hints = _resolve_hints(fn, localns)
    return BlockSpec(
        name=name or fn.__name__,
        category=category,
        doc=inspect.cleandoc(fn.__doc__ or ""),
        inputs=_build_inputs(fn, hints),
        outputs=_build_outputs(fn, hints, localns),
        fn=fn,
    )


def _caller_locals(depth: int = 2) -> dict[str, Any] | None:
    """The local scope at the block's definition site, if obtainable."""
    try:
        return sys._getframe(depth).f_locals
    except (AttributeError, ValueError):  # no frame introspection available
        return None


def block(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    category: str = "uncategorized",
    register_globally: bool = True,
):
    """Declare a function as a diagram block.

    Usable bare or with arguments::

        @block
        def invert(input: Mat) -> Mat: ...

        @block(category="filtering", name="gauss")
        def gaussian_blur(input: Mat, sigma: float = 1.5) -> Mat: ...

    The decorated function stays directly callable; the derived metadata is
    attached as ``fn.spec``.

    ``register_globally=False`` derives the spec without entering it into the
    process-wide registry, which is what tests and private block sets want.
    """
    # Captured here, at the definition site, so that types declared in a local
    # scope resolve. Both call forms reach this line from the same depth.
    localns = _caller_locals()

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        if isinstance(target, type):
            # Stateful blocks (flodiedi's VideoFile held a VideoCapture and a
            # mutex as members) will be classes. The port derivation differs --
            # __init__ carries the parameters, __call__ the ports -- so this is
            # refused explicitly rather than misinterpreted.
            raise BlockDefinitionError(
                f"{target.__name__!r} is a class. Class-based (stateful) blocks are "
                f"not implemented yet; use a function for now."
            )
        spec = _spec_from_function(
            target, name=name, category=category, localns=localns
        )
        target.spec = spec  # type: ignore[attr-defined]

        if register_globally:
            from .registry import register

            register(spec)
        return target

    if fn is not None:  # bare @block
        return decorate(fn)
    return decorate
