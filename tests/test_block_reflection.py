"""Ports are derived from type hints -- the premise of the whole port.

Each of these assertions replaces something flodiedi had to generate: a
``.fdf`` line, a ``Q_PROPERTY``, a ``qRegisterMetaType`` call, an
``inports.append(Port(...))`` statement.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import pytest

from pydiedi.core import Mat, Preview, block, registry
from pydiedi.core.block import BlockDefinitionError, PortKind


@pytest.fixture(autouse=True)
def isolated_registry():
    """Each test gets a clean registry, then the real blocks are restored."""
    saved = dict(registry.all_blocks())
    registry.clear()
    yield
    registry.clear()
    for spec in saved.values():
        registry.register(spec)


class Mode(Enum):
    fast = 0
    accurate = 1


def test_parameters_become_input_ports():
    @block(category="test")
    def resize(input: Mat, width: int, height: int = 480) -> Mat:
        return input

    spec = resize.spec
    assert [p.name for p in spec.inputs] == ["input", "width", "height"]
    assert all(p.kind is PortKind.IN for p in spec.inputs)


def test_missing_default_means_required():
    @block(category="test")
    def crop(input: Mat, size: int = 10) -> Mat:
        return input

    assert crop.spec.input("input").required is True
    assert crop.spec.input("size").required is False
    assert [p.name for p in crop.spec.required_inputs] == ["input"]


def test_defaults_are_captured():
    @block(category="test")
    def blur(input: Mat, sigma: float = 1.5, name: str = "x") -> Mat:
        return input

    assert blur.spec.input("sigma").default == 1.5
    assert blur.spec.input("name").default == "x"


def test_single_return_is_named_output():
    @block(category="test")
    def invert(input: Mat) -> Mat:
        return input

    assert [p.name for p in invert.spec.outputs] == ["output"]
    assert invert.spec.outputs[0].kind is PortKind.OUT


def test_named_tuple_return_gives_one_port_per_field():
    class Frame(NamedTuple):
        image: Mat
        fps: int

    @block(category="test")
    def video(path: Path) -> Frame:
        return Frame(None, 25)

    spec = video.spec
    assert [p.name for p in spec.outputs] == ["image", "fps"]
    assert spec.output("fps").type is int


def test_named_tuple_outputs_are_mapped_by_name_not_position():
    class Pair(NamedTuple):
        left: int
        right: int

    @block(category="test")
    def split() -> Pair:
        return Pair(left=1, right=2)

    assert split.spec.call({}) == {"left": 1, "right": 2}


def test_none_return_means_no_outputs():
    @block(category="test")
    def sink(input: Mat) -> None:
        return None

    assert sink.spec.outputs == ()
    assert sink.spec.call({"input": None}) == {}


def test_enum_parameter_exposes_choices():
    @block(category="test")
    def detect(input: Mat, mode: Mode = Mode.fast) -> Mat:
        return input

    assert detect.spec.input("mode").choices == ("fast", "accurate")
    assert detect.spec.input("input").choices is None


def test_docstring_becomes_block_doc():
    @block(category="test")
    def documented(input: Mat) -> Mat:
        """First line.

        More detail.
        """
        return input

    assert documented.spec.doc.startswith("First line.")
    assert "More detail." in documented.spec.doc


def test_block_name_defaults_to_function_name_and_can_be_overridden():
    @block(category="test")
    def gaussian_blur(input: Mat) -> Mat:
        return input

    @block(category="test", name="gauss")
    def other(input: Mat) -> Mat:
        return input

    assert gaussian_blur.spec.name == "gaussian_blur"
    assert other.spec.name == "gauss"
    assert set(registry.all_blocks()) == {"gaussian_blur", "gauss"}


def test_bare_decorator_works():
    @block
    def plain(input: Mat) -> Mat:
        return input

    assert plain.spec.name == "plain"
    assert plain.spec.category == "uncategorized"


def test_decorated_function_stays_callable():
    @block(category="test")
    def double(value: int = 2) -> int:
        return value * 2

    assert double(21) == 42


def test_signature_rendering():
    @block(category="test")
    def thr(input: Mat, level: float = 128, mode: Mode = Mode.fast) -> Mat:
        return input

    assert thr.spec.signature() == (
        "thr(input: Mat, level: float = 128, mode: Mode = fast) -> Mat"
    )


def test_preview_is_an_ordinary_output_type():
    @block(category="test")
    def show(input: Mat) -> Preview:
        return Preview(image=input)

    assert show.spec.output("output").type is Preview


# -- definitions that must be refused, loudly and at import time ----------


def test_unannotated_parameter_is_refused():
    with pytest.raises(BlockDefinitionError, match="no type annotation"):

        @block(category="test")
        def bad(input) -> Mat:  # type: ignore[no-untyped-def]
            return input


def test_missing_return_annotation_is_refused():
    with pytest.raises(BlockDefinitionError, match="no return annotation"):

        @block(category="test")
        def bad(input: Mat):  # type: ignore[no-untyped-def]
            return input


def test_plain_tuple_return_is_refused_because_ports_need_names():
    with pytest.raises(BlockDefinitionError, match="no names"):

        @block(category="test")
        def bad(input: Mat) -> tuple[Mat, int]:
            return input, 1


def test_varargs_are_refused():
    with pytest.raises(BlockDefinitionError, match=r"\*args"):

        @block(category="test")
        def bad(*args: Mat) -> Mat:
            return args[0]


def test_class_without_call_is_refused():
    with pytest.raises(BlockDefinitionError, match="needs a __call__ method"):

        @block(category="test")
        class NoCall:
            def run(self, input: Mat) -> Mat:
                return input


def test_class_with_required_init_arguments_is_refused():
    """Parameters belong on __call__, where they can be connected."""
    with pytest.raises(BlockDefinitionError, match="__init__ must be callable with no"):

        @block(category="test")
        class NeedsArgs:
            def __init__(self, path: str):
                self.path = path

            def __call__(self) -> int:
                return 1


def test_duplicate_block_name_is_refused():
    @block(category="test", name="clash")
    def first(input: Mat) -> Mat:
        return input

    with pytest.raises(registry.DuplicateBlockError, match="already registered"):

        @block(category="test", name="clash")
        def second(input: Mat) -> Mat:
            return input


def test_unknown_block_lookup_suggests_a_close_match():
    @block(category="test")
    def gaussian_blur(input: Mat) -> Mat:
        return input

    with pytest.raises(registry.UnknownBlockError, match="Did you mean 'gaussian_blur'"):
        registry.get("gaussian_blurr")


def test_wrong_output_count_is_reported():
    class Pair(NamedTuple):
        a: int
        b: int

    @block(category="test")
    def bad() -> Pair:
        return (1, 2, 3)  # type: ignore[return-value]

    with pytest.raises(TypeError, match="declares 2 outputs"):
        bad.spec.call({})


# -- stateful blocks, declared as classes ---------------------------------


def test_class_block_derives_ports_from_call():
    @block(category="test")
    class Counter:
        """Count how often it ran."""

        def __init__(self) -> None:
            self.n = 0

        def __call__(self, step: int = 1) -> int:
            self.n += step
            return self.n

    spec = Counter.spec
    assert spec.is_stateful is True
    assert [p.name for p in spec.inputs] == ["step"]
    assert [p.name for p in spec.outputs] == ["output"]
    assert spec.doc == "Count how often it ran."


def test_class_block_name_is_snake_cased():
    @block(category="test")
    class VideoFileReader:
        def __call__(self) -> int:
            return 1

    assert VideoFileReader.spec.name == "video_file_reader"


def test_class_block_name_can_be_overridden():
    @block(category="test", name="vid")
    class VideoFileReader:
        def __call__(self) -> int:
            return 1

    assert VideoFileReader.spec.name == "vid"


def test_self_is_not_a_port():
    @block(category="test")
    class Thing:
        def __call__(self, value: int = 0) -> int:
            return value

    assert "self" not in [p.name for p in Thing.spec.inputs]


def test_instance_keeps_state_between_calls():
    @block(category="test")
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0

        def __call__(self, add: int = 1) -> int:
            self.total += add
            return self.total

    spec = Accumulator.spec
    instance = spec.instantiate()
    assert spec.call({"add": 5}, instance) == {"output": 5}
    assert spec.call({"add": 3}, instance) == {"output": 8}


def test_two_instances_are_independent():
    @block(category="test")
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0

        def __call__(self, add: int = 1) -> int:
            self.total += add
            return self.total

    spec = Accumulator.spec
    a, b = spec.instantiate(), spec.instantiate()
    spec.call({"add": 10}, a)
    assert spec.call({"add": 1}, b) == {"output": 1}


def test_stateful_block_without_an_instance_is_a_clear_error():
    @block(category="test")
    class Thing:
        def __call__(self) -> int:
            return 1

    with pytest.raises(TypeError, match="is stateful and needs an instance"):
        Thing.spec.call({})


def test_close_is_called_when_present():
    closed: list[bool] = []

    @block(category="test")
    class Resource:
        def __call__(self) -> int:
            return 1

        def close(self) -> None:
            closed.append(True)

    spec = Resource.spec
    instance = spec.instantiate()
    spec.close(instance)
    assert closed == [True]


def test_close_is_optional():
    @block(category="test")
    class NoClose:
        def __call__(self) -> int:
            return 1

    spec = NoClose.spec
    spec.close(spec.instantiate())  # must not raise


def test_function_block_is_not_stateful():
    @block(category="test")
    def plain(value: int = 1) -> int:
        return value

    assert plain.spec.is_stateful is False
    assert plain.spec.instantiate() is None
