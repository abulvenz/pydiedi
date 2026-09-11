"""Reading and writing diagram files.

The format is YAML, chosen for one reason: diffs. A visual tool whose files
cannot be reviewed in version control becomes painful to maintain, and
flodiedi's XML was the worst case of that -- UUID node identity, four lines
per connection, and a persisted ``executionorder`` that changed on every run.

    version: 1
    name: wallpaper

    nodes:
      cam:       {block: capture, params: {index: 0}}
      gray_live: {block: cvt_color, params: {code: bgr2gray}}
      blur:      {block: gaussian_blur, params: {sigma: 1.5}}

    edges:
      - cam.camera_image -> gray_live.input
      - gray_live.output -> blur.input

    layout:
      cam: [-894, -272]

Node ids are chosen by the author, so a diagram with three ``cvt_color``
blocks reads as ``gray_live`` / ``gray_ref`` / ``gray_warp``. ``layout`` is
optional and carries no semantics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .graph import Edge, Graph, Node

__all__ = ["load", "loads", "dump", "dumps", "DiagramSyntaxError", "FORMAT_VERSION"]

FORMAT_VERSION = 1

_EDGE_RE = re.compile(
    r"^\s*(?P<src>[A-Za-z_][\w-]*)\.(?P<src_port>[A-Za-z_]\w*)"
    r"\s*->\s*"
    r"(?P<dst>[A-Za-z_][\w-]*)\.(?P<dst_port>[A-Za-z_]\w*)\s*$"
)
_ID_RE = re.compile(r"^[A-Za-z_][\w-]*$")
_LINE_KEY = "__line__"


class DiagramSyntaxError(ValueError):
    """A diagram file is malformed. Always names the offending line."""


class _LineLoader(yaml.SafeLoader):
    """A SafeLoader that records the source line of every mapping.

    Line numbers are what make the error messages actionable, and were exactly
    what flodiedi's loader lacked: an unknown block name crashed the editor via
    an unchecked ``createBlockByKey()`` result, with no hint where in the file
    the problem was.
    """


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode) -> dict[str, Any]:
    mapping = loader.construct_mapping(node, deep=True)
    mapping[_LINE_KEY] = node.start_mark.line + 1
    return mapping


_LineLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _line_of(raw: Any, fallback: int | None = None) -> int | None:
    if isinstance(raw, dict):
        value = raw.get(_LINE_KEY)
        if isinstance(value, int):
            return value
    return fallback


def _strip_lines(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_lines(v) for k, v in value.items() if k != _LINE_KEY}
    if isinstance(value, list):
        return [_strip_lines(v) for v in value]
    return value


# -- reading ---------------------------------------------------------------


def loads(text: str, *, source: str = "<string>") -> Graph:
    try:
        raw = yaml.load(text, Loader=_LineLoader)
    except yaml.YAMLError as exc:
        raise DiagramSyntaxError(f"{source}: {exc}") from exc

    if raw is None:
        raise DiagramSyntaxError(f"{source}: file is empty")
    if not isinstance(raw, dict):
        raise DiagramSyntaxError(
            f"{source}: expected a mapping at the top level, got {type(raw).__name__}"
        )

    version = raw.get("version")
    if version is None:
        raise DiagramSyntaxError(f"{source}: missing 'version' key (expected {FORMAT_VERSION})")
    if version != FORMAT_VERSION:
        raise DiagramSyntaxError(
            f"{source}: format version {version!r} is not supported "
            f"(this build reads version {FORMAT_VERSION})"
        )

    graph = Graph(name=str(raw.get("name", "untitled")))
    _read_nodes(raw, graph, source)
    _read_edges(raw, graph, source)
    _read_layout(raw, graph, source)
    return graph


def _read_nodes(raw: dict[str, Any], graph: Graph, source: str) -> None:
    nodes = raw.get("nodes") or {}
    if not isinstance(nodes, dict):
        raise DiagramSyntaxError(
            f"{source}: 'nodes' must be a mapping of id -> definition, "
            f"got {type(nodes).__name__}"
        )
    for node_id, definition in nodes.items():
        if node_id == _LINE_KEY:
            continue
        line = _line_of(definition, _line_of(nodes))
        at = f"{source}:{line}" if line else source
        if not _ID_RE.match(str(node_id)):
            raise DiagramSyntaxError(
                f"{at}: {node_id!r} is not a valid node id "
                f"(letters, digits, underscore and hyphen; must not start with a digit)"
            )
        if not isinstance(definition, dict):
            raise DiagramSyntaxError(
                f"{at}: node {node_id!r} must be a mapping with a 'block' key, "
                f"got {type(definition).__name__}"
            )
        block = definition.get("block")
        if not block:
            raise DiagramSyntaxError(f"{at}: node {node_id!r} has no 'block' key")

        params = definition.get("params") or {}
        if not isinstance(params, dict):
            raise DiagramSyntaxError(
                f"{at}: 'params' of node {node_id!r} must be a mapping, "
                f"got {type(params).__name__}"
            )
        unknown = set(definition) - {"block", "params", _LINE_KEY}
        if unknown:
            raise DiagramSyntaxError(
                f"{at}: node {node_id!r} has unknown key(s) "
                f"{', '.join(repr(k) for k in sorted(unknown))}; expected 'block', 'params'"
            )
        graph.add(
            Node(
                id=str(node_id),
                block=str(block),
                params=_strip_lines(params),
                line=line,
            )
        )


def _read_edges(raw: dict[str, Any], graph: Graph, source: str) -> None:
    edges = raw.get("edges") or []
    if not isinstance(edges, list):
        raise DiagramSyntaxError(
            f"{source}: 'edges' must be a list, got {type(edges).__name__}"
        )
    base_line = _line_of(raw)
    for index, entry in enumerate(edges):
        if not isinstance(entry, str):
            raise DiagramSyntaxError(
                f"{source}: edge #{index + 1} must be a string of the form "
                f"'node.port -> node.port', got {type(entry).__name__}"
            )
        match = _EDGE_RE.match(entry)
        if not match:
            raise DiagramSyntaxError(
                f"{source}: cannot parse edge {entry!r}. "
                f"Expected 'node.port -> node.port'."
            )
        graph.edges.append(
            Edge(
                src=match["src"],
                src_port=match["src_port"],
                dst=match["dst"],
                dst_port=match["dst_port"],
                line=base_line,
            )
        )


def _read_layout(raw: dict[str, Any], graph: Graph, source: str) -> None:
    layout = raw.get("layout") or {}
    if not isinstance(layout, dict):
        raise DiagramSyntaxError(
            f"{source}: 'layout' must be a mapping of id -> [x, y], "
            f"got {type(layout).__name__}"
        )
    for node_id, position in layout.items():
        if node_id == _LINE_KEY:
            continue
        if not (isinstance(position, (list, tuple)) and len(position) == 2):
            raise DiagramSyntaxError(
                f"{source}: layout of {node_id!r} must be [x, y], got {position!r}"
            )
        try:
            graph.layout[str(node_id)] = (float(position[0]), float(position[1]))
        except (TypeError, ValueError):
            raise DiagramSyntaxError(
                f"{source}: layout of {node_id!r} must be two numbers, got {position!r}"
            ) from None


def load(path: str | Path) -> Graph:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagramSyntaxError(f"cannot read {path}: {exc}") from exc
    graph = loads(text, source=str(path))
    graph.base_dir = path.parent.resolve()
    return graph


# -- writing ---------------------------------------------------------------


class _Flow(dict):
    """A mapping rendered inline, to keep node definitions to one line each."""


class _FlowList(list):
    """A sequence rendered inline, for ``[x, y]`` positions."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    _Flow,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=True
    ),
)
_Dumper.add_representer(
    _FlowList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def _plain(value: Any) -> Any:
    """Reduce a param value to something YAML can hold."""
    from enum import Enum

    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def dumps(graph: Graph) -> str:
    document: dict[str, Any] = {"version": FORMAT_VERSION, "name": graph.name}

    nodes: dict[str, Any] = {}
    for node_id, node in graph.nodes.items():
        definition: dict[str, Any] = {"block": node.block}
        if node.params:
            definition["params"] = _Flow(
                {k: _plain(v) for k, v in node.params.items()}
            )
        nodes[node_id] = _Flow(definition)
    document["nodes"] = nodes
    document["edges"] = [str(edge) for edge in graph.edges]
    if graph.layout:
        document["layout"] = {
            node_id: _FlowList([_number(x), _number(y)])
            for node_id, (x, y) in graph.layout.items()
        }

    return yaml.dump(
        document,
        Dumper=_Dumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


def _number(value: float) -> int | float:
    """Write 42.0 as 42, so hand-edited files stay tidy."""
    return int(value) if float(value).is_integer() else value


def dump(graph: Graph, path: str | Path) -> None:
    Path(path).write_text(dumps(graph), encoding="utf-8")
