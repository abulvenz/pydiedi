"""Command line interface.

``pydiedi blocks`` is the acceptance test for the whole premise: it prints
every block with its ports, types and defaults, derived purely from function
signatures. There is no registry file, no ``.fdf``, no generated code.

``pydiedi run`` replaces flodiedi's ``FlowExecute``, which crashed on a bad
path because it called ``FlowDiagram::load()`` -- a function that throws --
without a try/catch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import io, registry
from .core.block import DEFAULT_OUTPUT_NAME
from .core.executor import CycleError, Executor, NodeExecutionError
from .core.graph import ValidationError
from .core.types import CoercionError


def _cmd_blocks(args: argparse.Namespace) -> int:
    registry.discover()
    grouped = registry.by_category()
    if not grouped:
        print("no blocks registered", file=sys.stderr)
        return 1

    total = sum(len(specs) for specs in grouped.values())
    for category, specs in grouped.items():
        print(f"\n{category}")
        print("-" * len(category))
        for spec in specs:
            print(f"  {spec.signature()}")
            if args.verbose:
                if spec.doc:
                    print(f"      {spec.doc.splitlines()[0]}")
                for port in spec.inputs:
                    marker = "required" if port.required else "optional"
                    line = f"      in  {port.name}: {port.type_name} ({marker})"
                    if port.choices:
                        line += f" one of: {', '.join(port.choices)}"
                    print(line)
                for port in spec.outputs:
                    print(f"      out {port.name}: {port.type_name}")
    print(f"\n{total} blocks in {len(grouped)} categories")
    return 0


def _load_and_check(path: Path) -> tuple[object, int]:
    registry.discover()
    graph = io.load(path)
    graph.validate()
    return graph, 0


def _cmd_check(args: argparse.Namespace) -> int:
    graph, _ = _load_and_check(args.file)
    from .core.executor import topological_order

    order = topological_order(graph)  # type: ignore[arg-type]
    nodes = graph.nodes  # type: ignore[attr-defined]
    print(f"{args.file}: ok -- {len(nodes)} nodes, {len(graph.edges)} edges")  # type: ignore[attr-defined]
    print(f"execution order: {' -> '.join(order)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    graph, _ = _load_and_check(args.file)
    executor = Executor(graph, validate=False)  # type: ignore[arg-type]

    if args.verbose:
        print(f"order: {' -> '.join(executor.order)}", file=sys.stderr)

    result = executor.run(iterations=args.iterations, interval=args.interval)

    for index, preview in enumerate(result.previews):
        label = preview.title or f"preview {index + 1}"
        shape = "empty" if preview.image is None else "x".join(map(str, preview.image.shape))
        print(f"{label}: {shape}")
        if args.save_previews and preview.image is not None:
            import cv2

            args.save_previews.mkdir(parents=True, exist_ok=True)
            name = (preview.title or f"preview_{index + 1}").replace("/", "_")
            out = args.save_previews / f"{name}.png"
            cv2.imwrite(str(out), preview.image)
            print(f"  written to {out}")

    if args.verbose:
        print(f"{len(result.order)} nodes executed", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pydiedi",
        description="Visual dataflow for image processing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="more detail on stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    blocks = sub.add_parser("blocks", help="list available blocks with their ports")
    blocks.set_defaults(func=_cmd_blocks)

    check = sub.add_parser("check", help="validate a diagram without running it")
    check.add_argument("file", type=Path)
    check.set_defaults(func=_cmd_check)

    run = sub.add_parser("run", help="execute a diagram")
    run.add_argument("file", type=Path)
    run.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="sweeps to perform; 0 or less runs until interrupted (default: 1)",
    )
    run.add_argument(
        "--interval", type=float, default=0.0, help="seconds to wait between sweeps"
    )
    run.add_argument(
        "--save-previews",
        type=Path,
        metavar="DIR",
        help="write each preview to DIR as a PNG",
    )
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        io.DiagramSyntaxError,
        ValidationError,
        CycleError,
        CoercionError,
        registry.UnknownBlockError,
    ) as exc:
        # Expected, actionable failures: report them plainly without a
        # traceback, which is noise for a malformed diagram file.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except NodeExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("(run with -v for the full traceback)", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
