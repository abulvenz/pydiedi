# pydiedi

Sounds like pie-dee-eddy. Port of the famous [flodiedi](https://sourceforge.net/projects/flodiedi/) from launchpad.

Visual dataflow for image processing: build a graph of processing blocks, run it,
see what comes out. flodiedi was written 2010–2013 in C++/Qt4; this is the same
idea in Python, where most of its machinery turns out to be unnecessary.

**Status: phase 0 — the core runs headless. No GUI yet.**

## Why a rewrite rather than a port

flodiedi's 172 plugins comprise 58,765 lines of code, of which **3,576 are actual
logic** — 94 % is scaffolding. The `threshold` plugin is 319 lines (header,
source, CMakeLists) wrapping one OpenCV call. `gaussian_blur`, `canny`, `dilate`
and `resize` are 319 lines each for a *single* line of work.

That scaffolding exists for one reason: C++ cannot introspect itself. The
framework needs to know which ports a block has, of which type, with which
default — so flodiedi grew a declaration file (`functions.fdf`), a code
generator (`FlowFunctionParser`), `Q_PROPERTY` declarations,
`qRegisterMetaType` calls, `Q_INVOKABLE` constructors, `Q_EXPORT_PLUGIN2`
macros and a `CMakeLists.txt` per plugin, all to synthesise metadata.

In Python that metadata is simply present:

```python
@block(category="filtering")
def threshold(input: Mat, level: float = 128, max_value: float = 255,
              type: ThresholdType = ThresholdType.binary) -> Mat:
    """Binarise an image at a fixed level."""
    return cv2.threshold(input, level, max_value, type.value)[1]
```

Type hints *are* the ports. Defaults *are* the parameters. The docstring *is*
the help text. No declaration file, no generator, no build step, no `.so`.

`pydiedi blocks` prints every block with its ports, types and defaults, derived
purely from signatures — that is the whole premise, and it is the first
acceptance test.

## Install

```sh
uv sync
```

## Use

```sh
uv run pydiedi blocks                          # list blocks and their ports
uv run pydiedi blocks -v                       # with docs and enum choices
uv run pydiedi check  tests/fixtures/basic.yaml # validate without running
uv run pydiedi run    tests/fixtures/basic.yaml # execute
uv run pydiedi run diagram.yaml --save-previews out/
uv run pydiedi run diagram.yaml -n 0 --interval 0.03   # loop until interrupted
```

## The diagram format

```yaml
version: 1
name: basic

nodes:
  source:  {block: imread, params: {path: checkerboard.png}}
  gray:    {block: cvt_color, params: {code: bgr2gray}}
  blur:    {block: gaussian_blur, params: {kernel_size: 9, sigma: 2.0}}
  mask:    {block: threshold, params: {level: 100, type: binary_inv}}
  edges:   {block: canny, params: {threshold1: 40, threshold2: 120}}
  compare: {block: side_by_side, params: {title: mask_vs_edges}}

edges:
  - source.output -> gray.input
  - gray.output   -> blur.input
  - gray.output   -> edges.input      # fan-out: one output, two consumers
  - blur.output   -> mask.input
  - mask.output   -> compare.left
  - edges.output  -> compare.right

layout:                                # optional, carries no semantics
  source: [0, 0]
  gray:   [160, 0]
```

YAML was chosen for one reason: **reviewable diffs**. A visual tool whose files
cannot be read in version control becomes painful to maintain.

Four deliberate departures from flodiedi's XML:

| flodiedi | pydiedi |
|---|---|
| node identity is a `QUuid` | node id is a name the author picks |
| `__position_x/y` among the semantic properties | `layout` is its own section |
| `executionorder` persisted into the file | derived, never stored |
| four lines per connection | one line per edge |

The gain is not brevity. `SampleDiagrams/wallpaper.xml` contains four
`cvtColor` blocks, two `threshold` and two `multiply`, distinguishable only by
36-character UUIDs. Here they read `gray_live`, `gray_ref`, `gray_warp` — the
names carry the intent.

Relative paths resolve against the **diagram file**, not the working directory,
so a diagram and its data stay portable. flodiedi had no such notion, and the
result is visible in its own repository: `loadPointCloudplugin` shipped with
`/home/kochas/schnecke.pcd` compiled into it.

## Architecture

```
pydiedi/
├── core/          no Qt, no cv2 — graph, executor, blocks, file format
├── blocks/        the block library; cv2 and numpy, never a GUI toolkit
├── gui/           phase 3, PySide6
└── cli.py         headless runner
```

One rule shapes everything: **`core` never imports a GUI toolkit, and blocks
never create widgets.** A block that wants to display something returns a
`Preview` — a description of what to show. The Qt renderer turns it into a
`QGraphicsPixmapItem`, a web renderer would send a PNG, a headless run ignores
it.

flodiedi lost this property. Its model base class `FlowDiagramBlock` declared
`createExtraWidget()`, `createDisplayWidget()` and
`renderAdditionalStuff(QGraphicsItem*)`, so the model knew the view. Twenty
plugins used it, which forced a global `usesGui_` flag to special-case headless
execution and produced `setPixmap()` calls from the worker thread — tolerated by
Qt4, fatal in Qt5 and Qt6.

`tests/test_core_is_gui_free.py` enforces the rule with two independent checks:
a subprocess import that inspects `sys.modules`, and an AST scan that also
catches lazy or unreachable imports.

## Blocks

A block is a function. Every parameter becomes an input port; those with
defaults are optional and editable, those without must be connected. flodiedi
split these into `INPORT_` and `PARAM_` and then undermined the split itself —
`setInPort()` let the user promote any property to a port at runtime. Unifying
them is simpler and strictly more capable.

Several outputs use a `NamedTuple`, so ports have names and are wired by name
rather than by position:

```python
class Frame(NamedTuple):
    image: Mat
    fps: int

@block(category="imageio")
def video_file(path: Path) -> Frame: ...
```

Blocks may be declared anywhere — a module, a function, a notebook cell.

## Test

```sh
uv run pytest            # 112 tests
```

## Known limitations (phase 0)

- **No GUI.** Diagrams are written by hand or generated; phase 3 brings PySide6.
- **Stateful blocks are not supported.** flodiedi's `VideoFile` held a
  `VideoCapture` as a member; such blocks will be classes. The decorator
  currently refuses a class with an explicit message rather than
  misinterpreting it.
- **Execution is serial.** One sweep in topological order, as in flodiedi.
  Running independent branches in parallel is feasible — `cv2` releases the
  GIL — but correctness first.
- **Type compatibility is strict.** Identical types only. flodiedi's conversion
  table (`Plugins/datatypes`: `double->int`, `Mat->Mat1f`) arrives once there is
  a second Mat-like type to convert between.
- **Eight blocks.** The remaining ~140 are mechanical; the 3,576 lines of
  hand-written logic in flodiedi's plugins are the asset worth translating
  carefully rather than generating blindly.

## Licence

GPLv3, as flodiedi was. See [LICENSE](LICENSE).

flodiedi was written by Andreas Koch, Bernd Eckstein, Jan Winkler, Daniel Di
Marco, Tobias and others; parts carry University of Stuttgart copyright. None
of that code is reused here, but the semantics of the blocks are derived from
it.
