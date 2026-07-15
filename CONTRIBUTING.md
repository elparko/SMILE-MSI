# Contributing to SMILE MSI

Thanks for hacking on **SMILE MSI** (Python package: `smile_msi`). This guide covers
the practical bits — getting a dev environment, running the tests, and a couple of
gotchas that will otherwise eat an afternoon.

---

## Dev setup

We use [uv](https://docs.astral.sh/uv/) for environments and package management.

```bash
uv venv                              # create .venv (Python per requires-python >=3.10)
uv pip install -e '.[gui,dev]'       # editable install: engine + desktop app + test deps
```

The base install (`uv pip install -e .`) gives you the engine and the headless CLI.
The `[gui]` extra adds the desktop app (PySide6 + pyqtgraph), `[dev]` adds pytest, and
`[embed]` adds UMAP.

> [!IMPORTANT]
> **Don't use `uv run` right now.** A fresh `uv run …` currently fails to resolve the
> dependency tree (`zarr>=3` conflicts with the project's Python pin), so it never gets
> as far as running your command. Until that's sorted, invoke the **repo `.venv`
> interpreter directly** — e.g. `.venv/bin/python -m pytest …` or
> `.venv/bin/python -m smile_msi.gui` — rather than going through `uv run`. The
> `uv venv` / `uv pip install` steps above are fine; it's only `uv run` that's affected.

---

## Running the tests

The suite is **segmented** so you never wait on the slow Qt tests while iterating. Every
test is auto-tagged (in `tests/conftest.py`): `gui` (builds a Qt `MainWindow`) and `slow`
(heavy compute — UMAP / SHAP / cohort / 3D / registration). The `Makefile` wraps the common
loops (all set `QT_QPA_PLATFORM=offscreen`, the windowless Qt plugin):

```bash
make test           # the dev loop: ~450 pure-engine tests, no Qt, ~30s
make test-engine    # every non-Qt test (includes heavy engine compute), ~70s
make test-gui       # the Qt/MainWindow tests — slower; run before a PR
make test-analysis  # just the Analyze surface (dialog, gallery, provenance)
```

Under the hood these are pytest marker selections, so you can compose your own:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -m "not gui and not slow" -q   # fastest
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_analysis_dialog.py  # one area
```

A default **180s per-test timeout** turns a hung offscreen-Qt test into a failure instead of
a SIGKILL that takes the whole run down, and a `sessionfinish` hook exits with the real
status so a pyqtgraph teardown segfault can't turn a green run red.

### Gotcha: the full Qt suite is memory-heavy in one process

The ~500 GUI tests accumulate Qt/pyqtgraph state; running them **all in one process** can
exhaust memory. In CI (and locally) run the Qt tests **per file** (or with `-m gui` split
into batches), not as one `pytest tests/` invocation. `make test` (engine only) is always
safe to run whole.

### Gotcha: test-runner contention

Don't run a **background pytest watcher** at the same time as a GUI test run — the two
contend on the offscreen Qt platform and you'll get flaky, hard-to-attribute failures. Stop
any background watcher first, or scope it to a non-GUI subset (`-m "not gui"`).

---

## Running the app

```bash
.venv/bin/python -m smile_msi.gui
```

(Once installed, the `smile-msi` and `smile-msi-gui` console scripts launch the app
too, and `smile-msi data.imzML` opens a file directly.) Click **Load demo dataset** to
exercise every panel with a synthetic brain-like section — no data file needed.

---

## Code style

There is **no linter or formatter configured** for this repo. Please **match the style of
the surrounding code** in whatever file you're editing — naming, imports, spacing, and
docstring conventions. Keep diffs focused: change what the task needs and leave unrelated
reformatting out.

---

## Buttons & icons (one vocabulary, same place every time)

All toolbar/button icons come from **one** registry — `smile_msi/gui/icons.py` — so the
same action looks the same on every tab. Don't hand-pick a glyph or a one-off `QIcon`.

- Build an action button with the `common` factories, passing a **canonical action name**:

  ```python
  from .common import icon, icon_button, tool_button, primary_button

  icon_button("export", "Export CSV…", slot)        # text + leading download icon
  tool_button(name="export", tooltip="Export…", slot=fn)  # compact, icon-only
  primary_button("Run segmentation", fn)            # accent run button (▶ icon auto)
  some_existing_button.setIcon(icon("delete"))      # decorate a hand-built widget
  ```

- The vocabulary (one row per action) lives in `icons.SPEC`: `export, save, open, import,
  run, find, add, delete, remove, close, merge, settings, rename, copy, help, menu,
  submenu, overflow, refresh, expand, navigate, up, down`. Need a new verb? **Add a row
  there**, don't invent a glyph at the call site. `export` is download/export/save-image;
  `delete` is irreversible (pair with `confirm()`/`danger=True`); `remove` is reversible.

- **Placement:** the icon sits on the **leading (left) edge**; in a control row order reads
  *run/apply → add/save/export → navigate → delete/remove*, with **"More ▾" and the "?"
  glossary help pinned to the far right**. Icon-only buttons **must** keep a `tooltip` +
  `accessibleName` (GUI tests find buttons by name/text).

- Icons are vector (qtawesome, Font Awesome 6) and recolour to the light/dark palette
  automatically. If qtawesome is missing the app still runs — `icon()` falls back to the
  unicode glyph. Adding an icon never changes a button's `.text()`.

---

## Before you open a PR

- Run `make test` (the fast engine suite) green, plus the GUI files your change touches
  (`make test-analysis`, or the specific `tests/test_*_gui.py`).
- Keep the change scoped; don't reformat untouched code.
- Match the existing tone in user-facing docs (`README.md`, `USER_GUIDE.md`) if your
  change touches them.
