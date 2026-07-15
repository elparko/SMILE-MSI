"""Canonical action → icon vocabulary for the whole app.

This is the single source of truth that makes the SAME action look the SAME on every
tab (the recurring "download is ⤓ here but a plain 'Export…' there" inconsistency).
It mirrors the centralize-don't-one-off rule already used for ``ACCENT`` / ``colormap()``
/ ``eye_icon()`` in :mod:`smile_msi.gui.common`.

How it works
------------
* :func:`icon` resolves a canonical *action name* (``"export"``, ``"run"`` …) to a
  themed :class:`QIcon`, backed by `qtawesome <https://github.com/spyder-ide/qtawesome>`_
  (Font Awesome 6) and recoloured to the live palette so glyphs read in light *and* dark.
* If qtawesome is not installed the app still runs: :func:`icon` degrades to a pixmap
  painted from the action's **unicode fallback glyph** (the very glyphs the app used
  before — ⤓ ▶ ＋ …), so nothing goes blank.
* qtawesome is imported **lazily** (first :func:`icon` call, when a tab is built) so it
  never weighs on app startup — consistent with the lazy-GUI-import perf work.

Placement convention (so an action lives in the *same place* on every screen)
-----------------------------------------------------------------------------
* The icon sits on the **leading (left) edge** of its button — automatic via
  ``setIcon`` / ``ToolButtonTextBesideIcon``. Never trail an action icon.
* In a control row, order reads left→right: **primary run/apply → constructive
  (add / save / export) → navigation → destructive (delete / remove)**, with the
  **"More ▾" overflow and the "?" glossary help pinned to the far right**.
* Compact, universally-understood verbs (export ⤓, refresh ↺, remove ✕) are
  **icon-only** tool buttons (always with a tooltip + accessibleName); labelled
  actions keep **text + a leading icon**.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

# --------------------------------------------------------------------------- #
# The vocabulary — ONE table. Each canonical action maps to (qtawesome id,
# unicode fallback). Change an icon for the whole app by editing one row here.
# Synonyms are deliberately collapsed (download/export/save-image → "export";
# open/load/import-data → "open") so the same intent always renders the same.
# --------------------------------------------------------------------------- #
SPEC: dict[str, tuple[str, str]] = {
    # data out / in
    "export":   ("fa6s.download",           "⤓"),   # ⤓  the #1 icon (download/export/save-image)
    "save":     ("fa6s.floppy-disk",        "⤓"),   # persist into the app's own store
    "open":     ("fa6s.folder-open",        "\U0001F4C2"),  # 📂 open/load a dataset/session
    "import":   ("fa6s.file-import",        "\U0001F4C2"),  # pull in a reference library
    # compute / discovery
    "run":      ("fa6s.play",               "▶"),   # ▶  (== old RUN_GLYPH) execute now
    "find":     ("fa6s.magnifying-glass",   "\U0001F50D"),  # 🔍 discover/search/highlight
    "analysis": ("fa6s.flask",              "⚗"),  # ⚗ open an analysis/stats config page (NOT a folder)
    "pick":     ("fa6s.list-check",         "☑"),  # ☑ choose a subset from a list (regions/groups to scope an analysis)
    # object lifecycle
    "add":      ("fa6s.plus",               "＋"),   # ＋ create/append a new item
    "delete":   ("fa6s.trash-can",          "\U0001F5D1"),  # 🗑 irreversibly destroy (pair with confirm()/danger)
    "remove":   ("fa6s.xmark",              "✕"),   # ✕  reversibly detach/clear from a view
    "close":    ("fa6s.xmark",              "✕"),   # dismiss a dialog/panel
    "merge":    ("fa6s.code-merge",         "⨝"),   # combine existing items into one
    # configure
    "settings": ("fa6s.gear",               "⚙"),   # ⚙ open a config / colour / rename dialog
    "rename":   ("fa6s.pen",                "✎"),   # rename in place (opt. alt to settings)
    "copy":     ("fa6s.copy",               "⧉"),   # copy to clipboard
    # affordances / markers
    "help":     ("fa6s.circle-question",    "?"),        # glossary / about / guide
    "menu":     ("fa6s.chevron-down",       "▾"),   # ▾ (== old MENU_GLYPH) this button drops a menu
    "submenu":  ("fa6s.chevron-right",      "▸"),   # ▸ opens a submenu
    "overflow": ("fa6s.ellipsis",           "»"),   # » / ... compact overflow of extra actions
    "refresh":  ("fa6s.arrow-rotate-right", "↺"),   # ↺ recompute / reset-to-default / fit / reload
    "expand":   ("fa6s.angles-right",       "▸"),   # expand a composite into its parts
    "zoom-in":  ("fa6s.magnifying-glass-plus",  "＋"),  # increase magnification (NOT "add")
    "zoom-out": ("fa6s.magnifying-glass-minus", "－"),  # decrease magnification (NOT "remove"/✕)
    # navigation / reorder
    "navigate": ("fa6s.arrow-right",        "→"),   # → go-to / send-to another view
    "up":       ("fa6s.arrow-up",           "↑"),   # ↑ reorder up
    "down":     ("fa6s.arrow-down",         "↓"),   # ↓ reorder down
}

# Cache resolved icons per (name, colour) — qtawesome rendering and the fallback
# painter both cost a little, and the same handful of icons are reused on every tab.
# Cleared by on_theme_changed() so a live Light/Dark switch recolours every icon.
_CACHE: dict[tuple[str, str], QtGui.QIcon] = {}

_qta = None              # the qtawesome module once imported (lazily)
_qta_tried = False       # so a failed/absent import is attempted only once


def _qta_mod():
    """Import qtawesome lazily and at most once (protects app startup time)."""
    global _qta, _qta_tried
    if not _qta_tried:
        _qta_tried = True
        try:
            import qtawesome as _m
            _qta = _m
        except Exception:                       # noqa: BLE001 — absence is a supported mode
            _qta = None
    return _qta


def glyph(name: str) -> str:
    """The unicode fallback glyph for an action — for text contexts / no-qtawesome builds.

    e.g. ``glyph("run") == "▶"``. Used to keep ``RUN_GLYPH`` / ``MENU_GLYPH`` defined as a
    single vocabulary and to seed a tool button's fallback text."""
    spec = SPEC.get(name)
    return spec[1] if spec else ""


def _fg_color() -> str:
    """The current palette's button-text colour, so icons stay legible in light & dark."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return "#888888"
    return app.palette().color(QtGui.QPalette.ButtonText).name()


def _accent() -> str:
    """The app accent (selection green). Imported lazily to avoid a common↔icons cycle."""
    try:
        from .common import ACCENT
        return ACCENT
    except Exception:                           # noqa: BLE001
        return "#4FA56B"


def _glyph_icon(ch: str, color: str, size: int = 18) -> QtGui.QIcon:
    """Paint a unicode glyph into a transparent pixmap — the no-qtawesome fallback, so an
    icon-only button still shows its symbol instead of going blank."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.TextAntialiasing)
    p.setPen(QtGui.QColor(color))
    f = p.font()
    f.setPixelSize(int(size * 0.82))
    p.setFont(f)
    p.drawText(pm.rect(), QtCore.Qt.AlignCenter, ch)
    p.end()
    return QtGui.QIcon(pm)


def icon(name: str, *, color: str | None = None, active: bool = False) -> QtGui.QIcon:
    """Resolve a canonical action *name* to a themed :class:`QIcon`.

    ``color`` forces a hex colour; ``active=True`` tints with the accent (for a
    toggled-on / current state). With qtawesome present this is a crisp recoloured
    vector glyph; without it, a pixmap painted from the action's unicode fallback.
    Unknown names return a null icon (callers no-op on a null icon). Cached per
    (name, colour) and invalidated by :func:`on_theme_changed`."""
    spec = SPEC.get(name)
    if spec is None:
        return QtGui.QIcon()
    col = color or (_accent() if active else _fg_color())
    key = (name, col)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    qid, fallback = spec
    qta = _qta_mod()
    if qta is not None:
        try:
            ico = qta.icon(qid, color=col)
        except Exception:                       # noqa: BLE001 — bad id / no app yet → fall back
            ico = _glyph_icon(fallback, col)
    else:
        ico = _glyph_icon(fallback, col)
    _CACHE[key] = ico
    return ico


def on_theme_changed() -> None:
    """Drop the icon cache so the next repaint pulls palette-correct colours.

    Call from :func:`smile_msi.gui.common.set_theme` (the live Light/Dark switch) — the
    same path that re-themes open plots. Widgets re-pull their icon on the next paint;
    for already-built buttons that cache the QIcon, callers may re-set it, but most are
    rebuilt when their tab is revisited."""
    _CACHE.clear()
