"""Startup splash + a shared stage timeline.

The app used to show a blank screen for ~1s at launch: importing the GUI stack
(pyqtgraph, ~250 ms) and then building ``MainWindow`` (~800 ms, dominated by the
ion tab's pyqtgraph widgets) both run *before* the first window paints. This module
paints a lightweight splash within ~100 ms and narrates the startup as named stages
with per-stage elapsed times.

Two pieces:

* :class:`StageClock` — a tiny in-memory list of ``(name, start, end)`` records,
  shared by the startup path here and by background workers (see ``main._start_run``),
  and surfaced retrospectively in *Help ▸ Show timeline…*.
* :class:`StartupSplash` — a frameless, translucent progress card that renders the
  stage checklist live.

:func:`boot` is the real entry point (``smile-msi-gui``): it shows the splash *before*
importing ``main`` (hence before pyqtgraph), then hands the splash + clock to
``main.run``. Imports only PySide6 + stdlib + the light ``_appid`` helpers — never
pyqtgraph — so the splash is genuinely the first thing that runs.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from PySide6 import QtCore, QtGui, QtWidgets

# Inlined so this module never imports smile_msi.gui.common (which imports pyqtgraph and
# would defeat the whole point of painting before that ~250 ms import).
_ACCENT = "#4FA56B"
_MUTED_FG = "#9a9a9a"
_SHADOW_MARGIN = 18


class StageClock:
    """An ordered log of named stages with millisecond durations.

    Records are ``{"name", "start", "end"}`` (``end`` is ``None`` while a stage runs).
    ``start``/``end`` are :func:`time.perf_counter` values; :meth:`rows` converts them to
    display-ready ``(name, ms, running)`` tuples. A single optional ``listener`` callback
    fires on every open/close so a live view (the splash checklist, the loading card) can
    redraw without polling.
    """

    def __init__(self, listener=None):
        self.records: list[dict] = []
        self._listener = listener

    def set_listener(self, listener):
        self._listener = listener

    def _notify(self):
        if self._listener is not None:
            try:
                self._listener(self)
            except Exception:  # noqa: BLE001 — a redraw hiccup must never break startup
                pass

    def start(self, name: str) -> None:
        """Close the previous still-open stage and open ``name``. Idempotent-friendly:
        stages are appended in order, so re-``start``-ing a finished name adds a new row."""
        now = time.perf_counter()
        for r in self.records:
            if r["end"] is None:
                r["end"] = now
        self.records.append({"name": name, "start": now, "end": None})
        self._notify()

    def finish(self, name: str | None = None) -> None:
        """Close the last open stage (``name`` is accepted for symmetry but the clock is
        strictly sequential, so it always closes whichever stage is still running)."""
        now = time.perf_counter()
        for r in reversed(self.records):
            if r["end"] is None:
                r["end"] = now
                break
        self._notify()

    def finish_all(self) -> None:
        now = time.perf_counter()
        for r in self.records:
            if r["end"] is None:
                r["end"] = now
        self._notify()

    @contextmanager
    def stage(self, name: str):
        """``with clock.stage("Building workspace"): ...`` — times the block."""
        self.start(name)
        try:
            yield
        finally:
            self.finish(name)

    def rows(self):
        """``[(name, ms, running)]`` in start order — ``ms`` is elapsed-so-far for a
        still-running stage, final duration for a finished one."""
        now = time.perf_counter()
        out = []
        for r in self.records:
            end = r["end"] if r["end"] is not None else now
            out.append((r["name"], (end - r["start"]) * 1000.0, r["end"] is None))
        return out

    def total_ms(self) -> float:
        if not self.records:
            return 0.0
        now = time.perf_counter()
        end = max((r["end"] if r["end"] is not None else now) for r in self.records)
        return (end - min(r["start"] for r in self.records)) * 1000.0


class StartupSplash(QtWidgets.QWidget):
    """Frameless, translucent card that lists startup stages as they run.

    Deliberately a plain ``QWidget`` (not ``QSplashScreen``) so it can render a checklist
    with per-stage timing rather than a single bitmap. Stays on top, centres on the primary
    screen, and draws entirely with Qt primitives + text (no external asset) so it is
    frozen-bundle safe.
    """

    _PENDING, _RUNNING, _DONE = "○", "◐", "●"

    def __init__(self, stages=None):
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setFixedWidth(420 + 2 * _SHADOW_MARGIN)
        self._rows: dict[str, tuple] = {}   # name -> (glyph_label, name_label, ms_label)

        card = QtWidgets.QFrame(self)
        card.setObjectName("splashCard")
        card.setStyleSheet(
            "#splashCard{background:palette(window);"
            "border:1px solid palette(mid);border-radius:12px;}")
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 110))
        card.setGraphicsEffect(shadow)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(_SHADOW_MARGIN, _SHADOW_MARGIN, _SHADOW_MARGIN, _SHADOW_MARGIN)
        outer.addWidget(card)

        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(26, 22, 26, 20)
        lay.setSpacing(10)

        title = QtWidgets.QLabel("SMILE MSI")
        f = title.font()
        f.setPointSize(f.pointSize() + 6)
        f.setBold(True)
        title.setFont(f)
        lay.addWidget(title)

        sub = QtWidgets.QLabel("Starting up…")
        sub.setStyleSheet(f"color:{_MUTED_FG};")
        lay.addWidget(sub)
        self._subtitle = sub

        self._rows_box = QtWidgets.QVBoxLayout()
        self._rows_box.setSpacing(4)
        lay.addLayout(self._rows_box)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 0)                    # indeterminate marquee
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        bar.setStyleSheet(
            "QProgressBar{background:palette(midlight);border:none;border-radius:2px;}"
            f"QProgressBar::chunk{{background:{_ACCENT};border-radius:2px;}}")
        lay.addWidget(bar)

        for name in (stages or []):
            self._ensure_row(name)
        self.adjustSize()

    def _ensure_row(self, name: str):
        if name in self._rows:
            return self._rows[name]
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        glyph = QtWidgets.QLabel(self._PENDING)
        glyph.setFixedWidth(16)
        glyph.setStyleSheet(f"color:{_MUTED_FG};")
        lbl = QtWidgets.QLabel(name)
        lbl.setStyleSheet(f"color:{_MUTED_FG};")
        ms = QtWidgets.QLabel("")
        ms.setStyleSheet(f"color:{_MUTED_FG};")
        ms.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ms.setMinimumWidth(56)
        row.addWidget(glyph)
        row.addWidget(lbl, 1)
        row.addWidget(ms)
        self._rows_box.addLayout(row)
        self._rows[name] = (glyph, lbl, ms)
        self.adjustSize()
        return self._rows[name]

    # -- live drivers ------------------------------------------------------- #
    def render_clock(self, clock: "StageClock"):
        """Redraw every row from a :class:`StageClock` (its listener). Rows are created
        on demand so stages that weren't declared up front still appear."""
        for name, ms, running in clock.rows():
            glyph, lbl, ms_lbl = self._ensure_row(name)
            if running:
                glyph.setText(self._RUNNING)
                glyph.setStyleSheet(f"color:{_ACCENT};")
                lbl.setStyleSheet("color:palette(text);")
                ms_lbl.setText("")
                self._subtitle.setText(name + "…")
            else:
                glyph.setText(self._DONE)
                glyph.setStyleSheet(f"color:{_ACCENT};")
                lbl.setStyleSheet(f"color:{_MUTED_FG};")
                ms_lbl.setText(f"{ms:,.0f} ms")
        QtWidgets.QApplication.processEvents()

    def showEvent(self, e):  # noqa: N802 (Qt override)
        super().showEvent(e)
        scr = self.screen() or QtWidgets.QApplication.primaryScreen()
        if scr is not None:
            self.move(scr.availableGeometry().center() - self.rect().center())


def boot(argv=None, open_path=None):
    """Real GUI entry point: paint the splash before the heavy import, then run.

    Ordering is load-bearing:
      1. ``_name_macos_menubar`` / ``_set_windows_app_id`` must run *before* the
         ``QApplication`` (Cocoa) is created.
      2. The splash must ``show()`` + ``processEvents()`` *before* ``from .main import run``
         (the ~250 ms pyqtgraph import) so the first pixel lands within ~100 ms.
    """
    import sys
    from ._appid import _name_macos_menubar, _set_windows_app_id

    _name_macos_menubar()
    _set_windows_app_id()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv)

    clock = StageClock()
    splash = StartupSplash(stages=["Importing analysis engine", "Loading appearance",
                                   "Building workspace", "Scanning saved sessions"])
    clock.set_listener(splash.render_clock)
    splash.show()
    app.processEvents()                       # first pixel

    with clock.stage("Importing analysis engine"):
        from .main import run                  # the pyqtgraph import, now behind the splash

    return run(argv=argv, open_path=open_path, app=app, splash=splash, clock=clock)
