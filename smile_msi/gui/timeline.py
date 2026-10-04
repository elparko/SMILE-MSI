"""Performance timeline — surface, in-app, what the app spent time on.

Two sources feed one dialog:

* the in-memory :class:`~smile_msi.gui.splash.StageClock` (``win._startup_clock``),
  which holds the startup breakdown — imports, ``MainWindow`` construction, the
  saved-session scan, the launcher — as structured ``(name, ms)`` rows; and
* ``~/smile_msi_perf.log`` (written by ``MainWindow._perf``), which records every
  post-startup operation as ``START`` / ``STAGE`` / ``DONE`` / ``ERROR`` lines.

The log side is parsed by :func:`parse_perf_log` / :func:`summarize_operations`
— pure functions with no Qt dependency, so they are unit-testable on their own —
and :class:`TimelineDialog` renders both. Reached via **Help ▸ Show timeline…**.
"""
from __future__ import annotations

import os
import re

from PySide6 import QtCore, QtWidgets

from .common import MUTED_QSS, install_table_export, section_title

# ``[HH:MM:SS] KIND   <label>[ :: <stage>][  (1.23s)]`` — the shape _perf writes.
_LINE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s+"
    r"(?P<kind>START|STAGE|DONE|ERROR|PASS)\s+"
    r"(?P<rest>.*)$"
)
_SECONDS = re.compile(r"\((?P<sec>\d+(?:\.\d+)?)s\)\s*$")


def parse_perf_log(text: str) -> list[dict]:
    """Parse ``_perf`` output into ordered event dicts.

    Each event is ``{"time", "kind", "label", "stage", "seconds"}``. ``stage`` is set
    only for ``STAGE`` lines (the text after ``::``); ``seconds`` only when the line
    carried a ``(1.23s)`` suffix (``DONE`` / ``ERROR``). Unrecognized lines are skipped,
    so a truncated or hand-edited log never raises."""
    events: list[dict] = []
    for raw in text.splitlines():
        m = _LINE.match(raw.strip())
        if not m:
            continue
        kind = m.group("kind")
        rest = m.group("rest").strip()
        stage = None
        seconds = None
        sm = _SECONDS.search(rest)
        if sm:
            seconds = float(sm.group("sec"))
            rest = rest[: sm.start()].strip()
        if kind == "STAGE" and "::" in rest:
            label, stage = (p.strip() for p in rest.split("::", 1))
        else:
            label = rest
        events.append({"time": m.group("time"), "kind": kind, "label": label,
                       "stage": stage, "seconds": seconds})
    return events


def summarize_operations(events: list[dict]) -> list[dict]:
    """Fold a flat event list into one record per operation, newest last.

    Pairs each ``START <label>`` with the following ``DONE``/``ERROR`` for that label,
    collecting the ``STAGE`` names seen in between. Result rows are
    ``{"time", "label", "seconds", "stages", "status"}`` where ``status`` is
    ``"done"`` / ``"error"`` / ``"running"`` (still open at end of log)."""
    open_ops: dict[str, dict] = {}
    done: list[dict] = []
    for ev in events:
        label = ev["label"]
        if ev["kind"] == "START":
            open_ops[label] = {"time": ev["time"], "label": label, "seconds": None,
                               "stages": [], "status": "running"}
        elif ev["kind"] == "STAGE":
            op = open_ops.get(label)
            if op is not None and ev["stage"]:
                op["stages"].append(ev["stage"])
        elif ev["kind"] == "PASS":
            # a whole-slide read reported by the dataset layer (MSIDataset.on_pass): its own
            # completed row, so a GUI-thread stall shows up even with no START around it
            done.append({"time": ev["time"], "label": label, "seconds": ev["seconds"],
                         "stages": [], "status": "done"})
        elif ev["kind"] in ("DONE", "ERROR"):
            op = open_ops.pop(label, None)
            if op is None:      # a DONE with no matching START (log rotated mid-op)
                op = {"time": ev["time"], "label": label, "stages": []}
            op["seconds"] = ev["seconds"]
            op["status"] = "done" if ev["kind"] == "DONE" else "error"
            done.append(op)
    # operations still running at end of log keep their place after the finished ones
    done.extend(open_ops.values())
    return done


def read_perf_log(path: str | None = None) -> list[dict]:
    """Load and summarize the perf log at ``path`` (default: ``$SMILE_MSI_HOME`` /
    home ``smile_msi_perf.log``). Missing / unreadable log → ``[]``."""
    if path is None:
        home = os.environ.get("SMILE_MSI_HOME") or os.path.expanduser("~")
        path = os.path.join(home, "smile_msi_perf.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    return summarize_operations(parse_perf_log(text))


def _fmt_ms(ms: float) -> str:
    return f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _fmt_seconds(sec) -> str:
    return "—" if sec is None else (f"{sec:.2f} s" if sec >= 1 else f"{sec*1000:.0f} ms")


class TimelineDialog(QtWidgets.QDialog):
    """Read-only view of the startup breakdown + recent operations.

    ``startup_rows`` is ``StageClock.rows()`` (``[(name, ms, running)]``); ``operations``
    is :func:`summarize_operations` output. Both default to empty so the dialog is safe
    to open before anything has been recorded."""

    def __init__(self, parent=None, startup_rows=None, startup_total_ms=0.0,
                 operations=None):
        super().__init__(parent)
        self.setWindowTitle("SMILE MSI — Performance timeline")
        self.resize(520, 460)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(8)

        lay.addWidget(section_title("Startup"))
        startup_rows = list(startup_rows or [])
        if startup_rows:
            lay.addWidget(self._startup_table(startup_rows))
            total = QtWidgets.QLabel(f"Total: {_fmt_ms(startup_total_ms)}")
            total.setStyleSheet(MUTED_QSS)
            lay.addWidget(total)
        else:
            lay.addWidget(self._muted("No startup timing was recorded for this run."))

        lay.addWidget(section_title("Recent operations"))
        operations = list(operations or [])
        if operations:
            lay.addWidget(self._operations_table(operations), 1)
        else:
            lay.addWidget(self._muted(
                "No operations recorded yet — load a sample or apply preprocessing, "
                "then re-open this window."), 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)

    def _muted(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(MUTED_QSS)
        lbl.setWordWrap(True)
        return lbl

    def _startup_table(self, rows) -> QtWidgets.QTableWidget:
        t = _readonly_table(["Stage", "Duration"], len(rows))
        for r, (name, ms, running) in enumerate(rows):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            dur = "running…" if running else _fmt_ms(ms)
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(dur))
        t.resizeColumnsToContents()
        return t

    def _operations_table(self, operations) -> QtWidgets.QTableWidget:
        # newest first so the last thing the user did is at the top
        ops = list(reversed(operations))
        t = _readonly_table(["Time", "Operation", "Duration", "Stages"], len(ops))
        for r, op in enumerate(ops):
            t.setItem(r, 0, QtWidgets.QTableWidgetItem(op.get("time", "")))
            label = op["label"]
            if op.get("status") == "error":
                label = "⚠ " + label
            elif op.get("status") == "running":
                label = "… " + label
            t.setItem(r, 1, QtWidgets.QTableWidgetItem(label))
            t.setItem(r, 2, QtWidgets.QTableWidgetItem(_fmt_seconds(op.get("seconds"))))
            t.setItem(r, 3, QtWidgets.QTableWidgetItem(" → ".join(op.get("stages", []))))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        return t


def _readonly_table(headers, rows) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(rows, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    t.setAlternatingRowColors(True)
    install_table_export(t, stem="startup_timeline", title="Export timeline")
    return t
