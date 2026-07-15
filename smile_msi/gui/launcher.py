"""Startup launcher — choose how to begin a SMILE MSI session.

Shown once from :func:`smile_msi.gui.main.run` (never from ``MainWindow.__init__``,
so headless tests that build the window directly are unaffected). Lists the managed
per-sample sessions newest-first and offers Open dataset / Open session / Load demo /
Start fresh. The choice is reported back via :attr:`StartupDialog.result_action` as a
``(kind, payload)`` pair — ``"session"``/``"dataset"`` carry a path, ``"demo"``/``"fresh"``
carry ``None`` — which ``run()`` dispatches. Closing the dialog ⇒ start fresh.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import session
from . import common, filedialogs

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "smile_msi_logo.png"


def _relative_time(mtime) -> str:
    try:
        dt = max(0.0, time.time() - float(mtime))
    except (TypeError, ValueError):
        return ""
    for unit, secs in (("d", 86400), ("h", 3600), ("m", 60)):
        if dt >= secs:
            return f"{int(dt // secs)}{unit} ago"
    return "just now"


class StartupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SMILE MSI")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.result_action = ("fresh", None)   # (kind, payload): fresh | dataset | session | demo

        v = QtWidgets.QVBoxLayout(self)

        pix = QtGui.QPixmap(str(LOGO_PATH)) if LOGO_PATH.exists() else QtGui.QPixmap()
        if not pix.isNull():
            logo = QtWidgets.QLabel()
            logo.setAlignment(QtCore.Qt.AlignCenter)
            logo.setPixmap(pix.scaledToWidth(440, QtCore.Qt.SmoothTransformation))
            v.addWidget(logo)

        title = QtWidgets.QLabel(
            "<div align='center'><b>Spatial Mass Imaging of Lipid Environments</b><br>"
            "<span style='color:gray'>Open a sample to continue, or start fresh.</span></div>")
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(title)

        sessions = session.list_managed()
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.itemDoubleClicked.connect(self._open_selected)
        for s in sessions:
            px = f"{s['n_pixels']:,} px" if s.get("n_pixels") else "—"
            it = QtWidgets.QListWidgetItem(
                f"{s['name']}    ·    {px}  ·  {s['n_features']} features  ·  "
                f"{_relative_time(s['mtime'])}")
            it.setData(QtCore.Qt.UserRole, s["path"])
            self.list.addItem(it)
        if sessions:
            self.list.setCurrentRow(0)
            v.addWidget(QtWidgets.QLabel("Recent samples"))
            self.list.setToolTip(
                "Your auto-saved per-sample sessions, newest first. Double-click a row "
                "(or select it and press 'Open selected sample') to resume that analysis.")
            v.addWidget(self.list, 1)
        else:
            hint = QtWidgets.QLabel("No saved samples yet — open a dataset to begin.")
            hint.setEnabled(False)
            v.addWidget(hint)

        # When there are saved samples, "Open selected sample" below is the obvious next
        # step (accent-styled). With none, a first-time user has no sample to resume, so
        # make "Load demo" the standout start instead.
        row = QtWidgets.QHBoxLayout()
        b_data = common.button("Open dataset…", self._open_dataset, icon="open")
        b_data.setToolTip(
            "Import a new imaging dataset (.imzML, or a CSV/TSV/TXT peak table) and start "
            "a fresh session for it.")
        b_sess = common.button("Open session file…", self._open_session_file, icon="open")
        b_sess.setToolTip(
            "Reopen a saved analysis from a session .json file — regions, features, "
            "segmentation and report come back exactly as you left them.")
        b_demo = common.button("Load demo", self._load_demo, primary=not sessions, icon="open")
        b_demo.setToolTip(
            "Load the bundled demo dataset to explore SMILE MSI without importing your "
            "own data.")
        b_fresh = common.button("Start fresh", self._start_fresh, icon="open")
        b_fresh.setToolTip(
            "Open the app with no dataset loaded; import data later from the File menu.")
        for b in (b_data, b_sess, b_demo):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(b_fresh)
        v.addLayout(row)

        if sessions:
            b_open = common.primary_button("Open selected sample", self._open_selected, action="open")
            b_open.setToolTip(
                "Resume the sample highlighted in 'Recent samples' above (or double-click "
                "it in the list).")
            b_open.setDefault(True)
            v.addWidget(b_open)

        # footer: running version + an unobtrusive manual update check (Help menu mirror)
        from .. import __version__
        foot = QtWidgets.QHBoxLayout()
        ver = QtWidgets.QLabel(f"SMILE MSI v{__version__}")
        ver.setEnabled(False)
        foot.addWidget(ver)
        foot.addStretch(1)
        b_upd = QtWidgets.QPushButton("Check for updates…")
        b_upd.setIcon(common.icon("refresh"))
        b_upd.setToolTip("See if a newer SMILE MSI release is available on GitHub")
        b_upd.clicked.connect(self._check_for_updates)
        foot.addWidget(b_upd)
        v.addLayout(foot)

    def _check_for_updates(self):
        from .. import __version__
        from . import update_check
        update_check.check_for_updates(self, __version__)

    # --- each action records the choice, then closes the dialog --- #
    def _open_selected(self, *_):
        it = self.list.currentItem()
        if it is None:
            return
        self.result_action = ("session", it.data(QtCore.Qt.UserRole))
        self.accept()

    def _open_dataset(self):
        path, _ = filedialogs.get_open_file_name(
            self, "Open dataset", "", "Imaging data (*.imzML *.csv *.tsv *.txt)")
        if path:
            self.result_action = ("dataset", path)
            self.accept()

    def _open_session_file(self):
        path, _ = filedialogs.get_open_file_name(self, "Open session", "", "Session (*.json)")
        if path:
            self.result_action = ("session", path)
            self.accept()

    def _load_demo(self):
        self.result_action = ("demo", None)
        self.accept()

    def _start_fresh(self):
        self.result_action = ("fresh", None)
        self.accept()
