"""Data ▸ Manage caches… — what the managed store holds, what can go, and where it lives.

The store (``library.home_dir()``, default ``~/.smile-msi``) keeps one JSON session per
slide plus its companions: the fast cube (``.cube.zarr``, ~1 GB for a large slide, and up to
~7 GB of temporary spill while it builds), the analysis-run payloads and thumbnails. This
dialog lists them with sizes, marks what is safe to delete (orphaned companions, duplicate
sessions from the old fingerprint drift, legacy npz cubes, interrupted-build leftovers), and
lets the whole store be redirected to another disk.
"""
from __future__ import annotations

import os
import shutil

from PySide6 import QtCore, QtGui, QtWidgets

from .. import library, session
from . import filedialogs
from .common import MUTED_QSS, install_table_export, section_title


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


class CacheDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, build_in_flight=None, dataset_loaded=None):
        super().__init__(parent)
        self.setWindowTitle("Manage caches")
        self.resize(820, 520)
        # callables so the dialog reads the live state when the user acts, not at open time
        self._build_in_flight = build_in_flight or (lambda: False)
        self._dataset_loaded = dataset_loaded or (lambda: False)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(section_title("Managed store"))
        self.where = QtWidgets.QLabel()
        self.where.setWordWrap(True)
        self.where.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.where)
        hint = QtWidgets.QLabel(
            "Sessions hold your ROIs, feature lists and analyses — keep them. A cube is a "
            "cache: deleting one only means the next open of that slide rebuilds it. "
            "Building a large slide's cube needs roughly its .ibd size in temporary space "
            "here. To keep caches on another disk use Change folder… (or set "
            "$SMILE_MSI_HOME before launching).")
        hint.setWordWrap(True)
        hint.setStyleSheet(MUTED_QSS)
        lay.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Kind", "Status", "Size"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        install_table_export(self.table, self, stem="cache_entries", title="Export cache list")
        lay.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.b_removable = QtWidgets.QPushButton("Select removable")
        self.b_removable.clicked.connect(self._select_removable)
        self.b_delete = QtWidgets.QPushButton("Delete selected…")
        self.b_delete.clicked.connect(self._delete_selected)
        self.b_reveal = QtWidgets.QPushButton("Show folder")
        self.b_reveal.clicked.connect(self._reveal)
        self.b_move = QtWidgets.QPushButton("Change folder…")
        self.b_move.clicked.connect(self._change_folder)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(self.accept)
        for b in (self.b_removable, self.b_delete, self.b_reveal, self.b_move):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.refresh()

    # ---- data ------------------------------------------------------------ #
    def refresh(self):
        self.rows = session.cache_inventory()
        total = sum(r["size"] for r in self.rows)
        removable = sum(r["size"] for r in self.rows if r["removable"])
        home = library.home_dir()
        via = ("$SMILE_MSI_HOME" if os.environ.get("SMILE_MSI_HOME")
               else "redirect file" if library.home_redirect() else "default")
        self.where.setText(
            f"<b>{session.sessions_dir()}</b> ({via}) — {len(self.rows)} entries, "
            f"{_fmt_size(total)} in use, {_fmt_size(removable)} removable. "
            f"Store root: {home}")
        self.table.setRowCount(0)
        for r in self.rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            cells = [r["name"], r["kind"], r["status"], _fmt_size(r["size"])]
            for j, text in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(text)
                if j == 3:
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if r["removable"]:
                    it.setForeground(QtGui.QBrush(QtGui.QColor("#b8860b")))
                it.setData(QtCore.Qt.UserRole, r["path"])
                self.table.setItem(i, j, it)

    def _selected_paths(self) -> list[str]:
        return [self.table.item(ix.row(), 0).data(QtCore.Qt.UserRole)
                for ix in self.table.selectionModel().selectedRows()]

    # ---- actions ---------------------------------------------------------- #
    def _select_removable(self):
        self.table.clearSelection()
        for i, r in enumerate(self.rows):
            if r["removable"]:
                self.table.selectRow(i)
        if not any(r["removable"] for r in self.rows):
            QtWidgets.QMessageBox.information(self, "Manage caches", "Nothing to clean up.")

    def _delete_selected(self):
        paths = self._selected_paths()
        if not paths:
            return
        by_path = {r["path"]: r for r in self.rows}
        if self._build_in_flight() and any(by_path[p]["kind"] == "temporary" for p in paths):
            QtWidgets.QMessageBox.warning(
                self, "Manage caches",
                "A fast-cache build is running — its temporary files can't be deleted now.")
            return
        keep = [p for p in paths if not by_path[p]["removable"]]
        msg = f"Delete {len(paths)} item(s) ({_fmt_size(sum(by_path[p]['size'] for p in paths))})?"
        if keep:
            msg += (f"\n\n{len(keep)} of them are marked in use (sessions with your ROIs, or "
                    "cubes for slides you still open). Deleting a session loses its analysis; "
                    "deleting a cube only forces a rebuild.")
        if QtWidgets.QMessageBox.question(self, "Delete caches", msg) != QtWidgets.QMessageBox.Yes:
            return
        deleted, failed = session.delete_cache_paths(paths)
        self.refresh()
        note = f"Deleted {len(deleted)} item(s)."
        if failed:
            note += f" {len(failed)} could not be deleted (in use or permission denied)."
        QtWidgets.QMessageBox.information(self, "Manage caches", note)

    def _reveal(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(session.sessions_dir()))

    def _change_folder(self):
        if self._build_in_flight():
            QtWidgets.QMessageBox.warning(self, "Change folder",
                                          "Wait for the running fast-cache build to finish first.")
            return
        cur = library.home_dir()
        new = filedialogs.get_existing_directory(self, "Folder for SMILE MSI sessions and caches")
        if not new:
            return
        new = os.path.abspath(new)
        if os.path.abspath(cur) == new:
            return
        if os.environ.get("SMILE_MSI_HOME"):
            QtWidgets.QMessageBox.information(
                self, "Change folder",
                "$SMILE_MSI_HOME is set and overrides this setting. Unset it (or change it) "
                "and launch again.")
            return
        move_now = not self._dataset_loaded()
        msg = (f"Use\n{new}\nfor sessions and caches from now on?\n\n"
               + ("The current store will be moved there now." if move_now else
                  "A dataset is open, so the existing files stay where they are; close the "
                  "dataset and use this again to move them.")
               + "\n\nRestart SMILE MSI afterwards.")
        if QtWidgets.QMessageBox.question(self, "Change folder", msg) != QtWidgets.QMessageBox.Yes:
            return
        moved, failed = 0, []
        if move_now:
            os.makedirs(new, exist_ok=True)
            for fn in os.listdir(cur):
                if fn == "home":
                    continue              # the redirect pointer stays in the default folder
                src, dst = os.path.join(cur, fn), os.path.join(new, fn)
                try:
                    shutil.move(src, dst)
                    moved += 1
                except (OSError, shutil.Error):
                    failed.append(fn)
        library.set_home_redirect(new)
        self.refresh()
        note = f"Store now at {new}."
        if move_now:
            note += f" Moved {moved} item(s)."
        if failed:
            note += f" {len(failed)} could not be moved: {', '.join(failed[:5])}"
        note += " Restart SMILE MSI to finish."
        QtWidgets.QMessageBox.information(self, "Change folder", note)
