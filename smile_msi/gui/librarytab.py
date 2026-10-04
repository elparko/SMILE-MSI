"""LibraryTabMixin — the "Feature lists" manager: manage every feature list this sample has.

Demoted off the tab strip to a modeless dialog (plan 24 Phase 6), opened from the Features
⋯ menu's "Manage feature lists…" and by reveal_view("Feature lists").

Feature lists are no longer a global cross-dataset store; they live in the loaded
sample's session (auto-saved). This surface mirrors the dock's feature-set selector: it
lists the working scopes (the whole-slide set + each region's own picked list) *and*
the saved ★ lists, so the two surfaces never disagree about what exists. From here you
can load any of them into the working set, export to CSV, or delete a saved list.
Regions are managed from the right-hand Regions panel and persist with the session too,
so they have no tab here.
"""
from __future__ import annotations

import csv

from PySide6 import QtCore, QtGui, QtWidgets

from .common import add_copy_actions, confirm, fill_table, tab_page, icon
from . import filedialogs


class LibraryTabMixin:
    def _build_feature_lists_dialog(self):
        """Build the modeless Feature-lists manager (hidden). Constructed at window build so the
        mirror table (``lib_flist_table``) exists and refreshes with the dock's feature-set
        selector from the start — the same timing the old strip tab had — even before the dialog
        is first shown. Returns the (cached) dialog."""
        win = getattr(self, "_flist_window", None)
        if win is not None:
            return win
        win = QtWidgets.QDialog(self)
        win.setWindowTitle("Feature lists")
        win.setModal(False)
        lay = QtWidgets.QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._build_feature_lists_body())
        win.resize(600, 480)
        self._flist_window = win
        return win

    def _open_feature_lists_manager(self):
        """Feature-lists manager — demoted off the tab strip to the Features ⋯ menu's "Manage
        feature lists…" (plan 24 Phase 6), also reached by reveal_view("Feature lists"). Mirrors
        the dock's feature-set selector; the dialog is built at startup, so opening just refreshes
        and raises it."""
        win = self._build_feature_lists_dialog()
        self._refresh_library_tab()
        win.show()
        win.raise_()
        win.activateWindow()

    def _build_feature_lists_body(self):
        w, v = tab_page()
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(
            "<b>Feature lists</b> — the working sets (whole slide + each region), your "
            "saved ★ lists, and your ◆ lipid lists, kept with this sample's session"))
        top.addStretch(1)
        b_refresh = QtWidgets.QPushButton("Refresh")
        b_refresh.setIcon(icon("refresh"))
        b_refresh.clicked.connect(self._refresh_library_tab)
        top.addWidget(b_refresh)
        v.addLayout(top)

        self.lib_flist_table = QtWidgets.QTableWidget()
        self.lib_flist_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # ctrl/shift-click to select several lists at once (for bulk delete/export)
        self.lib_flist_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.lib_flist_table.cellDoubleClicked.connect(lambda *_: self._lib_load_feature_list())
        self.lib_flist_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.lib_flist_table.customContextMenuRequested.connect(self._lib_table_menu)
        for seq in (QtGui.QKeySequence.Delete, QtGui.QKeySequence("Backspace")):
            sc = QtGui.QShortcut(seq, self.lib_flist_table)
            sc.setContext(QtCore.Qt.WidgetShortcut)
            sc.activated.connect(self._lib_delete_feature_list)
        v.addWidget(self.lib_flist_table)

        fbar = QtWidgets.QHBoxLayout()
        b_fl_load = QtWidgets.QPushButton("Load into session")
        b_fl_load.setIcon(icon("open"))
        b_fl_load.clicked.connect(self._lib_load_feature_list)
        b_fl_comb = QtWidgets.QPushButton("Combine selected…")
        b_fl_comb.setIcon(icon("merge"))
        b_fl_comb.setToolTip("Merge the selected feature lists into a new ★ list — shared m/z "
                             "are kept once. Ctrl/⌘-click (or shift-click) two or more rows; "
                             "saved lists and working sets can be mixed.")
        b_fl_comb.clicked.connect(self._lib_combine_feature_lists)
        b_fl_exp = QtWidgets.QPushButton("Export…")
        b_fl_exp.setIcon(icon("export"))
        b_fl_exp.clicked.connect(self._lib_export_feature_list)
        b_fl_del = QtWidgets.QPushButton("Delete")
        b_fl_del.setObjectName("dangerAction")            # destructive verb → red tier
        b_fl_del.setIcon(icon("delete"))
        b_fl_del.setToolTip("Delete the selected saved ★ list(s). Working sets (the whole "
                            "slide and per-region lists) can't be deleted here — delete the "
                            "region from the Regions panel to drop its list.")
        b_fl_del.clicked.connect(self._lib_delete_feature_list)
        for b in (b_fl_load, b_fl_comb, b_fl_exp, b_fl_del):
            fbar.addWidget(b)
        fbar.addStretch(1)
        v.addLayout(fbar)

        self._refresh_library_tab()
        return w

    def _focus_tab(self, title):
        # reveal_view resolves both top-level and container-nested views by original label.
        self.reveal_view(title)

    def _refresh_library_tab(self):
        # Refresh button + tab init + per-sample reset. Funnel through the combo refresh,
        # which now also refills this tab's table (see _fill_library_table), so the two
        # surfaces can never drift apart again.
        self._refresh_feature_set_combo()

    def _fill_library_table(self):
        """Refill the 'Feature lists' tab table with *every* feature list this sample has —
        the working scopes (whole slide + each region's picked list) first, then the saved
        ★ lists, then the ◆ lipid lists — so it mirrors the dock's feature-set selector
        exactly. Each row stashes a ``(kind, name)`` tag on its first cell (``'scope'`` |
        ``'list'`` | ``'lipidlist'``) so load/export/delete dispatch without parsing the
        label. Kept separate from _refresh_feature_set_combo (which calls it) so there's no
        mutual recursion — this never touches the dock combo."""
        if not hasattr(self, "lib_flist_table"):
            return
        rows, tags = [], []
        scopes = (getattr(self, "_feature_scopes", {}) or {})
        for name in self._display_scope_names():
            kind_lbl = "whole slide" if name == "All slide" else "region"
            rows.append((name, kind_lbl, len(scopes.get(name, []))))
            tags.append(("scope", name))
        for name in sorted(self._feature_lists):
            rows.append((name, "saved ★", len(self._feature_lists.get(name, []))))
            tags.append(("list", name))
        for name in sorted(getattr(self, "_lipid_lists", {}) or {}):
            entries = self._lipid_lists.get(name) or []
            n_ions = sum(len(e.get("mzs", [])) for e in entries)
            rows.append((name, "lipid ◆", n_ions))
            tags.append(("lipidlist", name))
        fill_table(self.lib_flist_table, ["feature list", "type", "# features"], rows)
        for r, tag in enumerate(tags):                # carry (kind, name) past the visible text
            it = self.lib_flist_table.item(r, 0)
            if it is not None:
                it.setData(QtCore.Qt.UserRole, tag)

    @staticmethod
    def _lib_selected_meta(table):
        """``(kind, name)`` of the first selected row, or None."""
        rows = table.selectionModel().selectedRows()
        if not rows:
            return None
        it = table.item(rows[0].row(), 0)
        return it.data(QtCore.Qt.UserRole) if it else None

    @staticmethod
    def _lib_selected_metas(table):
        """Every selected row's ``(kind, name)`` tag, top-to-bottom (ctrl/shift multi-select)."""
        out = []
        for ix in sorted(table.selectionModel().selectedRows(), key=lambda i: i.row()):
            it = table.item(ix.row(), 0)
            tag = it.data(QtCore.Qt.UserRole) if it else None
            if tag:
                out.append(tag)
        return out

    def _lib_table_menu(self, pos):
        """Right-click menu for the Feature lists table — the same load / rename / set-as-
        default / combine / export / delete actions as the buttons below, but acting on the
        row(s) under the cursor. Right-clicking a row that isn't already selected selects it
        first (so a single right-click + action just works); right-clicking inside an
        existing multi-row selection keeps it (for Combine). Actions that don't apply to the
        current selection (e.g. renaming a working set, combining a single row) are shown
        disabled rather than hidden, so the menu always reads the same."""
        table = self.lib_flist_table
        row = table.rowAt(pos.y())
        sel_rows = {ix.row() for ix in table.selectionModel().selectedRows()}
        if row >= 0 and row not in sel_rows:
            table.selectRow(row)
        metas = self._lib_selected_metas(table)
        if not metas:
            return
        n = len(metas)
        one = metas[0] if n == 1 else None
        menu = QtWidgets.QMenu(table)
        a = menu.addAction(icon("open"), "Load into session", self._lib_load_feature_list)
        a.setEnabled(n == 1)
        menu.addSeparator()
        a = menu.addAction(icon("settings"), "Rename…", self._lib_rename_feature_list)
        a.setEnabled(one is not None and one[0] == "list")
        a = menu.addAction(icon("save"), "Set as default feature set", self._lib_set_default)
        a.setEnabled(one is not None and one[0] in ("scope", "list"))
        a = menu.addAction(icon("merge"), "Combine selected…", self._lib_combine_feature_lists)
        a.setEnabled(n >= 2)
        menu.addSeparator()
        a = menu.addAction(icon("export"), "Export…", self._lib_export_feature_list)
        a.setEnabled(n == 1)
        a = menu.addAction(icon("delete"), "Delete", self._lib_delete_feature_list)
        a.setEnabled(any(kind == "list" for kind, _ in metas))
        menu.addSeparator()
        add_copy_actions(menu, table)
        menu.exec(table.viewport().mapToGlobal(pos))

    def _lib_rename_feature_list(self):
        """Rename the selected saved ★ list. Working sets (whole slide / per-region) follow
        their region and can't be renamed here — point that out instead of failing quietly."""
        meta = self._lib_selected_meta(self.lib_flist_table)
        if not meta:
            self.statusBar().showMessage("Select a feature list to rename.")
            return
        kind, name = meta
        if kind != "list":
            self.statusBar().showMessage(
                "Only saved ★ lists can be renamed — a working set follows its region "
                "(rename the region in the Regions panel).")
            return
        self._rename_saved_list(name)                 # refreshes the combo + this tab

    def _lib_set_default(self):
        """Pin the selected row as the app-wide default feature set for new analyses."""
        meta = self._lib_selected_meta(self.lib_flist_table)
        if meta and meta[0] in ("scope", "list"):
            self._set_default_feature_set(meta)

    def _lib_features_for(self, kind, name):
        """Normalised ``[{mz, lipid, note}]`` for any kind — a saved ★ list is already in
        that shape; a working scope holds richer peak dicts, so annotate each m/z for
        export; a ◆ lipid list is flattened from its per-class m/z groups (labelled with
        the owning class)."""
        if kind == "scope":
            peaks = (getattr(self, "_feature_scopes", {}) or {}).get(name, [])
            return [{"mz": p.get("mz"), "lipid": self.annotate(p["mz"]) or "", "note": ""}
                    for p in peaks if p.get("mz") is not None]
        if kind == "lipidlist":
            entries = (getattr(self, "_lipid_lists", {}) or {}).get(name) or []
            return [{"mz": m, "lipid": e.get("class", ""), "note": ""}
                    for e in entries for m in e.get("mzs", [])]
        return list(self._feature_lists.get(name, []))

    def _lib_load_feature_list(self):
        meta = self._lib_selected_meta(self.lib_flist_table)
        if not meta:
            self.statusBar().showMessage("Select a feature list to load.")
            return
        kind, name = meta
        if kind == "scope":
            self._switch_feature_scope(name)      # make that working set active
        elif kind == "lipidlist":
            self._load_lipid_list(name)
        else:
            self._load_feature_list(name)
        self._focus_tab("Ion image")          # no "Feature list" tab — land on the main view

    def _lib_export_feature_list(self):
        meta = self._lib_selected_meta(self.lib_flist_table)
        if not meta:
            return
        kind, name = meta
        feats = self._lib_features_for(kind, name)
        path, _ = filedialogs.get_save_file_name(
            self, "Export feature list", f"{name}.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["mz", "lipid", "note"])
            for ft in feats:
                wtr.writerow([ft.get("mz", ""), ft.get("lipid", ""), ft.get("note", "")])
        self.statusBar().showMessage(f"Exported '{name}' → {path}")

    def _lib_combine_feature_lists(self):
        """Union the selected rows (saved ★ lists and/or working scopes) into a new saved ★
        list. _save_feature_list_to_library de-duplicates by m/z; the sources are untouched."""
        metas = self._lib_selected_metas(self.lib_flist_table)
        if len(metas) < 2:
            self.statusBar().showMessage(
                "Select two or more feature lists to combine (ctrl/⌘- or shift-click rows).")
            return
        merged = []
        for kind, name in metas:
            merged.extend(self._lib_features_for(kind, name))
        n_raw = len(merged)
        base = " + ".join(name for _kind, name in metas)
        saved = self._save_feature_list_to_library(base, merged, prompt=True)
        if not saved:                                       # user cancelled the name prompt
            return
        n_kept = len(self._feature_lists.get(saved, []))
        dup = n_raw - n_kept
        self._refresh_library_tab()
        self.statusBar().showMessage(
            f"Combined {len(metas)} lists into '{saved}' — {n_kept} unique feature(s)"
            + (f", {dup} duplicate m/z merged." if dup > 0 else "."))

    def _lib_delete_feature_list(self):
        metas = self._lib_selected_metas(self.lib_flist_table)
        if not metas:
            self.statusBar().showMessage("Select feature list(s) to delete.")
            return
        names = [name for kind, name in metas if kind == "list"]   # only saved ★ lists delete
        skipped = len(metas) - len(names)
        if not names:
            self.statusBar().showMessage(
                "Working sets and ◆ lipid lists can't be deleted here — delete a region from "
                "the Regions panel, or use Features ⋯ → 'Lipid lists…' to delete a lipid list.")
            return
        prompt = (f"Delete saved feature list '{names[0]}'?" if len(names) == 1
                  else f"Delete {len(names)} saved feature lists?\n\n" + ", ".join(names))
        if not confirm(self, "Delete feature lists", prompt):
            return
        for name in names:
            self._feature_lists.pop(name, None)
        self._mark_dirty()
        self._refresh_library_tab()
        msg = f"Deleted {len(names)} feature list(s)."
        if skipped:
            msg += f" ({skipped} working set(s) skipped — those can't be deleted here.)"
        self.statusBar().showMessage(msg)
