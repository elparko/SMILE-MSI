"""LipidListMixin — "lipid lists": the class-level twin of feature lists.

A *lipid list* is a named collection of lipid **classes** (PE, PC, SM, sulfatide, …), each
carrying a distinct colour and its member m/z. It behaves like a saved feature list — it lives
in the same right-dock 'Feature set' dropdown (prefixed ◆ so it never conflicts with the ★ ion
lists) and persists in the session — but loading one does two extra things:

* the working features become the union of every class's ions, each ion tinted its class colour
  (so all PE ions read red, all PC green, …); and
* the per-feature **color overlay** switches on, so the classes paint straight onto the tissue —
  each class a different colour on the nerve.

Lists are created from the Features ⋯ menu → 'New lipid list from identified features…' (the
visible features' identified ions, grouped by class) and edited (recolour / rename / delete)
from the same menu → 'Lipid lists…'.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from .common import REGION_PALETTE, icon, install_table_export


class LipidListMixin:
    # ----- model helpers ---------------------------------------------------- #
    @staticmethod
    def _norm_lipid_lists(raw):
        """Coerce a loaded ``{name: [{class,color,mzs}]}`` mapping into the canonical shape,
        dropping classes with no m/z and lists left empty."""
        out = {}
        for name, entries in (raw or {}).items():
            norm = []
            for e in (entries or []):
                mzs = [float(m) for m in (e.get("mzs") or [])]
                if mzs:
                    norm.append({"class": str(e.get("class", "")),
                                 "color": str(e.get("color", "")) or "#888888",
                                 "mzs": mzs})
            if norm:
                out[str(name)] = norm
        return out

    def _lipid_list_colors(self, classes):
        """A distinct colour per class: the vivid region palette first, then evenly-spaced
        hues for any overflow, so even a dozen-plus classes stay separable."""
        classes = list(classes)
        n = max(1, len(classes))
        out = {}
        for i, c in enumerate(classes):
            out[c] = (REGION_PALETTE[i] if i < len(REGION_PALETTE)
                      else pg.intColor(i, hues=n, maxValue=230).name())
        return out

    def save_lipid_list(self, classes_mzs, suggested=None):
        """Save ``{class: [m/z, …]}`` as a named lipid list (auto-coloured), register it in the
        feature-set selector, and return the chosen name (or None if cancelled/empty)."""
        cleaned = {str(c): sorted({float(m) for m in mzs})
                   for c, mzs in (classes_mzs or {}).items() if c and mzs}
        if not cleaned:
            self.statusBar().showMessage("No identified lipid classes to save.")
            return None
        default = suggested or f"Lipid list {len(self._lipid_lists) + 1}"
        name, ok = QtWidgets.QInputDialog.getText(self, "Save lipid list", "Name:", text=default)
        if not (ok and name.strip()):
            return None
        nm = name.strip()
        order = sorted(cleaned)
        colors = self._lipid_list_colors(order)
        self._lipid_lists[nm] = [{"class": c, "color": colors[c], "mzs": cleaned[c]} for c in order]
        self._mark_dirty()
        self._refresh_feature_set_combo()
        n_ions = sum(len(v) for v in cleaned.values())
        self.statusBar().showMessage(
            f"Saved lipid list '{nm}' — {len(order)} classes, {n_ions} ions. "
            "Pick it in the Feature set selector to paint the classes on the tissue.")
        return nm

    def _new_lipid_list_from_features(self):
        """Group the visible features' identified ions by lipid class and save them as a
        lipid list. A class composite (from a loaded lipid list) keeps its class and members."""
        by_class = {}
        for p in self._visible_peaks():
            if p.get("is_class"):
                mzs, cls = p.get("members") or [], p.get("lipid_class")
                by_class.setdefault(cls, []).extend(float(m) for m in mzs)
                continue
            mz = float(p["mz"])
            cls = p.get("lipid_class") or self.ann.class_of(mz)
            if cls:
                by_class.setdefault(cls, []).append(mz)
        if not by_class:
            self.statusBar().showMessage("None of the visible features is identified as a lipid "
                                         "— run Identify lipids first.")
            return None
        base = (getattr(self, "_active_feature_scope", None)
                or getattr(self, "_flist_name", None) or "Features").lstrip("★◆ ").strip()
        return self.save_lipid_list(by_class, suggested=f"{base} classes")

    # ----- load → class-coloured overlay ------------------------------------ #
    def _class_composite_peak(self, cls, mzs, color):
        """A synthetic 'feature' standing for a whole lipid class — it behaves like a
        single feature (selection, intensity window, ion image) but its image is the
        composite of its member ions (``members``). Carries no extracted column itself:
        the composite is rendered on demand (display) or expanded to real ion features
        (right-click → Expand into ions). ``mz`` is a representative (mean) for identity
        / sort / the spectrum cursor."""
        mzs = sorted(float(m) for m in mzs)
        rep = sum(mzs) / len(mzs) if mzs else 0.0
        cls = str(cls or "(unclassed)")
        return {"mz": rep, "is_class": True, "members": mzs, "lipid_class": cls,
                "label_override": cls, "color": color or "#888888",
                "intensity": 0.0, "rel_intensity": 0.0, "snr": 0.0}

    def _expand_class_to_ions(self, cls):
        """Replace a class composite in the working set with its individual member ions
        (real features), so browsing / analyses on that class become ion-level. Other
        classes stay as composites."""
        comp = next((p for p in self.peaks
                     if p.get("is_class") and p.get("lipid_class") == cls), None)
        if comp is None:
            return
        self.record_undo(f"expand {cls} into ions")
        ions = []
        for m in comp.get("members", []):
            p = self._peak_from_mz(float(m))
            p["color"] = comp.get("color")
            p["lipid_class"] = cls
            ions.append(p)
        rest = [p for p in self.peaks
                if not (p.get("is_class") and p.get("lipid_class") == cls)]
        self._set_peaks(rest + ions, reannotate=False, extract=False,
                        msg=f"Expanded {cls} into {len(ions)} ion feature(s).")
        self._show_lipid_tree(True)
        self._populate_lipid_tree(self._lipid_lists.get(self._active_lipid_list) or [])

    def _load_lipid_list(self, name):
        """Make ``name`` the active set: each lipid **class** becomes one composite
        feature (its member ions are loaded only when you expand the class), tinted its
        class colour, and the colour overlay switches on so every class paints on the
        tissue."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        entries = self._lipid_lists.get(name) or []
        if not entries:
            self.statusBar().showMessage(f"Lipid list '{name}' is empty.")
            return
        lo, hi = self.ds.mz_range
        peaks, n_classes, n_oob, n_ions = [], 0, 0, 0
        for e in entries:
            in_range = [float(m) for m in e.get("mzs", []) if lo <= float(m) <= hi]
            n_oob += len(e.get("mzs", [])) - len(in_range)
            if not in_range:
                continue
            n_classes += 1
            n_ions += len(in_range)
            peaks.append(self._class_composite_peak(e.get("class", ""), in_range, e.get("color")))
        if not peaks:
            self.statusBar().showMessage(
                f"Lipid list '{name}': none of its m/z fall in this sample's acquired range "
                f"{lo:.1f}–{hi:.1f}, so there's nothing to show here.")
            return
        self.record_undo("load lipid list")
        self._active_feature_scope = None
        self._active_lipid_list = name
        oob = f" · {n_oob} m/z out of range skipped" if n_oob else ""
        self._set_peaks(peaks, reannotate=False, extract=False,
                        msg=f"Loaded '{name}' — {n_classes} classes as composite features "
                            f"({n_ions} ions; right-click a class → Expand into ions){oob}.")
        self._set_flist_name(f"◆ {name}")
        # paint the classes on the tissue via the per-feature colour overlay
        chk = getattr(self, "color_overlay_chk", None)
        if chk is not None and not chk.isChecked():
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.blockSignals(False)
        self._toggle_color_overlay(True)               # greys colormap + re-renders the overlay
        # swap the flat per-ion table for the class-grouped tree (classes → ions, each
        # class acting as one ion for viewing — see LipidTreeMixin). Show it first so the
        # tree is the live page when _populate_lipid_tree syncs its eye states.
        self._show_lipid_tree(True)
        self._populate_lipid_tree(entries)
        self._set_feature_set_current()                # selector points at this lipid list

    # ----- manage (rename / delete / recolour) ------------------------------ #
    def _manage_lipid_lists(self):
        if not self._lipid_lists:
            self.statusBar().showMessage("No lipid lists yet — make one from the Features ⋯ menu "
                                         "→ New lipid list from identified features…")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Lipid lists")
        dlg.resize(580, 420)
        lay = QtWidgets.QHBoxLayout(dlg)
        names = QtWidgets.QListWidget()
        names.setMaximumWidth(190)
        names.addItems(sorted(self._lipid_lists))
        lay.addWidget(names)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Double-click a colour cell to recolour that class."))
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["class", "colour", "ions"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        install_table_export(table, dlg, stem="lipid_list", title="Export lipid list")
        right.addWidget(table, 1)
        btns = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton("Open on tissue")
        b_open.setObjectName("primaryAction")             # the dialog's primary open action
        b_open.setIcon(icon("open"))
        b_rename = QtWidgets.QPushButton("Rename…")
        b_rename.setIcon(icon("settings"))
        b_delete = QtWidgets.QPushButton("Delete")
        b_delete.setObjectName("dangerAction")            # destructive verb → red tier
        b_delete.setIcon(icon("delete"))
        btns.addWidget(b_open)
        btns.addStretch(1)
        btns.addWidget(b_rename)
        btns.addWidget(b_delete)
        right.addLayout(btns)
        lay.addLayout(right, 1)

        def current_name():
            it = names.currentItem()
            return it.text() if it is not None else None

        def fill_table():
            nm = current_name()
            table.setRowCount(0)
            for e in self._lipid_lists.get(nm, []) if nm else []:
                r = table.rowCount()
                table.insertRow(r)
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(e["class"]))
                sw = QtWidgets.QTableWidgetItem(e["color"])
                sw.setBackground(QtGui.QColor(e["color"]))
                sw.setForeground(QtGui.QColor("#111111"))
                table.setItem(r, 1, sw)
                table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(len(e["mzs"]))))

        def recolour(row, col):
            nm = current_name()
            if nm is None or col != 1 or not (0 <= row < len(self._lipid_lists.get(nm, []))):
                return
            entry = self._lipid_lists[nm][row]
            from . import colorpicker
            hexc = colorpicker.pick_color(dlg, initial=entry["color"],
                                          title=f"Colour for {entry['class']}")
            if not hexc:
                return
            entry["color"] = hexc
            self._mark_dirty()
            fill_table()
            if self._active_lipid_list == nm:
                self._load_lipid_list(nm)              # re-render the overlay with the new colour

        def open_list():
            nm = current_name()
            if nm:
                self._load_lipid_list(nm)

        def rename():
            nm = current_name()
            if not nm:
                return
            new, ok = QtWidgets.QInputDialog.getText(dlg, "Rename lipid list", "Name:", text=nm)
            new = new.strip() if ok else ""
            if new and new != nm:
                self._lipid_lists[new] = self._lipid_lists.pop(nm)
                if self._active_lipid_list == nm:
                    self._active_lipid_list = new
                self._mark_dirty()
                self._refresh_feature_set_combo()
                names.clear()
                names.addItems(sorted(self._lipid_lists))

        def delete():
            nm = current_name()
            if not nm:
                return
            self._lipid_lists.pop(nm, None)
            if self._active_lipid_list == nm:
                self._active_lipid_list = None
            self._mark_dirty()
            self._refresh_feature_set_combo()
            names.clear()
            names.addItems(sorted(self._lipid_lists))
            fill_table()

        names.currentItemChanged.connect(lambda *_: fill_table())
        table.cellDoubleClicked.connect(recolour)
        b_open.clicked.connect(open_list)
        b_rename.clicked.connect(rename)
        b_delete.clicked.connect(delete)
        if names.count():
            names.setCurrentRow(0)
        dlg.show()
        dlg.raise_()
        self._lipid_mgr_dlg = dlg                       # keep a reference so it isn't GC'd
