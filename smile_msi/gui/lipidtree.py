"""LipidTreeMixin — the class-grouped, expandable view of a loaded ◆ lipid list.

When a lipid list is the active feature set, the flat per-ion feature table is
swapped (in the dock's ``feat_stack``) for a tree grouped by lipid class. Each
class is a collapsible parent row that *behaves like a single ion for viewing* —
selecting it highlights the whole class on the tissue while the multi-colour
overlay stays on — and expands to its member ions, which keep their own
eye-toggle visibility and per-ion colour. Switching to any other feature set
(scope, ★ list, find peaks, clear) flips the stack back to the flat table; that
revert lives in ``features._set_peaks`` so every other path gets it for free.

The peaks themselves remain the single source of truth: the tree only mirrors
``self.peaks`` (``hidden`` / ``color`` / ``lipid_class``), so the colour overlay,
exports, window slider, and every analysis keep reading the same working set.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .common import MUTED_QSS, button, eye_icon, icon

_ROLE = QtCore.Qt.UserRole


class LipidTreeMixin:
    # ----- build ----------------------------------------------------------- #
    def _build_lipid_tree(self):
        """Create the class-grouped tree (page 1 of ``feat_stack``), wrapped in a page that
        carries its Show-all / Hide-all bar. Returns the page — ``feat_stack`` only ever adds
        it, so nothing outside this module needs to know the tree grew a sibling."""
        t = QtWidgets.QTreeWidget()
        t.setObjectName("lipidTree")
        t.setColumnCount(2)
        t.setHeaderLabels(["class · m/z", "lipid"])
        t.setUniformRowHeights(True)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        t.setMinimumHeight(320)
        t.header().setStretchLastSection(True)
        t.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        t.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._lipid_tree_menu)
        t.itemSelectionChanged.connect(self._lipid_tree_selected)
        t.itemChanged.connect(self._lipid_tree_item_changed)
        t.setToolTip(
            "Lipid classes. Click a class to highlight the whole class on the tissue "
            "(it acts as one ion); expand it to reach the member ions. Toggle the eye to "
            "show/hide an ion or a whole class; right-click to recolour or rename.")
        self.lipid_tree = t

        # A loaded lipid list is dozens of classes deep, and every row carries an eye. Hiding
        # all but one class was a click per row. These act on ``self.peaks`` — the source of
        # truth for `hidden` — not on the tree's check states, which are derived from it and
        # which _sync_lipid_tree would immediately overwrite.
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, tip, want in (
                ("Show all", "Show every ion and class in this list.", lambda _p: True),
                ("Hide all", "Hide every ion and class in this list.", lambda _p: False),
                ("Invert", "Show what is hidden and hide what is shown.",
                 lambda p: bool(p.get("hidden")))):
            row.addWidget(button(text, self._lipid_visibility_slot(want), tooltip=tip))
        row.addStretch(1)
        self._lipid_count = QtWidgets.QLabel("")
        self._lipid_count.setStyleSheet(MUTED_QSS)
        row.addWidget(self._lipid_count)
        col.addLayout(row)
        col.addWidget(t, 1)
        self._lipid_tree_page = page
        return page

    # ----- bulk visibility -------------------------------------------------- #
    def _lipid_visibility_slot(self, want):
        """A clicked(bool)-proof slot: PySide6 reads ``lambda w=want:`` as arity-1 and feeds
        the checked flag straight into ``w``, so the predicate would arrive as False."""
        return lambda *_: self._lipid_tree_set_visible(want)

    def _lipid_tree_set_visible(self, want):
        """Set every peak's ``hidden`` from ``want(peak) -> should_be_visible``, then run the
        one shared tail so the tree, the flat table, the readouts and the overlay all follow.

        Invert's predicate reads the very flag it is about to flip, so ``want`` is evaluated
        once per peak up front and the write loop only consumes the answer — it can never
        observe a flag an earlier iteration already changed."""
        peaks = getattr(self, "peaks", None) or []
        targets = [(p, bool(want(p))) for p in peaks]
        # flip a peak when its visibility (not hidden) already disagrees with the target
        flips = [(p, vis) for p, vis in targets if bool(p.get("hidden")) == vis]
        if not flips:
            return                                      # nothing to change → no undo entry
        self.record_undo("show/hide lipids")            # after the guard: a no-op leaves no entry
        for p, vis in flips:
            p["hidden"] = not vis
        self._after_tree_edit()

    def _refresh_lipid_count(self):
        lbl = getattr(self, "_lipid_count", None)
        if lbl is None:
            return
        peaks = getattr(self, "peaks", None) or []
        shown = sum(1 for p in peaks if not p.get("hidden"))
        lbl.setText(f"{shown} of {len(peaks)} shown")

    # ----- populate / sync from self.peaks --------------------------------- #
    def _populate_lipid_tree(self, entries=None):
        """Fill the class tree from the **working set** (``self.peaks``): each class
        composite is a parent row that behaves like one feature, with its member ions as
        view-only children; an expanded class (right-click → Expand into ions) instead
        shows its ions as real, eye-toggleable feature rows. ``entries`` is ignored (kept
        for call-site compatibility) — the tree is derived from the peaks themselves."""
        t = getattr(self, "lipid_tree", None)
        if t is None:
            return
        self._lipid_tree_building = True
        t.blockSignals(True)
        try:
            t.clear()
            # 1) class composites — one feature per class; members are view-only children
            for p in self.peaks:
                if not p.get("is_class"):
                    continue
                cls = p.get("lipid_class") or "(unclassed)"
                members = p.get("members") or []
                parent = QtWidgets.QTreeWidgetItem(t)
                parent.setData(0, _ROLE, ("class", cls))
                parent.setText(0, cls)
                parent.setText(1, f"{len(members)} ions · composite")
                parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable)
                for m in members:
                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setData(0, _ROLE, ("member", float(m)))   # view-only (no eye)
                    child.setText(0, f"{float(m):.4f}")
                    child.setText(1, self.annotate(m) or "—")
            # 2) expanded ions — real features carrying a class → grouped feature rows
            by_cls, flat = {}, []
            for p in self.peaks:
                if p.get("is_class"):
                    continue
                cls = p.get("lipid_class") or ""
                (by_cls.setdefault(cls, []).append(p) if cls else flat.append(p))
            for cls, ions in by_cls.items():
                parent = QtWidgets.QTreeWidgetItem(t)
                parent.setData(0, _ROLE, ("ionclass", cls))
                parent.setText(0, f"{cls} (ions)")
                parent.setText(1, f"{len(ions)} ions")
                parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable)
                for p in sorted(ions, key=lambda q: q["mz"]):
                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setData(0, _ROLE, ("ion", float(p["mz"])))
                    child.setText(0, f"{float(p['mz']):.4f}")
                    child.setText(1, self.annotate(p["mz"]) or "—")
                    child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
            for p in sorted(flat, key=lambda q: q["mz"]):
                row = QtWidgets.QTreeWidgetItem(t)
                row.setData(0, _ROLE, ("ion", float(p["mz"])))
                row.setText(0, f"{float(p['mz']):.4f}")
                row.setText(1, self.annotate(p["mz"]) or "—")
                row.setFlags(row.flags() | QtCore.Qt.ItemIsUserCheckable)
            t.collapseAll()
        finally:
            t.blockSignals(False)
            self._lipid_tree_building = False
        self._sync_lipid_tree()
        self._refresh_lipid_count()

    def _sync_lipid_tree(self):
        """Re-derive every check state + eye icon from ``self.peaks`` (the source of truth
        for hidden/colour). A class composite's eye = its own hidden flag; an expanded
        ion-class is checked when all its ions are visible. A no-op unless the tree shows."""
        t = getattr(self, "lipid_tree", None)
        stack = getattr(self, "feat_stack", None)
        page = getattr(self, "_lipid_tree_page", None)
        if t is None or (stack is not None and stack.currentWidget() is not page):
            return
        comp_by_cls = {p.get("lipid_class"): p for p in self.peaks if p.get("is_class")}
        by_mz = {round(float(p["mz"]), 4): p for p in self.peaks if not p.get("is_class")}
        prev = getattr(self, "_lipid_tree_building", False)
        self._lipid_tree_building = True
        t.blockSignals(True)
        try:
            for i in range(t.topLevelItemCount()):
                parent = t.topLevelItem(i)
                kind, val = parent.data(0, _ROLE)
                if kind == "class":                            # composite — its own hidden flag
                    comp = comp_by_cls.get(val)
                    vis = not (comp and comp.get("hidden"))
                    parent.setCheckState(0, QtCore.Qt.Checked if vis else QtCore.Qt.Unchecked)
                    parent.setIcon(0, eye_icon(self._feature_color(comp), vis))
                elif kind == "ionclass":                       # expanded ions — roll up children
                    n_vis, first = 0, None
                    for j in range(parent.childCount()):
                        child = parent.child(j)
                        p = by_mz.get(round(float(child.data(0, _ROLE)[1]), 4))
                        first = first or p
                        v = not (p and p.get("hidden"))
                        n_vis += int(v)
                        child.setCheckState(0, QtCore.Qt.Checked if v else QtCore.Qt.Unchecked)
                        child.setIcon(0, eye_icon(self._feature_color(p), v))
                    n = parent.childCount()
                    state = (QtCore.Qt.Checked if n and n_vis == n else
                             QtCore.Qt.Unchecked if n_vis == 0 else QtCore.Qt.PartiallyChecked)
                    parent.setCheckState(0, state)
                    parent.setIcon(0, eye_icon(self._feature_color(first), state != QtCore.Qt.Unchecked))
                else:                                          # flat ion top-level row
                    p = by_mz.get(round(float(val), 4))
                    v = not (p and p.get("hidden"))
                    parent.setCheckState(0, QtCore.Qt.Checked if v else QtCore.Qt.Unchecked)
                    parent.setIcon(0, eye_icon(self._feature_color(p), v))
        finally:
            t.blockSignals(False)
            self._lipid_tree_building = prev

    # ----- stack show/hide + selection mirror ------------------------------ #
    def _show_lipid_tree(self, on):
        """Flip the dock between the flat feature table and the class tree."""
        stack = getattr(self, "feat_stack", None)
        if stack is None:
            return
        stack.setCurrentWidget(self._lipid_tree_page if on else self.feat_table)

    def _select_lipid_tree_ion(self, mz):
        """Mirror the active ion into the tree selection (when the tree is showing) so a
        pick from the spectrum highlights the matching child without re-firing view code."""
        t = getattr(self, "lipid_tree", None)
        stack = getattr(self, "feat_stack", None)
        page = getattr(self, "_lipid_tree_page", None)
        if t is None or stack is None or stack.currentWidget() is not page or mz is None:
            return
        t.blockSignals(True)
        try:
            for i in range(t.topLevelItemCount()):
                parent = t.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    kind, val = child.data(0, _ROLE)
                    if kind in ("ion", "member") and abs(float(val) - float(mz)) < 1e-3:
                        parent.setExpanded(True)
                        t.setCurrentItem(child)
                        t.scrollToItem(child)
                        return
        finally:
            t.blockSignals(False)

    # ----- interaction: selection ------------------------------------------ #
    def _lipid_tree_selected(self):
        if getattr(self, "_lipid_tree_building", False):
            return
        items = self.lipid_tree.selectedItems()
        if not items:
            return
        kind, val = items[0].data(0, _ROLE)
        if kind == "class":
            comp = next((p for p in self.peaks
                         if p.get("is_class") and p.get("lipid_class") == val), None)
            if comp is not None:
                self.set_active_mz(float(comp["mz"]))     # composite renders as one feature
        elif kind in ("ion", "member"):
            self.set_active_mz(float(val))                # single ion (member = view-only preview)
        # an "ionclass" parent is just a group header — expand it to pick an ion

    def _focus_lipid_class(self, cls):
        """Highlight a whole lipid class on the tissue: brighten the composite of its visible
        ions over the multi-colour overlay (a static highlight — this used to pulse via a
        ~22 fps timer that saturated the UI thread), so the class reads as one ion. Drops any
        single-ion selection — the class is the subject now."""
        if self.ds is None:
            return
        mzs = [float(p["mz"]) for p in self.peaks
               if (p.get("lipid_class") or "") == cls and not p.get("hidden")]
        if not mzs:
            self.statusBar().showMessage(f"{cls}: all ions hidden — nothing to highlight.")
            return
        self.active_mz = None
        if hasattr(self, "_sync_coloc_start"):
            self._sync_coloc_start()
        self._highlight_active_peak()                  # active_mz None → clears the halo
        self._update_selected_feature_ui()
        if getattr(self, "_overlay_base", None) is None:   # ensure a base under the pulse
            self.refresh_ion_image()
        img = self.ds.composite_image(mzs, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm,
                                      weight=self.composite_weight)
        self._pulse_norm = self._windowed(img, 0.0, 100.0)
        self._draw_overlay_frame()                     # static highlight; no animation timer
        self._set_ion_tab_caption(f"focus: total {cls} ({len(mzs)} ions)")
        self.statusBar().showMessage(
            f"Highlighting {cls} — {len(mzs)} ion{'s' if len(mzs) != 1 else ''} as one.")

    # ----- interaction: visibility (eye toggle) ---------------------------- #
    def _lipid_tree_item_changed(self, item, col):
        if col != 0 or getattr(self, "_lipid_tree_building", False):
            return
        kind, val = item.data(0, _ROLE)
        vis = item.checkState(0) == QtCore.Qt.Checked
        changed = False
        if kind == "class":                            # composite — its own hidden flag
            comp = next((p for p in self.peaks
                         if p.get("is_class") and p.get("lipid_class") == val), None)
            if comp is not None and bool(comp.get("hidden")) == vis:
                comp["hidden"] = not vis
                changed = True
        elif kind == "ion":
            p = self._peak_for_mz(float(val))
            if p is not None and bool(p.get("hidden")) == vis:
                p["hidden"] = not vis
                changed = True
        elif kind == "ionclass":                       # whole expanded class
            self.record_undo("toggle lipid class")
            for p in self.peaks:
                if (not p.get("is_class") and (p.get("lipid_class") or "") == str(val)
                        and bool(p.get("hidden")) == vis):
                    p["hidden"] = not vis
                    changed = True
        if not changed:
            return
        self._after_tree_edit()

    def _after_tree_edit(self):
        """Shared tail for any tree edit that changes hidden/colour: re-sync the tree,
        the (hidden) flat table, the pipeline readouts, and the overlay."""
        self._sync_lipid_tree()
        self._refresh_lipid_count()
        self._decorate_feature_swatches()              # keep the flat table coherent
        self._sync_feature_consumers()
        if self.color_overlay_chk.isChecked():
            self.refresh_ion_image()

    # ----- interaction: context menu (recolour / rename / solo) ------------ #
    def _lipid_tree_menu(self, pos):
        t = self.lipid_tree
        item = t.itemAt(pos)
        if item is None:
            return
        kind, val = item.data(0, _ROLE)
        menu = QtWidgets.QMenu(t)
        if kind == "class":
            menu.addAction(f"Expand {val} into ions", lambda: self._expand_class_to_ions(str(val))).setIcon(icon("expand"))
            menu.addAction(f"Recolour {val}…", lambda: self._recolour_lipid_class(str(val))).setIcon(icon("settings"))
            menu.addAction(f"Show only {val}", lambda: self._solo_lipid_class(str(val))).setIcon(icon("find"))
        elif kind == "ionclass":
            menu.addAction(f"Recolour {val}…", lambda: self._recolour_lipid_class(str(val))).setIcon(icon("settings"))
            menu.addAction(f"Show only {val}", lambda: self._solo_lipid_class(str(val))).setIcon(icon("find"))
        elif kind == "member":                         # view-only member of a composite
            menu.addAction("View this ion", lambda: self.set_active_mz(float(val))).setIcon(icon("navigate"))
        else:                                          # a real ion feature
            menu.addAction("View this ion", lambda: self.set_active_mz(float(val))).setIcon(icon("navigate"))
            menu.addAction("Set colour…", lambda: self._recolour_lipid_ion(float(val))).setIcon(icon("settings"))
            menu.addAction("Rename label…", lambda: self._rename_feature_label(float(val))).setIcon(icon("settings"))
        menu.exec(t.viewport().mapToGlobal(pos))

    def _recolour_lipid_class(self, cls):
        peaks = [p for p in self.peaks if (p.get("lipid_class") or "") == cls]
        if not peaks:
            return
        from . import colorpicker
        hexc = colorpicker.pick_color(self, initial=self._feature_color(peaks[0]),
                                      title=f"Colour for {cls}")
        if not hexc:
            return
        self.record_undo("recolour lipid class")
        for p in peaks:
            p["color"] = hexc
        self._set_lipid_list_class_color(cls, hexc)    # persist on the saved list too
        self._after_tree_edit()
        self.statusBar().showMessage(f"{cls} recoloured {hexc}.")

    def _recolour_lipid_ion(self, mz):
        p = self._peak_for_mz(float(mz))
        if p is None:
            return
        from . import colorpicker
        hexc = colorpicker.pick_color(self, initial=self._feature_color(p), title="Ion colour")
        if not hexc:
            return
        self.record_undo("ion colour")
        p["color"] = hexc
        self._after_tree_edit()

    def _solo_lipid_class(self, cls):
        """Hide every other class so only ``cls`` shows on the tissue, then select it (the
        class composite renders as one feature)."""
        self.record_undo("solo lipid class")
        for p in self.peaks:
            p["hidden"] = (p.get("lipid_class") or "") != cls
        self._after_tree_edit()
        comp = next((p for p in self.peaks
                     if p.get("is_class") and p.get("lipid_class") == cls), None)
        if comp is not None:
            self.set_active_mz(float(comp["mz"]))
        self.statusBar().showMessage(f"Showing only {cls}.")

    def _set_lipid_list_class_color(self, cls, hexc):
        """Write a recoloured class back into the saved lipid list so the colour persists
        with the session (and on the next load)."""
        name = getattr(self, "_active_lipid_list", None)
        for e in (getattr(self, "_lipid_lists", {}).get(name) or []):
            if e.get("class") == cls:
                e["color"] = hexc
        self._mark_dirty()
