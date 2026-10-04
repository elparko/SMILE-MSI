"""Cohort view — the multi-file layer of the desktop app.

Two pieces, both backed by :mod:`smile_msi.cohort`:

* a **Samples** dock (left) — a roster of the files in the current cohort, bucketed by
  their **group** label (the level above a sample's regions). Double-click a sample to
  switch to it; the app flushes the current sample's auto-saved session and reloads the
  picked one, so only one dataset is ever in memory. Right-click to set a group or drop
  a sample.
* a **Cohort** tab — cross-sample / between-group comparison. It reads each sample's
  saved peaks (no dataset reloads), builds a sample x feature table, and runs a
  group-A-vs-group-B test (sample as replicate) into a volcano + table, plus a
  sample x feature heatmap.
* a **Cohort UMAP** tab — a *pooled* embedding: pixels from every sample, projected on a
  shared feature axis into ONE UMAP/t-SNE (via :func:`smile_msi.multivariate.pooled_embedding`),
  so the slides share a coordinate frame and the cloud can be coloured by sample or group.
  Unlike the per-tab comparison this **does** reload datasets (it needs the pixels).

Kept as a mixin so ``MainWindow`` composes it like the per-tab mixins; the heavy lifting
lives in the engine module and stays unit-testable without Qt.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import cohort as cohort_engine, library, session, profiles
from . import common
from .common import (REGION_PALETTE, colormap, fill_table, ControlBar, GUIDE_LINE, tab_page,
                     tool_button, confirm, table_placeholder, icon, primary_button,
                     menu_button, button, CheckPicker, NoScrollComboBox,
                     NoScrollDoubleSpinBox, NoScrollSpinBox)
from . import filedialogs

UNGROUPED = "Ungrouped"

# Cross-sample scaling strategies for the pooled UMAP embedding (mutually exclusive — a
# dropdown, not toggles). See smile_msi.multivariate.pooled_embedding(cohort_norm=).
SCALE_ALIGN = "Align samples (z-score)"      # per-sample z-score (the default)
SCALE_COHORT = "Cohort scale (global TIC)"   # one shared cohort-wide TIC anchor (Pixels unit)
SCALE_NONE = "None (global only)"            # a single global standardization only


_NEST_TOKEN_SPLIT = re.compile(r"[\s_\-]+")


def _nest_tokens(name: str) -> list[str]:
    """Lower-case tokens of a region/group name — split on whitespace, ``-`` and ``_``, with
    bracketing punctuation stripped, so ``'nerve1 endo (roi)'`` yields ``['nerve1','endo','roi']``
    rather than a token named ``'(roi)'``."""
    out = []
    for p in _NEST_TOKEN_SPLIT.split((name or "").strip()):
        p = p.strip("()[]{}.,;:").lower()
        if p:
            out.append(p)
    return out


class CohortMixin:
    # ===================================================================== #
    # Samples dock — the roster + switching
    # ===================================================================== #
    def _build_samples_panel(self):
        """Cohort roster as a collapsible **Samples** section at the top of the right-hand
        column — the level above a sample's regions. Restores the persisted 'Workspace'
        cohort so the roster survives restarts."""
        from .featurepanel import _CollapsibleSection
        self.cohort = cohort_engine.Cohort(name="Workspace")
        try:                                                   # restore last roster
            path = cohort_engine.cohort_path(self.cohort.name)
            if os.path.exists(path):
                self.cohort = cohort_engine.Cohort.load(path)
                if self.cohort.dedupe():                       # heal pre-fix duplicates
                    self._save_cohort()
        except Exception:  # noqa: BLE001 — roster restore is best-effort
            pass

        sec = _CollapsibleSection("Samples", on_toggle=self._relayout_right_sections)
        self._populate_samples_section(sec.body_layout)
        self._samples_section = sec          # kept so the Cohort tabs can reveal it on demand
        # which sample/cohort you're in is the top-level context → pin above Display
        self._right_sections_layout.insertWidget(self._section_index_offset, sec)
        self._right_sections.insert(0, sec)
        self._relayout_right_sections()
        self._refresh_sample_tree()

    def _populate_samples_section(self, lay):
        row = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton("Open files…")
        b_open.setIcon(icon("open"))
        b_open.setToolTip("Open one or more imzML files; the first becomes active and "
                          "each is added to the cohort")
        b_open.clicked.connect(self._cohort_open_files)
        b_add = QtWidgets.QPushButton("Add saved…")
        b_add.setIcon(icon("open"))
        b_add.setToolTip("Add samples you've already worked on (their auto-saved sessions)")
        b_add.clicked.connect(self._cohort_add_saved)
        b_regions = QtWidgets.QPushButton("Add regions…")
        b_regions.setIcon(icon("add"))
        b_regions.setToolTip("Add regions of the active sample as their own cohort samples, "
                             "so areas of one slide can be separate replicates in a comparison")
        b_regions.clicked.connect(self._cohort_add_regions)
        row.addWidget(b_open)
        row.addWidget(b_add)
        row.addWidget(b_regions)
        lay.addLayout(row)

        # find / bulk-group controls — the cohort-scale workflow: filter the roster, then
        # act on the matches in one step instead of clicking 30 samples one at a time.
        find_row = QtWidgets.QHBoxLayout()
        self.sample_filter = QtWidgets.QLineEdit()
        self.sample_filter.setPlaceholderText("Filter samples…")
        self.sample_filter.setClearButtonEnabled(True)
        self.sample_filter.setToolTip("Show only samples whose name, group, region or "
                                      "metadata contains this text.")
        self.sample_filter.textChanged.connect(self._refresh_sample_tree)
        find_row.addWidget(self.sample_filter, 1)
        b_selmatch = tool_button("⌖", "Select all matching samples — then 'Set group ▾' "
                                 "labels them all at once.", self._select_matching_samples)
        find_row.addWidget(b_selmatch)
        lay.addLayout(find_row)

        self.sample_tree = QtWidgets.QTreeWidget()
        self.sample_tree.setColumnCount(4)
        self.sample_tree.setHeaderLabels(["Name", "Group", "Region", "#px"])
        self.sample_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.sample_tree.setMaximumHeight(300)        # a touch taller now it's a real table
        self.sample_tree.setRootIsDecorated(False)    # groups are headers, not expandable arrows
        self.sample_tree.itemDoubleClicked.connect(self._cohort_tree_activated)
        self.sample_tree.setToolTip("Double-click a sample to switch to it. Click a column header "
                                    "to sort. Right-click (or 'Set group ▾') to group or remove.")
        self.sample_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.sample_tree.customContextMenuRequested.connect(self._cohort_tree_menu)
        hdr = self.sample_tree.header()
        hdr.setSectionsClickable(True)
        hdr.setSortIndicatorShown(True)
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hdr.sectionClicked.connect(self._on_sample_header_clicked)
        # (col, order) once the user clicks a header; None = the grouped default order. We sort
        # rows WITHIN each group on render rather than QTreeWidget.sortItems (which would reorder
        # the group headers too), so 'named groups first, Ungrouped last' always holds.
        self._sample_sort = None
        lay.addWidget(self.sample_tree)
        lay.addWidget(common.note("Double-click a sample to switch to it · select sample(s) and 'Set group ▾' to label them as the Group A / Group B compared in the Cohort tab."))

        # group assignment used to be right-click-only (undiscoverable); a visible button
        # makes the groups the Cohort tab's A/B dropdowns read from actually reachable.
        grp_row = QtWidgets.QHBoxLayout()
        b_group = QtWidgets.QToolButton()
        b_group.setText("Set group")                       # ▾ drawn by QSS (menuButton chevron)
        b_group.setObjectName("menuButton")                # match the Auto-group pill beside it
        b_group.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        b_group.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        b_group.setToolTip("Assign the selected sample(s) to a group — these become the "
                           "Group A / Group B choices in the Cohort tab.\n"
                           "(You can also right-click a sample.)")
        _group_menu = QtWidgets.QMenu(b_group)
        b_group.setMenu(_group_menu)                       # populated live from the selection on open
        _group_menu.aboutToShow.connect(lambda: self._cohort_fill_group_menu(_group_menu))
        grp_row.addWidget(b_group)
        # criteria-based grouping — derive groups in one pass (the FlowJo "group by
        # folder / keyword" move) instead of hand-labelling each sample.
        b_auto = menu_button("Auto-group", [
            ("By folder", self._cohort_group_by_folder),
            ("By name pattern…", self._cohort_group_by_pattern),
            ("By metadata…", self._cohort_group_by_meta),
        ], tooltip="Assign every sample's group at once from its folder, a piece of its "
                   "name, or a metadata column.")
        grp_row.addWidget(b_auto)
        b_batch = menu_button("Auto-batch", [
            ("By folder", self._cohort_batch_by_folder),
            ("By name pattern…", self._cohort_batch_by_pattern),
            ("By metadata…", self._cohort_batch_by_meta),
        ], tooltip="Assign every sample's acquisition batch at once (the technical axis for batch "
                   "correction) from its folder, a piece of its name, or a metadata column.")
        grp_row.addWidget(b_batch)
        b_meta = button("Import metadata…", self._cohort_import_metadata, icon="open",
                        tooltip="Load a CSV that maps each sample to metadata (and optionally "
                                "a group). Columns become metadata you can group and filter by.")
        grp_row.addWidget(b_meta)
        grp_row.addStretch(1)
        lay.addLayout(grp_row)

        # Cohort-level analysis option: how slides are scaled to be comparable before the pooled
        # UMAP embedding (Cohort UMAP tab). A dropdown because the strategies are mutually
        # exclusive; this is the home for cohort-wide options. Drives _cembed_run.
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Cross-sample scaling"))
        self.cembed_scaling_combo = NoScrollComboBox()
        self.cembed_scaling_combo.addItems([SCALE_ALIGN, SCALE_COHORT, SCALE_NONE])
        self.cembed_scaling_combo.setToolTip(
            "How slides are made comparable before the pooled UMAP embedding (Cohort UMAP tab):\n"
            "• Align samples (z-score) — per-sample z-score; removes each slide's overall "
            "offset/scale, but also flattens genuine cross-sample abundance differences.\n"
            "• Cohort scale (global TIC) — anchor every slide's TIC to one shared cohort-wide "
            "reference, so same-biology slides line up while real abundance differences survive "
            "(the Farrow-2025 recipe; Pixels unit).\n"
            "• None — a single global standardization across the pooled cloud only.")
        # connect AFTER addItems so the initial population doesn't fire into a half-built window
        self.cembed_scaling_combo.currentTextChanged.connect(self._cembed_sync_scaling)
        scale_row.addWidget(self.cembed_scaling_combo, 1)
        lay.addLayout(scale_row)

        b_cmp = QtWidgets.QPushButton("Batch compare →")
        b_cmp.setIcon(icon("navigate"))
        b_cmp.setToolTip("Open the Cohort tab to compare groups across samples")
        b_cmp.clicked.connect(self._goto_cohort_tab)
        lay.addWidget(b_cmp)

        b_correct = QtWidgets.QPushButton("Correct batches…")
        b_correct.setIcon(icon("settings"))
        b_correct.setToolTip("Run ComBat batch correction over the assigned batches and report "
                             "before/after QC (assign batches with Auto-batch ▾ first).")
        b_correct.clicked.connect(self._cohort_correct_batches)
        lay.addWidget(b_correct)

        b_batch_export = QtWidgets.QPushButton("Batch export across sections…")
        b_batch_export.setIcon(icon("export"))
        b_batch_export.setToolTip("Open the Export Studio to render ion images for chosen feature "
                                  "lists across these samples (sections) in one pass.")
        b_batch_export.clicked.connect(self.open_export_studio)
        lay.addWidget(b_batch_export)

    def _reveal_samples_panel(self):
        """Open the right-hand panel (if it was dragged shut) and expand + scroll to the
        Samples section. The Cohort / Cohort UMAP tabs tell you to add samples 'in the
        Samples panel' but live in the central area — without this the instruction points
        at a panel the user has to go hunt for. A button on those tabs calls this so the
        roster is one click away."""
        split = getattr(self, "center_split", None)
        if split is not None:
            sizes = split.sizes()
            if len(sizes) >= 2 and sizes[1] < 80:        # collapsed / too narrow to use → re-open
                want = self._right_panel_w or max(440, min(500, int(self.width() * 0.34)))
                split.setSizes([max(1, sum(sizes) - want), want])
        sec = getattr(self, "_samples_section", None)
        if sec is None:
            self.statusBar().showMessage("Samples panel isn't available yet.")
            return
        sec.header.setChecked(True)                      # expand (no-op if already open)
        scroll = getattr(self, "right_panel", None)
        if scroll is not None:                           # defer: scroll after the layout settles
            QtCore.QTimer.singleShot(0, lambda: scroll.ensureWidgetVisible(sec))
        # Always give feedback — when the panel was already open the reveal is invisible, so a
        # status line tells the user where to look and what the buttons there do.
        self.statusBar().showMessage(
            "Samples panel (right) ▸ build your cohort: Open files… · Add saved… · Add regions…")

    # ---- roster persistence + registration ------------------------------ #
    def _save_cohort(self) -> bool:
        """Persist the roster. Returns True on success, False if the write failed (read-only
        dir / full disk) — callers that report success to the user (e.g. relocation) must check
        this rather than assume the save stuck."""
        try:
            self.cohort.save()
            return True
        except Exception:  # noqa: BLE001 — best-effort, mirrors session autosave
            return False

    def _active_session_path(self):
        """Managed session path of the loaded sample (the lazily-assigned one, or the
        path it *would* auto-save to). ``None`` for the throwaway demo / no dataset."""
        if getattr(self, "_session_path", None):
            return self._session_path
        if self.ds is not None and self.ds.source != "synthetic":
            try:
                # resolve, not managed_path: point the roster at the file that actually holds
                # this slide's ROIs even if a past fingerprint drift saved them under another key
                return session.resolve_session_path(
                    self.ds.source, library.dataset_fingerprint(self.ds), self.ds.n_pixels)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _register_active_in_cohort(self):
        """Add the loaded sample to the cohort (idempotent). Called after an auto-save
        and after a session is applied, so any file you actually work on joins the roster.
        Skips the demo, which has no persisted session."""
        if getattr(self, "cohort", None) is None or self.ds is None:
            return
        if self.ds.source == "synthetic":
            return
        sp = self._active_session_path()
        if not sp:
            return
        sp_abs = os.path.abspath(sp)
        src_abs = os.path.abspath(self.ds.source) if self.ds.source else ""
        # if this file is already represented by region-samples, don't also add it whole
        # (that would double-count the same tissue in a group comparison). Match on the
        # stable source, falling back to the session path for older region samples.
        def _same_slide(s):
            if src_abs and s.source and os.path.abspath(s.source) == src_abs:
                return True
            return bool(s.session_path) and os.path.abspath(s.session_path) == sp_abs
        if any(s.region and _same_slide(s) for s in self.cohort.samples):
            self._refresh_sample_tree()
            return
        ref = cohort_engine.SampleRef(
            name=session._display_name(self.ds.source), session_path=sp,
            source=self.ds.source,
            fingerprint=library.dataset_fingerprint(self.ds), n_pixels=self.ds.n_pixels)
        before = self.cohort.find(ref.key())
        if before is not None:
            # same sample already on the roster — refresh its pointer to the current
            # session (keeping its group), never add a second entry
            before.session_path = sp
            before.fingerprint = ref.fingerprint
            before.n_pixels = ref.n_pixels
        else:
            self.cohort.add(ref)
        self._save_cohort()
        self._refresh_sample_tree()

    # ---- tree rendering -------------------------------------------------- #
    def _refresh_sample_tree(self):
        tree = getattr(self, "sample_tree", None)
        if tree is None:
            return
        # identity key of the loaded whole-slide sample, so it highlights regardless of the
        # session path (which keys on the stable source now, not the managed file)
        active_key = None
        if self.ds is not None and self.ds.source and self.ds.source != "synthetic":
            asp = self._active_session_path()
            if asp:
                active_key = cohort_engine.SampleRef(
                    name="", session_path=asp, source=self.ds.source).key()
        q = (getattr(self, "sample_filter", None).text().strip().lower()
             if getattr(self, "sample_filter", None) is not None else "")
        tree.blockSignals(True)
        tree.clear()
        # imported metadata (1.4) becomes real, sortable columns beyond the fixed four
        meta_cols = self.cohort.meta_keys()
        headers = ["Name", "Group", "Region", "#px"] + meta_cols
        tree.setColumnCount(len(headers))
        tree.setHeaderLabels(headers)
        sort_col, sort_order = getattr(self, "_sample_sort", None) or (0, QtCore.Qt.AscendingOrder)
        buckets = self.cohort.by_group()
        # show named groups first (in first-seen order), then the ungrouped bucket
        order = self.cohort.groups() + ([""] if "" in buckets else [])
        n_shown = 0
        for gi, g in enumerate(order):
            refs = [r for r in buckets.get(g, []) if self._sample_matches_filter(r, q)]
            if not refs:                          # hide groups with nothing matching the filter
                continue
            # sort rows WITHIN each group by the chosen column (default: name), so a specific
            # sample is easy to find; the group order itself is fixed above (never reordered)
            refs.sort(key=lambda r: self._sample_cell_value(r, sort_col, meta_cols),
                      reverse=(sort_order == QtCore.Qt.DescendingOrder))
            head = QtWidgets.QTreeWidgetItem([f"{g or UNGROUPED}  ({len(refs)})"])
            head.setFlags(head.flags() & ~QtCore.Qt.ItemIsSelectable)
            f = head.font(0)
            f.setBold(True)
            head.setFont(0, f)
            if g:
                hue = REGION_PALETTE[gi % len(REGION_PALETTE)]
                head.setForeground(0, QtGui.QBrush(QtGui.QColor(hue)))
            tree.addTopLevelItem(head)
            for r in refs:
                npx = (r.region_n_pixels if r.region else r.n_pixels) or 0
                px = f"{npx:,}" if npx else "—"       # header already says '#px' (no suffix → numeric sort)
                # region samples carry "file · region" names already; a ▸ marks them as a
                # sub-slide replicate, ▶ marks the loaded sample
                name = (f"▶ {r.name}" if (active_key and r.key() == active_key)
                        else (f"▸ {r.name}" if r.region else r.name))
                cells = [name, r.group or "", r.region or "", px] \
                    + [str((r.meta or {}).get(k, "")) for k in meta_cols]
                child = QtWidgets.QTreeWidgetItem(cells)
                child.setData(0, QtCore.Qt.UserRole, r.key())
                child.setToolTip(0, self._sample_tooltip(r))
                if active_key and r.key() == active_key:       # the loaded sample
                    cf = child.font(0)
                    cf.setBold(True)
                    child.setFont(0, cf)
                head.addChild(child)
                n_shown += 1
            head.setExpanded(True)
        tree.blockSignals(False)
        if getattr(self, "sample_filter", None) is not None:
            base = ("Show only samples whose name, group, region or metadata contains this text.")
            self.sample_filter.setToolTip(
                f"{n_shown} of {len(self.cohort.samples)} sample(s) match. " + base if q else base)
        self._refresh_cohort_group_combos()
        self._refresh_nested_combos()      # groups + the subject metadata key

    def _sample_cell_value(self, r, col, meta_cols):
        """Sort key for roster column ``col`` — numeric for #px, natural key otherwise (so
        'a2' precedes 'a10' and #px sorts 100 < 200, not lexically). Out-of-range columns
        (a stored sort whose meta column vanished) fall back to an empty natural key."""
        npx = (r.region_n_pixels if r.region else r.n_pixels) or 0
        base = [(r.name or ""), (r.group or ""), (r.region or ""), npx]
        if col < len(base):
            return base[col] if col == 3 else common._natural_key(str(base[col]).lower())
        i = col - len(base)
        key = meta_cols[i] if 0 <= i < len(meta_cols) else None
        return common._natural_key(str((r.meta or {}).get(key, "")).lower())

    def _sort_sample_tree(self, col, order):
        """Sort roster rows within each group by ``col`` (the group order itself is preserved),
        and remember it so a later refresh (set-group, add, filter) re-applies the sort."""
        self._sample_sort = (int(col), order)
        self.sample_tree.header().setSortIndicator(int(col), order)
        self._refresh_sample_tree()

    def _on_sample_header_clicked(self, col):
        """A roster column header was clicked → toggle ascending/descending on that column
        (first click ascending), then re-sort within each group."""
        col = int(col)
        prev = getattr(self, "_sample_sort", None)
        asc = QtCore.Qt.AscendingOrder
        order = (QtCore.Qt.DescendingOrder if prev and prev[0] == col and prev[1] == asc else asc)
        self._sort_sample_tree(col, order)

    # ---- filtering + metadata display ----------------------------------- #
    def _sample_matches_filter(self, ref, q):
        """True when the sample matches the roster filter ``q`` (already lower-cased).
        Matches across name, group, region, folder, and every metadata key/value, so a
        biologist can filter by 'control', a slide id, or 'sex=F' alike."""
        if not q:
            return True
        hay = [ref.name, ref.group, ref.region, ref.folder()]
        for k, v in (ref.meta or {}).items():
            hay += [str(k), str(v), f"{k}={v}"]
        return any(q in (s or "").lower() for s in hay)

    def _sample_meta_summary(self, ref, limit=3):
        """A compact 'k=v · k=v' string of the first few metadata fields, for the row."""
        bits = [f"{k}={v}" for k, v in (ref.meta or {}).items() if v not in (None, "")]
        return " · ".join(bits[:limit])

    def _sample_tooltip(self, ref):
        loc = (f"region '{ref.region}' of " if ref.region else "") + (ref.source or ref.session_path)
        bits = [f"{k}: {v}" for k, v in (ref.meta or {}).items() if v not in (None, "")]
        return loc + ("\n" + "  ·  ".join(bits) if bits else "")

    # ---- adding samples -------------------------------------------------- #
    def _cohort_open_files(self):
        paths, _ = filedialogs.get_open_file_names(
            self, "Open imzML files", "", "imzML (*.imzML)")
        if not paths:
            return
        # Register the not-yet-loaded files now (so they show immediately), then load the
        # first as the active sample. A file with no prior session shows up but only
        # carries peaks once you find peaks on it (which auto-saves its session).
        for p in paths[1:]:
            try:
                sp = session.managed_path(p, "")          # fingerprint unknown until loaded
                self.cohort.add(cohort_engine.SampleRef(
                    name=session._display_name(p), session_path=sp, source=p))
            except Exception:  # noqa: BLE001
                pass
        self._save_cohort()
        self._refresh_sample_tree()                   # show the queued files now, not after load
        self._flush_autosave()
        self._load_dataset_path(paths[0])
        self.statusBar().showMessage(
            f"Opening {session._display_name(paths[0])}"
            + (f" (+{len(paths) - 1} more added to the cohort)" if len(paths) > 1 else ""))

    def _cohort_add_saved(self):
        """Pick from the samples you've already worked on (their managed sessions)."""
        avail = sorted((r for r in cohort_engine.discover_samples()
                        if self.cohort.find(r.key()) is None),
                       key=lambda r: (r.name or "").lower())
        if not avail:
            QtWidgets.QMessageBox.information(
                self, "Add saved samples",
                "No other saved samples found. Open a file and find peaks to create one.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add saved samples to cohort")
        dlg.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel("Tick the samples to add. Set their group after adding."))
        by_key = {r.key(): r for r in avail}
        lst = common.CheckList(
            [(r.key(), r.name + (f"   ·  {r.n_pixels:,} px" if r.n_pixels else ""))
             for r in avail], noun="sample")
        lay.addWidget(lst, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        chosen = lst.checked_in_order()
        for key in chosen:
            self.cohort.add(by_key[key])
        added = len(chosen)
        if added:
            self._save_cohort()
            self._refresh_sample_tree()
            self.statusBar().showMessage(f"Added {added} sample(s) to the cohort.")

    def _region_measure_targets(self):
        """The m/z at which a region *without its own feature list* is measured when it joins
        the cohort, plus a short label naming where they came from.

        Locked to the segmentation's own feature set when there is one. The regions being
        promoted were cut from that segmentation, so measuring them at some other list's m/z
        would compare regions on features the clustering never saw — and the list the user
        picked peaks with before segmenting is precisely the one they meant. Falls back to the
        app-wide default list when no segmentation exists (drawn ROIs on a fresh slide).

        One rule for every promoted region, deliberately. Locking segmentation-derived regions
        to ``seg.peaks`` while drawn ROIs took the default list would seed a cohort with two
        feature axes — and :func:`cohort.consensus_targets` unions them silently (its
        ``min_prevalence`` defaults to 0), so the mismatch would never surface."""
        seg = getattr(self, "seg", None)
        raw = getattr(seg, "peaks", None)
        # `raw or []` would raise on an ndarray ("truth value is ambiguous"); entries may be
        # floats or peak dicts depending on which caller built the segmentation.
        seg_mzs = [] if raw is None else [float(p["mz"] if isinstance(p, dict) else p)
                                          for p in list(raw)]
        if seg_mzs:
            return sorted(set(seg_mzs)), "the feature list this segmentation was built on"
        return self._default_feature_targets(), "the default feature list"

    def _default_feature_targets(self):
        """Sorted-by-caller m/z of the app-wide *default* feature list — the same set new
        analyses default to (the list last actively selected / pinned, else the working
        set). Used to measure a region that has no feature list of its own when it joins the
        cohort, so all such regions share the one larger list."""
        data = (self._default_feature_set_data()
                if hasattr(self, "_default_feature_set_data") else None)
        if data and data[0] not in ("default", "consensus"):
            kind, name = data
            src = (getattr(self, "_feature_scopes", {}) if kind == "scope"
                   else getattr(self, "_feature_lists", {})) or {}
            mzs = [float(r["mz"]) for r in (src.get(name) or [])]
            if mzs:
                return mzs
        return list(self._visible_mzs()) if hasattr(self, "_visible_mzs") else []

    def _region_peaks_from_targets(self, mask, targets):
        """Build a region's feature-scope peak list by sampling the region's OWN mean
        spectrum (mask-scoped) at each m/z in ``targets`` (the default feature list). Each
        entry's ``intensity`` is the max within ±ppm of the target; ``rel_intensity`` is
        normalised to the region spectrum's peak — matching how :meth:`Dataset.pick_peaks`
        reports a region's own list, so a region measured this way is comparable to one whose
        peaks were found directly. Empty when there are no targets or the region has no
        signal."""
        targets = sorted({float(m) for m in (targets or []) if m and float(m) > 0})
        if not targets or mask is None:
            return []
        # Prefer the in-memory m/z cube (instant CSR.T@vec) over a full-resolution streaming
        # disk re-read of the region's pixels — we only sample at target m/z, so the cube's bin
        # resolution is enough (the nearest-bin fallback below covers coarse bins). Matches the
        # cube-first pattern used for every other region spectrum; falls back to the exact
        # streaming mean only when no cube has been built.
        cube_mean = getattr(self.ds, "cube_mean_spectrum", None)
        ms = cube_mean(mask) if cube_mean is not None else None
        if ms is None:
            ms = self.ds.mean_spectrum(mask=mask)
        axis, spec = ms
        axis = np.asarray(axis, dtype=float)
        spec = np.asarray(spec, dtype=float)
        if not spec.size or spec.max() <= 0:
            return []
        base = float(spec.max())
        tol = float(getattr(self, "ppm", 0.0) or 0.0) * 1e-6
        peaks = []
        for mz in targets:
            w = mz * tol
            lo = int(np.searchsorted(axis, mz - w))
            hi = int(np.searchsorted(axis, mz + w, side="right"))
            if hi <= lo:                               # no bin in window → nearest bin
                j = int(np.argmin(np.abs(axis - mz)))
                inten = float(spec[j])
            else:
                inten = float(spec[lo:hi].max())
            peaks.append({"mz": mz, "intensity": inten,
                          "rel_intensity": inten / base if base > 0 else 0.0})
        return peaks

    def _cohort_add_regions(self):
        """Promote regions of the *active* sample into the cohort as their own samples, so
        an area of a slide can act as an independent replicate. A region that already has a
        built feature list (a feature scope) uses it; a region without one is measured on the
        fly at the app-wide *default* feature list's m/z — within that region's own pixels —
        so it can join the cohort on the shared (larger) list without a separate Find-peaks
        pass, while staying a distinct replicate (its intensities are region-scoped, not the
        slide's). Either way the cohort reads the region's intensities from its scope with no
        dataset reload."""
        sp = self._active_session_path()
        if not sp or self.ds is None:
            QtWidgets.QMessageBox.information(
                self, "Add regions",
                "Load a sample first, then create regions (Segmentation tab) to add as samples.")
            return
        regions = getattr(self, "regions", []) or []
        scopes = getattr(self, "_feature_scopes", {}) or {}
        addable = [rg for rg in regions if self._region_pixel_mask(rg) is not None]
        if not addable:
            QtWidgets.QMessageBox.information(
                self, "Add regions",
                "No regions found on this sample. Create regions in the Segmentation tab "
                "(draw an ROI or cut the dendrogram), then add them as samples.")
            return
        base = session._display_name(self.ds.source)
        fp = library.dataset_fingerprint(self.ds)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add regions as cohort samples")
        dlg.setMinimumWidth(420)
        lay = QtWidgets.QVBoxLayout(dlg)
        _targets, _src_label = self._region_measure_targets()
        lay.addWidget(QtWidgets.QLabel(
            f"Regions of '{base}'. Ticked regions are added as samples; set their group "
            f"afterwards. A region without its own feature list is measured within that region "
            f"at the m/z of {_src_label} ({len(_targets):,} features). (The whole-slide entry "
            "for this file is removed so the same tissue isn't counted twice.)"))
        # Filter + All/None/Invert + shift-click-then-Space, instead of one click per region.
        npx_by_name, entries = {}, []
        for rg in addable:
            mask = self._region_pixel_mask(rg)
            npx = int(mask.sum()) if mask is not None else None
            npx_by_name[rg["name"]] = npx
            already = self.cohort.find(cohort_engine.SampleRef(
                name="", session_path=sp, source=self.ds.source,
                region=rg["name"]).key()) is not None
            tag = "own list" if rg["name"] in scopes else "default list"
            label = (f"{rg['name']}" + (f"   ·  {npx:,} px" if npx else "")
                     + ("   (already added)" if already else ""))
            entries.append((rg["name"], label, tag, not already))
        lst = common.CheckList(entries, noun="region")
        lay.addWidget(lst, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        # Resolve the checked regions (and their pixel masks) on the GUI thread — masks use
        # live region/segmentation state — then hand the heavy per-region spectrum measurement
        # off to a background job. Measuring a region streams its pixels from the cube, so doing
        # several inline froze the UI ("not responding"); this is the same _run path every other
        # cohort action uses, leaving the window responsive and the work cancellable.
        default_targets, _src = self._region_measure_targets()
        scopes_now = getattr(self, "_feature_scopes", {}) or {}
        plan = []
        for name in sorted(lst.checked_keys()):
            npx = npx_by_name.get(name)
            has_scope = name in scopes_now
            mask = None
            if not has_scope:
                rg = next((r for r in regions if r["name"] == name), None)
                mask = self._region_pixel_mask(rg) if rg is not None else None
            plan.append({"name": name, "npx": npx, "has_scope": has_scope, "mask": mask})
        if not plan:
            return
        ctx = {"sp": sp, "base": base, "fp": fp,
               "source": self.ds.source, "n_pixels": self.ds.n_pixels}

        def work(progress=None):
            # measure only the regions without their own feature list, at the default list's m/z
            need = [p for p in plan if not p["has_scope"]]
            measured, skipped = {}, []
            for k, p in enumerate(need):
                peaks = (self._region_peaks_from_targets(p["mask"], default_targets)
                         if p["mask"] is not None else [])
                if peaks:
                    measured[p["name"]] = peaks
                else:
                    skipped.append(p["name"])
                if progress is not None:
                    progress(k + 1, len(need))
            return {"measured": measured, "skipped": skipped}

        n = len(plan)
        self._run(work, want_progress=True,
                  busy=f"Measuring {n} region{'s' if n != 1 else ''}…",
                  on_done=lambda res: self._cohort_on_regions_measured(ctx, plan, res))

    def _cohort_on_regions_measured(self, ctx, plan, result):
        """Apply the off-thread region measurements back on the GUI thread: store each region's
        measured-at-default feature scope, add the regions to the cohort, drop this file's
        whole-slide entry (so a comparison doesn't count the same tissue twice), and refresh.
        Counterpart to the worker dispatched in :meth:`_cohort_add_regions`."""
        if not result:
            return
        measured = result.get("measured") or {}
        skipped = result.get("skipped") or []
        sp, base, fp = ctx["sp"], ctx["base"], ctx["fp"]
        source, n_pixels = ctx["source"], ctx["n_pixels"]
        added = built = 0
        for p in plan:
            name, npx = p["name"], p["npx"]
            if not p["has_scope"]:
                peaks = measured.get(name)
                if not peaks:                          # couldn't measure — already in skipped
                    continue
                self._feature_scopes[name] = peaks
                built += 1
            self.cohort.add(cohort_engine.SampleRef(
                name=f"{base} - {name}", session_path=sp, source=source, region=name,
                fingerprint=fp, n_pixels=n_pixels, region_n_pixels=npx))
            added += 1
        if built:
            # the regions were just given a measured-at-default feature scope — persist them to
            # this sample's session so the cohort can read each region's intensities back
            self._mark_dirty()
            self._flush_autosave()
            if hasattr(self, "_refresh_feature_set_combo"):
                self._refresh_feature_set_combo()
        if added:
            # this file is now represented by its regions — drop the whole-slide entry so a
            # group comparison doesn't count the same tissue both ways
            whole_key = cohort_engine.SampleRef(
                name="", session_path=sp, source=source).key()
            self.cohort.remove(whole_key)
            self._save_cohort()
            self._refresh_sample_tree()
            msg = f"Added {added} region(s) as cohort samples"
            if built:
                msg += f"; {built} measured at the default feature list"
            self.statusBar().showMessage(msg + ".")
        if skipped:
            QtWidgets.QMessageBox.warning(
                self, "Add regions",
                "Couldn't measure these regions — they have no feature list of their own and "
                "the default feature list is empty. Find peaks first, or set a default feature "
                "list:\n  • " + "\n  • ".join(skipped))

    # ---- switching + context menu --------------------------------------- #
    def _cohort_tree_activated(self, item, _col=0):
        key = item.data(0, QtCore.Qt.UserRole)
        if key:
            self._switch_to_sample(key)

    def _switch_to_sample(self, key):
        ref = self.cohort.find(key)
        if ref is None:
            return
        active = self._active_session_path()
        same_file = bool(active) and os.path.abspath(active) == os.path.abspath(ref.session_path)
        if same_file and not ref.region:
            self.statusBar().showMessage(f"{ref.name} is already the active sample.")
            return
        if same_file and ref.region:
            # the file is already loaded — just switch to the region's scope, no reload
            if self._activate_region_sample(ref.region):
                self.statusBar().showMessage(f"Showing region '{ref.region}'.")
            return
        if not os.path.exists(ref.session_path):
            # registered but never saved (opened, no peaks yet) — load the raw file instead
            if ref.source and os.path.exists(ref.source):
                self._flush_autosave()
                self._load_dataset_path(ref.source)
                return
            self.statusBar().showMessage(
                f"{ref.name} has no saved session yet — open it and find peaks first.")
            return
        self._flush_autosave()                       # persist the sample we're leaving
        # a region sample wants to land on its scope once the session finishes applying
        self._pending_region_sample = ref.region or None
        self._open_session_path(ref.session_path)    # reload its dataset + restore state
        self.statusBar().showMessage(f"Switching to {ref.name}…")

    def _activate_region_sample(self, name):
        """Make a promoted region the active view: switch to its feature scope and select
        it in the Segmentation tab. Returns True if the scope was found."""
        if not name:
            return False
        if name not in (getattr(self, "_feature_scopes", {}) or {}):
            self.statusBar().showMessage(
                f"Region '{name}' has no saved feature list on this sample anymore — "
                "rebuild it in the Segmentation tab.")
            return False
        self._switch_feature_scope(name)             # Features ▸ Feature-set selector path
        self._select_region_by_name(name)            # highlight it in the Segmentation list
        return True

    def _selected_sample_keys(self, at=None):
        """Keys of the selected sample rows; falls back to the row under ``at`` (a
        right-click position) when nothing is selected. Group headers carry no key."""
        items = [it for it in self.sample_tree.selectedItems()
                 if it.data(0, QtCore.Qt.UserRole)]
        if not items and at is not None:
            it = self.sample_tree.itemAt(at)
            if it is not None and it.data(0, QtCore.Qt.UserRole):
                items = [it]
        return [it.data(0, QtCore.Qt.UserRole) for it in items]

    def _cohort_group_menu(self, keys, into=None):
        """The assign-to-group / remove menu, shared by the right-click handler and the
        visible 'Set group' dropdown so both offer the same actions. Pass ``into`` to fill
        an existing (persistent) menu in place instead of building a throwaway one."""
        menu = into if into is not None else QtWidgets.QMenu(self)
        if into is not None:
            menu.clear()
        groups = self.cohort.groups()
        gm = menu.addMenu("Assign to group")
        for g in groups:
            gm.addAction(g, lambda _=False, gg=g: self._cohort_set_group(keys, gg))
        if groups:
            gm.addSeparator()
        gm.addAction("New group…", lambda: self._cohort_set_group(keys, None)).setIcon(icon("add"))
        gm.addAction("Ungroup", lambda: self._cohort_set_group(keys, "")).setIcon(icon("remove"))
        menu.addSeparator()
        menu.addAction("Remove from cohort",
                       lambda: self._cohort_remove(keys)).setIcon(icon("remove"))
        return menu

    def _cohort_fill_group_menu(self, menu):
        """Populate the 'Set group' dropdown live from the current selection. InstantPopup
        opens immediately, so (unlike the old click handler) it can't gate on selection —
        show a disabled hint when nothing is ticked instead of a status-bar message."""
        keys = self._selected_sample_keys()
        if not keys:
            menu.clear()
            menu.addAction("Select sample(s) in the list first").setEnabled(False)
            return
        self._cohort_group_menu(keys, into=menu)

    def _cohort_tree_menu(self, pos):
        keys = self._selected_sample_keys(at=pos)
        menu = self._cohort_group_menu(keys) if keys else QtWidgets.QMenu(self.sample_tree)
        if keys:
            menu.addSeparator()
        common.add_copy_actions(menu, self.sample_tree)
        menu.exec(self.sample_tree.viewport().mapToGlobal(pos))

    def _cohort_set_group(self, keys, group):
        if group is None:                            # prompt for a new label
            group, ok = QtWidgets.QInputDialog.getText(self, "New group", "Group name:")
            if not ok or not group.strip():
                return
            group = group.strip()
        for k in keys:
            self.cohort.set_group(k, group)
        self._save_cohort()
        self._refresh_sample_tree()

    def _cohort_remove(self, keys):
        if not keys:
            return
        n = len(keys)
        if not confirm(self, "Remove from cohort", f"Remove {n} sample(s) from the cohort? Their saved sessions on disk are kept; only the roster entry is dropped.", ok_text="Remove"):
            return
        unchecked = getattr(self, "_cohort_sample_unchecked", None)
        for k in keys:
            self.cohort.remove(k)
            if unchecked is not None:
                unchecked.discard(k)              # don't let a re-added same-key sample inherit
                                                  # the stale exclusion (it would be silently
                                                  # dropped from runs while shown in the roster)
        self._save_cohort()
        self._refresh_sample_tree()

    # ---- bulk selection + criteria grouping + metadata ------------------ #
    def _select_matching_samples(self):
        """Tick every sample currently shown (i.e. matching the filter) so 'Set group ▾'
        can label the whole filtered set in one step — the cohort-scale grouping flow."""
        tree = getattr(self, "sample_tree", None)
        if tree is None:
            return
        tree.clearSelection()
        n = 0
        it = QtWidgets.QTreeWidgetItemIterator(tree)
        while it.value():
            item = it.value()
            if item.data(0, QtCore.Qt.UserRole):       # a sample row (not a group header)
                item.setSelected(True)
                n += 1
            it += 1
        self.statusBar().showMessage(
            f"Selected {n} sample(s) — use 'Set group ▾' to label them." if n
            else "No samples match the filter.")

    def _autogroup_apply(self, changed, what):
        self._save_cohort()
        self._refresh_sample_tree()
        self.statusBar().showMessage(
            f"Grouped {changed} sample(s) by {what}." if changed
            else f"No samples were grouped by {what}.")

    def _cohort_group_by_folder(self):
        if not self.cohort.samples:
            self.statusBar().showMessage("Add samples to the cohort first.")
            return
        self._autogroup_apply(self.cohort.group_by_folder(), "folder")

    def _cohort_group_by_pattern(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Group by name pattern",
            "Regular expression matched against each sample name.\n"
            "The first capture group becomes the group (else the whole match).\n"
            "Examples:   ^(WT|KO)      batch\\d+      _(\\w+)_")
        if not ok or not text.strip():
            return
        try:
            changed = self.cohort.group_by_pattern(text.strip())
        except re.error as e:
            QtWidgets.QMessageBox.warning(self, "Group by pattern", f"Invalid pattern: {e}")
            return
        self._autogroup_apply(changed, "name pattern")

    def _cohort_group_by_meta(self):
        keys = self.cohort.meta_keys()
        if not keys:
            QtWidgets.QMessageBox.information(
                self, "Group by metadata",
                "These samples have no metadata yet. Use 'Import metadata…' to load a CSV first.")
            return
        key, ok = QtWidgets.QInputDialog.getItem(
            self, "Group by metadata", "Metadata field:", keys, 0, False)
        if not ok or not key:
            return
        self._autogroup_apply(self.cohort.group_by_meta(key), f"metadata '{key}'")

    # ---- batch assignment (the technical axis, for ComBat batch correction) ---- #
    def _autobatch_apply(self, changed, what):
        self._save_cohort()
        self.statusBar().showMessage(
            f"Assigned {changed} sample(s) to batches by {what} (for batch correction)."
            if changed else f"No samples were batched by {what}.")

    def _cohort_batch_by_folder(self):
        if not self.cohort.samples:
            self.statusBar().showMessage("Add samples to the cohort first.")
            return
        self._autobatch_apply(self.cohort.batch_by_folder(), "folder")

    def _cohort_batch_by_pattern(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Batch by name pattern",
            "Regular expression matched against each sample name.\n"
            "The first capture group becomes the batch (else the whole match).\n"
            "Examples:   run(\\d+)      _(\\d{8})_      (plateA|plateB)")
        if not ok or not text.strip():
            return
        try:
            changed = self.cohort.batch_by_pattern(text.strip())
        except re.error as e:
            QtWidgets.QMessageBox.warning(self, "Batch by pattern", f"Invalid pattern: {e}")
            return
        self._autobatch_apply(changed, "name pattern")

    def _cohort_batch_by_meta(self):
        keys = self.cohort.meta_keys()
        if not keys:
            QtWidgets.QMessageBox.information(
                self, "Batch by metadata",
                "These samples have no metadata yet. Use 'Import metadata…' to load a CSV first.")
            return
        key, ok = QtWidgets.QInputDialog.getItem(
            self, "Batch by metadata", "Metadata field (e.g. acquisition date / matrix lot):",
            keys, 0, False)
        if not ok or not key:
            return
        self._autobatch_apply(self.cohort.batch_by_meta(key), f"metadata '{key}'")

    def _cohort_correct_batches(self):
        """Run ComBat batch correction over the assigned batches and report before/after QC."""
        if getattr(self, "cohort", None) is None or len(self.cohort.samples) < 2:
            self.statusBar().showMessage("Add at least two samples to the cohort first.")
            return
        if not self.cohort.batches():
            self.statusBar().showMessage("Assign acquisition batches first (Auto-batch ▾).")
            return
        refs = self._cohort_included_refs()          # Samples ▾ subset (roster untouched)
        if len(refs) < 2:
            self.statusBar().showMessage("Tick at least two samples in the Samples ▾ picker "
                                         "to correct.")
            return
        seed = profiles.active_seed()

        def work(progress=None):
            sessions: dict = {}
            targets = cohort_engine.consensus_targets(
                refs, tol_ppm=cohort_engine.CONSENSUS_TOL_PPM, value="rel_intensity",
                sessions=sessions)
            if not targets:
                return {"error": "no_targets"}
            _out, summary, mixing = cohort_engine.correct_batches(
                refs, targets, protect=("group",), random_state=seed, sessions=sessions)
            return {"summary": summary, "mixing": mixing,
                    "n_targets": len(targets), "n_batches": len(self.cohort.batches())}

        self._run(work, want_progress=True, busy="Correcting batches…",
                  on_done=self._cohort_on_corrected)

    def _cohort_on_corrected(self, result):
        """Render the off-thread batch-correction QC (the before/after acceptance read)."""
        if not result:
            return
        if result.get("error") == "no_targets":
            self.statusBar().showMessage("No features shared across samples to correct — save "
                                         "peak lists on each sample first.")
            return
        s, m = result["summary"], result["mixing"]

        def _fmt(v):
            return "—" if v is None else f"{v:.2f}"

        QtWidgets.QMessageBox.information(
            self, "Batch correction QC",
            f"ComBat over {result['n_batches']} batches · {result['n_targets']} features\n\n"
            f"• Batch-variance reduced:  {s.var_reduction * 100:.0f}%\n"
            f"• kBET rejection (↓ better):  {m.rejection_before:.2f} → {m.rejection_after:.2f}\n"
            f"• Silhouette by batch (↓):  {_fmt(m.silhouette_batch_before)} → "
            f"{_fmt(m.silhouette_batch_after)}\n"
            f"• Silhouette by biology (preserve):  {_fmt(m.silhouette_biology_before)} → "
            f"{_fmt(m.silhouette_biology_after)}\n\n"
            "Diagnostic preview only — this correction is not applied; the batch comparison "
            "still runs on uncorrected data.")
        if hasattr(self, "record_step"):
            # DIAGNOSTIC/QC ONLY: the ComBat-corrected matrix is not persisted and the batch
            # comparison (_cohort_run) is computed on UNCORRECTED data. Record applied=False so
            # the exported Methods text says "assessed", not "removed" — never claim a correction
            # that didn't feed the reported result. (Gated in provenance.py.)
            self.record_step("batch_correction",
                             label="Batch-effect QC (diagnostic — not applied)",
                             params={"method": "combat", "applied": False,
                                     "n_batches": result["n_batches"],
                                     "n_features": result["n_targets"], "protected": ["group"],
                                     "var_reduction": float(s.var_reduction),
                                     "rejection_before": float(m.rejection_before),
                                     "rejection_after": float(m.rejection_after),
                                     "random_state": profiles.active_seed()})
        self.statusBar().showMessage(
            f"Batch correction QC: variance −{s.var_reduction * 100:.0f}%, "
            f"mixing {m.rejection_before:.2f}→{m.rejection_after:.2f}.")

    @staticmethod
    def _read_metadata_csv(path):
        """Read a CSV/TSV of sample metadata into (rows, columns); sniffs the delimiter."""
        with open(path, newline="", encoding="utf-8-sig") as f:
            head = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(head, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            cols = [c for c in (reader.fieldnames or []) if c]
            rows = [{k: v for k, v in r.items() if k} for r in reader]
        return rows, cols

    def _cohort_import_metadata(self):
        """Load a metadata CSV and attach it to the roster — match each row to a sample,
        fill its metadata, and optionally set its group, in one pass."""
        if not self.cohort.samples:
            self.statusBar().showMessage("Add samples to the cohort first.")
            return
        path, _ = filedialogs.get_open_file_name(
            self, "Import sample metadata (CSV)", "", "CSV / TSV (*.csv *.tsv *.txt)")
        if not path:
            return
        try:
            rows, cols = self._read_metadata_csv(path)
        except Exception as e:  # noqa: BLE001 — surface any parse error to the user
            QtWidgets.QMessageBox.warning(self, "Import metadata", f"Could not read the file:\n{e}")
            return
        if not rows or not cols:
            QtWidgets.QMessageBox.warning(self, "Import metadata", "The file has no usable rows.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Import sample metadata")
        form = QtWidgets.QFormLayout(dlg)
        form.addRow(QtWidgets.QLabel(
            f"{len(rows)} row(s), {len(cols)} column(s). Match each row to a sample by:"))
        key_combo = NoScrollComboBox()
        key_combo.addItems(cols)
        for i, c in enumerate(cols):                   # guess the identifier column
            if c.lower() in ("sample", "name", "file", "filename", "id", "source"):
                key_combo.setCurrentIndex(i)
                break
        form.addRow("Sample column:", key_combo)
        grp_combo = NoScrollComboBox()
        grp_combo.addItem("(none)")
        grp_combo.addItems(cols)
        for i, c in enumerate(cols):                   # guess the group column
            if c.lower() in ("group", "condition", "genotype", "treatment", "cohort"):
                grp_combo.setCurrentIndex(i + 1)
                break
        form.addRow("Group column:", grp_combo)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        key_col = key_combo.currentText()
        group_col = grp_combo.currentText() if grp_combo.currentIndex() > 0 else None
        matched = self.cohort.apply_metadata(rows, key_col=key_col, group_col=group_col)
        self._save_cohort()
        self._refresh_sample_tree()
        if matched:
            self.statusBar().showMessage(
                f"Applied metadata to {matched} of {len(self.cohort.samples)} sample(s).")
        else:
            QtWidgets.QMessageBox.information(
                self, "Import metadata",
                "No rows matched a sample. Check that the sample column holds names or "
                "filenames matching the roster (e.g. the imzML filename).")

    def _goto_cohort_tab(self):
        # Cohort is nested under the "Cohort" container now; reveal_view selects both the
        # container tab and its inner Cohort sub-tab by original label.
        self.reveal_view("Cohort")

    # ===================================================================== #
    # Cohort tab — cross-sample comparison
    # ===================================================================== #
    def _tab_cohort(self):
        w, v = tab_page()

        # ControlBar keeps this dense row (8 labelled controls + 2 buttons) from
        # overflowing under the dock — each label+control stays glued as one wrapping unit.
        bar = ControlBar()
        self.cohort_feat_combo = self._make_consensus_feature_combo(
            "Feature axis for the comparison:\n"
            "Consensus — a shared m/z axis clustered from every sample's peaks.\n"
            "Or any named feature set — the working set, a region scope, or a saved ★ list "
            "(matched against each sample's saved peaks).")
        self.cohort_value_combo = NoScrollComboBox()
        self.cohort_value_combo.addItems(["rel_intensity", "intensity"])
        self.cohort_value_combo.setToolTip("Per-peak value to compare. Relative intensity "
                                           "is the most comparable across slides.")
        self.cohort_norm_combo = NoScrollComboBox()
        self.cohort_norm_combo.addItems(["median", "sum", "none"])
        self.cohort_norm_combo.setToolTip(
            "Cross-slide normalization before the group test, so slide-to-slide total signal "
            "doesn't bias the comparison.\nmedian: equalize each sample's median (robust "
            "default) · sum: total-ion equalize · none: raw values.")
        self.cohort_tol_spin = NoScrollDoubleSpinBox()
        self.cohort_tol_spin.setRange(1.0, 200.0)
        self.cohort_tol_spin.setValue(20.0)
        self.cohort_tol_spin.setSuffix(" ppm")
        self.cohort_prev_spin = NoScrollDoubleSpinBox()
        self.cohort_prev_spin.setRange(0.0, 1.0)
        self.cohort_prev_spin.setSingleStep(0.1)
        self.cohort_prev_spin.setValue(0.5)
        self.cohort_prev_spin.setToolTip("Consensus only: keep ions present in at least "
                                         "this fraction of samples.")
        # Min prevalence only applies in Consensus mode; grey it out for a named feature set.
        def _sync_prev_enabled():
            self.cohort_prev_spin.setEnabled(self._cohort_combo_is_consensus(self.cohort_feat_combo))
        self.cohort_feat_combo.currentIndexChanged.connect(lambda *_: _sync_prev_enabled())
        _sync_prev_enabled()
        self.cohort_ga = NoScrollComboBox()
        self.cohort_gb = NoScrollComboBox()
        self.cohort_test_combo = NoScrollComboBox()
        # label -> group_comparison(method=) key; samples are the replicates here, so the
        # parametric t-tests are statistically appropriate (unlike per-pixel region tests).
        self._cohort_tests = {"Mann-Whitney U": "mwu", "Welch's t-test": "welch",
                              "Student's t-test": "student"}
        self.cohort_test_combo.addItems(list(self._cohort_tests))
        self.cohort_test_combo.setToolTip(
            "Two-group significance test across samples:\n"
            "Mann-Whitney U: rank-based, non-parametric (robust default).\n"
            "Welch's t-test: parametric, unequal variance · Student's: parametric, pooled "
            "variance.\nt-tests assume roughly normal per-sample values and need ≥2 samples per "
            "group.\nWith Paired ticked these become Wilcoxon signed-rank (rank) / paired t.")
        # Paired design: match each subject's two samples (e.g. left/right nerve,
        # treated/control of one donor) by a metadata key and run a within-subject test —
        # more powerful than the unpaired default when the design is repeated-measures.
        self.cohort_paired_check = QtWidgets.QCheckBox("Paired")
        self.cohort_paired_check.setToolTip(
            "Repeated-measures design: each subject contributes to BOTH groups. Samples are "
            "matched across Group A/B by the 'Pair by' metadata key, and the test becomes a "
            "within-subject Wilcoxon signed-rank (or paired t-test). Removes between-subject "
            "variance, so it's more correct and more powerful for paired data.")
        self.cohort_pair_combo = NoScrollComboBox()
        self.cohort_pair_combo.setToolTip(
            "Metadata key that identifies the subject shared by a Group A / Group B pair "
            "(e.g. 'subject' or 'donor'). Set it in the Samples panel's metadata.")
        self.cohort_pair_combo.setEnabled(False)
        self.cohort_paired_check.toggled.connect(self.cohort_pair_combo.setEnabled)
        b_samples = QtWidgets.QPushButton("Samples…")
        b_samples.setToolTip("Open the Samples panel to add the slides this comparison reads")
        b_samples.clicked.connect(self._reveal_samples_panel)
        b_run = primary_button("Run batch comparison")
        b_run.setToolTip("Run the Group A vs B test across every cohort sample (samples are the replicates). Needs ≥2 samples in the cohort.")
        self.b_cohort_run = b_run
        b_run.setEnabled(len(getattr(getattr(self, 'cohort', None), 'samples', []) or []) >= 2)
        b_run.clicked.connect(self._cohort_run)
        self.cohort_glossary = common.glossary_button([
            ["Consensus features", "A shared m/z axis clustered from every sample's peaks, so all slides span the same features."],
            ["Min prevalence", "Keep a consensus feature only if it appears in at least this fraction of samples."],
            ["Normalize", "Cross-slide scaling (median / sum / none) before the test so total-signal differences don't bias it."],
            ["log2 FC (B/A)", "log2 fold-change of the per-sample mean, Group B over Group A."],
            ["p / q (FDR)", "p = raw test p-value; q = Benjamini-Hochberg FDR-adjusted p (q≤0.05 = significant here)."],
            ["Mann-Whitney U", "Rank-based non-parametric two-group test (robust default)."],
            ["Welch's / Student's t", "Parametric two-group t-tests (unequal / pooled variance); need ≥2 samples per group."],
            ["Paired", "Repeated-measures design: match each subject's two samples by a metadata key and test within subject (Wilcoxon signed-rank / paired t). More powerful when the same subject is in both groups."]], parent=w)
        for lab, wdg in (("Features", self.cohort_feat_combo), ("Group A", self.cohort_ga),
                         ("vs B", self.cohort_gb), ("Test", self.cohort_test_combo)):
            bar.add_group(lab, wdg)
        bar.add_more(("Value:", self.cohort_value_combo), ("Normalize:", self.cohort_norm_combo),
                     ("Match:", self.cohort_tol_spin), ("Min prevalence:", self.cohort_prev_spin),
                     self.cohort_paired_check, ("Pair by:", self.cohort_pair_combo),
                     label="More")
        # Export popover — table CSV, a saved feature list built from the hits, and
        # publication images of the volcano + heatmap. Mirrors the Region-comparison tab so
        # the batch view isn't the odd one out. Armed only once a run has produced results.
        self.b_cohort_csv = button("Comparison table (CSV)…", self._cohort_export_csv,
                                   tooltip="Save every feature's log2 FC / p / q / per-group means.",
                                   icon="export")
        self.b_cohort_list = button("Feature list (fold-change / q cutoff)…",
                                    self._cohort_build_list_dialog, icon="save",
                                    tooltip="Turn this comparison into a saved ★ feature list: keep the "
                                            "ions that separate the groups, by fold-change and/or q≤0.05.")
        self.b_cohort_volcano_img = button("Volcano image…", self._cohort_export_volcano_dialog,
                                           icon="export",
                                           tooltip="Export the volcano as a publication figure — a "
                                                   "live-preview screen with format / DPI / theme / "
                                                   "labels and the shared Style preset (PNG/TIFF/PDF/SVG).")
        self.b_cohort_heat_img = button("Heatmap image…", self._cohort_export_heatmap_dialog,
                                        icon="export",
                                        tooltip="Export the sample × feature heatmap as a publication "
                                                "figure — live preview with colormap / scaling / theme / "
                                                "group strip and the shared Style preset.")
        self._cohort_export_btns = (self.b_cohort_csv, self.b_cohort_list,
                                    self.b_cohort_volcano_img, self.b_cohort_heat_img)
        for _b in self._cohort_export_btns:
            _b.setEnabled(False)
        bar.add(self._build_cohort_sample_picker(), b_samples, b_run, self.cohort_glossary)
        bar.add_more(*self._cohort_export_btns, label="Export", name="export")
        # Instructional hint / result summary on its own wrapping line below the controls.
        self.cohort_info = bar.set_status(
            "Click 'Samples…' to add ≥2 samples to the cohort, then label them: select "
            "samples and click 'Set group ▾'. The labels become Group A / Group B above. "
            "Each sample's saved peaks are used — no reloading needed.")
        v.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.cohort_table = QtWidgets.QTableWidget()
        self.cohort_table.setToolTip("Sortable results. Use the ⤓ button above to export the table as CSV.")
        self.cohort_table.setSortingEnabled(True)
        table_placeholder(self.cohort_table,
                          "Set Group A / B on the samples above and ‘Run batch comparison’.\n"
                          "The ranked features appear here.")
        split.addWidget(self.cohort_table)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        heat_w = pg.GraphicsLayoutWidget()
        self.cohort_heat_w = heat_w               # kept for image export
        self.cohort_heat_plot = heat_w.addPlot()
        self.cohort_heat_plot.setLabel("bottom", "feature (m/z)")
        self.cohort_heat_plot.setLabel("left", "sample")
        self.cohort_heat_plot.setToolTip("Sample × feature (m/z) grid; each column is min-max scaled so high- and low-abundance ions are both visible.")
        self.cohort_heat_plot.invertY(True)
        self.cohort_heat = pg.ImageItem()
        # The grid is built (n_samples × n_features) = (row, col), and the m/z ticks + x-range
        # below run out to n_features. Row-major maps row→y (sample) and col→x (feature) to
        # match; col-major would put samples on x and cram the image into x∈[0, n_samples],
        # smushing every column toward x=0 whenever there are more features than samples.
        self.cohort_heat.setOpts(axisOrder="row-major")
        self.cohort_heat_plot.addItem(self.cohort_heat)
        right.addWidget(heat_w)

        self.cohort_volcano = pg.PlotWidget()
        self.cohort_volcano.setLabel("bottom", "log2 fold-change  (B / A)")
        self.cohort_volcano.setToolTip("Each point is one feature. Click a point to view that ion on the active sample.")
        self.cohort_volcano.setLabel("left", "−log10 p")
        # volcano + caption are ONE splitter pane: the caption is a thin label under the plot,
        # not a resizable pane (straight into the splitter it grabbed ~140px and squashed it).
        volw = QtWidgets.QWidget()
        volv = QtWidgets.QVBoxLayout(volw)
        volv.setContentsMargins(0, 0, 0, 0)
        volv.setSpacing(2)
        volv.addWidget(self.cohort_volcano, 1)
        volv.addWidget(common.plot_caption("Click a volcano point to view that ion on the active sample."))
        right.addWidget(volw)
        right.setSizes([320, 280])
        split.addWidget(right)
        split.setSizes([560, 640])
        v.addWidget(split, 1)

        self._cohort_tabs.addTab(w, "Cohort")
        self._update_cohort_run_enabled()

    def _refresh_cohort_group_combos(self):
        ga, gb = getattr(self, "cohort_ga", None), getattr(self, "cohort_gb", None)
        if ga is None or gb is None:
            return
        groups = self.cohort.groups()
        for combo in (ga, gb):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if groups:
                combo.addItems(groups)
                if prev in groups:
                    combo.setCurrentText(prev)
            else:
                # an empty combo box has no popup — clicking it does nothing, which reads
                # as broken. Show a disabled hint so the control explains itself instead.
                combo.addItem("— no groups: set them in Samples —")
                item = combo.model().item(0)
                if item is not None:
                    item.setEnabled(False)
            combo.blockSignals(False)
        if len(groups) >= 2 and gb.currentText() == ga.currentText():
            gb.setCurrentIndex(1)
        self._refresh_cohort_pair_combo()
        self._update_cohort_run_enabled()

    def _refresh_cohort_pair_combo(self):
        """Populate the 'Pair by' selector from the roster's metadata keys, preserving the
        current pick. No metadata keys → the paired control is unavailable (self-explaining)."""
        combo = getattr(self, "cohort_pair_combo", None)
        if combo is None:
            return
        keys = self.cohort.meta_keys()
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if keys:
            combo.addItems(keys)
            if prev in keys:
                combo.setCurrentText(prev)
        else:
            combo.addItem("— no metadata: set it in Samples —")
            item = combo.model().item(0)
            if item is not None:
                item.setEnabled(False)
        combo.blockSignals(False)
        chk = getattr(self, "cohort_paired_check", None)
        if chk is not None and not keys:
            chk.setChecked(False)
            chk.setEnabled(False)
        elif chk is not None:
            chk.setEnabled(True)

    def _update_cohort_run_enabled(self):
        ok = len(getattr(getattr(self, 'cohort', None), 'samples', []) or []) >= 2
        b = getattr(self, 'b_cohort_run', None)
        if b is not None:
            b.setEnabled(ok)
        b = getattr(self, 'b_cembed_run', None)
        if b is not None:
            b.setEnabled(self._cembed_run_ok())

    def _cembed_run_ok(self) -> bool:
        """Whether the pooled embedding can run. *Sample means* reduce each slide to one point,
        so they need ≥2 cohort samples to be a cloud at all. *Pixels* and *Region means* are
        both meaningful on a single slide — a per-pixel UMAP of one section is the classic
        lipid-atlas "molecular histology" figure, and region means compares that slide's named
        regions — so allow them at n≥1 (the engine still needs ≥2 points and says so if it
        doesn't get them). This is what lets one slide be embedded without padding the cohort."""
        n = len(getattr(getattr(self, 'cohort', None), 'samples', []) or [])
        if n >= 2:
            return True
        unit = (self.cembed_unit_combo.currentText()
                if getattr(self, 'cembed_unit_combo', None) is not None else "")
        return n >= 1 and unit in ("Region means", "Pixels")

    def _cohort_run(self):
        if getattr(self, "cohort", None) is None or len(self.cohort.samples) < 2:
            self.statusBar().showMessage("Add at least two samples to the cohort first "
                                         "— use the Samples panel.")
            self._reveal_samples_panel()
            return
        # Snapshot every widget value on the GUI thread; the heavy batch pipeline then runs
        # off-thread so a big cohort (many sessions × features × scipy tests) can't freeze the
        # window (it used to block the event loop with no cancel).
        refs = self._cohort_included_refs()          # Samples ▾ subset (roster untouched)
        if len(refs) < 2:
            self.statusBar().showMessage("Tick at least two samples in the Samples ▾ picker "
                                         "to compare.")
            return
        value = self.cohort_value_combo.currentText()
        tol = float(self.cohort_tol_spin.value())
        prev = float(self.cohort_prev_spin.value())
        norm = self.cohort_norm_combo.currentText()
        is_consensus = self._cohort_combo_is_consensus(self.cohort_feat_combo)
        named_mzs = None if is_consensus else sorted(
            {round(float(m), 4) for m in self._combo_feature_mzs(self.cohort_feat_combo)})
        groups = self.cohort.groups()
        ga = self.cohort_ga.currentText(); ga = ga if ga in groups else ""
        gb = self.cohort_gb.currentText(); gb = gb if gb in groups else ""
        method = self._cohort_tests.get(self.cohort_test_combo.currentText(), "mwu")
        paired = bool(self.cohort_paired_check.isChecked())
        pair_by = self.cohort_pair_combo.currentText() if paired else None
        if paired and pair_by not in self.cohort.meta_keys():
            self.statusBar().showMessage(
                "Paired needs a 'Pair by' metadata key that matches subjects across the two "
                "groups. Set the metadata in Samples, or untick Paired.")
            return
        mode, id_ppm = self.mode_combo.currentText(), self.id_ppm
        # Per-group size guard: a 1-vs-N comparison is statistically vacuous (a two-sided test
        # can't reach significance), so refuse it up front with a specific message instead of
        # returning an all-NaN table the user reads as a clean negative. For a paired design the
        # unit is the *pair*, so also require ≥2 subjects present in both groups.
        if ga and gb and ga != gb:
            # Count the INCLUDED refs, not the full roster — the comparison runs on `refs`
            # (Samples ▾ subset), so counting self.cohort.by_group() let an untick bypass the
            # guard and run a vacuous 1-vs-N comparison anyway.
            nA = sum(1 for r in refs if r.group == ga)
            nB = sum(1 for r in refs if r.group == gb)
            if min(nA, nB) < 2:
                self.statusBar().showMessage(
                    f"Each group needs ≥2 samples for a valid test ({ga} has {nA}, {gb} has "
                    f"{nB}). Add replicates or pick other groups.")
                return
            if paired:
                def _pk(r):
                    return str((getattr(r, "meta", {}) or {}).get(pair_by, "")).strip().lower()
                ka = {_pk(r) for r in refs if r.group == ga and _pk(r)}
                kb = {_pk(r) for r in refs if r.group == gb and _pk(r)}
                n_pairs = len(ka & kb)
                if n_pairs < 2:
                    self.statusBar().showMessage(
                        f"Paired test needs ≥2 subjects present in both groups (matched on "
                        f"'{pair_by}'): found {n_pairs}. Check the metadata or untick Paired.")
                    return

        def work(progress=None):
            from .. import pipeline      # lazy: keeps pandas/scipy.stats out of GUI startup
            sessions: dict = {}          # one parse per session, shared by consensus + table
            targets = (cohort_engine.consensus_targets(
                refs, tol_ppm=tol, min_prevalence=prev, value=value, sessions=sessions)
                if is_consensus else named_mzs)
            if not targets:
                return {"error": "no_targets"}
            tbl = cohort_engine.batch_feature_table(
                refs, targets, tol_ppm=tol, value=value, normalize=norm, sessions=sessions)
            if tbl.empty:
                return {"error": "no_samples", "tbl": tbl}
            res = None
            if ga and gb and ga != gb:
                res = cohort_engine.group_comparison(tbl, ga, gb, method=method,
                                                     paired=paired, pair_by=pair_by)
                attrs = dict(res.attrs)
                res = pipeline.annotate_df(res, "mz", mode=mode, ppm=id_ppm)
                res.attrs.update(attrs)         # annotate_df may drop attrs (labels/test/warning)
            return {"tbl": tbl, "res": res, "targets": list(targets), "ga": ga, "gb": gb,
                    "method": method, "paired": paired, "pair_by": pair_by,
                    "value": value, "norm": norm, "tol": tol, "n_groups": len(groups)}

        self._run(work, want_progress=True, busy=f"Comparing {len(refs)} samples…",
                  on_done=self._cohort_on_compared)

    def _cohort_on_compared(self, result):
        """Render the off-thread batch result back on the GUI thread: heatmap + (when two
        groups are chosen) the ranked table, volcano, audit/report entry and a summary that
        surfaces how many samples were skipped and how many features were actually testable."""
        if not result:
            return
        if result.get("error") == "no_targets":
            self.statusBar().showMessage("No features to compare — pick peaks/a list or lower "
                                         "the min-prevalence threshold.")
            return
        tbl = result.get("tbl")
        skipped = list((tbl.attrs.get("skipped") if tbl is not None else None) or [])
        if result.get("error") == "no_samples" or tbl is None or tbl.empty:
            extra = f" ({len(skipped)} skipped: missing/corrupt or no saved peaks)" if skipped else ""
            self.statusBar().showMessage("No samples with saved peaks found in this cohort." + extra)
            return
        self._cohort_tbl = tbl
        self._render_cohort_heat(tbl)
        skip_note = f" · {len(skipped)} skipped" if skipped else ""
        targets = result.get("targets") or []
        ga, gb = result.get("ga"), result.get("gb")
        res = result.get("res")
        if res is not None:
            self._cohort_res = res
            test_name = res.attrs.get("test", "")
            warning = res.attrs.get("warning", "")
            n_testable = res.attrs.get("n_testable")
            extra = self.record_step(
                "cohort_comparison",
                label=f"Cohort group comparison ({test_name or result.get('method')})",
                params={"test": test_name or result.get("method"), "value": result.get("value"),
                        "normalize": result.get("norm"), "tol_ppm": result.get("tol"),
                        "n_samples": int(len(tbl)), "n_features": int(len(targets)),
                        "n_testable": (int(n_testable) if n_testable is not None else None),
                        "n_skipped": len(skipped), "group_a": ga, "group_b": gb,
                        "paired": bool(res.attrs.get("paired")),
                        "pair_by": res.attrs.get("pair_by"),
                        "n_pairs": res.attrs.get("n_pairs")},
                regions=self._cohort_region_inputs(self._cohort_included_refs(), ga, gb))
            self._log_analysis_to_report(
                "Cohort group comparison", res, regions=f"{ga} vs {gb}",
                title=f"Cohort comparison — {ga} vs {gb}",
                caption=f"{len(tbl)} samples · {len(targets)} features · {test_name}",
                source_extra=extra)
            self._render_cohort_table(res)
            self._render_cohort_volcano(res)
            sig = int((res["q_value"] <= 0.05).sum())     # NaN (untested) compares False
            nA = self._cohort_maxn(res, "n_A")
            nB = self._cohort_maxn(res, "n_B")
            testable = f" ({int(n_testable)} testable)" if n_testable is not None else ""
            unit = "pairs" if res.attrs.get("paired") else "n"
            paired_note = (f" · paired on '{res.attrs.get('pair_by')}'"
                           if res.attrs.get("paired") else "")
            off = float(res.attrs.get("mass_offset_ppm", 0.0) or 0.0)
            recal_note = f" · IDs auto-recalibrated {off:+.1f} ppm" if off else ""
            info = (f"{len(tbl)} samples · {len(targets)} features{skip_note} · {ga} ({unit}={nA}) "
                    f"vs {gb} ({unit}={nB}){paired_note} · {test_name} — {sig} at q≤0.05{testable}"
                    f"{recal_note}. Click a volcano point to view that ion on the active sample.")
            if warning:
                info = "⚠ " + warning + "  " + info
            self.cohort_info.setText(info)
            for _b in self._cohort_export_btns:
                _b.setEnabled(True)
        else:
            self._cohort_res = None                # overview only: no A-vs-B result to export
            self._render_cohort_overview(tbl)
            n_groups = result.get("n_groups", 0)
            hint = ("Assign at least two groups (select samples in the Samples panel → "
                    "'Set group ▾') and pick Group A vs B for a statistical comparison.") \
                   if n_groups < 2 else "Pick two different groups for Group A vs B."
            self.cohort_info.setText(
                f"{len(tbl)} samples · {len(targets)} features{skip_note}. {hint}")
            self.b_cohort_csv.setEnabled(True)
            self.b_cohort_heat_img.setEnabled(True)
            self.b_cohort_list.setEnabled(False)        # no A-vs-B result → nothing to rank
            self.b_cohort_volcano_img.setEnabled(False)
        self._goto_cohort_tab()

    @staticmethod
    def _cohort_maxn(res, col):
        """Largest per-group n across features, NaN-safe (untested features carry no n)."""
        v = res[col].dropna() if col in res else []
        return int(v.max()) if len(v) else 0

    @staticmethod
    def _cohort_region_inputs(refs, ga, gb):
        """The cohort replicates (each a labelled ROI/sample) that made up groups A and B,
        as audit region inputs — so a cross-slide comparison records *which* samples fed each
        side, tagged A/B. Cross-sample replicates have no current-slide mask, so they carry a
        pixel count + 'cohort replicate' origin (no mask hash) rather than a content fingerprint."""
        out = []
        for r in refs:
            g = getattr(r, "group", "") or ""
            role = "A" if g == ga else ("B" if g == gb else "")
            if not role:
                continue
            npx = getattr(r, "region_n_pixels", None) or getattr(r, "n_pixels", None)
            out.append((role, {"name": getattr(r, "name", "?"), "n_pixels": npx,
                               "via": "cohort replicate"}))
        return out

    def _render_cohort_heat(self, tbl):
        feat_cols = [c for c in tbl.columns if c != "group"]
        M = tbl[feat_cols].to_numpy(dtype=float)
        # per-feature min-max scaling for display so high- and low-abundance ions are
        # both visible; NaN (absent in a sample) renders at the column floor.
        scaled = np.full_like(M, np.nan)
        for j in range(M.shape[1]):
            col = M[:, j]
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                scaled[:, j] = 0.0
                continue
            lo, hi = np.nanmin(col), np.nanmax(col)
            scaled[:, j] = (col - lo) / (hi - lo) if hi > lo else (ok.astype(float))
        scaled = np.nan_to_num(scaled, nan=0.0)
        self.cohort_heat.setImage(scaled, autoLevels=False, levels=(0.0, 1.0))
        cmap = colormap(self.cmap_combo.currentText())
        try:
            self.cohort_heat.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:  # noqa: BLE001 — colormap API differences are non-fatal
            pass
        ax_l = self.cohort_heat_plot.getAxis("left")
        ax_l.setTicks([[(i + 0.5, name) for i, name in enumerate(tbl.index)]])
        targets = tbl.attrs.get("targets") or []
        step = max(1, len(targets) // 18)            # avoid an unreadable wall of labels
        ax_b = self.cohort_heat_plot.getAxis("bottom")
        ax_b.setTicks([[(i + 0.5, f"{targets[i]:.2f}") for i in range(0, len(targets), step)]])
        self.cohort_heat_plot.setRange(xRange=(0, max(1, len(targets))),
                                       yRange=(0, max(1, len(tbl.index))), padding=0)

    def _render_cohort_table(self, res):
        def g(v, spec):                       # untested features carry NaN p/q/fc → show '—'
            return "—" if not np.isfinite(v) else format(v, spec)

        def idc(x):                           # MS1 ID confidence as "82%", blank if unscored
            v = x.get("id_confidence", float("nan"))
            return f"{int(v)}%" if isinstance(v, (int, float)) and v == v else ""
        paired = bool(res.attrs.get("paired"))
        n_head = "nPairs" if paired else "nA/nB"
        n_comp = int(res.attrs.get("n_compared", 0))
        rows = []
        for _, x in res.iterrows():
            n_cell = f"{int(x['n_A'])}" if paired else f"{int(x['n_A'])}/{int(x['n_B'])}"
            det = x.get("n_detected")
            det_cell = f"{int(det)}/{n_comp}" if det == det and n_comp else "—"
            rows.append((f"{x['mz']:.4f}", x.get("best_lipid", "") or "(unidentified)",
                         x.get("best_adduct", ""), idc(x), det_cell, g(x['mean_A'], '.3g'),
                         g(x['mean_B'], '.3g'), g(x['log2_fc'], '+.2f'), g(x['p_value'], '.2e'),
                         g(x['q_value'], '.2e'), n_cell))
        fill_table(self.cohort_table,
                   ["m/z", "lipid", "adduct", "ID conf", "detected", f"mean {res.attrs.get('a_label', 'A')}",
                    f"mean {res.attrs.get('b_label', 'B')}", "log2 FC", "p", "q (FDR)", n_head],
                   rows)
        common.set_header_tooltips(self.cohort_table, {"m/z": "Feature m/z compared across samples.", "ID conf": "MS1 lipid-ID confidence (0–100): mass accuracy + sibling-adduct corroboration + runner-up gap. Image-free — how trustworthy the lipid name on a significant hit is.", "detected": "Reproducibility: in how many of the compared samples this ion is actually detected (value > 0). A significant hit detected in only a few samples is less trustworthy than one seen across the cohort.", "log2 FC": "log2 fold-change of the per-sample mean, Group B over Group A. + = higher in B.", "p": "Raw p-value of the two-group test across samples.", "q (FDR)": "Benjamini-Hochberg false-discovery-rate adjusted p; q≤0.05 is the significance cut here.", "nA/nB": "Number of samples contributing to Group A / Group B.", "nPairs": "Number of matched subject pairs contributing to this paired test."})

    def _render_cohort_overview(self, tbl):
        feat_cols = [c for c in tbl.columns if c != "group"]
        targets = tbl.attrs.get("targets") or []
        n = len(tbl)
        rows = []
        for t, c in zip(targets, feat_cols):
            col = tbl[c].to_numpy(dtype=float)
            present = int((~np.isnan(col)).sum())
            mean = float(np.nanmean(col)) if present else float("nan")
            rows.append((f"{t:.4f}", f"{present}/{n}", f"{mean:.3g}"))
        fill_table(self.cohort_table, ["m/z", "present in", "mean value"], rows)
        common.set_header_tooltips(self.cohort_table, {"present in": "How many samples this feature is detected in (prevalence across the cohort).", "mean value": "Mean of the chosen per-peak value across the samples where it is present."})

    def _render_cohort_volcano(self, res):
        self.cohort_volcano.clear()
        self.cohort_volcano.addLine(x=0, pen=pg.mkPen(GUIDE_LINE))
        fc_all = res["log2_fc"].to_numpy(dtype=float)
        p_all = res["p_value"].to_numpy(dtype=float)
        q_all = res["q_value"].to_numpy(dtype=float)
        ok = np.isfinite(fc_all) & np.isfinite(p_all)   # tested features only (finite p AND fc)
        fc = fc_all[ok]
        y = -np.log10(np.clip(p_all[ok], 1e-12, 1))
        mz = res["mz"].to_numpy()[ok]
        sig = q_all[ok] <= 0.05
        # Significance line on the (continuous) −log10 p axis. BH rejects exactly the features
        # with p ≤ p_threshold, where p_threshold is the largest p among the q≤0.05 hits — so a
        # dashed line there separates the significant (opaque) points from the rest EXACTLY,
        # unlike a raw p=0.05 line which wouldn't match the q-based colouring. When nothing
        # clears FDR, fall back to a faint uncorrected p=0.05 reference (every point is faded
        # anyway, so it reads as "nothing significant", not a mislabelled threshold).
        if sig.any():
            p_thr = float(np.nanmax(p_all[ok][sig]))
            self.cohort_volcano.addLine(y=-np.log10(max(p_thr, 1e-12)),
                                        pen=pg.mkPen(GUIDE_LINE, style=QtCore.Qt.DashLine))
        else:
            faint = pg.mkPen(GUIDE_LINE, style=QtCore.Qt.DotLine, width=0.7)
            self.cohort_volcano.addLine(y=-np.log10(0.05), pen=faint)
        self.cohort_volcano.setLabel("left", "−log₁₀ p  (dashed = FDR q≤0.05 threshold)")
        self.cohort_volcano.setLabel("bottom", "log₂ fold-change (B / A)")
        # Colour EVERY point by direction (down in A / up in B) — not just the significant
        # ones. With small cohorts nothing may clear q≤0.05, and an all-grey plot reads as
        # broken. Significance is shown by emphasis instead: significant ions are opaque,
        # larger and dark-ringed; the rest keep the same hue but fade back.
        a_qc, b_qc = QtGui.QColor(REGION_PALETTE[0]), QtGui.QColor(REGION_PALETTE[1])
        brushes, pens, sizes = [], [], []
        for f, s in zip(fc, sig):
            c = QtGui.QColor(b_qc if f > 0 else a_qc)
            c.setAlpha(235 if s else 110)
            brushes.append(pg.mkBrush(c))
            pens.append(pg.mkPen("#222", width=0.8) if s else pg.mkPen(None))
            sizes.append(11 if s else 7)
        sp = pg.ScatterPlotItem(x=fc, y=y, brush=brushes, size=sizes,
                                pen=pens, data=mz)
        sp.sigClicked.connect(self._cohort_volcano_clicked)
        self.cohort_volcano.addItem(sp)

    def _cohort_volcano_clicked(self, _sp, points):
        if points:
            self.set_active_mz(float(points[0].data()))
            self.reveal_view("Ion image")        # grouped-tab safe (index 0 is brittle post-IA)

    def _cohort_export_volcano_dialog(self):
        """Open the publication volcano export screen (live preview + style options) instead of
        dumping a raw screenshot of the on-screen pyqtgraph plot."""
        res = getattr(self, "_cohort_res", None)
        if res is None or not len(res):
            self.statusBar().showMessage("Run a Group A vs B comparison first.")
            return
        from .plotexport import VolcanoExportDialog
        VolcanoExportDialog(self, res, a_label=res.attrs.get("a_label", "A"),
                            b_label=res.attrs.get("b_label", "B"),
                            default_name="cohort_volcano.png").exec()

    def _cohort_export_heatmap_dialog(self):
        """Publication heatmap export screen (live preview + colormap/scaling/theme/group strip +
        Style preset), replacing the raw pyqtgraph screenshot."""
        tbl = getattr(self, "_cohort_tbl", None)
        if tbl is None or not len(tbl):
            self.statusBar().showMessage("Run a cohort comparison first.")
            return
        from .. import export, palettes
        from .plotexport import FigureExportDialog
        feat_cols = [c for c in tbl.columns if c != "group"]
        M = tbl[feat_cols].to_numpy(dtype=float)
        rows = [str(x) for x in tbl.index]
        targets = list(tbl.attrs.get("targets") or [])
        cols = ([f"{float(t):.2f}" for t in targets] if len(targets) == len(feat_cols)
                else [str(c).split("_")[-1] for c in feat_cols])
        groups = [str(g) for g in tbl["group"]] if "group" in tbl else None
        cmap0 = self.cmap_combo.currentText() if getattr(self, "cmap_combo", None) is not None else "viridis"

        def render(opts):
            return export.render_heatmap_figure(
                M, row_labels=rows, col_labels=cols, cmap=opts["cmap"], scale=opts["scale"],
                group_labels=(groups if opts["groups"] else None), title=opts["title"],
                theme=opts["theme"], dpi=opts["dpi"])

        options = [
            {"key": "title", "label": "Title", "kind": "text",
             "default": "Cohort heatmap (samples × features)"},
            {"key": "theme", "label": "Theme", "kind": "combo",
             "choices": [("Light", "light"), ("Dark", "dark")], "default": "light"},
            {"key": "cmap", "label": "Colormap", "kind": "combo",
             "choices": [(c, c) for c in palettes.SEQUENTIAL_CMAPS],
             "default": cmap0 if cmap0 in palettes.SEQUENTIAL_CMAPS else "viridis"},
            {"key": "scale", "label": "Scaling", "kind": "combo",
             "choices": [("Per feature", "col"), ("Per sample", "row"), ("Global", "none")],
             "default": "col",
             "tooltip": "Min-max normalise each feature (default, matches the on-screen view), "
                        "each sample, or globally."},
            {"key": "groups", "label": "Group colour strip", "kind": "check",
             "default": bool(groups), "enabled": bool(groups)},
        ]
        FigureExportDialog(self, render=render, options=options,
                           default_name="cohort_heatmap.png", title="Export heatmap").exec()

    def _cohort_export_csv(self):
        res = getattr(self, "_cohort_res", None)
        tbl = getattr(self, "_cohort_tbl", None)
        if res is None and tbl is None:
            return
        path, _ = filedialogs.get_save_file_name(
            self, "Export cohort comparison", "cohort_comparison.csv", "CSV (*.csv)")
        if not path:
            return
        (res if res is not None else tbl).to_csv(path, index=(res is None))
        self.statusBar().showMessage(f"Wrote {path}")

    def _cohort_build_list_dialog(self):
        """Batch comparison → saved ★ feature list. Keep the ions that separate the groups,
        cut by log2 fold-change (and/or q≤0.05) or a top-N by significance, preview live, save.
        Mirrors the Region-comparison 'Build feature list', keyed on fold-change instead of AUC."""
        res = getattr(self, "_cohort_res", None)
        if res is None or len(res) == 0 or "log2_fc" not in res.columns:
            self.statusBar().showMessage("Run a batch comparison (Group A vs B) first.")
            return
        df = res[np.isfinite(res["log2_fc"].to_numpy(dtype=float))]
        if len(df) == 0:
            self.statusBar().showMessage("No testable features to build a list from.")
            return
        na = res.attrs.get("a_label", "A"); nb = res.attrs.get("b_label", "B")
        has_q = "q_value" in df.columns

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Build feature list from batch comparison")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            f"log2 FC < 0 = higher in {na} · log2 FC > 0 = higher in {nb}.\n"
            "Pick which ions to keep, preview, then save as a ★ feature list."))
        ctl = QtWidgets.QHBoxLayout()
        target = NoScrollComboBox()
        target.addItems([f"Higher in {na}  (log2 FC ≤ −cut)", f"Higher in {nb}  (log2 FC ≥ +cut)",
                         "Either side (|log2 FC| ≥ cut)", "All tested ions"])
        target.setCurrentIndex(2)
        ctl.addWidget(QtWidgets.QLabel("Keep:")); ctl.addWidget(target)
        mode_combo = NoScrollComboBox()
        mode_combo.addItems(["by fold-change", "top N"])
        ctl.addWidget(mode_combo)
        fc_spin = NoScrollDoubleSpinBox()
        fc_spin.setRange(0.0, 10.0); fc_spin.setDecimals(2); fc_spin.setSingleStep(0.25)
        fc_spin.setValue(1.0); fc_spin.setPrefix("|log2 FC| ≥ ")
        n_spin = NoScrollSpinBox()
        n_spin.setRange(1, max(1, len(df))); n_spin.setValue(min(25, len(df)))
        n_spin.setVisible(False)
        ctl.addWidget(fc_spin); ctl.addWidget(n_spin)
        sig = QtWidgets.QCheckBox("q ≤ 0.05 only")
        sig.setEnabled(has_q); sig.setChecked(has_q)
        ctl.addWidget(sig)
        has_prev = "prevalence" in df.columns
        prev_spin = NoScrollSpinBox()
        prev_spin.setRange(0, 100); prev_spin.setSingleStep(10); prev_spin.setValue(0)
        prev_spin.setPrefix("detected ≥ "); prev_spin.setSuffix("%")
        prev_spin.setEnabled(has_prev)
        prev_spin.setToolTip("Reproducibility floor: keep only ions detected (value > 0) in at "
                             "least this fraction of the compared samples — cuts one-off hits.")
        ctl.addWidget(prev_spin)
        count_lbl = QtWidgets.QLabel(""); ctl.addWidget(count_lbl, 1)
        v.addLayout(ctl)
        preview = QtWidgets.QTableWidget(0, 4)
        preview.setHorizontalHeaderLabels(["m/z", "lipid", "log2 FC", "q"])
        preview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        preview.setMinimumSize(460, 300)
        common.install_table_export(preview, dlg, stem="feature_list_preview",
                                    title="Export preview")
        v.addWidget(preview, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        v.addWidget(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)

        def subset():
            d = df
            if sig.isChecked() and has_q:
                d = d[d["q_value"].astype(float) <= 0.05]
            if has_prev and prev_spin.value() > 0:      # reproducibility floor
                d = d[d["prevalence"].astype(float) * 100.0 >= prev_spin.value() - 1e-9]
            fcv = d["log2_fc"].astype(float)
            t = target.currentIndex()
            is_top = mode_combo.currentIndex() == 1
            cut = fc_spin.value()
            if t == 0:
                d = d.assign(__o=fcv).sort_values("__o")           # most negative first
                if not is_top:
                    d = d[d["log2_fc"].astype(float) <= -cut]
            elif t == 1:
                d = d.assign(__o=-fcv).sort_values("__o")          # most positive first
                if not is_top:
                    d = d[d["log2_fc"].astype(float) >= cut]
            elif t == 2:
                d = d.assign(__o=-fcv.abs()).sort_values("__o")    # biggest |FC| first
                if not is_top:
                    d = d[fcv.abs() >= cut]
            else:
                d = d.assign(__o=-fcv.abs()).sort_values("__o")
            if is_top:
                d = d.head(n_spin.value())
            return d.drop(columns="__o", errors="ignore")

        def refresh(*_):
            is_top = mode_combo.currentIndex() == 1
            fc_spin.setVisible(not is_top); n_spin.setVisible(is_top)
            d = subset()
            count_lbl.setText(f"{len(d)} ions")
            preview.setRowCount(len(d))
            for i, (_, x) in enumerate(d.iterrows()):
                q = x.get("q_value", float("nan"))
                cells = [f"{x['mz']:.4f}", (x.get("best_lipid", "") or ""),
                         format(float(x["log2_fc"]), "+.2f"),
                         ("—" if not np.isfinite(q) else format(float(q), ".1e"))]
                for j, txt in enumerate(cells):
                    preview.setItem(i, j, QtWidgets.QTableWidgetItem(txt))

        for wdg in (target, mode_combo):
            wdg.currentIndexChanged.connect(refresh)
        fc_spin.valueChanged.connect(refresh); n_spin.valueChanged.connect(refresh)
        sig.toggled.connect(refresh); prev_spin.valueChanged.connect(refresh)
        refresh()
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        d = subset()
        if len(d) == 0:
            self.statusBar().showMessage("No ions match — loosen the cutoff.")
            return
        feats = []
        for _, x in d.iterrows():
            q = x.get("q_value", float("nan"))
            qtag = "" if not np.isfinite(q) else f" · q {float(q):.1e}"
            feats.append({"mz": float(x["mz"]), "lipid": x.get("best_lipid", "") or "",
                          "note": f"{na} vs {nb} · log2 FC {float(x['log2_fc']):+.2f}{qtag}"})
        saved = self._save_feature_list_to_library(f"{na} vs {nb} markers", feats, prompt=True)
        if saved:
            self.statusBar().showMessage(f"Saved feature list '{saved}' ({len(feats)} ions).")

    # ===================================================================== #
    # Cohort nested stats — subject × compartment, hierarchical inference
    # ===================================================================== #
    def _tab_cohort_nested(self):
        """The nested-design screen: several compartments measured in every subject, subjects
        split into two groups. Sits beside the flat Cohort comparison because it answers a
        different question — not "does this ion differ between groups" but "does the group
        difference differ *by compartment*, once the subject each compartment came from is
        accounted for". See :mod:`smile_msi.hierstats`."""
        w, v = tab_page()
        bar = ControlBar()

        self.nest_feat_combo = self._make_consensus_feature_combo(
            "Feature axis for the nested comparison:\n"
            "Consensus — a shared m/z axis clustered from every sample's peaks.\n"
            "Or any named feature set — the working set, a region scope, or a saved ★ list.")
        self.nest_ga = NoScrollComboBox()
        self.nest_gb = NoScrollComboBox()
        for c in (self.nest_ga, self.nest_gb):
            c.setToolTip("The two groups compared. Subjects — not pixels, not compartments — "
                         "are the replicates, so each group needs ≥2 subjects.")
        # Subject and compartment are no longer picked from token-rule combos (two positional
        # knobs the user had to keep mutually opposite). Both are read from a per-sample mapping —
        # meta['subject'] / meta['compartment'] — that either button below writes: 'Auto-label'
        # seeds it from the cohort's own cross-slide structure in one click, 'Define
        # compartments…' lets it be set or corrected by hand.
        self.b_nest_auto = button(
            "Auto-label", self._nest_auto_apply,
            tooltip="Read each region-sample's SUBJECT (the nerve/donor — the unit of "
                    "replication), COMPARTMENT (the repeated measure inside each nerve) and "
                    "GROUP straight from its ROI label, using the cohort's own structure: a "
                    "token that names one slide and every region of it is the subject; a token "
                    "spanning slides is the compartment. One click; same result as accepting "
                    "'Define compartments…' unedited. Correct anything it gets wrong there.")
        self.b_nest_map = button(
            "Define compartments…", self._nest_design_dialog,
            tooltip="Map each region name to a compartment, and each group label to a group. "
                    "Use this when the names embed the nerve id, or when the group label mixes "
                    "the group with the compartment (e.g. 'Synk endo').")
        self.b_nest_class = button(
            "Lipid class test…", self._nest_class_dialog,
            tooltip="Roll the per-ion results up to lipid classes and test each class against "
                    "the rest of the lipidome, adjusted for the correlation between a class's "
                    "ions. Per-ion FDR is hopeless at these replicate counts; a class is where "
                    "a modest, coherent shift becomes testable.")
        self.b_nest_class.setEnabled(False)
        self._nest_comp_unchecked: set = set()
        self.nest_comp_picker = CheckPicker(
            "Compartments", self._nest_compartment_entries,
            lambda: self._nest_comp_unchecked,
            tooltip="Which compartments participate. This should list one entry per anatomical "
                    "compartment (endo / peri / epi) — if it lists one per nerve, change "
                    "'Compartment from'. Each must be present in both groups.",
            on_change=lambda: self._nest_update_design_summary(),
            noun="compartment", parent=w)

        self.nest_model_combo = NoScrollComboBox()
        self._nest_models = {"Mixed model (group × compartment)": "lmm",
                             "Stratified (test each compartment)": "stratified"}
        self.nest_model_combo.addItems(list(self._nest_models))
        self.nest_model_combo.setToolTip(
            "Mixed model: one fit per feature, y ~ group * compartment + (1 | subject). The "
            "subject random intercept encodes that a subject's compartments are correlated; "
            "the interaction asks whether the group difference DIFFERS by compartment.\n"
            "Stratified: an independent two-group test inside each compartment. Simpler to "
            "explain, but the compartments share subjects, so the three results aren't "
            "independent of each other.")
        self.nest_test_combo = NoScrollComboBox()
        self._nest_tests = {"Moderated t (empirical Bayes)": "modt", "Welch's t-test": "welch",
                            "Student's t-test": "student", "Mann-Whitney U": "mwu"}
        self.nest_test_combo.addItems(list(self._nest_tests))
        self.nest_test_combo.setToolTip(
            "Stratified only — the test run inside each compartment.\n"
            "Moderated t: limma-style empirical-Bayes variance shrinkage. Borrows variance "
            "across features, which is the only thing that buys usable power at small n.\n"
            "Mann-Whitney U: exact rank test. Honest, assumption-free, and at 3-vs-6 it "
            "cannot return an FDR-significant hit no matter how large the effect — the run "
            "will say so.")
        self.nest_test_combo.setEnabled(False)
        self.nest_model_combo.currentTextChanged.connect(
            lambda t: self.nest_test_combo.setEnabled(self._nest_models.get(t) == "stratified"))

        self.nest_summary_combo = NoScrollComboBox()
        self._nest_summaries = {"Median of pixels (robust)": "median",
                                "Mean of pixels (Cardinal meansTest)": "mean",
                                "Saved peak values (no reload)": "saved"}
        self.nest_summary_combo.addItems(list(self._nest_summaries))
        self.nest_summary_combo.setToolTip(
            "How each compartment's pixels collapse to one value per feature (pseudobulk).\n"
            "Median: robust to matrix hot-pixels and ablation artifacts — the safe default. "
            "Re-extracts pixels, so it reloads each slide once.\n"
            "Mean: the Cardinal meansTest convention. Also re-extracts.\n"
            "Saved peak values: reads what each region's peaks were saved with (a mean). "
            "Instant — no dataset loads — but you inherit that summary.")
        self.nest_norm_combo = NoScrollComboBox()
        self.nest_norm_combo.addItems(["tic", "median", "rms", "none"])
        self.nest_norm_combo.setToolTip(
            "Per-pixel normalization applied before the summary. Never computed against the "
            "compartment's own pixels — that would divide each compartment by its own mean and "
            "erase the differences being modelled.\n"
            "Only used by the pixel-summary modes.")
        self.nest_norm_scope_combo = NoScrollComboBox()
        self._nest_norm_scopes = {"On-tissue (all compartments)": "tissue",
                                  "Whole slide (incl. background)": "slide"}
        self.nest_norm_scope_combo.addItems(list(self._nest_norm_scopes))
        self.nest_norm_scope_combo.setToolTip(
            "Which pixels set the normalization's mean scale. That scale enters as ONE "
            "multiplicative constant per slide, so within-subject contrasts never care — but "
            "it differs between slides.\n"
            "On-tissue: the mean over the union of the slide's compartments. Independent of "
            "how much empty matrix the section sits in.\n"
            "Whole slide: the mean over every pixel, background included. A section with more "
            "background gets a different constant, which a mixed model's (1 | subject) term "
            "absorbs but a stratified test does not.")
        self.nest_transform_combo = NoScrollComboBox()
        self.nest_transform_combo.addItems(["log2", "none"])
        self.nest_transform_combo.setToolTip(
            "Variance stabilization before the model. Both tests assume roughly normal "
            "residuals and raw MSI intensities are strongly right-skewed, so log2(x + τ) is "
            "the default. Reported log2 fold-changes always come from the raw means either way.")
        self.nest_tol_spin = NoScrollDoubleSpinBox()
        self.nest_tol_spin.setRange(1.0, 200.0)
        self.nest_tol_spin.setValue(20.0)
        self.nest_tol_spin.setSuffix(" ppm")
        self.nest_prev_spin = NoScrollDoubleSpinBox()
        self.nest_prev_spin.setRange(0.0, 1.0)
        self.nest_prev_spin.setSingleStep(0.1)
        self.nest_prev_spin.setValue(0.5)
        self.nest_prev_spin.setToolTip("Consensus only: keep ions present in at least this "
                                       "fraction of samples.")

        b_samples = QtWidgets.QPushButton("Samples…")
        b_samples.setToolTip("Open the Samples panel to add the region-samples this model reads")
        b_samples.clicked.connect(self._reveal_samples_panel)
        self.b_nest_run = primary_button("Run nested comparison")
        self.b_nest_run.setToolTip(
            "Fit the nested model across every ticked region-sample. Needs ≥2 subjects per "
            "group and a subject metadata key.")
        self.b_nest_run.clicked.connect(self._nest_run)
        self.nest_glossary = common.glossary_button([
            ["Subject", "The unit of replication — the nerve/donor/animal. Pixels and compartments within one subject are sub-measurements, not replicates."],
            ["Compartment", "A named region measured in every subject (endoneurium, perineurium, epineurium)."],
            ["Pseudobulk", "Collapsing a compartment's pixels to one value per feature before any test. Cardinal's meansTest stance."],
            ["Mixed model", "y ~ group * compartment + (1 | subject). The random intercept absorbs each subject's overall level."],
            ["Interaction", "group × compartment: does the group difference itself differ across compartments?"],
            ["Simple effect", "The group difference WITHIN one compartment."],
            ["Moderated t", "Empirical-Bayes t-test (limma/Smyth 2004): shrinks each feature's variance toward a prior fitted across all features, buying power at small n."],
            ["Between-within df", "Denominator degrees of freedom split by whether a contrast varies between subjects (group) or within them (compartment). Conservative vs the normal approximation statsmodels reports by default."],
            ["Feasibility", "Whether an exact rank test on these group sizes could EVER clear BH-FDR across this many features. At 3-vs-6 it cannot."]], parent=w)

        for lab, wdg in (("Features", self.nest_feat_combo), ("Group A", self.nest_ga),
                         ("vs B", self.nest_gb), ("Model", self.nest_model_combo)):
            bar.add_group(lab, wdg)
        bar.add_more(("Test:", self.nest_test_combo), ("Summary:", self.nest_summary_combo),
                     ("Normalize:", self.nest_norm_combo),
                     ("Scale over:", self.nest_norm_scope_combo),
                     ("Transform:", self.nest_transform_combo),
                     ("Match:", self.nest_tol_spin), ("Min prevalence:", self.nest_prev_spin),
                     label="More")
        self.b_nest_csv = button("Results table (CSV)…", self._nest_export_csv, icon="export",
                                 tooltip="Save every feature's effects, p and q for each "
                                         "contrast, plus the model's provenance.")
        self.b_nest_tidy_csv = button(
            "Per-compartment intensities (CSV)…", self._nest_export_tidy_csv, icon="export",
            tooltip="One row per subject × compartment × feature — the raw pseudobulk means, "
                    "not the model's effects. Feeds an external mixed model or a second-opinion "
                    "analysis over the same pixels this tab used.")
        self.b_nest_list = button("Feature list (q cutoff)…", self._nest_build_list_dialog,
                                  icon="save",
                                  tooltip="Turn the hits from the plotted contrast into a "
                                          "saved ★ feature list.")
        self.b_nest_volcano_img = button("Volcano image…", self._nest_export_volcano_dialog,
                                         icon="export",
                                         tooltip="Export the volcano for the plotted contrast "
                                                 "as a publication figure.")
        self._nest_export_btns = (self.b_nest_csv, self.b_nest_tidy_csv, self.b_nest_list,
                                  self.b_nest_volcano_img)
        for _b in self._nest_export_btns:
            _b.setEnabled(False)
        bar.add(self._build_cohort_sample_picker(), self.nest_comp_picker, self.b_nest_auto,
                self.b_nest_map, b_samples, self.b_nest_run, self.b_nest_class,
                self.nest_glossary)
        bar.add_more(*self._nest_export_btns, label="Export", name="export")
        self.nest_info = bar.set_status(
            "Add region-samples to the cohort (Samples…) and label them with a group. Then click "
            "'Auto-label' to read each one's subject (the nerve) and compartment from its region "
            "name — or 'Define compartments…' to set them by hand — and pick Group A vs B to run.")
        v.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.nest_table = QtWidgets.QTableWidget()
        self.nest_table.setSortingEnabled(True)
        self.nest_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table_placeholder(self.nest_table,
                          "Pick Group A / B and a subject key above, then "
                          "'Run nested comparison'.\nRanked features appear here.")
        self.nest_table.itemSelectionChanged.connect(self._nest_table_selected)
        split.addWidget(self.nest_table)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        # Which contrast the volcano and the q-column shown in the table refer to. A nested fit
        # produces several families of p-values (interaction, averaged group effect, one simple
        # effect per compartment); plotting them all at once would be meaningless, so the view
        # commits to one at a time and says which.
        top = QtWidgets.QWidget()
        topv = QtWidgets.QVBoxLayout(top)
        topv.setContentsMargins(0, 0, 0, 0)
        topv.setSpacing(2)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("Contrast:"))
        self.nest_contrast_combo = NoScrollComboBox()
        self.nest_contrast_combo.setToolTip(
            "Which family of p-values the volcano and the table's q column show.\n"
            "Interaction: does the group difference differ across compartments?\n"
            "Group (all compartments): the averaged group effect.\n"
            "A compartment name: the group's simple effect inside that compartment.")
        self.nest_contrast_combo.currentTextChanged.connect(self._nest_contrast_changed)
        crow.addWidget(self.nest_contrast_combo, 1)
        topv.addLayout(crow)
        self.nest_volcano = pg.PlotWidget()
        self.nest_volcano.setLabel("bottom", "effect  (B − A)")
        self.nest_volcano.setLabel("left", "−log₁₀ p")
        topv.addWidget(self.nest_volcano, 1)
        topv.addWidget(common.plot_caption(
            "Click a point to view that ion, and to draw its per-subject values below."))
        right.addWidget(top)

        # The per-subject dot plot is the honest picture at this n: nine dots, not a p-value.
        self.nest_dots = pg.PlotWidget()
        self.nest_dots.setLabel("left", "intensity (summary)")
        self.nest_dots.setToolTip(
            "One dot per SUBJECT per compartment for the selected ion — the replicates the "
            "test actually ran on. With this many subjects, this plot is more trustworthy "
            "than any p-value beside it.")
        dotw = QtWidgets.QWidget()
        dotv = QtWidgets.QVBoxLayout(dotw)
        dotv.setContentsMargins(0, 0, 0, 0)
        dotv.setSpacing(2)
        dotv.addWidget(self.nest_dots, 1)
        self.nest_dots_caption = common.plot_caption(
            "Select a feature to see its per-subject values.")
        dotv.addWidget(self.nest_dots_caption)
        right.addWidget(dotw)
        right.setSizes([340, 300])
        split.addWidget(right)
        split.setSizes([620, 580])
        v.addWidget(split, 1)

        self._cohort_tabs.addTab(w, "Cohort nested stats")

    # --------------------------------------------------------------------- #
    # nested tab — inputs
    # --------------------------------------------------------------------- #
    def _nest_region_refs(self):
        """The ticked cohort samples that are *region* samples — the only ones that can carry a
        compartment. A whole-slide sample has no region name, so it cannot enter a nested model
        (it would be its own single compartment)."""
        return [r for r in self._cohort_included_refs() if (getattr(r, "region", "") or "")]

    #: metadata keys the "Auto-label" / "Define compartments…" mapping writes onto each SampleRef
    NEST_COMPARTMENT_KEY = "compartment"
    NEST_SUBJECT_KEY = "subject"

    def _nest_comp_from(self):
        # Always the per-sample mapping. The old positional-token combo ('exact' / 'last' /
        # 'first') is gone: 'Auto-label' bakes the same collapse into meta['compartment'] and
        # 'Define compartments…' overrides it, so there is one source of truth, not two knobs.
        return f"{cohort_engine.COMPARTMENT_BY_META}{self.NEST_COMPARTMENT_KEY}"

    def _nest_subject_by(self):
        # Likewise the subject is always meta['subject']; 'Auto-label' seeds it from the region
        # name (or the slide), 'Define compartments…' corrects it.
        return self.NEST_SUBJECT_KEY

    def _nest_compartment_entries(self):
        """Distinct compartment labels across the ticked region-samples, read from the mapping —
        the picker shows the labels the MODEL will see, not the raw region names, so '30
        compartments' is visibly a naming problem rather than a silent unestimable fit."""
        mode = self._nest_comp_from()
        names = sorted({cohort_engine.compartment_id(r, mode) for r in self._nest_region_refs()})
        return [(n, n) for n in names if n]

    def _nest_compartments(self):
        unchecked = getattr(self, "_nest_comp_unchecked", set())
        return [n for n, _ in self._nest_compartment_entries() if n not in unchecked]

    def _refresh_nested_combos(self):
        """Repopulate the group pickers from the live cohort. Called from the same places that
        refresh the flat cohort tab's combos. Subject and compartment are no longer combos — they
        come from the per-sample mapping — so only Group A/B need rebuilding here."""
        ga, gb = getattr(self, "nest_ga", None), getattr(self, "nest_gb", None)
        if ga is None or gb is None:
            return
        groups = self.cohort.groups()
        for combo in (ga, gb):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if groups:
                combo.addItems(groups)
                if prev in groups:
                    combo.setCurrentText(prev)
            combo.setEnabled(bool(groups))
            combo.blockSignals(False)
        if len(groups) >= 2 and gb.currentText() == ga.currentText():
            gb.setCurrentIndex(1)

        b = getattr(self, "b_nest_run", None)
        if b is not None:
            b.setEnabled(len(self._nest_region_refs()) >= 4)
        self._nest_update_design_summary()

    def _nest_auto_apply(self):
        """Seed subject, compartment and group for every region-sample from its ROI label, then
        write them onto the cohort — the one-click path the two token combos used to make the
        user assemble by hand.

        Reads which tokens name a subject and which name a compartment from the cohort's own
        cross-slide structure (:meth:`_nest_token_roles`), so it is order-free — ``'S01 endo'``
        and ``'endo_S01'`` resolve alike; on a lone slide it degrades to first-token = subject,
        last-token = compartment. It is exactly 'Define compartments…' accepted unedited, so
        whatever it gets wrong is correctable there."""
        refs = [r for r in self.cohort.samples if (getattr(r, "region", "") or "")]
        if not refs:
            QtWidgets.QMessageBox.information(
                self, "Auto-label",
                "This cohort has no region-samples. Load a slide, create regions, then use "
                "'Add regions as cohort samples' in the Samples panel.")
            return
        subj_vocab, comp_vocab = self._nest_token_roles(refs)
        # The compartment tokens to strip out of the group labels, so 'Synk endo' collapses to
        # 'Synk' — otherwise picking Group A/B would silently restrict the model to one
        # compartment. Fall back to the seeded compartment names on a lone slide.
        known = set(comp_vocab) or {self._nest_seed_compartment(rg, comp_vocab).lower()
                                    for rg in {r.region for r in refs}}
        known.discard("")
        subjects, comps = set(), set()
        for r in refs:
            meta = dict(r.meta or {})
            # Subject is resolved PER-REF from this ref's OWN source, not from a region→subject
            # table: nerves that share a region name ('endoneurium' on every slide, one nerve per
            # slide) still separate by slide, which no region-keyed mapping could express.
            subj = self._nest_seed_subject(
                r.region, getattr(r, "source", ""), subj_vocab).strip().lower()
            comp = self._nest_seed_compartment(r.region, comp_vocab)
            for key, val, seen in ((self.NEST_SUBJECT_KEY, subj, subjects),
                                   (self.NEST_COMPARTMENT_KEY, comp, comps)):
                if val:
                    meta[key] = val
                    seen.add(val)
                else:
                    meta.pop(key, None)
            r.meta = meta
            r.group = self._nest_seed_group(r.group or "", known)
        self._nest_commit_design()
        self.statusBar().showMessage(
            f"Auto-labelled {len(refs)} region-sample(s): {len(subjects)} subject(s), "
            f"{len(comps)} compartment(s), {len({r.group for r in refs if r.group})} group(s). "
            f"Correct any mistakes in 'Define compartments…'.")

    # --------------------------------------------------------------------- #
    # nested tab — the compartment / group mapping
    # --------------------------------------------------------------------- #
    @staticmethod
    def _nest_seed_compartment(region: str, vocab=None) -> str:
        """Best guess at the compartment inside a region name. A seed for the mapping dialog,
        never applied without the user seeing it.

        With a ``vocab`` from :meth:`_nest_token_roles` the guess is structural. Without one
        (a single-slide cohort, or a direct call) it degrades to the old rule — the last token —
        which is right only when the compartment happens to be written last."""
        toks = _nest_tokens(region)
        if not toks:
            return ""
        cands = [t for t in toks if t in (vocab or ())]
        if cands:
            # A bare replicate index ('1', '2') is shared across slides and varies within one,
            # so it looks exactly like a compartment. Prefer a word when the name offers both.
            pick = [t for t in cands if not t.isdigit()] or cands
            n_slides = (lambda t: vocab[t]) if isinstance(vocab, dict) else (lambda t: 0)
            return max(pick, key=lambda t: (n_slides(t), len(t)))
        return toks[-1]

    @staticmethod
    def _nest_seed_subject(region: str, source: str = "", vocab=None) -> str:
        """Best guess at the subject inside a region name. With a ``vocab`` from
        :meth:`_nest_token_roles`, the first token the cohort says names a subject — order-free,
        so ``'endo_S01'`` resolves as readily as ``'S01 endo'``. Without one, the first token
        when the name has more than one, else the slide it came from: a single-token region name
        like ``'endo'`` names no nerve, so the file has to."""
        toks = _nest_tokens(region)
        for t in toks:
            if t in (vocab or ()):
                return t
        if len(toks) > 1:
            return toks[0]
        return session._display_name(source).strip().lower() if source else ""

    @staticmethod
    def _nest_token_roles(refs):
        """Which region-name tokens name a **subject** and which name a **compartment**, read off
        the cohort's own structure rather than a word list.

        The design itself says which is which, and it says so in a way no naming convention can
        hide. A subject id names one nerve: it appears on exactly one slide, and on *every* region
        of that slide. A compartment is the repeated measure inside each nerve: it appears on two
        or more slides, and on only *some* of each slide's regions. Nothing here depends on the
        tokens being English, on their order, or on the absence of a trailing replicate index —
        the previous rule ("the compartment is the last token") silently read ``'2'`` out of
        ``'S01 endo 2'`` and ``'s01'`` out of ``'endo_S01'``, and then the group seeder, which
        strips *known* compartment tokens, failed with it.

        A token that spans several slides but never varies inside one (``'roi'``, a project code)
        is neither — it discriminates nothing.

        Returns ``(subject_tokens, {compartment_token: n_slides})``. Both are empty when the
        cohort cannot carry the signal — one slide, or one region per slide — and the seeds fall
        back to the positional rule."""
        from collections import defaultdict
        per_slide = defaultdict(set)                      # slide -> {region}
        tok_slides = defaultdict(set)                     # token -> {slide}
        tok_regions = defaultdict(set)                    # (slide, token) -> {region}
        for r in refs:
            rg = getattr(r, "region", "") or ""
            if not rg:
                continue
            src = getattr(r, "source", "") or ""
            per_slide[src].add(rg)
            for t in _nest_tokens(rg):
                tok_slides[t].add(src)
                tok_regions[(src, t)].add(rg)
        if len(per_slide) < 2 or all(len(rs) < 2 for rs in per_slide.values()):
            return set(), {}
        subjects, compartments = set(), {}
        for t, slides in tok_slides.items():
            varies = any(tok_regions[(s, t)] != per_slide[s] for s in slides)
            if len(slides) == 1 and not varies:
                subjects.add(t)
            elif len(slides) >= 2 and varies:
                compartments[t] = len(slides)
        return subjects, compartments

    def _nest_seed_group(self, label: str, compartments) -> str:
        """Best guess at the experimental group inside a group label like ``'Synk endo'``:
        the label with any known compartment token removed. Falls back to the label itself,
        which is correct whenever the group was already clean."""
        parts = [p for p in re.split(r"[\s_\-]+", (label or "").strip()) if p]
        keep = [p for p in parts if p.lower() not in compartments]
        return " ".join(keep) if keep else (label or "")

    def _nest_design_dialog(self):
        """Map region names → compartments and group labels → groups, then write both onto the
        samples (``meta['compartment']`` and ``group``) and persist the cohort.

        Region names are chosen inside one slide and group labels are typed by hand, so neither
        reliably carries the crossed design the model needs. Rather than infer it from a naming
        rule that the next dataset breaks, show every distinct value once, seed a guess, and let
        it be corrected. The result is stored per sample, so it survives reload and is visible
        in the roster."""
        refs = [r for r in self.cohort.samples if (getattr(r, "region", "") or "")]
        if not refs:
            QtWidgets.QMessageBox.information(
                self, "Define compartments",
                "This cohort has no region-samples. Load a slide, create regions, then use "
                "'Add regions as cohort samples' in the Samples panel.")
            return
        regions = sorted({r.region for r in refs})
        labels = sorted({(r.group or "") for r in refs})
        # Infer which tokens name subjects and which name compartments from the cohort's own
        # cross-slide structure, so the seeds survive names this codebase has never seen.
        subj_vocab, comp_vocab = self._nest_token_roles(refs)
        seeds = {rg: self._nest_seed_compartment(rg, comp_vocab) for rg in regions}
        # The group seeder strips *known* compartment tokens out of a label like 'Synk endo'.
        # Feed it the inferred vocabulary, not just the tokens that happened to seed — otherwise
        # a compartment the old rule mis-read stays inside every one of its group labels too.
        known = set(comp_vocab) or {v.lower() for v in seeds.values() if v}

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Define compartments and groups")
        dlg.setMinimumWidth(620)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            "The nested model needs three labels per sample: the <b>subject</b> (the nerve — "
            "the unit of replication), the <b>compartment</b> (the repeated measure inside each "
            "subject) and the <b>group</b> (what's compared between subjects).<br>"
            "Give the same compartment name to the same anatomical structure in every nerve, and "
            "one subject id per nerve. Leave a compartment blank to exclude those regions."))

        v.addWidget(QtWidgets.QLabel("<b>Region name → subject and compartment</b>"))
        t_comp = QtWidgets.QTableWidget(len(regions), 4)
        t_comp.setHorizontalHeaderLabels(["Region", "Samples", "Subject", "Compartment"])
        t_comp.horizontalHeader().setStretchLastSection(True)
        t_comp.verticalHeader().setVisible(False)
        first_ref = {r.region: r for r in reversed(refs)}      # any ref carrying that region
        for i, rg in enumerate(regions):
            n = sum(1 for r in refs if r.region == rg)
            meta = first_ref[rg].meta or {}
            comp = str(meta.get(self.NEST_COMPARTMENT_KEY, "")) or seeds[rg]
            subj = str(meta.get(self.NEST_SUBJECT_KEY, "")) or self._nest_seed_subject(
                rg, getattr(first_ref[rg], "source", ""), subj_vocab)
            for col, txt in ((0, rg), (1, str(n))):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t_comp.setItem(i, col, it)
            t_comp.setItem(i, 2, QtWidgets.QTableWidgetItem(subj))
            t_comp.setItem(i, 3, QtWidgets.QTableWidgetItem(comp))
        t_comp.resizeColumnsToContents()
        v.addWidget(t_comp, 1)

        v.addWidget(QtWidgets.QLabel("<b>Group label → group</b>"))
        t_grp = QtWidgets.QTableWidget(len(labels), 3)
        t_grp.setHorizontalHeaderLabels(["Current label", "Samples", "Group"])
        t_grp.horizontalHeader().setStretchLastSection(True)
        t_grp.verticalHeader().setVisible(False)
        for i, lb in enumerate(labels):
            n = sum(1 for r in refs if (r.group or "") == lb)
            for col, txt in ((0, lb or "(ungrouped)"), (1, str(n))):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t_grp.setItem(i, col, it)
            t_grp.setItem(i, 2, QtWidgets.QTableWidgetItem(self._nest_seed_group(lb, known)))
        t_grp.resizeColumnsToContents()
        v.addWidget(t_grp, 1)

        note = QtWidgets.QLabel()
        note.setWordWrap(True)
        v.addWidget(note)

        def _maps():
            smap = {regions[i]: (t_comp.item(i, 2).text() or "").strip().lower()
                    for i in range(len(regions))}
            cmap = {regions[i]: (t_comp.item(i, 3).text() or "").strip()
                    for i in range(len(regions))}
            gmap = {labels[i]: (t_grp.item(i, 2).text() or "").strip()
                    for i in range(len(labels))}
            return smap, cmap, gmap

        def _preview():
            smap, cmap, gmap = _maps()
            per, subjects = {}, set()
            for r in refs:
                c, s = cmap.get(r.region, ""), smap.get(r.region, "")
                if s:
                    subjects.add(s)
                if c and s:
                    per.setdefault(c, set()).add(s)
            groups = sorted({g for g in gmap.values() if g})
            bits = [f"{len(subjects)} subject(s)",
                    f"{len(per)} compartment(s): " +
                    (", ".join(f"{c} (in {len(s)} subjects)" for c, s in sorted(per.items()))
                     or "—"),
                    f"{len(groups)} group(s): {', '.join(groups) or '—'}"]
            lonely = [c for c, s in per.items() if len(s) < 2]
            if lonely:
                bits.append(f"⚠ {', '.join(lonely)} appear(s) in fewer than 2 subjects")
            if any(not s for s in smap.values()):
                bits.append("⚠ some regions have no subject — they'd be dropped")
            if len(groups) < 2:
                bits.append("⚠ the model needs two groups to compare")
            note.setText("  ·  ".join(bits))

        t_comp.itemChanged.connect(lambda *_: _preview())
        t_grp.itemChanged.connect(lambda *_: _preview())
        _preview()

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        smap, cmap, gmap = _maps()
        self._nest_apply_mapping(cmap, gmap, smap)

    def _nest_apply_mapping(self, cmap: dict, gmap: dict, smap: dict | None = None):
        """Write region→subject, region→compartment and group-label→group onto the cohort's
        samples.

        Stored per sample (``meta['subject']``, ``meta['compartment']`` and ``group``) rather
        than as a cohort-level dict, so it round-trips through the existing roster JSON, shows
        up in the Samples panel's columns, and can't drift out of sync with a sample that was
        added later."""
        refs = [r for r in self.cohort.samples if (getattr(r, "region", "") or "")]
        for r in refs:
            meta = dict(r.meta or {})
            for key, val in ((self.NEST_COMPARTMENT_KEY, cmap.get(r.region, "")),
                             (self.NEST_SUBJECT_KEY, (smap or {}).get(r.region, ""))):
                if val:
                    meta[key] = val
                elif smap is not None or key == self.NEST_COMPARTMENT_KEY:
                    meta.pop(key, None)      # blank = "no value", never a label named ''
            r.meta = meta
            r.group = gmap.get(r.group or "", r.group or "")
        self._nest_commit_design()
        n_comp = len({c for c in cmap.values() if c})
        self.statusBar().showMessage(
            f"Mapped {len(cmap)} region name(s) onto {n_comp} compartment(s) and "
            f"{len({g for g in gmap.values() if g})} group(s).")

    def _nest_commit_design(self):
        """Persist a just-changed subject/compartment/group labelling and refresh the tab.

        Any rendered result describes the model that ran under the OLD labels, so drop it rather
        than leave a stale table beside the new design."""
        self._nest_res = None
        for _b in self._nest_export_btns:
            _b.setEnabled(False)
        self.b_nest_class.setEnabled(False)
        self._save_cohort()
        self._refresh_sample_tree()               # → _refresh_nested_combos → repopulates groups
        self._nest_comp_unchecked = set()
        self._nest_update_design_summary()

    def _nest_design_state(self):
        """Resolve what the pickers currently describe into counts and structural problems.

        Every way a nested setup goes wrong is structural, and every one of them is visible
        here and invisible in the resulting q-values: a subject source that collapses all
        samples onto one id, a compartment name that embeds the nerve so each compartment has
        one subject, a group label that embeds the compartment so the A/B pair covers only one
        compartment. Shared by the live summary and the run guard, so they can never disagree
        about whether a design is fittable."""
        refs = self._nest_region_refs()
        ga, gb = self.nest_ga.currentText(), self.nest_gb.currentText()
        subject_by, mode = self._nest_subject_by(), self._nest_comp_from()
        comps = self._nest_compartments()
        contrast = bool(ga and gb and ga != gb)

        subs: dict = {}                       # group label -> {subject}
        all_subs: set = set()
        per_comp: dict = {}                   # compartment -> {subject}, over every group
        ab_cells: dict = {}                   # compartment -> group -> {subject}, A/B only
        unmapped, no_subject = [], []
        for r in refs:
            c = cohort_engine.compartment_id(r, mode)
            s = cohort_engine.subject_id(r, subject_by)
            if not c:
                unmapped.append(r.name)
                continue
            if not s:
                no_subject.append(r.name)
                continue
            all_subs.add(s)
            subs.setdefault(r.group or "", set()).add(s)
            per_comp.setdefault(c, set()).add(s)
            if contrast and r.group in (ga, gb) and c in comps:
                ab_cells.setdefault(c, {}).setdefault(r.group, set()).add(s)

        n_a, n_b = len(subs.get(ga, ())), len(subs.get(gb, ()))
        # The UNION, not the sum: when the subject source is wrong every sample collapses onto
        # one id, and summing per-group counts would hide exactly that.
        st = {"refs": refs, "ga": ga, "gb": gb, "contrast": contrast, "compartments": comps,
              "n_subjects": len(all_subs), "n_a": n_a, "n_b": n_b, "subject_by": subject_by,
              "unmapped": unmapped, "no_subject": no_subject,
              "shared": sorted(subs.get(ga, set()) & subs.get(gb, set())) if contrast else [],
              "missing_in_ab": [c for c in comps if c not in ab_cells] if contrast else [],
              "empty_cells": [c for c in comps
                              if c in ab_cells and not (ab_cells[c].get(ga) and ab_cells[c].get(gb))]
              if contrast else []}
        st["one_subject_per_compartment"] = bool(
            comps and st["n_subjects"] >= 2 and len(comps) >= st["n_subjects"])
        return st

    def _nest_blocking_problem(self, st) -> str:
        """The one message that explains why this design can't be fitted, or ``''``.

        Ordered so the *root* cause is named, not a downstream symptom: a bad subject source
        makes every compartment look lonely, and tangled group labels make compartments look
        missing."""
        if not st["refs"]:
            return ("Add region-samples to the cohort — a nested model needs named regions "
                    "within each slide, not whole slides.")
        # Nothing carries a compartment yet: the cohort has never been labelled. Name that, not a
        # downstream symptom, so the first thing the user sees is the button that fixes it.
        if len(st["unmapped"]) == len(st["refs"]):
            return ("Not labelled yet. Click 'Auto-label' to read each region's subject and "
                    "compartment from its name, or 'Define compartments…' to set them by hand.")
        if st["no_subject"]:
            return (f"{len(st['no_subject'])} region-sample(s) have no subject label "
                    f"(e.g. {st['no_subject'][0]}). Click 'Auto-label', or set it in 'Define "
                    f"compartments…' — every compartment must name its subject.")
        if st["n_subjects"] < 2:
            return ("Every region-sample resolves to the same subject. Click 'Auto-label' to "
                    "take the subject from the region name, or set it in 'Define compartments…'.")
        if not st["compartments"]:
            return ("No compartments resolved. Click 'Auto-label', or use 'Define compartments…' "
                    "to map each region name onto a compartment.")
        # A subject on both sides of the contrast means the subject labels are wrong, and wrong
        # subject labels ALSO make every compartment look lonely — so name that first, or the
        # user chases the symptom.
        if st["contrast"] and st["shared"]:
            return (f"Subject(s) {', '.join(st['shared'][:2])} appear in BOTH groups — the "
                    f"subject labels are wrong, or the group labels are. Re-run 'Auto-label', or "
                    f"fix them in 'Define compartments…'.")
        if st["one_subject_per_compartment"]:
            return (f"Every compartment appears in only one subject ({len(st['compartments'])} "
                    f"of them). Your region names embed the nerve id — click 'Auto-label' (it "
                    f"collapses 'S01 endo' and 'S02 endo' onto 'endo'), or use 'Define "
                    f"compartments…'.")
        if not st["contrast"]:
            return "Pick two different groups for Group A vs B."
        if st["missing_in_ab"]:
            missing = ", ".join(st["missing_in_ab"])
            covered = [c for c in st["compartments"] if c not in st["missing_in_ab"]]
            return (f"Groups '{st['ga']}' and '{st['gb']}' contain no {missing} samples — they "
                    f"only cover {', '.join(covered) or 'nothing'}. Your group labels embed the "
                    f"compartment ('Synk endo'), so picking them restricts the model to one "
                    f"compartment and nothing can be fitted. Use 'Define compartments…' to "
                    f"relabel the groups (Facial / Synk), then read the contrast you want "
                    f"('Group within {st['compartments'][0]}').")
        if st["empty_cells"]:
            return (f"Compartment(s) {', '.join(st['empty_cells'])} are missing from one of the "
                    f"two groups, so the group × compartment model is unestimable. Untick them, "
                    f"or fix the group labels.")
        if min(st["n_a"], st["n_b"]) < 2:
            return (f"Each group needs ≥2 subjects ({st['ga']} has {st['n_a']}, {st['gb']} has "
                    f"{st['n_b']}). Compartments of the same subject are not replicates.")
        return ""

    def _nest_update_design_summary(self):
        """Show the design the pickers describe, and any blocking problem, before anything runs."""
        info = getattr(self, "nest_info", None)
        if info is None or getattr(self, "_nest_res", None) is not None:
            return                              # a rendered result owns the status line
        st = self._nest_design_state()
        if not st["refs"]:
            info.setText(self._nest_blocking_problem(st))
            return
        comps = st["compartments"]
        parts = [f"{len(st['refs'])} region-samples · {st['n_subjects']} subjects · "
                 f"{len(comps)} compartments ({', '.join(comps[:4])}"
                 f"{'…' if len(comps) > 4 else ''})"]
        if st["contrast"]:
            parts.append(f"{st['ga']} n={st['n_a']} vs {st['gb']} n={st['n_b']}")
        text = " · ".join(parts) + "."
        problem = self._nest_blocking_problem(st)
        if problem:
            text = "⚠ " + problem + "  " + text
        elif st["unmapped"]:
            text = (f"⚠ {len(st['unmapped'])} region-sample(s) have no compartment and will be "
                    f"excluded.  " + text)
        info.setText(text)

    # --------------------------------------------------------------------- #
    # nested tab — run
    # --------------------------------------------------------------------- #
    def _nest_run(self):
        # One resolver decides whether this design is fittable, and it is the same one the live
        # summary shows — so the run can never accept what the summary flagged, nor refuse what
        # it called healthy. Before this, a run whose A/B groups covered a single compartment
        # was accepted, every mixed model came back rank-deficient, and the export was a table
        # of `converged = FALSE` under an authoritative-looking header.
        st = self._nest_design_state()
        if not st["refs"]:
            self.statusBar().showMessage(self._nest_blocking_problem(st))
            self._reveal_samples_panel()
            return
        problem = self._nest_blocking_problem(st)
        if problem:
            self.statusBar().showMessage(problem)
            self._nest_update_design_summary()
            return
        refs, ga, gb, comps = st["refs"], st["ga"], st["gb"], st["compartments"]
        subject_by = self._nest_subject_by()
        comp_from = self._nest_comp_from()
        if st["unmapped"]:      # not fatal: they're excluded, but never in silence
            self.statusBar().showMessage(
                f"{len(st['unmapped'])} region-sample(s) have no compartment "
                f"(e.g. {st['unmapped'][0]}) and will be excluded. Set them in "
                f"'Define compartments…' if that's not intended.")

        model = self._nest_models[self.nest_model_combo.currentText()]
        method = self._nest_tests[self.nest_test_combo.currentText()]
        summary = self._nest_summaries[self.nest_summary_combo.currentText()]
        norm = self.nest_norm_combo.currentText()
        norm_scope = self._nest_norm_scopes[self.nest_norm_scope_combo.currentText()]
        transform = self.nest_transform_combo.currentText()
        tol = float(self.nest_tol_spin.value())
        prev = float(self.nest_prev_spin.value())
        combo = self.nest_feat_combo
        is_consensus = self._cohort_combo_is_consensus(combo)
        named_mzs = None if is_consensus else sorted(
            {round(float(m), 4) for m in self._combo_feature_mzs(combo)})
        mode, id_ppm = self.mode_combo.currentText(), self.id_ppm

        def work(progress=None):
            from .. import pipeline
            sessions: dict = {}
            targets = (cohort_engine.consensus_targets(
                refs, tol_ppm=tol, min_prevalence=prev, value="rel_intensity",
                sessions=sessions) if is_consensus else named_mzs)
            if not targets:
                return {"error": "no_targets"}
            if summary == "saved":
                tbl = cohort_engine.batch_feature_table(refs, targets, tol_ppm=tol,
                                                        value="rel_intensity",
                                                        sessions=sessions)
            else:
                tbl = cohort_engine.pseudobulk_table(
                    refs, targets, tol_ppm=tol, norm=norm, summary=summary,
                    norm_scope=norm_scope, sessions=sessions,
                    progress=(lambda i, n, nm: progress(int(100 * i / max(1, n)),
                                                        f"Summarizing {nm}…"))
                    if progress is not None else None)
            if tbl.empty:
                return {"error": "no_samples", "tbl": tbl}
            res = cohort_engine.nested_comparison(
                tbl, ga, gb, subject_by=subject_by, model=model, compartments=comps,
                method=method, transform=transform, compartment_from=comp_from)
            attrs = dict(res.attrs)
            res = pipeline.annotate_df(res, "mz", mode=mode, ppm=id_ppm)
            res.attrs.update(attrs)          # annotate_df drops attrs (labels/test/warning)
            return {"tbl": tbl, "res": res, "targets": list(targets), "ga": ga, "gb": gb,
                    "model": model, "method": method, "summary": summary, "norm": norm,
                    "norm_scope": norm_scope, "transform": transform, "tol": tol,
                    "subject_by": subject_by, "compartment_from": comp_from,
                    "compartments": comps}

        busy = (f"Fitting {len(refs)} region-samples…" if summary == "saved"
                else f"Re-extracting pixels for {len(refs)} region-samples…")
        self._run(work, want_progress=True, busy=busy, on_done=self._nest_on_done)

    def _nest_on_done(self, result):
        if not result:
            return
        if result.get("error") == "no_targets":
            self.statusBar().showMessage("No features to compare — pick a feature set or lower "
                                         "the min-prevalence threshold.")
            return
        tbl = result.get("tbl")
        if result.get("error") == "no_samples" or tbl is None or tbl.empty:
            self.statusBar().showMessage("No region-samples could be read for this cohort.")
            return
        res = result.get("res")
        self._nest_res = res
        self._nest_tbl = tbl
        comps = res.attrs.get("compartments") or []
        self._nest_populate_contrasts(res)
        self._nest_render()

        skipped = list(res.attrs.get("skipped") or [])
        extra = self.record_step(
            "cohort_nested_comparison",
            label=f"Nested cohort comparison ({res.attrs.get('model')})",
            params={"model": res.attrs.get("model"), "test": res.attrs.get("test"),
                    "subject_by": res.attrs.get("subject_by"),
                    "summary": result.get("summary"), "normalize": result.get("norm"),
                    "transform": res.attrs.get("transform"), "tol_ppm": result.get("tol"),
                    "compartments": comps, "group_a": result.get("ga"),
                    "group_b": result.get("gb"),
                    "n_subjects_a": res.attrs.get("n_subjects_a"),
                    "n_subjects_b": res.attrs.get("n_subjects_b"),
                    "n_observations": res.attrs.get("n_observations"),
                    "n_features": res.attrs.get("n_features"),
                    "n_testable": res.attrs.get("n_testable"),
                    "df_method": res.attrs.get("df_method"),
                    "n_skipped": len(skipped)},
            regions=self._cohort_region_inputs(self._nest_region_refs(),
                                               result.get("ga"), result.get("gb")))
        self._log_analysis_to_report(
            "Cohort nested comparison", res,
            regions=f"{result.get('ga')} vs {result.get('gb')} × {', '.join(comps)}",
            title=f"Nested comparison — {result.get('ga')} vs {result.get('gb')}",
            caption=(f"{res.attrs.get('n_subjects_a')} vs {res.attrs.get('n_subjects_b')} "
                     f"subjects · {len(comps)} compartments · {res.attrs.get('test')}"),
            source_extra=extra)
        self._nest_record_run(result, res, comps)
        for _b in self._nest_export_btns:
            _b.setEnabled(True)
        # The class roll-up needs the mixed model's per-contrast standard errors, which the
        # stratified path doesn't report — so the button follows the model that actually ran.
        self.b_nest_class.setEnabled(res.attrs.get("model") == "lmm")
        self.reveal_view("Cohort nested stats")

    def _nest_record_run(self, result, res, comps):
        """Log the nested comparison into :attr:`MainWindow.cohort_run_store` too, so it shows
        up in the Analyses ▸ History tab like every other analysis. The registry marks cohort
        steps as screen-launch stubs (``registry._cohort_screen_only``) because they don't fit
        the single-slide ``AnalysisDialog`` — but they still deserve a durable, re-openable
        record, just filed under the cohort rather than whichever slide happens to be open."""
        store = getattr(self, "cohort_run_store", None)
        if store is None:
            return
        from .. import runs as runs_mod
        ga, gb = result.get("ga"), result.get("gb")
        run = runs_mod.new_run(
            "cohort_nested_comparison", title=f"Nested stats — {ga} vs {gb}",
            target="cohort", dataset=self.cohort.name,
            inputs={"group_a": ga, "group_b": gb, "subject_by": res.attrs.get("subject_by"),
                    "compartment_from": res.attrs.get("compartment_from"),
                    "compartments": comps, "tol_ppm": result.get("tol")},
            params={"model": res.attrs.get("model"), "test": res.attrs.get("test"),
                    "summary": result.get("summary"), "normalize": result.get("norm"),
                    "norm_scope": result.get("norm_scope"), "transform": res.attrs.get("transform")})
        run.status = "done"
        run.summary = (f"{res.attrs.get('n_subjects_a')} vs {res.attrs.get('n_subjects_b')} "
                       f"subjects · {len(comps)} compartments · {res.attrs.get('test')}")
        try:
            store.save_result(run, res)
        except Exception:                              # noqa: BLE001 — recording is best-effort
            import traceback
            traceback.print_exc()
            try:
                store.save(run)
            except Exception:                          # noqa: BLE001
                pass
        if hasattr(self, "_refresh_analysis_history"):
            self._refresh_analysis_history()

    def _nest_populate_contrasts(self, res):
        """Contrast picker entries → the (p, q, effect) column triple each one plots."""
        comps = res.attrs.get("compartments") or []
        entries = {}
        if res.attrs.get("model") == "lmm":
            entries["Interaction (group × compartment)"] = ("p_interaction", "q_interaction", None)
            entries["Group (all compartments)"] = ("p_group", "q_group", "effect_group")
        for c in comps:
            entries[f"Group within {c}"] = (f"p__{c}", f"q__{c}", f"effect__{c}")
        self._nest_contrasts = entries
        cb = self.nest_contrast_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(list(entries))
        cb.blockSignals(False)

    def _nest_contrast_changed(self, _text=None):
        if getattr(self, "_nest_res", None) is not None:
            self._nest_render()

    def _nest_current_contrast(self):
        entries = getattr(self, "_nest_contrasts", {}) or {}
        name = self.nest_contrast_combo.currentText()
        return name, entries.get(name, (None, None, None))

    def _nest_render(self):
        res = getattr(self, "_nest_res", None)
        if res is None:
            return
        self._nest_render_table(res)
        self._nest_render_volcano(res)
        self._nest_render_info(res)

    def _nest_render_info(self, res):
        comps = res.attrs.get("compartments") or []
        na, nb = res.attrs.get("n_subjects_a", 0), res.attrs.get("n_subjects_b", 0)
        a, b = res.attrs.get("a_label", "A"), res.attrs.get("b_label", "B")
        _name, (_p, qcol, _e) = self._nest_current_contrast()
        sig = 0
        if qcol and qcol in res.columns:
            sig = int((res[qcol].to_numpy(dtype=float) <= 0.05).sum())
        n_testable = res.attrs.get("n_testable", 0)
        skipped = list(res.attrs.get("skipped") or [])
        skip = f" · {len(skipped)} skipped" if skipped else ""
        summ = res.attrs.get("summary")
        summ_note = f" · {summ} pseudobulk" if summ else ""
        info = (f"{a} (n={na}) vs {b} (n={nb}) subjects · {res.attrs.get('n_observations', 0)} "
                f"profiles across {len(comps)} compartments{summ_note}{skip} · "
                f"{res.attrs.get('test', '')} — {sig} at q≤0.05 ({n_testable} testable).")
        # Two things the user must see before reading a single q-value.
        notes = []
        fea = res.attrs.get("feasibility") or {}
        if not fea.get("feasible", True):
            notes.append(
                f"a rank test on {fea['n_a']}-vs-{fea['n_b']} could not reject anything here "
                f"(exact p floor {fea['p_floor']:.3g}; BH admits ≤{fea['max_features']} "
                f"feature(s) at {fea['n_features']}). The moderated t / mixed model do not "
                f"share that floor, but treat this cohort as hypothesis-generating.")
        if res.attrs.get("warning"):
            notes.append(res.attrs["warning"])
        if notes:
            info = "⚠ " + "  ".join(notes) + "  " + info
        self.nest_info.setText(info)

    def _nest_render_table(self, res):
        comps = res.attrs.get("compartments") or []
        is_lmm = res.attrs.get("model") == "lmm"
        _name, (pcol, qcol, _e) = self._nest_current_contrast()

        def g(v, spec):
            return "—" if not np.isfinite(v) else format(v, spec)

        def idc(x):
            v = x.get("id_confidence", float("nan"))
            return f"{int(v)}%" if isinstance(v, (int, float)) and v == v else ""

        headers = ["m/z", "lipid", "adduct", "ID conf"]
        if is_lmm:
            headers += ["p interaction", "q interaction"]
        headers += [f"log2 FC {c}" for c in comps] + [f"q {c}" for c in comps]
        headers += ["p (shown)", "q (shown)"]
        # Sort by the contrast the view is showing, so the table and the volcano agree on
        # which features are "top". Untested features sink to the bottom rather than to zero.
        order = res.sort_values(pcol, kind="stable", na_position="last") \
            if pcol and pcol in res.columns else res
        rows = []
        for _, x in order.iterrows():
            cells = [f"{x['mz']:.4f}", x.get("best_lipid", "") or "(unidentified)",
                     x.get("best_adduct", ""), idc(x)]
            if is_lmm:
                cells += [g(x.get("p_interaction", np.nan), ".2e"),
                          g(x.get("q_interaction", np.nan), ".2e")]
            cells += [g(x.get(f"log2_fc__{c}", np.nan), "+.2f") for c in comps]
            cells += [g(x.get(f"q__{c}", np.nan), ".2e") for c in comps]
            cells += [g(x.get(pcol, np.nan) if pcol else np.nan, ".2e"),
                      g(x.get(qcol, np.nan) if qcol else np.nan, ".2e")]
            rows.append(tuple(cells))
        fill_table(self.nest_table, headers, rows)
        tips = {"m/z": "Feature m/z, shared across every subject and compartment.",
                "ID conf": "MS1 lipid-ID confidence (0–100).",
                "p interaction": "Joint Wald test of all group×compartment terms: does the group difference DIFFER across compartments?",
                "q interaction": "BH-FDR across features within the interaction family.",
                "p (shown)": "Raw p for the contrast picked in the Contrast box.",
                "q (shown)": "BH-FDR for that contrast, adjusted across features within its own family."}
        for c in comps:
            tips[f"log2 FC {c}"] = (f"τ-regularized log2 fold-change (B/A) of the raw per-subject "
                                    f"means inside {c}. + = higher in group B.")
            tips[f"q {c}"] = f"BH-FDR for the group's simple effect inside {c}."
        common.set_header_tooltips(self.nest_table, tips)

    @staticmethod
    def _nest_effect_spread(res, comps):
        """x-axis stand-in for the interaction, which is a joint test over K−1 terms and so has
        no single signed effect: how far the group difference travels across compartments. An
        unfitted feature's simple effects are all NaN — that's a NaN spread, not a zero one."""
        if not comps:
            return np.zeros(len(res))
        sims = np.vstack([res[f"effect__{c}"].to_numpy(dtype=float) for c in comps])
        allnan = np.all(np.isnan(sims), axis=0)
        out = np.full(sims.shape[1], np.nan)
        if (~allnan).any():
            good = sims[:, ~allnan]
            out[~allnan] = np.nanmax(good, axis=0) - np.nanmin(good, axis=0)
        return out

    def _nest_render_volcano(self, res):
        self.nest_volcano.clear()
        name, (pcol, qcol, ecol) = self._nest_current_contrast()
        if not pcol or pcol not in res.columns:
            return
        p_all = res[pcol].to_numpy(dtype=float)
        q_all = res[qcol].to_numpy(dtype=float) if qcol in res.columns else np.full(len(res), np.nan)
        # The interaction has no single signed effect (it's a joint test over K−1 terms), so the
        # x-axis becomes the spread of the compartment simple effects — how much the group
        # difference actually moves across compartments. Anything else would be a fake sign.
        comps = res.attrs.get("compartments") or []
        if ecol and ecol in res.columns:
            eff = res[ecol].to_numpy(dtype=float)
            scale = ", log2" if res.attrs.get("transform") == "log2" else ""
            xlabel = f"effect  (B − A){scale}"
        else:
            eff = self._nest_effect_spread(res, comps)
            xlabel = "spread of simple effects across compartments (max − min)"
        ok = np.isfinite(eff) & np.isfinite(p_all)
        if not ok.any():
            return
        x = eff[ok]
        y = -np.log10(np.clip(p_all[ok], 1e-12, 1))
        mz = res["mz"].to_numpy()[ok]
        sig = q_all[ok] <= 0.05
        self.nest_volcano.addLine(x=0, pen=pg.mkPen(GUIDE_LINE))
        if sig.any():
            p_thr = float(np.nanmax(p_all[ok][sig]))
            self.nest_volcano.addLine(y=-np.log10(max(p_thr, 1e-12)),
                                      pen=pg.mkPen(GUIDE_LINE, style=QtCore.Qt.DashLine))
        else:
            self.nest_volcano.addLine(y=-np.log10(0.05),
                                      pen=pg.mkPen(GUIDE_LINE, style=QtCore.Qt.DotLine, width=0.7))
        self.nest_volcano.setLabel("bottom", xlabel)
        self.nest_volcano.setLabel("left", f"−log₁₀ p   ({name})")
        a_qc, b_qc = QtGui.QColor(REGION_PALETTE[0]), QtGui.QColor(REGION_PALETTE[1])
        brushes, pens, sizes = [], [], []
        for f, s in zip(x, sig):
            c = QtGui.QColor(b_qc if f > 0 else a_qc)
            c.setAlpha(235 if s else 110)
            brushes.append(pg.mkBrush(c))
            pens.append(pg.mkPen("#222", width=0.8) if s else pg.mkPen(None))
            sizes.append(11 if s else 7)
        sp = pg.ScatterPlotItem(x=x, y=y, brush=brushes, size=sizes, pen=pens, data=mz)
        sp.sigClicked.connect(self._nest_volcano_clicked)
        self.nest_volcano.addItem(sp)

    def _nest_volcano_clicked(self, _sp, points):
        if points:
            mz = float(points[0].data())
            self.set_active_mz(mz)
            self._nest_render_dots(mz)

    def _nest_table_selected(self):
        rows = {i.row() for i in self.nest_table.selectedIndexes()}
        if len(rows) != 1:
            return
        item = self.nest_table.item(next(iter(rows)), 0)
        if item is None:
            return
        try:
            mz = float(item.text())
        except ValueError:
            return
        self._nest_render_dots(mz)

    def _nest_render_dots(self, mz):
        """One dot per subject per compartment for ``mz`` — the replicates the test ran on.

        Groups are offset within each compartment band and coloured like the volcano, and a
        short horizontal tick marks each group's mean. Subjects are what n counts, so nine dots
        is the whole dataset for one ion; a reader can see the overlap a q-value hides."""
        tbl = getattr(self, "_nest_tbl", None)
        res = getattr(self, "_nest_res", None)
        if tbl is None or res is None:
            return
        targets = list(tbl.attrs.get("targets") or [])
        if not targets:
            return
        idx = int(np.argmin(np.abs(np.asarray(targets, dtype=float) - mz)))
        col = [c for c in tbl.columns if c != "group"][idx]
        subject, compartment = cohort_engine.nested_design(tbl, res.attrs["subject_by"])
        comps = res.attrs.get("compartments") or []
        a, b = res.attrs.get("a_label", "A"), res.attrs.get("b_label", "B")
        vals = tbl[col].to_numpy(dtype=float)
        groups = tbl["group"].to_numpy(dtype=object)

        self.nest_dots.clear()
        colors = {a: QtGui.QColor(REGION_PALETTE[0]), b: QtGui.QColor(REGION_PALETTE[1])}
        ticks = []
        for k, comp in enumerate(comps):
            for gi, gname in enumerate((a, b)):
                sel = (compartment == comp) & (groups == gname) & np.isfinite(vals)
                if not sel.any():
                    continue
                yv = vals[sel]
                # deterministic jitter: spread the subjects across the group's half-band so
                # equal values don't overplot into one dot (no RNG — the plot must be stable)
                n = yv.size
                offs = (np.arange(n) - (n - 1) / 2.0) * (0.28 / max(1, n - 1)) if n > 1 else np.zeros(1)
                xv = np.full(n, k + (-0.18 if gi == 0 else 0.18)) + offs
                c = QtGui.QColor(colors[gname]); c.setAlpha(220)
                self.nest_dots.addItem(pg.ScatterPlotItem(
                    x=xv, y=yv, brush=pg.mkBrush(c), pen=pg.mkPen("#222", width=0.7), size=10))
                m = float(np.mean(yv))
                x0 = k + (-0.18 if gi == 0 else 0.18)
                self.nest_dots.plot([x0 - 0.12, x0 + 0.12], [m, m],
                                    pen=pg.mkPen(c.darker(140), width=2.5))
            ticks.append((k, comp))
        self.nest_dots.getAxis("bottom").setTicks([ticks])
        self.nest_dots.setLabel("left", f"m/z {mz:.4f}  ({res.attrs.get('summary') or 'value'})")
        self.nest_dots.setXRange(-0.6, max(0, len(comps) - 1) + 0.6, padding=0)
        row = res[np.isclose(res["mz"].to_numpy(dtype=float), mz)]
        note = ""
        if len(row):
            qi = row.iloc[0].get("q_interaction", float("nan"))
            if np.isfinite(qi):
                note = f" · interaction q = {qi:.2e}"
        self.nest_dots_caption.setText(
            f"m/z {mz:.4f} — one dot per subject; bars are group means. "
            f"◼ {a}  ◼ {b}{note}")

    # --------------------------------------------------------------------- #
    # nested tab — lipid class roll-up
    # --------------------------------------------------------------------- #
    def _nest_class_dialog(self):
        """Competitive, correlation-adjusted lipid-class test over the current nested result.

        Deliberately a self-contained modal reading ``_nest_tbl`` / ``_nest_res``: the whole
        surface is this method plus one engine call, so it can be relocated without untangling
        it from the tab."""
        res, tbl = getattr(self, "_nest_res", None), getattr(self, "_nest_tbl", None)
        if res is None or tbl is None or not len(res):
            self.statusBar().showMessage("Run a nested comparison first.")
            return
        if res.attrs.get("model") != "lmm":
            self.statusBar().showMessage(
                "The class test needs the mixed model's standard errors — re-run with "
                "Model = 'Mixed model (group × compartment)'.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Lipid class test")
        dlg.setMinimumSize(820, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel(
            "Each class is tested against <b>the rest of the lipidome</b> (competitive), so a "
            "per-slide scale offset that lifts every ion cancels.<br>A class is a chain-length "
            "series whose ions move together, so <b>r̄</b> (their mean residual correlation) "
            "inflates the variance by <b>VIF = 1 + (m−1)·r̄</b>. <b>n eff</b> is what those ions "
            "are really worth. Compare <b>p</b> with <b>p naive</b> to see the size of the lie."))

        row = QtWidgets.QHBoxLayout()
        contrast = NoScrollComboBox()
        for c in (res.attrs.get("compartments") or []):
            contrast.addItem(f"Group within {c}", c)
        contrast.addItem("Group (all compartments)", "group")
        contrast.setToolTip("Which per-ion statistic is rolled up. The interaction is a joint "
                            "test with no signed per-ion effect, so it cannot be rolled up.")
        min_conf = NoScrollSpinBox()
        min_conf.setRange(0, 100)
        min_conf.setValue(0)
        min_conf.setToolTip("Drop ions whose MS1 ID confidence is below this before forming the "
                            "classes. A third of accurate-mass IDs at 5 ppm can be chance "
                            "matches, and a class built from them is a class of noise.")
        min_ions = NoScrollSpinBox()
        min_ions.setRange(2, 50)
        min_ions.setValue(3)
        ranks = QtWidgets.QCheckBox("Rank test")
        ranks.setChecked(True)
        ranks.setToolTip("Rank-sum (robust to the heavy-tailed statistics a small-n contrast "
                         "produces). Unticked: limma's parametric t on the statistics.")
        adjust = QtWidgets.QCheckBox("Adjust for inter-ion correlation")
        adjust.setChecked(True)
        adjust.setToolTip("Untick to see the uncorrected result — the one that treats 19 "
                          "sulfatides as 19 independent votes. For comparison only.")
        for lab, wdg in (("Contrast:", contrast), ("Min ID conf:", min_conf),
                         ("Min ions:", min_ions)):
            row.addWidget(QtWidgets.QLabel(lab)); row.addWidget(wdg)
        row.addWidget(ranks); row.addWidget(adjust)
        row.addStretch(1)
        v.addLayout(row)

        table = QtWidgets.QTableWidget()
        table.setSortingEnabled(True)
        v.addWidget(table, 1)
        note = QtWidgets.QLabel()
        note.setWordWrap(True)
        v.addWidget(note)
        state: dict = {}

        def _recompute():
            try:
                out = cohort_engine.nested_class_comparison(
                    tbl, res, contrast=contrast.currentData(),
                    min_ions=int(min_ions.value()), min_confidence=float(min_conf.value()),
                    use_ranks=ranks.isChecked(),
                    inter_ion_cor=None if adjust.isChecked() else 0.0)
            except ValueError as exc:
                table.setRowCount(0)
                note.setText(f"⚠ {exc}")
                return
            state["out"] = out
            def g(v, spec):
                return "—" if not np.isfinite(v) else format(v, spec)
            has_fc = "median_log2_fc" in out.columns
            headers = ["class", "ions", "n eff", "r̄", "VIF", "dir"]
            headers += ["median log2 FC"] if has_fc else []
            headers += ["p", "q (FDR)", "p naive"]
            rows = []
            for _, x in out.iterrows():
                cells = [str(x["class"]), f"{int(x['n_ions'])}", f"{x['n_effective']:.1f}",
                         f"{x['r_bar']:+.2f}", f"{x['vif']:.1f}", x["direction"]]
                if has_fc:
                    cells.append(g(x["median_log2_fc"], "+.2f"))
                cells += [g(x["p_value"], ".3g"), g(x["q_value"], ".3g"),
                          g(x["p_uncorrected"], ".3g")]
                rows.append(tuple(cells))
            fill_table(table, headers, rows)
            common.set_header_tooltips(table, {
                "ions": "Ions assigned to this class after the ID-confidence filter.",
                "n eff": "m / VIF — how many independent ions this class is really worth.",
                "r̄": "Mean pairwise residual correlation among the class's ions. Estimated after removing each section's overall scale, so it measures class-specific coupling, not a global offset.",
                "VIF": "Variance inflation factor, 1 + (m−1)·r̄. The p-value is inflated by roughly its square root.",
                "dir": "Whether the class sits below or above the rest of the lipidome for this contrast.",
                "p": "Correlation-adjusted competitive p-value (Wu & Smyth 2012).",
                "q (FDR)": "Benjamini-Hochberg across the classes shown.",
                "p naive": "The same test at r̄ = 0 — what you would have reported by treating the class's ions as independent."})
            sig = int((out["q_value"].to_numpy(dtype=float) <= 0.05).sum())
            a, b = res.attrs.get("a_label", "A"), res.attrs.get("b_label", "B")
            note.setText(
                f"{len(out)} classes · {out.attrs['n_ions_annotated']} annotated ions of "
                f"{out.attrs['n_ions_background']} in the background · {a} vs {b} · "
                f"{out.attrs['test']} — {sig} at q≤0.05."
                + ("  ⚠ correlation adjustment OFF: these p-values treat a class's ions as "
                   "independent." if not adjust.isChecked() else ""))

        for w in (contrast, min_conf, min_ions):
            (w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox)
             else w.valueChanged).connect(lambda *_: _recompute())
        ranks.toggled.connect(lambda *_: _recompute())
        adjust.toggled.connect(lambda *_: _recompute())
        _recompute()

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        b_csv = bb.addButton("Export CSV…", QtWidgets.QDialogButtonBox.ActionRole)
        b_csv.clicked.connect(lambda: self._nest_class_export(state.get("out")))
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        dlg.exec()

    def _nest_class_export(self, out):
        if out is None or not len(out):
            return
        path, _ = filedialogs.get_save_file_name(
            self, "Export lipid class test", "class_test.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            for k in ("test", "contrast", "statistic", "a_label", "b_label", "use_ranks",
                      "min_vif", "inter_feature_cor", "center_observations", "min_confidence",
                      "min_ions", "n_ions_annotated", "n_ions_background"):
                if k in out.attrs:
                    fh.write(f"# {k}: {out.attrs[k]}\n")
            out.to_csv(fh, index=False)
        self.statusBar().showMessage(f"Wrote {path}")

    # --------------------------------------------------------------------- #
    # nested tab — exports
    # --------------------------------------------------------------------- #
    def _nest_export_csv(self):
        res = getattr(self, "_nest_res", None)
        if res is None or not len(res):
            self.statusBar().showMessage("Run a nested comparison first.")
            return
        path, _ = filedialogs.get_save_file_name(
            self, "Export nested comparison", "nested_comparison.csv", "CSV (*.csv)")
        if not path:
            return
        # The model is not recoverable from the columns alone, and a table of q-values whose
        # design is unrecorded is exactly the artifact this screen exists to prevent.
        with open(path, "w", newline="", encoding="utf-8") as fh:
            a = res.attrs
            for k in ("test", "model", "subject_by", "transform", "summary", "df_method",
                      "compartments", "a_label", "b_label", "n_subjects_a", "n_subjects_b",
                      "n_observations", "n_features", "n_testable", "tau"):
                if k in a:
                    fh.write(f"# {k}: {a[k]}\n")
            fea = a.get("feasibility") or {}
            if fea and not fea.get("feasible", True):
                fh.write(f"# rank_test_feasibility: {fea.get('reason')}\n")
            res.to_csv(fh, index=False)
        self.statusBar().showMessage(f"Wrote {path}")

    def _nest_export_tidy_csv(self):
        tbl = getattr(self, "_nest_tbl", None)
        if tbl is None or not len(tbl):
            self.statusBar().showMessage("Run a nested comparison first.")
            return
        res = getattr(self, "_nest_res", None)
        subject_by = res.attrs.get("subject_by") if res is not None else self._nest_subject_by()
        comp_from = (res.attrs.get("compartment_from") if res is not None
                    else self._nest_comp_from())
        path, _ = filedialogs.get_save_file_name(
            self, "Export per-compartment intensities", "compartment_intensities.csv",
            "CSV (*.csv)")
        if not path:
            return
        tidy = cohort_engine.pseudobulk_tidy(tbl, subject_by, comp_from)
        # Reuse the lipid IDs the nested run already annotated `res` with, keyed by mz, rather
        # than paying for a second annotation pass over every (subject, compartment, mz) row.
        if res is not None and "mz" in res.columns:
            ann_cols = [c for c in ("best_lipid", "best_class", "best_adduct", "best_ppm",
                                    "id_confidence") if c in res.columns]
            if ann_cols:
                tidy = tidy.merge(res[["mz", *ann_cols]].drop_duplicates("mz"), on="mz",
                                  how="left")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            for k in ("summary", "norm", "norm_scope", "value"):
                v = tidy.attrs.get(k)
                if v:
                    fh.write(f"# {k}: {v}\n")
            fh.write(f"# subject_by: {subject_by}\n# compartment_from: {comp_from}\n")
            tidy.to_csv(fh, index=False)
        self.statusBar().showMessage(f"Wrote {path}")

    def _nest_export_volcano_dialog(self):
        res = getattr(self, "_nest_res", None)
        if res is None or not len(res):
            self.statusBar().showMessage("Run a nested comparison first.")
            return
        from .plotexport import VolcanoExportDialog

        name, (pcol, qcol, ecol) = self._nest_current_contrast()
        comps = res.attrs.get("compartments") or []
        # VolcanoExportDialog speaks the shared log2_fc / p_value / q_value contract, so project
        # the chosen contrast onto it rather than teaching the dialog about nested columns.
        df = res.copy()
        df["log2_fc"] = (df[ecol] if ecol and ecol in df.columns
                         else self._nest_effect_spread(res, comps))
        df["p_value"], df["q_value"] = df[pcol], df[qcol]
        df.attrs.update(res.attrs)
        df.attrs["test"] = f"{res.attrs.get('test', '')} — {name}"
        VolcanoExportDialog(self, df, a_label=res.attrs.get("a_label", "A"),
                            b_label=res.attrs.get("b_label", "B"),
                            default_name="nested_volcano.png").exec()

    def _nest_build_list_dialog(self):
        res = getattr(self, "_nest_res", None)
        if res is None or not len(res):
            self.statusBar().showMessage("Run a nested comparison first.")
            return
        name, (_p, qcol, ecol) = self._nest_current_contrast()
        if not qcol or qcol not in res.columns:
            return
        q = res[qcol].to_numpy(dtype=float)
        hits = res[np.isfinite(q) & (q <= 0.05)]
        if not len(hits):
            self.statusBar().showMessage(f"No features clear q≤0.05 for '{name}'.")
            return
        feats = []
        for _, x in hits.iterrows():
            eff = x.get(ecol, float("nan")) if ecol else float("nan")
            tag = f" · effect {float(eff):+.2f}" if np.isfinite(eff) else ""
            feats.append({"mz": float(x["mz"]), "lipid": x.get("best_lipid", "") or "",
                          "note": f"{name} · q {float(x[qcol]):.1e}{tag}"})
        saved = self._save_feature_list_to_library(f"Nested — {name}", feats, prompt=True)
        if saved:
            self.statusBar().showMessage(f"Saved feature list '{saved}' ({len(feats)} ions).")

    # ===================================================================== #
    # Cohort UMAP — one embedding over pixels pooled from every sample
    # ===================================================================== #
    def _tab_cohort_embed(self):
        """A *pooled* embedding: pixels from every cohort sample, projected on a shared
        feature axis into ONE UMAP/t-SNE so the slides share a coordinate frame and the
        cloud can be coloured by sample or group — the cross-sample view the single-slide
        Components tab can't give (its axes aren't comparable between independent runs)."""
        w, v = tab_page()
        bar = ControlBar()
        self.cembed_feat_combo = self._make_consensus_feature_combo(
            "Feature axis for the embedding:\n"
            "Consensus — a shared m/z axis clustered from every sample's peaks (recommended, so "
            "every slide spans the same features).\nOr any named feature set — the working set, "
            "a region scope, or a saved ★ list.")
        self.cembed_method_combo = NoScrollComboBox()
        self.cembed_method_combo.addItems(["UMAP", "TSNE"])
        self.cembed_method_combo.setToolTip("UMAP if umap-learn is installed, else t-SNE.")
        # What one point IS: a pixel (heaviest, most detail) or an aggregate (one point per
        # region / per slide). Aggregates stream one cube at a time and keep only the means, so
        # memory stays flat for any cohort size — the scalable, cross-slide-comparable default.
        self.cembed_unit_combo = NoScrollComboBox()
        self.cembed_unit_combo.addItems(["Pixels", "Region means", "Sample means"])
        self.cembed_unit_combo.setToolTip(
            "What each point represents:\n"
            "Pixels — every (capped) pixel; most spatial detail, heaviest.\n"
            "Region means — one point per named region per sample: the mode for comparing "
            "selected regions ACROSS samples (pick which with 'Regions ▾'; needs saved regions). "
            "Runs even on a single slide with ≥2 regions; memory stays flat at any cohort size.\n"
            "Sample means — one point per slide (lightest; useful with many slides).")
        self.cembed_unit_combo.currentTextChanged.connect(lambda *_: self._cembed_sync_unit())
        self.cembed_color_combo = NoScrollComboBox()
        self.cembed_color_combo.addItems(["Group", "Sample", "Region", "Cluster"])
        self.cembed_color_combo.setToolTip(
            "Colour each point by its cohort group, by which sample it came from, by region "
            "(region means), or by an unsupervised Cluster of the embedding — the 'molecular "
            "histology' atlas colouring you can paint back onto tissue with 'Cluster map'.")
        self.cembed_color_combo.currentTextChanged.connect(lambda *_: self._draw_cembed())
        # Render style: a dense per-pixel cloud is drawn as a density composite (the atlas
        # look), not overplotted dots; few-point region/sample-mean clouds always use dots.
        self.cembed_style_combo = NoScrollComboBox()
        self.cembed_style_combo.addItems(["Density", "Dots"])
        self.cembed_style_combo.setToolTip(
            "Density — composite per-pixel points into a smooth shaded cloud (the LIPID-atlas "
            "look; pixels only). Dots — one marker per point (best for region/sample means).")
        self.cembed_style_combo.currentTextChanged.connect(lambda *_: self._draw_cembed())
        # Cluster controls — only used when Colour by = Cluster (or for the Cluster map).
        # Recluster is debounced so spinning k / switching method coalesces into one redraw.
        self._cembed_recluster_timer = QtCore.QTimer(self)
        self._cembed_recluster_timer.setSingleShot(True)
        self._cembed_recluster_timer.timeout.connect(self._cembed_recluster)
        self.cembed_cluster_method = NoScrollComboBox()
        self.cembed_cluster_method.addItems(["k-means", "HDBSCAN"])
        self.cembed_cluster_method.setToolTip(
            "How to cluster the embedding for the 'Cluster' colouring. k-means returns exactly "
            "'Clusters (k)' groups; HDBSCAN finds its own count and marks sparse points as "
            "Noise (needs the optional 'hdbscan' package, else falls back to k-means).")
        self.cembed_cluster_method.currentTextChanged.connect(self._schedule_cembed_recluster)
        self.cembed_cluster_k = NoScrollSpinBox()
        self.cembed_cluster_k.setRange(2, 40)
        self.cembed_cluster_k.setValue(8)
        self.cembed_cluster_k.setToolTip("Number of k-means clusters (molecular phenotypes) to "
                                         "split the embedding into.")
        self.cembed_cluster_k.valueChanged.connect(self._schedule_cembed_recluster)
        self.cembed_cap_spin = NoScrollSpinBox()
        self.cembed_cap_spin.setRange(50, 20000)
        self.cembed_cap_spin.setValue(3000)
        self.cembed_cap_spin.setToolTip("Random pixels kept per sample (Pixels unit only) — so a "
                                        "big slide doesn't swamp a small one and it stays tractable.")
        self.cembed_tol_spin = NoScrollDoubleSpinBox()
        self.cembed_tol_spin.setRange(1.0, 200.0)
        self.cembed_tol_spin.setValue(20.0)
        self.cembed_tol_spin.setSuffix(" ppm")
        self.cembed_prev_spin = NoScrollDoubleSpinBox()
        self.cembed_prev_spin.setRange(0.0, 1.0)
        self.cembed_prev_spin.setSingleStep(0.1)
        self.cembed_prev_spin.setValue(0.5)
        self.cembed_prev_spin.setToolTip("Consensus only: keep ions present in at least this "
                                         "fraction of samples.")
        # The user-facing scaling control now lives in the Samples panel as the
        # 'Cross-sample scaling' dropdown (cembed_scaling_combo). This checkbox is kept as a
        # hidden compatibility mirror — synced from that dropdown so anything still reading
        # cembed_norm_chk.isChecked() sees the z-score state — but is NOT added to the bar.
        self.cembed_norm_chk = QtWidgets.QCheckBox("Align samples")
        self.cembed_norm_chk.setChecked(True)
        self.cembed_norm_chk.setToolTip("Per-sample z-score before pooling (mirrors the Samples "
                                        "panel 'Cross-sample scaling' dropdown).")
        # UMAP knobs the embedding's *look* depends on — the kidney-atlas (Farrow 2025) figure
        # uses cosine · n_neighbors 250 · min_dist 0 over millions of pixels.
        self.cembed_metric = NoScrollComboBox()
        self.cembed_metric.addItems(["cosine", "euclidean", "correlation"])
        self.cembed_metric.setToolTip("UMAP distance metric. cosine compares spectral *shape* "
                                      "(recommended for MSI — the atlas look); euclidean compares "
                                      "magnitude.")
        self.cembed_nn = NoScrollSpinBox()
        self.cembed_nn.setRange(2, 500)
        self.cembed_nn.setValue(250)             # kidney-atlas default (Farrow 2025); clamped < n
        self.cembed_nn.setToolTip("UMAP n_neighbors — higher favours global structure (250 is the "
                                  "kidney-atlas default; auto-clamped below the point count).")
        self.cembed_mindist = NoScrollDoubleSpinBox()
        self.cembed_mindist.setRange(0.0, 1.0)
        self.cembed_mindist.setSingleStep(0.05)
        self.cembed_mindist.setValue(0.0)        # atlas default: packs points tightest
        self.cembed_mindist.setToolTip("UMAP min_dist — lower packs points tighter (0 = the "
                                       "kidney-atlas default).")
        self.cembed_regions_btn = self._build_cembed_region_picker()
        b_samples = QtWidgets.QPushButton("Samples…")
        b_samples.setToolTip("Open the Samples panel to add the slides this embedding pools")
        b_samples.clicked.connect(self._reveal_samples_panel)
        b_run = primary_button("Run pooled embedding")
        b_run.setToolTip("Pool pixels from every cohort sample onto a shared feature axis into "
                         "one UMAP/t-SNE. Needs ≥2 samples — except Region means, which can run "
                         "on a single slide with ≥2 saved named regions.")
        self.b_cembed_run = b_run
        b_run.setEnabled(self._cembed_run_ok())
        b_run.clicked.connect(self._cembed_run)
        for lab, wdg in (("Features", self.cembed_feat_combo), ("Method", self.cembed_method_combo),
                         ("Unit", self.cembed_unit_combo), ("Colour by", self.cembed_color_combo),
                         ("Style", self.cembed_style_combo),
                         ("Pixels/sample", self.cembed_cap_spin),
                         ("Match", self.cembed_tol_spin), ("Min prevalence", self.cembed_prev_spin)):
            bar.add_group(lab, wdg)
        bar.add_more(("Metric", self.cembed_metric), ("Neighbours", self.cembed_nn),
                     ("Min dist", self.cembed_mindist), label="UMAP params")
        bar.add_more(("Cluster by", self.cembed_cluster_method), ("Clusters (k)", self.cembed_cluster_k),
                     label="Clustering")
        bar.add(self.cembed_regions_btn, self._build_cohort_sample_picker(), b_samples, b_run)
        self.b_cembed_studio = QtWidgets.QPushButton("Edit & annotate…")
        self.b_cembed_studio.setIcon(icon("settings"))
        self.b_cembed_studio.setToolTip("Open UMAP Studio — turn this embedding into a "
                                        "publication-quality, density-shaded, annotated figure "
                                        "(donor / group / region / lipid colouring; export to PDF).")
        self.b_cembed_studio.setEnabled(False)
        self.b_cembed_studio.clicked.connect(self._open_cembed_studio)
        self.b_cembed_clustermap = QtWidgets.QPushButton("Cluster map")
        self.b_cembed_clustermap.setIcon(icon("analysis"))
        self.b_cembed_clustermap.setToolTip(
            "Paint the embedding's clusters back onto each slide's tissue — the linked "
            "spatial↔embedding 'molecular histology' figure. Pixels unit only.")
        self.b_cembed_clustermap.setEnabled(False)
        self.b_cembed_clustermap.clicked.connect(self._cembed_cluster_map)
        self.b_cembed_png = tool_button(
            name="export", tooltip="Export the embedding as a publication figure (live preview + "
                                    "colour-by / density-or-dots + Style preset).",
            slot=self._export_cembed_dialog)
        self.b_cembed_png.setEnabled(False)
        self.b_cembed_csv = tool_button(name="export", tooltip="Save coordinates as CSV…",
                                        slot=self._export_cembed_csv)
        self.b_cembed_csv.setEnabled(False)
        bar.add(self.b_cembed_studio, self.b_cembed_clustermap, self.b_cembed_png, self.b_cembed_csv)
        self.cembed_info = bar.set_status(
            "Click 'Samples…' to add ≥2 samples to the cohort and label their groups, then run: "
            "every sample's pixels are pooled on a shared feature axis into one embedding you "
            "can colour by group or sample.")
        v.addWidget(bar)

        self.cembed_plot = pg.PlotWidget()
        self.cembed_plot.setLabel("bottom", "embedding 1")
        self.cembed_plot.setLabel("left", "embedding 2")
        self.cembed_plot.setAspectLocked(False)
        self.cembed_legend = self.cembed_plot.addLegend(offset=(-10, 10))
        v.addWidget(self.cembed_plot, 1)
        self._cohort_tabs.addTab(w, "Cohort UMAP")
        self._update_cohort_run_enabled()
        self._cembed_sync_unit()

    def _cembed_run(self):
        if not self._cembed_run_ok():
            unit = self.cembed_unit_combo.currentText()
            if unit == "Region means":
                self.statusBar().showMessage("Region means needs a slide with ≥2 saved named "
                                             "regions — add a sample, or save regions on the "
                                             "Segmentation tab.")
            elif unit == "Sample means":
                self.statusBar().showMessage("Sample means is one point per slide — add a 2nd "
                                             "sample, or pick Pixels / Region means to embed a "
                                             "single slide.")
            else:
                self.statusBar().showMessage("Add at least one sample to the cohort first "
                                             "— use the Samples panel.")
            self._reveal_samples_panel()
            return
        refs = self._cohort_included_refs()          # Samples ▾ subset (roster untouched)
        if not refs:
            self.statusBar().showMessage("All samples are unticked in the Samples ▾ picker — "
                                         "tick at least one to embed.")
            return
        # Stale paths after a move / new machine: the loader silently skips any sample whose
        # raw .imzML is gone, so a relocated cohort would grind through every sample and die
        # deep in pooled_embedding with a vague "None of the N samples could be loaded". Catch
        # it up front — list what's missing and offer to relocate the data in place.
        if not self._cembed_preflight_paths(refs):
            return
        tol = float(self.cembed_tol_spin.value())
        targets = self._cohort_feature_targets(
            self.cembed_feat_combo, refs, tol_ppm=tol,
            min_prevalence=float(self.cembed_prev_spin.value()), value="rel_intensity")
        if len(targets) < 2:
            self.statusBar().showMessage("Need ≥2 shared features — lower min-prevalence, pick a "
                                         "named feature set, or find peaks for the active set.")
            return
        # Pre-flight: a named feature set whose m/z don't occur in any slide's peaks would
        # extract all-zeros and only fail deep in the pooling loop, after a multi-minute load.
        # Catch it here. Consensus targets are built FROM these slides' peaks, so they always
        # overlap — skip the probe for them. The session-peak check is a proxy (extraction reads
        # the raw cube, which may carry signal at an unpicked m/z), so this asks, not blocks.
        if not self._cohort_combo_is_consensus(self.cembed_feat_combo):
            n_with, n_over = cohort_engine.features_present(refs, targets, tol_ppm=tol)
            if n_with and not n_over and not self._confirm_foreign_features(
                    self.cembed_feat_combo.currentText(), len(targets), tol):
                return
        method = self.cembed_method_combo.currentText().lower()
        cap = int(self.cembed_cap_spin.value())
        # Cross-sample scaling from the Samples-panel dropdown (mutually exclusive strategies).
        scaling = (self.cembed_scaling_combo.currentText()
                   if getattr(self, "cembed_scaling_combo", None) is not None else SCALE_ALIGN)
        zscore = (scaling == SCALE_ALIGN)
        cohort_scale = (scaling == SCALE_COHORT)   # cohort-global TIC anchor (Pixels unit)
        metric = self.cembed_metric.currentText()
        nn = int(self.cembed_nn.value())
        mindist = float(self.cembed_mindist.value())
        unit = self.cembed_unit_combo.currentText()
        seed = profiles.active_seed()
        exclude = set(getattr(self, "_cembed_region_unchecked", set()))   # regions to leave out

        def work(progress=None):
            from .. import multivariate
            # Stream one cube at a time, evicted as its last sample is consumed (RAM ≤ ~1 cube).
            # Pixels pools whole-slide pixels → load the dense cube; the aggregate units touch
            # only the region/slide pixels → stay lazy and stream just those (no whole-slide
            # load), so Region means on one big slide is seconds, not minutes.
            loader = cohort_engine.SectionLoader(dense=(unit == "Pixels"))
            loader.prime(refs)
            try:
                if unit == "Pixels":
                    # Confine the pooled cloud to the picked regions when the user has narrowed
                    # the Regions ▾ selection; an empty exclude set keeps the whole-slide default.
                    pix_load = loader.load
                    if exclude:
                        regions_of = self._cembed_regions_of(whole_slide=False, exclude=exclude)
                        def pix_load(ref, _base=loader.load, _regions_of=regions_of):
                            res = _base(ref)
                            if res is None:
                                return None
                            ds = res[0]
                            entries = _regions_of(ref)   # (label, pixel_index) for ticked regions
                            parts = [np.asarray(p, dtype=int) for _, p in entries
                                     if p is not None and len(p)]
                            if not parts:                # all of this slide's regions unticked
                                return None
                            return ds, np.unique(np.concatenate(parts))
                    return multivariate.pooled_embedding(
                        refs, targets, loader=pix_load, method=method, tol_ppm=tol,
                        per_sample_cap=cap, standardize_per_sample=zscore,
                        cohort_norm=cohort_scale, progress=progress,
                        random_state=seed, metric=metric, n_neighbors=nn, min_dist=mindist,
                        keep_features=True,    # keep features → colour-by-lipid in Studio
                        keep_geometry=True)    # keep pixel row/col → spatial cluster back-map
                regions_of = self._cembed_regions_of(
                    whole_slide=(unit == "Sample means"), exclude=exclude)
                return multivariate.pooled_region_embedding(
                    refs, targets, loader=loader.load, regions_of=regions_of, unit="region",
                    method=method, tol_ppm=tol, standardize_per_sample=zscore, progress=progress,
                    random_state=seed, metric=metric, n_neighbors=nn, min_dist=mindist)
            finally:
                loader.close()

        n = len(refs)
        s = "" if n == 1 else "s"
        if unit == "Pixels":
            if exclude:                          # region-confined pool → name the scope
                kept = [r for r in self._cembed_region_names() if r not in exclude]
                scope = f" · {len(kept)} region type{'' if len(kept) == 1 else 's'}" if kept else ""
            else:
                scope = ""
            busy = f"Pooling pixels from {n} sample{s}{scope} · {method.upper()}…"
        elif unit == "Region means":
            picked = self._cembed_region_names()
            kept = [r for r in picked if r not in exclude]
            scope = (f"{len(kept)} region type{'' if len(kept) == 1 else 's'} across "
                     f"{n} sample{s}") if kept else f"named regions across {n} sample{s}"
            busy = f"Embedding region means · {scope} · {method.upper()}…"
        else:                                    # Sample means
            busy = f"Embedding {n} sample mean{s} · {method.upper()}…"
        # A modal loading popup (not just the status-bar bar) so a long pooled load over many
        # slides is unmissable and the half-built view can't be clicked; it streams one cube at
        # a time, advancing the bar per sample, and stays cancelable.
        self._run(work, want_progress=True, busy=busy, modal=True, title="Cohort UMAP",
                  on_done=lambda emb: self._render_cembed(emb, targets, n_input=len(refs)))

    def _confirm_foreign_features(self, set_name, n_targets, tol_ppm):
        """Warn (Yes/No) when none of a chosen feature set's m/z match any sample's saved peaks
        within tolerance — the "feature set from another dataset" mistake that otherwise fails
        minutes into the pooled load. Returns True to embed anyway (the probe reads picked peaks
        only, so a rare slide can still carry raw-cube signal at an unpicked m/z)."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Features don't match these slides")
        box.setText(f"None of the {n_targets} features in “{set_name}” match any sample's saved "
                    f"peaks within {tol_ppm:g} ppm.")
        box.setInformativeText(
            "This usually means the feature set is from a different dataset, or the slides differ "
            "in polarity / calibration. Switch Features to “Consensus (shared peaks)”, or widen "
            "the m/z tolerance.\n\nEmbed anyway?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)
        return box.exec() == QtWidgets.QMessageBox.Yes

    def _ask_relocate(self, missing, total) -> bool:
        """Warn that some sample files are missing here and ask whether to relocate (Yes/No).
        Split out so the dialog can be stubbed in tests. Returns True to open the folder picker."""
        shown = "\n".join(f"  • {m.name}" for m in missing[:12])
        more = f"\n  …and {len(missing) - 12} more" if len(missing) > 12 else ""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Sample files not found")
        box.setText(f"{len(missing)} of {total} sample(s) can't be found on this machine.")
        box.setInformativeText(
            f"{shown}{more}\n\nThis usually means the cohort was moved or copied from another "
            "folder or computer, so its saved file paths point somewhere that no longer exists.\n\n"
            "Pick the folder that now holds the raw .imzML files — they'll be matched by name, "
            "relocated, and saved back to the cohort. Relocate now?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        return box.exec() == QtWidgets.QMessageBox.Yes

    def _remap_unchecked_after_relocate(self, old_keys: dict) -> None:
        """Carry the Samples ▾ exclusion set across a relocation. Relocation rewrites a
        sample's ``source``, which changes its :meth:`SampleRef.key`; without remapping, an
        unticked sample silently re-enters runs (its old key no longer matches). ``old_keys``
        maps ``id(sample) -> pre-relocation key``."""
        unchecked = getattr(self, "_cohort_sample_unchecked", None)
        if not unchecked:
            return
        for s in self.cohort.samples:
            old = old_keys.get(id(s))
            new = s.key()
            if old is not None and old != new and old in unchecked:
                unchecked.discard(old)
                unchecked.add(new)

    def _cembed_preflight_paths(self, refs) -> bool:
        """Guard a pooled embedding against a moved / copied cohort with stale file paths.

        Lists exactly which included samples can't load and *why* — a missing imzML ``source``,
        or (for a region sample) a missing/unreadable session or lost region pixels — instead
        of grinding through every sample to a generic "None could be loaded". For the
        source-missing case it offers an in-place relocation (one folder pick, matched by name,
        persisted, with the Samples ▾ exclusions and a failed-save both handled). Returns True
        to proceed, False to abort."""
        problems = cohort_engine.unloadable_samples(refs)
        if not problems:
            return True
        # Source-missing samples are relocatable by pointing at the data folder; region-session
        # problems are NOT (sessions live in this PC's SMILE store, not the data folder).
        relocatable = [r for r, why in problems if "source file not found" in why]
        if relocatable:
            if not self._ask_relocate(relocatable, len(refs)):
                return False                          # declined — abort quietly, no second modal
            root = filedialogs.get_existing_directory(
                self, "Folder containing the raw .imzML files")
            if not root:
                return False                          # cancelled the picker — abort quietly
            # Snapshot keys BEFORE the source rewrite so the exclusion set can be remapped.
            old_keys = {id(s): s.key() for s in self.cohort.samples}
            report = cohort_engine.relocate_samples(
                self.cohort.samples, [root], fix_sessions=True)
            self._remap_unchecked_after_relocate(old_keys)
            saved = self._save_cohort() if (report.fixed or report.repointed_sessions) else True
            bits = [f"Relocated {len(report.fixed)} sample(s)"]
            if report.repointed_sessions:
                bits.append(f"repointed {report.repointed_sessions} session(s)")
            if report.ambiguous:
                bits.append(f"{len(report.ambiguous)} skipped (same file name — "
                            "can't tell which slide)")
            if not saved:
                bits.append("but the ROSTER COULD NOT BE SAVED (read-only dir / disk full) — "
                            "the fix will be lost on restart")
            self.statusBar().showMessage(" · ".join(bits) + ".")
            self._refresh_sample_tree()
            problems = cohort_engine.unloadable_samples(refs)
            if not problems:
                return True
        # Whatever still can't load: surface it precisely. Proceed if enough remain to embed.
        loadable = len(refs) - len(problems)
        if loadable >= 1:
            self.statusBar().showMessage(
                f"{loadable} of {len(refs)} sample(s) will embed; {len(problems)} can't load "
                "(see the Samples panel).")
            return True
        lines = "\n".join(f"  • {r.name} — {why}" for r, why in problems[:14])
        more = f"\n  …and {len(problems) - 14} more" if len(problems) > 14 else ""
        QtWidgets.QMessageBox.warning(
            self, "Samples can't be loaded",
            f"None of the {len(refs)} ticked sample(s) can load on this machine:\n\n{lines}{more}"
            "\n\nA region sample also needs its session file (in this PC's SMILE store) and its "
            "saved region pixels — if you moved machines, copy the .smile-msi/sessions folder "
            "too, or re-create the regions here.")
        return False

    def _cembed_regions_of(self, *, whole_slide, exclude=()):
        """The ``regions_of(sample)`` callback the region-mean embedding pools over — each
        returned ``(label, pixel_index)`` becomes one embedded point (its mean feature vector).
        ``whole_slide`` → one entry per sample (its whole slide, or its own region) = *Sample
        means*; else its named regions (a promoted region-sample → just its region) = *Region
        means*, falling back to the whole slide only when a slide carries no named regions at
        all. ``exclude`` is the set of region names the user deselected in the Regions picker
        (Region-means only). Sessions are cached so each is parsed once."""
        from .. import session as session_mod
        exclude = set(exclude or ())
        sessions: dict = {}

        def _data(ref):
            sp = getattr(ref, "session_path", "") or ""
            if sp not in sessions:
                try:
                    sessions[sp] = session_mod.load_session(sp)
                except (OSError, ValueError):
                    sessions[sp] = None
            return sessions[sp]

        def regions_of(ref):
            data = _data(ref)
            if not data:
                return []
            npx = int(data.get("n_pixels") or getattr(ref, "n_pixels", 0) or 0)
            reg = getattr(ref, "region", "") or ""
            if reg:                              # a promoted region-sample → just its own region
                if not whole_slide and reg in exclude:
                    return []
                pix = cohort_engine.region_pixels(data, reg)
                return [(ref.name if whole_slide else reg, pix)] if pix.size else []
            if whole_slide:                      # Sample means: the whole slide as one unit
                return [(ref.name, np.arange(npx))] if npx else []
            named = [str(r.get("name", "") or "") for r in (data.get("named_regions") or [])]
            out = [(n, cohort_engine.region_pixels(data, n)) for n in named if n not in exclude]
            out = [(n, p) for n, p in out if p is not None and p.size]
            if out:
                return out
            if named:                            # had regions but all deselected/gone → respect that
                return []
            return [(ref.name, np.arange(npx))] if npx else []   # no regions at all → whole slide
        return regions_of

    def _render_cembed(self, emb, targets, n_input=None):
        if emb is None or emb.coords.shape[0] == 0:
            self.statusBar().showMessage("No poolable samples — pooled embedding needs imzML-backed "
                                         "samples with saved peaks on a shared axis.")
            return
        self._cembed_emb = emb
        self._draw_cembed()
        self.b_cembed_png.setEnabled(True)
        self.b_cembed_csv.setEnabled(True)
        self.b_cembed_studio.setEnabled(True)
        # Back-mapping needs per-pixel tissue geometry (Pixels unit only).
        self.b_cembed_clustermap.setEnabled(getattr(emb, "px_row", None) is not None)
        n = emb.coords.shape[0]
        # Count drops against the INPUT refs (the Samples ▾ subset actually embedded), not the
        # full roster — else samples the user deliberately unticked are mislabelled as
        # "skipped (no signal)".
        base = n_input if n_input is not None else len(self.cohort.samples)
        miss = base - len(emb.sample_names)
        skipped = f" · {miss} sample(s) skipped (non-imzML / no signal)" if miss > 0 else ""
        unit_word = {"Pixels": "pooled pixels", "Region means": "region means",
                     "Sample means": "sample means"}.get(self.cembed_unit_combo.currentText(),
                                                         "points")
        self.cembed_info.setText(
            f"{len(emb.sample_names)} samples · {len(targets)} shared features · {n:,} {unit_word} "
            f"· {emb.method}{skipped}. Colour by group to compare biology across slides.")
        self._goto_cembed_tab()

    def _cembed_label_array(self, emb, color):
        """Per-point display label for the chosen Colour-by, as an ``(N,)`` object array."""
        if color == "Sample":
            return np.asarray(emb.sample_names, dtype=object)[emb.sample_id]
        if color == "Region" and getattr(emb, "region", None) is not None:
            return np.array([r or "(whole slide)" for r in emb.region], dtype=object)
        if color == "Cluster":
            return self._cembed_cluster_labels(emb)
        # Group — also the fallback for "Region" on a per-pixel embedding (no region per point)
        return np.array([g or UNGROUPED for g in emb.group], dtype=object)

    def _cembed_cluster_labels(self, emb):
        """Unsupervised cluster label per point (``'Cluster N'`` / ``'Noise'``), computed in
        the embedding plane and cached on the embedding — recomputed only when the cluster
        method / k changes. This is the 'molecular histology' grouping the atlas figures
        colour by; :meth:`_cembed_cluster_map` paints the same labels back onto tissue."""
        method = ("hdbscan" if self.cembed_cluster_method.currentText().lower().startswith("hdb")
                  else "kmeans")
        k = int(self.cembed_cluster_k.value())
        key = (method, k, int(emb.coords.shape[0]))
        if getattr(emb, "_cluster_key", None) == key and getattr(emb, "_cluster_labels", None) is not None:
            return emb._cluster_labels
        from .. import multivariate
        raw = multivariate.cluster_points(emb.coords, method=method, k=k,
                                          random_state=profiles.active_seed())
        labels = np.array(["Noise" if int(v) < 0 else f"Cluster {int(v) + 1}" for v in raw],
                          dtype=object)
        emb._cluster_labels = labels
        emb._cluster_raw = np.asarray(raw, dtype=int)
        emb._cluster_key = key
        return labels

    def _schedule_cembed_recluster(self, *_):
        """Debounce rapid cluster-control changes (spinning k, switching method) into a single
        recluster+redraw ~150 ms later, mirroring figeditor's _schedule_render."""
        self._cembed_recluster_timer.start(150)

    def _cembed_recluster(self):
        """Re-cluster + redraw when the cluster controls change — but only while the Cluster
        colouring is showing (otherwise the next draw/back-map picks up the change lazily)."""
        emb = getattr(self, "_cembed_emb", None)
        if emb is not None:
            emb._cluster_key = None              # invalidate cache → recompute on next read
        if self.cembed_color_combo.currentText() == "Cluster":
            self._draw_cembed()

    def _draw_cembed(self):
        emb = getattr(self, "_cembed_emb", None)
        if emb is None:
            return
        self.cembed_plot.clear()
        if self.cembed_legend is not None:
            self.cembed_legend.clear()
        color = self.cembed_color_combo.currentText()
        labels = self._cembed_label_array(emb, color)
        uniq = list(dict.fromkeys(labels.tolist()))     # first-seen order → stable colours
        color_key = {u: REGION_PALETTE[i % len(REGION_PALETTE)] for i, u in enumerate(uniq)}
        # Density-composite the dense per-pixel cloud (the atlas look); fall back to dots for
        # the few-point region/sample-mean clouds (a density raster of a handful of points is
        # meaningless) or whenever the user picks the Dots style.
        per_pixel = getattr(emb, "px_row", None) is not None
        want_density = self.cembed_style_combo.currentText() == "Density"
        if per_pixel and want_density and emb.coords.shape[0] >= 50:
            # The density compositor bakes a white background (the atlas/paper look), so pin
            # the plot to white here and restore the theme background for dots.
            self.cembed_plot.setBackground("w")
            self._draw_cembed_density(emb, labels, uniq, color_key)
        else:
            self.cembed_plot.setBackground(self.palette().color(QtGui.QPalette.Base))
            self._draw_cembed_dots(emb, labels, uniq, color_key)
        self.cembed_plot.autoRange()

    def _draw_cembed_dots(self, emb, labels, uniq, color_key):
        """One marker per point, coloured by category — the readable mode for the few-point
        region/sample-mean clouds (and available for pixels via the Style selector)."""
        size = 10 if getattr(emb, "region", None) is not None else 5   # aggregates: fewer, bigger
        for u in uniq:
            m = labels == u
            sp = pg.ScatterPlotItem(x=emb.coords[m, 0], y=emb.coords[m, 1], size=size,
                                    brush=pg.mkBrush(color_key[u]), pen=None, name=u)
            sp.sigClicked.connect(lambda _sp, _pts, lab=u: self.statusBar().showMessage(
                f"{self.cembed_color_combo.currentText()}: {lab}"))
            self.cembed_plot.addItem(sp)
            if self.cembed_legend is not None:
                self.cembed_legend.addItem(sp, f"{u}  ({int(m.sum()):,})")

    def _draw_cembed_density(self, emb, labels, uniq, color_key):
        """Composite the per-pixel cloud into a smooth density raster (the same compositor
        UMAP Studio uses), then drop one labelled centroid marker per category so the legend
        + on-plot labels survive — the inline preview of the publication figure."""
        from .. import umapstudio as us
        rgba = us.shade_categorical(emb.coords, labels, color_key, width=700, height=700,
                                    background="transparent")
        (x0, x1), (y0, y1) = us.shared_ranges(emb.coords)
        img = pg.ImageItem()
        img.setOpts(axisOrder="row-major")           # rgba is (H, W, 4); not a per-item data item
        img.setImage(rgba)
        img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.cembed_plot.addItem(img)
        for u in uniq:
            m = labels == u
            cx, cy = float(np.median(emb.coords[m, 0])), float(np.median(emb.coords[m, 1]))
            sp = pg.ScatterPlotItem(x=[cx], y=[cy], size=11, brush=pg.mkBrush(color_key[u]),
                                    pen=pg.mkPen("k", width=1), name=u)
            sp.sigClicked.connect(lambda _sp, _pts, lab=u: self.statusBar().showMessage(
                f"{self.cembed_color_combo.currentText()}: {lab}"))
            self.cembed_plot.addItem(sp)
            if self.cembed_legend is not None:
                self.cembed_legend.addItem(sp, f"{u}  ({int(m.sum()):,})")

    def _export_cembed_csv(self):
        """Export the embedding as a CSV — one row per point (pixel, or a region/sample mean):
        its 2-D coordinates, originating sample and cohort group (plus region + pixel count for
        the aggregate units) — so the cloud can be re-plotted or analysed outside the app."""
        emb = getattr(self, "_cembed_emb", None)
        if emb is None or emb.coords.shape[0] == 0:
            self.statusBar().showMessage("Run a pooled embedding first.")
            return
        path, _ = filedialogs.get_save_file_name(
            self, "Export embedding coordinates",
            f"{emb.method.lower()}_coords.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        names = np.asarray(emb.sample_names, dtype=object)
        samples = names[emb.sample_id]
        m = emb.method.lower()
        region = getattr(emb, "region", None)            # set for the region/sample-mean units
        sizes = getattr(emb, "sizes", None)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            head = [f"{m}_1", f"{m}_2", "sample", "group"]
            if region is not None:
                head += ["region", "n_pixels"]
            w.writerow(head)
            for i in range(emb.coords.shape[0]):
                row = [f"{emb.coords[i, 0]:.6g}", f"{emb.coords[i, 1]:.6g}",
                       samples[i], emb.group[i] or ""]
                if region is not None:
                    row += [region[i], int(sizes[i]) if sizes is not None else ""]
                w.writerow(row)
        # For the aggregate units (region / sample means) also write the intensity matrix the
        # cloud was built from — one row per point, one column per shared-axis m/z — so the
        # per-region mean spectra can be analysed directly (fold-changes, mixed models, …). The
        # stored values are the uncentred region means; ``expm1`` undoes the log1p the embedding
        # applied so the CSV carries linear (TIC-normalized) intensities.
        feats = getattr(emb, "features", None)
        if region is not None and feats is not None and getattr(feats, "size", 0):
            import os
            stem = (path[:-len("_coords.csv")] if path.endswith("_coords.csv")
                    else os.path.splitext(path)[0])
            mpath = stem + "_region_matrix.csv"
            targets = np.asarray(getattr(emb, "targets", []), dtype=float)
            linear = np.expm1(feats) if getattr(emb, "log1p", False) else np.asarray(feats)
            with open(mpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample", "region", "n_pixels"] + [f"{mz:.4f}" for mz in targets])
                for i in range(linear.shape[0]):
                    w.writerow([samples[i], region[i],
                                int(sizes[i]) if sizes is not None else ""]
                               + [f"{v:.6g}" for v in linear[i]])
            self.statusBar().showMessage(
                f"Wrote {path} and {mpath} — {linear.shape[0]}×{linear.shape[1]} region×m/z matrix")
            return
        self.statusBar().showMessage(f"Wrote {path}")

    def _goto_cembed_tab(self):
        self.reveal_view("Cohort UMAP")

    def _cembed_sync_scaling(self, *_):
        """Keep the legacy hidden 'Align samples' checkbox in step with the Samples-panel
        'Cross-sample scaling' dropdown, so anything still reading cembed_norm_chk sees the
        z-score state. The dropdown is the source of truth that _cembed_run reads."""
        chk = getattr(self, "cembed_norm_chk", None)
        if chk is not None:
            chk.setChecked(self.cembed_scaling_combo.currentText() == SCALE_ALIGN)

    def _cembed_sync_unit(self):
        """The Pixels/sample cap only applies to the per-pixel unit; the Regions picker only to
        Region means. Grey out what doesn't apply to the chosen unit."""
        unit = self.cembed_unit_combo.currentText()
        self.cembed_cap_spin.setEnabled(unit == "Pixels")
        btn = getattr(self, "cembed_regions_btn", None)
        if btn is not None:
            # Region means picks which regions become points; Pixels uses the same picker to
            # confine the pooled per-pixel cloud to chosen regions. Sample means has no use for it.
            btn.setEnabled(unit in ("Region means", "Pixels"))
        # Region means can run on a single multi-region slide, so the Run button's enabled
        # state depends on the chosen unit — re-evaluate whenever the unit changes.
        self._update_cohort_run_enabled()
        # Spell out what a "point" is for the chosen unit, since the region/sample distinction
        # is the easy thing to trip on: Region means is the one that compares regions.
        info = getattr(self, "cembed_info", None)
        if info is not None:
            if unit == "Region means":
                info.setText("Region means — each point is one named region of one sample. Use "
                             "'Regions ▾' to pick which named regions to compare; every matching "
                             "region from every sample is a point. Colour by Region to compare "
                             "tissue types, or by Sample/Group to see how they vary across slides.")
            elif unit == "Sample means":
                info.setText("Sample means — one point per slide (its whole-slide mean). Needs ≥2 "
                             "samples; pick 'Region means' to compare regions instead.")
            else:
                info.setText("Pixels — every (capped) pixel from every sample, pooled into one "
                             "embedding: the dense 'molecular histology' cloud. Colour by Cluster "
                             "for the atlas look, then 'Cluster map' to paint those clusters back "
                             "onto the tissue. Runs on a single slide too. Use 'Regions ▾' to pool "
                             "only chosen regions — untick any region to keep just the ticked ones "
                             "(background dropped); leave all ticked for the whole slide.")

    # ---- Region-means: pick which named regions across the cohort to embed ---- #
    def _build_cembed_region_picker(self):
        """A ``Regions ▾`` picker (searchable / sortable :class:`CheckPicker`) over every
        named region across the cohort — pick which become points in a Region-means
        embedding. Unticked names live in ``_cembed_region_unchecked`` (so a newly saved
        region defaults *on*); the list repopulates from the live cohort on each open."""
        if not hasattr(self, "_cembed_region_unchecked"):
            self._cembed_region_unchecked: set[str] = set()
        return CheckPicker(
            "Regions", lambda: [(n, n) for n in self._cembed_region_names()],
            lambda: self._cembed_region_unchecked, noun="region",
            tooltip="Choose which named regions to use across samples. Filter and All/None to "
                    "(un)tick the shown subset in one click. In Region means each ticked region "
                    "becomes a point; in Pixels the pooled per-pixel cloud is confined to the "
                    "ticked regions (untick any to drop it + background; leave all ticked for "
                    "the whole slide). Unticked names are left out.",
            parent=self)

    def _cembed_region_names(self):
        """Distinct named-region labels across the cohort — each slide's saved regions plus any
        promoted region-samples — in first-seen order. The menu of pickable regions."""
        from .. import session as session_mod
        names, seen, sessions = [], set(), {}
        for ref in (getattr(getattr(self, "cohort", None), "samples", None) or []):
            reg = getattr(ref, "region", "") or ""
            if reg:
                if reg not in seen:
                    seen.add(reg); names.append(reg)
                continue
            sp = getattr(ref, "session_path", "") or ""
            if sp not in sessions:
                try:
                    sessions[sp] = session_mod.load_session(sp)
                except Exception:  # noqa: BLE001 — a missing/corrupt session contributes no names
                    sessions[sp] = None
            data = sessions[sp]
            for r in ((data or {}).get("named_regions") or []):
                nm = str(r.get("name", "") or "")
                if nm and nm not in seen:
                    seen.add(nm); names.append(nm)
        return names

    # ---- per-run sample picker (shared across every cohort analysis) ----- #
    # Selecting which samples a run uses *without* removing them from the roster: a Samples ▾
    # popover (the twin of Regions ▾) backed by one shared unchecked-keys set, so the same
    # selection applies to the embedding, the comparison and batch correction. It's per-run UI
    # state — nothing is written to the saved cohort — so unticking is fully reversible and a
    # newly added sample defaults *in*.
    def _cohort_included_refs(self):
        """The cohort samples this analysis run uses — every roster sample minus the ones
        unticked in the Samples ▾ picker. A sample whose key isn't in the unchecked set
        defaults in, so freshly added samples participate automatically."""
        unchecked = getattr(self, "_cohort_sample_unchecked", set())
        return [r for r in self.cohort.samples if r.key() not in unchecked]

    def _build_cohort_sample_picker(self):
        """A ``Samples ▾`` picker (searchable / sortable :class:`CheckPicker`) over every
        cohort sample — pick which ones this run uses, leaving the roster untouched.
        Unticked keys live in the shared ``_cohort_sample_unchecked`` set; the list
        repopulates from the live cohort on each open. Build one per analysis toolbar (they
        share the set). Mirrors :meth:`_build_cembed_region_picker`."""
        if not hasattr(self, "_cohort_sample_unchecked"):
            self._cohort_sample_unchecked = set()
            self._cohort_sample_pickers = []
        picker = CheckPicker(
            "Samples", self._cohort_sample_entries,
            lambda: self._cohort_sample_unchecked, noun="sample",
            tooltip="Choose which cohort samples this run uses. Filter by name or group, then "
                    "All/None to (un)tick the shown subset in one click; sort by name or group. "
                    "Unticked samples are left out of this analysis but stay in the roster "
                    "(re-tick anytime) — the same selection applies to the embedding, "
                    "comparison and batch correction.",
            parent=self)
        self._cohort_sample_pickers.append(picker)
        return picker

    def _cohort_sample_entries(self):
        """``(key, label, group)`` for every cohort sample, in roster order — the menu of
        pickable samples. Region samples carry their ``file · region`` name; a ▸ marks them.
        The group lets the picker sort/filter by cohort group label."""
        out = []
        for r in (getattr(getattr(self, "cohort", None), "samples", None) or []):
            label = ("▸ " if getattr(r, "region", "") else "") + (r.name or r.key())
            out.append((r.key(), label, getattr(r, "group", "") or ""))
        return out

    def _cembed_embedding_data(self):
        """Build the pooled embedding's :class:`~smile_msi.umapstudio.EmbeddingData` with its
        colourable channels (Sample / Group / Region / Cluster + retained features), or ``None``
        if no embedding has been run. Shared by UMAP Studio and the publication export."""
        emb = getattr(self, "_cembed_emb", None)
        if emb is None or emb.coords.shape[0] == 0:
            return None
        from .. import umapstudio as us
        names = np.asarray(emb.sample_names, dtype=object)
        per_pixel_names = names[emb.sample_id]
        cat = {"Sample": per_pixel_names,
               "Group": np.array([g or UNGROUPED for g in emb.group], dtype=object)}
        # Region channel: use the embedding's OWN per-point region label (region/sample-mean
        # runs carry one per point in emb.region — the exact set of regions the run pooled
        # over) so Studio shows the same regions as the in-tab plot. Mirrors
        # _cembed_label_array. The per-pixel embedding has no per-point region, so fall back
        # to the cohort roster's sample→region map (promoted region-ROI samples).
        emb_region = getattr(emb, "region", None)
        if emb_region is not None:
            regions = np.array([r or "" for r in emb_region], dtype=object)
        else:
            region_of = {r.name: (r.region or "") for r in getattr(self.cohort, "samples", []) or []}
            regions = np.array([region_of.get(n, "") for n in per_pixel_names], dtype=object)
        if any(bool(r) for r in regions):
            cat["Region"] = np.where(regions == "", "(whole slide)", regions)
        # Unsupervised clusters of the embedding (the 'molecular histology' channel) — only
        # meaningful per-pixel, where the cloud is dense enough to carve into phenotypes.
        if getattr(emb, "px_row", None) is not None:
            cat["Cluster"] = self._cembed_cluster_labels(emb)
        return us.EmbeddingData(coords=emb.coords, method=emb.method, categorical=cat,
                                features=getattr(emb, "features", None), feature_mz=emb.targets,
                                counts=emb.counts)

    def _open_cembed_studio(self):
        """Hand the pooled embedding to UMAP Studio (Sample / Group / Region / Cluster + feature
        channels) — the interactive figure editor."""
        data = self._cembed_embedding_data()
        if data is None:
            self.statusBar().showMessage("Run a pooled embedding first.")
            return
        self.open_umap_studio(data)

    def _export_cembed_dialog(self):
        """Publication export of the pooled-embedding scatter (live preview + colour-by /
        density-or-dots + Style preset), replacing the raw pyqtgraph screenshot."""
        data = self._cembed_embedding_data()
        if data is None:
            self.statusBar().showMessage("Run a pooled embedding first.")
            return
        from .plotexport import EmbeddingExportDialog
        EmbeddingExportDialog(self, data, default_color_by="Group",
                              default_name="cohort_embedding.png",
                              title="Export embedding").exec()

    def _cembed_cluster_map(self):
        """Paint the embedding's clusters back onto each slide's tissue — the linked
        spatial↔embedding 'molecular histology' figure. One tissue panel per sample, coloured
        by the *same* cluster key as the UMAP, shown in a popup with a Save button."""
        emb = getattr(self, "_cembed_emb", None)
        if emb is None or getattr(emb, "px_row", None) is None:
            self.statusBar().showMessage("Run a Pixels-unit embedding first — the cluster map "
                                         "needs per-pixel tissue coordinates.")
            return
        labels = self._cembed_cluster_labels(emb)
        uniq = list(dict.fromkeys(labels.tolist()))
        color_key = {u: REGION_PALETTE[i % len(REGION_PALETTE)] for i, u in enumerate(uniq)}
        fig = self._build_cluster_map_figure(emb, labels, color_key)
        self._show_figure_dialog(fig, "Cluster map — clusters painted onto tissue")

    def _build_cluster_map_figure(self, emb, labels, color_key):
        """A Matplotlib figure: one tissue panel per sample with EVERY tissue pixel filled by a
        cluster colour, plus a shared cluster legend. The embedding only carries the capped
        random subset of pixels it was trained on, so painting just those leaves a big slide as
        a few-percent speckle on white (looks "blank"). We instead propagate each cluster label
        across the slide's full tissue mask by nearest painted neighbour, giving the solid
        'molecular histology' atlas. Pure (no Qt) so it's unit-testable and exports identically."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import to_rgb
        from matplotlib.lines import Line2D

        names = list(emb.sample_names)
        n = max(1, len(names))
        ncol = min(3, n)
        nrow = -(-n // ncol)                         # ceil
        # constrained_layout (not tight_layout) so empty/odd-sized panels and the side legend
        # never collapse onto each other — the old tight_layout overlapped titles when a panel
        # had nothing to draw.
        fig = Figure(figsize=(3.2 * ncol + 1.8, 3.2 * nrow + 0.4), dpi=150, facecolor="white",
                     layout="constrained")
        FigureCanvasAgg(fig)
        prow = np.asarray(emb.px_row, dtype=int)
        pcol = np.asarray(emb.px_col, dtype=int)
        sid = np.asarray(emb.sample_id, dtype=int)
        keys = list(color_key)
        code = {u: i for i, u in enumerate(keys)}
        codes = np.array([code[lbl] for lbl in labels], dtype=int)
        palette_rgb = np.array([to_rgb(color_key[u]) for u in keys], dtype=float)
        shapes = emb.shapes or {}
        tissue_rc = getattr(emb, "tissue_rc", None) or {}
        used_keys: set = set()
        for si, name in enumerate(names):
            ax = fig.add_subplot(nrow, ncol, si + 1)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            shape = shapes.get(name)
            sel = (sid == si) & (prow >= 0) & (pcol >= 0)
            if not shape or shape[0] <= 0 or shape[1] <= 0 or not sel.any():
                # Draw a visible placeholder instead of an invisible axis-off panel, so a slide
                # with no back-mappable geometry reads as "no data" rather than a layout glitch.
                ax.text(0.5, 0.5, "no tissue geometry", ha="center", va="center",
                        fontsize=8, color="0.5", transform=ax.transAxes)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            H, W = int(shape[0]), int(shape[1])
            rr, cc = prow[sel], pcol[sel]
            scodes = codes[sel]
            inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr, cc, scodes = rr[inb], cc[inb], scodes[inb]
            # Target pixels to paint: the full tissue mask if we captured it (solid atlas), else
            # just the painted subset (back-compat for embeddings without tissue_rc).
            trc = tissue_rc.get(name)
            if trc is not None and len(trc[0]):
                trow = np.asarray(trc[0], dtype=int); tcol = np.asarray(trc[1], dtype=int)
                tinb = (trow >= 0) & (trow < H) & (tcol >= 0) & (tcol < W)
                trow, tcol = trow[tinb], tcol[tinb]
                fill_codes = self._nn_fill_labels(rr, cc, scodes, trow, tcol)
            else:
                trow, tcol, fill_codes = rr, cc, scodes
            img = np.zeros((H, W, 4), dtype=float)   # transparent background
            img[trow, tcol, :3] = palette_rgb[fill_codes]
            img[trow, tcol, 3] = 1.0
            used_keys.update(int(c) for c in np.unique(fill_codes))
            ax.imshow(img, interpolation="nearest")
            cov = int(sel.sum())
            total = len(trow) if trc is not None else cov
            ax.set_xlabel(f"{cov:,} labelled · {total:,} px", fontsize=7, color="0.4")
        # Legend: only clusters that actually appear, in palette order.
        handles = [Line2D([0], [0], marker="s", linestyle="", markersize=8,
                          markerfacecolor=color_key[keys[i]], markeredgecolor="none",
                          label=str(keys[i]))
                   for i in range(len(keys)) if i in used_keys]
        if handles:
            fig.legend(handles=handles, title="Cluster", loc="outside right upper",
                       frameon=False, fontsize=8)
        fig.suptitle("Molecular-histology clusters mapped to tissue", fontsize=11)
        return fig

    @staticmethod
    def _nn_fill_labels(rr, cc, codes, trow, tcol):
        """Assign each target pixel ``(trow, tcol)`` the cluster code of its nearest *labelled*
        pixel ``(rr, cc, codes)`` in tissue space — the cheap label propagation that turns the
        capped subset into a solid per-slide segmentation. Uses a KD-tree when SciPy is present,
        else a vectorised brute-force fallback (fine for the modest pixel counts here)."""
        src = np.column_stack([rr, cc]).astype(float)
        tgt = np.column_stack([trow, tcol]).astype(float)
        try:
            from scipy.spatial import cKDTree
            _, idx = cKDTree(src).query(tgt, k=1)
        except Exception:  # noqa: BLE001 — SciPy missing/old → brute force
            idx = np.empty(len(tgt), dtype=int)
            step = 4096
            for s in range(0, len(tgt), step):
                d2 = ((tgt[s:s + step, None, :] - src[None, :, :]) ** 2).sum(axis=2)
                idx[s:s + step] = d2.argmin(axis=1)
        return codes[idx]

    def _show_figure_dialog(self, fig, title):
        """Show a Matplotlib figure in a modal popup with pan/zoom + a Save button."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(940, 660)
        lay = QtWidgets.QVBoxLayout(dlg)
        canvas = FigureCanvasQTAgg(fig)
        lay.addWidget(NavigationToolbar2QT(canvas, dlg))
        lay.addWidget(canvas, 1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        b_save = QtWidgets.QPushButton("Save…")
        b_save.setIcon(icon("save"))

        def _save():
            path, _ = filedialogs.get_save_file_name(
                self, "Save cluster map", "cluster_map.png",
                "PNG (*.png);;TIFF (*.tif);;PDF (*.pdf);;SVG (*.svg)")
            if path:
                from .. import export
                export.save_figure(fig, path, dpi=200)   # multi-format + editable vector text
                self.statusBar().showMessage(f"Wrote {path}")

        b_save.clicked.connect(_save)
        b_close = QtWidgets.QPushButton("Close")
        b_close.clicked.connect(dlg.accept)
        row.addWidget(b_save); row.addWidget(b_close)
        lay.addLayout(row)
        dlg.exec()
