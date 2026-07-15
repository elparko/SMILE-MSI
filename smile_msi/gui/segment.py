"""SegmentTabMixin — extracted from the monolithic MainWindow (no behavior change)."""
from __future__ import annotations


import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import prefs, spatial, profiles
from .common import (PALETTE, REGION_PALETTE, HILITE, MUTED_QSS, GUIDE_LINE, ControlBar,
                     _SortItem, dark_image_view, eye_icon, tab_page, confirm, glossary_button,
                     set_header_tooltips, icon, button, primary_button, check_table_bar,
                     NoScrollComboBox, NoScrollSlider, NoScrollSpinBox)
from .dendrogram import DendrogramView
from .scope import ScopeBar
from . import filedialogs

# Default padding around an ROI's bounding box when auto-cropping an export to it, as a
# fraction of the ROI's extent — so the region sits in a little context, not flush to the edge.
CROP_PAD_FRAC = 0.10

# How many segments the Detail slider may reach. User-configurable (the "Max" spinbox,
# persisted in prefs under ``segment_detail_max``); the actual ceiling is min(this, the
# tree's own max_clusters). Defaults to 24 — past it the 12-colour segment key repeats
# heavily and tiny clusters get statistically noisy, so it's a sensible standard while
# the user is free to raise it for genuinely fine-structured tissue.
DEFAULT_SEGMENT_DETAIL_MAX = 24
SEGMENT_DETAIL_MAX_KEY = "segment_detail_max"
SEGMENT_DETAIL_HARD_MAX = 200      # the tree itself supports up to ~n_micro; spinbox ceiling

# How many segments a *new* segmentation starts at — the standard interpreted count for a
# single tissue section. User-configurable (the "Default" spinbox, persisted under
# ``segment_default_count``); the actual starting cut is min(this, the tree's own size).
# Defaults to 8: the mid-point of the field's typical 3–10 interpreted range (and Cardinal's
# segmentation-vignette default), low enough to resolve simple tissue (nerve ≈ 3–5, kidney
# ≈ 4) after merging, high enough to separate a whole complex section (brain ≈ 10).
DEFAULT_SEGMENT_COUNT = 8
SEGMENT_COUNT_KEY = "segment_default_count"


def segment_detail_max() -> int:
    """The user's configured Detail-slider ceiling (persisted in prefs), clamped to a sane
    range. The effective ceiling on any given tree is ``min(this, tree.max_clusters)``."""
    try:
        v = int(prefs.get(SEGMENT_DETAIL_MAX_KEY, DEFAULT_SEGMENT_DETAIL_MAX))
    except (TypeError, ValueError):
        v = DEFAULT_SEGMENT_DETAIL_MAX
    return max(2, min(SEGMENT_DETAIL_HARD_MAX, v))


def segment_default_count() -> int:
    """The user's preferred starting number of segments for a new segmentation (persisted in
    prefs), clamped to a sane range. The actual starting cut is ``min(this, tree size)``."""
    try:
        v = int(prefs.get(SEGMENT_COUNT_KEY, DEFAULT_SEGMENT_COUNT))
    except (TypeError, ValueError):
        v = DEFAULT_SEGMENT_COUNT
    return max(2, min(SEGMENT_DETAIL_HARD_MAX, v))


def _clamp_span(lo, hi, n):
    """Shift a ``[lo, hi]`` interval to fit inside ``[0, n]``, preserving its length when it
    fits (so a fixed-size crop stays the requested size, just nudged inside the image)."""
    length = min(hi - lo, float(n))
    if lo < 0:
        lo, hi = 0.0, length
    elif hi > n:
        hi, lo = float(n), n - length
    return max(0.0, lo), min(float(n), hi)


class SegmentTabMixin:
    def _tab_segment(self):
        w, v = tab_page()
        # "Data in this analysis": the tree is built from exactly these features.
        self.seg_scope = ScopeBar(self, feature=True)
        v.addWidget(self.seg_scope)
        # A wrapping ControlBar (not a hand-built QHBoxLayout) so the row reflows instead
        # of overlapping/clipping once the Options popover and status readout are added.
        bar = ControlBar()
        b_seg = primary_button(
            "Run segmentation", self.do_segment,
            tooltip="Build one granularity tree over the tissue (or the chosen region), "
                    "then drag the Detail slider to explore coarse↔fine — no need to pick "
                    "k up front. Algorithm / Distance / region live under Options.")
        bar.add(b_seg)
        self.b_seg = b_seg
        self.b_seg.setEnabled(False)                    # enabled once a dataset + peaks exist
        # --- Detail slider: cut the prebuilt tree live (replaces k / Auto / drill-down) ---
        self.detail_slider = NoScrollSlider(QtCore.Qt.Horizontal)
        self.detail_slider.setRange(2, segment_detail_max())   # re-sized to the tree on build
        self.detail_slider.setValue(segment_default_count())
        self.detail_slider.setEnabled(False)            # enabled once a tree is built
        self.detail_slider.setMaximumWidth(220)
        self.detail_slider.setToolTip("Drag to re-segment live: left = a few big regions, "
                                      "right = many fine ones. One model, cut at any level.")
        self.detail_slider.valueChanged.connect(self._detail_changed)
        self.detail_slider.sliderPressed.connect(lambda: setattr(self, "_detail_dragging", True))
        self.detail_slider.sliderReleased.connect(self._detail_released)
        self.detail_lbl = QtWidgets.QLabel("")
        bar.add_group("Detail", self._muted(QtWidgets.QLabel("coarse")), self.detail_slider,
                      self._muted(QtWidgets.QLabel("fine")), self.detail_lbl)
        # The starting number of segments for a new segmentation (user-set, persisted). Setting
        # it also jumps the live cut here; the slider still explores freely without changing it.
        self.detail_default_spin = NoScrollSpinBox()
        self.detail_default_spin.setRange(2, SEGMENT_DETAIL_HARD_MAX)
        self.detail_default_spin.setValue(segment_default_count())
        self.detail_default_spin.setMaximumWidth(64)
        self.detail_default_spin.setToolTip(
            "Number of segments a new segmentation starts at, saved across sessions. The "
            "typical interpreted count for one tissue section is ~3–10 (8 is a good default; "
            "nerve/simple tissue often fewer). Setting this also moves the current cut here; "
            "the slider still explores freely. Capped by Max and the tree's own size.")
        self.detail_default_spin.valueChanged.connect(self._detail_default_changed)
        bar.add_group("Default", self.detail_default_spin)
        # How fine the Detail slider may go (user-set, persisted). The effective ceiling on
        # any tree is min(this, the tree's own max_clusters); raising it past ~24 is reachable
        # but the 12-colour segment key starts repeating and small clusters get noisy.
        self.detail_max_spin = NoScrollSpinBox()
        self.detail_max_spin.setRange(4, SEGMENT_DETAIL_HARD_MAX)
        self.detail_max_spin.setValue(segment_detail_max())
        self.detail_max_spin.setPrefix("≤ ")
        self.detail_max_spin.setMaximumWidth(72)
        self.detail_max_spin.setToolTip(
            "Most segments the Detail slider can reach (saved across sessions). Higher = finer, "
            "but the segment colour key has 12 colours so they repeat, and very small segments "
            "get statistically noisy. The tree's own size still caps this.")
        self.detail_max_spin.valueChanged.connect(self._detail_max_changed)
        bar.add_group("Max", self.detail_max_spin)
        # --- Options popover: how the tree is built + what it covers (applied on next run) -
        self.seg_method = NoScrollComboBox()
        self.seg_method.addItem("Bisecting k-means", "bisecting")    # default
        self.seg_method.addItem("Ward (agglomerative)", "ward")
        self.seg_method.setToolTip("Clustering algorithm. Bisecting k-means repeatedly splits "
                                   "the worst cluster in two (default); Ward merges bottom-up "
                                   "to minimise within-cluster variance. Takes effect "
                                   "on the next 'Run segmentation'.")
        self.seg_metric = NoScrollComboBox()
        self.seg_metric.addItem("Correlation", "correlation")        # default metric, first
        self.seg_metric.addItem("Euclidean", "euclidean")
        self.seg_metric.setToolTip("Distance metric. Correlation groups pixels by spectral "
                                   "shape (ignoring absolute intensity); Euclidean groups by "
                                   "overall profile. Takes effect on the next run.")
        # defaults come from the active analysis profile (mirrored into prefs), so the lab's
        # standard algorithm/metric is pre-selected; the user can still change them per run.
        for combo, pref, fallback in ((self.seg_method, profiles.SEG_METHOD_PREF, "bisecting"),
                                      (self.seg_metric, profiles.SEG_METRIC_PREF, "correlation")):
            i = combo.findData(prefs.get(pref, fallback))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.seg_region = NoScrollComboBox()
        self.seg_region.setToolTip("Segment only within this region's pixels (e.g. one ROI), "
                                   "independently of the rest of the slide — or the whole slide.")
        self._sync_seg_region_combo()
        bar.add_group("Within:", self.seg_region)
        bar.add_more(("Algorithm:", self.seg_method), ("Distance:", self.seg_metric), label="Options")
        b_save_seg = button("Export…", lambda: self.open_export_hub("seg"), icon="export",
                            tooltip="Export the segmentation map (PNG/TIFF/PDF/SVG, themed, "
                                    "with a region legend) via the Export hub")
        bar.add(b_save_seg)
        b_export_labels = button("Labels CSV…", self._export_seg_labels, icon="export",
                                 tooltip="Export the per-pixel cluster labels as a table "
                                         "(pixel, x, y, label) — the map as data, for "
                                         "cross-tool comparison or external analysis.")
        bar.add(b_export_labels)
        self.seg_info = bar.set_status("")              # k · silhouette · variance readout
        bar.add(glossary_button([
            ["Segmentation", "Unsupervised clustering of pixels by their spectra into regions ('segments')."],
            ["Granularity tree", "One hierarchy built once; the Detail slider cuts it into more/fewer segments without re-running."],
            ["Detail / cut", "Where the tree is cut: left = a few big segments, right = many fine ones."],
            ["Bisecting k-means", "Repeatedly splits the worst cluster in two (divisive)."],
            ["Ward", "Agglomerative clustering that merges clusters bottom-up to minimise within-cluster variance."],
            ["Correlation distance", "Groups pixels by spectral shape, ignoring absolute intensity."],
            ["Silhouette", "How well-separated the segments are (higher = cleaner clusters; ~1 best)."],
            ["PCA EV", "Explained variance of the PCA used in the Euclidean path — how much signal the embedding keeps."],
            ["Lineage (0·2)", "A drilled sub-cluster's path: '0·2' = sub-cluster 2 of cluster 0."],
            ["Increase detail here", "Locally re-cluster one segment into sub-clusters (right-click / double-click)."]], parent=self))
        v.addWidget(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # --- left: segmentation image + colour legend ------------------------
        leftw = QtWidgets.QWidget()
        leftv = QtWidgets.QVBoxLayout(leftw)
        leftv.setContentsMargins(0, 0, 0, 0)
        leftv.setSpacing(2)
        self.seg_view = pg.GraphicsLayoutWidget()
        dark_image_view(self.seg_view)   # segmentation map reads against black in both themes
        self.seg_view.setMinimumWidth(280)
        vb = self.seg_view.addViewBox()
        vb.invertY(True)
        vb.setAspectLocked(True)
        self.seg_img_item = pg.ImageItem()
        vb.addItem(self.seg_img_item)
        self.seg_vb = vb
        self._register_optical_view(vb, self.seg_img_item)   # optical backdrop behind clusters
        self.seg_view.scene().sigMouseClicked.connect(self._seg_image_clicked)
        self.seg_view.scene().sigMouseMoved.connect(self._seg_hover)
        self.seg_view.viewport().installEventFilter(self)  # clear the hover-peek on leave
        leftv.addWidget(self.seg_view, 1)
        self.seg_legend = QtWidgets.QLabel()         # colour key for what's on the image
        self.seg_legend.setWordWrap(True)
        self.seg_legend.setTextFormat(QtCore.Qt.RichText)
        self.seg_legend.setStyleSheet(MUTED_QSS)
        leftv.addWidget(self.seg_legend)
        leftw.setMinimumWidth(280)
        split.addWidget(leftw)

        # --- centre: the HCA tree the Detail slider cuts (drag the line = drag detail) --
        self.seg_dendro = DendrogramView()
        self.seg_dendro.cutChanged.connect(self._dendro_cut_changed)
        self.seg_dendro.branchSelectionChanged.connect(self._dendro_branches_changed)
        self.seg_dendro.makeRegionRequested.connect(self._dendro_make_region)
        self.seg_dendro.modeChanged.connect(self._dendro_mode_changed)
        self._dendro_branch_micros = frozenset()
        split.addWidget(self.seg_dendro)

        # --- middle: segment list (tick the boxes, then assign via dropdown) --
        midw = QtWidgets.QWidget()
        mid = QtWidgets.QVBoxLayout(midw)
        mid.setContentsMargins(0, 0, 0, 0)
        mid.addWidget(QtWidgets.QLabel("Segments at this detail — hover or arrow-key a row to "
                                       "peek, tick to keep it shown, then 'Add checked to…' or right-click"))
        self.seg_table = QtWidgets.QTableWidget()
        self.seg_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.seg_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.seg_table.setMouseTracking(True)                       # emit cellEntered on hover
        self.seg_table.viewport().setMouseTracking(True)
        self.seg_table.itemSelectionChanged.connect(self._update_seg_preview)
        self.seg_table.itemChanged.connect(self._seg_item_changed)
        self.seg_table.cellEntered.connect(self._seg_row_entered)   # hover a row → peek overlay
        self.seg_table.currentCellChanged.connect(self._seg_current_changed)  # ↑/↓ → peek
        self.seg_table.cellDoubleClicked.connect(self._seg_double_clicked)
        self.seg_table.setToolTip("Hover or arrow-key a row to preview it on the map. "
                                  "Double-click a segment to drill into it; right-click "
                                  "for Add/Remove/Merge and detail options.")
        self.seg_table.viewport().installEventFilter(self)          # clear the peek on leave
        self.seg_table.installEventFilter(self)                     # …and when focus leaves
        self.seg_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.seg_table.customContextMenuRequested.connect(self._seg_table_menu)
        # At fine detail this table is 24 clusters deep; 'Clear ticks' sat in the Edit popover
        # with no way back. All/None/Invert, and the map preview follows every one of them.
        self._seg_bulk = check_table_bar(self.seg_table, 0, "segment",
                                         on_change=self._update_seg_preview)
        mid.addWidget(self._seg_bulk)
        mid.addWidget(self.seg_table)
        mid.addWidget(self._muted(QtWidgets.QLabel(
            "Tick segments, then group them into a region. The 'region' column shows "
            "what's already assigned. Right-click a segment for Add/Remove/Merge/"
            "Increase-detail; double-click a row to drill into it.")))
        # --- region grouping: clear, big targets in two rows ------------------
        exp = QtWidgets.QSizePolicy.Expanding

        def _bigbtn(text, tip, slot, cls=QtWidgets.QPushButton):
            b = cls()
            b.setText(text)
            if tip:
                b.setToolTip(tip)
            b.setMinimumHeight(34)
            b.setSizePolicy(exp, QtWidgets.QSizePolicy.Fixed)
            if cls is QtWidgets.QPushButton:
                b.clicked.connect(slot)
            return b

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(6)
        b_newreg = _bigbtn("New region from checked",
                           "Create a new named region from the ticked/selected segments",
                           self._new_region_from_target)
        b_newreg.setIcon(icon("add"))
        self.assign_btn = _bigbtn("Add to region",        # ▾ drawn by QSS (menuButton chevron)
                                  "Add the ticked segments (or the selected rows if none are "
                                  "ticked) to an existing region. A segment belongs to one "
                                  "region at a time, so it moves out of any other region it "
                                  "was in.",
                                  None, cls=QtWidgets.QToolButton)
        self.assign_btn.setObjectName("menuButton")       # shared dropdown pill (single arrow)
        self.assign_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.assign_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.assign_btn.setIcon(icon("add"))
        self.assign_menu = QtWidgets.QMenu(self.assign_btn)
        self.assign_btn.setMenu(self.assign_menu)
        self._rebuild_assign_menu()
        b_rm = _bigbtn("Remove from region",
                       "Remove the ticked segment(s) from whatever region they belong to",
                       self._remove_checked)
        b_rm.setIcon(icon("remove"))
        for b in (b_newreg, self.assign_btn, b_rm):
            row1.addWidget(b)

        # Secondary edit ops (Merge / Reset / Clear ticks) live one click deeper in an
        # "Edit ▾" popover — the buttons keep their _bigbtn construction (so .clicked is
        # already wired) and their Phase-2 gate/danger style.
        b_merge = _bigbtn("Merge",
                          "Merge the ticked/selected segments into one — picking one sub-cluster "
                          "of a split merges the whole group (inverse of 'Increase detail here')",
                          lambda: self._merge_selected())
        b_merge.setIcon(icon("merge"))
        b_reset = _bigbtn("Reset",
                          "Discard merges/splits and return to the original segmentation at this detail",
                          self._reset_splits_confirm)
        b_reset.setIcon(icon("refresh"))
        b_clear = _bigbtn("Clear ticks", "Untick every segment", lambda: self._check_segments([]))
        b_clear.setIcon(icon("remove"))
        b_all_regions = _bigbtn(
            "Regions from all segments",
            "Create one region per segment at the current detail (the terminal branches of "
            "the tree), named '<name>_01', '<name>_02', … — turn a whole cut into a labelled "
            "region set in one step.",
            self._regions_from_all_segments)
        b_all_regions.setIcon(icon("add"))
        self.b_seg_all_regions = b_all_regions
        self.b_seg_all_regions.setEnabled(False)        # enabled once a tree is cut
        self.b_seg_merge = b_merge
        self.b_seg_merge.setEnabled(False)              # enabled once ≥2 clusters exist
        b_reset.setObjectName("dangerAction")
        edit_btn = QtWidgets.QToolButton()
        edit_btn.setText("Edit")                          # ▾ drawn by QSS (menuButton chevron)
        edit_btn.setObjectName("menuButton")              # shared dropdown pill (single arrow)
        edit_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        edit_btn.setMinimumHeight(34)
        edit_btn.setSizePolicy(exp, QtWidgets.QSizePolicy.Fixed)
        edit_menu = QtWidgets.QMenu(edit_btn)
        ebox = QtWidgets.QWidget()
        ecol = QtWidgets.QVBoxLayout(ebox)
        ecol.setContentsMargins(10, 8, 10, 8)
        ecol.setSpacing(6)
        for b in (b_all_regions, b_merge, b_reset, b_clear):
            ecol.addWidget(b)
            b.clicked.connect(edit_menu.close)          # parity with add_more auto-close
        ewa = QtWidgets.QWidgetAction(edit_menu)
        ewa.setDefaultWidget(ebox)
        edit_menu.addAction(ewa)
        edit_btn.setMenu(edit_menu)
        self.b_seg_edit = edit_btn

        mid.addLayout(row1)
        mid.addWidget(edit_btn)
        # Granularity is now the Detail slider above (one tree, cut live) — it replaces
        # the old k spinbox, Auto-k, and the manual Split/Reset/Show-all drill bar.
        # 'Increase detail here' (right-click a segment) still re-clusters one region
        # locally via _split_selected for interactive drill-in.
        midw.setMinimumWidth(280)
        split.addWidget(midw)
        # Named regions now live in the right-hand dock (Regions panel) so they're
        # reachable from every view; assign ticked segments to them via 'Add checked
        # to…', the right-click menu, or by clicking a region there.

        split.setChildrenCollapsible(False)              # no pane may vanish to zero width
        split.setSizes([560, 320, 440])
        v.addWidget(split, 1)                            # the canvas owns the vertical slack
        self.tabs.addTab(w, "Segmentation")
        self._refresh_seg_run_state()

    # ----- segment -> region builder --------------------------------------- #
    @staticmethod
    def _color_icon(hexc, size=14):
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(hexc))
        return QtGui.QIcon(pix)

    def _set_seg_image(self, color_for_cluster, dim=False, ignore_hidden=False):
        """Render the segmentation. ``color_for_cluster(cl)`` returns a hex color or
        None; clusters mapped to None are painted faint gray when ``dim`` else hidden.
        Hidden clusters (``self._seg_hidden``) are fully transparent unless
        ``ignore_hidden``. Off-tissue pixels stay transparent."""
        if self.seg is None:
            return
        hidden = set() if ignore_hidden else set(getattr(self, "_seg_hidden", ()))
        label_img = self.seg.label_image                 # (h,w) float, NaN off-tissue
        h, wd = label_img.shape
        rgba = np.zeros((h, wd, 4), dtype=np.ubyte)
        for cl in range(self.seg.n_clusters):
            if cl in hidden:                             # hidden -> transparent
                continue
            mask = (label_img == cl)
            hexc = color_for_cluster(cl)
            if hexc is None:
                if dim:
                    rgba[mask] = [70, 70, 70, 255]
            else:
                c = QtGui.QColor(hexc)
                rgba[mask] = [c.red(), c.green(), c.blue(), 255]
        self.seg_img_item.setImage(rgba)

    def _update_seg_legend(self, pairs, note=""):
        """Set the colour key under the image. ``pairs`` is [(hex, label), …]."""
        if getattr(self, "seg_legend", None) is None:
            return
        chips = [f'<span style="background:{hexc}">&nbsp;&nbsp;&nbsp;</span>&nbsp;{lab}'
                 for hexc, lab in pairs]
        text = " &nbsp; ".join(chips)
        if note:
            text = (text + " &nbsp; " if text else "") + f"<i>{note}</i>"
        self.seg_legend.setText(text)

    def _render_seg_base(self):
        self._set_seg_image(lambda cl: PALETTE[cl % len(PALETTE)])
        lin = getattr(self, "_seg_lineage", {})
        hidden = set(getattr(self, "_seg_hidden", ()))
        vis = [cl for cl in range(self.seg.n_clusters) if cl not in hidden] if self.seg else []
        pairs = [(PALETTE[cl % len(PALETTE)], lin.get(cl, str(cl))) for cl in vis[:16]]
        note = f"+{len(vis) - 16} more" if len(vis) > 16 else ""
        if hidden:
            note = (note + " · " if note else "") + f"{len(hidden)} hidden"
        self._update_seg_legend(pairs, note + (" · cluster colours" if not note else ""))

    def _preview_segments(self, segs, hover=None, pinned=None):
        """Highlight ``segs`` on the map (everything else dimmed; hide is ignored so
        ticked segments always show). ``pinned`` ones (ticked) stay shown and read with
        a ✓; the ``hover`` segment is the transient peek and renders solid white so it
        pops forward over the pinned ones."""
        hl = set(segs)
        pinned = set(pinned or ())

        def color_for(cl):
            if cl not in hl:
                return None
            if cl == hover and cl not in pinned:         # the peek pops white, on top
                return "#ffffff"
            return PALETTE[cl % len(PALETTE)]

        self._set_seg_image(color_for, dim=True, ignore_hidden=True)
        lin = getattr(self, "_seg_lineage", {})
        pairs = []
        for cl in sorted(hl):
            lab = lin.get(cl, str(cl))
            is_peek = cl == hover and cl not in pinned
            if is_peek:
                lab += " · hover"
            elif cl in pinned:
                lab += " ✓"
            pairs.append(("#ffffff" if is_peek else PALETTE[cl % len(PALETTE)], lab))
        note = ("✓ ticked stays shown · hover or arrow-key a row to peek"
                if (pinned or hover is not None) else "highlighted")
        self._update_seg_legend(pairs, note)

    def _preview_regions(self, ris):
        """Highlight every selected region together on the segmentation map. Only
        cluster-based regions paint here (ROI-mask regions live on the ion image);
        when the selection is all ROI regions this is a no-op on the seg view."""
        if self.seg is None:
            return
        cmap = {}
        for ri in ris:
            if not self.regions[ri].get("visible", True):
                continue
            for cl in self.regions[ri].get("segments", ()):
                cmap[cl] = self.regions[ri]["color"]
        if not cmap:
            return
        self._set_seg_image(lambda cl: cmap.get(cl), dim=True)
        self._update_seg_legend([(self.regions[ri]["color"], self.regions[ri]["name"])
                                 for ri in ris], "selected region(s)")

    def _show_all_regions(self):
        """'Show all' — paint every visible region's footprint onto the ion image (and the
        segmentation map too, when a segmentation exists). Unlike _render_region_map alone,
        this shows ROI-drawn regions even with no segmentation run, which is what the button
        previously failed to do."""
        ris = [i for i, rg in enumerate(self.regions)
               if rg.get("visible", True) and self._region_pixel_mask(rg) is not None]
        if not ris:
            self.statusBar().showMessage("No regions to show yet — draw an ROI → 'Add ROI', "
                                         "or group segmentation clusters into a region.")
            return
        self._show_region_overlay(ris)                    # footprints on the ion image
        if self.seg is not None:
            self._render_region_map()                     # and on the segmentation map
        self.reveal_view("Ion image")          # grouped-tab safe (raw index is brittle post-IA)
        self.statusBar().showMessage(f"Showing {len(ris)} region(s) on the ion image.")

    def _render_region_map(self):
        if self.seg is None:
            return
        cmap = {cl: rg["color"] for rg in self.regions
                if rg.get("visible", True) and self._region_on_active_slide(rg)
                for cl in rg["segments"]}
        if not cmap:
            self._render_seg_base()
            return
        self._set_seg_image(lambda cl: cmap.get(cl), dim=True)
        self._update_seg_legend([(rg["color"], rg["name"]) for rg in self.regions
                                 if rg["segments"] and rg.get("visible", True)
                                 and self._region_on_active_slide(rg)],
                                "unassigned = grey")

    def _rebuild_seg_table(self, keep_checked=False):
        """Recompute the per-cluster signature rows for the current leaves and refill
        the table (ordered by lineage so a parent's sub-clusters sit together). With
        ``keep_checked`` the ticks survive the refill (e.g. drilling into a segment
        shouldn't untick the ones you've pinned)."""
        if self.seg is None:
            return
        prev_checked = set(self._checked_segments()) if keep_checked else set()
        X = self.ds.feature_matrix(self.norm)
        # Label columns off the feature matrix's OWN peaks, not self.peaks: the GUI peak
        # list can drift from the matrix the segmentation was built on (e.g. a stats-derived
        # feature list), which used to make mzs[j] index out of range (IndexError).
        fpeaks = self.ds.feature_peaks
        mzs = list(fpeaks) if fpeaks is not None else [p["mz"] for p in self.peaks]
        lin = getattr(self, "_seg_lineage", {})
        rows = []
        for cl in range(self.seg.n_clusters):
            m = self.seg.mask(cl)
            if not m.any():
                continue                                  # a fully-reassigned id: skip
            means = X[m].mean(0)
            order = np.argsort(-means)[:3]
            top = ", ".join(self.annotate(mzs[j]) or f"m/z {mzs[j]:.3f}" for j in order)
            rows.append((cl, int(m.sum()), top))
        rows.sort(key=lambda row: lin.get(row[0], str(row[0])))
        # cache per-cluster signatures so hovering a blob can show them without the table
        self._seg_sig = {cl: (npx, top) for cl, npx, top in rows}
        self._fill_seg_table(rows)
        if prev_checked:                                  # re-tick survivors after the refill
            live = {cl for cl, _, _ in rows}
            self._check_segments([c for c in prev_checked if c in live], additive=True)

    def _fill_seg_table(self, rows):
        t = self.seg_table
        t.blockSignals(True)
        t.clear()
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels(["✓", "cluster", "pixels", "top lipids (enriched)", "region"])
        set_header_tooltips(t, {
            "✓": "Tick to keep this segment highlighted and include it when you build/assign a region.",
            "cluster": "Cluster id / lineage path — '0·2' means sub-cluster 2 produced by drilling into cluster 0.",
            "pixels": "Number of tissue pixels assigned to this segment.",
            "top lipids (enriched)": "The 3 features most enriched in this segment vs the rest (its molecular signature).",
            "region": "The named region this segment currently belongs to ('—' = unassigned)."})
        t.setRowCount(len(rows))
        lin = getattr(self, "_seg_lineage", {})
        hidden = set(getattr(self, "_seg_hidden", ()))
        gray = QtGui.QBrush(QtGui.QColor(GUIDE_LINE))
        for r, (cl, npx, top) in enumerate(rows):
            label = lin.get(cl, str(cl))                  # lineage path, e.g. "0", "0·2", "0·2·1"
            depth = label.count("·")                      # indent branched (drilled-in) children
            disp = ("    " * depth) + ("› " if depth else "") + label
            is_hidden = cl in hidden
            chk = QtWidgets.QTableWidgetItem()            # tick box + cluster color swatch
            chk.setIcon(self._color_icon("#555555" if is_hidden else PALETTE[cl % len(PALETTE)]))
            chk.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                         | QtCore.Qt.ItemIsUserCheckable)
            chk.setCheckState(QtCore.Qt.Unchecked)
            chk.setData(QtCore.Qt.UserRole, int(cl))
            t.setItem(r, 0, chk)
            for c, val in enumerate(((disp + ("  (hidden)" if is_hidden else "")),
                                     f"{npx:,}", top), start=1):
                it = _SortItem(str(val))
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setData(QtCore.Qt.UserRole, int(cl))
                if is_hidden:
                    it.setForeground(gray)
                t.setItem(r, c, it)
            t.setItem(r, 4, self._seg_region_cell(cl, gray if is_hidden else None))
        t.resizeColumnsToContents()
        t.blockSignals(False)
        self._seg_bulk.refresh_count()          # rows were rebuilt with signals blocked

    def _seg_region_of(self, cl):
        """The named region a cluster currently belongs to (a cluster lives in at most
        one), or ``None`` if it is still unassigned."""
        return next((rg for rg in self.regions if cl in rg.get("segments", set())), None)

    def _seg_region_cell(self, cl, hidden_brush=None):
        """A 'region' table cell for cluster ``cl``: the owning region's colour swatch +
        name, or a muted '—' when unassigned — so assigned segments read at a glance and
        aren't accidentally re-added elsewhere."""
        reg = self._seg_region_of(cl)
        it = _SortItem(reg["name"] if reg else "—")
        it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
        it.setData(QtCore.Qt.UserRole, int(cl))
        if reg:
            it.setIcon(self._color_icon(reg["color"]))
        if hidden_brush is not None or not reg:
            it.setForeground(hidden_brush or QtGui.QBrush(QtGui.QColor(GUIDE_LINE)))
        return it

    def _refresh_seg_assignments(self):
        """Update only the seg table's 'region' column in place, so membership shows live
        as segments are assigned/removed without recomputing cluster signatures."""
        t = getattr(self, "seg_table", None)
        if t is None or t.columnCount() < 5:
            return
        hidden = set(getattr(self, "_seg_hidden", ()))
        t.blockSignals(True)
        for r in range(t.rowCount()):
            base = t.item(r, 1)
            if base is None:
                continue
            cl = int(base.data(QtCore.Qt.UserRole))
            t.setItem(r, 4, self._seg_region_cell(
                cl, QtGui.QBrush(QtGui.QColor(GUIDE_LINE)) if cl in hidden else None))
        t.blockSignals(False)

    def _selected_segments(self):
        segs = []
        for ix in self.seg_table.selectionModel().selectedRows():
            it = self.seg_table.item(ix.row(), 1)
            if it is not None:
                segs.append(int(it.data(QtCore.Qt.UserRole)))
        return segs

    def _checked_segments(self):
        segs = []
        for r in range(self.seg_table.rowCount()):
            it = self.seg_table.item(r, 0)
            if it is not None and it.checkState() == QtCore.Qt.Checked:
                segs.append(int(it.data(QtCore.Qt.UserRole)))
        return segs

    def _check_segments(self, segs, additive=False):
        """Tick the rows whose cluster id is in ``segs`` (clearing the rest unless
        ``additive``); ``segs=[]`` clears all ticks."""
        want = set(segs)
        t = self.seg_table
        t.blockSignals(True)
        for r in range(t.rowCount()):
            it = t.item(r, 0)
            if it is None:
                continue
            cl = int(it.data(QtCore.Qt.UserRole))
            if cl in want:
                it.setCheckState(QtCore.Qt.Checked)
            elif not additive:
                it.setCheckState(QtCore.Qt.Unchecked)
        t.blockSignals(False)
        self._seg_bulk.refresh_count()
        self._update_seg_preview()

    def _update_seg_preview(self):
        """Ticked segments stay shown ('pinned') and isolate the map; the row/blob under
        the cursor overlays on top transiently. With nothing ticked, a hover just brings
        that one segment forward in white over the resting colour map (no dimming, no
        pulse — like selecting a feature). Nothing ticked or hovered → the region map."""
        checked = self._checked_segments()
        hover = getattr(self, "_seg_hover_cluster", None)
        if checked:                                      # a deliberate tick takes over the view
            self.region_list.clearSelection()
            shown = list(checked)
            if hover is not None and hover not in shown:
                shown.append(hover)
            self._preview_segments(shown, hover=hover, pinned=checked)
        elif hover is not None:                          # rest + hover: pop the blob white on top
            self._preview_hover_on_base(hover)
        else:
            self._render_region_map()                    # resting view

    def _preview_hover_on_base(self, hover):
        """Resting-state hover highlight: keep the current colour map (region map, or the
        cluster palette when no regions exist) and bring just the hovered segment forward
        in solid white. No dimming and no pulse — the equivalent of selecting a feature.
        The canvas tooltip / resting legend already name it, so this only recolours the
        image (leaving the legend untouched avoids layout shift as the cursor moves)."""
        cmap = {cl: rg["color"] for rg in self.regions if rg.get("visible", True)
                for cl in rg.get("segments", ())}
        has_regions = bool(cmap)

        def color_for(cl):
            if cl == hover:
                return "#ffffff"
            return cmap.get(cl) if has_regions else PALETTE[cl % len(PALETTE)]

        self._set_seg_image(color_for, dim=has_regions)

    def _seg_item_changed(self, item):
        if item.column() == 0:                           # a tick box toggled
            self._update_seg_preview()

    def _cluster_at_scene(self, scene_pos):
        """Cluster id under a scene position on the seg canvas, or None if off-tissue."""
        if self.seg is None:
            return None
        p = self.seg_vb.mapSceneToView(scene_pos)
        r, c = int(p.y()), int(p.x())
        lab = self.seg.label_image
        if not (0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and np.isfinite(lab[r, c])):
            return None
        return int(lab[r, c])

    def _seg_image_clicked(self, ev):
        """Click a blob to tick its segment (shift-click to add/remove from the tick set),
        feeding the 'Add checked to…' region-grouping workflow."""
        cl = self._cluster_at_scene(ev.scenePos())
        if cl is None:
            return
        checked = set(self._checked_segments())
        if bool(ev.modifiers() & QtCore.Qt.ShiftModifier):       # add/remove this seg
            new = (checked - {cl}) if cl in checked else (checked | {cl})
        else:                                                    # plain click toggles just this seg
            new = set() if checked == {cl} else {cl}
        self._check_segments(list(new))

    def _seg_hover(self, scene_pos):
        """Hover a blob → bring it forward in white on the map (like selecting a feature)
        and show its molecular signature (top enriched lipids + size) as the canvas
        tooltip, so you never need the ID table to know what you're about to tick."""
        if getattr(self, "seg_view", None) is None:
            return
        cl = self._cluster_at_scene(scene_pos)
        sig = getattr(self, "_seg_sig", {})
        if cl is None or cl not in sig:
            self.seg_view.setToolTip("")
            self._set_seg_hover(None)
            return
        npx, top = sig[cl]
        owner = next((rg["name"] for rg in self.regions if cl in rg["segments"]), None)
        tip = f"segment {cl} · {npx:,} px\n{top}"
        if owner:
            tip += f"\n→ in region: {owner}"
        self.seg_view.setToolTip(tip)
        self._set_seg_hover(cl)                           # pop the hovered blob white on the map

    # ----- hover-to-peek: overlay the row under the cursor on the map ------- #
    def _seg_row_entered(self, row, _col):
        """Hovering a row in the segment table overlays that segment on the map — a
        transient peek on top of the ticked (pinned) ones."""
        it = self.seg_table.item(row, 1)
        self._set_seg_hover(int(it.data(QtCore.Qt.UserRole)) if it is not None else None)

    def _seg_current_changed(self, row, _col, _prev_row, _prev_col):
        """Arrow-key (or click) navigation peeks the current row, mirroring mouse hover
        so keyboard users get the same live preview as they step through the list."""
        it = self.seg_table.item(row, 1) if row >= 0 else None
        self._set_seg_hover(int(it.data(QtCore.Qt.UserRole)) if it is not None else None)

    def _set_seg_hover(self, cl):
        """Set the transiently-overlaid segment (``None`` clears it). Re-renders only
        when the hovered cluster changes, so it stays cheap as the mouse moves."""
        if getattr(self, "_seg_hover_cluster", None) == cl:
            return
        self._seg_hover_cluster = cl
        self._update_seg_preview()

    def eventFilter(self, obj, event):
        # drop the hover-peek overlay once the pointer leaves the table — but keep it
        # while the table has keyboard focus, since arrow-key nav drives the same peek
        # (so nudging the mouse away mid-keyboarding doesn't kill the preview). Clear it
        # outright when focus leaves the table, so a keyboard peek doesn't linger.
        tbl = getattr(self, "seg_table", None)
        if tbl is not None:
            if (event.type() == QtCore.QEvent.Leave
                    and obj is tbl.viewport() and not tbl.hasFocus()):
                self._set_seg_hover(None)
            elif event.type() == QtCore.QEvent.FocusOut and obj is tbl:
                self._set_seg_hover(None)
        sv = getattr(self, "seg_view", None)               # leaving the canvas drops the peek too
        if sv is not None and event.type() == QtCore.QEvent.Leave and obj is sv.viewport():
            self._set_seg_hover(None)
        return super().eventFilter(obj, event)

    def _selected_region_index(self):
        # After _refresh_region_list() rebuilds the list, Qt keeps the highlight (restored
        # via setSelected) but resets currentRow to -1. Fall back to the first *selected*
        # row so "Find peaks (this region)" / rename / delete still see the highlighted
        # region — otherwise they act as if nothing is selected (e.g. spawning a stray ROI).
        row = self.region_list.currentRow()
        if 0 <= row < len(self.regions):
            return row
        sel = self._selected_region_indices()
        return sel[0] if sel else -1

    def _target_segments(self):
        """Segments to act on: ticked ones if any, else the highlighted/selected rows.

        Users often *select* rows rather than ticking the checkboxes; falling back to
        the selection makes assignment work either way."""
        return self._checked_segments() or self._selected_segments()

    def _assign_segments_to_region_index(self, seg_ids, ri):
        """Assign ``seg_ids`` to region ``ri`` (a segment belongs to one region at a
        time, so it is removed from every other region first). Refreshes the views,
        clears ticks, and reports status. No-op for an empty list / bad index."""
        segs = list(seg_ids)
        if not segs or not (0 <= ri < len(self.regions)):
            return
        self.record_undo("assign segments", domains=("regions",))
        for j, rg in enumerate(self.regions):            # a segment lives in one region at a time
            if j != ri:
                rg["segments"].difference_update(segs)
        self.regions[ri]["segments"].update(segs)
        self._check_segments([])                         # clear ticks once assigned
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        self._refresh_seg_assignments()                  # show membership in the seg table
        self.statusBar().showMessage(
            f"Added segment(s) {', '.join(map(str, sorted(segs)))} to {self.regions[ri]['name']}.")

    def _region_new(self):
        if self.seg is None:
            self.statusBar().showMessage("Run segmentation first to group clusters into a region "
                                         "(or draw an ROI → 'Add ROI').")
            return
        self.record_undo("new region", domains=("regions",))
        self._new_region(f"Region {len(self.regions) + 1}")
        self.statusBar().showMessage("New cluster region — tick segments then 'Add checked to…'.")

    def _region_rename(self):
        ri = self._selected_region_index()
        if ri < 0:
            return
        old = self.regions[ri]["name"]
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename region", "Name:", text=old)
        if ok and name.strip():
            self.record_undo("rename region", domains=("regions",))
            new = self._unique_region_name(name.strip())
            self.regions[ri]["name"] = new
            for rg in self.regions:                       # keep sub-region links valid
                if rg.get("parent") == old:
                    rg["parent"] = new
            self._rekey_region_scope(old, new)            # the region's feature list follows its name
            self._refresh_region_list()
            self._sync_region_combos()

    def _rekey_region_scope(self, old, new):
        """A region's per-sample feature list is keyed by the region *name* (see
        ion._on_peaks). Renaming the region must move that list to the new key, else it
        orphans under the old name in the Feature-set selector ("ROI names don't persist").
        Pass ``new=None`` when the region is deleted to drop the list. Keeps
        ``_active_feature_scope`` / ``_flist_name`` and the per-peak 'region' tags in sync,
        then rebuilds the selector + consumer labels."""
        scopes = getattr(self, "_feature_scopes", None)
        if scopes is not None and old in scopes:
            if new is None:
                scopes.pop(old, None)
            else:
                peaks = scopes.pop(old)
                for p in peaks:
                    if p.get("region") == old:
                        p["region"] = new
                scopes[new] = peaks
        if getattr(self, "_active_feature_scope", None) == old:
            self._active_feature_scope = new
            if new is not None:
                self._flist_name = f"{new} (sample)"
        if hasattr(self, "_refresh_feature_set_combo"):
            self._refresh_feature_set_combo()             # selector labels follow the rename
        if hasattr(self, "_sync_feature_consumers"):
            self._sync_feature_consumers()

    def _region_batch_rename(self, ris=None):
        """Add a prefix and/or suffix to several regions at once (ImageJ-style) — the fast
        way to label a batch like 'donor3_<name>'. Feature lists follow each rename."""
        ris = ris or self._selected_region_indices()
        if not ris:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Batch rename {len(ris)} regions")
        form = QtWidgets.QFormLayout(dlg)
        form.addRow(QtWidgets.QLabel(
            f"Add a prefix and/or suffix to {len(ris)} selected region name(s)."))
        pre = QtWidgets.QLineEdit()
        suf = QtWidgets.QLineEdit()
        form.addRow("Prefix:", pre)
        form.addRow("Suffix:", suf)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._apply_region_prefix_suffix(ris, pre.text(), suf.text())

    def _apply_region_prefix_suffix(self, ris, prefix, suffix):
        if not ris or (not prefix and not suffix):
            return
        self.record_undo("batch rename regions", domains=("regions",))
        for ri in ris:
            old = self.regions[ri]["name"]
            new = self._unique_region_name(f"{prefix}{old}{suffix}")
            if new == old:
                continue
            self.regions[ri]["name"] = new
            for rg in self.regions:                       # keep sub-region links valid
                if rg.get("parent") == old:
                    rg["parent"] = new
            self._rekey_region_scope(old, new)            # the region's feature list follows
        self._refresh_regions_all()
        self.statusBar().showMessage(f"Renamed {len(ris)} region(s).")

    def _region_batch_tag(self, ris=None):
        """Tag several regions with a group (A/B/…) in one step — the tag the A/B picker's
        one-click 'select Group A/B' reads, so you can set up an A-vs-B comparison fast."""
        ris = ris or self._selected_region_indices()
        if not ris:
            return
        existing = [g for g in dict.fromkeys((rg.get("group") or "").strip()
                                             for rg in self.regions) if g]
        labels = list(dict.fromkeys(["Group A", "Group B"] + existing)) + ["(clear)"]
        label, ok = QtWidgets.QInputDialog.getItem(
            self, "Tag regions with group",
            f"Assign a group to {len(ris)} region(s) — the A/B picker can then select a whole "
            "group in one click:", labels, 0, True)
        if not ok or not label:
            return
        self._apply_region_group(ris, "" if label == "(clear)" else label.strip())

    def _apply_region_group(self, ris, group):
        if not ris:
            return
        self.record_undo("tag regions", domains=("regions",))
        for ri in ris:
            self.regions[ri]["group"] = group
        self._refresh_regions_all()
        self.statusBar().showMessage(
            f"Tagged {len(ris)} region(s) as '{group}'." if group
            else f"Cleared the group on {len(ris)} region(s).")

    def _region_delete(self):
        ris = self._selected_region_indices() or (
            [self._selected_region_index()] if self._selected_region_index() >= 0 else [])
        gone = {self.regions[i]["name"] for i in ris if 0 <= i < len(self.regions)}
        if not gone:
            return
        n = len(gone)
        label = next(iter(gone)) if n == 1 else f'{n} regions'
        if not confirm(self, 'Delete region' + ('' if n == 1 else 's'),
                       f"Delete {label}? Their cluster groupings and any per-region feature "
                       "lists will be removed (sub-regions become top-level). This is "
                       "undoable with ⌘Z.", ok_text='Delete'):
            return
        self.record_undo("delete region", domains=("regions",))
        self.regions = [rg for rg in self.regions if rg["name"] not in gone]
        for rg in self.regions:                           # orphaned sub-regions become top-level
            if rg.get("parent") in gone:
                rg["parent"] = None
        for name in gone:
            self._rekey_region_scope(name, None)          # drop its now-orphaned feature list
        self.region_list.clearSelection()                 # drop the stale row→region selection
        self._hide_region_overlay()                       # and its on-image footprint
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        self._refresh_seg_assignments()                   # freed clusters show as unassigned again

    def _rebuild_assign_menu(self):
        """Populate the 'Add to region ▾' popup with the current regions (each with its
        colour swatch) plus a 'New region…' entry. The target segments are read when an
        item is clicked, so the menu never goes stale."""
        menu = getattr(self, "assign_menu", None)
        if menu is None:
            return
        menu.clear()
        for ri, rg in enumerate(self.regions):
            act = menu.addAction(self._color_icon(rg["color"]), rg["name"])
            act.triggered.connect(lambda _=False, ri=ri: self._assign_target_to_region(ri))
        if not self.regions:
            none = menu.addAction("(no regions yet)")
            none.setEnabled(False)
        menu.addSeparator()
        nr = menu.addAction(icon("add"), "New region from checked")
        nr.triggered.connect(self._new_region_from_target)

    def _new_region_from_target(self):
        """Create a brand-new region from the target segments (ticked, else selected)."""
        segs = self._target_segments()
        if not segs:
            self.statusBar().showMessage("Tick or select one or more segments first.")
            return
        self._new_region_from_segments(segs)

    def _assign_target_to_region(self, ri):
        """Add the target segments (ticked, else selected) to existing region ``ri``."""
        if not (0 <= ri < len(self.regions)):
            return
        segs = self._target_segments()
        if not segs:
            self.statusBar().showMessage("Tick or select one or more segments first.")
            return
        self._assign_segments_to_region_index(segs, ri)

    def _remove_target(self, segs=None):
        """Remove ``segs`` (default: target segments) from whatever region they are in."""
        segs = set(self._target_segments() if segs is None else segs)
        if not segs:
            self.statusBar().showMessage("Tick or select one or more segments first.")
            return
        self.record_undo("remove segments", domains=("regions",))
        for rg in self.regions:
            rg["segments"].difference_update(segs)
        self._check_segments([])
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        self._refresh_seg_assignments()
        self.statusBar().showMessage(
            f"Removed segment(s) {', '.join(map(str, sorted(segs)))} from their region(s).")

    def _remove_checked(self):
        self._remove_target()

    # ----- right-click context menu on the segment table ------------------- #
    def _seg_table_menu(self, pos):
        """Context menu over the target segments (ticked, else selected)."""
        if self.seg is None:
            return
        segs = self._target_segments()
        menu = QtWidgets.QMenu(self.seg_table)
        if segs:
            ids = ", ".join(map(str, sorted(segs)))
            menu.addAction(f"Target: segment(s) {ids}").setEnabled(False)
            menu.addSeparator()
            add_menu = menu.addMenu("Add to region ▸")
            for ri, rg in enumerate(self.regions):
                act = add_menu.addAction(self._color_icon(rg["color"]), rg["name"])
                act.triggered.connect(
                    lambda _=False, segs=list(segs), ri=ri:
                    self._assign_segments_to_region_index(segs, ri))
            if self.regions:
                add_menu.addSeparator()
            new_in_sub = add_menu.addAction(icon("add"), "New region…")
            new_in_sub.triggered.connect(
                lambda _=False, segs=list(segs): self._new_region_from_segments(segs))
            rm_act = menu.addAction(icon("remove"), "Remove from region")
            rm_act.setEnabled(any(s in rg["segments"] for rg in self.regions for s in segs))
            rm_act.triggered.connect(
                lambda _=False, segs=list(segs): self._remove_target(segs))
            menu.addSeparator()
            nfs = menu.addAction(icon("add"), "New region from selection")
            nfs.triggered.connect(
                lambda _=False, segs=list(segs): self._new_region_from_segments(segs))
            menu.addSeparator()
            split_menu = menu.addMenu("Increase detail here ▸")   # local drill-in
            split_menu.setIcon(icon("expand"))
            for kk in (2, 3, 4):
                act = split_menu.addAction(f"{kk} sub-clusters")
                act.triggered.connect(
                    lambda _=False, kk=kk, segs=list(segs): self._split_selected(kk, segs))
            if len(segs) >= 2:                                    # inverse of drill-in
                mg = menu.addAction(icon("merge"), "Merge into one segment")
                mg.triggered.connect(
                    lambda _=False, segs=list(segs): self._merge_selected(segs))
            menu.addSeparator()
            hl = menu.addAction(icon("find"), "Highlight (isolate)")
            hl.triggered.connect(lambda _=False, segs=list(segs): self._preview_segments(segs))
            hidden = set(getattr(self, "_seg_hidden", ()))
            if any(s not in hidden for s in segs):
                a_hide = menu.addAction(icon("remove"), "Hide")
                a_hide.triggered.connect(
                    lambda _=False, segs=list(segs): self._hide_segments(segs, True))
            if any(s in hidden for s in segs):
                a_show = menu.addAction(icon("find"), "Show")
                a_show.triggered.connect(
                    lambda _=False, segs=list(segs): self._hide_segments(segs, False))
        else:
            menu.addAction("Tick or select segment(s) first").setEnabled(False)
        menu.exec(self.seg_table.viewport().mapToGlobal(pos))

    def _regions_from_all_segments(self):
        """Turn every segment at the current detail (the tree's terminal branches) into its
        own named region in one step: prompt for a base name, then create '<base>_01',
        '<base>_02', … — one per live segment. Each segment lands in exactly one of the new
        regions (it is first cleared from any region it was already in)."""
        if self.seg is None:
            self.statusBar().showMessage("Run segmentation first to split the tree into regions.")
            return
        seg_ids = [cl for cl in range(self.seg.n_clusters) if self.seg.mask(cl).any()]
        if not seg_ids:
            self.statusBar().showMessage("No segments to turn into regions.")
            return
        base, ok = QtWidgets.QInputDialog.getText(
            self, "Regions from all segments",
            f"Create {len(seg_ids)} regions (one per segment), named '<name>_01', '<name>_02', …:",
            text="Region")
        base = base.strip() if ok else ""
        if not base:
            return
        self.record_undo("regions from all segments", domains=("regions",))
        for rg in self.regions:                          # a segment lives in one region at a time
            rg["segments"].difference_update(seg_ids)
        for n, cl in enumerate(seg_ids, start=1):
            self._new_region(name=f"{base}_{n:02d}", segments=[cl], select=False, refresh=False)
        self._check_segments([])
        self._refresh_regions_all()                      # one rebuild for the whole batch
        self._refresh_seg_assignments()                  # show membership in the seg table
        self.statusBar().showMessage(
            f"Created {len(seg_ids)} regions from all segments ({base}_01 … "
            f"{base}_{len(seg_ids):02d}).")

    def _new_region_from_segments(self, segs):
        """Create a fresh region (via the existing flow) then assign ``segs`` to it."""
        segs = list(segs)
        if not segs:
            return
        self._region_new()                               # appends + selects the new region
        if self.regions:
            self._assign_segments_to_region_index(segs, len(self.regions) - 1)

    # ----- hide / show / reset --------------------------------------------- #
    def _hide_segments(self, segs, hide=True):
        self.record_undo("hide/show segments", domains=("seg",))
        hidden = set(getattr(self, "_seg_hidden", set()))
        hidden = (hidden | set(segs)) if hide else (hidden - set(segs))
        self._seg_hidden = hidden
        self._check_segments([])                          # clear ticks so the view is clean
        self._rebuild_seg_table()
        self._render_region_map()
        ids = ", ".join(map(str, sorted(segs)))
        self.statusBar().showMessage(f"{'Hid' if hide else 'Showed'} segment(s) {ids}.")

    def _reset_splits(self):
        """Discard every drill-down split and return to the original segmentation."""
        orig = getattr(self, "_seg_original", None)
        if self.seg is None or orig is None:
            return
        self.record_undo("reset splits", domains=("seg", "regions"))
        labels0, k0, ev, sil = orig
        self.seg.labels = labels0.copy()
        self.seg.n_clusters = k0
        self.seg.label_image = self.ds.to_image(np.where(labels0 < 0, np.nan, labels0.astype(float)))
        self.seg.explained_variance = ev
        self.seg.silhouette = sil
        self._seg_lineage = {cl: str(cl) for cl in range(k0)}
        self._seg_hidden = set()
        for rg in self.regions:                           # drop members that were split children
            rg["segments"] = {s for s in rg["segments"] if s < k0}
        self._rebuild_seg_table()
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        self.statusBar().showMessage(f"Reset to the original {k0} segments.")

    def _reset_splits_confirm(self):
        if getattr(self, '_seg_original', None) is None or self.seg is None:
            return
        if confirm(self, 'Reset segmentation',
                   'Discard every drill-in split and merge and return to the original '
                   'segmentation at this detail? Cluster-region memberships that referenced '
                   'split/merged sub-clusters will be pruned.', ok_text='Reset'):
            self._reset_splits()

    def _refresh_seg_run_state(self):
        b = getattr(self, 'b_seg', None)
        if b is not None:
            ok = self.ds is not None and bool(getattr(self, 'peaks', None))
            b.setEnabled(ok)
            # Say *why* it's disabled instead of leaving a dead grey button.
            b.setToolTip("" if ok else "Find peaks first (Ion image ▸ Find peaks) — "
                         "segmentation groups pixels by their peak intensities.")
        bm = getattr(self, 'b_seg_merge', None)
        if bm is not None:
            bm.setEnabled(self.seg is not None and getattr(self.seg, 'n_clusters', 0) >= 2)
        bar = getattr(self, 'b_seg_all_regions', None)
        if bar is not None:
            bar.setEnabled(self.seg is not None and getattr(self.seg, 'n_clusters', 0) >= 1)

    def _refresh_region_list(self):
        lst = self.region_list
        # capture selection by name *before* reordering (rows still match the old order)
        sel_names = ({self.regions[r]["name"] for r in self._selected_region_indices()}
                     if self.regions else set())
        # remember scroll position too — clear()+rebuild resets it, which otherwise snaps a
        # long list back to the top on every edit (e.g. renaming a region far down the list)
        scroll = lst.verticalScrollBar().value()
        self._reorder_regions()                           # sub-regions directly under parents
        lst.blockSignals(True)
        lst.clear()
        gray = QtGui.QBrush(QtGui.QColor(GUIDE_LINE))
        by_name = {rg["name"]: rg for rg in self.regions}
        for ridx, rg in enumerate(self.regions):
            mask = self._region_pixel_mask(rg)
            npx = int(mask.sum()) if mask is not None else 0
            if rg.get("mask") is not None:                # ROI / sample region
                detail = "sub-ROI" if rg.get("parent") else "ROI"
            else:                                         # cluster-based region
                detail = "segs " + (", ".join(map(str, sorted(rg.get("segments", set())))) or "—")
            # walk the parent chain so each nesting level adds another indent step
            # (sub-sub-regions sit deeper than sub-regions); guard against cycles.
            depth, cur, seen = 0, rg, set()
            while cur.get("parent") and cur["parent"] not in seen and depth < 16:
                seen.add(cur["name"])
                cur = by_name.get(cur["parent"])
                if cur is None:
                    break
                depth += 1
            indent = ("    " * depth + "› ") if depth else ""
            visible = rg.get("visible", True)
            text = f"{indent}{rg['name']}  —  {detail} · {npx:,} px"
            grp = (rg.get("group") or "").strip()
            if grp:
                text += f"  ·  {grp}"                      # show the A/B group tag inline
            if not visible:
                text += "  (hidden)"
            # drawn eye / eye-with-slash (region colour when visible) — click it to toggle
            item = QtWidgets.QListWidgetItem(eye_icon(rg["color"], visible), text)
            item.setData(QtCore.Qt.UserRole, ridx)        # row→region index for drag-reorder sync
            if not visible:
                item.setForeground(gray)
            lst.addItem(item)
        for r in range(lst.count()):                      # restore selection by name (multi-select)
            if self.regions[r]["name"] in sel_names:
                lst.item(r).setSelected(True)
        lst.blockSignals(False)
        lst.verticalScrollBar().setValue(scroll)          # keep the view where it was (no snap-to-top)
        self._apply_region_filter()                       # re-hide rows that don't match the filter
        # the list rebuilt with signals blocked, so sync the on-image footprint overlay to
        # the (possibly changed) selection — else a just-deleted region keeps showing.
        sel = self._selected_region_indices()
        if sel:
            self._show_region_overlay(sel)
        else:
            self._hide_region_overlay()

    def _apply_region_filter(self, *_):
        """Hide region rows that don't match the filter box (by name + detail). Cheap —
        toggles row visibility in place, no rebuild — so it can run on every keystroke.
        Rows map 1:1 to ``self.regions`` after :meth:`_reorder_regions`."""
        lst = getattr(self, "region_list", None)
        if lst is None:
            return
        box = getattr(self, "region_filter", None)
        q = box.text().strip().lower() if box is not None else ""
        any_hidden = False
        for r in range(lst.count()):
            if r >= len(self.regions):
                continue
            rg = self.regions[r]
            hay = (rg.get("name", ""), rg.get("group", ""),
                   "sub-roi" if rg.get("parent") else ("roi" if rg.get("mask") is not None else "segs"))
            hide = (bool(q) and not any(q in str(s).lower() for s in hay)) \
                or not self._region_in_scope(rg)          # off-slide region, 'All samples' off
            lst.setRowHidden(r, hide)
            any_hidden = any_hidden or hide
        # Drag-reorder over a part-hidden list is ambiguous, so disable dragging whenever any
        # row is hidden (text filter OR the active-sample scope); re-enable once all rows show.
        lst.setDragEnabled(not any_hidden)

    def _region_list_selection_changed(self):
        if not self.region_list.selectedItems():      # selection cleared (incl. programmatic)
            self._clear_region_spectra()              # drop region traces; no seg<->region loop
            self._hide_region_overlay()
            return
        ris = self._selected_region_indices()
        if ris:
            self.seg_table.clearSelection()
            self._preview_regions(ris)
            self._show_region_overlay(ris)            # draw its footprint on the ion image
            # Selecting a region no longer recomputes its mean spectrum — that's the
            # disk/cube load people felt on every click while organising regions. Drop
            # any previously shown region trace so it doesn't linger out of sync; the
            # spectrum is computed on demand via the 'Show spectra' action.
            self._clear_region_spectra()
            if len(ris) == 1:
                # Selecting a region that has its own picked feature list activates that
                # list (opted-in behaviour) — switch the working set to its scope so every
                # pipeline follows the region you clicked. Regions with no list of their
                # own leave the current set untouched.
                name = self.regions[ris[0]]["name"]
                if name in getattr(self, "_feature_scopes", {}) and name != self._active_feature_scope:
                    self._switch_feature_scope(name)
                    self.statusBar().showMessage(f"Switched to '{name}' features.")

    def _region_on_active_slide(self, rg):
        """True when a region belongs on the current slide — it carries no ``sample`` tag
        (legacy session or a region just drawn before the field existed) or its tag matches
        the active dataset's source. Regions saved on another slide never paint on this
        tissue, so a leftover from a previous sample can't render over the wrong image."""
        tag = rg.get("sample") or ""
        return (not tag) or tag == (self.ds.source if self.ds is not None else "")

    def _region_show_all_samples(self):
        """The 'All samples' toggle's state — True when the region panel should show every
        loaded sample's regions. Defaults to True when the checkbox isn't built yet (or a
        headless caller lacks it) so nothing is hidden unexpectedly."""
        cb = getattr(self, "region_all_samples", None)
        return bool(cb.isChecked()) if cb is not None else True

    def _region_in_scope(self, rg):
        """Region passes the panel's sample scope: either 'All samples' is on, or the region
        belongs to the active slide. Drives the region list filter and the A/B / 'within'
        combos so they default to just this slide's regions (the region↔slide tie made
        visible)."""
        return self._region_show_all_samples() or self._region_on_active_slide(rg)

    def _on_region_scope_toggled(self, *_):
        """'All samples' flipped — re-hide/show region rows and rebuild the A/B + 'within'
        combos and the Find-peaks region selector so they all honour the new scope."""
        self._apply_region_filter()
        self._sync_region_combos()
        self._sync_pick_region_list()

    def _show_region_overlay(self, ris):
        """Paint the selected region(s) onto the ion image in their own colors, so picking
        a region in the sidebar shows where it sits (its drawn ROI / cluster footprint)."""
        item = getattr(self, "_region_overlay", None)
        if item is None or self.ds is None:
            return
        self._overlay_ris = list(ris)        # remembered so a rotate can re-render the same footprints
        h, w = self.ds.height, self.ds.width
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        drew = False
        for ri in ris:
            rg = self.regions[ri]
            if not self._region_on_active_slide(rg):      # never paint another slide's region
                continue
            m = self._region_pixel_mask(rg)
            if m is None or not m.any():
                continue
            on = self.ds.to_image(m.astype(float), fill=0.0) > 0.5
            col = QtGui.QColor(rg["color"])
            rgba[on, 0], rgba[on, 1], rgba[on, 2], rgba[on, 3] = col.red(), col.green(), col.blue(), 120
            drew = True
        if drew:
            item.setImage(rgba)
            item.show()
        else:
            item.hide()

    def _hide_region_overlay(self):
        self._overlay_ris = []
        item = getattr(self, "_region_overlay", None)
        if item is not None:
            item.hide()

    def _sync_seg_region_combo(self):
        """Keep the Segmentation-tab 'Within' selector in step with the region list:
        'Whole slide' plus one row per named region. Preserves the current pick by name
        across rebuilds (a vanished region falls back to whole slide)."""
        combo = getattr(self, "seg_region", None)
        if combo is None:
            return
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Whole slide", None)
        for rg in self.regions:
            if not self._region_in_scope(rg):            # default to the active slide's regions
                continue
            try:
                combo.addItem(self._color_icon(rg["color"]), rg["name"], rg["name"])
            except Exception:                            # icon is cosmetic
                combo.addItem(rg["name"], rg["name"])
        idx = combo.findData(cur) if cur is not None else 0
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _sync_region_combos(self):
        self._sync_seg_region_combo()
        if hasattr(self, "_sync_coloc_region_combo"):
            self._sync_coloc_region_combo()
        names = [rg["name"] for rg in self.regions if self._region_in_scope(rg)]
        # stats-tab A/B + the Region comparison tab's own A/B + the Lipid-classes dialog's
        # A/B — all optional (the stats + compare tabs were retired in plan 24 Phase 4), so
        # guard each and only sync the ones that exist.
        pairs = []
        for an, bn in (("region_a_combo", "region_b_combo"),
                       ("cmp_region_a", "cmp_region_b"), ("cc_region_a", "cc_region_b")):
            ca, cb = getattr(self, an, None), getattr(self, bn, None)
            if ca is not None and cb is not None:
                pairs.append((ca, cb))
        for a_combo, b_combo in pairs:
            for combo, default in ((a_combo, 0), (b_combo, 1)):
                cur = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(names)
                if cur in names:
                    combo.setCurrentText(cur)
                elif default < len(names):
                    combo.setCurrentIndex(default)
                combo.blockSignals(False)
            # default B to a different region than A so a fresh A-vs-B is meaningful
            # (adding regions one at a time can otherwise leave both pointing at the first)
            if len(names) >= 2 and a_combo.currentText() == b_combo.currentText():
                other = next((n for n in names if n != a_combo.currentText()), None)
                if other:
                    b_combo.blockSignals(True)
                    b_combo.setCurrentText(other)
                    b_combo.blockSignals(False)
            # hand each picker the regions' group tags so it can offer one-click
            # 'select all of Group A/B' (regions carry the cohort group set in the Flow)
            group_of = {rg["name"]: (rg.get("group") or "") for rg in self.regions}
            for combo in (a_combo, b_combo):
                if hasattr(combo, "set_item_groups"):
                    combo.set_item_groups(group_of)
        # Keep the stats-tab 'Group by' selector honest about what's available: grey out
        # the option that has no data, and default the per-group tests to the user's named
        # regions whenever ≥2 exist (falling back to the segmentation only when they don't).
        # Running segmentation clears the regions, which briefly makes 'Named regions'
        # unavailable; we must switch BACK to it once they're rebuilt — otherwise the combo
        # got stranded on 'Segmentation clusters' and silently ran the ROIs' analysis on
        # clusters (the reported bug). An *explicit* user pick (current ≠ the value we last
        # auto-set) is respected until that pick itself becomes unavailable.
        gb = getattr(self, "group_by_combo", None)
        if gb is not None:
            avail = {"Segmentation clusters": self.seg is not None,
                     "Named regions": len(names) >= 2}

            def _preferred():
                if avail["Named regions"]:
                    return "Named regions"            # the user's ROIs/curated anatomy win
                if avail["Segmentation clusters"]:
                    return "Segmentation clusters"
                return gb.currentText()
            gb.blockSignals(True)
            model = gb.model()
            for i in range(gb.count()):
                model.item(i).setEnabled(avail.get(gb.itemText(i), True))
            last_auto = getattr(self, "_group_by_auto_value", None)
            user_chose = last_auto is not None and gb.currentText() != last_auto
            if not user_chose or not avail.get(gb.currentText(), True):
                target = _preferred()
                if gb.currentText() != target:
                    gb.setCurrentText(target)
                self._group_by_auto_value = target    # remember what *we* set, to spot user edits
            gb.blockSignals(False)
        self._sync_pick_region_list()                     # Find peaks 'Region(s)' selector
        self._rebuild_assign_menu()                       # 'Add to region ▾' popup
        self._refresh_scope_bars()                        # group readouts track region/seg changes
        if hasattr(self, "_shap_refresh_groupings"):      # keep the SHAP 'Group by' picker live
            self._shap_refresh_groupings()                # (else it stays greyed until a mode toggle)
            self._shap_update_region_btn()
        if hasattr(self, "_refresh_action_states"):
            self._refresh_action_states()

    def _sync_pick_region_list(self):
        """Keep the Find-peaks 'Region(s)' selector in step with the region list: one
        checkable row per named region (with its colour swatch). Nothing ticked = whole
        slide. Built before any region exists, so guard and preserve the current ticks by
        name across rebuilds."""
        lw = getattr(self, "pick_region_list", None)
        if lw is None:
            return
        in_scope = [rg for rg in self.regions if self._region_in_scope(rg)]  # active slide's
        colors = {rg["name"]: rg["color"] for rg in in_scope}
        lw.set_icon_for(lambda n: self._color_icon(colors[n]))
        lw.set_entries([(rg["name"], rg["name"]) for rg in in_scope])

    def do_segment(self):
        """Build one granularity tree over the tissue (heavy, once) using the chosen
        algorithm / distance / region. The Detail slider then cuts it live — no up-front
        k, no manual over-segment-then-merge loop."""
        if not self.peaks:
            self.statusBar().showMessage("Find peaks first.")
            return
        mzs = self._scope_mzs(self.seg_scope)             # bar's subset, else every visible feature
        method = self.seg_method.currentData() or "bisecting"
        metric = self.seg_metric.currentData() or "euclidean"
        # 'Within': segment only inside a chosen region's pixels, independently of the rest
        # of the slide. Resolve the mask now — a cluster-backed region survives the
        # re-segmentation that _on_seg would otherwise clear, because the closure holds it.
        rname = self.seg_region.currentData() if getattr(self, "seg_region", None) else None
        mask, scope_msg = None, "the whole slide"
        if rname is not None:
            rg = next((r for r in self.regions if r["name"] == rname), None)
            m = self._region_pixel_mask(rg) if rg is not None else None
            if m is None or not np.asarray(m, bool).any():
                self.statusBar().showMessage(
                    f"'{rname}' has no pixels yet — segmenting the whole slide instead.")
            else:
                mask = np.asarray(m, bool)
                scope_msg = f"region '{rname}' ({int(mask.sum()):,} px)"
        ds, ppm, norm = self.ds, self.ppm, self.norm
        self._seg_method, self._seg_metric = method, metric   # for drill-in + provenance
        self._seg_scoped = mask is not None                   # ran within one region?
        self._seg_scope_region = rname if mask is not None else None   # ROI scope for the audit
        self._run(lambda: spatial.hierarchy(ds, mzs, tol_ppm=ppm, norm=norm,
                                            method=method, metric=metric, pixel_mask=mask,
                                            random_state=profiles.active_seed()),
                  on_done=self._on_hier, modal=True, title="Building segmentation…",
                  busy=f"Building the segmentation tree over {scope_msg}… this can take a minute.")

    def _on_hier(self, hier):
        """A fresh tree: size the Detail slider to it and cut at the current detail."""
        self.hier = hier
        # Slider ceiling = min(user-set Max, the tree's own cuttable size). The tree supports
        # up to n_micro (~200); the user's "Max" spinbox sets how fine they want to go.
        hi = int(min(segment_detail_max(), max(2, hier.max_clusters)))
        self.detail_slider.blockSignals(True)
        self.detail_slider.setRange(2, hi)
        k0 = int(min(max(2, segment_default_count()), hi))   # a fresh build starts at the default
        self.detail_slider.setValue(k0)
        self.detail_slider.setEnabled(True)
        self.detail_slider.blockSignals(False)
        self._detail_dragging = False
        self.seg_dendro.set_hierarchy(hier, maxk=hi)       # draw the tree the slider cuts
        self._dendro_branch_micros = frozenset()           # a fresh tree drops any lit branches
        seg = spatial.segmentation_at(self.ds, hier, k0, with_silhouette=True)
        self._on_seg(seg, record=True)                     # fresh cut: resets cluster regions
        self.seg_vb.autoRange()
        self.statusBar().showMessage(
            f"Tree built — {seg.n_clusters} segments. Drag Detail (coarse↔fine) to re-segment "
            "live, then click blobs or tick rows to build regions.")

    # ----- Detail slider: cut the prebuilt tree live ----------------------- #
    def _detail_default_changed(self, n):
        """User changed the default starting number of segments. Persist it and, if a tree is
        live, apply it now (the slider keeps exploring freely without disturbing this default;
        a fresh segmentation also starts here — see :meth:`_on_hier`)."""
        n = max(2, int(n))
        prefs.set(SEGMENT_COUNT_KEY, n)
        if getattr(self, "hier", None) is None:
            return                                   # no tree yet: applies on the next build
        k = int(min(n, self.detail_slider.maximum()))
        if k == self.detail_slider.value():
            return
        self._detail_dragging = False
        self.detail_slider.setValue(k)               # fires _detail_changed → _commit_cut

    def _detail_max_changed(self, hi):
        """User changed how fine the Detail slider may cut. Persist the ceiling and re-apply
        it to the live slider + dendrogram, clamped to what the current tree supports. With
        no tree yet it just persists, taking effect on the next build (see :meth:`_on_hier`)."""
        hi = max(2, int(hi))
        prefs.set(SEGMENT_DETAIL_MAX_KEY, hi)
        hier = getattr(self, "hier", None)
        if hier is None:
            return
        eff = int(min(hi, max(2, hier.max_clusters)))
        if eff == self.detail_slider.maximum():
            return                                   # tree's own size already binds — no change
        k = int(min(self.detail_slider.value(), eff))
        self.detail_slider.blockSignals(True)
        self.detail_slider.setRange(2, eff)
        self.detail_slider.setValue(k)
        self.detail_slider.blockSignals(False)
        self.seg_dendro.set_hierarchy(hier, maxk=eff)    # redraw the tree to the new depth
        self._sync_dendrogram(k)                         # place the cut line on it
        self._commit_cut(k)                              # re-render the seg view at the cut

    def _detail_changed(self, value):
        """Slider moved: while dragging just recolour the canvas (cheap); a keyboard/
        click change (no drag) commits straight away."""
        if getattr(self, "hier", None) is None:
            return
        self.detail_lbl.setText(f"{value} segments")
        if getattr(self, "_detail_dragging", False):
            self._preview_cut(value)
        else:
            self._commit_cut(value)

    def _detail_released(self):
        self._detail_dragging = False
        self._commit_cut(self.detail_slider.value())

    def _dendro_cut_changed(self, k):
        """The user dragged the dendrogram's cut line: route it through the Detail slider
        so it commits exactly like dragging the slider (the two are one cut)."""
        if getattr(self, "hier", None) is None:
            return
        k = int(max(2, min(int(k), self.detail_slider.maximum())))
        if k == self.detail_slider.value():
            self._commit_cut(k)                 # value unchanged → commit explicitly
        else:
            self.detail_slider.setValue(k)      # fires _detail_changed → _commit_cut

    def _sync_dendrogram(self, k=None):
        """Keep the dendrogram's cut line + branch colours on the current cut. ``k``
        defaults to the slider position (the source of truth for where the tree is cut)."""
        d = getattr(self, "seg_dendro", None)
        if d is not None and getattr(self, "hier", None) is not None:
            d.set_cut(int(self.detail_slider.value()) if k is None else int(k))

    # ----- dendrogram branch highlighting → regions ------------------------ #
    def _branch_pixel_mask(self, micros):
        """Per-(dataset)-pixel boolean mask for a set of micro-cluster ids, lifted off the
        live :class:`spatial.Hierarchy` exactly as ``segmentation_at`` scatters a cut back
        onto the full grid (so region-scoped trees land in the right pixels). ``None`` if
        there is no tree or no micros resolve to any pixel."""
        hier = getattr(self, "hier", None)
        micros = list(micros or ())
        if hier is None or not micros or self.ds is None:
            return None
        sub = np.isin(np.asarray(hier.micro_labels), micros)          # (n_used,) over segmented px
        full = self._lift_to_full(sub, False, bool)                   # scatter onto the full grid
        return full if full.any() else None

    def _refresh_branch_preview(self):
        """Paint the lit branches' pixels on the seg canvas iff the tree is in 'Highlight
        branches' mode and something is lit. Returns True when it painted (so callers can
        fall back to the resting view otherwise). The single place the amber map preview
        is (re)asserted — re-segmenting/recutting keeps the same cut-invariant micros, so
        this stays correct without recomputing the selection."""
        d = getattr(self, "seg_dendro", None)
        micros = getattr(self, "_dendro_branch_micros", frozenset())
        if d is None or not d.is_picking() or not micros:
            return False
        mask = self._branch_pixel_mask(micros)
        if mask is None:
            return False
        npx = int(np.asarray(mask, bool).sum())
        self._preview_pixel_mask(mask, f"highlighted branches · {npx:,} px", by_cluster=True)
        return True

    def _dendro_branches_changed(self, micros):
        """The lit branches changed: remember them and preview their pixels on the map
        (empty / cut-mode → fall back to the resting segment/region view)."""
        self._dendro_branch_micros = frozenset(int(m) for m in micros)
        if not self._refresh_branch_preview():
            self._update_seg_preview()

    def _dendro_mode_changed(self, _mode):
        """The tree's click mode toggled: keep the seg map in step — show the lit-branch
        preview in pick mode, restore the resting cluster/region view in cut mode."""
        if not self._refresh_branch_preview():
            self._update_seg_preview()

    def _preview_pixel_mask(self, mask, note="", color=HILITE, label="highlighted branches",
                            by_cluster=False):
        """Highlight an arbitrary per-pixel ``mask`` (e.g. lit tree branches, or a freshly
        built region) on the seg canvas — selected pixels in ``color``, the rest of the
        tissue dimmed for context. With ``by_cluster`` each lit pixel instead takes its own
        cluster's colour, so a highlighted branch shows in the same colour it has on the
        HCA tree (tree branch colours == seg-map cluster colours) rather than flat amber."""
        if self.seg is None or self.ds is None:
            return
        base = self.seg.label_image                       # (h,w) float, NaN off-tissue
        h, wd = base.shape
        rgba = np.zeros((h, wd, 4), dtype=np.ubyte)
        rgba[np.isfinite(base)] = [70, 70, 70, 255]       # dim context
        sel2d = self.ds.to_image(np.asarray(mask, dtype=float), fill=0.0) > 0.5
        if by_cluster:
            cls = np.unique(base[sel2d & np.isfinite(base)]).astype(int)
            for cl in cls:
                c = QtGui.QColor(PALETTE[int(cl) % len(PALETTE)])
                rgba[sel2d & (base == cl)] = [c.red(), c.green(), c.blue(), 255]
            self.seg_img_item.setImage(rgba)
            lin = getattr(self, "_seg_lineage", {})
            pairs = [(PALETTE[int(cl) % len(PALETTE)], lin.get(int(cl), str(int(cl))))
                     for cl in cls[:16]]
            self._update_seg_legend(pairs or [(color, label)], note)
            return
        c = QtGui.QColor(color)
        rgba[sel2d] = [c.red(), c.green(), c.blue(), 255]
        self.seg_img_item.setImage(rgba)
        self._update_seg_legend([(color, label)], note)

    def _dendro_make_region(self, micros):
        """Turn the lit tree branches into a new mask-backed region (their union of
        pixels), then clear the selection and show the new region's footprint on the seg
        canvas (it is mask-backed, so the cluster-keyed region map wouldn't surface it)."""
        micros = list(micros or ())
        mask = self._branch_pixel_mask(micros)
        if mask is None or not mask.any():
            self.statusBar().showMessage(
                "Highlight one or more branches on the tree first." if not micros
                else "The highlighted branches map to no segmented pixels.")
            return
        self.record_undo("new region from branches", domains=("regions",))
        idx = self._new_region(name=f"Branch region {len(self.regions) + 1}", mask=mask)
        self.seg_dendro.clear_branches()                  # done — reset the lit branches
        rg = self.regions[idx]
        npx = int(np.asarray(mask, bool).sum())
        # clear_branches() left the canvas on the resting view; paint the new region so the
        # headline action gives visible confirmation on the tab the user is on.
        self._preview_pixel_mask(rg["mask"], f"new region · {npx:,} px",
                                 color=rg["color"], label=rg["name"])
        self.statusBar().showMessage(
            f"Created '{rg['name']}' from highlighted tree branches ({npx:,} px).")

    @staticmethod
    def _var_text(obj):
        """Embedding-quality readout: PCA explained variance for the Euclidean path, or
        the metric name for correlation (where PCA is skipped, so EV is undefined/NaN)."""
        ev = float(getattr(obj, "explained_variance", float("nan")))
        return f"PCA EV {ev:.0%}" if np.isfinite(ev) else "correlation distance"

    def _lift_to_full(self, sub, fill, dtype):
        """Scatter a per-(segmented)-pixel array ``sub`` onto the full dataset grid, filling
        unsegmented pixels with ``fill`` — the canonical region-scoped-tree lift (mirrors
        ``spatial.segmentation_at``). Whole-slide trees (no ``pixel_mask``) pass through."""
        sub = np.asarray(sub, dtype=dtype)
        mask = getattr(getattr(self, "hier", None), "pixel_mask", None)
        if mask is None:
            return sub
        full = np.full(int(getattr(self.hier, "n_total", 0) or self.ds.n_pixels), fill, dtype=dtype)
        full[np.asarray(mask, dtype=bool)] = sub
        return full

    def _full_pixel_labels(self, sub_labels):
        """Lift segment labels to a full per-pixel float array for ``ds.to_image`` — NaN
        outside a region-scoped tree's mask. Whole-slide trees pass straight through."""
        return self._lift_to_full(sub_labels, np.nan, float)

    def _render_label_image(self, labels):
        """Paint the seg canvas straight from a per-pixel ``labels`` array, without
        touching self.seg / the table / regions — for live slider previews."""
        if self.ds is None:
            return
        img = self.ds.to_image(self._full_pixel_labels(labels))   # (h,w), NaN off-tissue/region
        h, wd = img.shape
        rgba = np.zeros((h, wd, 4), dtype=np.ubyte)
        finite = np.isfinite(img)
        if finite.any():
            # One NaN-safe palette-LUT gather instead of k full-frame `img == cl` scans
            # (3x at k=8, ~50x at k=200, on the live-slider GUI thread). LUT row 0 stays
            # transparent so off-tissue NaN and any stray negative label fall there.
            k = int(np.nanmax(img)) + 1
            lut = np.zeros((k + 1, 4), dtype=np.ubyte)
            for cl in range(k):
                c = QtGui.QColor(PALETTE[cl % len(PALETTE)])
                lut[cl + 1] = [c.red(), c.green(), c.blue(), 255]
            idx = np.zeros((h, wd), dtype=np.intp)
            idx[finite] = img[finite].astype(np.intp) + 1
            rgba = lut[idx]
        self.seg_img_item.setImage(rgba)

    def _preview_cut(self, k):
        """Live recolour for the cut at ``k`` segments (no commit) — sub-millisecond."""
        labels = spatial.cut(self.hier, k)
        self._render_label_image(labels)
        self._refresh_branch_preview()        # keep lit-branch pixels visible while dragging detail
        self._sync_dendrogram(k)
        n = int(labels.max()) + 1 if labels.size else 0
        self.seg_info.setText(f"k={n} · dragging… · {self._var_text(self.hier)}")

    def _commit_cut(self, k):
        """Commit the cut at ``k``: rebuild the segment table + regions off the new
        partition (a new granularity is a new partition, so cluster regions reset).
        The silhouette in segmentation_at is the slow part, so commit on a background
        thread and drop stale results (the canvas keeps showing the live preview); this
        keeps the slider from freezing the window on large datasets."""
        if getattr(self, "hier", None) is None:
            return
        ds, hier = self.ds, self.hier
        token = getattr(self, "_seg_cut_token", 0) + 1
        self._seg_cut_token = token

        def compute():
            return spatial.segmentation_at(ds, hier, k, with_silhouette=True)

        def on_done(seg):
            if token == getattr(self, "_seg_cut_token", 0):   # ignore superseded cuts
                self._on_seg(seg, record=False)

        self._run(compute, on_done=on_done, busy=f"Building {k} segments…")

    def _export_seg_labels(self):
        """Export the current segmentation as a per-pixel table (``pixel, x, y, label``)
        — the label map as data rather than an image, for cross-tool comparison or
        external analysis. ``pixel`` is the imzML acquisition order; off-tissue /
        out-of-region pixels carry ``label`` -1."""
        seg = getattr(self, "seg", None)
        if self.ds is None or seg is None or getattr(seg, "labels", None) is None:
            self.statusBar().showMessage("Run a segmentation first.")
            return
        path, _ = filedialogs.get_save_file_name(
            self, "Export segmentation labels", "segmentation_labels.csv",
            "CSV (*.csv);;Tab-separated (*.tsv);;Excel (*.xlsx)")
        if not path:
            return
        import pandas as pd
        xy = np.asarray(self.ds.coordinates, dtype=int)
        labels = np.asarray(seg.labels, dtype=int)
        n = min(len(xy), len(labels))                 # defensive: identical in practice
        df = pd.DataFrame({"pixel": range(n), "x": xy[:n, 0], "y": xy[:n, 1],
                           "label": labels[:n]})
        from .. import export
        export.write_table(df, path)
        k = int(len({int(v) for v in labels[labels >= 0]}))
        self.statusBar().showMessage(f"Wrote {path} ({n} pixels, {k} clusters).")

    def _on_seg(self, seg, record=True):
        self.seg = seg
        # a fresh segmentation invalidates cluster-based regions, but ROI/sample regions
        # are spatial selections independent of the clustering — keep those.
        self.regions = [rg for rg in self.regions if rg.get("mask") is not None]
        self._seg_lineage = {cl: str(cl) for cl in range(seg.n_clusters)}   # drill-down lineage
        self._seg_hidden = set()                      # cluster ids hidden from the image
        self._seg_hover_cluster = None                # transient hover-peek overlay
        self._seg_original = (seg.labels.copy(), int(seg.n_clusters),        # snapshot for Reset
                              float(seg.explained_variance), float(seg.silhouette))
        if record:
            self.record_step(
                "segmentation", label="Segmentation (hierarchical clustering)",
                params={"method": getattr(self, "_seg_method", "ward"),
                        "metric": getattr(self, "_seg_metric", "euclidean"),
                        "scoped": getattr(self, "_seg_scoped", False),
                        "n_clusters": int(seg.n_clusters),
                        "silhouette": round(float(seg.silhouette), 3),
                        "norm": self.norm, "tol_ppm": self.ppm},
                regions=([self._seg_scope_region]
                         if getattr(self, "_seg_scope_region", None) else []))
        self._render_seg_base()
        sil = "n/a" if not np.isfinite(seg.silhouette) else f"{seg.silhouette:.2f}"
        self.seg_info.setText(f"k={seg.n_clusters} · silhouette {sil} · {self._var_text(seg)}")
        self._rebuild_seg_table()
        self._refresh_seg_run_state()
        self._refresh_region_list()
        self._sync_region_combos()
        self._sync_dendrogram(seg.n_clusters)   # follow the committed cut, not a stale slider
        if hasattr(self, "_refresh_action_states"):
            self._refresh_action_states()
        # _render_seg_base just repainted clusters; if branches are still lit (pick mode),
        # re-assert their amber preview so the map and the tree don't fall out of step.
        self._refresh_branch_preview()
        self.statusBar().showMessage(
            f"{seg.n_clusters} segments. Click a blob or tick rows to group them into named "
            "regions on the right; right-click a segment for 'Increase detail here'.")
        self.segChanged.emit()

    # ----- local drill-in: re-cluster one segment ("increase detail here") - #
    def _seg_double_clicked(self, row, _col):
        """Double-click a segment row to drill into it (split it locally into 2)."""
        it = self.seg_table.item(row, 1)
        if it is not None:
            self._split_selected(segs=[int(it.data(QtCore.Qt.UserRole))])

    def _split_selected(self, k=None, segs=None):
        """Re-cluster the target segment(s) into sub-clusters — local drill-in for when
        one region needs finer detail than the global slider position. Runs in the
        background, then applies new labels + lineage and migrates any region that
        contained a split cluster."""
        if self.seg is None:
            self.statusBar().showMessage("Run segmentation first.")
            return
        targets = list(segs) if segs is not None else self._target_segments()
        if not targets:
            self.statusBar().showMessage("Tick or select a segment to split.")
            return
        self.record_undo("split segment", domains=("seg", "regions"))
        k = int(k or 2)                                   # default: split in two
        ds, mzs, ppm, norm = self.ds, self._scope_mzs(self.seg_scope), self.ppm, self.norm
        metric = getattr(self, "_seg_metric", "euclidean")   # drill-in matches the parent run
        labels0 = self.seg.labels

        def compute():
            labels = labels0.copy()
            next_id = int(labels.max()) + 1
            produced = {}                                # parent id -> [new child ids]
            for c in targets:
                mask_c = labels0 == c
                sub, _ = spatial.subcluster(ds, mzs, mask_c, k, tol_ppm=ppm, norm=norm,
                                            metric=metric)
                if sub is None or int(sub.max()) < 1:    # too small / homogeneous to split
                    continue
                idxs = np.flatnonzero(mask_c)
                newids = []
                for j in range(1, int(sub.max()) + 1):   # sub==0 keeps the parent id
                    labels[idxs[sub == j]] = next_id
                    newids.append(next_id)
                    next_id += 1
                produced[int(c)] = newids
            return labels, produced

        self._run(compute, on_done=self._on_split, busy=f"Splitting into {k} sub-clusters…")

    def _on_split(self, out):
        labels, produced = out
        if not produced:
            self.statusBar().showMessage("Couldn't split further — segment is too homogeneous.")
            return
        self.seg.labels = labels
        self.seg.n_clusters = int(labels.max()) + 1
        # off-region pixels (-1, region-scoped runs) stay transparent, like off-tissue
        self.seg.label_image = self.ds.to_image(np.where(labels < 0, np.nan, labels.astype(float)))
        lin = getattr(self, "_seg_lineage", {})
        for c, newids in produced.items():
            parent = lin.get(c, str(c))
            lin[c] = f"{parent}·1"                        # the retained bulk becomes child 1
            for j, nid in enumerate(newids, start=2):
                lin[nid] = f"{parent}·{j}"
            for rg in self.regions:                       # keep regions intact across a split
                if c in rg["segments"]:
                    rg["segments"].update(newids)
        self._seg_lineage = lin
        self._rebuild_seg_table(keep_checked=True)        # drilling in keeps your ticks
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_seg_base()
        self.statusBar().showMessage(
            f"Split complete — now {self.seg.n_clusters} segments. Drill further or "
            "group sub-clusters into regions.")

    # ----- local re-merge: fuse segments back together (inverse of split) --- #
    def _lineage_siblings(self, cl):
        """Every current cluster that shares ``cl``'s immediate split parent (its sibling
        group, including itself). For an un-split top-level cluster this is just ``{cl}``,
        so a lone selection never sweeps in unrelated regions — but picking one child of a
        split returns the whole group, so merging it re-fuses the split in one click."""
        lin = getattr(self, "_seg_lineage", {})
        path = lin.get(cl, str(cl))
        if "·" not in path:                               # not a split child
            return {cl}
        parent = path.rsplit("·", 1)[0]
        sibs = {c for c in range(self.seg.n_clusters)
                if (self.seg.labels == c).any()
                and "·" in lin.get(c, str(c))
                and lin.get(c, str(c)).rsplit("·", 1)[0] == parent}
        return sibs or {cl}

    def _merge_selected(self, segs=None):
        """Merge the target segment(s) into a single segment — the inverse of 'Increase
        detail here'. Any target that is a sub-cluster of a split is expanded to its whole
        sibling group, so selecting **one of a pair** (or any one sub-cluster) merges them
        all. Pixels take the id of the **largest** target (so a region keeps the dominant
        cluster's colour); other clusters' ids are untouched, so existing region
        assignments stay valid. Like splits, a fresh Detail cut re-partitions from the tree
        — use Reset / re-cut to undo."""
        if self.seg is None:
            self.statusBar().showMessage("Run segmentation first.")
            return
        raw = set(segs if segs is not None else self._target_segments())
        if not raw:
            self.statusBar().showMessage("Tick or select segment(s) to merge.")
            return
        targets = sorted(set().union(*(self._lineage_siblings(c) for c in raw)))
        if len(targets) < 2:
            self.statusBar().showMessage("Select two or more segments — or one sub-cluster of a "
                                         "split — to merge.")
            return
        self.record_undo("merge segments", domains=("seg", "regions"))
        labels = self.seg.labels
        keep = max(targets, key=lambda c: int((labels == c).sum()))   # absorb into the largest
        gone = [c for c in targets if c != keep]
        new = labels.copy()
        new[np.isin(labels, gone)] = keep
        self.seg.labels = new
        self.seg.n_clusters = int(new.max()) + 1
        self.seg.label_image = self.ds.to_image(new)
        # carry the merged ids through lineage, the hidden set, and any region memberships
        lin = getattr(self, "_seg_lineage", {})
        lin[keep] = "+".join(sorted({lin.get(c, str(c)) for c in targets}))
        for c in gone:
            lin.pop(c, None)
        self._seg_lineage = lin
        self._seg_hidden = {(keep if s in gone else s) for s in getattr(self, "_seg_hidden", set())}
        moved = []
        for rg in self.regions:
            if rg.get("segments") and (rg["segments"] & set(gone)):
                rg["segments"] = {(keep if s in gone else s) for s in rg["segments"]}
                moved.append(rg["name"])
        self._check_segments([])
        self._rebuild_seg_table()
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_seg_base()
        n_live = int(np.unique(new).size)
        note = f" · updated region(s): {', '.join(moved)}" if moved else ""
        self.statusBar().showMessage(
            f"Merged segments {', '.join(map(str, targets))} → {keep}. Now {n_live} segments{note}.")

    # ----- region model (cluster-based and/or pixel-mask "sample" regions) - #
    def _region_pixel_mask(self, rg, _seen=None):
        """Boolean per-pixel mask for a region. A region may be backed by a drawn
        ROI (``mask``) — used to select whole samples on a slide — and/or by a set of
        segmentation clusters (``segments``). The drawn mask wins when present, else
        the union of its clusters. A region with *neither* but with sub-regions is an
        **aggregate parent**: its mask is the union of its children's (a compartment
        grouping several fascicle ROIs resolves to all their pixels, without copying
        them — each child stays its own region for region-mean / ROI-replicate use).
        Returns ``None`` if nothing resolves yet."""
        m = rg.get("mask")
        if m is not None:
            return np.asarray(m, dtype=bool)
        segs = rg.get("segments")
        if not segs:
            return self._aggregate_child_mask(rg, _seen)
        if segs and self.seg is not None:
            # np.isin over every pixel is far from free, and this is called in tight loops
            # (scope-bar readouts on each feature toggle, stats, exports, overlays). Memoise
            # on the region against a content key — the live segmentation identity and the
            # exact cluster set — so it self-invalidates on re-segmentation or a region edit
            # without any separate invalidation pathway. (Cache key is dropped on save; the
            # session only serialises name/color/segments/mask.)
            key = (id(self.seg), frozenset(int(s) for s in segs))
            cached = rg.get("_mask_cache")
            if cached is not None and cached[0] == key:
                return cached[1]
            mask = np.isin(self.seg.labels, list(segs))
            rg["_mask_cache"] = (key, mask)
            return mask
        return None

    def _aggregate_child_mask(self, rg, _seen=None):
        """Union of the pixel masks of every region nested directly under ``rg`` (recursing
        into nested parents). ``None`` when it has no children that resolve to pixels. The
        ``_seen`` set guards against a malformed parent cycle. Computed on demand — children
        are individually cached — so an aggregate parent tracks any edit to its children."""
        name = rg.get("name")
        if name is None:
            return None
        if _seen is None:
            _seen = set()
        if name in _seen:
            return None
        _seen.add(name)
        union = None
        for ch in self.regions:
            if ch is rg or ch.get("parent") != name:
                continue
            mk = self._region_pixel_mask(ch, _seen)
            if mk is None:
                continue
            union = np.asarray(mk, dtype=bool).copy() if union is None else (union | mk)
        return union

    def _region_mask_2d(self, rg):
        """The region's pixel mask scattered onto the displayed image grid (H×W bool), or
        ``None`` if the region resolves to no pixels. Orientation-correct (goes through
        :meth:`MSI.to_image`), so it stays aligned after a Rotate."""
        m = self._region_pixel_mask(rg)
        if m is None:
            return None
        m2d = self.ds.to_image(np.asarray(m, dtype=float), fill=0.0) > 0.5
        return m2d if m2d.any() else None

    def _region_bbox_raw(self, rg, pad_frac=CROP_PAD_FRAC):
        """The region's mask bounding box as ``(r0, r1, c0, c1)`` display-grid bounds, padded
        by ``pad_frac`` of its extent (ignores any manually drawn crop). ``None`` if empty."""
        m2d = self._region_mask_2d(rg)
        if m2d is None:
            return None
        ys, xs = np.where(m2d)
        h, w = m2d.shape
        r0, r1, c0, c1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pr = int(round((r1 - r0) * pad_frac))
        pc = int(round((c1 - c0) * pad_frac))
        return (max(0, r0 - pr), min(h, r1 + pr), max(0, c0 - pc), min(w, c1 + pc))

    def _region_bbox(self, rg, pad_frac=CROP_PAD_FRAC):
        """``(r0, r1, c0, c1)`` crop bounds an export zooms in on. A manually set crop
        (``rg['crop']``, drawn in the Crop Studio) wins when it was made at the current
        orientation; otherwise the padded mask bounding box (:meth:`_region_bbox_raw`)."""
        crop = rg.get("crop")
        if crop and rg.get("crop_orient", self.ds.orientation) == self.ds.orientation:
            return tuple(int(round(float(v))) for v in crop)
        return self._region_bbox_raw(rg, pad_frac)

    def _crop_from_preset(self, rg, preset=None, pad_frac=None):
        """Crop bounds for ``rg`` derived from the **project crop standard** — an aspect ratio
        and/or a fixed physical size — centred on the ROI, so close-ups are reproducibly framed.
        Falls back to the padded bounding box when there is no preset. ``None`` if empty."""
        preset = preset if preset is not None else getattr(self, "_crop_preset", None)
        pf = pad_frac if pad_frac is not None else (preset or {}).get("pad_frac", CROP_PAD_FRAC)
        base = self._region_bbox_raw(rg, pf)
        if base is None or not preset:
            return base
        m2d = self._region_mask_2d(rg)
        ys, xs = np.where(m2d)
        h, w = m2d.shape
        cr = (int(ys.min()) + int(ys.max()) + 1) / 2.0    # ROI centre (row, col)
        cc = (int(xs.min()) + int(xs.max()) + 1) / 2.0
        r0, r1, c0, c1 = base
        hpx, wpx = float(r1 - r0), float(c1 - c0)          # height = rows, width = cols
        px = getattr(self.ds, "pixel_size_um", None)
        size_um = preset.get("size_um")
        if size_um and px:                                 # fixed physical size (most reproducible)
            wpx, hpx = float(size_um[0]) / px, float(size_um[1]) / px
        else:
            a = preset.get("aspect")                       # width/height; expand the deficient dim
            if a:
                if wpx / max(hpx, 1e-9) < a:
                    wpx = a * hpx
                else:
                    hpx = wpx / a
        r0, r1 = _clamp_span(cr - hpx / 2.0, cr + hpx / 2.0, h)
        c0, c1 = _clamp_span(cc - wpx / 2.0, cc + wpx / 2.0, w)
        return (int(round(r0)), int(round(r1)), int(round(c0)), int(round(c1)))

    def _unique_region_name(self, name):
        existing = {r["name"] for r in self.regions}
        if name not in existing:
            return name
        i = 2
        while f"{name} ({i})" in existing:
            i += 1
        return f"{name} ({i})"

    def _new_region(self, name=None, color=None, segments=None, mask=None, parent=None,
                    select=True, refresh=True):
        """Append a region (cluster-based and/or ROI-mask, optionally a sub-region of
        ``parent`` named region) and refresh every region view. Returns its index.

        ``refresh=False`` skips the (O(n)) list/combo/map rebuild so a batch caller adding
        many regions pays it once at the end via :meth:`_refresh_regions_all`, not per
        region (which made 'regions from all segments' / auto-detect O(n²))."""
        idx = len(self.regions)
        color = color or REGION_PALETTE[idx % len(REGION_PALETTE)]
        rg = {"name": self._unique_region_name(name or f"Region {idx + 1}"),
              "color": color,
              "segments": set(int(s) for s in (segments or ())),
              "mask": (np.asarray(mask, dtype=bool) if mask is not None else None),
              "parent": parent, "visible": True,
              # tie the region to the slide it's drawn on so it never renders on another
              # sample's tissue (regions persist per-session; this is the render-time guard).
              "sample": (self.ds.source if getattr(self, "ds", None) is not None else "")}
        self.regions.append(rg)
        if refresh:
            self._refresh_regions_all(rg["name"] if select else None)
        return next(i for i, r in enumerate(self.regions) if r is rg)

    def _refresh_regions_all(self, select_name=None):
        """The full post-edit region refresh — list (with sub-region reorder) + A/B combos
        + seg colour map. Batch callers run this once at the end instead of per region."""
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        if select_name is not None:
            self._select_region_by_name(select_name)
        self.regionsChanged.emit()

    @staticmethod
    def _effective_parent(rg, by_name):
        """The region's parent *name* only when it resolves to another existing region;
        otherwise None — so orphans (parent points at a deleted region), self-parents and
        cycles read as top-level, exactly the way :meth:`_nested_order` lays them out. Used
        everywhere sibling grouping must match the displayed nesting (move, context menu)."""
        p = rg.get("parent")
        return p if (p and p in by_name and p != rg["name"]) else None

    def _nested_order(self, regions):
        """Depth-first order: each sub-region directly under its parent, siblings keeping
        their relative order in ``regions``. Orphans / cycles fall back to top level. Pure —
        returns a new list and does NOT mutate ``self.regions`` (so callers can preview the
        resulting layout before committing)."""
        by_name = {rg["name"]: rg for rg in regions}
        children, roots = {}, []
        for rg in regions:
            p = self._effective_parent(rg, by_name)
            if p is not None:
                children.setdefault(p, []).append(rg)
            else:
                roots.append(rg)
        ordered, seen = [], set()

        def visit(rg):
            if id(rg) in seen:                            # cycle guard
                return
            seen.add(id(rg))
            ordered.append(rg)
            for ch in children.get(rg["name"], []):
                visit(ch)

        for rg in roots:
            visit(rg)
        for rg in regions:                                # any cycle remnants
            if id(rg) not in seen:
                ordered.append(rg)
        return ordered

    def _reorder_regions(self):
        """Order regions depth-first so each sub-region sits directly under its parent
        (siblings keep creation order). Keeps ``self.regions`` in the order the list shows,
        so row↔region stays 1:1."""
        self.regions = self._nested_order(self.regions)

    def _region_sync_order(self):
        """Persist a drag-reorder of the Regions list, fired (deferred) from
        :meth:`RegionList.dropEvent` once the drop has fully settled. Reads each row's
        region index (stamped in ``UserRole`` by :meth:`_refresh_region_list`) in the new
        top-to-bottom order, re-applies parent grouping so a moved parent keeps its
        sub-regions, and commits it as one undoable step. A child dragged outside its
        parent's block snaps back under it (use right-click ▸ 'Move into' to re-parent);
        such a no-net-change drag records NO undo step. Malformed stamps just resync the
        list to the model rather than dropping regions."""
        lst = getattr(self, "region_list", None)
        if lst is None:
            return
        before = [id(r) for r in self.regions]
        order, seen = [], set()
        for r in range(lst.count()):
            idx = lst.item(r).data(QtCore.Qt.UserRole)
            if not isinstance(idx, int) or not (0 <= idx < len(self.regions)) or idx in seen:
                self._refresh_region_list()               # stale/duplicate stamp → resync visual
                return
            seen.add(idx)
            order.append(self.regions[idx])
        if len(order) != len(self.regions):
            self._refresh_region_list()
            return
        nested = self._nested_order(order)                # re-snap sub-regions under parents
        if [id(r) for r in nested] == before:
            self._refresh_region_list()                   # no net change (incl. snap-back) → no undo
            return
        self.record_undo("reorder regions", domains=("regions",))
        self.regions = nested
        self._refresh_regions_all()

    def _region_move(self, ri, delta):
        """Move the region at row ``ri`` up (``delta`` < 0) or down (``delta`` > 0) among
        its *siblings* — regions that share its effective parent, so the level matches what
        the list shows (orphans move among the top-level rows). Sub-regions travel with
        their parent. No-op at the top/bottom of its level."""
        if not (0 <= ri < len(self.regions)):
            return
        rg = self.regions[ri]
        by_name = {r["name"]: r for r in self.regions}
        parent = self._effective_parent(rg, by_name)
        sibs = [r for r in self.regions if self._effective_parent(r, by_name) == parent]
        pos = next(k for k, r in enumerate(sibs) if r is rg)
        new = pos + delta
        if not (0 <= new < len(sibs)):
            return                                        # already first/last at its level
        other = sibs[new]
        j = next(k for k, r in enumerate(self.regions) if r is other)
        self.record_undo("reorder regions", domains=("regions",))
        self.regions[ri], self.regions[j] = self.regions[j], self.regions[ri]
        self._refresh_regions_all(select_name=rg["name"])
        self.statusBar().showMessage(f"Moved '{rg['name']}' {'up' if delta < 0 else 'down'}.")

    def _region_descendant_names(self, name):
        """Names of every region nested under ``name`` (children, grandchildren, …),
        following the ``parent`` links — used to keep 'Move into' from forming a cycle."""
        children = {}
        for rg in self.regions:
            p = rg.get("parent")
            if p:
                children.setdefault(p, []).append(rg["name"])
        out, stack = set(), list(children.get(name, []))
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(children.get(n, []))
        return out

    def _region_move_into(self, ris, parent_name):
        """Re-parent the regions at rows ``ris`` under the named region (or to top level
        when ``parent_name`` is None). Skips no-ops, and refuses to nest a region under
        itself or one of its own sub-regions (which would make a cycle)."""
        movers = [self.regions[i] for i in ris if 0 <= i < len(self.regions)]
        if not movers:
            return
        if parent_name is not None:
            blocked = set()
            for m in movers:
                blocked.add(m["name"])
                blocked |= self._region_descendant_names(m["name"])
            if parent_name in blocked:
                self.statusBar().showMessage(
                    "Can't move a region into itself or one of its sub-regions.")
                return
        changed = [m for m in movers if m.get("parent") != parent_name]
        if not changed:
            return
        self.record_undo("move region", domains=("regions",))
        for m in changed:
            m["parent"] = parent_name
        self._refresh_region_list()          # reorders so children sit under their parent
        self._sync_region_combos()
        self._render_region_map()
        names = ", ".join(m["name"] for m in changed)
        where = f"into '{parent_name}'" if parent_name else "to top level"
        self.statusBar().showMessage(f"Moved {names} {where}.")

    def _common_region_stem(self, names):
        """A sensible default parent name from sibling region names: their longest common
        prefix, trimmed of trailing separators / counters (so ``endoneurium_F1`` +
        ``endoneurium_F2`` → ``endoneurium``). ``""`` when there's no shared prefix."""
        names = [str(n) for n in names if n]
        if not names:
            return ""
        stem = names[0]
        for n in names[1:]:
            i = 0
            while i < len(stem) and i < len(n) and stem[i] == n[i]:
                i += 1
            stem = stem[:i]
        return stem.rstrip(" _-0123456789").strip()

    def _region_group_into_parent(self, ris=None, name=None):
        """Create a new **parent** region that aggregates the selected regions as its
        children (non-destructive). The parent owns no pixels of its own — its mask is the
        live union of its children (see :meth:`_region_pixel_mask`), so it acts as a
        compartment over several fascicle ROIs while each child stays an independent region
        for region-mean UMAP / ROI-replicate stats. Children keep their own colours.
        ``name`` skips the prompt (used by tests / scripting)."""
        ris = ris if ris is not None else (self._selected_region_indices() or [])
        movers = [self.regions[i] for i in ris if 0 <= i < len(self.regions)]
        if not movers:
            self.statusBar().showMessage("Select one or more regions to group.")
            return
        if name is None:
            default = self._common_region_stem([m["name"] for m in movers]) or "Compartment"
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Group into parent region", "Parent region name:", text=default)
            if not ok:
                return
        name = (name or "").strip()
        if not name:
            return
        # Nest the parent under the movers' shared parent (if they all share one) so an
        # existing hierarchy isn't flattened; else it lands at top level.
        shared = movers[0].get("parent")
        if not all(m.get("parent") == shared for m in movers):
            shared = None
        self.record_undo("group regions", domains=("regions",))
        pidx = self._new_region(name=name, parent=shared, select=False)
        pname = self.regions[pidx]["name"]
        for m in movers:
            m["parent"] = pname
        self._refresh_region_list()
        self._sync_region_combos()
        self._render_region_map()
        self._select_region_by_name(pname)
        self.statusBar().showMessage(
            f"Grouped {len(movers)} region(s) under '{pname}'.")

    def _select_region_by_name(self, name):
        lst = self.region_list
        for r in range(lst.count()):
            if r < len(self.regions) and self.regions[r]["name"] == name:
                lst.setCurrentRow(r)
                return

    def _region_toggle_draw(self, on):
        """The Regions-panel 'Draw ROI' mirrors the ion-tab ROI toggle and jumps to
        the ion image so you can draw without hunting for the control."""
        if getattr(self, "roi_chk", None) is not None and self.roi_chk.isChecked() != on:
            self.roi_chk.setChecked(on)
        if on:
            self.reveal_view("Ion image")      # grouped-tab safe (raw index is brittle post-IA)

    def _sync_region_draw_chk(self, on):
        """Reflect the ion-tab ROI toggle back onto the Regions-panel checkbox."""
        chk = getattr(self, "region_draw_chk", None)
        if chk is not None and chk.isChecked() != on:
            chk.blockSignals(True)
            chk.setChecked(on)
            chk.blockSignals(False)

    def _focus_ion_tab_for_roi(self):
        """Jump to the Ion image tab and turn on ROI drawing — ROIs are drawn on the
        image, then promoted to regions from the persistent Regions panel."""
        self.reveal_view("Ion image")          # grouped-tab safe (raw index is brittle post-IA)
        if getattr(self, "roi_chk", None) is not None and not self.roi_chk.isChecked():
            self.roi_chk.setChecked(True)

    def _region_from_roi(self):
        """Save the drawn ROI (rectangle or polygon) as a named region. The active feature
        list keeps driving the pipelines — build a region-scoped list on demand with the
        ion-image toolbar's '→ Feature list', or right-click ▸ 'Build feature list from
        this region'."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        mask = self._roi_mask()
        px = int(mask.sum()) if mask is not None else 0
        i = self._promote_roi_to_region()            # ROI → region (+ undo, selects it), or -1 if none
        if i < 0:
            self._focus_ion_tab_for_roi()
            self.statusBar().showMessage("Draw a non-empty ROI on the ion image, then 'Add ROI'.")
            return
        self.statusBar().showMessage(
            f"Region '{self.regions[i]['name']}' from ROI: {px:,} px. Build a region-scoped "
            "feature list with '→ Feature list' if you need one.")

    def _auto_detect_samples(self):
        """Auto-find tissue pieces on the slide (connected components of the TIC) and
        add one ROI 'sample' region per piece — the fast path when a slide carries
        several sections. Existing regions are kept."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        ds = self.ds

        def compute():
            return spatial.detect_samples(ds)
        self._run(compute, on_done=self._on_detect_samples, busy="Detecting samples on the slide…")

    def _on_detect_samples(self, masks):
        if not masks:
            self.statusBar().showMessage("No distinct samples found — the slide may hold one "
                                         "piece. Draw ROIs to split it, or adjust thresholds.")
            return
        self.record_undo("auto-detect samples", domains=("regions", "features"))
        base = len(self.regions)
        for i, m in enumerate(masks):
            self._new_region(f"ROI {base + i + 1}", mask=m, select=False, refresh=False)
        self._refresh_regions_all(f"ROI {base + 1}")     # one rebuild + select the first
        self.statusBar().showMessage(
            f"Auto-detected {len(masks)} region(s). Rename/recolour as needed; find peaks while "
            "a region is selected to build its own feature list.")

    def _subregion_from_roi(self):
        """Carve a sub-region out of the selected region using the current ROI
        (intersection with the parent so sub-regions stay inside their sample)."""
        ri = self._selected_region_index()
        if ri < 0:
            self.statusBar().showMessage("Select a parent region first, then draw an ROI inside it.")
            return
        if self.ds is None:
            return
        parent = self.regions[ri]
        roi = self._roi_mask()
        if roi is None or not roi.any():
            self._focus_ion_tab_for_roi()
            self.statusBar().showMessage("Draw an ROI inside the parent region, then 'Sub-region'.")
            return
        pmask = self._region_pixel_mask(parent)
        mask = (roi & pmask) if pmask is not None else roi
        if not mask.any():
            self.statusBar().showMessage("That ROI doesn't overlap the selected parent region.")
            return
        self.record_undo("sub-region", domains=("regions", "features"))
        i = self._new_region(f"{parent['name']} · sub", color=parent["color"], mask=mask,
                             parent=parent["name"])
        self._dismiss_roi()
        self.statusBar().showMessage(
            f"Sub-region '{self.regions[i]['name']}' of '{parent['name']}': {int(mask.sum()):,} px. "
            "Build a region-scoped feature list with '→ Feature list' if you need one.")

    def _selected_region_indices(self):
        rows = sorted({ix.row() for ix in self.region_list.selectionModel().selectedRows()})
        return [r for r in rows if 0 <= r < len(self.regions)]

    def _clear_region_spectra(self):
        """Drop every region mean-spectrum overlay (both per-region and combined)."""
        self._remove_overlays("Region", prefix=True)
        self._region_spectra_shown = False

    def _show_region_spectra(self, focus=True):
        """Overlay the mean spectrum of every *selected & visible* region on the
        spectrum plot — separately (one trace each, in its colour) or combined into
        one trace over the union of their pixels (the 'Combine' toggle). Called on
        selection change so the spectra always track the highlighted region(s); the
        traces show up under the spectrum's 'Traces…' button with their own show/hide toggles."""
        if self.ds is None or getattr(self, "region_list", None) is None:
            return
        self._clear_region_spectra()                      # replace, don't accumulate
        # Invalidate any in-flight background spectrum *now*, before the early returns below
        # — not just in the streaming branch. Otherwise hiding (eye off) or deselecting a
        # region while its mean is still streaming from disk lets that stale job finish the
        # token check and redraw the hidden region's trace.
        token = getattr(self, "_region_spec_token", 0) + 1
        self._region_spec_token = token
        ris = [ri for ri in self._selected_region_indices()
               if self.regions[ri].get("visible", True)]
        if not ris:
            if focus:                                     # button press with nothing chosen
                self.statusBar().showMessage("Select one or more regions in the list first, "
                                             "then 'Show spectra'.")
            return
        if focus:
            self.reveal_view("Ion image")      # grouped-tab safe (raw index is brittle post-IA)
            # un-collapse the spectrum panel so the region traces are actually visible
            if getattr(self, "spectrum", None) is not None and not self.spectrum.isVisible():
                self._toggle_spectrum_panel()
        combined = (getattr(self, "region_combine_chk", None) is not None
                    and self.region_combine_chk.isChecked())
        # Build the list of (mask, color, label) overlays to draw.
        jobs = []
        if combined:
            mask = None
            for ri in ris:
                m = self._region_pixel_mask(self.regions[ri])
                if m is not None:
                    mask = m if mask is None else (mask | m)
            if mask is None or not mask.any():
                return
            jobs.append((mask, self.regions[ris[0]]["color"], "Regions (combined)"))
            msg = f"Combined spectrum of {len(ris)} region(s): {int(mask.sum()):,} px."
        else:
            for ri in ris:
                rg = self.regions[ri]
                m = self._region_pixel_mask(rg)
                if m is not None and m.any():
                    jobs.append((m, rg["color"], f"Region: {rg['name']}"))
            msg = f"Overlaid {len(ris)} region spectrum(a) — toggle via the spectrum's 'Traces…' button."
        self._region_spectra_shown = True   # region traces are now (or about to be) displayed

        # Cube-served means are instant (draw now); any that must stream from disk run on a
        # background thread so selecting region(s) never freezes the window.
        stream = []
        for mask, color, label in jobs:
            fast = self.ds.cube_mean_spectrum(mask)
            if fast is not None:
                self._overlay_spectrum(fast[0], fast[1], color, label, fill=True)
            else:
                stream.append((mask, color, label))
        if not stream:
            if focus:
                self.statusBar().showMessage(msg)
            return
        # No cube yet → build it in the background so the NEXT selection is instant (this
        # one still streams once, off-thread). Self-guards against duplicate builds.
        if hasattr(self, "_autobuild_cache"):
            self._autobuild_cache()
        ds = self.ds
        if focus:
            self.statusBar().showMessage("Computing region spectrum(a)…")

        def compute():
            return [(ds.mean_spectrum(mask=m), color, label) for m, color, label in stream]

        def on_done(results):
            if token != getattr(self, "_region_spec_token", 0):
                return                                     # superseded by a newer selection
            for (axis, spec), color, label in results:
                self._overlay_spectrum(axis, spec, color, label, fill=True)
            if focus:
                self.statusBar().showMessage(msg)

        self._run(compute, on_done=on_done, busy="Region spectrum…")

    # ----- region colour + visibility (eye toggle) ------------------------- #
    def _region_set_color(self):
        ri = self._selected_region_index()
        if ri < 0:
            self.statusBar().showMessage("Select a region first.")
            return
        from . import colorpicker
        hexc = colorpicker.pick_color(self, initial=self.regions[ri]["color"],
                                      title=f"Colour for {self.regions[ri]['name']}")
        if hexc:
            self.record_undo("region colour", domains=("regions",))
            self.regions[ri]["color"] = hexc
            self._refresh_region_list()
            self._sync_region_combos()
            self._render_region_map()
            if getattr(self, "_region_spectra_shown", False):
                self._show_region_spectra(focus=False)    # recolour overlays already on screen
            self.statusBar().showMessage(f"Set {self.regions[ri]['name']} colour to {hexc}.")

    def _region_set_visible(self, ri, vis):
        if 0 <= ri < len(self.regions):
            self.record_undo("region visibility", domains=("regions",))
            self.regions[ri]["visible"] = bool(vis)
            self._refresh_region_list()
            self._render_region_map()
            # Toggling the eye only updates the footprint on the image — it no longer
            # recomputes mean spectra (use 'Show spectra' to refresh the traces).
            self._clear_region_spectra()

    def _region_eye_clicked(self, r):
        """The per-row eye glyph toggles that region's visibility on the image."""
        if 0 <= r < len(self.regions):
            self._region_set_visible(r, not self.regions[r].get("visible", True))

    def _region_list_menu(self, pos):
        ri = self._selected_region_index()
        menu = QtWidgets.QMenu(self.region_list)
        if ri >= 0:
            rg = self.regions[ri]
            menu.addAction(icon("save"), "Build feature list from this region",
                           lambda: self.features_from_region(ri))
            menu.addSeparator()
            menu.addAction(icon("settings"), "Edit close-up crop…", lambda: self.edit_region_crop(ri))
            menu.addAction(icon("find"), "Show spectra", lambda: self._show_region_spectra(focus=True))
            menu.addAction(icon("settings"), "Set colour…", self._region_set_color)
            vis = rg.get("visible", True)
            menu.addAction(icon("remove") if vis else icon("find"), "Hide" if vis else "Show",
                           lambda: self._region_set_visible(ri, not vis))
            menu.addAction(icon("settings"), "Rename…", self._region_rename)
            ris_sel = self._selected_region_indices() or [ri]
            if len(ris_sel) > 1:
                menu.addAction(icon("settings"), f"Batch rename {len(ris_sel)} (prefix/suffix)…",
                               lambda r=list(ris_sel): self._region_batch_rename(r))
            menu.addAction(icon("add"),
                           "Tag group…" if len(ris_sel) <= 1 else f"Tag {len(ris_sel)} with group…",
                           lambda r=list(ris_sel): self._region_batch_tag(r))
            # Reorder within the region's own level (its sub-regions travel with it). You
            # can also drag rows in the list to reorder.
            _bn = {r["name"]: r for r in self.regions}
            _par = self._effective_parent(rg, _bn)
            sibs = [r for r in self.regions if self._effective_parent(r, _bn) == _par]
            spos = next(k for k, r in enumerate(sibs) if r is rg)
            up = menu.addAction(icon("up"), "Move up", lambda: self._region_move(ri, -1))
            up.setEnabled(spos > 0)
            down = menu.addAction(icon("down"), "Move down", lambda: self._region_move(ri, +1))
            down.setEnabled(spos < len(sibs) - 1)
            menu.addSeparator()
            # Move into ▸ — re-parent the selected region(s) under another region (or back
            # to top level). Excludes the region itself and its own sub-regions (no cycles).
            ris = self._selected_region_indices() or [ri]
            movers = [self.regions[i] for i in ris]
            blocked = set()
            for m in movers:
                blocked.add(m["name"])
                blocked |= self._region_descendant_names(m["name"])
            sub = menu.addMenu("Move into")
            top = sub.addAction("Top level (no parent)",
                                lambda: self._region_move_into(ris, None))
            top.setEnabled(any(m.get("parent") for m in movers))
            targets = [r for r in self.regions if r["name"] not in blocked]
            if targets:
                sub.addSeparator()
                for t in targets:
                    sub.addAction(self._color_icon(t["color"]), t["name"],
                                  lambda n=t["name"]: self._region_move_into(ris, n))
            elif not top.isEnabled():
                sub.setEnabled(False)                      # nowhere to move it
            menu.addAction(
                icon("add"),
                "Group into parent region…" if len(ris) > 1 else "Wrap in parent region…",
                lambda r=list(ris): self._region_group_into_parent(r))
            menu.addSeparator()
            menu.addAction(icon("delete"), "Delete" if len(ris) <= 1 else f"Delete {len(ris)} regions",
                           self._region_delete)
        else:
            menu.addAction("Select a region first").setEnabled(False)
        menu.exec(self.region_list.viewport().mapToGlobal(pos))

