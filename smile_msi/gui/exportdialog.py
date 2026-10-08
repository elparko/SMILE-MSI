"""Unified Export hub — one dialog with maximum format + design optionality, plus the
``ExportMixin`` that wires it to the live app state and the pure :mod:`smile_msi.export`
engine.

The hub lets the user export *any* artifact — the active ion image (with the composited
corner spectrum overlay), a colour overlay, a gallery of every visible feature, spectra
(as a styled figure or a data table), the feature/statistics tables, the segmentation
map, or a **full multi-page PDF "book"** bundling images + methods + provenance — in a
choice of raster (PNG/TIFF/JPEG) and vector (PDF/SVG) formats, with theme, colormap,
corner placement, DPI, scale-bar and overlay-content controls.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import export, imaging, prefs, studio, stylelib
from .. import annotations as annot
from .common import (PALETTE, MUTED_QSS, RangeSliderField, icon, CheckList, NoScrollComboBox,
                     NoScrollDoubleSpinBox, NoScrollSpinBox, fig_to_pixmap)
from . import filedialogs

# label-content choices for the in-image legend (value = annotations mode; shared source)
LABEL_CHOICES = annot.LABEL_CHOICES


# scope key → (label, kind) where kind picks the format set + which option groups apply
SCOPES = [
    ("ion",     "Ion image — active m/z",            "image"),
    ("overlay", "Colour overlay (visible features)", "image"),
    ("gallery", "Batch ion images → folder",          "image"),
    ("component", "PCA / NMF component image",        "image"),
    ("specfig", "Spectrum figure (mean / ROI)",      "image"),
    ("specdata", "Spectra → data table",             "table"),
    ("features", "Feature table",                    "table"),
    ("stats",   "ROI statistics",                    "stats"),
    ("seg",     "Segmentation map",                  "image"),
    ("book",    "Full data book (PDF)",              "book"),
]

# Additive-friendly channel hues (well-separated on black) auto-assigned to overlay
# features that don't carry an explicit colour. Pure-ish primaries blend cleanly the way
# additive composites do — red+green→yellow, etc.
OVERLAY_CHANNEL_COLORS = ["#ff3b30", "#30d158", "#0a84ff", "#ffd60a",
                          "#ff2d95", "#64d2ff", "#ff9f0a", "#bf5af2"]

IMAGE_EXT = [("PNG (raster)", "png"), ("TIFF (raster, lossless)", "tiff"),
             ("JPEG (raster)", "jpg"), ("PDF (vector)", "pdf"), ("SVG (vector)", "svg")]
TABLE_EXT = [("CSV", "csv"), ("TSV", "tsv"), ("Excel (.xlsx)", "xlsx"),
             ("JSON", "json"), ("Markdown", "md")]
STATS_EXT = [("Excel report (formatted, charts)", "xlsx"), ("CSV (raw values)", "csv")]
BOOK_EXT = [("PDF report", "pdf")]

PREVIEW_SCOPES = ("ion", "overlay", "gallery", "component", "specfig", "seg")
PREVIEW_DPI = 100
PREVIEW_WIDTH = 560


class ExportDialog(QtWidgets.QDialog):
    """The export hub. Collects options and calls ``win.run_export(scope, opts)``."""

    def __init__(self, win, scope=None):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)
        # Scrollable content area so a tall configuration (e.g. the 'book' scope's report
        # sections + bundle preview, or the expanded 'Design options') can never push the
        # Export/Close buttons off-screen: everything below builds into ``root`` (a widget
        # inside a QScrollArea), while the button box is pinned to ``outer`` beneath it.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(540)
        # Options on the left, a live preview of the image on the right (image scopes only).
        self._split = split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(scroll)
        split.addWidget(self._build_preview_pane())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        root = QtWidgets.QVBoxLayout(content)

        # --- what to export ---
        form = QtWidgets.QFormLayout()
        self.scope_combo = NoScrollComboBox()
        for key, label, _kind in SCOPES:
            self.scope_combo.addItem(label, key)
        self.scope_combo.currentIndexChanged.connect(self._scope_changed)
        form.addRow("What to export", self.scope_combo)

        self.fmt_combo = NoScrollComboBox()
        form.addRow("Format", self.fmt_combo)

        # --- Style preset: a reusable design-system look shared by every figure this export
        # produces (fonts, type sizes, line weights, category colours). Applied on top of the
        # per-figure controls below via export.active_style at render time; importable/exportable
        # as a JSON "recipe" so a look tweaked elsewhere round-trips straight back in. ---
        self.preset_combo = NoScrollComboBox()
        self.preset_combo.setToolTip(
            "A reusable look — font, type sizes, line weights and category colours — applied to "
            "EVERY figure this export produces, on top of the design options below. Built-in "
            "looks plus any you save. Copy the recipe (JSON) to tweak it elsewhere, then Import "
            "the result.")
        self._refresh_preset_combo()
        self.preset_combo.activated.connect(self._apply_preset)
        copy_btn = QtWidgets.QPushButton("Copy")
        copy_btn.setToolTip("Copy this look's recipe (JSON) to the clipboard.")
        copy_btn.clicked.connect(self._copy_recipe)
        imp_btn = QtWidgets.QPushButton("Import…")
        imp_btn.setToolTip("Paste a recipe (JSON) to add it as a saved look.")
        imp_btn.clicked.connect(self._import_recipe)
        prow = QtWidgets.QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.setSpacing(4)
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(copy_btn)
        prow.addWidget(imp_btn)
        pwrap = QtWidgets.QWidget()
        pwrap.setLayout(prow)
        form.addRow("Style preset", pwrap)
        root.addLayout(form)

        # --- image / overlay design options ---
        self.design = QtWidgets.QGroupBox("Image design")
        dl = QtWidgets.QFormLayout(self.design)

        # PRIMARY tier (always visible): colormap + the two most-used Show toggles.
        # Secondary design parameters live one click deeper in the "Design options" disclosure.
        self.cmap_combo = NoScrollComboBox()
        items = [self.win.cmap_combo.itemText(i) for i in range(self.win.cmap_combo.count())] \
            if getattr(self.win, "cmap_combo", None) is not None else []
        self.cmap_combo.addItems(items or ["viridis", "inferno", "magma", "cividis", "turbo", "hot"])
        if getattr(self.win, "cmap_combo", None) is not None:
            self.cmap_combo.setCurrentText(self.win.cmap_combo.currentText())
        self.cmap_combo.setToolTip("Colour map for single-ion intensity images (ion / gallery / "
                                   "component). The colour overlay and segmentation map keep their "
                                   "own per-feature / per-cluster colours, so this is hidden for "
                                   "them.")
        self._design_form = dl
        dl.addRow("Colormap", self.cmap_combo)
        self.cmap_label = dl.labelForField(self.cmap_combo)

        # Build the secondary controls (constructions unchanged) for placement in the disclosure.
        self.theme_combo = NoScrollComboBox()
        self.theme_combo.addItems(["Light", "Dark glass"])   # light page background by default
        self.theme_combo.setToolTip("Page background of the exported figure: 'Light' for print/publication, 'Dark glass' for a black-background on-screen look.")
        self.label_combo = NoScrollComboBox()
        for text, _mode in LABEL_CHOICES:
            self.label_combo.addItem(text)
        self.label_combo.setToolTip("What each ion's legend label shows. A feature you've "
                                    "renamed (double-click its label on the image) keeps that "
                                    "custom text regardless of this choice.")
        self.corner_combo = NoScrollComboBox()
        self.corner_combo.addItems(["Bottom-right", "Bottom-left"])
        self.corner_combo.setToolTip("Which side of the footer band the intensity legend sits "
                                     "on; the scale bar takes the opposite side. Annotations "
                                     "always sit in a solid themed band beneath the image.")

        # Zoom the export in on a region (publication-style ROI close-up). Defaults to each
        # region's bounding box + padding, or the crop box you drew on the image. The region's
        # outline is traced over the data in its own colour.
        self.crop_combo = NoScrollComboBox()
        self.crop_combo.addItem("Full image (no crop)", None)
        for rg in (getattr(self.win, "regions", None) or []):
            if rg.get("visible", True) and self.win._region_bbox(rg) is not None:
                self.crop_combo.addItem(f"Crop to “{rg['name']}”", rg["name"])
        if self.crop_combo.count() > 1:
            self.crop_combo.addItem("Each visible region (separate files)", "__each__")
        self.crop_combo.setToolTip("Zoom the exported image in on a region of interest — the "
                                   "close-ups seen in publications. Uses the crop box you drew "
                                   "on the image, else the region's bounding box plus a margin.")
        self.crop_edit_btn = QtWidgets.QPushButton("Edit crop…")
        self.crop_edit_btn.setIcon(icon("settings"))
        self.crop_edit_btn.setToolTip("Open the Crop Studio to draw or adjust the close-up crop "
                                      "for the selected region. Non-destructive — the crop is "
                                      "saved on the region, not the data, and reopens here.")
        self.crop_edit_btn.clicked.connect(self._edit_crop_clicked)
        self.chk_crop_outline = QtWidgets.QCheckBox("Outline region")
        self.chk_crop_outline.setChecked(True)
        self.chk_crop_outline.setToolTip("Trace the cropped region's border in its colour. "
                                         "Untick for a close-up with no outline.")
        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addWidget(self.crop_combo, 1)
        crop_row.addWidget(self.chk_crop_outline)
        crop_row.addWidget(self.crop_edit_btn)

        # Split the original single "Show" HBox into a primary pair (Annotations + Scale bar,
        # on the main form) and a secondary "More overlays" set (in the disclosure). Every
        # checkbox object, default and the lone chk_card master connection are preserved.
        self.chk_card = QtWidgets.QCheckBox("Annotations")
        self.chk_card.setToolTip("Master switch for the intensity scale + side spectrum/legend.")
        self.chk_spec = QtWidgets.QCheckBox("Side spectrum")
        self.chk_spec.setToolTip("Show the mean spectrum in a margin beside the image.")
        self.chk_cbar = QtWidgets.QCheckBox("Intensity scale")
        self.chk_cbar.setToolTip("Small labelled colour scale in a bottom corner of the image.")
        self.chk_scale = QtWidgets.QCheckBox("Scale bar")
        self.chk_title = QtWidgets.QCheckBox("Title")
        for c in (self.chk_card, self.chk_spec, self.chk_cbar, self.chk_scale, self.chk_title):
            c.setChecked(True)
        self.chk_roi = QtWidgets.QCheckBox("Region outline")
        self.chk_roi.setToolTip("Trace every visible region's boundary over the composite, "
                                "each in its own colour. Colour overlay only.")
        self.chk_roi.setChecked(False)
        self.chk_card.toggled.connect(self._card_toggled)
        primary_toggles = QtWidgets.QHBoxLayout()
        for c in (self.chk_card, self.chk_scale):
            primary_toggles.addWidget(c)
        primary_toggles.addStretch(1)
        dl.addRow("Show", self._wrap(primary_toggles))
        more_toggles = QtWidgets.QHBoxLayout()
        for c in (self.chk_spec, self.chk_cbar, self.chk_title, self.chk_roi):
            more_toggles.addWidget(c)
        more_toggles.addStretch(1)

        self.dpi_spin = NoScrollSpinBox(); self.dpi_spin.setRange(72, 1200); self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setToolTip("Output resolution. 300 dpi is publication-quality; raise it for large prints.")
        self.width_spin = NoScrollDoubleSpinBox(); self.width_spin.setRange(2.0, 20.0)
        self.width_spin.setValue(7.0); self.width_spin.setSuffix(" in")
        self.width_spin.setToolTip("Physical width of the rendered figure in inches (height follows the image aspect).")
        # The scale-bar length is automatic (a round 1-2-5 length ≈20% of the slide, refit to
        # any crop) and shown read-only, so it can never be a wrong number typed by hand. The
        # "Scale bar" checkbox above toggles it; the pixel size below is what it measures against.
        self.scalebar_lbl = QtWidgets.QLabel("—")
        self.scalebar_lbl.setToolTip("Automatic scale-bar length (needs a known pixel size).")
        # Pixel size drives the scale bar; read from the imzML when present, but editable so a
        # file that omits it (no bar otherwise) or records it wrongly can still be made accurate.
        self.pixsize_spin = NoScrollDoubleSpinBox(); self.pixsize_spin.setRange(0, 100000)
        self.pixsize_spin.setDecimals(2); self.pixsize_spin.setSuffix(" µm/px")
        _px0 = getattr(getattr(self.win, "ds", None), "pixel_size_um", None)
        self.pixsize_spin.setValue(float(_px0) if _px0 else 0.0)
        self.pixsize_spin.setToolTip("Physical size of one pixel. Auto-filled from the imzML when "
                                     "recorded; set it here if the file lacks it or is wrong. "
                                     "0 = unknown (no scale bar can be drawn).")
        self.pixsize_spin.valueChanged.connect(self._pixel_size_changed)
        num = QtWidgets.QHBoxLayout()
        num.addWidget(QtWidgets.QLabel("Resolution")); num.addWidget(self.dpi_spin)
        num.addWidget(QtWidgets.QLabel("  Width")); num.addWidget(self.width_spin)
        num.addWidget(QtWidgets.QLabel("  Pixel size")); num.addWidget(self.pixsize_spin)
        num.addWidget(QtWidgets.QLabel("  Scale bar")); num.addWidget(self.scalebar_lbl)
        # caution shown when the bar may be inaccurate (non-square pixels / non-unit grid)
        self.pixsize_warn = QtWidgets.QLabel("")
        self.pixsize_warn.setWordWrap(True)
        self.pixsize_warn.setStyleSheet("color: #c1860a;")
        self.pixsize_warn.setVisible(False)

        # --- collapsible "Design options" disclosure (collapsed by default) ---
        self.design_more = QtWidgets.QGroupBox("Design options")
        self.design_more.setCheckable(True); self.design_more.setChecked(False)   # collapsed default
        dml = QtWidgets.QFormLayout(self.design_more)
        self._design_more_inner = QtWidgets.QWidget()
        dm = QtWidgets.QFormLayout(self._design_more_inner); dm.setContentsMargins(0, 0, 0, 0)
        dm.addRow("Card theme", self.theme_combo)
        dm.addRow("Label", self.label_combo)
        dm.addRow("Legend side", self.corner_combo)
        dm.addRow("Crop to region", self._wrap(crop_row))
        self.crop_combo.currentIndexChanged.connect(self._sync_crop_edit_btn)
        self._sync_crop_edit_btn()
        dm.addRow("More overlays", self._wrap(more_toggles))

        # --- intensity (contrast) window for the export ---------------------- #
        # By default each feature keeps the per-feature window set in the dock; tick the box
        # to override every exported image with one shared low/high window (drag or type it).
        self.chk_window = QtWidgets.QCheckBox("Override intensity window")
        self.chk_window.setToolTip("Off: each feature keeps the contrast window set in the dock. "
                                   "On: every exported image uses the low/high window below.")
        win_row = QtWidgets.QHBoxLayout()
        init_lo, init_hi = self._initial_export_window()
        self.window_slider = RangeSliderField(init_lo, init_hi)
        self.window_slider.setToolTip("Contrast window for the export — % of the hotspot-clip max "
                                      "(relative intensity). Drag the handles or type exact values.")
        self.window_slider.setEnabled(False)
        self.chk_window.toggled.connect(self.window_slider.setEnabled)
        win_row.addWidget(self.window_slider, 1)
        dm.addRow(self.chk_window)
        dm.addRow("Intensity", self._wrap(win_row))

        dm.addRow(self._wrap(num))
        dm.addRow(self.pixsize_warn)
        self._refresh_pixsize_warning()
        dml.addRow(self._design_more_inner)
        self.design_more.toggled.connect(self._design_more_inner.setVisible)
        self._design_more_inner.setVisible(False)
        dl.addRow(self.design_more)   # disclosure is the LAST row of the Image-design form

        root.addWidget(self.design)

        # --- which features to batch (Batch ion images scope only) ---
        # Pick exactly which features get an image; the crop/photo area above is chosen once
        # (via "Crop to region") and applied to every one of them.
        self.feat_group = QtWidgets.QGroupBox("Features to export")
        fg = QtWidgets.QVBoxLayout(self.feat_group)
        self.feat_list = CheckList(
            noun="feature",
            extra_actions=[
                ("Visible", "Tick exactly the features shown on the Ion image tab (respects "
                            "the filter).", lambda i: self._feat_is_visible(i)),
                ("Identified", "Tick exactly the features carrying a lipid annotation "
                               "(respects the filter).", lambda i: self._feat_is_identified(i))])
        self.feat_list.setMaximumHeight(260)
        self.feat_list.setToolTip("Tick the features to generate an image for. Each is rendered "
                                  "with the design + crop chosen above, into one folder.")
        fg.addWidget(self.feat_list)
        root.addWidget(self.feat_group)
        self._populate_feature_list()

        # --- spectra source picker (Spectrum figure / Spectra table scopes) ---
        # Pick which mean spectra to export — the whole-slide mean and/or each region —
        # then whether to overlay them in one output or write one file per spectrum, plus an
        # optional region-A − region-B difference (the FTU difference spectra of lipid atlases).
        self.spec_group = QtWidgets.QGroupBox("Spectra to export")
        sg = QtWidgets.QVBoxLayout(self.spec_group)
        self.spec_list = CheckList(noun="spectrum")
        self.spec_list.setMaximumHeight(180)
        self.spec_list.setToolTip("Tick which mean spectra to export — the whole-slide mean and/or "
                                  "each region. Each region's mean is averaged over its pixels.")
        sg.addWidget(self.spec_list)
        lay_row = QtWidgets.QHBoxLayout()
        lay_row.addWidget(QtWidgets.QLabel("Layout"))
        self.spec_layout = NoScrollComboBox()
        self.spec_layout.addItem("Overlaid — one output", "overlay")
        self.spec_layout.addItem("Separate — one file per spectrum", "separate")
        self.spec_layout.setToolTip("Overlaid: all ticked spectra in a single figure (or one table "
                                    "with a column per spectrum). Separate: one file per spectrum, "
                                    "named with the region.")
        lay_row.addWidget(self.spec_layout, 1)
        sg.addLayout(lay_row)
        self.chk_diff = QtWidgets.QCheckBox("Add difference spectrum (A − B)")
        self.chk_diff.setToolTip("Also export the signed difference between two regions' mean "
                                 "spectra — the FTU/compartment difference spectrum.")
        sg.addWidget(self.chk_diff)
        diff_row = QtWidgets.QHBoxLayout()
        self.diff_a = NoScrollComboBox()
        self.diff_b = NoScrollComboBox()
        diff_row.addWidget(QtWidgets.QLabel("A"))
        diff_row.addWidget(self.diff_a, 1)
        diff_row.addWidget(QtWidgets.QLabel("−  B"))
        diff_row.addWidget(self.diff_b, 1)
        self.diff_a.setEnabled(False)
        self.diff_b.setEnabled(False)
        self.chk_diff.toggled.connect(self.diff_a.setEnabled)
        self.chk_diff.toggled.connect(self.diff_b.setEnabled)
        sg.addLayout(diff_row)
        root.addWidget(self.spec_group)
        self._populate_spec_sources()

        # --- spectrum-figure axes (Spectrum figure / Spectra table scopes) ---
        self.spec_axes = QtWidgets.QGroupBox("Spectrum axes")
        sa = QtWidgets.QFormLayout(self.spec_axes)
        self.chk_xauto = QtWidgets.QCheckBox("Auto (crop to peaks, stop 50 m/z past the last one)")
        self.chk_xauto.setChecked(True)
        self.chk_xauto.setToolTip("Frame the x-axis to the data: start at the first peak and stop "
                                  "about 50 m/z past the last one (dropping the empty acquired "
                                  "tail). Uncheck to set an exact m/z window.")
        self.chk_xauto.toggled.connect(self._xauto_toggled)
        sa.addRow(self.chk_xauto)
        xr = QtWidgets.QHBoxLayout()
        mzr = getattr(self.win.ds, "mz_range", None) or (0.0, 1000.0)
        self.xmin_spin = NoScrollDoubleSpinBox(); self.xmin_spin.setRange(0, 100000)
        self.xmin_spin.setDecimals(2); self.xmin_spin.setSuffix(" m/z"); self.xmin_spin.setValue(float(mzr[0]))
        self.xmax_spin = NoScrollDoubleSpinBox(); self.xmax_spin.setRange(0, 100000)
        self.xmax_spin.setDecimals(2); self.xmax_spin.setSuffix(" m/z"); self.xmax_spin.setValue(float(mzr[1]))
        xr.addWidget(QtWidgets.QLabel("from")); xr.addWidget(self.xmin_spin)
        xr.addWidget(QtWidgets.QLabel("  to")); xr.addWidget(self.xmax_spin)
        xr.addStretch(1)
        sa.addRow("m/z range", self._wrap(xr))
        self.chk_peaks = QtWidgets.QCheckBox("Peak markers (▲ below the axis)")
        self.chk_peaks.setChecked(True)
        self.chk_peaks.setToolTip("Tick each detected peak's m/z with a small triangle just below "
                                  "the x-axis. Uncheck for a clean spectrum with no markers "
                                  "(figure export only).")
        sa.addRow(self.chk_peaks)
        self.chk_active = QtWidgets.QCheckBox("Mark selected m/z (vertical line)")
        self.chk_active.setChecked(False)
        self.chk_active.setToolTip("Draw a faint vertical line at the currently selected peak's "
                                   "m/z. Off by default for a clean publication spectrum "
                                   "(figure export only).")
        sa.addRow(self.chk_active)
        self.spec_style = NoScrollComboBox()
        self.spec_style.addItem("Profile — filled line", "line")
        self.spec_style.addItem("Sticks — centroid impulses", "sticks")
        self.spec_style.setToolTip("Profile: a filled continuous trace. Sticks: a thin vertical "
                                   "line per m/z rising from the baseline — the classic stacked "
                                   "centroid-spectra look (figure export only).")
        sa.addRow("Peak style", self.spec_style)
        self.chk_yshare = QtWidgets.QCheckBox("Standardize y-axis (one shared scale)")
        self.chk_yshare.setChecked(False)
        self.chk_yshare.setToolTip("Put every exported spectrum on one common intensity scale so "
                                   "peak heights are directly comparable. Most useful with the "
                                   "'Separate — one file per spectrum' layout (figure export only).")
        sa.addRow(self.chk_yshare)
        root.addWidget(self.spec_axes)
        self._xauto_toggled(True)

        # --- book sections ---
        self.book = QtWidgets.QGroupBox("Book sections")
        bv = QtWidgets.QVBoxLayout(self.book)
        # contents source: the curated Report-tab list, or auto-assembled from the live view
        src_row = QtWidgets.QHBoxLayout()
        self.book_source_combo = NoScrollComboBox()
        n_items = len(getattr(self.win, "report_items", None) or [])
        self.book_source_combo.addItem(
            f"Curated list — Report tab ({n_items} item{'s' if n_items != 1 else ''})", "curated")
        self.book_source_combo.addItem("Auto from current view", "auto")
        if n_items == 0:                       # nothing curated yet → default to the live view
            self.book_source_combo.setCurrentIndex(1)
        self.book_source_combo.setToolTip(
            "Curated: build the book from the ordered list you assembled on the Report tab.\n"
            "Auto: assemble it from whatever's live now (active ion, visible features, last comparison).")
        self.book_source_combo.currentIndexChanged.connect(self._book_source_changed)
        src_row.addWidget(QtWidgets.QLabel("Contents"))
        src_row.addWidget(self.book_source_combo)
        src_row.addStretch(1)
        bv.addLayout(src_row)
        self.sec_label = QtWidgets.QLabel("Sections")
        bv.addWidget(self.sec_label)
        bl = QtWidgets.QHBoxLayout()
        self.sec_chks = {}
        for key, label in [("summary", "Dataset summary"), ("methods", "Methods & provenance"),
                           ("gallery", "Ion images"), ("spectra", "Spectra"),
                           ("segmentation", "Segmentation"), ("stats", "Statistics")]:
            c = QtWidgets.QCheckBox(label); c.setChecked(True)
            self.sec_chks[key] = c
            bl.addWidget(c)
        bv.addLayout(bl)
        self.chk_bundle = QtWidgets.QCheckBox("Also write a data bundle (CSV folder beside the PDF)")
        self.chk_bundle.setChecked(True)
        self.chk_bundle.setToolTip("Writes the report's underlying data as a tight folder of "
                                   "CSV/Markdown files in a '<name>_data' folder next to the PDF. "
                                   "Pick exactly which files below — the preview shows what you'll get.")
        self.chk_bundle.toggled.connect(self._bundle_toggled)
        bv.addWidget(self.chk_bundle)

        # Per-file bundle toggles — each maps to a file in the reworked, slimmed bundle.
        self.bundle_box = QtWidgets.QWidget()
        bbl = QtWidgets.QHBoxLayout(self.bundle_box)
        bbl.setContentsMargins(16, 0, 0, 0)
        self.bundle_chks = {}
        for key, label in [("features", "Feature table"), ("region_stats", "Region intensities"),
                           ("statistics", "Statistics"), ("spectra", "Raw spectra (large)"),
                           ("methods", "Methods")]:
            c = QtWidgets.QCheckBox(label)
            c.setChecked(export.DEFAULT_BUNDLE_OPTIONS[key])
            c.toggled.connect(self._refresh_bundle_preview)
            self.bundle_chks[key] = c
            bbl.addWidget(c)
        bbl.addStretch(1)
        bv.addWidget(self.bundle_box)
        bv.addWidget(QtWidgets.QLabel("Preview — exactly what the bundle will contain (sample data):"))
        self.bundle_preview = QtWidgets.QPlainTextEdit()
        self.bundle_preview.setReadOnly(True)
        self.bundle_preview.setFixedHeight(150)
        self.bundle_preview.setStyleSheet(
            "font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px;")
        bv.addWidget(self.bundle_preview)
        root.addWidget(self.book)
        self._refresh_bundle_preview()

        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(MUTED_QSS)
        root.addWidget(self.hint)

        # --- buttons ---
        from .common import glossary_button
        gloss = glossary_button([
            ["Annotations", "The composited side panel: intensity scale + spectrum/legend that travels beside the image."],
            ["Hotspot clip", "The intensity percentile treated as 100% — the brightest pixels above it are clipped so a few hot pixels don't wash out the image."],
            ["Skyline (max)", "The maximum-intensity spectrum across all pixels (vs the mean spectrum)."],
            ["TIC normalization", "Each pixel's spectrum scaled to its total ion current so pixel-to-pixel brightness is comparable."],
            ["ppm", "Mass-extraction tolerance (parts-per-million) used to pull each ion image."],
            ["Data bundle", "A folder of CSV/Markdown files written beside the PDF with the report's underlying numbers."],
        ], parent=self)
        btns = QtWidgets.QDialogButtonBox()
        self.b_export = btns.addButton("Export…", QtWidgets.QDialogButtonBox.AcceptRole)
        self.b_export.setObjectName("primaryAction")      # the dialog's primary run action
        self.b_export.setIcon(icon("export"))
        # the Studio is the multi-section / multi-list batch layer above this single-subject hub
        b_studio = btns.addButton("Export Studio…", QtWidgets.QDialogButtonBox.ActionRole)
        b_studio.setIcon(icon("export"))
        b_studio.setToolTip("Batch-export ion images across several feature lists and sections "
                            "at once, plus their CSVs and analyses.")
        b_studio.clicked.connect(self._open_studio)
        btns.addButton(QtWidgets.QDialogButtonBox.Close)
        self.b_export.clicked.connect(self._do_export)
        btns.addButton(gloss, QtWidgets.QDialogButtonBox.HelpRole)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)              # pinned below the scroll area — always visible

        self._restore_prefs()              # default every control to the last export's choices
        if scope:                          # an explicit context scope wins over the saved one
            i = self.scope_combo.findData(scope)
            if i >= 0:
                self.scope_combo.setCurrentIndex(i)
        self._scope_changed()
        self._wire_preview(content)

        # Open at a comfortable size and never taller than the screen — a very tall scope
        # (book) then scrolls inside the dialog instead of overflowing off the bottom.
        scr = QtWidgets.QApplication.primaryScreen()
        avail_h = scr.availableGeometry().height() if scr is not None else 900
        self.setMaximumHeight(avail_h)
        avail_w = scr.availableGeometry().width() if scr is not None else 1400
        w = 600 + (PREVIEW_WIDTH if not self.preview_pane.isHidden() else 0)
        self.resize(min(w, avail_w - 40), min(760, avail_h - 80))
        split.setSizes([600, PREVIEW_WIDTH])
        self._sized = True

    # ----- live preview ----------------------------------------------------- #
    def _build_preview_pane(self):
        self.preview_pane = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(self.preview_pane)
        v.setContentsMargins(8, 8, 8, 8)
        head = QtWidgets.QLabel("Preview — what will be saved")
        head.setWordWrap(True)
        head.setStyleSheet(MUTED_QSS)
        v.addWidget(head)
        self.preview_img = QtWidgets.QLabel()
        self.preview_img.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_img.setWordWrap(True)
        self.preview_img.setMinimumSize(320, 280)
        # Ignored: the pixmap must follow the pane's size, never set it
        self.preview_img.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.preview_img.installEventFilter(self)
        v.addWidget(self.preview_img, 1)
        self.preview_caption = QtWidgets.QLabel("")
        self.preview_caption.setWordWrap(True)
        self.preview_caption.setStyleSheet(MUTED_QSS)
        v.addWidget(self.preview_caption)
        self._preview_pixmap = None
        self._sized = False
        self._wide_width = 0
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._render_preview)
        return self.preview_pane

    def _wire_preview(self, content):
        """Redraw the preview after any option change (debounced)."""
        for w in content.findChildren(QtWidgets.QComboBox):
            w.currentIndexChanged.connect(self._schedule_preview)
        for w in content.findChildren(QtWidgets.QCheckBox):
            w.toggled.connect(self._schedule_preview)
        for cls in (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox):
            for w in content.findChildren(cls):
                w.valueChanged.connect(self._schedule_preview)
        for w in content.findChildren(RangeSliderField):
            w.editingFinished.connect(self._schedule_preview)
        for w in content.findChildren(CheckList):
            w.changed.connect(self._schedule_preview)
        self._schedule_preview()

    def _schedule_preview(self, *_):
        if not self.preview_pane.isHidden():
            self._preview_timer.start()

    def _set_preview_visible(self, show):
        if self.preview_pane.isHidden() != show:
            return
        if self._sized:                    # after construction: grow/shrink by the pane's width
            if show:
                self.preview_pane.setVisible(True)
                self.resize(max(self.width() + PREVIEW_WIDTH, self._wide_width), self.height())
            else:
                self._wide_width = self.width()
                self.preview_pane.setVisible(False)
                self.resize(self.width() - self.preview_pane.width(), self.height())
        else:
            self.preview_pane.setVisible(show)

    def _render_preview(self):
        key = self.scope_combo.currentData()
        if self.preview_pane.isHidden() or key not in PREVIEW_SCOPES:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fig, caption = self.win.export_preview(key, self.options())
            self._preview_pixmap = fig_to_pixmap(fig) if fig is not None else None
        except Exception as e:  # noqa: BLE001 — a bad option combination must not break the dialog
            self._preview_pixmap, caption = None, f"Could not draw the preview: {e}"
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if self._preview_pixmap is None:
            self.preview_img.clear()
            self.preview_img.setText(caption)
            self.preview_caption.setText("")
        else:
            self.preview_caption.setText(caption)
            self._fit_preview()

    def _fit_preview(self):
        if self._preview_pixmap is not None:
            self.preview_img.setPixmap(self._preview_pixmap.scaled(
                self.preview_img.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def eventFilter(self, obj, event):
        if obj is self.preview_img and event.type() == QtCore.QEvent.Resize:
            self._fit_preview()
        return super().eventFilter(obj, event)

    @staticmethod
    def _wrap(layout):
        w = QtWidgets.QWidget(); w.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _card_toggled(self, on):
        for c in (self.chk_spec, self.chk_cbar, self.chk_title):
            c.setEnabled(on)

    def _sync_crop_edit_btn(self, *_):
        """'Edit crop…' is meaningful only when a single region is the crop target — not the
        full image and not the every-region batch."""
        data = self.crop_combo.currentData()
        self.crop_edit_btn.setEnabled(bool(data) and data != "__each__")
        self.chk_crop_outline.setEnabled(bool(data))

    def _edit_crop_clicked(self):
        """Open the Crop Studio for the region chosen in 'Crop to region', so its close-up crop
        can be drawn/adjusted without leaving the Export dialog."""
        name = self.crop_combo.currentData()
        if not name or name == "__each__":
            return
        rg = next((r for r in (getattr(self.win, "regions", None) or [])
                   if r.get("name") == name), None)
        if rg is not None:
            self.win._open_crop_studio(rg)
            self._schedule_preview()

    # ----- batch feature picker (gallery scope) ---------------------------- #
    def _populate_feature_list(self):
        """One checkable row per detected feature ('m/z · lipid'); visible features start
        checked. Each row stores its index into ``win.peaks`` so the export maps back exactly."""
        peaks = getattr(self.win, "peaks", None) or []
        entries = []
        for i, p in enumerate(peaks):
            mz = p.get("mz")
            lab = self.win._clean_label(mz) if mz is not None else ""
            text = (f"m/z {mz:.4f}" if mz is not None else "feature") + (f"  ·  {lab}" if lab else "")
            entries.append((i, text))
        self.feat_list.set_entries(entries)
        self.feat_list.set_checked([i for i, p in enumerate(peaks) if not p.get("hidden")])

    def _peak_at(self, i):
        """The peak a row's key points at, or None once the working set has moved under us."""
        peaks = getattr(self.win, "peaks", None) or []
        return peaks[i] if (i is not None and 0 <= i < len(peaks)) else None

    def _feat_is_visible(self, i):
        p = self._peak_at(i)
        return p is not None and not p.get("hidden")

    def _feat_is_identified(self, i):
        p = self._peak_at(i)
        return p is not None and p.get("mz") is not None and bool(self.win._clean_label(p["mz"]))

    def _selected_peaks(self):
        return [p for p in (self._peak_at(i) for i in self.feat_list.checked_in_order())
                if p is not None]

    def _xauto_toggled(self, on):
        self.xmin_spin.setEnabled(not on)
        self.xmax_spin.setEnabled(not on)

    # ----- spectra source picker (specfig / specdata scopes) --------------- #
    def _spec_sources(self):
        """``[(ident, label, color)]`` — the whole-slide mean plus every visible region that
        resolves to pixels. ``ident`` is ``'__mean__'`` or the region name."""
        out = [("__mean__", "Whole-slide mean", "#5aa9e6")]
        for rg in (getattr(self.win, "regions", None) or []):
            if rg.get("visible", True) and self.win._region_pixel_mask(rg) is not None:
                out.append((rg["name"], rg["name"], rg.get("color", "#f4a259")))
        return out

    def _populate_spec_sources(self):
        """Fill the source checklist + the A/B difference combos from the live regions."""
        self.diff_a.clear()
        self.diff_b.clear()
        sources = self._spec_sources()
        for ident, label, _color in sources:
            self.diff_a.addItem(label, ident)
            self.diff_b.addItem(label, ident)
        self.spec_list.set_entries([(ident, label) for ident, label, _c in sources])
        self.spec_list.set_checked([ident for ident, _l, _c in sources])
        # default difference to the first two regions when present (skip the mean at index 0)
        if self.diff_a.count() > 2:
            self.diff_a.setCurrentIndex(1)
            self.diff_b.setCurrentIndex(2)
        elif self.diff_b.count() > 1:
            self.diff_b.setCurrentIndex(1)

    def _selected_spec_idents(self):
        return self.spec_list.checked_in_order()      # mean first, then regions in panel order

    def _book_source_changed(self, *_):
        """In curated mode the body comes from the Report tab, so the per-section content
        toggles (ion images / spectra / segmentation / statistics) no longer apply — only
        the dataset-level summary + methods do. Grey the rest out to make that clear."""
        curated = self.book_source_combo.currentData() == "curated"
        for key in ("gallery", "spectra", "segmentation", "stats"):
            self.sec_chks[key].setEnabled(not curated)
        self.sec_label.setText("Sections (summary / methods apply; contents come from the "
                               "Report tab)" if curated else "Sections")

    def _pixel_size_changed(self, _value):
        """Write the edited pixel size back onto the dataset so every renderer (preview,
        export, crop studio) uses it, then refresh the inaccuracy caution. 0 clears it."""
        ds = getattr(self.win, "ds", None)
        if ds is None or not hasattr(ds, "set_pixel_size"):
            return
        v = float(self.pixsize_spin.value())
        ds.set_pixel_size(v if v > 0 else None)
        self._refresh_pixsize_warning()

    def _refresh_pixsize_warning(self):
        """Show the dataset's scale-bar caution (non-square pixels / non-unit grid), if any."""
        ds = getattr(self.win, "ds", None)
        msg = ds.pixel_size_warning() if (ds is not None and hasattr(ds, "pixel_size_warning")) else None
        self.pixsize_warn.setText(f"⚠ Scale bar may be inaccurate: {msg}." if msg else "")
        self.pixsize_warn.setVisible(bool(msg))
        # reflect the automatic length so the drawn bar is visible/verifiable, never typed
        from ..annotations import format_scalebar_label
        sb = ds.auto_scale_bar_um() if (ds is not None and hasattr(ds, "auto_scale_bar_um")) else None
        self.scalebar_lbl.setText(f"auto · {format_scalebar_label(sb)}" if sb
                                  else "unavailable — set a pixel size")

    def _bundle_opts(self):
        return {k: c.isChecked() for k, c in self.bundle_chks.items()}

    def _bundle_toggled(self, on):
        self.bundle_box.setEnabled(on)
        self.bundle_preview.setEnabled(on)
        self._refresh_bundle_preview()

    def _refresh_bundle_preview(self):
        if not self.chk_bundle.isChecked():
            self.bundle_preview.setPlainText("Data bundle off — only the PDF will be written.")
            return
        self.bundle_preview.setPlainText(export.bundle_preview(self._bundle_opts()))

    def _set_design_row_visible(self, field, label, visible):
        """Show/hide a whole ``QFormLayout`` row (label + field) in the Image-design form.

        Uses ``setRowVisible`` where the Qt build supports it (collapses the row gap too);
        falls back to toggling the two widgets directly on older PySide6.
        """
        form = getattr(self, "_design_form", None)
        if form is not None and hasattr(form, "setRowVisible"):
            try:
                form.setRowVisible(field, visible)
                return
            except (TypeError, RuntimeError):
                pass
        field.setVisible(visible)
        if label is not None:
            label.setVisible(visible)

    def _scope_changed(self, *_):
        key = self.scope_combo.currentData()
        kind = dict((k, kind) for k, _l, kind in SCOPES)[key]
        prev_fmt = self.fmt_combo.currentData()        # keep the chosen format if still offered
        self.fmt_combo.clear()
        table = {"image": IMAGE_EXT, "table": TABLE_EXT, "stats": STATS_EXT, "book": BOOK_EXT}[kind]
        for label, ext in table:
            self.fmt_combo.addItem(label, ext)
        if prev_fmt is not None:
            i = self.fmt_combo.findData(prev_fmt)
            if i >= 0:
                self.fmt_combo.setCurrentIndex(i)
        self.design.setVisible(kind == "image")
        # Colormap only styles single-ion intensity images; the overlay (per-feature hues) and
        # segmentation map (per-cluster colours) ignore it, so hide the row for them to avoid the
        # "exporting colour overlay but colormap says viridis" confusion.
        cmap_applies = key in ("ion", "gallery", "component")
        self._set_design_row_visible(self.cmap_combo, self.cmap_label, cmap_applies)
        self.feat_group.setVisible(key == "gallery")  # batch feature picker is gallery-only
        self.chk_roi.setVisible(key == "overlay")    # region outline applies to the composite only
        spec_scope = key in ("specfig", "specdata")  # source picker + m/z range apply to both
        self.spec_group.setVisible(spec_scope)
        self.spec_axes.setVisible(spec_scope)
        self.book.setVisible(kind == "book")
        if kind == "book":
            self._book_source_changed()
        # gallery/seg don't use a per-feature spectrum corner the same way; keep options but hint
        hints = {
            "ion": "Clean ion image with a small labelled intensity scale in a corner; the mean spectrum sits in a margin beside it.",
            "overlay": "Additive multi-ion composite (co-located ions mix, red+green→yellow) with per-channel colour ramps in a corner. Channels auto-take clean primary hues unless you've set a feature's colour. Tick 'Region outline' to trace your ROIs.",
            "component": "The active PCA/NMF component's score image, styled with the preset and "
                         "optionally cropped to a region (pick it under 'Crop to region'). Choose "
                         "the component on the Components tab first.",
            "gallery": "Batch-generate one designed image per ticked feature into a folder you choose. "
                       "Set the photo area once with 'Crop to region' above — it's applied to every image.",
            "specfig": "Publication spectra — pick the whole-slide mean and/or per-region means, "
                       "overlaid in one figure or one file each, with an optional A − B difference.",
            "specdata": "Raw m/z + intensity columns for the picked spectra (one column per "
                        "spectrum overlaid, or one file each), with an optional A − B difference.",
            "features": "The annotated feature list (m/z, lipid, class, adduct, ppm, confidence…).",
            "stats": "The region A-vs-B comparison: a formatted Excel report (charts + methods) or raw CSV.",
            "seg": "The segmentation map coloured by cluster / named region.",
            "book": "A single multi-page PDF: cover, dataset, methods + provenance + references, "
                    "ion-image gallery, spectra, segmentation, and statistics.",
        }
        self.hint.setText(hints.get(key, ""))
        # disable scopes that have no data
        ok, why = self.win.export_available(key)
        self.b_export.setEnabled(ok)
        if not ok:
            self.hint.setText(why)
        self._set_preview_visible(ok and key in PREVIEW_SCOPES)
        self._schedule_preview()

    def _initial_export_window(self):
        """Seed the export intensity window from the active feature's dock window (so the
        override starts where the live view is), falling back to the full 0–100 range."""
        try:
            p = self.win._peak_for_mz(self.win.active_mz)
            return self.win._feature_window(p)
        except Exception:  # noqa: BLE001 — no active feature / not picked yet
            return 0.0, 100.0

    def options(self):
        theme = "light" if self.theme_combo.currentText().startswith("Light") else "dark"
        # Annotations live in the bottom footer band now; only the legend's side is a choice
        # (scale bar takes the opposite side). auto_corner=False so the side is honoured.
        legend_left = self.corner_combo.currentText().endswith("left")
        corner = "lower left" if legend_left else "lower right"
        auto_corner = False
        return {
            "fmt": self.fmt_combo.currentData(),
            "theme": theme,
            "cmap": self.cmap_combo.currentText(),
            "corner": corner,
            "auto_corner": auto_corner,
            "label_mode": LABEL_CHOICES[self.label_combo.currentIndex()][1],
            "card": self.chk_card.isChecked(),
            "spectrum": self.chk_spec.isChecked(),
            "colorbar": self.chk_cbar.isChecked(),
            "scalebar_on": self.chk_scale.isChecked(),
            "title": self.chk_title.isChecked(),
            "roi_outline": self.chk_roi.isChecked(),
            "dpi": int(self.dpi_spin.value()),
            "width": float(self.width_spin.value()),
            "window_override": (self.window_slider.values()
                                if self.chk_window.isChecked() else None),
            "crop_region": self.crop_combo.currentData(),
            "crop_outline": self.chk_crop_outline.isChecked(),
            "batch_peaks": self._selected_peaks(),
            "spec_xauto": self.chk_xauto.isChecked(),
            "spec_xmin": float(self.xmin_spin.value()),
            "spec_xmax": float(self.xmax_spin.value()),
            "spec_peaks": self.chk_peaks.isChecked(),
            "spec_active": self.chk_active.isChecked(),
            "spec_style": self.spec_style.currentData(),
            "spec_yshare": self.chk_yshare.isChecked(),
            "spec_sources": self._selected_spec_idents(),
            "spec_layout": self.spec_layout.currentData(),
            "spec_diff": self.chk_diff.isChecked(),
            "spec_diff_a": self.diff_a.currentData(),
            "spec_diff_b": self.diff_b.currentData(),
            "sections": {k: c.isChecked() for k, c in self.sec_chks.items()},
            "book_source": self.book_source_combo.currentData(),
            "bundle": self.chk_bundle.isChecked(),
            "bundle_opts": self._bundle_opts(),
            "style_spec": self._style_spec_for_render(),
        }

    # ----- style-preset plumbing ------------------------------------------- #
    def _refresh_preset_combo(self, select=None):
        """(Re)fill the preset combo from the library, keeping the current selection (or
        ``select``). Signals are blocked so refilling doesn't fire ``_apply_preset``."""
        want = select or (self.preset_combo.currentText() if self.preset_combo.count() else "App Default")
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for nm in stylelib.names():
            self.preset_combo.addItem(nm)
        i = self.preset_combo.findText(want)
        self.preset_combo.setCurrentIndex(i if i >= 0 else 0)
        self.preset_combo.blockSignals(False)

    def _current_style_spec(self):
        return stylelib.get(self.preset_combo.currentText())

    def _style_spec_for_render(self):
        """The preset's recipe as a dict for :func:`export.active_style`, **minus the dimensions
        the dialog owns with its own widgets** (theme + image colormap) so those never fight the
        theme/colormap pickers. Everything else (fonts, type sizes, weights, palette, accent)
        rides along. ``None`` for the no-op 'App Default'."""
        spec = self._current_style_spec()
        if spec is None or spec.is_default():
            return None
        d = spec.to_dict()
        d.pop("theme", None)
        d.pop("image_cmap", None)
        return d if len(d) > 1 else None      # >1 because 'name' is always present

    def _apply_preset(self, *_):
        """Reflect the chosen look's theme + colormap into the dialog's own pickers so what you
        see matches what will render; the rest of the look rides along via ``style_spec``."""
        spec = self._current_style_spec()
        if spec is None:
            return
        if spec.theme:
            self._set_combo_text(self.theme_combo,
                                 "Light" if spec.theme in ("light", "print", "paper") else "Dark glass")
        if spec.image_cmap:
            i = self.cmap_combo.findText(spec.image_cmap)
            if i < 0:
                self.cmap_combo.addItem(spec.image_cmap)
                i = self.cmap_combo.findText(spec.image_cmap)
            if i >= 0:
                self.cmap_combo.setCurrentIndex(i)

    def _copy_recipe(self):
        spec = self._current_style_spec()
        if spec is None:
            return
        QtWidgets.QApplication.clipboard().setText(spec.to_json())
        self.win.statusBar().showMessage(f"Copied recipe for “{spec.name}” to the clipboard.")

    def _import_recipe(self):
        clip = QtWidgets.QApplication.clipboard().text()
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Import style recipe",
            "Paste a style recipe (JSON). It will be saved as a look you can reuse:", clip)
        if not ok or not text.strip():
            return
        try:
            spec = stylelib.import_json(text)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Import failed",
                                          f"That doesn't look like a valid recipe:\n{e}")
            return
        self._refresh_preset_combo(select=spec.name)
        self._apply_preset()
        self.win.statusBar().showMessage(f"Imported look “{spec.name}”.")

    # ----- remember the last-used selections across sessions --------------- #
    PREFS_KEY = "export_options"

    def _save_prefs(self):
        """Persist the design/scope selections (not the dataset-specific ones — crop region,
        ticked features and the m/z window depend on the loaded sample). Colormap is left out
        on purpose: in the dialog it mirrors the live ion-view colormap each time."""
        prefs.set(self.PREFS_KEY, {
            "scope": self.scope_combo.currentData(),
            "fmt": self.fmt_combo.currentData(),
            "theme": self.theme_combo.currentText(),
            "label": self.label_combo.currentIndex(),
            "corner": self.corner_combo.currentText(),
            "card": self.chk_card.isChecked(),
            "spectrum": self.chk_spec.isChecked(),
            "colorbar": self.chk_cbar.isChecked(),
            "scalebar_on": self.chk_scale.isChecked(),
            "title": self.chk_title.isChecked(),
            "roi_outline": self.chk_roi.isChecked(),
            "crop_outline": self.chk_crop_outline.isChecked(),
            "dpi": int(self.dpi_spin.value()),
            "width": float(self.width_spin.value()),
            "window_override_on": self.chk_window.isChecked(),
            "window_override": list(self.window_slider.values()),
            "spec_xauto": self.chk_xauto.isChecked(),
            "spec_peaks": self.chk_peaks.isChecked(),
            "spec_active": self.chk_active.isChecked(),
            "spec_style": self.spec_style.currentData(),
            "spec_yshare": self.chk_yshare.isChecked(),
            "spec_layout": self.spec_layout.currentData(),
            "sections": {k: c.isChecked() for k, c in self.sec_chks.items()},
            "book_source": self.book_source_combo.currentData(),
            "bundle": self.chk_bundle.isChecked(),
            "bundle_opts": self._bundle_opts(),
            "style_preset": self.preset_combo.currentText(),
        })

    def _restore_prefs(self):
        """Apply the last export's selections. Best-effort and per-key — a stale or missing
        value from an older build is skipped rather than breaking dialog construction."""
        d = prefs.get(self.PREFS_KEY) or {}
        if not isinstance(d, dict) or not d:
            return
        try:
            if "scope" in d:                       # scope rebuilds the format list + groups
                i = self.scope_combo.findData(d["scope"])
                if i >= 0:
                    self.scope_combo.setCurrentIndex(i)
            self._scope_changed()                  # ensure fmt_combo is populated for findData
            if "fmt" in d:
                i = self.fmt_combo.findData(d["fmt"])
                if i >= 0:
                    self.fmt_combo.setCurrentIndex(i)
            self._set_combo_text(self.theme_combo, d.get("theme"))
            self._set_combo_text(self.corner_combo, d.get("corner"))
            if isinstance(d.get("label"), int) and 0 <= d["label"] < self.label_combo.count():
                self.label_combo.setCurrentIndex(d["label"])
            for key, chk in (("card", self.chk_card), ("spectrum", self.chk_spec),
                             ("colorbar", self.chk_cbar), ("scalebar_on", self.chk_scale),
                             ("title", self.chk_title), ("roi_outline", self.chk_roi),
                             ("crop_outline", self.chk_crop_outline),
                             ("bundle", self.chk_bundle), ("spec_xauto", self.chk_xauto),
                             ("spec_peaks", self.chk_peaks),
                             ("spec_active", self.chk_active),
                             ("spec_yshare", self.chk_yshare)):
                if key in d:
                    chk.setChecked(bool(d[key]))
            if "spec_style" in d:
                i = self.spec_style.findData(d["spec_style"])
                if i >= 0:
                    self.spec_style.setCurrentIndex(i)
            if "dpi" in d:
                self.dpi_spin.setValue(int(d["dpi"]))
            if "width" in d:
                self.width_spin.setValue(float(d["width"]))
            if "spec_layout" in d:
                i = self.spec_layout.findData(d["spec_layout"])
                if i >= 0:
                    self.spec_layout.setCurrentIndex(i)
            if "window_override_on" in d:
                self.chk_window.setChecked(bool(d["window_override_on"]))
            wo = d.get("window_override")
            if isinstance(wo, (list, tuple)) and len(wo) == 2:
                self.window_slider.setValues(float(wo[0]), float(wo[1]))
            for key, chk in self.sec_chks.items():
                if key in (d.get("sections") or {}):
                    chk.setChecked(bool(d["sections"][key]))
            for key, chk in self.bundle_chks.items():
                if key in (d.get("bundle_opts") or {}):
                    chk.setChecked(bool(d["bundle_opts"][key]))
            if "book_source" in d:
                i = self.book_source_combo.findData(d["book_source"])
                if i >= 0:
                    self.book_source_combo.setCurrentIndex(i)
            if d.get("style_preset"):
                self._refresh_preset_combo(select=d["style_preset"])
                self._apply_preset()
        except Exception:  # noqa: BLE001 — restoring prefs must never break the dialog
            import traceback
            traceback.print_exc()

    @staticmethod
    def _set_combo_text(combo, text):
        if text:
            i = combo.findText(text)
            if i >= 0:
                combo.setCurrentIndex(i)

    def _open_studio(self):
        """Close the single-subject hub and open the multi-section Export Studio."""
        self.accept()
        self.win.open_export_studio()

    def _do_export(self):
        key = self.scope_combo.currentData()
        kind = dict((k, kd) for k, _l, kd in SCOPES).get(key)
        # a data bundle with every file unticked would silently write an empty folder
        if kind == "book" and self.chk_bundle.isChecked() and not any(self._bundle_opts().values()):
            self.win.statusBar().showMessage(
                "Pick at least one data-bundle file, or turn the data bundle off.")
            self.bundle_preview.setPlainText(
                "⚠ No bundle files selected — tick at least one above, or turn the data "
                "bundle off to write only the PDF.")
            return
        self._save_prefs()                 # remember these selections for next time
        if self.win.run_export(key, self.options()):
            self.accept()


# --------------------------------------------------------------------------- #
# Mixin — lives on MainWindow
# --------------------------------------------------------------------------- #
class ExportMixin:
    # ----- entry points ---------------------------------------------------- #
    def open_export_hub(self, scope=None):
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        dlg = ExportDialog(self, scope=scope)
        dlg.exec()

    def export_available(self, scope):
        """(ok, reason) — whether ``scope`` has data to export right now."""
        if self.ds is None:
            return False, "Load a dataset first."
        if scope in ("ion",) and self.active_mz is None:
            return False, "Select a feature (active m/z) first — click a peak or table row."
        if scope in ("overlay", "gallery", "features") and not self.peaks:
            return False, "Find peaks first to build the feature set."
        if scope == "component" and getattr(getattr(self, "_comp", None), "images", None) in (None, []):
            return False, "Run PCA or NMF first (Components tab) — embeddings have no score image."
        if scope == "stats" and getattr(self, "last_stats", None) is None:
            return False, "Run a region/ROI comparison first (Statistics tab)."
        if scope == "seg" and getattr(self, "seg", None) is None:
            return False, "Run segmentation first (Segmentation tab)."
        return True, ""

    # ----- dispatch -------------------------------------------------------- #
    def run_export(self, scope, opts):
        """Perform the export. Returns True on success/initiated (closes the hub)."""
        self._export_pending = []                          # paths saved this run (via _save_path)
        try:
            handler = {
                "ion": self._export_ion, "overlay": self._export_overlay,
                "component": self._export_component,
                "gallery": self._export_gallery, "specfig": self._export_spectrum_fig,
                "specdata": self._export_spectra_data, "features": self._export_feature_table,
                "stats": self._export_stats, "seg": self._export_segmentation,
                "book": self._export_book,
            }[scope]
            # The chosen Style preset is applied to every figure this handler renders in-process
            # (ion / overlay / component / gallery / spectrum / segmentation). Handlers that
            # render on a worker thread or process (book, studio) re-enter the context there.
            with export.active_style(opts.get("style_spec")):
                ok = bool(handler(opts))
            if ok and getattr(self, "prov", None) is not None:   # log outputs for the methods report
                for p in self._export_pending:
                    self.prov.output(p, description=f"{scope} export")
            if ok and scope != "book":               # mirror the export into the Report tab log
                try:
                    self._log_export_to_report(scope, opts)
                except Exception:  # noqa: BLE001 — logging must never break an export
                    import traceback
                    traceback.print_exc()
            return ok
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")
            return False

    def export_preview(self, scope, opts):
        """``(figure, caption)`` for the hub's preview: the first file the export would write,
        drawn by the same render code at a screen resolution. ``(None, reason)`` when there is
        nothing to draw. Never opens the Crop Studio — an unsaved crop previews as the region's
        bounding box."""
        opts = {**opts, "dpi": PREVIEW_DPI}
        targets = self._crop_targets(opts)
        rg = targets[0]
        many = f"{rg['name']} — 1 of {len(targets)} files, one per region" if len(targets) > 1 else ""
        with export.active_style(opts.get("style_spec")):
            if scope == "ion":
                return self._ion_render(opts)(rg), many
            if scope == "overlay":
                render = self._overlay_render(opts)
                return (render(rg), many) if render else (None, "No visible features to overlay.")
            if scope == "component":
                made = self._component_render(opts)
                return (made[0](rg), many) if made else (None, "Run PCA or NMF first.")
            if scope == "seg":
                return self._seg_render(opts)(rg), many
            if scope == "gallery":
                peaks = opts.get("batch_peaks") or self._visible_peaks()
                if not peaks:
                    return None, "No features ticked."
                p = peaks[0]
                n = len(peaks) * len(targets)
                return (self._gallery_figure(
                    p, rg, self._crop_kwargs(rg, opts.get("crop_outline", True)), opts),
                        f"m/z {float(p['mz']):.4f} — 1 of {n} images")
            if scope == "specfig":
                spectra = self._gather_spectra(opts)
                if not spectra:
                    return None, "No spectra ticked."
                render = self._spectrum_render(opts, spectra)
                if opts.get("spec_layout") == "separate":
                    return render(spectra[0]), f"{spectra[0][0]} — 1 of {len(spectra)} files"
                return render(None), ""
        return None, ""

    # ----- shared helpers -------------------------------------------------- #
    def _save_path(self, title, default, fmt):
        flt = f"{fmt.upper()} (*.{fmt})"
        path, _ = filedialogs.get_save_file_name(self, title, default, flt)
        if path and not path.lower().endswith("." + fmt):
            path += "." + fmt
        if path:                                           # remembered so run_export can log it
            self._export_pending = getattr(self, "_export_pending", [])
            self._export_pending.append(path)
        return path

    def _mean_spec_xy(self):
        """(axis, intensity) of the mean spectrum, cached for the export session."""
        cache = getattr(self, "_exp_mean_spec", None)
        if cache is None and self.ds is not None:
            axis, spec = self.ds.mean_spectrum()
            cache = (np.asarray(axis, float), np.asarray(spec, float))
            self._exp_mean_spec = cache
        return cache

    def _clean_label(self, mz):
        return (self.annotate(mz) or "").split(" (")[0].strip()

    def _subject_mask(self):
        """The tissue footprint (H×W boolean) so annotations land beside the subject; cached
        for the export session. None when there's no dataset."""
        m = getattr(self, "_exp_subject_mask", None)
        if m is None and self.ds is not None:
            m = np.isfinite(self.ds.to_image(np.ones(self.ds.n_pixels), fill=np.nan))
            self._exp_subject_mask = m
        return m

    def _design_kwargs(self, opts):
        """The subset of options that styles a rendered panel."""
        px = getattr(self.ds, "pixel_size_um", None)
        # length is automatic; the checkbox only toggles visibility (never a hand-typed size)
        sb = self.ds.auto_scale_bar_um() if (opts["scalebar_on"] and self.ds is not None) else None
        return dict(cmap=opts["cmap"], theme=opts["theme"], overlay_corner=opts["corner"],
                    overlay=opts["card"], show_spectrum=opts["spectrum"],
                    show_colorbar=opts["colorbar"], show_scalebar=opts["scalebar_on"],
                    show_title=opts["title"], width_in=opts["width"], dpi=opts["dpi"],
                    pixel_size_um=px, scale_bar_um=sb,
                    label_mode=opts.get("label_mode", annot.LABEL_MZ_PPM),
                    auto_corner=opts.get("auto_corner", True),
                    subject_mask=self._subject_mask())

    def _export_feature_window(self, p, opts):
        """Contrast window (% of the hotspot clip) to render feature ``p`` with: the export
        dialog's shared override when 'Override intensity window' is on, else the feature's
        own per-feature window set in the dock."""
        ov = (opts or {}).get("window_override")
        return tuple(ov) if ov is not None else self._feature_window(p)

    def _panel_kwargs(self, p, opts, include_spectrum=True):
        """Full render_ion_panel kwargs for feature ``p`` from current display settings."""
        mz = p["mz"]
        # A lipid-class feature renders the full composite of its member ions (honoring the
        # raw/balanced weight toggle), matching what the interactive views draw; a plain
        # feature renders its single ion. Previously every export used the representative
        # m/z's single ion, so class composites silently exported as one member.
        img = self._peak_image(p)
        lo, hi = self._export_feature_window(p, opts)   # relative-intensity window (% of clip)
        clip = float(self.contrast_spin.value())      # hotspot-clip percentile = the 100% anchor
        kw = dict(image=img, mz=mz, label=self._clean_label(mz), low=lo, high=hi, window=(lo, hi),
                  clip=clip, ppm=self.ppm, label_override=p.get("label_override"))
        kw.update(self._design_kwargs(opts))
        kw["mean_spectrum"] = self._mean_spec_xy() if include_spectrum else None
        return kw

    # ----- ROI cropping (zoom every image export in on a region) ----------- #
    @staticmethod
    def _safe_name(name):
        """Filesystem-safe filename stem. Delegates to the shared ``studio.safe_name`` so the
        GUI and engine produce identical, Windows-safe paths for the same label (also used by
        the Report tab via the composed window)."""
        return studio.safe_name(name)

    def _crop_kwargs(self, rg, outline=True):
        """Render kwargs that zoom a panel in on region ``rg``: crop to its bounds, trace its
        boundary in its own colour (unless ``outline`` is False), and fade the surround. The whole-slide side spectrum is
        dropped — it isn't region-specific and only shrinks the close-up — so a cropped panel
        is a clean, square-pixelled zoom. ``{}`` for the full image (``rg`` None)."""
        if rg is None:
            return {}
        bbox = self._region_bbox(rg)
        if bbox is None:
            return {}
        kw = {"crop": bbox, "outline_color": rg.get("color", "#ffffff"), "dim_outside": True,
              "show_spectrum": False, "mean_spectrum": None}
        om = self._region_mask_2d(rg)
        if om is not None:
            kw["outline_mask"] = om
            kw["show_outline"] = outline
        return kw

    def _crop_targets(self, opts):
        """The region(s) an image export should be cropped to, from the 'Crop to region'
        choice: ``[None]`` (full image), ``[one region]``, or every visible region."""
        cr = opts.get("crop_region")
        regions = getattr(self, "regions", None) or []
        if not cr:
            return [None]
        if cr == "__each__":
            tgts = [rg for rg in regions
                    if rg.get("visible", True) and self._region_bbox(rg) is not None]
            return tgts or [None]
        rg = next((r for r in regions if r["name"] == cr), None)
        return [rg] if (rg is not None and self._region_bbox(rg) is not None) else [None]

    def _open_crop_studio(self, rg):
        """Open the Crop Studio for region ``rg``; True if the user saved a crop."""
        from .cropstudio import CropStudio
        return CropStudio(self, rg).exec() == QtWidgets.QDialog.Accepted

    def edit_region_crop(self, ri):
        """Re-open the Crop Studio for region index ``ri`` (region list → 'Edit close-up crop…')."""
        if self.ds is None or not (0 <= ri < len(self.regions)):
            return
        rg = self.regions[ri]
        if self._region_bbox_raw(rg) is None:
            self.statusBar().showMessage(f"Region '{rg['name']}' has no pixels to crop to.")
            return
        self._open_crop_studio(rg)

    def _resolve_crop_targets(self, opts):
        """Crop targets for an export, opening the Crop Studio the **first time** a single
        region is exported as a close-up (no saved crop yet). Returns the target list, or
        ``None`` when the user cancels the Studio (so the export aborts cleanly)."""
        targets = self._crop_targets(opts)
        if len(targets) == 1 and targets[0] is not None:
            rg = targets[0]
            has_crop = bool(rg.get("crop")) and \
                rg.get("crop_orient", self.ds.orientation) == self.ds.orientation
            if not has_crop and not self._open_crop_studio(rg):
                return None
        return targets

    def _save_targets(self, opts, targets, render, base_name, title):
        """Render+save ``render(rg) -> Figure`` for each crop target: one chosen file for a
        single target (region name suffixed), or a chosen folder with one file per region
        when cropping to every region. Returns True on success."""
        if len(targets) > 1:
            d = filedialogs.get_existing_directory(self, f"{title} — choose a folder")
            if not d:
                return False
            n = 0
            for rg in targets:
                fig = render(rg)
                nm = f"{base_name}_{self._safe_name(rg['name'])}.{opts['fmt']}"
                export.save_figure(fig, os.path.join(d, nm), dpi=opts["dpi"])
                n += 1
                self.statusBar().showMessage(f"Region close-ups… {n}/{len(targets)}")
                QtWidgets.QApplication.processEvents()
            self._export_pending.append(d)
            self.statusBar().showMessage(f"Wrote {n} region close-ups to {d}")
            return True
        rg = targets[0]
        suffix = f"_{self._safe_name(rg['name'])}" if rg is not None else ""
        path = self._save_path(title, f"{base_name}{suffix}.{opts['fmt']}", opts["fmt"])
        if not path:
            return False
        export.save_figure(render(rg), path, dpi=opts["dpi"])
        self.statusBar().showMessage(f"Wrote {path}")
        return True

    def _region_intensity_table(self):
        """Per-region intensity of each visible feature — the numbers behind the ion images,
        for the data bundle. Columns: region, mz, lipid, n_px, mean, median, std. ``None``
        when there are no regions/features."""
        import pandas as pd
        regions = [rg for rg in (getattr(self, "regions", None) or [])
                   if rg.get("visible", True) and self._region_pixel_mask(rg) is not None]
        if not regions or not self.peaks:
            return None
        rows = []
        for p in self._visible_peaks():
            vec = self.ds.ion_vector(p["mz"], tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)
            lab = self._clean_label(p["mz"])
            for rg in regions:
                m = np.asarray(self._region_pixel_mask(rg), bool)
                v = vec[m]
                if not v.size:
                    continue
                rows.append({"region": rg["name"], "mz": round(float(p["mz"]), 4), "lipid": lab,
                             "n_px": int(v.size), "mean": round(float(np.nanmean(v)), 4),
                             "median": round(float(np.nanmedian(v)), 4),
                             "std": round(float(np.nanstd(v)), 4)})
        return pd.DataFrame(rows) if rows else None

    # ----- per-scope handlers ---------------------------------------------- #
    def _ion_render(self, opts):
        p = self._peak_for_mz(self.active_mz) or {"mz": self.active_mz}
        base = self._panel_kwargs(p, opts)

        def render(rg):
            kw = {**base, **self._crop_kwargs(rg, opts.get("crop_outline", True))}
            if rg is not None:
                kw["title"] = f"{rg['name']} · m/z {float(self.active_mz):.4f}"
            return export.render_ion_panel(**kw)
        return render

    def _export_ion(self, opts):
        render = self._ion_render(opts)
        targets = self._resolve_crop_targets(opts)
        if targets is None:
            return False
        return self._save_targets(opts, targets, render,
                                  f"ion_{self.active_mz:.4f}", "Save ion image")

    def _export_component(self, opts):
        """Export the active PCA/NMF component's score image — styled with the chosen preset
        and optionally cropped to a region. Score images aren't single-ion intensities, so
        the relative-% legend and side spectrum are dropped; colormap, scale bar, theme,
        crop and ROI outline all apply. The component's identity goes in the filename."""
        made = self._component_render(opts)
        if made is None:
            self.statusBar().showMessage("Run PCA or NMF first (Components tab).")
            return False
        render, c, idx = made
        targets = self._resolve_crop_targets(opts)
        if targets is None:
            return False
        return self._save_targets(opts, targets, render,
                                  f"{c.method.lower()}_component{idx}",
                                  f"Save {c.method} component image")

    def _component_render(self, opts):
        """``(render, components, index)`` for the active component, or None before PCA/NMF."""
        c = getattr(self, "_comp", None)
        if c is None or not getattr(c, "images", None):
            return None
        idx = max(0, min(int(getattr(self, "_comp_active", 0)), len(c.images) - 1))
        image = np.asarray(c.images[idx], float)
        ev = (f" · EV {c.explained_variance[idx]:.0%}"
              if len(getattr(c, "explained_variance", [])) else "")
        dkw = self._design_kwargs(opts)
        dkw.update(show_colorbar=False, show_spectrum=False)   # not an ion intensity

        def render(rg):
            kw = dict(image=image, mz=None, label="", mean_spectrum=None)
            kw.update(dkw)
            kw.update(self._crop_kwargs(rg, opts.get("crop_outline", True)))
            name = f"{rg['name']} · " if rg is not None else ""
            kw["title"] = f"{name}{c.method} component {idx}{ev}"
            return export.render_ion_panel(**kw)
        return render, c, idx

    def _export_overlay(self, opts):
        render = self._overlay_render(opts)
        if render is None:
            self.statusBar().showMessage("No visible features to overlay.")
            return False
        if opts.get("roi_outline") and not self._overlay_outlines():
            self.statusBar().showMessage("No visible regions to outline — exporting without one.")
        targets = self._resolve_crop_targets(opts)
        if targets is None:
            return False
        return self._save_targets(opts, targets, render, "overlay", "Save colour overlay")

    def _overlay_render(self, opts):
        """``render(rg)`` for the colour overlay, or None when no feature is visible."""
        peaks = self._overlay_peaks() or self._visible_peaks()
        if not peaks:
            return None
        chans = self._overlay_channels(peaks)
        rgb = self._compose_overlay_rgb(chans, opts)
        channels = []
        clip = float(self.contrast_spin.value())          # hotspot-clip percentile = the 100% anchor
        for p, hexc in chans:
            lo, hi = self._export_feature_window(p, opts)
            # extract on the apex (display centre) but label with the catalogued m/z,
            # so the exported overlay matches the interactive one (see _peak_image/_apex_mz)
            img = self.ds.ion_image(self._apex_mz(float(p["mz"])), tol_ppm=self.ppm,
                                    reduce=self.reduce, norm=self.norm)
            channels.append(dict(mz=float(p["mz"]), color=hexc, ppm=float(self.ppm),
                                 lo=float(lo), hi=float(hi),
                                 peak=imaging.relative_max(img, clip),
                                 label=self._clean_label(p["mz"]),
                                 label_override=p.get("label_override")))
        base_outlines = self._overlay_outlines() if opts.get("roi_outline") else []
        dk = self._design_kwargs(opts)
        dk.pop("crop", None)                               # crop is per-target below

        def render(rg):
            outs = list(base_outlines)
            crop = None
            if rg is not None:
                crop = self._region_bbox(rg)
                om = self._region_mask_2d(rg) if opts.get("crop_outline", True) else None
                if om is not None:
                    outs.append((om, rg.get("color", "#ffffff")))
            return export.render_overlay_panel(
                rgb, channels, outlines=(outs or None), crop=crop,
                show_legend=bool(opts["card"] and opts["colorbar"]), **dk)
        return render

    def _overlay_channels(self, peaks):
        """[(peak, hexcolor)] — each overlay channel's colour: the feature's explicit colour
        when the user set one, else the next clean additive-primary hue. Additive primaries
        blend in additive mode (red+green→yellow…)."""
        out, auto = [], 0
        for p in peaks:
            c = p.get("color")
            if c:
                out.append((p, c))
            else:
                out.append((p, OVERLAY_CHANNEL_COLORS[auto % len(OVERLAY_CHANNEL_COLORS)]))
                auto += 1
        return out

    def _compose_overlay_rgb(self, chans, opts=None):
        """Additive colour composite (uint8 RGB): each feature stretched by its
        own intensity window, tinted by its channel colour, and summed so co-located ions mix.
        ``chans`` is the ``[(peak, hexcolor)]`` from :meth:`_overlay_channels`; ``opts`` carries
        the export's shared window override (if any)."""
        from PySide6 import QtGui
        acc = np.zeros((self.ds.height, self.ds.width, 3), float)
        for p, hexc in chans:
            img = self._peak_image(p)   # class channel → its full composite, not one member
            norm = self._windowed(img, *self._export_feature_window(p, opts))
            col = QtGui.QColor(hexc)
            acc[..., 0] += norm * col.redF()
            acc[..., 1] += norm * col.greenF()
            acc[..., 2] += norm * col.blueF()
        return (np.clip(acc, 0, 1) * 255).astype(np.uint8)

    def _overlay_outlines(self):
        """[(mask2d, color)] region boundaries for the colour overlay — every visible region,
        traced in its own distinct colour. Empty if none."""
        outs = []
        for rg in (getattr(self, "regions", None) or []):
            if not rg.get("visible", True):
                continue
            m = self._region_pixel_mask(rg)
            if m is None:
                continue
            m2d = self.ds.to_image(np.asarray(m, dtype=float), fill=0.0) > 0.5
            if m2d.any():
                outs.append((m2d, rg.get("color", "#ff3b30")))
        return outs

    def _export_gallery(self, opts):
        peaks = opts.get("batch_peaks") or self._visible_peaks()   # ticked features, else visible
        if not peaks:                                     # else we'd make an empty folder + "Wrote 0"
            self.statusBar().showMessage("No features to export — find peaks (or tick some) first.")
            return False
        targets = self._resolve_crop_targets(opts)        # full image, one region, or each region
        if targets is None:
            return False
        d = filedialogs.get_existing_directory(self, "Choose a folder for the gallery")
        if not d:
            return False
        n, total = 0, len(peaks) * len(targets)
        for rg in targets:
            sub = d if rg is None else os.path.join(d, self._safe_name(rg["name"]))
            if rg is not None:
                os.makedirs(sub, exist_ok=True)
            crop_kw = self._crop_kwargs(rg, opts.get("crop_outline", True))
            for p in peaks:
                mz = p["mz"]
                fig = self._gallery_figure(p, rg, crop_kw, opts)
                # label_override first so a renamed feature / lipid-class composite (whose
                # representative m/z has no DB hit) names its file by its class, not "feature".
                label = p.get("label_override") or self._clean_label(mz) or "feature"
                name = f"ion_{mz:.4f}_{self._safe_name(label)}"
                export.save_figure(fig, os.path.join(sub, name + "." + opts["fmt"]), dpi=opts["dpi"])
                n += 1
                self.statusBar().showMessage(f"Gallery… {n}/{total}")
                QtWidgets.QApplication.processEvents()
        self._export_pending.append(d)
        self.statusBar().showMessage(f"Wrote {n} images to {d}")
        return True

    def _gallery_figure(self, p, rg, crop_kw, opts):
        mz = p["mz"]
        kw = {**self._panel_kwargs(p, opts), **crop_kw}
        if rg is not None:
            kw["title"] = f"{rg['name']} · m/z {mz:.4f}  {self._clean_label(mz)}".strip()
        return export.render_ion_panel(**kw)

    def _resolve_spec_source(self, ident):
        """``(name, axis, y, color)`` for a spectra-export source id — ``'__mean__'`` (the
        whole-slide mean) or a region name. ``None`` if the region resolves to no pixels."""
        if ident == "__mean__":
            ax, y = self._mean_spec_xy()
            return ("whole-slide mean", np.asarray(ax, float), np.asarray(y, float), "#5aa9e6")
        rg = next((r for r in (getattr(self, "regions", None) or [])
                   if r.get("name") == ident), None)
        if rg is None:
            return None
        mask = self._region_pixel_mask(rg)
        if mask is None or not np.asarray(mask, bool).any():
            return None
        ax, y = self.ds.mean_spectrum(mask=np.asarray(mask, bool))
        return (rg["name"], np.asarray(ax, float), np.asarray(y, float),
                rg.get("color", "#f4a259"))

    def _difference_spectrum(self, ident_a, ident_b):
        """``(name, axis, y, color)`` of mean(A) − mean(B). Both share the global m/z axis,
        so the subtraction is element-wise. ``None`` if either side is empty."""
        sa = self._resolve_spec_source(ident_a) if ident_a else None
        sb = self._resolve_spec_source(ident_b) if ident_b else None
        if sa is None or sb is None:
            return None
        n = min(len(sa[1]), len(sa[2]), len(sb[2]))
        diff = np.asarray(sa[2][:n], float) - np.asarray(sb[2][:n], float)
        return (f"{sa[0]} − {sb[0]}", np.asarray(sa[1][:n], float), diff, "#e15759")

    def _diff_labels(self, opts):
        """``(a_name, b_name)`` display names for the A − B difference (so the figure can say
        which direction is which), or ``None`` when no difference is being exported."""
        if not opts.get("spec_diff"):
            return None
        sa = self._resolve_spec_source(opts.get("spec_diff_a"))
        sb = self._resolve_spec_source(opts.get("spec_diff_b"))
        return (sa[0], sb[0]) if (sa and sb) else None

    def _gather_spectra(self, opts=None):
        """``[(name, axis, y, color)]`` for the spectra export — built from the dialog's ticked
        sources (whole-slide mean + regions) and, when requested, a region-A − region-B
        difference. With no explicit selection (legacy callers) falls back to the whole-slide
        mean plus the live ROI overlay."""
        opts = opts or {}
        idents = opts.get("spec_sources")
        spectra = []
        if idents:
            spectra = [s for s in (self._resolve_spec_source(i) for i in idents) if s is not None]
        else:
            spectra.append(("whole-slide mean", *self._mean_spec_xy(), "#5aa9e6"))
            try:
                mask = self._roi_mask()
                if mask is not None and mask.any():
                    ax, y = self.ds.mean_spectrum(mask=mask)
                    spectra.append(("ROI spectrum", np.asarray(ax), np.asarray(y), "#ffcc00"))
            except Exception:  # noqa: BLE001
                pass
        if opts.get("spec_diff"):
            d = self._difference_spectrum(opts.get("spec_diff_a"), opts.get("spec_diff_b"))
            if d is not None:
                spectra.append(d)
        return spectra

    @staticmethod
    def _spec_mz_range(opts):
        """The exact ``(lo, hi)`` m/z window when 'Auto' is off, else ``None`` (auto-crop)."""
        if not opts.get("spec_xauto", True):
            xmin, xmax = opts.get("spec_xmin", 0.0), opts.get("spec_xmax", 0.0)
            if xmax > xmin:
                return (float(xmin), float(xmax))
        return None

    def _suffix_path(self, base, name):
        """Insert a sanitized ``name`` before the extension of ``base`` (for one-file-per-
        spectrum exports), registering the result for provenance logging."""
        root, ext = os.path.splitext(base)
        p = f"{root}_{self._safe_name(name)}{ext}"
        self._export_pending = getattr(self, "_export_pending", [])
        self._export_pending.append(p)
        return p

    @staticmethod
    def _crop_spectrum_xy(s, mz_range):
        """Crop a ``(name, axis, y[, color])`` tuple to ``mz_range`` (for the data table)."""
        if not mz_range:
            return s
        ax = np.asarray(s[1], float)
        sel = (ax >= mz_range[0]) & (ax <= mz_range[1])
        return (s[0], ax[sel], np.asarray(s[2], float)[sel]) + tuple(s[3:])

    def _spectrum_render(self, opts, spectra):
        """``render(s)`` for one spectrum's own figure (separate layout), ``render(None)`` for
        every spectrum overlaid in one figure."""
        peaks_mz = ([p["mz"] for p in self._visible_peaks()]
                    if (self.peaks and opts.get("spec_peaks", True)) else None)
        diff_labels = self._diff_labels(opts)
        diff_name = f"{diff_labels[0]} − {diff_labels[1]}" if diff_labels else None
        kw = dict(peaks=peaks_mz, active_mz=(self.active_mz if opts.get("spec_active", False) else None),
                  theme=opts["theme"], width_in=max(opts["width"], 8.0), dpi=opts["dpi"],
                  mz_range=self._spec_mz_range(opts), draw_style=opts.get("spec_style", "line"),
                  # one shared y-scale across every exported spectrum, when asked
                  ylim=(export.spectra_ylim(spectra) if opts.get("spec_yshare") else None))

        def render(s):
            if s is None:
                return export.render_spectrum_figure(spectra, diff_labels=diff_labels, **kw)
            return export.render_spectrum_figure(
                [s], title=str(s[0]), diff_labels=(diff_labels if s[0] == diff_name else None), **kw)
        return render

    def _export_spectrum_fig(self, opts):
        spectra = self._gather_spectra(opts)
        if not spectra:
            self.statusBar().showMessage("No spectra selected to export.")
            return False
        render = self._spectrum_render(opts, spectra)
        fmt = opts["fmt"]
        if opts.get("spec_layout") == "separate":
            base = self._save_path("Save spectrum figures (one per spectrum)",
                                   f"spectrum.{fmt}", fmt)
            if not base:
                return False
            for s in spectra:
                export.save_figure(render(s), self._suffix_path(base, s[0]), dpi=opts["dpi"])
            self.statusBar().showMessage(f"Wrote {len(spectra)} spectrum figures")
            return True
        path = self._save_path("Save spectrum figure", f"spectrum.{fmt}", fmt)
        if not path:
            return False
        export.save_figure(render(None), path, dpi=opts["dpi"])
        self.statusBar().showMessage(f"Wrote {path}")
        return True

    def _export_spectra_data(self, opts):
        spectra = self._gather_spectra(opts)
        if not spectra:
            self.statusBar().showMessage("No spectra selected to export.")
            return False
        mz_range = self._spec_mz_range(opts)
        spectra = [self._crop_spectrum_xy(s, mz_range) for s in spectra]
        fmt = opts["fmt"]
        if opts.get("spec_layout") == "separate":
            base = self._save_path("Save spectra data (one per spectrum)", f"spectra.{fmt}", fmt)
            if not base:
                return False
            for s in spectra:
                export.write_spectra([s[:3]], self._suffix_path(base, s[0]), fmt=fmt)
            self.statusBar().showMessage(f"Wrote {len(spectra)} spectra files")
            return True
        path = self._save_path("Save spectra data", f"spectra.{fmt}", fmt)
        if not path:
            return False
        export.write_spectra([s[:3] for s in spectra], path, fmt=fmt)
        self.statusBar().showMessage(f"Wrote {path}")
        return True

    def _export_feature_table(self, opts):
        df = getattr(self, "feat_df", None)            # full annotated table (Identify lipids)
        if df is None:
            df = self._fallback_feature_df()           # else a basic table from picked peaks
        path = self._save_path("Save feature table",
                               f"{self._current_feature_set_name()}.{opts['fmt']}", opts["fmt"])
        if not path:
            return False
        header = self.audit_header_for_steps(
            ("spatial_feature_finding", "peak_picking"), analysis="Feature table (feature finding)")
        export.write_table(df, path, fmt=opts["fmt"], header_lines=header)
        self.statusBar().showMessage(f"Wrote {path} ({len(df)} features)")
        return True

    def _fallback_feature_df(self):
        import pandas as pd
        rows = []
        for p in self.peaks:
            rows.append({"mz": round(float(p["mz"]), 4), "lipid": self._clean_label(p["mz"]),
                         "rel_intensity": p.get("rel_intensity"), "snr": p.get("snr")})
        return pd.DataFrame(rows)

    def _export_stats(self, opts):
        if getattr(self, "last_stats", None) is None:
            return False
        la, lb = getattr(self, "_region_labels", ("Group A", "Group B"))
        if opts["fmt"] == "xlsx":
            path = self._save_path("Save Excel report", "roi_lipid_report.xlsx", "xlsx")
            if not path:
                return False
            from .. import pipeline  # lazy: keeps pandas/scipy.stats out of GUI startup
            pipeline.build_report(self.last_stats, path, a_label=la, b_label=lb,
                                  mode=self.mode_combo.currentText(),
                                  note="Exact rank-based per-pixel AUC between two regions.",
                                  provenance=getattr(self, "prov", None))
        else:
            path = self._save_path("Save statistics CSV", "roi_statistics.csv", "csv")
            if not path:
                return False
            header = self.audit_header_for_steps(
                ("roi_comparison", "statistics", "multigroup", "region_membership",
                 "class_comparison"), analysis="Region statistics")
            export.write_table(self.last_stats, path, fmt="csv", header_lines=header)
        self.statusBar().showMessage(f"Wrote {path}")
        return True

    def _export_segmentation(self, opts):
        if getattr(self, "seg", None) is None:
            return False
        targets = self._resolve_crop_targets(opts)
        if targets is None:
            return False
        return self._save_targets(opts, targets, self._seg_render(opts), "segmentation",
                                  "Save segmentation map")

    def _seg_render(self, opts):
        colors, _legend = self._seg_colors_legend()
        # honour the dialog's scale-bar checkbox; the renderer auto-sizes the bar to each
        # (possibly cropped) map, so a region close-up gets a correctly-scaled bar too.
        px = getattr(self.ds, "pixel_size_um", None) if opts.get("scalebar_on") else None

        def render(rg):
            title = f"Segmentation · {rg['name']}" if rg is not None else "Segmentation"
            return export.render_segmentation_figure(
                self.seg.label_image, colors, legend=_legend, title=title,
                theme=opts["theme"], width_in=opts["width"], dpi=opts["dpi"], pixel_size_um=px,
                crop=(self._region_bbox(rg) if rg is not None else None))
        return render

    def _seg_colors_legend(self):
        """Map clusters → colours (by named region if any, else by cluster) + a legend."""
        n = self.seg.n_clusters
        regions = getattr(self, "regions", []) or []
        cluster_regions = [r for r in regions
                           if r.get("segments") and r.get("visible", True)]
        if cluster_regions:
            colors, legend = {}, []
            for r in cluster_regions:
                for cl in r["segments"]:
                    colors[int(cl)] = r["color"]
                legend.append((r["color"], r["name"]))
            return colors, legend
        tree = getattr(self.seg, "colors", None) or []   # tree-aware colours when cut from a tree
        colors = {cl: (tree[cl] if cl < len(tree) else PALETTE[cl % len(PALETTE)])
                  for cl in range(n)}
        legend = [(colors[cl], f"cluster {cl}") for cl in range(min(n, 16))]
        return colors, legend

    # ----- the book -------------------------------------------------------- #
    def _document_header(self, opts):
        """The dataset-level scaffold shared by both books: cover meta, summary bullets,
        provenance and the section toggles. The body (auto sections vs the curated item
        list) is appended by the caller."""
        ds = self.ds
        name = os.path.basename(ds.source or "dataset")
        lo, hi = ds.mz_range
        from datetime import datetime
        return {
            "title": os.path.splitext(name)[0] or "MSI analysis",
            "subtitle": "MALDI mass spectrometry imaging — lipid analysis report",
            "cmap": opts["cmap"],
            "meta": {
                "Dataset": name, "Pixels": f"{ds.n_pixels:,} ({ds.width}×{ds.height})",
                "m/z range": f"{lo:.1f}–{hi:.1f}", "Polarity": ds.polarity or self.mode_combo.currentText(),
                "Normalization": self.norm, "Colormap": opts["cmap"],
                "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "dataset_lines": [
                f"{ds.n_pixels:,} pixels ({ds.width}×{ds.height})",
                f"m/z {lo:.2f}–{hi:.2f}",
                f"{ds.polarity or self.mode_combo.currentText()} ion mode",
                f"{self.norm.upper()}-normalized per pixel" if self.norm != "none" else "no normalization",
                f"{len(self.peaks)} detected features" if self.peaks else "no peaks picked yet",
            ],
            "provenance": getattr(self, "prov", None),
            "sections": opts["sections"],
        }

    def _attach_data_tables(self, doc):
        """Feature + per-region intensity tables travel with the doc so the CSV bundle can
        include them (built on the GUI thread — ``_fallback_feature_df`` touches the annotator)."""
        if self.peaks:
            fdf = getattr(self, "feat_df", None)
            doc["feature_table"] = fdf if fdf is not None else self._fallback_feature_df()
            rstats = self._region_intensity_table()
            if rstats is not None:
                doc["region_stats"] = rstats
        return doc

    def _collect_report_document(self, opts):
        """The data book built from the **curated Report-tab list** (ordered ``items``)."""
        doc = self._document_header(opts)
        doc["items"] = self.report_document_items(opts)
        return self._attach_data_tables(doc)

    def _collect_export_document(self, opts):
        sec = opts["sections"]
        doc = self._document_header(opts)
        if sec.get("gallery") and self.peaks:
            peaks = ([self._peak_for_mz(self.active_mz)] if self.active_mz else []) + self._visible_peaks()
            seen, panels = set(), []
            for p in peaks:
                if p is None or p["mz"] in seen:
                    continue
                seen.add(p["mz"])
                panels.append(self._panel_kwargs(p, opts))
                if len(panels) >= 8:
                    break
            doc["panels"] = panels
        if sec.get("spectra"):
            spectra = [("mean spectrum", *self._mean_spec_xy())]
            try:
                ax, y = self.ds.max_spectrum()
                spectra.append(("skyline (max)", np.asarray(ax), np.asarray(y), "#f4a259"))
            except Exception:  # noqa: BLE001
                pass
            doc["spectra"] = spectra
        if sec.get("segmentation") and getattr(self, "seg", None) is not None:
            colors, legend = self._seg_colors_legend()
            doc["segmentation"] = {"image": self.seg.label_image, "colors": colors,
                                   "legend": legend, "title": "Spatial segmentation"}
        if sec.get("stats") and getattr(self, "last_stats", None) is not None:
            la, lb = getattr(self, "_region_labels", ("Group A", "Group B"))
            doc["stats"] = {"df": self.last_stats, "a_label": la, "b_label": lb,
                            "title": "Discriminating lipids (rank AUC)"}
        return self._attach_data_tables(doc)

    def _export_book(self, opts):
        path = self._save_path("Save data book (PDF)", "MSI_report.pdf", "pdf")
        if not path:
            return False
        self.statusBar().showMessage("Building report book… (collecting figures)")
        QtWidgets.QApplication.processEvents()
        curated = opts.get("book_source") == "curated" and bool(getattr(self, "report_items", None))
        doc = self._collect_report_document(opts) if curated else self._collect_export_document(opts)
        folder = (os.path.splitext(path)[0] + "_data") if opts.get("bundle") else None

        bundle_opts = opts.get("bundle_opts")

        style_spec = opts.get("style_spec")

        def job():
            with export.active_style(style_spec):     # style the whole book on the worker thread
                export.build_book(doc, path, theme=opts["theme"], dpi=opts["dpi"])
            if folder:
                export.build_data_bundle(doc, folder, options=bundle_opts)  # CSVs beside the PDF
            return folder

        def done(written_folder):
            msg = f"Wrote {path}"
            if written_folder:
                msg += f"   ·   data bundle → {os.path.basename(written_folder)}/"
            self.statusBar().showMessage(msg)
        # render off the GUI thread (matplotlib OO API is thread-safe; doc holds plain arrays)
        self._run(job, on_done=done,
                  busy=("Building report book + data bundle…" if folder else "Building report book…"))
        return True
