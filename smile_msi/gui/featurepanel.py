"""RightPanelMixin — the right-hand dock.

Consolidates everything that used to be scattered across the *Ion image* sidebar
and the standalone *Feature list* / *Segmentation* tabs into one persistent panel
on the right edge of the window, stacked vertically:

* **Features** — the single working feature list. Highlighting a peak in the
  spectrum (or a row here) *selects* it and offers an **Add to list** button; the
  table also drives ion-image display, identification, saved lists, and the
  class / ratio composite tools.
* **Regions** — named regions built from segmentation clusters (moved off the
  Segmentation tab so they're reachable from any view).

Three things that used to crowd this panel now live next to the view they act on,
so the dock stays Display · Regions · Features: the **optical backdrop** setup is a
once-per-sample dialog (File → Image setup…), the **composite image** tools sit
behind the ion-image toolbar's *Composite…* button, and the **visible-spectra**
trace manager is the spectrum toolbar's *Traces…* button. Their builders
(``_populate_composite_section``, ``_populate_spectra_section``,
``_populate_optical_section``) still live here / in ``optical.py`` and are called
from those dialogs.

All the heavy logic still lives in the other mixins; this module only builds the
widgets (``feat_table``, ``feat_info``, ``feat_set_combo``, ``class_combo``,
``ratio_a/ratio_b``, ``region_list``, ``spectra_list`` …) and adds the
select-and-add interaction.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from . import common
from .common import (PALETTE, MUTED_FG, MUTED_QSS, ControlBar, FlowLayout, RangeSliderField,
                     eye_icon, set_header_tooltips, icon)


class FeatureTable(QtWidgets.QTableWidget):
    """Clicking a row must not scroll sideways. Qt scrolls to the clicked cell, so a click
    on the stretched lipid column, wider than the panel, jumped the view right."""

    def scrollTo(self, index, hint=QtWidgets.QAbstractItemView.EnsureVisible):
        bar = self.horizontalScrollBar()
        x = bar.value()
        super().scrollTo(index, hint)
        bar.setValue(x)


class RegionList(QtWidgets.QListWidget):
    """Region list whose left-most eye glyph toggles visibility on click.

    The eye replaces the old per-row checkbox. Clicking *on the eye* flips the
    region shown/hidden and leaves the selection untouched; clicking anywhere
    else on the row selects it (selection is what 'Find peaks (this region)',
    rename, delete, … act on). Keeping the two gestures separate avoids the
    old trap where a *checked* region read as 'not selected' to Find peaks.

    Rows reorder by dragging — single-row only (a disjoint multi-selection can't be
    expressed as one contiguous move and would desync the list from the model). The
    drop fires :attr:`reordered` once it has fully settled so the controller can
    persist the new order."""

    eyeClicked = QtCore.Signal(int)        # row whose eye was clicked
    reordered = QtCore.Signal()            # emitted (deferred) after a drag-drop reorder

    def startDrag(self, actions):
        # Only single-row drags reorder; with several rows selected (used for batch
        # ops / spectra overlay) don't start a drag at all.
        if len(self.selectedItems()) == 1:
            super().startDrag(actions)

    def dropEvent(self, ev):
        super().dropEvent(ev)              # let Qt perform the internal move
        # Re-derive + persist the order AFTER the drop fully unwinds, so rebuilding the
        # list (which clears it) can't re-enter Qt's in-progress drop machinery.
        QtCore.QTimer.singleShot(0, self.reordered.emit)

    def _eye_hit(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return -1
        rect = self.visualItemRect(item)
        # the eye sits at the far left; give it a comfortable click zone
        if rect.left() <= pos.x() <= rect.left() + self.iconSize().width() + 8:
            return self.row(item)
        return -1

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            r = self._eye_hit(ev.position().toPoint())
            if r >= 0:
                self.eyeClicked.emit(r)
                return                     # toggle only — don't change selection
        super().mousePressEvent(ev)


# Palette-driven styling so the dock reads cleanly in both light and dark themes:
# translucent greys adapt to whatever background the theme uses, and text colours
# come from the active palette rather than hard-coded hex.
_RIGHT_QSS = """
QToolButton#sectionHeader {
    text-align: left;
    padding: 6px 8px;
    border: none;
    border-radius: 5px;
    font-weight: 600;
    background: rgba(128,128,128,0.16);
}
QToolButton#sectionHeader:hover   { background: rgba(128,128,128,0.30); }
QToolButton#sectionHeader:pressed { background: rgba(128,128,128,0.22); }
QFrame#selFeature {
    border: 1px solid rgba(128,128,128,0.32);
    border-radius: 6px;
    background: rgba(128,128,128,0.10);
}
QTableWidget#featTable, QListWidget#regionList, QListWidget#spectraList {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 4px;
}
QLabel#datasetBanner {
    padding: 6px 8px;
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 5px;
    background: rgba(128,128,128,0.10);
}
"""

# A visible, grabbable grip on the central splitter handle — so the divider reads as a
# drag target and, once the right panel is dragged shut against the edge, the collapsed
# handle is still easy to find and drag back open.
_SPLIT_QSS = """
QSplitter#centerSplit::handle:horizontal {
    width: 8px;
    background: rgba(128,128,128,0.16);
    border-left: 1px solid rgba(128,128,128,0.28);
    border-right: 1px solid rgba(128,128,128,0.28);
}
QSplitter#centerSplit::handle:horizontal:hover { background: rgba(128,128,128,0.34); }
"""


class _CollapsibleSection(QtWidgets.QWidget):
    """A titled section whose body collapses to just its header bar when the
    header is clicked — lets the user fold away panels they aren't using."""

    def __init__(self, title, expanded=True, on_toggle=None, parent=None):
        super().__init__(parent)
        self._on_toggle = on_toggle
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QtWidgets.QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.header.toggled.connect(self._toggled)
        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 4, 4, 6)
        self.body_layout.setSpacing(5)
        self.body.setVisible(expanded)
        lay.addWidget(self.header)
        lay.addWidget(self.body)

    def is_expanded(self):
        return self.header.isChecked()

    def _toggled(self, on):
        self.body.setVisible(on)
        self.header.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        if self._on_toggle is not None:
            self._on_toggle()


class RightPanelMixin:
    # ----- panel assembly -------------------------------------------------- #
    def _build_right_panel(self):
        """Build the persistent Display · Regions · Features panel and place it as the
        right pane of the central splitter. Being a splitter pane (not a dock) means it
        collapses by dragging the divider to the right edge — and the handle stays grabbable
        to drag it back open — instead of being a dock you re-summon from a menu."""
        container = QtWidgets.QWidget()
        container.setStyleSheet(_RIGHT_QSS)
        clay = QtWidgets.QVBoxLayout(container)
        clay.setContentsMargins(5, 5, 5, 5)
        clay.setSpacing(5)
        self._right_sections_layout = clay

        # dataset summary banner — persistent reference data, pinned at the top
        self.info.setObjectName("datasetBanner")
        clay.addWidget(self.info)
        self._build_intake_strip(clay)              # dismissible "suggested settings" strip
        self._section_index_offset = clay.count()   # sections start after banner + intake strip

        # Order top→bottom: Display, then Regions ABOVE Features — you define a region,
        # then build/scope a feature list within it, so the panel flows the way the work
        # does (and the two no longer read as duplicate "pick a list" sections stacked).
        # The optical backdrop (a once-per-sample setup), the composite-image tools, and
        # the visible-spectra trace manager no longer live here — they've moved next to the
        # views they act on: File → Image setup…, the "Composite…" button on the ion-image
        # toolbar, and the "Traces…" button on the spectrum toolbar (see ion._tab_ion).
        disp = _CollapsibleSection("Display", on_toggle=self._relayout_right_sections)
        self._populate_display_section(disp.body_layout)
        reg = _CollapsibleSection("Regions", on_toggle=self._relayout_right_sections)
        self._populate_regions_section(reg.body_layout)
        feat = _CollapsibleSection("Features", on_toggle=self._relayout_right_sections)
        self._populate_features_section(feat.body_layout)
        self._right_sections = [disp, reg, feat]
        for s in self._right_sections:
            clay.addWidget(s)
        clay.addStretch(1)                       # absorbs slack; headers pin to the top
        self._relayout_right_sections()

        # Scroll the panel rather than squeezing sections: each section keeps its natural
        # height (so the feature table never overpaints the controls beneath it), and a
        # short window grows a scrollbar instead of overlapping widgets.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(container)
        # Comfortable floor so the feature table never gets crushed when open; the splitter
        # can still collapse the whole pane to the edge past this floor (setCollapsible).
        scroll.setMinimumWidth(380)
        self.right_panel = scroll
        self._right_panel_w = 0          # remembered open width (for re-expand); set on layout
        # Widen the floor to fit the content + scrollbar so no control row hides behind the
        # scrollbar at the panel's narrowest. Set it SYNCHRONOUSLY (before the split sizes are
        # computed), then re-fit once fonts/layout fully settle (display scaling can widen text).
        self._fit_right_panel_floor()
        self._assemble_center_split()
        self._refresh_feature_set_combo()
        QtCore.QTimer.singleShot(0, self._fit_right_panel_floor)

    def _assemble_center_split(self):
        """Put the analysis views (left) and the right panel in a horizontal splitter and
        make it the central widget. The right pane is collapsible, so dragging the divider
        to the edge hides the panel; the handle stays grabbable to drag it back."""
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setObjectName("centerSplit")
        split.setStyleSheet(_SPLIT_QSS)
        split.setHandleWidth(8)
        split.addWidget(self.tabs)
        split.addWidget(self.right_panel)
        split.setCollapsible(0, False)        # never hide the main view
        split.setCollapsible(1, True)         # the right panel can collapse to the edge
        split.setStretchFactor(0, 1)          # the view soaks up resize; the panel keeps width
        split.setStretchFactor(1, 0)
        self.center_split = split
        self.setCentralWidget(split)
        # Roomy default — about a third of the window, capped — deferred so the layout is
        # realized and setSizes actually takes effect.
        QtCore.QTimer.singleShot(0, self._init_center_split_sizes)

    def _init_center_split_sizes(self):
        """Give the right panel a roomy default width (≈⅓, capped) and remember it."""
        split = getattr(self, "center_split", None)
        if split is None:
            return
        total = max(self.width(), 900)
        # Open roomy enough that the multi-button rows show in full, but never below the
        # content-aware floor (so the rightmost controls never hide behind the scrollbar).
        floor = self.right_panel.minimumWidth() if getattr(self, "right_panel", None) else 440
        right = max(floor, min(580, int(total * 0.36)))
        self._right_panel_w = right
        split.setSizes([max(1, total - right), right])

    def _toggle_right_panel(self):
        """Collapse the right panel to the edge, or restore it to its last open width — the
        keyboard/menu twin of dragging the splitter divider."""
        split = getattr(self, "center_split", None)
        if split is None:
            return
        sizes = split.sizes()
        if len(sizes) < 2:
            return
        if sizes[1] > 0:                      # open → remember width, collapse to the edge
            self._right_panel_w = sizes[1]
            split.setSizes([sum(sizes), 0])
        else:                                 # collapsed → restore
            want = self._right_panel_w or max(440, min(500, int(self.width() * 0.34)))
            split.setSizes([max(1, sum(sizes) - want), want])

    def _relayout_right_sections(self):
        """Every section takes its natural height (stretch 0); the trailing spacer soaks
        up any extra space so headers stay pinned to the top and collapsed sections
        shrink to just their header. The enclosing QScrollArea handles the case where the
        sections together need more height than the panel has — it scrolls rather than
        squeezing a section below its content (which previously overlapped widgets)."""
        clay = self._right_sections_layout
        offset = getattr(self, "_section_index_offset", 0)
        for i in range(len(self._right_sections)):
            clay.setStretch(offset + i, 0)
        clay.setStretch(clay.count() - 1, 1)        # trailing spacer absorbs slack
        self._fit_right_panel_floor()

    def _fit_right_panel_floor(self):
        """Keep the right panel's minimum width wide enough that its content AND the vertical
        scrollbar both fit. Otherwise the rightmost controls — the multi-button rows like the
        Samples actions (Add region, Import metadata…) — overflow past the viewport and hide
        behind the scrollbar, unreachable with horizontal scrolling off. The floor tracks the
        content's own minimumSizeHint, so it adapts to font size / display scaling (wider on
        Windows at 125–150%) instead of a brittle hard-coded number."""
        scroll = getattr(self, "right_panel", None)
        inner = scroll.widget() if scroll is not None else None
        if inner is None:
            return
        extent = scroll.verticalScrollBar().sizeHint().width() or 16
        need = inner.minimumSizeHint().width() + extent + 4    # + frame/slack
        scroll.setMinimumWidth(max(380, need))

    def _reset_panel_layout(self):
        """Re-open the right-hand panel at a sensible width — a one-click fix if it was
        dragged shut against the edge."""
        self._init_center_split_sizes()
        self.statusBar().showMessage("Panel layout reset.")

    # ----- intake inspector (suggested-settings strip) -------------------- #
    # How the detected normalization state reads in the summary banner.
    _NORM_LABEL = {"raw": "raw", "tic": "TIC-norm", "rms": "RMS-norm",
                   "median": "median-norm", "unknown": ""}

    def _build_intake_strip(self, clay):
        """A dismissible strip under the banner that surfaces the intake inspector's
        suggested normalization / extraction tolerance for the just-loaded data (see
        :mod:`smile_msi.intake`). Hidden until a dataset is inspected; the user
        applies or dismisses — settings are never changed silently."""
        strip = QtWidgets.QFrame()
        strip.setObjectName("intakeStrip")
        strip.setStyleSheet(
            f"#intakeStrip {{ background: rgba(85,168,104,0.12);"
            f" border-left: 3px solid {common.ACCENT}; border-radius: 4px; }}")
        v = QtWidgets.QVBoxLayout(strip)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        self._intake_label = QtWidgets.QLabel()
        self._intake_label.setWordWrap(True)
        v.addWidget(self._intake_label)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self._intake_apply_btn = QtWidgets.QPushButton("Apply suggested")
        self._intake_apply_btn.setIcon(icon("save"))
        self._intake_apply_btn.clicked.connect(self._apply_intake_suggestion)
        dismiss = QtWidgets.QPushButton("Dismiss")
        dismiss.setIcon(icon("remove"))
        dismiss.clicked.connect(lambda: self._intake_strip.setVisible(False))
        row.addWidget(self._intake_apply_btn)
        row.addWidget(dismiss)
        row.addStretch(1)
        v.addLayout(row)
        strip.setVisible(False)
        self._intake_strip = strip
        self._intake_report = None
        clay.addWidget(strip)

    def _dataset_banner_html(self, ds, report=None):
        """The right-dock summary banner, enriched with the detected representation
        (centroid/profile) and normalization state when an intake report is given."""
        import os
        mzlo, mzhi = ds.mz_range
        bits = [f"polarity: {ds.polarity or '(unknown)'}"]
        if report is not None:
            if report.representation != "unknown":
                bits.append(report.representation)
            nl = self._NORM_LABEL.get(report.normalization, "")
            if nl:
                bits.append(nl)
        cal = getattr(self, "_calibration", None)
        # a per-region calibration rolls up to its WORST region: saying "calibrated" off the
        # best one would hide the region the mislabels are still coming from
        where = f" per region ({len(cal.get('regions') or [])})" if (cal or {}).get("per_region") else ""
        gone = self._calibration_unresolved_regions()
        if gone:
            cal_html = (f"<span style='color:{common.PALETTE[3]}'>calibration incomplete — "
                        f"{len(gone)} region(s) missing</span>")
        elif cal and cal.get("verified"):
            cal_html = (f"<span style='color:{common.PALETTE[2]}'><b>✓ calibrated{where}</b></span> "
                        f"(lock-mass, residual {cal.get('median_after', 0.0):+.2f} ppm)")
        elif cal and cal.get("median_after") is not None:
            cal_html = (f"<span style='color:{common.PALETTE[3]}'>lock-mass applied{where}, "
                        f"residual {cal['median_after']:+.2f} ppm</span>")
        elif cal:
            cal_html = "lock-mass recalibration applied (not verified)"
        else:
            cal_html = "<span style='color:#888'>not calibrated</span>"
        return (f"<b>{ds.n_pixels:,}</b> pixels · {ds.width}×{ds.height}<br>"
                f"m/z {mzlo:.1f}–{mzhi:.1f}<br>{' · '.join(bits)}<br>{cal_html}<br>"
                f"<i>{os.path.basename(ds.source)}</i>")

    def _show_intake_suggestion(self, report):
        """Populate + show the suggestion strip when the detected data implies a
        settings change (or the file's centroid/profile flag is wrong); hide otherwise
        so a correctly-configured load stays uncluttered."""
        self._intake_report = report
        strip = getattr(self, "_intake_strip", None)
        if strip is None:
            return
        if report is None:
            strip.setVisible(False)
            return
        cur_norm = self.norm_combo.currentText() if hasattr(self, "norm_combo") else "tic"
        actionable = (report.suggested_norm != cur_norm) or bool(report.warnings)
        if not actionable:
            strip.setVisible(False)
            return
        warn = "".join(
            f"<br><span style='color:{common.PALETTE[3]}'>⚠ {w}</span>"
            for w in report.warnings)
        self._intake_label.setText(
            "<b>Suggested settings</b><br>" + "<br>".join(report.notes) + warn)
        self._intake_apply_btn.setText(
            f"Apply  (normalization → {report.suggested_norm}, "
            f"{report.suggested_tol_ppm:.0f} ppm)")
        strip.setVisible(True)

    def _apply_intake_suggestion(self):
        """Apply the inspector's suggested normalization + extraction tolerance to the
        live controls (which re-render the viewer / invalidate the feature matrix)."""
        r = getattr(self, "_intake_report", None)
        if r is None:
            return
        if hasattr(self, "norm_combo"):
            self.norm_combo.setCurrentText(r.suggested_norm)
        if hasattr(self, "ppm_spin"):
            self.ppm_spin.setValue(float(r.suggested_tol_ppm))
        self._intake_strip.setVisible(False)
        self.statusBar().showMessage(
            f"Applied suggested settings — normalization: {r.suggested_norm}, "
            f"extraction tolerance: {r.suggested_tol_ppm:.0f} ppm.")

    # ----- Display section (live viewer render settings) ------------------- #
    def _populate_display_section(self, v):
        v.addWidget(self._note("Render settings — apply live to the ion-image viewer. "
                               "The colormap picker sits on the ion-image toolbar; "
                               "per-feature colors live in the Features panel."))
        # NB: self.cmap_combo is created on the ion-image toolbar (see ion._tab_ion), which
        # is built before this panel — every view shares that one combo.
        form = QtWidgets.QFormLayout()
        self.contrast_spin = common.NoScrollDoubleSpinBox()
        self.contrast_spin.setRange(80.0, 100.0)
        self.contrast_spin.setValue(99.0)
        self.contrast_spin.setSingleStep(0.5)
        self.contrast_spin.setToolTip("Clip the brightest pixels to remove hotspots (per-image percentile)")
        self.contrast_spin.valueChanged.connect(lambda *_: self._display_setting_changed())
        self.norm_combo = common.NoScrollComboBox()
        # "median" per-spectrum normalization is intentionally NOT offered here pending a
        # patent review (the per-spectrum median divisor reads on EP2388797B1's normalization
        # Markush list); the engine still supports norm="median" so older sessions keep loading.
        self.norm_combo.addItems(["tic", "none", "rms"])
        self.norm_combo.setToolTip("Per-pixel normalization applied to ion images and every analysis")
        self.norm_combo.currentTextChanged.connect(lambda *_: self._display_setting_changed())
        self.composite_combo = common.NoScrollComboBox()
        self.composite_combo.addItems(["raw sum", "balanced"])
        self.composite_combo.setToolTip(
            "How a lipid-class composite combines its member ions:\n"
            "• raw sum — true total-abundance map; the most abundant ion dominates\n"
            "• balanced — scale each ion to its own 99th-percentile before summing, so "
            "every member contributes (a 'where the class is present' map)")
        self.composite_combo.currentTextChanged.connect(lambda *_: self._display_setting_changed())
        form.addRow("Hotspot clip %", self.contrast_spin)
        form.addRow("Normalization", self.norm_combo)
        form.addRow("Class composite", self.composite_combo)
        v.addLayout(form)

    def _display_setting_changed(self):
        """Colormap / hotspot-clip / normalization changed → re-render and persist
        (these are part of the saved session settings).

        Debounced: dragging a spin box fires valueChanged on every tick, and each
        re-render re-extracts the ion image + recomputes percentiles. Coalesce the
        bursts so only the settled value triggers one redraw."""
        t = getattr(self, "_display_refresh_timer", None)
        if t is None:
            t = QtCore.QTimer(self)
            t.setSingleShot(True)
            t.setInterval(120)
            t.timeout.connect(self._apply_display_setting)
            self._display_refresh_timer = t
        t.start()

    def _apply_display_setting(self):
        self.refresh_ion_image()
        self._mark_dirty()

    @staticmethod
    def _muted(label):
        label.setStyleSheet(MUTED_QSS)
        return label

    @classmethod
    def _note(cls, text):
        """A muted, word-wrapping descriptive label. Wrapping is what keeps the panel
        shrinkable: an unwrapped sentence reports its full text width as a minimum and
        would pin the dock open."""
        lbl = cls._muted(QtWidgets.QLabel(text))
        lbl.setWordWrap(True)
        return lbl

    # ----- Features section (leads with the data) -------------------------- #
    def _populate_features_section(self, v):
        # --- the feature table is the star: build + place it first ----------
        self.feat_table = FeatureTable()
        self.feat_table.setObjectName("featTable")
        self.feat_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.feat_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.feat_table.setSortingEnabled(True)
        self.feat_table.verticalHeader().setVisible(False)
        self.feat_table.setShowGrid(False)                  # cleaner
        self.feat_table.setAlternatingRowColors(True)       # adapts to the palette
        self.feat_table.setMinimumHeight(320)     # ~11 rows; scrolls internally for more
        hh = self.feat_table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setHighlightSections(False)
        self.feat_table.setMouseTracking(True)              # emit cellEntered on hover
        self.feat_table.viewport().setMouseTracking(True)
        self.feat_table.itemSelectionChanged.connect(self._feature_selected)
        self.feat_table.cellEntered.connect(self._feat_hover)   # hover → Co-localization starting ion
        self.feat_table.cellDoubleClicked.connect(self._feature_double_clicked)
        self.feat_table.itemChanged.connect(self._feat_visibility_changed)  # m/z eye toggle
        for seq in (QtGui.QKeySequence.Delete, QtGui.QKeySequence("Backspace"),
                    QtGui.QKeySequence("Ctrl+Backspace")):
            sc = QtGui.QShortcut(seq, self.feat_table)
            sc.setContext(QtCore.Qt.WidgetShortcut)
            sc.activated.connect(self._remove_selected_features)
        self.feat_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.feat_table.customContextMenuRequested.connect(self._feat_table_menu)
        self.feat_table.setToolTip(
            "Click a row to select that feature (drives the ion image + spectrum cursor). "
            "Click the eye in the m/z column to show/hide it. Double-click to open it. "
            "Right-click a row for more actions (colour, co-localization, remove, export).")
        # Header tooltips persist across refills (the hook re-applies on every fill_table,
        # which lives in features.py / ion.py), so installing the map once here is enough;
        # only header keys that actually exist in a given fill get applied.
        common.set_header_tooltips(self.feat_table, {
            "m/z": "Measured mass-to-charge ratio of the feature (the picked peak centroid).",
            "lipid": "Best in-silico lipid annotation for this m/z.",
            "class": "Lipid class of the annotation (e.g. PC, PE, SM, sulfatide).",
            "adduct": "Ionization adduct assumed for the match (e.g. [M+H]+, [M+Na]+, [M-H]-).",
            "ppm": "Mass error between the measured m/z and the theoretical match, in parts-per-million.",
            "confidence": "Annotation confidence / FDR for the match.",
            "FDR": "Estimated false-discovery rate for the annotation.",
            "snr": "Signal-to-noise ratio of the detected peak."})
        # back-compat alias: older code/tests reach the picked-peak view via peak_table
        self.peak_table = self.feat_table
        self.feat_df = None

        # --- active-list summary banner sits right above the table ----------
        self.flist_status = QtWidgets.QLabel()
        self.flist_status.setWordWrap(True)
        self.flist_status.setStyleSheet("font-weight: 600;")
        v.addWidget(self.flist_status)

        # --- pick + identify + filter (compact, single row) -----------------
        tb = QtWidgets.QHBoxLayout()
        b_pick = QtWidgets.QPushButton("Find peaks…")
        b_pick.setIcon(icon("find"))
        b_pick.setToolTip("Detect peaks → build the working feature set (P). "
                          "Opens the picking settings.")
        b_pick.clicked.connect(self._open_pick_dialog)
        tb.addWidget(b_pick)
        b_build = QtWidgets.QPushButton("Identify lipids")
        b_build.setIcon(icon("run"))
        b_build.setToolTip("Annotate every feature as a lipid (with FDR)")
        b_build.clicked.connect(self.do_feature_list)
        tb.addWidget(b_build)
        self.feat_filter = QtWidgets.QLineEdit()
        self.feat_filter.setPlaceholderText("filter m/z, lipid, class…")
        self.feat_filter.setClearButtonEnabled(True)
        self.feat_filter.textChanged.connect(self._apply_feature_filter)
        tb.addWidget(self.feat_filter, 1)
        self.feat_idonly = QtWidgets.QCheckBox("id'd")
        self.feat_idonly.setToolTip("Show only identified features")
        self.feat_idonly.toggled.connect(self._apply_feature_filter)
        tb.addWidget(self.feat_idonly)
        v.addLayout(tb)

        # --- unified feature-set selector: one dropdown answers "what set am I
        #     looking at?" — working scopes (All slide / per-region samples) up top,
        #     then saved library lists (★) below a separator. Picking a scope switches
        #     to it; picking a saved list loads it. (Replaces the old separate
        #     "Features for:" scope row + "List:" saved-list row.)
        set_row = QtWidgets.QHBoxLayout()
        set_row.addWidget(self._muted(QtWidgets.QLabel("Feature set:")))
        self.feat_set_combo = common.NoScrollComboBox(lock_wheel=True)  # wheel must NEVER swap the
        # set (each switch is a full rebuild); only opening the dropdown and clicking may change it
        self.feat_set_combo.setToolTip(
            "Active feature set — switch between the whole slide, a region's own "
            "features (find peaks with a region selected), or a saved list (★).")
        self.feat_set_combo.activated.connect(self._on_feature_set_activated)
        set_row.addWidget(self.feat_set_combo, 1)
        v.addLayout(set_row)

        # --- display row: color overlay (all visible) · per-feature color. The colormap
        #     picker now lives on the ion-image toolbar (it drives the single-ion render). ---
        disp = QtWidgets.QHBoxLayout()
        self.color_overlay_chk = QtWidgets.QCheckBox("Color overlay")
        self.color_overlay_chk.setToolTip(
            "Show all eye-on features at once as an additive colour composite — each "
            "tinted by its own colour and intensity window — instead of a single ion "
            "image drawn with the colormap. Set a feature's colour with 'Color…' beside "
            "this box; co-located ions blend (red+green→yellow).")
        self.color_overlay_chk.toggled.connect(self._toggle_color_overlay)
        disp.addWidget(self.color_overlay_chk)
        self.b_feat_color = QtWidgets.QPushButton("Color…")
        self.b_feat_color.setIcon(icon("settings"))
        self.b_feat_color.setToolTip("Set the color of the selected feature(s) — shown in the "
                                     "list (eye) and used by the color overlay")
        self.b_feat_color.clicked.connect(self._set_feature_color)
        disp.addWidget(self.b_feat_color)
        # one-click colour palette across every eye-on feature (CVD-safe overlay presets +
        # categorical palettes from the shared smile_msi.palettes module)
        self.b_feat_palette = QtWidgets.QToolButton()
        self.b_feat_palette.setText("Palette")
        self.b_feat_palette.setObjectName("menuButton")   # shared dropdown pill + chevron
        self.b_feat_palette.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.b_feat_palette.setToolTip("Recolour all visible (eye-on) features at once from a "
                                       "colour-vision-safe overlay preset or a categorical palette")
        self.b_feat_palette.setMenu(self._build_palette_menu())
        disp.addWidget(self.b_feat_palette)
        disp.addStretch(1)
        v.addLayout(disp)

        # --- current-selection chip: shows the feature the spectrum/table is pointed
        #     at and an ✕ to deselect it (clears the cursor, halo + table highlight).
        #     Mirrors the Esc shortcut for users who'd rather click. Kept compact and
        #     pinned right above the table so "what am I looking at?" is always visible.
        selchip = QtWidgets.QHBoxLayout()
        selchip.setSpacing(4)
        self.feat_sel_chip = self._muted(QtWidgets.QLabel("No feature selected"))
        self.feat_sel_chip.setToolTip("The feature currently driving the spectrum cursor and ion image")
        # Ignored width policy: let the label clip rather than report its full text as a
        # minimum, which would pin the dock open (see _note()). Full text rides the tooltip.
        self.feat_sel_chip.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        selchip.addWidget(self.feat_sel_chip, 1)
        self.feat_deselect_btn = QtWidgets.QToolButton()
        self.feat_deselect_btn.setIcon(icon("remove"))
        self.feat_deselect_btn.setText("")
        self.feat_deselect_btn.setAccessibleName("Deselect feature")
        self.feat_deselect_btn.setAutoRaise(True)
        self.feat_deselect_btn.setToolTip("Deselect — clear the current feature (same as Esc)")
        self.feat_deselect_btn.setEnabled(False)
        self.feat_deselect_btn.clicked.connect(self._deselect_feature)
        selchip.addWidget(self.feat_deselect_btn)
        v.addLayout(selchip)

        # The flat per-ion table and the class-grouped lipid tree share one slot: a
        # loaded ◆ lipid list shows the tree (classes → ions, each class acting as one
        # ion for viewing); everything else shows the table. _set_peaks reverts to the
        # table, _load_lipid_list flips to the tree (see LipidTreeMixin).
        self.feat_stack = QtWidgets.QStackedWidget()
        self.feat_stack.addWidget(self.feat_table)            # page 0: flat per-ion table
        self.feat_stack.addWidget(self._build_lipid_tree())   # page 1: class-grouped tree
        v.addWidget(self.feat_stack, 1)

        # --- show/hide-all eye, right below the feature box -----------------
        # One "Toggle all" flip-flop meant you could not say what a click would do without
        # first reading every eye in the list. Say it explicitly instead, and show the count —
        # a long, part-hidden list must not be able to lie about what the analyses will see.
        toggle_row = QtWidgets.QHBoxLayout()
        _excl = ("Hidden features are excluded from segmentation, ROI stats, components, "
                 "and every other analysis.")
        for text, tip, want in (
                ("Show all", f"Show every feature. {_excl}", lambda _p: True),
                ("Hide all", f"Hide every feature. {_excl}", lambda _p: False),
                ("Invert", f"Show what is hidden and hide what is shown. {_excl}",
                 lambda p: bool(p.get("hidden")))):
            b = common.button(text, self._feature_visibility_slot(want), tooltip=tip)
            # Several panels now carry a Show all / Hide all / Invert trio, so name these:
            # findChildren by label alone would reach the lipid tree's or the regions menu's.
            b.setObjectName("featVis" + text.replace(" ", ""))
            if text == "Show all":
                b.setIcon(eye_icon("#9a9a9a", True))
            toggle_row.addWidget(b)
        b_clear = QtWidgets.QPushButton("Clear")
        b_clear.setIcon(icon("remove"))
        b_clear.setToolTip("Clear the feature list from view. The list isn't deleted — it "
                           "stays in the feature-set selector above and can be reopened "
                           "(or restored with ⌘Z).")
        b_clear.clicked.connect(self._clear_features)
        toggle_row.addWidget(b_clear)
        toggle_row.addStretch(1)
        self.feat_visible_count = self._muted(QtWidgets.QLabel(""))
        toggle_row.addWidget(self.feat_visible_count)
        v.addLayout(toggle_row)

        # --- identification detail (muted, below the table) -----------------
        self.feat_info = self._muted(QtWidgets.QLabel("Find peaks, then identify lipids."))
        self.feat_info.setWordWrap(True)
        v.addWidget(self.feat_info)

        # --- selected feature (driven by the highlighted spectrum peak) -----
        selbox = QtWidgets.QFrame()
        selbox.setObjectName("selFeature")
        sel = QtWidgets.QVBoxLayout(selbox)
        sel.setContentsMargins(8, 6, 8, 6)
        sel.setSpacing(4)
        self.sel_feature_label = QtWidgets.QLabel(
            "No feature selected — click a peak in the spectrum.")
        self.sel_feature_label.setWordWrap(True)
        sel.addWidget(self.sel_feature_label)
        selrow = QtWidgets.QHBoxLayout()
        self.add_feature_btn = QtWidgets.QPushButton("Add to list")
        self.add_feature_btn.setIcon(icon("add"))
        self.add_feature_btn.setToolTip("Add the highlighted m/z to the working feature list (⌘D)")
        self.add_feature_btn.setEnabled(False)
        self.add_feature_btn.clicked.connect(self._add_active_to_list)
        b_coloc = QtWidgets.QPushButton("Co-localized…")
        b_coloc.setIcon(icon("find"))
        b_coloc.setToolTip("Find ions co-localized with the selected feature")
        b_coloc.clicked.connect(self._coloc_selected)
        selrow.addWidget(self.add_feature_btn)
        selrow.addWidget(b_coloc)
        selrow.addStretch(1)
        sel.addLayout(selrow)
        # --- per-feature intensity window (dual-handle contrast slider) ---
        winrow = QtWidgets.QHBoxLayout()
        intensity_lbl = self._muted(QtWidgets.QLabel("Intensity"))
        intensity_lbl.setToolTip("Per-feature contrast window, as % of the hotspot-clip max "
                                 "(relative intensity). Drag the two handles to set "
                                 "the low/high display range for the selected feature.")
        winrow.addWidget(intensity_lbl)
        self.feat_window_slider = RangeSliderField(0.0, 100.0)
        self.feat_window_slider.setToolTip("Contrast window — % of the hotspot-clip max (relative intensity, "
                                           "so 90% here maps consistently). Drag the low/high handles, or "
                                           "type exact values, to set the displayed range for this feature. "
                                           "The image reloads once you let go / press Enter.")
        # Re-render only on a *committed* edit (drag release / typed value), so dragging the
        # window stays smooth instead of re-rendering the ion image on every pixel of motion.
        self.feat_window_slider.editingFinished.connect(lambda *_: self._feat_window_changed())
        winrow.addWidget(self.feat_window_slider, 1)
        sel.addLayout(winrow)
        self.feat_apply_all_chk = QtWidgets.QCheckBox("Apply window to all in current view")
        self.feat_apply_all_chk.setToolTip(
            "On: dragging the intensity window sets the same contrast for every feature "
            "shown in the list (after filtering). Off: it applies to the selected rows, "
            "or just the active feature.")
        sel.addWidget(self.feat_apply_all_chk)
        v.addWidget(selbox)

        # --- save / manage the current set (the selector above does the loading) ---
        mng = QtWidgets.QHBoxLayout()
        mng.addStretch(1)
        b_save = QtWidgets.QPushButton("Save as…")
        b_save.setIcon(icon("save"))
        b_save.setToolTip("Save the current working features as a new named list (★)")
        b_save.clicked.connect(self._save_feature_list_as)
        mng.addWidget(b_save)
        edit_btn = QtWidgets.QToolButton()
        edit_btn.setIcon(icon("overflow"))
        edit_btn.setText("")
        edit_btn.setObjectName("menuButton")              # shared dropdown pill + chevron (not native arrow)
        edit_btn.setAccessibleName("More feature-list actions")
        edit_btn.setToolTip("More: edit features, MS/MS, export")
        edit_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(edit_btn)
        menu.addAction("Add feature…", self._add_feature).setIcon(icon("add"))
        menu.addAction("New list from visible (eye-ticked) ions…",
                       self._save_visible_as_feature_list).setIcon(icon("save"))
        menu.addAction("Remove selected", self._remove_selected_features).setIcon(icon("remove"))
        menu.addAction("Clear list from view", self._clear_features).setIcon(icon("remove"))
        menu.addAction("Import targets…", self._import_targets).setIcon(icon("import"))
        menu.addSeparator()
        menu.addAction("Rename list", self._rename_feature_list).setIcon(icon("settings"))
        menu.addAction("Delete list", self._delete_feature_list).setIcon(icon("delete"))
        menu.addAction("Combine lists…", self._combine_feature_lists).setIcon(icon("merge"))
        menu.addAction("Set current list as default", self._set_default_feature_set).setIcon(icon("save"))
        menu.addAction("Manage feature lists…", self._open_feature_lists_manager).setIcon(icon("open"))
        menu.addSeparator()
        menu.addAction("New lipid list from identified features…",
                       self._new_lipid_list_from_features).setIcon(icon("add"))
        menu.addAction("Lipid lists…  (◆ class overlays)", self._manage_lipid_lists).setIcon(icon("settings"))
        menu.addSeparator()
        menu.addAction("Confirm with MS/MS…", self.do_msms_confirm).setIcon(icon("run"))
        menu.addAction("MS/MS library match…", self.do_msms_library).setIcon(icon("run"))
        menu.addSeparator()
        menu.addAction("Export…  (images · spectra · book)", lambda: self.open_export_hub()).setIcon(icon("export"))
        menu.addAction("Export feature table…", lambda: self.open_export_hub("features")).setIcon(icon("export"))
        menu.addAction("Export CSV (with analyses)…", self.export_features).setIcon(icon("export"))
        menu.addAction("Export methods report…", self.export_methods).setIcon(icon("export"))
        edit_btn.setMenu(menu)
        mng.addWidget(edit_btn)
        v.addLayout(mng)

    # ----- Composite ion-image tools (homed in the ion-toolbar dialog) ----- #
    def _populate_composite_section(self, v):
        """Total-class and A/B ratio composite images. These render one combined map into
        the single-ion Ion-image view (with the active colormap) — distinct from the
        Features 'Color overlay', which is an additive multi-ion RGB blend. Built into a
        small dialog opened from the ion-image toolbar's 'Composite…' button; the combos
        are filled by the feature pipeline (``_build_class_map`` / ``_populate_peak_combos``)."""
        v.addWidget(self._note(
            "Render one combined ion image into the Ion image view. 'Σ class' sums every "
            "ion of a lipid class; 'Ratio' divides one ion image by another (e.g. "
            "sulfatide / PC for myelin contrast). Find peaks first to fill the menus."))
        tools = QtWidgets.QHBoxLayout()
        tools.addWidget(QtWidgets.QLabel("Σ class"))
        self.class_combo = common.NoScrollComboBox()
        self.class_combo.setToolTip("Sum every ion of a lipid class into one composite image")
        tools.addWidget(self.class_combo, 1)
        b_class = QtWidgets.QPushButton("Show")
        b_class.setIcon(icon("run"))
        b_class.clicked.connect(self.do_class_image)
        tools.addWidget(b_class)
        v.addLayout(tools)
        ratio = QtWidgets.QHBoxLayout()
        ratio.addWidget(QtWidgets.QLabel("Ratio"))
        self.ratio_a = common.NoScrollComboBox()
        self.ratio_b = common.NoScrollComboBox()
        ratio.addWidget(self.ratio_a, 1)
        ratio.addWidget(QtWidgets.QLabel("/"))
        ratio.addWidget(self.ratio_b, 1)
        b_ratio = QtWidgets.QPushButton("A/B")
        b_ratio.setIcon(icon("run"))
        b_ratio.setToolTip("Ratio of two ion images (e.g. sulfatide/PC for myelin contrast)")
        b_ratio.clicked.connect(self.do_ratio_image)
        ratio.addWidget(b_ratio)
        v.addLayout(ratio)

    # ----- Regions section (ROI- or cluster-based, with sub-regions) ------- #
    def _populate_regions_section(self, v):
        v.addWidget(self._note(
            "Regions — draw an ROI on the ion image (rectangle or polygon) and 'Add ROI' to "
            "save it, or group segmentation clusters. Select one or many."))
        # filter box — find a region fast when a slide carries many (keeps the panel
        # usable as region counts grow); hides non-matching rows without rebuilding.
        self.region_filter = QtWidgets.QLineEdit()
        self.region_filter.setPlaceholderText("Filter regions…")
        self.region_filter.setClearButtonEnabled(True)
        self.region_filter.setToolTip("Show only regions whose name or detail contains this text.")
        self.region_filter.textChanged.connect(self._apply_region_filter)
        v.addWidget(self.region_filter)

        # regions are tied to their slide — show only the active sample's regions by default,
        # tick to see every loaded sample's regions at once (the "2500 global regions" pile).
        self.region_all_samples = QtWidgets.QCheckBox("All samples")
        self.region_all_samples.setToolTip(
            "Off: show only this slide's regions. On: show regions from every loaded sample.")
        self.region_all_samples.toggled.connect(self._on_region_scope_toggled)
        v.addWidget(self.region_all_samples)

        self.region_list = RegionList()
        self.region_list.setObjectName("regionList")
        self.region_list.setMinimumHeight(110)
        self.region_list.setIconSize(QtCore.QSize(16, 16))                  # predictable eye hit-zone
        self.region_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.region_list.setToolTip("Select one region, or several (Ctrl/⌘-click) to overlay or "
                                    "compare their spectra — or build one feature list over them "
                                    "together · click the eye to show/hide · double-click to rename "
                                    "· right-click for colour · drag to reorder")
        self.region_list.itemSelectionChanged.connect(self._region_list_selection_changed)
        self.region_list.eyeClicked.connect(self._region_eye_clicked)       # eye glyph = show/hide
        self.region_list.itemDoubleClicked.connect(lambda *_: self._region_rename())
        self.region_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.region_list.customContextMenuRequested.connect(self._region_list_menu)
        # Drag rows to reorder regions (single-row internal move); the settled drop drives
        # the persist + re-nest sync (see _region_sync_order). Sub-regions follow their
        # parent. Drag is auto-disabled while the filter hides rows (_apply_region_filter).
        self.region_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.region_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.region_list.setDropIndicatorShown(True)
        self.region_list.reordered.connect(self._region_sync_order)
        v.addWidget(self.region_list, 1)

        # --- primary actions stay visible; the rest fold into a "More ⚙" popover so the
        #     panel reads as two clear steps (draw → build) instead of a wall of buttons ---
        self.region_draw_chk = QtWidgets.QCheckBox("Draw ROI")
        self.region_draw_chk.setToolTip("Draw a rectangle or polygon on the ion image to outline a region")
        self.region_draw_chk.toggled.connect(self._region_toggle_draw)
        if getattr(self, "roi_chk", None) is not None:        # keep both ROI toggles in sync
            self.roi_chk.toggled.connect(self._sync_region_draw_chk)
        b_from_roi = QtWidgets.QPushButton("Add ROI")
        b_from_roi.setIcon(icon("add"))
        b_from_roi.setToolTip("Save the ROI you drew (rectangle or polygon) as a region. The "
                              "active feature list keeps driving the pipelines; build a "
                              "region-scoped list on demand with the ion-image toolbar's "
                              "'→ Feature list' button or right-click ▸ 'Build feature list'.")
        b_from_roi.clicked.connect(self._region_from_roi)

        # secondary controls — built the same way, just parked in the popover
        b_sub = QtWidgets.QPushButton("Sub-region")
        b_sub.setIcon(icon("add"))
        b_sub.setToolTip("Carve a sub-region from the selected region using the current ROI")
        b_sub.clicked.connect(self._subregion_from_roi)
        b_auto = QtWidgets.QPushButton("Auto-detect")
        b_auto.setIcon(icon("run"))
        b_auto.setToolTip("Find tissue pieces on the slide automatically → one region each")
        b_auto.clicked.connect(self._auto_detect_samples)
        b_spectra = QtWidgets.QPushButton("Show spectra")
        b_spectra.setIcon(icon("run"))
        b_spectra.setToolTip("Overlay the mean spectrum of each selected region on the spectrum plot")
        # NB: pass focus=True explicitly — clicked() would otherwise feed the button's
        # checked=False straight into the focus arg, so the plot never came to the front.
        b_spectra.clicked.connect(lambda *_: self._show_region_spectra(focus=True))
        self.region_combine_chk = QtWidgets.QCheckBox("Combine")
        self.region_combine_chk.setToolTip("One spectrum over the union of the selected regions")
        # Only re-draw when region traces are already on screen — toggling Combine should
        # never trigger a fresh spectrum compute from nothing (use 'Show spectra' for that).
        self.region_combine_chk.toggled.connect(
            lambda *_: self._show_region_spectra(focus=False)
            if getattr(self, "_region_spectra_shown", False) else None)
        b_clusters = QtWidgets.QPushButton("Clusters")
        b_clusters.setIcon(icon("add"))
        b_clusters.setToolTip("New region to fill from segmentation clusters")
        b_clusters.clicked.connect(self._region_new)
        b_colour = QtWidgets.QPushButton("Colour…")
        b_colour.setIcon(icon("settings"))
        b_colour.setToolTip("Pick a colour for the selected region")
        b_colour.clicked.connect(self._region_set_color)
        b_rename = QtWidgets.QPushButton("Rename")
        b_rename.setIcon(icon("settings"))
        b_rename.setToolTip("Rename the selected region (or just double-click it in the list).")
        b_rename.clicked.connect(self._region_rename)
        b_delete = QtWidgets.QPushButton("Delete")
        b_delete.setObjectName("dangerAction")            # destructive verb → red tier
        b_delete.setIcon(icon("delete"))
        b_delete.clicked.connect(self._region_delete)
        b_showall = QtWidgets.QPushButton("Show all")
        b_showall.setIcon(icon("find"))
        b_showall.setToolTip("Outline every region on the ion image (and the segmentation map, if any)")
        b_showall.clicked.connect(self._show_all_regions)
        b_moveup = QtWidgets.QPushButton("Move up")
        b_moveup.setIcon(icon("up"))
        b_moveup.setToolTip("Move the selected region up among its siblings (its sub-regions move "
                            "with it). You can also drag rows in the list to reorder.")
        b_moveup.clicked.connect(lambda: self._region_move(self._selected_region_index(), -1))
        b_movedown = QtWidgets.QPushButton("Move down")
        b_movedown.setIcon(icon("down"))
        b_movedown.setToolTip("Move the selected region down among its siblings.")
        b_movedown.clicked.connect(lambda: self._region_move(self._selected_region_index(), +1))

        bar = ControlBar()
        bar.add(self.region_draw_chk, b_from_roi)
        bar.add_more(
            "Spectra", b_spectra, self.region_combine_chk,
            "Make regions", b_sub, b_auto, b_clusters,
            "Selected region", b_moveup, b_movedown, b_colour, b_rename, b_delete, b_showall,
        )
        v.addWidget(bar)

        # --- one-click: build a feature list scoped to the selected region (or drawn ROI) ---
        b_feat = QtWidgets.QPushButton("Build feature list from region(s)")
        b_feat.setIcon(icon("save"))
        b_feat.setToolTip("Find peaks within the selected region (or the ROI you just drew) and "
                          "identify them — a feature list scoped to that sample (switch back via "
                          "the Features ▸ Feature set selector).\n"
                          "Select several regions (Ctrl/⌘-click) to build one list over them "
                          "together — peaks are picked across their combined pixels.")
        b_feat.clicked.connect(lambda: self.features_from_regions(self._selected_region_indices()))
        self.b_build_feat_from_region = b_feat
        b_feat.setEnabled(False)                      # needs a dataset + a selected region
        v.addWidget(b_feat)
        # Instant feedback: re-eval when the region selection changes. (Dataset/region-set
        # changes also flow through main.py's _refresh_action_states aggregator.)
        self.region_list.itemSelectionChanged.connect(self._update_build_region_enabled)
        self._update_build_region_enabled()
        # regions auto-save with the sample's session now — no manual library save/load.

    def _update_build_region_enabled(self):
        btn = getattr(self, 'b_build_feat_from_region', None)
        if btn is not None:
            btn.setEnabled(getattr(self, 'ds', None) is not None
                           and bool(self._selected_region_indices()))

    # ----- spectrum-trace manager (the spectrum toolbar's "Traces…" dialog) - #
    def _populate_spectra_section(self, v):
        self.spectra_list = QtWidgets.QListWidget()
        self.spectra_list.setObjectName("spectraList")
        self.spectra_list.setMinimumHeight(80)
        self.spectra_list.setToolTip("Tick to show/hide each trace overlaid on the spectrum")
        self.spectra_list.itemChanged.connect(self._spectra_item_changed)
        v.addWidget(self.spectra_list, 1)
        bar = FlowLayout()
        # Only 'Clear ROI' lives here: the spectrum toolbar already carries 'Clear pixels'
        # and 'Reset (R)' inches from the Traces… button, but its clear-ROI control is tucked
        # inside the Draw-ROI tool group (hidden unless drawing), so this is the one trace
        # action not otherwise reachable from the toolbar.
        b_roi = QtWidgets.QPushButton("Clear ROI")
        b_roi.setIcon(icon("remove"))
        b_roi.setToolTip("Remove the ROI mean trace overlaid on the spectrum")
        b_roi.clicked.connect(self._clear_roi)
        bar.addWidget(b_roi)
        v.addLayout(bar)

    # ----- per-feature color + intensity window ---------------------------- #
    def _feature_color(self, p):
        """The display color for a feature — its explicit ``color`` if set, else a
        stable default from the shared PALETTE keyed on its position in the list."""
        if p is None:
            return MUTED_FG
        c = p.get("color")
        if c:
            return c
        try:
            i = self.peaks.index(p)
        except ValueError:
            i = 0
        return PALETTE[i % len(PALETTE)]

    def _feature_window(self, p):
        """(low%, high%) contrast handles for a feature, as **relative intensity** — % of
        the hotspot-clip anchor (defaults 0 / 100 = full range up to the clip)."""
        if p is None:
            return 0.0, 100.0
        return float(p.get("lo", 0.0)), float(p.get("hi", 100.0))

    def _peak_for_mz(self, mz, tol=1e-3):
        if mz is None:
            return None
        for p in self.peaks:
            if abs(p["mz"] - mz) < tol:
                return p
        return None

    def _set_feature_color(self):
        rows = {ix.row() for ix in self.feat_table.selectionModel().selectedRows()}
        if not rows:
            rows = {it.row() for it in self.feat_table.selectedItems()}
        mzs = [m for m in (self._feat_mz_at(r) for r in rows) if m is not None]
        if not mzs:
            self.statusBar().showMessage("Select feature row(s) first, then pick a color.")
            return
        from . import colorpicker
        first = self._peak_for_mz(mzs[0])
        hexc = colorpicker.pick_color(self, initial=self._feature_color(first),
                                      title="Feature colour")
        if not hexc:
            return
        self.record_undo("feature colour")
        for m in mzs:
            p = self._peak_for_mz(m)
            if p is not None:
                p["color"] = hexc
        self._decorate_feature_swatches()
        self._sync_lipid_tree()                           # mirror the new colour into the tree
        if self.color_overlay_chk.isChecked():
            self.refresh_ion_image()
        self.statusBar().showMessage(f"Color {hexc} set on {len(mzs)} feature(s).")

    def _build_palette_menu(self):
        """The 'Palette ▾' menu: overlay presets (CVD-safe channel sets) + categorical
        palettes, each recolouring every eye-on feature in one click."""
        from .. import palettes
        menu = QtWidgets.QMenu(self)
        ov = menu.addMenu("Overlay presets (CVD-safe)")
        for name in palettes.OVERLAY_PRESETS:
            ov.addAction(name, lambda _=0, nm=name: self._apply_palette(nm, overlay=True))
        cat = menu.addMenu("Categorical")
        for name in palettes.CATEGORICAL_PALETTES:
            cat.addAction(name, lambda _=0, nm=name: self._apply_palette(nm, overlay=False))
        return menu

    def _apply_palette(self, name, *, overlay):
        """Recolour the visible (eye-on) features from palette ``name`` — cycled in list
        order. Overlay presets also turn the colour overlay on so the result is visible."""
        from .. import palettes
        vis = self._visible_peaks()
        if not vis:
            self.statusBar().showMessage("No visible features to recolour.")
            return
        cols = (palettes.overlay_colors(name, len(vis)) if overlay
                else palettes.categorical(name, len(vis)))
        if not cols:
            return
        self.record_undo("apply palette")
        for i, p in enumerate(vis):
            p["color"] = cols[i % len(cols)]
        if overlay and not self.color_overlay_chk.isChecked():
            self.color_overlay_chk.setChecked(True)       # show the composite (triggers a refresh)
        self._decorate_feature_swatches()
        self._sync_lipid_tree()
        if self.color_overlay_chk.isChecked():
            self.refresh_ion_image()
        self.statusBar().showMessage(f"Applied “{name}” palette to {len(vis)} feature(s).")

    def _toggle_color_overlay(self, on):
        # the colormap only applies to the single-ion view; grey it out in overlay mode
        self.cmap_combo.setEnabled(not on)
        self.refresh_ion_image()

    def _window_target_peaks(self):
        """Which features the intensity-window slider edits: every feature in the current
        (filtered) view when 'apply to all' is ticked, else all selected rows when more
        than one is highlighted, else just the active feature."""
        t = getattr(self, "feat_table", None)
        chk = getattr(self, "feat_apply_all_chk", None)
        if t is not None and chk is not None and chk.isChecked():
            peaks = [self._peak_for_mz(self._feat_mz_at(r))
                     for r in range(t.rowCount()) if not t.isRowHidden(r)]
            peaks = [p for p in peaks if p is not None]
            if peaks:
                return peaks
        if t is not None:
            rows = {ix.row() for ix in t.selectionModel().selectedRows()}
            if len(rows) > 1:
                peaks = [self._peak_for_mz(self._feat_mz_at(r)) for r in rows]
                peaks = [p for p in peaks if p is not None]
                if peaks:
                    return peaks
        p = self._peak_for_mz(self.active_mz)
        return [p] if p is not None else []

    def _feat_window_changed(self):
        """The intensity window was edited → store it on the target feature(s) and
        re-render. Edits one feature by default, or a whole batch when several rows are
        selected / 'apply to all' is on (see :meth:`_window_target_peaks`)."""
        lo, hi = self.feat_window_slider.values()
        targets = self._window_target_peaks()
        if not targets:
            return
        for p in targets:
            p["lo"], p["hi"] = float(lo), float(hi)
        self.refresh_ion_image()
        self._mark_dirty()                        # per-feature window is saved on the peak

    # ----- selected-feature interaction ------------------------------------ #
    def _update_selected_feature_ui(self):
        """Reflect the active m/z in the dock: header text, the Add-to-list button
        state, the per-feature color/window controls, and the synced table row."""
        if getattr(self, "sel_feature_label", None) is None:
            return
        has = self.active_mz is not None
        self.b_feat_color.setEnabled(has)
        self.feat_window_slider.setEnabled(has)
        chip = getattr(self, "feat_sel_chip", None)
        chip_btn = getattr(self, "feat_deselect_btn", None)
        if not has:
            self.sel_feature_label.setText(
                "No feature selected — click a peak in the spectrum.")
            self.add_feature_btn.setEnabled(False)
            self.add_feature_btn.setText("Add to list")
            if chip is not None:
                chip.setText("No feature selected")
                chip.setStyleSheet(MUTED_QSS)
            if chip_btn is not None:
                chip_btn.setEnabled(False)
            return
        mz = float(self.active_mz)
        lab = self.annotate(mz) or "(unannotated)"
        in_list = self._mz_in_list(mz)
        self.sel_feature_label.setText(f"<b>Selected:</b> m/z {mz:.4f}<br>{lab}")
        if chip is not None:
            chip.setText(f"Selected: m/z {mz:.4f} · {lab}")
            chip.setStyleSheet("font-weight: 600;")
            chip.setToolTip(f"m/z {mz:.4f} · {lab}")
        if chip_btn is not None:
            chip_btn.setEnabled(True)
        self.add_feature_btn.setEnabled(self.ds is not None and not in_list)
        self.add_feature_btn.setText("✓ In list" if in_list else "Add to list")
        lo, hi = self._feature_window(self._peak_for_mz(mz))
        self.feat_window_slider.blockSignals(True)
        self.feat_window_slider.setValues(lo, hi)
        self.feat_window_slider.blockSignals(False)
        self._select_feature_row(mz)
        self._select_lipid_tree_ion(mz)               # mirror into the class tree if it's showing

    def _mz_in_list(self, mz, tol=1e-3):
        return any(abs(p["mz"] - mz) < tol for p in self.peaks)

    def _select_feature_row(self, mz, tol=1e-3):
        """Highlight the table row matching ``mz`` without re-triggering selection."""
        t = getattr(self, "feat_table", None)
        if t is None:
            return
        t.blockSignals(True)
        target = -1
        for r in range(t.rowCount()):
            it = t.item(r, 0)
            try:
                if it is not None and abs(float(it.text()) - mz) < tol:
                    target = r
                    break
            except ValueError:
                continue
        if target >= 0 and not any(ix.row() == target
                                   for ix in t.selectionModel().selectedRows()):
            t.selectRow(target)
            t.scrollToItem(t.item(target, 0), QtWidgets.QAbstractItemView.EnsureVisible)
        t.blockSignals(False)

    def _add_active_to_list(self):
        """Add the highlighted spectrum peak to the working list."""
        if self.ds is None or self.active_mz is None:
            self.statusBar().showMessage("Highlight a peak in the spectrum first.")
            return
        mz = float(self.active_mz)
        if self._mz_in_list(mz):
            self.statusBar().showMessage(f"m/z {mz:.4f} is already in the feature list.")
            return
        self.record_undo("add to list")
        # reannotate=False / extract=False: defer ID + matrix like _add_feature — instant add,
        # 'Identify lipids' annotates, ion column pulled lazily on display.
        self._set_peaks(self.peaks + [self._peak_from_mz(mz)], reannotate=False, extract=False,
                        msg=f"Added m/z {mz:.4f} to the feature list.")
        self.set_active_mz(mz)

    def _coloc_selected(self):
        if self.active_mz is None:
            self.statusBar().showMessage("Select a feature (or click a peak) first.")
            return
        # Show the composite overlay of the co-localized ions + offer to make a list,
        # rather than dropping into the full correlation-matrix tab.
        self.open_coloc_overlay(float(self.active_mz))

    def _tab_index(self, needle):
        for i in range(self.tabs.count()):
            if needle in self.tabs.tabText(i):
                return i
        return -1

    # ----- per-feature visibility (eye toggle next to m/z) ----------------- #
    def _decorate_feature_swatches(self):
        """Put a drawn eye / eye-with-slash toggle in the m/z column (no emoji): the
        m/z cell is checkable — checked = visible (open eye, in the row colour),
        unchecked = hidden (greyed eye-with-slash, and the whole row greys out).
        Hidden features are excluded from the analyses (see ``_visible_mzs``)."""
        t = self.feat_table
        # disable sorting while we mutate cells: setting check state with live sorting
        # re-sorts mid-loop (stale row indices) and fires spurious itemChanged
        was_sorting = t.isSortingEnabled()
        t.setSortingEnabled(False)
        t.blockSignals(True)
        # Precompute lookups once: scanning self.peaks per row (_peak_for_mz) and
        # self.peaks.index(p) per row (_feature_color) made this O(n^2), and it reruns
        # on every visibility/selection change — laggy for a broad pick (up to 500).
        # The m/z column is round(mz, 4), so a 4-dp key matches the cell value exactly.
        pos = {id(p): i for i, p in enumerate(self.peaks)}
        by_mz = {}
        for p in self.peaks:
            by_mz.setdefault(round(float(p["mz"]), 4), p)   # keep first (matches _peak_for_mz)

        def color_of(p):
            if p is None:
                return MUTED_FG
            return p.get("color") or PALETTE[pos.get(id(p), 0) % len(PALETTE)]

        for r in range(t.rowCount()):
            it = t.item(r, 0)
            if it is None:
                continue
            mz = self._feat_mz_at(r)
            p = by_mz.get(round(mz, 4)) if mz is not None else None
            hidden = bool(p and p.get("hidden"))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked if hidden else QtCore.Qt.Checked)
            it.setIcon(eye_icon(color_of(p), not hidden))
            self._paint_row_hidden(t, r, hidden)
        t.setSortingEnabled(was_sorting)              # sort once, still signal-blocked
        t.blockSignals(False)
        # fill_table sized the m/z column for text only; the checkbox + eye icon we just
        # added need extra room or the m/z number gets clipped — re-fit it now.
        t.resizeColumnToContents(0)
        if t.columnWidth(0) < 96:
            t.setColumnWidth(0, 96)
        self._refresh_feature_visible_count()

    def _feat_visibility_changed(self, item):
        """The m/z cell's checkbox toggled → flip that feature's visibility. Ignore the
        itemChanged storm that ``fill_table``/``_decorate_feature_swatches`` emit while
        (re)building the rows — only genuine user toggles get through."""
        if item.column() != 0 or getattr(self, "_feat_repopulating", False):
            return
        mz = self._feat_mz_at(item.row())
        if mz is None:
            return
        hidden = item.checkState() != QtCore.Qt.Checked
        for p in self.peaks:
            if abs(p["mz"] - mz) < 1e-3 and bool(p.get("hidden")) != hidden:
                p["hidden"] = hidden
                # Style just this row — calling the full _decorate_feature_swatches here
                # toggles sorting inside an itemChanged slot, which eats the *next* click
                # (so re-showing a hidden feature appeared to do nothing).
                self._style_feature_row(item.row(), hidden)
                self._refresh_feature_visible_count()
                self._sync_feature_consumers()
                if self.color_overlay_chk.isChecked():
                    self.refresh_ion_image()              # add/remove it from the overlay
                break

    def _paint_row_hidden(self, t, row, hidden):
        """Grey out (or restore) every cell in a feature-table row by foreground colour."""
        gray = QtGui.QBrush(QtGui.QColor(common.GUIDE_LINE))
        default = QtGui.QBrush()
        for c in range(t.columnCount()):
            cell = t.item(row, c)
            if cell is not None:
                cell.setForeground(gray if hidden else default)

    def _style_feature_row(self, row, hidden):
        """Grey out (or restore) one row + flip its eye icon, without touching sort order
        — safe to run from inside the checkbox's itemChanged handler."""
        t = self.feat_table
        t.blockSignals(True)
        head = t.item(row, 0)
        if head is not None:
            head.setIcon(eye_icon(self._feature_color(self._peak_for_mz(self._feat_mz_at(row))),
                                  not hidden))
        self._paint_row_hidden(t, row, hidden)
        t.blockSignals(False)

    def _feature_visibility_slot(self, want):
        """A clicked(bool)-proof slot: PySide6 reads ``lambda w=want:`` as arity-1 and feeds
        the checked flag straight into ``w``, so the predicate would arrive as False."""
        return lambda *_: self._features_set_visible(want)

    def _features_set_visible(self, want):
        """Set every feature's ``hidden`` from ``want(peak) -> should_be_visible``.

        Invert's predicate reads the very flag it is about to flip, so ``want`` is evaluated
        once per peak up front and the write loop only consumes the answer."""
        peaks = self.peaks or []
        targets = [(p, bool(want(p))) for p in peaks]
        flips = [(p, vis) for p, vis in targets if bool(p.get("hidden")) == vis]
        if not flips:
            return                                        # nothing to change → no undo entry
        self.record_undo("show/hide features")            # after the guard, per its contract
        for p, vis in flips:
            p["hidden"] = not vis
        self._decorate_feature_swatches()
        self._sync_lipid_tree()                           # keep the class tree's eyes in step
        self._refresh_feature_visible_count()
        self._sync_feature_consumers()
        if self.color_overlay_chk.isChecked():
            self.refresh_ion_image()
        n_vis = sum(1 for p in peaks if not p.get("hidden"))
        self.statusBar().showMessage(f"{n_vis}/{len(peaks)} features visible "
                                     "(hidden ones are excluded from analyses).")

    def _toggle_all_features(self):
        """The old single flip-flop, kept as the keyboard/scripting entry point: all visible →
        hide all, otherwise show all."""
        if not self.peaks:
            return
        any_visible = any(not p.get("hidden") for p in self.peaks)
        self._features_set_visible(lambda _p: not any_visible)

    def _refresh_feature_visible_count(self):
        lbl = getattr(self, "feat_visible_count", None)
        if lbl is None:
            return
        peaks = self.peaks or []
        shown = sum(1 for p in peaks if not p.get("hidden"))
        lbl.setText(f"{shown} of {len(peaks)} shown" if peaks else "")

    def _visible_peaks(self):
        """Working features minus the ones the user hid via the eye toggle. Falls back
        to all features if everything is hidden (analyses never run on an empty set)."""
        vis = [p for p in self.peaks if not p.get("hidden")]
        return vis if vis else list(self.peaks)

    def _overlay_peaks(self):
        """Strictly the eye'd-on features — NO fall-back to all. The color overlay must
        go dark when everything is hidden (unlike the analyses, which use the fall-back)."""
        return [p for p in self.peaks if not p.get("hidden")]

    def _feature_mzs(self, p):
        """The m/z a feature covers — a class composite's member ions, else its one m/z.
        Lets a class composite flatten to its constituent ions when an analysis needs real
        per-pixel columns (display, by contrast, renders the class as one composite)."""
        if p is not None and p.get("is_class"):
            return [float(m) for m in (p.get("members") or [])]
        return [float(p["mz"])] if p is not None else []

    def _visible_mzs(self):
        out = []
        for p in self._visible_peaks():
            out.extend(self._feature_mzs(p))
        return out

    # ----- visible-spectra panel ------------------------------------------- #
    def _refresh_spectra_panel(self):
        lst = getattr(self, "spectra_list", None)
        if lst is None or getattr(self, "spectrum", None) is None:
            return
        lst.blockSignals(True)
        lst.clear()
        seen = []
        base = getattr(self, "_base_curve", None)
        if base is not None:
            seen.append(base)
        for it in self.spectrum.listDataItems():
            if getattr(it, "_overlay_name", None) is not None and it not in seen:
                seen.append(it)
        for it in seen:
            name = getattr(it, "_overlay_name", "spectrum")
            li = QtWidgets.QListWidgetItem(name)
            li.setFlags(li.flags() | QtCore.Qt.ItemIsUserCheckable)
            li.setCheckState(QtCore.Qt.Checked if it.isVisible() else QtCore.Qt.Unchecked)
            li.setData(QtCore.Qt.UserRole, it)
            color = getattr(it, "_overlay_color", None)
            if color:
                li.setIcon(self._color_icon(color))
            lst.addItem(li)
        if not seen:
            placeholder = QtWidgets.QListWidgetItem("(no spectra — load data)")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            lst.addItem(placeholder)
        lst.blockSignals(False)

    def _spectra_item_changed(self, item):
        it = item.data(QtCore.Qt.UserRole)
        if it is not None:
            it.setVisible(item.checkState() == QtCore.Qt.Checked)

    # ----- active feature list shared across every pipeline ---------------- #
    def _set_flist_name(self, name):
        """Record which feature list currently drives the pipelines and refresh
        every consumer label (Segmentation, Components, RGB, Montage…)."""
        self._flist_name = name
        self._sync_feature_consumers()

    def _features_summary(self):
        name = getattr(self, "_flist_name", None) or "working set"
        n = len(self.peaks)
        if n == 0:
            return "No features yet — find peaks or load a saved list."
        n_hidden = sum(1 for p in self.peaks if p.get("hidden"))
        if n_hidden:
            return f"{n - n_hidden}/{n} features visible · list: {name}"
        return f"{n} features · list: {name}"

    def _register_feature_consumer(self, label):
        """A pipeline tab registers a QLabel here so it always shows which feature
        list its Run button will operate on."""
        self._feature_consumer_labels.append(label)
        label.setText(self._features_summary())

    def _sync_feature_consumers(self):
        summary = self._features_summary()
        if getattr(self, "flist_status", None) is not None:
            self.flist_status.setText("▶ " + summary)
        for lbl in getattr(self, "_feature_consumer_labels", []):
            try:
                lbl.setText(summary)
            except RuntimeError:                      # label was destroyed
                pass
        self._refresh_scope_bars()                    # per-analysis "data in" readouts
        if hasattr(self, "_refresh_action_states"):   # re-eval every Run button's enabled state
            self._refresh_action_states()
