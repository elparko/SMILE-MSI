"""IonTabMixin — extracted from the monolithic MainWindow (no behavior change)."""
from __future__ import annotations


import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import (spatial, imaging)
from .common import (PIXEL_COLORS, ROI_COLOR, colormap, _spectrum_view, SpectrumViewBox,
                     lock_legend, fill_table, hex_to_rgba, ControlBar, dark_image_view,
                     pick_and_build, find_spatial, enable_pinch_zoom, tool_button,
                     signal_mz_range, plot_caption, confirm, icon, guarded, ACCENT, DANGER,
                     NoScrollComboBox, NoScrollSpinBox, NoScrollDoubleSpinBox, NoScrollSlider,
                     note, section_title)

# Refine pill entries → spatial.ring_mask mode (None = the filled mask as-is,
# "invert" = every acquired pixel the mask does NOT cover).
REFINE_MODES = (("Filled", None), ("Inner rim", "inner"), ("Outer collar", "outer"),
                ("Band", "band"), ("Everything else", "invert"))
ROI_SOURCE_THRESHOLD = "Threshold (signal)"
ROI_SOURCE_REGION = "Existing region"


def _apex_bin(axis, y, mz, rel_win=5e-5):
    """Index of the spectral apex within a tight window around ``mz`` in the spectrum
    ``(axis, y)``, or ``None`` when there's no spectrum. Single source of truth so the
    halo marker and the ion-image extraction centre always land on the same bin.

    ``rel_win`` (default 50 ppm) is the half-window as a fraction of ``mz`` — matching
    the halo's long-standing snap width — floored at three bins so it never collapses
    below the grid. A window that falls entirely outside the axis returns the nearest
    bin (mirrors the halo's old behaviour), which callers rendering real features never
    hit because a catalogued m/z is always inside the acquired range."""
    if mz is None or axis is None or y is None or not len(axis):
        return None
    mz = float(mz)
    bin_w = float(axis[1] - axis[0]) if len(axis) > 1 else 0.0
    win = max(bin_w * 3, mz * rel_win)
    lo = int(np.searchsorted(axis, mz - win))
    hi = int(np.searchsorted(axis, mz + win))
    if hi > lo:
        return lo + int(np.argmax(y[lo:hi]))
    return int(np.argmin(np.abs(axis - mz)))


class IonTabMixin:
    def _step_peak(self, delta):
        if not self.peaks:
            return
        mzs = sorted(p["mz"] for p in self.peaks)
        cur = self.active_mz if self.active_mz is not None else mzs[0]
        i = int(np.argmin([abs(m - cur) for m in mzs]))
        self.set_active_mz(mzs[(i + delta) % len(mzs)])

    @staticmethod
    def _toolbar_group(*widgets):
        """Pack widgets into a tight, borderless sub-widget so the whole group can be
        shown/hidden as a unit (used to reveal ROI / overlay controls contextually)."""
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        for x in widgets:
            h.addWidget(x)
        return w

    def _zoom_image(self, factor):
        """Scale the ion image's view about its centre. factor < 1 zooms in. The image's
        ViewBox is aspect-locked, so x and y scale together."""
        if getattr(self, "iv", None) is not None:
            self.iv.view.scaleBy((factor, factor))

    def _fit_image_view(self):
        """Fit the whole ion image back into the view (undo any zoom/pan)."""
        if getattr(self, "iv", None) is not None:
            self.iv.view.autoRange()

    def _pick_roi_color(self):
        """Let the user recolor the ROI drawing tool (the rich picker seeded with the current)."""
        from . import colorpicker
        hexc = colorpicker.pick_color(self, initial=getattr(self, "_roi_color", ROI_COLOR),
                                      title="ROI drawing colour")
        if hexc:
            self._set_roi_color(hexc)

    def _set_roi_color(self, hex_color):
        """Recolor every ROI-drawing element at once — the shape outlines, the click-draw
        polygon dots/line, the freehand/band footprint fills, and the ROI spectrum trace —
        so the whole tool stays one consistent color."""
        self._roi_color = hex_color
        pen = pg.mkPen(hex_color, width=2)
        for shp in self._roi_shapes():                 # rect / circle / polygon outlines
            shp.setPen(pen)
        self._poly_draw_markers.setPen(pen)
        self._poly_draw_markers.setBrush(pg.mkBrush(*hex_to_rgba(hex_color, 120)))
        self._poly_draw_line.setPen(pg.mkPen(hex_color, width=2, style=QtCore.Qt.DashLine))
        if getattr(self, "roi_color_btn", None) is not None:
            self.roi_color_btn.setIcon(self._color_icon(hex_color))
        self._render_brush_overlay()                   # repaint the freehand footprint
        painted = (getattr(self, "_brush_mask", None) is not None and self._brush_mask is not None
                   and self._brush_mask.any())
        if painted or any(s.isVisible() for s in self._roi_shapes()):
            self._roi_spectrum()                       # recolor the band fill + ROI trace
        self.statusBar().showMessage(f"ROI drawing color set to {hex_color}.")

    def _tab_ion(self):
        # image + spectrum — the feature table now lives in the right-hand dock
        self._roi_color = ROI_COLOR              # ROI drawing color (user-overridable below)
        rightw = QtWidgets.QWidget()
        rightw.setMinimumWidth(360)
        rv = QtWidgets.QVBoxLayout(rightw)
        rv.setContentsMargins(0, 0, 0, 0)
        # ROI toolbar — a wrapping ControlBar (FlowLayout) rather than a hand-built
        # QHBoxLayout: when the ion panel is narrow the old row overflowed past its
        # minimum and Qt shrank the children below their size hints, overlapping them
        # (the "Freehand (bru…ons" cram in the screenshot). The bar now reflows whole
        # groups onto a second row instead.
        bar = ControlBar()
        self.roi_chk = QtWidgets.QCheckBox("Draw ROI")
        self.roi_chk.setToolTip("Turn on ROI drawing to outline a region on the image (pick a shape or the freehand brush). With it off, clicking the image plots that pixel's spectrum instead.")
        self.roi_chk.toggled.connect(self._toggle_roi)
        self.roi_shape = NoScrollComboBox()
        self.roi_shape.addItems(["Rectangle", "Circle", "Polygon", "Polygon (click)",
                                 "Freehand (brush)"])
        # Sources beyond drawing: the mask starts from a signal threshold or from a region
        # that already exists, and the Refine pill then grows a rim / collar / band or
        # inverts it — the whole nerve-compartment build without the brush.
        self.roi_shape.insertSeparator(self.roi_shape.count())
        self.roi_shape.addItems([ROI_SOURCE_THRESHOLD, ROI_SOURCE_REGION])
        self.roi_shape.setToolTip("Where the ROI mask starts from:\n"
                                  "• Rectangle — drag a box\n"
                                  "• Circle — drag to size a circle (interior + edge included)\n"
                                  "• Polygon — drag the 4 vertices (right-click an edge to add one)\n"
                                  "• Polygon (click) — click to drop each vertex, double-click or "
                                  "click the first point to close\n"
                                  "• Freehand (brush) — drag on the image to paint pixels "
                                  "(set the brush radius); click to dab\n"
                                  "• Threshold (signal) — pixels where a feature, a lipid class or "
                                  "a sum of features reaches a cut (holes filled)\n"
                                  "• Existing region — start from a saved region, then refine it "
                                  "(e.g. grow an outer collar)")
        self.roi_shape.currentTextChanged.connect(self._roi_shape_changed)
        # ---- Threshold source: a small popup (signal · cut · fill · islands) ----
        self.thr_btn = QtWidgets.QToolButton()
        self.thr_btn.setText("Threshold")
        self.thr_btn.setObjectName("menuButton")
        self.thr_btn.setIcon(icon("settings"))
        self.thr_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.thr_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.thr_btn.setToolTip("Which signal to threshold and where to cut it. The mask "
                                "updates live on the image as you change it.")
        _thr_menu = QtWidgets.QMenu(self.thr_btn)
        _thr_box = QtWidgets.QWidget()
        _thr_col = QtWidgets.QVBoxLayout(_thr_box)
        _thr_col.setContentsMargins(12, 10, 12, 10); _thr_col.setSpacing(7)
        _thr_col.addWidget(section_title("Signal"))
        self.thr_signal = NoScrollComboBox()
        self.thr_signal.setMinimumWidth(220)
        self.thr_signal.setToolTip("The active feature, a lipid class (composite of its "
                                   "members) or a sum of features you pick")
        self.thr_signal.currentIndexChanged.connect(self._thr_signal_changed)
        self.thr_pick_btn = tool_button(name="settings", tooltip="Choose the features to sum…",
                                        slot=self._pick_threshold_sum)
        _srow = QtWidgets.QHBoxLayout(); _srow.setContentsMargins(0, 0, 0, 0); _srow.setSpacing(4)
        _srow.addWidget(self.thr_signal, 1); _srow.addWidget(self.thr_pick_btn)
        _thr_col.addLayout(_srow)
        self._thr_sum_mzs = []
        self._thr_cache = None
        _thr_col.addWidget(section_title("Cut"))
        self.thr_slider = NoScrollSlider(QtCore.Qt.Horizontal)
        self.thr_slider.setRange(0, 100)
        self.thr_slider.setValue(60)
        self.thr_slider.setToolTip("Percentile of the signal pixels (pixels with signal > 0)")
        self.thr_cut = NoScrollDoubleSpinBox()
        self.thr_cut.setRange(0, 100)
        self.thr_cut.setDecimals(0)
        self.thr_cut.setValue(60)
        self.thr_cut.setSuffix(" %")
        self.thr_cut.setToolTip("Keep pixels at or above this percentile of the signal pixels "
                                "(60 = the brightest 40 %). Tick 'absolute' to cut on intensity.")
        self.thr_absolute = QtWidgets.QCheckBox("absolute intensity")
        self.thr_absolute.setToolTip("Cut on raw intensity instead of a percentile")
        _crow = QtWidgets.QHBoxLayout(); _crow.setContentsMargins(0, 0, 0, 0); _crow.setSpacing(6)
        _crow.addWidget(self.thr_slider, 1); _crow.addWidget(self.thr_cut)
        _thr_col.addLayout(_crow)
        _thr_col.addWidget(self.thr_absolute)
        self.thr_slider.valueChanged.connect(self._thr_slider_moved)
        self.thr_cut.valueChanged.connect(self._thr_cut_changed)
        self.thr_absolute.toggled.connect(self._thr_absolute_toggled)
        _thr_col.addWidget(section_title("Clean up"))
        self.thr_fill = QtWidgets.QCheckBox("Fill holes")
        self.thr_fill.setChecked(True)
        self.thr_fill.setToolTip("Fill enclosed holes so a fascicle interior reads as solid")
        self.thr_fill.toggled.connect(lambda *_: self._roi_spectrum())
        self.thr_min_px = NoScrollSpinBox()
        self.thr_min_px.setRange(0, 100000)
        self.thr_min_px.setValue(0)
        self.thr_min_px.setPrefix("drop islands < ")
        self.thr_min_px.setSuffix(" px")
        self.thr_min_px.setToolTip("Drop connected islands smaller than this (0 = keep all)")
        self.thr_min_px.valueChanged.connect(lambda *_: self._roi_spectrum())
        _thr_col.addWidget(self.thr_fill)
        _thr_col.addWidget(self.thr_min_px)
        self.thr_info = note("—")
        _thr_col.addWidget(self.thr_info)
        _twa = QtWidgets.QWidgetAction(_thr_menu); _twa.setDefaultWidget(_thr_box); _thr_menu.addAction(_twa)
        _thr_menu.aboutToShow.connect(self._refresh_threshold_signals)
        self.thr_btn.setMenu(_thr_menu)
        self.thr_btn.hide()
        # ---- Existing-region source: pick the region the mask starts from ----
        self.src_region = NoScrollComboBox()
        self.src_region.setMinimumWidth(140)
        self.src_region.setToolTip("The saved region this ROI starts from — refine it with a "
                                   "rim / collar / band, or take everything else")
        self.src_region.currentIndexChanged.connect(lambda *_: self._roi_spectrum())
        self.src_region.hide()
        # brush radius (px) — only used by the Freehand brush
        self.brush_size = NoScrollSpinBox()
        self.brush_size.setRange(1, 80)
        self.brush_size.setValue(3)
        self.brush_size.setPrefix("brush ")
        self.brush_size.setSuffix(" px")
        self.brush_size.setToolTip("Freehand brush radius in pixels  ([ / ] to shrink / grow)")
        # brush action: paint pixels onto the ROI, or rub them back out
        self.brush_mode = NoScrollComboBox()
        self.brush_mode.addItems(["Draw", "Erase"])
        self.brush_mode.setToolTip("Freehand brush action:\n"
                                   "• Draw — add painted pixels to the ROI\n"
                                   "• Erase — rub painted pixels back out\n"
                                   "(⌘Z undoes the last stroke)")
        # Refine: keep the source mask as-is, reduce it to a rim / collar / band of this
        # width around its boundary, or invert it. Same widgets as the old hidden Border
        # popup (same names), now a visible pill with a µm readout beside the px width.
        self.border_spin = NoScrollSpinBox()
        self.border_spin.setRange(0, 500)
        self.border_spin.setValue(6)
        self.border_spin.setPrefix("width ")
        self.border_spin.setSuffix(" px")
        self.border_spin.setToolTip("Width of the rim / collar / band in pixels (the µm readout "
                                    "uses the slide's pixel size). Applies to every source, live.")
        self.border_spin.valueChanged.connect(self._refine_changed)
        self.border_mode = NoScrollComboBox()
        self.border_mode.addItems([label for label, _ in REFINE_MODES])
        self.border_mode.setToolTip("How the source mask becomes the ROI:\n"
                                    "• Filled — the mask as-is\n"
                                    "• Inner rim — a rim inside the boundary (drops the core)\n"
                                    "• Outer collar — a collar of tissue outside the boundary "
                                    "(perineurium around an endoneurium)\n"
                                    "• Band — both sides of the boundary\n"
                                    "• Everything else — every acquired pixel the mask does "
                                    "not cover (epineurium = not endo, not peri)")
        self.border_mode.currentTextChanged.connect(self._refine_changed)
        self.border_um = QtWidgets.QLabel("")
        self.border_um.setToolTip("The width in µm, from the slide's pixel size")
        self.refine_tools = self._toolbar_group(QtWidgets.QLabel("Refine"), self.border_mode,
                                                self.border_spin, self.border_um)
        self.refine_tools.hide()
        b_clear_roi = tool_button("✕", "Clear ROI — hide it, remove its spectrum overlay, and "
                                  "clear any regions assigned from it", self._clear_roi,
                                  name="remove")
        b_save = tool_button("⤓", "Save image — publication PNG of the current ion image "
                             "(colorbar + optional scale bar)", self.save_ion_image,
                             name="export")
        # The optical/histology backdrop is a global layer (set up via File → Image setup…,
        # toggled from View → Show optical image) shown behind every spatial view — see
        # gui/optical.py. Declutter: the ROI
        # sub-controls only appear once 'Draw ROI' is engaged, the brush options only for
        # the Freehand shape — so the default bar is just Draw ROI · Save.
        # swatch to recolor the ROI tool (default magenta — legible over viridis/inferno,
        # which go yellow at high intensity, so the outline and any draw-gaps stay visible)
        self.roi_color_btn = QtWidgets.QPushButton()
        self.roi_color_btn.setMaximumWidth(34)
        self.roi_color_btn.setIcon(self._color_icon(self._roi_color))
        self.roi_color_btn.setToolTip("ROI drawing color — click to change")
        self.roi_color_btn.clicked.connect(self._pick_roi_color)
        # Primary one-click action: turn the ROI you just drew straight into an annotated
        # feature list (saved as a region, peaks picked inside it, lipids identified) —
        # right here on the drawing toolbar, instead of hunting in the Regions panel.
        self.b_roi_features = QtWidgets.QPushButton("→ Feature list")
        self.b_roi_features.setIcon(icon("navigate"))
        self.b_roi_features.setToolTip(
            "Turn the ROI you drew into a feature list — saves it as a region, finds peaks "
            "inside it, and identifies the lipids. Switch back any time via the "
            "Features ▸ Feature set selector.")
        self.b_roi_features.setObjectName("primaryAction")  # headline one-click action of the ROI bar
        self.b_roi_features.clicked.connect(lambda: self.features_from_region(prefer_roi=True))
        # Save the built mask as a named region (name pre-filled from the source + refine).
        self.b_roi_region = QtWidgets.QPushButton("Save as region")
        self.b_roi_region.setIcon(icon("add"))
        self.b_roi_region.setToolTip("Save the current ROI mask (after Refine) as a named region "
                                     "in the Regions panel — the name is pre-filled from the "
                                     "source, e.g. 'endo · outer collar 6 px' or 'not endo'.")
        self.b_roi_region.clicked.connect(self._save_roi_as_region)
        # The whole nerve build in one dialog: threshold → outer collar → everything else.
        self.b_compartments = QtWidgets.QPushButton("Compartments…")
        self.b_compartments.setIcon(icon("add"))
        self.b_compartments.setToolTip("Build endoneurium / perineurium / epineurium regions in "
                                       "one go: threshold a signal for the endoneurium, grow a "
                                       "collar outside it for the perineurium, take everything "
                                       "else for the epineurium — previewed, then created as "
                                       "three grouped regions.")
        self.b_compartments.clicked.connect(self._open_compartments_dialog)
        # The shape picker carried the whole ROI vocabulary ("Freehand (brush)") yet got
        # squeezed to a few characters on the tight strip — give it room to show its label.
        self.roi_shape.setMinimumWidth(150)
        # Secondary ROI knobs all tuck one click deeper into a popover so the revealed ROI
        # sub-bar stays to just "→ Feature list · shape · ROI options ▾ · ✕" instead of
        # cramming the brush mode/size combos inline (where they truncated to "Dr" / "n 3
        # px" and ate the toolbar's stretch). The widgets are reparented as-is — same
        # objects, same signal connections.
        self.roi_opts_btn = QtWidgets.QToolButton()
        self.roi_opts_btn.setText("ROI options")          # ▾ drawn by QSS (menuButton chevron)
        self.roi_opts_btn.setObjectName("menuButton")     # shared dropdown pill (single arrow)
        self.roi_opts_btn.setIcon(icon("settings"))       # a configure-this menu (brush / size / colour)
        self.roi_opts_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.roi_opts_btn.setToolTip("Freehand brush action / size and the ROI drawing colour")
        self.roi_opts_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        _roi_menu = QtWidgets.QMenu(self.roi_opts_btn)
        _roi_box = QtWidgets.QWidget()
        _roi_col = QtWidgets.QVBoxLayout(_roi_box)
        _roi_col.setContentsMargins(12, 10, 12, 10); _roi_col.setSpacing(7)
        # Brush section — only meaningful for the Freehand shape, so the whole block
        # (title + mode + size) shows/hides as a unit via _sync_brush_tools.
        self.brush_tools = QtWidgets.QWidget()
        _brush_col = QtWidgets.QVBoxLayout(self.brush_tools)
        _brush_col.setContentsMargins(0, 0, 0, 0); _brush_col.setSpacing(7)
        _brush_col.addWidget(section_title("Brush"))
        _brush_col.addWidget(self.brush_mode)
        _brush_col.addWidget(self.brush_size)
        _roi_col.addWidget(self.brush_tools)
        _roi_col.addWidget(section_title("Colour"))
        _crow = QtWidgets.QHBoxLayout(); _crow.setContentsMargins(0,0,0,0); _crow.setSpacing(6)
        _crow.addWidget(QtWidgets.QLabel("ROI colour")); _crow.addWidget(self.roi_color_btn, 1)
        _roi_col.addLayout(_crow)
        _wa = QtWidgets.QWidgetAction(_roi_menu); _wa.setDefaultWidget(_roi_box); _roi_menu.addAction(_wa)
        self.roi_opts_btn.setMenu(_roi_menu)
        # Three flow items (source · refine · actions) rather than one, so a narrow panel
        # wraps them as whole groups instead of squeezing the widgets.
        self.roi_tools = self._toolbar_group(self.roi_shape, self.thr_btn, self.src_region)
        self.roi_tools.hide()
        self.roi_actions = self._toolbar_group(self.b_roi_region, self.b_roi_features,
                                               self.b_compartments, self.roi_opts_btn, b_clear_roi)
        self.roi_actions.hide()
        # image zoom controls — trackpad pinch / scroll-wheel zoom about the cursor are
        # always live (see enable_pinch_zoom below); these buttons add explicit zoom in/out
        # and a fit-to-image, handy while drawing an ROI.
        b_zoom_in = QtWidgets.QPushButton("＋")
        b_zoom_in.setMaximumWidth(34)
        b_zoom_in.setIcon(icon("zoom-in"))
        b_zoom_in.setText("")
        b_zoom_in.setToolTip("Zoom in  (or pinch / scroll on the image)")
        b_zoom_in.setAccessibleName("Zoom in")
        b_zoom_in.clicked.connect(lambda: self._zoom_image(1.0 / 1.3))
        b_zoom_out = QtWidgets.QPushButton("－")
        b_zoom_out.setMaximumWidth(34)
        b_zoom_out.setIcon(icon("zoom-out"))
        b_zoom_out.setText("")
        b_zoom_out.setToolTip("Zoom out  (or pinch / scroll on the image)")
        b_zoom_out.setAccessibleName("Zoom out")
        b_zoom_out.clicked.connect(lambda: self._zoom_image(1.3))
        b_fit = QtWidgets.QPushButton("Fit")
        b_fit.setIcon(icon("refresh"))
        b_fit.setToolTip("Fit the whole image to the view")
        b_fit.clicked.connect(self._fit_image_view)
        self.zoom_tools = self._toolbar_group(b_fit, b_zoom_in, b_zoom_out)
        # Rotate the slide flat — vertically-mounted tissue (e.g. a "Left_Vertical"
        # acquisition) wastes the wide viewport; this turns it 90° everywhere at once.
        self.rotate_btn = QtWidgets.QToolButton()
        self.rotate_btn.setText("90°")
        self.rotate_btn.setIcon(icon("refresh"))
        self.rotate_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.rotate_btn.setToolTip("Rotate the image 90° clockwise. Applies to every spatial view "
                                   "(ion image, segmentation, ROIs) at once and is saved with the "
                                   "session, so it stays put — lay a vertical slide flat with this.")
        self.rotate_btn.clicked.connect(self._rotate_image)
        # The toggle is always shown; the sub-tools are a separate flow item so they wrap
        # onto their own row when the panel is narrow (and, being hidden until 'Draw ROI'
        # is on, reserve no space meanwhile — FlowLayout skips empty items).
        bar.add(self.roi_chk)
        bar.add(self.roi_tools)
        bar.add(self.refine_tools)
        bar.add(self.roi_actions)
        # Colormap picker — promoted onto the render toolbar (it used to be tucked into the
        # right-hand panel). Drives the single-ion image; greyed out while the multi-channel
        # Color overlay is on, since each feature carries its own colour then. Created here
        # because the toolbar is built before the right panel — every other view shares this
        # same self.cmap_combo, so they keep working unchanged.
        self.cmap_combo = NoScrollComboBox()
        self.cmap_combo.addItems(["viridis", "inferno", "magma", "plasma", "cividis",
                                  "turbo", "hot", "jet", "gray"])
        self.cmap_combo.setMaximumWidth(140)   # room for the box border + drawn chevron gutter
        self.cmap_combo.setToolTip("Colormap for the single-ion image. viridis/cividis are "
                                   "perceptually uniform; magma/plasma/inferno for warm "
                                   "contrast; gray/hot/jet for publication or high contrast. "
                                   "Disabled while Color overlay is on.")
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._display_setting_changed())
        bar.add_group("Colormap", self.cmap_combo)
        # Composite images (total-class · A/B ratio) live in a small on-demand dialog opened
        # from here — right by the view they render into. Built once (hidden) so the class /
        # ratio combos exist for the feature pipeline to fill; see _build_composite_dialog.
        self.b_composite = QtWidgets.QPushButton("Composite…")
        self.b_composite.setIcon(icon("settings"))
        self.b_composite.setToolTip("Total-class and A/B ratio composite images, built from "
                                    "the feature list and drawn into this view")
        self.b_composite.clicked.connect(self._open_composite_dialog)
        bar.add(self.b_composite)
        # view transforms (rotate + zoom/fit) grouped together, then the export action
        bar.add_group(self.rotate_btn, self.zoom_tools)
        bar.add(b_save)
        self._build_composite_dialog()
        rv.addWidget(bar)
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.iv = pg.ImageView()
        # Drop pyqtgraph's built-in "ROI" + "Menu" buttons at the bottom of the viewer: the
        # app has its own ROI drawing tools, and clicking these reflowed the view (which used
        # to displace the in-image legend/scale bar). Also drop the right-side histogram /
        # colorbar panel — the in-image legend already shows the colour gradient
        # and max %, so that panel is redundant clutter.
        for _b in ("roiBtn", "menuBtn"):
            _w = getattr(self.iv.ui, _b, None)
            if _w is not None:
                _w.hide()
        self._show_histogram(False)
        dark_image_view(self.iv)   # tissue canvas reads against black in light *and* dark mode
        self.iv.view.invertY(True)
        self.iv.view.setAspectLocked(True)
        self._iv_pinch = enable_pinch_zoom(self.iv)   # trackpad pinch-to-zoom (kept off-GC)
        rc = self._roi_color                          # centralized ROI color (set above)
        self.roi = pg.RectROI([1, 1], [5, 5], pen=pg.mkPen(rc, width=2))
        self.roi.hide()
        self.iv.view.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self._roi_spectrum)
        # freehand polygon ROI for irregular tissue outlines (hidden until 'Polygon')
        self.poly_roi = pg.PolyLineROI([[2, 2], [12, 2], [12, 12], [2, 12]], closed=True,
                                       pen=pg.mkPen(rc, width=2))
        self.poly_roi.hide()
        self.iv.view.addItem(self.poly_roi)
        self.poly_roi.sigRegionChangeFinished.connect(self._roi_spectrum)
        # circular ROI — drag to size a circle over a feature; includes its interior + edge
        self.circle_roi = pg.CircleROI([2, 2], [10, 10], pen=pg.mkPen(rc, width=2))
        self.circle_roi.hide()
        self.iv.view.addItem(self.circle_roi)
        self.circle_roi.sigRegionChangeFinished.connect(self._roi_spectrum)
        # click-to-draw polygon: live vertex dots + a dashed outline, finalised into
        # poly_roi when the user closes the shape (so all masking logic is unchanged)
        self._poly_draw_pts = []
        self._poly_draw_markers = pg.ScatterPlotItem(size=11, symbol="o",
                                                     pen=pg.mkPen(rc, width=2),
                                                     brush=pg.mkBrush(*hex_to_rgba(rc, 120)))
        self._poly_draw_markers.setZValue(60)
        self._poly_draw_markers.hide()
        self.iv.view.addItem(self._poly_draw_markers)
        self._poly_draw_line = pg.PlotCurveItem(
            pen=pg.mkPen(rc, width=2, style=QtCore.Qt.DashLine))
        self._poly_draw_line.setZValue(59)
        self._poly_draw_line.hide()
        self.iv.view.addItem(self._poly_draw_line)
        # translucent footprint of the region(s) selected in the sidebar, in their colors
        self._region_overlay = pg.ImageItem()
        self._region_overlay.setZValue(40)
        self._region_overlay.hide()
        self.iv.view.addItem(self._region_overlay)
        # freehand-brush footprint (what the user paints), drawn above the region overlay
        self._brush_overlay = pg.ImageItem()
        self._brush_overlay.setZValue(45)
        self._brush_overlay.hide()
        self.iv.view.addItem(self._brush_overlay)
        self._brush_mask = None
        # band/ring footprint: fills the pixels an Inner/Outer/Band border actually keeps,
        # so a >0 px border is visible (the drawn shape outline always spans the full shape)
        self._band_overlay = pg.ImageItem()
        self._band_overlay.setZValue(46)
        self._band_overlay.hide()
        self.iv.view.addItem(self._band_overlay)
        # paint on left-drag when the Freehand brush is active; otherwise pan as usual
        self._orig_vb_drag = self.iv.view.mouseDragEvent
        self.iv.view.mouseDragEvent = self._brush_drag_event
        self.iv.getView().scene().sigMouseClicked.connect(self._image_clicked)
        # optical/histology backdrop sits behind the ion image in this same view (it also
        # backs the colour overlay, which renders into self.iv) — see gui/optical.py
        self._register_optical_view(self.iv.view, self.iv.getImageItem())
        # In-image legend + scale bar, painted beside the tissue subject and live
        # (WYSIWYG — the on-screen twin of the export). Mouse-transparent so it never blocks
        # ROI drawing / panning underneath; double-clicking a label renames it (_image_clicked).
        from .ionannotations import IonAnnotations
        self.ion_annot = IonAnnotations(self.iv)
        self.iv.setToolTip("Click a pixel to plot its spectrum  ·  double-click an ion label in the legend to rename it  ·  turn on 'Draw ROI' to outline a region.")
        right.addWidget(self.iv)
        self.spectrum = pg.PlotWidget(viewBox=SpectrumViewBox())  # box-drag zooms m/z only
        self.spectrum.setLabel("bottom", "m/z")
        self.spectrum.setLabel("left", "intensity")
        # pin the legend to the top-right corner (just inside the plot frame, below the
        # axis ticks) rather than pyqtgraph's default top-left where it crowds the Y axis.
        lock_legend(self.spectrum.addLegend(offset=(-10, 10)))   # legend stays pinned, can't be dragged off
        _spec_vb = _spectrum_view(self.spectrum)  # left-drag = X-only box zoom, baseline at 0
        # the found-peak rail (see _mark_peaks) lives in a thin gutter *below* the m/z
        # axis, so let Y open up just below the baseline. Y isn't mouse-pannable, so
        # dropping the yMin floor only frees the rail's gutter — the view never drifts.
        _spec_vb.setLimits(yMin=None)
        _spec_vb.sigYRangeChanged.connect(self._reposition_peak_rail)
        self._rail_busy = False
        self.spectrum.scene().sigMouseClicked.connect(self._spectrum_clicked)
        self.mz_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(ACCENT, width=2))
        self.mz_line.sigPositionChangeFinished.connect(self._line_moved)
        self.spectrum.addItem(self.mz_line)
        # halo on the selected peak's apex — makes the active feature pop when you pick
        # it from the feature list (in addition to the green cursor line)
        self.peak_marker = pg.ScatterPlotItem(size=16, symbol="o",
                                              pen=pg.mkPen(ACCENT, width=2.5),
                                              brush=pg.mkBrush(85, 168, 104, 70))
        self.peak_marker.setZValue(50)
        self.spectrum.addItem(self.peak_marker)
        # spectrum toolbar: collapse toggle · zoom controls · interaction hint
        specw = QtWidgets.QWidget()
        self.specw = specw
        sv = QtWidgets.QVBoxLayout(specw)
        sv.setContentsMargins(0, 0, 0, 0)
        bar2 = QtWidgets.QHBoxLayout()
        # chevron that collapses the spectrum plot down to this one-line header (handing
        # the space to the ion image) and restores it again. The header stays visible
        # while collapsed, so there's always a way to bring the spectrum back.
        self.spec_collapse_btn = QtWidgets.QToolButton()
        self.spec_collapse_btn.setArrowType(QtCore.Qt.DownArrow)
        self.spec_collapse_btn.setAutoRaise(True)
        self.spec_collapse_btn.setToolTip("Hide the spectrum panel")
        self.spec_collapse_btn.clicked.connect(self._toggle_spectrum_panel)
        bar2.addWidget(self.spec_collapse_btn)
        bar2.addWidget(QtWidgets.QLabel("Spectrum:"))
        # the zoom/clear tools + hint only make sense with the plot shown — grouped into
        # one widget so they hide together when the panel collapses.
        self._spec_tools = QtWidgets.QWidget()
        st = QtWidgets.QHBoxLayout(self._spec_tools)
        st.setContentsMargins(0, 0, 0, 0)
        b_reset = QtWidgets.QPushButton("Reset (R)")
        b_reset.setIcon(icon("refresh"))
        b_reset.setToolTip("Refit the spectrum to all visible traces  (R, or double-click the plot)")
        b_reset.clicked.connect(lambda: self._reset_spectrum_view())
        b_zin = QtWidgets.QPushButton("Zoom +")
        b_zin.setIcon(icon("zoom-in"))
        b_zin.setToolTip("Zoom in on the m/z axis  (or drag across the plot to zoom a window)")
        b_zin.clicked.connect(lambda: self.spectrum.getViewBox().scaleBy(x=0.7))
        b_zout = QtWidgets.QPushButton("Zoom −")
        b_zout.setIcon(icon("zoom-out"))
        b_zout.setToolTip("Zoom out on the m/z axis  (Reset refits to all traces)")
        b_zout.clicked.connect(lambda: self.spectrum.getViewBox().scaleBy(x=1.4))
        b_clear_px = QtWidgets.QPushButton("Clear pixels")
        b_clear_px.setIcon(icon("delete"))
        b_clear_px.setToolTip("Remove the per-pixel spectra overlaid by clicking the ion image "
                              "(click any pixel while the ROI tool is off)")
        b_clear_px.clicked.connect(self._clear_pixel_spectra)
        b_clear_all = QtWidgets.QPushButton("Clear all")
        b_clear_all.setIcon(icon("delete"))
        b_clear_all.setToolTip("Reset the spectrum to its base trace — remove every overlaid "
                               "selection at once: per-pixel clicks, the drawn-ROI mean, and "
                               "region means. The ROI and regions themselves are kept.")
        b_clear_all.clicked.connect(self._clear_spectrum_overlays)
        st.addWidget(b_reset)
        st.addWidget(b_zin)
        st.addWidget(b_zout)
        st.addWidget(b_clear_px)
        st.addWidget(b_clear_all)
        # Trace manager: every spectrum overlaid here (mean/skyline base, ROI mean, per-pixel
        # clicks, region means) with show/hide toggles. Lives behind this button instead of a
        # right-panel section, so it sits with the plot it manages.
        self.b_traces = QtWidgets.QPushButton("Traces…")
        self.b_traces.setIcon(icon("settings"))
        self.b_traces.setToolTip("Show or hide each spectrum overlaid on this plot — the "
                                 "mean/skyline base, ROI mean, per-pixel clicks, and region traces")
        self.b_traces.clicked.connect(self._open_traces_dialog)
        st.addWidget(self.b_traces)
        st.addWidget(QtWidgets.QLabel("  drag = zoom · click a peak to select · ←/→ step · P = pick"))
        bar2.addWidget(self._spec_tools)
        bar2.addStretch(1)
        sv.addLayout(bar2)
        sv.addWidget(self.spectrum)
        self._build_spectra_dialog()
        right.addWidget(specw)
        right.setChildrenCollapsible(False)
        right.setSizes([520, 280])
        self.spectrum_splitter = right
        self._ion_hint = plot_caption("Click a pixel to plot its spectrum  ·  double-click an ion label to rename it  ·  turn on 'Draw ROI' to outline a region.")
        self._spectrum_sizes = None        # remembered split sizes while collapsed
        rv.addWidget(right)
        rv.addWidget(self._ion_hint)
        self.tabs.addTab(rightw, "Ion image")

    # ----- on-demand panels relocated off the right dock ------------------- #
    def _build_composite_dialog(self):
        """Small non-modal dialog for the total-class / A/B ratio composite tools, opened
        from the ion-image toolbar's 'Composite…' button. Built once (hidden) so the class /
        ratio combos exist for the feature pipeline to populate even before it's first shown."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Composite images")
        lay = QtWidgets.QVBoxLayout(dlg)
        self._populate_composite_section(lay)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.hide)
        lay.addWidget(btns)
        self._composite_dialog = dlg

    def _open_composite_dialog(self):
        # The dialog's own note ("Find peaks first to fill the menus") and the do_* guards
        # cover the empty-state, so just open it — no contradictory "can't do this" flash.
        self._show_dialog(self._composite_dialog)

    def _build_spectra_dialog(self):
        """Small non-modal dialog for the visible-spectra trace manager, opened from the
        spectrum toolbar's 'Traces…' button. Built once (hidden) so ``spectra_list`` exists
        for ``_refresh_spectra_panel`` to keep current as overlays come and go."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Spectrum traces")
        lay = QtWidgets.QVBoxLayout(dlg)
        note = QtWidgets.QLabel(
            "Each spectrum overlaid on the plot — tick to show/hide. 'Clear ROI' removes the "
            "ROI mean trace; 'Clear pixels', 'Clear all' (reset to the base trace), and "
            "'Reset (R)' are on the spectrum toolbar.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self._populate_spectra_section(lay)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.hide)
        lay.addWidget(btns)
        self._spectra_dialog = dlg

    def _open_traces_dialog(self):
        self._refresh_spectra_panel()
        self._show_dialog(self._spectra_dialog)

    def _toggle_spectrum_panel(self, *_):
        """Collapse the bottom spectrum plot down to its one-line header (handing the
        freed height to the ion image), or restore it to its previous size. The header
        row with the chevron stays visible either way, so the panel can always come back."""
        split = self.spectrum_splitter
        if self.spectrum.isVisible():
            self._spectrum_sizes = split.sizes()          # remember to restore later
            self.spectrum.hide()
            self._spec_tools.hide()
            self.spec_collapse_btn.setArrowType(QtCore.Qt.RightArrow)
            self.spec_collapse_btn.setToolTip("Show the spectrum panel")
            # with the plot + tools hidden, specw's minimum is just the header row; asking
            # the splitter to give the bottom pane ~nothing clamps it to that header height.
            split.setSizes([max(1, sum(split.sizes())), 1])
        else:
            self.spectrum.show()
            self._spec_tools.show()
            self.spec_collapse_btn.setArrowType(QtCore.Qt.DownArrow)
            self.spec_collapse_btn.setToolTip("Hide the spectrum panel")
            split.setSizes(self._spectrum_sizes or [520, 280])

    def _projection(self):
        return "max" if self.proj_combo.currentText().startswith("skyline") else "mean"

    def _plot_mean_spectrum(self):
        if self.ds is None:
            return
        proj = self._projection()
        # Fast-mode backdrop: once the ion cache (cube) is built, serve mean *and* skyline
        # from it so they match the cube-served ROI/region overlays in scale & resolution
        # and the spectrum view stays instant — switching to skyline no longer streams every
        # pixel off disk on the GUI thread. Without a cube, fall back to the exact stream.
        cube_spec = (self.ds.cube_max_spectrum() if proj == "max"
                     else self.ds.cube_mean_spectrum())
        pending = False
        if cube_spec is not None:
            axis, spec = cube_spec
        elif (proj == "max" and getattr(self, "_cube_building", False)
              and self.ds.skyline_needs_disk_pass()):
            # The skyline would stream every spectrum off disk on the GUI thread (minutes
            # on a large slide). Show the already-primed mean and swap in the cube-served
            # skyline when the fast cache lands (_on_fast_cache_ready re-plots).
            axis, spec = self.ds.mean_spectrum()
            pending = True
            self._pending_spectrum_refresh = True
            self.statusBar().showMessage(
                "Showing the mean spectrum — the skyline appears when the fast cache finishes.")
        else:
            axis, spec = (self.ds.max_spectrum() if proj == "max" else self.ds.mean_spectrum())
        self._disp_axis = np.asarray(axis, dtype=float)
        self._disp_y = np.asarray(spec, dtype=float)
        self.spectrum.clear()
        self.spectrum.addItem(self.mz_line)
        self.spectrum.addItem(self.peak_marker)        # survives the clear()
        name = ("mean (skyline pending fast cache)" if pending
                else "skyline (max)" if proj == "max" else "mean")
        base = self.spectrum.plot(axis, spec, pen=pg.mkPen("#7aa6da", width=1), name=name)
        base._overlay_name = f"{name} spectrum"
        base._overlay_color = "#7aa6da"
        self._base_curve = base
        self.spectrum.setLabel("left", f"{name} intensity")
        # pin the pan/zoom limits to the full acquired range (you can still explore it
        # all), but open on just the m/z window that holds signal — the empty tail past
        # the last real peak is cropped out so the spectrum reads like a publication
        # figure rather than a sliver of peaks lost in dead space.
        vb = self.spectrum.getViewBox()
        # yMin stays unpinned so the found-peak rail's gutter below the baseline shows
        # (see _mark_peaks); Y isn't mouse-pannable, so the view can't drift down.
        vb.setLimits(xMin=float(axis[0]), xMax=float(axis[-1]), yMin=None)
        # Keep Y auto-fitting to the visible peaks — a one-shot autoRange() would *disable*
        # continuous auto-ranging, freezing Y to the base spectrum's scale so taller/shorter
        # overlays (region & feature spectra) never appear. Set X explicitly and re-assert
        # the auto-Y that _spectrum_view promises.
        lo, hi = signal_mz_range(axis, spec) or (float(axis[0]), float(axis[-1]))
        vb.setXRange(lo, hi, padding=0)
        vb.enableAutoRange(axis="y", enable=True)
        vb.setAutoVisible(y=True)
        self._highlight_active_peak()                  # restore the halo on the active peak
        self._refresh_spectra_panel()

    def _reset_spectrum_view(self):
        """Fit the spectrum to all visible traces, then re-assert auto-Y. A bare
        autoRange() would *disable* continuous auto-ranging (pyqtgraph turns it off
        after a one-shot fit), freezing Y to whatever's shown — so later overlays or
        X-zooms wouldn't refit. Used by the Reset buttons and the double-click reset."""
        vb = self.spectrum.getViewBox()
        vb.autoRange()
        vb.enableAutoRange(axis="y", enable=True)
        vb.setAutoVisible(y=True)

    def _spectrum_clicked(self, ev):
        """Single click snaps the line to the apex of the peak you clicked on;
        double-click resets the view."""
        if self.ds is None:
            return
        vb = self.spectrum.getViewBox()
        if ev.double():
            self._reset_spectrum_view()
            return
        x = float(vb.mapSceneToView(ev.scenePos()).x())
        self.set_active_mz(self._snap_mz(x))

    def _snap_mz(self, x):
        """Snap a clicked/dragged m/z to the apex of the nearest peak in the
        *displayed* spectrum, so the line lands exactly on the peak you aimed at.
        When that apex coincides with a picked peak, return the peak's centroid so
        the ion image stays tied to the feature list."""
        axis, y = self._disp_axis, self._disp_y
        if axis is None or y is None or not len(axis):
            if self.peaks:                                   # fall back to picked peaks
                pmz = np.array([p["mz"] for p in self.peaks])
                return float(pmz[np.argmin(np.abs(pmz - x))])
            return float(x)
        # search window: ~2% of the visible width (so it tracks zoom), >= one bin
        (x0, x1), _ = self.spectrum.getViewBox().viewRange()
        bin_w = float(axis[1] - axis[0]) if len(axis) > 1 else 0.0
        win = max((x1 - x0) * 0.02, bin_w)
        lo = int(np.searchsorted(axis, x - win))
        hi = int(np.searchsorted(axis, x + win))
        apex = lo + int(np.argmax(y[lo:hi])) if hi > lo else int(np.argmin(np.abs(axis - x)))
        mz = float(axis[apex])
        if self.peaks:                                       # prefer a coincident feature
            pmz = np.array([p["mz"] for p in self.peaks])
            j = int(np.argmin(np.abs(pmz - mz)))
            if abs(float(pmz[j]) - mz) <= max(win, bin_w):
                return float(pmz[j])
        return mz

    def _build_pixel_lookup(self):
        # A (height x width) grid mapping displayed (row, col) -> pixel index (-1 = no pixel).
        # Vectorized scatter, not a Python dict comprehension: the dict cost is O(n_pixels) at
        # Python speed and a rotate rebuilds it, so on a big slide (~1M px) it alone took
        # seconds. The array build + click lookup are effectively instant.
        rows, cols = self.ds._pixel_rows_cols()
        grid = np.full((self.ds.height, self.ds.width), -1, dtype=np.int64)
        grid[rows, cols] = np.arange(len(rows))
        self._pix_lookup = grid

    # ----- ROI drawing + pixel/region spectra ------------------------------ #
    def _roi_shapes(self):
        return (self.roi, self.circle_roi, self.poly_roi)

    def _active_roi(self):
        s = self.roi_shape.currentText() if getattr(self, "roi_shape", None) is not None else "Rectangle"
        if s.startswith("Polygon"):
            return self.poly_roi
        if s == "Circle":
            return self.circle_roi
        return self.roi

    def _poly_click_mode(self):
        return (getattr(self, "roi_shape", None) is not None
                and self.roi_shape.currentText() == "Polygon (click)")

    # ----- sources beyond drawing (threshold / existing region) ------------- #
    def _threshold_mode(self):
        return (getattr(self, "roi_shape", None) is not None
                and self.roi_shape.currentText() == ROI_SOURCE_THRESHOLD)

    def _region_source_mode(self):
        return (getattr(self, "roi_shape", None) is not None
                and self.roi_shape.currentText() == ROI_SOURCE_REGION)

    def _source_mode(self):
        """True when the ROI mask comes from a signal threshold or a saved region rather
        than a drawn shape — there is no outline then, so the footprint is always painted."""
        return self._threshold_mode() or self._region_source_mode()

    def _sync_source_tools(self):
        thr, reg = self._threshold_mode(), self._region_source_mode()
        if getattr(self, "thr_btn", None) is not None:
            self.thr_btn.setVisible(thr)
        if getattr(self, "src_region", None) is not None:
            self.src_region.setVisible(reg)
        if thr:
            self._refresh_threshold_signals()
        if reg:
            self._refresh_source_regions()

    def _refresh_source_regions(self):
        combo = self.src_region
        keep = combo.currentText()
        names = [rg["name"] for rg in (self.regions or [])]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if keep in names:
            combo.setCurrentText(keep)
        combo.blockSignals(False)
        combo.setEnabled(bool(names))
        combo.setToolTip("Draw or save a region first" if not names else
                         "The saved region this ROI starts from — refine it with a rim / "
                         "collar / band, or take everything else")

    def _region_source_mask(self):
        name = self.src_region.currentText()
        rg = next((r for r in (self.regions or []) if r.get("name") == name), None)
        if rg is None:
            return None
        mask = self._region_pixel_mask(rg)
        return mask if mask is not None and mask.any() else None

    @staticmethod
    def _peak_label(p):
        lip = (p.get("lipid") or "").strip() if p else ""
        return lip if lip else (f"m/z {float(p['mz']):.4f}" if p else "—")

    def _active_peak(self):
        if self.active_mz is None:
            return None
        return next((p for p in (self.peaks or [])
                     if abs(float(p["mz"]) - float(self.active_mz)) < 1e-4), None)

    def _signal_options(self, sum_mzs=()):
        """``[(label, kind)]`` for a signal picker: the active feature, each lipid class
        (composite of its members), and a hand-picked sum of features."""
        ap = self._active_peak()
        active_label = self._peak_label(ap) if ap else (
            f"m/z {self.active_mz:.4f}" if self.active_mz is not None else "none selected")
        out = [(f"Active feature ({active_label})", ("active",))]
        cmap = getattr(self, "_class_map", {}) or {}
        for cls in sorted(cmap, key=lambda c: -len(cmap[c])):
            out.append((f"Class: {cls} ({len(cmap[cls])} ions)", ("class", cls)))
        n = len(sum_mzs)
        out.append((f"Sum of features… ({n} chosen)" if n else "Sum of features…", ("sum",)))
        return out

    @staticmethod
    def _fill_signal_combo(combo, options):
        keep = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, kind in options:
            combo.addItem(label, kind)
        idx = next((i for i in range(combo.count()) if combo.itemData(i) == keep), 0)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _refresh_threshold_signals(self):
        self._fill_signal_combo(self.thr_signal, self._signal_options(self._thr_sum_mzs))

    def _thr_signal_changed(self, *_):
        self._thr_cache = None
        kind = self.thr_signal.currentData() or ("active",)
        if kind[0] == "sum" and not self._thr_sum_mzs:
            self._pick_threshold_sum()
            return
        self._roi_spectrum()

    def _pick_threshold_sum(self):
        chosen = self._pick_feature_sum(self._thr_sum_mzs)
        if chosen is None:
            return
        self._thr_sum_mzs = chosen
        self._thr_cache = None
        self._refresh_threshold_signals()
        self._roi_spectrum()

    def _pick_feature_sum(self, initial=()):
        """Multi-pick the features whose per-pixel intensities are summed into one signal.
        Returns the chosen m/z list, or ``None`` when cancelled / nothing to pick from."""
        peaks = list(self.peaks or [])
        if not peaks:
            self.statusBar().showMessage("Find peaks first — there are no features to sum.")
            return None
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Sum of features")
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(note("Tick the features to add together (e.g. the sulfatide species)."))
        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        chosen = {round(m, 4) for m in initial}
        for p in peaks:
            it = QtWidgets.QListWidgetItem(f"{self._peak_label(p)}   ({float(p['mz']):.4f})")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if round(float(p["mz"]), 4) in chosen
                             else QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, float(p["mz"]))
            lst.addItem(it)
        v.addWidget(lst, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.resize(380, 460)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return [float(lst.item(i).data(QtCore.Qt.UserRole)) for i in range(lst.count())
                if lst.item(i).checkState() == QtCore.Qt.Checked]

    def _thr_slider_moved(self, v):
        if self.thr_absolute.isChecked():
            return
        self.thr_cut.blockSignals(True)
        self.thr_cut.setValue(float(v))
        self.thr_cut.blockSignals(False)
        self._roi_spectrum()

    def _thr_cut_changed(self, v):
        if not self.thr_absolute.isChecked():
            self.thr_slider.blockSignals(True)
            self.thr_slider.setValue(int(round(float(v))))
            self.thr_slider.blockSignals(False)
        self._roi_spectrum()

    def _thr_absolute_toggled(self, on):
        self.thr_slider.setEnabled(not on)
        self.thr_cut.blockSignals(True)
        if on:
            vec, _ = self._threshold_signal_vector()
            top = float(np.nanmax(vec)) if vec is not None and vec.size else 1.0
            self.thr_cut.setSuffix("")
            self.thr_cut.setDecimals(3)
            self.thr_cut.setRange(0.0, max(top, 1.0) * 10)
            pos = vec[vec > 0] if vec is not None else np.array([])
            self.thr_cut.setValue(float(np.percentile(pos, self.thr_slider.value())) if pos.size else 0.0)
        else:
            self.thr_cut.setSuffix(" %")
            self.thr_cut.setDecimals(0)
            self.thr_cut.setRange(0, 100)
            self.thr_cut.setValue(float(self.thr_slider.value()))
        self.thr_cut.blockSignals(False)
        self._roi_spectrum()

    def _threshold_signal_vector(self):
        """``(per-pixel signal, label)`` for the Threshold source — memoised on its inputs
        so dragging the cut never re-extracts."""
        if self.ds is None:
            return None, ""
        kind = self.thr_signal.currentData() or ("active",)
        vec, label, key = self._signal_vector_for(kind, self._thr_sum_mzs, compute=False)
        if vec is None and key is not None:
            cached = self._thr_cache
            if cached is not None and cached[0] == key:
                return cached[1], label
            vec, label, key = self._signal_vector_for(kind, self._thr_sum_mzs)
            self._thr_cache = (key, vec)
        return vec, label

    def _signal_vector_for(self, kind, sum_mzs=(), compute=True):
        """``(vector, label, cache key)`` for a signal kind (``("active",)``, ``("class",
        name)`` or ``("sum",)``) at the bar's ppm / reduce / norm — the active feature (class
        composites included), a lipid class, or the given sum. With ``compute=False`` only the
        label and key come back (vector ``None``) so a caller can consult its own cache."""
        if self.ds is None:
            return None, "", None
        weight = self.composite_weight
        if kind[0] == "class":
            mzs = [self._apex_mz(m) for m in (getattr(self, "_class_map", {}) or {}).get(kind[1], [])]
            label = f"class {kind[1]}"
        elif kind[0] == "sum":
            mzs = [self._apex_mz(m) for m in sum_mzs]
            label = f"sum of {len(mzs)} features"
        else:
            ap = self._active_peak()
            if ap is not None and ap.get("is_class"):
                mzs = [self._apex_mz(m) for m in (ap.get("members") or [])]
            elif self.active_mz is not None:
                mzs = [self._apex_mz(float(self.active_mz))]
            else:
                mzs = []
            label = self._peak_label(ap) if ap else (
                f"m/z {self.active_mz:.4f}" if self.active_mz is not None else "")
        if not mzs:
            return None, label, None
        key = (kind, tuple(round(float(m), 5) for m in mzs), self.ppm, self.reduce, self.norm,
               weight, id(self.ds), getattr(self.ds, "_cache_gen", 0))
        if not compute:
            return None, label, key
        if len(mzs) == 1 and kind[0] == "active":
            vec = self.ds.ion_vector(mzs[0], tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)
        else:
            vec = self.ds.composite_vector(mzs, tol_ppm=self.ppm, reduce=self.reduce,
                                           norm=self.norm, weight=weight)
        return np.asarray(vec, dtype=float), label, key

    def _threshold_source_mask(self):
        vec, _ = self._threshold_signal_vector()
        if vec is None:
            if getattr(self, "thr_info", None) is not None:
                self.thr_info.setText("Select a feature (or pick a class / sum) to threshold.")
            return None
        absolute = self.thr_absolute.isChecked()
        mask = spatial.threshold_mask(self.ds, vec, float(self.thr_cut.value()),
                                      percentile=not absolute,
                                      fill_holes=self.thr_fill.isChecked(),
                                      min_pixels=int(self.thr_min_px.value()))
        n = int(mask.sum())
        self.thr_info.setText(f"{n:,} of {self.ds.n_pixels:,} px kept")
        return mask if n else None

    # ----- refine (rim / collar / band / invert) ----------------------------- #
    def _refine_kind(self):
        if getattr(self, "border_mode", None) is None:
            return None
        return dict(REFINE_MODES).get(self.border_mode.currentText())

    def _refine_active(self):
        """True when Refine changes the footprint (a ring of >0 px, or the inversion)."""
        kind = self._refine_kind()
        if kind is None:
            return False
        return kind == "invert" or float(self.border_spin.value()) > 0

    def _refine_tag(self):
        kind = self._refine_kind()
        if kind is None:
            return ""
        if kind == "invert":
            return "  ·  everything else"
        w = float(self.border_spin.value())
        return f"  ·  {self.border_mode.currentText().lower()} {w:.0f} px" if w > 0 else ""

    def _sync_refine_tools(self):
        kind = self._refine_kind()
        ring = kind in ("inner", "outer", "band")
        if getattr(self, "border_spin", None) is not None:
            self.border_spin.setEnabled(ring)
        if getattr(self, "border_um", None) is not None:
            px = getattr(self.ds, "pixel_size_um", None) if self.ds is not None else None
            w = float(self.border_spin.value())
            self.border_um.setText(f"= {w * float(px):g} µm" if (px and ring) else
                                   ("(no pixel size)" if ring else ""))

    def _refine_changed(self, *_):
        self._sync_refine_tools()
        self._roi_spectrum()

    def _roi_source_label(self):
        """A name for the current ROI build, pre-filled into 'Save as region' — the source
        ('ST 42:2;O3 ≥ p60', 'endo', 'ROI 3') plus the refine ('· outer collar 6 px', 'not …')."""
        if self._threshold_mode():
            _, label = self._threshold_signal_vector()
            cut = float(self.thr_cut.value())
            base = f"{label or 'signal'} {'> ' + format(cut, 'g') if self.thr_absolute.isChecked() else '≥ p' + format(cut, 'g')}"
        elif self._region_source_mode():
            base = self.src_region.currentText() or "region"
        else:
            base = f"ROI {len(self.regions) + 1}"
        kind = self._refine_kind()
        if kind == "invert":
            return f"not {base}"
        if kind and float(self.border_spin.value()) > 0:
            return f"{base} · {self.border_mode.currentText().lower()} {float(self.border_spin.value()):g} px"
        return base

    def _open_compartments_dialog(self):
        """Open (building once) the endo / peri / epi Compartments dialog; returns it."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return None
        dlg = getattr(self, "_compartments_dialog", None)
        if dlg is None:
            from .compartments import CompartmentsDialog
            dlg = self._compartments_dialog = CompartmentsDialog(self)
        dlg.refresh()
        self._show_dialog(dlg)
        return dlg

    def _save_roi_as_region(self):
        """'Save as region': the current mask (source + refine) becomes a named region, the
        name prompted with a pre-fill from the build. A refined *existing region* nests under
        its source. Returns the new region's index, or -1."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return -1
        mask = self._roi_mask()
        if mask is None or not mask.any():
            self.statusBar().showMessage("Nothing to save — draw an ROI, or pick a Threshold / "
                                         "Existing-region source that keeps some pixels.")
            return -1
        default = self._roi_source_label()
        name, ok = QtWidgets.QInputDialog.getText(self, "Save as region", "Region name:", text=default)
        if not ok or not name.strip():
            return -1
        kind = self._refine_kind()
        parent = (self.src_region.currentText()
                  if self._region_source_mode() and kind in ("inner", "outer", "band")
                  and float(self.border_spin.value()) > 0 else None)
        self.record_undo("ROI → region", domains=("regions",))
        ri = self._new_region(name.strip(), mask=mask, parent=parent)
        if self.prov is not None:
            self.prov.step("region_derive", source=self.roi_shape.currentText(),
                           op=kind or "filled", width_px=float(self.border_spin.value()),
                           cut=(float(self.thr_cut.value()) if self._threshold_mode() else None),
                           region=self.regions[ri]["name"])
        self._dismiss_roi()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"Region '{self.regions[ri]['name']}' saved: {int(mask.sum()):,} px.")
        return ri

    # ----- freehand brush -------------------------------------------------- #
    def _brush_mode(self):
        return (getattr(self, "roi_shape", None) is not None
                and self.roi_shape.currentText().startswith("Freehand"))

    def _brush_active(self):
        """Brush is live only while 'Draw ROI' is on and the Freehand shape is picked."""
        return (self._brush_mode() and getattr(self, "roi_chk", None) is not None
                and self.roi_chk.isChecked())

    def _brush_drag_event(self, ev, axis=None):
        # Optical-alignment drag takes precedence: left-drag moves the photo backdrop
        # (the brush/ROI tools are off while aligning). See gui/optical.py.
        if self._optical_drag_active() and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            p = self.iv.view.mapSceneToView(ev.scenePos())
            pl = self.iv.view.mapSceneToView(ev.lastScenePos())
            self._optical_drag_by(p.x() - pl.x(), p.y() - pl.y())
            if ev.isFinish():
                self._mark_dirty()
            return
        if not self._brush_active() or ev.button() != QtCore.Qt.LeftButton:
            return self._orig_vb_drag(ev, axis)             # normal pan/zoom
        ev.accept()
        if ev.isStart():
            self._brush_begin_stroke()                      # one undo step per drag
        p = self.iv.view.mapSceneToView(ev.scenePos())
        self._brush_stamp(p.x(), p.y())
        if ev.isFinish():
            self._roi_spectrum()

    def _brush_erasing(self):
        return (getattr(self, "brush_mode", None) is not None
                and self.brush_mode.currentText() == "Erase")

    def _brush_begin_stroke(self):
        """Snapshot the freehand mask before a stroke so ⌘Z can roll it back."""
        action = "erase" if self._brush_erasing() else "brush"
        self.record_undo(f"{action} stroke", domains=("roi",))

    def _brush_stamp(self, vx, vy):
        """Paint or erase a disk of the current brush radius (view-x = pixel col,
        view-y = row), depending on the Draw/Erase mode."""
        if self.ds is None:
            return
        erase = self._brush_erasing()
        if self._brush_mask is None or self._brush_mask.shape[0] != self.ds.n_pixels:
            if erase:
                return                                      # nothing painted yet to erase
            self._brush_mask = np.zeros(self.ds.n_pixels, dtype=bool)
        r = float(self.brush_size.value())
        rows, cols = self.ds._pixel_rows_cols()
        d2 = (cols.astype(float) + 0.5 - vx) ** 2 + (rows.astype(float) + 0.5 - vy) ** 2
        disk = d2 <= r * r
        if erase:
            self._brush_mask &= ~disk
        else:
            self._brush_mask |= disk
        self._render_brush_overlay()

    def _render_brush_overlay(self):
        item = getattr(self, "_brush_overlay", None)
        if item is None or self.ds is None or self._brush_mask is None or not self._brush_mask.any():
            if item is not None:
                item.hide()
            return
        h, w = self.ds.height, self.ds.width
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        on = self.ds.to_image(self._brush_mask.astype(float), fill=0.0) > 0.5
        r, g, b, _ = hex_to_rgba(self._roi_color)
        # fairly opaque so painted pixels read as solid colour — any gaps/holes in the
        # freehand stroke then stand out clearly against the filled area
        rgba[on, 0], rgba[on, 1], rgba[on, 2], rgba[on, 3] = r, g, b, 165
        item.setImage(rgba)
        item.show()

    def _clear_brush(self):
        self._brush_mask = None
        if getattr(self, "_brush_overlay", None) is not None:
            self._brush_overlay.hide()
        if getattr(self, "_band_overlay", None) is not None:
            self._band_overlay.hide()

    def _roi_shape_changed(self, *_):
        self._sync_brush_tools()                      # brush options only for the Freehand shape
        self._sync_source_tools()                     # threshold panel / region picker per source
        if getattr(self, "roi_chk", None) is not None and self.roi_chk.isChecked():
            self._toggle_roi(True)                    # re-show with the newly-chosen shape

    def _sync_brush_tools(self):
        """Brush mode/size only make sense for the Freehand shape — hide them otherwise."""
        if getattr(self, "brush_tools", None) is not None:
            self.brush_tools.setVisible(self._brush_mode())

    def _toggle_roi(self, on):
        for grp in ("roi_tools", "refine_tools", "roi_actions"):
            if getattr(self, grp, None) is not None:
                getattr(self, grp).setVisible(on)     # ROI sub-controls only while drawing
        self._sync_brush_tools()
        self._sync_source_tools()
        self._sync_refine_tools()
        for shp in self._roi_shapes():                # only the active shape is shown
            shp.hide()
        self._poly_draw_cancel()                      # leaving any in-progress click-draw
        if not on:
            self._remove_overlays("ROI")              # drop the stale ROI spectrum overlay
            self._clear_brush()
            self.statusBar().showMessage("ROI tool off — click any pixel on the image to "
                                         "overlay its spectrum ('Clear pixels' to reset).")
            return
        if self._source_mode():
            self._clear_brush()                       # no drawn footprint; the band shows the mask
            self._roi_spectrum()
            self.statusBar().showMessage(
                "Threshold: pick the signal and cut in 'Threshold ▾'; Refine grows a rim / "
                "collar or inverts; 'Save as region' keeps it."
                if self._threshold_mode() else
                "Existing region: pick the source region, then Refine (outer collar = a "
                "perineurium around it; everything else = the rest) and 'Save as region'.")
            return
        if self._brush_mode():
            self._clear_brush()                       # fresh canvas; drag on the image to paint
            self.statusBar().showMessage("Freehand brush: drag to paint (or Erase) pixels · "
                                         "[ / ] resize brush · ⌘Z undoes a stroke · "
                                         "'Add ROI' to save · 'Clear ROI' to reset.")
            return
        if getattr(self, "_brush_overlay", None) is not None:
            self._brush_overlay.hide()                # leaving brush for a geometric shape
        if self._poly_click_mode():
            self._poly_draw_start()                   # click out a fresh polygon
            return
        active = self._active_roi()
        active.show()
        if self.ds is not None:
            w, h = self.ds.width, self.ds.height
            if active is self.roi:
                active.setPos([w * 0.3, h * 0.3])
                active.setSize([max(3, w * 0.3), max(3, h * 0.3)])
            elif active is self.circle_roi:
                d = max(4, min(w, h) * 0.3)
                active.setPos([w * 0.35, h * 0.35])
                active.setSize([d, d])
            else:
                self.poly_roi.setPoints([[w * 0.3, h * 0.3], [w * 0.6, h * 0.3],
                                         [w * 0.6, h * 0.6], [w * 0.3, h * 0.6]])
            self._roi_spectrum()

    # ----- click-to-draw polygon ------------------------------------------- #
    def _poly_draw_start(self):
        self._poly_draw_pts = []
        self._poly_draw_update()
        self._poly_draw_markers.show()
        self._poly_draw_line.show()
        self.statusBar().showMessage("Click to drop polygon points · double-click or click the "
                                     "first point to close · 'Clear ROI' to cancel.")

    def _poly_draw_add_point(self, x, y):
        self._poly_draw_pts.append([float(x), float(y)])
        self._poly_draw_markers.show()                # re-show after an Esc-cancel + restart
        self._poly_draw_line.show()
        self._poly_draw_update()

    def _poly_draw_update(self):
        pts = self._poly_draw_pts
        if pts:
            xs = [p[0] for p in pts] + ([pts[0][0]] if len(pts) > 2 else [])
            ys = [p[1] for p in pts] + ([pts[0][1]] if len(pts) > 2 else [])
            self._poly_draw_markers.setData([p[0] for p in pts], [p[1] for p in pts])
            self._poly_draw_line.setData(xs, ys)        # dashed outline, closed once >2 pts
        else:
            self._poly_draw_markers.clear()
            self._poly_draw_line.clear()

    def _poly_draw_finish(self):
        if len(self._poly_draw_pts) < 3:
            self.statusBar().showMessage("Need at least 3 points to close the polygon.")
            return
        pts = self._poly_draw_pts
        self.poly_roi.setPoints(pts)                    # hand off to the real polygon ROI
        self.poly_roi.show()
        self._poly_draw_cancel()                        # hide the scratch dots/outline
        self._roi_spectrum()
        self.statusBar().showMessage(f"Polygon closed ({len(pts)} points) — drag vertices to "
                                     "adjust, or send it to a region.")

    def _poly_draw_cancel(self):
        self._poly_draw_pts = []
        if getattr(self, "_poly_draw_markers", None) is not None:
            self._poly_draw_markers.clear()
            self._poly_draw_markers.hide()
            self._poly_draw_line.clear()
            self._poly_draw_line.hide()

    def _roi_mask(self):
        """Per-pixel mask for the current ROI, after applying the border/ring setting."""
        return self._apply_border(self._roi_mask_filled())

    def _apply_border(self, mask):
        """Apply the Refine pill: reduce a filled mask to a rim / collar / band of the chosen
        width around its edge, or invert it to 'everything else'. 'Filled' (or a 0 px width)
        returns the mask untouched."""
        if mask is None or self.ds is None:
            return mask
        kind = self._refine_kind()
        if kind is None:
            return mask
        if kind == "invert":
            inv = spatial.invert_mask(self.ds, mask)
            return inv if inv.any() else mask
        w = float(self.border_spin.value()) if getattr(self, "border_spin", None) is not None else 0.0
        if w <= 0:
            return mask
        ring = spatial.ring_mask(self.ds, mask, w, mode=kind)
        return ring if ring.any() else mask           # never silently empty the ROI

    def _roi_mask_filled(self):
        """Boolean per-pixel mask for the current ROI shape (filled, pre-border). A
        click-to-draw polygon still in progress (≥3 points placed) is honoured even
        before it's closed, so 'Add ROI' captures the polygon you drew instead of
        falling back to the rectangle."""
        if self.ds is None:
            return None
        if self._threshold_mode():
            return self._threshold_source_mask()
        if self._region_source_mode():
            return self._region_source_mask()
        if self._brush_mode():
            bm = getattr(self, "_brush_mask", None)
            return bm if bm is not None and bm.any() else None
        pts = getattr(self, "_poly_draw_pts", None)
        if pts and len(pts) >= 3:
            return self._poly_points_mask(pts)
        if self.poly_roi.isVisible():
            return self._poly_roi_mask()
        if self.circle_roi.isVisible():
            return self._circle_roi_mask()
        pos, size = self.roi.pos(), self.roi.size()
        # ROI coords are the viewbox coords. With pyqtgraph in row-major mode the ImageItem
        # maps array index (row, col) → view (x=col, y=row), so view-x = pixel COL and
        # view-y = pixel ROW. Test cols against the x-extent and rows against the y-extent,
        # or the mask comes out transposed.
        x0, y0 = int(pos.x()), int(pos.y())
        x1, y1 = int(pos.x() + size.x()), int(pos.y() + size.y())
        mask = np.zeros(self.ds.n_pixels, dtype=bool)
        rows, cols = self.ds._pixel_rows_cols()
        sel = (cols >= x0) & (cols < x1) & (rows >= y0) & (rows < y1)
        mask[sel] = True
        return mask

    def _poly_points_mask(self, verts):
        """Per-pixel mask from a list of (x, y) polygon vertices (viewbox coords), by
        point-in-polygon over pixel centres. view-x = pixel COL, view-y = pixel ROW, so
        the centre for pixel (row, col) is (x=col, y=row) — using (row, col) transposes it."""
        from matplotlib.path import Path
        verts = [(float(x), float(y)) for x, y in verts]
        if len(verts) < 3:
            return None
        rows, cols = self.ds._pixel_rows_cols()
        centres = np.column_stack([cols.astype(float) + 0.5, rows.astype(float) + 0.5])
        inside = Path(verts).contains_points(centres)
        mask = np.zeros(self.ds.n_pixels, dtype=bool)
        mask[inside] = True
        return mask

    def _poly_roi_mask(self):
        """Per-pixel mask for the closed polygon ROI, using its local vertices + origin."""
        st = self.poly_roi.getState()
        origin = st.get("pos")
        ox, oy = float(origin.x()), float(origin.y())
        return self._poly_points_mask([(p.x() + ox, p.y() + oy) for p in st.get("points", [])])

    def _circle_roi_mask(self):
        """Per-pixel mask for the circular ROI (interior + edge). view-x = pixel COL,
        view-y = pixel ROW, so the circle centre/radii are tested against (col, row)."""
        pos, size = self.circle_roi.pos(), self.circle_roi.size()
        wv, hv = float(size.x()), float(size.y())
        if wv <= 0 or hv <= 0:
            return None
        cx, cy = float(pos.x()) + wv / 2.0, float(pos.y()) + hv / 2.0
        rx, ry = wv / 2.0, hv / 2.0
        rows, cols = self.ds._pixel_rows_cols()
        nx = (cols.astype(float) + 0.5 - cx) / rx          # view-x = col
        ny = (rows.astype(float) + 0.5 - cy) / ry          # view-y = row
        mask = np.zeros(self.ds.n_pixels, dtype=bool)
        mask[nx * nx + ny * ny <= 1.0] = True
        return mask

    def _reorient_roi_shapes(self):
        """Rotate the active geometric ROI tools (rectangle / circle / polygon) along with
        the image on a 90°-CW orientation step, so they keep covering the *same tissue
        pixels* instead of staying put in view space while the picture turns under them.
        (The freehand brush is a per-pixel mask, so it already re-rasterizes rotated through
        ``to_image``.) View coords are row-major (view-x = col, view-y = row); a single
        90°-CW step maps a view point (x, y) → (W − y, x), where W is the *new* image width
        (= the old height). Verified to leave each ROI's selected-pixel set unchanged."""
        if self.ds is None:
            return
        W = float(self.ds.width)                          # new width = old height (rows)

        def T(x, y):
            return (W - y, x)

        if self.roi.isVisible():                          # rectangle → new axis-aligned box
            pos, size = self.roi.pos(), self.roi.size()
            nx, ny = T(pos.x(), pos.y() + size.y())       # bottom-left corner → new top-left
            self.roi.setPos([nx, ny]); self.roi.setSize([size.y(), size.x()])
        if self.circle_roi.isVisible():                   # circle → move centre, keep radius
            pos, size = self.circle_roi.pos(), self.circle_roi.size()
            r = size.x() / 2.0
            cx, cy = T(pos.x() + r, pos.y() + r)
            self.circle_roi.setPos([cx - r, cy - r])
        if self.poly_roi.isVisible():                     # polygon → transform every vertex
            st = self.poly_roi.getState(); o = st["pos"]
            pts = [T(p.x() + o.x(), p.y() + o.y()) for p in st["points"]]
            self.poly_roi.setPos([0.0, 0.0]); self.poly_roi.setPoints(pts)
        drawing = getattr(self, "_poly_draw_pts", None)   # click-to-draw still in progress
        if drawing:
            self._poly_draw_pts = [list(T(x, y)) for x, y in drawing]
            self._poly_draw_update()

    def _roi_spectrum(self):
        brush = (self._brush_mode() and getattr(self, "_brush_mask", None) is not None
                 and self._brush_mask.any())
        source = (self._source_mode() and getattr(self, "roi_chk", None) is not None
                  and self.roi_chk.isChecked())
        if self.ds is None or not (brush or source or any(s.isVisible() for s in self._roi_shapes())):
            self._render_band_overlay(None)        # nothing selected → drop the band fill
            return
        mask = self._roi_mask()
        if mask is None or mask.sum() == 0:
            self._render_band_overlay(None)
            self._remove_overlays("ROI")
            if source:
                self.statusBar().showMessage("ROI: no pixels — loosen the cut, or pick another "
                                             "signal / region.")
            return
        self._render_band_overlay(mask)            # show which pixels the refine keeps (cheap, instant)
        tag = self._refine_tag()
        npx = int(mask.sum())
        # Prefer the fast cube path (in-RAM, instant). Without the cube the masked mean
        # streams every pixel from disk — run that OFF the GUI thread so toggling/resizing
        # an ROI never freezes the window; a token drops stale results from rapid edits.
        fast = self.ds.cube_mean_spectrum(mask)
        if fast is not None:
            self._overlay_spectrum(fast[0], fast[1], self._roi_color, "ROI")
            self.statusBar().showMessage(f"ROI: {npx} px{tag}  ·  use 'Clear ROI' to remove it.")
            return
        # No cube yet → build it in the background so the NEXT ROI is instant (this one
        # still streams once, off-thread). Cheap to call repeatedly: it self-guards.
        if hasattr(self, "_autobuild_cache"):
            self._autobuild_cache()
        ds = self.ds
        token = getattr(self, "_roi_spec_token", 0) + 1
        self._roi_spec_token = token
        self.statusBar().showMessage(f"ROI: {npx} px{tag}  ·  computing spectrum…")

        def compute():
            return ds.mean_spectrum(mask=mask)

        def on_done(res):
            if token != getattr(self, "_roi_spec_token", 0):
                return                             # superseded by a newer ROI edit
            self._overlay_spectrum(res[0], res[1], self._roi_color, "ROI")
            self.statusBar().showMessage(f"ROI: {npx} px{tag}  ·  use 'Clear ROI' to remove it.")

        self._run(compute, on_done=on_done, busy="ROI spectrum…")

    def _render_band_overlay(self, mask):
        """Fill the pixels the ROI actually keeps so a >0 px Inner/Outer/Band border is
        visible on the image. With border 0 (whole shape) the drawn outline already shows
        the selection, so the fill stays hidden. In freehand mode the band replaces the
        full painted footprint while it's shown, and restores it once the border is 0."""
        item = getattr(self, "_band_overlay", None)
        if item is None:
            return
        show = self._refine_active() or self._source_mode()
        if self.ds is None or mask is None or not np.any(mask) or not show:
            item.hide()
            if self._brush_mode():
                self._render_brush_overlay()       # restore the full painted footprint
            return
        h, w = self.ds.height, self.ds.width
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        on = self.ds.to_image(mask.astype(float), fill=0.0) > 0.5
        r, g, b, _ = hex_to_rgba(self._roi_color)
        rgba[on, 0], rgba[on, 1], rgba[on, 2], rgba[on, 3] = r, g, b, 150
        item.setImage(rgba)
        item.show()
        if getattr(self, "_brush_overlay", None) is not None:
            self._brush_overlay.hide()             # the band replaces the full painted footprint

    def _remove_overlays(self, name, prefix=False):
        """Remove overlay traces tagged via item._overlay_name. When prefix=True,
        remove every overlay whose name *starts with* `name` (e.g. all 'ROI*')."""
        n = 0
        for item in list(self.spectrum.listDataItems()):
            tag = getattr(item, "_overlay_name", None)
            if tag is None:
                continue
            if (str(tag).startswith(name) if prefix else tag == name):
                self.spectrum.removeItem(item)
                n += 1
        self._refresh_spectra_panel()
        return n

    def _overlay_spectrum(self, axis, spec, color, name, fill=False):
        # remove any prior overlay with this name, then plot. `fill` shades the area under
        # the trace in its color, so region/ROI spectra read as colored peaks that match
        # the region's footprint on the image.
        self._remove_overlays(name)
        kw = {"pen": pg.mkPen(color, width=1.5 if fill else 1), "name": name}
        if fill:
            c = QtGui.QColor(color)
            c.setAlpha(70)
            kw["fillLevel"] = 0.0
            kw["brush"] = c
        curve = self.spectrum.plot(axis, spec, **kw)
        curve._overlay_name = name
        curve._overlay_color = color
        self._refresh_spectra_panel()

    def _clear_roi(self):
        """Reset all ROI state: uncheck 'Draw ROI', hide the ROI shape, and remove
        every ROI spectrum overlay."""
        bm = getattr(self, "_brush_mask", None)
        if bm is not None and bm.any():               # painted pixels are about to be lost
            if not confirm(self, "Clear ROI", "Clear the painted ROI? The hand-painted pixels will be removed (⌘Z can still undo this).", ok_text="Clear ROI"):
                return
            self.record_undo("clear ROI", domains=("roi",))
        # uncheck without re-triggering side effects beyond the toggle handler
        if getattr(self, "roi_chk", None) is not None:
            self.roi_chk.setChecked(False)
        for shp in self._roi_shapes():
            shp.hide()
        self._poly_draw_cancel()
        self._clear_brush()
        n = self._remove_overlays("ROI", prefix=True)
        self.statusBar().showMessage(
            f"Cleared ROI ({n} overlay{'s' if n != 1 else ''} removed).")

    def _dismiss_roi(self):
        """Put the drawing ROI away (shapes + in-progress polygon + its spectrum) without
        touching named regions or the A/B masks — used after promoting an ROI to a named
        region, so the yellow ROI is replaced by the region in its own color."""
        for shp in self._roi_shapes():
            shp.hide()
        self._poly_draw_cancel()
        self._clear_brush()
        chk = getattr(self, "roi_chk", None)
        if chk is not None and chk.isChecked():
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._remove_overlays("ROI")

    def _image_clicked(self, ev):
        if self.ds is None:
            return
        vb = self.iv.getView()
        # Double-clicking a legend label renames that ion (WYSIWYG edit) — checked before the
        # ROI/pixel handlers and only when not mid-draw, so it never interferes with drawing.
        annot = getattr(self, "ion_annot", None)
        drawing_now = getattr(self, "roi_chk", None) is not None and self.roi_chk.isChecked()
        if ev.double() and annot is not None and not drawing_now and not self._brush_active():
            mz = annot.label_hit(ev.scenePos())
            if mz is not None:
                ev.accept()
                self._rename_feature_label(mz)
                return
        # freehand brush: a single click drops one dab (drags paint via mouseDragEvent)
        if self._brush_active():
            if ev.button() != QtCore.Qt.LeftButton:
                return
            ev.accept()
            self._brush_begin_stroke()                # one undo step per dab
            p = vb.mapSceneToView(ev.scenePos())
            self._brush_stamp(p.x(), p.y())
            self._roi_spectrum()
            return
        # click-to-draw polygon takes priority over the pixel-spectrum click
        if getattr(self, "roi_chk", None) is not None and self.roi_chk.isChecked() \
                and self._poly_click_mode():
            if ev.button() != QtCore.Qt.LeftButton:
                return
            ev.accept()
            p = vb.mapSceneToView(ev.scenePos())
            if ev.double():
                self._poly_draw_finish()
                return
            pts = self._poly_draw_pts
            if len(pts) >= 3 and abs(p.x() - pts[0][0]) < 1.2 and abs(p.y() - pts[0][1]) < 1.2:
                self._poly_draw_finish()               # clicked back on the first point
                return
            self._poly_draw_add_point(p.x(), p.y())
            return
        # With the ROI tool OFF, a click on the image inspects that pixel's spectrum —
        # this is the default now (no checkbox). While drawing, a bare click belongs to
        # the ROI, so instead clear the active feature so nothing keeps pulsing — deselect
        # from outside the list. (Rectangle ROIs are dragged via handles, not bare clicks.)
        drawing = getattr(self, "roi_chk", None) is not None and self.roi_chk.isChecked()
        if drawing or self._pix_lookup is None:
            # a Threshold source thresholds the *active* feature — a stray click must not
            # blank it (and the mask with it)
            if self.active_mz is not None and not self._source_mode():
                self._deselect_feature()
            return
        p = vb.mapSceneToView(ev.scenePos())
        r, c = int(p.y()), int(p.x())
        lut = self._pix_lookup
        if not (0 <= r < lut.shape[0] and 0 <= c < lut.shape[1]):
            return
        i = int(lut[r, c])
        if i < 0:                                     # clicked off-tissue (no acquired pixel)
            return
        mz, inten = self.ds.get_spectrum(i)
        n = sum(1 for it in self.spectrum.listDataItems()
                if str(getattr(it, "_overlay_name", "")).startswith("pixel"))
        color = PIXEL_COLORS[n % len(PIXEL_COLORS)]
        self._overlay_spectrum(mz, inten, color, f"pixel ({c},{r})")
        self.statusBar().showMessage(f"Pixel ({c},{r}) spectrum overlaid. "
                                     "Use 'Clear pixels' on the spectrum toolbar to reset.")

    def _clear_pixel_spectra(self):
        n = 0
        for item in list(self.spectrum.listDataItems()):
            if str(getattr(item, "_overlay_name", "")).startswith("pixel"):
                self.spectrum.removeItem(item)
                n += 1
        self._refresh_spectra_panel()
        self.statusBar().showMessage(f"Cleared {n} pixel spectr{'um' if n == 1 else 'a'}."
                                     if n else "No pixel spectra to clear.")

    def _clear_spectrum_overlays(self):
        """Reset the spectrum to its base trace: remove every overlaid selection in one go —
        the per-pixel click spectra, the drawn-ROI mean, and region means. The base mean/skyline
        trace stays, and the underlying ROI and regions are left intact (only their spectrum
        overlays are removed — re-selecting a region re-draws its trace)."""
        n = (self._remove_overlays("pixel", prefix=True)
             + self._remove_overlays("ROI", prefix=True)
             + self._remove_overlays("Region", prefix=True))
        self.statusBar().showMessage(
            f"Cleared {n} spectrum overlay{'' if n == 1 else 's'} — reset to the base trace."
            if n else "No spectrum overlays to clear.")

    # ----- peak picking ---------------------------------------------------- #
    def _promote_roi_to_region(self):
        """Promote a non-empty drawn ROI to a new named region (recording the 'ROI →
        region' undo step) and return its index, or -1 when no ROI is drawn. Dismisses
        the drawing ROI so it's replaced by the region in its own colour."""
        roi = self._roi_mask()
        if roi is None or not roi.any():
            return -1
        self.record_undo("ROI → region", domains=("regions", "features"))
        ri = self._new_region(f"ROI {len(self.regions) + 1}", mask=roi)
        self._dismiss_roi()
        return ri

    def _region_for_feature_build(self, ri=None, prefer_roi=False):
        """Resolve which region a feature list should be built from, so 'ROI → feature
        list' is a single click no matter how far along you are:

        * an explicit integer index (a menu/programmatic caller) is used as-is;
        * with ``prefer_roi`` (the ion-image toolbar button, fired right after drawing),
          a freshly drawn ROI wins — that's the unambiguous intent there;
        * otherwise the region selected in the Regions list;
        * otherwise a drawn ROI is promoted to a new named region on the spot, giving the
          resulting list a persistent home.

        Returns the region index, or -1 when there's neither a selection nor a drawn ROI
        (the caller shows the hint). Promotion records the 'ROI → region' undo step; the
        caller records its own feature step (suppressed while nested)."""
        if isinstance(ri, int) and not isinstance(ri, bool):
            return ri
        if prefer_roi:
            ri = self._promote_roi_to_region()
            if ri >= 0:
                return ri
        ri = self._selected_region_index()
        if ri >= 0:
            return ri
        return self._promote_roi_to_region()

    def do_pick_peaks(self):
        if self.ds is None:
            return
        region, mask = self._pick_region_mask()
        if region is False:                               # stale/empty region — message shown
            return
        self._pending_pick_region = region
        self._pending_pick_save = (getattr(self, "pick_save_list_chk", None) is not None
                                   and self.pick_save_list_chk.isChecked())
        self._run(pick_and_build, self.ds, float(self.snr_spin.value()),
                  float(self.minrel_spin.value()), self.ppm, self.reduce,
                  projection=self._projection(), mask=mask,
                  prominence=float(self.prominence_spin.value()),
                  max_peaks=int(self.maxpeaks_spin.value()),
                  on_done=self._on_peaks, want_progress=True,
                  busy=(f"Picking peaks in {region}…" if region else "Picking peaks & building features…"))

    def _on_peaks(self, peaks):
        region = getattr(self, "_pending_pick_region", None)
        self._pending_pick_region = None
        # only the Find-peaks dialog arms this; session restore / region-build paths leave it
        # False so reloading a sample never spawns duplicate ★ lists
        save_list = getattr(self, "_pending_pick_save", False)
        self._pending_pick_save = False
        scope = region or "All slide"
        for p in peaks:                                   # tag for the per-sample scope
            p["region"] = region
        self._feature_scopes[scope] = peaks               # store / replace this scope's features
        self.peaks = peaks
        self._active_feature_scope = scope
        self._flist_name = "All slide" if region is None else f"{region} (sample)"
        n_detected = int(getattr(peaks, "n_detected", len(peaks)))
        if self.prov is not None:
            self.prov.step("peak_picking", snr=float(self.snr_spin.value()),
                           min_rel=float(self.minrel_spin.value()),
                           prominence=float(self.prominence_spin.value()), norm=self.norm,
                           tol_ppm=self.ppm, n_peaks=len(peaks), scope=scope,
                           max_peaks=int(self.maxpeaks_spin.value()), n_detected=n_detected)
        self._populate_peak_table()
        self._populate_peak_combos()
        self._build_class_map()
        self._mark_peaks()
        self._refresh_feature_set_combo()
        if peaks:
            self.set_active_mz(peaks[0]["mz"])
        self._mark_dirty()                                # picking isn't undoable → hook here
        # say so when the cap bound: a capped count is the cap, not a measurement of this
        # region, and two regions that both stop at it have not been shown to agree
        capped = (f" of {n_detected} detected — raise Max peaks to keep the rest"
                  if n_detected > len(peaks) else "")
        gb = float(getattr(peaks, "cache_gb", 0.0) or 0.0)
        heavy = (f" Feature cache ≈ {gb:.1f} GB; set Max peaks or raise Min rel. intensity "
                 "if segmentation or PCA runs out of memory." if gb > 2.0 else "")
        msg = (f"{len(peaks)} peaks picked{capped}{f' in {region}' if region else ''}. "
               "Features ready." + heavy
               + (" Switch samples with the Features ▸ Feature set selector." if region else ""))
        if save_list and peaks:
            saved = self._save_feature_list_to_library(
                f"{region} peaks" if region else "All slide peaks", peaks, prompt=True)
            if saved:                                     # None → user cancelled the name prompt
                msg += f" Saved as feature list '{saved}' (★)."
        self.statusBar().showMessage(msg)
        if hasattr(self, "_refresh_action_states"):
            self._refresh_action_states()

    def _pick_region_mask(self):
        """(scope_name, mask) for the Find-peaks 'Region(s)' selector. Nothing ticked →
        (None, None) = whole dataset. One ticked → that region. Several ticked → the union
        of their pixels, scoped to a combined 'A + B' name so the picker builds one list
        over them together. Returns (False, False) on a stale/empty selection so the caller
        can bail. Shared by the plain picker and the spatial finder."""
        lw = getattr(self, "pick_region_list", None)
        if lw is None:
            return None, None
        names = lw.checked_in_order()                     # row order → the 'A + B' scope name
        if not names:                                     # nothing ticked → whole slide
            return None, None
        ris = [j for n in names
               for j, rg in enumerate(self.regions) if rg["name"] == n]
        if not ris:
            self.statusBar().showMessage("The ticked region(s) no longer exist — pick another.")
            return False, False
        mask = None
        for r in ris:                                     # union of the ticked regions' pixels
            m = self._region_pixel_mask(self.regions[r])
            if m is not None:
                mask = m if mask is None else (mask | m)
        if mask is None or not mask.any():
            self.statusBar().showMessage("The ticked region(s) have no pixels.")
            return False, False
        scope = " + ".join(self.regions[r]["name"] for r in ris)
        return scope, mask

    def do_find_spatial_features(self):
        """Spatially-aware feature finder: mean candidates → frequency
        (reproducibility) gate → spatial-denoise gate → per-feature auto-width. Routes
        the survivors through the normal peak-scope plumbing."""
        if self.ds is None:
            return
        region, mask = self._pick_region_mask()
        if region is False:                               # stale/empty region — message shown
            return
        self._pending_pick_region = region
        self._pending_pick_save = (getattr(self, "pick_save_list_chk", None) is not None
                                   and self.pick_save_list_chk.isChecked())
        proj = {"mean": "mean", "skyline (max)": "max", "both": "both"}.get(
            self.spatial_proj_combo.currentText(), "mean")
        fdr_max = {"off": None, "≤ 5%": 0.05, "≤ 10%": 0.10, "≤ 20%": 0.20}.get(
            self.spatial_fdr_combo.currentText())
        # snapshot every setting that produced this feature set for the audit trail (the
        # finder runs off-thread; _on_spatial_features records once it lands)
        self._pending_spatial_audit = {
            "snr": float(self.snr_spin.value()),
            "min_rel_intensity": float(self.minrel_spin.value()),
            "min_frequency": float(self.spatial_freq_spin.value()) / 100.0,
            "min_morans": float(self.spatial_morans_spin.value()),
            "tol_ppm": self.ppm, "norm": self.norm, "reduce": self.reduce,
            "max_candidates": int(self.spatial_maxcand_spin.value()),
            "projection": proj, "prominence": float(self.prominence_spin.value()),
            "collapse_isotopes": self.spatial_deiso_chk.isChecked(),
            "fdr_max": fdr_max, "mode": self.mode_combo.currentText()}
        self._pending_spatial_regions = (region.split(" + ") if region else [])
        self._run(find_spatial, self.ds, float(self.snr_spin.value()),
                  float(self.minrel_spin.value()), self.ppm, self.reduce,
                  float(self.spatial_freq_spin.value()) / 100.0,
                  float(self.spatial_morans_spin.value()), norm=self.norm, mask=mask,
                  max_candidates=int(self.spatial_maxcand_spin.value()),
                  prominence=float(self.prominence_spin.value()), projection=proj,
                  collapse_isotopes=self.spatial_deiso_chk.isChecked(),
                  fdr_max=fdr_max, mode=self.mode_combo.currentText(), fdr_ppm=self.id_ppm,
                  on_done=self._on_spatial_features, want_progress=True,
                  busy=(f"Finding spatial features in {region}…" if region
                        else "Finding spatial features (mean → frequency → spatial denoise)…"))

    def _on_spatial_features(self, result):
        region = getattr(self, "_pending_pick_region", None)
        self._on_peaks(result.peaks)                      # store scope + populate working set
        src = (result.params or {}).get("projection", "mean")
        src = {"max": "skyline", "both": "mean+skyline"}.get(src, "mean")
        cap = (f" (capped from {result.n_detected} detected)"
               if result.candidates_capped else "")
        funnel = (f"{result.n_candidates} {src} candidates{cap} → {result.n_after_frequency} "
                  f"reproducible → {result.n_after_morans} spatially structured")
        if result.n_after_collapse >= 0 and result.n_after_collapse != result.n_after_morans:
            funnel += f" → {result.n_after_collapse} after isotope collapse"
        fdr = (result.params or {}).get("fdr")
        if fdr:
            lv = fdr.get("levels", {})
            funnel += (f" → {fdr.get('n_kept', len(result.peaks))} at FDR≤{fdr.get('q_max')} "
                       f"({fdr.get('n_unannotated_kept', 0)} unknown kept; "
                       f"IDs @5/10/20%: {lv.get(0.05, 0)}/{lv.get(0.10, 0)}/{lv.get(0.20, 0)})")
        width = ("" if result.suggested_ppm != result.suggested_ppm   # NaN guard
                 else f" · auto-width ≈ {result.suggested_ppm:.0f} ppm")
        if not result.peaks:
            self.statusBar().showMessage(
                f"No spatial features survived ({funnel}). Lower Min frequency / Min Moran's I.")
            return
        self.statusBar().showMessage(
            f"Spatial features{f' in {region}' if region else ''}: {funnel}{width}. "
            "Identify lipids to annotate them.")
        recs = [{"mz": round(float(p["mz"]), 4), "lipid": self.annotate(float(p["mz"])) or ""}
                for p in result.peaks]
        # record how this feature set was generated: every finder setting + the funnel
        # counts (which gate dropped what) + the ROIs it was scoped to → audit trail + Report
        audit = dict(getattr(self, "_pending_spatial_audit", {}) or {})
        audit.update({"n_candidates": int(result.n_candidates),
                      "n_detected": int(result.n_detected),
                      "candidates_capped": bool(result.candidates_capped),
                      "n_after_frequency": int(result.n_after_frequency),
                      "n_after_morans": int(result.n_after_morans),
                      "n_features": len(result.peaks)})
        if result.n_after_collapse is not None and result.n_after_collapse >= 0:
            audit["n_after_collapse"] = int(result.n_after_collapse)
        extra = self.record_step(
            "spatial_feature_finding", label="Spatial feature finding",
            params=audit, regions=getattr(self, "_pending_spatial_regions", []))
        self._log_analysis_to_report(
            "Spatial features", recs,
            regions=(f"within “{region}”" if region else "Whole slide — every tissue pixel"),
            title="Spatial features",
            caption=f"{len(recs)} spatially-structured features · {funnel}",
            source_extra=extra)

    def features_from_region(self, ri=None, prefer_roi=False):
        """One click: pick peaks restricted to a region's pixels *and* identify them,
        producing an annotated feature list scoped to that region. Pass nothing and it
        works on the selected region, or promotes the ROI you just drew — so 'ROI →
        feature list' is a single button. ``prefer_roi`` (the ion-image toolbar button)
        favours the just-drawn ROI over any list selection. The result is stored as the
        region's per-sample scope (switch back with the Features ▸ Feature set selector)."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        ri = self._region_for_feature_build(ri, prefer_roi=prefer_roi)
        if ri < 0:
            self._focus_ion_tab_for_roi()
            self.statusBar().showMessage("Select a region, or draw an ROI on the image, "
                                         "then build its feature list.")
            return
        rg = self.regions[ri]
        mask = self._region_pixel_mask(rg)
        if mask is None or not mask.any():
            self.statusBar().showMessage(f"Region '{rg['name']}' has no pixels.")
            return
        self.record_undo("region features")          # suppressed when nested in an ROI→region action
        self._pending_pick_region = rg["name"]
        self._run(pick_and_build, self.ds, float(self.snr_spin.value()),
                  float(self.minrel_spin.value()), self.ppm, self.reduce,
                  projection=self._projection(), mask=mask,
                  prominence=float(self.prominence_spin.value()),
                  max_peaks=int(self.maxpeaks_spin.value()),
                  on_done=self._on_region_features, want_progress=True,
                  busy=f"Building feature list for {rg['name']}…")

    def features_from_regions(self, ris):
        """Build ONE feature list over several regions taken *together* — peaks picked
        across the union of their pixels, identified, and stored as a single per-sample
        scope named for the regions (e.g. 'Cortex + Medulla'). This is 'work on regions
        A and B as one list', distinct from the Combine *spectrum* overlay. With fewer
        than two regions selected it falls back to the single-region / drawn-ROI path so
        the one button covers both cases."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        ris = [r for r in (ris or []) if 0 <= r < len(self.regions)]
        if len(ris) < 2:
            self.features_from_region()      # single selected region, or promote a drawn ROI
            return
        mask = None
        for r in ris:                        # union of the selected regions' pixel masks
            m = self._region_pixel_mask(self.regions[r])
            if m is not None:
                mask = m if mask is None else (mask | m)
        if mask is None or not mask.any():
            self.statusBar().showMessage("The selected regions have no pixels.")
            return
        scope = " + ".join(self.regions[r]["name"] for r in ris)
        self.record_undo("region features")
        self._pending_pick_region = scope
        self._run(pick_and_build, self.ds, float(self.snr_spin.value()),
                  float(self.minrel_spin.value()), self.ppm, self.reduce,
                  projection=self._projection(), mask=mask,
                  prominence=float(self.prominence_spin.value()),
                  max_peaks=int(self.maxpeaks_spin.value()),
                  on_done=self._on_region_features, want_progress=True,
                  busy=f"Building feature list for {scope}…")

    def _on_region_features(self, peaks):
        self._on_peaks(peaks)            # store the per-sample scope + populate working set
        if peaks:
            self.do_feature_list()       # annotate → full feature list for the region

    def _populate_peak_table(self):
        """Quick (un-annotated) view of the working features in the dock table:
        m/z, lipid, intensity, S/N. The full annotated view replaces this once
        'Identify lipids' runs (see features._render_feature_table). Keeping lipid
        in column 1 means the filter / 'identified only' toggle work in both views.

        A working set can be committed from a stats step (e.g. 'Keep features by AUC'),
        whose peaks carry only m/z + their stats columns — no intensity/S-N. Those cells
        show '—' rather than KeyError-ing out of the whole refresh (which silently left
        the list + selector stale when the exception was swallowed by the flow runner)."""
        def cell(p, key, fmt):
            v = p.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) and v == v else "—"
        rows = [(f"{p['mz']:.4f}", self.annotate(p["mz"]) or "—",
                 cell(p, "intensity", ".0f"), cell(p, "snr", ".1f")) for p in self.peaks]
        self._feat_repopulating = True
        try:
            fill_table(self.feat_table, ["m/z", "lipid", "intensity", "S/N"], rows)
            self._decorate_feature_swatches()
        finally:
            self._feat_repopulating = False
        self._apply_feature_filter()
        self._sync_feature_consumers()
        self._update_selected_feature_ui()

    # vertical fraction of the visible peak height reserved for the rail gutter below 0
    _RAIL_FRAC = 0.06

    def _mark_peaks(self):
        prior = getattr(self, "_peak_scatter", None)     # drop the old markers first
        if prior is not None:
            self.spectrum.removeItem(prior)
            self._peak_scatter = None
        self._peak_rail_mz = None
        if not self.peaks:
            return
        pmz = np.array([p["mz"] for p in self.peaks], dtype=float)
        self._peak_rail_mz = pmz
        # hang the markers in a thin gutter *below* the m/z axis (below the y=0 baseline)
        # as up-pointing arrows that point back up at their peaks — a "peak
        # rail" that never overlaps the traces. _reposition_peak_rail keeps the gutter
        # depth tracking the visible y-scale as you zoom. Still clickable.
        vb = self.spectrum.getViewBox()
        y1 = vb.viewRange()[1][1]
        g = (y1 if y1 > 0 else 1.0) * self._RAIL_FRAC
        sc = pg.ScatterPlotItem(x=pmz, y=np.full(len(pmz), -g), symbol="t1", size=12,
                                brush=pg.mkBrush(DANGER),
                                pen=pg.mkPen("#ffffff", width=0.5), hoverable=True,
                                hoverBrush=pg.mkBrush("#ffcc00"), hoverSize=16, data=pmz)
        sc.setZValue(40)
        sc.sigClicked.connect(self._marker_clicked)      # click an arrow to select that peak
        self.spectrum.addItem(sc)
        self._peak_scatter = sc
        self._reposition_peak_rail()

    def _reposition_peak_rail(self, *args):
        """Keep the found-peak rail parked a fixed fraction below the baseline as the
        auto-Y rescales (X-zoom changes the visible peak height). Driven by the view's
        sigYRangeChanged; guarded against the re-entrant signal that moving the markers
        itself emits. The rail's negative y also pulls the auto-range bottom down, which
        is what carves the gutter under the axis in the first place."""
        sc = getattr(self, "_peak_scatter", None)
        pmz = getattr(self, "_peak_rail_mz", None)
        if sc is None or pmz is None or len(pmz) == 0 or getattr(self, "_rail_busy", False):
            return
        y1 = self.spectrum.getViewBox().viewRange()[1][1]
        if y1 <= 0:
            return
        g = y1 * self._RAIL_FRAC
        self._rail_busy = True
        try:
            sc.setData(x=pmz, y=np.full(len(pmz), -g), data=pmz)
        finally:
            self._rail_busy = False

    def _marker_clicked(self, _scatter, points):
        if points:
            self.set_active_mz(float(points[0].data()))

    def _line_moved(self):
        self.set_active_mz(self._snap_mz(float(self.mz_line.value())))

    # ----- ion image ------------------------------------------------------- #
    def set_active_mz(self, mz):
        self.active_mz = float(mz)
        self.mz_line.setValue(self.active_mz)
        self._highlight_active_peak()             # halo on the apex
        self._ensure_mz_visible(self.active_mz)   # pan the spectrum so the peak is on-screen
        if hasattr(self, "_sync_coloc_start"):    # mirror the target into the co-localization picker
            self._sync_coloc_start()
        self._update_selected_feature_ui()
        self.refresh_ion_image()
        if hasattr(self, "_sync_cmp_active"):     # let the Region-comparison view follow the active ion
            self._sync_cmp_active(self.active_mz)
        if hasattr(self, "_sync_stats_active"):   # …and the ROI-stats ROC curve
            self._sync_stats_active(self.active_mz)
        if (self._threshold_mode() and getattr(self, "roi_chk", None) is not None
                and self.roi_chk.isChecked()):    # a live threshold follows the active feature
            self._thr_cache = None
            self._roi_spectrum()
        self._mark_dirty()                        # the active selection is restorable state

    def _on_escape(self):
        """Esc cascade: cancel an in-progress ROI draw, else clear a shown ROI, else
        deselect the active feature (so nothing keeps pulsing)."""
        if getattr(self, "_poly_draw_pts", None):
            self._poly_draw_cancel()
            self.statusBar().showMessage("ROI drawing cancelled.")
        elif self.roi.isVisible() or self.poly_roi.isVisible():
            self._clear_roi()
        elif self.active_mz is not None:
            self._deselect_feature()

    def _deselect_feature(self):
        """Drop the active feature: stop the pulse, clear the halo + table selection, and
        redraw the overlay without it. Gives a way to have nothing selected/pulsing."""
        self.active_mz = None
        self._stop_pulse()
        self._pulse_norm = None
        if hasattr(self, "_sync_coloc_start"):
            self._sync_coloc_start()                  # clear the co-localization picker too
        self._highlight_active_peak()                 # active_mz None → clears the halo
        t = getattr(self, "feat_table", None)
        if t is not None and t.selectionModel().selectedRows():
            t.blockSignals(True)
            t.clearSelection()
            t.blockSignals(False)
        self._update_selected_feature_ui()            # → "No feature selected"
        if getattr(self, "color_overlay_chk", None) is not None and self.color_overlay_chk.isChecked():
            self._draw_overlay_frame()                # static overlay, no pulse
        self.statusBar().showMessage("Feature deselected.")

    def _highlight_active_peak(self):
        """Sit the halo marker on the apex of the spectrum near the active m/z, so a
        feature picked from the list lights up the corresponding peak."""
        pm = getattr(self, "peak_marker", None)
        if pm is None:
            return
        axis, y = self._disp_axis, self._disp_y
        j = _apex_bin(axis, y, self.active_mz)
        if j is None:
            pm.clear()
            return
        pm.setData([float(axis[j])], [float(y[j])])

    def _apex_mz(self, mz):
        """Snap a catalogued feature m/z to its true spectral apex, for **display
        extraction only** — never the stored/label m/z, which stays the catalogued value
        (identity, annotation, quantification all key off it).

        A picked-peak m/z is a 3-point intensity-weighted centroid (``msi._centroid``),
        so on an asymmetric peak it can sit up to ~1 bin off the real apex. With the tight
        extraction window (default 10 ppm) that offset slides the boxcar off the peak and
        blanks the ion image — the reported "labelled feature, empty image". Centring the
        extraction on the apex of the displayed mean/max spectrum keeps the image on the
        peak and coincident with the halo marker (both go through :func:`_apex_bin`).
        Falls back to ``mz`` unchanged when no spectrum is loaded."""
        j = _apex_bin(self._disp_axis, self._disp_y, mz)
        return float(self._disp_axis[j]) if j is not None else mz

    def _ensure_mz_visible(self, mz):
        """If the active m/z sits outside the current zoom window, pan (keep the zoom
        width) so the selected peak scrolls into view."""
        if self._disp_axis is None:
            return
        vb = self.spectrum.getViewBox()
        (x0, x1), _ = vb.viewRange()
        if mz < x0 or mz > x1:
            half = (x1 - x0) / 2.0
            vb.setXRange(mz - half, mz + half, padding=0)

    def _rotate_image(self):
        """Rotate the dataset's display orientation 90° CW and redraw every view."""
        if self.ds is None:
            self.statusBar().showMessage("Load a dataset first.")
            return
        k = self.ds.rotate90(clockwise=True)
        self._reorient_views()
        if hasattr(self, "_mark_dirty"):
            self._mark_dirty()                   # persist orientation with the session
        self.statusBar().showMessage(f"Rotated 90° → {k * 90}°. Orientation saved with the session.")

    def _reorient_views(self):
        """Redraw all spatial views after an orientation change and refit their viewports.
        Each call is guarded so a not-yet-built view can't wedge the rotation."""
        safe = guarded                   # best-effort refresh; a dead view must not wedge rotation
        # the pixel→index lookup is keyed by *displayed* (row, col); a rotation changes
        # that mapping, so rebuild it or clicking a pixel reads the wrong spectrum.
        safe(self._build_pixel_lookup)
        safe(self.refresh_ion_image)
        if getattr(self, "seg", None) is not None:
            safe(self._render_seg_base)
        safe(self._render_region_map)
        safe(self._render_brush_overlay)         # per-pixel ROI mask → re-rasterized rotated
        # geometric ROI tools live in view coords, not as masks, so rotate their geometry to
        # track the tissue, then refresh the selection band + ROI spectrum for the new pose.
        safe(self._reorient_roi_shapes)
        safe(self._roi_spectrum)
        # region footprints painted on the ion view: re-rasterize the *same* regions so they
        # rotate with the base image instead of staying in the old orientation (misaligned).
        ov = getattr(self, "_region_overlay", None)
        ris = getattr(self, "_overlay_ris", None)
        if ov is not None and ov.isVisible() and ris:
            safe(self._show_region_overlay, list(ris))
        safe(self._refresh_optical)
        for vb in (getattr(getattr(self, "iv", None), "view", None),
                   getattr(self, "seg_vb", None)):
            if vb is not None:
                guarded(vb.autoRange)

    def _apply_cmap(self, cmap):
        """Apply the colormap to the ion ImageView only when it actually changed.

        ``ImageView.setColorMap`` rebuilds the gradient editor's 256 stops (tens of
        thousands of graphics-item ops) every call — ~16 ms of pure waste on EVERY
        ion redraw, which made clicking through a long feature list lag. The colormap
        persists on the view across ``setImage``, so re-apply it only on a real change."""
        if cmap != getattr(self, "_applied_cmap", None):
            self.iv.setColorMap(colormap(cmap))
            self._applied_cmap = cmap

    def refresh_ion_image(self):
        if self.ds is None:
            return
        if getattr(self, "color_overlay_chk", None) is not None and self.color_overlay_chk.isChecked():
            self._render_color_overlay()
            return
        self._stop_pulse()                       # no pulse outside the overlay
        cmap = self.cmap_combo.currentText()
        high = float(self.contrast_spin.value())
        # The image tracks the *active* feature, but when nothing is selected we keep the
        # last-shown ion image as a static backdrop — deselecting, or building a region's
        # own feature list (auto-detect / "Add ROI"), clears active_mz. That backdrop must
        # still re-render on a Rotate; otherwise it freezes at the old orientation while the
        # ROI / region overlays rotate under it, so the overlays float off the tissue.
        mz = self.active_mz if self.active_mz is not None else getattr(self, "_displayed_mz", None)
        if mz is None:
            # Nothing has been displayed yet: fall back to the TIC as a tissue backdrop so the
            # ion view is never blank and still rotates — no need to pick a peak first to make
            # Rotate do anything (the reported "I have to select a peak for it to rotate").
            img = self.ds.tic_image()
            auto = (getattr(self, "_iv_img_shape", None) != img.shape)
            self._iv_img_shape = img.shape
            self.iv.setImage(imaging.quantile_clip(img, high=high), autoLevels=True, autoRange=auto)
            self._apply_cmap(cmap)
            self._refresh_optical()
            self._show_scale_legend([])          # TIC has no single-peak % readout
            self._set_ion_tab_caption("Total ion current")
            return
        self._displayed_mz = mz
        p = self._peak_for_mz(mz)
        if self._park_ion_until_cache(mz, p):
            return
        # A class composite renders the composite of its member ions (acts as one feature);
        # a normal feature renders its single ion image.
        img = (self._peak_image(p) if p is not None
               else self.ds.ion_image(mz, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm))
        # Preserve the user's zoom/pan across peak changes — only refit the view when the
        # tissue grid itself changes (a new dataset or a rotation), not on every new ion.
        auto = (getattr(self, "_iv_img_shape", None) != img.shape)
        self._iv_img_shape = img.shape
        if p is not None and ("lo" in p or "hi" in p):
            # honor the feature's own intensity window (percentile contrast handles)
            self.iv.setImage(self._windowed(img, *self._feature_window(p)), autoLevels=False,
                             levels=(0.0, 1.0), autoRange=auto)
        else:
            self.iv.setImage(imaging.quantile_clip(img, high=high), autoLevels=True, autoRange=auto)
        self._apply_cmap(cmap)
        self._refresh_optical()                  # keep the backdrop opacity after re-imaging
        if p is not None and p.get("is_class"):
            cls, n = p.get("lipid_class", "class"), len(p.get("members", []))
            self._show_scale_legend([])
            self._set_ion_tab_caption(f"{cls} · composite ({n} ions)")
            self.statusBar().showMessage(f"{cls} — composite of {n} member ion(s).")
        else:
            # Peak intensity as a % of the global Hotspot clip (= 100%). Anchored to that fixed
            # setting, NOT the per-feature contrast window — so dragging the intensity-window /
            # histogram handles changes the displayed contrast but never moves the peak readout.
            self._show_scale_legend([(mz, None, imaging.relative_max(img, high))])
            lab = self.annotate(mz)
            self._set_ion_tab_caption(f"m/z {mz:.3f}")
            self.statusBar().showMessage(f"m/z {mz:.4f}  {lab}".strip())

    def _set_ion_tab_caption(self, detail):
        """Keep the Ion-image tab title short and *stable* ('Ion image') so the live view
        detail never widens the tab strip and pushes other tabs off the right edge. The
        detail (m/z · TIC · overlay · ratio …) rides the tab's tooltip instead — it's also
        in the status bar and the right-panel 'Selected' chip, so nothing is lost."""
        if self.tabs.tabText(0) != "Ion image":
            self.tabs.setTabText(0, "Ion image")
        strip = getattr(self, "tab_strip", None)
        if strip is not None and strip.count() > 0:
            strip.setTabToolTip(0, f"Ion image — {detail}" if detail else "Ion image")

    def _park_ion_until_cache(self, mz, p):
        """While the fast cache is still building, an m/z with no extracted column would
        stream every spectrum off disk *on the GUI thread* — minutes on a large slide, per
        click. Park the render instead and let ``_on_fast_cache_ready`` redraw it from the
        cube. Returns True when the render was parked."""
        if not getattr(self, "_cube_building", False) or self.ds is None:
            return False
        ds, ppm, red = self.ds, self.ppm, self.reduce
        if p is not None and p.get("is_class"):
            needs = any(ds.ion_needs_disk_pass(self._apex_mz(m), ppm, red)
                        for m in (p.get("members") or []))
        elif p is not None:
            needs = all(ds.ion_needs_disk_pass(m, ppm, red)
                        for m in (self._apex_mz(p["mz"]), float(p["mz"])))
        else:
            needs = ds.ion_needs_disk_pass(float(mz), ppm, red)
        if not needs:
            return False
        self._pending_ion_refresh = True
        self.statusBar().showMessage(
            f"m/z {mz:.4f} — the ion image appears when the fast cache finishes building "
            "(drawing it now would read every spectrum off disk).")
        return True

    def _peak_image(self, p):
        """Displayed image for a feature: a class composite renders the composite of its
        member ions (so a class behaves as one feature everywhere it's drawn); a normal
        feature renders its single ion image."""
        if p is not None and p.get("is_class"):
            members = [self._apex_mz(m) for m in (p.get("members") or [])]
            return self.ds.composite_image(members, tol_ppm=self.ppm,
                                           reduce=self.reduce, norm=self.norm,
                                           weight=self.composite_weight)
        mz = self._apex_mz(p["mz"])
        # Before the cube exists, an apex-snapped m/z that misses the extracted feature
        # column would stream the whole slide off disk; the catalogued m/z's exact column
        # (already extracted at this tolerance) is the right image until the cube is ready.
        if (mz != float(p["mz"]) and self.ds.ion_needs_disk_pass(mz, self.ppm, self.reduce)
                and not self.ds.ion_needs_disk_pass(float(p["mz"]), self.ppm, self.reduce)):
            mz = float(p["mz"])
        return self.ds.ion_image(mz, tol_ppm=self.ppm, reduce=self.reduce, norm=self.norm)

    def _windowed(self, img, lo, hi):
        """Normalize an ion image to [0,1] for display using the feature's
        contrast window. ``lo``/``hi`` are **relative intensity** — percentages of the
        hotspot-clip anchor (the same 100% the peak readout uses), not pixel percentiles —
        so a handle of 90 maps consistently across uses. Off-tissue stays
        NaN/transparent."""
        clip = float(self.contrast_spin.value()) if getattr(self, "contrast_spin", None) is not None else 99.0
        return imaging.relative_window(img, lo, hi, clip=clip)

    def _overlay_key(self):
        """A cheap signature of everything the base composite depends on, so it's only
        recomputed when the visible set / colors / windows / normalization change — not
        on every selection (which keeps selecting a new feature snappy)."""
        return (self.norm, self.reduce, self.ppm,           # base depends on these too (cache-bust on change)
                getattr(self.ds, "orientation", 0),         # rotation re-scatters every channel → re-composite
                tuple((round(p["mz"], 4), self._feature_color(p),
                       *self._feature_window(p)) for p in self._overlay_peaks()))

    color_overlay_max = 10   # cap the multi-channel overlay — too many colours turn to mush

    def _render_color_overlay(self):
        """Multi-channel overlay: each visible feature drawn at once, tinted by its color
        and stretched by its own intensity window, combined with a per-pixel *maximum*
        (lighten). Capped at :attr:`color_overlay_max` (the strongest by intensity) so it
        stays legible, and cleared back to the plain TIC backdrop — not a black frame —
        when every feature's eye is off. The base composite is cached, so re-rendering only
        recomputes it when the visible set / colours / windows change."""
        # Peak % is anchored to the global Hotspot clip (a fixed setting), not each feature's
        # contrast window — so it stays put when the intensity-window handles are dragged.
        clip = float(self.contrast_spin.value()) if getattr(self, "contrast_spin", None) is not None else 99.0
        vis = self._overlay_peaks()
        if not vis:
            # all eyes off → clear the overlay back to the TIC backdrop (don't show black)
            self._overlay_base = self._overlay_base_key = None
            self._stop_pulse()
            self._pulse_norm = None
            img = self.ds.tic_image()
            self._iv_img_shape = img.shape
            self.iv.setImage(imaging.quantile_clip(img, high=clip), autoLevels=True)
            self._apply_cmap(self.cmap_combo.currentText())
            self._refresh_optical()
            self._show_scale_legend([])
            self._set_ion_tab_caption("color overlay — all hidden (showing TIC)")
            self.statusBar().showMessage("Color overlay cleared — every feature is hidden (showing TIC).")
            return
        capped = len(vis) > self.color_overlay_max
        peaks = (sorted(vis, key=lambda p: float(p.get("intensity", 0.0) or 0.0),
                        reverse=True)[:self.color_overlay_max] if capped else vis)
        key = (self._overlay_key(), self.color_overlay_max)   # the cap participates in the cache key
        if getattr(self, "_overlay_base", None) is None or key != getattr(self, "_overlay_base_key", None):
            acc, used, rows = None, 0, []
            for p in peaks:
                img = self._peak_image(p)        # composite for a class feature, else single ion
                win = self._feature_window(p)
                norm = self._windowed(img, *win)
                colour = self._feature_color(p)
                col = QtGui.QColor(colour)
                tint = np.dstack([norm * col.redF(), norm * col.greenF(), norm * col.blueF()])
                acc = tint if acc is None else np.maximum(acc, tint)
                rows.append((p["mz"], colour, imaging.relative_max(img, clip)))
                used += 1
            if acc is None:
                acc = np.zeros((self.ds.height, self.ds.width, 3), dtype=float)
            self._overlay_base, self._overlay_base_key = acc, key
            self._overlay_legend_rows = rows
            cap_note = f" — top {used} of {len(vis)} (cap {self.color_overlay_max})" if capped else ""
            self._set_ion_tab_caption(f"color overlay ({used} features){cap_note}")
            self.statusBar().showMessage(f"Color overlay: {used} feature(s){cap_note}, "
                                         "each by its own color + intensity window.")
        self._show_scale_legend(getattr(self, "_overlay_legend_rows", []))
        self._update_pulse_target()
        self._draw_overlay_frame()

    def _update_pulse_target(self):
        """The selected-feature pulse was removed: a ~22 fps QTimer here repainted the whole
        colour overlay every tick (a full ImageView.setImage), which saturated the Qt UI thread
        and made everything — even repositioning the overlay — lag for as long as the overlay was
        on. The selected feature is already drawn in its own colour on the composite, so no extra
        per-selection highlight is added now."""
        self._pulse_norm = None
        self._stop_pulse()

    def _stop_pulse(self):
        # No animation timer remains (the pulse was removed); kept as a safe no-op guard for the
        # call sites that "stop the pulse" on deselect / when leaving the overlay.
        t = getattr(self, "_pulse_timer", None)
        if t is not None and t.isActive():
            t.stop()

    def _show_histogram(self, show):
        ui = getattr(self.iv, "ui", None)
        if ui is not None and hasattr(ui, "histogram"):
            ui.histogram.setVisible(show)

    # ----- in-image legend + scale bar (WYSIWYG) --------------- #
    def _subject_bbox_view(self):
        """Tissue bounding box ``(r0, r1, c0, c1)`` in data/array coords (row range, col
        range), cached per image shape so the legend can sit beside the subject. The overlay
        maps it into the view (col→view-x, row→view-y). Returns None when there's no dataset."""
        if self.ds is None:
            return None
        key = (self.ds.n_pixels, self.ds.height, self.ds.width, getattr(self.ds, "orientation", None))
        if getattr(self, "_subject_bbox_key", None) != key:
            occ = np.isfinite(self.ds.to_image(np.ones(self.ds.n_pixels), fill=np.nan))
            rows = np.where(occ.any(axis=1))[0]
            cols = np.where(occ.any(axis=0))[0]
            self._subject_bbox_cache = (
                (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
                if rows.size and cols.size else None)
            self._subject_bbox_key = key
        return self._subject_bbox_cache

    def _live_scale_bar_um(self):
        """A round scale-bar length (1/2/5 × 10ⁿ µm) ≈ 20% of the slide width, or None when
        the pixel size is unknown. Shares annotations.nice_scalebar_um with the export and
        crop-studio renderers so every scale bar in the app sizes (and labels) identically."""
        px = getattr(self.ds, "pixel_size_um", None)
        if not px:
            return None
        from ..annotations import nice_scalebar_um
        return nice_scalebar_um(self.ds.width * float(px))

    def _show_scale_legend(self, rows):
        """Drive the in-image overlay. ``rows`` is a list of ``(mz, color-or-None, peak%)``
        tuples (single-ion → one row, colour None; colour overlay → one per visible feature).
        Enriched here with each feature's contrast window, lipid name, and any rename."""
        annot = getattr(self, "ion_annot", None)
        if annot is None:
            return
        if not rows:
            annot.clear()
            return
        cmap = self.cmap_combo.currentText() if getattr(self, "cmap_combo", None) is not None else "viridis"
        rich = []
        for mz, color, pct in rows:
            p = self._peak_for_mz(mz)
            lo, hi = self._feature_window(p)
            rich.append({"mz": mz, "color": color, "cmap": (None if color else cmap),
                         "lo": lo, "hi": hi, "peak": pct, "ppm": self.ppm,
                         "name": self._clean_label(mz),
                         "override": (p.get("label_override") if p else None)})
        annot.set_data(rich, subject=self._subject_bbox_view(),
                       pixel_size_um=getattr(self.ds, "pixel_size_um", None),
                       scale_bar_um=self._live_scale_bar_um())

    def _rename_feature_label(self, mz):
        """Rename the legend label for the feature at ``mz`` (a wrong lipid, or trim to taste).
        Stored on the peak as ``label_override`` so it persists with the session and shows in
        exports too. Blank restores the default label."""
        p = self._peak_for_mz(mz)
        if p is None:
            return
        cur = p.get("label_override") or self._clean_label(mz) or f"{mz:.4f} m/z"
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Rename label", f"Label for m/z {mz:.4f}\n(blank = default):",
            QtWidgets.QLineEdit.Normal, str(cur))
        if not ok:
            return
        self.record_undo("rename label")
        text = text.strip()
        if text:
            p["label_override"] = text
        else:
            p.pop("label_override", None)
        self._mark_dirty()
        self.refresh_ion_image()
        self.statusBar().showMessage(
            f"Label for m/z {mz:.4f} set to “{text}”." if text else
            f"Label for m/z {mz:.4f} reset to default.")

    def _reuse_buf(self, name, shape, dtype):
        """Return a cached scratch array of (shape, dtype), reallocating only when the
        shape changes — so redrawing the colour overlay doesn't churn the allocator/GC."""
        b = getattr(self, name, None)
        if b is None or b.shape != shape or b.dtype != dtype:
            b = np.empty(shape, dtype=dtype)
            setattr(self, name, b)
        return b

    def _draw_overlay_frame(self):
        """Paint the cached colour-overlay composite once. If `_pulse_norm` is set (a lipid-class
        focus highlight) it's brightened toward white statically — this used to animate at ~22 fps
        via a timer, which saturated the UI thread; it's now a single static frame per change."""
        base = getattr(self, "_overlay_base", None)
        if base is None:
            return
        pn = getattr(self, "_pulse_norm", None)
        f = self._reuse_buf("_pulse_fbuf", base.shape, float)
        if pn is not None:
            a = self._reuse_buf("_pulse_abuf", pn.shape, float)
            np.multiply(pn, 0.75, out=a)                       # push highlighted pixels toward white
            np.add(base, a[..., None], out=f)
        else:
            np.copyto(f, base)
        np.clip(f, 0.0, 1.0, out=f)
        np.multiply(f, 255.0, out=f)
        # ping-pong the uint8 buffer handed to pyqtgraph: it reads the array lazily on the next
        # paint, so the in-flight frame must not be the one we overwrite on the following draw.
        idx = getattr(self, "_pulse_u8_idx", 0) ^ 1
        self._pulse_u8_idx = idx
        out = self._reuse_buf(f"_pulse_u8buf{idx}", base.shape, np.ubyte)
        np.copyto(out, f, casting="unsafe")
        # Explicit 0–255 levels (don't inherit the single-ion render's stale (0,1)); and
        # autoRange/autoHistogramRange OFF so drawing the overlay doesn't re-zoom the view or
        # bounce the histogram.
        self.iv.setImage(out, autoLevels=False, levels=(0, 255),
                         autoRange=False, autoHistogramRange=False)

    def save_ion_image(self):
        """Open the Export hub scoped to the current ion view — single ion image (with the
        composited corner spectrum/colour-scale card) or the colour overlay — so the user
        picks the format (PNG/TIFF/JPEG/PDF/SVG) and design options."""
        if self.ds is None:
            return
        overlay = getattr(self, "color_overlay_chk", None) is not None and self.color_overlay_chk.isChecked()
        self.open_export_hub(scope="overlay" if overlay else "ion")

