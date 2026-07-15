"""Crop Studio — the dedicated workflow for setting a region's export close-up.

Opening it (automatically the first time a region is exported as a close-up, or via the
region list's "Edit close-up crop…") shows the ion image with the region outlined and a
draggable, eight-handle crop rectangle, beside a **live preview of the exact exported panel**
(outline + intensity scale + scale bar) — so the crop is WYSIWYG. The crop is saved on the
region (``rg['crop']`` / ``rg['crop_orient']``) and reused by every later export.

To reproduce consistent panels across a project, a **project standard** (aspect ratio + an
optional fixed physical size) can be set as the default and **applied to every region** at
once, so each close-up is framed the same way.

Coordinates follow the rest of the app: in pyqtgraph's row-major display the ion image is
shown so that view-x = pixel *column* and view-y = pixel *row* (see
:meth:`IonTabMixin._roi_mask_filled`), so the crop rectangle maps to ``(r0, r1, c0, c1)``
display-grid bounds — exactly what :func:`export.render_ion_panel` crops to.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import annotations as annot, export, imaging
from .common import (colormap, dark_image_view, fig_to_pixmap, ION_CANVAS_BG, MUTED_QSS, icon,
                     NoScrollComboBox, NoScrollSlider)

# Aspect presets offered in the project-standard combo: label → width/height (None = free).
ASPECTS = [("Free", None), ("1:1 square", 1.0), ("4:3", 4 / 3), ("3:2", 3 / 2), ("16:9", 16 / 9)]


class CropStudio(QtWidgets.QDialog):
    """Set/refine the export close-up crop for one region, with a live preview."""

    def __init__(self, win, region, *, allow_apply_all=True):
        super().__init__(win)
        self.win = win
        self.rg = region
        self.ds = win.ds
        self.setWindowTitle(f"Crop Studio — {region.get('name', 'region')}")
        self.setMinimumSize(940, 560)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(140)             # debounce the matplotlib render on drag
        self._preview_timer.timeout.connect(self._render_preview)

        root = QtWidgets.QVBoxLayout(self)

        # ---- top controls: reference ion + padding ----
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Reference ion:"))
        self.ref_combo = NoScrollComboBox()
        for label, mz in self._ref_options():
            self.ref_combo.addItem(label, mz)
        self.ref_combo.currentIndexChanged.connect(self._ref_changed)
        top.addWidget(self.ref_combo, 1)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Padding:"))
        self.pad_slider = NoScrollSlider(QtCore.Qt.Horizontal)
        self.pad_slider.setRange(0, 50); self.pad_slider.setValue(10)
        self.pad_slider.setFixedWidth(120)
        self.pad_slider.setToolTip("Margin around the ROI when fitting the crop to its bounds.")
        self.pad_slider.valueChanged.connect(self._pad_changed)
        top.addWidget(self.pad_slider)
        self.pad_lbl = QtWidgets.QLabel("10%")
        top.addWidget(self.pad_lbl)
        b_fit = QtWidgets.QPushButton("Fit to ROI")
        b_fit.setIcon(icon("refresh"))
        b_fit.setToolTip("Reset the crop box to the region's bounding box plus the padding.")
        b_fit.clicked.connect(self._fit_to_roi)
        top.addWidget(b_fit)
        root.addLayout(top)

        # ---- image (editable) | preview (WYSIWYG) ----
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.iv = pg.ImageView()
        self.iv.ui.histogram.hide(); self.iv.ui.roiBtn.hide(); self.iv.ui.menuBtn.hide()
        dark_image_view(self.iv)         # crop source ion image reads against black in both themes
        self.iv.view.invertY(True); self.iv.view.setAspectLocked(True)
        self._tint = pg.ImageItem(); self._tint.setZValue(20); self.iv.view.addItem(self._tint)
        self.crop_roi = pg.RectROI([1, 1], [5, 5],
                                   pen=pg.mkPen("#ffd60a", width=2.5, style=QtCore.Qt.DashLine))
        self.crop_roi.handlePen = pg.mkPen("#1b1e24", width=1.0)
        for h_pos, h_center in (([0, 0], [1, 1]), ([1, 0], [0, 1]), ([0, 1], [1, 0]),
                                ([0.5, 0], [0.5, 1]), ([0.5, 1], [0.5, 0]),
                                ([0, 0.5], [1, 0.5]), ([1, 0.5], [0, 0.5])):
            self.crop_roi.addScaleHandle(h_pos, h_center)   # resize from any corner/edge
        self.crop_roi.setZValue(30)
        self.iv.view.addItem(self.crop_roi)
        self._dotify_handles()                           # small solid round grab dots
        self.crop_roi.sigRegionChanged.connect(self._box_changed)
        split.addWidget(self.iv)

        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QtWidgets.QLabel("Export preview"))
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumWidth(300)
        # The preview shows the rendered ion-image crop, which is exported on a dark canvas —
        # keep it on the shared dark ION_CANVAS_BG (not the light panel) so the WYSIWYG matches.
        self.preview.setStyleSheet(f"background:{ION_CANVAS_BG}; border:1px solid palette(mid);")
        rv.addWidget(self.preview, 1)
        self.size_lbl = QtWidgets.QLabel("")
        self.size_lbl.setStyleSheet(MUTED_QSS)
        rv.addWidget(self.size_lbl)
        split.addWidget(right)
        split.setSizes([560, 360])
        root.addWidget(split, 1)

        # ---- project standard ----
        std = QtWidgets.QGroupBox("Project standard — reproduce consistent panels")
        sl = QtWidgets.QHBoxLayout(std)
        sl.addWidget(QtWidgets.QLabel("Aspect:"))
        self.aspect_combo = NoScrollComboBox()
        for label, val in ASPECTS:
            self.aspect_combo.addItem(label, val)
        self.aspect_combo.currentIndexChanged.connect(self._aspect_changed)
        sl.addWidget(self.aspect_combo)
        self.lock_size = QtWidgets.QCheckBox("Same physical size for every region")
        self.lock_size.setToolTip("When applying to all regions, use this crop box's physical "
                                   "size (µm) centred on each ROI — so every close-up is the same "
                                   "scale. Needs a known pixel size.")
        self.lock_size.setEnabled(getattr(self.ds, "pixel_size_um", None) is not None)
        sl.addWidget(self.lock_size)
        sl.addStretch(1)
        b_default = QtWidgets.QPushButton("Set as project default")
        b_default.setIcon(icon("save"))
        b_default.setToolTip("Remember this aspect / size as the default framing for new region crops.")
        b_default.clicked.connect(self._set_default)
        sl.addWidget(b_default)
        sl.addWidget(QtWidgets.QLabel("Apply to:"))
        b_one = QtWidgets.QPushButton("This region")
        b_one.setIcon(icon("settings"))
        b_one.setToolTip("Frame just this region to the standard (aspect, and the same physical "
                         "size if locked). Review it, then Save crop.")
        b_one.clicked.connect(self._apply_one)
        sl.addWidget(b_one)
        self.b_all = QtWidgets.QPushButton("All regions")
        self.b_all.setIcon(icon("settings"))
        self.b_all.setEnabled(allow_apply_all and len(getattr(win, "regions", []) or []) > 1)
        self.b_all.setToolTip("Frame every region's close-up the same way at once.")
        self.b_all.clicked.connect(self._apply_all)
        sl.addWidget(self.b_all)
        root.addWidget(std)

        # ---- dialog buttons ----
        btns = QtWidgets.QDialogButtonBox()
        b_save = btns.addButton("Save crop", QtWidgets.QDialogButtonBox.AcceptRole)
        b_save.setIcon(icon("save"))
        btns.addButton(QtWidgets.QDialogButtonBox.Cancel)
        b_save.clicked.connect(self._save)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # seed an existing aspect/size preset onto the controls, then load the image + box
        self._apply_preset_to_controls(getattr(win, "_crop_preset", None))
        self._ref_changed()
        self._seed_box()

    # ------------------------------------------------------------------ #
    # reference image
    # ------------------------------------------------------------------ #
    def _ref_options(self):
        opts = []
        active = getattr(self.win, "active_mz", None)
        peaks = self.win._visible_peaks() if getattr(self.win, "peaks", None) else []
        seen = set()
        order = ([self.win._peak_for_mz(active)] if active else []) + list(peaks)
        for p in order:
            if p is None or p["mz"] in seen:
                continue
            seen.add(p["mz"])
            lab = (self.win._clean_label(p["mz"]) or "").strip()
            opts.append((f"m/z {p['mz']:.4f}  {lab}".strip(), float(p["mz"])))
            if len(opts) >= 40:
                break
        opts.append(("Total ion image (TIC)", None))
        return opts

    def _ref_image(self):
        mz = self.ref_combo.currentData()
        if mz is None:
            return imaging.quantile_clip(self.ds.tic_image(), high=99), "viridis", (0.0, 1.0)
        img = self.ds.ion_image(self.win._apex_mz(mz), tol_ppm=self.win.ppm,
                                reduce=self.win.reduce, norm=self.win.norm)
        return imaging.quantile_clip(img, high=99.0), self._cmap(), (0.0, 1.0)

    def _cmap(self):
        c = getattr(self.win, "cmap_combo", None)
        return c.currentText() if c is not None else "viridis"

    def _ref_changed(self):
        img, cmap, levels = self._ref_image()
        self.iv.setImage(img, autoLevels=False, levels=levels, autoRange=False)
        self.iv.setColorMap(colormap(cmap))
        self._draw_region_tint()
        self._schedule_preview()

    def _draw_region_tint(self):
        m2d = self.win._region_mask_2d(self.rg)
        if m2d is None:
            self._tint.clear(); return
        h, w = m2d.shape
        rgba = np.zeros((h, w, 4), np.ubyte)
        col = QtGui.QColor(self.rg.get("color", "#ff3b30"))
        # Outline ring only (no interior fill) — a translucent fill would blend with bright
        # pixels and misrepresent the ion colours; the ring delineates the ROI cleanly while
        # the true colormap shows through, matching the export preview on the right.
        edge = m2d & ~_erode(m2d)
        rgba[edge] = (col.red(), col.green(), col.blue(), 235)
        self._tint.setImage(rgba)

    # ------------------------------------------------------------------ #
    # crop box <-> region geometry
    # ------------------------------------------------------------------ #
    def _seed_box(self):
        """Show the crop box from the region's saved crop, else the project preset, else its
        padded bounding box."""
        crop = self.rg.get("crop") if self.rg.get("crop_orient", self.ds.orientation) == self.ds.orientation else None
        if crop is None:
            crop = self.win._crop_from_preset(self.rg, pad_frac=self.pad_slider.value() / 100.0)
        if crop is None:
            return
        self._set_box(*crop)
        self._zoom_to_box()                              # open zoomed in on the region

    def _set_box(self, r0, r1, c0, c1):
        self.crop_roi.blockSignals(True)
        self.crop_roi.setPos([float(c0), float(r0)])     # x = col, y = row
        self.crop_roi.setSize([max(1.0, c1 - c0), max(1.0, r1 - r0)])
        self.crop_roi.blockSignals(False)
        self._box_changed()

    def _box_crop(self):
        """Current box as clamped ``(r0, r1, c0, c1)`` display-grid bounds."""
        pos, size = self.crop_roi.pos(), self.crop_roi.size()
        h, w = self.ds.height, self.ds.width
        c0, c1 = sorted((pos.x(), pos.x() + size.x()))   # view-x = col
        r0, r1 = sorted((pos.y(), pos.y() + size.y()))   # view-y = row
        r0, r1 = max(0, min(int(round(r0)), h)), max(0, min(int(round(r1)), h))
        c0, c1 = max(0, min(int(round(c0)), w)), max(0, min(int(round(c1)), w))
        if r1 - r0 < 1 or c1 - c0 < 1:
            return None
        return (r0, r1, c0, c1)

    def _box_changed(self):
        crop = self._box_crop()
        if crop is None:
            return
        r0, r1, c0, c1 = crop
        wpx, hpx = c1 - c0, r1 - r0                      # export width = cols, height = rows
        px = getattr(self.ds, "pixel_size_um", None)
        if px:
            self.size_lbl.setText(f"{wpx}×{hpx} px   ·   {wpx * px:.0f}×{hpx * px:.0f} µm")
        else:
            self.size_lbl.setText(f"{wpx}×{hpx} px")
        self._schedule_preview()

    def _dotify_handles(self):
        """Render the resize handles as small, solid round dots instead of the default large
        diamonds: a round (many-sided) path with a thick stroke fills to a clean dot — robust
        across pyqtgraph versions (no per-instance paint override, which Qt virtuals ignore)."""
        dot = pg.mkPen("#ffd60a", width=4.0)
        hov = pg.mkPen("#ffffff", width=5.0)
        for h in self.crop_roi.handles:
            it = h["item"]
            try:
                it.sides = 20                            # ~circle
                it.radius = 4.0                          # small grab dot
                it.pen = it.currentPen = dot
                it.hoverPen = hov
                it.buildPath()
                it.update()
            except Exception:  # noqa: BLE001 — degrade to default handles if internals differ
                pass

    def _zoom_to_box(self, padding=0.45):
        """Frame the view on the crop box (+ a margin) so the Studio opens zoomed in on the
        region instead of the whole slide. Coords: view-x = col, view-y = row."""
        crop = self._box_crop()
        if crop is None:
            return
        r0, r1, c0, c1 = crop
        self.iv.view.setRange(xRange=(c0, c1), yRange=(r0, r1), padding=padding)

    def _schedule_preview(self):
        self._preview_timer.start()

    def _render_preview(self):
        crop = self._box_crop()
        if crop is None:
            return
        mz = self.ref_combo.currentData()
        try:
            img, _cmap, _lv = self._ref_image_raw()
            m2d = self.win._region_mask_2d(self.rg)
            px = getattr(self.ds, "pixel_size_um", None)
            # a nice round bar sized to *this crop's* physical width, not a blind 200µm that
            # a small close-up would have to clamp (and then mislabel)
            r0, r1, c0, c1 = crop
            sb = annot.nice_scalebar_um((c1 - c0) * float(px)) if px else None
            fig = export.render_ion_panel(
                img, mz=mz, label=(self.win._clean_label(mz) if mz is not None else ""),
                cmap=self._cmap(), low=0.0, high=99.0, window=(0, 99), crop=crop,
                outline_mask=m2d, outline_color=self.rg.get("color", "#ffffff"),
                dim_outside=True, show_spectrum=False, mean_spectrum=None,
                pixel_size_um=px, scale_bar_um=sb, theme="light", dpi=110, width_in=4.6)
            pm = fig_to_pixmap(fig)
            export._close(fig)
            avail = self.preview.size()
            self.preview.setPixmap(pm.scaled(avail, QtCore.Qt.KeepAspectRatio,
                                             QtCore.Qt.SmoothTransformation))
        except Exception as e:  # noqa: BLE001 — a preview hiccup must never break the dialog
            self.preview.setText(f"(preview unavailable: {type(e).__name__})")

    def _ref_image_raw(self):
        """The *un-clipped* reference image for rendering the preview (render_ion_panel does
        its own percentile contrast), plus cmap/levels — mirrors :meth:`_ref_image`."""
        mz = self.ref_combo.currentData()
        if mz is None:
            return self.ds.tic_image(), "viridis", (0.0, 1.0)
        return (self.ds.ion_image(self.win._apex_mz(mz), tol_ppm=self.win.ppm,
                                  reduce=self.win.reduce, norm=self.win.norm),
                self._cmap(), (0.0, 1.0))

    # ------------------------------------------------------------------ #
    # padding / aspect controls
    # ------------------------------------------------------------------ #
    def _pad_changed(self, v):
        self.pad_lbl.setText(f"{v}%")

    def _fit_to_roi(self):
        bbox = self.win._region_bbox_raw(self.rg, pad_frac=self.pad_slider.value() / 100.0)
        if bbox is not None:
            self._set_box(*bbox)
            self._apply_aspect()
            self._zoom_to_box()

    def _aspect_changed(self):
        self._apply_aspect()

    def _apply_aspect(self):
        a = self.aspect_combo.currentData()              # width/height, or None (free)
        crop = self._box_crop()
        if a is None or crop is None:
            return
        r0, r1, c0, c1 = crop
        hpx = r1 - r0
        cr, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
        new_w = a * hpx                                  # width (cols) = aspect × height (rows)
        self._set_box(cr - hpx / 2.0, cr + hpx / 2.0, cc - new_w / 2.0, cc + new_w / 2.0)

    # ------------------------------------------------------------------ #
    # project standard
    # ------------------------------------------------------------------ #
    def _apply_preset_to_controls(self, preset):
        if not preset:
            return
        a = preset.get("aspect")
        idx = next((i for i, (_l, v) in enumerate(ASPECTS) if v == a), 0)
        self.aspect_combo.setCurrentIndex(idx)
        self.pad_slider.setValue(int(round(preset.get("pad_frac", 0.10) * 100)))
        self.lock_size.setChecked(bool(preset.get("size_um")))

    def _current_preset(self):
        crop = self._box_crop()
        px = getattr(self.ds, "pixel_size_um", None)
        size_um = None
        if self.lock_size.isChecked() and crop and px:
            r0, r1, c0, c1 = crop
            size_um = ((c1 - c0) * px, (r1 - r0) * px)   # (width_um, height_um)
        return {"aspect": self.aspect_combo.currentData(),
                "pad_frac": self.pad_slider.value() / 100.0, "size_um": size_um}

    def _set_default(self):
        self.win._crop_preset = self._current_preset()
        self.win._mark_dirty()
        self.win.statusBar().showMessage("Saved as the project's default crop framing.")

    def _apply_one(self):
        """Frame just this region to the standard (and remember it as the default). Reflects
        in the crop box so the user can review before Save crop persists it."""
        preset = self._current_preset()
        self.win._crop_preset = preset
        crop = self.win._crop_from_preset(self.rg, preset=preset)
        if crop is not None:
            self._set_box(*crop)
            self._zoom_to_box()
        self.win.statusBar().showMessage(
            f"Framed '{self.rg.get('name', 'region')}' to the standard — Save crop to keep it.")

    def _apply_all(self):
        preset = self._current_preset()
        self.win._crop_preset = preset
        regions = [r for r in (self.win.regions or []) if self.win._region_bbox_raw(r) is not None]
        n = 0
        for r in regions:
            crop = self.win._crop_from_preset(r, preset=preset)
            if crop is not None:
                r["crop"] = crop
                r["crop_orient"] = self.ds.orientation
                n += 1
        self.win._mark_dirty()
        self._seed_box()                                 # reflect the standard on this region too
        self.win.statusBar().showMessage(f"Applied the project crop standard to {n} region(s).")

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    def _save(self):
        crop = self._box_crop()
        if crop is None:
            self.win.statusBar().showMessage("Crop box is empty — drag out a region first.")
            return
        self.rg["crop"] = crop
        self.rg["crop_orient"] = self.ds.orientation
        self.win._mark_dirty()
        self.win.statusBar().showMessage(f"Saved close-up crop for '{self.rg.get('name', 'region')}'.")
        self.accept()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _erode(mask):
    """A 1-pixel binary erosion (4-neighbour) without SciPy — for the ROI edge ring."""
    m = np.asarray(mask, bool)
    e = m.copy()
    e[:-1, :] &= m[1:, :]; e[1:, :] &= m[:-1, :]
    e[:, :-1] &= m[:, 1:]; e[:, 1:] &= m[:, :-1]
    return e
