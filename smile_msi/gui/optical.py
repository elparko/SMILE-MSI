"""OpticalMixin — a global optical / histology image layer.

Loads a microscopy photo of the slide (TIFF / JPEG / PNG) and shows it as an
aligned **backdrop** behind the spatial views — the single-ion image, the
multi-channel colour overlay (both share the Ion-image ``ImageView``), and the
segmentation map. One global **Show optical** toggle and one **Spectra opacity**
slider control it everywhere; manual alignment (move · scale · rotate · flip)
lines the photo up with the MSI grid.

Design notes
------------
* The photo is a separate ``pg.ImageItem`` added *behind* each view's main image
  (``ZValue`` far negative, ``ignoreBounds`` so it never disturbs auto-range).
  Showing the ion data *over* it is just lowering the main image item's opacity —
  so this works in every view without touching their render code.
* Alignment is stored as ``{tx, ty, scale, angle, flipx, flipy}`` and baked into a
  ``QTransform`` (see :meth:`_optical_transform`). Identity params place the photo
  stretched-to-fit and centred on the data, matching the old behaviour as a
  starting point; the controls then nudge it into register.
* The image itself is never embedded in the session — only its **path** plus the
  alignment/toggle/opacity, mirroring how the dataset is referenced by source.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import filedialogs
from .common import (icon, tool_button, NoScrollDoubleSpinBox, NoScrollSlider)


class OpticalMixin:
    # ----- state ---------------------------------------------------------- #
    @staticmethod
    def _default_optical_align() -> dict:
        return {"tx": 0.0, "ty": 0.0, "scale": 1.0, "angle": 0.0,
                "flipx": False, "flipy": False}

    def _init_optical_state(self):
        self._optical = None                 # RGB uint8 ndarray, or None
        self._optical_path = None            # source file (persisted; re-read on open)
        self._optical_on = False             # global show/hide
        self._optical_spectra_alpha = 0.5    # opacity of the ion/seg layer over the photo
        self._optical_align = self._default_optical_align()
        self._optical_views = []             # [{'view','item','main'}] registered backdrops
        self._optical_widgets_loading = False  # guard against control↔state feedback loops

    # ----- per-view backdrop registry ------------------------------------- #
    def _register_optical_view(self, view, main_item):
        """Attach a backdrop ImageItem to ``view`` (a pyqtgraph ViewBox), sitting behind
        ``main_item`` (the view's foreground image, whose opacity we lower to reveal the
        photo). Each spatial tab calls this once while it builds."""
        item = pg.ImageItem()
        item.setZValue(-1000)                # behind every data/overlay layer
        item.setVisible(False)
        try:
            view.addItem(item, ignoreBounds=True)   # never widen auto-range to the photo
        except TypeError:                            # older pyqtgraph signature
            view.addItem(item)
        if self._optical is not None:
            item.setImage(self._optical, autoLevels=False)
        self._optical_views.append({"view": view, "item": item, "main": main_item})
        self._refresh_optical()

    def _optical_transform(self):
        """QTransform mapping the photo's pixels onto the MSI grid, or None when there's
        nothing to place. The photo is fit + nudged in the *unrotated* (orientation-0) tissue
        frame, then the dataset's display rotation is applied on top — so the backdrop turns
        **with** the tissue on a Rotate instead of just re-stretching to the new aspect.
        Identity alignment fits the photo to the data extent, centred; scale/rotation/flip act
        about that centre, then (tx, ty) shift it in grid pixels."""
        if self._optical is None or self.ds is None:
            return None
        hp, wp = int(self._optical.shape[0]), int(self._optical.shape[1])
        if hp <= 0 or wp <= 0:
            return None
        a = self._optical_align
        # pg ImageItem (row-major): photo array axis0 (rows, hp) → view-y, axis1 (cols, wp)
        # → view-x, matching the ion/seg images. Fit to the UN-rotated extent (h0, w0).
        h0, w0 = self.ds._base_hw
        base_sx = w0 / wp
        base_sy = h0 / hp
        sx = base_sx * a["scale"] * (-1.0 if a["flipx"] else 1.0)
        sy = base_sy * a["scale"] * (-1.0 if a["flipy"] else 1.0)
        # Fit + manual alignment in the UN-rotated (orientation-0) frame: photo px → base view.
        fit = QtGui.QTransform()
        fit.translate(w0 / 2.0 + a["tx"], h0 / 2.0 + a["ty"])   # view-space offset (orient 0)
        fit.rotate(a["angle"])
        fit.scale(sx, sy)
        fit.translate(-wp / 2.0, -hp / 2.0)                    # centre the photo
        # Display rotation: base-view → current rotated view. Built directly from the ion
        # image's own pixel mapping (one 90°-CW step sends view (x, y) → (Wnew − y, x)), so it
        # tracks to_image exactly — no QTransform.rotate() sign/invertY ambiguity.
        k = int(getattr(self.ds, "orientation", 0)) % 4
        orient = {
            0: QtGui.QTransform(1, 0, 0, 1, 0.0, 0.0),         # identity
            1: QtGui.QTransform(0, 1, -1, 0, float(h0), 0.0),  # (X,Y) → (h0 − Y, X)
            2: QtGui.QTransform(-1, 0, 0, -1, float(w0), float(h0)),  # (w0 − X, h0 − Y)
            3: QtGui.QTransform(0, -1, 1, 0, 0.0, float(w0)),  # (Y, w0 − X)
        }[k]
        return fit * orient                                    # photo → base view → rotated view

    def _refresh_optical(self):
        """Re-place / show / hide the backdrop in every registered view and set the
        foreground opacity. Cheap; call after any optical state change."""
        xform = self._optical_transform()
        show = self._optical is not None and self._optical_on and xform is not None
        for e in getattr(self, "_optical_views", []):
            item, main = e["item"], e["main"]
            if show:
                item.setTransform(xform)
                item.setVisible(True)
                if main is not None:
                    main.setOpacity(self._optical_spectra_alpha)
            else:
                item.setVisible(False)
                if main is not None:
                    main.setOpacity(1.0)

    def _set_optical_image(self, img, path):
        self._optical = img
        self._optical_path = path
        for e in getattr(self, "_optical_views", []):
            if img is not None:
                e["item"].setImage(img, autoLevels=False)
        self._refresh_optical()

    # ----- right-dock controls -------------------------------------------- #
    def _populate_optical_section(self, v):
        v.addWidget(self._note(
            "Show a microscopy / histology photo of the slide as a backdrop behind the "
            "ion image, colour overlay, and segmentation. Load it, then line it up with "
            "the alignment controls — it's saved with the sample."))

        load_row = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load image…")
        b_load.setIcon(icon("open"))
        b_load.setToolTip("Load an optical image (TIFF / JPEG / PNG) of this slide")
        b_load.clicked.connect(self.load_optical)
        load_row.addWidget(b_load)
        self.optical_name_lbl = self._muted(QtWidgets.QLabel("No image loaded"))
        self.optical_name_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                            QtWidgets.QSizePolicy.Preferred)
        load_row.addWidget(self.optical_name_lbl, 1)
        b_clear = tool_button(name="remove", tooltip="Remove the optical image",
                              slot=self._remove_optical)
        b_clear.setAutoRaise(True)
        load_row.addWidget(b_clear)
        v.addLayout(load_row)

        self.optical_show_chk = QtWidgets.QCheckBox("Show optical")
        self.optical_show_chk.setToolTip("Toggle the optical backdrop in every spatial view")
        self.optical_show_chk.setEnabled(False)
        self.optical_show_chk.toggled.connect(self._set_optical_on)
        v.addWidget(self.optical_show_chk)

        op_row = QtWidgets.QHBoxLayout()
        op_row.addWidget(self._muted(QtWidgets.QLabel("Spectra opacity")))
        self.optical_alpha_slider = NoScrollSlider(QtCore.Qt.Horizontal)
        self.optical_alpha_slider.setRange(0, 100)
        self.optical_alpha_slider.setValue(int(round(self._optical_spectra_alpha * 100)))
        self.optical_alpha_slider.setToolTip(
            "Opacity of the ion image / segmentation drawn over the photo. Lower = more "
            "of the photo shows through.")
        self.optical_alpha_slider.valueChanged.connect(self._optical_alpha_changed)
        op_row.addWidget(self.optical_alpha_slider, 1)
        v.addLayout(op_row)

        # --- alignment sub-panel: shown only once an image is loaded ----------
        self.optical_align_toggle = QtWidgets.QToolButton()
        self.optical_align_toggle.setIcon(icon("menu"))
        self.optical_align_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.optical_align_toggle.setText("Align (advanced) ▾")
        self.optical_align_toggle.setAutoRaise(True); self.optical_align_toggle.setCheckable(True)
        self.optical_align_toggle.setToolTip("Manual alignment: move, scale, rotate, and flip the photo to register it with the MSI grid.")
        self.optical_align_toggle.setVisible(False)   # only meaningful once an image is loaded
        def _toggle_align(on):
            self.optical_align_box.setVisible(bool(on) and self._optical is not None)
            self.optical_align_toggle.setText("Align (advanced) ▴" if on else "Align (advanced) ▾")
        self.optical_align_toggle.toggled.connect(_toggle_align)
        v.addWidget(self.optical_align_toggle)

        self.optical_align_box = QtWidgets.QWidget()
        ab = QtWidgets.QVBoxLayout(self.optical_align_box)
        ab.setContentsMargins(0, 2, 0, 0)
        ab.setSpacing(4)

        mode_row = QtWidgets.QHBoxLayout()
        self.optical_drag_chk = QtWidgets.QCheckBox("Drag to move")
        self.optical_drag_chk.setToolTip(
            "While on, drag on the ion image to move the photo (the spinners below set "
            "exact values). Turn off to pan/zoom the view normally.")
        mode_row.addWidget(self.optical_drag_chk)
        mode_row.addStretch(1)
        b_reset = QtWidgets.QPushButton("Reset")
        b_reset.setIcon(icon("refresh"))
        b_reset.setToolTip("Reset alignment to fit-and-centre")
        b_reset.clicked.connect(self._reset_optical_align)
        mode_row.addWidget(b_reset)
        ab.addLayout(mode_row)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.optical_x_spin = self._align_spin(-1e5, 1e5, 5.0, " px")
        self.optical_y_spin = self._align_spin(-1e5, 1e5, 5.0, " px")
        self.optical_scale_spin = self._align_spin(1.0, 2000.0, 5.0, " %")
        self.optical_scale_spin.setValue(100.0)
        self.optical_rot_spin = self._align_spin(-180.0, 180.0, 1.0, " °")
        self.optical_rot_spin.setWrapping(True)
        for sp in (self.optical_x_spin, self.optical_y_spin,
                   self.optical_scale_spin, self.optical_rot_spin):
            sp.valueChanged.connect(self._optical_align_from_widgets)
        form.addRow("Move X", self.optical_x_spin)
        form.addRow("Move Y", self.optical_y_spin)
        form.addRow("Scale", self.optical_scale_spin)
        form.addRow("Rotate", self.optical_rot_spin)
        ab.addLayout(form)

        flip_row = QtWidgets.QHBoxLayout()
        self.optical_fliph_btn = QtWidgets.QPushButton("Flip H")
        self.optical_fliph_btn.setIcon(icon("settings"))
        self.optical_fliph_btn.setCheckable(True)
        self.optical_fliph_btn.toggled.connect(self._optical_align_from_widgets)
        self.optical_flipv_btn = QtWidgets.QPushButton("Flip V")
        self.optical_flipv_btn.setIcon(icon("settings"))
        self.optical_flipv_btn.setCheckable(True)
        self.optical_flipv_btn.toggled.connect(self._optical_align_from_widgets)
        flip_row.addWidget(self.optical_fliph_btn)
        flip_row.addWidget(self.optical_flipv_btn)
        ab.addLayout(flip_row)

        b_land = QtWidgets.QPushButton("Estimate from landmarks…")
        b_land.setToolTip("Estimate the alignment from matching photo↔ion landmark points, "
                          "then verify it against the overlay.")
        b_land.clicked.connect(self._open_landmark_dialog)
        ab.addWidget(b_land)

        self.optical_align_box.setVisible(False)
        v.addWidget(self.optical_align_box)

    @staticmethod
    def _align_spin(lo, hi, step, suffix):
        sp = NoScrollDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(1)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)
        return sp

    # ----- control handlers ----------------------------------------------- #
    def _set_optical_on(self, on):
        if self._optical_widgets_loading:
            return
        self._optical_on = bool(on)
        self._refresh_optical()
        self._sync_optical_show_controls()      # keep dialog checkbox + View toggle in step
        self._mark_dirty()

    def _sync_optical_show_controls(self):
        """Mirror the current show/enabled state onto both the Image-setup dialog's 'Show
        optical' checkbox and the View-menu quick toggle, without re-firing their handlers
        (they share :meth:`_set_optical_on`, so the guard prevents a feedback loop)."""
        self._optical_widgets_loading = True
        try:
            has = self._optical is not None
            chk = getattr(self, "optical_show_chk", None)
            if chk is not None:
                chk.setEnabled(has)
                chk.setChecked(self._optical_on)
            act = getattr(self, "optical_show_action", None)
            if act is not None:
                act.setEnabled(has)
                act.setChecked(self._optical_on)
        finally:
            self._optical_widgets_loading = False

    def _optical_alpha_changed(self, value):
        if self._optical_widgets_loading:
            return
        self._optical_spectra_alpha = max(0.0, min(1.0, value / 100.0))
        # Only the foreground opacity changed — set it directly instead of a full
        # _refresh_optical() (which recomputes + re-applies the transform on every slider step).
        if self._optical is not None and self._optical_on:
            for e in getattr(self, "_optical_views", []):
                main = e["main"]
                if main is not None:
                    main.setOpacity(self._optical_spectra_alpha)
        self._mark_dirty()

    def _optical_align_from_widgets(self, *_):
        if self._optical_widgets_loading:
            return
        a = self._optical_align
        a["tx"] = float(self.optical_x_spin.value())
        a["ty"] = float(self.optical_y_spin.value())
        a["scale"] = max(0.01, self.optical_scale_spin.value() / 100.0)
        a["angle"] = float(self.optical_rot_spin.value())
        a["flipx"] = self.optical_fliph_btn.isChecked()
        a["flipy"] = self.optical_flipv_btn.isChecked()
        self._refresh_optical()
        self._mark_dirty()

    def _apply_optical_coeffs(self, coeffs):
        """Apply registration-estimated ``{tx,ty,scale,angle,flipx,flipy}`` coefficients to the
        optical alignment and sync the controls (plan 05). The user then verifies the result
        visually against the live overlay (a wrong landmark set degrades visibly, never silently)."""
        if self._optical is None:
            return
        self._optical_align = {**self._default_optical_align(), **dict(coeffs)}
        self._optical_widgets_to_align()
        self._refresh_optical()
        self._mark_dirty()

    def _open_landmark_dialog(self):
        """Estimate the optical alignment from landmark pairs (plan 05)."""
        if self._optical is None:
            self.statusBar().showMessage("Load an optical image first (File ▸ Image setup…).")
            return
        from .registerdialog import LandmarkDialog
        LandmarkDialog(self).exec()

    def _optical_widgets_to_align(self):
        """Push the alignment dict back into the controls (after a drag / reset / restore),
        guarded so the resulting valueChanged signals don't re-enter the handlers."""
        if getattr(self, "optical_x_spin", None) is None:
            return
        a = self._optical_align
        self._optical_widgets_loading = True
        try:
            self.optical_x_spin.setValue(a["tx"])
            self.optical_y_spin.setValue(a["ty"])
            self.optical_scale_spin.setValue(a["scale"] * 100.0)
            self.optical_rot_spin.setValue(a["angle"])
            self.optical_fliph_btn.setChecked(a["flipx"])
            self.optical_flipv_btn.setChecked(a["flipy"])
        finally:
            self._optical_widgets_loading = False

    def _reset_optical_align(self):
        self.record_undo("Reset optical alignment", ("optical",))
        self._optical_align = self._default_optical_align()
        self._optical_widgets_to_align()
        self._refresh_optical()
        self._mark_dirty()
        self.statusBar().showMessage("Optical alignment reset.")

    def _sync_optical_controls(self):
        """Bring every optical control in line with the current state (no signals fired)."""
        self._sync_optical_show_controls()        # 'Show optical' checkbox + View-menu toggle
        self._optical_widgets_loading = True
        try:
            has = self._optical is not None
            if getattr(self, "optical_alpha_slider", None) is not None:
                self.optical_alpha_slider.setValue(int(round(self._optical_spectra_alpha * 100)))
            if getattr(self, "optical_name_lbl", None) is not None:
                self.optical_name_lbl.setText(
                    os.path.basename(self._optical_path) if (has and self._optical_path)
                    else ("(loaded)" if has else "No image loaded"))
            if getattr(self, "optical_align_toggle", None) is not None:
                self.optical_align_toggle.setVisible(has)
                if not has:
                    self.optical_align_toggle.setChecked(False)
            if getattr(self, "optical_align_box", None) is not None:
                self.optical_align_box.setVisible(has and self.optical_align_toggle.isChecked())
        finally:
            self._optical_widgets_loading = False
        self._optical_widgets_to_align()

    # ----- drag-to-move (driven by the ion view's drag handler) ----------- #
    def _optical_drag_active(self):
        return (self._optical is not None and self._optical_on
                and getattr(self, "optical_drag_chk", None) is not None
                and self.optical_drag_chk.isChecked())

    def _optical_drag_by(self, dx, dy):
        """Shift the photo by (dx, dy) view-pixels — the live drag on the ion image."""
        a = self._optical_align
        a["tx"] += float(dx)
        a["ty"] += float(dy)
        self._optical_widgets_to_align()
        self._refresh_optical()

    # ----- load / remove --------------------------------------------------- #
    def load_optical(self):
        path, _ = filedialogs.get_open_file_name(
            self, "Load optical / slide image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        self._load_optical_path(path, reset_align=True, turn_on=True)

    def _load_optical_path(self, path, reset_align=True, turn_on=True):
        from PIL import Image
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
        except Exception as e:  # noqa: BLE001 — bad file shouldn't crash the app
            self.statusBar().showMessage(f"Couldn't load optical image: {e}")
            return False
        if reset_align:
            self._optical_align = self._default_optical_align()
        self._set_optical_image(img, path)
        if turn_on:
            self._optical_on = True
        self._sync_optical_controls()
        self._refresh_optical()
        self._mark_dirty()
        self.statusBar().showMessage(
            f"Loaded optical image {img.shape[1]}×{img.shape[0]}. Toggle 'Show optical', "
            "set spectra opacity, and align it (File → Image setup…).")
        return True

    def _remove_optical(self):
        if self._optical is None:
            return
        self.record_undo("Remove optical image", ("optical",))
        self._optical = None
        self._optical_path = None
        self._optical_on = False
        self._optical_align = self._default_optical_align()
        self._sync_optical_controls()
        self._refresh_optical()
        self._mark_dirty()
        self.statusBar().showMessage("Optical image removed.")

    def _clear_optical(self):
        """Drop optical state for a freshly loaded sample (a session restore re-applies it
        afterwards). Unlike :meth:`_remove_optical` this doesn't mark the session dirty."""
        self._optical = None
        self._optical_path = None
        self._optical_on = False
        self._optical_align = self._default_optical_align()
        self._sync_optical_controls()
        self._refresh_optical()

    # ----- session persistence -------------------------------------------- #
    def _optical_session_dict(self):
        """Serialize the optical layer (path + alignment + display), or None when no image
        is loaded. The pixels aren't stored — only the path, like the dataset itself."""
        if not self._optical_path:
            return None
        a = self._optical_align
        return {"path": self._optical_path, "on": bool(self._optical_on),
                "alpha": float(self._optical_spectra_alpha),
                "tx": float(a["tx"]), "ty": float(a["ty"]), "scale": float(a["scale"]),
                "angle": float(a["angle"]), "flipx": bool(a["flipx"]),
                "flipy": bool(a["flipy"])}

    def _apply_optical_session(self, data):
        """Restore the optical layer from a session dict (called inside _apply_session)."""
        self._clear_optical()
        opt = (data or {}).get("optical")
        if not opt:
            return
        self._optical_align = {
            "tx": float(opt.get("tx", 0.0)), "ty": float(opt.get("ty", 0.0)),
            "scale": float(opt.get("scale", 1.0)), "angle": float(opt.get("angle", 0.0)),
            "flipx": bool(opt.get("flipx", False)), "flipy": bool(opt.get("flipy", False))}
        self._optical_spectra_alpha = float(opt.get("alpha", 0.5))
        path = opt.get("path")
        if path and os.path.exists(path):
            from PIL import Image
            try:
                img = np.asarray(Image.open(path).convert("RGB"))
                self._set_optical_image(img, path)
                self._optical_on = bool(opt.get("on", True))
            except Exception:  # noqa: BLE001
                pass
        elif path:
            self.statusBar().showMessage(
                f"Optical image not found at {path} — re-load it via File → Image setup….")
        self._sync_optical_controls()
        self._refresh_optical()
