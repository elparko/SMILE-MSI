"""Compartments dialog — endoneurium / perineurium / epineurium regions in one go.

The nerve-compartment recipe as a single small dialog on the Draw-ROI bar: threshold a signal
(the active feature, a lipid class, or a sum of features such as the sulfatides) for the
**endoneurium**, grow a collar of N px outside it for the **perineurium**, and take every other
acquired pixel for the **epineurium**. The three masks preview as one RGB overlay; *Create* adds
them as three regions tagged as their own groups, so the group pickers (Multi-group features,
Discriminating features, Region comparison) list them at once.

Everything here calls the same engine helpers the bar and the script console use
(:func:`smile_msi.spatial.threshold_mask`, :func:`ring_mask`, :func:`invert_mask`).
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import spatial
from .common import (REGION_PALETTE, NoScrollComboBox, NoScrollDoubleSpinBox, NoScrollSpinBox,
                     button, hex_to_rgba, note, primary_button, section_title, tool_button)

DEFAULT_NAMES = ("endo", "peri", "epi")
DEFAULT_CUT = 60          # percentile of signal pixels
DEFAULT_COLLAR_PX = 6     # 30 µm on a 5 µm slide
PREVIEW_PX = 320


class CompartmentsDialog(QtWidgets.QDialog):
    """Non-modal: pick the signal, the cut and the collar width; preview; Create."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Compartments by construction")
        self.setModal(False)
        self._sum_mzs = []
        self._cache = None
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)
        form = QtWidgets.QVBoxLayout()
        form.setSpacing(6)

        form.addWidget(section_title("Endoneurium = signal ≥ cut"))
        self.signal = NoScrollComboBox()
        self.signal.setMinimumWidth(240)
        self.signal.setToolTip("The active feature, a lipid class composite, or a sum of "
                               "features you pick (e.g. the sulfatide species)")
        self.signal.currentIndexChanged.connect(self._signal_changed)
        self.pick_btn = tool_button(name="settings", tooltip="Choose the features to sum…",
                                    slot=self._pick_sum)
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
        row.addWidget(self.signal, 1); row.addWidget(self.pick_btn)
        form.addLayout(row)
        self.cut = NoScrollDoubleSpinBox()
        self.cut.setRange(0, 100)
        self.cut.setDecimals(0)
        self.cut.setValue(DEFAULT_CUT)
        self.cut.setPrefix("cut at percentile ")
        self.cut.setToolTip("Keep pixels at or above this percentile of the signal pixels "
                            "(60 = the brightest 40 %)")
        self.cut.valueChanged.connect(self._changed)
        form.addWidget(self.cut)
        self.fill = QtWidgets.QCheckBox("Fill holes")
        self.fill.setChecked(True)
        self.fill.toggled.connect(self._changed)
        form.addWidget(self.fill)
        self.min_px = NoScrollSpinBox()
        self.min_px.setRange(0, 100000)
        self.min_px.setPrefix("drop islands < ")
        self.min_px.setSuffix(" px")
        self.min_px.valueChanged.connect(self._changed)
        form.addWidget(self.min_px)

        form.addWidget(section_title("Perineurium = collar outside it"))
        self.collar = NoScrollSpinBox()
        self.collar.setRange(1, 500)
        self.collar.setValue(DEFAULT_COLLAR_PX)
        self.collar.setPrefix("width ")
        self.collar.setSuffix(" px")
        self.collar.valueChanged.connect(self._changed)
        self.collar_um = QtWidgets.QLabel("")
        row = QtWidgets.QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        row.addWidget(self.collar); row.addWidget(self.collar_um, 1)
        form.addLayout(row)

        form.addWidget(section_title("Epineurium = everything else"))
        form.addWidget(note("Every other acquired pixel (off-tissue included — trim it with "
                            "a region crop, or threshold the TIC first)."))

        form.addWidget(section_title("Region names"))
        self.names = []
        for default in DEFAULT_NAMES:
            e = QtWidgets.QLineEdit(default)
            e.textChanged.connect(self._changed)
            self.names.append(e)
            form.addWidget(e)
        self.counts = note("")
        form.addWidget(self.counts)
        form.addStretch(1)
        foot = QtWidgets.QHBoxLayout()
        self.b_create = primary_button("Create regions", self._create,
                                       tooltip="Add the three regions (tagged as groups)")
        foot.addWidget(self.b_create)
        foot.addStretch(1)
        foot.addWidget(button("Close", self.hide))
        form.addLayout(foot)
        outer.addLayout(form)

        self.preview = QtWidgets.QLabel()
        self.preview.setMinimumSize(PREVIEW_PX, PREVIEW_PX)
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background: #000;")
        outer.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    def refresh(self):
        """Re-list the signals (the class map may have changed) and re-preview."""
        self.win._fill_signal_combo(self.signal, self.win._signal_options(self._sum_mzs))
        px = getattr(self.win.ds, "pixel_size_um", None) if self.win.ds is not None else None
        self.collar_um.setText(f"= {self.collar.value() * float(px):g} µm" if px
                               else "(no pixel size)")
        self._cache = None
        self._changed()

    def _signal_changed(self, *_):
        self._cache = None
        kind = self.signal.currentData() or ("active",)
        if kind[0] == "sum" and not self._sum_mzs:
            self._pick_sum()
            return
        self._changed()

    def _pick_sum(self):
        chosen = self.win._pick_feature_sum(self._sum_mzs)
        if chosen is None:
            return
        self._sum_mzs = chosen
        self._cache = None
        self.win._fill_signal_combo(self.signal, self.win._signal_options(self._sum_mzs))
        self._changed()

    def _signal_vector(self):
        kind = self.signal.currentData() or ("active",)
        vec, label, key = self.win._signal_vector_for(kind, self._sum_mzs, compute=False)
        if key is None:
            return None, label
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1], label
        vec, label, key = self.win._signal_vector_for(kind, self._sum_mzs)
        self._cache = (key, vec)
        return vec, label

    def colors(self):
        base = len(self.win.regions)
        return [REGION_PALETTE[(base + k) % len(REGION_PALETTE)] for k in range(3)]

    def masks(self):
        """``(endo, peri, epi)`` per-pixel masks, or ``None`` when the signal is unset or
        the cut keeps no pixels."""
        ds = self.win.ds
        if ds is None:
            return None
        vec, _ = self._signal_vector()
        if vec is None:
            return None
        endo = spatial.threshold_mask(ds, vec, float(self.cut.value()), percentile=True,
                                      fill_holes=self.fill.isChecked(),
                                      min_pixels=int(self.min_px.value()))
        if not endo.any():
            return None
        peri = spatial.ring_mask(ds, endo, float(self.collar.value()), mode="outer")
        epi = spatial.invert_mask(ds, endo | peri)
        return endo, peri, epi

    def _changed(self, *_):
        px = getattr(self.win.ds, "pixel_size_um", None) if self.win.ds is not None else None
        self.collar_um.setText(f"= {self.collar.value() * float(px):g} µm" if px
                               else "(no pixel size)")
        trio = self.masks()
        names = [e.text().strip() for e in self.names]
        ok = trio is not None and all(names) and len(set(names)) == 3
        self.b_create.setEnabled(ok)
        if trio is None:
            self.counts.setText("Select a feature (or a class / sum) — the cut keeps no pixels yet.")
            self.preview.clear()
            return
        n = [int(m.sum()) for m in trio]
        self.counts.setText("  ·  ".join(f"{nm or '?'} {c:,} px" for nm, c in zip(names, n))
                            + ("" if len(set(names)) == 3 else "  —  names must differ"))
        self._render(trio)

    def _render(self, trio):
        ds = self.win.ds
        h, w = ds.height, ds.width
        rgba = np.zeros((h, w, 4), dtype=np.ubyte)
        for mask, hexc in zip(trio, self.colors()):
            on = ds.to_image(mask.astype(float), fill=0.0) > 0.5
            r, g, b, _ = hex_to_rgba(hexc)
            rgba[on, 0], rgba[on, 1], rgba[on, 2], rgba[on, 3] = r, g, b, 255
        img = QtGui.QImage(rgba.tobytes(), w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        pm = QtGui.QPixmap.fromImage(img).scaled(PREVIEW_PX, PREVIEW_PX, QtCore.Qt.KeepAspectRatio,
                                                 QtCore.Qt.FastTransformation)
        self.preview.setPixmap(pm)

    # ------------------------------------------------------------------ #
    def _create(self):
        """Three regions, each tagged as its own group, in one undo step. Returns the names."""
        win = self.win
        trio = self.masks()
        if trio is None:
            win.statusBar().showMessage("Nothing to create — the cut keeps no pixels.")
            return []
        names = [e.text().strip() for e in self.names]
        colors = self.colors()
        win.record_undo("compartments", domains=("regions",))
        made = []
        for name, mask, color in zip(names, trio, colors):
            i = win._new_region(name, color=color, mask=mask, refresh=False)
            win.regions[i]["group"] = win.regions[i]["name"]
            made.append(win.regions[i]["name"])
        win._refresh_regions_all(made[0])
        _, label = self._signal_vector()
        if win.prov is not None:
            win.prov.step("region_derive", source="compartments", op="threshold+outer+invert",
                          signal=label, cut=float(self.cut.value()),
                          width_px=float(self.collar.value()), regions=list(made))
        win._mark_dirty()
        counts = [int(m.sum()) for m in trio]
        win.statusBar().showMessage(
            "Compartments created: " + ", ".join(f"{n} ({c:,} px)" for n, c in zip(made, counts))
            + " — tagged as groups for Multi-group / Discriminating / Region comparison.")
        self.hide()
        return made
