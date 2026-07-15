"""Landmark-based optical ↔ MSI registration (plan 05).

Estimate an affine / similarity / rigid transform from landmark point pairs (the same
feature's pixel position on the photo and on the ion image), then pre-fill the optical
alignment controls so the user can verify it visually against the live overlay. Drives
:mod:`smile_msi.registration`.

The interactive canvas-click picking of landmarks is a planned enhancement; this dialog
takes typed coordinates so the engine is usable today and the estimate is verifiable.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from .. import profiles, registration
from .common import note, section_title

_MIN_PAIRS = {"similarity": 2, "rigid": 2, "affine": 3, "piecewise-affine": 3}


class LandmarkDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Estimate alignment from landmarks")
        self.resize(480, 470)
        self.result = None

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(section_title("Landmark pairs"))
        root.addWidget(note("Enter matching points: the same feature's pixel position on the "
                            "photo and on the ion image. Estimate fits a transform and pre-fills "
                            "the alignment controls — verify it against the overlay and fine-tune."))

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Photo X", "Photo Y", "Ion X", "Ion Y"])
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)
        for _ in range(3):
            self._add_row()

        rowb = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Add row")
        b_add.clicked.connect(self._add_row)
        b_del = QtWidgets.QPushButton("Remove row")
        b_del.clicked.connect(self._del_row)
        rowb.addWidget(b_add)
        rowb.addWidget(b_del)
        rowb.addStretch(1)
        root.addLayout(rowb)

        self.status = QtWidgets.QLabel("")
        root.addWidget(self.status)

        rowx = QtWidgets.QHBoxLayout()
        b_est = QtWidgets.QPushButton("Estimate && apply")
        b_est.setObjectName("primaryAction")
        b_est.clicked.connect(self._estimate)
        rowx.addStretch(1)
        rowx.addWidget(b_est)
        root.addLayout(rowx)

    def _add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(4):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))

    def _del_row(self):
        r = self.table.currentRow()
        if r < 0:
            r = self.table.rowCount() - 1
        if r >= 0:
            self.table.removeRow(r)

    def _pairs(self):
        src, dst = [], []
        for r in range(self.table.rowCount()):
            try:
                px = float(self.table.item(r, 0).text())
                py = float(self.table.item(r, 1).text())
                ix = float(self.table.item(r, 2).text())
                iy = float(self.table.item(r, 3).text())
            except (TypeError, ValueError, AttributeError):
                continue
            src.append([px, py])
            dst.append([ix, iy])
        return np.asarray(src, dtype=float), np.asarray(dst, dtype=float)

    def _estimate(self):
        src, dst = self._pairs()
        kind = profiles.active().get("registration_kind", "affine")
        need = _MIN_PAIRS.get(kind, 3)
        if len(src) < need:
            self.status.setText(f"Need at least {need} landmark pairs for a {kind} transform.")
            return
        try:
            res = registration.estimate_landmark_transform(src, dst, kind=kind)
        except Exception as e:  # noqa: BLE001 — surface a degenerate landmark set, don't crash
            self.status.setText(f"Estimate failed: {e}")
            return
        self.result = res
        coeffs = registration.to_qtransform_coeffs(res)
        if hasattr(self.main, "_apply_optical_coeffs"):
            self.main._apply_optical_coeffs(coeffs)
        if hasattr(self.main, "record_step"):
            self.main.record_step("registration", label="Optical landmark registration",
                                  params={"kind": kind, "n_landmarks": res.n_landmarks,
                                          "rmse": float(res.rmse)})
        self.status.setText(f"Estimated {kind} transform — RMSE {res.rmse:.2f} px "
                            f"({res.n_landmarks} landmarks). Verify against the overlay.")
