"""Absolute quantification (Data ▸ Quantification (calibration)…).

Fit an on-tissue calibration curve from concentration-tagged standard regions and
back-calculate absolute concentrations. Drives :mod:`smile_msi.quantify`. Out-of-range
pixels are flagged, never clamped; the fitted model persists in the session.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .. import profiles, quantify
from .common import (note, section_title, check_table_bar, filter_combo,
                     NoScrollComboBox)


class QuantifyDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Quantification — on-tissue calibration")
        self.resize(620, 660)
        self._model = None

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(section_title("Calibration standards"))
        root.addWidget(note("Tag named regions as calibration standards with a known "
                            "concentration, pick the analyte ion, and fit a curve. Out-of-range "
                            "pixels are flagged, never clamped."))

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Use", "Region", "Concentration"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self._bulk = check_table_bar(self.table, 0, "region")
        root.addWidget(self._bulk)
        root.addWidget(self.table, 1)

        form = QtWidgets.QFormLayout()
        self.analyte = filter_combo(
            tooltip="The ion being quantified. Type an m/z or a lipid name to filter — the "
                    "working set can run to hundreds of ions.")
        self.is_combo = filter_combo(
            tooltip="An internal-standard ion to normalise against, or '(none)'. Type an m/z "
                    "or a lipid name to filter.")
        form.addRow("Analyte m/z", self.analyte)
        form.addRow("Internal standard m/z", self.is_combo)
        prof = profiles.active()
        self.units = QtWidgets.QLineEdit(str(prof.get("quant_units", "a.u.")))
        form.addRow("Units", self.units)
        self.weighting = NoScrollComboBox()
        self.weighting.addItems(["none", "1/x", "1/x2"])
        self.weighting.setCurrentText(str(prof.get("quant_weighting", "none")))
        form.addRow("Curve weighting", self.weighting)
        self.through0 = QtWidgets.QCheckBox("Force through origin")
        self.through0.setChecked(str(prof.get("quant_through_origin", "no")) == "yes")
        form.addRow("", self.through0)
        root.addLayout(form)

        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(160)
        root.addWidget(self.report)

        row = QtWidgets.QHBoxLayout()
        b_fit = QtWidgets.QPushButton("Fit calibration")
        b_fit.setObjectName("primaryAction")
        b_fit.clicked.connect(self._fit)
        b_apply = QtWidgets.QPushButton("Apply to map")
        b_apply.clicked.connect(self._apply)
        row.addWidget(b_fit)
        row.addStretch(1)
        row.addWidget(b_apply)
        root.addLayout(row)

    # ---- populate from the open sample ----------------------------------- #
    def load_from_main(self):
        regions = list(getattr(self.main, "regions", []) or [])
        self.table.setRowCount(len(regions))
        for i, r in enumerate(regions):
            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
            chk.setCheckState(QtCore.Qt.Unchecked)
            self.table.setItem(i, 0, chk)
            nm = QtWidgets.QTableWidgetItem(str(r.get("name", f"region {i}")))
            nm.setFlags(nm.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(i, 1, nm)
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(""))
        self._bulk.refresh_count()                 # the bar was built over an empty table
        # Label each ion "m/z · lipid" and carry the m/z as item data: the label is what the
        # completer matches (so "PC 34" finds it), the data is what the fit consumes.
        ions = []
        for pk in (getattr(self.main, "peaks", []) or []):
            mz = float(pk["mz"])
            lab = (self.main._clean_label(mz) if hasattr(self.main, "_clean_label") else "") or ""
            ions.append((f"{mz:.4f}" + (f"   ·  {lab}" if lab else ""), mz))
        self.analyte.clear()
        for label, mz in ions:
            self.analyte.addItem(label, mz)
        self.is_combo.clear()
        self.is_combo.addItem("(none)", None)
        for label, mz in ions:
            self.is_combo.addItem(label, mz)
        self._model = None
        self.report.clear()

    @staticmethod
    def _combo_mz(combo):
        """The m/z the box is *showing*, or None.

        Resolved from the text, never from ``currentData()`` alone: these combos are editable
        (that is what makes them filterable), and typing over one leaves ``currentIndex()``
        parked on the previously chosen row. Trusting the data role there would quantify the
        old ion while the box displayed the new one — wrong, and silently so. ``findText``
        first so an exact pick keeps full m/z precision rather than the 4dp of its label;
        otherwise fall back to the leading number of whatever was typed."""
        text = (combo.currentText() or "").strip()
        i = combo.findText(text)
        if i >= 0:
            data = combo.itemData(i)
            return None if data is None else float(data)    # '(none)' carries data None
        try:
            return float(text.split()[0])
        except (IndexError, ValueError):
            return None

    def _levels(self):
        levels = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() != QtCore.Qt.Checked:
                continue
            try:
                conc = float(self.table.item(i, 2).text())
            except (TypeError, ValueError):
                continue
            r = self.main.regions[i]
            mask = self.main._region_pixel_mask(r)
            if mask is None:
                continue
            levels.append(quantify.CalLevel(str(r.get("name", "")), conc, mask))
        return levels

    # ---- actions --------------------------------------------------------- #
    def _fit(self):
        if getattr(self.main, "ds", None) is None:
            self.main.statusBar().showMessage("Load a dataset first.")
            return
        levels = self._levels()
        if len(levels) < 2:
            self.report.setPlainText("Select at least two standard regions with concentrations.")
            return
        analyte = self._combo_mz(self.analyte)
        if analyte is None:
            self.report.setPlainText("Pick an analyte ion.")
            return
        is_mz = self._combo_mz(self.is_combo)
        self._model = quantify.fit_calibration(
            self.main.ds, levels, analyte, is_mz=is_mz,
            units=self.units.text().strip() or "a.u.",
            weighting=self.weighting.currentText(),
            through_origin=self.through0.isChecked())
        rep = quantify.calibration_report(self._model)
        self.report.setPlainText("Calibration model\n"
                                 + "\n".join(f"{k}: {v}" for k, v in rep.items()))

    def _apply(self):
        if self._model is None:
            self.report.append("\nFit a calibration first.")
            return
        import numpy as np
        conc, in_range = quantify.apply_calibration(self.main.ds, self._model)
        n_out = int((~np.asarray(in_range)).sum())
        med = float(np.nanmedian(conc))
        models = list(getattr(self.main, "_calibration_models", []) or [])
        models.append(self._model.to_dict())
        self.main._calibration_models = models
        if hasattr(self.main, "_mark_dirty"):
            self.main._mark_dirty()
        if hasattr(self.main, "record_step"):
            self.main.record_step("quantification", label="Absolute quantification",
                                  params=quantify.calibration_report(self._model))
        self.report.append(f"\nApplied: median {med:.4g} {self._model.units}; "
                           f"{n_out} pixel(s) out of calibrated range (flagged).")
        self.main.statusBar().showMessage("Calibration applied; model saved to session.")
