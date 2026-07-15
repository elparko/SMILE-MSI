"""Acquisition metadata & reporting (File ▸ Acquisition metadata & reporting…).

Edit the acquisition/reporting metadata that travels with a standards-compliant imzML
export, validate it against the MIAMSIE minimum-reporting checklist, and export. Drives
:mod:`smile_msi.standards`; the metadata persists in the session (``acquisition_meta``).
"""
from __future__ import annotations

from PySide6 import QtWidgets

from .. import profiles, standards
from .common import (ACCENT, button, icon_button, note, primary_button, section_title,
                     NoScrollComboBox)

_TEXT_FIELDS = [
    ("organism", "Organism"), ("tissue", "Tissue"), ("condition", "Condition / disease"),
    ("sample_prep", "Sample preparation"), ("storage", "Storage"),
    ("matrix", "MALDI matrix"), ("matrix_application", "Matrix application"),
    ("instrument", "Instrument"), ("mass_analyzer", "Mass analyzer"),
    ("mass_resolution", "Mass resolution"), ("calibration", "Calibration"),
]
_NUM_FIELDS = [
    ("section_thickness_um", "Section thickness (µm)"),
    ("laser_spot_um", "Laser spot / pixel size (µm)"),
]
_LEVEL_COLORS = {"pass": ACCENT, "warn": "#C9A227", "fail": "#C0524A"}


class ReportingDialog(QtWidgets.QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Acquisition metadata & reporting")
        self.resize(560, 660)
        self._fields: dict[str, QtWidgets.QWidget] = {}

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(section_title("Acquisition metadata"))
        root.addWidget(note("Recorded into a standards-compliant imzML export (cvParams) and "
                            "checked against the MIAMSIE minimum-reporting guideline — for "
                            "deposition / METASPACE annotation."))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(inner)
        for key, label in _TEXT_FIELDS:
            w = QtWidgets.QLineEdit()
            self._fields[key] = w
            form.addRow(label, w)
        ino = NoScrollComboBox()
        ino.addItems(["MALDI", "DESI", "SIMS", "other"])
        self._fields["ionization"] = ino
        form.addRow("Ionization", ino)
        for key, label in _NUM_FIELDS:
            w = QtWidgets.QLineEdit()
            w.setPlaceholderText("number")
            self._fields[key] = w
            form.addRow(label, w)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        self.status = QtWidgets.QLabel("")
        root.addWidget(self.status)
        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(150)
        root.addWidget(self.report)

        row = QtWidgets.QHBoxLayout()
        b_val = button("Validate (MIAMSIE)", self._validate)
        b_save = primary_button("Save", self._save, action="save")
        b_exp = icon_button("export", "Save && export imzML…", self._export)
        row.addWidget(b_val)
        row.addStretch(1)
        row.addWidget(b_save)
        row.addWidget(b_exp)
        root.addLayout(row)

    # ---- data <-> widgets ------------------------------------------------ #
    def load_from_main(self):
        """Prefill from the open session's saved metadata, or seed from the active profile."""
        meta = dict(getattr(self.main, "_acquisition_meta", None) or {})
        if not meta:
            prof = profiles.active()
            meta = {"instrument": prof.get("report_instrument", ""),
                    "mass_analyzer": prof.get("report_mass_analyzer", ""),
                    "matrix": prof.get("report_matrix", ""),
                    "ionization": prof.get("report_ionization", "MALDI")}
        for key, w in self._fields.items():
            v = meta.get(key, "")
            if isinstance(w, QtWidgets.QComboBox):
                i = w.findText(str(v or "MALDI"))
                w.setCurrentIndex(i if i >= 0 else 0)
            else:
                w.setText("" if v in (None, "") else str(v))
        self.status.setText("")
        self.report.clear()

    def _collect(self) -> standards.AcquisitionMeta:
        d: dict = {}
        for key, _ in _TEXT_FIELDS:
            d[key] = self._fields[key].text().strip()
        d["ionization"] = self._fields["ionization"].currentText()
        for key, _ in _NUM_FIELDS:
            t = self._fields[key].text().strip()
            try:
                d[key] = float(t) if t else None
            except ValueError:
                d[key] = None
        return standards.AcquisitionMeta.from_dict(d)

    # ---- actions --------------------------------------------------------- #
    def _save(self):
        self.main._acquisition_meta = self._collect().to_dict()
        if hasattr(self.main, "_mark_dirty"):
            self.main._mark_dirty()
        self.main.statusBar().showMessage("Acquisition metadata saved.")

    def _validate(self):
        res = standards.validate_reporting(getattr(self.main, "prov", None), self._collect())
        self.status.setText(
            f"MIAMSIE: {res.level.upper()}  ({res.n_ok} ok · {res.n_warn} warn · "
            f"{res.n_missing_required} missing required)")
        self.status.setStyleSheet(
            f"color:{_LEVEL_COLORS.get(res.level, '#888')};font-weight:bold;")
        self.report.setPlainText(res.to_markdown())

    def _export(self):
        self._save()
        if getattr(self.main, "ds", None) is None:
            self.main.statusBar().showMessage("Load a dataset first.")
            return
        self.main.export_imzml()
