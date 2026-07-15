"""Export the feature list with every analysis's per-feature results joined on as columns.

The feature list is the spine of the exported CSV — one row per feature, exactly the rows on
screen. Each analysis you have run adds a namespaced block of columns beside it, so a list can
carry nothing but its annotation, or its annotation plus a region comparison, a PCA and a
co-localization, and land in R / Excel / pandas ready to analyse without a single join.

The picker greys out the analyses that *cannot* attach (a lipid-class comparison is keyed by
class, not m/z; a per-ion segmentation describes one ion) and says why, so an absent column
block never reads as a failed analysis. The reshaping and joining are
:mod:`smile_msi.featuretable`'s job; this is only the picker around them.
"""
from __future__ import annotations

import traceback

from PySide6 import QtWidgets

from . import filedialogs
from .common import CheckList, NoScrollDoubleSpinBox, button, note
from .. import featuretable


class FeatureExportDialog(QtWidgets.QDialog):
    """Pick which analyses ride along, then write one CSV."""

    def __init__(self, parent, base, attachments, *, default_name="feature_list"):
        super().__init__(parent)
        self.setWindowTitle("Export feature list")
        self.setModal(True)
        self.resize(620, 520)
        self._base = base
        self._attachments = list(attachments)
        self._default_name = default_name
        self.path = None
        self.notes = []

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(note(
            f"{len(base)} features × {len(base.columns)} columns. Tick the analyses whose "
            "per-feature results should ride along as extra columns — each is prefixed with the "
            "name shown, so nothing collides."))

        usable = [a for a in self._attachments if a.usable]
        self._checks = CheckList(
            [(a.key, self._label(a), self._group(a), a.usable) for a in self._attachments],
            checked=[a.key for a in usable],           # 'carry everything' is the default
            noun="analysis")
        self._checks.changed.connect(self._refresh_summary)
        v.addWidget(self._checks, 1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Match m/z within"))
        self._ppm = NoScrollDoubleSpinBox()      # the join tolerance: a hover-scroll must not move it
        self._ppm.setRange(0.1, 100.0)
        self._ppm.setDecimals(1)
        self._ppm.setSingleStep(0.5)
        self._ppm.setValue(featuretable.DEFAULT_JOIN_PPM)
        self._ppm.setSuffix(" ppm")
        self._ppm.setToolTip(
            "How close an analysis's m/z must be to a feature's to count as the same ion. "
            "Widen this if an analysis ran before the feature list was recalibrated.")
        row.addWidget(self._ppm)
        row.addStretch(1)
        self._summary = QtWidgets.QLabel("")
        row.addWidget(self._summary)
        v.addLayout(row)

        foot = QtWidgets.QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(button("Cancel", self.reject))
        foot.addWidget(button("Export…", self._export, primary=True,
                              tooltip="Choose a file and write the joined table"))
        v.addLayout(foot)
        self._refresh_summary()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _label(a):
        return f"{a.title}   [{a.prefix}.*]" if a.usable else a.title

    @staticmethod
    def _group(a):
        if not a.usable:
            return a.reason or "no per-feature result"
        bits = [f"{len(a.columns)} columns"]
        if a.reason:
            bits.append(a.reason)
        elif a.subtitle:
            bits.append(a.subtitle)
        return " · ".join(bits)

    def _selected(self):
        keys = self._checks.checked_keys()
        return [a for a in self._attachments if a.usable and a.key in keys]

    def _refresh_summary(self):
        sel = self._selected()
        extra = sum(len(a.columns) for a in sel)
        n = len(sel)
        self._summary.setText(
            f"{len(self._base)} features × {len(self._base.columns) + extra} columns "
            f"({n} {'analysis' if n == 1 else 'analyses'}, +{extra})")

    # ------------------------------------------------------------------ #
    def _export(self):
        try:
            df, self.notes = featuretable.join_by_mz(self._base, self._selected(),
                                                     tol_ppm=float(self._ppm.value()))
        except Exception as exc:                       # noqa: BLE001 — a bad join must not kill the app
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not build the table:\n{exc}")
            return
        path, _ = filedialogs.get_save_file_name(self, "Save feature list",
                                                 f"{self._default_name}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        except Exception as exc:                       # noqa: BLE001
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not write {path}:\n{exc}")
            return
        self.path, self.n_features, self.n_columns = path, len(df), len(df.columns)
        self.accept()
