"""Spatial multi-omics co-mapping (Data ▸ Spatial multi-omics…).

Import a second-modality grid measured on the same tissue (spatial transcriptomics /
proteomics) and run metabolite↔gene spatial correlation against the open MSI slide.
Drives :mod:`smile_msi.comap`.

The dialog imports a modality grid via :func:`smile_msi.comap.load_modality` (a generic
CSV / Space Ranger directory needs only core deps; an ``.h5ad`` file needs the optional
``comap`` extra and surfaces a clear ImportError if it is missing). With
``transform=None`` it assumes the modality shares the MSI pixel grid (same-section,
co-acquired) — a note states this; plan-05 registration plugs a real transform in later.

Pick the MSI metabolite ions (m/z peaks) and a modality feature subset, then run
:func:`smile_msi.comap.cross_modality_correlation` and read the strongest
metabolite↔feature pairs off :meth:`smile_msi.comap.CrossCorr.top_pairs`. The correlation
metric comes from the active analysis profile (``comap_corr_method``). Each run records a
``cross_modality_correlation`` provenance step.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .. import profiles
from . import filedialogs
from .common import (CheckList, install_table_export, note, section_title,
                     NoScrollComboBox, NoScrollSpinBox)

_FILE_FILTER = (
    "Modality grids (*.csv *.tsv *.txt *.h5ad);;"
    "Generic table (*.csv *.tsv *.txt);;"
    "AnnData (*.h5ad);;All files (*)"
)


class CoMapDialog(QtWidgets.QDialog):
    """Cross-modality co-mapping against the open MSI slide."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Spatial multi-omics — co-mapping")
        self.resize(680, 720)
        self.grid = None          # smile_msi.comap.ModalityGrid once imported
        self._result = None       # smile_msi.comap.CrossCorr from the last run

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(section_title("Second modality"))
        root.addWidget(note(
            "Import a second assay measured on the same tissue (spatial transcriptomics / "
            "proteomics). A generic CSV/TSV with x/y columns or a Space Ranger directory "
            "needs no extra; an .h5ad (AnnData) file needs the 'comap' extra "
            "(uv sync --extra comap)."))

        # ---- import row -------------------------------------------------- #
        imp = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("No modality grid imported")
        b_browse = QtWidgets.QPushButton("Import grid…")
        b_browse.clicked.connect(self._browse)
        self.kind = NoScrollComboBox()
        self.kind.addItems(["transcriptomics", "proteomics", "generic"])
        imp.addWidget(QtWidgets.QLabel("Kind"))
        imp.addWidget(self.kind)
        imp.addWidget(self.path_edit, 1)
        imp.addWidget(b_browse)
        root.addLayout(imp)

        self.grid_status = QtWidgets.QLabel("")
        self.grid_status.setWordWrap(True)
        root.addWidget(self.grid_status)

        root.addWidget(note(
            "Same-grid assumption: the modality is treated as sharing the MSI pixel grid "
            "(transform = none), valid for same-section co-acquired data. Serial-section "
            "registration plugs a transform in later."))

        # ---- features --------------------------------------------------- #
        root.addWidget(section_title("Modality features"))
        root.addWidget(note(
            "Tick the genes / proteins / channels to correlate against every MSI ion. "
            "Leave all unticked to use every modality feature."))
        self.feat_list = CheckList(noun="feature")
        self.feat_list.setMaximumHeight(240)
        root.addWidget(self.feat_list)

        # ---- options ---------------------------------------------------- #
        form = QtWidgets.QFormLayout()
        self.region = NoScrollComboBox()
        form.addRow("Restrict to region", self.region)
        self.method_label = QtWidgets.QLabel("")
        form.addRow("Correlation (from profile)", self.method_label)
        self.topk = NoScrollSpinBox()
        self.topk.setRange(1, 500)
        self.topk.setValue(20)
        form.addRow("Top pairs to show", self.topk)
        root.addLayout(form)

        # ---- results ---------------------------------------------------- #
        root.addWidget(section_title("Top metabolite ↔ feature pairs"))
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["MSI m/z", "Modality feature", "Correlation"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        install_table_export(self.table, self, stem="comap_pairs",
                             title="Export metabolite ↔ feature pairs")
        root.addWidget(self.table, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        row = QtWidgets.QHBoxLayout()
        b_run = QtWidgets.QPushButton("Run correlation")
        b_run.setObjectName("primaryAction")
        b_run.clicked.connect(self._run)
        row.addStretch(1)
        row.addWidget(b_run)
        root.addLayout(row)

    # ---- profile / population ------------------------------------------- #
    def _corr_method(self) -> str:
        try:
            return str(profiles.active_params().get("comap_corr_method", "spearman"))
        except Exception:  # noqa: BLE001 — profile store optional in tests
            return "spearman"

    def _min_pixels(self) -> int:
        try:
            return int(profiles.active_params().get("comap_min_pixels", 20))
        except Exception:  # noqa: BLE001
            return 20

    def load_from_main(self):
        """Refresh region choices and the profile-driven method label from the host."""
        self.method_label.setText(self._corr_method())
        self.region.clear()
        self.region.addItem("Detected foreground", None)
        for i, r in enumerate(getattr(self.main, "regions", []) or []):
            self.region.addItem(str(r.get("name", f"region {i}")), i)

    # ---- import --------------------------------------------------------- #
    def _browse(self):
        path, _ = filedialogs.get_open_file_name(
            self, "Import modality grid", filter=_FILE_FILTER)
        if path:
            self._import(path)

    def _import(self, path: str):
        from .. import comap

        try:
            grid = comap.load_modality(path, kind=self.kind.currentText())
        except ImportError as e:                 # missing .h5ad backend → clear message
            self.grid_status.setText(str(e))
            self.grid_status.setStyleSheet("color:#C0524A;")
            return
        except (ValueError, OSError) as e:       # malformed / unreadable file
            self.grid_status.setText(f"Could not import: {e}")
            self.grid_status.setStyleSheet("color:#C0524A;")
            return
        self.grid = grid
        self.path_edit.setText(path)
        self.grid_status.setStyleSheet("")
        self.grid_status.setText(
            f"Imported {grid.n_features} feature(s) at {grid.n_loc} location(s) "
            f"({grid.kind}).")
        self._populate_features(grid)

    def _populate_features(self, grid):
        self.feat_list.set_entries([(str(n), str(n)) for n in grid.feature_names])

    def _selected_features(self) -> list:
        return self.feat_list.checked_in_order()          # empty → all features

    def _region_mask(self):
        idx = self.region.currentData()
        if idx is None:
            return None
        regions = getattr(self.main, "regions", []) or []
        if idx >= len(regions) or not hasattr(self.main, "_region_pixel_mask"):
            return None
        return self.main._region_pixel_mask(regions[idx])

    # ---- run ------------------------------------------------------------ #
    def _run(self):
        if getattr(self.main, "ds", None) is None:
            self.main.statusBar().showMessage("Load a dataset first.")
            return
        raw_peaks = list(getattr(self.main, "peaks", []) or [])
        if not raw_peaks:
            self.status.setText("No MSI features — find peaks first.")
            return
        # spatial.feature_matrix wants plain m/z floats, not peak dicts.
        peaks = [float(p["mz"]) if isinstance(p, dict) else float(p) for p in raw_peaks]
        if self.grid is None:
            self.status.setText("Import a modality grid first.")
            return

        from .. import comap

        feat_sel = self._selected_features()
        grid = self.grid
        if feat_sel:
            cols = [grid.feature_names.index(n) for n in feat_sel]
            sub_vals = grid.dense()[:, cols]
            grid = comap.ModalityGrid(
                kind=grid.kind, feature_names=list(feat_sel),
                coords_um=grid.coords_um, values=sub_vals,
                spot_diameter_um=grid.spot_diameter_um, source=grid.source,
                transform=grid.transform, meta=dict(grid.meta))

        method = self._corr_method()
        # cross_modality_correlation supports pearson / spearman; cosine → pearson here.
        corr_method = method if method in ("pearson", "spearman") else "pearson"
        mask = self._region_mask()
        try:
            res = comap.cross_modality_correlation(
                self.main.ds, peaks, grid, transform=None,
                method=corr_method, mask=mask, min_pixels=self._min_pixels())
        except ValueError as e:                  # too few overlapping pixels, etc.
            self.status.setText(str(e))
            return
        self._result = res
        self._show_pairs(res)

        if hasattr(self.main, "record_step"):
            region_name = self.region.currentText()
            self.main.record_step(
                "cross_modality_correlation",
                label="MSI - modality correlation",
                params={
                    "modality_kind": grid.kind,
                    "modality_source": grid.source,
                    "n_features": grid.n_features,
                    "method": corr_method,
                    "tol_ppm": 50.0,
                    "norm": "tic",
                    "transform": "identity",
                    "n_pixels": int(res.n_pixels),
                    "region": region_name,
                })
        self.main.statusBar().showMessage(
            f"Cross-modality correlation over {res.n_pixels} pixel(s).")

    def _show_pairs(self, res):
        pairs = res.top_pairs(int(self.topk.value()))
        self.table.setRowCount(len(pairs))
        for r, (mz, feat, score) in enumerate(pairs):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{mz:.4f}"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(feat)))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{score:+.3f}"))
        self.status.setText(
            f"{res.method.capitalize()} correlation · {res.matrix.shape[0]} ion(s) × "
            f"{res.matrix.shape[1]} feature(s) over {res.n_pixels} shared pixel(s).")
