"""Single-cell / subcellular spatial metabolomics (Data ▸ Single-cell profiling…).

Segment cells on the co-registered optical / histology image, map every MSI pixel
onto the cells it overlaps through a registration transform, build an area-weighted
per-cell metabolite matrix, and embed + cluster the cells — the SpaceM paradigm.
Drives the pure engine :mod:`smile_msi.singlecell`.

The optical↔MSI transform is taken from a registration :class:`RegistrationResult`
(``main._registration.matrix``) when present, else derived from the manual fit
(``main._optical_align``) in the un-rotated base frame — exactly the ``fit`` portion
of :meth:`OpticalMixin._optical_transform` (photo-px → base-px, before the display
rotation), so the mapping stays rotation-invariant.

This is a self-contained :class:`QDialog` that takes the main window as its host and
only *reads* / *calls* what it needs (``main._optical``, ``main.ds``, ``main.regions``,
``main._optical_align``, optional ``main._registration``, optional
``main.record_step`` / ``main._run`` / ``main._region_pixel_mask`` /
``main._new_region``). The menu / opener wiring is the integrator's job.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import profiles, singlecell
from .common import (note, section_title, NoScrollComboBox, NoScrollDoubleSpinBox,
                     NoScrollSpinBox)


# Backends that need an optional uv extra (lazy-imported by the engine; we surface a
# clear hint in the combo so the requirement is visible before a run fails).
_BACKEND_HINTS = {
    "watershed": "",
    "cellpose": "needs `uv sync --extra cellpose`",
    "stardist": "needs `uv sync --extra stardist`",
}


def _align_matrix(align: dict, hp: int, wp: int, h0: int, w0: int) -> np.ndarray:
    """Derive the base-frame 3x3 affine (photo-px (col,row) -> MSI base-px (x,y)) from
    the manual ``_optical_align`` dict, mirroring the ``fit`` portion of
    :meth:`OpticalMixin._optical_transform` (the move/scale/rotate/flip evaluated in
    the un-rotated orientation-0 frame, **without** the trailing display rotation).

    Composition (column-vector / homogeneous): ``T1 @ R @ S @ T2`` where
    ``T1 = translate(w0/2 + tx, h0/2 + ty)``, ``R = rotate(angle)``,
    ``S = scale(sx, sy)``, ``T2 = translate(-wp/2, -hp/2)`` — exactly the Qt fit chain.
    """
    a = dict(align or {})
    scale = float(a.get("scale", 1.0))
    base_sx = (w0 / wp) if wp else 1.0
    base_sy = (h0 / hp) if hp else 1.0
    sx = base_sx * scale * (-1.0 if a.get("flipx") else 1.0)
    sy = base_sy * scale * (-1.0 if a.get("flipy") else 1.0)
    ang = np.deg2rad(float(a.get("angle", 0.0)))
    tx = float(a.get("tx", 0.0))
    ty = float(a.get("ty", 0.0))

    def _T(dx, dy):
        m = np.eye(3)
        m[0, 2] = dx
        m[1, 2] = dy
        return m

    T1 = _T(w0 / 2.0 + tx, h0 / 2.0 + ty)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    S = np.diag([sx, sy, 1.0])
    T2 = _T(-wp / 2.0, -hp / 2.0)
    return T1 @ R @ S @ T2


class SingleCellDialog(QtWidgets.QDialog):
    """Per-cell metabolite profiling from a co-registered optical image."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Single-cell profiling")
        self.resize(680, 720)

        # transient engine objects (held on the dialog; the integrator may also stash
        # them on the window / session)
        self._cells = None          # singlecell.CellSet
        self._cellmap = None        # singlecell.PixelCellMap
        self._cellmat = None        # singlecell.CellMatrix
        self._coords = None         # (n_cells, 2) embedding
        self._labels = None         # (n_cells,) cluster ids
        self._transform = None      # 3x3 base-frame affine used
        self._transform_source = "" # 'registration' | 'optical-align'

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(section_title("Single-cell spatial metabolomics"))
        root.addWidget(note(
            "Segment cells on the co-registered optical / histology image, map each MSI "
            "pixel onto the cells it overlaps (area-weighted, SpaceM), then build a per-cell "
            "metabolite matrix and cluster the cells. Per-cell profiles are area-weighted "
            "estimates — watch the cells-per-pixel readout when the MSI pixel is larger than "
            "a cell."))

        prof = profiles.active()

        # ---- segmentation controls -------------------------------------- #
        form = QtWidgets.QFormLayout()
        self.backend = NoScrollComboBox()
        for b in singlecell.SEGMENT_BACKENDS:
            hint = _BACKEND_HINTS.get(b, "")
            self.backend.addItem(f"{b}  ({hint})" if hint else b, b)
        self.backend.setCurrentIndex(
            max(0, self.backend.findData(str(prof.get("sc_backend", "watershed")))))
        form.addRow("Cell backend", self.backend)

        self.channel = NoScrollComboBox()
        self.channel.addItems(["auto", "gray", "r", "g", "b"])
        form.addRow("Stain channel", self.channel)

        self.min_diam = NoScrollDoubleSpinBox()
        self.min_diam.setRange(0.5, 1000.0)
        self.min_diam.setSingleStep(1.0)
        self.min_diam.setValue(float(prof.get("sc_min_diameter_um", 6.0)))
        self.min_diam.setSuffix(" µm")
        form.addRow("Min cell diameter", self.min_diam)

        self.invert = QtWidgets.QCheckBox("Dark structures on bright (e.g. H&E nuclei)")
        form.addRow("", self.invert)

        self.weighting = NoScrollComboBox()
        for w in singlecell.WEIGHTINGS:
            self.weighting.addItem(w, w)
        self.weighting.setCurrentIndex(
            max(0, self.weighting.findData(str(prof.get("sc_weighting", "area")))))
        form.addRow("Pixel→cell weighting", self.weighting)

        self.pixel_norm = NoScrollComboBox()
        self.pixel_norm.addItems(["tic", "none", "rms", "median"])
        self.pixel_norm.setCurrentText(str(prof.get("sc_pixel_norm", "tic")))
        form.addRow("Per-pixel norm", self.pixel_norm)

        self.cell_norm = NoScrollComboBox()
        self.cell_norm.addItems(["none", "tic", "median"])
        self.cell_norm.setCurrentText(str(prof.get("sc_cell_norm", "none")))
        form.addRow("Per-cell norm", self.cell_norm)

        self.n_clusters = NoScrollSpinBox()
        self.n_clusters.setRange(2, 40)
        self.n_clusters.setValue(8)
        form.addRow("Cell clusters (k)", self.n_clusters)
        root.addLayout(form)

        # ---- cell-cluster scatter --------------------------------------- #
        self.canvas = None
        self.fig = None
        self.ax = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(4.0, 3.0), tight_layout=True)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_off()
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setMinimumHeight(220)
            root.addWidget(self.canvas, 1)
        except Exception:  # noqa: BLE001 — matplotlib optional; degrade to a label
            self._scatter_fallback = QtWidgets.QLabel("(matplotlib unavailable — no scatter)")
            self._scatter_fallback.setAlignment(QtCore.Qt.AlignCenter)
            root.addWidget(self._scatter_fallback, 1)

        # ---- status / readout ------------------------------------------- #
        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(120)
        root.addWidget(self.report)

        # ---- action buttons --------------------------------------------- #
        row = QtWidgets.QHBoxLayout()
        self.b_run = QtWidgets.QPushButton("Run single-cell profiling")
        self.b_run.setObjectName("primaryAction")
        self.b_run.clicked.connect(self._run)
        self.b_regions = QtWidgets.QPushButton("Make regions from cell clusters")
        self.b_regions.setEnabled(False)
        self.b_regions.clicked.connect(self._make_regions)
        row.addWidget(self.b_run)
        row.addStretch(1)
        row.addWidget(self.b_regions)
        root.addLayout(row)

    # ---- populate from the open sample ----------------------------------- #
    def load_from_main(self):
        """Refresh enabled-state + status from the host window's optical / dataset."""
        self._cells = self._cellmap = self._cellmat = None
        self._coords = self._labels = None
        self.b_regions.setEnabled(False)
        self.report.clear()
        if self.ax is not None:
            self.ax.clear()
            self.ax.set_axis_off()
            if self.canvas is not None:
                self.canvas.draw_idle()

        has_opt = getattr(self.main, "_optical", None) is not None
        has_ds = getattr(self.main, "ds", None) is not None
        self.b_run.setEnabled(has_opt and has_ds)
        if not has_ds:
            self.report.setPlainText("Load a dataset first.")
        elif not has_opt:
            self.report.setPlainText(
                "Load a co-registered optical / histology image first "
                "(File → Image setup…), then align it to the MSI grid.")
        else:
            src = "registration" if self._registration_matrix() is not None \
                else "manual optical alignment"
            self.report.setPlainText(
                f"Ready. Transform source: {src}. Pick a backend and run.")

    # ---- transform resolution -------------------------------------------- #
    def _registration_matrix(self):
        """The authoritative optical→MSI base-px affine from a stored RegistrationResult,
        or ``None`` (then we fall back to the manual alignment)."""
        reg = getattr(self.main, "_registration", None)
        mat = getattr(reg, "matrix", None) if reg is not None else None
        if mat is None:
            return None
        try:
            M = np.asarray(mat, dtype=float)
        except (TypeError, ValueError):
            return None
        return M if M.shape == (3, 3) else None

    def _resolve_transform(self):
        """Return ``(matrix, source)`` — the registration matrix if present, else the
        base-frame affine derived from the manual ``_optical_align``."""
        M = self._registration_matrix()
        if M is not None:
            return M, "registration"
        optical = self.main._optical
        hp, wp = int(optical.shape[0]), int(optical.shape[1])
        h0, w0 = self.main.ds._base_hw
        align = getattr(self.main, "_optical_align", None) \
            or {"tx": 0.0, "ty": 0.0, "scale": 1.0, "angle": 0.0,
                "flipx": False, "flipy": False}
        return _align_matrix(align, hp, wp, int(h0), int(w0)), "optical-align"

    # ---- region confinement ---------------------------------------------- #
    def _scope_mask(self):
        """Bool[n_pix] of picked-region pixels to confine the run to, or ``None`` for the
        whole slide. (No region picker in this dialog yet; reserved hook.)"""
        return None

    # ---- run ------------------------------------------------------------- #
    def _gather(self):
        """Collect the parameter bundle from the widgets."""
        seed = profiles.active_seed()
        return dict(
            backend=self.backend.currentData() or "watershed",
            channel=self.channel.currentText(),
            min_diameter_um=float(self.min_diam.value()),
            invert=self.invert.isChecked(),
            weighting=self.weighting.currentData() or "area",
            pixel_norm=self.pixel_norm.currentText(),
            cell_norm=self.cell_norm.currentText(),
            k=int(self.n_clusters.value()),
            random_state=int(seed),
        )

    def _peaks(self):
        """The m/z list for the cell matrix — the dataset's cached feature peaks."""
        peaks = getattr(self.main.ds, "feature_peaks", None)
        return None if peaks is None else np.asarray(peaks, dtype=float)

    def _run(self):
        if getattr(self.main, "ds", None) is None or getattr(self.main, "_optical", None) is None:
            self._status("Load a dataset and a co-registered optical image first.")
            return
        peaks = self._peaks()
        if peaks is None or peaks.size == 0:
            self._status("No features yet — find peaks / build features first, then re-run.")
            return
        try:
            transform, source = self._resolve_transform()
        except Exception as e:  # noqa: BLE001 — bad alignment shouldn't crash the app
            self._status(f"Couldn't resolve the optical→MSI transform: {e}")
            return
        self._transform = transform
        self._transform_source = source
        params = self._gather()
        ds = self.main.ds
        optical = self.main._optical
        scope_mask = self._scope_mask()
        opt_px = getattr(self.main, "_optical_pixel_size_um", None)

        def work(progress=None):
            cells = singlecell.segment_cells(
                optical, backend=params["backend"], channel=params["channel"],
                min_diameter_um=params["min_diameter_um"],
                optical_pixel_size_um=opt_px, invert=params["invert"],
                random_state=params["random_state"])
            if progress:
                progress(1, 4)
            mapping = singlecell.map_pixels_to_cells(
                ds, cells, transform, optical_pixel_size_um=opt_px,
                weighting=params["weighting"])
            if progress:
                progress(2, 4)
            cellmat = singlecell.build_cell_matrix(
                ds, mapping, peaks, pixel_norm=params["pixel_norm"],
                cell_norm=params["cell_norm"])
            if progress:
                progress(3, 4)
            coords, labels, used = singlecell.cell_segments(
                cellmat, k=params["k"], random_state=params["random_state"])
            if progress:
                progress(4, 4)
            return cells, mapping, cellmat, coords, labels, used

        self.b_run.setEnabled(False)
        runner = getattr(self.main, "_run", None)
        if callable(runner):
            runner(work, on_done=lambda res: self._on_done(res, params, scope_mask),
                   want_progress=True, busy="Single-cell profiling…")
        else:                                   # no host worker → inline
            try:
                res = work()
            except Exception as e:  # noqa: BLE001
                self.b_run.setEnabled(True)
                self._status(f"Single-cell profiling failed: {e}")
                return
            self._on_done(res, params, scope_mask)

    def _on_done(self, res, params, scope_mask):
        self.b_run.setEnabled(True)
        cells, mapping, cellmat, coords, labels, used = res
        self._cells = cells
        self._cellmap = mapping
        self._cellmat = cellmat
        self._coords = np.asarray(coords)
        self._labels = np.asarray(labels)

        cpp = np.asarray(mapping.cells_per_pixel, dtype=float)
        touched = cpp[cpp > 0]
        mean_cpp = float(touched.mean()) if touched.size else 0.0
        n_cells = int(cells.n_cells)
        n_kept, n_peaks = cellmat.matrix.shape

        self._draw_scatter(used)

        lines = [
            f"Segmented {n_cells} cell(s) ({cells.backend}).",
            f"Per-cell matrix: {n_kept} × {n_peaks}  "
            f"(cells kept after min-support filtering).",
            f"Embedding: {used}; {self.n_clusters.value()} clusters.",
            f"Mean cells per touched MSI pixel: {mean_cpp:.2f}"
            + ("  — MSI pixel >> cell; profiles are area-weighted estimates."
               if mean_cpp > 1.5 else "."),
            f"Transform source: {self._transform_source}.",
        ]
        self.report.setPlainText("\n".join(lines))
        self.b_regions.setEnabled(n_kept > 0 and self._labels is not None
                                  and self._labels.size > 0)

        # provenance / audit
        rec = getattr(self.main, "record_step", None)
        if callable(rec):
            try:
                rec("single_cell", label="Single-cell segmentation", params={
                    "backend": params["backend"], "channel": params["channel"],
                    "min_diameter_um": params["min_diameter_um"],
                    "invert": params["invert"], "weighting": params["weighting"],
                    "pixel_norm": params["pixel_norm"], "cell_norm": params["cell_norm"],
                    "n_cells": n_cells, "n_cells_kept": int(n_kept),
                    "mean_cells_per_pixel": round(mean_cpp, 3),
                    "transform_source": self._transform_source,
                    "random_state": params["random_state"],
                })
            except Exception:  # noqa: BLE001 — audit must never break the analysis
                pass
        msg = getattr(self.main, "statusBar", None)
        if callable(msg):
            self.main.statusBar().showMessage(
                f"Single-cell profiling: {n_kept} cell profiles, "
                f"{self.n_clusters.value()} clusters.")

    def _draw_scatter(self, used):
        if self.ax is None or self.canvas is None:
            return
        self.ax.clear()
        coords = self._coords
        labels = self._labels
        if coords is not None and coords.shape[0] > 0:
            self.ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                            s=10, linewidths=0)
            self.ax.set_title(f"Cells ({used})", fontsize=9)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    # ---- roll cell clusters up into regions ------------------------------ #
    def _make_regions(self):
        if self._cellmap is None or self._labels is None or self._cellmat is None:
            self._status("Run single-cell profiling first.")
            return
        new_region = getattr(self.main, "_new_region", None)
        if not callable(new_region):
            self._status("This window can't create regions.")
            return
        n_pix = int(getattr(self.main.ds, "n_pixels", 0))
        if n_pix <= 0:
            self._status("Dataset has no pixels.")
            return

        # kept cell id -> compact index in the PixelCellMap
        id_to_idx = {int(v): i for i, v in enumerate(self._cellmap.cell_ids)}
        kept_ids = np.asarray(self._cellmat.cell_ids, dtype=int)
        labels = np.asarray(self._labels, dtype=int)
        made = 0
        for cl in sorted(set(int(c) for c in labels)):
            mask = np.zeros(n_pix, dtype=bool)
            for ci_id in kept_ids[labels == cl]:
                idx = id_to_idx.get(int(ci_id))
                if idx is None:
                    continue
                pix = self._cellmap.cell_pixels[idx]
                if pix.size:
                    mask[pix] = True
            if not mask.any():
                continue
            try:
                new_region(name=f"Cell cluster {cl + 1}", mask=mask,
                           select=False, refresh=False)
                made += 1
            except TypeError:                       # tolerant of a minimal _new_region
                new_region(name=f"Cell cluster {cl + 1}", mask=mask)
                made += 1
        refresh = getattr(self.main, "_refresh_regions", None) \
            or getattr(self.main, "_refresh_region_list", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # noqa: BLE001
                pass
        self._status(f"Created {made} region(s) from cell clusters.")
        msg = getattr(self.main, "statusBar", None)
        if callable(msg):
            self.main.statusBar().showMessage(
                f"Created {made} region(s) from cell clusters.")

    # ---- helpers --------------------------------------------------------- #
    def _status(self, text):
        self.report.append(text) if self.report.toPlainText() else self.report.setPlainText(text)
